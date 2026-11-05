#!/usr/bin/env python3
"""Bulk AlphaFold 3 runner for very large sequence libraries (millions of jobs).

Reads library part files (the same .tsv/.tsv.gz that assemble_inputs.py reads; needs columns
`combo_id` and `af3_protein_entry`, i.e. the AF3 protein entry incl. non-canonical residues as CCD
`modifications`), and runs AF3 inference in batches on every GPU you give it.

Design for scale
  * Static sharding: row i goes to worker (i % total_workers). No coordinator, no queue, so it works
    the same on one workstation or as a Slurm array (use --shard_index / --num_shards).
  * Batching: one run_alphafold.py call handles --batch_size inputs, so weights loading + JIT
    compilation are paid once per batch, not per sequence.
  * Resumable: finished ids are appended to a per-worker ledger AFTER their scores are written.
    Kill it any time and re-run the same command.
  * Disk: scores for every job go to results_*.tsv. Structures are screened: only jobs with
    --screen_metric > --min_score (default mean_plddt > 90) get a PDB (needs `pip install gemmi`); all others are
    deleted right after scoring. Hits are
    packed into one tar per batch, never millions of loose folders.
  * Failures: a batch that loses jobs is bisected until the bad inputs are isolated and logged to
    failed_*.tsv; good ones are not lost.

AF3 runs on Linux only (Docker/WSL2/cluster); run this where run_alphafold.py works.

Example (single node, 4 GPUs):
  python run_af3_bulk.py --parts "library/part_*.tsv.gz" --workdir out \
      --af3_dir ~/alphafold3 --model_dir ~/af3_weights --gpus 0,1,2,3
Slurm array (each task = one node):
  python run_af3_bulk.py ... --gpus 0,1,2,3 --shard_index $SLURM_ARRAY_TASK_ID --num_shards $SLURM_ARRAY_TASK_COUNT
"""
import argparse, csv, glob, gzip, io, json, multiprocessing as mp, os, re, shutil, subprocess, sys, tarfile, time

csv.field_size_limit(10**7)
RESULT_COLS = ["id", "ranking_score", "ptm", "iptm", "mean_plddt", "fraction_disordered", "has_clash", "seconds"]


def sanitize(name):
    """AF3 names its output folder this way."""
    return re.sub(r"[^a-z0-9_\-.]", "_", name.lower())


def open_text(path):
    if path.endswith(".xz"):
        import lzma
        return lzma.open(path, "rt")
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def iter_rows(files, worker, total):
    """Yield (id, raw_row) for rows owned by `worker`. Deterministic across runs."""
    i = -1
    for path in files:
        delim = "," if path.replace(".gz", "").replace(".xz", "").endswith(".csv") else "\t"
        with open_text(path) as f:
            for row in csv.DictReader(f, delimiter=delim):
                i += 1
                if i % total == worker:
                    yield row


def build_job(row, seeds, user_ccd):
    """AF3 input dict for one library row, single-sequence mode (no MSA/templates)."""
    if row.get("af3_protein_entry"):
        entry = json.loads(row["af3_protein_entry"])
    else:  # plain-sequence fallback: columns id, sequence
        entry = {"protein": {"id": "A", "sequence": row["sequence"]}}
    entry["protein"].update({"unpairedMsa": "", "pairedMsa": "", "templates": []})
    job = {"name": row.get("combo_id") or row["id"], "modelSeeds": seeds,
           "sequences": [entry], "dialect": "alphafold3", "version": 1}
    if user_ccd:
        job["userCCD"] = user_ccd
    return job


def read_scores(out_dir, name, seconds):
    d = os.path.join(out_dir, sanitize(name))
    s = os.path.join(d, f"{sanitize(name)}_summary_confidences.json")
    c = os.path.join(d, f"{sanitize(name)}_confidences.json")
    if not (os.path.isfile(s) and os.path.isfile(c)):
        return None
    summ = json.load(open(s))
    pl = json.load(open(c)).get("atom_plddts") or [0]
    r = os.path.join(d, f"{sanitize(name)}_ranking_scores.csv")
    return {"id": name, "ranking_score": summ.get("ranking_score", ""), "ptm": summ.get("ptm", ""),
            "iptm": summ.get("iptm", ""), "mean_plddt": round(sum(pl) / len(pl), 2),
            "fraction_disordered": summ.get("fraction_disordered", ""),
            "has_clash": summ.get("has_clash", ""), "seconds": round(seconds, 2)}


def cif_to_pdb(cif_path, pdb_path):
    """Write a PDB file from an AF3 model.cif. Returns True on success."""
    try:
        import gemmi
        st = gemmi.read_structure(cif_path)
        st.setup_entities()
        st.shorten_chain_names()
        st.write_pdb(pdb_path)
        return True
    except ImportError:
        pass
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        io_ = PDBIO()
        io_.set_structure(MMCIFParser(QUIET=True).get_structure("m", cif_path))
        io_.save(pdb_path)
        return True
    except ImportError:
        sys.exit("Need `pip install gemmi` (or biopython) to write PDB files.")


def passes(score, a):
    v = score.get(a.screen_metric)
    try:
        return float(v) > a.min_score
    except (TypeError, ValueError):
        return False  # metric missing (e.g. iptm for a single chain) -> not kept


def run_af3(jobs, a, gpu, tmp):
    """Run AF3 on a list of jobs; return {name: scores}. Files stay in tmp for structure harvesting."""
    in_dir, out_dir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
    for d in (in_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for j in jobs:
        json.dump(j, open(os.path.join(in_dir, sanitize(j["name"]) + ".json"), "w"))
    cmd = [sys.executable, os.path.join(a.af3_dir, "run_alphafold.py"),
           f"--input_dir={in_dir}", f"--output_dir={out_dir}", f"--model_dir={a.model_dir}",
           "--run_data_pipeline=false", "--run_inference=true",
           f"--num_diffusion_samples={a.num_diffusion_samples}", f"--num_recycles={a.num_recycles}"]
    if a.db_dir:
        cmd.append(f"--db_dir={a.db_dir}")
    if a.jax_cache:
        cmd.append(f"--jax_compilation_cache_dir={a.jax_cache}")
    cmd += a.extra_args.split()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    per = (time.time() - t0) / max(1, len(jobs))
    got = {}
    for j in jobs:
        s = read_scores(out_dir, j["name"], per)
        if s:
            got[j["name"]] = s
    return got, out_dir, p.stderr[-2000:] if len(got) < len(jobs) else ""


def process(jobs, a, gpu, tmp, sink, depth=0):
    """Run jobs; on partial failure bisect so one bad input cannot sink a whole batch."""
    got, out_dir, err = run_af3(jobs, a, gpu, tmp)
    sink["ok"](got, out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)  # everything not saved by the screen is deleted here
    missing = [j for j in jobs if j["name"] not in got]
    if not missing:
        return
    if len(jobs) == 1:
        sink["fail"](jobs[0]["name"], err.replace("\n", " | "))
        return
    if len(missing) == len(jobs):  # nothing worked: split to find the culprit(s)
        mid = len(jobs) // 2
        process(jobs[:mid], a, gpu, tmp, sink, depth + 1)
        process(jobs[mid:], a, gpu, tmp, sink, depth + 1)
    else:  # some worked: just retry the missing ones
        process(missing, a, gpu, tmp, sink, depth + 1)


def worker(args):
    a, wid, total, gpu = args
    res_dir = os.path.join(a.workdir, "results")
    os.makedirs(res_dir, exist_ok=True)
    ledger_p = os.path.join(res_dir, f"done_w{wid}.txt")
    done = set(open(ledger_p).read().split()) if os.path.exists(ledger_p) else set()
    res_p = os.path.join(res_dir, f"results_w{wid}.tsv")
    new_file = not os.path.exists(res_p)
    res_f = open(res_p, "a", newline="")
    res = csv.DictWriter(res_f, RESULT_COLS, delimiter="\t")
    if new_file:
        res.writeheader()
    led = open(ledger_p, "a")
    fail = open(os.path.join(res_dir, f"failed_w{wid}.tsv"), "a")
    tmp = os.path.join(a.workdir, "tmp", f"w{wid}")
    os.makedirs(tmp, exist_ok=True)
    hit_dir = os.path.join(a.workdir, "hits_pdb", f"w{wid}")
    os.makedirs(hit_dir, exist_ok=True)
    state = {"batches": 0, "n": len(done), "hits": 0}

    def on_ok(got, out_dir):
        for name, s in got.items():
            res.writerow(s)
        res_f.flush(); os.fsync(res_f.fileno())
        keep = [n for n, s in got.items() if passes(s, a)]
        if keep:  # screen: only hits get a PDB; all other structures vanish with out_dir
            pdbs = []
            for name in keep:
                cif = os.path.join(out_dir, sanitize(name), f"{sanitize(name)}_model.cif")
                if os.path.isfile(cif):
                    p = os.path.join(out_dir, f"{sanitize(name)}.pdb")
                    cif_to_pdb(cif, p)
                    pdbs.append(p)
            if a.loose_pdb:
                for p in pdbs:
                    shutil.copy(p, os.path.join(hit_dir, os.path.basename(p)))
            else:
                tp = os.path.join(hit_dir, f"b{int(time.time()*1000)}_{len(pdbs)}.tar")
                with tarfile.open(tp, "w") as t:
                    for p in pdbs:
                        t.add(p, arcname=os.path.basename(p))
            state["hits"] += len(pdbs)
        led.write("\n".join(got) + ("\n" if got else "")); led.flush(); os.fsync(led.fileno())
        state["n"] += len(got)

    def on_fail(name, err):
        fail.write(f"{name}\t{err}\n"); fail.flush()
        led.write(name + "\n"); led.flush()  # don't retry forever; failed_*.tsv lists them

    sink = {"ok": on_ok, "fail": on_fail}
    batch = []
    t_start = time.time()

    def flush():
        if not batch:
            return
        process(list(batch), a, gpu, tmp, sink)
        batch.clear()
        state["batches"] += 1
        el = time.time() - t_start
        print(f"[w{wid} gpu{gpu}] total done {state['n']}, hits kept {state['hits']}  ({el/60:.1f} min this session)", flush=True)

    files = sorted(sum((glob.glob(p) for p in a.parts), []))
    for row in iter_rows(files, wid, total):
        name = row.get("combo_id") or row.get("id")
        if name in done:
            continue
        batch.append(build_job(row, a.seeds, a.user_ccd))
        if len(batch) >= a.batch_size:
            flush()
        if a.limit and state["n"] >= a.limit:
            break
    flush()
    shutil.rmtree(tmp, ignore_errors=True)
    return wid, state["n"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", required=True, help="glob(s) of library part files (.tsv/.tsv.gz/.csv)")
    ap.add_argument("--workdir", required=True, help="output + resume state lives here")
    ap.add_argument("--af3_dir", required=True, help="folder containing run_alphafold.py")
    ap.add_argument("--model_dir", required=True, help="AF3 weights folder")
    ap.add_argument("--db_dir", default="", help="only needed if AF3 insists; unused with empty MSA")
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids on this node, e.g. 0,1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=1,
                    help="AF3 processes sharing each GPU; raise to fill a big card (e.g. 143 GB H200)")
    ap.add_argument("--auto_workers", action="store_true",
                    help="pick workers_per_gpu = GPU VRAM // --vram_per_worker_gb (capped by CPU cores)")
    ap.add_argument("--vram_per_worker_gb", type=float, default=12.0,
                    help="peak VRAM one worker needs; measure with nvidia-smi on a --limit run")
    ap.add_argument("--screen_metric", default="mean_plddt", choices=["iptm", "ptm", "ranking_score", "mean_plddt"])
    ap.add_argument("--min_score", type=float, default=90.0, help="keep PDB only if metric > this")
    ap.add_argument("--loose_pdb", action="store_true", help="write hits as loose .pdb files instead of one tar per batch")
    ap.add_argument("--shard_index", type=int, default=0, help="this node's index (Slurm array task id)")
    ap.add_argument("--num_shards", type=int, default=1, help="number of nodes")
    ap.add_argument("--batch_size", type=int, default=200, help="inputs per run_alphafold.py call")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--num_diffusion_samples", type=int, default=1, help="AF3 default is 5; 1 is ~much faster")
    ap.add_argument("--num_recycles", type=int, default=10)
    ap.add_argument("--user_ccd", default="", help="path to mmCIF text defining custom CCD components (userCCD)")
    ap.add_argument("--jax_cache", default="", help="persistent JAX compilation cache dir (saves minutes per batch)")
    ap.add_argument("--limit", type=int, default=0, help="stop each worker after N jobs (for timing tests)")
    ap.add_argument("--extra_args", default="", help="extra flags passed to run_alphafold.py")
    a = ap.parse_args()

    a.user_ccd = open(a.user_ccd).read() if a.user_ccd else ""
    gpus = [g for g in a.gpus.split(",") if g != ""]
    if a.auto_workers:
        try:
            mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits",
                                  f"--id={gpus[0]}"], capture_output=True, text=True).stdout.split()[0]
            a.workers_per_gpu = max(1, int(float(mem) / 1024 // a.vram_per_worker_gb))
            cpu_cap = max(1, (os.cpu_count() or 1) // (2 * len(gpus)))
            a.workers_per_gpu = min(a.workers_per_gpu, cpu_cap)
        except Exception as e:
            print(f"auto_workers failed ({e}); using workers_per_gpu={a.workers_per_gpu}")
    print(f"workers_per_gpu = {a.workers_per_gpu}")
    per_node = len(gpus) * a.workers_per_gpu
    total = per_node * a.num_shards
    jobs = []
    for k in range(per_node):
        wid = a.shard_index * per_node + k
        jobs.append((a, wid, total, gpus[k % len(gpus)]))
    os.makedirs(a.workdir, exist_ok=True)
    print(f"{len(jobs)} workers on this node, {total} globally; workdir={a.workdir}", flush=True)
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        for wid, n in pool.imap_unordered(worker, jobs):
            print(f"worker {wid} finished, {n} jobs recorded", flush=True)
    print(f"Done. Scores: {a.workdir}/results/results_w*.tsv (same header; concatenate). Failures: failed_w*.tsv")


if __name__ == "__main__":
    main()
