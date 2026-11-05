"""Turn rows of the library into ready-to-run input files for AF3, Boltz-2 and RF3.
Usage: python3 assemble_inputs.py <part.tsv.gz> <combo_id> [<combo_id> ...]
Writes af3/<id>.json, boltz2/<id>.yaml, rf3/<id>.json in the current folder.
Empty MSA is used everywhere (single-sequence mode); change if you have MSAs.
"""
import gzip, json, os, sys, csv
csv.field_size_limit(10**7)
part, ids = sys.argv[1], set(sys.argv[2:])
for d in ("af3", "boltz2", "rf3"):
    os.makedirs(d, exist_ok=True)
with gzip.open(part, "rt") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        i = r["combo_id"]
        if i not in ids:
            continue
        prot = json.loads(r["af3_protein_entry"])
        prot["protein"].update({"unpairedMsa": "", "pairedMsa": "", "templates": []})
        json.dump({"name": i, "modelSeeds": [1], "sequences": [prot], "dialect": "alphafold3", "version": 1},
                  open(f"af3/{i}.json", "w"))
        open(f"boltz2/{i}.yaml", "w").write("version: 1\nsequences:\n  - " + r["boltz2_protein_entry"] + "\n")
        json.dump({"name": i, "components": [{"seq": r["rf3_seq"], "chain_id": "A"}]}, open(f"rf3/{i}.json", "w"))
