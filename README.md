# NCAA peptide library -> AlphaFold 3 bulk screen

10,000,000 unique peptides (length 6-30) built from 32 real CCD non-canonical amino acids,
split into 14 tar files (`ncaa_small_NNof14.tar`). Each tar holds `lib_L??.tsv.xz` files (one per length),
the library README and `assemble_inputs.py`.

## Run on a cluster (Linux, AF3 installed, weights available)
```
mkdir -p lib && for t in ncaa_small_*of14.tar; do tar -xf $t -C lib; done   # or extract only the parts you need
pip install gemmi                                                          # PDB writing, in the AF3 env

# find the best number of workers on one GPU first (watch nvidia-smi + jobs/min):
python run_af3_bulk.py --parts "lib/lib_L*.tsv.xz" --workdir test --af3_dir /path/alphafold3 \
    --model_dir /path/af3_weights --gpus 0 --workers_per_gpu 8 --batch_size 100 --limit 200

# full run, one node (all GPUs, auto worker count):
python run_af3_bulk.py --parts "lib/lib_L*.tsv.xz" --workdir out --af3_dir ... --model_dir ... \
    --gpus 0,1,2,3 --auto_workers --vram_per_worker_gb 8 --jax_cache $SCRATCH/jaxcache
# many nodes: add  --shard_index $SLURM_ARRAY_TASK_ID --num_shards $SLURM_ARRAY_TASK_COUNT
```
Single chains have no ipTM, so the screen is `mean_plddt > 90` (change with `--screen_metric/--min_score`).
Every job's scores go to `results/results_w*.tsv`; only hits get a PDB (`hits_pdb/`, one tar per batch).
The run is resumable: re-run the same command after any interruption. Use the same `--parts` and
GPU/worker/shard settings when resuming (rows are assigned to workers by row number).
Measured on an RTX 4060 laptop: ~6 s/job, 68% of jobs pass pLDDT > 90.
