1. From the repo root, overlay the package:

```bash
unzip -o FIR_GRAND_hybridmeta_v4.zip
```

2. Sanity-check before running:

```bash
python -m py_compile HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py
bash -n run_ms_hybridmeta_blockbias_scout.sbatch run_ms_hybridmeta_blockbias_winner.sbatch
```

3. Run the scout first:

```bash
sbatch run_ms_hybridmeta_blockbias_scout.sbatch
```

4. After it finishes, inspect the new files in the result folder:

- `*_summary.csv`
- `*_summary_tails.csv`
- `*_summary_diagnostics.csv`

5. Rank the gains quickly:

```bash
python analyze_hybrid_diagnostics.py \
  results/run_ms_hybridmeta_blockbias_scout.sbatch_<jobid>/*_summary.csv \
  results/run_ms_hybridmeta_blockbias_scout.sbatch_<jobid>/*_summary_diagnostics.csv
```

6. Then run the narrowed winner job:

```bash
sbatch run_ms_hybridmeta_blockbias_winner.sbatch
```
