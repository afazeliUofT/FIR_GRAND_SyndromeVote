1. From the repo root:

```bash
unzip -o /path/to/FIR_GRAND_blockbias_v3.zip
chmod +x run_ms_receiver7_blockbias_scout.sbatch run_ms_receiver7_blockbias_winner.sbatch analyze_hybrid_gain.py
```

2. Run the scout first:

```bash
sbatch run_ms_receiver7_blockbias_scout.sbatch
```

3. After it completes, rank the best FER-gain region:

```bash
python analyze_hybrid_gain.py results/run_ms_receiver7_blockbias_scout.sbatch_<jobid>/sionna_tdl_sionna5g_k1024_n2048_qm1_hybrid_*_summary.csv
```

4. Then run the narrowed winner job:

```bash
sbatch run_ms_receiver7_blockbias_winner.sbatch
```

The new scripts do not replace your main simulator. They add a more focused operating point and a small result-analysis helper.
