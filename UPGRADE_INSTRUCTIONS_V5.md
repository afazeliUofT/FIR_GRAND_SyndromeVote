# Upgrade instructions (v5)

From the repo root:

```bash
unzip -o /path/to/FIR_GRAND_errfix_v5.zip
chmod +x run_ms_receiver7_localbias_scout.sbatch \
         run_ms_receiver7_localbias_winner.sbatch \
         run_ms_receiver7_winner.sbatch \
         run_main_quiet.py \
         tools/filter_tf_stderr.py
bash -n run_ms_receiver7_localbias_scout.sbatch
bash -n run_ms_receiver7_localbias_winner.sbatch
bash -n run_ms_receiver7_winner.sbatch
python -m py_compile run_main_quiet.py tools/filter_tf_stderr.py
```

Then rerun the affected job first:

```bash
sbatch run_ms_receiver7_localbias_scout.sbatch
```

If you also want the cleaned narrow runs:

```bash
sbatch run_ms_receiver7_localbias_winner.sbatch
sbatch run_ms_receiver7_winner.sbatch
```
