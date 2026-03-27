Apply from the repo root:

```bash
unzip -o FIR_GRAND_errfix_v6.zip
chmod +x run_ms_receiver7_localbias_scout.sbatch \
         run_ms_receiver7_localbias_winner.sbatch \
         run_ms_receiver7_winner.sbatch \
         run_main_quiet.py \
         tools/filter_tf_stderr.py
```

Then rerun the scout job:

```bash
sbatch run_ms_receiver7_localbias_scout.sbatch
```

If you want to verify the launch script before submitting:

```bash
bash -n run_ms_receiver7_localbias_scout.sbatch
python -m py_compile run_main_quiet.py tools/filter_tf_stderr.py
```
