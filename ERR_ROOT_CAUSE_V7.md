# Root cause fixed in v7

The repeated `.err` issue is caused by **corrupted text-file formatting** in the committed repo state:

- several `.sbatch` launchers were flattened into one line;
- `run_main_quiet.py` was flattened into one line;
- `tools/filter_tf_stderr.py` was flattened into one line.

That corruption makes the helpers and launchers unreliable or outright invalid, which is why the same stderr issue kept reappearing.

This package restores those files as proper multiline scripts.

Files replaced:
- `run_main_quiet.py`
- `tools/filter_tf_stderr.py`
- `run_ms_receiver7_localbias_scout.sbatch`
- `run_ms_receiver7_localbias_winner.sbatch`
- `run_ms_receiver7_winner.sbatch`
- `run_ms_hybridmeta_blockbias_scout.sbatch`
- `run_ms_hybridmeta_blockbias_winner.sbatch`

Recommended rerun:
- `sbatch run_ms_receiver7_localbias_scout.sbatch`
- `sbatch run_ms_hybridmeta_blockbias_scout.sbatch`
