# ERR log diagnosis (v5)

## What is confirmed from the live repo
- The repo is **not** at the latest package state. It contains the v2 local-bias files, but none of the later v3/v4 files.
- The only committed log pair currently visible is `run_ms_receiver7_localbias_scout.sbatch-29806492.{out,err}`.
- That run also produced committed result files (`summary.csv` and `summary_tails.csv`), so the job completed and the `.err` is almost certainly **non-fatal stderr noise**, not a crash.

## Most likely cause of the `.err`
The current receiver7 Slurm scripts run CPU jobs with `USE_GPU=0`, but they only set `TF_CPP_MIN_LOG_LEVEL` to `2`, and the main Python entry also only uses `setdefault(...)` for `CUDA_VISIBLE_DEVICES` / `TF_CPP_MIN_LOG_LEVEL`.

With GPU-enabled TensorFlow wheels on CPU nodes, this still commonly emits non-fatal stderr lines such as:
- cuDNN/cuFFT/cuBLAS factory registration warnings
- `failed call to cuInit`
- pre-absl TensorFlow logger messages

## What v5 changes
1. Adds `run_main_quiet.py` so the environment is forced before TensorFlow import.
2. Raises the CPU-mode TensorFlow log threshold to `3`.
3. Adds `tools/filter_tf_stderr.py` to drop only known benign CPU-node TensorFlow GPU-probe noise while preserving real exceptions.
4. Updates the three receiver7 Slurm entry points to use the wrapper + stderr filter.

This is meant to make `.err` either empty or contain only actionable errors.
