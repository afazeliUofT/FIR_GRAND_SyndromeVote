# ERR fix v6

## What I confirmed from the live repo

- The repo is **not fully aligned** with the latest quiet-launch package.
- `run_ms_receiver7_localbias_winner.sbatch` and `run_ms_receiver7_winner.sbatch` already call `run_main_quiet.py` and route stderr through `tools/filter_tf_stderr.py`.
- `run_ms_receiver7_localbias_scout.sbatch` is still on the older direct launch path (`python HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py ...`) and still uses a weaker CPU-only log setup.
- The only committed `.err` in `logs/` belongs to that scout job, while the matching `results/` directory contains completed summary files. That points to **non-fatal TensorFlow/Sionna stderr noise**, not a failed simulation.

## What this overlay changes

- updates `run_ms_receiver7_localbias_scout.sbatch` to the same quiet CPU-mode launcher style as the newer scripts,
- re-includes `run_main_quiet.py` and `tools/filter_tf_stderr.py` so the repo is self-consistent,
- refreshes the two Receiver-7 winner launchers so all three Receiver-7 jobs use the same stderr policy.

## Expected effect

On CPU runs, benign TensorFlow GPU-probe warnings should stop polluting `.err`.
Real Python exceptions and shell failures will still appear.
