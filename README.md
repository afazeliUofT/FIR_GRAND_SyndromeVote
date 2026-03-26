# FIR_GRAND Receiver-7 overlay

This overlay does two things that the current repo does not do reliably:

1. It enables a stronger **Receiver-7 / basis-GRAND + block-debias restart** path.
2. It fixes two evaluation issues that were holding the hybrid back:
   - the rescue stage now probes **multiple LDPC snapshots**, not just the final stage-1 snapshot;
   - the Slurm scripts now default to **paired decoder frame streams** (`PAIR_DECODER_STREAMS=1`), so FER comparisons against legacy LDPC are directly comparable on the same frame bank.

## Important repo-state note
The committed `logs/` and `results/` folders are **stale older runs**. They do **not** validate Receiver-7.
The current repo scripts already mention Receiver-7, but the committed outputs still come from older harsh-channel runs without `hybbgr*` results.
See `CURRENT_RESULTS_REVIEW.md`.

## What changed in the code
The main simulator (`HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`) now supports:

- `RUN_RECEIVER1=0/1` so the plain GRAND hybrid can be disabled when you only want LDPC vs Receiver-7,
- `GRAND_RESCUE_SNAPSHOT_ITERS="..."` to rescue from several stage-1 snapshots,
- cumulative stage-2 accounting across repeated snapshot tries,
- `PAIR_DECODER_STREAMS=1` for common-random-number / same-frame decoder comparisons.

## Included files
- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`
- `run_ms_scout.sbatch`
- `run_ms_fair.sbatch`
- `run_ms_receiver7_winner.sbatch`
- `README.md`
- `UPGRADE_INSTRUCTIONS.md`
- `CURRENT_RESULTS_REVIEW.md`
- `RECEIVER7_NOTES.md`
- `REMOVE_FROM_REPO.txt`

## Which Slurm to run
If your goal is **best FER and the cleanest Receiver-7 vs legacy LDPC comparison**, run:

```bash
sbatch run_ms_receiver7_winner.sbatch
```

That run is intentionally narrow:
- `ldpc100` vs `hybbgr100`
- same stage-1 depth
- same frame bank (`PAIR_DECODER_STREAMS=1`)
- aggressive Receiver-7 settings
- multi-snapshot rescue schedule ending at iteration 100

## Optional follow-up runs
Use these only after the winner run if you want broader mapping:

```bash
sbatch run_ms_scout.sbatch
sbatch run_ms.sbatch
sbatch run_ms_fair.sbatch
```

## Cleanup
Before committing new results, remove the stale tracked outputs listed in `REMOVE_FROM_REPO.txt`.
