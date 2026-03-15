# FIR_GRAND Receiver-7 basis-GRAND upgrade

This package upgrades the current Receiver-6 soft-anchor path with a new **Receiver-7 / basis-GRAND + block-debias restart** path.

## Why this upgrade
The latest `run_ms_scout.sbatch` and `run_ms.sbatch` results show two things:

1. The hybrid occasionally wins, but only in a **very high-FER regime**.
2. The current strongest path (`hybahr15`) spends a lot of stage-2 effort, yet rescues only a small fraction of its own stage-1 failures.

That means the bottleneck is not simply:
- more LDPC iterations,
- more raw GRAND patterns,
- or a slightly different SNR point.

The residual errors look **structured**: correlated reliability distortions, wrong-BP-basin states, and grouped local disagreement patterns. Receiver-7 attacks that directly.

## What Receiver-7 does
Receiver-7 adds a new hybrid decoder family:

- `hybbgr4`
- `hybbgr8`
- `hybbgr15`

Its stage-2 flow is:

1. syndrome-vote + check-cover front-end
2. build a small library of **structured basis patterns**
   - unsatisfied-check neighborhoods
   - short ranked windows
   - posterior-vs-channel disagreement groups
   - top singleton suspects
3. run GRAND over **combinations of those basis vectors**
4. use a **block-debias anchored restart** on the full LDPC graph
5. fall back to the existing peel / GRAND machinery

This is still GRAND-based, but it is much more efficient than spending hundreds of thousands of tests on raw independent bit-flip patterns.

## Included files
- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py` — upgraded simulator with Receiver-7
- `run_ms.sbatch` — targeted moderate-impairment run intended to show a practical hybrid advantage
- `run_ms_scout.sbatch` — broad scout run to find the actual winning region
- `run_ms_fair.sbatch` — calmer comparison run, still with Receiver-7 available
- `UPGRADE_INSTRUCTIONS.md` — exact replace/add/remove steps
- `RECEIVER7_NOTES.md` — tuning notes and new environment variables
- `CURRENT_RESULTS_REVIEW.md` — concise diagnosis of why the current Receiver-6 runs are not convincing
- `REMOVE_FROM_REPO.txt` — stale files to delete for a clean repo

## Important operating-point change
The previous scout/stress runs were too harsh:
- even `ldpc100` stayed at very high FER across the whole sweep,
- so the hybrid only won in a regime that is not a persuasive demonstration.

The new `run_ms.sbatch` and `run_ms_scout.sbatch` intentionally move to a **milder but still 5G-compatible imperfect-CSI regime**, where:
- failures are still structured,
- but they are not so global that every decoder is broken.

That is a better place to show a meaningful hybrid win.

## Practical recommendation
Run in this order:
1. `sbatch run_ms_scout.sbatch`
2. inspect the FER crossover region
3. `sbatch run_ms.sbatch`
4. optionally `sbatch run_ms_fair.sbatch`

The scout run is important because the right region should be identified from data, not guessed.
