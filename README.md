# FIR_GRAND Receiver-6 soft-anchor upgrade

This package extends the current Receiver-1 / Receiver-2 / Receiver-3 / Receiver-4 / Receiver-5 pipeline with a stronger stage-2 path aimed at the failure mode seen in the latest runs:

- post-BP trapping-set / wrong-fixed-point states,
- structured LLR bias from imperfect CSI,
- local residuals for which exact OSD/GF(2) reprocessing often has zero useful free dimension.

## What is new

New decoders:
- `hybahr4`
- `hybahr8`
- `hybahr15`

They keep all existing outputs and add a stronger stage-2 path:
1. syndrome-vote + check-cover front-end,
2. soft local hypothesis enumeration on the residual candidate set,
3. ranking by syndrome reduction plus LLR cost,
4. anchored full-graph LDPC restarts from the original channel LLRs,
5. peel / weighted GF(2) fallback,
6. GRAND fallback.

## Why this is needed

The latest Receiver-5 runs show that the local OSD path is often not really engaging:
- `osd_free_dim_mean` is essentially zero,
- `restart_num_runs_mean` is zero,
- and the final rescue fraction is far below what is needed to catch `ldpc20` consistently.

That means the dominant failures are not behaving like small exact local error patterns. They look more like wrong-BP-basin failures. Receiver-6 attacks that regime directly by generating *soft* local hypotheses even when the local parity subsystem is not nicely solvable.

## Included files

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py` — upgraded simulator with Receiver-6
- `run_ms.sbatch` — stressed NR-like run with Receiver-6 enabled and a broader default sweep
- `run_ms_fair.sbatch` — closer apples-to-apples run with Receiver-6 enabled
- `run_ms_scout.sbatch` — broad scout sweep to find the actual sweet spot first
- `UPGRADE_INSTRUCTIONS.md` — exact replace/add steps
- `RECEIVER6_NOTES.md` — what the new knobs do and what to tweak first
- `CURRENT_RUN_DIAGNOSIS.md` — concise diagnosis of why the current Receiver-5 runs do not win yet

## Important note

`run_ms.sbatch` and `run_ms_scout.sbatch` intentionally target a reliability-distorting 5G-compatible operating region. They are better for demonstrating where a hybrid can help, but they are not apples-to-apples with the static perfect-CSI TDL-A setup.

Use `run_ms_fair.sbatch` when you want a closer comparison to the calmer baseline setup.
