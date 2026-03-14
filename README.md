# FIR_GRAND Receiver-5 OSD + Anchored-Restart upgrade

This package extends the current Receiver-1 / Receiver-2 / Receiver-3 / Receiver-4 pipeline with a stronger stage-2 path aimed at the failure mode seen in the recent runs: post-BP trapping-set / pseudo-codeword states that are not well handled by exact local GRAND alone.

## What is new

New decoders:

- `hybosd4`
- `hybosd8`
- `hybosd15`

They keep all existing outputs and add a stronger stage-2 path:

1. syndrome-vote + check-cover front-end
2. optional channel-vs-posterior disagreement augmentation of the candidate set
3. local OSD / MRB-style candidate generation on the induced local parity subsystem
4. anchored **full-graph** LDPC restarts from the original channel LLRs
5. peel / weighted GF(2) fallback
6. GRAND fallback

## New CSI mode

The upgraded Python also adds a new `SIONNA_CSI_MODE=nr_imperfect` mode. It starts from the existing block-LS proxy and adds structured phase / amplitude / interpolation bias so the simulation can stress the receiver in a more NR-like imperfect-CSI regime.

## Included files

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py` — complete upgraded simulator
- `run_ms.sbatch` — stress / demonstration script intended to maximize the probability of a visible hybrid win
- `run_ms_fair.sbatch` — fairer comparison close to the current static TDL-A / perfect-CSI setup
- `UPGRADE_INSTRUCTIONS.md` — exact replace / add steps for your repository
- `RECEIVER5_TUNING_NOTES.md` — what the new knobs do and what to tweak first

## Important note

`run_ms.sbatch` changes the operating point on purpose:

- stronger selectivity and mobility
- residual CFO
- imperfect CSI via `nr_imperfect`
- SNR sweep concentrated around the observed near-crossover region

That is the best script for demonstrating a hybrid advantage, but it is **not** apples-to-apples with the old static perfect-CSI TDL-A run. Use `run_ms_fair.sbatch` for a closer comparison.
