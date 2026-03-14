# FIR_GRAND Receiver-4 Chase-Triggered GRAND upgrade

This package adds a stronger hybrid receiver on top of the existing Receiver-1/2/3 pipeline.

## What is new

New decoders:
- `hybctg4`
- `hybctg8`
- `hybctg15`

They keep all existing simulation outputs and add a stronger stage-2 path:
1. syndrome-vote + check-cover front-end
2. Chase-style local candidate generation on the least-reliable residual bits
3. short LDPC re-decoding ("polish" pass) under each strong local hypothesis
4. peel/GF(2) pre-solver
5. GRAND fallback

This is designed to attack post-BP trapping-set / near-codeword failures that plain local GRAND often misses.

## Included files

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py` — complete upgraded simulator
- `run_ms_fair.sbatch` — fair comparison under a channel close to the current repo setup
- `run_ms.sbatch` — more stressful but still 5G-compatible channel setup intended to increase the probability that the hybrid stage-2 clearly helps over legacy LDPC

## Important note

`run_ms.sbatch` changes the operating point:
- stronger frequency selectivity
- mobility
- residual CFO
- imperfect-CSI proxy LLR generation

So it is useful for demonstrating the hybrid advantage, but it is not an apples-to-apples comparison to the old perfect-CSI TDL-A run.
