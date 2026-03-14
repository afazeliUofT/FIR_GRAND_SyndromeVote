# Receiver-5 tuning notes

## What Receiver-5 is trying to fix

The recent runs suggest the residual post-BP failures are not mainly “small exact local error patterns.” They look more like trapping-set / pseudo-codeword failures with structured soft-information errors. Receiver-5 therefore tries to:

- build a better candidate set with syndrome-vote and disagreement bits
- generate *code-aware* local candidates with OSD instead of only exact low-weight local flips
- restart the **full** LDPC decoder from the original channel LLRs, but with strong anchors on the local hypothesis

## First knobs to tune if `hybosd15` is still close but not winning

1. Increase global restart effort:
   - `GRAND_OSD_RESTART_MAX_CANDIDATES`
   - `GRAND_OSD_RESTART_ITERS`

2. Increase local OSD breadth:
   - `GRAND_OSD_RATIO`
   - `GRAND_OSD_MAX_BITS`
   - `GRAND_OSD_ENUM_MAX_BITS`
   - `GRAND_OSD_MAX_CANDIDATES`

3. Strengthen the candidate-set front-end:
   - `GRAND_OSD_CHECK_COVER_K`
   - `GRAND_OSD_DISAGREEMENT_BITS`

4. Strengthen restart anchors:
   - `GRAND_OSD_RESTART_GAIN`
   - `GRAND_OSD_RESTART_DUAL_GAIN`
   - `GRAND_OSD_RESTART_ABS_FLOOR`
   - `GRAND_OSD_RESTART_ANCHOR_ALL=1` for a more aggressive mode

## Best first target

If runtime is limited, focus on `hybosd15`. It is the strongest candidate to beat `ldpc20`.

## Channel / operating-point advice

Use `run_ms.sbatch` first if your goal is to demonstrate a visible hybrid advantage. That script intentionally pushes the system into a more frequency-selective, mobile, imperfect-CSI regime where a code-validity outer stage has a better chance to help.

## Sweet-spot SNR range

Start with the bundled default sweep in `run_ms.sbatch`. It concentrates around the previously observed near-crossover region. If you need a tighter sweep, try a smaller step around the best region that emerges from the first run.

## What not to over-interpret

A BER win without a FER win is not yet a convincing system-level win. Treat the goal as beating `ldpc20` on **FER** first, then check BER and timing.
