# Review of the currently committed results

## Bottom line
The currently committed `logs/` and `results/` directories are **not the right evidence** for Receiver-7.
They are stale outputs from older, harsher runs, and they do not contain any `hybbgr*` results.

## What the stale outputs show

### 1) `run_ms.sbatch_27796723`
The committed summary contains:
- `ldpc4`, `ldpc8`, `ldpc15`, `ldpc20`, `ldpc100`
- `hyb*`, `hybsv*`, `hybptg*`, `hybctg*`, `hybosd*`, `hybahr*`
- **no `hybbgr*` entries**

Observed behavior from that stale result set:
- `ldpc100` is best at almost every SNR point.
- The only noticeable FER win over `ldpc100` is `hybahr15` at the lowest SNR point, and the gain is tiny (about 0.014 absolute FER).
- That tiny FER gain comes with a huge stage-2 cost (roughly milliseconds per frame instead of a few hundred microseconds).

### 2) `run_ms_scout.sbatch_27768739`
Again the committed summary contains only:
- `ldpc15`, `ldpc20`, `ldpc100`
- `hyb15`, `hybptg15`, `hybctg15`, `hybosd15`, `hybahr15`
- **no `hybbgr*` entries**

Observed behavior from that stale result set:
- Some hybrid variants beat `ldpc100` at a few points,
- but only in a **very-high-FER regime**,
- so the wins are not persuasive as a practical replacement story.

## Why the stale runs are not convincing

1. **Receiver-7 was not actually validated** in the committed outputs.
2. The strongest stale hybrid path still used only **15 LDPC stage-1 iterations**.
3. The rescue stage was launched from only **one final snapshot**, which is often the worst place to try localized rescue.
4. The committed comparisons used **different decoder RNG streams**, so decoder-vs-decoder FER gaps include extra Monte Carlo noise.

## What this overlay fixes

- adds multi-snapshot rescue via `GRAND_RESCUE_SNAPSHOT_ITERS`,
- adds same-frame decoder comparisons via `PAIR_DECODER_STREAMS=1`,
- adds `RUN_RECEIVER1=0/1` so the dedicated winner run can focus on `ldpc100` vs `hybbgr100`,
- adds `run_ms_receiver7_winner.sbatch` with an aggressive Receiver-7 configuration aimed at best FER.
