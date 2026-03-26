# Why Receiver-7 barely helped in `run_ms_receiver7_winner.sbatch`

## What the winner run actually showed

In the committed run, `hybbgr100` beat `ldpc100` at every tested SNR, but only very slightly:

- FER gain was only about **0.0007 to 0.0024 absolute**.
- That is only about **1.6% to 3.7% relative**.
- The hybrid paid for that with about **122x to 198x** the baseline decode time.

So the issue is **not** that the code is crashing or that Receiver-7 is completely broken.
The issue is that the **current operating point is wrong for GRAND-style rescue**.

## Diagnosis

The current run uses:

- `LDPC_ITERS=100`
- `STAGE1_ITERS=100`
- `PAIR_DECODER_STREAMS=1`
- `GRAND_RESCUE_SNAPSHOT_ITERS=4,8,12,15,20,40,60,80,100`
- `TDL=C`, delay spread `8e-7`, speed `[8,14]` m/s, `CFO=120`, `CSI_MODE=nr_imperfect`

This means the hybrid is being asked to rescue **only the tail of failures that remain after a full 100-iteration BP decode**.
That tail is usually **not sparse enough** for GRAND to shine.

The summary shows that stage-2 only improves its own stage-1 FER by roughly **2% to 3%**. That means only a small fraction of failed frames are still compatible with a small structured pattern search.

In other words, the residual failure mode here is mostly:

- wrong-BP-basin convergence,
- dense reliability bias from imperfect CSI,
- and difficult trapping-set style failures after BP already used many iterations.

That is a poor regime for a heavy stage-2 combinatorial search.

## Practical solution

To make the hybrid a convincing winner, do **not** compare against `ldpc100` in this regime.
Instead:

1. Compare at **earlier BP budgets**, e.g. 15 or 20 iterations.
2. Keep rescue snapshots only in the **early-to-mid** range: `4,8,12,15,20`.
3. Move to a **localized imperfect-CSI** regime rather than a globally hard tail.
4. Let Receiver-7 attack frames **before** BP has fully hardened into its wrong basin.

That is what the new Slurm files in this package do.

## New target regime

The new runs intentionally use a more local-bias channel-estimation stress pattern:

- lower Doppler than the previous winner run,
- coarser pilot stride,
- larger CSI block structure,
- moderate phase/amplitude distortion,
- and earlier hybrid entry.

This is still a Sionna-generated 5G-compatible imperfect-CSI scenario, but it is much more aligned with what local/basis GRAND can actually fix.
