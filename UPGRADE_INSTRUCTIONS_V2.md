1. From the repo root, overlay the package:

   unzip -o /path/to/FIR_GRAND_receiver7_localbias_v2.zip

2. Run the scout first:

   sbatch run_ms_receiver7_localbias_scout.sbatch

3. Inspect which decoder/iteration pair gives the biggest FER gap.

4. Then run the narrower winner job:

   sbatch run_ms_receiver7_localbias_winner.sbatch

## Why these new runs are different

The old winner run compared against `ldpc100`, which leaves only the hardest residual failures for stage-2.
That is the wrong place to demonstrate a large GRAND gain.

The new runs instead:

- compare at 15 or 20 BP iterations,
- rescue only from early/mid snapshots,
- and use a localized imperfect-CSI channel regime.

That gives the hybrid a much better chance to win **substantially** in FER.
