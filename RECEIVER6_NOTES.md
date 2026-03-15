# Receiver-6 tuning notes

## Main new knobs

Receiver-6 environment variables all start with `GRAND_AHR_`.

### Candidate-set growth
- `GRAND_AHR_RATIO` : how many soft-ranked variables to gather relative to the stage-2 search size `L`
- `GRAND_AHR_MAX_BITS` : cap on the soft candidate pool size
- `GRAND_AHR_CORE_MAX_BITS` : how many least-reliable bits are allowed in the core hypothesis engine
- `GRAND_AHR_CORE_MAX_WEIGHT` : maximum number of forced flips inside the core hypothesis engine
- `GRAND_AHR_MAX_CANDIDATES` : max number of soft hypotheses kept before restart/polish

### Scoring
- `GRAND_AHR_SAT_PENALTY` : weight on residual unsatisfied checks after a local hypothesis
- `GRAND_AHR_LLR_WEIGHT` : weight on reliability cost of the local hypothesis

### Restart strength
- `GRAND_AHR_RESTART_MAX_CANDIDATES` : how many strong local hypotheses trigger full-graph restarts
- `GRAND_AHR_RESTART_ITERS` : LDPC iterations used in each anchored restart
- `GRAND_AHR_RESTART_GAIN` : LLR magnitude injected on anchored bits
- `GRAND_AHR_RESTART_DUAL_GAIN` : stronger gain used for core bits / higher-confidence anchors
- `GRAND_AHR_RESTART_ABS_FLOOR` : minimum absolute LLR magnitude used for anchors
- `GRAND_AHR_RESTART_ALPHA` : damping factor in anchored restart mixing
- `GRAND_AHR_RESTART_ANCHOR_ALL` : when `1`, all selected bits can be anchored; when `0`, only the strongest subset is anchored

### Fallback search
- `GRAND_AHR_MAX_WEIGHT`, `GRAND_AHR_MAX_PATTERNS`, `GRAND_AHR_BOOST_MAX_WEIGHT`, `GRAND_AHR_BOOST_MAX_PATTERNS`
  control the final GRAND fallback budget.

## What to tweak first

1. If `hybahr15` still barely improves stage-1, increase:
   - `GRAND_AHR_RESTART_MAX_CANDIDATES`
   - `GRAND_AHR_RESTART_ITERS`
   - `GRAND_AHR_RESTART_GAIN`

2. If the restart looks too aggressive and BER worsens, decrease:
   - `GRAND_AHR_RESTART_GAIN`
   - `GRAND_AHR_RESTART_DUAL_GAIN`
   - `GRAND_AHR_RESTART_ANCHOR_ALL`

3. If the decoder is not finding enough distinct local hypotheses, increase:
   - `GRAND_AHR_RATIO`
   - `GRAND_AHR_MAX_BITS`
   - `GRAND_AHR_MAX_CANDIDATES`

4. If runtime explodes without meaningful rescue improvement, reduce:
   - `GRAND_AHR_MAX_CANDIDATES`
   - `GRAND_AHR_RESTART_MAX_CANDIDATES`
   - the GRAND fallback budgets

## Suggested experiment order

1. Run `run_ms_scout.sbatch` first to find the actual winning window.
2. Re-run `run_ms.sbatch` centered on the best 2–3 SNR points.
3. Only then use `run_ms_fair.sbatch` for the closer comparison.
