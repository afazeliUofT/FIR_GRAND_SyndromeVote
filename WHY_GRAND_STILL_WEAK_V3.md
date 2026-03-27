# Why the current hybrid is still not convincingly beating legacy LDPC

The repo data now point to a more specific failure mode than the first Receiver-7 diagnosis.

## Strong clues from the committed repo

1. The `ldpc100` vs `hybbgr100` winner run improved FER only slightly, while paying a huge stage-2 cost.
2. The v2 localbias scout moved to a milder imperfect-CSI regime, but it still compared at `15` and `20` BP iterations and still kept Receiver-6 and Receiver-7 in the same sweep.
3. That means the comparison is still too deep into the BP regime. GRAND is being asked to fix the hardest residual tail instead of the earlier, more structured block-bias failures.

## Updated diagnosis

The residual failures are most likely **clustered reliability-bias failures**, not sparse independent bit errors:

- imperfect CSI injects subcarrier-block bias (`SIONNA_CSI_BLOCK_SC`),
- coarse pilot spacing spreads the bias over a wider region,
- by `15` to `20` BP iterations, much of the easy headroom is already gone,
- the remaining failures are not well matched to raw bit-flip or small-list GRAND search.

## Practical consequence

To give the hybrid a real chance to win, the job should do all of the following together:

1. compare at earlier BP iterations (`8`, `12`, maybe `15`),
2. use a channel/CSI regime that creates *blockwise* reliability distortion, not dense late-tail damage,
3. tune Receiver-7 toward block hypotheses rather than many scattered singleton hypotheses.

That is exactly what the new block-bias scout and winner jobs do.
