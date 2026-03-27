# Diagnosis behind the next upgrade

The committed results now give a much clearer story:

- Plain hybrid GRAND (`hyb*`) is weak because generic local-pattern search is not the main source of gain.
- Receiver-7 / basis-GRAND (`hybbgr*`) helps, but the localbias scout shows Receiver-6 / anchored restart (`hybahr*`) is consistently stronger.
- That means the dominant residual after shallow LDPC is not a sparse algebraic pattern; it is a **biased reliability pocket** caused by imperfect CSI, where a restart mechanism can re-enter a better BP basin.
- The old `ldpc100` vs `hybbgr100` comparison was the wrong operating point. By 100 iterations, BP has already consumed almost all easy gain.

So the next package does three concrete things:

1. adds **Receiver-8 / `hybmeta`**, which uses a stronger Receiver-6 style primary profile and then falls back to Receiver-7 only on the hard residue;
2. adds a **block-bias channel regime** where quasi-static CSI mismatch is stronger and more localized, which should favor the restart-based hybrid;
3. writes a new `*_summary_diagnostics.csv` file so the next run directly reports:
   - stage-2 invocation rate,
   - true-fix rate if invoked,
   - stage-1 error span and block concentration,
   - snapshot success histogram,
   - primary-vs-fallback success share.
