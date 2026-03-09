# Receiver 3+ tuning notes

## Default intent

The default sbatch file is tuned to maximize BER/BLER gain probability on the checked-in SIONNA_TDL setup, not strict equal-complexity fairness.

Main defaults:

- `RUN_RECEIVER3=1`
- `GRAND_PTG_LLR_SOURCE=mixed`
- `GRAND_PTG_CHECK_COVER_K=2`
- `GRAND_PTG_MAX_WEIGHT=8`
- `GRAND_PTG_MAX_PATTERNS=40000`
- `GRAND_PTG_BOOST_MAX_WEIGHT=8`
- `GRAND_PTG_BOOST_MAX_PATTERNS=250000`
- `GRAND_PTG_PEEL_RATIO=1.75`
- `GRAND_PTG_PEEL_MAX_BITS=48`
- `GRAND_PTG_PEEL_DENSE_MAX_VARS=28`
- `GRAND_PTG_PEEL_MAX_FREE_ENUM=12`
- `GRAND_PTG_PEEL_EXTRA_LLR_BITS=8`

## What to tune first

1. `GRAND_PTG_PEEL_MAX_BITS`
2. `GRAND_PTG_PEEL_DENSE_MAX_VARS`
3. `GRAND_PTG_PEEL_EXTRA_LLR_BITS`
4. `GRAND_PTG_MAX_PATTERNS`
5. `GRAND_PTG_BOOST_MAX_PATTERNS`

## Practical interpretation

- If BER/FER still barely moves, enlarge the peel candidate set first.
- If peel gets close but not enough, enlarge the exact-solve limit next.
- Only after that should you spend much more on pure GRAND enumeration.

## Safer ranges to try

- `GRAND_PTG_PEEL_MAX_BITS=56`
- `GRAND_PTG_PEEL_DENSE_MAX_VARS=32`
- `GRAND_PTG_PEEL_EXTRA_LLR_BITS=12`
- `GRAND_PTG_MAX_PATTERNS=60000`
- `GRAND_PTG_BOOST_MAX_PATTERNS=400000`

## One caution

A stronger local linear solve can occasionally find a valid but wrong nearby codeword if the local candidate set is too broad and too ambiguous.
That is why the default uses weighted nullspace enumeration and keeps the exact-solve dimension bounded.
