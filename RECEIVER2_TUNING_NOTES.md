# Receiver 2 tuning notes

## What the new front-end does

Receiver 2 changes only the **search-set construction** before GRAND enumeration:

1. build the union of variables adjacent to unsatisfied checks
2. compute vote count `u_v`
3. compute `eta_v = u_v / (|LLR_v| + epsilon)`
4. seed the list with up to `k_cc` least reliable neighbours per unsatisfied check
5. fill the remaining budget using the global syndrome-vote ranking

The downstream pattern generation and syndrome testing are unchanged.

## Decoder labels

- `hyb4`, `hyb8`, `hyb15` = Receiver 1 (current baseline)
- `hybsv4`, `hybsv8`, `hybsv15` = Receiver 2 (new)

## Included defaults

The modified `run_ms.sbatch` uses these Receiver-2 defaults:

```bash
RUN_RECEIVER2=1
GRAND_SV_CHECK_COVER_K=1
GRAND_SV_EPSILON=1e-3
GRAND_SV_MAX_WEIGHT=7
GRAND_SV_MAX_PATTERNS=20000
GRAND_SV_BOOST_MAX_WEIGHT=7
GRAND_SV_BOOST_MAX_PATTERNS=120000
GRAND_SV_LLR_SOURCE=channel
```

These settings are meant to maximize the chance that Receiver 2 closes or beats the BER/BLER gap to strong legacy LDPC baselines.

## If you want the strongest Receiver-2 attempt first

Try the included sbatch file as-is.

If the resulting `hybsv*` curves are still not ahead of `ldpc20` / `ldpc100`, the first knobs to increase are:

```bash
GRAND_SV_MAX_PATTERNS
GRAND_SV_BOOST_MAX_PATTERNS
GRAND_SV_MAX_WEIGHT
GRAND_SV_BOOST_MAX_WEIGHT
```

in that order.

## If you want a fair Receiver-1 vs Receiver-2 comparison

Use matching budgets:

```bash
GRAND_SV_MAX_WEIGHT=$GRAND_MAX_WEIGHT
GRAND_SV_MAX_PATTERNS=$GRAND_MAX_PATTERNS
GRAND_SV_BOOST_MAX_WEIGHT=$GRAND_BOOST_MAX_WEIGHT
GRAND_SV_BOOST_MAX_PATTERNS=$GRAND_BOOST_MAX_PATTERNS
```

## Honest expectation

This patch gives Receiver 2 a **materially better front-end** than the current code.

It should improve performance per tested pattern, especially when the stage-1 failure is localized but the LLR-only ranking misses repeated offenders.

It still does **not** guarantee that the new hybrid will beat `ldpc20` or `ldpc100` under the current 5G NR + TDL setup. If you still do not get a clean win after this patch, the next levers are:

1. larger Receiver-2 budgets
2. snapshot choices centered on `8` and `15`
3. a paper-matched AWGN/BPSK experiment instead of the current TDL setup
4. Receiver 3 peel/inactivation before GRAND
