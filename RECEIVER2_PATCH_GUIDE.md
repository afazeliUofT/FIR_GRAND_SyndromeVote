# Receiver 2 drop-in guide: syndrome-vote + check-cover front-end

This package adds **Receiver 2** from the paper to the existing hybrid pipeline while keeping the current outputs intact:

- existing legacy baselines remain: `ldpc4`, `ldpc8`, `ldpc15`, `ldpc20`, `ldpc100`
- existing Receiver-1 hybrids remain: `hyb4`, `hyb8`, `hyb15`
- new Receiver-2 hybrids are added: `hybsv4`, `hybsv8`, `hybsv15`
- `summary.csv`, `summary_tails.csv`, raw pickle output, stage-1 timing, total timing, and per-frame GRAND logs are all preserved

## Fastest way to use this package

Unzip this package into the repo root and let these two files replace the originals:

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`

That is the cleanest path because the changes span multiple anchors.

---

## Exact anchor texts and what was changed

### File: `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`

### 1) Extend `ClusterGrandConfig`

**Exact anchor text to search:**

```python
max_syndrome_weight_for_grand: Optional[int] = None
```

**Drop-in added immediately after that anchor:**

```python

    # Receiver front-end selection
    selection_mode: str = "llr"            # "llr" (Receiver 1) or "syndrome_vote" (Receiver 2)
    sv_epsilon: float = 1e-3               # epsilon in eta_v = u_v / (rho_v + epsilon)
    sv_check_cover_k: int = 0              # k_cc ; 0 disables check-cover seeding
```

---

### 2) Insert Receiver-2 helper functions before the union-of-clusters GRAND function

**Exact anchor text to search:**

```python
### CELL number 27 ### updated
```

**Drop-in inserted immediately before that anchor:**

- `_auto_pick_grand_search_size(...)`
- `_select_search_vars_llr(...)`
- `_select_search_vars_syndrome_vote(...)`

This is the actual Receiver-2 front-end. It implements:

- vote count `u_v`
- score `eta_v = u_v / (rho_v + epsilon)`
- check-cover seeding with `k_cc`
- fill-to-budget using global syndrome-vote ranking

---

### 3) Replace the pure-LLR search-set builder inside `run_local_grand_on_union_of_clusters(...)`

**Exact anchor text to search:**

```python
# AUTO-pick L if max_bits_from_cluster is None
```

**What changed:**

The original LLR-only `search_vars` selection block was replaced with a front-end switch:

- Receiver 1: `selection_mode="llr"`
- Receiver 2: `selection_mode="syndrome_vote"`

The downstream pattern generator, batch evaluator, counters, and syndrome membership test are unchanged.

---

### 4) Attach front-end metadata to the GRAND result object

**Exact anchor text to search:**

```python
setattr(res, "llr_source_used", str(llr_source_used))
```

**Drop-in added right after that anchor in the final result block:**

```python
    setattr(res, "selection_mode_used", str(selection_mode_used))
    setattr(res, "sv_seeded_count", int(sv_seeded_count))
    setattr(res, "sv_neighbor_visits", int(sv_neighbor_visits))
    setattr(res, "sv_score_len", int(sv_score_len))
```

This keeps raw-pickle inspection possible without changing the existing CSV schema.

---

### 5) Add Receiver-2 GRAND configs

**Exact anchor text to search:**

```python
grand_cfg_awgn_boost = ClusterGrandConfig(
```

**What changed:**

Immediately after the baseline `grand_cfg_awgn` and `grand_cfg_awgn_boost`, the modified file adds:

- `RUN_RECEIVER2`
- `GRAND_SV_USE_BOOST`
- `grand_cfg_awgn_sv`
- `grand_cfg_awgn_sv_boost`

These are the Receiver-2 configs used by the new `hybsv*` decoders.

---

### 6) Make boost config explicit per hybrid run

**Exact anchor text to search:**

```python
def run_hybrid_ldpc_grand_adaptive(
```

**What changed:**

The function signature now accepts:

```python
grand_cfg_boost: Optional[ClusterGrandConfig] = None
```

This allows Receiver 1 and Receiver 2 to use different boost budgets while preserving the same logging and timing path.

---

### 7) Generalize the boost path

**Exact anchor text to search:**

```python
# If baseline GRAND exhausted its search and failed, optionally boost
```

**What changed:**

That block now uses the passed-in `grand_cfg_boost` instead of assuming the baseline boost config.

This is the key reason the same hybrid driver can run both:

- Receiver 1 (`hyb*`)
- Receiver 2 (`hybsv*`)

without duplicating the Monte-Carlo engine.

---

### 8) Add Receiver-2 scenarios to the sweep runner

**Exact anchor text to search:**

```python
# Scenario 3: Complete hybrid (Stage-1 LDPC + GRAND rescue)
```

**What changed:**

That existing Receiver-1 loop remains intact.

Immediately after it, the modified file adds a new loop gated by `RUN_RECEIVER2`:

- `hybsv4`
- `hybsv8`
- `hybsv15`

These run the same stage-1 LDPC, same snapshots, same logging, same save path, but with the Receiver-2 front-end.

---

### 9) Small hardware-model extension for Receiver 2

**Exact anchor text to search:**

```python
# (3) Sort union bits by |LLR| : O(n log2 n)
```

**What changed:**

That part of `grand_hw_cycles_from_result(...)` now treats the sort as a generic front-end ranking sort and adds a small extra arithmetic charge for syndrome-vote score formation.

The existing timing model is otherwise preserved.

---

## File: `run_ms.sbatch`

### 10) Add Receiver-2 environment knobs

**Exact anchor text to search:**

```bash
# ===== GRAND knobs =====
```

**What changed:**

The modified sbatch keeps the existing Receiver-1 exports and adds:

```bash
export RUN_RECEIVER2=${RUN_RECEIVER2:-1}
export GRAND_SV_LLR_SOURCE=${GRAND_SV_LLR_SOURCE:-${GRAND_LLR_SOURCE:-channel}}
export GRAND_SV_CHECK_COVER_K=${GRAND_SV_CHECK_COVER_K:-1}
export GRAND_SV_EPSILON=${GRAND_SV_EPSILON:-1e-3}
export GRAND_SV_MAX_WEIGHT=${GRAND_SV_MAX_WEIGHT:-7}
export GRAND_SV_MAX_PATTERNS=${GRAND_SV_MAX_PATTERNS:-20000}
export GRAND_SV_USE_BOOST=${GRAND_SV_USE_BOOST:-1}
export GRAND_SV_BOOST_MAX_WEIGHT=${GRAND_SV_BOOST_MAX_WEIGHT:-7}
export GRAND_SV_BOOST_MAX_PATTERNS=${GRAND_SV_BOOST_MAX_PATTERNS:-120000}
export GRAND_SV_BATCH_SIZE=${GRAND_SV_BATCH_SIZE:-${GRAND_BATCH_SIZE:-256}}
```

These defaults are intentionally more aggressive for Receiver 2 so the new variant has a real chance to beat the stronger legacy baselines in BER/BLER.

---

### 11) Add Receiver-2 logging echoes

**Exact anchor text to search:**

```bash
echo "[SBATCH CONFIG] TARGET_FRAME_ERRORS=${TARGET_FRAME_ERRORS}  MAX_FRAMES=${MAX_FRAMES}"
```

**Drop-in added immediately after that line:**

```bash
echo "[SBATCH CONFIG] RUN_RECEIVER2=${RUN_RECEIVER2}  GRAND_SV_CHECK_COVER_K=${GRAND_SV_CHECK_COVER_K}  GRAND_SV_EPSILON=${GRAND_SV_EPSILON}"
echo "[SBATCH CONFIG] GRAND_SV_MAX_WEIGHT=${GRAND_SV_MAX_WEIGHT}  GRAND_SV_MAX_PATTERNS=${GRAND_SV_MAX_PATTERNS}"
echo "[SBATCH CONFIG] GRAND_SV_BOOST_MAX_WEIGHT=${GRAND_SV_BOOST_MAX_WEIGHT}  GRAND_SV_BOOST_MAX_PATTERNS=${GRAND_SV_BOOST_MAX_PATTERNS}"
```

---

## New decoder labels in the result CSVs

After the patch, the result files will include all old decoders plus:

- `hybsv4`
- `hybsv8`
- `hybsv15`

These are Receiver 2 runs.

---

## Fair-vs-aggressive comparison note

The included sbatch defaults for Receiver 2 are **aggressive** on purpose:

- higher `GRAND_SV_MAX_WEIGHT`
- higher `GRAND_SV_MAX_PATTERNS`
- much larger boost budget

That is the right choice if your near-term goal is to maximize BER/BLER improvement.

If you want a strict same-budget Receiver-1 vs Receiver-2 comparison, set:

```bash
GRAND_SV_MAX_WEIGHT=$GRAND_MAX_WEIGHT
GRAND_SV_MAX_PATTERNS=$GRAND_MAX_PATTERNS
GRAND_SV_BOOST_MAX_WEIGHT=$GRAND_BOOST_MAX_WEIGHT
GRAND_SV_BOOST_MAX_PATTERNS=$GRAND_BOOST_MAX_PATTERNS
```

---

## Expected behavior after patch

- existing CSV/tails files still write in the same format
- existing baseline curves still appear
- Receiver 2 appears as additional hybrid curves
- the pattern engine, syndrome tester, batch-parallel evaluator, and hardware-time path remain the same
- only the stage-2 **front-end candidate selection** changes for `hybsv*`
