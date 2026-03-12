# Receiver 3+ drop-in guide

This package keeps the current outputs and adds a stronger hybrid family on top of your existing code:

- Existing legacy baselines remain: `ldpc4`, `ldpc8`, `ldpc15`, `ldpc20`, `ldpc100`
- Existing Receiver 1 hybrids remain: `hyb4`, `hyb8`, `hyb15`
- Existing Receiver 2 hybrids remain: `hybsv4`, `hybsv8`, `hybsv15`
- New stronger hybrids are added: `hybptg4`, `hybptg8`, `hybptg15`

`hybptg*` implements a stronger stage-2 path:

1. syndrome-vote + check-cover front-end  
2. larger peel candidate set `V_peel`  
3. peel reduction of degree-1 equations  
4. exact weighted GF(2) solve on the reduced local system when it is still small enough  
5. fallback to the existing GRAND engine if the pre-solver cannot certify a correction

## Files to replace

Replace these two files in the repo root:

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`

## Main code changes

### 1) `ClusterGrandConfig` gains Receiver-3 knobs

Added fields:

- `pre_solver_mode`
- `peel_candidate_ratio`
- `peel_max_bits`
- `peel_dense_max_vars`
- `peel_max_free_enum`
- `peel_extra_llr_bits`

### 2) New helper functions

Added:

- `_resolve_sort_llr_vector(...)`
- `_auto_pick_peel_candidate_size(...)`
- `_select_presolver_vars(...)`
- `_build_local_subsystem_for_candidate(...)`
- `_peel_reduce_system(...)`
- `_gf2_weighted_solve(...)`
- `_run_presolver_peel_gf2(...)`
- `run_local_rescue_with_optional_presolver(...)`

### 3) Existing GRAND engine is preserved

`run_local_grand_on_union_of_clusters(...)` is still the same main enumerative engine.
The only direct change inside it is support for `llr_source="mixed"`.

### 4) Hybrid stage-2 call sites now go through the wrapper

`run_hybrid_ldpc_grand_adaptive(...)` now calls:

- `run_local_rescue_with_optional_presolver(...)`

instead of calling the GRAND engine directly.

That means:

- Receiver 1 / Receiver 2 still behave as before if `pre_solver_mode="none"`
- Receiver 3+ first tries the pre-solver, then falls back to GRAND

### 5) Hardware timing still works

The existing stage-2 timing output is preserved.
The stage-2 cycle model now additionally charges:

- peel candidate reliability work
- peel edge work
- dense GF(2) XOR work

The existing summary CSV columns stay valid because peel cost is folded into the stage-2 / GRAND bucket.

### 6) New configs added

Added:

- `RUN_RECEIVER3`
- `GRAND_PTG_USE_BOOST`
- `grand_cfg_awgn_ptg`
- `grand_cfg_awgn_ptg_boost`

The new decoder labels are:

- `hybptg4`
- `hybptg8`
- `hybptg15`

## Why this is stronger than the current Receiver 2 code

The current Receiver 2 upgrade only changes candidate ranking before GRAND.
It still relies on bounded pattern enumeration.

This stronger drop-in adds a non-enumerative local solve before GRAND.
That matters most when the current runs already hit the GRAND pattern ceiling.
