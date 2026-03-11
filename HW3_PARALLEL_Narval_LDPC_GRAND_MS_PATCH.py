"""
Receiver-4R drop-in for FIR_GRAND_SyndromeVote.

This file is a *full replacement* for HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py.
It loads the currently committed base script from the local git clone, then patches in
a stronger hybrid receiver:

Receiver 4R = Receiver 3+ exact rescue
            + multi-snapshot candidate mining
            + beam-restart LDPC re-decodes on biased LLRs
            + exact-success semantics preserved for the existing result pipeline.

Why this wrapper approach?
- It preserves all existing scenarios and outputs (ldpc*, hyb*, hybsv*, hybptg*).
- It adds new stronger scenarios hybrr4 / hybrr8 / hybrr15 without forcing you to
  paste the full 5k+ line base script again.
- It keeps the current CSV / tails / pickle / timing flow intact because the original
  result aggregation functions still run.

IMPORTANT:
- This file assumes the repo is a local git clone.
- It uses `git show` to read the currently committed base script from HEAD/origin.
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_WRAPPER_SENTINEL = "Receiver-4R drop-in for FIR_GRAND_SyndromeVote"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _this_relpath() -> str:
    return Path(__file__).name


def _run_git_show(refspec: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "show", refspec],
            cwd=str(_repo_root()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out
    except Exception:
        return None


def _load_base_source() -> Tuple[str, str]:
    relpath = _this_relpath()

    refs: List[str] = []
    env_ref = (os.environ.get("BASE_GIT_REF", "") or "").strip()
    if env_ref:
        refs.append(env_ref)

    refs.extend(
        [
            f"origin/main:{relpath}",
            f"HEAD:{relpath}",
            f"HEAD~1:{relpath}",
            f"main:{relpath}",
        ]
    )

    seen = set()
    for refspec in refs:
        if refspec in seen:
            continue
        seen.add(refspec)
        src = _run_git_show(refspec)
        if not src:
            continue
        if _WRAPPER_SENTINEL in src:
            continue
        if "def run_awgn_sweep_for_code" not in src:
            continue
        if "ClusterGrandConfig" not in src:
            continue
        return src, refspec

    raise RuntimeError(
        "Could not load the committed base script from git. "
        "Make sure this directory is a git clone and that the original "
        "HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py is still available in HEAD or origin/main."
    )


def _load_base_module() -> Tuple[types.ModuleType, str]:
    src, refspec = _load_base_source()
    mod = types.ModuleType("fir_grand_base")
    mod.__file__ = str(_repo_root() / _this_relpath())
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod, refspec


base, _BASE_REFSPEC = _load_base_module()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _stable_unique_keep_order(vals: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for x in vals:
        xi = int(x)
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
    return out


_ORIG_RUN_LOCAL_RESCUE = base.run_local_rescue_with_optional_presolver
_ORIG_GRAND_HW_CYCLES = base.grand_hw_cycles_from_result


def _resolve_restart_snapshot_list(
    requested_snapshot_iter: int,
    cfg: Any,
    frame: Any,
) -> List[int]:
    raw = getattr(cfg, "rescue_snapshot_list", None)
    vals: List[int] = []

    if raw is None:
        vals = [int(requested_snapshot_iter)]
    elif isinstance(raw, str):
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        try:
            vals = [int(x) for x in raw]
        except Exception:
            vals = [int(requested_snapshot_iter)]

    if int(requested_snapshot_iter) not in vals:
        vals = [int(requested_snapshot_iter)] + vals

    snaps = getattr(frame, "snapshots", {}) or {}
    llr_keys = set((snaps.get("llr", {}) or {}).keys())
    out = [it for it in _stable_unique_keep_order(vals) if int(it) in llr_keys]
    if not out:
        out = [int(requested_snapshot_iter)]
    return out


def _resolve_restart_seed_llr(
    llr_snapshot: np.ndarray,
    llr_channel: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
    mode_l = (mode or "mixed").strip().lower()

    if mode_l == "posterior":
        return llr_snapshot.copy()

    if mode_l == "channel":
        if llr_channel is not None:
            return np.asarray(llr_channel, dtype=np.float32).copy()
        return llr_snapshot.copy()

    if mode_l in ("mixed", "hybrid", "minabs"):
        if llr_channel is None:
            return llr_snapshot.copy()
        ch = np.asarray(llr_channel, dtype=np.float32)
        abs_mix = np.minimum(np.abs(llr_snapshot), np.abs(ch)).astype(np.float32, copy=False)
        sign_ref = np.sign(llr_snapshot).astype(np.float32, copy=False)
        zero_mask = (sign_ref == 0)
        if np.any(zero_mask):
            sign_ref = sign_ref.copy()
            sign_ref[zero_mask] = np.sign(ch[zero_mask]).astype(np.float32, copy=False)
            zero_mask = (sign_ref == 0)
            if np.any(zero_mask):
                sign_ref[zero_mask] = 1.0
        return (sign_ref * abs_mix).astype(np.float32, copy=False)

    if mode_l in ("channel_plus_posterior", "sum"):
        if llr_channel is None:
            return llr_snapshot.copy()
        ch = np.asarray(llr_channel, dtype=np.float32)
        return (ch + llr_snapshot).astype(np.float32, copy=False)

    return llr_snapshot.copy()


def _apply_restart_bias(
    base_llr: np.ndarray,
    hard_bits_snapshot: np.ndarray,
    flip_vars: Sequence[int],
    alpha: float,
    bias: float,
    force_abs: float,
) -> Tuple[np.ndarray, int]:
    out = np.asarray(base_llr, dtype=np.float32).copy()
    hard = np.asarray(hard_bits_snapshot, dtype=np.uint8)

    abs0 = np.abs(out).astype(np.float32, copy=False)
    bias_ops = 0

    for v in flip_vars:
        vi = int(v)
        desired_bit = 1 - int(hard[vi])
        desired_sign = 1.0 if desired_bit == 0 else -1.0
        mag = max(float(force_abs), float(alpha) * float(abs0[vi]) + float(bias))
        out[vi] = np.float32(desired_sign * mag)
        bias_ops += 3

    return out, int(bias_ops)


def _candidate_local_scores(
    candidate_vars: np.ndarray,
    unsat_checks: np.ndarray,
    code_cfg: Any,
    llr_for_sort: np.ndarray,
) -> Dict[int, Tuple[int, float]]:
    unsat_set = set(int(j) for j in np.asarray(unsat_checks, dtype=np.int32).tolist())
    out: Dict[int, Tuple[int, float]] = {}
    llr_abs = np.abs(np.asarray(llr_for_sort, dtype=np.float32))
    for v in np.asarray(candidate_vars, dtype=np.int32).tolist():
        vi = int(v)
        checks = code_cfg.vars_to_checks[vi]
        vote = 0
        for j in checks:
            if int(j) in unsat_set:
                vote += 1
        out[vi] = (int(vote), float(llr_abs[vi]))
    return out


def _build_restart_patterns_for_snapshot(
    frame: Any,
    sim_cfg: Any,
    snapshot_iter: int,
    cfg: Any,
) -> Dict[str, Any]:
    snaps = frame.snapshots
    llr_snaps = snaps.get("llr", {}) or {}
    syn_snaps = snaps.get("syndrome", {}) or {}
    hard_snaps = snaps.get("hard_bits", {}) or {}

    if snapshot_iter not in llr_snaps or snapshot_iter not in syn_snaps or snapshot_iter not in hard_snaps:
        return {
            "snapshot_iter": int(snapshot_iter),
            "patterns": [],
            "hard_bits_snapshot": None,
            "llr_snapshot": None,
            "syndrome": None,
            "union_size": 0,
            "search_size": 0,
            "llr_sort_len": 0,
            "selection_mode_used": str(getattr(cfg, "selection_mode", "llr")),
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": 0,
        }

    syndrome = np.asarray(syn_snaps[snapshot_iter], dtype=np.uint8)
    hard_bits_snapshot = np.asarray(hard_snaps[snapshot_iter], dtype=np.uint8)
    llr_snapshot = np.asarray(llr_snaps[snapshot_iter], dtype=np.float32)

    initial_syndrome_weight = int(syndrome.sum())
    if initial_syndrome_weight <= 0:
        return {
            "snapshot_iter": int(snapshot_iter),
            "patterns": [],
            "hard_bits_snapshot": hard_bits_snapshot,
            "llr_snapshot": llr_snapshot,
            "syndrome": syndrome,
            "union_size": 0,
            "search_size": 0,
            "llr_sort_len": 0,
            "selection_mode_used": str(getattr(cfg, "selection_mode", "llr")),
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": 0,
        }

    llr_for_sort, _ = base._resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = base.build_allowed_mask_from_config(frame, sim_cfg, int(snapshot_iter), cfg)
    clusters = base.find_variable_clusters_from_syndrome(syndrome, sim_cfg.code)

    if not clusters:
        return {
            "snapshot_iter": int(snapshot_iter),
            "patterns": [],
            "hard_bits_snapshot": hard_bits_snapshot,
            "llr_snapshot": llr_snapshot,
            "syndrome": syndrome,
            "union_size": 0,
            "search_size": 0,
            "llr_sort_len": 0,
            "selection_mode_used": str(getattr(cfg, "selection_mode", "llr")),
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": 0,
        }

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full <= 0:
        return {
            "snapshot_iter": int(snapshot_iter),
            "patterns": [],
            "hard_bits_snapshot": hard_bits_snapshot,
            "llr_snapshot": llr_snapshot,
            "syndrome": syndrome,
            "union_size": 0,
            "search_size": 0,
            "llr_sort_len": 0,
            "selection_mode_used": str(getattr(cfg, "selection_mode", "llr")),
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": 0,
        }

    L_search = int(getattr(cfg, "restart_candidate_pool_size", 0) or 0)
    if L_search <= 0:
        L_grand = int(base._auto_pick_grand_search_size(L_full, cfg))
        L_search = int(base._auto_pick_peel_candidate_size(L_full, L_grand, cfg))
    L_search = max(1, min(L_search, L_full))

    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    if hasattr(base, "_select_presolver_vars"):
        candidate_vars, meta = base._select_presolver_vars(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=sim_cfg.code,
            llr_for_sort=llr_for_sort,
            L_peel=L_search,
            cfg=cfg,
        )
    else:
        selection_mode = str(getattr(cfg, "selection_mode", "llr") or "llr").strip().lower()
        if selection_mode in ("syndrome_vote", "sv", "receiver2"):
            candidate_vars, meta = base._select_search_vars_syndrome_vote(
                union_vars=union_vars,
                unsat_checks=unsat_checks,
                code_cfg=sim_cfg.code,
                llr_for_sort=llr_for_sort,
                L=L_search,
                cfg=cfg,
            )
        else:
            candidate_vars, meta = base._select_search_vars_llr(
                union_vars=union_vars,
                llr_for_sort=llr_for_sort,
                L=L_search,
            )

    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    if candidate_vars.size <= 0:
        return {
            "snapshot_iter": int(snapshot_iter),
            "patterns": [],
            "hard_bits_snapshot": hard_bits_snapshot,
            "llr_snapshot": llr_snapshot,
            "syndrome": syndrome,
            "union_size": int(L_full),
            "search_size": 0,
            "llr_sort_len": int(L_full),
            "selection_mode_used": str(meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))),
            "sv_seeded_count": int(meta.get("sv_seeded_count", 0)),
            "sv_neighbor_visits": int(meta.get("sv_neighbor_visits", 0)),
            "sv_score_len": int(meta.get("sv_score_len", L_full)),
        }

    scores = _candidate_local_scores(
        candidate_vars=candidate_vars,
        unsat_checks=unsat_checks,
        code_cfg=sim_cfg.code,
        llr_for_sort=llr_for_sort,
    )

    B = int(getattr(cfg, "restart_top_bits", 0) or 0)
    if B <= 0:
        B = min(10, int(candidate_vars.size))
    B = max(1, min(B, int(candidate_vars.size)))

    ranked_bits = sorted(
        [int(v) for v in candidate_vars.tolist()],
        key=lambda v: (
            -int(scores.get(v, (0, 0.0))[0]),
            float(scores.get(v, (0, 0.0))[1]),
            int(v),
        ),
    )[:B]

    max_order = int(getattr(cfg, "restart_max_order", 2) or 2)
    max_order = max(1, min(max_order, len(ranked_bits)))

    per_snapshot_cap = int(getattr(cfg, "restart_max_candidates_per_snapshot", 0) or 0)
    if per_snapshot_cap <= 0:
        per_snapshot_cap = max(4, int(getattr(cfg, "restart_max_candidates", 8) or 8))

    pattern_items: List[Tuple[Tuple[Any, ...], Tuple[int, ...]]] = []

    for w in range(1, max_order + 1):
        for comb in itertools.combinations(ranked_bits, w):
            touched_checks = set()
            total_vote = 0
            llr_cost = 0.0
            for v in comb:
                vote, abs_llr = scores.get(int(v), (0, 0.0))
                total_vote += int(vote)
                llr_cost += float(abs_llr)
                for j in sim_cfg.code.vars_to_checks[int(v)]:
                    if int(syndrome[int(j)]) == 1:
                        touched_checks.add(int(j))
            key = (
                -len(touched_checks),
                -int(total_vote),
                float(llr_cost),
                int(w),
                tuple(int(v) for v in comb),
            )
            pattern_items.append((key, tuple(int(v) for v in comb)))

    pattern_items.sort(key=lambda t: t[0])
    patterns = [tuple(item[1]) for item in pattern_items[:per_snapshot_cap]]

    return {
        "snapshot_iter": int(snapshot_iter),
        "patterns": patterns,
        "hard_bits_snapshot": hard_bits_snapshot,
        "llr_snapshot": llr_snapshot,
        "syndrome": syndrome,
        "union_size": int(L_full),
        "search_size": int(candidate_vars.size),
        "llr_sort_len": int(L_full),
        "selection_mode_used": str(meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))),
        "sv_seeded_count": int(meta.get("sv_seeded_count", 0)),
        "sv_neighbor_visits": int(meta.get("sv_neighbor_visits", 0)),
        "sv_score_len": int(meta.get("sv_score_len", L_full)),
    }


def _ldpc_restart_cycles_from_decode(
    iter_used: int,
    syndrome: np.ndarray,
    code_cfg: Any,
    hw_model: Any,
    early_stop: bool = True,
) -> int:
    final_vn2cn_executed = bool((not early_stop) or (int(np.asarray(syndrome).sum()) != 0))
    return int(
        base.ldpc_hw_cycles_frame(
            int(iter_used),
            code_cfg,
            hw_model,
            final_vn2cn_executed=final_vn2cn_executed,
        )
    )


def _make_restart_decoder_cfg(stage1_max_iters: int) -> Any:
    extra_iters = int(_env_int("GRAND_RR_EXTRA_LDPC_ITERS", 8))
    total_iters = int(max(1, stage1_max_iters + extra_iters))
    alpha = float(_env_float("GRAND_RR_LDPC_ALPHA", 0.8))
    return base.DecoderConfig(max_iters=total_iters, alpha=alpha, early_stop=True)


def _synthesize_success_result_from_restart(
    frame: Any,
    sim_cfg: Any,
    stage_snapshot_iter: int,
    cfg: Any,
    exact_fail_result: Any,
    chosen_snapshot_iter: int,
    hard_bits_after: np.ndarray,
    syndrome_after: np.ndarray,
    restart_meta: Dict[str, Any],
) -> Any:
    if exact_fail_result is None:
        exact_fail_result = types.SimpleNamespace()

    stage_hard = np.asarray(frame.snapshots["hard_bits"][int(stage_snapshot_iter)], dtype=np.uint8)
    stage_syndrome = np.asarray(frame.snapshots["syndrome"][int(stage_snapshot_iter)], dtype=np.uint8)

    flipped_vs_stage = np.flatnonzero(hard_bits_after != stage_hard).astype(np.int32)
    final_bit_errors = int(np.count_nonzero(hard_bits_after != np.asarray(frame.c_bits, dtype=np.uint8)))

    res = base.ClusterGrandResult(
        success=bool(int(np.asarray(syndrome_after).sum()) == 0),
        pattern_weight=int(flipped_vs_stage.size),
        flipped_vars=flipped_vs_stage,
        patterns_tested=int(getattr(exact_fail_result, "patterns_tested", 0)) + int(restart_meta["candidates_tried"]),
        initial_syndrome_weight=int(stage_syndrome.sum()),
        final_syndrome_weight=int(np.asarray(syndrome_after).sum()),
        initial_bit_errors=int(np.count_nonzero(stage_hard != np.asarray(frame.c_bits, dtype=np.uint8))),
        final_bit_errors=int(final_bit_errors),
        total_v2c_edge_visits=int(getattr(exact_fail_result, "total_v2c_edge_visits", 0)),
        total_unique_checks_visited=int(getattr(exact_fail_result, "total_unique_checks_visited", 0)),
        total_unique_checks_toggled=int(getattr(exact_fail_result, "total_unique_checks_toggled", 0)),
        patterns_generated=int(getattr(exact_fail_result, "patterns_generated", 0)) + int(restart_meta["patterns_generated"]),
    )

    for attr, default in [
        ("patterns_evaluated", int(getattr(exact_fail_result, "patterns_evaluated", getattr(exact_fail_result, "patterns_tested", 0)))),
        ("total_v2c_edge_visits_evaluated", int(getattr(exact_fail_result, "total_v2c_edge_visits_evaluated", getattr(exact_fail_result, "total_v2c_edge_visits", 0)))),
        ("total_unique_checks_visited_evaluated", int(getattr(exact_fail_result, "total_unique_checks_visited_evaluated", getattr(exact_fail_result, "total_unique_checks_visited", 0)))),
        ("total_unique_checks_toggled_evaluated", int(getattr(exact_fail_result, "total_unique_checks_toggled_evaluated", getattr(exact_fail_result, "total_unique_checks_toggled", 0)))),
        ("union_size", int(getattr(exact_fail_result, "union_size", restart_meta.get("union_size", 0)))),
        ("search_size", int(getattr(exact_fail_result, "search_size", restart_meta.get("search_size", 0)))),
        ("llr_sort_len", int(getattr(exact_fail_result, "llr_sort_len", restart_meta.get("llr_sort_len", 0)))),
        ("sum_pattern_weights_generated", int(getattr(exact_fail_result, "sum_pattern_weights_generated", 0))),
        ("cluster_unsat_edges", int(getattr(exact_fail_result, "cluster_unsat_edges", 0))),
        ("cluster_pair_edges", int(getattr(exact_fail_result, "cluster_pair_edges", 0))),
        ("num_batches_evaluated", int(getattr(exact_fail_result, "num_batches_evaluated", 0))),
        ("positions_packed_evaluated", int(getattr(exact_fail_result, "positions_packed_evaluated", 0))),
        ("batch_size_used", int(getattr(exact_fail_result, "batch_size_used", 0))),
        ("llr_source_used", str(getattr(exact_fail_result, "llr_source_used", getattr(cfg, "llr_source", "posterior")))),
        ("selection_mode_used", str(getattr(exact_fail_result, "selection_mode_used", getattr(cfg, "selection_mode", "llr")))),
        ("sv_seeded_count", int(getattr(exact_fail_result, "sv_seeded_count", 0))),
        ("sv_neighbor_visits", int(getattr(exact_fail_result, "sv_neighbor_visits", 0))),
        ("sv_score_len", int(getattr(exact_fail_result, "sv_score_len", 0))),
        ("pre_solver_attempted", int(getattr(exact_fail_result, "pre_solver_attempted", 1))),
        ("pre_solver_success", int(getattr(exact_fail_result, "pre_solver_success", 0))),
        ("peel_candidate_size", int(getattr(exact_fail_result, "peel_candidate_size", 0))),
        ("peel_residual_vars", int(getattr(exact_fail_result, "peel_residual_vars", 0))),
        ("peel_residual_rows", int(getattr(exact_fail_result, "peel_residual_rows", 0))),
        ("peel_edge_work", int(getattr(exact_fail_result, "peel_edge_work", 0))),
        ("peel_dense_xor_ops", int(getattr(exact_fail_result, "peel_dense_xor_ops", 0))),
        ("peel_free_dim", int(getattr(exact_fail_result, "peel_free_dim", 0))),
        ("peel_extra_llr_added", int(getattr(exact_fail_result, "peel_extra_llr_added", 0))),
        ("pre_solver_mode_used", str(getattr(exact_fail_result, "pre_solver_mode_used", "peel_gf2_restart"))),
    ]:
        setattr(res, attr, default)

    setattr(res, "patterns_evaluated", int(getattr(res, "patterns_evaluated", 0)) + int(restart_meta["candidates_tried"]))
    setattr(res, "num_batches_evaluated", int(getattr(res, "num_batches_evaluated", 0)) + int(restart_meta["candidates_tried"]))
    setattr(
        res,
        "sum_pattern_weights_generated",
        int(getattr(res, "sum_pattern_weights_generated", 0)) + int(restart_meta["sum_candidate_orders"]),
    )
    setattr(
        res,
        "positions_packed_evaluated",
        int(getattr(res, "positions_packed_evaluated", 0)) + int(restart_meta["sum_candidate_orders"]),
    )

    setattr(res, "restart_attempted", 1)
    setattr(res, "restart_success", 1)
    setattr(res, "restart_candidates_tried", int(restart_meta["candidates_tried"]))
    setattr(res, "restart_patterns_generated", int(restart_meta["patterns_generated"]))
    setattr(res, "restart_snapshot_used", int(chosen_snapshot_iter))
    setattr(res, "restart_hw_cycles_ldpc", int(restart_meta["restart_hw_cycles_ldpc"]))
    setattr(res, "restart_bias_operations", int(restart_meta["restart_bias_operations"]))
    setattr(res, "restart_extra_iters_used", int(restart_meta["restart_extra_iters_used"]))
    setattr(res, "restart_best_unsat_weight", 0)
    return res


def _run_restart_beam_after_exact_failure(
    frame: Any,
    sim_cfg: Any,
    stage_snapshot_iter: int,
    cfg: Any,
    exact_fail_result: Any,
) -> Any:
    snapshot_list = _resolve_restart_snapshot_list(stage_snapshot_iter, cfg, frame)
    stage1_max_iters = int(stage_snapshot_iter)
    dec_cfg_restart = _make_restart_decoder_cfg(stage1_max_iters)

    max_candidates_total = int(getattr(cfg, "restart_max_candidates", 8) or 8)
    max_candidates_total = max(1, max_candidates_total)

    restart_seed_llr = str(getattr(cfg, "restart_seed_llr", "mixed") or "mixed")
    restart_bias_alpha = float(getattr(cfg, "restart_bias_alpha", 1.35) or 1.35)
    restart_bias_abs = float(getattr(cfg, "restart_force_abs", 8.0) or 8.0)
    restart_bias_offset = float(getattr(cfg, "restart_bias_offset", 0.75) or 0.75)

    hw_model = getattr(base, "HW_MODEL", None)
    if hw_model is None:
        hw_model = base.load_hw_model_from_env()

    candidate_items: List[Dict[str, Any]] = []

    for priority, snap_it in enumerate(snapshot_list):
        built = _build_restart_patterns_for_snapshot(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=int(snap_it),
            cfg=cfg,
        )
        patterns = list(built.get("patterns", []))
        for patt in patterns:
            candidate_items.append(
                {
                    "priority": int(priority),
                    "snapshot_iter": int(snap_it),
                    "pattern": tuple(int(v) for v in patt),
                    "built": built,
                }
            )

    if not candidate_items:
        if exact_fail_result is not None:
            setattr(exact_fail_result, "restart_attempted", 1)
            setattr(exact_fail_result, "restart_success", 0)
            setattr(exact_fail_result, "restart_candidates_tried", 0)
            setattr(exact_fail_result, "restart_hw_cycles_ldpc", 0)
        return exact_fail_result

    candidate_items = candidate_items[:max_candidates_total]

    patterns_generated = len(candidate_items)
    candidates_tried = 0
    sum_candidate_orders = 0
    restart_hw_cycles_ldpc = 0
    restart_bias_operations = 0
    best_unsat_weight = math.inf
    best_llr_cost = math.inf

    for item in candidate_items:
        built = item["built"]
        snap_it = int(item["snapshot_iter"])
        pattern = tuple(int(v) for v in item["pattern"])
        hard_bits_snapshot = built["hard_bits_snapshot"]
        llr_snapshot = built["llr_snapshot"]

        if hard_bits_snapshot is None or llr_snapshot is None:
            continue

        seed_llr = _resolve_restart_seed_llr(
            llr_snapshot=llr_snapshot,
            llr_channel=getattr(frame, "llr_channel", None),
            mode=restart_seed_llr,
        )
        biased_llr, bias_ops = _apply_restart_bias(
            base_llr=seed_llr,
            hard_bits_snapshot=hard_bits_snapshot,
            flip_vars=pattern,
            alpha=restart_bias_alpha,
            bias=restart_bias_offset,
            force_abs=restart_bias_abs,
        )

        hard_bits_after, llr_post_after, syndrome_after, iter_used = base.ldpc_min_sum_decode(
            biased_llr,
            sim_cfg.code,
            dec_cfg_restart,
            snapshot_iters=[],
            snapshots={},
        )

        _ = llr_post_after
        syn_w_after = int(np.asarray(syndrome_after).sum())
        llr_cost = float(np.abs(biased_llr[np.asarray(pattern, dtype=np.int32)]).sum()) if pattern else 0.0

        restart_hw_cycles_ldpc += _ldpc_restart_cycles_from_decode(
            iter_used=int(iter_used),
            syndrome=syndrome_after,
            code_cfg=sim_cfg.code,
            hw_model=hw_model,
            early_stop=bool(dec_cfg_restart.early_stop),
        )
        restart_bias_operations += int(bias_ops)
        candidates_tried += 1
        sum_candidate_orders += int(len(pattern))

        if syn_w_after < best_unsat_weight or (syn_w_after == best_unsat_weight and llr_cost < best_llr_cost):
            best_unsat_weight = syn_w_after
            best_llr_cost = llr_cost

        if syn_w_after == 0:
            restart_meta = {
                "candidates_tried": int(candidates_tried),
                "patterns_generated": int(patterns_generated),
                "sum_candidate_orders": int(sum_candidate_orders),
                "restart_hw_cycles_ldpc": int(restart_hw_cycles_ldpc),
                "restart_bias_operations": int(restart_bias_operations),
                "restart_extra_iters_used": int(iter_used),
                "union_size": int(built.get("union_size", 0)),
                "search_size": int(built.get("search_size", 0)),
                "llr_sort_len": int(built.get("llr_sort_len", 0)),
            }
            return _synthesize_success_result_from_restart(
                frame=frame,
                sim_cfg=sim_cfg,
                stage_snapshot_iter=int(stage_snapshot_iter),
                cfg=cfg,
                exact_fail_result=exact_fail_result,
                chosen_snapshot_iter=int(snap_it),
                hard_bits_after=np.asarray(hard_bits_after, dtype=np.uint8),
                syndrome_after=np.asarray(syndrome_after, dtype=np.uint8),
                restart_meta=restart_meta,
            )

    if exact_fail_result is not None:
        setattr(exact_fail_result, "restart_attempted", 1)
        setattr(exact_fail_result, "restart_success", 0)
        setattr(exact_fail_result, "restart_candidates_tried", int(candidates_tried))
        setattr(exact_fail_result, "restart_patterns_generated", int(patterns_generated))
        setattr(exact_fail_result, "restart_hw_cycles_ldpc", int(restart_hw_cycles_ldpc))
        setattr(exact_fail_result, "restart_bias_operations", int(restart_bias_operations))
        setattr(exact_fail_result, "restart_best_unsat_weight", int(best_unsat_weight) if math.isfinite(best_unsat_weight) else -1)
        setattr(exact_fail_result, "restart_extra_iters_used", int(dec_cfg_restart.max_iters))
    return exact_fail_result


def _cfg_clone_with_exact_presolver(cfg: Any) -> Any:
    new_cfg = copy.copy(cfg)
    mode = str(getattr(cfg, "pre_solver_mode", "none") or "none").strip().lower()
    if mode in ("peel_gf2_restart", "receiver4", "r4", "restart_beam", "beam_restart", "receiver4r", "rr"):
        setattr(new_cfg, "pre_solver_mode", "peel_gf2")
    return new_cfg


def run_local_rescue_with_optional_presolver_receiver4(
    frame: Any,
    sim_cfg: Any,
    snapshot_iter: int,
    cfg: Any,
) -> Any:
    mode = str(getattr(cfg, "pre_solver_mode", "none") or "none").strip().lower()

    if mode not in ("peel_gf2_restart", "receiver4", "r4", "restart_beam", "beam_restart", "receiver4r", "rr"):
        return _ORIG_RUN_LOCAL_RESCUE(frame=frame, sim_cfg=sim_cfg, snapshot_iter=snapshot_iter, cfg=cfg)

    cfg_exact = _cfg_clone_with_exact_presolver(cfg)
    res_exact = _ORIG_RUN_LOCAL_RESCUE(frame=frame, sim_cfg=sim_cfg, snapshot_iter=snapshot_iter, cfg=cfg_exact)

    if res_exact is not None and bool(getattr(res_exact, "success", False)):
        setattr(res_exact, "restart_attempted", 0)
        setattr(res_exact, "restart_success", 0)
        setattr(res_exact, "restart_candidates_tried", 0)
        setattr(res_exact, "restart_hw_cycles_ldpc", 0)
        setattr(res_exact, "pre_solver_mode_used", "peel_gf2_restart")
        return res_exact

    res_restart = _run_restart_beam_after_exact_failure(
        frame=frame,
        sim_cfg=sim_cfg,
        stage_snapshot_iter=int(snapshot_iter),
        cfg=cfg,
        exact_fail_result=res_exact,
    )
    if res_restart is not None:
        setattr(res_restart, "pre_solver_mode_used", "peel_gf2_restart")
    return res_restart


def grand_hw_cycles_from_result_receiver4(
    result: Any,
    sim_cfg: Any,
    hw: Any,
) -> int:
    cycles = int(_ORIG_GRAND_HW_CYCLES(result, sim_cfg, hw))
    cycles += int(getattr(result, "restart_hw_cycles_ldpc", 0))
    return int(cycles)


def _cluster_cfg_from_env(prefix: str, defaults: Dict[str, Any], *, pre_solver_mode: str) -> Any:
    cfg = base.ClusterGrandConfig(
        max_weight=int(_env_int(f"{prefix}_MAX_WEIGHT", int(defaults["max_weight"]))),
        max_patterns=int(_env_int(f"{prefix}_MAX_PATTERNS", int(defaults["max_patterns"]))),
        max_bits_from_cluster=None,
        verbose=False,
        llr_source=str(os.environ.get(f"{prefix}_LLR_SOURCE", str(defaults["llr_source"]))).strip().lower(),
        pattern_overgen_ratio=float(_env_float(f"{prefix}_OVERGEN", float(defaults["pattern_overgen_ratio"]))),
        max_syndrome_weight_for_grand=None,
        batch_size=int(_env_int(f"{prefix}_BATCH_SIZE", int(defaults["batch_size"]))),
        selection_mode=str(os.environ.get(f"{prefix}_SELECTION_MODE", str(defaults["selection_mode"]))).strip().lower(),
        sv_epsilon=float(_env_float(f"{prefix}_EPSILON", float(defaults["sv_epsilon"]))),
        sv_check_cover_k=int(_env_int(f"{prefix}_CHECK_COVER_K", int(defaults["sv_check_cover_k"]))),
        pre_solver_mode=str(pre_solver_mode),
        peel_candidate_ratio=float(_env_float(f"{prefix}_PEEL_RATIO", float(defaults["peel_candidate_ratio"]))),
        peel_max_bits=int(_env_int(f"{prefix}_PEEL_MAX_BITS", int(defaults["peel_max_bits"]))),
        peel_dense_max_vars=int(_env_int(f"{prefix}_PEEL_DENSE_MAX_VARS", int(defaults["peel_dense_max_vars"]))),
        peel_max_free_enum=int(_env_int(f"{prefix}_PEEL_MAX_FREE_ENUM", int(defaults["peel_max_free_enum"]))),
        peel_extra_llr_bits=int(_env_int(f"{prefix}_PEEL_EXTRA_LLR_BITS", int(defaults["peel_extra_llr_bits"]))),
    )

    setattr(cfg, "rescue_snapshot_list", str(os.environ.get(f"{prefix}_SNAPSHOTS", str(defaults["rescue_snapshot_list"]))).strip())
    setattr(cfg, "restart_seed_llr", str(os.environ.get(f"{prefix}_RESTART_SEED_LLR", str(defaults["restart_seed_llr"]))).strip().lower())
    setattr(cfg, "restart_top_bits", int(_env_int(f"{prefix}_RESTART_TOP_BITS", int(defaults["restart_top_bits"]))))
    setattr(cfg, "restart_max_order", int(_env_int(f"{prefix}_RESTART_MAX_ORDER", int(defaults["restart_max_order"]))))
    setattr(cfg, "restart_max_candidates", int(_env_int(f"{prefix}_RESTART_MAX_CANDIDATES", int(defaults["restart_max_candidates"]))))
    setattr(cfg, "restart_max_candidates_per_snapshot", int(_env_int(f"{prefix}_RESTART_MAX_CANDIDATES_PER_SNAPSHOT", int(defaults["restart_max_candidates_per_snapshot"]))))
    setattr(cfg, "restart_candidate_pool_size", int(_env_int(f"{prefix}_RESTART_CANDIDATE_POOL_SIZE", int(defaults["restart_candidate_pool_size"]))))
    setattr(cfg, "restart_bias_alpha", float(_env_float(f"{prefix}_RESTART_BIAS_ALPHA", float(defaults["restart_bias_alpha"]))))
    setattr(cfg, "restart_bias_offset", float(_env_float(f"{prefix}_RESTART_BIAS_OFFSET", float(defaults["restart_bias_offset"]))))
    setattr(cfg, "restart_force_abs", float(_env_float(f"{prefix}_RESTART_FORCE_ABS", float(defaults["restart_force_abs"]))))
    return cfg


def _receiver4_defaults() -> Dict[str, Any]:
    return {
        "max_weight": 8,
        "max_patterns": 40000,
        "llr_source": "mixed",
        "pattern_overgen_ratio": 1.02,
        "batch_size": 256,
        "selection_mode": "syndrome_vote",
        "sv_epsilon": 1e-3,
        "sv_check_cover_k": 2,
        "peel_candidate_ratio": 1.75,
        "peel_max_bits": 56,
        "peel_dense_max_vars": 32,
        "peel_max_free_enum": 12,
        "peel_extra_llr_bits": 8,
        "rescue_snapshot_list": "8,15,4",
        "restart_seed_llr": "mixed",
        "restart_top_bits": 10,
        "restart_max_order": 3,
        "restart_max_candidates": 8,
        "restart_max_candidates_per_snapshot": 4,
        "restart_candidate_pool_size": 24,
        "restart_bias_alpha": 1.35,
        "restart_bias_offset": 0.75,
        "restart_force_abs": 8.0,
    }


def _receiver4_boost_defaults() -> Dict[str, Any]:
    d = _receiver4_defaults().copy()
    d["max_patterns"] = 180000
    d["restart_max_candidates"] = 12
    d["restart_max_candidates_per_snapshot"] = 6
    d["restart_candidate_pool_size"] = 32
    d["restart_force_abs"] = 10.0
    return d


base.RUN_RECEIVER4 = bool(_env_int("RUN_RECEIVER4", 1))
base.GRAND_RR_USE_BOOST = bool(_env_int("GRAND_RR_USE_BOOST", 1))
base.grand_cfg_awgn_rr = _cluster_cfg_from_env(
    "GRAND_RR",
    _receiver4_defaults(),
    pre_solver_mode="peel_gf2_restart",
)
base.grand_cfg_awgn_rr_boost = _cluster_cfg_from_env(
    "GRAND_RR_BOOST",
    _receiver4_boost_defaults(),
    pre_solver_mode="peel_gf2_restart",
)

if "GRAND_RR_BOOST_LLR_SOURCE" not in os.environ:
    base.grand_cfg_awgn_rr_boost.llr_source = str(base.grand_cfg_awgn_rr.llr_source)
if "GRAND_RR_BOOST_CHECK_COVER_K" not in os.environ:
    base.grand_cfg_awgn_rr_boost.sv_check_cover_k = int(base.grand_cfg_awgn_rr.sv_check_cover_k)
if "GRAND_RR_BOOST_EPSILON" not in os.environ:
    base.grand_cfg_awgn_rr_boost.sv_epsilon = float(base.grand_cfg_awgn_rr.sv_epsilon)


def run_awgn_sweep_for_code_receiver4(
    code_cfg: Any,
    interleaver: Any,
    snr_sweep: List[float],
    mc_cfg_local: Any,
    output_dir: str,
    alpha: float = 0.8,
) -> Dict[float, Dict[str, Any]]:
    channel_name = "SIONNA_TDL"

    stage1_list_env = str(os.environ.get("STAGE1_ITERS", "4,8,15"))
    stage1_list = [int(x) for x in stage1_list_env.split(",") if x.strip()]
    stage1_list = sorted(set([it for it in stage1_list if it > 0]))

    ldpc_list_env = str(os.environ.get("LDPC_ITERS", "4,8,15,20,100"))
    ldpc_list = [int(x) for x in ldpc_list_env.split(",") if x.strip()]
    ldpc_list = sorted(set([it for it in ldpc_list if it > 0]))

    base_seed = base._env_int("RNG_SEED_GLOBAL", 12345) + base._stable_u32_seed_from_string(code_cfg.code_name)
    run_receiver2 = bool(base._env_int("RUN_RECEIVER2", 0))
    run_receiver3 = bool(base._env_int("RUN_RECEIVER3", 0))
    run_receiver4 = bool(_env_int("RUN_RECEIVER4", 1))

    results: Dict[float, Dict[str, Any]] = {}

    total_threads = int(getattr(base, "NUMBA_THREADS", 0) or base._detect_num_threads())
    use_parallel, n_jobs, threads_per_job, backend = base._snr_parallel_plan(snr_sweep, total_threads)

    if use_parallel:
        print(
            f"[run_awgn_sweep_for_code] SNR-parallel ON: "
            f"n_jobs={n_jobs}, threads/worker={threads_per_job}, backend={backend}"
        )
    else:
        if getattr(base, "NUMBA_AVAILABLE", False) and getattr(base, "set_num_threads", None) is not None:
            try:
                base.set_num_threads(total_threads)
            except Exception:
                pass
        print(f"[run_awgn_sweep_for_code] SNR-parallel OFF (serial); Numba threads={total_threads}")

    def _rr_snapshot_list_for_stage(stage_it: int) -> List[int]:
        raw = str(getattr(base.grand_cfg_awgn_rr, "rescue_snapshot_list", "8,15,4"))
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
        vals = _stable_unique_keep_order([int(stage_it)] + vals)
        vals = [it for it in vals if it <= max(stage1_list)]
        return vals or [int(stage_it)]

    def _run_one_snr(snr_db: float):
        snr_db = float(snr_db)

        if use_parallel and getattr(base, "NUMBA_AVAILABLE", False) and getattr(base, "set_num_threads", None) is not None:
            try:
                base.set_num_threads(int(threads_per_job))
            except Exception:
                pass

        sim_cfg_ldpc = base.SimulationConfig(
            code=code_cfg,
            channel=base.ChannelConfig(name=channel_name, snr_db=snr_db),
            interleaver=interleaver,
            rng_seed_global=int(base_seed),
            snapshot_iters=[],
        )

        per_snr: Dict[str, Any] = {}

        # Scenario 1: Legacy LDPC-only baselines.
        for it in ldpc_list:
            dec_name = f"ldpc{int(it)}"
            seed = int((base_seed + 1_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
            dec_cfg = base.DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
            per_snr[dec_name] = base.run_ldpc_min_sum_adaptive(
                sim_cfg=sim_cfg_ldpc,
                dec_cfg=dec_cfg,
                mc_cfg=mc_cfg_local,
                rng_seed=seed,
                label=dec_name,
            )

        # Scenario 3: Receiver 1.
        for it in stage1_list:
            dec_name = f"hyb{int(it)}"
            seed = int((base_seed + 10_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
            dec_cfg = base.DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
            sim_cfg_hyb = base.SimulationConfig(
                code=code_cfg,
                channel=base.ChannelConfig(name=channel_name, snr_db=snr_db),
                interleaver=interleaver,
                rng_seed_global=int(base_seed),
                snapshot_iters=[int(it)],
            )
            per_snr[dec_name] = base.run_hybrid_ldpc_grand_adaptive(
                sim_cfg=sim_cfg_hyb,
                dec_cfg_stage1=dec_cfg,
                grand_cfg=base.grand_cfg_awgn,
                snapshot_iter=int(it),
                mc_cfg=mc_cfg_local,
                rng_seed=seed,
                label=dec_name,
                grand_cfg_boost=(base.grand_cfg_awgn_boost if getattr(base, "GRAND_USE_BOOST", False) else None),
            )

        # Scenario 4: Receiver 2.
        if run_receiver2:
            for it in stage1_list:
                dec_name = f"hybsv{int(it)}"
                seed = int((base_seed + 20_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
                dec_cfg = base.DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
                sim_cfg_hyb = base.SimulationConfig(
                    code=code_cfg,
                    channel=base.ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=[int(it)],
                )
                per_snr[dec_name] = base.run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=base.grand_cfg_awgn_sv,
                    snapshot_iter=int(it),
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(base.grand_cfg_awgn_sv_boost if getattr(base, "GRAND_SV_USE_BOOST", False) else None),
                )

        # Scenario 5: Receiver 3+.
        if run_receiver3:
            for it in stage1_list:
                dec_name = f"hybptg{int(it)}"
                seed = int((base_seed + 30_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
                dec_cfg = base.DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
                sim_cfg_hyb = base.SimulationConfig(
                    code=code_cfg,
                    channel=base.ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=[int(it)],
                )
                per_snr[dec_name] = base.run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=base.grand_cfg_awgn_ptg,
                    snapshot_iter=int(it),
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(base.grand_cfg_awgn_ptg_boost if getattr(base, "GRAND_PTG_USE_BOOST", False) else None),
                )

        # Scenario 6: Receiver 4R.
        if run_receiver4:
            for it in stage1_list:
                dec_name = f"hybrr{int(it)}"
                seed = int((base_seed + 40_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
                dec_cfg = base.DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
                snap_list = _rr_snapshot_list_for_stage(int(it))
                sim_cfg_hyb = base.SimulationConfig(
                    code=code_cfg,
                    channel=base.ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=snap_list,
                )
                per_snr[dec_name] = base.run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=base.grand_cfg_awgn_rr,
                    snapshot_iter=int(it),
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(base.grand_cfg_awgn_rr_boost if getattr(base, "GRAND_RR_USE_BOOST", False) else None),
                )

        return snr_db, per_snr

    if use_parallel:
        tasks = base.Parallel(
            n_jobs=n_jobs,
            backend=backend,
            prefer="processes",
            batch_size=1,
        )(base.delayed(_run_one_snr)(snr_db) for snr_db in snr_sweep)
        for snr_db, per_snr in tasks:
            results[float(snr_db)] = per_snr
    else:
        for snr_db in snr_sweep:
            snr_db_f, per_snr = _run_one_snr(snr_db)
            results[float(snr_db_f)] = per_snr

    prefix = base._sanitize_prefix(f"{channel_name.lower()}_{code_cfg.code_name}_hybrid")
    base.save_awgn_results(results, output_dir=output_dir, prefix=prefix)
    return results


base.run_local_rescue_with_optional_presolver = run_local_rescue_with_optional_presolver_receiver4
base.grand_hw_cycles_from_result = grand_hw_cycles_from_result_receiver4
base.run_awgn_sweep_for_code = run_awgn_sweep_for_code_receiver4

globals().update(
    {
        name: obj
        for name, obj in base.__dict__.items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
run_local_rescue_with_optional_presolver = run_local_rescue_with_optional_presolver_receiver4
grand_hw_cycles_from_result = grand_hw_cycles_from_result_receiver4
run_awgn_sweep_for_code = run_awgn_sweep_for_code_receiver4


if __name__ == "__main__":
    print(f"[Receiver-4R wrapper] loaded base script from git ref: {_BASE_REFSPEC}")
    base._run_experiments_main()
