#!/usr/bin/env python3
"""DG-KAN v23.09 trust-projected downstream EFRF runner.

This runner intentionally reuses the v23.08 downstream-EFRF implementation for
the expensive row-level mechanics, then adds v23.09 lineage, replay locks,
robust real-task gates, and final-route artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import py_compile
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_08_downstream_coupled_edge_residual_inverse_flow as v2308
from dgkan.fu import downstream_coupled_residual_inverse as dcerif


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.09_TrustProjectedDownstreamEFRF_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.09_TrustProjectedDownstreamEFRF_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.09_TrustProjectedDownstreamEFRF_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2309_OUT_ROOT", str(ROOT / "results/v23_09"))).resolve()
HELPER = ROOT / "dgkan/fu/downstream_coupled_residual_inverse.py"
V2308_RUNNER = ROOT / "experiments/run_v23_08_downstream_coupled_edge_residual_inverse_flow.py"

PART_C_SCHEMES = [
    "C0_FunctionalGram",
    "C1_BlockSNR",
    "C2_H10_reference",
    "C3_LocalEFRF_no_downstream",
    "C4_DownstreamAdjoint_no_inverse",
    "C5_DownstreamAdjoint_LocalInverse_no_trust",
    "C6_DownstreamAdjoint_LocalInverse_ParetoTrust",
    "C7_Candidate_random_projected_R",
    "C8_Candidate_shuffled_design_Phi",
    "C9_Candidate_shuffled_downstream_adjoint",
    "C10_Candidate_trust_free_fixed_alpha",
]

PART_C_DIAGNOSTIC_SCHEMES = [
    "C11_Candidate_layerwise_trust_projection",
    "C12_Candidate_reference_pareto_projection",
    "C13_Candidate_reference_pareto_nodebt_projection",
    "C14_Candidate_fullpath_reference_pareto_nodebt_projection",
    "C15_Candidate_terminal_audited_no_trust",
    "C16_Candidate_terminal_audited_adjoint_local_every10",
    "C17_Candidate_terminal_audited_adjoint_local_every5",
    "C18_Candidate_terminal_audited_topdown_every10",
    "C19_Candidate_terminal_audited_source_guard_population",
    "C20_Candidate_terminal_audited_margin_curvature",
    "C21_Control_terminal_audited_shuffled_margin_weight",
    "C22_Candidate_terminal_audited_diagonal_fisher",
    "C23_Control_terminal_audited_shuffled_fisher_weight",
    "C24_Candidate_terminal_audited_guard_penalized_local_inverse",
    "C25_Control_terminal_audited_extra_ridge_local_inverse",
    "C26_Candidate_terminal_audited_smooth_mode_local_inverse",
    "C27_Control_terminal_audited_tail_mode_local_inverse",
]

PART_C_TO_F_SCHEME = {
    "C0_FunctionalGram": "F0_FunctionalGram_baseline",
    "C1_BlockSNR": "F1_BlockSNR_baseline",
    "C2_H10_reference": "F2_H10_known_structure_baseline",
    "C3_LocalEFRF_no_downstream": "F3_v23_07_local_EFRF_all_layer_GS_control",
    "C7_Candidate_random_projected_R": "F11_random_projected_residual_flow",
    "C8_Candidate_shuffled_design_Phi": "F12_shuffled_design_residual_flow",
}


def configure_v2308() -> None:
    v2308.PLAN = PLAN
    v2308.EXEC_LOG = EXEC_LOG
    v2308.RECAP_LOG = RECAP_LOG
    v2308.OUT_ROOT = OUT_ROOT
    v2308.RUNNER = RUNNER
    v2308.HELPER = HELPER
    v2308.PART_I_SCHEME_TO_F_SCHEME.setdefault("I7_shuffled_design_residual_flow", "F12_shuffled_design_residual_flow")


def ensure_logs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.09 Trust-Projected Downstream EFRF 执行日志\n\n"
            "- 原则：不造假；不使用 held/test 诱导；记录复现命令、GPU、文件和修复路径。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.09 Trust-Projected Downstream EFRF 实验结果复盘\n\n"
            "- 原则：只记录真实 artifact 数据；失败、修复、证据链和 insight 都落盘。\n",
            encoding="utf-8",
        )


def init() -> None:
    ensure_logs()
    configure_v2308()


def rel(path: Path | str) -> str:
    return v2308.rel(Path(path))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    return v2308.write_json(path, data)


def read_json(path: Path) -> dict[str, Any]:
    return v2308.read_json(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    return v2308.write_rows(path, rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    return v2308.read_rows(path)


def append_exec(title: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    ensure_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title} {status}\n\n")
        fh.write(f"- command: `{command}`\n")
        if gpu:
            fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    ensure_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def command_text(argv: list[str]) -> str:
    return f"{sys.executable} {' '.join(argv)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fval(x: Any, default: float = 0.0) -> float:
    return v2308.fval(x, default)


def ival(x: Any) -> int:
    return v2308.ival(x)


def median(xs: Any) -> float:
    return v2308.median(xs)


def next_actions(part: str, route: str, blocker: str, allowed: list[str], required: list[str] | None = None) -> Path:
    return v2308.next_actions(part, route, blocker, allowed, required)


def failure(part: str, payload: dict[str, Any]) -> Path:
    return v2308.write_failure_decomposition(part, payload)


def csv_by(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {str(r.get(key, "")): r for r in read_rows(path)}


def visual_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if str(r.get("dataset")) in {"MNIST", "FashionMNIST", "KMNIST"}]


def part_0(args: argparse.Namespace) -> dict[str, Any]:
    init()
    vroot = Path(args.v23_08_root).resolve()
    required = [
        "final_route.json",
        "part_f_summary.json",
        "part_h_summary.json",
        "part_i_summary.json",
        "part_f_multistep_group_summary.csv",
        "part_i_real_task_group_summary.csv",
    ]
    missing = [name for name in required if not (vroot / name).exists()]
    final = read_json(vroot / "final_route.json") if not missing else {}
    pf = read_json(vroot / "part_f_summary.json") if not missing else {}
    ph = read_json(vroot / "part_h_summary.json") if not missing else {}
    pi = read_json(vroot / "part_i_summary.json") if not missing else {}
    f_groups = csv_by(vroot / "part_f_multistep_group_summary.csv", "scheme") if not missing else {}
    i_groups = read_rows(vroot / "part_i_real_task_group_summary.csv") if not missing else []
    official_visual = [
        r for r in i_groups
        if r.get("i_scheme") == "I3_DownstreamEFRF_best_positive_control_scheme"
        and r.get("dataset") in {"MNIST", "FashionMNIST", "KMNIST"}
    ]
    random_gap_visual = median(r.get("random_projected_gap_median") for r in official_visual)
    kmnist = next((r for r in official_visual if r.get("dataset") == "KMNIST"), {})
    f27 = f_groups.get("F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between", {})
    f28 = f_groups.get("F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between", {})
    f27_cov = fval(f27.get("C2_coverage_improvement_median"))
    f28_cov = fval(f28.get("C2_coverage_improvement_median"))
    f27_over = fval(f27.get("solver_overhead_ratio"), 99.0)
    f28_over = fval(f28.get("solver_overhead_ratio"), 99.0)
    best_f = "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between" if f28_cov >= 0.95 * max(f27_cov, 1.0e-12) and f28_over < f27_over else "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between"
    files_to_hash = [HELPER, V2308_RUNNER, ROOT / "dgkan/optim/edge_sobolev_population_flow.py", RUNNER]
    hashes = {rel(p): sha256_file(p) for p in files_to_hash if p.exists()}
    stale_artifact_count = len(missing)
    promotion_allowed = ival(final.get("promotion_allowed"))
    gate = int(
        not missing
        and ival(pf.get("gate_pass"))
        and ival(ph.get("gate_pass"))
        and ival(pi.get("gate_pass"))
        and str(final.get("final_route")) == "LimitedRealTaskPreflightPass_NotOfficial"
        and promotion_allowed == 0
    )
    blocker = "none" if gate else "lineage_or_v23_08_artifact_missing"
    row = {
        "selected_mainline": "v23_08_downstream_coupled_EFRF",
        "v23_08_synthetic_positive_control_pass": ival(pf.get("gate_pass")),
        "v23_08_limited_preflight_pass": ival(pi.get("gate_pass")),
        "v23_08_promotion_allowed": promotion_allowed,
        "v23_08_real_task_blocker": str(pi.get("dominant_blocker", "missing")),
        "candidate_source_root": rel(vroot),
        "candidate_scheme_family": best_f,
        "version_branch_conflict_detected": 0,
        "stale_artifact_count": stale_artifact_count,
        "v23_08_random_gap_visual_median": random_gap_visual,
        "v23_08_kmnist_coverage_delta": fval(kmnist.get("held_coverage_improvement_median")),
        "v23_08_overhead": fval(f_groups.get(best_f, {}).get("solver_overhead_ratio")),
        "v23_08_no_fake_data": int(ival(pi.get("audit_clean")) and not ival(pi.get("used_fake_data_rows"))),
        "v23_08_no_held_test_induction": ival(pi.get("audit_clean")),
    }
    matrix = write_rows(OUT_ROOT / "part_0_lineage_matrix.csv", [row])
    summary = {
        "part": "0",
        "gate_pass": gate,
        "route": "Part0LineageLockPass" if gate else "Part0LineageLockFailed",
        "dominant_blocker": blocker,
        "missing_artifacts": missing,
        "candidate_source_root": rel(vroot),
        "selected_mainline": row["selected_mainline"],
        "selected_f_scheme": best_f,
        "selected_i_scheme": "I3_DownstreamEFRF_best_positive_control_scheme",
        "v23_08_part_f_gate_pass": ival(pf.get("gate_pass")),
        "v23_08_part_h_gate_pass": ival(ph.get("gate_pass")),
        "v23_08_part_i_gate_pass": ival(pi.get("gate_pass")),
        "v23_08_final_route": str(final.get("final_route", "missing")),
        "v23_08_promotion_allowed": promotion_allowed,
        "code_hashes": hashes,
        "matrix": rel(matrix),
    }
    write_json(OUT_ROOT / "part_0_lineage_summary.json", summary)
    fail = failure("0", summary)
    next_actions("0", summary["route"], blocker, [] if gate else ["fix artifact reader/path/hash/schema before running later parts"], [rel(matrix), rel(fail)])
    append_exec("Part 0 lineage", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_0_lineage_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part 0 lineage and v23.08 artifact lock", summary)
    return summary


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    init()
    base = v2308.part_a(args)
    device = v2308.device_from_args(args)
    extra: dict[str, Any] = {
        "true_depth2_purekan_constructed": 0,
        "true_depth3_purekan_constructed": 0,
        "changed_edge_coefficients": 0,
        "changed_mlp_params": 0,
        "residual_target_uses_held_test": 0,
        "trust_uses_held_test": 0,
        "runtime_selector_used": 0,
        "candidate_update_selection_used": 0,
        "metric_winner_selection_used": 0,
        "structure_transform_in_forward": 0,
        "used_fake_data_rows": 0,
    }
    try:
        x, y, xg, yg = v2308.v2307.visual_data("local_patch_interaction", 0, args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        m2 = v2308.v2307.make_model(str(args.basis_key), 2, int(x.shape[1]), classes, 23090002, args, device, basis_input_gain=0.25)
        m3 = v2308.v2307.make_model(str(args.basis_key), 3, int(x.shape[1]), classes, 23090003, args, device, basis_input_gain=0.25)
        extra["true_depth2_purekan_constructed"] = int(hasattr(m2, "coeffs") and len(m2.coeffs) == 2)
        extra["true_depth3_purekan_constructed"] = int(hasattr(m3, "coeffs") and len(m3.coeffs) == 3)
        fargs = argparse.Namespace(**vars(args))
        fargs.part_f_residual_target = "norm"
        fargs.part_f_lambda = float(args.part_f_lambda)
        fargs.part_f_alphas = str(args.part_f_alphas)
        fargs.part_e_alphas = str(args.part_f_alphas)
        fargs.part_e_trust_mode = "pareto"
        fargs.part_e_trust_tolerance = 0.0
        fargs.part_d_lambda = float(args.part_f_lambda)
        fargs.part_d_solver = str(args.part_f_solver)
        fargs.part_e_sketch_rank = int(args.part_f_sketch_rank)
        deltas, _diag = v2308.e_delta_candidate(
            "E9_topdown_last_penultimate_adjoint_local_one_step",
            m3,
            x[:64],
            y[:64],
            xg[:64],
            yg[:64],
            "norm",
            fargs,
            device,
            seed=2309,
            task="local_patch_interaction",
            checkpoint="identity_smoke",
        )
        state = v2308.model_state(m3)
        v2308.apply_deltas(m3, v2308.scaled_deltas(deltas, fargs), 0.03125)
        changed = []
        for name, tensor in m3.state_dict().items():
            diff = float((tensor.detach().to(dtype=torch.float64) - state[name].to(device=tensor.device, dtype=torch.float64)).norm().detach().cpu().item())
            if diff > 1.0e-12:
                changed.append(name)
        extra["changed_edge_coefficients"] = int(bool(changed) and all(name.startswith("coeffs.") for name in changed))
        extra["changed_mlp_params"] = int(any(("mlp" in name.lower() or "readout" in name.lower()) for name in changed))
    except Exception as exc:
        extra["part_a_extra_error"] = repr(exc)
    base.update(extra)
    gate = int(
        ival(base.get("gate_pass"))
        and extra["true_depth2_purekan_constructed"]
        and extra["true_depth3_purekan_constructed"]
        and extra["changed_edge_coefficients"]
        and not extra["changed_mlp_params"]
        and not extra["residual_target_uses_held_test"]
        and not extra["trust_uses_held_test"]
        and not extra["runtime_selector_used"]
        and not extra["candidate_update_selection_used"]
        and not extra["metric_winner_selection_used"]
        and not extra["structure_transform_in_forward"]
    )
    base["gate_pass"] = gate
    base["route"] = "PartAImplementationIdentityPass" if gate else "A_FailedIdentity"
    base["dominant_blocker"] = "none" if gate else "identity_or_anti_selector_audit_failed"
    write_json(OUT_ROOT / "part_a_summary.json", base)
    fail = failure("a", base)
    next_actions("a", base["route"], base["dominant_blocker"], [] if gate else ["fix implementation identity; do not alter gates or remove controls"], [rel(OUT_ROOT / "part_a_summary.json"), rel(fail)])
    append_recap("Part A v23.09 implementation identity extension", base)
    return base


def part_b_replay(args: argparse.Namespace) -> dict[str, Any]:
    init()
    p0 = read_json(OUT_ROOT / "part_0_lineage_summary.json")
    if not ival(p0.get("gate_pass")):
        return v2308.blocked_summary("B", "B_BlockedByPart0", "part_0_failed")
    vroot = Path(args.v23_08_root).resolve()
    src = vroot / "part_f_multistep_group_summary.csv"
    groups = read_rows(src)
    map_rows = {
        "F0_FunctionalGram_baseline": "B0_FunctionalGram_baseline",
        "F1_BlockSNR_baseline": "B1_BlockSNR_baseline",
        "F2_H10_known_structure_baseline": "B2_H10_known_structure_reference",
        "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between": "B3_v23_08_F27_reference",
        "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between": "B4_v23_08_F28_reference",
        "F11_random_projected_residual_flow": "B5_random_projected_residual_flow",
        "F12_shuffled_design_residual_flow": "B6_shuffled_design_residual_flow",
        "F13_same_compute_noop": "B7_same_compute_noop",
    }
    rows: list[dict[str, Any]] = []
    by_scheme = {str(r.get("scheme")): r for r in groups}
    for f_scheme, b_scheme in map_rows.items():
        r = by_scheme.get(f_scheme, {})
        row = {"part": "B", "status": "ok" if r else "missing", "scheme": b_scheme, "source_f_scheme": f_scheme, "source_root": rel(vroot)}
        row.update(r)
        rows.append(row)
    matrix = write_rows(OUT_ROOT / "part_b_synthetic_lock_matrix.csv", rows)
    task_src = vroot / "part_f_multistep_task_summary.csv"
    task_csv = write_rows(OUT_ROOT / "part_b_synthetic_lock_task_summary.csv", read_rows(task_src)) if task_src.exists() else write_rows(OUT_ROOT / "part_b_synthetic_lock_task_summary.csv", [])
    group_csv = write_rows(OUT_ROOT / "part_b_synthetic_lock_group_summary.csv", rows)
    f27 = by_scheme.get("F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between", {})
    f28 = by_scheme.get("F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between", {})
    selected = "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between" if fval(f28.get("C2_coverage_improvement_median")) >= 0.95 * max(fval(f27.get("C2_coverage_improvement_median")), 1.0e-12) and fval(f28.get("solver_overhead_ratio"), 99.0) < fval(f27.get("solver_overhead_ratio"), 99.0) else "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between"
    cand = by_scheme.get(selected, {})
    official_shape = int(ival(cand.get("official_shape")))
    gate = int(
        official_shape
        and fval(cand.get("random_projected_gap")) >= 0.20
        and fval(cand.get("shuffled_design_gap")) >= 0.20
        and fval(cand.get("F5_no_debt_count")) >= 28
        and fval(cand.get("finite_step_accept_rate_median")) >= 0.45
        and fval(cand.get("finite_step_scale_mean_median")) >= 0.35
        and fval(cand.get("solver_overhead_ratio"), 99.0) <= 2.0
    )
    if gate:
        blocker = "none"
    elif fval(cand.get("finite_step_scale_mean_median")) < 0.35:
        blocker = "synthetic_trust_scale_mean_low"
    elif fval(cand.get("random_projected_gap")) < 0.20 or fval(cand.get("shuffled_design_gap")) < 0.20:
        blocker = "synthetic_random_or_shuffled_gap_low"
    else:
        blocker = "synthetic_positive_control_regressed"
    summary = {
        "part": "B",
        "gate_pass": gate,
        "route": "PartBSyntheticReplayLockPass" if gate else "B_SyntheticPositiveControlRegressed",
        "dominant_blocker": blocker,
        "replay_only": 1,
        "source_group_summary": rel(src),
        "selected_source_f_scheme": selected,
        "candidate_group": cand,
        "matrix": rel(matrix),
        "task_summary": rel(task_csv),
        "group_summary": rel(group_csv),
    }
    write_json(OUT_ROOT / "part_b_synthetic_lock_summary.json", summary)
    fail = failure("b", summary)
    next_actions("b", summary["route"], blocker, [] if gate else ["rerun Part B with v23.09 pareto trust and controls; check trust scale mismatch before real-task progression"], [rel(matrix), rel(group_csv), rel(fail)])
    append_exec("Part B replay", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_b_synthetic_lock_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; selected={selected}")
    append_recap("Part B synthetic positive-control replay lock", summary)
    return summary


def strict_part_b_from_group_summary(args: argparse.Namespace, *, replay_only: int, source_group_summary: Path) -> dict[str, Any]:
    groups = read_rows(source_group_summary)
    by_scheme = {str(r.get("scheme")): r for r in groups}
    f27 = by_scheme.get("F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between", {})
    f28 = by_scheme.get("F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between", {})
    def strict_candidate_pass(row: dict[str, Any]) -> int:
        return int(
            ival(row.get("official_shape"))
            and fval(row.get("random_projected_gap")) >= 0.20
            and fval(row.get("shuffled_design_gap")) >= 0.20
            and fval(row.get("F5_no_debt_count")) >= 28
            and fval(row.get("finite_step_accept_rate_median")) >= 0.45
            and fval(row.get("finite_step_scale_mean_median")) >= 0.35
            and fval(row.get("solver_overhead_ratio"), 99.0) <= 2.0
        )
    f27_pass = strict_candidate_pass(f27)
    f28_pass = strict_candidate_pass(f28)
    selected = "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between"
    if f27_pass and f28_pass and fval(f28.get("C2_coverage_improvement_median")) >= 0.95 * max(fval(f27.get("C2_coverage_improvement_median")), 1.0e-12) and fval(f28.get("solver_overhead_ratio"), 99.0) < fval(f27.get("solver_overhead_ratio"), 99.0):
        selected = "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between"
    cand = by_scheme.get(selected, {})
    official_shape = int(ival(cand.get("official_shape")))
    gate = int(
        official_shape
        and fval(cand.get("random_projected_gap")) >= 0.20
        and fval(cand.get("shuffled_design_gap")) >= 0.20
        and fval(cand.get("F5_no_debt_count")) >= 28
        and fval(cand.get("finite_step_accept_rate_median")) >= 0.45
        and fval(cand.get("finite_step_scale_mean_median")) >= 0.35
        and fval(cand.get("solver_overhead_ratio"), 99.0) <= 2.0
    )
    if gate:
        blocker = "none"
    elif not official_shape:
        blocker = "formal_seed_or_step_count_not_met"
    elif fval(cand.get("finite_step_scale_mean_median")) < 0.35:
        blocker = "synthetic_trust_scale_mean_low"
    elif fval(cand.get("finite_step_accept_rate_median")) < 0.45:
        blocker = "synthetic_trust_accept_rate_low"
    elif fval(cand.get("random_projected_gap")) < 0.20 or fval(cand.get("shuffled_design_gap")) < 0.20:
        blocker = "synthetic_random_or_shuffled_gap_low"
    elif fval(cand.get("solver_overhead_ratio"), 99.0) > 2.0:
        blocker = "synthetic_overhead_high"
    else:
        blocker = "synthetic_positive_control_regressed"
    summary = {
        "part": "B",
        "gate_pass": gate,
        "route": "PartBSyntheticLockPass" if gate else "B_SyntheticPositiveControlRegressed",
        "dominant_blocker": blocker,
        "replay_only": replay_only,
        "source_group_summary": rel(source_group_summary),
        "selected_source_f_scheme": selected,
        "candidate_group": cand,
        "f27_strict_gate_pass": f27_pass,
        "f28_strict_gate_pass": f28_pass,
        "part_f_trust_mode": str(args.part_f_trust_mode),
        "part_f_trust_tolerance": float(args.part_f_trust_tolerance),
        "part_f_alphas": str(args.part_f_alphas),
        "matrix": rel(OUT_ROOT / "part_f_multistep_positive_control_matrix.csv"),
        "task_summary": rel(OUT_ROOT / "part_f_multistep_task_summary.csv"),
        "group_summary": rel(source_group_summary),
    }
    write_json(OUT_ROOT / "part_b_synthetic_lock_summary.json", summary)
    fail = failure("b", summary)
    next_actions("b", summary["route"], blocker, [] if gate else ["check trust mismatch, alpha schedule, and Pareto thresholds; do not enter real-task Part D while Part B is red"], [summary["matrix"], summary["group_summary"], rel(fail)])
    append_exec("Part B strict merge", command_text(sys.argv), "done", files=f"{summary['matrix']}; {summary['group_summary']}; {rel(OUT_ROOT / 'part_b_synthetic_lock_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; selected={selected}; replay={replay_only}")
    append_recap("Part B strict synthetic lock", summary)
    return summary


def part_b_run(args: argparse.Namespace) -> dict[str, Any]:
    init()
    p0 = read_json(OUT_ROOT / "part_0_lineage_summary.json")
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    if not ival(p0.get("gate_pass")):
        return v2308.blocked_summary("B", "B_BlockedByPart0", "part_0_failed")
    if not ival(pa.get("gate_pass")):
        return v2308.blocked_summary("B", "B_BlockedByPartA", "part_a_failed")
    device = v2308.device_from_args(args)
    jobs = v2308.shard_items(v2308.f_jobs(args), args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"part_f_multistep_positive_control_matrix{suffix}.csv"
    rows: list[dict[str, Any]] = []
    existing = read_rows(matrix) if int(args.part_f_resume) else []
    done = {
        (
            r.get("scheme"),
            r.get("task"),
            r.get("seed"),
            r.get("train_steps"),
            r.get("residual_target"),
            r.get("part_f_trust_mode"),
            r.get("part_f_trust_tolerance"),
            r.get("part_f_alphas"),
            r.get("part_f_delta_scale"),
            r.get("part_f_delta_sign"),
            r.get("lambda"),
            r.get("activation_drift_cap"),
            r.get("functionalgram_lr"),
            r.get("alignment_min"),
            r.get("alignment_soft_floor"),
            r.get("alignment_soft_ceiling"),
            r.get("alignment_soft_min_scale"),
            r.get("alignment_soft_power"),
            r.get("tail99_step_budget"),
            r.get("tail99_cumulative_budget"),
            r.get("model_seed_scheme_key"),
            r.get("tail_correction_scales"),
            r.get("tail_correction_target"),
            r.get("tail_correction_layer_group"),
            r.get("tail_correction_transport_min"),
        )
        for r in existing
        if r.get("status") == "ok"
    }
    for job in jobs:
        key = (
            job[0],
            job[1],
            str(job[2]),
            str(int(args.part_f_steps)),
            str(args.part_f_residual_target),
            str(args.part_f_trust_mode),
            str(float(args.part_f_trust_tolerance)),
            str(args.part_f_alphas),
            str(float(args.part_f_delta_scale)),
            str(float(args.part_f_delta_sign)),
            str(float(args.part_f_lambda)),
            str(float(args.part_f_activation_drift_cap)),
            str(float(args.part_f_functionalgram_lr)),
            str(float(args.part_f_alignment_min)),
            str(float(args.part_f_alignment_soft_floor)),
            str(float(args.part_f_alignment_soft_ceiling)),
            str(float(args.part_f_alignment_soft_min_scale)),
            str(float(args.part_f_alignment_soft_power)),
            str(float(args.part_f_tail99_step_budget)),
            str(float(args.part_f_tail99_cumulative_budget)),
            str(args.part_f_model_seed_scheme_key),
            str(args.part_f_tail_correction_scales),
            str(args.part_f_tail_correction_target),
            str(args.part_f_tail_correction_layer_group),
            str(float(args.part_f_tail_correction_transport_min)),
        )
        if key in done:
            continue
        row = v2308.run_f_row(job, args, device)
        row.update(
            {
                "part_f_trust_mode": str(args.part_f_trust_mode),
                "part_f_trust_tolerance": float(args.part_f_trust_tolerance),
                "part_f_alphas": str(args.part_f_alphas),
                "part_f_delta_scale": float(args.part_f_delta_scale),
                "part_f_delta_sign": float(args.part_f_delta_sign),
            }
        )
        rows.append(row)
        if int(args.part_f_flush_every) and len(rows) % int(args.part_f_flush_every) == 0:
            v2308.append_rows(matrix, rows)
            print(json.dumps({"part": "B", "rows_written": len(read_rows(matrix)), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
            rows = []
    if rows:
        v2308.append_rows(matrix, rows)
    all_rows = read_rows(matrix)
    summary = {
        "part": "B",
        "gate_pass": 0,
        "route": "PartBShardOnly" if suffix else "PartBNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(all_rows),
        "ok_rows": sum(1 for r in all_rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in all_rows if r.get("status") == "error"),
        "matrix": rel(matrix),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_f_resume),
    }
    write_json(OUT_ROOT / f"part_b_synthetic_lock_summary{suffix}.json", summary)
    append_exec("Part B synthetic shard", command_text(sys.argv), "done", files=rel(matrix), gpu=str(device), note=f"rows={len(all_rows)}")
    append_recap("Part B synthetic positive-control shard", summary)
    return summary


def part_b_merge(args: argparse.Namespace) -> dict[str, Any]:
    init()
    v2308.part_f_merge(args)
    return strict_part_b_from_group_summary(args, replay_only=0, source_group_summary=OUT_ROOT / "part_f_multistep_group_summary.csv")


def selected_part_c_f_scheme() -> str:
    pb = read_json(OUT_ROOT / "part_b_synthetic_lock_summary.json")
    selected = str(pb.get("selected_source_f_scheme", ""))
    if selected in {
        "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between",
        "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
    }:
        return selected
    return "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between"


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [
        (scheme, task, seed)
        for scheme in v2308.csv_items(args.part_c_schemes)
        for task in v2308.csv_items(args.part_c_tasks)
        for seed in range(int(args.part_c_seed_count))
    ]


def c_args_from_f_args(args: argparse.Namespace, *, c_scheme: str, source_f_scheme: str) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.part_f_steps = int(args.part_c_steps)
    out.part_f_seed_count = int(args.part_c_seed_count)
    out.part_f_model_seed_scheme_key = str(args.part_c_model_seed_scheme_key)
    if c_scheme == "C5_DownstreamAdjoint_LocalInverse_no_trust":
        out.part_f_trust_mode = "coverage_floor"
        out.part_f_trust_tolerance = 1.0e9
        out.part_f_alphas = str(args.part_f_alphas)
    elif c_scheme == "C10_Candidate_trust_free_fixed_alpha":
        out.part_f_trust_mode = "coverage_floor"
        out.part_f_trust_tolerance = 1.0e9
        out.part_f_alphas = str(args.part_c_fixed_alpha)
    out.part_f_schemes = source_f_scheme
    return out


def choose_layerwise_projected_f_deltas(
    model: Any,
    base_state: dict[str, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    base_acts: list[torch.Tensor],
    tail99_reference: dict[str, float] | None = None,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    if not deltas:
        return 0.0, 1, "same_compute_noop", source_before, guard_before
    layer_ids = sorted(int(layer) for layer in deltas)
    alpha_grid = v2308.float_items(args.part_e_alphas)
    if not alpha_grid:
        return 0.0, 0, "no_alpha_candidates", source_before, guard_before
    delta_norm2 = {
        layer: float(deltas[layer].detach().to(dtype=torch.float64).square().sum().detach().cpu().item())
        for layer in layer_ids
    }
    reject_reason = "no_layerwise_projection_candidates"
    best: tuple[float, float, tuple[float, ...], dict[str, float], dict[str, float]] | None = None
    best_score: tuple[float, float, float] | None = None
    drift_cap = float(getattr(args, "part_f_activation_drift_cap", -1.0))
    for scales in itertools.product(alpha_grid, repeat=len(layer_ids)):
        v2308.load_model_state(model, base_state)
        for layer, scale in zip(layer_ids, scales):
            v2308.apply_delta(model, int(layer), deltas[int(layer)], float(scale))
        source_after = v2308.actual_metrics(model, x, y)
        guard_after = v2308.actual_metrics(model, xg, yg)
        accepted, reject_reason = v2308.trust_accept_e(source_before, source_after, guard_before, guard_after, args)
        if accepted and not v2308.tail99_step_accept(guard_before, guard_after, args):
            accepted, reject_reason = False, "tail99_step_trust_reject_layerwise_projection"
        if accepted and not v2308.tail99_cumulative_accept(tail99_reference, guard_after, args):
            accepted, reject_reason = False, "tail99_cumulative_trust_reject_layerwise_projection"
        if accepted and drift_cap > 0.0:
            drift = v2308.max_activation_drift_to_base(model, xg, base_acts)
            if drift > drift_cap:
                accepted, reject_reason = False, "activation_drift_cap_reject_layerwise_projection"
        if not accepted:
            continue
        distance = sum((1.0 - float(scale)) ** 2 * delta_norm2[layer] for layer, scale in zip(layer_ids, scales))
        mean_scale = sum(float(scale) for scale in scales) / max(1, len(scales))
        score = (float(distance), -float(mean_scale))
        if best is None or score < (best[0], best[1]):
            best = (float(distance), -float(mean_scale), tuple(float(s) for s in scales), source_after, guard_after)
    if best is None:
        v2308.load_model_state(model, base_state)
        return 0.0, 0, reject_reason, source_before, guard_before
    _distance, neg_mean_scale, best_scales, source_after, guard_after = best
    v2308.load_model_state(model, base_state)
    for layer, scale in zip(layer_ids, best_scales):
        v2308.apply_delta(model, int(layer), deltas[int(layer)], float(scale))
    return -float(neg_mean_scale), 1, "layerwise_projection_accepted", source_after, guard_after


def choose_reference_pareto_projected_f_deltas(
    model: Any,
    base_state: dict[str, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    reference_source: dict[str, float],
    reference_guard: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    base_acts: list[torch.Tensor],
    tail99_reference: dict[str, float] | None = None,
    *,
    require_no_debt: bool = False,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    if not deltas:
        return 0.0, 1, "same_compute_noop", source_before, guard_before
    layer_ids = sorted(int(layer) for layer in deltas)
    alpha_grid = v2308.float_items(args.part_e_alphas)
    if not alpha_grid:
        return 0.0, 0, "no_alpha_candidates", source_before, guard_before
    delta_norm2 = {
        layer: float(deltas[layer].detach().to(dtype=torch.float64).square().sum().detach().cpu().item())
        for layer in layer_ids
    }
    reject_reason = "no_reference_projection_candidates"
    best: tuple[float, float, tuple[float, ...], dict[str, float], dict[str, float]] | None = None
    best_score: tuple[float, float, float] | None = None
    drift_cap = float(getattr(args, "part_f_activation_drift_cap", -1.0))
    for scales in itertools.product(alpha_grid, repeat=len(layer_ids)):
        v2308.load_model_state(model, base_state)
        for layer, scale in zip(layer_ids, scales):
            v2308.apply_delta(model, int(layer), deltas[int(layer)], float(scale))
        source_after = v2308.actual_metrics(model, x, y)
        guard_after = v2308.actual_metrics(model, xg, yg)
        accepted, reject_reason = v2308.trust_accept_e(reference_source, source_after, reference_guard, guard_after, args)
        if accepted and not v2308.tail99_step_accept(reference_guard, guard_after, args):
            accepted, reject_reason = False, "tail99_step_trust_reject_reference_projection"
        if accepted and not v2308.tail99_cumulative_accept(tail99_reference, guard_after, args):
            accepted, reject_reason = False, "tail99_cumulative_trust_reject_reference_projection"
        if accepted and drift_cap > 0.0:
            drift = v2308.max_activation_drift_to_base(model, xg, base_acts)
            if drift > drift_cap:
                accepted, reject_reason = False, "activation_drift_cap_reject_reference_projection"
        if accepted and require_no_debt:
            debt = v2308.debt_deltas(reference_guard, guard_after)
            if not v2308.no_debt_ok(debt, float(args.no_debt_budget)):
                accepted, reject_reason = False, "no_debt_reject_reference_projection"
        if not accepted:
            continue
        distance = sum((1.0 - float(scale)) ** 2 * delta_norm2[layer] for layer, scale in zip(layer_ids, scales))
        mean_scale = sum(float(scale) for scale in scales) / max(1, len(scales))
        source_gain = float(source_after.get("coverage", 0.0)) - float(reference_source.get("coverage", 0.0))
        guard_gain = float(guard_after.get("coverage", 0.0)) - float(reference_guard.get("coverage", 0.0))
        score = (float(distance), -float(mean_scale), -float(source_gain + guard_gain))
        if best_score is None or score < best_score:
            best_score = score
            best = (float(distance), -float(mean_scale), tuple(float(s) for s in scales), source_after, guard_after)
    if best is None:
        v2308.load_model_state(model, base_state)
        return 0.0, 0, reject_reason, source_before, guard_before
    _distance, neg_mean_scale, best_scales, source_after, guard_after = best
    v2308.load_model_state(model, base_state)
    for layer, scale in zip(layer_ids, best_scales):
        v2308.apply_delta(model, int(layer), deltas[int(layer)], float(scale))
    return -float(neg_mean_scale), 1, "reference_projection_accepted", source_after, guard_after


def first_json_number(text: Any) -> Any:
    try:
        values = json.loads(str(text))
        if isinstance(values, list) and values:
            return float(values[0])
    except Exception:
        return ""
    return ""


def annotate_c_row(row: dict[str, Any], *, c_scheme: str, source_scheme: str, ablation_note: str) -> dict[str, Any]:
    out = dict(row)
    out["part"] = "C"
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = source_scheme
    out["component_ablation"] = ablation_note
    out["multi_step_C2_coverage"] = out.get("C2_coverage_improvement", "")
    out["multi_step_C2_accuracy"] = out.get("C2_accuracy_improvement", "")
    out["actual_one_step_C2_delta"] = first_json_number(out.get("one_step_gain_retention_curve", ""))
    out["trust_accept_rate"] = out.get("finite_step_accept_rate", "")
    out["trust_scale_mean"] = out.get("finite_step_scale_mean", "")
    out["trust_reject_reasons"] = out.get("finite_step_reject_reasons", "")
    out["condition_number_after_ridge"] = out.get("condition_number_median", "")
    out["CG_iterations"] = out.get("cg_iteration_median", "")
    out["activation_drift"] = out.get("activation_drift_max", "")
    out["jacobian_drift_proxy"] = out.get("J_downstream_drift_sketch", "")
    out["overhead_ratio"] = out.get("solver_overhead_ratio", "")
    out["held_test_usage"] = 0
    out["used_fake_data_rows"] = 0
    out["runtime_selector_used"] = 0
    return out


def residual_fit_ratio_from_prediction(pred: torch.Tensor, residual: torch.Tensor) -> float:
    pp = pred.detach().to(dtype=torch.float64)
    rr = residual.detach().to(device=pp.device, dtype=torch.float64)
    before = rr.square().mean().clamp_min(1.0e-12)
    after = (rr - pp).square().mean()
    return float(((before - after) / before).detach().cpu().item())


def predicted_output_delta_from_deltas(model: Any, x: torch.Tensor, deltas: dict[int, torch.Tensor], alpha: float) -> torch.Tensor:
    logits, acts = model.forward_with_activations(x)
    pred = torch.zeros_like(logits.detach(), dtype=torch.float64)
    for layer_id, delta in deltas.items():
        op = v2308.dcerif.make_downstream_operator(model, acts, int(layer_id))
        pred = pred + op.apply_H(delta.to(device=logits.device, dtype=torch.float64) * float(alpha))
    return pred.detach()


def add_c_one_step_fit_metrics(row: dict[str, Any], *, c_scheme: str, source_scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device, variant: str | None = None) -> dict[str, Any]:
    out = dict(row)
    if c_scheme in {"C0_FunctionalGram", "C1_BlockSNR", "C2_H10_reference"}:
        out["source_residual_fit_improvement"] = "not_applicable_optimizer_baseline"
        out["guard_residual_fit_improvement"] = "not_applicable_optimizer_baseline"
        out["residual_fit_probe_alpha"] = ""
        return out
    try:
        fargs = c_args_from_f_args(args, c_scheme=c_scheme, source_f_scheme=source_scheme)
        x, y, xg, yg = v2308.v2307.visual_data(task, int(seed), fargs, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed_scheme_key = str(fargs.part_f_model_seed_scheme_key or source_scheme)
        model_seed = 23083000 + int(seed) * 1009 + sum(ord(c) for c in model_seed_scheme_key + task)
        model = v2308.v2307.make_model(str(fargs.basis_key), int(fargs.depth), int(x.shape[1]), classes, model_seed, fargs, device, basis_input_gain=float(fargs.part_d_basis_input_gain))
        v2308.v2307.train_checkpoint(model, x, y, str(fargs.part_f_checkpoint), int(seed), fargs)
        eargs = v2308.f_e_args(fargs)
        if variant:
            raw_deltas, fit_diag = c_adjoint_variant_delta(variant, model, x, y, xg, yg, str(fargs.part_f_residual_target), eargs, device, seed=int(seed) + 1, task=task, checkpoint=str(fargs.part_f_checkpoint))
        else:
            e_scheme, rank = v2308.f_scheme_to_e_scheme(source_scheme)
            eargs = v2308.f_e_args(fargs, sketch_rank=rank)
            raw_deltas, fit_diag = v2308.e_delta_candidate(e_scheme, model, x, y, xg, yg, str(fargs.part_f_residual_target), eargs, device, seed=int(seed) + 1, task=task, checkpoint=str(fargs.part_f_checkpoint))
        deltas = v2308.scaled_deltas(raw_deltas, eargs)
        _ls, rs = v2308.output_residual(model, x, y, str(fargs.part_f_residual_target))
        _lg, rg = v2308.output_residual(model, xg, yg, str(fargs.part_f_residual_target))
        alpha = 1.0
        if c_scheme == "C10_Candidate_trust_free_fixed_alpha":
            alpha = float(args.part_c_fixed_alpha)
        source_pred = predicted_output_delta_from_deltas(model, x, deltas, alpha)
        guard_pred = predicted_output_delta_from_deltas(model, xg, deltas, alpha)
        out["source_residual_fit_improvement"] = residual_fit_ratio_from_prediction(source_pred, rs)
        out["guard_residual_fit_improvement"] = residual_fit_ratio_from_prediction(guard_pred, rg)
        out["residual_fit_probe_alpha"] = alpha
        out["residual_fit_probe_solve_residual_max"] = fval(fit_diag.get("solve_residual_max"))
        out["residual_fit_probe_condition_number_after_ridge_max"] = fval(fit_diag.get("condition_number_after_ridge_max"))
    except Exception as exc:
        out["source_residual_fit_improvement"] = "error"
        out["guard_residual_fit_improvement"] = "error"
        out["residual_fit_probe_error"] = repr(exc)
    return out


def margin_curvature_weights(logits: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, device: torch.device, *, seed: int, shuffle: bool) -> torch.Tensor:
    logits64 = logits.detach().to(device=device, dtype=torch.float64)
    yy = y.detach().to(device=device, dtype=torch.long)
    idx = torch.arange(int(logits64.shape[0]), device=device)
    correct = logits64[idx, yy]
    masked = logits64.clone()
    masked[idx, yy] = -torch.inf
    other = masked.max(dim=1).values
    margin = correct - other
    tau = float(getattr(args, "part_c_margin_curvature_tau", 0.0))
    temperature = max(1.0e-6, float(getattr(args, "part_c_margin_curvature_temperature", 1.0)))
    weights = torch.sigmoid((tau - margin) / temperature)
    weights = weights / weights.mean().clamp_min(1.0e-8)
    if shuffle:
        gen = torch.Generator(device=device).manual_seed(23091000 + int(seed))
        weights = weights[torch.randperm(int(weights.shape[0]), device=device, generator=gen)]
    return weights.reshape(-1, 1).to(dtype=torch.float64)


def fisher_curvature_weights(logits: torch.Tensor, args: argparse.Namespace, device: torch.device, *, seed: int, shuffle: bool) -> torch.Tensor:
    logits64 = logits.detach().to(device=device, dtype=torch.float64)
    temperature = max(1.0e-6, float(getattr(args, "part_c_fisher_curvature_temperature", 1.0)))
    prob = torch.softmax(logits64 / temperature, dim=1).to(dtype=torch.float64)
    eps = max(1.0e-12, float(getattr(args, "part_c_fisher_curvature_eps", 1.0e-3)))
    strength = min(1.0, max(0.0, float(getattr(args, "part_c_fisher_curvature_strength", 0.5))))
    max_weight = max(1.0, float(getattr(args, "part_c_fisher_curvature_max_weight", 3.0)))
    fisher_diag = (prob * (1.0 - prob)).clamp_min(eps)
    normalized_curvature = fisher_diag / fisher_diag.mean().clamp_min(eps)
    raw = torch.rsqrt(1.0 + strength * normalized_curvature)
    raw = raw.clamp(max=max_weight)
    raw = raw / raw.mean().clamp_min(1.0e-8)
    weights = raw
    if shuffle:
        gen = torch.Generator(device=device).manual_seed(23092000 + int(seed))
        weights = weights[torch.randperm(int(weights.shape[0]), device=device, generator=gen)]
    return weights.to(dtype=torch.float64)


def scaled_extra_metric(metric: torch.Tensor, reference_normal: torch.Tensor, strength: float) -> torch.Tensor:
    ref_trace = torch.trace(reference_normal.to(dtype=torch.float64)).abs().clamp_min(1.0e-12)
    metric_trace = torch.trace(metric.to(dtype=torch.float64)).abs().clamp_min(1.0e-12)
    return float(strength) * (ref_trace / metric_trace) * metric.to(dtype=torch.float64)


def solve_downstream_guard_penalized(
    model: Any,
    source_acts: list[torch.Tensor],
    guard_acts: list[torch.Tensor],
    layer_id: int,
    residual: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    mode: str,
) -> tuple[torch.Tensor, dcerif.DownstreamOperator, dict[str, Any]]:
    source_op = dcerif.make_downstream_operator(model, source_acts, int(layer_id))
    h_source = source_op.explicit_matrix()
    gram = v2308.v2307.expand_edge_gram(model, int(layer_id), str(args.basis_key), 0.0, args, device=device)
    metric = dcerif.expanded_edge_metric(gram, source_op.layer_output_dim)
    source_normal = h_source.T @ h_source
    normal = source_normal + float(args.part_d_lambda) * metric
    penalty_strength = max(0.0, float(getattr(args, "part_c_guard_penalty_strength", 0.1)))
    penalty_trace = 0.0
    if mode == "guard_penalized_local_inverse":
        guard_op = dcerif.make_downstream_operator(model, guard_acts, int(layer_id))
        h_guard = guard_op.explicit_matrix()
        scale = float(h_source.shape[0]) / float(max(1, int(h_guard.shape[0])))
        penalty = penalty_strength * scale * (h_guard.T @ h_guard)
        normal = normal + penalty
        penalty_trace = float(torch.trace(penalty).detach().cpu().item())
    elif mode == "extra_ridge_local_inverse":
        penalty = scaled_extra_metric(metric, source_normal, penalty_strength)
        normal = normal + penalty
        penalty_trace = float(torch.trace(penalty).detach().cpu().item())
    rhs = h_source.T @ residual.to(device=device, dtype=torch.float64).reshape(-1, 1)
    flat, diag = dcerif.solve_ridge_normal_exact(normal, rhs)
    _mn, _mx, cond = v2308.eig_condition(normal)
    diag.update(
        {
            "condition_number_after_ridge": cond,
            "cg_iterations": 0.0,
            "rank_H": float(torch.linalg.matrix_rank(h_source, tol=1.0e-8).detach().cpu().item()),
            "guard_penalty_strength": penalty_strength,
            "guard_penalty_trace": penalty_trace,
            "guard_penalty_mode": mode,
        }
    )
    return flat.reshape(source_op.basis_dim, source_op.layer_output_dim), source_op, diag


def solve_local_hidden_guard_penalized(
    model: Any,
    source_acts: list[torch.Tensor],
    guard_acts: list[torch.Tensor],
    hidden_target: torch.Tensor,
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    phi_source = model.layer_phi(source_acts[int(layer_id)], int(layer_id)).to(dtype=torch.float64)
    gram = v2308.v2307.expand_edge_gram(model, int(layer_id), str(args.basis_key), 0.0, args, device=device)
    source_normal = phi_source.T @ phi_source
    normal = source_normal + float(args.part_d_lambda) * gram.to(device=device, dtype=torch.float64)
    penalty_strength = max(0.0, float(getattr(args, "part_c_guard_penalty_strength", 0.1)))
    penalty_trace = 0.0
    if mode == "guard_penalized_local_inverse":
        phi_guard = model.layer_phi(guard_acts[int(layer_id)], int(layer_id)).to(dtype=torch.float64)
        scale = float(phi_source.shape[0]) / float(max(1, int(phi_guard.shape[0])))
        penalty = penalty_strength * scale * (phi_guard.T @ phi_guard)
        normal = normal + penalty
        penalty_trace = float(torch.trace(penalty).detach().cpu().item())
    elif mode == "extra_ridge_local_inverse":
        penalty = scaled_extra_metric(gram, source_normal, penalty_strength)
        normal = normal + penalty
        penalty_trace = float(torch.trace(penalty).detach().cpu().item())
    rhs = phi_source.T @ hidden_target.to(device=device, dtype=torch.float64)
    sol, diag = dcerif.solve_ridge_normal_exact(normal, rhs)
    _mn, _mx, cond = v2308.eig_condition(normal)
    diag.update(
        {
            "condition_number_after_ridge": cond,
            "cg_iterations": 0.0,
            "rank_Phi": float(torch.linalg.matrix_rank(phi_source, tol=1.0e-8).detach().cpu().item()),
            "guard_penalty_strength": penalty_strength,
            "guard_penalty_trace": penalty_trace,
            "guard_penalty_mode": mode,
        }
    )
    return sol, diag


def mask_delta_spectral_modes(delta: torch.Tensor, k: int, *, mode: str, fraction: float) -> tuple[torch.Tensor, dict[str, float]]:
    kk = max(1, int(k))
    rows = int(delta.shape[0]) // kk
    if rows <= 0:
        return delta, {
            "spectral_mode_k": float(kk),
            "spectral_mode_keep_count": 0.0,
            "spectral_mode_energy_retained": 0.0,
        }
    keep_count = max(1, min(kk, int(round(float(fraction) * float(kk)))))
    shaped = delta.detach().clone().reshape(rows, kk, -1)
    mask = torch.zeros((kk,), device=delta.device, dtype=delta.dtype)
    if str(mode) == "tail":
        mask[kk - keep_count :] = 1.0
    else:
        mask[:keep_count] = 1.0
    masked = shaped * mask.reshape(1, kk, 1)
    before = shaped.to(dtype=torch.float64).square().sum().clamp_min(1.0e-12)
    after = masked.to(dtype=torch.float64).square().sum()
    return masked.reshape_as(delta), {
        "spectral_mode_k": float(kk),
        "spectral_mode_keep_count": float(keep_count),
        "spectral_mode_fraction": float(fraction),
        "spectral_mode_energy_retained": float((after / before).detach().cpu().item()),
    }


def c_adjoint_variant_delta(
    variant: str,
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    target: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
    task: str,
    checkpoint: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    eargs = argparse.Namespace(**vars(args))
    eargs.part_d_lambda = float(args.part_e_lambda)
    last = len(model.coeffs) - 1
    pen = max(0, last - 1)
    fit_x = torch.cat([x, xg], dim=0) if variant == "source_guard_population" else x
    fit_y = torch.cat([y, yg], dim=0) if variant == "source_guard_population" else y
    fit_logits, acts = model.forward_with_activations(fit_x)
    guard_logits = None
    guard_acts = None
    if variant in {"guard_penalized_local_inverse", "extra_ridge_local_inverse"}:
        guard_logits, guard_acts = model.forward_with_activations(xg)
    residual_override = None
    weight_diag: dict[str, Any] = {}
    if variant in {"margin_curvature", "shuffled_margin_weight"}:
        _base_logits, residual = v2308.output_residual(model, fit_x, fit_y, target)
        weights = margin_curvature_weights(fit_logits, fit_y, args, device, seed=int(seed), shuffle=variant == "shuffled_margin_weight")
        residual_override = residual.to(device=device, dtype=torch.float64) * weights
        weight_diag = {
            "margin_curvature_weight_mean": float(weights.mean().detach().cpu().item()),
            "margin_curvature_weight_min": float(weights.min().detach().cpu().item()),
            "margin_curvature_weight_max": float(weights.max().detach().cpu().item()),
            "margin_curvature_weight_std": float(weights.std(unbiased=False).detach().cpu().item()),
            "margin_curvature_weight_shuffled": int(variant == "shuffled_margin_weight"),
        }
    if variant in {"diagonal_fisher_curvature", "shuffled_fisher_weight"}:
        _base_logits, residual = v2308.output_residual(model, fit_x, fit_y, target)
        weights = fisher_curvature_weights(fit_logits, args, device, seed=int(seed), shuffle=variant == "shuffled_fisher_weight")
        base = residual.to(device=device, dtype=torch.float64)
        weighted = base * weights
        base_rms = base.square().mean().sqrt().clamp_min(1.0e-12)
        weighted_rms = weighted.square().mean().sqrt().clamp_min(1.0e-12)
        residual_override = weighted * (base_rms / weighted_rms)
        weight_diag = {
            "fisher_curvature_weight_mean": float(weights.mean().detach().cpu().item()),
            "fisher_curvature_weight_min": float(weights.min().detach().cpu().item()),
            "fisher_curvature_weight_max": float(weights.max().detach().cpu().item()),
            "fisher_curvature_weight_std": float(weights.std(unbiased=False).detach().cpu().item()),
            "fisher_curvature_weight_shuffled": int(variant == "shuffled_fisher_weight"),
            "fisher_curvature_strength": float(getattr(args, "part_c_fisher_curvature_strength", 0.5)),
            "fisher_curvature_temperature": float(getattr(args, "part_c_fisher_curvature_temperature", 1.0)),
            "fisher_curvature_eps": float(getattr(args, "part_c_fisher_curvature_eps", 1.0e-3)),
            "fisher_curvature_max_weight": float(getattr(args, "part_c_fisher_curvature_max_weight", 3.0)),
            "fisher_curvature_mode": "damped_normalized_diagonal",
            "fisher_curvature_residual_rms_before": float(base_rms.detach().cpu().item()),
            "fisher_curvature_residual_rms_after": float(residual_override.square().mean().sqrt().detach().cpu().item()),
        }
    if variant in {"guard_penalized_local_inverse", "extra_ridge_local_inverse"}:
        _base_logits, rr = v2308.output_residual(model, fit_x, fit_y, target)
        delta_last, op_last, dlast = solve_downstream_guard_penalized(model, acts, guard_acts or [], last, rr, eargs, device, mode=variant)
    else:
        delta_last, op_last, rr, dlast = v2308.solve_downstream(model, fit_x, fit_y, last, target, eargs, device, seed=int(seed), residual_override=residual_override)
    leftover = rr - op_last.apply_H(delta_last).detach()
    hidden_target = v2308.downstream_adjoint_hidden_target(model, acts[pen + 1], pen + 1, leftover)
    gen = torch.Generator(device=device).manual_seed(23090000 + int(seed) * 1009 + sum(ord(c) for c in task + checkpoint + variant))
    if variant == "shuffled_downstream_adjoint":
        perm = torch.randperm(int(hidden_target.shape[0]), generator=gen, device=device)
        hidden_target = hidden_target[perm]
    if variant == "adjoint_no_inverse":
        phi = model.layer_phi(acts[pen], pen)
        delta_pen = v2308.v2307.gradient_like_delta(phi, hidden_target, None)
        fit = v2308.v2307.improvement_metrics(phi, hidden_target.to(device=phi.device, dtype=torch.float64), delta_pen)
        dpen = {
            "solve_residual": 0.0,
            "condition_number_after_ridge": 0.0,
            "predicted_guard_fit_mean": fval(fit.get("fit_improvement_ratio")),
            "rank_Phi": float(torch.linalg.matrix_rank(phi.to(dtype=torch.float64), tol=1.0e-8).detach().cpu().item()),
        }
        solver_type = "last_exact_plus_adjoint_gradient_no_local_inverse"
    else:
        if variant in {"guard_penalized_local_inverse", "extra_ridge_local_inverse"}:
            delta_pen, dpen = solve_local_hidden_guard_penalized(model, acts, guard_acts or [], hidden_target, pen, eargs, device, mode=variant)
        else:
            delta_pen, dpen = v2308.solve_local_hidden(model, acts, hidden_target, pen, eargs, device)
        solver_type = "last_exact_plus_shuffled_adjoint_local" if variant == "shuffled_downstream_adjoint" else f"last_exact_plus_{variant}_adjoint_local"
    spectral_diag: dict[str, Any] = {}
    if variant in {"spectral_smooth_modes", "spectral_tail_modes"}:
        mode = "tail" if variant == "spectral_tail_modes" else "smooth"
        fraction = float(getattr(args, "part_c_spectral_mode_fraction", 2.0 / 3.0))
        raw_last_norm = float(delta_last.detach().to(dtype=torch.float64).norm().detach().cpu().item())
        raw_pen_norm = float(delta_pen.detach().to(dtype=torch.float64).norm().detach().cpu().item())
        delta_last, last_sdiag = mask_delta_spectral_modes(delta_last, int(getattr(model, "k", 1)), mode=mode, fraction=fraction)
        delta_pen, pen_sdiag = mask_delta_spectral_modes(delta_pen, int(getattr(model, "k", 1)), mode=mode, fraction=fraction)
        spectral_diag = {
            "spectral_mode_variant": mode,
            "spectral_mode_fraction": fraction,
            "spectral_mode_keep_count": last_sdiag.get("spectral_mode_keep_count", 0.0),
            "spectral_mode_k": last_sdiag.get("spectral_mode_k", 0.0),
            "spectral_last_energy_retained": last_sdiag.get("spectral_mode_energy_retained", 0.0),
            "spectral_penultimate_energy_retained": pen_sdiag.get("spectral_mode_energy_retained", 0.0),
            "spectral_last_delta_norm_before": raw_last_norm,
            "spectral_penultimate_delta_norm_before": raw_pen_norm,
            "spectral_last_delta_norm_after": float(delta_last.detach().to(dtype=torch.float64).norm().detach().cpu().item()),
            "spectral_penultimate_delta_norm_after": float(delta_pen.detach().to(dtype=torch.float64).norm().detach().cpu().item()),
        }
    return {last: delta_last, pen: delta_pen}, {
        "layer_group": f"topdown_last_penultimate_{variant}",
        "solver_type": solver_type,
        "selected_layers": f"{last},{pen}",
        "solve_residual_max": max(fval(dlast.get("solve_residual")), fval(dpen.get("solve_residual"))),
        "condition_number_after_ridge_max": max(fval(dlast.get("condition_number_after_ridge")), fval(dpen.get("condition_number_after_ridge"))),
        "rank_H_max": fval(dlast.get("rank_H")),
        "rank_Phi_max": fval(dpen.get("rank_Phi")),
        "adjoint_hidden_target_norm": float(hidden_target.norm().detach().cpu().item()),
        "component_variant": variant,
        "population_residual_sample_count": int(fit_x.shape[0]),
        "guard_penalty_strength": fval(dlast.get("guard_penalty_strength")),
        "guard_penalty_trace_last": fval(dlast.get("guard_penalty_trace")),
        "guard_penalty_trace_penultimate": fval(dpen.get("guard_penalty_trace")),
        **weight_diag,
        **spectral_diag,
    }


def run_c_adjoint_variant_row(c_scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device, *, variant: str, selected_f_scheme: str) -> dict[str, Any]:
    interval = 30 if "every30" in selected_f_scheme or "F28" in selected_f_scheme else 20
    internal_scheme = f"C_internal_{variant}_every{interval}"
    fargs = c_args_from_f_args(args, c_scheme=c_scheme, source_f_scheme=internal_scheme)
    original_map = v2308.f_scheme_to_e_scheme
    original_delta = v2308.e_delta_candidate
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_adjoint = internal_scheme in v2308.ADJOINT_LOCAL_HYBRID_SCHEMES
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES

    def patched_map(scheme: str) -> tuple[str, int | None]:
        if scheme == internal_scheme:
            return "E9_topdown_last_penultimate_adjoint_local_one_step", None
        return original_map(scheme)

    def patched_delta(e_scheme: str, model: Any, x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, target: str, eargs: argparse.Namespace, edev: torch.device, *, seed: int, task: str, checkpoint: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        if e_scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
            return c_adjoint_variant_delta(variant, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)
        return original_delta(e_scheme, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)

    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = interval
        v2308.ADJOINT_LOCAL_HYBRID_SCHEMES.add(internal_scheme)
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        v2308.f_scheme_to_e_scheme = patched_map
        v2308.e_delta_candidate = patched_delta
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.f_scheme_to_e_scheme = original_map
        v2308.e_delta_candidate = original_delta
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_adjoint:
            v2308.ADJOINT_LOCAL_HYBRID_SCHEMES.discard(internal_scheme)
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    note = "downstream adjoint gradient proxy without local inverse" if variant == "adjoint_no_inverse" else "deterministically shuffled downstream adjoint before local inverse"
    annotated = annotate_c_row(row, c_scheme=c_scheme, source_scheme=internal_scheme, ablation_note=note)
    return add_c_one_step_fit_metrics(annotated, c_scheme=c_scheme, source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device, variant=variant)


def run_c_layerwise_projection_row(c_scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device, *, selected_f_scheme: str) -> dict[str, Any]:
    fargs = c_args_from_f_args(args, c_scheme="C6_DownstreamAdjoint_LocalInverse_ParetoTrust", source_f_scheme=selected_f_scheme)
    original_choose = v2308.choose_alpha_for_f_deltas
    try:
        v2308.choose_alpha_for_f_deltas = choose_layerwise_projected_f_deltas
        row = v2308.run_f_row((selected_f_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.choose_alpha_for_f_deltas = original_choose
    annotated = annotate_c_row(
        row,
        c_scheme=c_scheme,
        source_scheme=selected_f_scheme,
        ablation_note="candidate with Pareto trust implemented as per-layer alpha projection toward the EFRF step",
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C6_DownstreamAdjoint_LocalInverse_ParetoTrust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["component_ablation"] = "candidate with Pareto trust implemented as per-layer alpha projection toward the EFRF step"
    return out


def run_c_reference_projection_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
    require_no_debt: bool = False,
    protect_between_optimizer: bool = False,
) -> dict[str, Any]:
    fargs = c_args_from_f_args(args, c_scheme="C6_DownstreamAdjoint_LocalInverse_ParetoTrust", source_f_scheme=selected_f_scheme)
    original_choose = v2308.choose_alpha_for_f_deltas
    original_edge = v2308.choose_alpha_edge_optimizer_f
    reference_source: dict[str, float] | None = None
    reference_guard: dict[str, float] | None = None

    def patched_choose(
        model: Any,
        base_state: dict[str, torch.Tensor],
        deltas: dict[int, torch.Tensor],
        source_before: dict[str, float],
        guard_before: dict[str, float],
        x: torch.Tensor,
        y: torch.Tensor,
        xg: torch.Tensor,
        yg: torch.Tensor,
        choose_args: argparse.Namespace,
        base_acts: list[torch.Tensor],
        tail99_reference: dict[str, float] | None = None,
    ) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
        nonlocal reference_source, reference_guard
        if reference_source is None:
            reference_source = dict(source_before)
        if reference_guard is None:
            reference_guard = dict(guard_before)
        projected = choose_reference_pareto_projected_f_deltas(
            model,
            base_state,
            deltas,
            source_before,
            guard_before,
            reference_source,
            reference_guard,
            x,
            y,
            xg,
            yg,
            choose_args,
            base_acts,
            tail99_reference=tail99_reference,
            require_no_debt=require_no_debt,
        )
        if int(projected[1]):
            return projected
        if require_no_debt:
            return projected
        return original_choose(
            model,
            base_state,
            deltas,
            source_before,
            guard_before,
            x,
            y,
            xg,
            yg,
            choose_args,
            base_acts,
            tail99_reference=tail99_reference,
        )

    def patched_edge_optimizer(
        model: Any,
        base_state: dict[str, torch.Tensor],
        source_before: dict[str, float],
        guard_before: dict[str, float],
        x: torch.Tensor,
        y: torch.Tensor,
        xg: torch.Tensor,
        yg: torch.Tensor,
        edge_args: argparse.Namespace,
        *,
        use_population_gate: bool,
        adamw: bool = False,
        base_acts: list[torch.Tensor] | None = None,
        guard_logits_before: torch.Tensor | None = None,
        alignment_reference: torch.Tensor | None = None,
        alignment_records: list[float] | None = None,
        alignment_soft_scale_records: list[float] | None = None,
        tail99_reference: dict[str, float] | None = None,
    ) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
        nonlocal reference_source, reference_guard
        if not protect_between_optimizer:
            return original_edge(
                model,
                base_state,
                source_before,
                guard_before,
                x,
                y,
                xg,
                yg,
                edge_args,
                use_population_gate=use_population_gate,
                adamw=adamw,
                base_acts=base_acts,
                guard_logits_before=guard_logits_before,
                alignment_reference=alignment_reference,
                alignment_records=alignment_records,
                alignment_soft_scale_records=alignment_soft_scale_records,
                tail99_reference=tail99_reference,
            )
        if reference_source is None:
            reference_source = dict(source_before)
        if reference_guard is None:
            reference_guard = dict(guard_before)
        reject_reason = "no_optimizer_alpha_candidates"
        for alpha in v2308.float_items(edge_args.part_f_alphas):
            one_alpha_args = argparse.Namespace(**vars(edge_args))
            one_alpha_args.part_f_alphas = str(float(alpha))
            out = original_edge(
                model,
                base_state,
                source_before,
                guard_before,
                x,
                y,
                xg,
                yg,
                one_alpha_args,
                use_population_gate=use_population_gate,
                adamw=adamw,
                base_acts=base_acts,
                guard_logits_before=guard_logits_before,
                alignment_reference=alignment_reference,
                alignment_records=alignment_records,
                alignment_soft_scale_records=alignment_soft_scale_records,
                tail99_reference=tail99_reference,
            )
            selected_alpha, accepted, reason, source_after, guard_after = out
            if int(accepted):
                debt = v2308.debt_deltas(reference_guard, guard_after)
                if v2308.no_debt_ok(debt, float(edge_args.no_debt_budget)):
                    return float(selected_alpha), 1, "optimizer_full_path_nodebt_accepted", source_after, guard_after
                reject_reason = "no_debt_reject_optimizer_full_path"
                v2308.load_model_state(model, base_state)
                continue
            reject_reason = str(reason)
        v2308.load_model_state(model, base_state)
        return 0.0, 0, reject_reason, source_before, guard_before

    try:
        v2308.choose_alpha_for_f_deltas = patched_choose
        v2308.choose_alpha_edge_optimizer_f = patched_edge_optimizer
        row = v2308.run_f_row((selected_f_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.choose_alpha_for_f_deltas = original_choose
        v2308.choose_alpha_edge_optimizer_f = original_edge
    ablation_note = (
        "candidate with full-path reference Pareto/no-debt feasibility over EFRF refresh and between-step optimizer updates"
        if protect_between_optimizer
        else "candidate with strict reference Pareto projection plus no-debt feasibility in the guard metrics, without fallback to the original selector"
        if require_no_debt
        else "candidate with Pareto trust projected against initial source/guard feasible region with per-layer alpha candidates"
    )
    annotated = annotate_c_row(
        row,
        c_scheme=c_scheme,
        source_scheme=selected_f_scheme,
        ablation_note=ablation_note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C6_DownstreamAdjoint_LocalInverse_ParetoTrust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["component_ablation"] = ablation_note
    return out


def terminal_audit_transform(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(row)
    audit_pass = int(ival(out.get("F5_no_debt_pass")) and ival(out.get("component_non_positive")))
    out["terminal_audit_pass"] = audit_pass
    out["terminal_audit_reject"] = int(not audit_pass)
    if audit_pass:
        return out
    zero_fields = [
        "C2_coverage_improvement",
        "C2_accuracy_improvement",
        "local_patch_coverage_improvement",
        "rotation_coverage_improvement",
        "guard_nll_delta",
        "train_nll_delta",
        "source_guard_c2_gap",
        "Brier_delta",
        "ECE_delta",
        "tail95_delta",
        "tail99_delta",
    ]
    for key in zero_fields:
        out[key] = 0.0
    out["one_step_gain_retention_curve"] = json.dumps([0.0])
    out["F5_no_debt_pass"] = 1
    out["component_non_positive"] = 1
    out["margin10_delta_correct_sign"] = 1
    out["finite_step_accept_count"] = 0
    out["finite_step_accept_rate"] = 0.0
    out["finite_step_scale_mean"] = 0.0
    out["finite_step_skip_count"] = int(args.part_c_steps)
    out["finite_step_reject_reasons"] = json.dumps({"terminal_audit_rollback": 1}, sort_keys=True)
    out["terminal_audit_original_C2_coverage_improvement"] = row.get("C2_coverage_improvement", "")
    out["terminal_audit_original_tail99_delta"] = row.get("tail99_delta", "")
    return out


def run_c_terminal_audited_no_trust_row(c_scheme: str, task: str, seed: int, args: argparse.Namespace, device: torch.device, *, selected_f_scheme: str) -> dict[str, Any]:
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=selected_f_scheme)
    row = v2308.run_f_row((selected_f_scheme, task, int(seed)), fargs, device)
    audited = terminal_audit_transform(row, args)
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=selected_f_scheme,
        ablation_note="candidate with downstream/local-inverse trajectory accepted by terminal train/guard no-debt audit and rolled back on audit failure",
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["component_ablation"] = "candidate with downstream/local-inverse trajectory accepted by terminal train/guard no-debt audit and rolled back on audit failure"
    return out


def run_c_terminal_audited_adjoint_local_interval_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
    interval: int,
) -> dict[str, Any]:
    internal_scheme = f"C_internal_adjoint_local_every{int(interval)}_terminal_audit"
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=internal_scheme)
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_adjoint = internal_scheme in v2308.ADJOINT_LOCAL_HYBRID_SCHEMES
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES
    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = int(interval)
        v2308.ADJOINT_LOCAL_HYBRID_SCHEMES.add(internal_scheme)
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_adjoint:
            v2308.ADJOINT_LOCAL_HYBRID_SCHEMES.discard(internal_scheme)
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    audited = terminal_audit_transform(row, args)
    note = f"candidate with adjoint-local downstream refresh every {int(interval)} steps, no stepwise trust, and terminal train/guard no-debt audit with rollback"
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=internal_scheme,
        ablation_note=note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = internal_scheme
    out["component_ablation"] = note
    return out


def run_c_terminal_audited_source_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    source_f_scheme: str,
    ablation_note: str,
) -> dict[str, Any]:
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=source_f_scheme)
    row = v2308.run_f_row((source_f_scheme, task, int(seed)), fargs, device)
    audited = terminal_audit_transform(row, args)
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=source_f_scheme,
        ablation_note=ablation_note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=source_f_scheme, task=task, seed=seed, args=args, device=device)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = source_f_scheme
    out["component_ablation"] = ablation_note
    return out


def run_c_terminal_audited_source_guard_population_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
) -> dict[str, Any]:
    interval = 30 if "every30" in selected_f_scheme or "F28" in selected_f_scheme else 20
    internal_scheme = f"C_internal_source_guard_population_every{interval}"
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=internal_scheme)
    original_map = v2308.f_scheme_to_e_scheme
    original_delta = v2308.e_delta_candidate
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES

    def patched_map(scheme: str) -> tuple[str, int | None]:
        if scheme == internal_scheme:
            return "E9_topdown_last_penultimate_adjoint_local_one_step", None
        return original_map(scheme)

    def patched_delta(e_scheme: str, model: Any, x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, target: str, eargs: argparse.Namespace, edev: torch.device, *, seed: int, task: str, checkpoint: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        if e_scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
            return c_adjoint_variant_delta("source_guard_population", model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)
        return original_delta(e_scheme, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)

    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = interval
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        v2308.f_scheme_to_e_scheme = patched_map
        v2308.e_delta_candidate = patched_delta
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.f_scheme_to_e_scheme = original_map
        v2308.e_delta_candidate = original_delta
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    audited = terminal_audit_transform(row, args)
    note = "diagnostic candidate whose downstream/local inverse residual is fit on concatenated train+guard population data, with no stepwise trust and terminal train/guard no-debt audit"
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=internal_scheme,
        ablation_note=note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device, variant="source_guard_population")
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = internal_scheme
    out["component_ablation"] = note
    out["guard_used_in_update_objective"] = 1
    out["promotion_allowed"] = 0
    return out


def run_c_terminal_audited_margin_curvature_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
    shuffled_weight: bool = False,
) -> dict[str, Any]:
    interval = 30 if "every30" in selected_f_scheme or "F28" in selected_f_scheme else 20
    variant = "shuffled_margin_weight" if shuffled_weight else "margin_curvature"
    internal_scheme = f"C_internal_{variant}_every{interval}"
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=internal_scheme)
    original_map = v2308.f_scheme_to_e_scheme
    original_delta = v2308.e_delta_candidate
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES

    def patched_map(scheme: str) -> tuple[str, int | None]:
        if scheme == internal_scheme:
            return "E9_topdown_last_penultimate_adjoint_local_one_step", None
        return original_map(scheme)

    def patched_delta(e_scheme: str, model: Any, x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, target: str, eargs: argparse.Namespace, edev: torch.device, *, seed: int, task: str, checkpoint: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        if e_scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
            return c_adjoint_variant_delta(variant, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)
        return original_delta(e_scheme, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)

    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = interval
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        v2308.f_scheme_to_e_scheme = patched_map
        v2308.e_delta_candidate = patched_delta
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.f_scheme_to_e_scheme = original_map
        v2308.e_delta_candidate = original_delta
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    audited = terminal_audit_transform(row, args)
    note = (
        "control with train-only margin-curvature weights deterministically shuffled across source samples before downstream/local inverse, terminal-audited"
        if shuffled_weight
        else "candidate with train-only margin-curvature weighted residual inverse, no stepwise trust, and terminal train/guard no-debt audit"
    )
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=internal_scheme,
        ablation_note=note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device, variant=variant)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = internal_scheme
    out["component_ablation"] = note
    out["guard_used_in_update_objective"] = 0
    out["margin_curvature_tau"] = float(getattr(args, "part_c_margin_curvature_tau", 0.0))
    out["margin_curvature_temperature"] = float(getattr(args, "part_c_margin_curvature_temperature", 1.0))
    out["margin_curvature_weight_shuffled"] = int(shuffled_weight)
    return out


def run_c_terminal_audited_fisher_curvature_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
    shuffled_weight: bool = False,
) -> dict[str, Any]:
    interval = 30 if "every30" in selected_f_scheme or "F28" in selected_f_scheme else 20
    variant = "shuffled_fisher_weight" if shuffled_weight else "diagonal_fisher_curvature"
    internal_scheme = f"C_internal_{variant}_every{interval}"
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=internal_scheme)
    original_map = v2308.f_scheme_to_e_scheme
    original_delta = v2308.e_delta_candidate
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES

    def patched_map(scheme: str) -> tuple[str, int | None]:
        if scheme == internal_scheme:
            return "E9_topdown_last_penultimate_adjoint_local_one_step", None
        return original_map(scheme)

    def patched_delta(e_scheme: str, model: Any, x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, target: str, eargs: argparse.Namespace, edev: torch.device, *, seed: int, task: str, checkpoint: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        if e_scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
            return c_adjoint_variant_delta(variant, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)
        return original_delta(e_scheme, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)

    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = interval
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        v2308.f_scheme_to_e_scheme = patched_map
        v2308.e_delta_candidate = patched_delta
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.f_scheme_to_e_scheme = original_map
        v2308.e_delta_candidate = original_delta
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    audited = terminal_audit_transform(row, args)
    note = (
        "control with train-only damped diagonal-Fisher/probability-curvature weights deterministically shuffled across source samples before downstream/local inverse, terminal-audited"
        if shuffled_weight
        else "candidate with train-only damped diagonal-Fisher/probability-curvature weighted residual inverse, residual RMS preserved, no stepwise trust, and terminal train/guard no-debt audit"
    )
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=internal_scheme,
        ablation_note=note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device, variant=variant)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = internal_scheme
    out["component_ablation"] = note
    out["guard_used_in_update_objective"] = 0
    out["fisher_curvature_strength"] = float(getattr(args, "part_c_fisher_curvature_strength", 0.5))
    out["fisher_curvature_temperature"] = float(getattr(args, "part_c_fisher_curvature_temperature", 1.0))
    out["fisher_curvature_eps"] = float(getattr(args, "part_c_fisher_curvature_eps", 1.0e-3))
    out["fisher_curvature_max_weight"] = float(getattr(args, "part_c_fisher_curvature_max_weight", 3.0))
    out["fisher_curvature_weight_shuffled"] = int(shuffled_weight)
    return out


def run_c_terminal_audited_guard_penalty_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
    control_extra_ridge: bool = False,
) -> dict[str, Any]:
    interval = 30 if "every30" in selected_f_scheme or "F28" in selected_f_scheme else 20
    variant = "extra_ridge_local_inverse" if control_extra_ridge else "guard_penalized_local_inverse"
    internal_scheme = f"C_internal_{variant}_every{interval}"
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=internal_scheme)
    original_map = v2308.f_scheme_to_e_scheme
    original_delta = v2308.e_delta_candidate
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES

    def patched_map(scheme: str) -> tuple[str, int | None]:
        if scheme == internal_scheme:
            return "E9_topdown_last_penultimate_adjoint_local_one_step", None
        return original_map(scheme)

    def patched_delta(e_scheme: str, model: Any, x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, target: str, eargs: argparse.Namespace, edev: torch.device, *, seed: int, task: str, checkpoint: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        if e_scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
            return c_adjoint_variant_delta(variant, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)
        return original_delta(e_scheme, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)

    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = interval
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        v2308.f_scheme_to_e_scheme = patched_map
        v2308.e_delta_candidate = patched_delta
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.f_scheme_to_e_scheme = original_map
        v2308.e_delta_candidate = original_delta
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    audited = terminal_audit_transform(row, args)
    note = (
        "control with same-strength extra metric ridge instead of guard operator penalty, terminal-audited"
        if control_extra_ridge
        else "candidate with train/guard guard-operator penalty inside downstream and local hidden normal equations, terminal-audited"
    )
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=internal_scheme,
        ablation_note=note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device, variant=variant)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = internal_scheme
    out["component_ablation"] = note
    out["guard_used_in_update_objective"] = int(not control_extra_ridge)
    out["guard_penalty_strength"] = float(getattr(args, "part_c_guard_penalty_strength", 0.1))
    out["guard_penalty_control_extra_ridge"] = int(control_extra_ridge)
    out["promotion_allowed"] = 0
    return out


def run_c_terminal_audited_spectral_mode_row(
    c_scheme: str,
    task: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    *,
    selected_f_scheme: str,
    tail_control: bool = False,
) -> dict[str, Any]:
    interval = 30 if "every30" in selected_f_scheme or "F28" in selected_f_scheme else 20
    variant = "spectral_tail_modes" if tail_control else "spectral_smooth_modes"
    internal_scheme = f"C_internal_{variant}_every{interval}"
    fargs = c_args_from_f_args(args, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_f_scheme=internal_scheme)
    original_map = v2308.f_scheme_to_e_scheme
    original_delta = v2308.e_delta_candidate
    old_interval = v2308.HYBRID_REFRESH_INTERVALS.get(internal_scheme)
    had_candidate = internal_scheme in v2308.F_CANDIDATE_SCHEMES

    def patched_map(scheme: str) -> tuple[str, int | None]:
        if scheme == internal_scheme:
            return "E9_topdown_last_penultimate_adjoint_local_one_step", None
        return original_map(scheme)

    def patched_delta(e_scheme: str, model: Any, x: torch.Tensor, y: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, target: str, eargs: argparse.Namespace, edev: torch.device, *, seed: int, task: str, checkpoint: str) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        if e_scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
            return c_adjoint_variant_delta(variant, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)
        return original_delta(e_scheme, model, x, y, xg, yg, target, eargs, edev, seed=seed, task=task, checkpoint=checkpoint)

    try:
        v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = interval
        v2308.F_CANDIDATE_SCHEMES.add(internal_scheme)
        v2308.f_scheme_to_e_scheme = patched_map
        v2308.e_delta_candidate = patched_delta
        row = v2308.run_f_row((internal_scheme, task, int(seed)), fargs, device)
    finally:
        v2308.f_scheme_to_e_scheme = original_map
        v2308.e_delta_candidate = original_delta
        if old_interval is None:
            v2308.HYBRID_REFRESH_INTERVALS.pop(internal_scheme, None)
        else:
            v2308.HYBRID_REFRESH_INTERVALS[internal_scheme] = old_interval
        if not had_candidate:
            v2308.F_CANDIDATE_SCHEMES.discard(internal_scheme)
    audited = terminal_audit_transform(row, args)
    note = (
        "control retaining the same number of high/tail Chebyshev coefficient modes before terminal audit"
        if tail_control
        else "candidate retaining smooth low/mid Chebyshev coefficient modes before terminal train/guard no-debt audit"
    )
    annotated = annotate_c_row(
        audited,
        c_scheme=c_scheme,
        source_scheme=internal_scheme,
        ablation_note=note,
    )
    out = add_c_one_step_fit_metrics(annotated, c_scheme="C5_DownstreamAdjoint_LocalInverse_no_trust", source_scheme=selected_f_scheme, task=task, seed=seed, args=args, device=device, variant=variant)
    out["c_scheme"] = c_scheme
    out["scheme"] = c_scheme
    out["source_scheme"] = internal_scheme
    out["component_ablation"] = note
    out["guard_used_in_update_objective"] = 0
    out["spectral_mode_fraction"] = float(getattr(args, "part_c_spectral_mode_fraction", 2.0 / 3.0))
    out["spectral_mode_tail_control"] = int(tail_control)
    out["promotion_allowed"] = 0
    return out


def run_c_row(job: tuple[str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    c_scheme, task, seed = job
    selected_f_scheme = selected_part_c_f_scheme()
    try:
        if c_scheme == "C4_DownstreamAdjoint_no_inverse":
            return run_c_adjoint_variant_row(c_scheme, task, seed, args, device, variant="adjoint_no_inverse", selected_f_scheme=selected_f_scheme)
        if c_scheme == "C9_Candidate_shuffled_downstream_adjoint":
            return run_c_adjoint_variant_row(c_scheme, task, seed, args, device, variant="shuffled_downstream_adjoint", selected_f_scheme=selected_f_scheme)
        if c_scheme == "C11_Candidate_layerwise_trust_projection":
            return run_c_layerwise_projection_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme)
        if c_scheme == "C12_Candidate_reference_pareto_projection":
            return run_c_reference_projection_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme)
        if c_scheme == "C13_Candidate_reference_pareto_nodebt_projection":
            return run_c_reference_projection_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, require_no_debt=True)
        if c_scheme == "C14_Candidate_fullpath_reference_pareto_nodebt_projection":
            return run_c_reference_projection_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, require_no_debt=True, protect_between_optimizer=True)
        if c_scheme == "C15_Candidate_terminal_audited_no_trust":
            return run_c_terminal_audited_no_trust_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme)
        if c_scheme == "C16_Candidate_terminal_audited_adjoint_local_every10":
            return run_c_terminal_audited_adjoint_local_interval_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, interval=10)
        if c_scheme == "C17_Candidate_terminal_audited_adjoint_local_every5":
            return run_c_terminal_audited_adjoint_local_interval_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, interval=5)
        if c_scheme == "C18_Candidate_terminal_audited_topdown_every10":
            return run_c_terminal_audited_source_row(
                c_scheme,
                task,
                seed,
                args,
                device,
                source_f_scheme="F15_hybrid_EFRF_every10_steps_FunctionalGram_between",
                ablation_note="candidate with topdown downstream refresh every 10 steps, FunctionalGram between steps, no stepwise trust, and terminal train/guard no-debt audit with rollback",
            )
        if c_scheme == "C19_Candidate_terminal_audited_source_guard_population":
            return run_c_terminal_audited_source_guard_population_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme)
        if c_scheme == "C20_Candidate_terminal_audited_margin_curvature":
            return run_c_terminal_audited_margin_curvature_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, shuffled_weight=False)
        if c_scheme == "C21_Control_terminal_audited_shuffled_margin_weight":
            return run_c_terminal_audited_margin_curvature_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, shuffled_weight=True)
        if c_scheme == "C22_Candidate_terminal_audited_diagonal_fisher":
            return run_c_terminal_audited_fisher_curvature_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, shuffled_weight=False)
        if c_scheme == "C23_Control_terminal_audited_shuffled_fisher_weight":
            return run_c_terminal_audited_fisher_curvature_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, shuffled_weight=True)
        if c_scheme == "C24_Candidate_terminal_audited_guard_penalized_local_inverse":
            return run_c_terminal_audited_guard_penalty_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, control_extra_ridge=False)
        if c_scheme == "C25_Control_terminal_audited_extra_ridge_local_inverse":
            return run_c_terminal_audited_guard_penalty_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, control_extra_ridge=True)
        if c_scheme == "C26_Candidate_terminal_audited_smooth_mode_local_inverse":
            return run_c_terminal_audited_spectral_mode_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, tail_control=False)
        if c_scheme == "C27_Control_terminal_audited_tail_mode_local_inverse":
            return run_c_terminal_audited_spectral_mode_row(c_scheme, task, seed, args, device, selected_f_scheme=selected_f_scheme, tail_control=True)
        if c_scheme in {"C5_DownstreamAdjoint_LocalInverse_no_trust", "C6_DownstreamAdjoint_LocalInverse_ParetoTrust", "C10_Candidate_trust_free_fixed_alpha"}:
            source_f = selected_f_scheme
        else:
            source_f = PART_C_TO_F_SCHEME[c_scheme]
        fargs = c_args_from_f_args(args, c_scheme=c_scheme, source_f_scheme=source_f)
        row = v2308.run_f_row((source_f, task, int(seed)), fargs, device)
        note = {
            "C5_DownstreamAdjoint_LocalInverse_no_trust": "candidate with Pareto trust disabled by accept-all coverage tolerance",
            "C6_DownstreamAdjoint_LocalInverse_ParetoTrust": "selected Part B candidate with normal Pareto trust",
            "C10_Candidate_trust_free_fixed_alpha": f"candidate with trust disabled and fixed alpha={args.part_c_fixed_alpha}",
        }.get(c_scheme, "reused fixed baseline/control from v23.08 row mechanics")
        annotated = annotate_c_row(row, c_scheme=c_scheme, source_scheme=source_f, ablation_note=note)
        return add_c_one_step_fit_metrics(annotated, c_scheme=c_scheme, source_scheme=source_f, task=task, seed=seed, args=args, device=device)
    except Exception as exc:
        return {"part": "C", "status": "error", "scheme": c_scheme, "c_scheme": c_scheme, "task": task, "seed": int(seed), "error_message": repr(exc)}


def part_c_run(args: argparse.Namespace) -> dict[str, Any]:
    init()
    pb = read_json(OUT_ROOT / "part_b_synthetic_lock_summary.json")
    if not ival(pb.get("gate_pass")):
        return v2308.blocked_summary("C", "C_BlockedByPartB", "part_b_not_passed")
    device = v2308.device_from_args(args)
    jobs = v2308.shard_items(part_c_jobs(args), args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"part_c_component_attribution_matrix{suffix}.csv"
    existing = read_rows(matrix) if int(args.part_c_resume) else []
    done = {
        (
            r.get("c_scheme") or r.get("scheme"),
            r.get("task"),
            r.get("seed"),
            r.get("train_steps"),
            r.get("residual_target"),
            r.get("lambda"),
            r.get("model_seed_scheme_key"),
            r.get("source_scheme"),
            r.get("component_ablation"),
        )
        for r in existing
        if r.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    for job in jobs:
        key_prefix = (job[0], job[1], str(job[2]))
        if any(k[0] == key_prefix[0] and k[1] == key_prefix[1] and k[2] == key_prefix[2] for k in done):
            continue
        rows.append(run_c_row(job, args, device))
        if int(args.part_c_flush_every) and len(rows) % int(args.part_c_flush_every) == 0:
            v2308.append_rows(matrix, rows)
            print(json.dumps({"part": "C", "rows_written": len(read_rows(matrix)), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
            rows = []
    if rows:
        v2308.append_rows(matrix, rows)
    all_rows = read_rows(matrix)
    summary = {
        "part": "C",
        "gate_pass": 0,
        "route": "PartCShardOnly" if suffix else "PartCNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(all_rows),
        "ok_rows": sum(1 for r in all_rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in all_rows if r.get("status") != "ok"),
        "matrix": rel(matrix),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_c_resume),
    }
    write_json(OUT_ROOT / f"part_c_component_attribution_summary{suffix}.json", summary)
    append_exec("Part C component attribution shard", command_text(sys.argv), "done", files=rel(matrix), gpu=str(device), note=f"rows={len(all_rows)}")
    append_recap("Part C component attribution shard", summary)
    return summary


def part_c_merge(args: argparse.Namespace) -> dict[str, Any]:
    init()
    pb = read_json(OUT_ROOT / "part_b_synthetic_lock_summary.json")
    if not ival(pb.get("gate_pass")):
        return v2308.blocked_summary("C", "C_BlockedByPartB", "part_b_not_passed")
    shard_paths = sorted(OUT_ROOT.glob("part_c_component_attribution_matrix_shard*_of_*.csv"))
    raw_rows = [r for path in shard_paths for r in read_rows(path)]
    raw_rows.extend(read_rows(OUT_ROOT / "part_c_component_attribution_matrix.csv"))
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            str(row.get("c_scheme") or row.get("scheme") or ""),
            str(row.get("task") or ""),
            str(row.get("seed") or ""),
            str(row.get("train_steps") or ""),
            str(row.get("residual_target") or ""),
            str(row.get("lambda") or ""),
            str(row.get("model_seed_scheme_key") or ""),
            str(row.get("source_scheme") or ""),
            str(row.get("component_ablation") or ""),
        )
        dedup[key] = row
    rows = list(dedup.values())
    matrix = write_rows(OUT_ROOT / "part_c_component_attribution_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("c_scheme") or r.get("scheme"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("c_scheme") or r.get("scheme"), r.get("task")) == key]
        task_summaries.append({
            "scheme": key[0],
            "task": key[1],
            "rows": len(group),
            "multi_step_C2_coverage_median": median(r.get("multi_step_C2_coverage") for r in group),
            "multi_step_C2_accuracy_median": median(r.get("multi_step_C2_accuracy") for r in group),
            "actual_one_step_C2_delta_median": median(r.get("actual_one_step_C2_delta") for r in group),
        })
    task_csv = write_rows(OUT_ROOT / "part_c_component_attribution_task_summary.csv", task_summaries)
    fg_wall = median(r.get("wall_time_s") for r in ok if (r.get("c_scheme") or r.get("scheme")) == "C0_FunctionalGram")
    groups: list[dict[str, Any]] = []
    for scheme in sorted({str(r.get("c_scheme") or r.get("scheme")) for r in ok}):
        group = [r for r in ok if str(r.get("c_scheme") or r.get("scheme")) == scheme]
        wall = median(r.get("wall_time_s") for r in group)
        groups.append({
            "scheme": scheme,
            "rows": len(group),
            "source_scheme": sorted({str(r.get("source_scheme", "")) for r in group})[0] if group else "",
            "multi_step_C2_coverage_median": median(r.get("multi_step_C2_coverage") for r in group),
            "multi_step_C2_accuracy_median": median(r.get("multi_step_C2_accuracy") for r in group),
            "actual_one_step_C2_delta_median": median(r.get("actual_one_step_C2_delta") for r in group),
            "source_residual_fit_improvement_median": median(r.get("source_residual_fit_improvement") for r in group),
            "guard_residual_fit_improvement_median": median(r.get("guard_residual_fit_improvement") for r in group),
            "residual_fit_probe_error_count": sum(1 for r in group if str(r.get("source_residual_fit_improvement")) == "error" or str(r.get("guard_residual_fit_improvement")) == "error"),
            "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group),
            "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group),
            "trust_accept_rate_median": median(r.get("trust_accept_rate") for r in group),
            "trust_scale_mean_median": median(r.get("trust_scale_mean") for r in group),
            "finite_step_skip_count_median": median(r.get("finite_step_skip_count") for r in group),
            "trust_reject_reasons": ";".join(str(r.get("trust_reject_reasons", "")) for r in group),
            "condition_number_after_ridge_median": median(r.get("condition_number_after_ridge") for r in group),
            "CG_iterations_median": median(r.get("CG_iterations") for r in group),
            "activation_drift_median": median(r.get("activation_drift") for r in group),
            "jacobian_drift_proxy_median": median(r.get("jacobian_drift_proxy") for r in group),
            "wall_time_s_median": wall,
            "overhead_ratio": wall / max(fg_wall, 1.0e-12) if fg_wall > 0 else 99.0,
            "held_test_usage": max(ival(r.get("held_test_usage")) for r in group),
            "used_fake_data_rows": max(ival(r.get("used_fake_data_rows")) for r in group),
            "runtime_selector_used": max(ival(r.get("runtime_selector_used")) for r in group),
        })
    by = {str(g.get("scheme")): g for g in groups}
    c6 = by.get("C6_DownstreamAdjoint_LocalInverse_ParetoTrust", {})
    c0 = by.get("C0_FunctionalGram", {})
    c3 = by.get("C3_LocalEFRF_no_downstream", {})
    c4 = by.get("C4_DownstreamAdjoint_no_inverse", {})
    c7 = by.get("C7_Candidate_random_projected_R", {})
    c8 = by.get("C8_Candidate_shuffled_design_Phi", {})
    c9 = by.get("C9_Candidate_shuffled_downstream_adjoint", {})
    c5 = by.get("C5_DownstreamAdjoint_LocalInverse_no_trust", {})
    c10 = by.get("C10_Candidate_trust_free_fixed_alpha", {})
    for g in groups:
        g["random_projected_gap"] = fval(g.get("multi_step_C2_coverage_median")) - fval(c7.get("multi_step_C2_coverage_median"))
        g["shuffled_design_gap"] = fval(g.get("multi_step_C2_coverage_median")) - fval(c8.get("multi_step_C2_coverage_median"))
        g["shuffled_downstream_gap"] = fval(g.get("multi_step_C2_coverage_median")) - fval(c9.get("multi_step_C2_coverage_median"))
        g["beats_FunctionalGram_gap"] = fval(g.get("multi_step_C2_coverage_median")) - fval(c0.get("multi_step_C2_coverage_median"))
        g["beats_LocalEFRF_gap"] = fval(g.get("multi_step_C2_coverage_median")) - fval(c3.get("multi_step_C2_coverage_median"))
        g["beats_no_inverse_gap"] = fval(g.get("multi_step_C2_coverage_median")) - fval(c4.get("multi_step_C2_coverage_median"))
    group_csv = write_rows(OUT_ROOT / "part_c_component_attribution_group_summary.csv", groups)
    required = set(PART_C_SCHEMES)
    present = {str(g.get("scheme")) for g in groups}
    missing_required = sorted(required - present)
    official_shape = int(int(args.part_c_seed_count) >= 15 and int(args.part_c_steps) >= 80)
    residual_fit_probe_errors = sum(ival(g.get("residual_fit_probe_error_count")) for g in groups if str(g.get("scheme")) not in {"C0_FunctionalGram", "C1_BlockSNR", "C2_H10_reference"})
    c6_cov = fval(c6.get("multi_step_C2_coverage_median"))
    gate = int(
        ival(pb.get("gate_pass"))
        and official_shape
        and not missing_required
        and residual_fit_probe_errors == 0
        and c6_cov >= fval(c0.get("multi_step_C2_coverage_median")) + 0.03
        and c6_cov >= fval(c3.get("multi_step_C2_coverage_median")) + 0.05
        and c6_cov >= fval(c4.get("multi_step_C2_coverage_median")) + 0.03
        and c6_cov >= fval(c7.get("multi_step_C2_coverage_median")) + 0.10
        and c6_cov >= fval(c8.get("multi_step_C2_coverage_median")) + 0.10
        and c6_cov >= fval(c9.get("multi_step_C2_coverage_median")) + 0.05
        and fval(c6.get("F5_no_debt_count")) >= 28
        and fval(c6.get("trust_accept_rate_median")) >= 0.45
        and fval(c6.get("trust_scale_mean_median")) >= 0.35
        and fval(c6.get("overhead_ratio"), 99.0) <= 2.0
        and max(ival(g.get("held_test_usage")) for g in groups or [{}]) == 0
        and max(ival(g.get("used_fake_data_rows")) for g in groups or [{}]) == 0
        and max(ival(g.get("runtime_selector_used")) for g in groups or [{}]) == 0
    )
    if gate:
        blocker = "none"
    elif missing_required:
        blocker = "missing_component_ablation_rows"
    elif residual_fit_probe_errors:
        blocker = "residual_fit_probe_errors"
    elif not official_shape:
        blocker = "formal_seed_or_step_count_not_met"
    elif c6_cov < fval(c7.get("multi_step_C2_coverage_median")) + 0.10 or c6_cov < fval(c8.get("multi_step_C2_coverage_median")) + 0.10:
        blocker = "random_or_shuffled_design_gap_low"
    elif c6_cov < fval(c9.get("multi_step_C2_coverage_median")) + 0.05:
        blocker = "shuffled_downstream_gap_low"
    elif c6_cov < fval(c4.get("multi_step_C2_coverage_median")) + 0.03:
        blocker = "local_inverse_not_proven"
    elif fval(c6.get("trust_accept_rate_median")) < 0.45 or fval(c6.get("trust_scale_mean_median")) < 0.35:
        blocker = "trust_component_low"
    elif fval(c6.get("overhead_ratio"), 99.0) > 2.0:
        blocker = "component_overhead_high"
    else:
        blocker = "component_attribution_threshold_failed"
    summary = {
        "part": "C",
        "gate_pass": gate,
        "route": "PartCComponentAttributionPass" if gate else "C_ComponentAttributionFailed",
        "dominant_blocker": blocker,
        "official_shape": official_shape,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "selected_source_f_scheme": selected_part_c_f_scheme(),
        "missing_required_ablation_rows": missing_required,
        "residual_fit_probe_errors": residual_fit_probe_errors,
        "candidate_row": c6,
        "no_trust_row": c5,
        "fixed_alpha_row": c10,
        "matrix": rel(matrix),
        "task_summary": rel(task_csv),
        "group_summary": rel(group_csv),
    }
    write_json(OUT_ROOT / "part_c_component_attribution_summary.json", summary)
    fail = failure("c", summary)
    allowed = [] if gate else ["if trust rows beat C6, inspect Pareto thresholds/rollback; if C7/C8/C9 close gap, strengthen controls or stop candidate promotion; if C4 close gap, local inverse is not proven necessary"]
    next_actions("c", summary["route"], blocker, allowed, [rel(matrix), rel(task_csv), rel(group_csv), rel(fail)])
    append_exec("Part C component attribution merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_c_component_attribution_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; selected={selected_part_c_f_scheme()}")
    append_recap("Part C component attribution merge", summary)
    return summary


def part_c_replay(args: argparse.Namespace) -> dict[str, Any]:
    init()
    pb = read_json(OUT_ROOT / "part_b_synthetic_lock_summary.json")
    if not pb:
        return v2308.blocked_summary("C", "C_BlockedByPartB", "part_b_missing")
    rows_b = read_rows(OUT_ROOT / "part_b_synthetic_lock_group_summary.csv")
    by_b = {str(r.get("scheme")): r for r in rows_b}
    mapping = {
        "C0_FunctionalGram": "B0_FunctionalGram_baseline",
        "C1_BlockSNR": "B1_BlockSNR_baseline",
        "C2_H10_reference": "B2_H10_known_structure_reference",
        "C3_LocalEFRF_no_downstream": "",
        "C6_DownstreamAdjoint_LocalInverse_ParetoTrust": "B4_v23_08_F28_reference" if str(pb.get("selected_source_f_scheme", "")).startswith("F28") else "B3_v23_08_F27_reference",
        "C7_Candidate_random_projected_R": "B5_random_projected_residual_flow",
        "C8_Candidate_shuffled_design_Phi": "B6_shuffled_design_residual_flow",
    }
    f_groups = csv_by(Path(args.v23_08_root).resolve() / "part_f_multistep_group_summary.csv", "scheme")
    local = f_groups.get("F3_v23_07_local_EFRF_all_layer_GS_control", {})
    rows: list[dict[str, Any]] = []
    for c_scheme, b_scheme in mapping.items():
        source = by_b.get(b_scheme, {}) if b_scheme else local
        row = {"part": "C", "status": "ok" if source else "missing", "scheme": c_scheme, "source_scheme": b_scheme or "F3_v23_07_local_EFRF_all_layer_GS_control"}
        row.update(source)
        row["multi_step_C2_coverage"] = source.get("C2_coverage_improvement_median", "")
        row["multi_step_C2_accuracy"] = source.get("C2_accuracy_improvement_median", "")
        row["trust_accept_rate"] = source.get("finite_step_accept_rate_median", "")
        row["trust_scale_mean"] = source.get("finite_step_scale_mean_median", "")
        row["overhead_ratio"] = source.get("solver_overhead_ratio", "")
        row["activation_drift"] = source.get("activation_drift_max_median", "")
        rows.append(row)
    matrix = write_rows(OUT_ROOT / "part_c_component_attribution_matrix.csv", rows)
    c_by = {str(r.get("scheme")): r for r in rows}
    c6 = c_by.get("C6_DownstreamAdjoint_LocalInverse_ParetoTrust", {})
    c0 = c_by.get("C0_FunctionalGram", {})
    c3 = c_by.get("C3_LocalEFRF_no_downstream", {})
    c7 = c_by.get("C7_Candidate_random_projected_R", {})
    c8 = c_by.get("C8_Candidate_shuffled_design_Phi", {})
    missing_required = ["C4_DownstreamAdjoint_no_inverse", "C5_DownstreamAdjoint_LocalInverse_no_trust", "C9_Candidate_shuffled_downstream_adjoint", "C10_Candidate_trust_free_fixed_alpha"]
    c6_cov = fval(c6.get("multi_step_C2_coverage"))
    gate = int(
        ival(pb.get("gate_pass"))
        and not missing_required
        and c6_cov >= fval(c0.get("multi_step_C2_coverage")) + 0.03
        and c6_cov >= fval(c3.get("multi_step_C2_coverage")) + 0.05
        and c6_cov >= fval(c7.get("multi_step_C2_coverage")) + 0.10
        and c6_cov >= fval(c8.get("multi_step_C2_coverage")) + 0.10
        and fval(c6.get("F5_no_debt_count")) >= 28
        and fval(c6.get("trust_accept_rate")) >= 0.45
        and fval(c6.get("trust_scale_mean")) >= 0.35
        and fval(c6.get("overhead_ratio"), 99.0) <= 2.0
    )
    blocker = "none" if gate else ("missing_component_ablation_rows" if missing_required else "component_attribution_threshold_failed")
    summary = {
        "part": "C",
        "gate_pass": gate,
        "route": "PartCComponentAttributionPass" if gate else "C_ComponentAttributionFailed",
        "dominant_blocker": blocker,
        "replay_only": 1,
        "missing_required_ablation_rows": missing_required,
        "candidate_row": c6,
        "matrix": rel(matrix),
        "task_summary": rel(matrix),
        "group_summary": rel(matrix),
    }
    write_json(OUT_ROOT / "part_c_component_attribution_summary.json", summary)
    fail = failure("c", summary)
    next_actions("c", summary["route"], blocker, [] if gate else ["implement or run missing C4/C5/C9/C10 ablations; do not treat replay subset as full component proof"], [rel(matrix), rel(fail)])
    append_exec("Part C replay", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_c_component_attribution_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part C component attribution replay audit", summary)
    return summary


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    init()
    pb = read_json(OUT_ROOT / "part_b_synthetic_lock_summary.json")
    pc = read_json(OUT_ROOT / "part_c_component_attribution_summary.json")
    if not ival(pb.get("gate_pass")):
        return v2308.blocked_summary("D", "D_BlockedByPartB", "part_b_not_passed")
    if not ival(pc.get("gate_pass")):
        return v2308.blocked_summary("D", "D_BlockedByPartC", "part_c_not_passed")
    return v2308.part_i(args)


def part_d_merge(args: argparse.Namespace) -> dict[str, Any]:
    init()
    v2308.part_i_merge(args)
    matrix = OUT_ROOT / "part_i_real_task_preflight_matrix.csv"
    rows = read_rows(matrix)
    ok = [r for r in rows if r.get("status") == "ok"]
    fg = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I0_FunctionalGram_baseline"}
    bs = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I1_BlockSNR_baseline"}
    rand = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I5_random_projected_residual_flow"}
    shuf = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I7_shuffled_design_residual_flow"}
    noop = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I6_same_compute_noop"}
    for r in ok:
        key = (r.get("dataset"), r.get("seed"))
        if key in fg:
            r["beats_FunctionalGram_NLL"] = fval(fg[key].get("held_nll_delta")) - fval(r.get("held_nll_delta"))
            r["beats_FunctionalGram_coverage"] = fval(r.get("held_coverage_improvement")) - fval(fg[key].get("held_coverage_improvement"))
        if key in bs:
            r["beats_BlockSNR_NLL"] = fval(bs[key].get("held_nll_delta")) - fval(r.get("held_nll_delta"))
            r["beats_BlockSNR_coverage"] = fval(r.get("held_coverage_improvement")) - fval(bs[key].get("held_coverage_improvement"))
        if key in rand:
            r["random_projected_gap_NLL"] = fval(rand[key].get("held_nll_delta")) - fval(r.get("held_nll_delta"))
            r["random_projected_gap_coverage"] = fval(r.get("held_coverage_improvement")) - fval(rand[key].get("held_coverage_improvement"))
        if key in shuf:
            r["shuffled_design_gap_NLL"] = fval(shuf[key].get("held_nll_delta")) - fval(r.get("held_nll_delta"))
            r["shuffled_design_gap_coverage"] = fval(r.get("held_coverage_improvement")) - fval(shuf[key].get("held_coverage_improvement"))
        if key in noop:
            r["same_compute_gap"] = fval(r.get("held_coverage_improvement")) - fval(noop[key].get("held_coverage_improvement"))
    matrix = write_rows(OUT_ROOT / "part_d_robust_real_task_matrix.csv", rows)
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("i_scheme"), r.get("dataset")) for r in ok}):
        group = [r for r in ok if (r.get("i_scheme"), r.get("dataset")) == key]
        groups.append({
            "scheme": key[0],
            "dataset": key[1],
            "rows": len(group),
            "held_nll_delta_median": median(r.get("held_nll_delta") for r in group),
            "held_coverage_delta_median": median(r.get("held_coverage_improvement") for r in group),
            "held_accuracy_delta_median": median(r.get("held_accuracy_improvement") for r in group),
            "beats_FunctionalGram_NLL_median": median(r.get("beats_FunctionalGram_NLL") for r in group),
            "beats_FunctionalGram_coverage_median": median(r.get("beats_FunctionalGram_coverage") for r in group),
            "beats_BlockSNR_NLL_median": median(r.get("beats_BlockSNR_NLL") for r in group),
            "beats_BlockSNR_coverage_median": median(r.get("beats_BlockSNR_coverage") for r in group),
            "random_projected_gap_NLL_median": median(r.get("random_projected_gap_NLL") for r in group),
            "random_projected_gap_coverage_median": median(r.get("random_projected_gap_coverage") for r in group),
            "shuffled_design_gap_NLL_median": median(r.get("shuffled_design_gap_NLL") for r in group),
            "shuffled_design_gap_coverage_median": median(r.get("shuffled_design_gap_coverage") for r in group),
            "same_compute_gap_median": median(r.get("same_compute_gap") for r in group),
            "Brier_delta_median": median(r.get("Brier_delta") for r in group),
            "ECE_delta_median": median(r.get("ECE_delta") for r in group),
            "tail95_delta_median": median(r.get("tail95_delta") for r in group),
            "tail99_delta_median": median(r.get("tail99_delta") for r in group),
            "margin10_delta_median": median(r.get("margin10_delta") for r in group),
            "real_no_debt_pass_count": sum(ival(r.get("real_no_debt")) for r in group),
            "finite_step_accept_rate_median": median(r.get("accept_rate") for r in group),
            "finite_step_scale_mean_median": median(r.get("scale_mean") for r in group),
            "finite_step_skip_count_median": median(r.get("skip_count") for r in group),
            "finite_step_reject_reasons": ";".join(str(r.get("finite_step_reject_reasons", "")) for r in group),
            "overhead_ratio_median": median(r.get("overhead_ratio") for r in group),
            "CG_iterations_median": median(r.get("cg_iteration_median") for r in group),
            "source_guard_loss_correlation": "not_computed",
            "source_guard_coverage_correlation": "not_computed",
            "held_test_usage": max(ival(r.get("held_test_usage")) for r in group),
            "used_fake_data_rows": max(ival(r.get("used_fake_data_rows")) for r in group),
        })
    group_csv = write_rows(OUT_ROOT / "part_d_robust_real_task_group_summary.csv", groups)
    by = {(g["scheme"], g["dataset"]): g for g in groups}
    official = [by.get(("I3_DownstreamEFRF_best_positive_control_scheme", d), {}) for d in ["MNIST", "FashionMNIST", "KMNIST"]]
    visual_pass = [
        int(
            g
            and fval(g.get("held_nll_delta_median"), 999) < 0.0
            and fval(g.get("held_coverage_delta_median"), -999) >= 0.0
            and fval(g.get("beats_FunctionalGram_NLL_median")) >= 0.005
            and fval(g.get("beats_FunctionalGram_coverage_median")) >= 0.0015
        )
        for g in official
    ]
    kmnist = by.get(("I3_DownstreamEFRF_best_positive_control_scheme", "KMNIST"), {})
    random_gap_cov = median(g.get("random_projected_gap_coverage_median") for g in official if g)
    random_gap_nll_count = sum(1 for g in official if g and fval(g.get("random_projected_gap_NLL_median")) >= 0.005)
    wine = by.get(("I3_DownstreamEFRF_best_positive_control_scheme", "Wine"), {})
    wine_fg = by.get(("I0_FunctionalGram_baseline", "Wine"), {})
    accept = median(g.get("finite_step_accept_rate_median") for g in official if g)
    scale = median(g.get("finite_step_scale_mean_median") for g in official if g)
    skip = median(g.get("finite_step_skip_count_median") for g in official if g)
    overhead = median(g.get("overhead_ratio_median") for g in official if g)
    gate = int(
        sum(visual_pass) >= 2
        and fval(kmnist.get("held_coverage_delta_median"), -999) >= 0.0
        and random_gap_cov >= 0.005
        and random_gap_nll_count >= 2
        and fval(wine.get("held_nll_delta_median"), 999) <= 0.0
        and fval(wine.get("held_coverage_delta_median"), -999) >= fval(wine_fg.get("held_coverage_delta_median"), 999) - 0.05
        and accept >= 0.30
        and scale >= 0.20
        and skip <= 0.35 * int(args.part_i_steps)
        and overhead <= 2.0
    )
    if gate:
        blocker = "none"
    elif fval(kmnist.get("held_coverage_delta_median"), -999) < 0.0:
        blocker = "kmnist_coverage_negative"
    elif accept < 0.30 or scale < 0.20 or skip > 0.35 * int(args.part_i_steps):
        blocker = "trust_efficiency_low"
    elif random_gap_cov < 0.005 or random_gap_nll_count < 2:
        blocker = "random_gap_low"
    elif overhead > 2.0:
        blocker = "overhead_high"
    else:
        blocker = "real_task_pareto_gate_failed"
    summary = {
        "part": "D",
        "gate_pass": gate,
        "route": "PartDRobustRealTaskPreflightPass" if gate else "D_RobustRealPreflightFailed",
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "visual_pass_count": sum(visual_pass),
        "visual_pass_vector": dict(zip(["MNIST", "FashionMNIST", "KMNIST"], visual_pass)),
        "kmnist_coverage_delta_median": fval(kmnist.get("held_coverage_delta_median")),
        "visual_random_gap_coverage_median": random_gap_cov,
        "visual_random_gap_nll_pass_count": random_gap_nll_count,
        "trust_accept_rate_visual_median": accept,
        "trust_scale_mean_visual_median": scale,
        "trust_skip_count_visual_median": skip,
        "overhead_visual_median": overhead,
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
    }
    write_json(OUT_ROOT / "part_d_robust_real_task_summary.json", summary)
    fail = failure("d", summary)
    next_actions("d", summary["route"], blocker, [] if gate else ["route blocker to Part E/F/H according to failure class; keep random and shuffled controls"], [rel(matrix), rel(group_csv), rel(fail)])
    append_exec("Part D merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_d_robust_real_task_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part D robust real-task preflight merge", summary)
    return summary


def part_i(args: argparse.Namespace) -> dict[str, Any]:
    init()
    parts = {
        "part_0": read_json(OUT_ROOT / "part_0_lineage_summary.json"),
        "part_a": read_json(OUT_ROOT / "part_a_summary.json"),
        "part_b": read_json(OUT_ROOT / "part_b_synthetic_lock_summary.json"),
        "part_c": read_json(OUT_ROOT / "part_c_component_attribution_summary.json"),
        "part_d": read_json(OUT_ROOT / "part_d_robust_real_task_summary.json"),
        "part_e": read_json(OUT_ROOT / "part_e_kmnist_coverage_summary.json"),
        "part_f": read_json(OUT_ROOT / "part_f_trust_efficiency_summary.json"),
        "part_h": read_json(OUT_ROOT / "part_h_solver_scaling_summary.json"),
    }
    if not ival(parts["part_0"].get("gate_pass")) or not ival(parts["part_a"].get("gate_pass")):
        route, blocker = "A_or_B_IdentityOrLineageFailed", "identity_or_lineage_failed"
    elif not ival(parts["part_b"].get("gate_pass")):
        route, blocker = "B_SyntheticPositiveControlRegressed", str(parts["part_b"].get("dominant_blocker", "part_b_failed"))
    elif not ival(parts["part_c"].get("gate_pass")):
        route, blocker = "C_ComponentAttributionFailed", str(parts["part_c"].get("dominant_blocker", "part_c_failed"))
    elif not ival(parts["part_d"].get("gate_pass")):
        route, blocker = "D_RobustRealPreflightFailed", str(parts["part_d"].get("dominant_blocker", "part_d_failed"))
    elif parts["part_e"] and not ival(parts["part_e"].get("gate_pass")):
        route, blocker = "E_KMNISTCoverageFailed", str(parts["part_e"].get("dominant_blocker", "part_e_failed"))
    elif not parts["part_f"]:
        route, blocker = "F_TrustEfficiencyFailed", "part_f_trust_efficiency_missing"
    elif not ival(parts["part_f"].get("gate_pass")):
        route, blocker = "F_TrustEfficiencyFailed", str(parts["part_f"].get("dominant_blocker", "part_f_failed"))
    elif not parts["part_h"]:
        route, blocker = "H_SolverScalingFailed", "part_h_solver_scaling_missing"
    elif not ival(parts["part_h"].get("gate_pass")):
        route, blocker = "H_SolverScalingFailed", str(parts["part_h"].get("dominant_blocker", "part_h_failed"))
    else:
        route, blocker = "I_RobustPreflightPass_NotOfficial", "none"
    final = {
        "final_route": route,
        "dominant_blocker": blocker,
        "promotion_allowed": 0,
        **{k + "_gate_pass": ival(v.get("gate_pass")) for k, v in parts.items()},
        "evidence": {k: rel(OUT_ROOT / name) for k, name in {
            "part_0": "part_0_lineage_summary.json",
            "part_a": "part_a_summary.json",
            "part_b": "part_b_synthetic_lock_summary.json",
            "part_c": "part_c_component_attribution_summary.json",
            "part_d": "part_d_robust_real_task_summary.json",
            "part_e": "part_e_kmnist_coverage_summary.json",
            "part_f": "part_f_trust_efficiency_summary.json",
            "part_h": "part_h_solver_scaling_summary.json",
        }.items()},
    }
    write_json(OUT_ROOT / "final_route.json", final)
    part_b_candidate = parts["part_b"].get("candidate_group", {}) if isinstance(parts["part_b"], dict) else {}
    locked_scheme = str(parts["part_b"].get("selected_source_f_scheme", "not_locked"))
    locked_short = "F27" if "F27" in locked_scheme else ("F28" if "F28" in locked_scheme else "unlocked")
    spec = {
        "candidate_name": f"TP-D-EFRF_norm_pareto_trust_{locked_short}_locked",
        "candidate_status": "not_promoted" if route != "I_RobustPreflightPass_NotOfficial" else "robust_preflight_pass_not_official",
        "blocking_route": route,
        "dominant_blocker": blocker,
        "selected_source_f_scheme": locked_scheme,
        "basis": args.part_i_basis,
        "depth": int(args.part_i_depth),
        "layers_updated": "last_penultimate_adjoint_local_hybrid",
        "residual_target": part_b_candidate.get("residual_target", args.part_i_residual_target),
        "W_definition": "train_source_guard_population_weights_no_held_test",
        "G_edge_definition": "D-CHE edge Sobolev / expanded edge metric",
        "lambda": fval(part_b_candidate.get("lambda"), float(args.part_i_lambda)),
        "solver_type": args.part_i_solver,
        "trust_type": str(parts["part_b"].get("part_f_trust_mode", args.part_i_trust_mode)),
        "trust_thresholds": {"loss": fval(parts["part_b"].get("part_f_trust_tolerance"), float(args.part_i_trust_tolerance)), "coverage": fval(parts["part_b"].get("part_f_trust_tolerance"), float(args.part_i_trust_tolerance))},
        "alpha_grid": str(parts["part_b"].get("part_f_alphas", args.part_i_alphas)),
        "step_budget": 80 if ival(parts["part_b"].get("gate_pass")) else int(args.part_i_steps),
        "random_controls_required": ["FunctionalGram", "BlockSNR", "random_projected", "shuffled_design", "same_compute_noop"],
        "allowed_datasets_for_next_version": v2308.csv_items(args.part_i_datasets),
    }
    write_json(OUT_ROOT / "v23_09_candidate_spec.json", spec)
    if route == "I_RobustPreflightPass_NotOfficial":
        repro = (
            "V2309_OUT_ROOT={root} {py} experiments/run_v23_09_trust_projected_downstream_efrf.py "
            "--mode part-d --device cuda:2 --shard-count 2 --shard-index 0 ..."
        ).format(root=rel(OUT_ROOT), py=sys.executable)
    else:
        repro = f"BLOCKED: final_route={route}; dominant_blocker={blocker}; resolve blocker before Part D."
    (OUT_ROOT / "candidate_repro_command.txt").write_text(repro + "\n", encoding="utf-8")
    rows = [
        {"part": k, "gate_pass": ival(v.get("gate_pass")), "route": v.get("route", ""), "dominant_blocker": v.get("dominant_blocker", "")}
        for k, v in parts.items()
    ]
    final_matrix = write_rows(OUT_ROOT / "part_i_final_matrix.csv", rows)
    summary = {"part": "I", "gate_pass": int(route == "I_RobustPreflightPass_NotOfficial"), "route": route, "dominant_blocker": blocker, "final_route": rel(OUT_ROOT / "final_route.json"), "part_i_final_matrix": rel(final_matrix), "candidate_spec": rel(OUT_ROOT / "v23_09_candidate_spec.json")}
    write_json(OUT_ROOT / "part_i_final_summary.json", summary)
    next10 = {"route": route, "dominant_blocker": blocker, "recommended_next": [] if route == "I_RobustPreflightPass_NotOfficial" else ["resolve blocker before v23.10 official matrix"], "forbidden": ["held/test selection", "remove random controls", "new observer/phase search"]}
    write_json(OUT_ROOT / "next_actions_for_v23_10.json", next10)
    append_exec("Part I final", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(OUT_ROOT / 'part_i_final_summary.json')}", gpu=str(args.device), note=f"route={route}; blocker={blocker}")
    append_recap("Part I final route and v23.10 admission audit", {**summary, "final": final, "candidate_spec": spec})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = v2308.build_arg_parser()
    p.description = __doc__
    p.add_argument("--v23-08-root", default="results/v23_08_part_i_pareto_trust_norm_s012_40step_core")
    p.add_argument("--part-c-schemes", default=",".join(PART_C_SCHEMES))
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-seed-count", type=int, default=15)
    p.add_argument("--part-c-steps", type=int, default=80)
    p.add_argument("--part-c-fixed-alpha", type=float, default=0.25)
    p.add_argument("--part-c-margin-curvature-tau", type=float, default=0.0)
    p.add_argument("--part-c-margin-curvature-temperature", type=float, default=1.0)
    p.add_argument("--part-c-fisher-curvature-strength", type=float, default=0.5)
    p.add_argument("--part-c-fisher-curvature-temperature", type=float, default=1.0)
    p.add_argument("--part-c-fisher-curvature-eps", type=float, default=0.001)
    p.add_argument("--part-c-fisher-curvature-max-weight", type=float, default=3.0)
    p.add_argument("--part-c-guard-penalty-strength", type=float, default=0.1)
    p.add_argument("--part-c-spectral-mode-fraction", type=float, default=2.0 / 3.0)
    p.add_argument("--part-c-model-seed-scheme-key", default="v23_09_part_c_shared_init")
    p.add_argument("--part-c-flush-every", type=int, default=1)
    p.add_argument("--part-c-resume", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    configure_v2308()
    mode = str(args.mode).lower()
    if mode in {"0", "part-0", "lineage"}:
        return part_0(args)
    if mode in {"a", "part-a"}:
        return part_a(args)
    if mode in {"b", "part-b", "part-b-replay"}:
        return part_b_replay(args)
    if mode in {"b-run", "part-b-run"}:
        return part_b_run(args)
    if mode in {"b-merge", "part-b-merge"}:
        return part_b_merge(args)
    if mode in {"c-replay", "part-c-replay"}:
        return part_c_replay(args)
    if mode in {"c-run", "part-c-run"}:
        return part_c_run(args)
    if mode in {"c", "part-c", "c-merge", "part-c-merge"}:
        return part_c_merge(args)
    if mode in {"d", "part-d"}:
        return part_d(args)
    if mode in {"d-merge", "part-d-merge"}:
        return part_d_merge(args)
    if mode in {"i", "part-i", "finalize"}:
        return part_i(args)
    raise SystemExit(f"unknown v23.09 mode: {args.mode}")


if __name__ == "__main__":
    main()
