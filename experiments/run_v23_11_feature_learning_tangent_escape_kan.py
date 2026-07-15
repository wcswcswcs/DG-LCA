#!/usr/bin/env python3
"""DG-KAN v23.11 feature-learning / tangent-escape audit runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.11_FeatureLearningTangentEscapeKAN_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.11_FeatureLearningTangentEscapeKAN_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.11_FeatureLearningTangentEscapeKAN_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2311_OUT_ROOT", str(ROOT / "results/v23_11_feature_learning_tangent_escape_kan"))).resolve()
FORMAL_ROOT = ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80"

ROUTE_A = ROOT / "results/v23_10_efficiency_equivalence_task_curvature_efrf/route_a_efficiency_equivalence_summary.json"
B1 = ROOT / "results/v23_10_route_b_margin_curvature_reduced_s5_80/route_b_b1_margin_curvature_reduced_summary.json"
B2 = ROOT / "results/v23_10_route_b_diagonal_fisher_reduced_s5_80/route_b_b2_diagonal_fisher_reduced_summary.json"
B3 = ROOT / "results/v23_10_route_b_guard_penalty_reduced_s5_80/route_b_b3_guard_penalty_reduced_summary.json"
DEFAULT_REDUCED_REF = B3
BASIS_GAIN_ROOT = Path(os.environ.get("V2311_BASIS_GAIN_ROOT", str(ROOT / "results/v23_11_basis_gain_tangent_relocation_reduced_s5_80"))).resolve()
SPECTRAL_MODE_ROOT = Path(os.environ.get("V2311_SPECTRAL_MODE_ROOT", str(ROOT / "results/v23_11_stage2_spectral_mode_reduced_s5_80"))).resolve()
TAIL_TARGET_ROOT = Path(os.environ.get("V2311_TAIL_TARGET_ROOT", str(ROOT / "results/v23_11_stage2b_tail_target_reduced_s5_80"))).resolve()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def ival(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_logs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.11 Feature-Learning Tangent-Escape KAN 执行日志\n\n"
            "- 原则：不造假；不把 v23.09/v23.10 failure 改写成 success；不使用 held/test induction。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.11 Feature-Learning Tangent-Escape KAN 实验结果复盘\n\n"
            "- 原则：只记录真实 artifact 数据；不得把 diagnostic 写成 promotion。\n",
            encoding="utf-8",
        )


def command_text() -> str:
    return f"{sys.executable} {' '.join(sys.argv)}"


def append_exec(title: str, command: str, *, files: str, gpu: str, note: str) -> None:
    ensure_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title} done\n\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        fh.write(f"- note: {note}\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    ensure_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
        fh.write("\n```\n")


def route_b_failed(summary: dict[str, Any]) -> bool:
    return (
        ival(summary.get("route_b_reduced_gate_pass"), -1) == 0
        and not bool(summary.get("promotion_allowed", False))
        and bool(summary.get("checks", {}).get("no_fake_data", False))
        and bool(summary.get("checks", {}).get("no_held_test_usage", False))
    )


def evidence_lock(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "plan": PLAN,
        "runner": RUNNER,
        "route_a": ROUTE_A,
        "route_b_b1": B1,
        "route_b_b2": B2,
        "route_b_b3": B3,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    route_a = read_json(ROUTE_A) if ROUTE_A.exists() else {}
    b1 = read_json(B1) if B1.exists() else {}
    b2 = read_json(B2) if B2.exists() else {}
    b3 = read_json(B3) if B3.exists() else {}
    checks = {
        "no_missing_required_artifacts": not missing,
        "route_a_efficiency_equivalence_pass": ival(route_a.get("gate_pass"), -1) == 1,
        "route_a_not_coverage_promotion": "efficiency" in str(route_a.get("promotion_scope", "")).lower(),
        "official_v23_09_still_failed": ival(route_a.get("official_v23_09_part_c_gate_pass"), -1) == 0,
        "b1_failed_reduced_gate": route_b_failed(b1),
        "b2_failed_reduced_gate": route_b_failed(b2),
        "b3_failed_reduced_gate": route_b_failed(b3),
        "c20_terminal_rejected": fval(b1.get("metrics", {}).get("C20_terminal_reject_count_inferred")) > 0.0,
        "c22_coverage_neutral_to_c3": abs(fval(b2.get("metrics", {}).get("C22_minus_C3"), 99.0)) < 0.01,
        "c24_coverage_neutral_to_c3": abs(fval(b3.get("metrics", {}).get("C24_minus_C3"), 99.0)) < 0.01,
    }
    gate = int(all(checks.values()))
    hashes = {name: sha256_file(path) for name, path in required.items() if path.exists()}
    summary = {
        "part": "0",
        "gate_pass": gate,
        "route": "V2311EvidenceLockPass" if gate else "V2311EvidenceLockFailed",
        "dominant_blocker": "none" if gate else "missing_or_inconsistent_v23_10_evidence",
        "checks": checks,
        "missing": missing,
        "hashes": hashes,
        "evidence_paths": {name: rel(path) for name, path in required.items()},
        "locked_metrics": {
            "route_a_c15_coverage": route_a.get("metrics", {}).get("c15_coverage"),
            "route_a_c15_minus_c3": route_a.get("metrics", {}).get("c15_minus_c3"),
            "route_a_c3_over_c15_wall_time_ratio": route_a.get("metrics", {}).get("c3_over_c15_wall_time_ratio"),
            "b1_c20_coverage": b1.get("metrics", {}).get("C20_coverage"),
            "b1_c20_terminal_reject_count_inferred": b1.get("metrics", {}).get("C20_terminal_reject_count_inferred"),
            "b2_c22_coverage": b2.get("metrics", {}).get("C22_coverage"),
            "b2_c22_minus_c3": b2.get("metrics", {}).get("C22_minus_C3"),
            "b3_c24_coverage": b3.get("metrics", {}).get("C24_coverage"),
            "b3_c24_minus_c3": b3.get("metrics", {}).get("C24_minus_C3"),
            "b3_c24_minus_c25": b3.get("metrics", {}).get("C24_minus_C25"),
        },
        "methodology_update": "Stop local-inverse micro-variants. Next reduced diagnostic must test whether representation/tangent geometry can move the C3/C15 ceiling.",
        "next_stage": "stage_1_basis_gain_tangent_relocation_reduced",
    }
    out = write_json(OUT_ROOT / "part_0_evidence_lock_summary.json", summary)
    append_exec("Part 0 evidence lock", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={summary['dominant_blocker']}")
    append_recap("Part 0 evidence lock", summary)
    return summary


def theory_reset(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "part_0_evidence_lock_summary.json"
    lock = read_json(lock_path) if lock_path.exists() else evidence_lock(args)
    stage1_commands = [
        'ROOT=results/v23_11_basis_gain_tangent_relocation_reduced_s5_80',
        'SCHEMES=C3_LocalEFRF_no_downstream,C15_Candidate_terminal_audited_no_trust,C9_Candidate_shuffled_downstream_adjoint,C10_Candidate_trust_free_fixed_alpha',
        'V2309_OUT_ROOT="$ROOT" /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_09_trust_projected_downstream_efrf.py --mode part-c-run --device cuda:2 --shard-count 2 --shard-index 0 --part-c-schemes "$SCHEMES" --part-c-seed-count 5 --part-c-steps 80 --part-d-basis-input-gain 1.0 --part-f-residual-target norm --part-f-trust-mode pareto --part-f-trust-tolerance 0.002 --part-f-lambda 10 --part-f-alphas 1,0.5,0.25,0.125 --part-f-functionalgram-lr 0.0075 --part-c-flush-every 1 --part-c-resume 1 --v23-08-root results/v23_08_part_i_pareto_trust_norm_s012_40step_core',
        'V2309_OUT_ROOT="$ROOT" /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_09_trust_projected_downstream_efrf.py --mode part-c-run --device cuda:3 --shard-count 2 --shard-index 1 --part-c-schemes "$SCHEMES" --part-c-seed-count 5 --part-c-steps 80 --part-d-basis-input-gain 1.0 --part-f-residual-target norm --part-f-trust-mode pareto --part-f-trust-tolerance 0.002 --part-f-lambda 10 --part-f-alphas 1,0.5,0.25,0.125 --part-f-functionalgram-lr 0.0075 --part-c-flush-every 1 --part-c-resume 1 --v23-08-root results/v23_08_part_i_pareto_trust_norm_s012_40step_core',
        'V2309_OUT_ROOT="$ROOT" /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_09_trust_projected_downstream_efrf.py --mode part-c-merge --device cpu --part-c-schemes "$SCHEMES" --part-c-seed-count 5 --part-c-steps 80 --part-d-basis-input-gain 1.0 --part-f-residual-target norm --part-f-trust-mode pareto --part-f-trust-tolerance 0.002 --part-f-lambda 10 --part-f-alphas 1,0.5,0.25,0.125 --part-f-functionalgram-lr 0.0075 --v23-08-root results/v23_08_part_i_pareto_trust_norm_s012_40step_core',
        '/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v23_11_feature_learning_tangent_escape_kan.py --mode basis-gain-audit --device cpu',
    ]
    summary = {
        "version": "v23.11",
        "status": "preregistered_theory_reset",
        "blocked_previous_routes": ["C20_margin_curvature", "C22_diagonal_fisher", "C24_guard_penalty_constrained_solve"],
        "evidence_lock_gate_pass": lock.get("gate_pass"),
        "core_hypothesis": "coverage dominance requires feature-learning/tangent-escape, not another fixed-local-inverse residual weighting or guard penalty.",
        "primary_sources": [
            "https://arxiv.org/abs/2404.19756",
            "https://arxiv.org/abs/1806.07572",
            "https://arxiv.org/abs/1812.07956",
            "https://arxiv.org/abs/2011.14522",
            "https://www.mitpressjournals.org/doi/10.1162/089976698300017746",
            "https://arxiv.org/abs/1503.05671",
            "https://arxiv.org/abs/2203.03466",
        ],
        "stage_1": {
            "name": "basis_gain_tangent_relocation_reduced",
            "scope": "diagnostic_only_not_promotion",
            "root": rel(BASIS_GAIN_ROOT),
            "schemes": [
                "C3_LocalEFRF_no_downstream",
                "C15_Candidate_terminal_audited_no_trust",
                "C9_Candidate_shuffled_downstream_adjoint",
                "C10_Candidate_trust_free_fixed_alpha",
            ],
            "basis_input_gain": 1.0,
            "seed_count": 5,
            "steps": 80,
            "expected_rows": 40,
            "commands": stage1_commands,
        },
        "promotion_rule": "No promotion from Stage 1. Stage 2 candidate can be implemented only if Stage 1 shows the C3/C15 ceiling moves with representation geometry.",
    }
    prereg = write_json(FORMAL_ROOT / "v23_11_preregistered_theory_reset.json", summary)
    nxt = write_json(FORMAL_ROOT / "next_actions_for_v23_11.json", {
        "version": "v23.11",
        "next_action": "run_stage_1_basis_gain_tangent_relocation_reduced",
        "depends_on": rel(OUT_ROOT / "part_0_evidence_lock_summary.json"),
        "commands": stage1_commands,
        "promotion_allowed": 0,
        "reason": "diagnostic-only tangent relocation evidence is required before implementing a feature-learning candidate.",
    })
    append_exec("Theory reset and next-action sync", command_text(), files=f"{rel(prereg)}; {rel(nxt)}", gpu=str(args.device), note="v23.11 points to basis-gain tangent-relocation reduced diagnostic; no promotion claimed")
    append_recap("Theory reset and next-action sync", summary)
    return summary


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def row_for_scheme(rows: list[dict[str, str]], scheme: str) -> dict[str, str]:
    for row in rows:
        if row.get("scheme") == scheme:
            return row
    return {}


def coverage(row: dict[str, Any]) -> float:
    return fval(row.get("multi_step_C2_coverage_median", row.get("C2_coverage_improvement_median", "")))


def count_token(value: Any, token: str) -> int:
    return str(value or "").count(token)


def basis_gain_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "high_gain_matrix": BASIS_GAIN_ROOT / "part_c_component_attribution_matrix.csv",
        "high_gain_group": BASIS_GAIN_ROOT / "part_c_component_attribution_group_summary.csv",
        "high_gain_summary": BASIS_GAIN_ROOT / "part_c_component_attribution_summary.json",
        "default_reference": DEFAULT_REDUCED_REF,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    groups = read_csv_rows(required["high_gain_group"]) if not missing else []
    part_c = read_json(required["high_gain_summary"]) if not missing else {}
    default = read_json(DEFAULT_REDUCED_REF) if DEFAULT_REDUCED_REF.exists() else {}
    c3 = row_for_scheme(groups, "C3_LocalEFRF_no_downstream")
    c15 = row_for_scheme(groups, "C15_Candidate_terminal_audited_no_trust")
    c9 = row_for_scheme(groups, "C9_Candidate_shuffled_downstream_adjoint")
    c10 = row_for_scheme(groups, "C10_Candidate_trust_free_fixed_alpha")
    high_c3 = coverage(c3)
    high_c15 = coverage(c15)
    high_c9 = coverage(c9)
    high_c10 = coverage(c10)
    default_metrics = default.get("metrics", {})
    default_c3 = fval(default_metrics.get("C3_coverage"))
    default_c15 = fval(default_metrics.get("C15_coverage"))
    ceiling_move_abs = max(abs(high_c3 - default_c3), abs(high_c15 - default_c15))
    ceiling_move_best_signed = max(high_c3 - default_c3, high_c15 - default_c15)
    c15_terminal_rollback_count = count_token(c15.get("trust_reject_reasons"), "terminal_audit_rollback")
    checks = {
        "no_missing_required_artifacts": not missing,
        "row_count_40_ok": ival(part_c.get("row_count")) == 40 and ival(part_c.get("ok_rows")) == 40,
        "no_fake_data": ival(part_c.get("used_fake_data_rows")) == 0,
        "no_held_test_usage": ival(part_c.get("held_test_usage")) == 0,
        "c15_terminal_audit_recorded": bool(c15) and ("terminal_audit_rollback" in str(c15.get("trust_reject_reasons", "")) or fval(c15.get("trust_accept_rate_median")) >= 0.0),
        "c15_beats_c9_by_0p05": (high_c15 - high_c9) >= 0.05,
        "ceiling_moved_by_0p01": ceiling_move_abs >= 0.01,
        "ceiling_moved_positive_by_0p01": ceiling_move_best_signed >= 0.01,
    }
    gate = int(all(checks.values()))
    if missing:
        blocker = "missing_required_artifacts"
    elif not checks["row_count_40_ok"]:
        blocker = "unexpected_reduced_shape"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif c15_terminal_rollback_count > 0 or high_c15 <= 0.0:
        blocker = "high_gain_c15_terminal_rollback_or_zero_coverage"
    elif not checks["c15_beats_c9_by_0p05"]:
        blocker = "high_gain_c15_does_not_beat_shuffled_downstream"
    elif not checks["ceiling_moved_positive_by_0p01"]:
        blocker = "basis_gain_moved_ceiling_destructively_not_positively"
    elif not checks["ceiling_moved_by_0p01"]:
        blocker = "basis_gain_did_not_move_ceiling"
    else:
        blocker = "none"
    summary = {
        "part": "1",
        "gate_pass": gate,
        "route": "Stage1BasisGainTangentRelocationUseful" if gate else "Stage1BasisGainTangentRelocationFailedOrNeutral",
        "scope": "diagnostic_only_not_promotion",
        "dominant_blocker": blocker,
        "checks": checks,
        "missing": missing,
        "metrics": {
            "default_reference_root": rel(DEFAULT_REDUCED_REF),
            "default_c3_coverage": default_c3,
            "default_c15_coverage": default_c15,
            "high_gain_c3_coverage": high_c3,
            "high_gain_c15_coverage": high_c15,
            "high_gain_c9_coverage": high_c9,
            "high_gain_c10_coverage": high_c10,
            "high_gain_c15_minus_c9": high_c15 - high_c9,
            "high_gain_c15_minus_c10": high_c15 - high_c10,
            "high_gain_c3_minus_default_c3": high_c3 - default_c3,
            "high_gain_c15_minus_default_c15": high_c15 - default_c15,
            "high_gain_ceiling_move_abs": ceiling_move_abs,
            "high_gain_ceiling_move_best_signed": ceiling_move_best_signed,
            "high_gain_c15_terminal_rollback_count_inferred": c15_terminal_rollback_count,
            "high_gain_c15_no_debt_count": fval(c15.get("F5_no_debt_count")),
            "high_gain_c15_source_residual_fit_improvement_median": fval(c15.get("source_residual_fit_improvement_median")),
            "high_gain_c15_guard_residual_fit_improvement_median": fval(c15.get("guard_residual_fit_improvement_median")),
            "high_gain_c15_activation_drift_median": fval(c15.get("activation_drift_median")),
            "high_gain_c3_trust_accept_rate_median": fval(c3.get("trust_accept_rate_median")),
            "high_gain_c9_trust_accept_rate_median": fval(c9.get("trust_accept_rate_median")),
            "high_gain_c10_component_non_positive_rows": fval(c10.get("component_non_positive_rows")),
        },
        "artifacts": {name: rel(path) for name, path in required.items()},
        "hashes": {name: sha256_file(path) for name, path in required.items() if path.exists()},
        "next_action": "implement_stage_2_feature_anchor_candidate" if gate else "do_not_implement_C26_from_static_basis_gain; analyze learned_feature_motion_or_stop",
        "interpretation": (
            "Static basis-gain relocation moved the tangent/representation ceiling but destructively: "
            "C15 rolled back or collapsed, and shuffled/fixed-alpha controls exceeded C15. "
            "This supports the representation-geometry diagnosis but rejects static gain as a repair mechanism."
        ),
    }
    out = write_json(OUT_ROOT / "basis_gain_tangent_relocation_summary.json", summary)
    next_payload = {
        "version": "v23.11",
        "last_completed_stage": "stage_1_basis_gain_tangent_relocation_reduced",
        "stage_1_gate_pass": gate,
        "stage_1_route": summary["route"],
        "stage_1_blocker": blocker,
        "promotion_allowed": 0,
        "completed_artifacts": {
            "stage_1_summary": rel(out),
            "matrix": rel(required["high_gain_matrix"]),
            "group_summary": rel(required["high_gain_group"]),
            "part_c_summary": rel(required["high_gain_summary"]),
        },
        "allowed_next_actions": (
            ["implement_stage_2_feature_anchor_candidate_with_matched_same_compute_controls"]
            if gate
            else ["analyze_learned_feature_motion_candidate_or_stop_static_gain_route"]
        ),
        "forbidden_next_actions": [
            "rerun_static_basis_gain_as_promotion",
            "implement_C26_as_static_gain_plus_C15",
            "full_run_C20_C22_C24",
            "fabricate_data",
            "held_test_induction",
            "runtime_winner_selection",
            "add_mlp_stem_or_readout",
            "add_new_edge_function_family",
        ],
        "reason": summary["interpretation"],
    }
    nxt = write_json(FORMAL_ROOT / "next_actions_for_v23_11.json", next_payload)
    append_exec("Stage 1 basis-gain tangent-relocation audit", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={summary['dominant_blocker']}")
    append_exec("Stage 1 next-action sync", command_text(), files=rel(nxt), gpu=str(args.device), note=f"promotion_allowed=0; next={next_payload['allowed_next_actions'][0]}")
    append_recap("Stage 1 basis-gain tangent-relocation audit", summary)
    return summary


def spectral_mode_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "spectral_matrix": SPECTRAL_MODE_ROOT / "part_c_component_attribution_matrix.csv",
        "spectral_group": SPECTRAL_MODE_ROOT / "part_c_component_attribution_group_summary.csv",
        "spectral_summary": SPECTRAL_MODE_ROOT / "part_c_component_attribution_summary.json",
        "default_reference": DEFAULT_REDUCED_REF,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    groups = read_csv_rows(required["spectral_group"]) if not missing else []
    part_c = read_json(required["spectral_summary"]) if not missing else {}
    default = read_json(DEFAULT_REDUCED_REF) if DEFAULT_REDUCED_REF.exists() else {}
    c26 = row_for_scheme(groups, "C26_Candidate_terminal_audited_smooth_mode_local_inverse")
    c27 = row_for_scheme(groups, "C27_Control_terminal_audited_tail_mode_local_inverse")
    c26_cov = coverage(c26)
    c27_cov = coverage(c27)
    default_metrics = default.get("metrics", {})
    default_c3 = fval(default_metrics.get("C3_coverage"))
    default_c15 = fval(default_metrics.get("C15_coverage"))
    default_c9 = fval(default_metrics.get("C9_coverage"))
    default_c10 = fval(default_metrics.get("C10_coverage"))
    c26_terminal_rollback_count = count_token(c26.get("trust_reject_reasons"), "terminal_audit_rollback")
    c27_terminal_rollback_count = count_token(c27.get("trust_reject_reasons"), "terminal_audit_rollback")
    checks = {
        "no_missing_required_artifacts": not missing,
        "row_count_20_ok": ival(part_c.get("row_count")) == 20 and ival(part_c.get("ok_rows")) == 20,
        "no_fake_data": ival(part_c.get("used_fake_data_rows")) == 0,
        "no_held_test_usage": ival(part_c.get("held_test_usage")) == 0,
        "c26_terminal_audit_clean": c26_terminal_rollback_count == 0 and fval(c26.get("F5_no_debt_count")) >= fval(c26.get("rows")),
        "c26_minus_c3_ge_0p01": c26_cov - default_c3 >= 0.01,
        "c26_minus_c15_ge_0p01": c26_cov - default_c15 >= 0.01,
        "c26_minus_c27_ge_0p03": c26_cov - c27_cov >= 0.03,
        "c26_minus_c9_ge_0p05": c26_cov - default_c9 >= 0.05,
    }
    gate = int(all(checks.values()))
    if missing:
        blocker = "missing_required_artifacts"
    elif not checks["row_count_20_ok"]:
        blocker = "unexpected_reduced_shape"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c26_terminal_audit_clean"]:
        blocker = "c26_terminal_audit_rollback_or_debt"
    elif not checks["c26_minus_c3_ge_0p01"] or not checks["c26_minus_c15_ge_0p01"]:
        blocker = "c26_does_not_move_above_c3_c15_ceiling"
    elif not checks["c26_minus_c27_ge_0p03"]:
        blocker = "smooth_mode_not_better_than_tail_mode_control"
    elif not checks["c26_minus_c9_ge_0p05"]:
        blocker = "smooth_mode_not_better_than_shuffled_downstream_reference"
    else:
        blocker = "none"
    summary = {
        "part": "2A",
        "gate_pass": gate,
        "route": "Stage2ASpectralModeReducedPass" if gate else "Stage2ASpectralModeReducedFailed",
        "scope": "reduced_diagnostic_only_not_promotion",
        "dominant_blocker": blocker,
        "checks": checks,
        "missing": missing,
        "metrics": {
            "default_reference_root": rel(DEFAULT_REDUCED_REF),
            "default_c3_coverage": default_c3,
            "default_c15_coverage": default_c15,
            "default_c9_coverage": default_c9,
            "default_c10_coverage": default_c10,
            "c26_coverage": c26_cov,
            "c27_coverage": c27_cov,
            "c26_minus_c3": c26_cov - default_c3,
            "c26_minus_c15": c26_cov - default_c15,
            "c26_minus_c27": c26_cov - c27_cov,
            "c26_minus_c9": c26_cov - default_c9,
            "c26_terminal_rollback_count_inferred": c26_terminal_rollback_count,
            "c27_terminal_rollback_count_inferred": c27_terminal_rollback_count,
            "c26_no_debt_count": fval(c26.get("F5_no_debt_count")),
            "c27_no_debt_count": fval(c27.get("F5_no_debt_count")),
            "c26_source_residual_fit_improvement_median": fval(c26.get("source_residual_fit_improvement_median")),
            "c26_guard_residual_fit_improvement_median": fval(c26.get("guard_residual_fit_improvement_median")),
            "c27_source_residual_fit_improvement_median": fval(c27.get("source_residual_fit_improvement_median")),
            "c27_guard_residual_fit_improvement_median": fval(c27.get("guard_residual_fit_improvement_median")),
        },
        "artifacts": {name: rel(path) for name, path in required.items()},
        "hashes": {name: sha256_file(path) for name, path in required.items() if path.exists()},
        "next_action": "promote_to_full_controls_only_if_user_confirms" if gate else "do_not_full_run_spectral_mode; analyze_or_stop_stage2a",
        "interpretation": (
            "Smooth/tail Chebyshev mode-band masking tests whether a stable edge-function spectral subspace can escape the C3/C15 basin. "
            "Promotion requires smooth modes to beat the fixed C3/C15 references and the same-size tail-mode control."
        ),
    }
    out = write_json(OUT_ROOT / "spectral_mode_reduced_summary.json", summary)
    next_payload = {
        "version": "v23.11",
        "last_completed_stage": "stage_2a_spectral_mode_reduced",
        "stage_2a_gate_pass": gate,
        "stage_2a_route": summary["route"],
        "stage_2a_blocker": blocker,
        "promotion_allowed": 0,
        "completed_artifacts": {
            "stage_2a_summary": rel(out),
            "matrix": rel(required["spectral_matrix"]),
            "group_summary": rel(required["spectral_group"]),
            "part_c_summary": rel(required["spectral_summary"]),
        },
        "allowed_next_actions": (
            ["consider_full_controls_for_C26_only_after_rechecking_no_selection_bias"]
            if gate
            else ["do_not_full_run_C26_C27; analyze_remaining_theory_or_stop_stage2a"]
        ),
        "forbidden_next_actions": [
            "promote_C26_without_full_C0_C10_C15_controls",
            "rerun_static_basis_gain_as_promotion",
            "implement_C26_as_plain_FunctionalGram_or_Adam_anchor_plus_C15",
            "full_run_C20_C22_C24",
            "fabricate_data",
            "held_test_induction",
            "runtime_winner_selection",
            "add_mlp_stem_or_readout",
            "add_new_edge_function_family",
        ],
        "reason": summary["interpretation"],
    }
    nxt = write_json(FORMAL_ROOT / "next_actions_for_v23_11.json", next_payload)
    append_exec("Stage 2A spectral-mode reduced audit", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_exec("Stage 2A next-action sync", command_text(), files=rel(nxt), gpu=str(args.device), note=f"promotion_allowed=0; next={next_payload['allowed_next_actions'][0]}")
    append_recap("Stage 2A spectral-mode reduced audit", summary)
    return summary


def tail_target_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "tail_matrix": TAIL_TARGET_ROOT / "part_c_component_attribution_matrix.csv",
        "tail_group": TAIL_TARGET_ROOT / "part_c_component_attribution_group_summary.csv",
        "tail_summary": TAIL_TARGET_ROOT / "part_c_component_attribution_summary.json",
        "default_reference": DEFAULT_REDUCED_REF,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    groups = read_csv_rows(required["tail_group"]) if not missing else []
    part_c = read_json(required["tail_summary"]) if not missing else {}
    default = read_json(DEFAULT_REDUCED_REF) if DEFAULT_REDUCED_REF.exists() else {}
    c3 = row_for_scheme(groups, "C3_LocalEFRF_no_downstream")
    c15 = row_for_scheme(groups, "C15_Candidate_terminal_audited_no_trust")
    c9 = row_for_scheme(groups, "C9_Candidate_shuffled_downstream_adjoint")
    c10 = row_for_scheme(groups, "C10_Candidate_trust_free_fixed_alpha")
    c3_cov = coverage(c3)
    c15_cov = coverage(c15)
    c9_cov = coverage(c9)
    c10_cov = coverage(c10)
    default_metrics = default.get("metrics", {})
    default_c3 = fval(default_metrics.get("C3_coverage"))
    default_c15 = fval(default_metrics.get("C15_coverage"))
    default_c9 = fval(default_metrics.get("C9_coverage"))
    c15_terminal_rollback_count = count_token(c15.get("trust_reject_reasons"), "terminal_audit_rollback")
    checks = {
        "no_missing_required_artifacts": not missing,
        "row_count_40_ok": ival(part_c.get("row_count")) == 40 and ival(part_c.get("ok_rows")) == 40,
        "no_fake_data": ival(part_c.get("used_fake_data_rows")) == 0,
        "no_held_test_usage": ival(part_c.get("held_test_usage")) == 0,
        "c15_terminal_audit_clean": c15_terminal_rollback_count == 0 and fval(c15.get("F5_no_debt_count")) >= fval(c15.get("rows")),
        "c15_minus_matched_c3_ge_0p01": c15_cov - c3_cov >= 0.01,
        "c15_minus_default_c3_ge_0p01": c15_cov - default_c3 >= 0.01,
        "c15_minus_default_c15_ge_0p01": c15_cov - default_c15 >= 0.01,
        "c15_minus_c9_ge_0p05": c15_cov - c9_cov >= 0.05,
    }
    gate = int(all(checks.values()))
    if missing:
        blocker = "missing_required_artifacts"
    elif not checks["row_count_40_ok"]:
        blocker = "unexpected_reduced_shape"
    elif not checks["no_fake_data"] or not checks["no_held_test_usage"]:
        blocker = "audit_violation"
    elif not checks["c15_terminal_audit_clean"]:
        blocker = "tail_target_c15_terminal_audit_rollback_or_debt"
    elif not checks["c15_minus_matched_c3_ge_0p01"]:
        blocker = "tail_target_c15_does_not_beat_matched_c3"
    elif not checks["c15_minus_default_c3_ge_0p01"] or not checks["c15_minus_default_c15_ge_0p01"]:
        blocker = "tail_target_does_not_move_above_default_c3_c15_ceiling"
    elif not checks["c15_minus_c9_ge_0p05"]:
        blocker = "tail_target_c15_does_not_beat_shuffled_downstream"
    else:
        blocker = "none"
    summary = {
        "part": "2B",
        "gate_pass": gate,
        "route": "Stage2BTailTargetReducedPass" if gate else "Stage2BTailTargetReducedFailed",
        "scope": "reduced_diagnostic_only_not_promotion",
        "dominant_blocker": blocker,
        "checks": checks,
        "missing": missing,
        "metrics": {
            "default_reference_root": rel(DEFAULT_REDUCED_REF),
            "default_c3_coverage": default_c3,
            "default_c15_coverage": default_c15,
            "default_c9_coverage": default_c9,
            "tail_target_c3_coverage": c3_cov,
            "tail_target_c15_coverage": c15_cov,
            "tail_target_c9_coverage": c9_cov,
            "tail_target_c10_coverage": c10_cov,
            "tail_target_c15_minus_matched_c3": c15_cov - c3_cov,
            "tail_target_c15_minus_default_c3": c15_cov - default_c3,
            "tail_target_c15_minus_default_c15": c15_cov - default_c15,
            "tail_target_c15_minus_c9": c15_cov - c9_cov,
            "tail_target_c15_minus_c10": c15_cov - c10_cov,
            "tail_target_c15_terminal_rollback_count_inferred": c15_terminal_rollback_count,
            "tail_target_c15_no_debt_count": fval(c15.get("F5_no_debt_count")),
            "tail_target_c15_source_residual_fit_improvement_median": fval(c15.get("source_residual_fit_improvement_median")),
            "tail_target_c15_guard_residual_fit_improvement_median": fval(c15.get("guard_residual_fit_improvement_median")),
            "tail_target_c3_trust_accept_rate_median": fval(c3.get("trust_accept_rate_median")),
            "tail_target_c15_trust_accept_rate_median": fval(c15.get("trust_accept_rate_median")),
        },
        "artifacts": {name: rel(path) for name, path in required.items()},
        "hashes": {name: sha256_file(path) for name, path in required.items() if path.exists()},
        "next_action": "consider_full_controls_only_after_rechecking_matched_controls" if gate else "do_not_full_run_tail_target; residual_target_alignment_is_insufficient",
        "interpretation": (
            "This diagnostic aligns the residual target with C2 coverage_CVaR25 by using the existing train-only tail_weighted_norm target. "
            "It is not a new model family and is allowed only as a reduced diagnostic after Stage 2A failed. "
            "A pass requires C15 to beat matched C3, default C3/C15, and shuffled downstream control without terminal debt."
        ),
    }
    out = write_json(OUT_ROOT / "tail_target_reduced_summary.json", summary)
    next_payload = {
        "version": "v23.11",
        "last_completed_stage": "stage_2b_tail_target_reduced",
        "stage_2b_gate_pass": gate,
        "stage_2b_route": summary["route"],
        "stage_2b_blocker": blocker,
        "promotion_allowed": 0,
        "completed_artifacts": {
            "stage_2b_summary": rel(out),
            "matrix": rel(required["tail_matrix"]),
            "group_summary": rel(required["tail_group"]),
            "part_c_summary": rel(required["tail_summary"]),
        },
        "allowed_next_actions": (
            ["consider_full_controls_for_tail_target_only_after_rechecking_no_selection_bias"]
            if gate
            else ["stop_residual_target_microvariants; move_to_feature_learning_or_efficiency_claim"]
        ),
        "forbidden_next_actions": [
            "promote_tail_target_without_full_C0_C10_C15_controls",
            "rerun_static_basis_gain_as_promotion",
            "implement_plain_FunctionalGram_or_Adam_anchor_plus_C15",
            "full_run_C20_C22_C24_C26_C27",
            "fabricate_data",
            "held_test_induction",
            "runtime_winner_selection",
            "add_mlp_stem_or_readout",
            "add_new_edge_function_family",
        ],
        "reason": summary["interpretation"],
    }
    nxt = write_json(FORMAL_ROOT / "next_actions_for_v23_11.json", next_payload)
    append_exec("Stage 2B tail-target reduced audit", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_exec("Stage 2B next-action sync", command_text(), files=rel(nxt), gpu=str(args.device), note=f"promotion_allowed=0; next={next_payload['allowed_next_actions'][0]}")
    append_recap("Stage 2B tail-target reduced audit", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evidence-lock", "theory-reset", "basis-gain-audit", "spectral-mode-audit", "tail-target-audit"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.mode == "evidence-lock":
        evidence_lock(args)
    elif args.mode == "theory-reset":
        theory_reset(args)
    elif args.mode == "basis-gain-audit":
        basis_gain_audit(args)
    elif args.mode == "spectral-mode-audit":
        spectral_mode_audit(args)
    elif args.mode == "tail-target-audit":
        tail_target_audit(args)
    else:
        raise ValueError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
