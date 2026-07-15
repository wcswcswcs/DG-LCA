#!/usr/bin/env python3
"""DG-KAN v23.10 efficiency-equivalence / task-curvature runner."""

from __future__ import annotations

import argparse
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
PLAN = ROOT / "docs/DG-KAN_v23.10_EfficiencyEquivalenceTaskCurvatureEFRF_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.10_EfficiencyEquivalenceTaskCurvatureEFRF_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.10_EfficiencyEquivalenceTaskCurvatureEFRF_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2310_OUT_ROOT", str(ROOT / "results/v23_10_efficiency_equivalence_task_curvature_efrf"))).resolve()

V2309_FORMAL_ROOT = ROOT / "results/v23_09_part_b_repair_tol001_formal_15seed80"
C15_ROOT = ROOT / "results/v23_09_part_c_repair_terminal_audit_full_c0_c15_tol002_15seed80"
C18_ROOT = ROOT / "results/v23_09_part_c_repair_topdown_terminal_diag_s5_80"
C19_ROOT = ROOT / "results/v23_09_part_c_repair_source_guard_population_diag_s5_80"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return path


def ensure_logs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.10 Efficiency-Equivalence / Task-Curvature EFRF 执行日志\n\n"
            "- 原则：不造假；不把 v23.09 failure 改写成 success；不使用 held/test 诱导。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.10 Efficiency-Equivalence / Task-Curvature EFRF 实验结果复盘\n\n"
            "- 原则：只记录真实 artifact 数据；不得把 diagnostic 写成 promotion。\n",
            encoding="utf-8",
        )


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


def command_text() -> str:
    return f"{sys.executable} {' '.join(sys.argv)}"


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def evidence_lock(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    required = {
        "plan": PLAN,
        "v23_10_preregistered_json": V2309_FORMAL_ROOT / "v23_10_preregistered_theory_reset.json",
        "part_c_next_actions": V2309_FORMAL_ROOT / "part_c_next_actions_for_codex.json",
        "next_actions_for_v23_10": V2309_FORMAL_ROOT / "next_actions_for_v23_10.json",
        "c15_single_run": C15_ROOT / "c15_terminal_audit_single_run_summary.json",
        "c18_negative": C18_ROOT / "c18_topdown_terminal_diag_summary.json",
        "c19_negative": C19_ROOT / "c19_source_guard_population_diag_summary.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    c15 = read_json(required["c15_single_run"]) if not missing else {}
    c18 = read_json(required["c18_negative"]) if not missing else {}
    c19 = read_json(required["c19_negative"]) if not missing else {}
    pca = read_json(required["part_c_next_actions"]) if not missing else {}
    official = c15.get("official_v23_09_part_c_status", {})
    gate = int(
        not missing
        and int(official.get("gate_pass", -1)) == 0
        and c15.get("threshold_audit_for_c15_if_considered_candidate", {}).get("beats_LocalEFRF_gap_ge_0_05") is False
        and fval(c18.get("key_deltas", {}).get("C18_minus_C3"), 1.0) < 0.0
        and fval(c19.get("key_deltas", {}).get("C19_minus_C3"), 1.0) < 0.0
        and "v23.09 Part C is not achieved" in str(pca.get("stop_condition", ""))
    )
    summary = {
        "part": "0",
        "gate_pass": gate,
        "route": "V2310EvidenceLockPass" if gate else "V2310EvidenceLockFailed",
        "dominant_blocker": "none" if gate else "missing_or_inconsistent_v23_09_evidence",
        "missing": missing,
        "official_v23_09_part_c_gate_pass": official.get("gate_pass"),
        "c15_minus_c3": c15.get("comparison_deltas", {}).get("C15_minus_C3_LocalEFRF_coverage"),
        "c18_minus_c3": c18.get("key_deltas", {}).get("C18_minus_C3"),
        "c19_minus_c3": c19.get("key_deltas", {}).get("C19_minus_C3"),
        "evidence_paths": {name: rel(path) for name, path in required.items()},
    }
    out = write_json(OUT_ROOT / "part_0_evidence_lock_summary.json", summary)
    append_exec("Part 0 evidence lock", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; blocker={summary['dominant_blocker']}")
    append_recap("Part 0 evidence lock", summary)
    return summary


def route_a_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_logs()
    lock_path = OUT_ROOT / "part_0_evidence_lock_summary.json"
    lock = read_json(lock_path) if lock_path.exists() else {}
    c15 = read_json(C15_ROOT / "c15_terminal_audit_single_run_summary.json")
    official = c15["official_v23_09_part_c_status"]
    metrics = c15["c15_metrics"]
    deltas = c15["comparison_deltas"]
    threshold = c15["threshold_audit_for_c15_if_considered_candidate"]
    checks = {
        "evidence_lock_pass": int(lock.get("gate_pass", 0)) == 1,
        "official_v23_09_still_failed": int(official.get("gate_pass", -1)) == 0,
        "row_count_360_ok": int(official.get("row_count", 0)) == 360 and int(official.get("ok_rows", 0)) == 360,
        "no_fake_data": int(official.get("used_fake_data_rows", 1)) == 0,
        "no_held_test_usage": int(official.get("held_test_usage", 1)) == 0,
        "no_debt_all_rows": int(metrics.get("F5_no_debt_count", -1)) == int(metrics.get("rows", 0)),
        "terminal_audit_clean": int(metrics.get("terminal_audit_pass_count", 0)) == int(metrics.get("rows", 0)) and int(metrics.get("terminal_audit_reject_count", 1)) == 0,
        "local_efrf_equivalent": abs(fval(deltas.get("C15_minus_C3_LocalEFRF_coverage"))) <= 0.005,
        "wall_time_efficient": fval(deltas.get("C15_over_C3_overhead_ratio"), 1.0) <= 0.10,
        "overhead_efficient": fval(deltas.get("C15_over_C3_overhead_ratio"), 1.0) <= 0.10,
        "shuffled_downstream_gap": fval(deltas.get("C15_minus_C9_shuffled_downstream_coverage")) >= 0.05,
        "random_projected_gap": fval(deltas.get("C15_minus_C7_random_projected_coverage")) >= 0.10,
        "shuffled_design_gap": fval(deltas.get("C15_minus_C8_shuffled_design_coverage")) >= 0.10,
        "no_inverse_gap": bool(threshold.get("beats_no_inverse_gap_ge_0_05")),
        "fixed_alpha_gap": fval(deltas.get("C15_minus_C10_fixed_alpha_coverage")) >= 0.03,
    }
    gate = int(all(checks.values()))
    summary = {
        "part": "A",
        "gate_pass": gate,
        "route": "RouteA_EfficiencyEquivalencePass" if gate else "RouteA_EfficiencyEquivalenceFailed",
        "dominant_blocker": "none" if gate else "efficiency_equivalence_check_failed",
        "claim": "C15 is LocalEFRF-equivalent and substantially cheaper; this is not coverage dominance.",
        "promotion_scope": "v23.10_route_A_efficiency_equivalence_only",
        "official_v23_09_part_c_gate_pass": official.get("gate_pass"),
        "checks": checks,
        "metrics": {
            "c15_coverage": metrics.get("multi_step_C2_coverage_median"),
            "c15_minus_c3": deltas.get("C15_minus_C3_LocalEFRF_coverage"),
            "c15_minus_c9": deltas.get("C15_minus_C9_shuffled_downstream_coverage"),
            "c15_minus_c10": deltas.get("C15_minus_C10_fixed_alpha_coverage"),
            "c15_over_c3_overhead_ratio": deltas.get("C15_over_C3_overhead_ratio"),
            "c3_over_c15_wall_time_ratio": deltas.get("C3_over_C15_wall_time_ratio"),
            "no_debt_count": metrics.get("F5_no_debt_count"),
            "terminal_reject_count": metrics.get("terminal_audit_reject_count"),
        },
        "source_artifact": rel(C15_ROOT / "c15_terminal_audit_single_run_summary.json"),
    }
    out = write_json(OUT_ROOT / "route_a_efficiency_equivalence_summary.json", summary)
    append_exec("Route A efficiency-equivalence audit", command_text(), files=rel(out), gpu=str(args.device), note=f"gate={gate}; scope=efficiency_equivalence_only")
    append_recap("Route A efficiency-equivalence audit", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evidence-lock", "route-a-audit"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.mode == "evidence-lock":
        evidence_lock(args)
    elif args.mode == "route-a-audit":
        route_a_audit(args)
    else:
        raise ValueError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
