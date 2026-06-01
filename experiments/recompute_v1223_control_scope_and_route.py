#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp


def _refresh_actuator_scope(out_dir: Path) -> dict[str, Any]:
    rows = exp.read_csv_rows(out_dir / "v1223_actuator_response_dictionary.csv")
    if not rows:
        rows = exp.read_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv")
    for row in rows:
        gap = exp.matched_control_gap_roleaware(rows, row)
        row["control_gap"] = gap if math.isfinite(gap) else ""
        row["matched_control_gap"] = row["control_gap"]
        eligible = exp.safe_int(row.get("diagnostic_only"), 0) == 0 and exp.safe_int(row.get("uses_label"), 0) == 0 and exp.safe_int(row.get("uses_ce_vector"), 0) == 0
        exploration = eligible and exp.safe_int(row.get("role_safe_movement_pass"), 0) and exp.safe_int(row.get("release_audit_pass"), 0) and math.isfinite(gap) and gap >= 0.002
        official = exploration and gap >= 0.005
        row["exploratory_release_gate"] = int(exploration)
        row["official_release_gate"] = int(official)
        row["control_scope"] = "dataset_seed_budget_sign"
        row["failure_reason"] = "" if exploration else "role_movement_or_release_or_control_gap_gate_failed"
    safety_rows = [dict(r, stage="V1223_ACTUATOR_SAFETY_ROLEAWARE") for r in rows]
    control_rows = [dict(r, stage="V1223_ACTUATOR_CONTROLS") for r in rows if str(r.get("basis_type")) == "matched_control"]
    exp.write_csv_rows(out_dir / "v1223_actuator_response_dictionary.csv", rows)
    exp.write_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv", safety_rows)
    exp.write_csv_rows(out_dir / "v1223_actuator_controls.csv", control_rows)
    non_control = [r for r in rows if str(r.get("basis_type")) != "matched_control"]
    safe_keys = {(r.get("dataset", ""), r.get("seed", "")) for r in non_control if exp.safe_int(r.get("role_safe_movement_pass"), 0)}
    release_keys = {(r.get("dataset", ""), r.get("seed", "")) for r in non_control if exp.safe_int(r.get("release_audit_pass"), 0)}
    exploratory_keys = {(r.get("dataset", ""), r.get("seed", "")) for r in non_control if exp.safe_int(r.get("exploratory_release_gate"), 0)}
    official_rows = [r for r in non_control if exp.safe_int(r.get("official_release_gate"), 0)]
    return {
        "actuator_rows": len(rows),
        "actuator_role_safe_movement_rows": sum(exp.safe_int(r.get("role_safe_movement_pass"), 0) for r in non_control),
        "actuator_role_safe_dataset_seed_count": len(safe_keys),
        "actuator_release_audit_rows": sum(exp.safe_int(r.get("release_audit_pass"), 0) for r in non_control),
        "actuator_release_dataset_seed_count": len(release_keys),
        "actuator_exploratory_release_dataset_seed_count": len(exploratory_keys),
        "actuator_official_gate_rows": len(official_rows),
        "actuator_exploratory_success": int(len(exploratory_keys) >= 3),
        "actuator_official_gate_pass": int(len(official_rows) >= 8),
        "control_scope_fix_applied": 1,
    }


def _refresh_route(out_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    state.update(exp.write_functional_b_v4(out_dir, state))
    route, minimum_success, fail_reason = exp.decide_route(state)
    official_success = int(route.startswith("S5"))
    final_stop_allowed = int(
        official_success
        or (
            exp.safe_int(state.get("hard_budget_exhausted"), 0) == 1
            and exp.safe_int(state.get("mandatory_exploration_levels_executed"), 0) == 1
            and route == "R4-FunctionalMechanismNoGoAfterAllFallbacks"
        )
    )
    state.update({
        "route": route,
        "minimum_success": minimum_success,
        "fail_reason": fail_reason,
        "official_success_reached": official_success,
        "final_stop_allowed": final_stop_allowed,
    })
    final_audit_path = out_dir / "v1223_final_stop_audit.json"
    final_audit: dict[str, Any]
    if final_audit_path.exists():
        final_audit = json.loads(final_audit_path.read_text())
    else:
        final_audit = {"stage": "V1223_FINAL_STOP_AUDIT"}
    final_audit.update({
        "route": route,
        "final_stop_allowed": final_stop_allowed,
        "official_success_reached": official_success,
        "hard_budget_exhausted": state.get("hard_budget_exhausted", 0),
        "mandatory_exploration_levels_executed": state.get("mandatory_exploration_levels_executed", 0),
        "explicit_user_stop_flag": 0,
        "control_scope_fix_applied": 1,
    })
    exp.write_json(final_audit_path, final_audit)
    zip_path = exp.package_zip(out_dir)
    state["code_review_packet"] = exp.rel(zip_path)
    state["code_review_packet_sha256"] = exp.sha256_file(zip_path)
    req = exp.write_required_manifest(out_dir)
    state.update(req)
    final_audit["required_artifact_missing_count"] = req.get("required_artifact_missing_count", "")
    exp.write_json(final_audit_path, final_audit)
    exp.write_hash_manifest(out_dir)
    exp.write_json(out_dir / "v1223_route_decision.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    state = json.loads((out_dir / "v1223_route_decision.json").read_text())
    state.update(_refresh_actuator_scope(out_dir))
    state = _refresh_route(out_dir, state)
    keys = [
        "route",
        "minimum_success",
        "fail_reason",
        "official_success_reached",
        "final_stop_allowed",
        "functional_p3_pass_count",
        "p4_open",
        "p4_pass",
        "actuator_exploratory_release_dataset_seed_count",
        "actuator_official_gate_rows",
        "actuator_exploratory_success",
        "actuator_official_gate_pass",
        "control_scope_fix_applied",
        "code_review_packet_sha256",
        "required_artifact_missing_count",
    ]
    print(json.dumps({k: state.get(k) for k in keys}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
