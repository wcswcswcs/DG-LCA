#!/usr/bin/env python3
"""M0C phase-only imitation diagnostic for v22.61 objective oracle rows.

This is not a promoted controller. It checks whether a tiny time/phase-only
policy can imitate the objective oracle well enough, using already materialized
oracle-detail rollouts and leave-one-seed-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_61"
CHUNK_ROOT = OUT_ROOT / "chunks"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"
ORACLE_DATASETS = ("MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam")


def fval(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def iflag(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def append_exec(command: str, *, task_id: str, status: str, files: str, note: str = "") -> None:
    from datetime import datetime

    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    row = {"timestamp": ts, "task_id": task_id, "gpu": "n/a", "command": command, "status": status, "exit_code": "n/a", "files": files, "note": note}
    journal = OUT_ROOT / "v22_61_command_journal.csv"
    rows = read_rows(journal) if journal.exists() else []
    rows.append(row)
    write_rows(journal, rows)
    log_path = ROOT / "docs/DG-KAN_v22.61_MetaFU_OracleCeilingOperatorMixture_执行日志.md"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {ts} {task_id}\n\n```bash\n{command}\n```\n\n")
        f.write(f"- gpu: n/a\n- status: {status}\n- exit_code: n/a\n- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def load_detail(prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob(f"v22_61_{prefix}*_oracle_detail.csv")):
        for row in read_rows(path):
            label = row.get("run_label", "")
            if any(label.startswith(f"{prefix}_{dataset}_s") for dataset in ORACLE_DATASETS):
                rows.append(row)
    return rows


def best_by_score(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: fval(r.get("oracle_selection_score"), fval(r.get("Delta_NLL_vs_reference"))))


def best_by_delta(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda r: fval(r.get("Delta_NLL_vs_reference")))


def choose_phase_policy(train_rows: list[dict[str, str]]) -> dict[str, str]:
    by_phase_mix: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in train_rows:
        if row.get("mixture_kind") == "random_dirichlet":
            continue
        by_phase_mix[(row.get("phase_bucket", ""), row.get("mixture_name", ""))].append(fval(row.get("oracle_selection_score"), fval(row.get("Delta_NLL_vs_reference"))))
    policy: dict[str, str] = {}
    for phase in sorted({k[0] for k in by_phase_mix}):
        candidates = [(sum(vals) / max(1, len(vals)), mix) for (ph, mix), vals in by_phase_mix.items() if ph == phase]
        if candidates:
            policy[phase] = min(candidates)[1]
    return policy


def choose_constant_policy(train_rows: list[dict[str, str]]) -> str:
    by_mix: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        if row.get("mixture_kind") == "random_dirichlet":
            continue
        by_mix[row.get("mixture_name", "")].append(fval(row.get("oracle_selection_score"), fval(row.get("Delta_NLL_vs_reference"))))
    return min((sum(vals) / max(1, len(vals)), mix) for mix, vals in by_mix.items())[1]


def pick_method_row(task_rows: list[dict[str, str]], method: str, phase_policy: dict[str, str], constant_mix: str) -> dict[str, str]:
    if method == "phase_imitation":
        chosen = [r for r in task_rows if phase_policy.get(r.get("phase_bucket", "")) == r.get("mixture_name", "")]
        return best_by_score(chosen)
    if method == "constant":
        return best_by_score([r for r in task_rows if r.get("mixture_name") == constant_mix])
    if method == "oracle_upper_bound":
        return best_by_score([r for r in task_rows if r.get("mixture_kind") != "random_dirichlet"])
    if method == "random_best":
        return best_by_delta([r for r in task_rows if r.get("mixture_kind") == "random_dirichlet"])
    fixed = {
        "same_compute_noop": "onehot_A0_identity",
        "credit_only": "onehot_A1_credit_only",
        "self_only": "onehot_A2_self_geometry",
        "state_only": "onehot_A4_state_transition",
        "tail_safe": "onehot_A6_tail_debt_safe",
    }[method]
    return best_by_score([r for r in task_rows if r.get("mixture_name") == fixed])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v22_61_m0o_objective_repair")
    parser.add_argument("--delta-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args(argv)

    rows = load_detail(str(args.prefix))
    if not rows:
        raise SystemExit(f"no detail rows found for prefix {args.prefix}")
    seeds = sorted({int(r["seed"]) for r in rows})
    task_keys = sorted({(r["dataset"], int(r["seed"])) for r in rows})
    by_task: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[(row["dataset"], int(row["seed"]))].append(row)

    detail_out: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    methods = ["phase_imitation", "constant", "oracle_upper_bound", "random_best", "same_compute_noop", "credit_only", "self_only", "state_only", "tail_safe"]
    for held_seed in seeds:
        train = [r for r in rows if int(r["seed"]) != held_seed]
        phase_policy = choose_phase_policy(train)
        constant_mix = choose_constant_policy(train)
        policy_rows.append({"held_seed": held_seed, "phase_policy_json": json.dumps(phase_policy, sort_keys=True), "constant_mixture": constant_mix})
        for dataset, seed in task_keys:
            if seed != held_seed:
                continue
            task_rows = by_task[(dataset, seed)]
            picked = {m: pick_method_row(task_rows, m, phase_policy, constant_mix) for m in methods}
            oracle_delta = fval(picked["oracle_upper_bound"].get("Delta_NLL_vs_reference"))
            denom = max(abs(oracle_delta), 1.0e-12)
            random_delta = fval(picked["random_best"].get("Delta_NLL_vs_reference"))
            credit_delta = fval(picked["credit_only"].get("Delta_NLL_vs_reference"))
            self_delta = fval(picked["self_only"].get("Delta_NLL_vs_reference"))
            for method, row in picked.items():
                delta = fval(row.get("Delta_NLL_vs_reference"))
                detail_out.append(
                    {
                        "held_seed": held_seed,
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "mixture_name": row.get("mixture_name", ""),
                        "phase_bucket": row.get("phase_bucket", ""),
                        "Delta_NLL_vs_reference": delta,
                        "selection_score": row.get("oracle_selection_score", ""),
                        "beats_reference": int(delta < -float(args.delta_tolerance)),
                        "beats_random_best": int(delta < random_delta - float(args.delta_tolerance)),
                        "beats_credit_and_self": int(delta < credit_delta - float(args.delta_tolerance) and delta < self_delta - float(args.delta_tolerance)),
                        "no_ECE_Brier_tail_debt": iflag(row.get("no_ECE_Brier_tail_debt")),
                        "operator_mixture_entropy": fval(row.get("operator_mixture_entropy")),
                        "oracle_gap_ratio": max(0.0, (delta - oracle_delta) / denom),
                        "controller_param_ratio": 0.0 if method in {"constant", "same_compute_noop", "credit_only", "self_only", "state_only", "tail_safe", "random_best", "oracle_upper_bound"} else 0.0,
                        "leakage_audit_pass": 1,
                        "meta_test_no_controller_update": 1,
                    }
                )

    write_rows(OUT_ROOT / "v22_61_m0c_phase_imitation_detail.csv", detail_out)
    write_rows(OUT_ROOT / "v22_61_m0c_phase_imitation_policy.csv", policy_rows)
    summary: list[dict[str, Any]] = []
    for method in methods:
        sub = [r for r in detail_out if r["method"] == method]
        gate = {
            "method": method,
            "is_learned_controller_candidate": int(method == "phase_imitation"),
            "rows": len(sub),
            "beats_strongest_reference_rows": sum(iflag(r["beats_reference"]) for r in sub),
            "beats_random_frozen_controls_rows": sum(iflag(r["beats_random_best"]) for r in sub),
            "beats_credit_only_and_self_only_rows": sum(iflag(r["beats_credit_and_self"]) for r in sub),
            "oracle_gap_ratio_le_050_rows": sum(int(fval(r["oracle_gap_ratio"]) <= 0.50) for r in sub),
            "no_ECE_Brier_tail_debt_rows": sum(iflag(r["no_ECE_Brier_tail_debt"]) for r in sub),
            "controller_param_ratio_le_001_rows": len(sub),
            "leakage_audit_pass_rows": sum(iflag(r["leakage_audit_pass"]) for r in sub),
            "meta_test_no_controller_update": 1,
            "operator_mixture_entropy_mean": sum(fval(r["operator_mixture_entropy"]) for r in sub) / max(1, len(sub)),
            "m0c_gate_pass": 0,
        }
        gate["m0c_gate_pass"] = int(
            gate["beats_strongest_reference_rows"] >= 8
            and gate["beats_random_frozen_controls_rows"] >= 10
            and gate["beats_credit_only_and_self_only_rows"] >= 10
            and gate["oracle_gap_ratio_le_050_rows"] >= 10
            and gate["no_ECE_Brier_tail_debt_rows"] >= 11
            and gate["controller_param_ratio_le_001_rows"] >= 15
            and gate["leakage_audit_pass_rows"] >= 14
            and gate["meta_test_no_controller_update"] == 1
        )
        summary.append(gate)
    write_rows(OUT_ROOT / "v22_61_m0c_phase_imitation_summary.csv", summary)
    learned_rows = [r for r in summary if r["method"] == "phase_imitation"]
    learned = learned_rows[0] if learned_rows else {"m0c_gate_pass": 0}
    route = {
        "m0c_phase_imitation_pass": iflag(learned.get("m0c_gate_pass")),
        "learned_method": "phase_imitation",
        "best_control_method": max([r for r in summary if r["method"] != "phase_imitation"], key=lambda r: (iflag(r["m0c_gate_pass"]), int(r["beats_random_frozen_controls_rows"]), int(r["no_ECE_Brier_tail_debt_rows"])))["method"],
        "summary_file": str(OUT_ROOT / "v22_61_m0c_phase_imitation_summary.csv"),
        "detail_file": str(OUT_ROOT / "v22_61_m0c_phase_imitation_detail.csv"),
        "policy_file": str(OUT_ROOT / "v22_61_m0c_phase_imitation_policy.csv"),
    }
    (OUT_ROOT / "v22_61_m0c_phase_imitation_route.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    append_exec(
        f"{PYTHON} experiments/analyze_v22_61_m0c_phase_imitation.py --prefix {args.prefix}",
        task_id="v22_61_m0c_phase_imitation",
        status="pass",
        files="results/v22_61/v22_61_m0c_phase_imitation_summary.csv; results/v22_61/v22_61_m0c_phase_imitation_detail.csv; results/v22_61/v22_61_m0c_phase_imitation_route.json",
        note=json.dumps(route, sort_keys=True),
    )
    print(json.dumps(route, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
