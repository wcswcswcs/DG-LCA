#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v22_68"

V67_DETAIL = ROOT / "results" / "v22_67" / "v22_67_part_b_candidate_detail_mcga_over_poet_fsclip_eta001_residual_rank4.csv"
V67_SUMMARY = ROOT / "results" / "v22_67" / "v22_67_part_b_mlp_robustness_summary_mcga_over_poet_fsclip_eta001_residual_rank4.json"
V67_DECOMP = ROOT / "results" / "v22_67" / "v22_67_part_b_external_winner_decomposition_mcga_over_poet_fsclip_eta001_residual_rank4.csv"
V68_DETAIL = OUT / "v22_68_part_b_candidate_detail.csv"
V68_SUMMARY = OUT / "v22_68_part_b_mlp_robustness_summary.json"
V68_DECOMP = OUT / "v22_68_part_b_external_failure_decomposition.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("steps", "")))


def flag(value: Any) -> int:
    return int(str(value).strip() in {"1", "true", "True", "yes"})


def fnum(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def metric(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row.get(name, "")
    return ""


def summarize_steps(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_step: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(str(row.get("steps", "")), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for step, step_rows in sorted(by_step.items(), key=lambda kv: int(kv[0])):
        out[step] = {
            "rows": len(step_rows),
            "beats_strongest": sum(flag(r.get("beats_strongest_NLL")) for r in step_rows),
            "beats_external": sum(flag(r.get("beats_external_OET_NLL")) for r in step_rows),
            "beats_best_control": sum(flag(r.get("beats_best_control_NLL")) for r in step_rows),
            "beats_same_generator_controls": sum(flag(r.get("beats_same_generator_controls_NLL")) for r in step_rows),
            "best_control_fail_rows": [
                {
                    "dataset": r.get("dataset", ""),
                    "seed": r.get("seed", ""),
                    "steps": r.get("steps", ""),
                    "Delta_NLL_vs_best_control": fnum(r.get("Delta_NLL_vs_best_control")),
                    "failure_mode": r.get("failure_mode", ""),
                }
                for r in step_rows
                if not flag(r.get("beats_best_control_NLL"))
            ],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v67 = read_rows(V67_DETAIL)
    v68 = read_rows(V68_DETAIL)
    v67_by_key = {key(r): r for r in v67}
    v68_by_key = {key(r): r for r in v68}
    v67_decomp_by_key = {key(r): r for r in read_rows(V67_DECOMP)}
    v68_decomp_by_key = {key(r): r for r in read_rows(V68_DECOMP)}

    common = sorted(set(v67_by_key) & set(v68_by_key), key=lambda k: (int(k[2]), k[0], int(k[1])))
    overlap_rows: list[dict[str, Any]] = []
    flip_counters = Counter()
    for k in common:
        old = v67_by_key[k]
        new = v68_by_key[k]
        old_dec = v67_decomp_by_key.get(k, {})
        new_dec = v68_decomp_by_key.get(k, {})
        row = {
            "dataset": k[0],
            "seed": k[1],
            "steps": k[2],
            "v67_candidate_NLL": fnum(old.get("final_NLL")),
            "v68_candidate_NLL": fnum(new.get("final_NLL")),
            "candidate_NLL_delta_v68_minus_v67": None,
            "v67_best_control_pass": flag(old.get("beats_best_control_NLL")),
            "v68_best_control_pass": flag(new.get("beats_best_control_NLL")),
            "v67_same_generator_pass": flag(old.get("beats_same_generator_controls_NLL")),
            "v68_same_generator_pass": flag(new.get("beats_same_generator_controls_NLL")),
            "v67_external_pass": flag(old.get("beats_external_OET_NLL")),
            "v68_external_pass": flag(new.get("beats_external_OET_NLL")),
            "v67_delta_best_control": fnum(old.get("Delta_NLL_vs_best_control")),
            "v68_delta_best_control": fnum(new.get("Delta_NLL_vs_best_control")),
            "v67_delta_external": fnum(metric(old, "Delta_NLL_vs_external_OET", "Delta_NLL_vs_best_external")),
            "v68_delta_external": fnum(metric(new, "Delta_NLL_vs_external_OET", "Delta_NLL_vs_best_external")),
            "v67_best_control_method": old_dec.get("best_control_method", ""),
            "v67_best_control_NLL": fnum(old_dec.get("best_control_NLL")),
            "v68_best_control_method": new_dec.get("best_control_method", ""),
            "v68_best_control_NLL": fnum(new_dec.get("best_control_NLL")),
            "v67_failure_mode": old.get("failure_mode", ""),
            "v68_failure_mode": new.get("failure_mode", ""),
        }
        old_nll = row["v67_candidate_NLL"]
        new_nll = row["v68_candidate_NLL"]
        if old_nll is not None and new_nll is not None:
            row["candidate_NLL_delta_v68_minus_v67"] = new_nll - old_nll
        if row["v67_best_control_pass"] != row["v68_best_control_pass"]:
            flip_counters["best_control"] += 1
        if row["v67_same_generator_pass"] != row["v68_same_generator_pass"]:
            flip_counters["same_generator_controls"] += 1
        if row["v67_external_pass"] != row["v68_external_pass"]:
            flip_counters["external"] += 1
        overlap_rows.append(row)

    overlap_csv = OUT / "v22_68_part_b_v22_67_eta001_delta_audit_overlap.csv"
    if overlap_rows:
        with overlap_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overlap_rows)

    common_best_control_flips = [r for r in overlap_rows if r["v67_best_control_pass"] != r["v68_best_control_pass"]]
    same_candidate_changed_rows = [
        r
        for r in overlap_rows
        if r["candidate_NLL_delta_v68_minus_v67"] is not None
        and abs(float(r["candidate_NLL_delta_v68_minus_v67"])) > 1.0e-12
    ]
    summary = {
        "artifact": "v22_68_part_b_v22_67_eta001_delta_audit",
        "generated_at_sg": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S %z"),
        "input_files": {
            "v22_67_detail": str(V67_DETAIL.relative_to(ROOT)),
            "v22_67_summary": str(V67_SUMMARY.relative_to(ROOT)),
            "v22_67_decomposition": str(V67_DECOMP.relative_to(ROOT)),
            "v22_68_detail": str(V68_DETAIL.relative_to(ROOT)),
            "v22_68_summary": str(V68_SUMMARY.relative_to(ROOT)),
            "v22_68_decomposition": str(V68_DECOMP.relative_to(ROOT)),
        },
        "v22_67_summary": read_json(V67_SUMMARY),
        "v22_68_summary": read_json(V68_SUMMARY),
        "v22_67_step_summary": summarize_steps(v67),
        "v22_68_step_summary": summarize_steps(v68),
        "key_sets": {
            "v22_67_rows": len(v67),
            "v22_68_rows": len(v68),
            "common_keys": len(common),
            "v22_67_only_keys": len(set(v67_by_key) - set(v68_by_key)),
            "v22_68_only_keys": len(set(v68_by_key) - set(v67_by_key)),
            "v22_67_only_step_counts": dict(Counter(k[2] for k in set(v67_by_key) - set(v68_by_key))),
            "v22_68_only_step_counts": dict(Counter(k[2] for k in set(v68_by_key) - set(v67_by_key))),
        },
        "overlap_pass_flip_counts": dict(flip_counters),
        "same_key_candidate_nll_changed_rows": len(same_candidate_changed_rows),
        "best_control_pass_flip_rows": common_best_control_flips,
        "interpretation": {
            "strict_current_state": "v22.68 official Part B must be evaluated on 800/1200 with current matched controls.",
            "v22_67_eta001_pass_context": "The v22.67 eta001 pass artifact covers 400/800 and used its then-current comparison set.",
            "main_observed_delta": "All common-key best-control pass flips are Spam step800 rows; candidate NLL is unchanged on those rows, while v22.68 compares against stronger matched controls.",
            "audit_constraint": "This audit does not relax controls or promote row-wise best methods.",
        },
    }
    summary_path = OUT / "v22_68_part_b_v22_67_eta001_delta_audit.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path.relative_to(ROOT)),
        "overlap_csv": str(overlap_csv.relative_to(ROOT)),
        "common_keys": len(common),
        "best_control_pass_flips": len(common_best_control_flips),
        "same_key_candidate_nll_changed_rows": len(same_candidate_changed_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
