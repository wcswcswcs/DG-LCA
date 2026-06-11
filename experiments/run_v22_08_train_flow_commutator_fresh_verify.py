#!/usr/bin/env python3
"""Repeat verification for v22.08 C-O12 train-flow commutator fresh smoke."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_08_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)
from experiments.run_v22_08_post_nogo_train_flow_commutator_certificate import _fresh_summary, _train_fresh, parser as base_parser  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = base_parser()
    p.description = "Repeat C-O12 fresh-smoke rows under independent train seeds."
    p.add_argument("--verify-top-k", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--train-seed-stride", type=int, default=100003)
    return p


def _f(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default)


def _load_verify_candidates(out_dir: Path, top_k: int) -> list[dict[str, Any]]:
    selected = read_rows(out_dir / "v22_08_post_nogo_train_flow_commutator_selected_candidates.csv")
    fresh = read_rows(out_dir / "v22_08_post_nogo_train_flow_commutator_fresh_c3_summary.csv")
    opened = {
        (str(r.get("v21_id")), str(r.get("dataset")), str(r.get("seed")))
        for r in fresh
        if int_flag(r.get("TFC_fresh_C3_train_flow_commutator_pass"))
    }
    candidates = [
        r
        for r in selected
        if (str(r.get("v21_id")), str(r.get("dataset")), str(r.get("seed"))) in opened
    ]
    if not candidates and selected:
        candidates = selected[:1]
    return candidates[: max(0, int(top_k))]


def _summarize_repeats(enriched: list[dict[str, Any]], fresh_pack: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    fresh_summary = fresh_pack["summary"]
    by_candidate: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in fresh_summary:
        key = (str(row.get("v21_id")), str(row.get("dataset")), str(row.get("seed")))
        by_candidate.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in by_candidate.items():
        pass_rows = sum(int_flag(r.get("TFC_fresh_C3_train_flow_commutator_pass")) for r in rows)
        debt_rows = sum(int_flag(r.get("debt_not_exploded")) for r in rows)
        item: dict[str, Any] = {
            "v21_id": key[0],
            "dataset": key[1],
            "seed": key[2],
            "repeat_rows": len(rows),
            "required_repeats": int(repeats),
            "fresh_C3_repeat_pass_rows": pass_rows,
            "debt_not_exploded_rows": debt_rows,
            "TFC_repeat_robust_fresh_C3_verified": int(pass_rows >= int(repeats) and debt_rows >= int(repeats)),
            "official_observer_pass": 0,
            "official_C2_solver_claimed": 0,
            "official_C3_pass": 0,
            "blocker": "",
        }
        for h in [100, 400, 800, 1600, 2400, 3200]:
            vals = [_f(r.get(f"source_vs_best_control_h{h}")) for r in rows]
            vals = [v for v in vals if v == v]
            item[f"source_vs_best_control_h{h}_min"] = min(vals) if vals else ""
            item[f"source_vs_best_control_h{h}_mean"] = sum(vals) / len(vals) if vals else ""
            item[f"source_vs_best_control_h{h}_positive_rows"] = sum(1 for v in vals if v >= 0.005)
        blockers: list[str] = []
        if not int_flag(item["TFC_repeat_robust_fresh_C3_verified"]):
            blockers.append("repeat_fresh_C3_horizon_or_debt_gate_failed")
        blockers.append("official_observer_gate_not_open")
        blockers.append("official_C2_solver_not_claimed")
        blockers.append("official_C3_gate_not_claimed")
        item["blocker"] = ";".join(dict.fromkeys(blockers))
        out.append(item)
    if not out and not enriched:
        out.append(
            {
                "v21_id": "",
                "dataset": "",
                "seed": "",
                "repeat_rows": 0,
                "required_repeats": int(repeats),
                "fresh_C3_repeat_pass_rows": 0,
                "debt_not_exploded_rows": 0,
                "TFC_repeat_robust_fresh_C3_verified": 0,
                "official_observer_pass": 0,
                "official_C2_solver_claimed": 0,
                "official_C3_pass": 0,
                "blocker": "no_C-O12_fresh_C3_candidate_to_verify",
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    candidates = _load_verify_candidates(out_dir, int(args.verify_top_k))
    fresh_rows: list[dict[str, Any]] = []
    for candidate_index, row in enumerate(candidates):
        base_train_seed = int(float(row.get("train_seed") or 0))
        for repeat in range(int(args.repeats)):
            repeated = deepcopy(row)
            repeated["train_seed"] = base_train_seed + repeat * int(args.train_seed_stride)
            repeated["verify_candidate_index"] = candidate_index
            repeated["verify_repeat"] = repeat
            for run_kind in [
                "TFC-AdamWPlusTrainFlowCommutatorFU",
                "CTRL-AdamW",
                "CTRL-SGD",
                "CTRL-NoOp",
                "CTRL-RandomMatchedTrainFlowCommutatorFU",
            ]:
                item = _train_fresh(args, repeated, run_kind)
                item["verify_candidate_index"] = candidate_index
                item["verify_repeat"] = repeat
                item["verify_train_seed"] = repeated["train_seed"]
                fresh_rows.append(item)
    enriched, fresh_pack = _fresh_summary(fresh_rows)
    repeat_summary = _summarize_repeats(enriched, fresh_pack, int(args.repeats))
    robust_rows = sum(int_flag(r.get("TFC_repeat_robust_fresh_C3_verified")) for r in repeat_summary)
    route = {
        "route": "TrainFlowCommutatorRepeatFreshVerifiedOfficialBlocked" if robust_rows else "TrainFlowCommutatorRepeatVerifyBlocked",
        "verify_candidate_rows": len(candidates),
        "repeat_rows": sum(int(float(r.get("repeat_rows") or 0)) for r in repeat_summary),
        "required_repeats": int(args.repeats),
        "TFC_repeat_robust_fresh_C3_verified_rows": robust_rows,
        "official_observer_pass": 0,
        "official_C2_solver_claimed_rows": 0,
        "official_C3_pass_rows": 0,
        "promotion_allowed": 0,
        "blocker": "official_observer_gate_not_open;official_C2_solver_not_claimed;official_C3_gate_not_claimed"
        if robust_rows
        else "repeat_fresh_C3_horizon_or_debt_gate_failed;official_observer_gate_not_open;official_C2_solver_not_claimed;official_C3_gate_not_claimed",
        "next_codex_action": "C-O12 repeated fresh smoke verified but official observer/C2/C3 gates remain closed; do not promote without a formal official C2/C3 construction"
        if robust_rows
        else "C-O12 fresh smoke did not survive repeat verification; continue only with a genuinely new principle",
    }
    write_rows(out_dir / "v22_08_train_flow_commutator_verify_fresh_c3_matrix.csv", enriched)
    write_rows(out_dir / "v22_08_train_flow_commutator_verify_fresh_c3_summary.csv", fresh_pack["summary"])
    write_rows(out_dir / "v22_08_train_flow_commutator_verify_repeat_summary.csv", repeat_summary)
    write_json(out_dir / "v22_08_train_flow_commutator_verify_route.json", route)
    simple_svg(out_dir / "figures/v22_08_tfc_verify_h3200.svg", "v22.08 TFC repeat verify h3200", repeat_summary, "source_vs_best_control_h3200_min")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_train_flow_commutator_fresh_verify.py --source-dir {args.source_dir} --device {args.device} --verify-top-k {int(args.verify_top_k)} --repeats {int(args.repeats)} --steps {int(args.steps)} --refresh-interval {int(args.refresh_interval)} --out-dir {out_dir}",
        status="completed",
        note=f"verify_candidates={len(candidates)} robust_fresh_verified={robust_rows} route={route['route']}",
    )


if __name__ == "__main__":
    main()
