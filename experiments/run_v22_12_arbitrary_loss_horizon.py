#!/usr/bin/env python3
"""v22.12 S5 arbitrary-loss horizon training for operator FU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.loss_interface import GenericUpstreamCotangent, stable_random_delta_like  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import HORIZONS, _control_updates, _evaluate_attempt, _loss_adapters  # noqa: E402
from experiments.run_v22_12_common import PYTHON, append_exec, ensure_out, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2212)
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _adapters(payload: dict[str, Any], seed: int) -> list[tuple[str, Any, Any]]:
    adapters = list(_loss_adapters(payload))
    logits = payload["logits"].detach().float()
    stable = stable_random_delta_like(logits, seed=seed, kind="stable")
    adapters.append(("Delta-StableRandom", GenericUpstreamCotangent(stable, "Delta-StableRandom"), None))
    return adapters


def _attempts() -> list[dict[str, Any]]:
    return [
        {"attempt": "operator_initial_commit_adamw", "lr_scale": 1.0, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0},
        {"attempt": "adapter_balanced_replay_400x0p04_stop1600", "lr_scale": 0.5, "periodic_interval": 400, "periodic_scale": 0.04, "periodic_stop_step": 1600},
        {"attempt": "source_loss_aware_replay_400x0p02_stop3200", "lr_scale": 0.2, "periodic_interval": 400, "periodic_scale": 0.02, "periodic_stop_step": 3200},
        {"attempt": "cotangent_norm_normalized_lr0p1", "lr_scale": 0.1, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0},
        {"attempt": "terminal_preservation_800x0p015_stop4800_lr0p05", "lr_scale": 0.05, "periodic_interval": 800, "periodic_scale": 0.015, "periodic_stop_step": 4800},
        {"attempt": "terminal_preservation_400x0p025_stop6400_lr0p03", "lr_scale": 0.03, "periodic_interval": 400, "periodic_scale": 0.025, "periodic_stop_step": 6400},
        {"attempt": "terminal_preservation_200x0p018_stop6400_lr0p02", "lr_scale": 0.02, "periodic_interval": 200, "periodic_scale": 0.018, "periodic_stop_step": 6400},
        {"attempt": "terminal_preservation_100x0p010_stop6400_lr0p01", "lr_scale": 0.01, "periodic_interval": 100, "periodic_scale": 0.010, "periodic_stop_step": 6400},
        {
            "attempt": "source_loss_pid20_lr0p0005_clip0p03",
            "lr_scale": 0.0005,
            "periodic_interval": 0,
            "periodic_scale": 0.0,
            "periodic_stop_step": 0,
            "pid": (20, 0.025, 0.002, 0.010, 0.030, 6400),
        },
    ]


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    commit_route = read_json(source_dir / "v22_12_operator_metric_commit_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(commit_route.get("S4_operator_metric_commit_pass_rows")) <= 0:
        route = {
            "route": "S5-BlockedBeforeArbitraryLossOperatorHorizon",
            "S4_operator_metric_commit_pass_rows": int_flag(commit_route.get("S4_operator_metric_commit_pass_rows")),
            "C3_source_formation_pass_rows": 0,
            "C4_terminal_retention_pass_rows": 0,
            "C3_adapter_pass_count": 0,
            "C4_adapter_pass_count": 0,
            "source_loss_nonnegative_rows": 0,
            "promotion_allowed": 0,
            "blocker": "S4_operator_metric_commit_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_12_operator_metric_commit_payload.pt")
        controls = _control_updates(payload, payload["update_vec"].detach().float(), int(args.seed))
        for adapter_idx, (adapter_name, adapter, task_data) in enumerate(_adapters(payload, int(args.seed))):
            for idx, attempt in enumerate(_attempts()):
                row, control_rows = _evaluate_attempt(payload, attempt, adapter_name, adapter, task_data, controls, int(args.seed) + adapter_idx * 1000 + idx * 100)
                row["horizon_status"] = "v22_12_arbitrary_loss_operator_training_horizon"
                row["reason"] = "continued training uses generic LossInterface cotangent; operator FU does not branch on adapter name"
                row["uses_adapter_name_branch"] = 0
                row["uses_loss_interface_cotangent_for_direction"] = 1
                row["operator_attempt_family"] = "loss_interface_operator"
                rows.append(row)
                for control in control_rows:
                    control["horizon_status"] = "control_v22_12"
                    rows.append(control)
        fu_rows = [r for r in rows if str(r.get("variant")) == "FU"]
        non_random = [r for r in fu_rows if str(r.get("loss_adapter_name")) != "Delta-StableRandom"]
        c3_adapters = sorted({str(r.get("loss_adapter_name")) for r in non_random if int_flag(r.get("C3_source_formation_pass"))})
        c4_adapters = sorted({str(r.get("loss_adapter_name")) for r in non_random if int_flag(r.get("C4_terminal_retention_pass"))})
        c3 = sum(int_flag(r.get("C3_source_formation_pass")) for r in fu_rows)
        c4 = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in fu_rows)
        source_loss_nonnegative = 0
        for r in fu_rows:
            try:
                source_loss_nonnegative += int(float(r.get("source_loss_h3200", -999.0)) >= -1.0e-6 and float(r.get("source_loss_h4800", -999.0)) >= -1.0e-6)
            except Exception:
                pass
        best = max(fu_rows, key=lambda r: float(r.get("source_func_h3200") or -999.0)) if fu_rows else {}
        if len(c3_adapters) >= 2 and len(c4_adapters) >= 1:
            route_name = "S5-ArbitraryLossOperatorHorizonPass"
        elif len(c3_adapters) == 1:
            route_name = "S5-AdapterSpecificSourceOpened_NotUniversal"
        elif any(int_flag(r.get("TargetRetentionOnly_NotTaskUseful")) for r in fu_rows):
            route_name = "S5-TargetRetentionOnly_NotTaskUseful"
        else:
            route_name = "S5-ArbitraryLossOperatorHorizonNoGo"
        route = {
            "route": route_name,
            "S4_operator_metric_commit_pass_rows": int_flag(commit_route.get("S4_operator_metric_commit_pass_rows")),
            "FU_attempt_rows": len(fu_rows),
            "C3_source_formation_pass_rows": c3,
            "C4_terminal_retention_pass_rows": c4,
            "C3_adapter_pass_count": len(c3_adapters),
            "C4_adapter_pass_count": len(c4_adapters),
            "C3_adapter_pass_list": ";".join(c3_adapters),
            "C4_adapter_pass_list": ";".join(c4_adapters),
            "source_loss_nonnegative_rows": source_loss_nonnegative,
            "selected_attempt": best.get("attempt", ""),
            "selected_loss_adapter": best.get("loss_adapter_name", ""),
            "best_source_func_h3200": best.get("source_func_h3200", ""),
            "best_source_loss_h3200": best.get("source_loss_h3200", ""),
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass", 1)) for r in fu_rows)),
            "uses_labels_for_direction": 0,
            "uses_loss_interface_cotangent_for_direction": 1,
            "uses_loss_formula_specific_direction": 0,
            "uses_adapter_name_branch": 0,
            "promotion_allowed": 0,
            "blocker": "" if route_name == "S5-ArbitraryLossOperatorHorizonPass" else ";".join(dict.fromkeys(part for r in fu_rows for part in str(r.get("blocker", "")).split(";") if part)),
        }
    write_rows(out_dir / "v22_12_arbitrary_loss_horizon_matrix.csv", rows)
    write_json(out_dir / "v22_12_arbitrary_loss_horizon_route.json", route)
    simple_svg(out_dir / "figures/v22_12_horizon_source_func.svg", "v22.12 arbitrary-loss source_func", rows, "source_func_h3200")
    simple_svg(out_dir / "figures/v22_12_horizon_source_loss.svg", "v22.12 arbitrary-loss source_loss", rows, "source_loss_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"route={route['route']} c3_adapters={route.get('C3_adapter_pass_count')} c4_adapters={route.get('C4_adapter_pass_count')} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
