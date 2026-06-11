#!/usr/bin/env python3
"""v22.12 targeted repair diagnostics that do not mutate official gates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_11_arbitrary_loss_horizon import _control_updates, _evaluate_attempt  # noqa: E402
import experiments.run_v22_11_arbitrary_loss_horizon as horizon_v22_11  # noqa: E402
from experiments.run_v22_12_arbitrary_loss_horizon import _adapters  # noqa: E402
from experiments.run_v22_12_common import PYTHON, append_exec, ensure_out, int_flag, write_rows  # noqa: E402
from experiments.run_v22_12_operator_fu import _build_operator_rows, _commit_operator, _make_probe  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=2212)
    return p


def _attempt(name: str, lr: float, pid: tuple[int, float, float, float, float, int]) -> dict[str, Any]:
    return {
        "attempt": name,
        "lr_scale": float(lr),
        "periodic_interval": 0,
        "periodic_scale": 0.0,
        "periodic_stop_step": 0,
        "pid": pid,
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    seed = int(args.seed)
    # Diagnostic only: keeps the gate-relevant C3/C4 horizons while skipping h6400.
    horizon_v22_11.HORIZONS = [100, 400, 800, 1600, 3200, 4800]
    cases = [
        {
            "diagnostic_id": "O10_scale0p16_nonrandom_source_loss_balanced_pid20",
            "operator_id": "O10_NonRandomSourceLossBalancedOperator",
            "norm_scale": 0.16,
            "attempt": _attempt("source_loss_pid20_lr0p0005_clip0p03", 0.0005, (20, 0.025, 0.002, 0.010, 0.030, 6400)),
            "adapters": {"Delta-LossCEAdapter", "Delta-MSEAdapter", "Delta-RankingAdapter"},
            "purpose": "non-random raw consensus plus loss-adapter smooth consensus; tests whether source_loss-aware operator opens second adapter",
            "commit_seed_offset": 0,
            "control_seed_offset": 0,
            "eval_seed_offset": 0,
            "fixed_eval_seed": 1,
        },
        {
            "diagnostic_id": "O4_scale0p08_formal_rowmean_pid_boundary",
            "operator_id": "O4_LowNDSOperator",
            "norm_scale": 0.08,
            "attempt": _attempt("pid10_kp0p035_ki0p003_kd0p012_clip0p05_lr0p001", 0.001, (10, 0.035, 0.003, 0.012, 0.050, 6400)),
            "adapters": {"Delta-MSEAdapter", "Delta-RankingAdapter"},
            "purpose": "current formal target plus PID retention repair",
        },
        {
            "diagnostic_id": "O4_scale0p16_loss_positive_retention_shortfall",
            "operator_id": "O4_LowNDSOperator",
            "norm_scale": 0.16,
            "attempt": _attempt("pid5_kp0p040_ki0p003_kd0p012_clip0p060_lr0p001", 0.001, (5, 0.040, 0.003, 0.012, 0.060, 6400)),
            "adapters": {"Delta-MSEAdapter", "Delta-RankingAdapter"},
            "purpose": "scale repair where ranking source_loss stays positive but source_func remains short",
        },
        {
            "diagnostic_id": "O1_scale0p16_retention_positive_loss_shortfall",
            "operator_id": "O1_UpstreamCotangentDrift",
            "norm_scale": 0.16,
            "attempt": _attempt("pid10_kp0p035_ki0p003_kd0p012_clip0p05_lr0p001", 0.001, (10, 0.035, 0.003, 0.012, 0.050, 6400)),
            "adapters": {"Delta-RankingAdapter"},
            "purpose": "high-gain O1 boundary where ranking source_func turns positive but loss neutrality fails",
        },
    ]
    out_rows: list[dict[str, Any]] = []
    for case_idx, case in enumerate(cases):
        commit_seed = seed + int(case.get("commit_seed_offset", case_idx * 100))
        control_seed = seed + int(case.get("control_seed_offset", case_idx * 257))
        eval_seed_base = seed + int(case.get("eval_seed_offset", case_idx * 1000))
        model, x, logits, labels = _make_probe(seed)
        operator_rows, tensors, _cotangent_rows = _build_operator_rows(logits, labels, seed, float(case["norm_scale"]))
        op_id = str(case["operator_id"])
        target = tensors[op_id].detach().float()
        op_row = next((r for r in operator_rows if str(r.get("operator_id")) == op_id), {})
        commit_model, commit_x, commit_logits, commit_labels = _make_probe(seed)
        commit_rows, update, commit_route = _commit_operator(commit_model, commit_x, target, commit_seed)
        if update is None:
            out_rows.append(
                {
                    "diagnostic_id": case["diagnostic_id"],
                    "operator_id": op_id,
                    "norm_scale": case["norm_scale"],
                    "attempt": case["attempt"]["attempt"],
                    "loss_adapter_name": "",
                    "diagnostic_horizons": ";".join(str(h) for h in horizon_v22_11.HORIZONS),
                    "abbreviated_horizon": 1,
                    "S2_operator_atom_pass": op_row.get("S2_operator_atom_pass", ""),
                    "S4_operator_metric_commit_pass_rows": commit_route.get("S4_operator_metric_commit_pass_rows", 0),
                    "C3_source_formation_pass": 0,
                    "C4_terminal_retention_pass": 0,
                    "blocker": commit_route.get("blocker", "commit_failed"),
                    "purpose": case["purpose"],
                }
            )
            continue
        payload = {
            "seed": seed,
            "model_config": {"input_dim": 8, "hidden": 64, "classes": 5},
            "model_state": commit_model.state_dict(),
            "x": commit_x.detach().float(),
            "logits": commit_logits.detach().float(),
            "labels_for_loss_adapter_only": commit_labels.detach().long(),
            "target_delta": target,
            "update_vec": update.detach().float(),
        }
        controls = _control_updates(payload, update.detach().float(), control_seed)
        for adapter_idx, (adapter_name, adapter, task_data) in enumerate(_adapters(payload, seed)):
            if adapter_name not in case["adapters"]:
                continue
            row, _control_rows = _evaluate_attempt(
                payload,
                case["attempt"],
                adapter_name,
                adapter,
                task_data,
                controls,
                eval_seed_base if int(case.get("fixed_eval_seed", 0)) else eval_seed_base + adapter_idx * 100,
            )
            out_rows.append(
                {
                    "diagnostic_id": case["diagnostic_id"],
                    "operator_id": op_id,
                    "norm_scale": case["norm_scale"],
                    "attempt": case["attempt"]["attempt"],
                    "loss_adapter_name": adapter_name,
                    "diagnostic_horizons": ";".join(str(h) for h in horizon_v22_11.HORIZONS),
                    "abbreviated_horizon": 1,
                    "S2_operator_atom_pass": op_row.get("S2_operator_atom_pass", ""),
                    "operator_NDS": op_row.get("NDS", ""),
                    "operator_expected_loss_linear_gain": op_row.get("expected_loss_linear_gain", ""),
                    "S4_operator_metric_commit_pass_rows": commit_route.get("S4_operator_metric_commit_pass_rows", 0),
                    "source_func_h800": row.get("source_func_h800", ""),
                    "source_func_h1600": row.get("source_func_h1600", ""),
                    "source_func_h3200": row.get("source_func_h3200", ""),
                    "source_func_h4800": row.get("source_func_h4800", ""),
                    "source_loss_h800": row.get("source_loss_h800", ""),
                    "source_loss_h1600": row.get("source_loss_h1600", ""),
                    "source_loss_h3200": row.get("source_loss_h3200", ""),
                    "source_loss_h4800": row.get("source_loss_h4800", ""),
                    "row_positive_count_h3200": row.get("row_positive_count_h3200", ""),
                    "C3_source_formation_pass": row.get("C3_source_formation_pass", 0),
                    "C4_terminal_retention_pass": row.get("C4_terminal_retention_pass", 0),
                    "blocker": row.get("blocker", ""),
                    "purpose": case["purpose"],
                }
            )
    write_rows(out_dir / "v22_12_repair_diagnostic_summary.csv", out_rows)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_repair_diagnostics.py --out-dir {out_dir} --seed {seed}",
        status="completed",
        note="wrote v22_12_repair_diagnostic_summary.csv; diagnostic only, official route unchanged",
    )


if __name__ == "__main__":
    main()
