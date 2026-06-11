#!/usr/bin/env python3
"""GPU diagnostic sweep for v22.13 operator horizon repairs.

This runner is explicitly diagnostic: it writes preview rows and never promotes
them as official C3/C4 evidence.  The full horizon runner must be rerun for any
promising candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_13_operator_horizon as horizon  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, int_flag, read_json, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2213)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--operator-ids", default="LIO2_SplitCoherentControlNull")
    p.add_argument("--norm-scales", default="0.08,0.16,0.32,0.64,1.28,2.56")
    p.add_argument("--adapters", default="Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter")
    p.add_argument("--attempts", default="role_blind_initial_commit_lr1,slow_source_state_fixed_800x0p006")
    p.add_argument("--horizons", default="100,400,800")
    return p


def _items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in _items(text)]


def _ints(text: str) -> list[int]:
    return [int(x) for x in _items(text)]


def _preview_pass(row: dict[str, Any], horizons: list[int]) -> int:
    needed = [h for h in horizons if h <= 800]
    if not needed:
        needed = horizons
    for h in needed:
        try:
            source_func = float(row.get(f"source_func_h{h}", "-999"))
            source_loss = float(row.get(f"source_loss_h{h}", "-999"))
        except Exception:
            return 0
        if source_func < 0.005 or source_loss < -1.0e-6:
            return 0
    return 1


def _evaluate_preview(
    payload: dict[str, Any],
    attempt: dict[str, Any],
    adapter_name: str,
    adapter: Any,
    task_data: Any,
    controls: list[tuple[str, Any, str]],
    seed: int,
    device: Any,
    horizons: list[int],
) -> dict[str, Any]:
    commit_update = payload["update_vec"].detach().float().to(device)
    task_data = horizon._move_task_data(task_data, device)
    fu_snaps = horizon._train_variant_gpu(
        payload,
        update_vec=commit_update,
        optimizer_name="AdamW",
        adapter=adapter,
        task_data=task_data,
        seed=seed,
        lr_scale=float(attempt["lr_scale"]),
        periodic_update_vec=commit_update if int(attempt["periodic_interval"]) > 0 else None,
        periodic_interval=int(attempt["periodic_interval"]),
        periodic_scale=float(attempt["periodic_scale"]),
        periodic_stop_step=int(attempt.get("periodic_stop_step", 0)),
        retention_weight=float(attempt.get("retention_weight", 0.0)),
        retention_start_step=int(attempt.get("retention_start_step", 0)),
        retention_stop_step=int(attempt.get("retention_stop_step", 0)),
        device=device,
    )
    control_snaps = {}
    for idx, (control_name, update, optimizer_name) in enumerate(controls):
        periodic_vec = update.detach().float().to(device) if int(attempt["periodic_interval"]) > 0 and float(update.detach().float().norm().item()) > 0.0 else None
        control_snaps[control_name] = horizon._train_variant_gpu(
            payload,
            update_vec=update.detach().float().to(device),
            optimizer_name=optimizer_name,
            adapter=adapter,
            task_data=task_data,
            seed=seed + idx + 1,
            lr_scale=float(attempt["lr_scale"]),
            periodic_update_vec=periodic_vec,
            periodic_interval=int(attempt["periodic_interval"]),
            periodic_scale=float(attempt["periodic_scale"]),
            periodic_stop_step=int(attempt.get("periodic_stop_step", 0)),
            retention_weight=float(attempt.get("retention_weight", 0.0)),
            retention_start_step=int(attempt.get("retention_start_step", 0)),
            retention_stop_step=int(attempt.get("retention_stop_step", 0)),
            device=device,
        )
    best_func = {h: max(snaps[h]["target_retention_score"] for snaps in control_snaps.values()) for h in horizons}
    best_loss = {h: min(snaps[h]["loss_value"] for snaps in control_snaps.values()) for h in horizons}
    source_func = {h: fu_snaps[h]["target_retention_score"] - best_func[h] for h in horizons}
    source_loss = {h: best_loss[h] - fu_snaps[h]["loss_value"] for h in horizons}
    row: dict[str, Any] = {
        "horizon_status": "v22_13_gpu_repair_preview",
        "loss_adapter_name": adapter_name,
        "attempt": attempt["attempt"],
        "variant": "FU",
        "control_count": len(control_snaps),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "uses_loss_formula_specific_direction": 0,
        "periodic_interval": attempt["periodic_interval"],
        "periodic_scale": attempt["periodic_scale"],
        "periodic_stop_step": attempt.get("periodic_stop_step", 0),
        "lr_scale": attempt["lr_scale"],
        "device": str(device),
        "C3_source_formation_pass": 0,
        "C4_terminal_retention_pass": 0,
        "blocker": "diagnostic_preview_not_official_gate",
    }
    for h in horizons:
        row[f"source_func_h{h}"] = source_func[h]
        row[f"source_loss_h{h}"] = source_loss[h]
        row[f"row_positive_count_h{h}"] = sum(int(fu_snaps[h]["target_retention_score"] > snaps[h]["target_retention_score"] + 0.005) for snaps in control_snaps.values())
        row[f"FU_loss_value_h{h}"] = fu_snaps[h]["loss_value"]
        row[f"best_control_loss_value_h{h}"] = best_loss[h]
        row[f"FU_target_retention_score_h{h}"] = fu_snaps[h]["target_retention_score"]
        row[f"best_control_target_retention_score_h{h}"] = best_func[h]
    return row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = horizon._device(args.device)
    horizons = _ints(args.horizons)
    horizon.HORIZONS = horizons
    payload_path = source_dir / "v22_13_operator_commit_payload.pt"
    commit_route = read_json(source_dir / "v22_13_operator_commit_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(commit_route.get("S4_operator_metric_commit_pass_rows")) <= 0 or not payload_path.exists():
        route = {
            "route": "RepairSweepBlockedBeforeRun",
            "preview_pass_rows": 0,
            "best_candidate": "",
            "blocker": "operator_commit_payload_missing_or_S4_failed",
            "promotion_allowed": 0,
        }
    else:
        payload = horizon._load(payload_path)
        adapter_allow = set(_items(args.adapters))
        attempt_allow = set(_items(args.attempts))
        attempts = [a for a in horizon._attempts() if str(a.get("attempt")) in attempt_allow]
        adapters = [(name, adapter, data, seen) for name, adapter, data, seen in horizon._adapters(payload, int(args.seed)) if name in adapter_allow]
        for operator_id in _items(args.operator_ids):
            for norm_scale in _floats(args.norm_scales):
                for adapter_idx, (display, adapter, task_data, seen) in enumerate(adapters):
                    adapter_seed = int(args.seed) + adapter_idx * 1000 + int(norm_scale * 1000)
                    adapter_payload, adapter_commit_diag = horizon._payload_for_adapter(payload, operator_id, adapter, task_data, adapter_seed, device, norm_scale)
                    controls = horizon._control_updates_gpu(adapter_payload, adapter_payload["update_vec"], adapter_seed, device)
                    for attempt_idx, attempt in enumerate(attempts):
                        row = _evaluate_preview(adapter_payload, attempt, display, adapter, task_data, controls, adapter_seed + attempt_idx * 100, device, horizons)
                        row = horizon._decorate(row, seen, 0, operator_id)
                        row["diagnostic_only"] = 1
                        row["repair_sweep_preview_pass"] = _preview_pass(row, horizons)
                        row["norm_scale"] = norm_scale
                        row["max_preview_horizon"] = max(horizons)
                        row.update({f"adapter_commit_{k}": v for k, v in adapter_commit_diag.items() if k in {"projection_residual_Gf", "ActuationR2", "function_displacement_cos_with_target", "operator_family", "NDS_reduction", "control_projection_after"}})
                        rows.append(row)
        def score(row: dict[str, Any]) -> tuple[int, float, float]:
            try:
                sf = float(row.get(f"source_func_h{max(horizons)}", "-999"))
            except Exception:
                sf = -999.0
            try:
                sl = float(row.get(f"source_loss_h{max(horizons)}", "-999"))
            except Exception:
                sl = -999.0
            return (int_flag(row.get("repair_sweep_preview_pass")), sf, sl)

        best = max(rows, key=score) if rows else {}
        route = {
            "route": "RepairSweepPreviewPassFound" if any(int_flag(r.get("repair_sweep_preview_pass")) for r in rows) else "RepairSweepNoPreviewPass",
            "preview_rows": len(rows),
            "preview_pass_rows": sum(int_flag(r.get("repair_sweep_preview_pass")) for r in rows),
            "best_candidate": ";".join(str(best.get(k, "")) for k in ["operator_id", "norm_scale", "loss_adapter_name", "attempt"]),
            f"best_source_func_h{max(horizons)}": best.get(f"source_func_h{max(horizons)}", ""),
            f"best_source_loss_h{max(horizons)}": best.get(f"source_loss_h{max(horizons)}", ""),
            "device": str(device),
            "diagnostic_only": 1,
            "promotion_allowed": 0,
            "blocker": "" if rows else "no_rows",
        }
    write_rows(out_dir / "v22_13_operator_repair_sweep.csv", rows)
    write_json(out_dir / "v22_13_operator_repair_sweep_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_13_operator_repair_sweep.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)} --device {args.device} --operator-ids {args.operator_ids} --norm-scales {args.norm_scales} --adapters {args.adapters} --attempts {args.attempts} --horizons {args.horizons}",
        status="completed" if rows else "blocked",
        note=f"route={route['route']} preview_pass_rows={route.get('preview_pass_rows')} best={route.get('best_candidate')} device={device} diagnostic_only=1",
    )


if __name__ == "__main__":
    main()
