#!/usr/bin/env python
"""v15.8 H=800 long-horizon confirmation.

This extension reuses the already executed Line T/W official artifacts to pick
the same top-2 candidates, then runs only the Line H long-horizon cases at
800 steps. It does not introduce new directions, controllers, action banks, or
audit-directed search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "v15_8_dynamics_harness_decoupled_decay_recovery_allbasis" / "official_v158"
sys.path.insert(0, str(ROOT))

from experiments import run_v158_dynamics_harness_decoupled_decay_recovery_allbasis as v158  # noqa: E402


def choose_top_methods(out_dir: Path) -> list[str]:
    t_rows = v158.read_rows(out_dir / "v158_line_t_recovery_dynamics.csv")
    w_rows = v158.read_rows(out_dir / "v158_line_w_decay_recovery.csv")
    t_summary = v158.summarize_methods(t_rows, v158.T_CANDIDATES, "line_t")
    w_summary = v158.summarize_methods(w_rows, v158.W_CANDIDATES, "line_w")
    top_methods: list[str] = []
    for summary in [t_summary, w_summary]:
        method = str(summary.get("line_t_best_method", summary.get("line_w_best_method", "")))
        if method and method not in top_methods:
            top_methods.append(method)
    return top_methods[:2] or ["T1-G7R-PulseOnce-then-AdamWRecovery", "W1-G7R-GlobalDecoupledWeightDecayRecovery"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--long-horizon-steps", type=int, default=800)
    ap.add_argument("--trace-interval", type=int, default=40)
    ap.add_argument("--split-count", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--readout-weight-decay", type=float, default=0.001)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lambda-noise", type=float, default=0.25)
    ap.add_argument("--linec-seeds", default="0")
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=64)
    ap.add_argument("--real-linec", type=int, default=1)
    ap.add_argument("--dche-candidate", default="D-CHE20-DegreeNormalizedReadoutHealthSubstrate")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = v158.resolve_cuda_device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("v15.8 H800 extension requires CUDA; refusing cpu execution")

    top_methods = choose_top_methods(out_dir)
    splits = v158.load_splits(args, device)
    rows_raw: list[dict[str, object]] = []
    horizon_raw: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    for split in splits:
        dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = split
        for base_method in top_methods:
            method = f"H-{base_method}"
            result = v158.train_dynamics_case(
                "H800",
                method,
                "D-CHE",
                dataset,
                seed,
                xtr,
                ytr,
                xva,
                yva,
                xte,
                yte,
                input_dim,
                output_dim,
                args,
                device,
                int(args.long_horizon_steps),
            )
            rows_raw.append(result["row"])
            horizon_raw.extend(result["horizon_rows"])
            direction_rows.extend(result["direction_rows"])

    h_controls = {f"H-{m}" for m in top_methods if m in v158.T_CONTROLS or m in v158.W_CONTROLS}
    h_controls.update({"H-T0-D-CHE-AdamW", "H-WCTRL-AdamWOnlyMatchedDecay"})
    rows = v158.enrich_rows(rows_raw, h_controls)
    horizons = v158.enrich_horizon_rows(horizon_raw, h_controls)
    summary = v158.summarize_methods(rows, sorted({str(r.get("method")) for r in rows}), "line_h800")
    route = {
        "stage": "V158_H800_EXTENSION",
        "top_methods": ",".join(top_methods),
        "line_h800_rows": len(rows),
        "line_h800_horizon_rows": len(horizons),
        "line_h800_best_method": summary.get("line_h800_best_method"),
        "line_h800_gate_pass": summary.get("line_h800_gate_pass"),
        "line_h800_real_lite_pass_count": summary.get("line_h800_real_lite_pass_count"),
        "line_h800_source_vs_best_control_mean": summary.get("line_h800_source_vs_best_control_mean"),
        "line_h800_bad_event_fraction": summary.get("line_h800_bad_event_fraction"),
        "line_h800_tail_debt_recovery_rate": summary.get("line_h800_tail_debt_recovery_rate"),
        "line_h800_LineC_debt_recovery_rate": summary.get("line_h800_LineC_debt_recovery_rate"),
        "promotion_allowed": 0,
        "uses_audit_metric_for_direction": 0,
        "controller_executed": 0,
        "action_bank_used_as_search_space": 0,
        "reset_route_used": 0,
        "cpu_offload_used": 0,
    }
    v158.write_rows(out_dir / "v158_line_h800_long_horizon.csv", rows)
    v158.write_rows(out_dir / "v158_line_h800_horizon_recovery.csv", horizons)
    v158.write_rows(out_dir / "v158_line_h800_method_summary.csv", summary.get("line_h800_method_rows", []))
    v158.write_rows(out_dir / "v158_line_h800_direction_provenance.csv", direction_rows)
    v158.write_json(out_dir / "v158_line_h800_route_recheck.json", route)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
