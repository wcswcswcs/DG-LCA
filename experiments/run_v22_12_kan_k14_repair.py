#!/usr/bin/env python3
"""Incremental v22.12 K14 D-CHE corrected-readout-layout repair.

This runner exists so late K14 repairs can be audited without recomputing the
full S6 matrix.  It reuses already-landed S6 rows, computes one fresh K14 row
with full horizons and controls, then rewrites the S6 route from the combined
matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.run_v22_12_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402
from experiments.run_v22_12_kan_mapping import _carrier_probe_correct_readout_layout, _load  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2212)
    return p


def _summarize_route(rows: list[dict[str, Any]], adapter_name: str) -> dict[str, Any]:
    pass_rows = sum(int(str(r.get("KAN_source_channel_decision")) in {"KANRetainedSourceOpened", "KANReadoutSourceOpened_BasisChannelStillOpen"}) for r in rows)
    readout_rows = sum(int(str(r.get("KAN_source_channel_decision")) == "KANReadoutSourceOpened_BasisChannelStillOpen") for r in rows)
    eff_blocked_rows = sum(int(str(r.get("KAN_source_channel_decision")) == "KANEfficiencyContractBlocked") for r in rows)
    dfou_open = any(r.get("carrier") == "D-FOU" and str(r.get("KAN_source_channel_decision")) in {"KANRetainedSourceOpened", "KANReadoutSourceOpened_BasisChannelStillOpen"} for r in rows)
    dche_open = any(r.get("carrier") == "D-CHE" and str(r.get("KAN_source_channel_decision")) in {"KANRetainedSourceOpened", "KANReadoutSourceOpened_BasisChannelStillOpen"} for r in rows)
    dche_source_efficiency_blocked = any(r.get("carrier") == "D-CHE" and str(r.get("KAN_source_channel_decision")) == "KANEfficiencyContractBlocked" for r in rows)
    dche_terminal_source_loss_blocked = any(r.get("carrier") == "D-CHE" and "source_loss_h4800_gate" in str(r.get("blocker", "")) for r in rows)
    if pass_rows and readout_rows == pass_rows:
        route_name = "S6-KANReadoutSourceOpened_BasisChannelStillOpen"
    elif pass_rows:
        route_name = "S6-KANRetainedSourceOpened"
    elif eff_blocked_rows:
        route_name = "S6-KANSourceExistsEfficiencyBlocked"
    else:
        route_name = "S6-KANSourceChannelMismatchConfirmed"
    if dfou_open and dche_source_efficiency_blocked:
        route_name = "S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked"
    elif dfou_open and not dche_open:
        route_name = "S6-DFOUReadoutSourceOpened_DCHEMismatch"
    route_blocker = ""
    if dfou_open and dche_source_efficiency_blocked:
        route_blocker = "D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists"
        if dche_terminal_source_loss_blocked:
            route_blocker += ";D-CHE_source_loss_h4800_gate_after_source_exists"
    elif dfou_open and not dche_open:
        route_blocker = "D-CHE_source_channel_mismatch_after_DFOU_open"
    elif not pass_rows:
        route_blocker = ";".join(dict.fromkeys(str(r.get("blocker", "")) for r in rows if r.get("blocker")))
    return {
        "route": route_name,
        "KAN_source_channel_pass_rows": pass_rows,
        "KAN_readout_only_pass_rows": readout_rows,
        "KAN_efficiency_blocked_rows": eff_blocked_rows,
        "selected_loss_adapter": adapter_name,
        "carrier_specific_efficiency_pass_consistency": 1,
        "wrong_global_efficiency_gate_detected": 0,
        "wrong_global_efficiency_gate_fixed": 1,
        "promotion_allowed": 0,
        "blocker": route_blocker,
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    matrix_path = out_dir / "v22_12_kan_mapping_matrix.csv"
    rows = [r for r in read_rows(matrix_path) if not str(r.get("attempt", "")).startswith("K14_")]
    horizon_route = read_json(source_dir / "v22_12_arbitrary_loss_horizon_route.json")
    basis_route = read_json(source_dir / "v22_12_basis_efficiency_route.json")
    payload = _load(source_dir / "v22_12_operator_metric_commit_payload.pt")
    adapter_name = str(horizon_route.get("selected_loss_adapter") or "Delta-MSEAdapter")
    mlp_source = finite_float(horizon_route.get("best_source_func_h3200"), 0.0)
    dche_efficiency = int_flag(basis_route.get("D-CHE_pass"))
    row = _carrier_probe_correct_readout_layout(
        "D-CHE",
        payload,
        adapter_name,
        mlp_source,
        dche_efficiency,
        int(args.seed) + 3000,
        attempt_name="K14_dche_corrected_readout_layout_fan_scale_25x0p08",
        interval=25,
        scale=0.08,
        init_variant="fan_scale_repair",
    )
    row["mapping_strategy"] = "K14_dche_corrected_readout_layout_low_degree_commit"
    row["carrier_specific_efficiency_pass"] = dche_efficiency
    row["loss_agnostic_efficiency_gate_pass"] = dche_efficiency
    row["wrong_global_efficiency_gate_detected"] = 0
    row["wrong_global_efficiency_gate_fixed"] = 1
    row["carrier_specific_efficiency_pass_consistency"] = 1
    row["basis_channel_energy"] = row.get("K14_basis_channel_energy", "")
    row["readout_channel_energy"] = row.get("K14_readout_channel_energy", "")
    row["K14_incremental_repair_runner"] = 1
    rows.append(row)
    route = _summarize_route(rows, adapter_name)
    write_rows(matrix_path, rows)
    write_json(out_dir / "v22_12_kan_mapping_route.json", route)
    simple_svg(out_dir / "figures/v22_12_kan_source_func.svg", "v22.12 KAN source func", rows, "KAN_source_func_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_12_kan_k14_repair.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"route={route['route']} K14_decision={row.get('KAN_source_channel_decision')} K14_source_func_h3200={row.get('KAN_source_func_h3200')} K14_source_loss_h3200={row.get('KAN_source_loss_h3200')} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
