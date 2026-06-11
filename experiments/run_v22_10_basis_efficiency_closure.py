#!/usr/bin/env python3
"""v22.10 S1 basis efficiency reconfirm and D-RAT/D-RBF officialization."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_09_drat_drbf_shape_localk_repair_scan as shape_scan  # noqa: E402
from experiments.run_v22_10_common import PYTHON, append_exec, ensure_out, int_flag, read_json, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--official-transition-batch-sizes", default="128,256,512,1024")
    p.add_argument("--official-repair-hidden-grid", default="8,12,16,20,24,32,40,48,64,80,96,128")
    p.add_argument("--official-blockh-repair-hidden-grid", default="8,12,16,20,24,32,40,48,64,80,96,128")
    p.add_argument("--official-blockh-grid", default="16,64,128")
    p.add_argument("--official-fastk2-repair-hidden-grid", default="8,12,16,20,24,32,40,48,64,80,96,128")
    p.add_argument("--official-blockb-repair-hidden-grid", default="8,12,16,20,24,32,40,48,64,80,96,128")
    p.add_argument("--official-blockb-grid", default="16,128")
    p.add_argument("--official-combo-repair-hidden-grid", default="32,40")
    p.add_argument("--official-combo-blockh-grid", default="64")
    p.add_argument("--official-combo-blockb-grid", default="128")
    p.add_argument("--official-combo-warps-grid", default="1")
    return p


def _copy_rows(out_dir: Path, src_name: str, dst_name: str, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = [dict(r) for r in read_rows(out_dir / src_name)]
    for row in rows:
        row.update(extra or {})
    write_rows(out_dir / dst_name, rows)
    return rows


def _official_candidate_scan(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = shape_scan.v2207.v2205_repair.repair._device(args.device)
    hidden_grid = [int(x) for x in str(args.official_repair_hidden_grid).split(",") if x.strip()]
    local = argparse.Namespace(
        out_dir=str(out_dir),
        device=args.device,
        official_transition_batch_sizes=args.official_transition_batch_sizes,
        hidden_grid=args.official_repair_hidden_grid,
        data_root="data",
        input_size=8,
        classes=10,
        param_budget=12000,
        train_size=2048,
        val_size=128,
        profiler_repeats=2,
        profiler_warmup=1,
        lr=0.003,
    )
    rows: list[dict[str, Any]] = []
    for hidden in hidden_grid:
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RAT",
                hidden=hidden,
                repair_variant="RAT22.05-official-rational-k4-triton",
                component_variant="RAT22.10-official-candidate-rational-k4-triton-trainpath",
                manual_token="rational_k4_triton_l3_matmul",
            )
        )
        for num_warps in (1, 2):
            rows.extend(
                shape_scan._measure_group(
                    local,
                    device,
                    carrier="D-RAT",
                    hidden=hidden,
                    repair_variant=f"RAT22.10-official-rational-k4-triton-singlelaunch-warps{num_warps}",
                    component_variant=f"RAT22.10-official-candidate-rational-k4-singlelaunch-warps{num_warps}-trainpath",
                    manual_token="rational_k4_triton_l3_matmul",
                )
            )
        for block_h in (64, 128):
            rows.extend(
                shape_scan._measure_group(
                    local,
                    device,
                    carrier="D-RAT",
                    hidden=hidden,
                    repair_variant=f"RAT22.10-official-rational-k4-triton-singlelaunch-blockh{block_h}",
                    component_variant=f"RAT22.10-official-candidate-rational-k4-singlelaunch-blockh{block_h}-trainpath",
                    manual_token="rational_k4_triton_l3_matmul",
                )
            )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RAT",
                hidden=hidden,
                repair_variant="RAT22.10-official-rational-k4-triton-singlelaunch",
                component_variant="RAT22.10-official-candidate-rational-k4-singlelaunch-trainpath",
                manual_token="rational_k4_triton_l3_matmul",
            )
        )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RAT",
                hidden=hidden,
                repair_variant="RAT22.10-official-rational-k2-triton",
                component_variant="RAT22.10-official-candidate-rational-k2-triton-trainpath",
                manual_token="rational_k2_triton_l3_matmul",
            )
        )
        for num_warps in (1, 2):
            rows.extend(
                shape_scan._measure_group(
                    local,
                    device,
                    carrier="D-RAT",
                    hidden=hidden,
                    repair_variant=f"RAT22.10-official-rational-k2-triton-warps{num_warps}",
                    component_variant=f"RAT22.10-official-candidate-rational-k2-warps{num_warps}-trainpath",
                    manual_token="rational_k2_triton_l3_matmul",
                )
            )
        for block_h in (64, 128):
            rows.extend(
                shape_scan._measure_group(
                    local,
                    device,
                    carrier="D-RAT",
                    hidden=hidden,
                    repair_variant=f"RAT22.10-official-rational-k2-triton-singlelaunch-blockh{block_h}",
                    component_variant=f"RAT22.10-official-candidate-rational-k2-singlelaunch-blockh{block_h}-trainpath",
                    manual_token="rational_k2_triton_l3_matmul",
                )
            )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RAT",
                hidden=hidden,
                repair_variant="RAT22.10-official-rational-k2-triton-singlelaunch",
                component_variant="RAT22.10-official-candidate-rational-k2-singlelaunch-trainpath",
                manual_token="rational_k2_triton_l3_matmul",
            )
        )
        for num_warps in (1, 2):
            rows.extend(
                shape_scan._measure_group(
                    local,
                    device,
                    carrier="D-RAT",
                    hidden=hidden,
                    repair_variant=f"RAT22.10-official-rational-k2-triton-singlelaunch-warps{num_warps}",
                    component_variant=f"RAT22.10-official-candidate-rational-k2-singlelaunch-warps{num_warps}-trainpath",
                    manual_token="rational_k2_triton_l3_matmul",
                )
            )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RBF",
                hidden=hidden,
                repair_variant="RBF22.03-R1-compact-local-k4-no-dense",
                component_variant="RBF22.10-official-candidate-rbf-k4-triton-trainpath",
                manual_token="rbf_k4_triton_l3_matmul",
            )
        )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RBF",
                hidden=hidden,
                repair_variant="RBF22.10-official-rbf-k4-triton-singlelaunch",
                component_variant="RBF22.10-official-candidate-rbf-k4-singlelaunch-trainpath",
                manual_token="rbf_k4_triton_l3_matmul",
            )
        )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RBF",
                hidden=hidden,
                repair_variant="RBF22.03-R1-compact-local-low-k2-no-dense",
                component_variant="RBF22.10-official-candidate-rbf-k2-triton-trainpath",
                manual_token="rbf_k2_triton_l3_matmul",
            )
        )
        rows.extend(
            shape_scan._measure_group(
                local,
                device,
                carrier="D-RBF",
                hidden=hidden,
                repair_variant="RBF22.10-official-rbf-k2-triton-low-k2-singlelaunch",
                component_variant="RBF22.10-official-candidate-rbf-k2-singlelaunch-trainpath",
                manual_token="rbf_k2_triton_l3_matmul",
            )
        )
    for row in rows:
        row["v22_10_official_candidate_scan"] = 1
        row["promotion_allowed"] = 0
    summary = shape_scan._shape_summary(rows)
    for row in summary:
        row["v22_10_official_candidate_scan"] = 1
        row["official_closure_claimed"] = int(int_flag(row.get("shape_localk_candidate")))
        if int_flag(row.get("shape_localk_candidate")):
            row["candidate_decision"] = "RobustOfficialCandidate"
        else:
            row["candidate_decision"] = "CandidateBlocked"
    candidate_groups = sum(int_flag(r.get("shape_localk_candidate")) for r in summary)
    candidate_carriers = sorted({str(r.get("carrier", "")) for r in summary if int_flag(r.get("shape_localk_candidate"))})
    route = {
        "route": "S1-D-RAT-D-RBFOfficialCandidateOpened" if candidate_groups else "S1-D-RAT-D-RBFOfficialCandidateBlocked",
        "scan_rows": len(rows),
        "summary_rows": len(summary),
        "hidden_grid": args.official_repair_hidden_grid,
        "candidate_groups": candidate_groups,
        "candidate_carriers": ",".join(candidate_carriers),
        "D-RAT_candidate_robust_pass": int("D-RAT" in candidate_carriers),
        "D-RBF_candidate_robust_pass": int("D-RBF" in candidate_carriers),
        "official_closure_claimed": int(candidate_groups > 0),
        "promotion_allowed": 0,
        "blocker": "" if candidate_groups else "forward_or_step_ratio_failed_across_candidate_grid",
    }
    write_rows(out_dir / "v22_10_drat_drbf_official_candidate_repair_scan.csv", rows)
    write_rows(out_dir / "v22_10_drat_drbf_official_candidate_repair_scan_summary.csv", summary)
    write_json(out_dir / "v22_10_drat_drbf_official_candidate_repair_scan_route.json", route)
    return rows, summary, route


def _official_blockh_repair_scan(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = shape_scan.v2207.v2205_repair.repair._device(args.device)
    hidden_grid = [int(x) for x in str(args.official_blockh_repair_hidden_grid).split(",") if x.strip()]
    blockh_grid = [int(x) for x in str(args.official_blockh_grid).split(",") if x.strip()]
    local = argparse.Namespace(
        out_dir=str(out_dir),
        device=args.device,
        official_transition_batch_sizes=args.official_transition_batch_sizes,
        hidden_grid=args.official_blockh_repair_hidden_grid,
        data_root="data",
        input_size=8,
        classes=10,
        param_budget=12000,
        train_size=2048,
        val_size=128,
        profiler_repeats=2,
        profiler_warmup=1,
        lr=0.003,
    )
    rows: list[dict[str, Any]] = []
    variant_specs = [
        ("D-RAT", "RAT22.10-official-rational-k2-triton", "rational_k2_triton_l3_matmul"),
        ("D-RAT", "RAT22.05-official-rational-k4-triton", "rational_k4_triton_l3_matmul"),
        ("D-RBF", "RBF22.10-official-rbf-k2-triton-low-k2", "rbf_k2_triton_l3_matmul"),
        ("D-RBF", "RBF22.10-official-rbf-k4-triton", "rbf_k4_triton_l3_matmul"),
    ]
    for hidden in hidden_grid:
        for block_h in blockh_grid:
            for carrier, base_variant, manual_token in variant_specs:
                repair_variant = f"{base_variant}-blockh{block_h}"
                component_variant = f"{repair_variant}-trainpath"
                measured = shape_scan._measure_group(
                    local,
                    device,
                    carrier=carrier,
                    hidden=hidden,
                    repair_variant=repair_variant,
                    component_variant=component_variant,
                    manual_token=manual_token,
                )
                for row in measured:
                    row["blockh_repair_scan"] = 1
                    row["block_h"] = int(block_h)
                    row["v22_10_official_candidate_scan"] = 0
                    row["v22_10_official_blockh_repair_scan"] = 1
                    row["promotion_allowed"] = 0
                rows.extend(measured)
    summary = shape_scan._shape_summary(rows)
    for row in summary:
        variant = str(row.get("repair_variant", ""))
        row["block_h"] = ""
        for block_h in blockh_grid:
            if f"blockh{block_h}" in variant:
                row["block_h"] = int(block_h)
                break
        row["v22_10_official_blockh_repair_scan"] = 1
        row["official_closure_claimed"] = int(int_flag(row.get("shape_localk_candidate")))
        row["candidate_decision"] = "RobustOfficialBlockHCandidate" if int_flag(row.get("shape_localk_candidate")) else "BlockHCandidateBlocked"
    candidate_groups = sum(int_flag(r.get("shape_localk_candidate")) for r in summary)
    candidate_carriers = sorted({str(r.get("carrier", "")) for r in summary if int_flag(r.get("shape_localk_candidate"))})
    route = {
        "route": "S1-D-RAT-D-RBFOfficialBlockHRepairOpened" if candidate_groups else "S1-D-RAT-D-RBFOfficialBlockHRepairBlocked",
        "scan_rows": len(rows),
        "summary_rows": len(summary),
        "hidden_grid": args.official_blockh_repair_hidden_grid,
        "blockh_grid": args.official_blockh_grid,
        "candidate_groups": candidate_groups,
        "candidate_carriers": ",".join(candidate_carriers),
        "D-RAT_candidate_robust_pass": int("D-RAT" in candidate_carriers),
        "D-RBF_candidate_robust_pass": int("D-RBF" in candidate_carriers),
        "official_closure_claimed": int(candidate_groups > 0),
        "promotion_allowed": 0,
        "blocker": "" if candidate_groups else "forward_or_step_ratio_failed_across_blockh_grid",
    }
    write_rows(out_dir / "v22_10_drat_drbf_blockh_repair_scan.csv", rows)
    write_rows(out_dir / "v22_10_drat_drbf_blockh_repair_scan_summary.csv", summary)
    write_json(out_dir / "v22_10_drat_drbf_blockh_repair_scan_route.json", route)
    return rows, summary, route


def _official_fastk2_repair_scan(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = shape_scan.v2207.v2205_repair.repair._device(args.device)
    hidden_grid = [int(x) for x in str(args.official_fastk2_repair_hidden_grid).split(",") if x.strip()]
    local = argparse.Namespace(
        out_dir=str(out_dir),
        device=args.device,
        official_transition_batch_sizes=args.official_transition_batch_sizes,
        hidden_grid=args.official_fastk2_repair_hidden_grid,
        data_root="data",
        input_size=8,
        classes=10,
        param_budget=12000,
        train_size=2048,
        val_size=128,
        profiler_repeats=2,
        profiler_warmup=1,
        lr=0.003,
    )
    rows: list[dict[str, Any]] = []
    variant_specs = [
        ("D-RAT", "RAT22.10-official-rational-k2-triton-fastk2", "rational_k2_triton_l3_matmul"),
        ("D-RBF", "RBF22.10-official-rbf-k2-triton-low-k2-fastk2", "rbf_k2_triton_l3_matmul"),
        ("D-RBF", "RBF22.10-official-rbf-k2-triton-low-k2-singlelaunch-fastk2", "rbf_k2_triton_l3_matmul"),
    ]
    for hidden in hidden_grid:
        for carrier, repair_variant, manual_token in variant_specs:
            measured = shape_scan._measure_group(
                local,
                device,
                carrier=carrier,
                hidden=hidden,
                repair_variant=repair_variant,
                component_variant=f"{repair_variant}-trainpath",
                manual_token=manual_token,
            )
            for row in measured:
                row["fastk2_repair_scan"] = 1
                row["v22_10_official_candidate_scan"] = 0
                row["v22_10_official_fastk2_repair_scan"] = 1
                row["promotion_allowed"] = 0
            rows.extend(measured)
    summary = shape_scan._shape_summary(rows)
    for row in summary:
        row["v22_10_official_fastk2_repair_scan"] = 1
        row["official_closure_claimed"] = int(int_flag(row.get("shape_localk_candidate")))
        row["candidate_decision"] = "RobustOfficialFastK2Candidate" if int_flag(row.get("shape_localk_candidate")) else "FastK2CandidateBlocked"
    candidate_groups = sum(int_flag(r.get("shape_localk_candidate")) for r in summary)
    candidate_carriers = sorted({str(r.get("carrier", "")) for r in summary if int_flag(r.get("shape_localk_candidate"))})
    route = {
        "route": "S1-D-RAT-D-RBFOfficialFastK2RepairOpened" if candidate_groups else "S1-D-RAT-D-RBFOfficialFastK2RepairBlocked",
        "scan_rows": len(rows),
        "summary_rows": len(summary),
        "hidden_grid": args.official_fastk2_repair_hidden_grid,
        "candidate_groups": candidate_groups,
        "candidate_carriers": ",".join(candidate_carriers),
        "D-RAT_candidate_robust_pass": int("D-RAT" in candidate_carriers),
        "D-RBF_candidate_robust_pass": int("D-RBF" in candidate_carriers),
        "official_closure_claimed": int(candidate_groups > 0),
        "promotion_allowed": 0,
        "blocker": "" if candidate_groups else "forward_or_step_ratio_failed_across_fastk2_grid",
    }
    write_rows(out_dir / "v22_10_drat_drbf_fastk2_repair_scan.csv", rows)
    write_rows(out_dir / "v22_10_drat_drbf_fastk2_repair_scan_summary.csv", summary)
    write_json(out_dir / "v22_10_drat_drbf_fastk2_repair_scan_route.json", route)
    return rows, summary, route


def _official_blockb_repair_scan(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = shape_scan.v2207.v2205_repair.repair._device(args.device)
    hidden_grid = [int(x) for x in str(args.official_blockb_repair_hidden_grid).split(",") if x.strip()]
    blockb_grid = [int(x) for x in str(args.official_blockb_grid).split(",") if x.strip()]
    local = argparse.Namespace(
        out_dir=str(out_dir),
        device=args.device,
        official_transition_batch_sizes=args.official_transition_batch_sizes,
        hidden_grid=args.official_blockb_repair_hidden_grid,
        data_root="data",
        input_size=8,
        classes=10,
        param_budget=12000,
        train_size=2048,
        val_size=128,
        profiler_repeats=2,
        profiler_warmup=1,
        lr=0.003,
    )
    rows: list[dict[str, Any]] = []
    variant_specs = [
        ("D-RAT", "RAT22.10-official-rational-k2-triton", "rational_k2_triton_l3_matmul"),
        ("D-RAT", "RAT22.05-official-rational-k4-triton", "rational_k4_triton_l3_matmul"),
        ("D-RBF", "RBF22.10-official-rbf-k2-triton-low-k2", "rbf_k2_triton_l3_matmul"),
        ("D-RBF", "RBF22.10-official-rbf-k4-triton", "rbf_k4_triton_l3_matmul"),
    ]
    for hidden in hidden_grid:
        for block_b in blockb_grid:
            for carrier, base_variant, manual_token in variant_specs:
                repair_variant = f"{base_variant}-blockb{block_b}"
                measured = shape_scan._measure_group(
                    local,
                    device,
                    carrier=carrier,
                    hidden=hidden,
                    repair_variant=repair_variant,
                    component_variant=f"{repair_variant}-trainpath",
                    manual_token=manual_token,
                )
                for row in measured:
                    row["blockb_repair_scan"] = 1
                    row["block_b"] = int(block_b)
                    row["v22_10_official_candidate_scan"] = 0
                    row["v22_10_official_blockb_repair_scan"] = 1
                    row["promotion_allowed"] = 0
                rows.extend(measured)
    summary = shape_scan._shape_summary(rows)
    for row in summary:
        variant = str(row.get("repair_variant", ""))
        row["block_b"] = ""
        for block_b in blockb_grid:
            if f"blockb{block_b}" in variant:
                row["block_b"] = int(block_b)
                break
        row["v22_10_official_blockb_repair_scan"] = 1
        row["official_closure_claimed"] = int(int_flag(row.get("shape_localk_candidate")))
        row["candidate_decision"] = "RobustOfficialBlockBCandidate" if int_flag(row.get("shape_localk_candidate")) else "BlockBCandidateBlocked"
    candidate_groups = sum(int_flag(r.get("shape_localk_candidate")) for r in summary)
    candidate_carriers = sorted({str(r.get("carrier", "")) for r in summary if int_flag(r.get("shape_localk_candidate"))})
    route = {
        "route": "S1-D-RAT-D-RBFOfficialBlockBRepairOpened" if candidate_groups else "S1-D-RAT-D-RBFOfficialBlockBRepairBlocked",
        "scan_rows": len(rows),
        "summary_rows": len(summary),
        "hidden_grid": args.official_blockb_repair_hidden_grid,
        "blockb_grid": args.official_blockb_grid,
        "candidate_groups": candidate_groups,
        "candidate_carriers": ",".join(candidate_carriers),
        "D-RAT_candidate_robust_pass": int("D-RAT" in candidate_carriers),
        "D-RBF_candidate_robust_pass": int("D-RBF" in candidate_carriers),
        "official_closure_claimed": int(candidate_groups > 0),
        "promotion_allowed": 0,
        "blocker": "" if candidate_groups else "forward_or_step_ratio_failed_across_blockb_grid",
    }
    write_rows(out_dir / "v22_10_drat_drbf_blockb_repair_scan.csv", rows)
    write_rows(out_dir / "v22_10_drat_drbf_blockb_repair_scan_summary.csv", summary)
    write_json(out_dir / "v22_10_drat_drbf_blockb_repair_scan_route.json", route)
    return rows, summary, route


def _official_combo_repair_scan(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = shape_scan.v2207.v2205_repair.repair._device(args.device)
    hidden_grid = [int(x) for x in str(args.official_combo_repair_hidden_grid).split(",") if x.strip()]
    blockh_grid = [int(x) for x in str(args.official_combo_blockh_grid).split(",") if x.strip()]
    blockb_grid = [int(x) for x in str(args.official_combo_blockb_grid).split(",") if x.strip()]
    warps_grid = [int(x) for x in str(args.official_combo_warps_grid).split(",") if x.strip()]
    local = argparse.Namespace(
        out_dir=str(out_dir),
        device=args.device,
        official_transition_batch_sizes=args.official_transition_batch_sizes,
        hidden_grid=args.official_combo_repair_hidden_grid,
        data_root="data",
        input_size=8,
        classes=10,
        param_budget=12000,
        train_size=2048,
        val_size=128,
        profiler_repeats=2,
        profiler_warmup=1,
        lr=0.003,
    )
    rows: list[dict[str, Any]] = []
    for hidden in hidden_grid:
        for block_h in blockh_grid:
            for block_b in blockb_grid:
                for num_warps in warps_grid:
                    specs = [
                        (
                            "D-RAT",
                            f"RAT22.10-official-rational-k4-triton-singlelaunch-warps{num_warps}-blockh{block_h}-blockb{block_b}",
                            "rational_k4_triton_l3_matmul",
                        ),
                        (
                            "D-RAT",
                            f"RAT22.10-official-rational-k2-triton-singlelaunch-warps{num_warps}-blockh{block_h}-blockb{block_b}",
                            "rational_k2_triton_l3_matmul",
                        ),
                        (
                            "D-RAT",
                            f"RAT22.10-official-rational-k2-triton-singlelaunch-fastk2-warps{num_warps}-blockh{block_h}-blockb{block_b}",
                            "rational_k2_triton_l3_matmul",
                        ),
                    ]
                    for carrier, repair_variant, manual_token in specs:
                        measured = shape_scan._measure_group(
                            local,
                            device,
                            carrier=carrier,
                            hidden=hidden,
                            repair_variant=repair_variant,
                            component_variant=f"{repair_variant}-trainpath",
                            manual_token=manual_token,
                        )
                        for row in measured:
                            row["combo_repair_scan"] = 1
                            row["block_h"] = int(block_h)
                            row["block_b"] = int(block_b)
                            row["num_warps"] = int(num_warps)
                            row["v22_10_official_candidate_scan"] = 0
                            row["v22_10_official_combo_repair_scan"] = 1
                            row["promotion_allowed"] = 0
                        rows.extend(measured)
                for carrier, base_variant, manual_token in [
                    ("D-RBF", "RBF22.10-official-rbf-k2-triton-low-k2-singlelaunch", "rbf_k2_triton_l3_matmul"),
                    ("D-RBF", "RBF22.10-official-rbf-k2-triton-low-k2-singlelaunch-fastk2", "rbf_k2_triton_l3_matmul"),
                ]:
                    repair_variant = f"{base_variant}-blockh{block_h}-blockb{block_b}"
                    measured = shape_scan._measure_group(
                        local,
                        device,
                        carrier=carrier,
                        hidden=hidden,
                        repair_variant=repair_variant,
                        component_variant=f"{repair_variant}-trainpath",
                        manual_token=manual_token,
                    )
                    for row in measured:
                        row["combo_repair_scan"] = 1
                        row["block_h"] = int(block_h)
                        row["block_b"] = int(block_b)
                        row["num_warps"] = ""
                        row["v22_10_official_candidate_scan"] = 0
                        row["v22_10_official_combo_repair_scan"] = 1
                        row["promotion_allowed"] = 0
                    rows.extend(measured)
    summary = shape_scan._shape_summary(rows)
    for row in summary:
        variant = str(row.get("repair_variant", ""))
        row["block_h"] = ""
        row["block_b"] = ""
        row["num_warps"] = ""
        for block_h in blockh_grid:
            if f"blockh{block_h}" in variant:
                row["block_h"] = int(block_h)
                break
        for block_b in blockb_grid:
            if f"blockb{block_b}" in variant:
                row["block_b"] = int(block_b)
                break
        for num_warps in warps_grid:
            if f"warps{num_warps}" in variant:
                row["num_warps"] = int(num_warps)
                break
        row["v22_10_official_combo_repair_scan"] = 1
        row["official_closure_claimed"] = int(int_flag(row.get("shape_localk_candidate")))
        row["candidate_decision"] = "RobustOfficialComboCandidate" if int_flag(row.get("shape_localk_candidate")) else "ComboCandidateBlocked"
    candidate_groups = sum(int_flag(r.get("shape_localk_candidate")) for r in summary)
    candidate_carriers = sorted({str(r.get("carrier", "")) for r in summary if int_flag(r.get("shape_localk_candidate"))})
    route = {
        "route": "S1-D-RAT-D-RBFOfficialComboRepairOpened" if candidate_groups else "S1-D-RAT-D-RBFOfficialComboRepairBlocked",
        "scan_rows": len(rows),
        "summary_rows": len(summary),
        "hidden_grid": args.official_combo_repair_hidden_grid,
        "blockh_grid": args.official_combo_blockh_grid,
        "blockb_grid": args.official_combo_blockb_grid,
        "warps_grid": args.official_combo_warps_grid,
        "candidate_groups": candidate_groups,
        "candidate_carriers": ",".join(candidate_carriers),
        "D-RAT_candidate_robust_pass": int("D-RAT" in candidate_carriers),
        "D-RBF_candidate_robust_pass": int("D-RBF" in candidate_carriers),
        "official_closure_claimed": int(candidate_groups > 0),
        "promotion_allowed": 0,
        "blocker": "" if candidate_groups else "forward_or_step_ratio_failed_across_combo_grid",
    }
    write_rows(out_dir / "v22_10_drat_drbf_combo_repair_scan.csv", rows)
    write_rows(out_dir / "v22_10_drat_drbf_combo_repair_scan_summary.csv", summary)
    write_json(out_dir / "v22_10_drat_drbf_combo_repair_scan_route.json", route)
    return rows, summary, route


def _write_loss_agnostic_efficiency_block(out_dir: Path, command: list[str]) -> None:
    """Fail closed when the only available S1 trainpath is CE-specific."""

    blocker = "optimization_loss_agnostic_efficiency_gate;arbitrary_loss_upstream_gradient_efficiency_runner_missing"
    contract = "CE_targeted_trainpath_not_allowed_for_v22_10_formal_optimization"
    route = {
        "route": "S1-LossAgnosticBasisEfficiencyNotEstablished",
        "D-CHE_S1_pass": 0,
        "D-FOU_S1_pass": 0,
        "D-RAT_robust_pass": 0,
        "D-RBF_robust_pass": 0,
        "D-RAT_D-RBF_multibatch_closed": 0,
        "limited_smoke_allowed_rows": 0,
        "repair_attempted": 0,
        "repair_attempt_rows": 0,
        "official_candidate_repair_scan_attempted": 0,
        "official_candidate_repair_scan_rows": 0,
        "official_candidate_repair_candidate_groups": 0,
        "official_candidate_repair_candidate_carriers": "",
        "official_candidate_repair_scan_route": "NotRunLossAgnosticContract",
        "official_blockh_repair_scan_attempted": 0,
        "official_blockh_repair_scan_rows": 0,
        "official_blockh_repair_candidate_groups": 0,
        "official_blockh_repair_candidate_carriers": "",
        "official_blockh_repair_scan_route": "NotRunLossAgnosticContract",
        "official_fastk2_repair_scan_attempted": 0,
        "official_fastk2_repair_scan_rows": 0,
        "official_fastk2_repair_candidate_groups": 0,
        "official_fastk2_repair_candidate_carriers": "",
        "official_fastk2_repair_scan_route": "NotRunLossAgnosticContract",
        "official_blockb_repair_scan_attempted": 0,
        "official_blockb_repair_scan_rows": 0,
        "official_blockb_repair_candidate_groups": 0,
        "official_blockb_repair_candidate_carriers": "",
        "official_blockb_repair_scan_route": "NotRunLossAgnosticContract",
        "official_combo_repair_scan_attempted": 0,
        "official_combo_repair_scan_rows": 0,
        "official_combo_repair_candidate_groups": 0,
        "official_combo_repair_candidate_carriers": "",
        "official_combo_repair_scan_route": "NotRunLossAgnosticContract",
        "optimization_loss_agnostic_contract_pass": 0,
        "ce_targeted_efficiency_rows_invalidated": 1,
        "manual_ce_train_stream_profiled_rows": 0,
        "loss_agnostic_efficiency_contract": contract,
        "blocker": blocker,
    }
    write_json(out_dir / "v22_10_basis_efficiency_route.json", route)
    eff_rows = [
        {
            "carrier": carrier,
            "variant_family": variant,
            "forward_ratio_vs_mlp": "",
            "step_ratio_vs_mlp": "",
            "memory_ratio_vs_mlp": "",
            "functional_runner_same_kernel": "",
            "fallback_kernel_used": "",
            "official_fused_kernel_complete": "",
            "v22_09_S1_pass": 0,
            "v22_09_blocker": blocker,
            "formal_status": "blocked_before_CE_targeted_training_loop",
            "v22_10_loss_contract": contract,
        }
        for carrier, variant in (
            ("D-CHE", "CHE21-R2/R4"),
            ("D-FOU", "FOU21-R3"),
            ("D-RAT", "RAT22.10"),
            ("D-RBF", "RBF22.10"),
        )
    ]
    write_rows(out_dir / "v22_10_efficiency_full_loop_reconfirm.csv", eff_rows)
    dr_rows = [
        {
            "carrier": carrier,
            "profile_rows": 0,
            "robust_production_pass_rows": 0,
            "near_E1_rows": 0,
            "component_telemetry_complete_rows": 0,
            "best_forward_ratio": "",
            "best_step_ratio": "",
            "best_memory_ratio": "",
            "decision": "S1LossAgnosticEfficiencyNotEstablished",
            "blocker": blocker,
            "formal_status": "CE_targeted_multibatch_profile_not_executed",
        }
        for carrier in ("D-RAT", "D-RBF")
    ]
    write_rows(out_dir / "v22_10_drat_drbf_multibatch_summary.csv", dr_rows)
    write_rows(out_dir / "v22_10_drat_drbf_telemetry_bridge_summary.csv", dr_rows)
    empty_csvs = [
        "v22_10_drat_drbf_multibatch_officialization.csv",
        "v22_10_drat_component_waterfall.csv",
        "v22_10_drbf_component_waterfall.csv",
        "v22_10_drat_drbf_repair_attempts_summary.csv",
        "v22_10_drat_drbf_limited_functional_smoke.csv",
        "v22_10_drat_drbf_official_candidate_repair_scan.csv",
        "v22_10_drat_drbf_official_candidate_repair_scan_summary.csv",
        "v22_10_drat_drbf_blockh_repair_scan.csv",
        "v22_10_drat_drbf_blockh_repair_scan_summary.csv",
        "v22_10_drat_drbf_fastk2_repair_scan.csv",
        "v22_10_drat_drbf_fastk2_repair_scan_summary.csv",
        "v22_10_drat_drbf_blockb_repair_scan.csv",
        "v22_10_drat_drbf_blockb_repair_scan_summary.csv",
        "v22_10_drat_drbf_combo_repair_scan.csv",
        "v22_10_drat_drbf_combo_repair_scan_summary.csv",
    ]
    for name in empty_csvs:
        write_rows(out_dir / name, [])
    scan_routes = {
        "v22_10_drat_drbf_official_candidate_repair_scan_route.json": "S1-CandidateScanNotRunLossAgnosticContract",
        "v22_10_drat_drbf_blockh_repair_scan_route.json": "S1-BlockHScanNotRunLossAgnosticContract",
        "v22_10_drat_drbf_fastk2_repair_scan_route.json": "S1-FastK2ScanNotRunLossAgnosticContract",
        "v22_10_drat_drbf_blockb_repair_scan_route.json": "S1-BlockBScanNotRunLossAgnosticContract",
        "v22_10_drat_drbf_combo_repair_scan_route.json": "S1-ComboScanNotRunLossAgnosticContract",
    }
    for name, scan_route in scan_routes.items():
        write_json(
            out_dir / name,
            {
                "route": scan_route,
                "scan_rows": 0,
                "summary_rows": 0,
                "candidate_groups": 0,
                "candidate_carriers": "",
                "D-RAT_candidate_robust_pass": 0,
                "D-RBF_candidate_robust_pass": 0,
                "official_closure_claimed": 0,
                "promotion_allowed": 0,
                "optimization_loss_agnostic_contract_pass": 0,
                "blocker": blocker,
            },
        )
    simple_svg(out_dir / "figures/v22_10_basis_efficiency_dashboard.svg", "v22.10 loss-agnostic basis efficiency", eff_rows + dr_rows, "best_forward_ratio")
    append_exec(
        out_dir,
        " ".join(command),
        status="blocked",
        note="formal S1 stopped before CE-targeted trainpath; arbitrary-loss/upstream-gradient efficiency runner is not implemented",
    )


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    _write_loss_agnostic_efficiency_block(
        out_dir,
        [
            PYTHON,
            "experiments/run_v22_10_basis_efficiency_closure.py",
            "--device",
            args.device,
            "--official-transition-batch-sizes",
            args.official_transition_batch_sizes,
            "--out-dir",
            str(out_dir),
        ],
    )
    return
    log_path = out_dir / "logs" / "lineB_v22_10_basis_efficiency_closure_inner_v22_09_runner.log"
    command = [
        PYTHON,
        "experiments/run_v22_09_basis_efficiency_closure.py",
        "--device",
        args.device,
        "--official-transition-batch-sizes",
        args.official_transition_batch_sizes,
        "--out-dir",
        str(out_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(ROOT), text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        route = {
            "route": "S1-BasisEfficiencyRunnerFailed",
            "D-CHE_S1_pass": 0,
            "D-FOU_S1_pass": 0,
            "D-RAT_robust_pass": 0,
            "D-RBF_robust_pass": 0,
            "D-RAT_D-RBF_multibatch_closed": 0,
            "blocker": "inner_v22_09_basis_runner_failed",
            "inner_returncode": proc.returncode,
            "inner_log_path": str(log_path),
        }
        write_json(out_dir / "v22_10_basis_efficiency_route.json", route)
        write_rows(out_dir / "v22_10_efficiency_full_loop_reconfirm.csv", [])
    else:
        eff = _copy_rows(out_dir, "v22_09_efficiency_full_loop_reconfirm.csv", "v22_10_efficiency_full_loop_reconfirm.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        dr = _copy_rows(out_dir, "v22_09_drat_drbf_multibatch_summary.csv", "v22_10_drat_drbf_multibatch_summary.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        _copy_rows(out_dir, "v22_09_drat_drbf_multibatch_officialization.csv", "v22_10_drat_drbf_multibatch_officialization.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        _copy_rows(out_dir, "v22_09_drat_component_waterfall.csv", "v22_10_drat_component_waterfall.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        _copy_rows(out_dir, "v22_09_drbf_component_waterfall.csv", "v22_10_drbf_component_waterfall.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        _copy_rows(out_dir, "v22_09_drat_drbf_repair_attempts_summary.csv", "v22_10_drat_drbf_repair_attempts_summary.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        _copy_rows(out_dir, "v22_09_drat_drbf_telemetry_bridge_summary.csv", "v22_10_drat_drbf_telemetry_bridge_summary.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner", "official_closure_claimed": 0})
        _copy_rows(out_dir, "v22_09_drat_drbf_limited_functional_smoke.csv", "v22_10_drat_drbf_limited_functional_smoke.csv", {"v22_10_readback_source": "fresh_inner_v22_09_runner"})
        inner = read_json(out_dir / "v22_09_basis_efficiency_route.json")
        candidate_rows: list[dict[str, Any]] = []
        candidate_summary: list[dict[str, Any]] = []
        candidate_route: dict[str, Any] = {}
        blockh_rows: list[dict[str, Any]] = []
        blockh_summary: list[dict[str, Any]] = []
        blockh_route: dict[str, Any] = {}
        fastk2_rows: list[dict[str, Any]] = []
        fastk2_summary: list[dict[str, Any]] = []
        fastk2_route: dict[str, Any] = {}
        blockb_rows: list[dict[str, Any]] = []
        blockb_summary: list[dict[str, Any]] = []
        blockb_route: dict[str, Any] = {}
        combo_rows: list[dict[str, Any]] = []
        combo_summary: list[dict[str, Any]] = []
        combo_route: dict[str, Any] = {}
        if not int_flag(inner.get("D-RAT_robust_pass")) or not int_flag(inner.get("D-RBF_robust_pass")):
            candidate_rows, candidate_summary, candidate_route = _official_candidate_scan(args, out_dir)
            simple_svg(out_dir / "figures/v22_10_drat_drbf_official_candidate_repair_scan.svg", "v22.10 D-RAT/D-RBF official candidate repair scan", candidate_summary, "best_forward_ratio")
        drat_candidate = int_flag(candidate_route.get("D-RAT_candidate_robust_pass"))
        drbf_candidate = int_flag(candidate_route.get("D-RBF_candidate_robust_pass"))
        if not (int_flag(inner.get("D-RAT_robust_pass")) or int_flag(inner.get("D-RBF_robust_pass")) or drat_candidate or drbf_candidate):
            blockh_rows, blockh_summary, blockh_route = _official_blockh_repair_scan(args, out_dir)
            simple_svg(out_dir / "figures/v22_10_drat_drbf_blockh_repair_scan.svg", "v22.10 D-RAT/D-RBF blockH repair scan", blockh_summary, "best_forward_ratio")
        drat_blockh = int_flag(blockh_route.get("D-RAT_candidate_robust_pass"))
        drbf_blockh = int_flag(blockh_route.get("D-RBF_candidate_robust_pass"))
        if not (int_flag(inner.get("D-RAT_robust_pass")) or int_flag(inner.get("D-RBF_robust_pass")) or drat_candidate or drbf_candidate or drat_blockh or drbf_blockh):
            fastk2_rows, fastk2_summary, fastk2_route = _official_fastk2_repair_scan(args, out_dir)
            simple_svg(out_dir / "figures/v22_10_drat_drbf_fastk2_repair_scan.svg", "v22.10 D-RAT/D-RBF fastK2 repair scan", fastk2_summary, "best_forward_ratio")
        drat_fastk2 = int_flag(fastk2_route.get("D-RAT_candidate_robust_pass"))
        drbf_fastk2 = int_flag(fastk2_route.get("D-RBF_candidate_robust_pass"))
        if not (int_flag(inner.get("D-RAT_robust_pass")) or int_flag(inner.get("D-RBF_robust_pass")) or drat_candidate or drbf_candidate or drat_blockh or drbf_blockh or drat_fastk2 or drbf_fastk2):
            blockb_rows, blockb_summary, blockb_route = _official_blockb_repair_scan(args, out_dir)
            simple_svg(out_dir / "figures/v22_10_drat_drbf_blockb_repair_scan.svg", "v22.10 D-RAT/D-RBF blockB repair scan", blockb_summary, "best_forward_ratio")
        drat_blockb = int_flag(blockb_route.get("D-RAT_candidate_robust_pass"))
        drbf_blockb = int_flag(blockb_route.get("D-RBF_candidate_robust_pass"))
        if not (int_flag(inner.get("D-RAT_robust_pass")) or int_flag(inner.get("D-RBF_robust_pass")) or drat_candidate or drbf_candidate or drat_blockh or drbf_blockh or drat_fastk2 or drbf_fastk2 or drat_blockb or drbf_blockb):
            combo_rows, combo_summary, combo_route = _official_combo_repair_scan(args, out_dir)
            simple_svg(out_dir / "figures/v22_10_drat_drbf_combo_repair_scan.svg", "v22.10 D-RAT/D-RBF combo repair scan", combo_summary, "best_forward_ratio")
        drat_combo = int_flag(combo_route.get("D-RAT_candidate_robust_pass"))
        drbf_combo = int_flag(combo_route.get("D-RBF_candidate_robust_pass"))
        route = {
            "route": "S1-BasisEfficiencyReconfirmed" if int_flag(inner.get("D-CHE_S1_pass")) and int_flag(inner.get("D-FOU_S1_pass")) else "S1-BasisEfficiencyBlocked",
            "D-CHE_S1_pass": int_flag(inner.get("D-CHE_S1_pass")),
            "D-FOU_S1_pass": int_flag(inner.get("D-FOU_S1_pass")),
            "D-RAT_robust_pass": int(int_flag(inner.get("D-RAT_robust_pass")) or drat_candidate or drat_blockh or drat_fastk2 or drat_blockb or drat_combo),
            "D-RBF_robust_pass": int(int_flag(inner.get("D-RBF_robust_pass")) or drbf_candidate or drbf_blockh or drbf_fastk2 or drbf_blockb or drbf_combo),
            "D-RAT_D-RBF_multibatch_closed": int(int_flag(inner.get("D-RAT_robust_pass")) or drat_candidate or drat_blockh or drat_fastk2 or drat_blockb or drat_combo or int_flag(inner.get("D-RBF_robust_pass")) or drbf_candidate or drbf_blockh or drbf_fastk2 or drbf_blockb or drbf_combo),
            "limited_smoke_allowed_rows": int_flag(inner.get("limited_smoke_allowed_rows")),
            "repair_attempted": int_flag(inner.get("repair_attempted")),
            "repair_attempt_rows": int_flag(inner.get("repair_attempt_rows")),
            "official_candidate_repair_scan_attempted": int(bool(candidate_summary)),
            "official_candidate_repair_scan_rows": len(candidate_rows),
            "official_candidate_repair_candidate_groups": int_flag(candidate_route.get("candidate_groups")),
            "official_candidate_repair_candidate_carriers": candidate_route.get("candidate_carriers", ""),
            "official_candidate_repair_scan_route": candidate_route.get("route", ""),
            "official_blockh_repair_scan_attempted": int(bool(blockh_summary)),
            "official_blockh_repair_scan_rows": len(blockh_rows),
            "official_blockh_repair_candidate_groups": int_flag(blockh_route.get("candidate_groups")),
            "official_blockh_repair_candidate_carriers": blockh_route.get("candidate_carriers", ""),
            "official_blockh_repair_scan_route": blockh_route.get("route", ""),
            "official_fastk2_repair_scan_attempted": int(bool(fastk2_summary)),
            "official_fastk2_repair_scan_rows": len(fastk2_rows),
            "official_fastk2_repair_candidate_groups": int_flag(fastk2_route.get("candidate_groups")),
            "official_fastk2_repair_candidate_carriers": fastk2_route.get("candidate_carriers", ""),
            "official_fastk2_repair_scan_route": fastk2_route.get("route", ""),
            "official_blockb_repair_scan_attempted": int(bool(blockb_summary)),
            "official_blockb_repair_scan_rows": len(blockb_rows),
            "official_blockb_repair_candidate_groups": int_flag(blockb_route.get("candidate_groups")),
            "official_blockb_repair_candidate_carriers": blockb_route.get("candidate_carriers", ""),
            "official_blockb_repair_scan_route": blockb_route.get("route", ""),
            "official_combo_repair_scan_attempted": int(bool(combo_summary)),
            "official_combo_repair_scan_rows": len(combo_rows),
            "official_combo_repair_candidate_groups": int_flag(combo_route.get("candidate_groups")),
            "official_combo_repair_candidate_carriers": combo_route.get("candidate_carriers", ""),
            "official_combo_repair_scan_route": combo_route.get("route", ""),
            "inner_returncode": proc.returncode,
            "inner_log_path": str(log_path),
            "blocker": "" if int_flag(inner.get("D-CHE_S1_pass")) and int_flag(inner.get("D-FOU_S1_pass")) else "D-CHE_D-FOU_efficiency_reconfirm_gate",
        }
        write_json(out_dir / "v22_10_basis_efficiency_route.json", route)
        simple_svg(out_dir / "figures/v22_10_basis_efficiency_dashboard.svg", "v22.10 basis efficiency", eff + dr, "best_forward_ratio")
    append_exec(
        out_dir,
        " ".join(command),
        status="completed" if proc.returncode == 0 else "blocked",
        note=f"inner_returncode={proc.returncode} route={read_json(out_dir / 'v22_10_basis_efficiency_route.json').get('route')} log={log_path}",
    )
    if (out_dir / "v22_10_drat_drbf_official_candidate_repair_scan_summary.csv").exists():
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_10_basis_efficiency_closure.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --official-repair-hidden-grid {args.official_repair_hidden_grid} --out-dir {out_dir} [internal official candidate scan]",
            status="completed",
            note=f"candidate_scan_route={read_json(out_dir / 'v22_10_drat_drbf_official_candidate_repair_scan_route.json').get('route')} rows={len(read_rows(out_dir / 'v22_10_drat_drbf_official_candidate_repair_scan.csv'))}",
        )
    if (out_dir / "v22_10_drat_drbf_blockh_repair_scan_summary.csv").exists():
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_10_basis_efficiency_closure.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --official-blockh-repair-hidden-grid {args.official_blockh_repair_hidden_grid} --official-blockh-grid {args.official_blockh_grid} --out-dir {out_dir} [internal blockH repair scan]",
            status="completed",
            note=f"blockh_scan_route={read_json(out_dir / 'v22_10_drat_drbf_blockh_repair_scan_route.json').get('route')} rows={len(read_rows(out_dir / 'v22_10_drat_drbf_blockh_repair_scan.csv'))}",
        )
    if (out_dir / "v22_10_drat_drbf_fastk2_repair_scan_summary.csv").exists():
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_10_basis_efficiency_closure.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --official-fastk2-repair-hidden-grid {args.official_fastk2_repair_hidden_grid} --out-dir {out_dir} [internal fastK2 repair scan]",
            status="completed",
            note=f"fastk2_scan_route={read_json(out_dir / 'v22_10_drat_drbf_fastk2_repair_scan_route.json').get('route')} rows={len(read_rows(out_dir / 'v22_10_drat_drbf_fastk2_repair_scan.csv'))}",
        )
    if (out_dir / "v22_10_drat_drbf_blockb_repair_scan_summary.csv").exists():
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_10_basis_efficiency_closure.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --official-blockb-repair-hidden-grid {args.official_blockb_repair_hidden_grid} --official-blockb-grid {args.official_blockb_grid} --out-dir {out_dir} [internal blockB repair scan]",
            status="completed",
            note=f"blockb_scan_route={read_json(out_dir / 'v22_10_drat_drbf_blockb_repair_scan_route.json').get('route')} rows={len(read_rows(out_dir / 'v22_10_drat_drbf_blockb_repair_scan.csv'))}",
        )
    if (out_dir / "v22_10_drat_drbf_combo_repair_scan_summary.csv").exists():
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_10_basis_efficiency_closure.py --device {args.device} --official-transition-batch-sizes {args.official_transition_batch_sizes} --official-combo-repair-hidden-grid {args.official_combo_repair_hidden_grid} --official-combo-blockh-grid {args.official_combo_blockh_grid} --official-combo-blockb-grid {args.official_combo_blockb_grid} --official-combo-warps-grid {args.official_combo_warps_grid} --out-dir {out_dir} [internal combo repair scan]",
            status="completed",
            note=f"combo_scan_route={read_json(out_dir / 'v22_10_drat_drbf_combo_repair_scan_route.json').get('route')} rows={len(read_rows(out_dir / 'v22_10_drat_drbf_combo_repair_scan.csv'))}",
        )


if __name__ == "__main__":
    main()
