#!/usr/bin/env python3
"""v22.06 final route, execution log, recap, and packet builder."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_06_common import (  # noqa: E402
    PYTHON,
    V2206_RECAP_DOC,
    append_exec,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    run_cmd,
    sha256_file,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-clean-self-test", type=int, default=1)
    return p


def _artifact_index(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            resolved = path.resolve()
            try:
                artifact = str(resolved.relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({"artifact": artifact, "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _preserve_manual_recap_section(path: Path) -> list[str]:
    """Keep the human-written lab notes across finalizer reruns."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    start = text.find("## 人工复盘笔记 / Observation / Insight")
    if start < 0:
        return []
    end = text.find("\n## Artifact Index", start)
    if end < 0:
        end = len(text)
    section = text[start:end].strip("\n")
    return section.splitlines() + [""] if section else []


def _run_clean_self_test(out_dir: Path) -> tuple[int, str]:
    zip_path, _bundle = build_packet(out_dir)
    clean_root = out_dir / "clean_unzip_self_test"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(clean_root)
    source = clean_root / "02_SOURCE_TREE"
    command = [
        PYTHON,
        "experiments/run_v22_06_s013_truth_gate.py",
        "--mode",
        "all",
        "--source-root",
        str(source),
        "--self-contained-import-check",
        "1",
        "--out-dir",
        str(out_dir),
    ]
    code, log = run_cmd(command, cwd=source, timeout=1200)
    (out_dir / "v22_06_clean_unzip_self_test.log").write_text(log, encoding="utf-8")
    return code, str(source)


def _route(out_dir: Path) -> dict[str, Any]:
    code_rows = read_rows(out_dir / "v22_06_code_truth_gate.csv")
    code_pass = bool(code_rows) and all(int_flag(r.get("pass")) for r in code_rows)
    eff = read_json(out_dir / "v22_06_efficiency_route.json")
    dr = read_json(out_dir / "v22_06_drat_drbf_officialization_route.json")
    metric = read_json(out_dir / "v22_06_metric_solver_route.json")
    terminal = read_json(out_dir / "v22_06_terminal_preservation_route.json")
    kan = read_json(out_dir / "v22_06_kan_source_mapping_route.json")
    dche = int_flag(eff.get("D-CHE_S1_pass"))
    dfou = int_flag(eff.get("D-FOU_S1_pass"))
    dr_pass = any(int_flag(v.get("production_fused_official_rows")) for v in dr.values() if isinstance(v, dict))
    functional_route = str(metric.get("functional_route", ""))
    blockers = []
    if not code_pass:
        blockers.append("S0.13")
    if not dche or not dfou:
        blockers.append("D-CHE_or_D-FOU_S1")
    if not dr_pass:
        blockers.append("D-RAT_D-RBF_official")
    if functional_route not in {"C4-H3200SourceOpened", "F4-KANProductiveTerminalSource"}:
        blockers.append("functional_metric_solver_source_chain")
    if str(terminal.get("decision", "")).startswith("C5Blocked"):
        blockers.append(str(terminal.get("blocker", "C5")))
    if str(kan.get("decision", "")) in {"KANMappingNotEntered", "KANSourceChannelWriterMissing"}:
        blockers.append(str(kan.get("blocker", "KAN_mapping")))
    route = functional_route or "F0-MetricNoEffect"
    if not code_pass:
        route = "R0-CodeMetricInvalid"
    elif functional_route == "C4-H3200SourceOpened":
        route = "C4-H3200SourceOpened_C5PendingOrBlocked"
    return {
        "route": route,
        "promotion_allowed": 0,
        "blocking_metric": ";".join(x for x in blockers if x),
        "S0_13_pass": int(code_pass),
        "D-CHE_S1_pass": dche,
        "D-FOU_S1_pass": dfou,
        "DRAT_DRBF_any_production_official_pass": int(dr_pass),
        "functional_route": functional_route,
        "terminal_preservation_decision": terminal.get("decision", ""),
        "KAN_mapping_decision": kan.get("decision", ""),
        "next_codex_action": "repair C1/C3/C4 blocker before h4800/KAN mapping" if functional_route != "C4-H3200SourceOpened" else "run C5 terminal preservation and KAN channel mapping",
    }


def _write_required_figures(out_dir: Path) -> None:
    placeholders = {
        "code_truth_dashboard.svg": ("v22.06 code truth dashboard", read_rows(out_dir / "v22_06_code_truth_gate.csv"), "pass"),
        "semantic_alias_heatmap.svg": ("v22.06 semantic alias heatmap", read_rows(out_dir / "v22_06_functional_alias_matrix.csv"), "semantic_alias"),
        "function_displacement_alias_heatmap.svg": ("v22.06 function displacement alias heatmap", read_rows(out_dir / "v22_06_function_displacement_alias_matrix.csv"), "function_displacement_cosine"),
        "source_observability_predictor_auc.svg": ("v22.06 source observability", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "C1_observability_pass"),
        "split_transfer_target_contrast.svg": ("v22.06 split transfer target contrast", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "B2_transfer_gain_mean"),
        "NDS_vs_terminal_erosion.svg": ("v22.06 NDS vs terminal erosion", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "source_h3200_mean"),
        "Sobolev_RKHS_energy_vs_source.svg": ("v22.06 Sobolev/RKHS vs source", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "source_h800_mean"),
        "optimizer_cumulative_projection_h3200_to_h4800.svg": ("v22.06 optimizer projection", read_rows(out_dir / "v22_06_terminal_preservation_summary.csv"), "removed_destructive_component_norm"),
        "short_long_source_state_trace.svg": ("v22.06 short/long source state", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "source_h1600_mean"),
        "debt_transition_h3200_to_h4800.svg": ("v22.06 debt transition", read_rows(out_dir / "v22_06_terminal_preservation_summary.csv"), "source_h4800"),
        "signal_channel_reservoir_projection.svg": ("v22.06 signal/reservoir projection", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "source_to_reservoir_leakage_mean"),
        "info_volume_and_fold_proxy_trace.svg": ("v22.06 info volume/fold proxy", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "projection_residual_Gf_mean"),
        "MLP_source_decomposition_hidden_readout.svg": ("v22.06 MLP source decomposition", read_rows(out_dir / "v22_06_metric_solver_summary.csv"), "source_h800_mean"),
        "GPU_utilization_timeline.svg": ("v22.06 GPU utilization timeline", read_rows(out_dir / "v22_06_command_journal.csv"), ""),
    }
    for name, (title, rows, metric) in placeholders.items():
        path = out_dir / "figures" / name
        if not path.exists():
            simple_svg(path, title, rows, metric)


def _source_artifact_status(out_dir: Path) -> list[dict[str, Any]]:
    specs = [
        (
            "metric_solver_source_smoke_v6_full",
            "registration_blocker: new M264 specs were absent from mlp allowlist; not used as final M264 evidence",
        ),
        (
            "metric_solver_source_smoke_v6_full_fresh",
            "fresh T6 dual-memory / early-warm-carry readback after allowlist repair",
        ),
        (
            "metric_solver_source_smoke_v7_hidden_block",
            "fresh T7 hidden-block solver readback",
        ),
        (
            "metric_solver_source_smoke_v8_adaptive_hidden_block",
            "fresh T8 adaptive hidden-block solver readback",
        ),
        (
            "metric_solver_source_smoke_v9_c3_gated_hidden_block",
            "fresh T9 C3-gated hidden-block final readback",
        ),
        (
            "metric_solver_source_smoke_v10_compensated_hidden_block",
            "initial T10 compensated hidden-block readback before trace compensation whitelist repair",
        ),
        (
            "metric_solver_source_smoke_v10_compensated_hidden_block_fresh",
            "fresh T10 compensated hidden-block readback after trace compensation whitelist repair",
        ),
        (
            "metric_solver_source_smoke_v11_periodic_compensated_hidden_block",
            "fresh T10 I3 periodic boundary carry readback",
        ),
        (
            "metric_solver_source_smoke_v12_sgd_bootstrap_compensated_hidden_block",
            "fresh T10 SGD-bootstrap then compensated hidden carry readback",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v12_readback",
            "combined official readback for T9/T10 hidden-block, compensation, periodic, and bootstrap smokes",
        ),
        (
            "metric_solver_source_smoke_v13_optimizer_transport",
            "fresh T10 optimizer-state transport smoke from plan 11.5",
        ),
        (
            "metric_solver_source_smoke_v14_optimizer_transport_strength_sanity",
            "fresh T10 optimizer-state transport low-strength sanity smoke",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v14_readback",
            "combined official readback for T9/T10 hidden-block, compensation, periodic, bootstrap, optimizer transport, and low-strength sanity smokes",
        ),
        (
            "metric_solver_source_smoke_v15_early_observable_source_channel",
            "diagnostic whitelist blocker: T11 target construction diagnostics were absent from raw trace; not used as final T11 evidence",
        ),
        (
            "metric_solver_source_smoke_v15_early_observable_source_channel_fresh",
            "fresh T11 train-only early-observable source-channel smoke after trace whitelist repair",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v15_readback",
            "combined official readback including fresh T11 early-observable source-channel smoke",
        ),
        (
            "metric_solver_source_smoke_v16_train_split_control_gate",
            "fresh T11 train-split control-relative gate smoke after T11 source formation failed",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v16_readback",
            "combined official readback including T11 control-relative gate smoke",
        ),
        (
            "metric_solver_source_smoke_v17_adamw_compatible_c4",
            "control-missing blocker: ran only new FU specs without matched controls; source_h* empty and not used as final AdamW-compatible C4 evidence",
        ),
        (
            "metric_solver_source_smoke_v17_adamw_compatible_c4_fresh",
            "fresh AdamW-compatible C4 smoke with matched controls: T10 AdamW bootstrap+transport and T11 AdamW-relative gate",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v17_readback",
            "combined official readback including AdamW-compatible C4 integration smoke",
        ),
        (
            "metric_solver_source_smoke_v18_early_warm_then_transport",
            "registration_blocker: new phase-composed specs were absent from mlp allowlist; only matched controls ran, not used as final v18 evidence",
        ),
        (
            "metric_solver_source_smoke_v18_early_warm_then_transport_fresh",
            "fresh T10 phase-composed C4 smoke after allowlist repair: early direct warm source writer then optimizer-state transport",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v18_readback",
            "combined official readback including phase-composed early-warm then optimizer-transport smoke",
        ),
        (
            "metric_solver_source_smoke_v19_soft_compensated_hidden_block",
            "fresh T12 soft-compensated hidden-block smoke: train-only C3-preserving partial readout compensation",
        ),
        (
            "metric_solver_source_smoke_combined_v9_to_v19_readback",
            "combined official readback including T12 soft-compensated hidden-block smoke",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, decision in specs:
        matrix = read_rows(out_dir / name / "v21_01_source_retention_matrix.csv")
        ids = sorted({str(r.get("v21_id", "")) for r in matrix if str(r.get("v21_id", "")).strip()})
        rows.append(
            {
                "artifact": name,
                "command_journal": f"{name}/v21_01_command_journal.csv" if (out_dir / name / "v21_01_command_journal.csv").exists() else "",
                "matrix_rows": len(matrix),
                "unique_v21_ids": len(ids),
                "measured_rows": sum(1 for r in matrix if str(r.get("execution_status", "")).lower() in {"measured", "completed", ""}),
                "decision": decision,
            }
        )
    return rows


def _row_by_id(rows: list[dict[str, Any]], v21_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("v21_id", "")) == v21_id:
            return row
    return {}


def _repair_readback(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [
        ("T6 dual-memory early warm", "MLP-V2206-C4-T6G0-EarlyWarmCarry-stop1200-lr50"),
        ("T7 hidden block carry", "MLP-V2206-C4-T7G0-SourceStateCarry"),
        ("T8 adaptive hidden early warm", "MLP-V2206-C4-T8G0-EarlyWarmCarry-stop800-lr50"),
        ("T9 C3-gated hidden carry", "MLP-V2206-C4-T9G0-SourceStateCarry"),
        ("T9 C3-gated hidden early warm", "MLP-V2206-C4-T9G0-EarlyWarmCarry-stop800-lr50"),
        ("T10 compensated hidden carry", "MLP-V2206-C4-T10G0-SourceStateCarry"),
        ("T10 compensated hidden early warm", "MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50"),
        ("T10 compensated hidden early warm 1200", "MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50"),
        ("T10 periodic boundary alt100", "MLP-V2206-C4-T10G0-PeriodicCarry-alt100-lr50"),
        ("T10 periodic boundary alt200", "MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50"),
        ("T10 SGD bootstrap 400", "MLP-V2206-C4-T10G0-SGDBootstrap400Carry-lr50"),
        ("T10 SGD bootstrap 800", "MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50"),
        ("T10 optimizer transport lr50", "MLP-V2206-C4-T10G0-OptTransport-lr50"),
        ("T10 optimizer transport lr100", "MLP-V2206-C4-T10G0-OptTransport-lr100"),
        ("T10 optimizer transport lr150", "MLP-V2206-C4-T10G0-OptTransport-lr150"),
        ("T10 optimizer transport lr300", "MLP-V2206-C4-T10G0-OptTransport-lr300"),
        ("T10 early warm400 then transport", "MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150"),
        ("T10 early warm800 then transport", "MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150"),
        ("T11 early observable solver", "MLP-V2206-S2-T11G0-EarlyObservableSolver"),
        ("T11 early observable warm carry", "MLP-V2206-C4-T11G0-EarlyWarmCarry-stop800-lr50"),
        ("T11 early observable optimizer transport", "MLP-V2206-C4-T11G0-OptTransport-lr150"),
        ("T11 train-split control gate", "MLP-V2206-C4-T11G0-ControlRelativeGate-stop800-lr50"),
        ("T10 AdamW bootstrap transport", "MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150"),
        ("T11 AdamW-relative gate", "MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50"),
        ("T12 soft-compensated solver", "MLP-V2206-S2-T12G0-SoftCompensatedHiddenBlockSolver"),
        ("T12 soft-compensated carry", "MLP-V2206-C4-T12G0-SourceStateCarry"),
        ("T12 soft-compensated transport", "MLP-V2206-C4-T12G0-OptTransport-lr150"),
    ]
    out: list[dict[str, Any]] = []
    for attempt, v21_id in wanted:
        row = _row_by_id(metric_rows, v21_id)
        out.append(
            {
                "attempt": attempt,
                "v21_id": v21_id,
                "source_h100": row.get("source_h100_mean", ""),
                "source_h400": row.get("source_h400_mean", ""),
                "source_h800": row.get("source_h800_mean", ""),
                "source_h1600": row.get("source_h1600_mean", ""),
                "source_h3200": row.get("source_h3200_mean", ""),
                "C1": row.get("C1_observability_pass", ""),
                "C3": row.get("C3_actuation_pass", ""),
                "C4": row.get("C4_h3200_source_pass", ""),
                "ActuationR2": row.get("ActuationR2_mean", ""),
                "projection_residual": row.get("projection_residual_Gf_mean", ""),
                "B2_transfer": row.get("B2_transfer_gain_mean", ""),
                "hidden_fraction": row.get("block_source_hidden_residual_fraction_mean", ""),
                "hidden_cap_ratio": row.get("block_source_hidden_function_cap_ratio_mean", ""),
                "compensation_scale": row.get("block_source_hidden_compensation_scale_mean", ""),
                "compensation_mix": row.get("block_source_hidden_compensation_mix_mean", ""),
                "compensation_residual": row.get("block_source_hidden_compensation_residual_ratio_mean", ""),
                "transport_active": row.get("optimizer_transport_active_mean", ""),
                "transport_strength": row.get("optimizer_transport_strength_mean", ""),
                "transport_before": row.get("optimizer_transport_projection_before_mean", ""),
                "transport_after": row.get("optimizer_transport_projection_after_mean", ""),
                "control_gate_active": row.get("train_split_control_gate_active_mean", ""),
                "control_gate_accept": row.get("train_split_control_gate_accept_mean", ""),
                "control_gate_margin": row.get("train_split_control_gate_signal_margin_mean", ""),
                "adamw_bootstrap_active": row.get("adamw_bootstrap_writer_active_mean", ""),
                "adamw_gate_active": row.get("adamw_control_gate_active_mean", ""),
                "adamw_gate_accept": row.get("adamw_control_gate_accept_mean", ""),
                "adamw_gate_margin": row.get("adamw_control_gate_signal_margin_mean", ""),
                "failure": row.get("failure_taxonomy", ""),
            }
        )
    return out


def _dataset_readback(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "MLP-V2206-C4-T7G0-SourceStateCarry",
        "MLP-V2206-C4-T9G0-SourceStateCarry",
        "MLP-V2206-C4-T10G0-SourceStateCarry",
        "MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50",
        "MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50",
        "MLP-V2206-C4-T10G0-PeriodicCarry-alt100-lr50",
        "MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50",
        "MLP-V2206-C4-T10G0-SGDBootstrap400Carry-lr50",
        "MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50",
        "MLP-V2206-C4-T10G0-OptTransport-lr50",
        "MLP-V2206-C4-T10G0-OptTransport-lr100",
        "MLP-V2206-C4-T10G0-OptTransport-lr150",
        "MLP-V2206-C4-T10G0-OptTransport-lr300",
        "MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150",
        "MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150",
        "MLP-V2206-S2-T11G0-EarlyObservableSolver",
        "MLP-V2206-C4-T11G0-EarlyWarmCarry-stop800-lr50",
        "MLP-V2206-C4-T11G0-OptTransport-lr150",
        "MLP-V2206-C4-T11G0-ControlRelativeGate-stop800-lr50",
        "MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150",
        "MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50",
        "MLP-V2206-S2-T12G0-SoftCompensatedHiddenBlockSolver",
        "MLP-V2206-C4-T12G0-SourceStateCarry",
        "MLP-V2206-C4-T12G0-OptTransport-lr150",
    }
    out: list[dict[str, Any]] = []
    for row in matrix_rows:
        v21_id = str(row.get("v21_id", ""))
        if v21_id not in wanted:
            continue
        out.append(
            {
                "v21_id": v21_id,
                "dataset": row.get("dataset", ""),
                "source_h100": row.get("source_h100", ""),
                "source_h400": row.get("source_h400", ""),
                "source_h800": row.get("source_h800", ""),
                "source_h1600": row.get("source_h1600", ""),
                "source_h3200": row.get("source_h3200", ""),
                "R3200/1600": row.get("retention_h3200_over_h1600", ""),
                "retained_flag": row.get("retained_flag", ""),
            }
        )
    return sorted(out, key=lambda r: (str(r.get("v21_id", "")), str(r.get("dataset", ""))))


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir).resolve()
    if int(args.run_clean_self_test):
        code, source = _run_clean_self_test(out_dir)
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_06_s013_truth_gate.py --mode all --source-root {source} --self-contained-import-check 1 --out-dir {out_dir}",
            status="completed" if code == 0 else "blocked",
            note=f"clean_unzip_returncode={code}",
        )
    _write_required_figures(out_dir)
    route = _route(out_dir)
    write_json(out_dir / "v22_06_final_route.json", route)
    write_rows(out_dir / "v22_06_final_route.csv", [route])
    artifact_rows = _artifact_index(out_dir)
    write_rows(out_dir / "v22_06_artifact_index.csv", artifact_rows)
    build_packet(out_dir)

    code_rows = read_rows(out_dir / "v22_06_code_truth_gate.csv")
    semantic_rows = read_rows(out_dir / "v22_06_functional_semantic_contract.csv")
    alias_rows = read_rows(out_dir / "v22_06_semantic_noncollapse_summary.csv")
    eff_rows = read_rows(out_dir / "v22_06_efficiency_full_loop_reconfirm.csv")
    dr_rows = read_rows(out_dir / "v22_06_drat_drbf_officialization_summary.csv")
    metric_rows = read_rows(out_dir / "v22_06_metric_solver_summary.csv")
    terminal_rows = read_rows(out_dir / "v22_06_terminal_preservation_summary.csv")
    kan_rows = read_rows(out_dir / "v22_06_kan_source_mapping_summary.csv")
    source_status_rows = _source_artifact_status(out_dir)
    repair_rows = _repair_readback(metric_rows)
    dataset_rows = _dataset_readback(read_rows(out_dir / "v22_06_metric_solver_raw_matrix.csv"))
    best_metric = (
        sorted(
            metric_rows,
            key=lambda r: (
                int_flag(r.get("C4_h3200_source_pass")),
                int_flag(r.get("C4_early_source_pass")),
                int_flag(r.get("C3_actuation_pass")),
                int_flag(r.get("C1_observability_pass")),
                finite_float(r.get("source_h3200_mean"), -999.0),
                finite_float(r.get("source_h800_mean"), -999.0),
            ),
            reverse=True,
        )[0]
        if metric_rows
        else {}
    )
    manual_recap_lines = _preserve_manual_recap_section(V2206_RECAP_DOC)

    recap = [
        "# DG-KAN v22.06 Training-Dynamics MetricGeometryFU BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- promotion_allowed: {route.get('promotion_allowed')}",
        f"- blocking_metric: `{route.get('blocking_metric')}`",
        f"- S0.13 pass: {route.get('S0_13_pass')}",
        f"- D-CHE/D-FOU S1 pass: {route.get('D-CHE_S1_pass')} / {route.get('D-FOU_S1_pass')}",
        f"- D-RAT/D-RBF production official any pass: {route.get('DRAT_DRBF_any_production_official_pass')}",
        f"- functional_route: `{route.get('functional_route')}`",
        f"- terminal_preservation_decision: `{route.get('terminal_preservation_decision')}`",
        f"- KAN_mapping_decision: `{route.get('KAN_mapping_decision')}`",
        f"- next_codex_action: {route.get('next_codex_action')}",
        "",
        "## Part A S0.13 Code / Metric / Mechanism Gate",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 40),
        "",
        "### Functional Semantic Contract",
        "",
        md_table(semantic_rows, ["mechanism_id", "target_constructor_type", "metric_operator_type", "solver_type", "uses_future_or_validation", "uses_audit_metric_for_direction", "is_proxy"], 20),
        "",
        "### Semantic Alias Summary",
        "",
        md_table(alias_rows, ["pairs", "semantic_alias_pairs", "undeclared_alias_pairs", "pass", "gradient_norm"], 10),
        "",
        "## Part B Basis Efficiency / Officialization",
        "",
        md_table(eff_rows, ["carrier", "forward_ratio_vs_mlp", "step_ratio_vs_mlp", "memory_ratio_vs_mlp", "v22_06_S1_pass", "v22_06_decision", "v22_06_blocker"], 20),
        "",
        md_table(dr_rows, ["carrier", "profile_rows", "production_fused_official_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "",
        "## Part C Metric-As-Geometry Solver",
        "",
        f"- best_metric_v21_id: `{best_metric.get('v21_id', '')}`",
        f"- best_metric_solver_name: `{best_metric.get('metric_solver_name', '')}`",
        f"- best source_h800: {best_metric.get('source_h800_mean', '')}",
        f"- best source_h3200: {best_metric.get('source_h3200_mean', '')}",
        f"- best C1/C3/C4: {best_metric.get('C1_observability_pass', '')}/{best_metric.get('C3_actuation_pass', '')}/{best_metric.get('C4_h3200_source_pass', '')}",
        "",
        md_table(metric_rows, ["v21_id", "metric_solver_name", "ActuationR2_mean", "projection_residual_Gf_mean", "B2_transfer_gain_mean", "source_to_reservoir_leakage_mean", "source_h100_mean", "source_h400_mean", "source_h800_mean", "source_h1600_mean", "source_h3200_mean", "C1_observability_pass", "C3_actuation_pass", "C4_early_source_pass", "C4_h3200_source_pass", "failure_taxonomy"], 20),
        "",
        "### Continuation Artifact Readback",
        "",
        md_table(source_status_rows, ["artifact", "command_journal", "matrix_rows", "unique_v21_ids", "measured_rows", "decision"], 20),
        "",
        "### M264-M269 Repair Readback",
        "",
        md_table(repair_rows, ["attempt", "v21_id", "source_h100", "source_h400", "source_h800", "source_h1600", "source_h3200", "C1", "C3", "C4", "ActuationR2", "projection_residual", "B2_transfer", "hidden_fraction", "hidden_cap_ratio", "compensation_scale", "compensation_mix", "compensation_residual", "transport_active", "transport_strength", "transport_before", "transport_after", "control_gate_active", "control_gate_accept", "control_gate_margin", "adamw_bootstrap_active", "adamw_gate_active", "adamw_gate_accept", "adamw_gate_margin", "failure"], 40),
        "",
        "### Dataset-Level Source Readback",
        "",
        md_table(dataset_rows, ["v21_id", "dataset", "source_h100", "source_h400", "source_h800", "source_h1600", "source_h3200", "R3200/1600", "retained_flag"], 20),
        "",
        "## Part D Terminal Preservation / KAN Mapping",
        "",
        md_table(terminal_rows, ["v21_id", "source_h3200", "source_h4800", "R4800_over_3200", "C5_productive_terminal_source", "preservation_status", "blocker"], 20),
        "",
        md_table(kan_rows, ["mapping_status", "blocker", "best_metric_v21_id", "KAN_source_channel_decision"], 20),
        "",
        "## 修改记录",
        "",
        "- 新增 `dgkan/fu/metric_solver.py`：实现 v22.06 S1 readout exact metric solver，输出 ActuationR2、projection residual、B2 transfer、metric/source-channel diagnostics。",
        "- 新增 `dgkan/fu/jacobian_sketch.py` 与 `dgkan/fu/basis_channel_metric.py`：用于函数位移 alias readback 和 hidden/readout/reservoir source energy audit。",
        "- 扩展 `dgkan/fu/mechanisms.py`：注册 M259-M268 v22.06 metric-as-geometry solver mechanisms，并声明 semantic contract。",
        "- 新增 M262/M263：按 C4 blocker 修复方向补 T4 SmoothManifold / T5 LowNDS readout exact solver；用于验证 low-Sobolev/low-NDS target 是否能打开 source formation。",
        "- 新增 M264/T6 DualMemorySource：从 train split short loss-cotangent 与 B1/B2 class-axis long memory agreement 生成 train-only target，测试 cross-split source memory 是否能补 C4。",
        "- 新增 M265/T7 HiddenBlockSource：在 readout exact solve 后加入 train-gradient hidden residual block，并诚实记录 `hidden_residual_is_exact_solve=0` 与 hidden fraction，用来测试 source 是否被 readout-only 限制卡住。",
        "- 新增 M266/T8 AdaptiveHiddenBlock：用 function displacement cap 限制 hidden residual，测试保住 C3 solver evidence 后是否仍能打开 source。",
        "- 新增 M267/T9 C3GatedHiddenBlock：对 hidden residual 做 train-only C3 gate/backtracking；tiny semantic audit 中与 M264 fallback alias 已声明，不伪装 non-collapse。",
        "- 新增 M268/T10 CompensatedHiddenBlock：先写入 hidden residual，再用 readout exact compensation 把当前 train split 的函数位移拉回 metric target，测试 hidden source 与 C3 solver evidence 是否能同时成立。",
        "- 按计划 11.5 新增 T10 optimizer-state transport integration：不新增同义 mechanism，而是在 runner 中把 T10 source carry 注入当前 SGD flat gradient，并移除反 source 的负投影；trace 落盘 transport before/after projection。",
        "- 按计划 11.6 新增 M269/T11 EarlyObservableSourceChannel：从当前 train batch 的两个 class-conditional split 生成 class-wise source axis，并用 mid-debt window 加权；方向不读取 validation/test/future，也不使用 source audit horizon。",
        "- 按计划 11.7 新增 T11 I4 train-split control-relative gate：每次 FU 提交前用同 batch 模拟 FU candidate 与 SGD candidate 的 B1/B2/corrupt gain，只有 FU 在 train-only signal margin 上胜出才写入。",
        "- 新增 v17 AdamW-compatible C4 integration：`v2206_adamw_bootstrap_then_optimizer_transport` 用 AdamW 走早期 800 步后切换 T10 transport，`v2206_adamw_control_relative_gate` 要求 T11 FU+AdamW 在 train split 上胜过 AdamW candidate 才写入；用于审计早期 h100/h400 负值是否来自 optimizer baseline mismatch。",
        "- 新增 v18 phase-composed C4 integration：`v2206_early_warm_then_optimizer_transport` 在 h400/h800 前用 direct warm writer 形成 source，随后切换同一 T10 carry direction 的 optimizer-state transport；这是相位组合检验，不是继续提高 transport 强度。",
        "- 新增 M270/T12 SoftCompensatedHiddenBlock：在 T10 的 hidden residual + readout compensation 内加入 train-only partial compensation mix 选择；目标是在 C3 过线前提下减少 readout compensation 对 hidden source 的抵消。",
        "- 扩展 `experiments/run_v17_common.py`：补 v22.06 solver trace whitelist，新增 I1 early-source-warm-carry、I2 source-state-carry、I3 periodic boundary carry、SGD-bootstrap-then-carry 与 optimizer-state transport 分支，并写入 warm writer / source carry / hidden residual / compensation / transport diagnostics。",
        "- 扩展 `experiments/run_v21_01_source_retention.py`：注册 `MLP-V2206-*` source-retention specs、C4 source-state carry specs、early warm carry specs、hidden-block specs、T10 optimizer transport specs 与 T11 early-observable specs，并修复 `--scope mlp` allowlist 缺口；第一次 v6 full allowlist miss 只作为 registration blocker 记录。",
        "- 扩展 `experiments/run_v22_06_metric_solver_fu.py`：纳入 M264-M269 IDs、T10 optimizer transport 与 T11 early-observable spec IDs，并汇总 early-warm/source-carry/hidden-residual/compensation/transport/T11 class-axis diagnostics；summary 只读 runner artifact，不生成实验数值。",
        "- 调整 `dgkan/kernels/fused_rbf.py` low-D launch block sizing；D-RAT/D-RBF officialization 使用 production manual CE trainpath probe 复核，不采用 micro benchmark 替代。",
        "- 新增 v22.06 runner：S0.13 truth gate、metric solver summary、terminal preservation gate、KAN mapping gate、D-RAT/D-RBF officialization、efficiency reconfirm、finalize。",
        "",
        "## 分析 / Insight / 证据链",
        "",
        "- S0.13 的核心证据来自 clean unzip packet 内运行；若 clean unzip/import/contract 任一失败，route 会停在 `R0-CodeMetricInvalid`。",
        "- M259-M261 的 semantic contract 均标记 `is_proxy=0`，因为 update 由 readout exact solve 构造，并通过函数位移 alias matrix 读回；它们不是 v22.05 flat-gradient metric projection。",
        "- C1/C3/C4 分开判定：C1 看 train-only B2 transfer 和 source/reservoir leakage，C3 看 ActuationR2/projection residual，C4 才看 matched-control source retention。任何一段失败都不进入 promotion。",
        f"- 本轮最佳 functional row 是 `{best_metric.get('v21_id', '')}`：source_h3200_mean={best_metric.get('source_h3200_mean', '')}，但 source_h800_mean={best_metric.get('source_h800_mean', '')}，C4_h3200_source_pass={best_metric.get('C4_h3200_source_pass', '')}；因此只能写作 C3/C4 进展，不能写成 retained source promotion。",
        "- 第一次 v6 full 只产生 42 个 matrix rows、缺少新 M264 specs；这是 allowlist registration blocker，不作为 M264 科学负结果。修复 allowlist 后 fresh v6/v7/v8/v9 均重新落盘。",
        "- T6 dual-memory + I1 early warm carry 能把 best h3200 推到正数，但 h100/h400/h800 仍为负，说明它打开的是 late/local recovery，不是 C4 要求的连续 early source chain。",
        "- T7 hidden block 是本轮 source 读数最强的方向：source_h800_mean 与 source_h3200_mean 转正，但 ActuationR2/projection residual/B2 破坏 C1/C3，因此不能把 hidden residual 写成 metric solver success。",
        "- T8 adaptive cap 与 T9 C3 gate 尝试修复 T7 的 solver evidence；它们能恢复 C1/C3，但 hidden residual 被压低或归零，source 又回到负值。",
        "- T10 readout compensation 保住了 C1/C3 solver evidence，并在 trace 中落盘 nonzero compensation norm / scale / residual ratio；但 early source chain 仍未打开，说明“hidden residual + exact readout compensation”只解决了 solver-evidence 张力，没有解决 pre-h800 source formation。",
        "- T10 periodic boundary carry 与 SGD-bootstrap-then-carry 均为 fresh 3-dataset smoke：它们产生了若干 dataset-local late positives，但没有让 h100/h400/h800 与 h3200 同时满足 dataset-invariant C4 gate。",
        "- T10 optimizer-state transport 是 plan 11.5 的 optimizer-conflict fallback：它只改写当前 optimizer gradient，不额外直接 apply FU；若 transport projection after 明显上升但 C4 仍失败，则说明 source washout 不只是当前 step 的反 source 投影。",
        "- T11 early-observable source-channel 是对 C1 的直接修复：它先测试当前 train split 是否存在可重复 class-axis source，再进入 warm carry / optimizer transport；若 T11 仍不能让 h100/h400/h800 同时转正，则 blocker 会继续定位在 early source observability，而不是 terminal preservation。",
        "- T11 I4 control-relative gate 是对 T11 的 follow-up：若它拒绝大多数 FU 或通过后仍 h100/h400 为负，则说明当前 train split B1/B2/corrupt proxy 仍不足以预测 matched-control source formation。",
        "- AdamW-compatible C4 是对当前 blocker 的直接检验：若 AdamW bootstrap 打开早期 source 但 AdamW-relative gate 没有 FU margin，则只能说明 optimizer baseline 解释了 early source，不能说明 FunctionalUpdate 本身达成 promotion；若二者都失败，则排除简单 optimizer mismatch。",
        "- v18 phase-composed C4 检验的是“先形成早期 source，再写入 optimizer state”是否比纯 transport 更接近 C4；若它仍不能让 h100/h400/h800 连续过线，则 blocker 不是单纯相位切换，而是 target/source observer 本身尚未产生早期 dataset-invariant signal。",
        "- T12 soft compensation 检验的是 T7/T10 之间的张力：若 partial compensation 能保住 C1/C3 但仍无法把 h100/h400/h800 同时转正，则早期 source blocker 不是 full readout compensation 单独造成。",
        "- Dataset-level evidence 仍不稳定：T7 让 KMNIST 与部分 Fashion/MNIST horizon 转正，但 MNIST h100/h400 与 Fashion h3200 仍负；T9 C3-gated rows 在 MNIST h3200 局部转正，但 KMNIST/Fashion h3200 仍负。",
        "- low-NDS/SmoothManifold target 修复没有解决 source formation：T4/T5 仍表现为 ActuationR2 高、B2 transfer 正，但 matched-control source 读数为负。",
        "- 因 C4 h3200 source gate 未过，本轮没有启动 h4800/full 4GPU continuation；这是 fail-closed，而不是漏跑。",
        "- Terminal preservation 与 KAN mapping 没有在 h3200 source gate 之前强行启动；如果复盘显示 `blocked_before_C5` 或 `KANMappingNotEntered`，这是按计划 fail-closed。",
        "- D-RAT/D-RBF officialization 只读取/重跑 official manual CE trainpath probe；micro-near rows 不会被写成 production fused pass。",
        "",
        *manual_recap_lines,
        "## Artifact Index",
        "",
        md_table(artifact_rows, ["artifact", "exists", "size_bytes", "sha256"], 160),
    ]
    V2206_RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_06_finalize.py --out-dir {out_dir}", status="completed", note=f"route={route['route']} promotion={route['promotion_allowed']}")
    # Rebuild after writing docs and appending the final execution-log entry so
    # both zip deliverables contain the latest audit trail.
    build_packet(out_dir)


if __name__ == "__main__":
    main()
