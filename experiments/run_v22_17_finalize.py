#!/usr/bin/env python3
"""Finalize v22.17 artifacts, figures, route, and Chinese execution/recap logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_17_common import (  # noqa: E402
    EXEC_DOC,
    FIG_ROOT,
    OUT_ROOT,
    PYTHON,
    RECAP_DOC,
    append_exec,
    artifact_index,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    sha256_file,
    source_packet_paths,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _all(rows: list[dict[str, Any]], field: str) -> bool:
    return bool(rows) and all(int_flag(r.get(field)) for r in rows)


def _any_model_pass(rows: list[dict[str, Any]], model: str) -> bool:
    return any(r.get("model_name") == model and int(r.get("exit_code", 1)) == 0 for r in rows)


def _count_flag(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int_flag(r.get(field)) for r in rows)


def _refresh_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "plan": 0,
            "strict": 0,
            "candidate": 0,
            "skipped": 0,
            "max_nll": "",
            "min_gain": "",
        }
    return {
        "rows": len(rows),
        "plan": _count_flag(rows, "plan_mlp_controllability_criteria_pass"),
        "strict": _count_flag(rows, "strict_source_gain_over_base_pass"),
        "candidate": _count_flag(rows, "mlp_controllability_official_candidate_pass"),
        "skipped": sum(int(r.get("controller_refresh_skipped_count") or 0) for r in rows),
        "max_nll": max(float(r.get("controller_nll_delta_vs_base_h4800") or 0.0) for r in rows),
        "min_gain": min(float(r.get("controller_minus_base_source_loss_h4800") or 0.0) for r in rows),
    }


def _is_control_variant(model_name: str) -> bool:
    upper = str(model_name).upper()
    return any(token in upper for token in ["CONTROL", "RANDOM", "SIGNFLIP", "CORRUPT"])


def _base_model_for_delta(model_name: str) -> str:
    if model_name.startswith("DGMLP"):
        return "DGMLP"
    if model_name.startswith("DGKAN_DFOU"):
        return "DGKAN_DFOU"
    if model_name.startswith("DGKAN_DCHE"):
        return "DGKAN_DCHE"
    return model_name


def _mlp_fu_variant_stats(rows: list[dict[str, Any]], *, controls: bool = False) -> list[dict[str, Any]]:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    variants = sorted(
        {
            r.get("model_name", "")
            for r in rows
            if str(r.get("model_name", "")).startswith("DGMLP_FU")
            and _is_control_variant(str(r.get("model_name", ""))) == bool(controls)
        }
    )
    out: list[dict[str, Any]] = []
    for variant in variants:
        vals: list[dict[str, float]] = []
        for row in rows:
            if row.get("model_name") != variant:
                continue
            base = by_key.get((row.get("dataset"), str(row.get("seed")), "DGMLP"))
            if not base:
                continue
            vals.append(
                {
                    "nll_delta": finite_float(row.get("final_test_loss_NLL")) - finite_float(base.get("final_test_loss_NLL")),
                    "accuracy_delta": finite_float(row.get("final_test_accuracy")) - finite_float(base.get("final_test_accuracy")),
                    "tail_q99_delta": finite_float(row.get("tail_loss_q99")) - finite_float(base.get("tail_loss_q99")),
                    "auc_loss_time_delta": finite_float(row.get("AUC_loss_time")) - finite_float(base.get("AUC_loss_time")),
                }
            )
        if not vals:
            continue
        n = len(vals)
        nll_pass = sum(v["nll_delta"] <= 0.02 for v in vals)
        acc_pass = sum(v["accuracy_delta"] >= -0.005 for v in vals)
        tail_pass = sum(v["tail_q99_delta"] <= 0.05 for v in vals)
        out.append(
            {
                "model_name": variant,
                "rows": n,
                "nll_noharm_pass": nll_pass,
                "accuracy_noharm_pass": acc_pass,
                "tail_q99_noharm_pass": tail_pass,
                "all_three_noharm_pass": sum(
                    v["nll_delta"] <= 0.02 and v["accuracy_delta"] >= -0.005 and v["tail_q99_delta"] <= 0.05
                    for v in vals
                ),
                "nll_improvement_pass": sum(v["nll_delta"] < -0.01 for v in vals),
                "auc_improvement_pass": sum(v["auc_loss_time_delta"] < 0 for v in vals),
                "accuracy_improvement_pass": sum(v["accuracy_delta"] >= 0 for v in vals),
                "max_nll_delta": max(v["nll_delta"] for v in vals),
                "min_accuracy_delta": min(v["accuracy_delta"] for v in vals),
                "max_tail_q99_delta": max(v["tail_q99_delta"] for v in vals),
                "max_auc_loss_time_delta": max(v["auc_loss_time_delta"] for v in vals),
                "metric_noharm_candidate_pass": int(n >= 9 and nll_pass >= 7 and acc_pass >= 7 and tail_pass >= 7),
                "metric_improvement_candidate_pass": int(
                    n >= 9
                    and sum(v["nll_delta"] < -0.01 for v in vals) >= 5
                    and sum(v["auc_loss_time_delta"] < 0 for v in vals) >= 5
                    and sum(v["accuracy_delta"] >= 0 for v in vals) >= 4
                ),
            }
        )
    return out


def _kan_fu_variant_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r.get("dataset"), str(r.get("seed")), r.get("model_name")): r for r in rows}
    variants = sorted(
        {
            r.get("model_name", "")
            for r in rows
            if str(r.get("model_name", "")).startswith("DGKAN_") and "_FU" in str(r.get("model_name", ""))
        }
    )
    out: list[dict[str, Any]] = []
    for variant in variants:
        vals: list[dict[str, float]] = []
        dense_flags: list[int] = []
        for row in rows:
            if row.get("model_name") != variant:
                continue
            base_name = _base_model_for_delta(str(row.get("model_name", "")))
            base = by_key.get((row.get("dataset"), str(row.get("seed")), base_name))
            if not base:
                continue
            vals.append(
                {
                    "nll_delta": finite_float(row.get("final_test_loss_NLL")) - finite_float(base.get("final_test_loss_NLL")),
                    "accuracy_delta": finite_float(row.get("final_test_accuracy")) - finite_float(base.get("final_test_accuracy")),
                    "tail_q99_delta": finite_float(row.get("tail_loss_q99")) - finite_float(base.get("tail_loss_q99")),
                    "auc_loss_time_delta": finite_float(row.get("AUC_loss_time")) - finite_float(base.get("AUC_loss_time")),
                    "controller_overhead_ratio": finite_float(row.get("controller_overhead_ratio"), 0.0),
                    "basis_channel_energy_fraction": finite_float(row.get("basis_channel_energy_fraction"), 0.0),
                }
            )
            dense_flags.append(int_flag(row.get("uses_dense_output_jacobian_official")))
        if not vals:
            continue
        n = len(vals)
        nll_pass = sum(v["nll_delta"] <= 0.02 for v in vals)
        acc_pass = sum(v["accuracy_delta"] >= -0.005 for v in vals)
        tail_pass = sum(v["tail_q99_delta"] <= 0.05 for v in vals)
        out.append(
            {
                "model_name": variant,
                "rows": n,
                "nll_noharm_pass": nll_pass,
                "accuracy_noharm_pass": acc_pass,
                "tail_q99_noharm_pass": tail_pass,
                "all_three_noharm_pass": sum(
                    v["nll_delta"] <= 0.02 and v["accuracy_delta"] >= -0.005 and v["tail_q99_delta"] <= 0.05
                    for v in vals
                ),
                "nll_improvement_pass": sum(v["nll_delta"] < -0.01 for v in vals),
                "auc_improvement_pass": sum(v["auc_loss_time_delta"] < 0 for v in vals),
                "accuracy_improvement_pass": sum(v["accuracy_delta"] >= 0 for v in vals),
                "max_nll_delta": max(v["nll_delta"] for v in vals),
                "min_accuracy_delta": min(v["accuracy_delta"] for v in vals),
                "max_tail_q99_delta": max(v["tail_q99_delta"] for v in vals),
                "max_auc_loss_time_delta": max(v["auc_loss_time_delta"] for v in vals),
                "max_controller_overhead_ratio": max(v["controller_overhead_ratio"] for v in vals),
                "min_basis_channel_energy_fraction": min(v["basis_channel_energy_fraction"] for v in vals),
                "uses_dense_output_jacobian_official_any": int(any(dense_flags)),
                "metric_noharm_candidate_pass": int(n >= 9 and nll_pass >= 7 and acc_pass >= 7 and tail_pass >= 7),
                "metric_improvement_candidate_pass": int(
                    n >= 9
                    and sum(v["nll_delta"] < -0.01 for v in vals) >= 5
                    and sum(v["auc_loss_time_delta"] < 0 for v in vals) >= 5
                    and sum(v["accuracy_delta"] >= 0 for v in vals) >= 4
                ),
                "strict_basis_native_official_pass": 0,
            }
        )
    return out


def _build_packet() -> Path:
    packet = OUT_ROOT / "v22_17_code_review_packet.zip"
    if packet.exists():
        packet.unlink()
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in source_packet_paths():
            if not path.exists():
                continue
            arc = path.resolve().relative_to(ROOT)
            z.write(path, arc)
            manifest.append({"packet_path": str(arc), "source_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        for path in sorted(OUT_ROOT.rglob("v22_17_*")):
            if path.is_file() and path.suffix.lower() != ".zip" and "_code_packet_unzip" not in path.parts:
                arc = Path("artifacts") / path.relative_to(OUT_ROOT)
                z.write(path, arc)
                manifest.append({"packet_path": str(arc), "source_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(OUT_ROOT / "v22_17_required_source_files.csv", manifest)
    return packet


def _compile_packet(packet: Path) -> None:
    target = OUT_ROOT / "_code_packet_unzip"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as z:
        z.extractall(target)
    proc = subprocess.run(
        [PYTHON, "-m", "compileall", "-q", "experiments", "dgkan"],
        cwd=str(target),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (OUT_ROOT / "v22_17_clean_unzip_compileall.log").write_text(proc.stdout + f"\nexit_code={proc.returncode}\n", encoding="utf-8")
    proc2 = subprocess.run(
        [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0,'.'); import experiments.run_v22_17_common; import dgkan.integration.kanbefair_adapter; print('import_closure_ok')",
        ],
        cwd=str(target),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (OUT_ROOT / "v22_17_clean_unzip_import_closure.log").write_text(proc2.stdout + f"\nexit_code={proc2.returncode}\n", encoding="utf-8")


def _write_deferred() -> None:
    continual_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_continual_learning_matrix.csv")
    continual_baseline_smoke = any(r.get("model_name") == "MLP" and str(r.get("exit_code")) == "0" for r in continual_rows) and any(
        r.get("model_name") == "KAN" and str(r.get("exit_code")) == "0" for r in continual_rows
    )
    deferred = [
        {"item": "G2 vision expanded EMNIST/Cifar10/SVHN", "reason": "not run in this smoke pass; requires longer external benchmark budget", "algorithm_failure": 0},
        {"item": "G3 UCI/tabular expanded", "reason": "ucimlrepo/network availability not used for official claim in this pass", "algorithm_failure": 0},
        {"item": "G5 text/audio/symbolic", "reason": "text/audio dependencies/cache are diagnostic/deferred; symbolic not part of strict FC-PureKAN claim here", "algorithm_failure": 0},
        {"item": "Part E MLP+FU expanded improvement", "reason": "3x3 h440 MLP+FU metric no-harm, trained controls, and h4800 long MLP+FU bridge are recorded; h4800 long still does not pass improvement candidate gate, and expanded datasets remain incomplete", "algorithm_failure": 0},
        {"item": "official actionable risk predictor subgroup/deployment", "reason": "all-scope diagnostic risk gate is evaluated separately; subgroup and production deployment remain deferred", "algorithm_failure": 0},
        {"item": "official analytic/sketched controllability controller", "reason": "basis JVP/VJP gradcheck plus gradient/split-consensus/JVP-history H4800 sketch audits are recorded; best MLP refresh repair passed strict source-gain on 9/9 small-subset smoke rows with no task-NLL breach; KAN native-basis h4800 exploration now has source/full-loop-efficiency/no-harm 9/9, but task/AUC improvement, KAN-vs-MLP+FU comparison, expanded/continual evidence, and official promotion remain incomplete", "algorithm_failure": 0},
    ]
    if not continual_baseline_smoke:
        deferred.insert(2, {"item": "G4 Class_MNIST continual learning", "reason": "not run; bridge smoke prioritized after A/G0/G1", "algorithm_failure": 0})
    else:
        deferred.insert(2, {"item": "G4 Class_MNIST continual FU improvement", "reason": "KANbeFair exact Class_MNIST original MLP/KAN dry-run exits 0; DGMLP_FU/DGKAN_FU forgetting improvement not run/proven", "algorithm_failure": 0})
    write_rows(OUT_ROOT / "v22_17_deferred_items.csv", deferred)
    empty_with_reason = [{"status": "deferred", "reason": "not run in v22.17 smoke pass; see v22_17_deferred_items.csv"}]
    for name in [
        "v22_17_basis_jvp_vjp_gradcheck.csv",
        "v22_17_mlp_fu_source_dynamics.csv",
        "v22_17_mlp_fu_overguidance_matrix.csv",
        "v22_17_mlp_fu_controls_matrix.csv",
        "v22_17_mlp_fu_expanded_dataset_matrix.csv",
        "v22_17_kanbefair_continual_learning_matrix.csv",
        "v22_17_kanbefair_symbolic_diagnostic_matrix.csv",
    ]:
        path = OUT_ROOT / name
        if not path.exists():
            write_rows(path, empty_with_reason)


def _gate_summary() -> tuple[list[dict[str, Any]], str]:
    prov = read_json(OUT_ROOT / "v22_17_kanbefair_provenance.json")
    import_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_import_smoke.csv")
    complexity_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_model_complexity_smoke.csv")
    baseline_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_baseline_reproduction.csv")
    bridge_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix.csv")
    trained_control_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_trained_controls_3x3_h440.csv")
    jvp_refresh_bridge_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_jvp_refresh_bridge_3x3_h440.csv")
    mlp_fu_long_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_mlp_fu_long_3x3_h4800.csv")
    continual_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_continual_learning_matrix.csv")
    kan_jvp_refresh_rows = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_kan_jvp_refresh_bridge_3x3_h120.csv")
    kan_native_basis_summary = read_rows(OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120.csv")
    kan_native_basis_h4800_summary = read_rows(OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800.csv")
    mlp_fu_stats = _mlp_fu_variant_stats(bridge_rows)
    mlp_fu_control_stats = _mlp_fu_variant_stats(trained_control_rows, controls=True)
    mlp_fu_jvp_refresh_stats = _mlp_fu_variant_stats(jvp_refresh_bridge_rows)
    mlp_fu_long_stats = _mlp_fu_variant_stats(mlp_fu_long_rows)
    kan_jvp_refresh_stats = _kan_fu_variant_stats(kan_jvp_refresh_rows)
    if mlp_fu_stats:
        write_rows(OUT_ROOT / "v22_17_mlp_fu_smoke_pass_summary.csv", mlp_fu_stats)
    if mlp_fu_control_stats:
        write_rows(OUT_ROOT / "v22_17_mlp_fu_trained_control_pass_summary.csv", mlp_fu_control_stats)
    if mlp_fu_jvp_refresh_stats:
        write_rows(OUT_ROOT / "v22_17_mlp_fu_jvp_refresh_bridge_summary.csv", mlp_fu_jvp_refresh_stats)
    if mlp_fu_long_stats:
        write_rows(OUT_ROOT / "v22_17_mlp_fu_long_h4800_pass_summary.csv", mlp_fu_long_stats)
    if kan_jvp_refresh_stats:
        write_rows(OUT_ROOT / "v22_17_kan_jvp_refresh_bridge_diagnostic_summary.csv", kan_jvp_refresh_stats)
    source_ablation = read_rows(OUT_ROOT / "v22_17_source_candidate_ablation.csv")
    source_official_rows = read_rows(OUT_ROOT / "v22_17_source_usefulness_official_matrix.csv")
    risk_rows = read_rows(OUT_ROOT / "v22_17_risk_prediction_matrix.csv")
    ctrl_rows = read_rows(OUT_ROOT / "v22_17_controllability_manifold_matrix.csv")
    refresh_scale005 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_scale005_3x3.csv")
    refresh_best = _refresh_summary(refresh_scale005)
    worktree_import = any(r.get("target") == "worktree_patched" and int_flag(r.get("import_ok")) for r in import_rows)
    baseline_mlp = _any_model_pass(baseline_rows, "MLP")
    baseline_kan = _any_model_pass(baseline_rows, "KAN")
    bridge_pass = bool(bridge_rows)
    continual_baseline_smoke = any(r.get("model_name") == "MLP" and str(r.get("exit_code")) == "0" for r in continual_rows) and any(
        r.get("model_name") == "KAN" and str(r.get("exit_code")) == "0" for r in continual_rows
    )
    source_proxy = any(finite_float(r.get("U_beats_random_fraction"), 0.0) > 0.5 for r in source_ablation)
    source_official = any(int_flag(r.get("official_source_usefulness_pass")) for r in source_official_rows)
    risk_official = any(int_flag(r.get("official_pass")) for r in risk_rows)
    ctrl_has_rows = bool(ctrl_rows)
    mlp_noharm_variants = [r for r in mlp_fu_stats if int_flag(r.get("metric_noharm_candidate_pass"))]
    all_mlp_fu_stats = [*mlp_fu_stats, *mlp_fu_jvp_refresh_stats, *mlp_fu_long_stats]
    mlp_improvement_variants = [r for r in all_mlp_fu_stats if int_flag(r.get("metric_improvement_candidate_pass"))]
    jvp_noharm_variants = [r for r in mlp_fu_jvp_refresh_stats if int_flag(r.get("metric_noharm_candidate_pass"))]
    long_noharm_variants = [r for r in mlp_fu_long_stats if int_flag(r.get("metric_noharm_candidate_pass"))]
    long_improvement_variants = [r for r in mlp_fu_long_stats if int_flag(r.get("metric_improvement_candidate_pass"))]
    kan_jvp_noharm_variants = [r for r in kan_jvp_refresh_stats if int_flag(r.get("metric_noharm_candidate_pass"))]
    kan_native_basis_smoke_rows = [
        r
        for r in kan_native_basis_summary
        if int(r.get("rows") or 0) >= 9
        and int(r.get("basis_energy_ge_050") or 0) >= 9
        and int(r.get("source_loss_h120_ge_0") or 0) >= 9
        and int(r.get("controls_fail") or 0) >= 9
        and int(r.get("nll_noharm_vs_kan") or 0) >= 9
        and int(r.get("overhead_ratio_le_025") or 0) >= 9
    ]
    kan_native_basis_h4800_rows = [
        r
        for r in kan_native_basis_h4800_summary
        if int(r.get("rows") or 0) >= 9
        and int(r.get("basis_energy_ge_050") or 0) >= 9
        and int(r.get("source_loss_h4800_ge_0") or 0) >= 9
        and int(r.get("controls_fail") or 0) >= 9
        and int(r.get("nll_noharm_vs_kan") or 0) >= 9
        and int(r.get("overhead_ratio_le_025") or 0) >= 9
        and int(r.get("full_loop_ratio_le_3") or 0) >= 9
    ]
    control_improvement_variants = [r for r in mlp_fu_control_stats if int_flag(r.get("metric_improvement_candidate_pass"))]
    control_noharm_variants = [r for r in mlp_fu_control_stats if int_flag(r.get("metric_noharm_candidate_pass"))]
    noharm_names = ",".join(r.get("model_name", "") for r in mlp_noharm_variants[:4])
    control_noharm_names = ",".join(r.get("model_name", "") for r in control_noharm_variants[:4])
    best_nll_improve = max((int(r.get("nll_improvement_pass") or 0) for r in all_mlp_fu_stats), default=0)
    best_auc_improve = max((int(r.get("auc_improvement_pass") or 0) for r in all_mlp_fu_stats), default=0)
    best_acc_improve = max((int(r.get("accuracy_improvement_pass") or 0) for r in all_mlp_fu_stats), default=0)
    best_long_nll_improve = max((int(r.get("nll_improvement_pass") or 0) for r in mlp_fu_long_stats), default=0)
    best_long_auc_improve = max((int(r.get("auc_improvement_pass") or 0) for r in mlp_fu_long_stats), default=0)
    best_long_acc_improve = max((int(r.get("accuracy_improvement_pass") or 0) for r in mlp_fu_long_stats), default=0)
    long_noharm_names = ",".join(r.get("model_name", "") for r in long_noharm_variants[:4])
    best_control_nll_improve = max((int(r.get("nll_improvement_pass") or 0) for r in mlp_fu_control_stats), default=0)
    best_control_auc_improve = max((int(r.get("auc_improvement_pass") or 0) for r in mlp_fu_control_stats), default=0)
    best_control_acc_improve = max((int(r.get("accuracy_improvement_pass") or 0) for r in mlp_fu_control_stats), default=0)
    rows = [
        {"gate": "A provenance/patch", "pass": int(int_flag(prov.get("kanbefair_original_raw_unchanged")) and int_flag(prov.get("hardcoded_chdir_patched"))), "evidence": "v22_17_kanbefair_provenance.json"},
        {"gate": "A import/model complexity", "pass": int(worktree_import and _all(complexity_rows, "complexity_ok")), "evidence": "v22_17_kanbefair_import_smoke.csv + model_complexity_smoke.csv"},
        {"gate": "G1 KANbeFair baseline dry-run", "pass": int(baseline_mlp and baseline_kan), "evidence": "v22_17_kanbefair_baseline_reproduction.csv"},
        {"gate": "B task-useful source official", "pass": int(source_official), "evidence": "source_usefulness_official_matrix.csv + source_candidate_ablation.csv"},
        {"gate": "C actionable risk official", "pass": int(risk_official), "evidence": "risk_prediction_matrix.csv"},
        {
            "gate": "E MLP+FU no-harm metric smoke candidate",
            "pass": int(bool(mlp_noharm_variants)),
            "evidence": f"v22_17_mlp_fu_smoke_pass_summary.csv; metric-noharm variants={noharm_names or 'none'}; control no-harm variants={control_noharm_names or 'none'}, so official control-fail interpretation remains conservative",
        },
        {
            "gate": "E JVP-refresh bridge no-harm smoke candidate",
            "pass": int(bool(jvp_noharm_variants)),
            "evidence": "v22_17_mlp_fu_jvp_refresh_bridge_summary.csv; DGMLP_FU_JVP_REFRESH no-harm 9/9 on NLL/accuracy/tail; improvement gate not passed",
        },
        {
            "gate": "E MLP+FU h4800 long improvement check",
            "pass": int(bool(long_improvement_variants)),
            "evidence": f"v22_17_mlp_fu_long_h4800_pass_summary.csv; h4800 no-harm variants={long_noharm_names or 'none'}; long best counts: NLL<-0.01 {best_long_nll_improve}/9, AUC<0 {best_long_auc_improve}/9, acc>=0 {best_long_acc_improve}/9",
        },
        {
            "gate": "G4 Class_MNIST continual baseline dry-run",
            "pass": int(continual_baseline_smoke),
            "evidence": "v22_17_kanbefair_continual_learning_matrix.csv; original KANbeFair MLP/KAN exact protocol dry-run only, not DG-KAN/FU forgetting proof",
        },
        {
            "gate": "F KAN JVP-refresh bridge diagnostic",
            "pass": int(bool(kan_jvp_noharm_variants)),
            "evidence": "v22_17_kan_jvp_refresh_bridge_diagnostic_summary.csv; no-harm/AUC diagnostic only, strict basis-native official remains 0 due bridge dense-cache/h120/high overhead",
        },
        {
            "gate": "F KAN native-basis h120 smoke",
            "pass": int(bool(kan_native_basis_smoke_rows)),
            "evidence": "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120.csv; matched MLP hidden48 param ratio 0.943; basis>=0.5/source>=0/controls-fail/NLL-noharm/controller-overhead<=0.25 all 9/9, but full_loop<=3 is 1/9 and h3200/h4800 missing",
        },
        {
            "gate": "F KAN native-basis h4800 exploration",
            "pass": int(bool(kan_native_basis_h4800_rows)),
            "evidence": "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800.csv; D-CHE h8 matched-FLOPs MLP80 source_h4800/basis>=0.5/full_loop<=3/overhead<=0.25/controls-fail/NLL-noharm all 9/9; AUC improve vs KAN only 1/9 and official_pass remains 0",
        },
        {
            "gate": "E MLP+FU improvement smoke",
            "pass": int(bool(mlp_improvement_variants)),
            "evidence": f"best improvement counts across variants: NLL<-0.01 {best_nll_improve}/9, AUC<0 {best_auc_improve}/9, acc>=0 {best_acc_improve}/9",
        },
        {
            "gate": "E trained controls improvement fail",
            "pass": int(bool(mlp_fu_control_stats) and not control_improvement_variants),
            "evidence": f"v22_17_mlp_fu_trained_control_pass_summary.csv; control best counts: NLL<-0.01 {best_control_nll_improve}/9, AUC<0 {best_control_auc_improve}/9, acc>=0 {best_control_acc_improve}/9",
        },
        {
            "gate": "D MLP strict refresh smoke candidate",
            "pass": int(refresh_best["rows"] > 0 and refresh_best["candidate"] == refresh_best["rows"]),
            "evidence": f"jvp_useful_history refresh1600 vgate strict scale0.05: plan {refresh_best['plan']}/{refresh_best['rows']}, strict source-gain {refresh_best['strict']}/{refresh_best['rows']}, official_candidate {refresh_best['candidate']}/{refresh_best['rows']}, max NLL delta {refresh_best['max_nll']} in 3x3 smoke",
        },
        {"gate": "D controllability official", "pass": 0, "evidence": "MLP D criteria are met in a 3x3 small-subset refresh audit; KAN D-CHE h8 h4800 source/full-loop efficiency exploration is 9/9, but official end-to-end task/AUC improvement, KAN-vs-MLP+FU comparison, and full non-smoke integration remain incomplete"},
        {"gate": "E/F/G DG bridge smoke", "pass": int(bridge_pass), "evidence": "v22_17_kanbefair_dgkan_bridge_matrix.csv"},
    ]
    if not rows[0]["pass"] or not rows[1]["pass"] or not rows[2]["pass"]:
        route = "R0-CodeOrIntegrationFailed"
    elif bridge_pass and not ((source_official or source_proxy) and risk_official and ctrl_has_rows):
        route = "R9-KANbeFairBridgeReady_InternalMechanismPending"
    else:
        route = "R10-KANbeFairExternalNoHarm_MechanismPartial"
    write_rows(OUT_ROOT / "v22_17_gate_summary.csv", rows)
    write_json(OUT_ROOT / "v22_17_final_route.json", {"route": route, "official_promotion": 0, "timestamp": now_sg()})
    return rows, route


def _summary_stats() -> list[dict[str, Any]]:
    rows = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix.csv")
    out: list[dict[str, Any]] = []
    by_key = {(r.get("dataset"), r.get("seed"), r.get("model_name")): r for r in rows}
    def best(prefix: str, dataset: str, seed: str) -> dict[str, Any] | None:
        candidates = [
            r
            for r in rows
            if r.get("dataset") == dataset
            and r.get("seed") == seed
            and r.get("model_name", "").startswith(prefix)
            and not _is_control_variant(r.get("model_name", ""))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: finite_float(r.get("final_test_loss_NLL"), float("inf")))
    for dataset in sorted({r.get("dataset", "") for r in rows}):
        seed = "0"
        mlp = by_key.get((dataset, seed, "DGMLP"))
        mlpfu = best("DGMLP_FU", dataset, seed)
        kan = by_key.get((dataset, seed, "DGKAN_DFOU"))
        kanfu = best("DGKAN_DFOU_FU", dataset, seed)
        out.append(
            {
                "dataset": dataset,
                "MLP_NLL": "" if not mlp else mlp.get("final_test_loss_NLL", ""),
                "best_MLPFU_model": "" if not mlpfu else mlpfu.get("model_name", ""),
                "MLPFU_NLL": "" if not mlpfu else mlpfu.get("final_test_loss_NLL", ""),
                "MLPFU_delta_NLL": "" if not mlp or not mlpfu else finite_float(mlpfu.get("final_test_loss_NLL")) - finite_float(mlp.get("final_test_loss_NLL")),
                "KAN_NLL": "" if not kan else kan.get("final_test_loss_NLL", ""),
                "best_KANFU_model": "" if not kanfu else kanfu.get("model_name", ""),
                "KANFU_NLL": "" if not kanfu else kanfu.get("final_test_loss_NLL", ""),
                "KANFU_delta_NLL": "" if not kan or not kanfu else finite_float(kanfu.get("final_test_loss_NLL")) - finite_float(kan.get("final_test_loss_NLL")),
                "KANFU_vs_MLPFU_NLL_delta": "" if not kanfu or not mlpfu else finite_float(kanfu.get("final_test_loss_NLL")) - finite_float(mlpfu.get("final_test_loss_NLL")),
                "MLPFU_acc": "" if not mlpfu else mlpfu.get("final_test_accuracy", ""),
                "KANFU_acc": "" if not kanfu else kanfu.get("final_test_accuracy", ""),
            }
        )
    write_rows(OUT_ROOT / "v22_17_four_square_summary.csv", out)
    return out


def _svg(path: Path, title: str, lines: list[str]) -> None:
    height = 80 + 22 * len(lines)
    safe_lines = [line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines]
    body = "\n".join(f'<text x="24" y="{70 + 22 * i}" font-size="14" fill="#1f2937">{line}</text>' for i, line in enumerate(safe_lines))
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">\n'
            '<rect width="1100" height="100%" fill="#f8fafc"/>\n'
            f'<text x="24" y="36" font-size="22" font-family="sans-serif" fill="#111827">{title}</text>\n'
            f'<g font-family="monospace">{body}</g>\n'
            "</svg>\n"
        ),
        encoding="utf-8",
    )


def _write_figures(route: str, summary: list[dict[str, Any]]) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    continual = read_rows(OUT_ROOT / "v22_17_kanbefair_continual_learning_matrix.csv")
    dataset_lines = [
        f"{r['dataset']}: bestMLPFU={r.get('best_MLPFU_model','')} dNLL={r['MLPFU_delta_NLL']} | bestKANFU={r.get('best_KANFU_model','')} dNLL={r['KANFU_delta_NLL']} | KANFU-MLPFU={r['KANFU_vs_MLPFU_NLL_delta']}"
        for r in summary
    ]
    _svg(FIG_ROOT / "v22_17_project_status_dashboard.svg", "v22.17 status dashboard", [f"route={route}", "official_promotion=0", *dataset_lines])
    _svg(FIG_ROOT / "v22_17_source_usefulness_distribution.svg", "source usefulness", ["See v22_17_source_usefulness_official_matrix.csv", "Future labels are analysis-only; not used as runtime direction."])
    _svg(FIG_ROOT / "v22_17_actionable_vs_naive_washout_auc.svg", "actionable washout", ["See risk_prediction_matrix.csv.", "Neutral-control false alarm is recorded when available."])
    _svg(FIG_ROOT / "v22_17_raw_vs_reachable_manifold_residual.svg", "raw vs reachable residual", ["Finite-diff diagnostic only.", "See controllability_manifold_matrix.csv."])
    _svg(FIG_ROOT / "v22_17_mlp_fu_four_square_matrix.svg", "MLP/KAN x AdamW/FU smoke", dataset_lines or ["No bridge rows."])
    _svg(FIG_ROOT / "v22_17_kan_basis_energy_vs_source_loss.svg", "KAN basis energy vs source loss", ["See kan_basis_controllability_matrix.csv."])
    _svg(FIG_ROOT / "v22_17_efficiency_component_waterfall.svg", "efficiency smoke", ["See real_efficiency_controller_loop.csv.", "Dense output Jacobian official path used: 0."])
    _svg(FIG_ROOT / "v22_17_kanbefair_dataset_grid.svg", "KANbeFair dataset grid", dataset_lines or ["No dataset rows."])
    continual_lines = [
        f"{r.get('model_name')}: avg_acc={r.get('average_accuracy')} bwt={r.get('average_backward_transfer')} F0={r.get('Forgetting_T0_after_T2', '')} F1={r.get('Forgetting_T1_after_T2', '')}"
        for r in continual
        if r.get("status") == "pass"
    ]
    _svg(FIG_ROOT / "v22_17_continual_forgetting_curves.svg", "continual forgetting", continual_lines or ["Deferred; Class_MNIST exact protocol not run in this smoke pass."])
    _svg(FIG_ROOT / "v22_17_task_nll_accuracy_auc_separated.svg", "task NLL/accuracy/AUC", dataset_lines or ["No task rows."])


def _append_docs(route: str, gates: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    provenance = read_json(OUT_ROOT / "v22_17_kanbefair_provenance.json")
    patches = read_rows(OUT_ROOT / "v22_17_kanbefair_worktree_patch_manifest.csv")
    baseline = read_rows(OUT_ROOT / "v22_17_kanbefair_baseline_reproduction.csv")
    continual = read_rows(OUT_ROOT / "v22_17_kanbefair_continual_learning_matrix.csv")
    bridge = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix.csv")
    mlp_fu_stats = read_rows(OUT_ROOT / "v22_17_mlp_fu_smoke_pass_summary.csv")
    mlp_fu_control_stats = read_rows(OUT_ROOT / "v22_17_mlp_fu_trained_control_pass_summary.csv")
    mlp_fu_jvp_refresh_stats = read_rows(OUT_ROOT / "v22_17_mlp_fu_jvp_refresh_bridge_summary.csv")
    jvp_refresh_bridge = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_jvp_refresh_bridge_3x3_h440.csv")
    mlp_fu_long_stats = read_rows(OUT_ROOT / "v22_17_mlp_fu_long_h4800_pass_summary.csv")
    mlp_fu_long_bridge = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_mlp_fu_long_3x3_h4800.csv")
    kan_jvp_refresh_stats = read_rows(OUT_ROOT / "v22_17_kan_jvp_refresh_bridge_diagnostic_summary.csv")
    kan_jvp_refresh_bridge = read_rows(OUT_ROOT / "v22_17_kanbefair_dgkan_bridge_matrix_kan_jvp_refresh_bridge_3x3_h120.csv")
    kan_native_basis_best_summary = read_rows(OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120.csv")
    kan_native_basis_best_raw = read_rows(OUT_ROOT / "v22_17_kan_basis_native_audit_kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120.csv")
    kan_native_basis_h4800_summary = read_rows(OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800.csv")
    kan_native_basis_h4800_raw = read_rows(OUT_ROOT / "v22_17_kan_basis_native_audit_kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800.csv")
    kan_native_repair_sets = [
        ("dense-cache bridge h120 diagnostic", OUT_ROOT / "v22_17_kan_jvp_refresh_bridge_diagnostic_summary.csv"),
        ("native basis-only interval20 h120", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_basisonly_interval20_3x3_h120.csv"),
        ("native scale0.25 interval20 h120", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale025_interval20_3x3_h120.csv"),
        ("native scale0.10 interval20 h120", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval20_3x3_h120.csv"),
        ("native scale0.10 interval40 h120", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval40_3x3_h120.csv"),
        ("native scale0.10 interval40 h120 matched MLP48", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120.csv"),
        ("native scale0.10 interval40 h120 matched FLOPs MLP160", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_scale010_interval40_matchedflops160_3x3_h120.csv"),
        ("native D-CHE scale0.10 interval40 h120 matched FLOPs MLP160", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_scale010_interval40_matchedflops160_3x3_h120.csv"),
        ("native D-CHE scale0.10 interval120 h120 matched FLOPs MLP160", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_scale010_interval120_matchedflops160_3x3_h120.csv"),
        ("native D-CHE h8 scale0.10 interval120 h120 matched FLOPs MLP80", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_h8_scale010_interval120_matchedflops80_3x3_h120.csv"),
        ("native D-CHE h8 scale0.10 interval4800 h4800 matched FLOPs MLP80", OUT_ROOT / "v22_17_kan_basis_native_audit_summary_kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800.csv"),
    ]
    kan_native_repair_comparison: list[dict[str, Any]] = []
    for label, path in kan_native_repair_sets:
        for row in read_rows(path)[:1]:
            kan_native_repair_comparison.append(
                {
                    "repair": label,
                    "model_name": row.get("model_name", ""),
                    "rows": row.get("rows", ""),
                    "basis_ge_050": row.get("basis_energy_ge_050", ""),
                    "source_ge_0": row.get("source_loss_h4800_ge_0", row.get("source_loss_horizon_ge_0", row.get("source_loss_h120_ge_0", ""))),
                    "controls_fail": row.get("controls_fail", ""),
                    "nll_noharm_vs_kan": row.get("nll_noharm_vs_kan", row.get("nll_noharm_pass", "")),
                    "auc_improve_vs_kan": row.get("auc_improve_vs_kan", row.get("auc_improvement_pass", "")),
                    "overhead_le_025": row.get("overhead_ratio_le_025", ""),
                    "full_loop_le_3": row.get("full_loop_ratio_le_3", ""),
                    "max_overhead": row.get("max_controller_overhead_ratio", ""),
                    "max_full_loop_ratio": row.get("max_full_loop_ratio_vs_mlp", ""),
                    "max_param_ratio_vs_mlp": row.get("max_param_ratio_vs_mlp", ""),
                    "max_flops_ratio_vs_mlp": row.get("max_flops_ratio_vs_mlp", ""),
                    "min_basis_energy": row.get("min_basis_channel_energy_fraction", ""),
                    "max_nll_delta_vs_kan": row.get("max_nll_delta_vs_kan", row.get("max_nll_delta", "")),
                    "blockers": row.get("blockers", ""),
                }
            )
    risk = read_rows(OUT_ROOT / "v22_17_risk_prediction_matrix.csv")
    ctrl = read_rows(OUT_ROOT / "v22_17_controllability_manifold_matrix.csv")
    gradcheck = read_rows(OUT_ROOT / "v22_17_basis_jvp_vjp_gradcheck.csv")
    sketch = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit.csv")
    split_sketch = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_split_consensus.csv")
    split_pos_sketch = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_split_consensus_positive.csv")
    split_pos_h20 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_split_consensus_positive_h20.csv")
    jvp_useful_sketch = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history.csv")
    jvp_useful_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_3x3.csv")
    jvp_refresh400_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh400_3x3.csv")
    jvp_refresh1600_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_3x3.csv")
    jvp_refresh1600_vgate_strict_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_3x3.csv")
    jvp_refresh1600_vgate_strict_scale05_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_scale05_3x3.csv")
    jvp_refresh1600_vgate_strict_scale025_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_scale025_3x3.csv")
    jvp_refresh1600_vgate_strict_scale01_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_scale01_3x3.csv")
    jvp_refresh1600_vgate_strict_scale005_3x3 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_scale005_3x3.csv")
    jvp_gain_sketch = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_gain_history.csv")
    jvp_useful_h20 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_h20.csv")
    jvp_gain_h20 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_gain_history_h20.csv")
    jvp_weighted_h20 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_weighted_history_h20.csv")
    jvp_linesearch_h20 = read_rows(OUT_ROOT / "v22_17_controllability_sketch_audit_jvp_useful_history_linesearch_h20.csv")
    deferred = read_rows(OUT_ROOT / "v22_17_deferred_items.csv")
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n# v22.17 最终复盘追加\n\n生成时间：{now_sg()}\n\n")
        f.write(f"Final route：`{route}`；official promotion：`0`。\n\n")
        f.write("## 1. 代码与 KANbeFair 集成证据\n\n")
        f.write(f"- zip sha256：`{provenance.get('kanbefair_zip_sha256', '')}`\n")
        f.write(f"- raw root：`{provenance.get('kanbefair_raw_root', '')}`\n")
        f.write(f"- worktree root：`{provenance.get('kanbefair_worktree_root', '')}`\n")
        f.write(f"- raw unchanged：`{provenance.get('kanbefair_original_raw_unchanged', '')}`\n")
        f.write("\n补丁审计：\n\n")
        f.write(md_table(patches, ["patch_id", "file", "detected", "patched", "patch_sha256", "reason"], max_rows=20))
        f.write("\nGate summary：\n\n")
        f.write(md_table(gates, ["gate", "pass", "evidence"], max_rows=20))
        f.write("\n## 2. KANbeFair baseline reproduction\n\n")
        f.write(md_table(baseline, ["dataset", "model_name", "attempt", "exit_code", "dry_run", "fallback_used", "test_metric_from_original", "param_count", "flops"], max_rows=20))
        if continual:
            f.write("\nKANbeFair Class_MNIST continual baseline dry-run（exact protocol: digits 0-2 / 3-5 / 6-8, num_classes=9）：\n\n")
            f.write(md_table(continual, ["dataset", "model_name", "dry_run", "exit_code", "average_accuracy", "average_backward_transfer", "ACC_after_task_2_on_task_0", "ACC_after_task_2_on_task_1", "ACC_after_task_2_on_task_2", "Forgetting_T0_after_T2", "Forgetting_T1_after_T2", "dgkan_official_evidence"], max_rows=20))
            f.write("\n说明：该表只证明 KANbeFair 原始 continual protocol 可执行；未接入 DGMLP_FU/DGKAN_FU，不能作为 FU forgetting improvement 证据。\n")
        f.write("\n## 3. DG bridge unified task metrics\n\n")
        f.write(md_table(summary, ["dataset", "MLP_NLL", "best_MLPFU_model", "MLPFU_NLL", "MLPFU_delta_NLL", "KAN_NLL", "best_KANFU_model", "KANFU_NLL", "KANFU_delta_NLL", "KANFU_vs_MLPFU_NLL_delta"], max_rows=20))
        if mlp_fu_stats:
            f.write("\nPart E MLP+FU 3x3 smoke pass summary（单一 variant 统计，不跨 seed/dataset 挑模型）：\n\n")
            f.write(md_table(mlp_fu_stats, ["model_name", "rows", "nll_noharm_pass", "accuracy_noharm_pass", "tail_q99_noharm_pass", "all_three_noharm_pass", "nll_improvement_pass", "auc_improvement_pass", "accuracy_improvement_pass", "max_nll_delta", "min_accuracy_delta", "max_tail_q99_delta", "metric_noharm_candidate_pass", "metric_improvement_candidate_pass"], max_rows=20))
            f.write(
                "\n说明：`metric_noharm_candidate_pass=1` 只表示 NLL/accuracy/tail 三个 metric gate 在 3x3 smoke 上达标；"
                "trained controls 需要单独解释，因此这里不提升为 Part E official no-harm。\n"
            )
        if mlp_fu_control_stats:
            f.write("\nPart E MLP+FU trained controls 3x3 smoke summary：\n\n")
            f.write(md_table(mlp_fu_control_stats, ["model_name", "rows", "nll_noharm_pass", "accuracy_noharm_pass", "tail_q99_noharm_pass", "all_three_noharm_pass", "nll_improvement_pass", "auc_improvement_pass", "accuracy_improvement_pass", "max_nll_delta", "min_accuracy_delta", "max_tail_q99_delta", "metric_noharm_candidate_pass", "metric_improvement_candidate_pass"], max_rows=20))
            f.write(
                "\n说明：trained controls 没有任何一个达到 improvement candidate gate；但部分 controls 也达到 no-harm metric gate，"
                "尤其 signflip control 有 NLL<-0.01 的 3/9 部分信号，因此 no-harm 的 control-fail 条件仍需保守解释。\n"
            )
        if mlp_fu_jvp_refresh_stats:
            f.write("\nPart E audit-derived JVP-refresh bridge 3x3 smoke summary：\n\n")
            f.write(md_table(mlp_fu_jvp_refresh_stats, ["model_name", "rows", "nll_noharm_pass", "accuracy_noharm_pass", "tail_q99_noharm_pass", "all_three_noharm_pass", "nll_improvement_pass", "auc_improvement_pass", "accuracy_improvement_pass", "max_nll_delta", "min_accuracy_delta", "max_tail_q99_delta", "max_auc_loss_time_delta", "metric_noharm_candidate_pass", "metric_improvement_candidate_pass"], max_rows=20))
            f.write("\nJVP-refresh bridge 原始行节选：\n\n")
            f.write(md_table(jvp_refresh_bridge, ["dataset", "seed", "model_name", "final_test_loss_NLL", "NLL_delta_vs_MLP_AdamW", "accuracy_delta_vs_MLP_AdamW", "AUC_loss_time_delta_vs_MLP_AdamW", "gate_accept_count", "gate_reject_count", "gate_reject_reason_counts", "mean_jvp_refresh_source_loss_gain", "mean_jvp_refresh_update_ratio", "controller_overhead_ratio"], max_rows=30))
        if mlp_fu_long_stats:
            f.write("\nPart E MLP+FU h4800 long bridge summary（512/256 fixed-step, 单一 variant 统计）：\n\n")
            f.write(md_table(mlp_fu_long_stats, ["model_name", "rows", "nll_noharm_pass", "accuracy_noharm_pass", "tail_q99_noharm_pass", "all_three_noharm_pass", "nll_improvement_pass", "auc_improvement_pass", "accuracy_improvement_pass", "max_nll_delta", "min_accuracy_delta", "max_tail_q99_delta", "max_auc_loss_time_delta", "metric_noharm_candidate_pass", "metric_improvement_candidate_pass"], max_rows=20))
            f.write(
                "\n说明：h4800 long 是针对 Part E `MLP+FU 接近 AdamW 但不优` blocker 的长程修复尝试；"
                "`DGMLP_FU_ACTIONABLE`、`DGMLP_FU_SPLIT_ACTIONABLE`、`DGMLP_FU_SPLIT_VIRTUAL` 在 NLL/accuracy/tail no-harm 上为 9/9，"
                "`DGMLP_FU_SPLIT_RELEASE` 出现 NLL<-0.01 的 3/9 与 AUC<0 的 5/9，但 tail/no-harm 不稳且未达到 improvement gate。\n"
            )
            f.write("\nPart E MLP+FU h4800 long raw rows：\n\n")
            f.write(md_table(mlp_fu_long_bridge, ["dataset", "seed", "model_name", "final_test_loss_NLL", "NLL_delta_vs_MLP_AdamW", "accuracy_delta_vs_MLP_AdamW", "AUC_loss_time_delta_vs_MLP_AdamW", "source_loss_h_final", "gate_accept_count", "gate_reject_count", "gate_reject_reason_counts", "controller_overhead_ratio"], max_rows=50))
        if kan_jvp_refresh_stats:
            f.write("\nPart F KAN JVP-refresh bridge 3x3 h120 diagnostic summary：\n\n")
            f.write(md_table(kan_jvp_refresh_stats, ["model_name", "rows", "nll_noharm_pass", "accuracy_noharm_pass", "tail_q99_noharm_pass", "all_three_noharm_pass", "nll_improvement_pass", "auc_improvement_pass", "accuracy_improvement_pass", "max_nll_delta", "min_accuracy_delta", "max_tail_q99_delta", "max_auc_loss_time_delta", "max_controller_overhead_ratio", "min_basis_channel_energy_fraction", "uses_dense_output_jacobian_official_any", "metric_noharm_candidate_pass", "metric_improvement_candidate_pass", "strict_basis_native_official_pass"], max_rows=20))
            f.write(
                "\n说明：该 KAN 行是 bridge fixed-step diagnostic，`uses_dense_output_jacobian_official_any=0` 仅表示记录字段未走 dense-output official flag；"
                "但当前实现仍是 bridge/dense-cache 诊断、h120 短程且 controller overhead 约 0.97，不能提升为 strict basis-native official。\n"
            )
            f.write("\nKAN JVP-refresh bridge 原始行节选：\n\n")
            f.write(md_table(kan_jvp_refresh_bridge, ["dataset", "seed", "model_name", "final_test_loss_NLL", "NLL_delta_vs_KAN_AdamW", "accuracy_delta_vs_KAN_AdamW", "AUC_loss_time_delta_vs_KAN_AdamW", "gate_accept_count", "gate_reject_count", "gate_reject_reason_counts", "mean_jvp_refresh_source_loss_gain", "mean_jvp_refresh_update_ratio", "basis_channel_energy_fraction", "readout_channel_energy_fraction", "uses_dense_output_jacobian_official", "controller_overhead_ratio"], max_rows=30))
        if kan_native_repair_comparison:
            f.write("\nPart F strict KAN native-basis audit 修复对比：\n\n")
            f.write(md_table(kan_native_repair_comparison, ["repair", "rows", "basis_ge_050", "source_ge_0", "controls_fail", "nll_noharm_vs_kan", "auc_improve_vs_kan", "overhead_le_025", "full_loop_le_3", "max_overhead", "max_full_loop_ratio", "max_param_ratio_vs_mlp", "max_flops_ratio_vs_mlp", "min_basis_energy", "max_nll_delta_vs_kan", "blockers"], max_rows=20))
            f.write(
                "\n说明：native audit 使用 `native_no_dense_basis` PrimitiveKAN、basis selector、`uses_dense_output_jacobian_official=0`；"
                "controllability rows 中的 finite-diff J 仅作 audit diagnostic，不作为 official fast path。"
                "`scale0.10 interval40 + matched MLP hidden48` 是当前最好的 h120 native-basis smoke：KAN 参数量是 matched MLP 的 0.943，"
                "basis>=0.5、source_loss_h120>=0、controls fail、NLL no-harm、controller overhead<=0.25 均为 9/9；"
                "但 full-loop ratio<=3 只有 1/9，KAN FLOPs 仍是 matched-param MLP 的 3.77 倍。"
                "额外 matched-FLOPs MLP hidden160 diagnostic 中 FLOPs 比约 0.999，full-loop ratio<=3 仍只有 1/9，max full-loop ratio=4.441753301931999。"
                "随后 D-CHE h8 + matched-FLOPs MLP80 + interval4800 的 h4800 长程实验把 source_h4800、basis>=0.5、full-loop<=3、controller overhead<=0.25、controls fail、NLL no-harm 都推进到 9/9，max full-loop ratio=2.952011975627768，max overhead=0.0015814291144632278；"
                "但 AUC improve vs KAN 只有 1/9，且 `KAN_vs_MLPFU_delta`、MLP+FU improvement 与 external expanded proof 仍未完成，所以 official 仍为 0。\n"
            )
        if kan_native_basis_best_summary:
            f.write("\nPart F best native-basis h120 summary（scale0.10 interval40 + matched MLP hidden48）：\n\n")
            f.write(md_table(kan_native_basis_best_summary, ["model_name", "rows", "smoke_exploration_candidate_pass", "official_pass", "basis_energy_ge_050", "source_loss_h120_ge_0", "full_loop_ratio_le_3", "overhead_ratio_le_025", "controls_fail", "nll_noharm_vs_kan", "auc_improve_vs_kan", "max_full_loop_ratio_vs_mlp", "max_controller_overhead_ratio", "max_param_ratio_vs_mlp", "max_flops_ratio_vs_mlp", "min_basis_channel_energy_fraction", "max_nll_delta_vs_kan", "blockers"], max_rows=20))
            f.write("\nPart F best native-basis h120 raw FU rows：\n\n")
            f.write(md_table([r for r in kan_native_basis_best_raw if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")], ["dataset", "seed", "basis_channel_energy_fraction", "KAN_source_loss_h120", "controls_pass_count", "NLL_delta_vs_KAN_AdamW_native", "AUC_loss_time_delta_vs_KAN_AdamW_native", "full_loop_ratio_vs_mlp", "param_ratio_vs_mlp", "flops_ratio_vs_mlp", "controller_overhead_ratio", "gate_accept_count", "gate_reject_count", "refresh_interval_skip_count", "mean_jvp_reachable_projection_residual", "blocker"], max_rows=20))
        if kan_native_basis_h4800_summary:
            f.write("\nPart F best native-basis h4800 exploration summary（D-CHE h8 scale0.10 interval4800 + matched FLOPs MLP80）：\n\n")
            f.write(md_table(kan_native_basis_h4800_summary, ["model_name", "rows", "smoke_exploration_candidate_pass", "official_pass", "basis_energy_ge_050", "source_loss_h4800_ge_0", "full_loop_ratio_le_3", "overhead_ratio_le_025", "controls_fail", "nll_noharm_vs_kan", "auc_improve_vs_kan", "max_full_loop_ratio_vs_mlp", "max_controller_overhead_ratio", "max_param_ratio_vs_mlp", "max_flops_ratio_vs_mlp", "min_basis_channel_energy_fraction", "max_nll_delta_vs_kan", "blockers"], max_rows=20))
            f.write("\nPart F best native-basis h4800 raw FU rows：\n\n")
            f.write(md_table([r for r in kan_native_basis_h4800_raw if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")], ["dataset", "seed", "basis_channel_energy_fraction", "KAN_source_loss_h4800", "controls_pass_count", "NLL_delta_vs_KAN_AdamW_native", "AUC_loss_time_delta_vs_KAN_AdamW_native", "NLL_delta_vs_MLP_AdamW_same_run", "AUC_loss_time_delta_vs_MLP_AdamW_same_run", "full_loop_ratio_vs_mlp", "param_ratio_vs_mlp", "flops_ratio_vs_mlp", "controller_overhead_ratio", "gate_accept_count", "gate_reject_count", "refresh_interval_skip_count", "mean_jvp_reachable_projection_residual", "blocker"], max_rows=20))
        f.write("\n原始 bridge rows：\n\n")
        f.write(md_table(bridge, ["dataset", "model_name", "final_test_loss_NLL", "final_test_accuracy", "AUC_loss_time", "ECE", "Brier", "tail_loss_q99", "controller_overhead_ratio", "smoke_only"], max_rows=40))
        f.write("\n## 4. Mechanism evidence chain\n\n")
        b_gate = next((g for g in gates if g.get("gate") == "B task-useful source official"), {})
        c_gate = next((g for g in gates if g.get("gate") == "C actionable risk official"), {})
        f.write(
            f"- B/source-usefulness：读取 `v22_17_source_usefulness_official_matrix.csv`，gate pass={b_gate.get('pass', '')}；future labels 只用于 analysis，不用于 runtime direction。\n"
        )
        f.write(
            f"- C/actionable-risk：读取 `v22_17_risk_prediction_matrix.csv`，gate pass={c_gate.get('pass', '')}；matched-run 离线 label 与 neutral-control false alarm 已记录。\n\n"
        )
        f.write(md_table(risk, ["risk_model", "risk_scope", "horizon", "AUC_actionable_H100", "AUC_actionable_H200", "positive_label_rate_actionable_H100", "cross_dataset_AUC", "cross_seed_AUC", "false_alarm_rate_on_task_neutral_source", "task_neutral_eval_rows_H100", "risk_model_trained", "official_pass", "blocker"], max_rows=12))
        f.write("\n- D/controllability：记录了 finite-diff J diagnostic residual；若已运行 `v22_17_basis_jvp_vjp_gradcheck.csv`，该文件只证明/否定 basis VJP/JVP gradcheck，不等价于 analytic/sketched official controllability controller。\n\n")
        f.write(md_table(ctrl, ["dataset", "model_name", "dim", "useful_source_count", "raw_history_projection_residual", "jacobian_reachable_projection_residual", "controllability_ratio", "controls_pass_count"], max_rows=40))
        f.write("\nSketched J-space H3200/H4800 controller audit 节选：\n\n")
        f.write(md_table(sketch, ["dataset", "seed", "model_name", "jacobian_reachable_projection_residual", "raw_history_projection_residual", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "controls_pass_count", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        if split_sketch or split_pos_sketch or split_pos_h20:
            f.write("\nSketched J-space split-consensus source-candidate 修复尝试：\n\n")
            f.write(
                "本轮针对 `NoSourceGainVsBase` blocker 做了两类真实修复尝试："
                "`split_consensus` 用 batch 两半 CE 梯度的一致方向替代单 batch 梯度，"
                "`split_consensus_positive` 进一步只保留 split cosine > 0 的历史方向。"
                "controller 采用 `add_to_base`，目标仍是 `base_effect`，没有使用未来标签作为 runtime direction。\n\n"
            )
        if split_sketch:
            f.write("`split_consensus` H4800 结果：\n\n")
            f.write(md_table(split_sketch, ["dataset", "seed", "source_candidate", "target_split_consensus_cosine", "mean_history_split_consensus_cosine", "jacobian_reachable_projection_residual", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        if split_pos_sketch:
            f.write("\n`split_consensus_positive` H4800 结果：\n\n")
            f.write(md_table(split_pos_sketch, ["dataset", "seed", "source_candidate", "target_split_consensus_cosine", "mean_history_split_consensus_cosine", "jacobian_reachable_projection_residual", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        if split_pos_h20:
            f.write("\n`split_consensus_positive` H20 探针：\n\n")
            f.write(md_table(split_pos_h20, ["dataset", "seed", "source_candidate", "target_split_consensus_cosine", "mean_history_split_consensus_cosine", "jacobian_reachable_projection_residual", "source_loss_h20", "controller_minus_base_source_loss_h20", "controller_nll_delta_vs_base_h20", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        if jvp_useful_sketch or jvp_gain_sketch or jvp_useful_h20 or jvp_gain_h20 or jvp_weighted_h20 or jvp_linesearch_h20:
            f.write("\nSketched J-space JVP-scored source-history 修复尝试：\n\n")
            f.write(
                "`jvp_useful_history` 使用当前 batch 的 CE cotangent 与当前 JVP 对历史方向打分，选择即时 source_loss 为正且最高的历史方向；"
                "`jvp_gain_history` 进一步要求历史方向当前分数超过当前普通 gradient，否则退回 gradient。"
                "`jvp_weighted_history` 尝试按当前正 source score 对历史方向做加权平均。"
                "三者均不读取 future/test label，不按 dataset/seed 分支，仍用 strict source-gain-over-base gate 做候选审计。\n\n"
            )
        if jvp_useful_h20:
            f.write("`jvp_useful_history` H20 探针：\n\n")
            f.write(md_table(jvp_useful_h20, ["dataset", "seed", "source_candidate", "target_history_source_loss", "mean_history_source_loss", "jacobian_reachable_projection_residual", "source_loss_h20", "controller_minus_base_source_loss_h20", "controller_nll_delta_vs_base_h20", "blocker"], max_rows=20))
        if jvp_gain_h20:
            f.write("\n`jvp_gain_history` H20 探针：\n\n")
            f.write(md_table(jvp_gain_h20, ["dataset", "seed", "source_candidate", "target_history_source_loss", "target_current_base_source_loss", "target_history_gain_vs_current_base", "jacobian_reachable_projection_residual", "source_loss_h20", "controller_minus_base_source_loss_h20", "controller_nll_delta_vs_base_h20", "blocker"], max_rows=20))
        if jvp_weighted_h20:
            f.write("\n`jvp_weighted_history` H20 no-go 探针：\n\n")
            f.write(md_table(jvp_weighted_h20, ["dataset", "seed", "source_candidate", "target_history_source_loss", "target_current_base_source_loss", "target_history_gain_vs_current_base", "jacobian_reachable_projection_residual", "source_loss_h20", "controller_minus_base_source_loss_h20", "controller_nll_delta_vs_base_h20", "blocker"], max_rows=20))
        if jvp_linesearch_h20:
            f.write("\n`jvp_useful_history` current-batch source-gain line-search H20 no-go 探针：\n\n")
            f.write(md_table(jvp_linesearch_h20, ["dataset", "seed", "source_candidate", "controller_line_search_enabled", "line_search_selected_scale", "line_search_best_train_source_gain", "line_search_best_virtual_loss_delta", "source_loss_h20", "controller_minus_base_source_loss_h20", "controller_nll_delta_vs_base_h20", "blocker"], max_rows=20))
        if jvp_useful_sketch:
            f.write("\n`jvp_useful_history` H4800 结果：\n\n")
            f.write(md_table(jvp_useful_sketch, ["dataset", "seed", "source_candidate", "target_history_source_loss", "mean_history_source_loss", "jacobian_reachable_projection_residual", "random_basis_projection_residual", "shuffled_basis_projection_residual", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "controls_pass_count", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        if jvp_useful_3x3:
            plan_count = sum(int(row.get("plan_mlp_controllability_criteria_pass") or 0) for row in jvp_useful_3x3)
            strict_count = sum(int(row.get("strict_source_gain_over_base_pass") or 0) for row in jvp_useful_3x3)
            f.write("\n`jvp_useful_history` H4800 3x3 扩展 smoke：\n\n")
            f.write(f"- plan 原文 MLP D criteria pass：{plan_count}/{len(jvp_useful_3x3)}\n")
            f.write(f"- strict source-gain-over-base pass：{strict_count}/{len(jvp_useful_3x3)}\n\n")
            f.write(md_table(jvp_useful_3x3, ["dataset", "seed", "source_candidate", "jacobian_reachable_projection_residual", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "controls_pass_count", "plan_mlp_controllability_criteria_pass", "strict_source_gain_over_base_pass", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        refresh_sets = [
            ("refresh400_full_step", jvp_refresh400_3x3),
            ("refresh1600_full_step", jvp_refresh1600_3x3),
            ("refresh1600_vgate_strict", jvp_refresh1600_vgate_strict_3x3),
            ("refresh1600_vgate_strict_scale0.5", jvp_refresh1600_vgate_strict_scale05_3x3),
            ("refresh1600_vgate_strict_scale0.25", jvp_refresh1600_vgate_strict_scale025_3x3),
            ("refresh1600_vgate_strict_scale0.1", jvp_refresh1600_vgate_strict_scale01_3x3),
            ("refresh1600_vgate_strict_scale0.05", jvp_refresh1600_vgate_strict_scale005_3x3),
        ]
        refresh_rows: list[dict[str, Any]] = []
        for label, rows_ in refresh_sets:
            if rows_:
                n = len(rows_)
                refresh_rows.append(
                    {
                        "repair": label,
                        "rows": n,
                        "plan_pass": f"{sum(int(r.get('plan_mlp_controllability_criteria_pass') or 0) for r in rows_)}/{n}",
                        "strict_source_gain_pass": f"{sum(int(r.get('strict_source_gain_over_base_pass') or 0) for r in rows_)}/{n}",
                        "official_candidate_pass": f"{sum(int(r.get('mlp_controllability_official_candidate_pass') or 0) for r in rows_)}/{n}",
                        "refresh_skipped_total": sum(int(r.get("controller_refresh_skipped_count") or 0) for r in rows_),
                        "max_nll_delta_h4800": max(float(r.get("controller_nll_delta_vs_base_h4800") or 0.0) for r in rows_),
                        "min_source_gain_h4800": min(float(r.get("controller_minus_base_source_loss_h4800") or 0.0) for r in rows_),
                    }
                )
        if refresh_rows:
            f.write("\n`jvp_useful_history` full-loop refresh 修复对比：\n\n")
            f.write(md_table(refresh_rows, ["repair", "rows", "plan_pass", "strict_source_gain_pass", "official_candidate_pass", "refresh_skipped_total", "max_nll_delta_h4800", "min_source_gain_h4800"], max_rows=20))
        if jvp_refresh1600_vgate_strict_scale005_3x3:
            f.write("\n最佳当前修复 `refresh1600_vgate_strict_scale0.05` H4800 3x3 明细：\n\n")
            f.write(md_table(jvp_refresh1600_vgate_strict_scale005_3x3, ["dataset", "seed", "refresh_scale", "controller_refresh_count", "controller_refresh_skipped_count", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "plan_mlp_controllability_criteria_pass", "strict_source_gain_over_base_pass", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        if jvp_gain_sketch:
            f.write("\n`jvp_gain_history` H4800 结果：\n\n")
            f.write(md_table(jvp_gain_sketch, ["dataset", "seed", "source_candidate", "target_history_source_loss", "target_current_base_source_loss", "target_history_gain_vs_current_base", "jacobian_reachable_projection_residual", "source_loss_h4800", "controller_minus_base_source_loss_h4800", "controller_nll_delta_vs_base_h4800", "controls_pass_count", "mlp_controllability_official_candidate_pass", "blocker"], max_rows=20))
        f.write("\nBasis JVP/VJP gradcheck 节选：\n\n")
        f.write(md_table(gradcheck, ["dataset", "model_name", "implementation_mode", "selector", "uses_dense_basis_tensor_spec", "grad_relerr", "func_jvp_relerr_max", "jvp_vjp_gradcheck_pass", "official_basis_jvp_vjp_gradcheck_pass"], max_rows=40))
        f.write("\n## 4.1 本轮修复修改审计\n\n")
        f.write(
            "- 新增 `experiments/run_v22_17_basis_jvp_vjp_gradcheck.py`：用真实 KANbeFair batch 对 KAN basis/readout/all selector 做 CE backward、autograd reference、`torch.func.jvp` 与 finite-diff auxiliary sketch 检查；`official_basis_jvp_vjp_gradcheck_pass` 要求 selector=basis 且 `uses_dense_basis_tensor_spec=0`。\n"
            "- 修改 `experiments/run_v22_17_common.py`：把 basis gradcheck 与 sketched controllability audit 脚本纳入 required source packet，便于审计源码包复现。\n"
            "- 新增并多轮修改 `experiments/run_v22_17_controllability_sketch_audit.py`：加入 real KANbeFair batch 的 sketched J-space controller audit、optimizer state clone、first-step controller update semantics、`controller_minus_base_source_loss_h4800` gate、`jspace_target`、`controller_compose`、`split_consensus`、`split_consensus_positive`、`jvp_useful_history`、`jvp_gain_history`、`jvp_weighted_history` source candidate、current-batch source-gain line-search、周期 refresh、refresh virtual-loss gate、`controller_only` refresh mode 与 `refresh_scale`。\n"
            "- 修改 `experiments/run_v22_17_kanbefair_dgkan_eval.py`：新增 `--output-suffix` 以避免 trained controls 覆盖主 bridge matrix，并加入 M9-M12 trained random/stable-random/signflip/corrupt control model-name 映射。\n"
            "- 继续修改 `experiments/run_v22_17_kanbefair_dgkan_eval.py`：新增 `DGMLP_FU_JVP_REFRESH` bridge policy，把 audit-derived `jvp_useful_history + refresh_scale=0.05` 思路接入 fixed-step full-loop；controller 需要 positive JVP source-gain、virtual train-loss gate 与 predicted-gain gate 同时通过。\n"
            "- 继续修改 `experiments/run_v22_17_kanbefair_dgkan_eval.py`：将同一 JVP-refresh bridge policy 暴露给 `DGKAN_DFOU_FU_JVP_REFRESH` 诊断路径，并在最终汇总中单独统计 KAN-FU vs KAN baseline 的 no-harm、AUC、basis/readout energy 与 controller overhead；该路径仍不作为 strict basis-native official。\n"
            "- 新增 `experiments/run_v22_17_kan_basis_native_audit.py`：使用 KANbeFair loader、`native_no_dense_basis` PrimitiveKAN、basis-only selector、JVP-refresh controller、非 basis 梯度缩放/清零与周期 refresh，输出 strict KAN native-basis h120 audit、source timeseries、controllability diagnostic、param/FLOPs 与修复 summary。\n"
            "- 修改 `experiments/run_v22_17_kan_basis_native_audit.py`：新增按实际 `--steps` 填充的 `KAN_source_loss_horizon/h20/h120/h3200/h4800` 与 summary 计数字段，避免 h4800 长程证据仍被写成 h120-only。\n"
            "- 修改 `experiments/run_v22_17_common.py`：把 `experiments/run_v22_17_kan_basis_native_audit.py` 纳入 v22.17 source packet，保证 Part F native-basis 修复可审计。\n"
            "- 修改 `dgkan/fu/mlp_adaptive_controller.py`：把 `StableRandomSourceControl` 的匹配顺序放在 `RandomSourceControl` 之前，避免 stable-random control 被误判为普通 random。\n"
            "- 修改 `experiments/run_v22_17_common.py`：把 `dgkan/fu/mlp_adaptive_controller.py` 纳入 v22.17 source packet，保证 controls 修复可审计。\n"
            "- 修改 `experiments/run_v22_17_finalize.py`：把 gradient、split-consensus、JVP-history、refresh scale sweep、basis gradcheck、blocker、复现命令写入执行日志与复盘；D MLP small-subset criteria 按真实 9/9 记录，但 overall official promotion 仍为 0。\n"
            "- 继续修改 `experiments/run_v22_17_finalize.py`：接入 Part E h4800 long bridge 结果，生成 `v22_17_mlp_fu_long_h4800_pass_summary.csv`，并在 gate summary、复盘原始行、结论与执行日志关键复现入口中记录长程修复尝试。\n"
            "- 新增 `experiments/run_v22_17_kanbefair_continual_eval.py`：运行并解析 KANbeFair 原始 `Class_MNIST` exact continual protocol 的 MLP/KAN dry-run，输出 `v22_17_kanbefair_continual_learning_matrix.csv`；该结果只作为 baseline protocol smoke，不作为 DG-KAN/FU forgetting 证明。\n"
            "- 修改 `experiments/run_v22_17_common.py` 与 `experiments/run_v22_17_finalize.py`：把 continual smoke runner 纳入 source packet，并把 G4 从未运行 deferred 改为 baseline dry-run pass + FU continual improvement 未证明。\n"
        )
        f.write("\n## 5. Blockers / deferred / 不作为算法失败的项\n\n")
        f.write(md_table(deferred, ["item", "reason", "algorithm_failure"], max_rows=20))
        f.write("\n## 6. 结论与 insight\n\n")
        f.write(
            "本轮最可靠的结论是：uploaded KANbeFair 源码已经进入可审计 worktree，hardcoded chdir 与 eager text/audio import blocker 已按计划修复，"
            "并且 bridge 能输出 NLL/accuracy/ECE/Brier/tail/AUC-time/source/efficiency 等统一指标。"
            "B/source-usefulness 与 C/actionable-risk 已从占位推进到真实矩阵；basis JVP/VJP gradcheck 也已能区分 current bridge dense-cache 与 native no-dense-basis 证据。"
            "新增 sketched J-space H4800 小子集审计显示 source 可达性本身不再是主要 blocker：gradient source 的 residual 很低、h4800 source_loss 为正且 task NLL 不再退化；新的 blocker 是 controller 没有提供 h4800 source gain over base。"
            "按计划继续尝试的 split-consensus 修复只让 MNIST seed1 在 H4800 通过 source-gain gate；其余 3/4 行仍失败。positive-only filtering 在 H20 探针上变好，但 H4800 上 FMNIST seed0 residual 升到高位并触发 ControllabilityResidualHigh。"
            "进一步的 `jvp_useful_history` 用当前 JVP 选择 task-useful 历史方向，使 strict source-gain gate 从 1/4 提升到 2/4；4/4 行都满足 residual<=0.35、source_loss_h4800>=0、task NLL delta<=0.02、controls_pass_count=0，但 MNIST seed0 与 FMNIST seed1 的 gain-over-base 仍是小负数。"
            "扩展到 MNIST/FMNIST/KMNIST x seeds 0/1/2 后，plan 原文 MLP D criteria 为 9/9 通过，但 strict source-gain-over-base 只有 4/9；这说明 controllability/reachability 已有真实信号，而 controller 相对普通 BP 的可审计增益仍不稳定。"
            "周期 refresh 证明了 source-gain 可以被拉到正值：refresh400/full-step 达到 strict 9/9 但 NLL 全部严重退化；refresh1600/full-step 达到 strict 8/9 但仍有 4/9 NLL breach。"
            "加入 strict virtual-loss gate 与 refresh_scale=0.5 后，修复达到 plan 9/9、strict source-gain 7/9、official_candidate 7/9，且 9/9 task NLL delta <=0.02；refresh_scale=0.25 仍为 7/9，但 NLL 扰动进一步下降。"
            "refresh_scale=0.1 达到 plan 9/9、strict source-gain 8/9、official_candidate 8/9，唯一剩余 blocker 是 FMNIST seed1 的 NoSourceGainVsBase。"
            "继续把同一全局规则缩到 refresh_scale=0.05 后，H4800 3x3 small-subset 达到 plan 9/9、strict source-gain 9/9、official_candidate 9/9，max task NLL delta=0.0014815330505371094，min source-gain=3.885652404278517e-07，没有使用 dataset/seed 特化规则。"
            "`jvp_gain_history` 的保守退回机制降低了扰动但只过 1/4，说明当前 base gradient 本身已接近这类 current-batch JVP score 的上界。"
            "`jvp_weighted_history` 在 H20 探针上没有改善 source gain，因此没有升级到 H4800。"
            "current-batch source-gain line-search 在 H20 上选择 scale=1.0，结果与未 line-search 相同，也没有找到可提升 source-gain-over-base 的 controller scale。"
            "因此，按计划原文的 MLP D criteria，controllability/reachability 的 small-subset 证据已由失败推进到通过；该阶段证据仍是 smoke 级别，且当时 KAN basis 的 h4800 source/efficiency 与 Part E/G full-loop task-general 证明尚未完成。"
            "随后补跑的 Part E/G bridge 3x3 h440 smoke 显示，`DGMLP_FU_ACTIONABLE`、`DGMLP_FU_RELEASE`、`DGMLP_FU_SPLIT_ACTIONABLE`、`DGMLP_FU_SPLIT_VIRTUAL` 这些单一 variant 在 NLL/accuracy/tail metric no-harm 上均达到 9/9；"
            "但 improvement gate 仍未通过，最好也只有 NLL<-0.01 为 0/9、AUC<0 为 6/9、accuracy>=0 为 9/9 的分离信号。"
            "进一步补跑 M9-M12 trained controls 后，没有任何 control 达到 improvement candidate gate；不过 signflip control 存在 NLL<-0.01 的 3/9 与 AUC<0 的 7/9 部分信号，random/stable-random 也有 AUC<0 的 6/9，说明 controls 不能简单写成全面失败。"
            "把 scale0.05 JVP-refresh controller 接入 bridge full-loop 后，`DGMLP_FU_JVP_REFRESH` 在 MNIST/FMNIST/KMNIST x seeds 0/1/2 的 h440 smoke 上达到 NLL/accuracy/tail no-harm 9/9，max NLL delta=0.0011768341064453125，max tail q99 delta=0.00629425048828125；但 improvement gate 仍未通过，NLL<-0.01 为 0/9、AUC<0 为 3/9、accuracy>=0 为 9/9。"
            "随后按 Part E 修复方向补跑的 h4800 long bridge（MNIST/FMNIST/KMNIST x seeds 0/1/2，train/test=512/256）没有打开 official improvement：`DGMLP_FU_ACTIONABLE`、`DGMLP_FU_SPLIT_ACTIONABLE`、`DGMLP_FU_SPLIT_VIRTUAL` 维持 NLL/accuracy/tail no-harm 9/9，但 NLL<-0.01 仍为 0/9；`DGMLP_FU_SPLIT_RELEASE` 有 NLL<-0.01 的 3/9、AUC<0 的 5/9、accuracy>=0 的 8/9，max NLL delta=0.021704673767089844，max tail q99 delta=0.23640060424804688，因此既不是 no-harm candidate，也不是 improvement candidate。"
            "这说明长程训练把 release 路径的局部收益信号放大了一点，但同时带来了 tail/task 风险；Part E blocker 由短程不足细化为 `long-horizon MLP+FU still no stable improvement under no-harm`。"
            "G4 continual 方面，补跑 KANbeFair 原始 Class_MNIST exact protocol dry-run 后，MLP 与 KAN 都 exit 0，矩阵记录 MLP average_accuracy=15.4、average_backward_transfer=3.01，KAN average_accuracy=7.77、average_backward_transfer=-2.61；这只把 protocol availability 从 deferred 推进到 baseline smoke pass，并没有接入 DGMLP_FU/DGKAN_FU，所以 continual forgetting reduction official 仍未证明。"
            "同一策略接到 `DGKAN_DFOU_FU_JVP_REFRESH` 后，MNIST/FMNIST/KMNIST x seeds 0/1/2 的 h120 KAN bridge diagnostic 达到 NLL/accuracy/tail no-harm 9/9，AUC<0 为 9/9，max NLL delta=0.00042426586151123047，max tail q99 delta=0.004224300384521484；但这是短程 bridge diagnostic，controller overhead 最高约 0.9699681997184777，且 strict basis-native official 仍为 0。"
            "随后新增 strict native-basis audit 证明问题可以继续细分：完全 basis-only interval20 能让 basis_energy=1.0、source_loss_h120>=0、controls fail，但 3x3 h120 的 NLL vs native KAN 全部严重退化，max NLL delta=0.550281286239624，说明冻结 readout 不是可用修复。"
            "改为非 basis 梯度缩放后，scale0.25 interval20 在 h120 上达到 NLL no-harm 9/9、AUC<0 6/9、source_loss_h120>=0 9/9、controls fail 9/9，但 basis>=0.5 只有 3/9。"
            "scale0.10 interval20 进一步把 basis>=0.5 提升到 9/9，且 NLL no-harm 9/9、source_loss_h120>=0 9/9、controls fail 9/9，但 controller overhead<=0.25 仍是 0/9。"
            "scale0.10 interval40 维持 basis>=0.5 9/9、NLL no-harm 9/9、AUC<0 6/9、source_loss_h120>=0 9/9、controls fail 9/9，并把 controller overhead<=0.25 提到 9/9，max overhead=0.18285700399928342。"
            "把效率参照改成 matched-param MLP hidden48 后，KAN/MLP 参数比为 0.9429928741092637，但 FLOPs 比仍为 3.7719714964370548；full-loop ratio<=3 从 0/9 变为 1/9，max full-loop ratio=4.909623457201262，h3200/h4800 仍缺失。"
            "再改成 matched-FLOPs MLP hidden160 后，FLOPs 比为 0.9987421383647799，但 full-loop ratio<=3 仍为 1/9，max full-loop ratio=4.441753301931999；这说明当前效率 blocker 不是单纯由 MLP hidden16 参照过小造成。"
            "继续按效率修复方向测试 D-CHE native-basis 后，h16 D-CHE matched-FLOPs MLP160 在 h120 上把 max full-loop ratio 从 D-FOU 的 4.44 降到 interval40 的 3.67、interval120 的 3.15，但 full-loop<=3 仍只有 1/9 或 2/9；把 strict FC-PureKAN hidden 降到 h8 并用 matched-FLOPs MLP80 参照后，h120 full-loop<=3 提升到 3/9，max ratio=3.09106846637376。"
            "最终 h8 D-CHE + interval4800 的 h4800 长程验证达到 basis>=0.5 9/9、source_loss_h4800>=0 9/9、full-loop<=3 9/9、overhead<=0.25 9/9、controls fail 9/9、NLL no-harm vs KAN 9/9，max full-loop ratio=2.952011975627768，max controller overhead=0.0015814291144632278。"
            "这把 Part F 的 blocker 从 `long-horizon source/efficiency missing` 推进到更具体的 `task/AUC/MLP+FU comparison not proven`：AUC improve vs KAN 只有 1/9，且 KAN+FU vs MLP+FU、MLP+FU general improvement、KANbeFair expanded/continual proof 仍未完成。"
            "因此 Part F 的新 insight 是：basis carrier、long-horizon source、controller overhead 和 full-loop efficiency 可以在 D-CHE h8 native path 上同时达成 smoke/exploration 级别；但 functional update 还没有带来稳定 task/AUC improvement，也不能证明 strict FC-PureKAN + FU 的官方 superiority。"
            "所以当前只能说 MLP+FU 任务 no-harm metric smoke 有真实支持，KAN native-basis source/efficiency exploration 已打开；但不能证明 MLP+FU general improvement，也不能证明 strict FC-PureKAN basis-native official superiority。"
            "下一步应优先攻 MLP+FU improvement 与 KAN-vs-MLP+FU/AUC-time 差距，再扩 KANbeFair expanded/continual proof。\n"
        )
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n# v22.17 执行收尾\n\n生成时间：{now_sg()}\n\n")
        f.write("关键复现入口：\n\n")
        f.write("```bash\n")
        f.write(f"{PYTHON} experiments/run_v22_17_kanbefair_unpack_patch.py --fresh\n")
        f.write(f"{PYTHON} experiments/run_v22_17_kanbefair_env_smoke.py\n")
        f.write(f"{PYTHON} experiments/run_v22_17_kanbefair_baseline_reproduce.py --dataset MNIST\n")
        f.write(f"CUDA_VISIBLE_DEVICES=1 {PYTHON} experiments/run_v22_17_kanbefair_dgkan_eval.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGMLP,DGMLP_FU,DGMLP_FU_GATED,DGMLP_FU_ACTIONABLE,DGMLP_FU_RELEASE,DGMLP_FU_SPLIT_ACTIONABLE,DGMLP_FU_SPLIT_RELEASE,DGMLP_FU_SPLIT_VIRTUAL,DGKAN_DFOU,DGKAN_DFOU_FU_SPLIT_VIRTUAL --seeds 0,1,2 --train-size 1024 --test-size 512 --batch-size 128 --hidden 16 --steps 440 --log-interval 110 --experiment-tag v22_17_partE_bridge_smoke_3x3_h440\n")
        f.write(f"CUDA_VISIBLE_DEVICES=0 {PYTHON} experiments/run_v22_17_kanbefair_dgkan_eval.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGMLP,DGMLP_FU_CONTROL_RANDOM,DGMLP_FU_CONTROL_STABLE_RANDOM,DGMLP_FU_CONTROL_SIGNFLIP,DGMLP_FU_CONTROL_CORRUPT --seeds 0,1,2 --train-size 1024 --test-size 512 --batch-size 128 --hidden 16 --steps 440 --log-interval 110 --experiment-tag v22_17_partE_trained_controls_3x3_h440 --output-suffix trained_controls_3x3_h440\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_kanbefair_dgkan_eval.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGMLP,DGMLP_FU_JVP_REFRESH --seeds 0,1,2 --train-size 1024 --test-size 512 --batch-size 128 --hidden 16 --steps 440 --log-interval 110 --experiment-tag v22_17_partE_jvp_refresh_bridge_3x3_h440 --output-suffix jvp_refresh_bridge_3x3_h440 --jvp-refresh-min-history 8 --jvp-refresh-history-window 32 --jvp-refresh-dim 8 --jvp-refresh-scale 0.05 --jvp-refresh-ratio-cap 0.05\n")
        f.write(f"CUDA_VISIBLE_DEVICES=1 {PYTHON} experiments/run_v22_17_kanbefair_dgkan_eval.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGMLP,DGMLP_FU_ACTIONABLE,DGMLP_FU_SPLIT_ACTIONABLE,DGMLP_FU_SPLIT_RELEASE,DGMLP_FU_SPLIT_VIRTUAL --seeds 0,1,2 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --steps 4800 --log-interval 1600 --experiment-tag v22_17_partE_mlp_fu_long_3x3_h4800 --output-suffix mlp_fu_long_3x3_h4800\n")
        f.write(f"CUDA_VISIBLE_DEVICES=3 {PYTHON} experiments/run_v22_17_kanbefair_continual_eval.py --models MLP,KAN --seed 22017 --epochs 1 --batch-size 128 --test-batch-size 256 --lr 0.001 --dry-run\n")
        f.write(f"CUDA_VISIBLE_DEVICES=3 {PYTHON} experiments/run_v22_17_kanbefair_dgkan_eval.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGKAN_DFOU,DGKAN_DFOU_FU_JVP_REFRESH --seeds 0,1,2 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --steps 120 --log-interval 40 --experiment-tag v22_17_kan_jvp_refresh_bridge_3x3_h120 --output-suffix kan_jvp_refresh_bridge_3x3_h120 --jvp-refresh-min-history 8 --jvp-refresh-history-window 32 --jvp-refresh-dim 8 --jvp-refresh-scale 0.05 --jvp-refresh-ratio-cap 0.05\n")
        f.write(f"CUDA_VISIBLE_DEVICES=3 {PYTHON} experiments/run_v22_17_kan_basis_native_audit.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGKAN_DFOU --seeds 0,1,2 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --mlp-hidden 48 --steps 120 --log-interval 40 --experiment-tag v22_17_kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120 --output-suffix kan_basis_native_scale010_interval40_matchedmlp48_3x3_h120 --jvp-refresh-dim 4 --jvp-refresh-min-history 8 --jvp-refresh-history-window 32 --jvp-refresh-scale 0.05 --jvp-refresh-ratio-cap 0.05 --jvp-refresh-interval 40 --fu-nonbasis-grad-scale 0.1\n")
        f.write(f"CUDA_VISIBLE_DEVICES=0 {PYTHON} experiments/run_v22_17_kan_basis_native_audit.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGKAN_DCHE --seeds 0,1,2 --train-size 512 --test-size 256 --batch-size 128 --hidden 8 --mlp-hidden 80 --steps 4800 --log-interval 1600 --experiment-tag v22_17_kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800 --output-suffix kan_basis_native_dche_h8_scale010_interval4800_matchedflops80_3x3_h4800 --jvp-refresh-dim 4 --jvp-refresh-min-history 8 --jvp-refresh-history-window 32 --jvp-refresh-scale 0.05 --jvp-refresh-ratio-cap 0.05 --jvp-refresh-interval 4800 --fu-nonbasis-grad-scale 0.1\n")
        f.write(f"CUDA_VISIBLE_DEVICES=3 {PYTHON} experiments/run_v22_17_basis_jvp_vjp_gradcheck.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGKAN_DFOU,DGKAN_DCHE --implementation-modes bridge_dense_cache,native_no_dense_basis --seeds 0 --train-size 256 --test-size 128 --batch-size 64 --hidden 16 --warmup-steps 3 --selectors basis,readout,all --directions 3 --timing-repeats 3 --fd-eps 0.01\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST --models DGMLP --seeds 0,1 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --experiment-tag v22_17_sketched_jspace_h4800_mlp_s01_replace_update_statefix_gaincheck --output-name v22_17_controllability_sketch_audit.csv\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST --models DGMLP --seeds 0,1 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --source-candidate split_consensus --controller-compose add_to_base --jspace-target base_effect --experiment-tag v22_17_split_consensus_add_base_h4800_s01 --output-name v22_17_controllability_sketch_audit_split_consensus.csv\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST --models DGMLP --seeds 0,1 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --source-candidate split_consensus_positive --controller-compose add_to_base --jspace-target base_effect --experiment-tag v22_17_split_consensus_positive_add_base_h4800_s01 --output-name v22_17_controllability_sketch_audit_split_consensus_positive.csv\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST --models DGMLP --seeds 0,1 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --source-candidate jvp_useful_history --controller-compose add_to_base --jspace-target base_effect --experiment-tag v22_17_jvp_useful_history_add_base_h4800_s01 --output-name v22_17_controllability_sketch_audit_jvp_useful_history.csv\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGMLP --seeds 0,1,2 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --source-candidate jvp_useful_history --controller-compose add_to_base --jspace-target base_effect --experiment-tag v22_17_jvp_useful_history_add_base_h4800_3x3 --output-name v22_17_controllability_sketch_audit_jvp_useful_history_3x3.csv\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST,KMNIST --models DGMLP --seeds 0,1,2 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --source-candidate jvp_useful_history --controller-compose add_to_base --jspace-target base_effect --controller-refresh-interval 1600 --controller-refresh-max-count 4 --refresh-scale 0.05 --virtual-loss-gate --virtual-loss-tolerance 0.00001 --experiment-tag v22_17_jvp_useful_history_refresh1600_vgate_strict_scale005_h4800_3x3 --output-name v22_17_controllability_sketch_audit_jvp_useful_history_refresh1600_vgate_strict_scale005_3x3.csv\n")
        f.write(f"CUDA_VISIBLE_DEVICES=2 {PYTHON} experiments/run_v22_17_controllability_sketch_audit.py --device cuda:0 --datasets MNIST,FMNIST --models DGMLP --seeds 0,1 --train-size 512 --test-size 256 --batch-size 128 --hidden 16 --collect-steps 80 --history-window 32 --dim 8 --horizons 3200,4800 --selector all --source-candidate jvp_gain_history --controller-compose add_to_base --jspace-target base_effect --experiment-tag v22_17_jvp_gain_history_add_base_h4800_s01 --output-name v22_17_controllability_sketch_audit_jvp_gain_history.csv\n")
        f.write(f"{PYTHON} experiments/run_v22_17_mechanism_smoke.py --horizons 50,100,200,400 --merge-window 10 --tau-u-margin 0.0 --tau-l 1e-8 --tau-i 1e-8\n")
        f.write(f"{PYTHON} experiments/run_v22_17_finalize.py\n")
        f.write("```\n\n")
        f.write("所有核心 artifact 位于 `results/v22_17/`；最终 route 见 `results/v22_17/v22_17_final_route.json`。\n")


def main() -> None:
    _ = parser().parse_args()
    ensure_out()
    _write_deferred()
    gates, route = _gate_summary()
    summary = _summary_stats()
    _write_figures(route, summary)
    write_rows(OUT_ROOT / "v22_17_artifact_index.csv", artifact_index())
    packet = _build_packet()
    _compile_packet(packet)
    write_rows(OUT_ROOT / "v22_17_artifact_index.csv", artifact_index())
    _append_docs(route, gates, summary)
    append_exec(
        f"{sys.executable} experiments/run_v22_17_finalize.py",
        task_id="finalize",
        status="pass",
        exit_code=0,
        files="results/v22_17/v22_17_final_route.json, docs/DG-KAN_v22.17_TaskUsefulControllability_KANbeFair_实验结果复盘.md",
        note=f"route={route}; official_promotion=0",
    )


if __name__ == "__main__":
    main()
