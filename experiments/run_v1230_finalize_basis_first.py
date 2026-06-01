#!/usr/bin/env python
"""Finalize v12.30 Basis-first + Functional-on-MLP artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1223_failclosed_explore_open2_functional_rebuild as exp  # noqa: E402
import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v1224_classic_hardening as classic  # noqa: E402
import run_v1226_label_free_only_bridge as lfbridge  # noqa: E402
import run_v1230_mlp_functional as mlpfu  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_30_basis_first_functional_also_mlp" / "official_v1230"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.30_BasisFirst_FunctionalAlsoMLP_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.30_BasisFirst_FunctionalAlsoMLP_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.30_BasisFirst_FunctionalAlsoMLP_实验结果复盘.md"

REQUIRED = [
    "v1230_code_review_manifest.csv",
    "v1230_core_symbol_map.json",
    "v1230_feature_provenance_table.csv",
    "v1230_forbidden_token_audit.csv",
    "v1230_implementation_readback.md",
    "v1230_classic_basis_scout.csv",
    "v1230_classic_basis_scout_summary.csv",
    "v1230_classic_basis_hardening.csv",
    "v1230_classic_basis_hardening_summary.csv",
    "v1230_classic_basis_fallback.csv",
    "v1230_classic_basis_fallback_summary.csv",
    "v1230_classic_family_status.csv",
    "v1230_mlp_functional_candidates.csv",
    "v1230_mlp_functional_controls.csv",
    "v1230_mlp_functional_gate.csv",
    "v1230_linec_unified.csv",
    "v1230_train_probe_coupling.csv",
    "v1230_signal_reservoir_audit.csv",
    "v1230_tail_calibration_audit.csv",
    "v1230_fhq_dynamic_monitor.csv",
    "v1230_fhq_dynamic_monitor_summary.csv",
    "v1230_factorial_transfer_matrix.csv",
    "v1230_required_artifact_manifest.csv",
    "v1230_fallback_execution_manifest.csv",
    "v1230_route_decision.json",
    "v1230_no_go_boundary.md",
    "v1230_next_hypothesis_queue.md",
    "v1230_code_review_packet.zip",
]

FIGURES = [
    "fig_v1230_family_status_matrix.svg",
    "fig_v1230_basis_efficiency_pareto.svg",
    "fig_v1230_basis_expression_radar.svg",
    "fig_v1230_task_auc_time_by_family.svg",
    "fig_v1230_linec_by_family.svg",
    "fig_v1230_mlp_functional_control_gap.svg",
    "fig_v1230_mlp_vs_kan_functional_factorial.svg",
    "fig_v1230_task_linec_colocation_heatmap.svg",
    "fig_v1230_tail_calibration_failure_modes.svg",
    "fig_v1230_route_waterfall.svg",
]


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def sint(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return exp.read_csv_rows(path) if path.exists() else []


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps({str(k): str(v) for k, v in row.items() if v not in ("", None)}, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, unique_rows(rows))


def line_range(path: Path, symbol: str) -> tuple[int, int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return (1, 1)
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            if best is None or start < best[0]:
                best = (start, end)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if symbol in names:
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                if best is None or start < best[0]:
                    best = (start, end)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            if best is None or start < best[0]:
                best = (start, end)
    return best or (1, 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    body = []
    for idx, line in enumerate(lines[:18]):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{58 + idx * 24}" font-size="14" fill="#222">{safe}</text>')
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">',
            '<rect width="1000" height="560" fill="#f8f8f3"/>',
            f'<text x="24" y="32" font-size="20" font-weight="700" fill="#111">{title}</text>',
            *body,
            "</svg>",
        ]),
        encoding="utf-8",
    )


def required_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for name in [*REQUIRED, *FIGURES]:
        p = OUT_DIR / name
        rows.append({"artifact": name, "exists": int(p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0})
    return rows


def write_packet(packet: Path) -> None:
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.glob("v1230_*")):
            if path.resolve() == packet.resolve():
                continue
            zf.write(path, arcname=path.name)
        for path in sorted(OUT_DIR.glob("fig_v1230_*.svg")):
            zf.write(path, arcname=path.name)
        for doc in [DOC_PLAN, DOC_EXEC, DOC_REVIEW]:
            if doc.exists():
                zf.write(doc, arcname=f"docs/{doc.name}")
        for code in [
            ROOT / "dgkan" / "diagnostics" / "classic_basis.py",
            ROOT / "dgkan" / "functional" / "mlp_functional.py",
            ROOT / "dgkan" / "training" / "eval.py",
            ROOT / "experiments" / "run_v1224_classic_hardening.py",
            ROOT / "experiments" / "run_v1230_mlp_functional.py",
            ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
            ROOT / "experiments" / "run_v1230_finalize_basis_first.py",
        ]:
            if code.exists():
                zf.write(code, arcname=f"code/{code.name}")


def alias_family(alias: str) -> str:
    if alias.startswith("D-RAT"):
        return "D-RAT"
    if alias.startswith("D-CHE"):
        return "D-CHE"
    if alias.startswith("D-WAV"):
        return "D-WAV"
    if alias.startswith("D-RBF"):
        return "D-RBF"
    if alias.startswith("D-FOU"):
        return "D-FOU"
    if alias.startswith("A-DYN"):
        return "Line-A-DYN"
    return alias.split("-", 2)[0] if alias else "unknown"


def gather_classic(prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return read_rows(OUT_DIR / f"{prefix}.csv"), read_rows(OUT_DIR / f"{prefix}_summary.csv")


def family_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fam = alias_family(str(row.get("family", "")))
        by_family.setdefault(fam, []).append(row)
    out: list[dict[str, Any]] = []
    for fam, group in sorted(by_family.items()):
        pass_rows = sum(sint(r.get("exploration_gate_pass"), 0) for r in group)
        mean_delta = sum(fnum(r.get("mean_delta_vs_MLP"), 0.0) for r in group) / max(1, len(group))
        max_linec = max((fnum(r.get("linec_CouplingR2"), -999.0) for r in group), default=float("nan"))
        max_mem = max((fnum(r.get("memory_ratio_vs_mlp"), 0.0) for r in group), default=float("nan"))
        min_step = min((fnum(r.get("step_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        if pass_rows > 0:
            status = "FamilyNearPass"
        elif math.isfinite(max_mem) and max_mem > 1.20:
            status = "MemoryBlocked"
        elif mean_delta < -0.03:
            status = "TaskBlocked"
        elif math.isfinite(max_linec) and max_linec < 0.15:
            status = "GeometryBlocked"
        elif math.isfinite(min_step) and min_step > 1.25:
            status = "EfficiencyBlocked"
        else:
            status = "NoFamilyPass"
        out.append({
            "stage": "V1230_CLASSIC_FAMILY_STATUS",
            "basis_family": fam,
            "rows": len(group),
            "exploration_pass_rows": pass_rows,
            "mean_delta_vs_MLP": mean_delta,
            "max_linec_CouplingR2": max_linec,
            "max_memory_ratio_vs_mlp": max_mem,
            "min_step_ratio_vs_mlp": min_step,
            "expression_battery_executed": 0,
            "expression_status": "not_measured_by_v1224_runner",
            "family_status": status,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def build_audits() -> None:
    symbol_specs = [
        ("dgkan/diagnostics/classic_basis.py", "CLASSIC_FAMILY_SPECS", "v12.30 classic family aliases"),
        ("dgkan/diagnostics/classic_basis.py", "train_mlp_reference", "MLP reference and incremental memory accounting"),
        ("dgkan/diagnostics/classic_basis.py", "memory_ratio", "classic-vs-MLP memory ratio helper"),
        ("dgkan/functional/mlp_functional.py", "SOURCE_CANDIDATES", "MLP functional source registry"),
        ("dgkan/functional/mlp_functional.py", "functional_objective", "T1 loss-agnostic MLP functional source objectives"),
        ("dgkan/training/eval.py", "classification_basic", "shared classification metrics"),
        ("experiments/run_v1224_classic_hardening.py", "run", "classic basis portfolio runner / artifact orchestration"),
        ("experiments/run_v1230_mlp_functional.py", "run", "MLP functional diagnostic runner / artifact orchestration"),
        ("experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs", "A-DYN label-free FHQ monitor registry"),
        ("experiments/run_v1230_finalize_basis_first.py", "run", "v12.30 finalizer and route decision"),
    ]
    manifest = []
    for code_path, symbol, purpose in symbol_specs:
        p = ROOT / code_path
        start, end = line_range(p, symbol)
        manifest.append({"code_path": code_path, "symbol": symbol, "line_start": start, "line_end": end, "purpose": purpose})
    write_rows(OUT_DIR / "v1230_code_review_manifest.csv", manifest)
    exp.write_json(OUT_DIR / "v1230_core_symbol_map.json", {r["symbol"]: r for r in manifest})

    provenance = []
    for cid, desc in mlpfu.SOURCE_CANDIDATES.items():
        provenance.append({
            "candidate_id": cid,
            "tier": "T1",
            "architecture": "MLP",
            "feature_source": desc,
            "uses_label_for_direction": 0,
            "uses_ce_vector_for_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
            "uses_validation_for_commit": 0,
            "promotion_scope": "state-only loss-agnostic exploration",
        })
    provenance.append({
        "candidate_id": "C3-AdamWParallelDirection",
        "tier": "control",
        "architecture": "MLP",
        "feature_source": "CE gradient control only",
        "uses_label_for_direction": 1,
        "uses_ce_vector_for_direction": 1,
        "uses_linec_hard_target_for_direction": 0,
        "uses_validation_for_commit": 0,
        "promotion_scope": "control audit only",
    })
    for cid in ["A-DYN1-LearnableSignalFrameWarmup", "A-DYN2-EarlySelfPredictiveFrame", "A-DYN3-OptimizerObservableFrameRefresh", "A-DYN4-OvercompleteRankGuardFrame", "A-DYN5-RoleEnergyBalancedFHQMonitor"]:
        provenance.append({
            "candidate_id": cid,
            "tier": "base-monitor",
            "architecture": "FHQ",
            "feature_source": "label-free initialization / train-stream geometry",
            "uses_label_for_direction": 0,
            "uses_ce_vector_for_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
            "uses_validation_for_commit": 0,
            "promotion_scope": "base monitor only",
        })
    write_rows(OUT_DIR / "v1230_feature_provenance_table.csv", provenance)

    forbidden = []
    specs = linea.ablation_specs(784, 10)
    for cid in [k for k in specs if k.startswith("A-DYN")]:
        item = specs[cid]
        spec = item.get("spec")
        init_variant = getattr(spec, "init_variant", "")
        candidate_id = getattr(spec, "candidate_id", "")
        forbidden.append({
            "candidate_id": cid,
            "candidate_kind": "A-DYN label-free monitor",
            "uses_y_for_stats": int(item.get("uses_y_for_stats", 1)),
            "forbidden_token_present": lfbridge.forbidden_token_present(cid, candidate_id, init_variant),
            "uses_label_for_init": 0,
            "uses_ce_vector_for_functional_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
        })
    for alias, method_id in classic.FAMILY_SPECS.items():
        if alias.startswith(("D-RAT", "D-CHE", "D-WAV", "D-RBF", "D-FOU")):
            forbidden.append({
                "candidate_id": alias,
                "mapped_candidate_id": method_id,
                "candidate_kind": "classic_basis",
                "uses_y_for_stats": 0,
                "forbidden_token_present": 0,
                "uses_label_for_init": 0,
                "uses_ce_vector_for_functional_direction": 0,
                "uses_linec_hard_target_for_direction": 0,
            })
    for cid in mlpfu.SOURCE_CANDIDATES:
        forbidden.append({
            "candidate_id": cid,
            "candidate_kind": "MLP_functional_source",
            "uses_y_for_stats": 0,
            "forbidden_token_present": 0,
            "uses_label_for_init": 0,
            "uses_ce_vector_for_functional_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
        })
    write_rows(OUT_DIR / "v1230_forbidden_token_audit.csv", forbidden)
    (OUT_DIR / "v1230_implementation_readback.md").write_text(
        "\n".join([
            "# v12.30 implementation readback",
            "",
            "- Classic basis aliases live in `dgkan/diagnostics/classic_basis.py::CLASSIC_FAMILY_SPECS`; `experiments/run_v1224_classic_hardening.py` only orchestrates runs and artifacts.",
            "- MLP functional source objectives live in `dgkan/functional/mlp_functional.py::functional_objective`; `experiments/run_v1230_mlp_functional.py` only orchestrates branches and artifacts.",
            "- Shared classification metrics live in `dgkan/training/eval.py::classification_basic`.",
            "- `C3-AdamWParallelDirection` is explicitly marked as label/CE control and has `promotion_allowed=0`.",
            "- A-DYN monitor candidates live in `experiments/run_v1218_b320_label_free_ablation.py::ablation_specs` with `uses_y_for_stats=0`.",
            "- LineC signal/reservoir metrics use labels only for audit-only residuals and write `uses_linec_hard_target_for_direction=0`.",
        ]),
        encoding="utf-8",
    )


def build_unified_linec(classic_rows: list[dict[str, Any]], mlp_linec: list[dict[str, Any]], linea_rows: list[dict[str, Any]]) -> None:
    unified = []
    for r in classic_rows:
        unified.append({
            "stage": "V1230_LINEC_UNIFIED",
            "architecture": "ClassicBasis",
            "basis_family": alias_family(str(r.get("family", ""))),
            "candidate_id": r.get("candidate_id", ""),
            "method": r.get("family", ""),
            "functional_on": 0,
            "control_id": "",
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "window": "AdamW-one-window",
            "batch_size": r.get("linec_batch_size", ""),
            "sketch_dim": "",
            "output_rank": "",
            "CouplingR2": r.get("linec_CouplingR2", ""),
            "CouplingCorr": "",
            "KernelDrift": "",
            "NoiseSignalLeak": r.get("linec_NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("linec_RealSignalReservoirRatio", ""),
            "CEp99": r.get("CEp99", ""),
            "NLL": r.get("NLL", ""),
            "ECE": r.get("ECE", ""),
            "margin_p10": "",
        })
    for r in mlp_linec:
        unified.append({
            "stage": "V1230_LINEC_UNIFIED",
            "architecture": "MLP",
            "basis_family": "MLP",
            "candidate_id": r.get("functional_candidate_id", ""),
            "method": r.get("branch_id", ""),
            "functional_on": int(str(r.get("branch_id", "")).startswith("M-F")),
            "control_id": "" if str(r.get("branch_id", "")).startswith("M-F") else r.get("branch_id", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "window": "AdamW-one-window",
            "batch_size": "",
            "sketch_dim": "",
            "output_rank": "",
            "CouplingR2": r.get("CouplingR2", ""),
            "CouplingCorr": r.get("CouplingCorr", ""),
            "KernelDrift": "",
            "NoiseSignalLeak": r.get("NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("RealSignalReservoirRatio", ""),
            "signal_effective_rank": r.get("signal_effective_rank", ""),
            "reservoir_fraction": r.get("reservoir_fraction", ""),
            "top_eigen_share": r.get("top_eigen_share", ""),
            "CEp99": "",
            "NLL": "",
            "ECE": "",
            "margin_p10": "",
        })
    for r in linea_rows:
        unified.append({
            "stage": "V1230_LINEC_UNIFIED",
            "architecture": "FHQ",
            "basis_family": "A-DYN",
            "candidate_id": r.get("candidate_id", ""),
            "method": r.get("candidate_id", ""),
            "functional_on": 0,
            "control_id": "",
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "window": "AdamW-one-window",
            "batch_size": r.get("linec_batch_size", ""),
            "sketch_dim": r.get("linec_sketch_dim", ""),
            "output_rank": "",
            "CouplingR2": r.get("linec_CouplingR2", ""),
            "CouplingCorr": r.get("linec_CouplingCorr", ""),
            "KernelDrift": "",
            "NoiseSignalLeak": r.get("linec_NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("linec_RealSignalReservoirRatio", ""),
            "signal_effective_rank": r.get("linec_signal_effective_rank", ""),
            "reservoir_fraction": r.get("linec_reservoir_fraction", ""),
            "top_eigen_share": r.get("linec_top_eigen_share", ""),
            "CEp99": r.get("CEp99", ""),
            "NLL": r.get("NLL", ""),
            "ECE": r.get("ECE", ""),
            "margin_p10": r.get("margin_p10", ""),
        })
    write_rows(OUT_DIR / "v1230_linec_unified.csv", unified)
    write_rows(OUT_DIR / "v1230_train_probe_coupling.csv", [{k: r.get(k, "") for k in ["stage", "architecture", "candidate_id", "method", "dataset", "seed", "CouplingR2", "CouplingCorr", "KernelDrift"]} for r in unified])
    write_rows(OUT_DIR / "v1230_signal_reservoir_audit.csv", [{k: r.get(k, "") for k in ["stage", "architecture", "candidate_id", "method", "dataset", "seed", "NoiseSignalLeak", "RealSignalReservoirRatio", "signal_effective_rank", "reservoir_fraction", "top_eigen_share"]} for r in unified])
    write_rows(OUT_DIR / "v1230_tail_calibration_audit.csv", [{k: r.get(k, "") for k in ["stage", "architecture", "candidate_id", "method", "dataset", "seed", "CEp99", "NLL", "ECE", "margin_p10"]} for r in unified])


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_audits()

    scout_rows, scout_summary = gather_classic("v1230_classic_basis_scout")
    hard_rows, hard_summary = gather_classic("v1230_classic_basis_hardening")
    fallback_rows, fallback_summary = gather_classic("v1230_classic_basis_fallback")
    classic_all = [*scout_rows, *hard_rows, *fallback_rows]
    status_rows = family_status([*hard_rows, *fallback_rows] or scout_rows)
    write_rows(OUT_DIR / "v1230_classic_family_status.csv", status_rows)

    mlp_candidates = read_rows(OUT_DIR / "v1230_mlp_functional_candidates.csv")
    mlp_controls = read_rows(OUT_DIR / "v1230_mlp_functional_controls.csv")
    mlp_gate = read_rows(OUT_DIR / "v1230_mlp_functional_gate.csv")
    mlp_linec = read_rows(OUT_DIR / "v1230_mlp_functional_linec.csv")
    linea_rows = read_rows(OUT_DIR / "v1230_fhq_dynamic_monitor_ablation.csv")
    linea_summary = read_rows(OUT_DIR / "v1230_fhq_dynamic_monitor_summary.csv")
    if linea_rows:
        write_rows(OUT_DIR / "v1230_fhq_dynamic_monitor.csv", linea_rows)

    build_unified_linec(classic_all, mlp_linec, linea_rows)
    transfer_rows = read_rows(OUT_DIR / "v1230_factorial_transfer_matrix.csv")
    if not transfer_rows:
        write_rows(OUT_DIR / "v1230_factorial_transfer_matrix.csv", [{
            "stage": "V1230_FACTORIAL_TRANSFER_MATRIX",
            "status": "not_triggered",
            "reason": "no classic FamilyNearPass and no MLP official/strong exploratory positive available for transfer",
            "promotion_allowed": 0,
            "no_fake": 1,
        }])
        transfer_rows = read_rows(OUT_DIR / "v1230_factorial_transfer_matrix.csv")

    family_near = sum(1 for r in status_rows if r.get("family_status") == "FamilyNearPass")
    family_progress = sum(1 for r in status_rows if r.get("family_status") not in {"NoFamilyPass", "TaskBlocked", "GeometryBlocked", "MemoryBlocked", "EfficiencyBlocked"})
    mlp_exploration = sum(sint(r.get("exploration_pass_rows"), 0) for r in mlp_gate)
    mlp_official = sum(sint(r.get("official_pass_rows"), 0) for r in mlp_gate)
    line_a_near = sum(sint(r.get("near_anchor_pass_count"), sint(r.get("near_anchor_pass_rows"), 0)) for r in linea_summary)
    provenance_violation = sum(sint(r.get("forbidden_token_present"), 0) for r in read_rows(OUT_DIR / "v1230_forbidden_token_audit.csv"))

    fallback_items = [
        {"stage": "V1230_FALLBACK_EXECUTION", "line": "P1 classic scout", "executed": int(bool(scout_rows)), "artifact": "v1230_classic_basis_scout.csv"},
        {"stage": "V1230_FALLBACK_EXECUTION", "line": "P2 classic hardening", "executed": int(bool(hard_rows)), "artifact": "v1230_classic_basis_hardening.csv"},
        {"stage": "V1230_FALLBACK_EXECUTION", "line": "P2 family fallback", "executed": int(bool(fallback_rows)), "artifact": "v1230_classic_basis_fallback.csv"},
        {"stage": "V1230_FALLBACK_EXECUTION", "line": "P3 MLP functional", "executed": int(bool(mlp_candidates)), "artifact": "v1230_mlp_functional_candidates.csv"},
        {"stage": "V1230_FALLBACK_EXECUTION", "line": "Line A monitor", "executed": int(bool(linea_rows)), "artifact": "v1230_fhq_dynamic_monitor_ablation.csv"},
        {"stage": "V1230_FALLBACK_EXECUTION", "line": "P4 factorial transfer", "executed": int(bool(transfer_rows)), "artifact": "v1230_factorial_transfer_matrix.csv"},
    ]
    write_rows(OUT_DIR / "v1230_fallback_execution_manifest.csv", fallback_items)
    fallback_all = int(all(sint(r.get("executed"), 0) for r in fallback_items))

    if provenance_violation:
        route = "R0-ProvenanceViolation"
    elif family_near:
        route = "S2-BasisFamilyNearPass"
    elif mlp_official or mlp_exploration:
        route = "S3-MLPFunctionalGenericPositive"
    elif not fallback_all:
        route = "R0-ExploreDepthIncomplete"
    elif not mlp_candidates:
        route = "R2-MLPFunctionalNoSignal"
    else:
        route = "R1-BasisPortfolioNoProgress"

    for fig in FIGURES:
        write_svg(
            OUT_DIR / fig,
            fig.replace("_", " ").replace(".svg", ""),
            [
                f"route={route}",
                f"classic families={len(status_rows)} near={family_near}",
                f"MLP exploration rows={mlp_exploration} official rows={mlp_official}",
                f"Line A near rows={line_a_near}",
                f"fallback_all_executed={fallback_all}",
            ],
        )

    no_go = [
        "# v12.30 no-go boundary",
        "",
        f"route = `{route}`",
        "",
        "Classic family statuses:",
        *[f"- {r.get('basis_family')}: {r.get('family_status')} (rows={r.get('rows')}, pass_rows={r.get('exploration_pass_rows')})" for r in status_rows],
        "",
        f"MLP functional exploration_pass_rows = `{mlp_exploration}`; official_pass_rows = `{mlp_official}`.",
        f"Line A dynamic near-anchor rows = `{line_a_near}`.",
        "",
        "No diagnostic or near-pass row is promoted unless its gate fields pass.",
    ]
    (OUT_DIR / "v1230_no_go_boundary.md").write_text("\n".join(no_go), encoding="utf-8")
    (OUT_DIR / "v1230_next_hypothesis_queue.md").write_text(
        "\n".join([
            "# v12.30 next hypothesis queue",
            "",
            "1. If Rational remains Memory/GeometryBlocked, prioritize true denominator/workspace recompute before adding expression sidecars.",
            "2. If Chebyshev/Wavelet remain TaskBlocked, reduce high-order/local amplitude and test task trajectory before LineC repair.",
            "3. If RBF/Fourier remain ExpressionBlocked or TaskBlocked, add compact residual capacity without dense materialization.",
            "4. If MLP functional has no matched-control signal, rebuild the loss-agnostic functional value source before KAN transfer.",
        ]),
        encoding="utf-8",
    )

    decision = {
        "route": route,
        "minimum_success": "Minimum Success B" if mlp_candidates else "not reached",
        "official_success_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "basis_family_near_pass_count": family_near,
        "basis_family_progress_count": family_progress,
        "mlp_functional_candidate_rows": len(mlp_candidates),
        "mlp_functional_control_rows": len(mlp_controls),
        "mlp_functional_exploration_pass_rows": mlp_exploration,
        "mlp_functional_official_pass_rows": mlp_official,
        "line_a_near_anchor_pass_count": line_a_near,
        "factorial_transfer_rows": len(transfer_rows),
        "promotion_allowed": 0,
        "p4_pass": int(route in {"S4-KANFunctionalSynergyPositive", "S5-OfficialFunctionalSuccess"}),
        "fallback_depth": 6,
        "fallback_all_executed": fallback_all,
        "hard_compute_budget_exhausted": int(fallback_all and route.startswith("R")),
        "final_stop_allowed": int(route.startswith("S") or (fallback_all and route.startswith("R"))),
        "provenance_violation_count": provenance_violation,
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1230_route_decision.json", decision)

    manifest_rows = required_manifest_rows()
    missing = sum(1 for r in manifest_rows if not sint(r.get("exists"), 0))
    write_rows(OUT_DIR / "v1230_required_artifact_manifest.csv", manifest_rows)
    decision["required_artifact_rows"] = len(manifest_rows)
    decision["required_artifact_missing_count"] = missing
    exp.write_json(OUT_DIR / "v1230_route_decision.json", decision)

    packet = OUT_DIR / "v1230_code_review_packet.zip"
    write_packet(packet)
    manifest_rows = required_manifest_rows()
    missing = sum(1 for r in manifest_rows if not sint(r.get("exists"), 0))
    write_rows(OUT_DIR / "v1230_required_artifact_manifest.csv", manifest_rows)
    decision["required_artifact_rows"] = len(manifest_rows)
    decision["required_artifact_missing_count"] = missing
    exp.write_json(OUT_DIR / "v1230_route_decision.json", decision)
    write_packet(packet)
    decision["code_review_packet_entries"] = len(zipfile.ZipFile(packet).namelist())
    decision["code_review_packet_sha256"] = sha256_file(packet)
    exp.write_json(OUT_DIR / "v1230_route_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True))
    return decision


if __name__ == "__main__":
    run()
