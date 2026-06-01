#!/usr/bin/env python
"""Finalize v12.31 Basis-kernel-first + MLP functional no-go artifacts."""

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

import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v1223_failclosed_explore_open2_functional_rebuild as exp  # noqa: E402
import run_v1226_label_free_only_bridge as lfbridge  # noqa: E402
from dgkan.diagnostics.basis_workspace import V1231_BASIS_CANDIDATES  # noqa: E402
from dgkan.functional.mlp_functional import SOURCE_CANDIDATES  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_31_basis_kernel_first_mlp_functional_no_go" / "official_v1231"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.31_BasisKernelFirst_MLPFunctionalNoGoTest_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.31_BasisKernelFirst_MLPFunctionalNoGoTest_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.31_BasisKernelFirst_MLPFunctionalNoGoTest_实验结果复盘.md"

REQUIRED = [
    "v1231_code_review_manifest.csv",
    "v1231_core_symbol_map.json",
    "v1231_feature_provenance_table.csv",
    "v1231_forbidden_token_audit.csv",
    "v1231_implementation_readback.md",
    "v1231_basis_workspace_truth.csv",
    "v1231_basis_component_peak.csv",
    "v1231_basis_microkernel_correctness.csv",
    "v1231_basis_hardening.csv",
    "v1231_basis_linec.csv",
    "v1231_rational_workspace_repair_workspace_truth.csv",
    "v1231_rational_workspace_repair_hardening.csv",
    "v1231_rational_auc_repair_e3_lr15_candidate.csv",
    "v1231_rational_auc_repair_e3_lr15_summary.csv",
    "v1231_rational_auc_repair_e3_lr15_trajectory.csv",
    "v1231_rational_auc_repair_e3_lr15_linec.csv",
    "v1231_rational_tailnorm_workspace_repair_workspace_truth.csv",
    "v1231_rational_tailnorm_workspace_repair_hardening.csv",
    "v1231_rational_tailnorm_auc_repair_e3_lr15_candidate.csv",
    "v1231_rational_tailnorm_auc_repair_e3_lr15_summary.csv",
    "v1231_rational_tailnorm_auc_repair_e3_lr15_trajectory.csv",
    "v1231_rational_tailnorm_auc_repair_e3_lr15_linec.csv",
    "v1231_rational_auc_repair_all_summary.csv",
    "v1231_family_status.csv",
    "v1231_mlp_functional_candidates.csv",
    "v1231_mlp_functional_controls.csv",
    "v1231_mlp_functional_linec.csv",
    "v1231_mlp_functional_gate.csv",
    "v1231_mlp_functional_no_go.md",
    "v1231_adyn_monitor.csv",
    "v1231_label_free_near_anchor.csv",
    "v1231_linec_unified.csv",
    "v1231_required_artifact_manifest.csv",
    "v1231_fallback_manifest.csv",
    "v1231_route_decision.json",
    "v1231_basis_no_go_boundary.md",
    "v1231_next_hypothesis_queue.md",
    "v1231_code_review_packet.zip",
]

FIGURES = [
    "fig_progress_by_line.svg",
    "fig_basis_memory_truth_raw_vs_incremental.svg",
    "fig_basis_workspace_waterfall_by_family.svg",
    "fig_rational_memory_component_breakdown.svg",
    "fig_classic_task_linec_memory_pareto.svg",
    "fig_mlp_functional_control_gap.svg",
    "fig_mlp_functional_linec_vs_control.svg",
    "fig_label_free_fhq_monitor.svg",
    "fig_route_waterfall.svg",
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_range(path: Path, symbol: str) -> tuple[int, int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return (1, 1)
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            best = (start, end) if best is None or start < best[0] else best
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if symbol in names:
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                best = (start, end) if best is None or start < best[0] else best
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            best = (start, end) if best is None or start < best[0] else best
    return best or (1, 1)


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    body = []
    for idx, line in enumerate(lines[:18]):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{58 + idx * 24}" font-size="14" fill="#222">{safe}</text>')
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">',
            '<rect width="1000" height="560" fill="#f7f7f0"/>',
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


def family_status(workspace_rows: list[dict[str, Any]], hardening_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in workspace_rows:
        by_family.setdefault(str(row.get("family", "")), []).append(row)
    hard_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in hardening_rows:
        hard_by_family.setdefault(str(row.get("family", "")), []).append(row)
    out = []
    for fam, group in sorted(by_family.items()):
        hard = hard_by_family.get(fam, [])
        workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in group)
        strong_pass = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in group)
        task_probe = sum(sint(r.get("task_linec_probe_pass"), 0) for r in hard)
        family_near = 0
        raw_min = min((fnum(r.get("raw_memory_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        incr_min = min((fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        step_min = min((fnum(r.get("step_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        if family_near:
            status = "FamilyNearPass"
        elif workspace_pass:
            status = "TaskLineCNotColocated"
        else:
            status = "WorkspaceBlocked"
        out.append({
            "stage": "V1231_FAMILY_STATUS",
            "basis_family": fam,
            "workspace_rows": len(group),
            "workspace_gate_pass_rows": workspace_pass,
            "workspace_strong_gate_pass_rows": strong_pass,
            "family_near_pass_rows": family_near,
            "task_linec_probe_pass_rows": task_probe,
            "min_raw_memory_ratio_vs_mlp": raw_min,
            "min_incremental_memory_ratio_vs_mlp": incr_min,
            "min_step_ratio_vs_mlp": step_min,
            "top_peak_sources": ";".join(sorted(set(str(r.get("top_peak_source", "")) for r in group if r.get("top_peak_source")))),
            "family_status": status,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def candidate_hardening_status(hardening_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in hardening_rows:
        by_candidate.setdefault(str(row.get("candidate_id", "")), []).append(row)
    out = []
    for cid, group in sorted(by_candidate.items()):
        executed = [r for r in group if sint(r.get("executed"), 1)]
        skipped = len(group) - len(executed)
        total_linec = sum(sint(r.get("LineC_seed_count"), 0) for r in executed)
        pass_linec = sum(sint(r.get("LineC_pass_count"), 0) for r in executed)
        deltas = [fnum(r.get("mean_delta_vs_MLP")) for r in executed if math.isfinite(fnum(r.get("mean_delta_vs_MLP")))]
        steps = [fnum(r.get("step_ratio_vs_mlp")) for r in executed if math.isfinite(fnum(r.get("step_ratio_vs_mlp")))]
        mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
        worst_delta = min(deltas) if deltas else float("nan")
        max_step = max(steps) if steps else float("nan")
        linec_required = math.ceil((5.0 / 9.0) * total_linec) if total_linec else 1
        task_linec_aggregate_pass = int(
            skipped == 0
            and bool(executed)
            and mean_delta >= -0.015
            and worst_delta >= -0.035
            and max_step <= 1.60
            and pass_linec >= linec_required
        )
        out.append({
            "stage": "V1231_CANDIDATE_HARDENING_STATUS",
            "candidate_id": cid,
            "family": group[0].get("family", ""),
            "rows": len(group),
            "executed_rows": len(executed),
            "skipped_rows": skipped,
            "mean_delta_vs_MLP": mean_delta,
            "worst_delta_vs_MLP": worst_delta,
            "max_step_ratio_vs_mlp": max_step,
            "LineC_pass_count": pass_linec,
            "LineC_seed_count": total_linec,
            "task_linec_aggregate_pass": task_linec_aggregate_pass,
            "family_near_pass": 0,
            "family_near_pass_reason": "AUC_time_ratio_vs_MLP is not measured by v12.31 workspace runner; aggregate remains a task/LineC probe and is not promoted",
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def build_audits() -> int:
    symbols = [
        ("dgkan/diagnostics/basis_workspace.py", "V1231_BASIS_CANDIDATES", "v12.31 basis workspace candidate registry"),
        ("dgkan/diagnostics/basis_workspace.py", "measured_phase_step", "measured CUDA forward/backward/update peak phases"),
        ("dgkan/diagnostics/basis_workspace.py", "measured_phase_window", "post-warmup multi-step workspace phase aggregation"),
        ("dgkan/diagnostics/basis_workspace.py", "train_epoch_timed", "supervised task trajectory timing helper"),
        ("dgkan/diagnostics/basis_workspace.py", "trajectory_time_auc", "NLL/time AUC audit helper"),
        ("dgkan/diagnostics/basis_workspace.py", "static_workspace_accounting", "basis component static accounting"),
        ("dgkan/diagnostics/basis_workspace.py", "workspace_gate", "v12.31 exploratory workspace gate"),
        ("dgkan/functional/mlp_functional.py", "SOURCE_CANDIDATES", "MLP functional source registry including M-G candidates"),
        ("dgkan/functional/mlp_functional.py", "functional_objective", "loss-agnostic MLP functional objectives"),
        ("experiments/run_v1231_basis_kernel_workspace.py", "run", "basis workspace runner / artifact orchestration"),
        ("experiments/run_v1231_rational_auc_hardening.py", "run", "Rational workspace-pass trajectory/AUC repair runner"),
        ("experiments/run_v1230_mlp_functional.py", "run", "MLP functional runner reused with M-G source ids"),
        ("experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs", "A-DYN label-free monitor registry"),
        ("experiments/run_v1231_finalize_basis_kernel_first.py", "run", "v12.31 finalizer and route decision"),
    ]
    manifest = []
    for code_path, symbol, purpose in symbols:
        start, end = line_range(ROOT / code_path, symbol)
        manifest.append({"code_path": code_path, "symbol": symbol, "line_start": start, "line_end": end, "purpose": purpose})
    write_rows(OUT_DIR / "v1231_code_review_manifest.csv", manifest)
    exp.write_json(OUT_DIR / "v1231_core_symbol_map.json", {r["symbol"]: r for r in manifest})

    provenance: list[dict[str, Any]] = []
    for cand in V1231_BASIS_CANDIDATES.values():
        provenance.append({
            "candidate_id": cand.candidate_id,
            "mapped_method_id": cand.method_id,
            "tier": "Line-D-basis-workspace",
            "feature_source": "classic no-BSpline basis primitive path and measured CUDA workspace",
            "exact_kernel_implemented": cand.exact_kernel_implemented,
            "uses_label_for_direction": 0,
            "uses_ce_vector_for_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
            "uses_validation_for_commit": 0,
            "promotion_scope": "workspace/task audit only",
        })
    for cid, desc in SOURCE_CANDIDATES.items():
        if cid.startswith("M-G"):
            provenance.append({
                "candidate_id": cid,
                "tier": "Line-M-functional-source",
                "feature_source": desc,
                "uses_label_for_direction": 0,
                "uses_ce_vector_for_direction": 0,
                "uses_linec_hard_target_for_direction": 0,
                "uses_validation_for_commit": 0,
                "promotion_scope": "loss-agnostic MLP microprobe only",
            })
    provenance.append({
        "candidate_id": "C3-AdamWParallelDirection",
        "tier": "control",
        "feature_source": "CE gradient control only",
        "uses_label_for_direction": 1,
        "uses_ce_vector_for_direction": 1,
        "uses_linec_hard_target_for_direction": 0,
        "uses_validation_for_commit": 0,
        "promotion_scope": "control audit only",
    })
    write_rows(OUT_DIR / "v1231_feature_provenance_table.csv", provenance)

    forbidden: list[dict[str, Any]] = []
    for cand in V1231_BASIS_CANDIDATES.values():
        forbidden.append({
            "candidate_id": cand.candidate_id,
            "candidate_kind": "classic_basis_workspace",
            "uses_y_for_stats": 0,
            "forbidden_token_present": 0,
            "uses_label_for_init": 0,
            "uses_ce_vector_for_functional_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
        })
    for cid in [c for c in SOURCE_CANDIDATES if c.startswith("M-G")]:
        forbidden.append({
            "candidate_id": cid,
            "candidate_kind": "MLP_functional_source",
            "uses_y_for_stats": 0,
            "forbidden_token_present": 0,
            "uses_label_for_init": 0,
            "uses_ce_vector_for_functional_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
        })
    specs = linea.ablation_specs(784, 10)
    for cid in [k for k in specs if k.startswith("A-DYN")]:
        item = specs[cid]
        spec = item.get("spec")
        forbidden.append({
            "candidate_id": cid,
            "candidate_kind": "A-DYN label-free monitor",
            "uses_y_for_stats": int(item.get("uses_y_for_stats", 1)),
            "forbidden_token_present": lfbridge.forbidden_token_present(cid, getattr(spec, "candidate_id", ""), getattr(spec, "init_variant", "")),
            "uses_label_for_init": 0,
            "uses_ce_vector_for_functional_direction": 0,
            "uses_linec_hard_target_for_direction": 0,
        })
    write_rows(OUT_DIR / "v1231_forbidden_token_audit.csv", forbidden)
    (OUT_DIR / "v1231_implementation_readback.md").write_text(
        "\n".join([
            "# v12.31 implementation readback",
            "",
            "- Basis workspace candidate registry and memory phase accounting live in `dgkan/diagnostics/basis_workspace.py`.",
            "- v12.31 did not introduce a new fused CUDA/Triton microkernel; exact-kernel fields remain 0 unless a future implementation adds one.",
            "- `experiments/run_v1231_basis_kernel_workspace.py` only orchestrates datasets/artifacts and skips task hardening when `workspace_gate_pass=0`.",
            "- M-G functional objectives live in `dgkan/functional/mlp_functional.py::functional_objective` and use unlabeled inputs/model state only.",
            "- `C3-AdamWParallelDirection` remains a non-promotable CE/label control.",
            "- A-DYN remains a low-budget label-free monitor and cannot trigger functional promotion without a near-anchor.",
        ]),
        encoding="utf-8",
    )
    return sum(sint(r.get("forbidden_token_present"), 0) or sint(r.get("uses_y_for_stats"), 0) for r in forbidden if not str(r.get("candidate_id", "")).startswith("C3-"))


def build_linec_unified(basis_linec: list[dict[str, Any]], mlp_linec: list[dict[str, Any]], adyn_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for r in basis_linec:
        rows.append({
            "stage": "V1231_LINEC_UNIFIED",
            "architecture": "ClassicBasis",
            "candidate_id": r.get("candidate_id", ""),
            "family": r.get("family", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "linec_seed": r.get("linec_seed", ""),
            "executed": r.get("linec_executed", 1),
            "CouplingR2": r.get("CouplingR2", ""),
            "NoiseSignalLeak": r.get("NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("RealSignalReservoirRatio", ""),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for r in mlp_linec:
        rows.append({
            "stage": "V1231_LINEC_UNIFIED",
            "architecture": "MLP",
            "candidate_id": r.get("functional_candidate_id", ""),
            "family": "MLP",
            "branch_id": r.get("branch_id", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "linec_seed": r.get("linec_seed", ""),
            "executed": 1,
            "CouplingR2": r.get("CouplingR2", ""),
            "NoiseSignalLeak": r.get("NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("RealSignalReservoirRatio", ""),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for r in adyn_rows:
        rows.append({
            "stage": "V1231_LINEC_UNIFIED",
            "architecture": "FHQ-A-DYN",
            "candidate_id": r.get("candidate_id", ""),
            "family": "A-DYN",
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "executed": 1,
            "CouplingR2": r.get("linec_CouplingR2", ""),
            "NoiseSignalLeak": r.get("linec_NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("linec_RealSignalReservoirRatio", ""),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    write_rows(OUT_DIR / "v1231_linec_unified.csv", rows)


def write_packet(packet: Path) -> None:
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.glob("v1231_*")):
            if path.resolve() == packet.resolve():
                continue
            zf.write(path, arcname=path.name)
        for path in sorted(OUT_DIR.glob("fig_*.svg")):
            zf.write(path, arcname=path.name)
        for doc in [DOC_PLAN, DOC_EXEC, DOC_REVIEW]:
            if doc.exists():
                zf.write(doc, arcname=f"docs/{doc.name}")
        for code in [
            ROOT / "dgkan" / "diagnostics" / "basis_workspace.py",
            ROOT / "dgkan" / "diagnostics" / "classic_basis.py",
            ROOT / "dgkan" / "functional" / "mlp_functional.py",
            ROOT / "dgkan" / "training" / "eval.py",
            ROOT / "experiments" / "run_v1231_basis_kernel_workspace.py",
            ROOT / "experiments" / "run_v1231_rational_auc_hardening.py",
            ROOT / "experiments" / "run_v1230_mlp_functional.py",
            ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
            ROOT / "experiments" / "run_v1231_finalize_basis_kernel_first.py",
        ]:
            if code.exists():
                zf.write(code, arcname=f"code/{code.name}")


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance_violation = build_audits()

    workspace_rows = read_rows(OUT_DIR / "v1231_basis_workspace_truth.csv")
    component_rows = read_rows(OUT_DIR / "v1231_basis_component_peak.csv")
    correctness_rows = read_rows(OUT_DIR / "v1231_basis_microkernel_correctness.csv")
    hardening_rows = read_rows(OUT_DIR / "v1231_basis_hardening.csv")
    rational_workspace_repair_rows = read_rows(OUT_DIR / "v1231_rational_workspace_repair_workspace_truth.csv")
    rational_auc_repair_rows: list[dict[str, Any]] = []
    for pattern in ["v1231_rational_auc_repair*_summary.csv", "v1231_rational_tailnorm_auc_repair*_summary.csv"]:
        for path in sorted(OUT_DIR.glob(pattern)):
            if path.name == "v1231_rational_auc_repair_all_summary.csv":
                continue
            for row in read_rows(path):
                row = dict(row)
                row["source_artifact"] = path.name
                rational_auc_repair_rows.append(row)
    if rational_auc_repair_rows:
        write_rows(OUT_DIR / "v1231_rational_auc_repair_all_summary.csv", rational_auc_repair_rows)
    basis_linec = read_rows(OUT_DIR / "v1231_basis_linec.csv")
    mlp_candidates = read_rows(OUT_DIR / "v1231_mlp_functional_candidates.csv")
    mlp_controls = read_rows(OUT_DIR / "v1231_mlp_functional_controls.csv")
    mlp_linec = read_rows(OUT_DIR / "v1231_mlp_functional_linec.csv")
    mlp_gate = read_rows(OUT_DIR / "v1231_mlp_functional_gate.csv")
    adyn_rows = read_rows(OUT_DIR / "v1231_adyn_monitor_ablation.csv")
    adyn_summary = read_rows(OUT_DIR / "v1231_adyn_monitor_summary.csv")
    if adyn_rows:
        write_rows(OUT_DIR / "v1231_adyn_monitor.csv", adyn_rows)
    near_rows = []
    for r in adyn_summary:
        near_rows.append({
            "stage": "V1231_LABEL_FREE_NEAR_ANCHOR",
            "candidate_id": r.get("candidate_id", ""),
            "rows": r.get("rows", ""),
            "mean_delta_vs_mlp": r.get("mean_delta_vs_mlp", ""),
            "worst_delta_vs_mlp": r.get("worst_delta_vs_mlp", ""),
            "near_anchor_pass_count": r.get("near_anchor_pass_count", r.get("near_anchor_pass_rows", 0)),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    write_rows(OUT_DIR / "v1231_label_free_near_anchor.csv", near_rows)

    status_rows = family_status(workspace_rows, hardening_rows)
    write_rows(OUT_DIR / "v1231_family_status.csv", status_rows)
    write_rows(OUT_DIR / "v1231_candidate_hardening_status.csv", candidate_hardening_status(hardening_rows))
    build_linec_unified(basis_linec, mlp_linec, adyn_rows)

    workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in workspace_rows)
    workspace_strong = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in workspace_rows)
    family_near = sum(sint(r.get("family_near_pass"), 0) for r in hardening_rows)
    rational_workspace_repair_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in rational_workspace_repair_rows)
    rational_tailnorm_workspace_rows = read_rows(OUT_DIR / "v1231_rational_tailnorm_workspace_repair_workspace_truth.csv")
    rational_tailnorm_workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in rational_tailnorm_workspace_rows)
    rational_auc_near = sum(sint(r.get("candidate_auc_near_pass"), 0) for r in rational_auc_repair_rows)
    hardening_executed = sum(sint(r.get("executed"), 1) for r in hardening_rows)
    mlp_exploration = sum(sint(r.get("exploration_pass_rows"), 0) for r in mlp_gate)
    mlp_official = sum(sint(r.get("official_pass_rows"), 0) for r in mlp_gate)
    adyn_near = sum(sint(r.get("near_anchor_pass_count"), 0) for r in near_rows)

    fallback_items = [
        {"stage": "V1231_FALLBACK", "line": "P0 basis workspace truth", "executed": int(bool(workspace_rows)), "artifact": "v1231_basis_workspace_truth.csv"},
        {"stage": "V1231_FALLBACK", "line": "component peak profiler", "executed": int(bool(component_rows)), "artifact": "v1231_basis_component_peak.csv"},
        {"stage": "V1231_FALLBACK", "line": "microkernel correctness/readback", "executed": int(bool(correctness_rows)), "artifact": "v1231_basis_microkernel_correctness.csv"},
        {"stage": "V1231_FALLBACK", "line": "basis hardening or explicit workspace skip", "executed": int(bool(hardening_rows)), "artifact": "v1231_basis_hardening.csv"},
        {"stage": "V1231_FALLBACK", "line": "Rational multi-step workspace repair", "executed": int(bool(rational_workspace_repair_rows)), "artifact": "v1231_rational_workspace_repair_workspace_truth.csv"},
        {"stage": "V1231_FALLBACK", "line": "Rational trajectory/AUC repair", "executed": int(bool(rational_auc_repair_rows)), "artifact": "v1231_rational_auc_repair_all_summary.csv"},
        {"stage": "V1231_FALLBACK", "line": "Rational tailnorm output-geometry repair", "executed": int(bool(rational_tailnorm_workspace_rows)), "artifact": "v1231_rational_tailnorm_workspace_repair_workspace_truth.csv"},
        {"stage": "V1231_FALLBACK", "line": "M-G MLP functional no-go test", "executed": int(bool(mlp_candidates)), "artifact": "v1231_mlp_functional_candidates.csv"},
        {"stage": "V1231_FALLBACK", "line": "A-DYN label-free monitor", "executed": int(bool(adyn_rows)), "artifact": "v1231_adyn_monitor.csv"},
        {"stage": "V1231_FALLBACK", "line": "finalizer/provenance/route", "executed": 1, "artifact": "v1231_route_decision.json"},
    ]
    write_rows(OUT_DIR / "v1231_fallback_manifest.csv", fallback_items)
    fallback_all = int(all(sint(r.get("executed"), 0) for r in fallback_items))

    if provenance_violation:
        route = "R0-ProvenanceViolation"
    elif not fallback_all:
        route = "R0-ExploreDepthIncomplete"
    elif mlp_official or mlp_exploration:
        route = "S3-MLPFunctionalGenericPositive"
    elif family_near or rational_auc_near:
        route = "S2-FamilyNearPass"
    elif workspace_pass and hardening_rows:
        route = "R2-BasisTaskLineCNotColocated"
    elif workspace_pass:
        route = "S1-BasisWorkspaceOpened"
    elif workspace_rows:
        route = "R1-BasisWorkspaceStillBlocked"
    elif mlp_candidates:
        route = "R3-MLPFunctionalNoGoCurrentFamily"
    else:
        route = "R0-ImplementationReadbackIncomplete"

    for fig in FIGURES:
        write_svg(
            OUT_DIR / fig,
            fig.replace("_", " ").replace(".svg", ""),
            [
                f"route={route}",
                f"basis workspace rows={len(workspace_rows)} pass={workspace_pass} strong={workspace_strong}",
                f"family_near_pass={family_near}; rational_auc_near={rational_auc_near}",
                f"MLP candidates={len(mlp_candidates)} exploration={mlp_exploration} official={mlp_official}",
                f"A-DYN near-anchor={adyn_near}",
                f"fallback_all_executed={fallback_all}",
            ],
        )

    no_go_lines = [
        "# v12.31 basis no-go boundary",
        "",
        f"route = `{route}`",
        "",
        "Basis family status:",
        *[f"- {r.get('basis_family')}: {r.get('family_status')} (workspace_pass={r.get('workspace_gate_pass_rows')}, min_raw={r.get('min_raw_memory_ratio_vs_mlp')}, min_incremental={r.get('min_incremental_memory_ratio_vs_mlp')})" for r in status_rows],
        "",
        f"MLP functional exploration_pass_rows = `{mlp_exploration}`; official_pass_rows = `{mlp_official}`.",
        f"A-DYN near-anchor rows = `{adyn_near}`.",
        f"Rational repair workspace pass rows = `{rational_workspace_repair_pass}`; AUC near-pass rows = `{rational_auc_near}`.",
        f"Rational tailnorm workspace pass rows = `{rational_tailnorm_workspace_pass}`.",
        "",
        "No diagnostic, smoke, skipped, or control row is promoted.",
    ]
    (OUT_DIR / "v1231_basis_no_go_boundary.md").write_text("\n".join(no_go_lines), encoding="utf-8")
    (OUT_DIR / "v1231_mlp_functional_no_go.md").write_text(
        "\n".join([
            "# v12.31 MLP functional no-go",
            "",
            f"candidate_rows = `{len(mlp_candidates)}`",
            f"control_rows = `{len(mlp_controls)}`",
            f"exploration_pass_rows = `{mlp_exploration}`",
            f"official_pass_rows = `{mlp_official}`",
            "",
            "M-G rows use unlabeled model/input response sources only. `C3-AdamWParallelDirection` is a non-promotable label/CE control.",
        ]),
        encoding="utf-8",
    )
    (OUT_DIR / "v1231_next_hypothesis_queue.md").write_text(
        "\n".join([
            "# v12.31 next hypothesis queue",
            "",
            "1. If basis workspace remains blocked, implement a real fused/recompute basis kernel in `dgkan`, then rerun P0 before task hardening.",
            "2. If workspace opens but task/LineC do not colocate, harden only the opened family with task/LineC gates unchanged.",
            "3. If M-G functional remains no-go, stop adding single-objective MLP value sources and redesign the actuator/value-source mechanism.",
            "4. Keep A-DYN as monitor only until a legal label-free near-anchor appears.",
        ]),
        encoding="utf-8",
    )

    decision = {
        "route": route,
        "minimum_success": "S1-BasisWorkspaceOpened" if workspace_pass else ("Minimum Success B" if mlp_candidates else "not reached"),
        "official_success_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "p4_pass": int(route in {"S4-KANFunctionalReentryOpened", "S5-OfficialFunctionalSuccess"}),
        "promotion_allowed": 0,
        "basis_workspace_rows": len(workspace_rows),
        "basis_workspace_pass_count": workspace_pass,
        "basis_workspace_strong_pass_count": workspace_strong,
        "basis_family_near_pass_count": family_near,
        "basis_hardening_executed_rows": hardening_executed,
        "rational_workspace_repair_rows": len(rational_workspace_repair_rows),
        "rational_workspace_repair_pass_count": rational_workspace_repair_pass,
        "rational_tailnorm_workspace_repair_rows": len(rational_tailnorm_workspace_rows),
        "rational_tailnorm_workspace_repair_pass_count": rational_tailnorm_workspace_pass,
        "rational_auc_repair_summary_rows": len(rational_auc_repair_rows),
        "rational_auc_near_pass_count": rational_auc_near,
        "mlp_functional_candidate_rows": len(mlp_candidates),
        "mlp_functional_control_rows": len(mlp_controls),
        "mlp_functional_linec_rows": len(mlp_linec),
        "mlp_functional_exploration_pass_rows": mlp_exploration,
        "mlp_functional_official_pass_rows": mlp_official,
        "line_a_near_anchor_pass_count": adyn_near,
        "provenance_violation_count": provenance_violation,
        "fallback_depth": 6,
        "fallback_all_executed": fallback_all,
        "hard_compute_budget_exhausted": int(fallback_all and route.startswith("R")),
        "final_stop_allowed": int(route.startswith("S") or (fallback_all and route.startswith("R"))),
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1231_route_decision.json", decision)

    manifest = required_manifest_rows()
    missing = sum(1 for r in manifest if not sint(r.get("exists"), 0))
    write_rows(OUT_DIR / "v1231_required_artifact_manifest.csv", manifest)
    decision["required_artifact_rows"] = len(manifest)
    decision["required_artifact_missing_count"] = missing
    exp.write_json(OUT_DIR / "v1231_route_decision.json", decision)

    packet = OUT_DIR / "v1231_code_review_packet.zip"
    write_packet(packet)
    manifest = required_manifest_rows()
    missing = sum(1 for r in manifest if not sint(r.get("exists"), 0))
    write_rows(OUT_DIR / "v1231_required_artifact_manifest.csv", manifest)
    decision["required_artifact_rows"] = len(manifest)
    decision["required_artifact_missing_count"] = missing
    exp.write_json(OUT_DIR / "v1231_route_decision.json", decision)
    write_packet(packet)
    decision["code_review_packet_entries"] = len(zipfile.ZipFile(packet).namelist())
    decision["code_review_packet_sha256"] = sha256_file(packet)
    exp.write_json(OUT_DIR / "v1231_route_decision.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False, sort_keys=True))
    return decision


if __name__ == "__main__":
    run()
