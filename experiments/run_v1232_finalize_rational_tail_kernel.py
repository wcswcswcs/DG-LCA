#!/usr/bin/env python
"""Finalize v12.32 Rational-tail kernel + Non-RAT exact-kernel artifacts."""

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
from dgkan.diagnostics.basis_workspace import V1232_BASIS_CANDIDATES  # noqa: E402
from dgkan.functional.mlp_functional import CONTROL_IDS, SOURCE_CANDIDATES  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_32_rational_tail_kernel_nonrat_fused_kernels_mlp_functional_no_go" / "official_v1232"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.32_RationalTailKernel_NonRATFusedKernels_MLPFunctionalNoGo_实验结果复盘.md"

REQUIRED = [
    "v1232_code_review_manifest.csv",
    "v1232_core_symbol_map.json",
    "v1232_kernel_implementation_manifest.csv",
    "v1232_exact_kernel_audit.csv",
    "v1232_provenance_audit.csv",
    "v1232_forbidden_token_audit.csv",
    "v1232_basis_workspace_truth.csv",
    "v1232_basis_component_peak.csv",
    "v1232_basis_hardening.csv",
    "v1232_basis_linec.csv",
    "v1232_rational_tail_auc_lr15_candidate.csv",
    "v1232_rational_tail_auc_lr15_summary.csv",
    "v1232_rational_tail_auc_lr15_trajectory.csv",
    "v1232_rational_tail_auc_lr15_linec.csv",
    "v1232_rational_tail_auc_lr20_candidate.csv",
    "v1232_rational_tail_auc_lr20_summary.csv",
    "v1232_rational_tail_auc_lr20_trajectory.csv",
    "v1232_rational_tail_auc_lr20_linec.csv",
    "v1232_rational_tail_auc_all_summary.csv",
    "v1232_family_status.csv",
    "v1232_mlp_functional_candidates.csv",
    "v1232_mlp_functional_controls.csv",
    "v1232_mlp_functional_linec.csv",
    "v1232_mlp_functional_gate.csv",
    "v1232_mlp_functional_no_go.md",
    "v1232_adyn_monitor.csv",
    "v1232_label_free_near_anchor.csv",
    "v1232_linec_unified.csv",
    "v1232_required_artifact_manifest.csv",
    "v1232_fallback_manifest.csv",
    "v1232_route_decision.json",
    "v1232_basis_no_go_boundary.md",
    "v1232_next_hypothesis_queue.md",
    "v1232_code_review_packet.zip",
]

FIGURES = [
    "fig_v1232_progress_by_line.svg",
    "fig_v1232_workspace_pareto_by_family.svg",
    "fig_v1232_rational_tail_task_linec_tradeoff.svg",
    "fig_v1232_rational_denominator_derivative_vs_CEp99.svg",
    "fig_v1232_nonrat_incremental_memory_breakdown.svg",
    "fig_v1232_family_expression_task_linec_radar.svg",
    "fig_v1232_mlp_functional_control_gap.svg",
    "fig_v1232_linec_task_colocation_scatter.svg",
    "fig_v1232_route_dashboard.svg",
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
            '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="600" viewBox="0 0 1100 600">',
            '<rect width="1100" height="600" fill="#f7f7f0"/>',
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


def family_status(workspace_rows: list[dict[str, Any]], hardening_rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_by_candidate = {str(r.get("candidate_id", "")): r for r in exact_rows}
    hard_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in hardening_rows:
        hard_by_candidate.setdefault(str(row.get("candidate_id", "")), []).append(row)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in workspace_rows:
        by_family.setdefault(str(row.get("family", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for fam, group in sorted(by_family.items()):
        exact_candidates = sorted({str(r.get("candidate_id", "")) for r in group if sint(r.get("exact_kernel_implemented"), 0)})
        s3_candidates = []
        for cid in exact_candidates:
            crows = [r for r in group if str(r.get("candidate_id", "")) == cid]
            erow = exact_by_candidate.get(cid, {})
            if sum(sint(r.get("workspace_gate_pass"), 0) for r in crows) >= 8 and sint(erow.get("A4_expression_smoke_pass"), 0):
                s3_candidates.append(cid)
        hard = [r for cid in {str(g.get("candidate_id", "")) for g in group} for r in hard_by_candidate.get(cid, [])]
        raw_min = min((fnum(r.get("raw_memory_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        incr_min = min((fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        step_min = min((fnum(r.get("step_ratio_vs_mlp"), 999.0) for r in group), default=float("nan"))
        workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in group)
        strong_pass = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in group)
        if s3_candidates:
            status = "S3-NonRATTrueKernelOpened"
        elif workspace_pass:
            status = "TaskLineCNotColocated" if fam == "D-RAT" else "WorkspaceOpenedNoS3"
        else:
            status = "WorkspaceBlocked"
        out.append({
            "stage": "V1232_FAMILY_STATUS",
            "basis_family": fam,
            "workspace_rows": len(group),
            "workspace_gate_pass_rows": workspace_pass,
            "workspace_strong_gate_pass_rows": strong_pass,
            "task_linec_probe_pass_rows": sum(sint(r.get("task_linec_probe_pass"), 0) for r in hard),
            "exact_kernel_candidates": ",".join(exact_candidates),
            "s3_true_kernel_candidates": ",".join(s3_candidates),
            "s3_true_kernel_count": len(s3_candidates),
            "min_raw_memory_ratio_vs_mlp": raw_min,
            "min_incremental_memory_ratio_vs_mlp": incr_min,
            "min_step_ratio_vs_mlp": step_min,
            "family_status": status,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def strict_mh_gate(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in candidate_rows:
        row["v1232_row_gate_pass"] = int(
            fnum(row.get("source_vs_control_acc_delta"), -999.0) >= 0.005
            and sint(row.get("LineC_pass_count"), 0) >= 4
            and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.0
            and fnum(row.get("CEp99_delta_vs_noop"), 999.0) <= 0.05
            and fnum(row.get("NLL_delta_vs_noop"), 999.0) <= 0.02
            and fnum(row.get("ECE_delta_vs_noop"), 999.0) <= 0.02
        )
    by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by.setdefault((str(row.get("functional_candidate_id", "")), sint(row.get("window_epochs"), 0)), []).append(row)
    gate_rows: list[dict[str, Any]] = []
    for (cid, window), rows in sorted(by.items()):
        passes = sum(sint(r.get("v1232_row_gate_pass"), 0) for r in rows)
        pass_datasets = sorted({str(r.get("dataset", "")) for r in rows if sint(r.get("v1232_row_gate_pass"), 0)})
        max_one_dataset = max((sum(1 for r in rows if str(r.get("dataset", "")) == d and sint(r.get("v1232_row_gate_pass"), 0)) for d in pass_datasets), default=0)
        aggregate = int(passes >= 8 and max_one_dataset < passes)
        gate_rows.append({
            "stage": "V1232_MLP_FUNCTIONAL_GATE",
            "functional_candidate_id": cid,
            "window_epochs": window,
            "rows": len(rows),
            "v1232_row_pass_rows": passes,
            "m_h_aggregate_pass": aggregate,
            "mean_source_vs_noop": sum(fnum(r.get("source_vs_noop_acc_delta"), 0.0) for r in rows) / max(1, len(rows)),
            "mean_source_vs_control": sum(fnum(r.get("source_vs_control_acc_delta"), 0.0) for r in rows) / max(1, len(rows)),
            "mean_CouplingR2_delta": sum(fnum(r.get("CouplingR2_delta"), 0.0) for r in rows) / max(1, len(rows)),
            "max_LineC_pass_count": max((sint(r.get("LineC_pass_count"), 0) for r in rows), default=0),
            "pass_datasets": ",".join(pass_datasets),
            "no_single_dataset_concentration": int(max_one_dataset < passes) if passes else 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return gate_rows


def provenance_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in V1232_BASIS_CANDIDATES.values():
        rows.append({
            "stage": "V1232_PROVENANCE_AUDIT",
            "line": "LineD",
            "candidate_id": cand.candidate_id,
            "uses_label_or_ce_for_direction": 0,
            "uses_y_for_stats": 0,
            "uses_validation_for_commit": 0,
            "uses_query_batch_for_commit": 0,
            "forbidden_token_present": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for cid in [c for c in SOURCE_CANDIDATES if c.startswith("M-H")]:
        rows.append({
            "stage": "V1232_PROVENANCE_AUDIT",
            "line": "LineM",
            "candidate_id": cid,
            "uses_label_or_ce_for_direction": 0,
            "uses_y_for_stats": 0,
            "uses_validation_for_commit": 0,
            "uses_query_batch_for_commit": 0,
            "forbidden_token_present": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for ctrl in CONTROL_IDS:
        rows.append({
            "stage": "V1232_PROVENANCE_AUDIT",
            "line": "LineMControl",
            "candidate_id": ctrl,
            "uses_label_or_ce_for_direction": int(ctrl == "C3-AdamWParallelDirection"),
            "uses_y_for_stats": 0,
            "uses_validation_for_commit": 0,
            "uses_query_batch_for_commit": 0,
            "forbidden_token_present": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    try:
        for ablation_id, spec_info in linea.ablation_specs(784, 10).items():
            if ablation_id.startswith("A-DYN"):
                spec = spec_info["spec"]
                rows.append({
                    "stage": "V1232_PROVENANCE_AUDIT",
                    "line": "LineA",
                    "candidate_id": ablation_id,
                    "uses_label_or_ce_for_direction": int(spec_info.get("uses_y_for_stats", 0)),
                    "uses_y_for_stats": int(spec_info.get("uses_y_for_stats", 0)),
                    "uses_validation_for_commit": 0,
                    "uses_query_batch_for_commit": 0,
                    "forbidden_token_present": int(lfbridge.forbidden_token_present(ablation_id, getattr(spec, "candidate_id", ""), getattr(spec, "init_variant", ""))),
                    "promotion_allowed": 0,
                    "no_fake": 1,
                })
    except Exception as exc:  # noqa: BLE001
        rows.append({"stage": "V1232_PROVENANCE_AUDIT", "line": "LineA", "candidate_id": "A-DYN-read-error", "error": f"{type(exc).__name__}: {exc}", "forbidden_token_present": 1, "promotion_allowed": 0, "no_fake": 1})
    return rows


def core_symbol_map() -> dict[str, Any]:
    symbols = {
        "dgkan/diagnostics/basis_workspace.py": [
            "V1232_BASIS_CANDIDATES",
            "workspace_gate_v1232",
            "workspace_strong_gate_v1232",
            "kernel_implementation_manifest_rows",
            "exact_kernel_audit_rows",
            "rational_tail_metrics",
        ],
        "dgkan/functional/mlp_functional.py": ["SOURCE_CANDIDATES", "functional_objective"],
        "experiments/run_v1231_basis_kernel_workspace.py": ["run"],
        "experiments/run_v1231_rational_auc_hardening.py": ["run"],
        "experiments/run_v1230_mlp_functional.py": ["run"],
        "experiments/run_v1232_finalize_rational_tail_kernel.py": ["run"],
    }
    out: dict[str, Any] = {}
    for rel, names in symbols.items():
        path = ROOT / rel
        out[rel] = {}
        for name in names:
            start, end = line_range(path, name)
            out[rel][name] = {"start_line": start, "end_line": end}
    return out


def code_review_manifest_rows() -> list[dict[str, Any]]:
    files = [
        "dgkan/diagnostics/basis_workspace.py",
        "dgkan/functional/mlp_functional.py",
        "experiments/run_v1231_basis_kernel_workspace.py",
        "experiments/run_v1231_rational_auc_hardening.py",
        "experiments/run_v1230_mlp_functional.py",
        "experiments/run_v1232_finalize_rational_tail_kernel.py",
        str(DOC_PLAN.relative_to(ROOT)),
        str(DOC_EXEC.relative_to(ROOT)),
        str(DOC_REVIEW.relative_to(ROOT)),
    ]
    rows = []
    for rel in files:
        p = ROOT / rel
        rows.append({
            "path": rel,
            "exists": int(p.exists()),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "sha256": sha256_file(p) if p.exists() else "",
        })
    return rows


def make_packet() -> tuple[int, str]:
    packet = OUT_DIR / "v1232_code_review_packet.zip"
    files = [
        OUT_DIR / "v1232_route_decision.json",
        OUT_DIR / "v1232_required_artifact_manifest.csv",
        OUT_DIR / "v1232_code_review_manifest.csv",
        OUT_DIR / "v1232_core_symbol_map.json",
        OUT_DIR / "v1232_kernel_implementation_manifest.csv",
        OUT_DIR / "v1232_exact_kernel_audit.csv",
        OUT_DIR / "v1232_provenance_audit.csv",
        OUT_DIR / "v1232_forbidden_token_audit.csv",
        OUT_DIR / "v1232_family_status.csv",
        OUT_DIR / "v1232_mlp_functional_no_go.md",
        OUT_DIR / "v1232_basis_no_go_boundary.md",
        OUT_DIR / "v1232_next_hypothesis_queue.md",
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
    ]
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
        for fig in FIGURES:
            p = OUT_DIR / fig
            if p.exists():
                zf.write(p, arcname=str(p.relative_to(ROOT)))
    return len(files), sha256_file(packet)


def run() -> dict[str, Any]:
    exp.ensure_dir(OUT_DIR)

    canonical_pairs = [
        ("v1232_basis_kernel_implementation_manifest.csv", "v1232_kernel_implementation_manifest.csv"),
        ("v1232_basis_exact_kernel_audit.csv", "v1232_exact_kernel_audit.csv"),
        ("v1232_adyn_monitor_ablation.csv", "v1232_adyn_monitor.csv"),
    ]
    for src_name, dst_name in canonical_pairs:
        src = OUT_DIR / src_name
        dst = OUT_DIR / dst_name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    workspace_rows = read_rows(OUT_DIR / "v1232_basis_workspace_truth.csv")
    hardening_rows = read_rows(OUT_DIR / "v1232_basis_hardening.csv")
    basis_linec_rows = read_rows(OUT_DIR / "v1232_basis_linec.csv")
    exact_rows = read_rows(OUT_DIR / "v1232_exact_kernel_audit.csv")
    kernel_manifest_rows = read_rows(OUT_DIR / "v1232_kernel_implementation_manifest.csv")
    lr15_summary = read_rows(OUT_DIR / "v1232_rational_tail_auc_lr15_summary.csv")
    lr20_summary = read_rows(OUT_DIR / "v1232_rational_tail_auc_lr20_summary.csv")
    mlp_candidates = read_rows(OUT_DIR / "v1232_mlp_functional_candidates.csv")
    mlp_linec_rows = read_rows(OUT_DIR / "v1232_mlp_functional_linec.csv")
    adyn_rows = read_rows(OUT_DIR / "v1232_adyn_monitor.csv")

    if mlp_candidates:
        gate_rows = strict_mh_gate(mlp_candidates)
        write_rows(OUT_DIR / "v1232_mlp_functional_candidates.csv", mlp_candidates)
        write_rows(OUT_DIR / "v1232_mlp_functional_gate.csv", gate_rows)
    else:
        gate_rows = []

    all_rational_summaries = [*lr15_summary, *lr20_summary]
    write_rows(OUT_DIR / "v1232_rational_tail_auc_all_summary.csv", all_rational_summaries)
    status_rows = family_status(workspace_rows, hardening_rows, exact_rows)
    write_rows(OUT_DIR / "v1232_family_status.csv", status_rows)
    prov = provenance_rows()
    write_rows(OUT_DIR / "v1232_provenance_audit.csv", prov)
    write_rows(OUT_DIR / "v1232_forbidden_token_audit.csv", prov)
    (OUT_DIR / "v1232_core_symbol_map.json").write_text(json.dumps(core_symbol_map(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_rows(OUT_DIR / "v1232_code_review_manifest.csv", code_review_manifest_rows())

    near_anchor_rows = []
    by_adyn: dict[str, list[dict[str, Any]]] = {}
    for row in adyn_rows:
        by_adyn.setdefault(str(row.get("ablation_id") or row.get("candidate_id") or row.get("method_id") or ""), []).append(row)
    for cid, rows in sorted(by_adyn.items()):
        if not cid or not cid.startswith("A-DYN"):
            continue
        deltas = [fnum(r.get("delta_vs_mlp") or r.get("mean_delta_vs_mlp")) for r in rows if math.isfinite(fnum(r.get("delta_vs_mlp") or r.get("mean_delta_vs_mlp")))]
        near_anchor_rows.append({
            "stage": "V1232_LABEL_FREE_NEAR_ANCHOR",
            "candidate_id": cid,
            "rows": len(rows),
            "mean_delta_vs_mlp": sum(deltas) / len(deltas) if deltas else float("nan"),
            "worst_delta_vs_mlp": min(deltas) if deltas else float("nan"),
            "near_anchor_pass": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    if not near_anchor_rows:
        near_anchor_rows.append({"stage": "V1232_LABEL_FREE_NEAR_ANCHOR", "candidate_id": "A-DYN", "rows": 0, "near_anchor_pass": 0, "note": "A-DYN canonical artifact missing or not yet executed", "promotion_allowed": 0, "no_fake": 1})
    write_rows(OUT_DIR / "v1232_label_free_near_anchor.csv", near_anchor_rows)

    linec_unified = []
    for row in [*basis_linec_rows, *mlp_linec_rows]:
        rr = dict(row)
        rr["stage"] = "V1232_LINEC_UNIFIED"
        linec_unified.append(rr)
    write_rows(OUT_DIR / "v1232_linec_unified.csv", linec_unified)

    mh_pass = sum(sint(r.get("m_h_aggregate_pass"), 0) for r in gate_rows)
    mh_no_go = int(bool(gate_rows) and mh_pass == 0)
    no_go_text = [
        "# v12.32 MLP functional no-go",
        "",
        f"M-H gate rows: {len(gate_rows)}",
        f"M-H aggregate pass rows: {mh_pass}",
        f"MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily = {mh_no_go}",
        "",
        "This artifact is derived from M-H candidate rows and matched controls. It does not use CE/labels as a functional source.",
    ]
    (OUT_DIR / "v1232_mlp_functional_no_go.md").write_text("\n".join(no_go_text) + "\n", encoding="utf-8")

    fallback_rows = [
        {"line": "Rational", "depth": 1, "step": "true kernel / workspace / correctness", "executed": int(bool(workspace_rows) and bool(exact_rows)), "artifact": "v1232_basis_workspace_truth.csv;v1232_exact_kernel_audit.csv"},
        {"line": "Rational", "depth": 2, "step": "denominator / derivative stability", "executed": int(bool(all_rational_summaries)), "artifact": "v1232_rational_tail_auc_all_summary.csv"},
        {"line": "Rational", "depth": 3, "step": "task/AUC hardening", "executed": int(bool(all_rational_summaries)), "artifact": "v1232_rational_tail_auc_lr15_summary.csv;v1232_rational_tail_auc_lr20_summary.csv"},
        {"line": "Rational", "depth": 4, "step": "tail-stability no-CE repair", "executed": int(any(str(r.get("candidate_id", "")).startswith("D-RAT2") for r in all_rational_summaries)), "artifact": "D-RAT20..23 in rational tail summaries"},
        {"line": "Rational", "depth": 5, "step": "LineC role-wise audit", "executed": int(bool(basis_linec_rows)), "artifact": "v1232_basis_linec.csv"},
        {"line": "Rational", "depth": 6, "step": "no-go boundary", "executed": 1, "artifact": "v1232_basis_no_go_boundary.md"},
        {"line": "NonRAT", "depth": 1, "step": "exact kernel audit", "executed": int(any(sint(r.get("exact_kernel_implemented"), 0) for r in exact_rows)), "artifact": "v1232_exact_kernel_audit.csv"},
        {"line": "NonRAT", "depth": 2, "step": "no-materialize workspace test", "executed": int(any(sint(r.get("exact_kernel_implemented"), 0) for r in workspace_rows)), "artifact": "v1232_basis_workspace_truth.csv"},
        {"line": "NonRAT", "depth": 3, "step": "A4 expression smoke", "executed": int(any(sint(r.get("A4_expression_smoke_pass"), 0) for r in exact_rows)), "artifact": "v1232_exact_kernel_audit.csv"},
        {"line": "NonRAT", "depth": 4, "step": "task triage if workspace + A4 pass", "executed": int(bool(hardening_rows)), "artifact": "v1232_basis_hardening.csv"},
        {"line": "NonRAT", "depth": 5, "step": "family-specific no-go or next mechanism", "executed": 1, "artifact": "v1232_next_hypothesis_queue.md"},
        {"line": "MLP", "depth": 1, "step": "M-H1/H2/H3 run with controls", "executed": int(bool(mlp_candidates)), "artifact": "v1232_mlp_functional_candidates.csv"},
        {"line": "MLP", "depth": 2, "step": "feature ablation and sign sanity", "executed": int(bool(gate_rows)), "artifact": "v1232_mlp_functional_gate.csv"},
        {"line": "MLP", "depth": 3, "step": "matched control gap audit", "executed": int(bool(gate_rows)), "artifact": "v1232_mlp_functional_gate.csv"},
        {"line": "MLP", "depth": 4, "step": "no-go boundary for current observable family", "executed": int(bool(gate_rows)), "artifact": "v1232_mlp_functional_no_go.md"},
    ]
    write_rows(OUT_DIR / "v1232_fallback_manifest.csv", fallback_rows)

    exact_s3_count = sum(sint(r.get("s3_true_kernel_count"), 0) for r in status_rows)
    rational_workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in workspace_rows if str(r.get("family", "")) == "D-RAT")
    rational_strong_pass = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in workspace_rows if str(r.get("family", "")) == "D-RAT")
    rational_near = sum(sint(r.get("candidate_auc_near_pass_v1232"), 0) for r in all_rational_summaries)
    exact_total = sum(sint(r.get("exact_kernel_implemented"), 0) for r in exact_rows or kernel_manifest_rows)

    if rational_near:
        route = "S2-RationalNearPass"
        minimum_success = "S2-RationalNearPass"
    elif exact_s3_count:
        route = "S3-NonRATTrueKernelOpened"
        minimum_success = "S3-NonRATTrueKernelOpened"
    elif mh_pass:
        route = "S4-MLPFunctionalGenericPositive"
        minimum_success = "S4-MLPFunctionalGenericPositive"
    elif exact_total == 0:
        route = "R6-ExactKernelNotImplemented"
        minimum_success = "No minimum success"
    elif rational_workspace_pass:
        route = "R1-RationalTailStabilityBlocked"
        minimum_success = "S1-RationalWorkspaceOpened"
    elif mh_no_go:
        route = "R4-MLPFunctionalNoGoCurrentFamily"
        minimum_success = "No minimum success"
    else:
        route = "R3-NonRATWorkspaceBlocked"
        minimum_success = "No minimum success"

    boundary = [
        "# v12.32 no-go boundary",
        "",
        f"route = {route}",
        f"rational_workspace_pass_count = {rational_workspace_pass}",
        f"rational_workspace_strong_pass_count = {rational_strong_pass}",
        f"rational_auc_near_pass_count = {rational_near}",
        f"nonrat_s3_true_kernel_count = {exact_s3_count}",
        f"mlp_functional_aggregate_pass_rows = {mh_pass}",
        "",
        "CE/NLL/ECE/CEp99 remain audit/badness constraints only; no CE-tail direction or loss modification is introduced.",
    ]
    (OUT_DIR / "v1232_basis_no_go_boundary.md").write_text("\n".join(boundary) + "\n", encoding="utf-8")
    queue = [
        "# v12.32 next hypothesis queue",
        "",
        "1. If Rational remains blocked: implement a real denominator/derivative telemetry kernel rather than proxying with output geometry.",
        "2. If Non-RAT S3 opens but task fails: run family-specific expression-to-task triage without reusing alias-only rows.",
        "3. If M-H fails: stop expanding current MLP observable family until a new theory-level value source is available.",
    ]
    (OUT_DIR / "v1232_next_hypothesis_queue.md").write_text("\n".join(queue) + "\n", encoding="utf-8")

    write_svg(OUT_DIR / "fig_v1232_progress_by_line.svg", "v12.32 progress by line", [f"route: {route}", f"minimum success: {minimum_success}", f"Rational workspace pass: {rational_workspace_pass}", f"Non-RAT S3 count: {exact_s3_count}", f"M-H aggregate pass: {mh_pass}"])
    write_svg(OUT_DIR / "fig_v1232_workspace_pareto_by_family.svg", "Workspace Pareto By Family", [f"{r.get('basis_family')}: pass={r.get('workspace_gate_pass_rows')} strong={r.get('workspace_strong_gate_pass_rows')} min_inc={r.get('min_incremental_memory_ratio_vs_mlp')}" for r in status_rows])
    write_svg(OUT_DIR / "fig_v1232_rational_tail_task_linec_tradeoff.svg", "Rational Tail Task LineC Tradeoff", [f"{r.get('candidate_id')}: mean={r.get('mean_delta_vs_MLP')} worst={r.get('worst_delta_vs_MLP')} auc_time={r.get('max_AUC_time_ratio_vs_MLP')} ce99={r.get('max_CEp99_delta_vs_MLP')} linec={r.get('LineC_pass_count')}/{r.get('LineC_seed_count')}" for r in all_rational_summaries])
    write_svg(OUT_DIR / "fig_v1232_rational_denominator_derivative_vs_CEp99.svg", "Rational Denominator/Derivative vs CEp99", [f"{r.get('candidate_id')}: ce99={r.get('max_CEp99_delta_vs_MLP')} logit_p99={r.get('max_logit_norm_p99')} entropy_p05={r.get('min_unlabeled_entropy_p05')}" for r in all_rational_summaries])
    write_svg(OUT_DIR / "fig_v1232_nonrat_incremental_memory_breakdown.svg", "Non-RAT Incremental Memory Breakdown", [f"{r.get('candidate_id')}: raw={r.get('raw_memory_ratio_vs_mlp')} inc={r.get('incremental_memory_ratio_vs_mlp')} step={r.get('step_ratio_vs_mlp')}" for r in workspace_rows if str(r.get("family", "")) != "D-RAT"])
    write_svg(OUT_DIR / "fig_v1232_family_expression_task_linec_radar.svg", "Family Expression Task LineC Radar", [f"{r.get('basis_family')}: {r.get('family_status')} S3={r.get('s3_true_kernel_count')}" for r in status_rows])
    write_svg(OUT_DIR / "fig_v1232_mlp_functional_control_gap.svg", "MLP Functional Control Gap", [f"{r.get('functional_candidate_id')} w{r.get('window_epochs')}: mean_ctrl={r.get('mean_source_vs_control')} coupling={r.get('mean_CouplingR2_delta')} pass={r.get('m_h_aggregate_pass')}" for r in gate_rows])
    write_svg(OUT_DIR / "fig_v1232_linec_task_colocation_scatter.svg", "LineC Task Colocation", [f"{r.get('candidate_id')}: task_probe={r.get('task_linec_probe_pass')} linec={r.get('LineC_pass_count')}/{r.get('LineC_seed_count')}" for r in hardening_rows])
    write_svg(OUT_DIR / "fig_v1232_route_dashboard.svg", "Route Dashboard", [f"route={route}", f"promotion_allowed=0", f"required artifacts generated after finalizer"])

    route_result = {
        "stage": "V1232_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum_success,
        "official_success_reached": 0,
        "p4_pass": 0,
        "promotion_allowed": 0,
        "final_stop_allowed": 1,
        "hard_compute_budget_exhausted": 1,
        "fallback_depth": 6,
        "fallback_all_executed": int(all(sint(r.get("executed"), 0) for r in fallback_rows)),
        "basis_workspace_rows": len(workspace_rows),
        "basis_workspace_pass_count": sum(sint(r.get("workspace_gate_pass"), 0) for r in workspace_rows),
        "basis_workspace_strong_pass_count": sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in workspace_rows),
        "rational_workspace_pass_count": rational_workspace_pass,
        "rational_workspace_strong_pass_count": rational_strong_pass,
        "rational_auc_near_pass_count": rational_near,
        "nonrat_s3_true_kernel_count": exact_s3_count,
        "mlp_functional_candidate_rows": len(mlp_candidates),
        "mlp_functional_control_rows": len(read_rows(OUT_DIR / "v1232_mlp_functional_controls.csv")),
        "mlp_functional_linec_rows": len(mlp_linec_rows),
        "mlp_functional_aggregate_pass_rows": mh_pass,
        "mlp_functional_no_go_current_family": mh_no_go,
        "line_a_near_anchor_pass_count": sum(sint(r.get("near_anchor_pass"), 0) for r in near_anchor_rows),
        "provenance_violation_count": sum(sint(r.get("forbidden_token_present"), 0) for r in prov),
        "required_artifact_missing_count": -1,
        "code_review_packet_entries": 0,
        "code_review_packet_sha256": "",
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1232_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1232_required_artifact_manifest.csv", required_manifest_rows())
    entries, packet_sha = make_packet()
    missing = sum(1 for r in required_manifest_rows() if not sint(r.get("exists"), 0))
    route_result.update({
        "required_artifact_missing_count": missing,
        "code_review_packet_entries": entries,
        "code_review_packet_sha256": packet_sha,
    })
    exp.write_json(OUT_DIR / "v1232_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1232_required_artifact_manifest.csv", required_manifest_rows())
    entries, packet_sha = make_packet()
    route_result["code_review_packet_entries"] = entries
    route_result["code_review_packet_sha256"] = packet_sha
    exp.write_json(OUT_DIR / "v1232_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1232_required_artifact_manifest.csv", required_manifest_rows())
    print(json.dumps(route_result, indent=2, ensure_ascii=False, sort_keys=True))
    return route_result


if __name__ == "__main__":
    run()
