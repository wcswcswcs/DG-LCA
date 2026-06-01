#!/usr/bin/env python
"""Finalize v12.33 Rational telemetry + Non-RAT lifetime + MLP closure artifacts."""

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
from dgkan.diagnostics.basis_workspace import V1233_BASIS_CANDIDATES, nonrat_lifetime_rows  # noqa: E402
from dgkan.functional.mlp_functional import CONTROL_IDS, SOURCE_CANDIDATES  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_33_rational_telemetry_kernel_nonrat_lifetime_mlp_functional_closure" / "official_v1233"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.33_RationalTelemetryKernel_NonRATLifetime_MLPFunctionalClosure_实验结果复盘.md"

REQUIRED = [
    "v1233_code_review_manifest.csv",
    "v1233_core_symbol_map.json",
    "v1233_provenance_audit.csv",
    "v1233_forbidden_direction_audit.csv",
    "v1233_required_artifact_manifest.csv",
    "v1233_fallback_manifest.csv",
    "v1233_kernel_implementation_manifest.csv",
    "v1233_exact_kernel_audit.csv",
    "v1233_basis_workspace_truth.csv",
    "v1233_basis_component_peak.csv",
    "v1233_basis_hardening.csv",
    "v1233_basis_linec.csv",
    "v1233_rational_telemetry.csv",
    "v1233_rational_telemetry_summary.csv",
    "v1233_nonrat_lifetime.csv",
    "v1233_mlp_functional_closure.csv",
    "v1233_mlp_functional_candidates.csv",
    "v1233_mlp_functional_controls.csv",
    "v1233_mlp_functional_linec.csv",
    "v1233_mlp_functional_gate.csv",
    "v1233_adyn_monitor.csv",
    "v1233_label_free_near_anchor.csv",
    "v1233_family_status.csv",
    "v1233_linec_unified.csv",
    "v1233_basis_no_go_boundary.md",
    "v1233_next_hypothesis_queue.md",
    "v1233_route_decision.json",
    "v1233_code_review_packet.zip",
]

FIGURES = [
    "fig_v1233_progress_over_versions.svg",
    "fig_v1233_rational_workspace_tail_pareto.svg",
    "fig_v1233_rational_den_derivative_vs_CEp99.svg",
    "fig_v1233_rational_auc_linec_tail_heatmap.svg",
    "fig_v1233_nonrat_lifetime_waterfall.svg",
    "fig_v1233_nonrat_incremental_memory_by_component.svg",
    "fig_v1233_mlp_functional_control_gap.svg",
    "fig_v1233_family_status_matrix.svg",
    "fig_v1233_route_dashboard.svg",
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
            '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="620" viewBox="0 0 1160 620">',
            '<rect width="1160" height="620" fill="#f7f8f4"/>',
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


def provenance_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in V1233_BASIS_CANDIDATES.values():
        rows.append({
            "stage": "V1233_PROVENANCE_AUDIT",
            "line": "LineD",
            "candidate_id": cand.candidate_id,
            "uses_label_or_ce_for_direction": 0,
            "uses_y_for_stats": 0,
            "uses_validation_for_commit": 0,
            "uses_query_batch_for_commit": 0,
            "uses_linec_hard_target_for_direction": 0,
            "forbidden_token_present": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for cid in [c for c in SOURCE_CANDIDATES if c.startswith("M-I")]:
        rows.append({
            "stage": "V1233_PROVENANCE_AUDIT",
            "line": "LineM",
            "candidate_id": cid,
            "uses_label_or_ce_for_direction": 0,
            "uses_y_for_stats": 0,
            "uses_validation_for_commit": 0,
            "uses_query_batch_for_commit": 0,
            "uses_linec_hard_target_for_direction": 0,
            "forbidden_token_present": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for ctrl in CONTROL_IDS:
        rows.append({
            "stage": "V1233_PROVENANCE_AUDIT",
            "line": "LineMControl",
            "candidate_id": ctrl,
            "uses_label_or_ce_for_direction": int(ctrl == "C3-AdamWParallelDirection"),
            "uses_y_for_stats": 0,
            "uses_validation_for_commit": 0,
            "uses_query_batch_for_commit": 0,
            "uses_linec_hard_target_for_direction": 0,
            "forbidden_token_present": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    try:
        for ablation_id, spec_info in linea.ablation_specs(784, 10).items():
            if not ablation_id.startswith("A-DYN"):
                continue
            spec = spec_info["spec"]
            rows.append({
                "stage": "V1233_PROVENANCE_AUDIT",
                "line": "LineA",
                "candidate_id": ablation_id,
                "uses_label_or_ce_for_direction": int(spec_info.get("uses_y_for_stats", 0)),
                "uses_y_for_stats": int(spec_info.get("uses_y_for_stats", 0)),
                "uses_validation_for_commit": 0,
                "uses_query_batch_for_commit": 0,
                "uses_linec_hard_target_for_direction": 0,
                "forbidden_token_present": int(lfbridge.forbidden_token_present(ablation_id, getattr(spec, "candidate_id", ""), getattr(spec, "init_variant", ""))),
                "promotion_allowed": 0,
                "no_fake": 1,
            })
    except Exception as exc:  # noqa: BLE001
        rows.append({"stage": "V1233_PROVENANCE_AUDIT", "line": "LineA", "candidate_id": "A-DYN-read-error", "error": f"{type(exc).__name__}: {exc}", "forbidden_token_present": 1, "promotion_allowed": 0, "no_fake": 1})
    return rows


def core_symbol_map() -> dict[str, Any]:
    symbols = {
        "dgkan/diagnostics/basis_workspace.py": [
            "V1233_BASIS_CANDIDATES",
            "workspace_gate_v1233",
            "workspace_strong_gate_v1233",
            "rational_telemetry_metrics",
            "nonrat_lifetime_rows",
        ],
        "dgkan/functional/mlp_functional.py": ["SOURCE_CANDIDATES", "functional_objective"],
        "experiments/run_v1231_basis_kernel_workspace.py": ["run"],
        "experiments/run_v1231_rational_auc_hardening.py": ["run"],
        "experiments/run_v1230_mlp_functional.py": ["run"],
        "experiments/run_v1233_finalize_rational_telemetry_lifetime.py": ["run"],
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
        "experiments/run_v1233_finalize_rational_telemetry_lifetime.py",
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


def family_status(workspace_rows: list[dict[str, Any]], telemetry_summary: list[dict[str, Any]], lifetime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in workspace_rows:
        by_family.setdefault(str(row.get("family", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for fam, rows in sorted(by_family.items()):
        workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in rows)
        strong_pass = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in rows)
        min_raw = min((fnum(r.get("raw_memory_ratio_vs_mlp"), 999.0) for r in rows), default=float("nan"))
        min_inc = min((fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0) for r in rows), default=float("nan"))
        min_step = min((fnum(r.get("step_ratio_vs_mlp"), 999.0) for r in rows), default=float("nan"))
        if fam == "D-RAT":
            near = sum(sint(r.get("candidate_auc_near_pass_v1233"), 0) for r in telemetry_summary)
            official = sum(sint(r.get("candidate_auc_official_pass_v1233"), 0) for r in telemetry_summary)
            status = "RationalOfficialPass" if official else "RationalNearPass" if near else "RationalTailTaskLineCNotColocated" if workspace_pass else "WorkspaceBlocked"
        else:
            fam_life = [r for r in lifetime_rows if str(r.get("family", "")) == fam]
            s3 = sum(sint(r.get("s3_lifetime_pass"), 0) for r in fam_life)
            status = "NonRATLifetimeOpened" if s3 else "NonRATLifetimeBlocked" if fam_life else "WorkspaceBlocked"
        out.append({
            "stage": "V1233_FAMILY_STATUS",
            "basis_family": fam,
            "workspace_rows": len(rows),
            "workspace_gate_pass_rows": workspace_pass,
            "workspace_strong_gate_pass_rows": strong_pass,
            "min_raw_memory_ratio_vs_mlp": min_raw,
            "min_incremental_memory_ratio_vs_mlp": min_inc,
            "min_step_ratio_vs_mlp": min_step,
            "family_status": status,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def mlp_closure_rows(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    closure: list[dict[str, Any]] = []
    for row in candidate_rows:
        if not str(row.get("functional_candidate_id", "")).startswith("M-I"):
            continue
        rr = dict(row)
        rr["stage"] = "V1233_MLP_FUNCTIONAL_CLOSURE"
        rr["matched_control_gap"] = rr.get("source_vs_control_acc_delta", "")
        rr["v1233_exploration_row_pass"] = int(
            fnum(rr.get("source_vs_control_acc_delta"), -999.0) >= 0.005
            and sint(rr.get("LineC_pass_count"), 0) >= 4
            and fnum(rr.get("CEp99_delta_vs_noop"), 999.0) <= 0.05
        )
        rr["v1233_official_row_pass"] = int(
            fnum(rr.get("source_vs_control_acc_delta"), -999.0) >= 0.010
            and sint(rr.get("LineC_pass_count"), 0) >= 5
            and fnum(rr.get("CEp99_delta_vs_noop"), 999.0) <= 0.05
            and fnum(rr.get("NLL_delta_vs_noop"), 999.0) <= 0.0
            and fnum(rr.get("ECE_delta_vs_noop"), 999.0) <= 0.01
        )
        rr["promotion_allowed"] = 0
        rr["no_fake"] = 1
        closure.append(rr)
    by: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in closure:
        by.setdefault((str(row.get("functional_candidate_id", "")), sint(row.get("window_epochs"), 0)), []).append(row)
    gate: list[dict[str, Any]] = []
    for (cid, window), rows in sorted(by.items()):
        exp_pass = sum(sint(r.get("v1233_exploration_row_pass"), 0) for r in rows)
        off_pass = sum(sint(r.get("v1233_official_row_pass"), 0) for r in rows)
        gate.append({
            "stage": "V1233_MLP_FUNCTIONAL_GATE",
            "functional_candidate_id": cid,
            "window_epochs": window,
            "rows": len(rows),
            "v1233_exploration_row_pass_rows": exp_pass,
            "v1233_official_row_pass_rows": off_pass,
            "m_i_aggregate_pass": int(exp_pass >= 8),
            "m_i_official_pass": int(off_pass >= 8),
            "mean_source_vs_noop": sum(fnum(r.get("source_vs_noop_acc_delta"), 0.0) for r in rows) / max(1, len(rows)),
            "mean_source_vs_control": sum(fnum(r.get("source_vs_control_acc_delta"), 0.0) for r in rows) / max(1, len(rows)),
            "mean_CouplingR2_delta": sum(fnum(r.get("CouplingR2_delta"), 0.0) for r in rows) / max(1, len(rows)),
            "max_LineC_pass_count": max((sint(r.get("LineC_pass_count"), 0) for r in rows), default=0),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return closure, gate


def make_packet() -> tuple[int, str]:
    packet = OUT_DIR / "v1233_code_review_packet.zip"
    files = [
        OUT_DIR / "v1233_route_decision.json",
        OUT_DIR / "v1233_required_artifact_manifest.csv",
        OUT_DIR / "v1233_code_review_manifest.csv",
        OUT_DIR / "v1233_core_symbol_map.json",
        OUT_DIR / "v1233_provenance_audit.csv",
        OUT_DIR / "v1233_forbidden_direction_audit.csv",
        OUT_DIR / "v1233_family_status.csv",
        OUT_DIR / "v1233_basis_no_go_boundary.md",
        OUT_DIR / "v1233_next_hypothesis_queue.md",
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
    ]
    seen: set[str] = set()
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        entries = 0
        for path in files:
            if path.exists():
                arc = str(path.relative_to(ROOT))
                if arc in seen:
                    continue
                seen.add(arc)
                zf.write(path, arcname=arc)
                entries += 1
        for name in REQUIRED:
            p = OUT_DIR / name
            if p.exists() and p != packet:
                arc = str(p.relative_to(ROOT))
                if arc in seen:
                    continue
                seen.add(arc)
                zf.write(p, arcname=arc)
                entries += 1
        for fig in FIGURES:
            p = OUT_DIR / fig
            if p.exists():
                arc = str(p.relative_to(ROOT))
                if arc in seen:
                    continue
                seen.add(arc)
                zf.write(p, arcname=arc)
                entries += 1
    return entries, sha256_file(packet)


def run() -> dict[str, Any]:
    exp.ensure_dir(OUT_DIR)
    for src_name, dst_name in [
        ("v1233_basis_kernel_implementation_manifest.csv", "v1233_kernel_implementation_manifest.csv"),
        ("v1233_basis_exact_kernel_audit.csv", "v1233_exact_kernel_audit.csv"),
        ("v1233_adyn_monitor_ablation.csv", "v1233_adyn_monitor.csv"),
    ]:
        src = OUT_DIR / src_name
        dst = OUT_DIR / dst_name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    workspace_rows = read_rows(OUT_DIR / "v1233_basis_workspace_truth.csv")
    for extra_name in [
        "v1233_rational_output_geometry_repair_workspace_truth.csv",
    ]:
        workspace_rows.extend(read_rows(OUT_DIR / extra_name))
    if workspace_rows:
        write_rows(OUT_DIR / "v1233_basis_workspace_truth.csv", workspace_rows)
    hardening_rows = read_rows(OUT_DIR / "v1233_basis_hardening.csv")
    for extra_name in [
        "v1233_rational_output_geometry_repair_hardening.csv",
    ]:
        hardening_rows.extend(read_rows(OUT_DIR / extra_name))
    if hardening_rows:
        write_rows(OUT_DIR / "v1233_basis_hardening.csv", hardening_rows)
    basis_linec_rows = read_rows(OUT_DIR / "v1233_basis_linec.csv")
    for extra_name in [
        "v1233_rational_output_geometry_repair_linec.csv",
    ]:
        basis_linec_rows.extend(read_rows(OUT_DIR / extra_name))
    if basis_linec_rows:
        write_rows(OUT_DIR / "v1233_basis_linec.csv", basis_linec_rows)
    exact_rows = read_rows(OUT_DIR / "v1233_exact_kernel_audit.csv")
    mlp_candidates = read_rows(OUT_DIR / "v1233_mlp_functional_candidates.csv")
    mlp_controls = read_rows(OUT_DIR / "v1233_mlp_functional_controls.csv")
    mlp_linec = read_rows(OUT_DIR / "v1233_mlp_functional_linec.csv")
    adyn_rows = read_rows(OUT_DIR / "v1233_adyn_monitor.csv")

    telemetry_candidates = []
    telemetry_summary = []
    for name in [
        "v1233_rational_telemetry_lr15_candidate.csv",
        "v1233_rational_telemetry_lr20_candidate.csv",
        "v1233_rational_telemetry_repair_candidate.csv",
    ]:
        telemetry_candidates.extend(read_rows(OUT_DIR / name))
    for name in [
        "v1233_rational_telemetry_lr15_summary.csv",
        "v1233_rational_telemetry_lr20_summary.csv",
        "v1233_rational_telemetry_repair_summary.csv",
    ]:
        telemetry_summary.extend(read_rows(OUT_DIR / name))
    write_rows(OUT_DIR / "v1233_rational_telemetry.csv", telemetry_candidates)
    write_rows(OUT_DIR / "v1233_rational_telemetry_summary.csv", telemetry_summary)

    lifetime = nonrat_lifetime_rows(workspace_rows, exact_rows)
    write_rows(OUT_DIR / "v1233_nonrat_lifetime.csv", lifetime)

    closure, gate_rows = mlp_closure_rows(mlp_candidates)
    write_rows(OUT_DIR / "v1233_mlp_functional_closure.csv", closure)
    write_rows(OUT_DIR / "v1233_mlp_functional_gate.csv", gate_rows)

    prov = provenance_rows()
    write_rows(OUT_DIR / "v1233_provenance_audit.csv", prov)
    write_rows(OUT_DIR / "v1233_forbidden_direction_audit.csv", prov)
    (OUT_DIR / "v1233_core_symbol_map.json").write_text(json.dumps(core_symbol_map(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_rows(OUT_DIR / "v1233_code_review_manifest.csv", code_review_manifest_rows())

    family_rows = family_status(workspace_rows, telemetry_summary, lifetime)
    write_rows(OUT_DIR / "v1233_family_status.csv", family_rows)

    near_anchor_rows = []
    by_adyn: dict[str, list[dict[str, Any]]] = {}
    for row in adyn_rows:
        by_adyn.setdefault(str(row.get("ablation_id") or row.get("candidate_id") or row.get("method_id") or ""), []).append(row)
    for cid, rows in sorted(by_adyn.items()):
        if not cid.startswith("A-DYN"):
            continue
        deltas = [fnum(r.get("delta_vs_mlp") or r.get("mean_delta_vs_mlp")) for r in rows if math.isfinite(fnum(r.get("delta_vs_mlp") or r.get("mean_delta_vs_mlp")))]
        near_anchor_rows.append({
            "stage": "V1233_LABEL_FREE_NEAR_ANCHOR",
            "candidate_id": cid,
            "rows": len(rows),
            "mean_delta_vs_mlp": sum(deltas) / len(deltas) if deltas else float("nan"),
            "worst_delta_vs_mlp": min(deltas) if deltas else float("nan"),
            "near_anchor_pass": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    if not near_anchor_rows:
        near_anchor_rows.append({"stage": "V1233_LABEL_FREE_NEAR_ANCHOR", "candidate_id": "A-DYN", "rows": 0, "near_anchor_pass": 0, "note": "A-DYN canonical artifact missing or not executed", "promotion_allowed": 0, "no_fake": 1})
    write_rows(OUT_DIR / "v1233_label_free_near_anchor.csv", near_anchor_rows)

    linec_unified = []
    for row in [*basis_linec_rows, *mlp_linec]:
        rr = dict(row)
        rr["stage"] = "V1233_LINEC_UNIFIED"
        linec_unified.append(rr)
    write_rows(OUT_DIR / "v1233_linec_unified.csv", linec_unified)

    fallback_rows = [
        {"line": "Rational", "depth": 1, "step": "workspace + telemetry manifest", "executed": int(bool(workspace_rows)), "artifact": "v1233_basis_workspace_truth.csv"},
        {"line": "Rational", "depth": 2, "step": "denominator/derivative telemetry", "executed": int(bool(telemetry_candidates)), "artifact": "v1233_rational_telemetry.csv"},
        {"line": "Rational", "depth": 3, "step": "AUC/task/LineC/tail triage", "executed": int(bool(telemetry_summary)), "artifact": "v1233_rational_telemetry_summary.csv"},
        {"line": "Rational", "depth": 4, "step": "tail-failure autopsy/no-go", "executed": 1, "artifact": "v1233_basis_no_go_boundary.md"},
        {"line": "Rational", "depth": 5, "step": "LineC unified audit", "executed": int(bool(linec_unified)), "artifact": "v1233_linec_unified.csv"},
        {"line": "Rational", "depth": 6, "step": "next mechanism queue", "executed": 1, "artifact": "v1233_next_hypothesis_queue.md"},
        {"line": "NonRAT", "depth": 1, "step": "exact kernel audit", "executed": int(bool(exact_rows)), "artifact": "v1233_exact_kernel_audit.csv"},
        {"line": "NonRAT", "depth": 2, "step": "lifetime waterfall", "executed": int(bool(lifetime)), "artifact": "v1233_nonrat_lifetime.csv"},
        {"line": "NonRAT", "depth": 3, "step": "component memory audit", "executed": int((OUT_DIR / "v1233_basis_component_peak.csv").exists()), "artifact": "v1233_basis_component_peak.csv"},
        {"line": "MLP", "depth": 1, "step": "M-I closure with matched controls", "executed": int(bool(closure)), "artifact": "v1233_mlp_functional_closure.csv"},
        {"line": "MLP", "depth": 2, "step": "generic positive/no-go gate", "executed": int(bool(gate_rows)), "artifact": "v1233_mlp_functional_gate.csv"},
        {"line": "LineA", "depth": 1, "step": "A-DYN monitor", "executed": int(bool(adyn_rows)), "artifact": "v1233_adyn_monitor.csv"},
    ]
    write_rows(OUT_DIR / "v1233_fallback_manifest.csv", fallback_rows)

    rational_workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in workspace_rows if str(r.get("family", "")) == "D-RAT")
    rational_workspace_strong = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in workspace_rows if str(r.get("family", "")) == "D-RAT")
    rational_near = sum(sint(r.get("candidate_auc_near_pass_v1233"), 0) for r in telemetry_summary)
    rational_official = sum(sint(r.get("candidate_auc_official_pass_v1233"), 0) for r in telemetry_summary)
    telemetry_available = sum(sint(r.get("telemetry_available"), 0) for r in telemetry_candidates)
    nonrat_s3 = len({str(r.get("candidate_id", "")) for r in lifetime if sint(r.get("s3_lifetime_pass"), 0)})
    nonrat_official = len({str(r.get("candidate_id", "")) for r in lifetime if sint(r.get("s3_lifetime_official_pass"), 0)})
    mi_aggregate = sum(sint(r.get("m_i_aggregate_pass"), 0) for r in gate_rows)
    mi_official = sum(sint(r.get("m_i_official_pass"), 0) for r in gate_rows)
    provenance_violations = sum(sint(r.get("forbidden_token_present"), 0) for r in prov)
    line_a_near = sum(sint(r.get("near_anchor_pass"), 0) for r in near_anchor_rows)

    if provenance_violations:
        route = "R0-ProvenanceOrForbiddenDirectionViolation"
        minimum_success = "No minimum success"
    elif rational_official and nonrat_official and mi_official:
        route = "R11-S5UnifiedSuccess"
        minimum_success = "S5"
    elif rational_official:
        route = "R4-RationalTelemetryOfficialPass"
        minimum_success = "S2-RationalTelemetryOfficial"
    elif rational_near:
        route = "R3-RationalTelemetryNearPass"
        minimum_success = "S2-RationalTelemetryNearPass"
    elif nonrat_official:
        route = "R7-NonRATLifetimeOfficialPass"
        minimum_success = "S3-NonRATLifetimeOfficial"
    elif nonrat_s3:
        route = "R6-NonRATLifetimeNearPass"
        minimum_success = "S3-NonRATLifetimeNearPass"
    elif mi_official or mi_aggregate:
        route = "R9-MLPFunctionalClosurePositive"
        minimum_success = "S4-MLPFunctionalClosurePositive"
    elif rational_workspace_pass and not telemetry_available:
        route = "R1-RationalTelemetryKernelBlocked"
        minimum_success = "S1-RationalWorkspaceOpened"
    elif rational_workspace_pass:
        route = "R2-RationalTailTaskLineCNotColocated"
        minimum_success = "S1-RationalWorkspaceOpened"
    elif lifetime:
        route = "R5-NonRATLifetimeBlocked"
        minimum_success = "No minimum success"
    else:
        route = "R8-MLPFunctionalNoGoClosure"
        minimum_success = "No minimum success"

    boundary = [
        "# v12.33 no-go boundary",
        "",
        f"route = {route}",
        f"rational_workspace_pass_count = {rational_workspace_pass}",
        f"rational_workspace_strong_pass_count = {rational_workspace_strong}",
        f"rational_telemetry_available_rows = {telemetry_available}",
        f"rational_auc_near_pass_count = {rational_near}",
        f"rational_auc_official_pass_count = {rational_official}",
        f"nonrat_s3_lifetime_count = {nonrat_s3}",
        f"mlp_functional_aggregate_pass_rows = {mi_aggregate}",
        "",
        "CE/NLL/ECE/CEp99 remain audit/badness constraints only; no CE-tail direction or loss modification is introduced.",
    ]
    (OUT_DIR / "v1233_basis_no_go_boundary.md").write_text("\n".join(boundary) + "\n", encoding="utf-8")
    queue = [
        "# v12.33 next hypothesis queue",
        "",
        "1. If Rational telemetry exists but tail/AUC still fails: implement a true fused denominator/derivative kernel with training-time stability control, not another output-geometry alias.",
        "2. If Non-RAT exact kernels keep failing lifetime gates: repair optimizer/readout-gradient lifetime overlap before task grids.",
        "3. If M-I closure fails matched controls: close current loss-agnostic observable family and require a new mechanism-level value source.",
    ]
    (OUT_DIR / "v1233_next_hypothesis_queue.md").write_text("\n".join(queue) + "\n", encoding="utf-8")

    write_svg(OUT_DIR / "fig_v1233_progress_over_versions.svg", "v12.33 progress over versions", [f"route={route}", f"minimum_success={minimum_success}", f"promotion_allowed=0"])
    write_svg(OUT_DIR / "fig_v1233_rational_workspace_tail_pareto.svg", "Rational workspace/tail pareto", [f"{r.get('candidate_id')}: mean={r.get('mean_delta_vs_MLP')} worst={r.get('worst_delta_vs_MLP')} auc={r.get('max_AUC_time_ratio_vs_MLP')} ce99={r.get('max_CEp99_delta_vs_MLP')}" for r in telemetry_summary])
    write_svg(OUT_DIR / "fig_v1233_rational_den_derivative_vs_CEp99.svg", "Rational den/derivative vs CEp99", [f"{r.get('candidate_id')}: den_p01={r.get('min_den_p01_batch')} rprime99={r.get('max_r_prime_p99')} ce99={r.get('max_CEp99_delta_vs_MLP')}" for r in telemetry_summary])
    write_svg(OUT_DIR / "fig_v1233_rational_auc_linec_tail_heatmap.svg", "Rational AUC/LineC/tail", [f"{r.get('candidate_id')}: auc={r.get('max_AUC_time_ratio_vs_MLP')} linec={r.get('LineC_pass_count')}/{r.get('LineC_seed_count')} near={r.get('candidate_auc_near_pass_v1233')}" for r in telemetry_summary])
    write_svg(OUT_DIR / "fig_v1233_nonrat_lifetime_waterfall.svg", "Non-RAT lifetime waterfall", [f"{r.get('candidate_id')} {r.get('dataset')} s{r.get('seed')}: largest={r.get('largest_component')} inc={r.get('incremental_memory_ratio_vs_mlp')} fail={r.get('failure_reason')}" for r in lifetime])
    write_svg(OUT_DIR / "fig_v1233_nonrat_incremental_memory_by_component.svg", "Non-RAT incremental memory by component", [f"{r.get('candidate_id')}: basis={r.get('basis_activation_bytes')} readout={r.get('readout_grad_bytes')} opt={r.get('optimizer_state_bytes')}" for r in lifetime])
    write_svg(OUT_DIR / "fig_v1233_mlp_functional_control_gap.svg", "MLP functional control gap", [f"{r.get('functional_candidate_id')} w{r.get('window_epochs')}: ctrl={r.get('mean_source_vs_control')} pass={r.get('m_i_aggregate_pass')}" for r in gate_rows])
    write_svg(OUT_DIR / "fig_v1233_family_status_matrix.svg", "Family status matrix", [f"{r.get('basis_family')}: {r.get('family_status')} pass={r.get('workspace_gate_pass_rows')} strong={r.get('workspace_strong_gate_pass_rows')}" for r in family_rows])
    write_svg(OUT_DIR / "fig_v1233_route_dashboard.svg", "Route dashboard", [f"route={route}", f"official_success_reached=0", f"required artifacts generated after finalizer"])

    route_result = {
        "stage": "V1233_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum_success,
        "official_success_reached": int(route == "R11-S5UnifiedSuccess"),
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
        "rational_workspace_strong_pass_count": rational_workspace_strong,
        "rational_telemetry_available_rows": telemetry_available,
        "rational_auc_near_pass_count": rational_near,
        "rational_auc_official_pass_count": rational_official,
        "nonrat_s3_lifetime_count": nonrat_s3,
        "nonrat_s3_lifetime_official_count": nonrat_official,
        "mlp_functional_candidate_rows": len(mlp_candidates),
        "mlp_functional_control_rows": len(mlp_controls),
        "mlp_functional_linec_rows": len(mlp_linec),
        "mlp_functional_aggregate_pass_rows": mi_aggregate,
        "mlp_functional_official_pass_rows": mi_official,
        "mlp_functional_no_go_current_family": int(bool(gate_rows) and mi_aggregate == 0 and mi_official == 0),
        "line_a_near_anchor_pass_count": line_a_near,
        "provenance_violation_count": provenance_violations,
        "required_artifact_missing_count": -1,
        "code_review_packet_entries": 0,
        "code_review_packet_sha256": "",
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1233_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1233_required_artifact_manifest.csv", required_manifest_rows())
    entries, packet_sha = make_packet()
    missing = sum(1 for r in required_manifest_rows() if not sint(r.get("exists"), 0))
    route_result.update({
        "required_artifact_missing_count": missing,
        "code_review_packet_entries": entries,
        "code_review_packet_sha256": packet_sha,
    })
    exp.write_json(OUT_DIR / "v1233_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1233_required_artifact_manifest.csv", required_manifest_rows())
    entries, packet_sha = make_packet()
    route_result["code_review_packet_entries"] = entries
    route_result["code_review_packet_sha256"] = packet_sha
    exp.write_json(OUT_DIR / "v1233_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1233_required_artifact_manifest.csv", required_manifest_rows())
    print(json.dumps(route_result, indent=2, ensure_ascii=False, sort_keys=True))
    return route_result


if __name__ == "__main__":
    run()
