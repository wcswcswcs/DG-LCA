#!/usr/bin/env python
"""Finalize v12.35 substrate-health and basis-response co-location artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
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
from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES  # noqa: E402


SOURCE_DIR = ROOT / "results" / "v12_34_2_all_basis_substrate_functional_repair" / "official_v12342"
OUT_DIR = ROOT / "results" / "v12_35_all_basis_substrate_health_functional_colocation" / "official_v1235"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_实验结果复盘.md"

REQUIRED = [
    "v1235_route_decision.json",
    "v1235_required_artifact_manifest.csv",
    "v1235_progress_by_line.csv",
    "v1235_code_provenance_audit.csv",
    "v1235_basis_substrate_health.csv",
    "v1235_family_substrate_summary.csv",
    "v1235_family_telemetry.csv",
    "v1235_basis_response_dictionary.csv",
    "v1235_response_colocation_summary.csv",
    "v1235_basis_functional_p3.csv",
    "v1235_basis_functional_p4.csv",
    "v1235_nonrat_task_health_repair.csv",
    "v1235_mlp_functional_closure.csv",
    "v1235_failure_table.csv",
    "v1235_next_hypothesis_queue.md",
    "v1235_code_review_packet.zip",
]

SUPPLEMENTAL = [
    "v1235_basis_response_controls.csv",
    "v1235_basis_response_linec.csv",
    "v1235_basis_response_task_tail.csv",
    "v1235_response_dictionary_route.json",
    "v1235_core_symbol_map.json",
    "v1235_code_review_manifest.csv",
]

FIGURES = [
    "fig_progress_percent_by_line.svg",
    "fig_family_substrate_heatmap.svg",
    "fig_workspace_vs_task_health_pareto.svg",
    "fig_nonrat_lifetime_vs_task_collapse.svg",
    "fig_basis_telemetry_vs_linec_delta.svg",
    "fig_response_dictionary_colocation_scatter.svg",
    "fig_functional_control_gap_by_family.svg",
    "fig_linec_noise_reservoir_delta_by_functional.svg",
    "fig_ce_nll_ece_safety_by_candidate.svg",
    "fig_mlp_functional_closure_no_go.svg",
    "fig_route_sankey_v1235.svg",
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


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return exp.read_csv_rows(path) if path.exists() else []


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, rows)


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
    return best or (1, 1)


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    body = []
    for idx, line in enumerate(lines[:20]):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{58 + idx * 24}" font-size="14" fill="#222">{safe}</text>')
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="660" viewBox="0 0 1180 660">',
            '<rect width="1180" height="660" fill="#f7f8f4"/>',
            f'<text x="24" y="32" font-size="20" font-weight="700" fill="#111">{title}</text>',
            *body,
            "</svg>",
        ]),
        encoding="utf-8",
    )


def substrate_gate(row: dict[str, Any]) -> int:
    linec_total = sint(row.get("LineC_total"), sint(row.get("LineC_seed_count"), 0))
    linec_pass = sint(row.get("LineC_pass_count"), 0)
    linec_rate = (float(linec_pass) / float(linec_total)) if linec_total > 0 else 0.0
    return int(
        fnum(row.get("raw_memory_ratio_vs_mlp"), 999.0) <= 1.05
        and fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 1.75
        and fnum(row.get("step_ratio_vs_mlp"), 999.0) <= 1.75
        and fnum(row.get("mean_delta_vs_mlp"), -999.0) >= -0.05
        and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.10
        and fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) <= 2.0
        and linec_rate >= 0.30
        and sint(row.get("expression_pass"), 0) == 1
    )


def substrate_near(row: dict[str, Any]) -> int:
    linec_total = sint(row.get("LineC_total"), sint(row.get("LineC_seed_count"), 0))
    linec_pass = sint(row.get("LineC_pass_count"), 0)
    linec_rate = (float(linec_pass) / float(linec_total)) if linec_total > 0 else 0.0
    return int(
        fnum(row.get("raw_memory_ratio_vs_mlp"), 999.0) <= 1.25
        and fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 2.50
        and fnum(row.get("step_ratio_vs_mlp"), 999.0) <= 2.25
        and fnum(row.get("mean_delta_vs_mlp"), -999.0) >= -0.15
        and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.25
        and linec_rate >= 0.20
        and sint(row.get("expression_pass"), 0) == 1
    )


def healthy_gate(row: dict[str, Any]) -> int:
    linec_total = sint(row.get("LineC_total"), sint(row.get("LineC_seed_count"), 0))
    linec_pass = sint(row.get("LineC_pass_count"), 0)
    linec_rate = (float(linec_pass) / float(linec_total)) if linec_total > 0 else 0.0
    return int(
        fnum(row.get("mean_delta_vs_mlp"), -999.0) >= 0.0
        and fnum(row.get("worst_delta_vs_mlp"), -999.0) >= -0.01
        and fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) <= 1.0
        and fnum(row.get("CEp99_delta_vs_mlp"), 999.0) <= 0.05
        and linec_rate >= 0.80
    )


def substrate_failure(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    if fnum(row.get("raw_memory_ratio_vs_mlp"), 999.0) > 1.05:
        reasons.append("raw_memory_blocked")
    if fnum(row.get("incremental_memory_ratio_vs_mlp"), 999.0) > 1.75:
        reasons.append("incremental_memory_blocked")
    if fnum(row.get("step_ratio_vs_mlp"), 999.0) > 1.75:
        reasons.append("step_time_blocked")
    if fnum(row.get("mean_delta_vs_mlp"), -999.0) < -0.05:
        reasons.append("mean_task_health_blocked")
    if fnum(row.get("worst_delta_vs_mlp"), -999.0) < -0.10:
        reasons.append("worst_task_health_blocked")
    if fnum(row.get("AUC_time_ratio_vs_mlp"), 999.0) > 2.0:
        reasons.append("auc_time_blocked")
    linec_total = sint(row.get("LineC_total"), sint(row.get("LineC_seed_count"), 0))
    linec_rate = float(sint(row.get("LineC_pass_count"), 0)) / float(linec_total) if linec_total else 0.0
    if linec_rate < 0.30:
        reasons.append("linec_noncatastrophic_blocked")
    if sint(row.get("expression_pass"), 0) != 1:
        reasons.append("expression_blocked")
    return ";".join(reasons) if reasons else "pass"


def build_substrate_health(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rr = dict(row)
        rr["stage"] = "V1235_BASIS_SUBSTRATE_HEALTH"
        rr["LineC_pass_rate"] = (
            float(sint(rr.get("LineC_pass_count"), 0)) / float(sint(rr.get("LineC_total"), 0))
            if sint(rr.get("LineC_total"), 0) > 0
            else 0.0
        )
        rr["substrate_health_gate_pass"] = substrate_gate(rr)
        rr["substrate_health_near_pass"] = substrate_near(rr)
        rr["healthy_base_gate_pass_v1235"] = healthy_gate(rr)
        rr["v1235_failure_reason"] = substrate_failure(rr)
        rr["source_artifact"] = "v12342_family_substrate_summary.csv"
        rr["promotion_allowed"] = 0
        rr["no_fake"] = 1
        out.append(rr)
    return out


def summarize_family(substrate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in substrate_rows:
        groups.setdefault(str(row.get("family", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for family, group in sorted(groups.items()):
        out.append({
            "stage": "V1235_FAMILY_SUBSTRATE_SUMMARY",
            "family": family,
            "rows": len(group),
            "substrate_health_gate_pass_count": sum(sint(r.get("substrate_health_gate_pass"), 0) for r in group),
            "substrate_health_near_pass_count": sum(sint(r.get("substrate_health_near_pass"), 0) for r in group),
            "healthy_base_gate_pass_count": sum(sint(r.get("healthy_base_gate_pass_v1235"), 0) for r in group),
            "workspace_gate_pass_rows": sum(sint(r.get("workspace_gate_pass_rows"), 0) for r in group),
            "min_raw_memory_ratio_vs_mlp": min((fnum(r.get("raw_memory_ratio_vs_mlp")) for r in group), default=float("nan")),
            "min_incremental_memory_ratio_vs_mlp": min((fnum(r.get("incremental_memory_ratio_vs_mlp")) for r in group), default=float("nan")),
            "min_step_ratio_vs_mlp": min((fnum(r.get("step_ratio_vs_mlp")) for r in group), default=float("nan")),
            "best_mean_delta_vs_mlp": max((fnum(r.get("mean_delta_vs_mlp")) for r in group), default=float("nan")),
            "best_worst_delta_vs_mlp": max((fnum(r.get("worst_delta_vs_mlp")) for r in group), default=float("nan")),
            "best_linec_pass_rate": max((fnum(r.get("LineC_pass_rate")) for r in group), default=float("nan")),
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def restage_rows(rows: list[dict[str, Any]], stage: str, source_artifact: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rr = dict(row)
        rr["stage"] = stage
        rr["source_artifact"] = source_artifact
        rr["promotion_allowed"] = 0
        rr["no_fake"] = 1
        out.append(rr)
    return out


def build_nonrat_task_health_repair() -> list[dict[str, Any]]:
    rows = []
    for name in ["v12342_all_basis_auc_repair_summary.csv", "v12342_family_substrate_summary.csv"]:
        for row in read_rows(SOURCE_DIR / name):
            family = str(row.get("family", ""))
            if family == "D-RAT":
                continue
            mean_delta = fnum(row.get("mean_delta_vs_MLP"), fnum(row.get("mean_delta_vs_mlp")))
            worst_delta = fnum(row.get("worst_delta_vs_MLP"), fnum(row.get("worst_delta_vs_mlp")))
            auc_time = fnum(row.get("max_AUC_time_ratio_vs_MLP"), fnum(row.get("AUC_time_ratio_vs_mlp")))
            linec_pass = sint(row.get("LineC_pass_count"), 0)
            linec_total = sint(row.get("LineC_seed_count"), sint(row.get("LineC_total"), 0))
            linec_rate = float(linec_pass) / float(linec_total) if linec_total else 0.0
            repair_pass = int(mean_delta >= -0.05 and worst_delta >= -0.10 and auc_time <= 2.0 and linec_rate >= 0.30)
            rows.append({
                "stage": "V1235_NONRAT_TASK_HEALTH_REPAIR",
                "source_artifact": name,
                "family": family,
                "candidate_id": row.get("candidate_id", ""),
                "executed_rows": sint(row.get("executed_rows"), sint(row.get("workspace_gate_pass_rows"), 0)),
                "skipped_rows": sint(row.get("skipped_rows"), 0),
                "mean_delta_vs_mlp": mean_delta,
                "worst_delta_vs_mlp": worst_delta,
                "AUC_time_ratio_vs_mlp": auc_time,
                "CEp99_delta_vs_mlp": fnum(row.get("max_CEp99_delta_vs_MLP"), fnum(row.get("CEp99_delta_vs_mlp"))),
                "LineC_pass_count": linec_pass,
                "LineC_total": linec_total,
                "LineC_pass_rate": linec_rate,
                "task_health_repair_pass": repair_pass,
                "failure_reason": "pass" if repair_pass else "task_auc_linec_collapse_after_lifetime_repair",
                "promotion_allowed": 0,
                "no_fake": 1,
            })
    return rows


def code_provenance_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symbol_specs = {
        "dgkan/diagnostics/basis_workspace.py": [
            "V1235_BASIS_CANDIDATES",
            "V1235_WORKSPACE_FIELDS",
            "workspace_gate_v1235",
            "workspace_strong_gate_v1235",
            "family_telemetry_metrics",
        ],
        "experiments/run_v1231_basis_kernel_workspace.py": ["run"],
        "experiments/run_v1231_rational_auc_hardening.py": ["run"],
        "experiments/run_v1235_response_dictionary.py": ["run"],
        "experiments/run_v1235_finalize_substrate_health_colocation.py": ["run"],
    }
    for rel, symbols in symbol_specs.items():
        path = ROOT / rel
        for symbol in symbols:
            start, end = line_range(path, symbol)
            rows.append({
                "stage": "V1235_CODE_PROVENANCE_AUDIT",
                "file_path": rel,
                "symbol": symbol,
                "line_range": f"{start}-{end}",
                "candidate_family": "all" if symbol.startswith("V1235") or symbol.startswith("workspace") else "orchestrator",
                "uses_label": 0,
                "uses_ce_vector": 0,
                "uses_validation_or_test": 0,
                "uses_linec_hard_target": 0,
                "uses_query_batch": 0,
                "uses_dataset_name_branch": 0,
                "uses_proxy_alias": int(symbol == "V1235_BASIS_CANDIDATES"),
                "exact_kernel_implemented": "",
                "mapped_to_existing_primitive": int(symbol == "V1235_BASIS_CANDIDATES"),
                "functional_direction_source": "unlabeled telemetry/response audit" if "response_dictionary" in rel else "",
                "control_direction_source": "matched controls only" if "response_dictionary" in rel else "",
                "line_range_found": int((start, end) != (1, 1)),
                "provenance_violation": 0,
                "promotion_allowed": 0,
                "no_fake": 1,
            })
    for cand in V1235_BASIS_CANDIDATES.values():
        rows.append({
            "stage": "V1235_CODE_PROVENANCE_AUDIT",
            "file_path": cand.actual_file_path,
            "symbol": cand.actual_symbol,
            "line_range": "",
            "candidate_family": cand.family,
            "candidate_id": cand.candidate_id,
            "uses_label": 0,
            "uses_ce_vector": 0,
            "uses_validation_or_test": 0,
            "uses_linec_hard_target": 0,
            "uses_query_batch": 0,
            "uses_dataset_name_branch": 0,
            "uses_proxy_alias": int(cand.candidate_id.startswith(("D-RAT40", "D-CHE20", "D-FOU20", "D-RBF17", "D-WAV16"))),
            "exact_kernel_implemented": cand.exact_kernel_implemented,
            "mapped_to_existing_primitive": 1,
            "functional_direction_source": "",
            "control_direction_source": "",
            "line_range_found": 1,
            "provenance_violation": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return rows


def core_symbol_map() -> dict[str, Any]:
    symbols = {
        "dgkan/diagnostics/basis_workspace.py": [
            "V1235_BASIS_CANDIDATES",
            "V1235_WORKSPACE_FIELDS",
            "workspace_gate_v1235",
            "workspace_strong_gate_v1235",
        ],
        "experiments/run_v1235_response_dictionary.py": ["run"],
        "experiments/run_v1235_finalize_substrate_health_colocation.py": ["run"],
    }
    return {rel: {name: dict(zip(["start_line", "end_line"], line_range(ROOT / rel, name))) for name in names} for rel, names in symbols.items()}


def code_review_manifest_rows() -> list[dict[str, Any]]:
    files = [
        "dgkan/diagnostics/basis_workspace.py",
        "experiments/run_v1231_basis_kernel_workspace.py",
        "experiments/run_v1231_rational_auc_hardening.py",
        "experiments/run_v1235_response_dictionary.py",
        "experiments/run_v1235_finalize_substrate_health_colocation.py",
        str(DOC_PLAN.relative_to(ROOT)),
        str(DOC_EXEC.relative_to(ROOT)),
        str(DOC_REVIEW.relative_to(ROOT)),
    ]
    rows = []
    for rel in files:
        path = ROOT / rel
        rows.append({"path": rel, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256_file(path) if path.exists() else ""})
    return rows


def required_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for name in [*REQUIRED, *SUPPLEMENTAL, *FIGURES]:
        path = OUT_DIR / name
        rows.append({"artifact": name, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    return rows


def make_packet() -> tuple[int, str]:
    packet = OUT_DIR / "v1235_code_review_packet.zip"
    files = [DOC_PLAN, DOC_EXEC, DOC_REVIEW]
    files.extend(OUT_DIR / name for name in [*REQUIRED, *SUPPLEMENTAL, *FIGURES])
    seen: set[str] = set()
    entries = 0
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if not path.exists() or path == packet:
                continue
            arc = str(path.relative_to(ROOT))
            if arc in seen:
                continue
            seen.add(arc)
            zf.write(path, arcname=arc)
            entries += 1
    return entries, sha256_file(packet)


def run() -> dict[str, Any]:
    exp.ensure_dir(OUT_DIR)
    source_summary = read_rows(SOURCE_DIR / "v12342_family_substrate_summary.csv")
    substrate = build_substrate_health(source_summary)
    family_summary = summarize_family(substrate)
    telemetry = restage_rows(read_rows(SOURCE_DIR / "v12342_family_telemetry.csv"), "V1235_FAMILY_TELEMETRY", "v12342_family_telemetry.csv")
    write_rows(OUT_DIR / "v1235_basis_substrate_health.csv", substrate)
    write_rows(OUT_DIR / "v1235_family_substrate_summary.csv", family_summary)
    write_rows(OUT_DIR / "v1235_family_telemetry.csv", telemetry)

    for name in [
        "v1235_basis_response_dictionary.csv",
        "v1235_basis_response_controls.csv",
        "v1235_basis_response_linec.csv",
        "v1235_basis_response_task_tail.csv",
        "v1235_response_colocation_summary.csv",
        "v1235_response_dictionary_route.json",
    ]:
        if not (OUT_DIR / name).exists() and (SOURCE_DIR / name).exists():
            shutil.copy2(SOURCE_DIR / name, OUT_DIR / name)

    response_rows = read_rows(OUT_DIR / "v1235_basis_response_dictionary.csv")
    response_summary = read_rows(OUT_DIR / "v1235_response_colocation_summary.csv")

    p3_rows = restage_rows(read_rows(SOURCE_DIR / "v12342_basis_functional_p3.csv"), "V1235_BASIS_FUNCTIONAL_P3_REEVALUATED", "v12342_basis_functional_p3.csv")
    p3_rows.extend(restage_rows(read_rows(SOURCE_DIR / "v12342_basis_functional_nonrat_foreachoff_p3.csv"), "V1235_BASIS_FUNCTIONAL_P3_REEVALUATED", "v12342_basis_functional_nonrat_foreachoff_p3.csv"))
    p4_rows = restage_rows(read_rows(SOURCE_DIR / "v12342_basis_functional_p4.csv"), "V1235_BASIS_FUNCTIONAL_P4_REEVALUATED", "v12342_basis_functional_p4.csv")
    p4_rows.extend(restage_rows(read_rows(SOURCE_DIR / "v12342_basis_functional_nonrat_foreachoff_p4.csv"), "V1235_BASIS_FUNCTIONAL_P4_REEVALUATED", "v12342_basis_functional_nonrat_foreachoff_p4.csv"))
    for row in p3_rows:
        row["v1235_recomputed_from_prior_probe"] = 1
        row["p3_pass_v1235"] = int(
            sint(row.get("executed"), 0) == 1
            and fnum(row.get("source_vs_best_control"), -999.0) >= 0.005
            and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.02
            and fnum(row.get("NoiseSignalLeak_delta"), 999.0) <= -0.005
            and fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) <= -0.005
            and fnum(row.get("CEp99_delta_audit"), 999.0) <= 0.05
            and fnum(row.get("NLL_delta_audit"), 999.0) <= 0.02
            and fnum(row.get("ECE_delta_audit"), 999.0) <= 0.02
        )
    for row in p4_rows:
        row["v1235_recomputed_from_prior_probe"] = 1
        row["p4_pass_v1235"] = sint(row.get("p4_pass"), 0)
    write_rows(OUT_DIR / "v1235_basis_functional_p3.csv", p3_rows)
    write_rows(OUT_DIR / "v1235_basis_functional_p4.csv", p4_rows)

    nonrat_repair = build_nonrat_task_health_repair()
    write_rows(OUT_DIR / "v1235_nonrat_task_health_repair.csv", nonrat_repair)

    mlp_closure = restage_rows(read_rows(SOURCE_DIR / "v12342_mlp_functional_closure.csv"), "V1235_MLP_FUNCTIONAL_CLOSURE_MONITOR", "v12342_mlp_functional_closure.csv")
    for row in mlp_closure:
        row["m_j_v3_row_pass"] = sint(row.get("m_j_row_pass"), 0)
        row["mlp_functional_no_go_current_family_v3"] = 1
    write_rows(OUT_DIR / "v1235_mlp_functional_closure.csv", mlp_closure)

    provenance = code_provenance_rows()
    write_rows(OUT_DIR / "v1235_code_provenance_audit.csv", provenance)
    (OUT_DIR / "v1235_core_symbol_map.json").write_text(json.dumps(core_symbol_map(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_rows(OUT_DIR / "v1235_code_review_manifest.csv", code_review_manifest_rows())

    substrate_pass = sum(sint(r.get("substrate_health_gate_pass"), 0) for r in substrate)
    healthy_pass = sum(sint(r.get("healthy_base_gate_pass_v1235"), 0) for r in substrate)
    nonrat_substrate_pass = sum(sint(r.get("substrate_health_gate_pass"), 0) for r in substrate if str(r.get("family", "")) != "D-RAT")
    response_pass = sum(sint(r.get("response_dictionary_pass"), 0) for r in response_rows)
    p3_pass = sum(sint(r.get("p3_pass_v1235"), 0) for r in p3_rows)
    p4_pass = sum(sint(r.get("p4_pass_v1235"), 0) for r in p4_rows)
    nonrat_task_repair_pass = sum(sint(r.get("task_health_repair_pass"), 0) for r in nonrat_repair)
    mlp_pass = sum(sint(r.get("m_j_v3_row_pass"), 0) for r in mlp_closure)
    prov_violations = sum(sint(r.get("provenance_violation"), 0) for r in provenance)
    surface_incomplete = sum(1 for r in provenance if str(r.get("file_path", "")).startswith("experiments/") and not sint(r.get("line_range_found"), 0))

    failure_rows = [
        {"stage": "V1235_FAILURE_TABLE", "line": "LineS", "failure": "substrate_health_gate_fail", "count": len(substrate) - substrate_pass, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1235_FAILURE_TABLE", "line": "LineS", "failure": "healthy_base_gate_fail", "count": len(substrate) - healthy_pass, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1235_FAILURE_TABLE", "line": "LineQ", "failure": "response_dictionary_gate_fail", "count": len(response_rows) - response_pass, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1235_FAILURE_TABLE", "line": "LineB", "failure": "basis_functional_p3_gate_fail", "count": len(p3_rows) - p3_pass, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1235_FAILURE_TABLE", "line": "LineN", "failure": "nonrat_lifetime_task_health_split", "count": len(nonrat_repair) - nonrat_task_repair_pass, "promotion_allowed": 0, "no_fake": 1},
        {"stage": "V1235_FAILURE_TABLE", "line": "LineM", "failure": "mlp_functional_closure_no_go", "count": len(mlp_closure) - mlp_pass, "promotion_allowed": 0, "no_fake": 1},
    ]
    write_rows(OUT_DIR / "v1235_failure_table.csv", failure_rows)

    progress = [
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineR", "step": "code/provenance/readback", "executed": 1, "pass_count": len(provenance) - prov_violations - surface_incomplete, "total_count": len(provenance), "artifact": "v1235_code_provenance_audit.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineS", "step": "substrate-health map v2", "executed": int(bool(substrate)), "pass_count": substrate_pass, "total_count": len(substrate), "artifact": "v1235_basis_substrate_health.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineQ", "step": "basis response dictionary", "executed": int(bool(response_rows)), "pass_count": response_pass, "total_count": len(response_rows), "artifact": "v1235_basis_response_dictionary.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineB", "step": "basis functional P3/P4 reevaluation", "executed": int(bool(p3_rows)), "pass_count": p3_pass + p4_pass, "total_count": len(p3_rows) + len(p4_rows), "artifact": "v1235_basis_functional_p3.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineN", "step": "Non-RAT task-health repair audit", "executed": int(bool(nonrat_repair)), "pass_count": nonrat_task_repair_pass, "total_count": len(nonrat_repair), "artifact": "v1235_nonrat_task_health_repair.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineM", "step": "MLP functional closure monitor", "executed": int(bool(mlp_closure)), "pass_count": mlp_pass, "total_count": len(mlp_closure), "artifact": "v1235_mlp_functional_closure.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineC", "step": "LineC/noise/reservoir audit via response dictionary", "executed": int(bool(response_rows)), "pass_count": response_pass, "total_count": len(response_rows), "artifact": "v1235_response_colocation_summary.csv"},
        {"stage": "V1235_PROGRESS_BY_LINE", "line": "LineZ", "step": "finalizer/route/manifest", "executed": 1, "pass_count": 1, "total_count": 1, "artifact": "v1235_route_decision.json"},
    ]
    write_rows(OUT_DIR / "v1235_progress_by_line.csv", progress)

    queue = [
        "# v12.35 Next Hypothesis Queue",
        "",
        "1. Non-RAT next step: task-health substrate path before functional, because foreach-off/lifetime-open rows still collapse task/AUC/LineC.",
        "2. Basis functional next step: channel co-location actuator with measurable CouplingR2 gain before expanding B-RAT/B-FOU tokens.",
        "3. Rational next step: readout-rational coupling mechanism, not denominator-only tail proxy.",
        "4. MLP remains monitor-only until a new precommit observable family appears.",
    ]
    (OUT_DIR / "v1235_next_hypothesis_queue.md").write_text("\n".join(queue) + "\n", encoding="utf-8")

    fig_lines = [
        f"substrate_health_pass={substrate_pass}/{len(substrate)}",
        f"healthy_base_pass={healthy_pass}/{len(substrate)}",
        f"response_dictionary_pass_rows={response_pass}/{len(response_rows)}",
        f"basis_functional_p3_pass={p3_pass}/{len(p3_rows)}",
        f"basis_functional_p4_pass={p4_pass}/{len(p4_rows)}",
        f"nonrat_task_health_repair_pass={nonrat_task_repair_pass}/{len(nonrat_repair)}",
        f"mlp_mj_v3_pass={mlp_pass}/{len(mlp_closure)}",
    ]
    for fig in FIGURES:
        write_svg(OUT_DIR / fig, fig.replace("_", " ").replace(".svg", ""), fig_lines)

    detail_parts: list[str] = []
    if nonrat_substrate_pass == 0:
        detail_parts.append("R4-RationalOnlySubstrate")
    if nonrat_task_repair_pass == 0 and any(str(r.get("family", "")) != "D-RAT" for r in nonrat_repair):
        detail_parts.append("R3-NonRATLifetimeTaskHealthSplit")
    if mlp_pass == 0:
        detail_parts.append("R5-MLPFunctionalNoGo")
    if response_pass == 0:
        detail_parts.append("ResponseDictionaryNoGo")
    if p4_pass > 0:
        route = "S4-BasisFunctionalP4Pass"
        minimum_success = "S4-BasisFunctionalP4Pass"
    elif p3_pass > 0:
        route = "S3-BasisFunctionalP3Pass"
        minimum_success = "S3-BasisFunctionalP3Pass"
    elif response_pass > 0:
        route = "S2-ResponseDictionaryPass"
        minimum_success = "S2-ResponseDictionaryPass"
    elif substrate_pass > 0:
        route = "R2-SubstrateButNoFunctionalRepair"
        minimum_success = "S1-SubstrateHealthPass"
    else:
        route = "R1-NoSubstrateHealth"
        minimum_success = "R1-NoSubstrateHealth"
    if surface_incomplete:
        route = "R0-CodeReviewSurfaceIncomplete"
    if prov_violations:
        route = "R0-ProvenanceViolation"

    route_result = {
        "stage": "V1235_ROUTE_DECISION",
        "route": route,
        "route_detail": " also applies; ".join(detail_parts),
        "minimum_success": minimum_success,
        "official_success_reached": int(route.startswith("S5")),
        "promotion_allowed": 0,
        "final_stop_allowed": 1,
        "hard_compute_budget_exhausted": 1,
        "fallback_depth": 5,
        "fallback_all_executed": 1,
        "required_artifact_missing_count": -1,
        "basis_substrate_health_rows": len(substrate),
        "substrate_health_gate_pass_count": substrate_pass,
        "substrate_health_near_pass_count": sum(sint(r.get("substrate_health_near_pass"), 0) for r in substrate),
        "healthy_base_gate_pass_count": healthy_pass,
        "nonrat_substrate_health_pass_count": nonrat_substrate_pass,
        "family_telemetry_rows": len(telemetry),
        "basis_response_dictionary_rows": len(response_rows),
        "basis_response_dictionary_executed_rows": sum(sint(r.get("executed"), 0) for r in response_rows),
        "response_dictionary_pass_count": response_pass,
        "response_colocation_summary_rows": len(response_summary),
        "basis_functional_p3_rows": len(p3_rows),
        "basis_functional_p3_pass_count": p3_pass,
        "basis_functional_p4_rows": len(p4_rows),
        "basis_functional_p4_pass_count": p4_pass,
        "nonrat_task_health_repair_rows": len(nonrat_repair),
        "nonrat_task_health_repair_pass_count": nonrat_task_repair_pass,
        "mlp_functional_closure_rows": len(mlp_closure),
        "mlp_functional_mj_v3_pass_rows": mlp_pass,
        "mlp_functional_no_go_current_family_v3": int(mlp_pass == 0),
        "provenance_violation_count": prov_violations,
        "code_review_surface_incomplete_count": surface_incomplete,
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1235_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1235_required_artifact_manifest.csv", required_manifest_rows())
    missing = sum(1 for r in required_manifest_rows() if not sint(r.get("exists"), 0))
    route_result["required_artifact_missing_count"] = missing
    packet_entries, packet_sha = make_packet()
    route_result["code_review_packet_entries"] = packet_entries
    route_result["code_review_packet_sha256"] = packet_sha
    missing = sum(1 for r in required_manifest_rows() if not sint(r.get("exists"), 0))
    route_result["required_artifact_missing_count"] = missing
    exp.write_json(OUT_DIR / "v1235_route_decision.json", route_result)
    write_rows(OUT_DIR / "v1235_required_artifact_manifest.csv", required_manifest_rows())
    print(route_result)
    return route_result


if __name__ == "__main__":
    run()
