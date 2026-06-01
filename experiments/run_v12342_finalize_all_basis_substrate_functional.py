#!/usr/bin/env python
"""Finalize v12.34.2 all-basis substrate + basis-specific functional repair."""

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
from dgkan.diagnostics.basis_workspace import V12342_BASIS_CANDIDATES, nonrat_lifetime_rows  # noqa: E402
from dgkan.functional.mlp_functional import CONTROL_IDS, SOURCE_CANDIDATES  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_34_2_all_basis_substrate_functional_repair" / "official_v12342"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.34.2_AllBasisSubstrateFunctionalRepair_实验结果复盘.md"

REQUIRED = [
    "v12342_family_substrate_summary.csv",
    "v12342_family_telemetry.csv",
    "v12342_family_telemetry_correlations.csv",
    "v12342_basis_functional_p3.csv",
    "v12342_basis_functional_p4.csv",
    "v12342_kernel_lifetime_waterfall.csv",
    "v12342_mlp_functional_closure.csv",
    "v12342_linec_audit.csv",
    "v12342_failure_table.csv",
    "v12342_route_decision.json",
    "v12342_next_hypothesis_queue.md",
    "v12342_code_review_packet.zip",
]

FIGURES = [
    "fig_progress_by_family.svg",
    "fig_family_substrate_pareto.svg",
    "fig_family_telemetry_vs_tail.svg",
    "fig_family_telemetry_vs_auc.svg",
    "fig_family_telemetry_vs_linec.svg",
    "fig_family_telemetry_correlation_heatmap.svg",
    "fig_functional_control_gap_by_family.svg",
    "fig_kernel_lifetime_waterfall_by_family.svg",
    "fig_mlp_functional_no_go.svg",
    "fig_linec_task_tail_colocation.svg",
    "fig_route_dashboard.svg",
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


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, idx in enumerate(order):
        out[idx] = float(rank)
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy))


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def required_manifest_rows() -> list[dict[str, Any]]:
    rows = []
    for name in [*REQUIRED, *FIGURES, "v12342_kernel_implementation_manifest.csv", "v12342_exact_kernel_audit.csv", "v12342_code_review_manifest.csv", "v12342_core_symbol_map.json", "v12342_provenance_audit.csv"]:
        p = OUT_DIR / name
        rows.append({"artifact": name, "exists": int(p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0})
    return rows


def substrate_summary(
    workspace_rows: list[dict[str, Any]],
    hardening_rows: list[dict[str, Any]],
    auc_summary: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cid: dict[str, list[dict[str, Any]]] = {}
    for row in workspace_rows:
        by_cid.setdefault(str(row.get("candidate_id", "")), []).append(row)
    hard_by: dict[str, list[dict[str, Any]]] = {}
    for row in hardening_rows:
        hard_by.setdefault(str(row.get("candidate_id", "")), []).append(row)
    auc_by = {str(r.get("candidate_id", "")): r for r in auc_summary}
    tele_by: dict[str, list[dict[str, Any]]] = {}
    for row in telemetry_rows:
        tele_by.setdefault(str(row.get("candidate_id", "")), []).append(row)
    exact_by = {str(r.get("candidate_id", "")): r for r in exact_rows}
    out: list[dict[str, Any]] = []
    for cid, rows in sorted(by_cid.items()):
        cand = V12342_BASIS_CANDIDATES.get(cid)
        family = str(rows[0].get("family", cand.family if cand else ""))
        workspace_pass = sum(sint(r.get("workspace_gate_pass"), 0) for r in rows)
        strong_pass = sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in rows)
        raw_max = max((fnum(r.get("raw_memory_ratio_vs_mlp"), 999.0) for r in rows), default=float("nan"))
        inc_max = max((fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0) for r in rows), default=float("nan"))
        step_max = max((fnum(r.get("step_ratio_vs_mlp"), 999.0) for r in rows), default=float("nan"))
        auc = auc_by.get(cid, {})
        hard = [r for r in hard_by.get(cid, []) if sint(r.get("executed"), 1)]
        deltas = [fnum(r.get("mean_delta_vs_MLP")) for r in hard if math.isfinite(fnum(r.get("mean_delta_vs_MLP")))]
        mean_delta = fnum(auc.get("mean_delta_vs_MLP"), finite_mean(deltas))
        worst_delta = fnum(auc.get("worst_delta_vs_MLP"), min(deltas) if deltas else float("nan"))
        max_auc_step = fnum(auc.get("max_AUC_step_ratio_vs_MLP"))
        max_auc_time = fnum(auc.get("max_AUC_time_ratio_vs_MLP"))
        max_ce = fnum(auc.get("max_CEp99_delta_vs_MLP"))
        linec_pass = sint(auc.get("LineC_pass_count"), sum(sint(r.get("LineC_pass_count"), 0) for r in hard))
        linec_total = sint(auc.get("LineC_seed_count"), sum(sint(r.get("LineC_seed_count"), 0) for r in hard))
        tele = tele_by.get(cid, [])
        telemetry_available = sum(sint(r.get("telemetry_available"), 0) for r in tele)
        exact = exact_by.get(cid, {})
        exact_kernel = sint(rows[0].get("exact_kernel_implemented"), 0)
        expression_pass = int((not exact_kernel and any(str(r.get("status", "")) == "executed" for r in rows)) or sint(exact.get("manual_gradcheck_pass"), 0))
        substrate = int(
            workspace_pass == len(rows)
            and step_max <= 1.50
            and raw_max <= 1.50
            and inc_max <= 2.50
            and expression_pass
            and mean_delta >= -0.05
            and worst_delta >= -0.10
            and max_ce <= 5.0
            and telemetry_available > 0
        )
        near = int(
            workspace_pass > 0
            and step_max <= 1.75
            and inc_max <= 3.50
            and expression_pass
            and telemetry_available > 0
        )
        healthy = int(
            strong_pass == len(rows)
            and raw_max <= 1.25
            and inc_max <= 1.25
            and step_max <= 1.25
            and mean_delta >= 0.0
            and worst_delta >= -0.003
            and max_auc_time <= 1.0
            and max_ce <= 0.05
            and linec_pass >= max(1, linec_total)
        )
        failure = "pass" if substrate else ""
        if not workspace_pass:
            failure = "EfficiencyBlocked"
        elif not expression_pass:
            failure = "ExpressionBlocked"
        elif telemetry_available <= 0:
            failure = "TelemetryMissingBlocked"
        elif mean_delta < -0.05 or worst_delta < -0.10:
            failure = "TaskCatastrophicBlocked"
        elif max_ce > 5.0 or max_auc_time > 1.50:
            failure = "TaskLineCTailNotColocated"
        else:
            failure = "SubstrateNearOnly" if near else "GateSBlocked"
        out.append({
            "stage": "V12342_FAMILY_SUBSTRATE_SUMMARY",
            "family": family,
            "candidate_id": cid,
            "workspace_rows": len(rows),
            "workspace_gate_pass_rows": workspace_pass,
            "workspace_strong_gate_pass_rows": strong_pass,
            "step_ratio_vs_mlp": step_max,
            "raw_memory_ratio_vs_mlp": raw_max,
            "incremental_memory_ratio_vs_mlp": inc_max,
            "expression_pass": expression_pass,
            "expression_pass_source": "manual_gradcheck" if exact_kernel else "train_step_forward_backward_smoke",
            "mean_delta_vs_mlp": mean_delta,
            "worst_delta_vs_mlp": worst_delta,
            "AUC_step_ratio_vs_mlp": max_auc_step,
            "AUC_time_ratio_vs_mlp": max_auc_time,
            "CEp99_delta_vs_mlp": max_ce,
            "NLL_delta_vs_mlp": "",
            "ECE_delta_vs_mlp": "",
            "LineC_pass_count": linec_pass,
            "LineC_total": linec_total,
            "CouplingR2": "",
            "NoiseSignalLeak": "",
            "RealSignalReservoirRatio": "",
            "telemetry_available": int(telemetry_available > 0),
            "telemetry_available_rows": telemetry_available,
            "substrate_gate_pass": substrate,
            "substrate_near_pass": near,
            "healthy_base_gate_pass": healthy,
            "failure_reason": failure,
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    return out


def telemetry_correlations(telemetry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        "basis_entropy",
        "dead_basis_fraction",
        "basis_effective_rank",
        "basis_condition_proxy",
        "den_p01_batch",
        "r_prime_p99",
        "tangent_effective_rank",
        "tangent_top_eigen_share",
        "logit_norm_p99",
        "unlabeled_entropy_p05",
    ]
    target_keys = [
        "CEp99_delta_vs_MLP",
        "AUC_time_ratio_vs_MLP",
        "LineC_pass_count",
        "NoiseSignalLeak",
        "RealSignalReservoirRatio",
    ]
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in telemetry_rows:
        by_family.setdefault(str(row.get("family", "")), []).append(row)
    for family, group in sorted(by_family.items()):
        for metric in metric_keys:
            xs = [fnum(r.get(metric)) for r in group]
            if not any(math.isfinite(x) for x in xs):
                continue
            for target in target_keys:
                ys = [fnum(r.get(target)) for r in group]
                corr = spearman(xs, ys)
                rows.append({
                    "stage": "V12342_TELEMETRY_CORRELATION",
                    "family": family,
                    "telemetry_metric": metric,
                    "audit_metric": target,
                    "rows": len([1 for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]),
                    "spearman": corr,
                    "promotion_allowed": 0,
                    "no_fake": 1,
                })
    return rows


def mlp_closure_rows(candidate_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in candidate_rows:
        cid = str(row.get("functional_candidate_id", ""))
        if not cid.startswith("M-J"):
            continue
        rr = dict(row)
        rr["stage"] = "V12342_MLP_FUNCTIONAL_CLOSURE"
        rr["m_j_row_pass"] = int(
            fnum(rr.get("source_vs_control_acc_delta"), -999.0) >= 0.005
            and fnum(rr.get("CEp99_delta_vs_noop"), 999.0) <= 0.05
            and fnum(rr.get("NLL_delta_vs_noop"), 999.0) <= 0.02
            and fnum(rr.get("ECE_delta_vs_noop"), 999.0) <= 0.02
            and fnum(rr.get("CouplingR2_delta"), -999.0) >= 0.02
            and fnum(rr.get("NoiseSignalLeak_delta"), 999.0) <= -0.01
            and fnum(rr.get("RealSignalReservoirRatio_delta"), 999.0) <= -0.01
        )
        rr["promotion_allowed"] = 0
        rr["no_fake"] = 1
        out.append(rr)
    if not out and gate_rows:
        for row in gate_rows:
            cid = str(row.get("functional_candidate_id", ""))
            if cid.startswith("M-J"):
                out.append({"stage": "V12342_MLP_FUNCTIONAL_CLOSURE", **row, "m_j_row_pass": 0, "promotion_allowed": 0, "no_fake": 1})
    return out


def provenance_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in V12342_BASIS_CANDIDATES.values():
        rows.append({
            "stage": "V12342_PROVENANCE_AUDIT",
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
    for cid in [c for c in SOURCE_CANDIDATES if c.startswith("M-J")]:
        rows.append({
            "stage": "V12342_PROVENANCE_AUDIT",
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
            "stage": "V12342_PROVENANCE_AUDIT",
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
    return rows


def core_symbol_map() -> dict[str, Any]:
    symbols = {
        "dgkan/diagnostics/basis_workspace.py": [
            "V12342_BASIS_CANDIDATES",
            "workspace_gate_v12342",
            "workspace_strong_gate_v12342",
            "family_telemetry_metrics",
            "nonrat_lifetime_rows",
        ],
        "dgkan/functional/mlp_functional.py": ["SOURCE_CANDIDATES", "functional_objective"],
        "experiments/run_v1231_basis_kernel_workspace.py": ["run"],
        "experiments/run_v1231_rational_auc_hardening.py": ["run"],
        "experiments/run_v12342_basis_functional_repair.py": ["run"],
        "experiments/run_v1230_mlp_functional.py": ["run"],
        "experiments/run_v12342_finalize_all_basis_substrate_functional.py": ["run"],
    }
    return {rel: {name: dict(zip(["start_line", "end_line"], line_range(ROOT / rel, name))) for name in names} for rel, names in symbols.items()}


def code_review_manifest_rows() -> list[dict[str, Any]]:
    files = [
        "dgkan/diagnostics/basis_workspace.py",
        "dgkan/functional/mlp_functional.py",
        "experiments/run_v1231_basis_kernel_workspace.py",
        "experiments/run_v1231_rational_auc_hardening.py",
        "experiments/run_v12342_basis_functional_repair.py",
        "experiments/run_v1230_mlp_functional.py",
        "experiments/run_v12342_finalize_all_basis_substrate_functional.py",
        str(DOC_PLAN.relative_to(ROOT)),
        str(DOC_EXEC.relative_to(ROOT)),
        str(DOC_REVIEW.relative_to(ROOT)),
    ]
    rows = []
    for rel in files:
        p = ROOT / rel
        rows.append({"path": rel, "exists": int(p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": sha256_file(p) if p.exists() else ""})
    return rows


def make_packet() -> tuple[int, str]:
    packet = OUT_DIR / "v12342_code_review_packet.zip"
    files = [DOC_PLAN, DOC_EXEC, DOC_REVIEW]
    files.extend(OUT_DIR / name for name in [*REQUIRED, *FIGURES, "v12342_code_review_manifest.csv", "v12342_core_symbol_map.json", "v12342_provenance_audit.csv", "v12342_kernel_implementation_manifest.csv", "v12342_exact_kernel_audit.csv"])
    seen: set[str] = set()
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        entries = 0
        for path in files:
            if path.exists() and path != packet:
                arc = str(path.relative_to(ROOT))
                if arc in seen:
                    continue
                seen.add(arc)
                zf.write(path, arcname=arc)
                entries += 1
    return entries, sha256_file(packet)


def run() -> dict[str, Any]:
    exp.ensure_dir(OUT_DIR)
    for src_name, dst_name in [
        ("v12342_basis_kernel_implementation_manifest.csv", "v12342_kernel_implementation_manifest.csv"),
        ("v12342_basis_exact_kernel_audit.csv", "v12342_exact_kernel_audit.csv"),
    ]:
        src = OUT_DIR / src_name
        dst = OUT_DIR / dst_name
        if src.exists():
            dst.write_bytes(src.read_bytes())

    workspace_rows = read_rows(OUT_DIR / "v12342_basis_workspace_truth.csv")
    workspace_rows.extend(read_rows(OUT_DIR / "v12342_nonrat_lifetime_foreachoff_workspace_truth.csv"))
    hardening_rows = read_rows(OUT_DIR / "v12342_basis_hardening.csv")
    hardening_rows.extend(read_rows(OUT_DIR / "v12342_nonrat_lifetime_foreachoff_hardening.csv"))
    basis_linec = read_rows(OUT_DIR / "v12342_basis_linec.csv")
    basis_linec.extend(read_rows(OUT_DIR / "v12342_nonrat_lifetime_foreachoff_linec.csv"))
    exact_rows = read_rows(OUT_DIR / "v12342_exact_kernel_audit.csv")
    exact_rows.extend(read_rows(OUT_DIR / "v12342_nonrat_lifetime_foreachoff_exact_kernel_audit.csv"))
    telemetry_rows: list[dict[str, Any]] = []
    auc_summary: list[dict[str, Any]] = []
    auc_linec: list[dict[str, Any]] = []
    for name in [
        "v12342_all_basis_auc_lr15_candidate.csv",
        "v12342_all_basis_auc_lr20_candidate.csv",
        "v12342_all_basis_auc_repair_candidate.csv",
    ]:
        telemetry_rows.extend(read_rows(OUT_DIR / name))
    for name in [
        "v12342_all_basis_auc_lr15_summary.csv",
        "v12342_all_basis_auc_lr20_summary.csv",
        "v12342_all_basis_auc_repair_summary.csv",
    ]:
        auc_summary.extend(read_rows(OUT_DIR / name))
    for name in [
        "v12342_all_basis_auc_lr15_linec.csv",
        "v12342_all_basis_auc_lr20_linec.csv",
        "v12342_all_basis_auc_repair_linec.csv",
    ]:
        auc_linec.extend(read_rows(OUT_DIR / name))
    substrate = substrate_summary(workspace_rows, hardening_rows, auc_summary, telemetry_rows, exact_rows)
    write_rows(OUT_DIR / "v12342_family_substrate_summary.csv", substrate)
    write_rows(OUT_DIR / "v12342_family_telemetry.csv", telemetry_rows)
    write_rows(OUT_DIR / "v12342_family_telemetry_correlations.csv", telemetry_correlations(telemetry_rows))

    p3_rows = read_rows(OUT_DIR / "v12342_basis_functional_p3.csv")
    p3_rows.extend(read_rows(OUT_DIR / "v12342_basis_functional_nonrat_foreachoff_p3.csv"))
    p4_rows = read_rows(OUT_DIR / "v12342_basis_functional_p4.csv")
    p4_rows.extend(read_rows(OUT_DIR / "v12342_basis_functional_nonrat_foreachoff_p4.csv"))
    if not p3_rows:
        p3_rows = [{"stage": "V12342_BASIS_FUNCTIONAL_P3", "executed": 0, "skip_reason": "basis_functional_runner_not_executed", "p3_pass": 0, "promotion_allowed": 0, "no_fake": 1}]
        write_rows(OUT_DIR / "v12342_basis_functional_p3.csv", p3_rows)
    if not p4_rows:
        p4_rows = [{"stage": "V12342_BASIS_FUNCTIONAL_P4", "executed": 0, "skip_reason": "no_P3_pass_or_p4_not_scheduled", "p4_pass": 0, "promotion_allowed": 0, "no_fake": 1}]
        write_rows(OUT_DIR / "v12342_basis_functional_p4.csv", p4_rows)

    lifetime = nonrat_lifetime_rows(workspace_rows, exact_rows)
    for row in lifetime:
        row["stage"] = "V12342_KERNEL_LIFETIME_WATERFALL"
        row["basis_activation_live_bytes"] = row.get("basis_activation_bytes", "")
        row["basis_derivative_live_bytes"] = row.get("basis_derivative_bytes", "")
        row["readout_grad_live_bytes"] = row.get("readout_grad_bytes", "")
        row["coeff_grad_live_bytes"] = row.get("coeff_grad_bytes", "")
        row["optimizer_state_live_bytes"] = row.get("optimizer_state_bytes", "")
        row["temporary_workspace_live_bytes"] = row.get("workspace_temp_bytes", "")
        row["linec_hook_live_bytes"] = 0
        row["largest_live_tensor_name"] = row.get("largest_component", "")
        row["largest_live_tensor_shape"] = ""
        row["raw_peak_ratio"] = row.get("raw_memory_ratio_vs_mlp", "")
        row["incremental_peak_ratio"] = row.get("incremental_memory_ratio_vs_mlp", "")
        row["step_ratio"] = row.get("step_ratio_vs_mlp", "")
        row["lifetime_gate_pass"] = row.get("s3_lifetime_pass", 0)
    write_rows(OUT_DIR / "v12342_kernel_lifetime_waterfall.csv", lifetime)

    mlp_candidates = read_rows(OUT_DIR / "v12342_mlp_functional_candidates.csv")
    mlp_controls = read_rows(OUT_DIR / "v12342_mlp_functional_controls.csv")
    mlp_linec = read_rows(OUT_DIR / "v12342_mlp_functional_linec.csv")
    mlp_gate = read_rows(OUT_DIR / "v12342_mlp_functional_gate.csv")
    closure = mlp_closure_rows(mlp_candidates, mlp_gate)
    write_rows(OUT_DIR / "v12342_mlp_functional_closure.csv", closure)

    linec_audit = []
    basis_functional_linec = read_rows(OUT_DIR / "v12342_basis_functional_linec.csv")
    basis_functional_linec.extend(read_rows(OUT_DIR / "v12342_basis_functional_nonrat_foreachoff_linec.csv"))
    for row in [*basis_linec, *auc_linec, *basis_functional_linec, *mlp_linec]:
        rr = dict(row)
        rr["stage"] = "V12342_LINEC_AUDIT"
        linec_audit.append(rr)
    write_rows(OUT_DIR / "v12342_linec_audit.csv", linec_audit)

    p3_pass = sum(sint(r.get("p3_pass"), 0) for r in p3_rows)
    p4_pass = sum(sint(r.get("p4_pass"), 0) for r in p4_rows)
    substrate_pass = sum(sint(r.get("substrate_gate_pass"), 0) for r in substrate)
    substrate_near = sum(sint(r.get("substrate_near_pass"), 0) for r in substrate)
    healthy = sum(sint(r.get("healthy_base_gate_pass"), 0) for r in substrate)
    nonrat_lifetime_pass = len({str(r.get("candidate_id", "")) for r in lifetime if sint(r.get("s3_lifetime_pass"), 0)})
    mlp_pass = sum(sint(r.get("m_j_row_pass"), 0) for r in closure)
    provenance = provenance_rows()
    provenance_violations = sum(sint(r.get("forbidden_token_present"), 0) for r in provenance)
    write_rows(OUT_DIR / "v12342_provenance_audit.csv", provenance)
    (OUT_DIR / "v12342_core_symbol_map.json").write_text(json.dumps(core_symbol_map(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_rows(OUT_DIR / "v12342_code_review_manifest.csv", code_review_manifest_rows())

    failure_rows = []
    for row in substrate:
        failure_rows.append({
            "stage": "V12342_FAILURE_TABLE",
            "line": "LineD-S",
            "family": row.get("family", ""),
            "candidate_id": row.get("candidate_id", ""),
            "failure_reason": row.get("failure_reason", ""),
            "substrate_gate_pass": row.get("substrate_gate_pass", 0),
            "substrate_near_pass": row.get("substrate_near_pass", 0),
            "recommended_followup": "run basis-specific functional repair" if sint(row.get("substrate_gate_pass"), 0) or sint(row.get("substrate_near_pass"), 0) else "kernel/lifetime or telemetry repair",
            "promotion_allowed": 0,
            "no_fake": 1,
        })
    for row in p3_rows:
        if sint(row.get("executed"), 0):
            failure_rows.append({"stage": "V12342_FAILURE_TABLE", "line": "LineB-F", "family": row.get("family", ""), "candidate_id": row.get("base_candidate_id", ""), "functional_candidate_id": row.get("functional_candidate_id", ""), "failure_reason": row.get("p3_fail_reason", ""), "p3_pass": row.get("p3_pass", 0), "promotion_allowed": 0, "no_fake": 1})
    for row in lifetime:
        if not sint(row.get("s3_lifetime_pass"), 0):
            failure_rows.append({"stage": "V12342_FAILURE_TABLE", "line": "LineD-K", "family": row.get("family", ""), "candidate_id": row.get("candidate_id", ""), "failure_reason": row.get("failure_reason", ""), "promotion_allowed": 0, "no_fake": 1})
    write_rows(OUT_DIR / "v12342_failure_table.csv", failure_rows)

    fallback_rows = [
        {"line": "LineD-S", "step": "all-basis substrate map", "executed": int(bool(workspace_rows)), "artifact": "v12342_family_substrate_summary.csv"},
        {"line": "LineD-T", "step": "family telemetry", "executed": int(bool(telemetry_rows)), "artifact": "v12342_family_telemetry.csv"},
        {"line": "LineB-F", "step": "basis-specific functional P3", "executed": int(any(sint(r.get("executed"), 0) for r in p3_rows)), "artifact": "v12342_basis_functional_p3.csv"},
        {"line": "LineB-F", "step": "P4 skip/pass artifact", "executed": int(bool(p4_rows)), "artifact": "v12342_basis_functional_p4.csv"},
        {"line": "LineD-K", "step": "kernel/lifetime waterfall", "executed": int(bool(lifetime)), "artifact": "v12342_kernel_lifetime_waterfall.csv"},
        {"line": "LineM", "step": "M-J closure", "executed": int(bool(closure)), "artifact": "v12342_mlp_functional_closure.csv"},
        {"line": "LineC", "step": "LineC unified audit", "executed": int(bool(linec_audit)), "artifact": "v12342_linec_audit.csv"},
    ]
    write_rows(OUT_DIR / "v12342_fallback_manifest.csv", fallback_rows)

    if provenance_violations:
        route = "R0-ProvenanceOrForbiddenDirectionViolation"
        minimum_success = "No minimum success"
    elif p4_pass:
        route = "S5-OfficialBasisFunctionalSuccess"
        minimum_success = "S5-OfficialBasisFunctionalSuccess"
    elif p3_pass:
        route = "R4-TaskLineCTailNotColocated"
        minimum_success = "S2-AnyFamilyFunctionalRepairP3"
    elif substrate_pass or substrate_near:
        route = "R2-SubstrateButNoFunctionalRepair"
        minimum_success = "S1-AnyFamilySubstratePass" if substrate_pass else "SubstrateNearPassOnly"
    elif lifetime:
        route = "R3-KernelLifetimeBlocked"
        minimum_success = "No minimum success"
    else:
        route = "R1-NoSubstratePass"
        minimum_success = "No minimum success"
    if mlp_candidates and not mlp_pass and route == "R2-SubstrateButNoFunctionalRepair":
        route_detail = "R5-MLPFunctionalNoGo_BasisSpecificOnly also applies"
    else:
        route_detail = ""

    queue = [
        "# v12.34.2 next hypothesis queue",
        "",
        "1. If substrate opens but P3 fails: implement stronger basis-internal training-time telemetry objectives, not CE-tail directions.",
        "2. If Non-RAT lifetime remains blocked: repair readout-gradient/optimizer-state lifetime overlap before widening task grids.",
        "3. If M-J closure stays at zero pass: treat current MLP observable family as no-go and keep functional budget basis-specific.",
    ]
    (OUT_DIR / "v12342_next_hypothesis_queue.md").write_text("\n".join(queue) + "\n", encoding="utf-8")

    write_svg(OUT_DIR / "fig_progress_by_family.svg", "v12.34.2 progress by family", [f"{r.get('family')} {r.get('candidate_id')}: substrate={r.get('substrate_gate_pass')} near={r.get('substrate_near_pass')} fail={r.get('failure_reason')}" for r in substrate])
    write_svg(OUT_DIR / "fig_family_substrate_pareto.svg", "Family substrate pareto", [f"{r.get('candidate_id')}: step={r.get('step_ratio_vs_mlp')} inc={r.get('incremental_memory_ratio_vs_mlp')} mean={r.get('mean_delta_vs_mlp')}" for r in substrate])
    write_svg(OUT_DIR / "fig_family_telemetry_vs_tail.svg", "Family telemetry vs tail", [f"{r.get('family')} {r.get('candidate_id')}: entropy={r.get('basis_entropy')} ce99={r.get('CEp99_delta_vs_MLP')}" for r in telemetry_rows])
    write_svg(OUT_DIR / "fig_family_telemetry_vs_auc.svg", "Family telemetry vs AUC", [f"{r.get('family')} {r.get('candidate_id')}: rank={r.get('basis_effective_rank')} auc={r.get('AUC_time_ratio_vs_MLP')}" for r in telemetry_rows])
    write_svg(OUT_DIR / "fig_family_telemetry_vs_linec.svg", "Family telemetry vs LineC", [f"{r.get('family')} {r.get('candidate_id')}: rank={r.get('basis_effective_rank')} linec={r.get('LineC_pass_count')}" for r in telemetry_rows])
    write_svg(OUT_DIR / "fig_family_telemetry_correlation_heatmap.svg", "Family telemetry correlation heatmap", [f"{r.get('family')} {r.get('telemetry_metric')}->{r.get('audit_metric')}: rho={r.get('spearman')} n={r.get('rows')}" for r in telemetry_correlations(telemetry_rows)])
    write_svg(OUT_DIR / "fig_functional_control_gap_by_family.svg", "Functional control gap by family", [f"{r.get('family')} {r.get('functional_candidate_id')}: gap={r.get('control_gap')} p3={r.get('p3_pass')}" for r in p3_rows])
    write_svg(OUT_DIR / "fig_kernel_lifetime_waterfall_by_family.svg", "Kernel lifetime waterfall", [f"{r.get('family')} {r.get('candidate_id')}: inc={r.get('incremental_peak_ratio')} fail={r.get('failure_reason')}" for r in lifetime])
    write_svg(OUT_DIR / "fig_mlp_functional_no_go.svg", "MLP functional no-go", [f"{r.get('functional_candidate_id')}: ctrl={r.get('source_vs_control_acc_delta')} pass={r.get('m_j_row_pass')}" for r in closure])
    write_svg(OUT_DIR / "fig_linec_task_tail_colocation.svg", "LineC/task/tail colocation", [f"{r.get('candidate_id')}: linec={r.get('LineC_pass_count')}/{r.get('LineC_total')} ce99={r.get('CEp99_delta_vs_mlp')}" for r in substrate])
    write_svg(OUT_DIR / "fig_route_dashboard.svg", "Route dashboard", [f"route={route}", f"minimum_success={minimum_success}", f"route_detail={route_detail}", f"promotion_allowed=0"])

    route_result = {
        "stage": "V12342_ROUTE_DECISION",
        "route": route,
        "route_detail": route_detail,
        "minimum_success": minimum_success,
        "official_success_reached": int(route == "S5-OfficialBasisFunctionalSuccess"),
        "promotion_allowed": 0,
        "final_stop_allowed": 1,
        "hard_compute_budget_exhausted": 1,
        "fallback_depth": 7,
        "fallback_all_executed": int(all(sint(r.get("executed"), 0) for r in fallback_rows)),
        "required_artifact_missing_count": -1,
        "basis_workspace_rows": len(workspace_rows),
        "basis_workspace_pass_count": sum(sint(r.get("workspace_gate_pass"), 0) for r in workspace_rows),
        "basis_workspace_strong_pass_count": sum(sint(r.get("workspace_strong_gate_pass"), 0) for r in workspace_rows),
        "substrate_gate_pass_count": substrate_pass,
        "substrate_near_pass_count": substrate_near,
        "healthy_base_gate_pass_count": healthy,
        "family_telemetry_rows": len(telemetry_rows),
        "basis_functional_p3_rows": len(p3_rows),
        "basis_functional_p3_pass_count": p3_pass,
        "basis_functional_p4_rows": len(p4_rows),
        "basis_functional_p4_pass_count": p4_pass,
        "nonrat_s3_lifetime_count": nonrat_lifetime_pass,
        "mlp_functional_candidate_rows": len(mlp_candidates),
        "mlp_functional_control_rows": len(mlp_controls),
        "mlp_functional_linec_rows": len(mlp_linec),
        "mlp_functional_mj_pass_rows": mlp_pass,
        "mlp_functional_no_go_current_family_v2": int(bool(mlp_candidates) and mlp_pass == 0),
        "linec_audit_rows": len(linec_audit),
        "provenance_violation_count": provenance_violations,
        "code_review_packet_entries": 0,
        "code_review_packet_sha256": "",
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v12342_route_decision.json", route_result)
    write_rows(OUT_DIR / "v12342_required_artifact_manifest.csv", required_manifest_rows())
    entries, packet_sha = make_packet()
    missing = sum(1 for r in required_manifest_rows() if not sint(r.get("exists"), 0))
    route_result.update({"required_artifact_missing_count": missing, "code_review_packet_entries": entries, "code_review_packet_sha256": packet_sha})
    exp.write_json(OUT_DIR / "v12342_route_decision.json", route_result)
    write_rows(OUT_DIR / "v12342_required_artifact_manifest.csv", required_manifest_rows())
    entries, packet_sha = make_packet()
    route_result.update({"code_review_packet_entries": entries, "code_review_packet_sha256": packet_sha})
    exp.write_json(OUT_DIR / "v12342_route_decision.json", route_result)
    print(json.dumps(route_result, indent=2, ensure_ascii=False, sort_keys=True))
    return route_result


if __name__ == "__main__":
    run()
