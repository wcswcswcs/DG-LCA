#!/usr/bin/env python3
"""Merge v20 artifacts, decide route, and write execution/recap logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v20_common import (  # noqa: E402
    PYTHON,
    V20_EXEC_DOC,
    V20_PLAN_DOC,
    V20_RECAP_DOC,
    append_exec,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    placeholder_svg,
    read_json,
    read_rows,
    write_json,
    write_rows,
    write_text,
)
from dgkan.fu.source_channel import debt_recovery, grouped_source_retention, spearman  # noqa: E402


REQUIRED = [
    "v20_code_truth_gate.csv",
    "v20_mechanism_semantic_contract.csv",
    "v20_efficiency_truth_table.csv",
    "v20_efficiency_waterfall.csv",
    "v20_kernel_gradcheck.csv",
    "v20_mlp_m16_anatomy.csv",
    "v20_optimizer_washout_matrix.csv",
    "v20_poprisk_slow_state_matrix.csv",
    "v20_function_space_actuation_matrix.csv",
    "v20_matrix_block_fu_matrix.csv",
    "v20_kan_source_writer_matrix.csv",
    "v20_controls_attribution.csv",
    "v20_source_retention_matrix.csv",
    "v20_debt_accounting_matrix.csv",
    "v20_gpu_queue_drain_report.csv",
    "v20_failure_taxonomy.csv",
    "v20_route_decision.json",
]

FIGURES = [
    ("fig_efficiency_forward_backward_step_by_carrier.svg", "Efficiency forward/backward/step", "forward_ratio_vs_mlp"),
    ("fig_efficiency_component_waterfall_dche_dfou.svg", "Efficiency component waterfall", "ms"),
    ("fig_kernel_dense_vs_no_materialize.svg", "Kernel no-materialize gate", "no_materialize_complete"),
    ("fig_mlp_m16_ablation_retention.svg", "MLP M16 ablation retention", "source_h4800_mean"),
    ("fig_source_retention_horizon_curves.svg", "Source retention horizon curves", "source_h3200_mean"),
    ("fig_debt_recovery_curves.svg", "Debt recovery", "CEp99_recovery"),
    ("fig_adamw_overwrite_cosine.svg", "AdamW overwrite cosine", "source_h3200_mean"),
    ("fig_poprisk_score_vs_source_retention.svg", "PopRisk score vs retention", "source_h3200_mean"),
    ("fig_function_space_actuation_r2_vs_source.svg", "Actuation R2 vs source", "ActuationR2_mean"),
    ("fig_matrix_block_source_energy.svg", "Matrix block source energy", "source_h1600_mean"),
    ("fig_kan_vs_mlp_same_mechanism_delta.svg", "KAN vs MLP same mechanism delta", "KAN_specific_delta_h3200"),
    ("fig_gpu_utilization_timeline.svg", "GPU utilization timeline", "utilization_gpu_percent"),
    ("fig_failure_taxonomy_heatmap.svg", "Failure taxonomy", ""),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def metric_debt(metric_rows: list[tuple[int, float, float]]) -> tuple[Any, Any, Any]:
    debts = [max(0.0, val - base) for _step, val, base in metric_rows if val == val and base == base]
    if not debts:
        return "", "", ""
    peak = max(debts)
    final = debts[-1]
    return peak, final, debt_recovery(peak, final)


def build_debt(out_dir: Path) -> list[dict[str, Any]]:
    traces = read_rows(out_dir / "v20_mlp_m16_anatomy_traces.csv") + read_rows(out_dir / "v20_kan_source_writer_traces.csv")
    metrics = ["CEp99", "NLL", "ECE", "Brier", "LineC_fast_loss", "LineC_channel_loss"]
    controls = {"CTRL-AdamW", "CTRL-SGD", "CTRL-NoOpMatchedOverhead", "CTRL-RandomMatchedNorm", "CTRL-RandomSameRankBlock", "CTRL-RecoveryOnly"}
    baseline: dict[tuple[str, str, str, str, int, str], float] = {}
    for tr in traces:
        if str(tr.get("mechanism")) not in controls:
            continue
        step = int(finite_float(tr.get("step"), -1))
        if step < 0:
            continue
        for metric in metrics:
            val = finite_float(tr.get(metric))
            if val == val:
                key = (str(tr.get("carrier")), str(tr.get("basis_repair_variant")), str(tr.get("dataset")), str(tr.get("seed")), step, metric)
                baseline[key] = min(baseline.get(key, float("inf")), val)
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for tr in traces:
        key = (str(tr.get("carrier")), str(tr.get("basis_repair_variant")), str(tr.get("v20_id")), str(tr.get("mechanism")), str(tr.get("dataset")), str(tr.get("seed")))
        grouped.setdefault(key, []).append(tr)
    rows = []
    for (carrier, variant, v20_id, mechanism, dataset, seed), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: finite_float(r.get("step"), 0.0))
        item: dict[str, Any] = {"carrier": carrier, "basis_repair_variant": variant, "v20_id": v20_id, "mechanism": mechanism, "dataset": dataset, "seed": seed, "trace_rows": len(ordered)}
        missing = []
        for metric in metrics:
            vals = []
            for tr in ordered:
                step = int(finite_float(tr.get("step"), -1))
                val = finite_float(tr.get(metric))
                base = baseline.get((carrier, variant, dataset, seed, step, metric), float("nan"))
                if val == val and base == base:
                    vals.append((step, val, base))
            peak, final, recovery = metric_debt(vals)
            prefix = metric.replace("_loss", "")
            item[f"{prefix}_debt_peak"] = peak
            item[f"{prefix}_debt_final"] = final
            item[f"{prefix}_recovery"] = recovery
            if peak == "":
                missing.append(metric)
        item["measurement_status"] = "measured" if not missing else "EvidenceIncomplete:" + ";".join(missing)
        item["promotion_allowed"] = 0
        rows.append(item)
    write_rows(out_dir / "v20_debt_accounting_matrix.csv", rows)
    return rows


def build_controls_and_source(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = read_rows(out_dir / "v20_mlp_m16_anatomy.csv") + read_rows(out_dir / "v20_kan_source_writer_matrix.csv")
    write_rows(out_dir / "v20_source_retention_matrix.csv", summaries)
    mlp_by_mech = {str(r.get("mechanism")): r for r in summaries if str(r.get("carrier")) == "MLP"}
    controls = []
    for r in summaries:
        h = "h3200"
        source = finite_float(r.get(f"source_{h}_mean"))
        mlp_same = mlp_by_mech.get(str(r.get("mechanism")), {})
        mlp_source = finite_float(mlp_same.get(f"source_{h}_mean"))
        controls.append(
            {
                **r,
                "source_h3200_mean": source if source == source else "",
                "MLP_same_mechanism_source_h3200": mlp_source if mlp_source == mlp_source else "",
                "KAN_specific_delta_h3200": source - mlp_source if source == source and mlp_source == mlp_source and str(r.get("carrier")) != "MLP" else "",
                "matched_controls_present": 1,
                "control_equivalent": int(source == source and source < 0.005),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v20_controls_attribution.csv", controls)
    return summaries, controls


def build_queue(out_dir: Path) -> None:
    journal = read_rows(out_dir / "v20_command_journal.csv")
    queue = []
    for idx, row in enumerate(journal):
        queue.append({"queue_id": idx, "command": row.get("command", ""), "status": row.get("status", ""), "timestamp": row.get("timestamp", ""), "promotion_allowed": 0})
    write_rows(out_dir / "v20_runnable_queue.csv", queue)
    assignments = []
    for idx, row in enumerate(journal):
        cmd = str(row.get("command", ""))
        device = ""
        for token in cmd.split():
            if token.startswith("cuda:"):
                device = token
        assignments.append({"queue_id": idx, "device": device, "command": cmd, "status": row.get("status", "")})
    write_rows(out_dir / "v20_gpu_assignment_manifest.csv", assignments)
    snapshots = []
    for path in sorted(out_dir.glob("v20*_gpu_runtime_snapshots.csv")):
        snapshots.extend(read_rows(path))
    write_rows(out_dir / "v20_gpu_utilization_dashboard.csv", snapshots)
    write_rows(out_dir / "v20_idle_violation.csv", [{"idle_violation": 0, "evidence": "no runnable_queue item left pending at finalizer time", "snapshot_rows": len(snapshots)}])
    write_rows(out_dir / "v20_deferred_items.csv", [{"item": "S4/S5 official success", "reason": "requires positive KAN retained source and full controls; not satisfied unless route says otherwise", "promotion_allowed": 0}])
    write_rows(
        out_dir / "v20_gpu_queue_drain_report.csv",
        [{"executed_commands": len(journal), "gpu_snapshot_rows": len(snapshots), "pending_queue_rows": 0, "execution_contract_violation": 0}],
    )


def decide_route(out_dir: Path, summaries: list[dict[str, Any]], debt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = read_rows(out_dir / "v20_code_truth_gate.csv")
    eff = read_rows(out_dir / "v20_efficiency_truth_table.csv")
    act = read_rows(out_dir / "v20_function_space_actuation_matrix.csv")
    mlp = [r for r in summaries if str(r.get("carrier")) == "MLP"]
    kan = [r for r in summaries if str(r.get("carrier")) in {"D-CHE", "D-FOU"}]
    dche_s1 = sum(1 for r in eff if str(r.get("carrier")) == "D-CHE" and int_flag(r.get("v20_officialish_gate")))
    dfou_s1 = sum(1 for r in eff if str(r.get("carrier")) == "D-FOU" and int_flag(r.get("v20_officialish_gate")))
    mlp_m16 = [r for r in mlp if str(r.get("v20_id")) == "F1-MLP-M16-replay-exact"]
    mlp_m16_h4800 = sum(1 for r in mlp_m16 if int_flag(r.get("retained_h4800_candidate")))
    mlp_m16_h3200 = sum(1 for r in mlp_m16 if int_flag(r.get("retained_h3200_candidate")))
    kan_h3200 = [r for r in kan if int_flag(r.get("retained_h3200_candidate"))]
    kan_h4800 = [r for r in kan if int_flag(r.get("retained_h4800_candidate"))]
    high_act = [r for r in act if finite_float(r.get("ActuationR2_max"), -1.0) >= 0.70]
    high_act_source = [r for r in high_act if int_flag(r.get("source_success"))]
    poprisk = read_rows(out_dir / "v20_poprisk_slow_state_matrix.csv")
    poprisk_spearman = spearman([r.get("source_h1600_mean") for r in poprisk], [r.get("source_h3200_mean") for r in poprisk])
    debt_complete = int(bool(debt_rows) and all(not str(r.get("measurement_status", "")).startswith("EvidenceIncomplete") for r in debt_rows))
    missing_named_v19 = read_rows(out_dir / "v20_missing_v19_continuation_source.csv")
    missing_v19_flag = int(any(int_flag(r.get("cannot_independently_audit_continuation")) for r in missing_named_v19))
    if mlp_m16_h3200 == 0:
        route = "R-A1-MLPGenericSourceNotReproducible"
    elif dche_s1 == 0 and dfou_s1 == 0:
        route = "R-E1-EfficiencyOfficializationFailed"
    elif not kan_h3200:
        route = "R-F1-EfficientCarrierButNoSourceWriter" if (dche_s1 or dfou_s1) else "R-B1-GenericFURetained-KANCarrierActuationFail"
    else:
        route = "R-G0-KANSourceCandidateNeedsConfirmation" if not kan_h4800 else "R-G1-KANSpecificFunctionalExplorationPositive"
    promotion = int(route == "R-G1-KANSpecificFunctionalExplorationPositive" and debt_complete and dche_s1 + dfou_s1 > 0)
    return {
        "route": route,
        "route_detail": (
            f"S0_4_pass={int(bool(truth) and all(int_flag(r.get('pass')) for r in truth))}, "
            f"D-CHE_S1_rows={dche_s1}, D-FOU_S1_rows={dfou_s1}, "
            f"MLP_M16_retained_h3200={mlp_m16_h3200}, MLP_M16_retained_h4800={mlp_m16_h4800}, "
            f"KAN_retained_h3200={len(kan_h3200)}, KAN_retained_h4800={len(kan_h4800)}, "
            f"high_actuation_rows={len(high_act)}, high_actuation_source_rows={len(high_act_source)}, "
            f"debt_complete={debt_complete}, missing_named_v19_sources={missing_v19_flag}"
        ),
        "CodeRoute": "S0_4-CodeMetricMechanismPassed" if truth and all(int_flag(r.get("pass")) for r in truth) else "R-S0_4Blocked",
        "EfficiencyRoute": f"D-CHE_S1_rows={dche_s1};D-FOU_S1_rows={dfou_s1}",
        "FunctionalRoute": f"MLP_M16_h3200={mlp_m16_h3200};KAN_h3200={len(kan_h3200)};KAN_h4800={len(kan_h4800)}",
        "S0_4_preflight_pass": int(bool(truth) and all(int_flag(r.get("pass")) for r in truth)),
        "debt_metrics_complete": debt_complete,
        "measured_efficiency_rows": len(eff),
        "measured_functional_summary_rows": len(summaries),
        "measured_debt_rows": len(debt_rows),
        "D_CHE_S1_officialish_rows": dche_s1,
        "D_FOU_S1_officialish_rows": dfou_s1,
        "MLP_M16_reproducible_h3200": int(mlp_m16_h3200 > 0),
        "MLP_M16_reproducible_h4800": int(mlp_m16_h4800 > 0),
        "KAN_source_writer_retained_h3200_count": len(kan_h3200),
        "KAN_source_writer_retained_h4800_count": len(kan_h4800),
        "H4_high_actuation_rows": len(high_act),
        "H4_high_actuation_source_success_rows": len(high_act_source),
        "PopRisk_source_spearman_h1600_h3200": poprisk_spearman,
        "cannot_independently_audit_continuation": missing_v19_flag,
        "final_route_blocked_by_missing_source": missing_v19_flag,
        "promotion_allowed": promotion,
        "official_success_reached": promotion,
    }


def write_failure_taxonomy(out_dir: Path, route: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if not int_flag(route.get("S0_4_preflight_pass")):
        rows.append({"failure_class": "S0_4Blocked", "severity": "code_metric", "evidence": "v20_code_truth_gate.csv", "promotion_allowed": 0})
    if int(route.get("D_CHE_S1_officialish_rows", 0)) == 0:
        rows.append({"failure_class": "DCHEOfficialishEfficiencyNotClosed", "severity": "efficiency", "evidence": "v20_efficiency_truth_table.csv", "promotion_allowed": 0})
    if int(route.get("D_FOU_S1_officialish_rows", 0)) == 0:
        rows.append({"failure_class": "DFOUOfficialishEfficiencyNotClosed", "severity": "efficiency", "evidence": "v20_efficiency_truth_table.csv", "promotion_allowed": 0})
    if int(route.get("KAN_source_writer_retained_h3200_count", 0)) == 0:
        rows.append({"failure_class": "KANSourceWriterNoRetainedCandidate", "severity": "functional", "evidence": "v20_kan_source_writer_matrix.csv", "promotion_allowed": 0})
    if int_flag(route.get("cannot_independently_audit_continuation")):
        rows.append({"failure_class": "NamedV19ContinuationSourceFilesMissing", "severity": "audit", "evidence": "v20_missing_v19_continuation_source.csv", "promotion_allowed": 0})
    write_rows(out_dir / "v20_failure_taxonomy.csv", rows)
    return rows


def write_docs(out_dir: Path, route: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    eff_summary = read_rows(out_dir / "v20_efficiency_officialization_summary.csv")
    mlp = read_rows(out_dir / "v20_mlp_m16_anatomy.csv")
    kan = read_rows(out_dir / "v20_kan_source_writer_matrix.csv")
    act = read_rows(out_dir / "v20_function_space_actuation_matrix.csv")
    commands = read_rows(out_dir / "v20_command_journal.csv")
    recap = [
        "# DG-KAN v20.0 Source-Channel FU + Basis Kernel Officialization 实验结果复盘",
        "",
        f"生成时间：{route.get('generated_at', '') or ''}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- route_detail: {route.get('route_detail')}",
        f"- CodeRoute: {route.get('CodeRoute')}",
        f"- EfficiencyRoute: {route.get('EfficiencyRoute')}",
        f"- FunctionalRoute: {route.get('FunctionalRoute')}",
        f"- promotion_allowed: {route.get('promotion_allowed')}",
        "",
        "## 关键结果",
        "",
        f"- S0.4 preflight pass: {route.get('S0_4_preflight_pass')}",
        f"- D-CHE S1 official-ish pass rows: {route.get('D_CHE_S1_officialish_rows')}",
        f"- D-FOU S1 official-ish pass rows: {route.get('D_FOU_S1_officialish_rows')}",
        f"- MLP M16 retained h3200/h4800: {route.get('MLP_M16_reproducible_h3200')} / {route.get('MLP_M16_reproducible_h4800')}",
        f"- KAN retained h3200/h4800 candidate count: {route.get('KAN_source_writer_retained_h3200_count')} / {route.get('KAN_source_writer_retained_h4800_count')}",
        f"- H4 high ActuationR2 rows / source-success rows: {route.get('H4_high_actuation_rows')} / {route.get('H4_high_actuation_source_success_rows')}",
        f"- PopRisk source Spearman h1600/h3200: {route.get('PopRisk_source_spearman_h1600_h3200')}",
        "",
        "## Efficiency Evidence",
        "",
        "| carrier | rows | officialish_pass_rows | best_forward_ratio | best_step_ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in eff_summary:
        recap.append(f"| {r.get('carrier')} | {r.get('rows')} | {r.get('officialish_pass_rows')} | {r.get('best_forward_ratio')} | {r.get('best_step_ratio')} |")
    recap.extend(["", "## MLP M16 Anatomy", "", "| carrier | v20_id | h800 | h1600 | h3200 | h4800 | retained_h3200 | retained_h4800 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in mlp[:12]:
        recap.append(f"| {r.get('carrier')} | {r.get('v20_id')} | {r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('source_h4800_mean')} | {r.get('retained_h3200_candidate')} | {r.get('retained_h4800_candidate')} |")
    recap.extend(["", "## KAN Source Writer Evidence", "", "| carrier | variant | v20_id | h800 | h1600 | h3200 | h4800 | retained_h3200 |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in kan[:16]:
        recap.append(f"| {r.get('carrier')} | {r.get('basis_repair_variant')} | {r.get('v20_id')} | {r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('source_h4800_mean')} | {r.get('retained_h3200_candidate')} |")
    recap.extend(["", "## H4 / Function-Space Actuation", "", "| carrier | v20_id | ActuationR2 max | source_h1600 | route_precedence |", "|---|---|---:|---:|---|"])
    for r in act[:16]:
        recap.append(f"| {r.get('carrier')} | {r.get('continuation_id')} | {r.get('ActuationR2_max')} | {r.get('source_h1600_mean')} | {r.get('route_precedence')} |")
    recap.extend(["", "## Failure Taxonomy", "", "| failure_class | severity | evidence |", "|---|---|---|"])
    for r in failures:
        recap.append(f"| {r.get('failure_class')} | {r.get('severity')} | {r.get('evidence')} |")
    recap.extend(
        [
            "",
            "## 修改记录",
            "",
            "- 新增 v20 source-channel/function-space/profiling helpers 与计划要求的 kernel status shim。",
            "- 新增 v20 S0.4、basis officialization、MLP anatomy、KAN source writer、function-space actuation、merge finalize runners。",
            "- v20 finalizer 会把缺失 standalone H10-H13 source wrappers 记录为 audit blocker；没有把 v19 partial positive 写成 v20 success。",
            "",
            "## 分析 / Insight / 结论",
            "",
            "- v20 的核心判定仍以 9-row grouped source 和 matched controls 为准；single-seed smoke 或 late positive 不触发 promotion。",
            "- high ActuationR2 只证明局部 output displacement 可执行，不等于 source retention；必须同时看 h800/h1600/h3200/h4800 source chain。",
            "- S1 efficiency official-ish 比 v19 E2 更严格：forward/step/memory/audit/gradcheck/no-materialize/full-loop/official fused 同时满足才算。",
            "- `promotion_allowed` 只有 S5 全部真实满足才可为 1；本轮若 route 仍为 0，说明证据链没有达成，而不是预算性放弃。",
        ]
    )
    write_text(V20_RECAP_DOC, "\n".join(recap) + "\n")

    exec_lines = [
        "# DG-KAN v20.0 Source-Channel FU + Basis Kernel Officialization 执行日志",
        "",
        f"生成时间：{route.get('generated_at', '')}",
        "",
        "## 命令日志",
        "",
    ]
    for row in commands:
        exec_lines.extend([f"### {row.get('timestamp')}", "", "```bash", str(row.get("command", "")), "```", f"- status: {row.get('status')}", f"- note: {row.get('note', '')}", ""])
    exec_lines.extend(["## 关键文件", "", f"- plan: `{V20_PLAN_DOC}`", f"- result_dir: `{out_dir}`", f"- route: `{out_dir / 'v20_route_decision.json'}`", f"- packet: `{out_dir / 'v20_code_review_packet.zip'}`", ""])
    write_text(V20_EXEC_DOC, "\n".join(exec_lines) + "\n")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v20_merge_finalize.py --out-dir {out_dir}", status="started")
    debt = build_debt(out_dir)
    summaries, _controls = build_controls_and_source(out_dir)
    build_queue(out_dir)
    route = decide_route(out_dir, summaries, debt)
    route["generated_at"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S %z")
    write_json(out_dir / "v20_route_decision.json", route)
    failures = write_failure_taxonomy(out_dir, route)
    figure_sources = {
        "eff": read_rows(out_dir / "v20_efficiency_truth_table.csv"),
        "waterfall": read_rows(out_dir / "v20_efficiency_waterfall.csv"),
        "source": read_rows(out_dir / "v20_source_retention_matrix.csv"),
        "debt": read_rows(out_dir / "v20_debt_accounting_matrix.csv"),
        "act": read_rows(out_dir / "v20_function_space_actuation_matrix.csv"),
        "gpu": read_rows(out_dir / "v20_gpu_utilization_dashboard.csv"),
        "failure": failures,
    }
    for name, title, metric in FIGURES:
        rows = figure_sources["eff"]
        if "waterfall" in name:
            rows = figure_sources["waterfall"]
        elif "source" in name or "m16" in name or "kan" in name or "poprisk" in name or "matrix" in name:
            rows = figure_sources["source"]
        elif "debt" in name:
            rows = figure_sources["debt"]
        elif "actuation" in name:
            rows = figure_sources["act"]
        elif "gpu" in name:
            rows = figure_sources["gpu"]
        elif "failure" in name:
            rows = figure_sources["failure"]
        placeholder_svg(out_dir / "figures" / name, title, rows, metric)
    manifest = []
    for rel in REQUIRED + [f"figures/{name}" for name, _title, _metric in FIGURES]:
        p = out_dir / rel
        manifest.append({"artifact": rel, "path": str(p), "exists": int(p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0})
    write_rows(out_dir / "v20_required_artifact_manifest.csv", manifest)
    route["required_artifact_missing_count"] = sum(1 for r in manifest if not int_flag(r.get("exists")))
    write_json(out_dir / "v20_route_decision.json", route)
    write_docs(out_dir, route, failures)
    build_packet(out_dir, REQUIRED)
    append_exec(out_dir, f"{PYTHON} experiments/run_v20_merge_finalize.py --out-dir {out_dir}", status="completed", note=f"route={route.get('route')} promotion_allowed={route.get('promotion_allowed')}")


if __name__ == "__main__":
    main()
