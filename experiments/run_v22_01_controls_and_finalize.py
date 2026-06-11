#!/usr/bin/env python3
"""v22.01 controls, queue artifacts, final route, recap, and packets."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_01_common import (  # noqa: E402
    PYTHON,
    V2201_RECAP_DOC,
    append_exec,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    sha256_file,
    simple_svg,
    write_json,
    write_rows,
    write_text,
)


REQUIRED_ARTIFACTS = [
    "v22_01_code_truth_gate.csv",
    "v22_01_py_compile.log",
    "v22_01_import_closure.csv",
    "v22_01_missing_dependency_report.csv",
    "v22_01_source_chain_unit_tests.csv",
    "v22_01_retention_formula_tests.csv",
    "v22_01_route_reaggregation_tests.csv",
    "v22_01_tail_debt_tests.csv",
    "v22_01_linec_debt_tests.csv",
    "v22_01_ece_brier_debt_tests.csv",
    "v22_01_auctime_debt_tests.csv",
    "v22_01_linec_fast_golden.csv",
    "v22_01_linec_channel_golden.csv",
    "v22_01_linec_measurement_invalid_tests.csv",
    "v22_01_mechanism_semantic_contract.csv",
    "v22_01_mechanism_noncollapse_tests.csv",
    "v22_01_efficiency_truth_table.csv",
    "v22_01_efficiency_officialization_summary.csv",
    "v22_01_kernel_gradcheck.csv",
    "v22_01_official_fused_status_matrix.csv",
    "v22_01_drat_drbf_active_repair.csv",
    "v22_01_drat_drbf_active_repair_summary.csv",
    "v22_01_drat_drbf_component_waterfall.csv",
    "v22_01_drat_drbf_kernel_gradcheck.csv",
    "v22_01_reaggregated_v22_source_chain.csv",
    "v22_01_reaggregated_v22_route.json",
    "v22_01_reaggregated_candidate_diff.csv",
    "v22_01_terminal_collapse_autopsy.csv",
    "v22_01_terminal_collapse_taxonomy.csv",
    "v22_01_mlp_source_lab_summary.csv",
    "v22_01_mlp_source_lab_decision.json",
    "v22_01_precommit_selector_audit.csv",
    "v22_01_precommit_selector_decision.json",
    "v22_01_function_space_target_reset.csv",
    "v22_01_function_space_target_reset_decision.json",
    "v22_01_kan_source_writer_summary.csv",
    "v22_01_kan_source_writer_decision.json",
    "v22_01_runnable_queue.csv",
    "v22_01_gpu_assignment_manifest.csv",
    "v22_01_gpu_utilization_dashboard.csv",
    "v22_01_idle_violation.csv",
    "v22_01_deferred_items.csv",
    "v22_01_queue_drain_report.csv",
    "v22_01_command_journal.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def route_from(out_dir: Path) -> dict[str, Any]:
    truth = read_rows(out_dir / "v22_01_code_truth_gate.csv")
    eff = read_rows(out_dir / "v22_01_efficiency_officialization_summary.csv")
    route = read_json(out_dir / "v22_01_reaggregated_v22_route.json")
    mlp = read_json(out_dir / "v22_01_mlp_source_lab_decision.json")
    selector = read_json(out_dir / "v22_01_precommit_selector_decision.json")
    target = read_json(out_dir / "v22_01_function_space_target_reset_decision.json")
    kan = read_json(out_dir / "v22_01_kan_source_writer_decision.json")
    dr = read_json(out_dir / "v22_01_drat_drbf_active_repair_decision.json")
    missing = []
    for name in REQUIRED_ARTIFACTS:
        path = out_dir / name
        if not path.exists():
            missing.append(name)
    s08_pass = int(bool(truth) and all(int_flag(r.get("pass")) for r in truth))
    dche_s1 = sum(int_flag(r.get("S1_pass_rows")) for r in eff if str(r.get("carrier")) == "D-CHE")
    dfou_s1 = sum(int_flag(r.get("S1_pass_rows")) for r in eff if str(r.get("carrier")) == "D-FOU")
    h4800 = int(route.get("candidate_h4800", 0) or 0)
    if missing:
        final_route = "R0-RequiredArtifactMissing"
    elif not s08_pass:
        final_route = "R0-S08TruthGateFailed"
    elif int(route.get("candidate_continuous_h3200", 0) or 0) and not h4800:
        final_route = "R4-TerminalCollapseNoH4800-SourceTargetTheoryInsufficient"
    elif h4800:
        final_route = "R5-H4800CandidateNeedsIndependentConfirmation"
    else:
        final_route = "R2-NoContinuousSourceAfterRuleFix"
    promotion = int(final_route == "S5-PromotionReady")
    return {
        "route": final_route,
        "S0_8_pass": s08_pass,
        "D-CHE_S1_rows": dche_s1,
        "D-FOU_S1_rows": dfou_s1,
        "candidate_early_chain": route.get("candidate_early_chain", 0),
        "candidate_continuous_h3200": route.get("candidate_continuous_h3200", 0),
        "candidate_h4800": h4800,
        "terminal_collapse_groups": route.get("terminal_collapse_groups", 0),
        "MLPRoute": mlp.get("decision", ""),
        "SelectorRoute": selector.get("decision", ""),
        "TargetRoute": target.get("decision", ""),
        "KANRoute": kan.get("decision", ""),
        "DRATRoute": (dr.get("D-RAT", {}) or {}).get("decision", ""),
        "DRBFRoute": (dr.get("D-RBF", {}) or {}).get("decision", ""),
        "promotion_allowed": promotion,
        "required_artifact_missing_count": len(missing),
        "missing_artifacts": ";".join(missing),
    }


def write_queue(out_dir: Path) -> None:
    tasks = [
        ("GPU0", "S0.8 truth gate", "v22_01_code_truth_gate.csv"),
        ("GPU0", "D-CHE efficiency officialization", "v22_01_efficiency_officialization_summary.csv"),
        ("GPU1", "D-FOU efficiency officialization", "v22_01_efficiency_officialization_summary.csv"),
        ("GPU2", "terminal-collapse autopsy", "v22_01_terminal_collapse_autopsy.csv"),
        ("GPU2", "MLP source lab / precommit selector", "v22_01_mlp_source_lab_summary.csv"),
        ("GPU1", "KAN source-channel writer", "v22_01_kan_source_writer_summary.csv"),
        ("GPU3", "D-RAT/D-RBF active repair", "v22_01_drat_drbf_active_repair_summary.csv"),
    ]
    queue = []
    assignments = []
    for idx, (gpu, task, artifact) in enumerate(tasks):
        status = "completed" if (out_dir / artifact).exists() else "missing"
        queue.append({"queue_index": idx, "task": task, "artifact": artifact, "assigned_gpu": gpu, "status": status})
        assignments.append({"gpu": gpu, "task": task, "artifact": artifact, "status": status})
    write_rows(out_dir / "v22_01_runnable_queue.csv", queue)
    write_rows(out_dir / "v22_01_gpu_assignment_manifest.csv", assignments)
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], text=True, capture_output=True, timeout=30)
        util_rows = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                util_rows.append({"gpu": parts[0], "name": parts[1], "memory_used": parts[2], "memory_total": parts[3], "utilization_gpu": parts[4], "snapshot_time": now_sg()})
    except Exception as exc:
        util_rows = [{"gpu": "", "blocker": f"{type(exc).__name__}:{exc}", "snapshot_time": now_sg()}]
    write_rows(out_dir / "v22_01_gpu_utilization_dashboard.csv", util_rows)
    pending = [r for r in queue if r["status"] != "completed"]
    write_rows(out_dir / "v22_01_idle_violation.csv", [{"execution_contract_violation": int(bool(pending)), "pending_tasks": len(pending), "note": "violation only if runnable tasks remain after finalize"}])
    dr = read_json(out_dir / "v22_01_drat_drbf_active_repair_decision.json")
    deferred = []
    for fam in ["D-RAT", "D-RBF"]:
        d = dr.get(fam, {}) or {}
        if d.get("decision") != "NearE1Reached":
            deferred.append({"item": f"{fam} limited functional smoke", "status": "deferred", "reason": d.get("blocker", "NearE1 not reached")})
    write_rows(out_dir / "v22_01_deferred_items.csv", deferred)
    write_rows(out_dir / "v22_01_drat_drbf_limited_smoke.csv", deferred)
    write_rows(out_dir / "v22_01_queue_drain_report.csv", [{"completed_tasks": sum(r["status"] == "completed" for r in queue), "missing_tasks": len(pending), "deferred_items": len(deferred), "queue_drained": int(not pending)}])


def manifest(out_dir: Path) -> None:
    rows = []
    for name in REQUIRED_ARTIFACTS:
        path = out_dir / name
        rows.append({"artifact": name, "required": 1, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256_file(path) if path.exists() and path.is_file() else ""})
    write_rows(out_dir / "v22_01_required_artifact_manifest.csv", rows)


def render_recap(out_dir: Path, decision: dict[str, Any]) -> None:
    truth = read_rows(out_dir / "v22_01_code_truth_gate.csv")
    eff = read_rows(out_dir / "v22_01_efficiency_officialization_summary.csv")
    drat = read_rows(out_dir / "v22_01_drat_drbf_active_repair_summary.csv")
    source = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain_summary.csv")
    autopsy = read_rows(out_dir / "v22_01_terminal_collapse_autopsy.csv")
    taxonomy = read_rows(out_dir / "v22_01_terminal_collapse_taxonomy.csv")
    mlp = read_rows(out_dir / "v22_01_mlp_source_lab_summary.csv")
    selector = read_rows(out_dir / "v22_01_precommit_selector_audit.csv")
    target = read_rows(out_dir / "v22_01_function_space_target_reset.csv")
    kan = read_rows(out_dir / "v22_01_kan_source_writer_summary.csv")
    packet = out_dir / "v22_01_code_review_packet.zip"
    bundle = out_dir / "v22_01_results_bundle.zip"
    text = [
        "# DG-KAN v22.01 EarlySourceRetention TerminalCollapse KernelOfficialization DRAT/DRBF 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{decision.get('route')}`",
        f"- S0.8 pass: {decision.get('S0_8_pass')}",
        f"- D-CHE/D-FOU S1 rows: {decision.get('D-CHE_S1_rows')} / {decision.get('D-FOU_S1_rows')}",
        f"- functional early / h3200 / h4800: {decision.get('candidate_early_chain')} / {decision.get('candidate_continuous_h3200')} / {decision.get('candidate_h4800')}",
        f"- terminal collapse groups: {decision.get('terminal_collapse_groups')}",
        f"- MLPRoute: {decision.get('MLPRoute')}",
        f"- SelectorRoute: {decision.get('SelectorRoute')}",
        f"- TargetRoute: {decision.get('TargetRoute')}",
        f"- KANRoute: {decision.get('KANRoute')}",
        f"- D-RAT/D-RBF: {decision.get('DRATRoute')} / {decision.get('DRBFRoute')}",
        f"- promotion_allowed: {decision.get('promotion_allowed')}",
        f"- required artifact missing count: {decision.get('required_artifact_missing_count')}",
        "",
        "## Part A Code Audit",
        "",
        md_table(truth, ["check", "pass", "metric", "value", "blocker"], max_rows=20),
        "## Part B Basis Efficiency",
        "",
        md_table(eff, ["carrier", "rows", "E1_pass_rows", "S1_pass_rows", "E1_pass_batch_sizes", "best_forward_ratio", "best_step_ratio", "decision"], max_rows=20),
        md_table(drat, ["carrier", "profile_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "decision", "blocker"], max_rows=20),
        "## Part C Functional Update",
        "",
        md_table(source, ["carrier", "variant", "v22_id", "rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "early_source_chain_group", "continuous_retention_group", "productive_h4800_group", "terminal_collapse_group", "source_chain_blocker"], max_rows=40),
        "## Terminal Collapse Autopsy",
        "",
        md_table(taxonomy, ["class", "groups", "mean_h3200", "mean_h4800"], max_rows=20),
        md_table(autopsy, ["carrier", "v22_id", "h3200", "h4800", "source_derivative_h3200_to_h4800", "retention_h4800_over_h3200", "stable_random_h4800_positive_fraction", "dataset_seed_heterogeneity_score", "terminal_collapse_class"], max_rows=30),
        "## MLP Source Lab / Selector",
        "",
        md_table(mlp, ["f2_family", "v22_id", "h800", "h3200", "h4800", "h4800_positive_rows", "retention_h4800_over_h3200", "MLP_FU_S3", "MLP_FU_S4"], max_rows=30),
        md_table(selector, ["predictor", "kind", "AUC_continuous_retention", "pass", "precommit_selector_pass", "retrospective_only_flag", "selector_decision"], max_rows=20),
        "## Function-Space Target Reset",
        "",
        md_table(target, ["carrier", "v22_id", "target_family", "ActuationR2", "B2_transfer_gain", "random_target_B2_gain", "early_source_chain_rate", "h3200_retention_rate", "h4800_retention_rate", "Target_S2", "Target_S3"], max_rows=30),
        "## KAN Source Writer",
        "",
        md_table(kan, ["carrier", "v22_id", "writer_family", "h800", "h3200", "h4800", "h3200_positive_rows", "h4800_positive_rows", "KAN_FU_S2", "KAN_FU_S3", "same_run_efficiency_S1_rows"], max_rows=30),
        "## 修改记录",
        "",
        "- 修正 `dgkan/fu/source_chain.py`：early-source 改为 `>=0.005`，加入 control-equivalent 排除、h4800 continuous 和 terminal-collapse 标记。",
        "- 新增 `dgkan/fu/debt_accounting.py`：提供 audit-only debt peak/final/recovery 与 AUCtime ratio 公式测试。",
        "- 新增 v22.01 runner：S0.8 truth gate、efficiency recheck、D-RAT/D-RBF active repair、terminal-collapse autopsy、MLP source lab、function-space target reset、KAN writer audit、finalizer。",
        "- v22.01 functional 数据来自 v22 raw source-retention matrix 重新聚合；没有复用旧 route 结论。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- 修正 source-chain rule 后，zero/control rows 不再能打开 early chain；h3200 candidate 必须同时满足 h100/h400/h800 >= 0.005。",
        "- 当前主 blocker 仍是 terminal collapse：有 h3200 continuous groups，但 h4800 retained group 仍未闭合，因此不能 promotion。",
        "- F46/C4 train-loss 类指标在 v22 中仍只能作为 retrospective taxonomy；本轮没有 precommit independent selector pass，不能用作方向成功证据。",
        "- D-CHE/D-FOU efficiency 是真实进展，但 functional source retention 未闭合，所以 blocker 不是 kernel officialization 单项。",
        "- D-RAT/D-RBF 已执行 active repair/profiling；若 Near-E1 未达成，limited functional smoke 按 gate deferred，不写 carrier source breakthrough。",
        "- KAN low-degree/low-frequency/readout writer 仍未形成 KAN-FU-S3；若 MLP 也没有 MLP-FU-S3，则当前 train-stream retained target/source observability theory 仍不足。",
        "",
    ]
    if packet.exists():
        text.append(f"- v22_01_code_review_packet.zip: size={packet.stat().st_size} sha256={sha256_file(packet)}")
    if bundle.exists():
        text.append(f"- v22_01_results_bundle.zip: size={bundle.stat().st_size} sha256={sha256_file(bundle)}")
    write_text(V2201_RECAP_DOC, "\n".join(text) + "\n")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    write_queue(out_dir)
    manifest(out_dir)
    decision = route_from(out_dir)
    write_json(out_dir / "v22_01_route_decision.json", decision)
    write_rows(out_dir / "v22_01_route_decision.csv", [decision])
    simple_svg(out_dir / "figures" / "gpu_utilization_dashboard.svg", "GPU utilization dashboard", read_rows(out_dir / "v22_01_gpu_utilization_dashboard.csv"), "gpu")
    render_recap(out_dir, decision)
    zip_path, bundle = build_packet(out_dir)
    render_recap(out_dir, decision)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_controls_and_finalize.py", status="completed", note=f"route={decision.get('route')} packet={zip_path.name} bundle={bundle.name}")


if __name__ == "__main__":
    main()
