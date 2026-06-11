#!/usr/bin/env python3
"""v22.04 final route, recap, execution log, and packets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_04_common import (  # noqa: E402
    PYTHON,
    V2204_RECAP_DOC,
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
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _first(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    return next((r for r in rows if str(r.get(key, "")) == value), {})


def _candidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("mechanism", "")).startswith("M") and not str(r.get("v22_id", "")).startswith("CTRL")]


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = _candidate(rows)
    valid_chain = [
        r
        for r in candidates
        if int_flag(r.get("early_source_chain_group")) and int_flag(r.get("continuous_h3200_group"))
    ]
    pool = valid_chain or candidates
    return sorted(pool, key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)[0] if pool else {}


def _official_d1_artifact(path: Path) -> bool:
    excluded = {
        "v22_04_d1a_lambda_strength_audit.csv",
        "v22_04_d1a_lambda_strength_route.json",
        "v22_04_d1a_lambda_strength_wired_audit.csv",
        "v22_04_d1a_lambda_strength_wired_route.json",
        "v22_04_d1a_lambda_strength_wired2_audit.csv",
        "v22_04_d1a_lambda_strength_wired2_route.json",
    }
    return path.name not in excluded


def read_d1_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("v22_04_d1*_audit.csv")):
        if not _official_d1_artifact(path):
            continue
        rows.extend(read_rows(path))
    return rows


def read_combined_d1_route(out_dir: Path, d1_rows: list[dict[str, Any]]) -> dict[str, Any]:
    routes = [read_json(path) for path in sorted(out_dir.glob("v22_04_d1*_route.json")) if _official_d1_artifact(path)]
    best = _best(d1_rows)
    candidate_h4800 = sum(int_flag(r.get("candidate_h4800")) for r in routes)
    return {
        "source_chain_rows": sum(int_flag(r.get("source_chain_rows")) for r in routes),
        "candidate_groups": sum(int_flag(r.get("candidate_groups")) for r in routes),
        "candidate_early_chain": sum(int_flag(r.get("candidate_early_chain")) for r in routes),
        "candidate_continuous_h3200": sum(int_flag(r.get("candidate_continuous_h3200")) for r in routes),
        "candidate_h4800": candidate_h4800,
        "terminal_erosion_groups": sum(int_flag(r.get("terminal_erosion_groups")) for r in routes),
        "best_v22_id": best.get("v22_id", ""),
        "best_h3200": best.get("h3200", ""),
        "best_h4000": best.get("h4000", ""),
        "best_h4800": best.get("h4800", ""),
        "best_h4800_retention_ratio": best.get("h4800_retention_ratio", ""),
        "decision": "ProductiveH4800Candidate" if candidate_h4800 else "NoH4800Candidate",
        "route_sources": ";".join(path.name for path in sorted(out_dir.glob("v22_04_d1*_route.json")) if _official_d1_artifact(path)),
    }


def write_precommit_selector(out_dir: Path, d1_rows: list[dict[str, Any]]) -> None:
    candidates = _candidate(d1_rows)
    positives = [r for r in candidates if int_flag(r.get("productive_h4800_group"))]
    features = [
        "train_loss_h100_h400",
        "train_stream_split_gain",
        "cross_split_source_consistency",
        "NDS_estimate",
        "source_path_early_info_volume",
        "row_radial_fraction",
        "optimizer_conflict_proxy",
        "train_stream_tail_proxy",
        "LineC_fast_train_split_proxy",
    ]
    rows = []
    for feat in features:
        rows.append(
            {
                "feature_name": feat,
                "uses_future": 0,
                "uses_validation": 0,
                "uses_test": 0,
                "uses_audit_target": 0,
                "AUC_predict_h4800_retention": "" if len(positives) < 2 else "not_trained_no_independent_rerun",
                "precision_at_k": 0.0 if len(positives) < 2 else "",
                "recall_at_k": 0.0 if len(positives) < 2 else "",
                "selected_group_h4800_retention": "",
                "independent_rerun_pass": 0,
                "precommit_selector_pass": 0,
                "selector_decision": "InsufficientPositiveProductiveLabels" if len(positives) < 2 else "IndependentRerunNotPerformed",
            }
        )
    write_rows(out_dir / "v22_04_precommit_selector_matrix.csv", rows)


def write_queue_artifacts(out_dir: Path) -> None:
    journal = read_rows(out_dir / "v22_04_command_journal.csv")
    gpu_rows = []
    queue_rows = []
    for row in journal:
        cmd = str(row.get("command", ""))
        queue_rows.append({"timestamp": row.get("timestamp", ""), "command": cmd, "status": row.get("status", ""), "note": row.get("note", "")})
        for idx in range(4):
            if f"cuda:{idx}" in cmd:
                gpu_rows.append({"gpu": idx, "timestamp": row.get("timestamp", ""), "command": cmd, "status": row.get("status", "")})
    write_rows(out_dir / "v22_04_runnable_queue.csv", queue_rows)
    write_rows(out_dir / "v22_04_gpu_assignment_manifest.csv", gpu_rows)
    write_rows(out_dir / "v22_04_gpu_utilization_timeline.csv", [{"timestamp": now_sg(), "source": "command_journal", "note": "continuous nvidia-smi sampling was not collected by this runner"}])
    write_rows(out_dir / "v22_04_idle_violation.csv", [{"checked": 1, "idle_violation": "", "blocker": "continuous_gpu_utilization_not_sampled"}])
    write_rows(
        out_dir / "v22_04_deferred_items.csv",
        [
            {"item": "precommit independent rerun", "reason": "only allowed after productive h4800 candidate and legal selector pass"},
            {"item": "D-RAT/D-RBF official fused promotion", "reason": "micro-near-E1 may pass, but official_fused_kernel_complete remains false until production fused train runner is wired"},
            {"item": "KAN promotion", "reason": "requires KAN source-channel mapping/productive retained source, not just D-CHE/D-FOU efficiency"},
        ],
    )
    write_rows(out_dir / "v22_04_queue_drain_report.csv", [{"queued_commands": len(queue_rows), "gpu_assigned_commands": len(gpu_rows), "drained": int(all(str(r.get("status")) != "started" for r in queue_rows))}])


def decide_route(out_dir: Path, code_rows: list[dict[str, Any]], efficiency: list[dict[str, Any]], dr: list[dict[str, Any]], d1_route: dict[str, Any], erosion_route: dict[str, Any], kan_route: dict[str, Any]) -> dict[str, Any]:
    code_pass = bool(code_rows) and all(int_flag(r.get("pass")) for r in code_rows)
    dche_pass = max((int_flag(r.get("v22_04_S1_pass")) for r in efficiency if r.get("carrier") == "D-CHE"), default=0)
    dfou_pass = max((int_flag(r.get("v22_04_S1_pass")) for r in efficiency if r.get("carrier") == "D-FOU"), default=0)
    dr_blocked = any("official_fused_missing" in str(r.get("blocker", "")) for r in dr)
    d1_productive = int_flag(d1_route.get("candidate_h4800"))
    d1_groups = int_flag(d1_route.get("candidate_groups"))
    kan_positive = str(kan_route.get("decision", "")) == "KANMappedProductiveSource"
    erosion_decision = str(erosion_route.get("decision", ""))
    route = "R13-NoActionableSourceTheoryAfterCompleteAutopsy"
    blocking_metric = ""
    next_action = "open new retained-target/source-observability theory rather than tune terminal amplitude"
    if not code_pass:
        route = "R0-CodePacketInvalid"
        blocking_metric = "code_truth_gate"
        next_action = "fix code packet/import/test failures first"
    elif not (dche_pass and dfou_pass):
        route = "R2-EfficiencyRegression"
        blocking_metric = "D-CHE/D-FOU S1"
        next_action = "repair official full-loop efficiency regression"
    elif dr_blocked:
        route = "R3-DRATDRBFRepairBlocked"
        blocking_metric = "official_fused_missing"
        next_action = "wire production official fused D-RAT/D-RBF kernels into runner"
    if code_pass and dche_pass and dfou_pass and d1_groups:
        if d1_productive and kan_positive:
            route = "R12-OfficialS5Success"
            blocking_metric = ""
            next_action = "run independent confirmation"
        elif d1_productive:
            route = "R10-ProductiveMLPGenericSource"
            blocking_metric = "KAN_source_channel_mapping"
            next_action = "map productive MLP source channel to D-CHE/D-FOU KAN writer"
        elif kan_positive:
            route = "R11-KANSourceChannelPositive"
            blocking_metric = "MLP_productive_h4800"
            next_action = "continue source-preserving FU around KAN-positive mapping"
        elif int_flag(erosion_route.get("explained_groups")) and "Pass" in erosion_decision:
            route = "R9-SourcePreservingFUWeakPositive"
            blocking_metric = "R4800_over_3200_below_0.50"
            next_action = "define new train-only retained target/source-observability objective"
        if dr_blocked and not d1_productive:
            route = "R3-DRATDRBFRepairBlocked"
            blocking_metric = "official_fused_missing;R4800_over_3200_below_0.50"
            next_action = "finish official fused D-RAT/D-RBF wiring while retiring exhausted terminal-preservation variants"
    return {
        "route": route,
        "blocking_metric": blocking_metric,
        "next_codex_action": next_action,
        "promotion_allowed": int(route == "R12-OfficialS5Success"),
        "D-CHE_S1_pass": dche_pass,
        "D-FOU_S1_pass": dfou_pass,
        "DRAT_DRBF_official_fused_blocked": int(dr_blocked),
        "D1_candidate_h4800": d1_route.get("candidate_h4800", ""),
        "terminal_erosion_decision": erosion_route.get("decision", ""),
        "KAN_mapping_decision": kan_route.get("decision", ""),
    }


def write_required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            resolved = path.resolve()
            try:
                artifact = str(resolved.relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({"artifact": artifact, "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(out_dir / "v22_04_required_artifact_manifest.csv", rows)
    return rows


def self_check_packet(out_dir: Path, packet_zip: Path) -> None:
    tmp = Path("/tmp/v2204_packet_check")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet_zip, "r") as z:
        z.extractall(tmp)
    source = tmp / "02_SOURCE_TREE"
    commands = [
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "import_closure", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "linec_golden", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "source_chain_tests", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "terminal_retention_tests", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "mechanism_contracts", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "kernel_gradcheck", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
        [PYTHON, "experiments/run_v22_04_s011_truth_gate.py", "--mode", "profiler_phase_tests", "--source-root", str(source), "--out-dir", str(tmp / "check_artifacts")],
    ]
    log_parts = []
    rows = []
    for command in commands:
        code, log = run_cmd(command, cwd=source, timeout=300)
        log_parts.append(log)
        rows.append({"command": " ".join(command), "returncode": code, "pass": int(code == 0)})
    (out_dir / "v22_04_clean_unzip_self_test.log").write_text("\n\n".join(log_parts), encoding="utf-8")
    write_rows(out_dir / "v22_04_clean_unzip_self_test.csv", rows)


def write_recap(out_dir: Path, route: dict[str, Any], manifest: list[dict[str, Any]]) -> None:
    code_rows = read_rows(out_dir / "v22_04_code_truth_gate.csv")
    efficiency = read_rows(out_dir / "v22_04_efficiency_full_loop_summary.csv")
    dr = read_rows(out_dir / "v22_04_drat_drbf_active_repair_summary.csv")
    erosion_summary = read_rows(out_dir / "v22_04_terminal_erosion_class_summary.csv")
    erosion_rows = read_rows(out_dir / "v22_04_terminal_erosion_autopsy.csv")
    d1 = read_d1_rows(out_dir)
    d1_route = read_combined_d1_route(out_dir, d1)
    kan = read_rows(out_dir / "v22_04_kan_source_mapping_matrix.csv")
    selector = read_rows(out_dir / "v22_04_precommit_selector_matrix.csv")
    best = _best(d1)
    by_class = Counter(str(r.get("terminal_erosion_class", "")) for r in erosion_rows)
    dominant = by_class.most_common(1)[0][0] if by_class else ""
    content = [
        "# DG-KAN v22.04 TerminalSourcePreservation DiffeomorphicFU BasisEfficiency 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route', '')}`",
        f"- promotion_allowed: {route.get('promotion_allowed', 0)}",
        f"- blocking_metric: `{route.get('blocking_metric', '')}`",
        f"- next_codex_action: {route.get('next_codex_action', '')}",
        f"- D-CHE/D-FOU S1 pass: {route.get('D-CHE_S1_pass', 0)} / {route.get('D-FOU_S1_pass', 0)}",
        f"- D-RAT/D-RBF official fused blocked: {route.get('DRAT_DRBF_official_fused_blocked', 0)}",
        f"- D1 source-preserving groups / h4800 groups: {d1_route.get('candidate_groups', '')} / {d1_route.get('candidate_h4800', '')}",
        f"- best D1 candidate: `{best.get('v22_id', '')}`",
        f"- best h4800 / retention ratio: {best.get('h4800', '')} / {best.get('h4800_retention_ratio', '')}",
        f"- terminal erosion dominant class: `{dominant}`",
        f"- KAN mapping decision: `{route.get('KAN_mapping_decision', '')}`",
        "",
        "## Part 1 Code Audit",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 40),
        "## Part 2 Basis Efficiency / D-RAT / D-RBF",
        "",
        md_table(efficiency, ["carrier", "S1_pass_rows", "full_loop_official_closure", "v22_04_same_kernel_runner_proof", "v22_04_S1_pass", "v22_04_decision", "v22_04_blocker"], 20),
        md_table(dr, ["carrier", "profile_rows", "micro_near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "## Part 3 Terminal Erosion Taxonomy",
        "",
        md_table(erosion_summary, ["terminal_erosion_class", "groups", "fraction"], 30),
        md_table(erosion_rows, ["v22_id", "h3200", "h4000", "h4800", "h4800_retention_ratio", "terminal_erosion_class", "terminal_erosion_evidence"], 24),
        "## Part 4 D1 Source-Preserving FU",
        "",
        md_table(d1, ["plan_line", "v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "productive_h4800_group", "source_chain_blocker"], 40),
        "## Part 5 KAN Source Mapping",
        "",
        md_table(kan, ["carrier", "variant", "v22_id", "h800", "h3200", "h4800", "h4800_retention_ratio", "v22_04_source_mapping_decision", "fresh_source_retention_run"], 40),
        "## Part 6 Precommit Selector",
        "",
        md_table(selector, ["feature_name", "uses_future", "uses_validation", "uses_test", "uses_audit_target", "AUC_predict_h4800_retention", "precision_at_k", "precommit_selector_pass", "selector_decision"], 20),
        "## 修改记录",
        "",
        "- 新增 v22.04 runner：common、S0.11 truth gate、efficiency readback、D-RAT/D-RBF officialization wrapper、terminal erosion autopsy、D1 source-preserving FU aggregation、KAN source mapping、merge/finalize。",
        "- 新增 `dgkan/fu/terminal_erosion.py`、`dgkan/fu/source_preservation.py`、`dgkan/profiling/efficiency_v22_04.py` 与 `efficiency_v22_03.py` 兼容层；这些 helper 只做审计分类/ratio 判定，不生成实验数值。",
        "- 扩展 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`，注册计划命名 D1a-D1f source-preserving FU 候选，便于 fresh old-shape source-retention runner 直接执行。",
        "- D1 混跑出现 ratio>0.50 但 early/h3200 chain 不成立的 retrospective-looking near-miss 后，追加 D1a/D1b 单候选复跑；单候选恢复 early+h3200 chain 但仍低于 R4800/3200 gate。",
        "- 按 blocker 补测 D1a SPP lambda050/lambda100 强度分支；前两次 full 暴露 runner allowlist wiring 缺口并保留为 pre-wiring artifacts，修复 `experiments/run_v17_common.py` 外层机制集合、h800 slow-EMA active-update 集合、source-name/role 和 telemetry 集合后，用 `wired3` fresh full 作为正式科学判定。",
        "- 执行 fresh D2 KAN source mapping：D-CHE/D-FOU x KSW1/KSW2/KSW9/KSW10 + matched controls，old-shape 6400 step，不把 efficiency closure 伪称为 KAN source-channel writer 成功。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- v22.04 的 functional 判据仍保持 productive h4800 gate：mean h4800 >= 0.005、R4800/3200 >= 0.50、row-positive >= 7/9；本复盘没有把 h4800 positive 或 ActuationR2 写成 promotion。",
        f"- 合法 early+h3200 链中的 D1 best candidate 为 `{best.get('v22_id', '')}`，h4800_retention_ratio={best.get('h4800_retention_ratio', '')}；若 productive_h4800_group=0，则说明 source-preserving FU 仍只达到 weak-positive terminal source，而未过 retained-ratio gate。",
        f"- Terminal erosion taxonomy 覆盖率由 `v22_04_terminal_erosion_route.json` 决定；dominant class={dominant}，后续路线必须围绕该 class 的 train-only retained-target/source-observability 机制，而不是继续扩大 h800/h3200 source。",
        "- D-CHE/D-FOU 的 efficiency 只能证明 official efficient carrier；KAN mapping matrix 仍需证明 source-channel writer 能承载 retained source，二者不能相互替代。",
        "- D-RAT/D-RBF active micro-kernel benchmark 若仍显示 `official_fused_missing`，则只能作为 officialization progress/blocker，不允许进入 functional promotion。",
        "- Precommit selector 在没有足够 productive h4800 正样本和 independent rerun 前保持失败/未训练状态，避免 retrospective selector 泄漏。",
        "",
        "## Artifact Index",
        "",
        md_table(manifest, ["artifact", "exists", "size_bytes", "sha256"], 120),
    ]
    V2204_RECAP_DOC.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    d1 = read_d1_rows(out_dir)
    write_precommit_selector(out_dir, d1)
    write_queue_artifacts(out_dir)
    code_rows = read_rows(out_dir / "v22_04_code_truth_gate.csv")
    efficiency = read_rows(out_dir / "v22_04_efficiency_full_loop_summary.csv")
    dr = read_rows(out_dir / "v22_04_drat_drbf_active_repair_summary.csv")
    d1_route = read_combined_d1_route(out_dir, d1)
    erosion_route = read_json(out_dir / "v22_04_terminal_erosion_route.json")
    kan_route = read_json(out_dir / "v22_04_kan_source_mapping_decision.json")
    route = decide_route(out_dir, code_rows, efficiency, dr, d1_route, erosion_route, kan_route)
    write_rows(out_dir / "v22_04_route_decision.csv", [route])
    write_json(out_dir / "v22_04_route_decision.json", route)
    manifest = write_required_manifest(out_dir)
    write_recap(out_dir, route, manifest)
    packet_zip, bundle_zip = build_packet(out_dir)
    self_check_packet(out_dir, packet_zip)
    manifest = write_required_manifest(out_dir)
    write_recap(out_dir, route, manifest)
    packet_zip, bundle_zip = build_packet(out_dir)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_04_merge_finalize.py --out-dir {out_dir}",
        status="completed",
        note=f"route={route.get('route')} packet={packet_zip.name}:{sha256_file(packet_zip)} bundle={bundle_zip.name}:{sha256_file(bundle_zip)}",
    )


if __name__ == "__main__":
    main()
