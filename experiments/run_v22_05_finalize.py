#!/usr/bin/env python3
"""v22.05 final route, recap, execution log, and packet builder."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_05_common import (  # noqa: E402
    PYTHON,
    V2205_RECAP_DOC,
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
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-clean-self-test", type=int, default=1)
    return p


def _route(code_rows: list[dict[str, Any]], eff: dict[str, Any], dr: dict[str, Any], metric: dict[str, Any], kan: dict[str, Any], idle: list[dict[str, Any]]) -> dict[str, Any]:
    code_pass = bool(code_rows) and all(int_flag(r.get("pass")) for r in code_rows)
    dche = int_flag(eff.get("D-CHE_S1_pass"))
    dfou = int_flag(eff.get("D-FOU_S1_pass"))
    dr_blocked = any(str(v.get("blocker", "")).find("official_fused_missing") >= 0 for v in dr.values() if isinstance(v, dict))
    metric_route = str(metric.get("functional_route", ""))
    kan_decision = str(kan.get("decision", ""))
    idle_violation = any(int_flag(r.get("idle_violation")) for r in idle)
    route = "R0-CodeMetricInvalid"
    blocking = []
    next_action = "fix S0.12 before science rows"
    if code_pass:
        if not (dche and dfou):
            route = "R2-EfficiencyReconfirmBlocked"
            blocking.append("D-CHE_or_D-FOU_S1")
            next_action = "repair D-CHE/D-FOU full-loop efficiency before metric FU promotion"
        elif metric_route == "F2-MLPProductiveTerminalSource":
            route = "F2-MLPProductiveTerminalSource"
            next_action = "run independent rerun and KAN same-metric mapping"
        elif metric_route == "F1-MetricRetentionImproved":
            route = "F1-MetricRetentionImproved"
            next_action = "extend the improving metric to independent rerun and h6400"
        elif kan_decision in {"KANSourceChannelOpened", "KANProductiveTerminalSource"}:
            route = "F3-KANSourceChannelOpened" if kan_decision == "KANSourceChannelOpened" else "F4-KANProductiveTerminalSource"
            next_action = "extend KAN mapping to 3x3 and same-metric MLP controls"
        elif dr_blocked:
            route = "R3-DRATDRBFRepairBlocked"
            blocking.append("official_fused_missing")
            next_action = "finish production official fused D-RAT/D-RBF train runner wiring"
        else:
            route = metric_route or "F0-MetricNoEffect"
            next_action = "run terminal erosion autopsy and propose one new metric family"
    else:
        blocking.append("S0.12")
    if idle_violation:
        blocking.append("gpu_idle_contract")
    return {
        "route": route,
        "promotion_allowed": int(route in {"F4-KANProductiveTerminalSource"} and not blocking),
        "blocking_metric": ";".join(blocking),
        "next_codex_action": next_action,
        "S0_pass": int(code_pass),
        "D-CHE_S1_pass": dche,
        "D-FOU_S1_pass": dfou,
        "DRAT_DRBF_official_fused_blocked": int(dr_blocked),
        "functional_route": metric_route,
        "KAN_mapping_decision": kan_decision,
        "idle_violation": int(idle_violation),
    }


def _artifact_index(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            resolved = path.resolve()
            try:
                artifact = str(resolved.relative_to(ROOT))
            except ValueError:
                artifact = str(path)
            rows.append({"artifact": artifact, "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _run_clean_self_test(out_dir: Path) -> tuple[int, str]:
    zip_path, _bundle = build_packet(out_dir)
    clean_root = out_dir / "clean_unzip_self_test"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(clean_root)
    source = clean_root / "02_SOURCE_TREE"
    command = [
        PYTHON,
        "experiments/run_v22_05_s012_truth_gate.py",
        "--mode",
        "all",
        "--source-root",
        str(source),
        "--self-contained-import-check",
        "1",
        "--out-dir",
        str(out_dir),
    ]
    code, log = run_cmd(command, cwd=source, timeout=900)
    (out_dir / "v22_05_clean_unzip_self_test.log").write_text(log, encoding="utf-8")
    return code, str(source)


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir).resolve()
    if int(args.run_clean_self_test):
        code, source = _run_clean_self_test(out_dir)
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_05_s012_truth_gate.py --mode all --source-root {source} --self-contained-import-check 1 --out-dir {out_dir}",
            status="completed" if code == 0 else "blocked",
            note=f"clean_unzip_returncode={code}",
        )

    code_rows = read_rows(out_dir / "v22_05_code_truth_gate.csv")
    eff_route = read_json(out_dir / "v22_05_efficiency_route.json")
    dr_route = read_json(out_dir / "v22_05_drat_drbf_repair_route.json")
    metric_route = read_json(out_dir / "v22_05_metric_first_route.json")
    kan_route = read_json(out_dir / "v22_05_kan_source_mapping_route.json")
    idle_rows = read_rows(out_dir / "v22_05_idle_violation.csv")
    route = _route(code_rows, eff_route, dr_route, metric_route, kan_route, idle_rows)
    write_json(out_dir / "v22_05_final_route.json", route)
    write_rows(out_dir / "v22_05_final_route.csv", [route])

    metric_rows = read_rows(out_dir / "v22_05_metric_first_mlp_summary.csv")
    autopsy_rows = read_rows(out_dir / "v22_05_metric_autopsy_summary.csv")
    correlation_rows = read_rows(out_dir / "v22_05_metric_energy_retention_correlations.csv")
    verdict_rows = read_rows(out_dir / "v22_05_metric_failure_verdict.csv")
    eff_rows = read_rows(out_dir / "v22_05_efficiency_full_loop_reconfirm.csv")
    dr_rows = read_rows(out_dir / "v22_05_drat_drbf_repair_summary.csv")
    kan_rows = read_rows(out_dir / "v22_05_kan_source_mapping_matrix.csv")
    alias_rows = read_rows(out_dir / "v22_05_semantic_noncollapse_summary.csv")
    schema_rows = read_rows(out_dir / "v22_05_mechanism_contracts.csv")
    artifact_rows = _artifact_index(out_dir)
    write_rows(out_dir / "v22_05_artifact_index.csv", artifact_rows)

    best_metric = sorted(metric_rows, key=lambda r: finite_float(r.get("R4800_over_3200"), -999.0), reverse=True)[0] if metric_rows else {}
    simple_svg(out_dir / "figures/code_truth_dashboard.svg", "v22.05 code truth dashboard", code_rows, "pass")
    simple_svg(out_dir / "figures/mechanism_semantic_alias_heatmap.svg", "v22.05 semantic alias heatmap", read_rows(out_dir / "v22_05_semantic_alias_matrix.csv"), "semantic_alias")
    simple_svg(out_dir / "figures/controls_attribution_heatmap.svg", "v22.05 controls attribution", metric_rows, "control_equivalent_fraction")
    simple_svg(out_dir / "figures/optimizer_projection_on_source_trace.svg", "v22.05 optimizer projection proxy", metric_rows, "metric_projection_cosine")
    simple_svg(out_dir / "figures/debt_transition_h3200_to_h4800.svg", "v22.05 debt transition", metric_rows, "LineC_debt_final")
    simple_svg(out_dir / "figures/MLP_source_decomposition_hidden_readout.svg", "v22.05 MLP source decomposition", metric_rows, "source_h4800_mean")

    recap = [
        "# DG-KAN v22.05 MetricFirst FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- promotion_allowed: {route.get('promotion_allowed')}",
        f"- blocking_metric: `{route.get('blocking_metric')}`",
        f"- next_codex_action: {route.get('next_codex_action')}",
        f"- S0 pass: {route.get('S0_pass')}",
        f"- D-CHE/D-FOU S1 pass: {route.get('D-CHE_S1_pass')} / {route.get('D-FOU_S1_pass')}",
        f"- D-RAT/D-RBF official fused blocked: {route.get('DRAT_DRBF_official_fused_blocked')}",
        f"- functional_route: `{route.get('functional_route')}`",
        f"- KAN mapping decision: `{route.get('KAN_mapping_decision')}`",
        "",
        "## Part A S0.12 Code / Metric / Mechanism Gate",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 30),
        "",
        "### Mechanism Schema And Semantic Alias",
        "",
        md_table(alias_rows, ["pairs", "semantic_alias_pairs", "undeclared_alias_pairs", "pass", "gradient_norm"], 10),
        "",
        md_table(schema_rows, ["mechanism", "mechanism_family", "source_pattern", "uses_optimizer_primary", "uses_function_space_metric", "uses_sobolev_metric", "uses_rkhs_metric", "uses_fisher_metric", "semantic_noncollapse_group", "semantic_alias_representative"], 24),
        "",
        "## Part B Basis Efficiency / D-RAT / D-RBF",
        "",
        md_table(eff_rows, ["carrier", "forward_ratio_vs_mlp", "step_ratio_vs_mlp", "memory_ratio_vs_mlp", "functional_overhead_ratio", "v22_05_S1_pass", "v22_05_decision", "v22_05_blocker"], 20),
        "",
        md_table(dr_rows, ["carrier", "profile_rows", "micro_near_E1_rows", "E1_official_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "",
        "## Part C Metric-First MLP FU",
        "",
        f"- best_metric_v21_id: `{best_metric.get('v21_id', '')}`",
        f"- best_metric_name: `{best_metric.get('metric_name', '')}`",
        f"- best R4800/3200: {best_metric.get('R4800_over_3200', '')}",
        f"- best h4800: {best_metric.get('source_h4800_mean', '')}",
        f"- best row_h4800_positive_count: {best_metric.get('row_h4800_positive_count', '')}",
        "",
        md_table(metric_rows, ["v21_id", "metric_name", "rows", "source_h800_mean", "source_h3200_mean", "source_h4800_mean", "R4800_over_3200", "row_h4800_positive_count", "R4800_improvement_vs_v2204_best", "H_C1_exploration_S2_pass", "failure_taxonomy"], 30),
        "",
        "### Metric Failure Autopsy",
        "",
        md_table(verdict_rows, ["metric_families_tried", "metric_rows", "source_formation_failed_rows", "terminal_retention_evaluable_rows", "control_equivalent_rows", "productive_h4800_rows", "best_attempt_label", "best_v21_id", "best_source_h100_mean", "best_source_h3200_mean", "best_source_h4800_mean", "dominant_failure_mode", "strongest_logged_energy_correlation_to_h4800", "strongest_logged_energy_correlation_pearson"], 5),
        "",
        md_table(autopsy_rows, ["attempt_label", "v21_id", "source_h100_mean", "source_h800_mean", "source_h3200_mean", "source_h4800_mean", "source_h6400_mean", "R4800_over_3200", "early_source_chain_open", "terminal_retention_evaluable", "control_equivalent_fraction", "metric_energy_mean", "NDS", "LineC_debt_final", "hidden_source_fraction", "readout_source_fraction", "autopsy_failure_mode"], 40),
        "",
        md_table(correlation_rows, ["attempt_label", "x_metric", "y_metric", "n", "pearson", "spearman"], 40),
        "",
        "## Part D KAN Source Mapping",
        "",
        md_table(kan_rows, ["carrier", "variant", "v22_id", "h800", "h3200", "h4800", "h4800_retention_ratio", "KAN_h100_h400_h800_h3200_chain_open", "KAN_productive_terminal_source", "v22_05_source_mapping_decision"], 40),
        "",
        "## 修改记录",
        "",
        "- 新增 v22.05 runner：common、S0.12 truth gate、efficiency reconfirm、D-RAT/D-RBF repair wrapper、metric-first FU summarizer、metric failure autopsy、KAN mapping readback、queue sampler、finalize。",
        "- 新增 `dgkan/fu/function_space_metrics.py`、`sobolev_metric.py`、`rkhs_metric.py`、`fisher_metric.py`、`metric_projection.py`；这些 helper 只用 train-batch/current-gradient 生成 metric-projected update 和 audit energy，不读取 validation/test/future。",
        "- 扩展 `dgkan/fu/mechanisms.py`：新增 M220-M227 metric-first mechanisms；在 metric-only early-source failure 后追加 M228-M235 source-memory repair mechanisms；修正 contract schema，把 `source_pattern` 与 bool columns 分离，并给 v22.04 terminal momentum-like aliases 标注 `semantic_noncollapse_group`。",
        "- 扩展 `experiments/run_v21_01_source_retention.py` 与 `experiments/run_v17_common.py`：注册 MLP-MF-G0..G8 与 MLP-MFSM-G0..G8 specs，并把 metric diagnostics 写入 horizon trace/enriched matrix，方便审计。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- S0.12 以 clean unzip self-contained import 为 hard gate；若 `self_contained_import_check=0` 或 import/compile/schema/alias 任一失败，不允许继续把 science rows 写成 promotion。",
        "- v22.05 的 metric-first mechanisms 是 parameter-pullback proxy，而不是完整显式 Jacobian solver；因此复盘只把它们作为 metric-first 实验推进，不把它们伪称为精确 $J^T G_f J$ 求逆。",
        "- D-CHE/D-FOU efficiency route 仍只证明 carrier efficiency；KAN source-channel 是否打开必须看 KAN source chain rows，不能用 efficiency 替代 functional evidence。",
        "- D-RAT/D-RBF 若仍为 `official_fused_missing`，只能写 active repair progress/blocker，不能进入 functional promotion。",
        "- Metric-first FU 若未过 R4800/3200 gate，failure taxonomy 以落盘 `NDS`、Sobolev/RKHS energy、control-equivalent 和 h4800 source 读数裁决，避免继续扩大旧 terminal-preservation 名字池。",
        "- 本轮若 autopsy 显示 `terminal_retention_evaluable_rows=0`，则失败点不是 h3200 后保留，而是 metric update 没能形成 early/h3200 source；后续应先做 train-only loss-cotangent/source-channel metric target，再谈 terminal projection。",
        "",
        "## Artifact Index",
        "",
        md_table(artifact_rows, ["artifact", "exists", "size_bytes", "sha256"], 160),
    ]
    V2205_RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")
    packet, bundle = build_packet(out_dir)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_finalize.py --out-dir {out_dir} --run-clean-self-test {int(args.run_clean_self_test)}",
        status="completed",
        note=f"route={route.get('route')} packet={packet.name} bundle={bundle.name}",
    )


if __name__ == "__main__":
    main()
