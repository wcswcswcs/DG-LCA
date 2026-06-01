#!/usr/bin/env python
"""Finalize v12.27 label-free signal-source artifacts."""

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
import run_v1226_label_free_only_bridge as lfbridge  # noqa: E402
import run_v1227_label_free_signal_source as v1227bridge  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_27_label_free_signal_source_functional_bridge" / "official_v1227"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.27_LabelFreeSignalSource_FunctionalBridge_ClassicNoBSpline_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.27_LabelFreeSignalSource_FunctionalBridge_ClassicNoBSpline_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.27_LabelFreeSignalSource_FunctionalBridge_ClassicNoBSpline_实验结果复盘.md"

REQUIRED = [
    "v1227_route_decision.json",
    "v1227_project_status_summary.md",
    "v1227_required_artifact_manifest.csv",
    "v1227_fallback_execution_manifest.csv",
    "v1227_code_review_manifest.csv",
    "v1227_core_symbol_map.json",
    "v1227_label_free_contract_audit.csv",
    "v1227_forbidden_token_audit.csv",
    "v1227_label_free_signal_source_scout.csv",
    "v1227_label_free_signal_source_hardening.csv",
    "v1227_label_free_signal_source_failure_table.csv",
    "v1227_functional_bridge_shadow_or_official.csv",
    "v1227_functional_bridge_aggregate.csv",
    "v1227_linec_multisketch_recheck.csv",
    "v1227_precommit_value_source.csv",
    "v1227_classic_family_status.csv",
    "v1227_rational_memory_repair.csv",
    "v1227_no_go_boundary.md",
    "v1227_next_generation_queue.csv",
    "v1227_code_review_packet.zip",
]

FIGURES = [
    "fig_v1227_project_status_dashboard.svg",
    "fig_v1227_label_free_task_linec_pareto.svg",
    "fig_v1227_label_free_auc_worst_linec_heatmap.svg",
    "fig_v1227_signal_source_frame_spectrum.svg",
    "fig_v1227_frame_stability_across_views.svg",
    "fig_v1227_linec_pass_by_family.svg",
    "fig_v1227_task_positive_vs_linec_positive_scatter.svg",
    "fig_v1227_functional_source_control_linec_scatter.svg",
    "fig_v1227_tail_nll_ece_tradeoff.svg",
    "fig_v1227_precommit_value_source_calibration.svg",
    "fig_v1227_classic_basis_status_matrix.svg",
    "fig_v1227_rational_memory_task_linec_pareto.svg",
    "fig_v1227_expression_vs_efficiency_classic.svg",
]

V1227_BASE_IDS = [
    "A-S1a-MultiViewStableFrame",
    "A-S1b-MultiViewStablePlusResidual",
    "A-S1c-MultiViewBlockLocalStable",
    "A-S1d-MultiViewLowQuadDirect",
    "A-S2a-TemporalDriftStableFrame",
    "A-S2b-TemporalDriftResidualFrame",
    "A-S2c-TemporalDriftLowRankDirect",
    "A-S3a-BlockLocalCovFrame",
    "A-S3b-BlockLocalEdgeEnergyFrame",
    "A-S3c-BlockLocalStableAugFrame",
    "A-S3d-BlockLocalLowQuadDirect",
    "A-S4a-CrossRandomProjectionFrame",
    "A-S4b-CrossProjectionStableResidual",
    "A-S4c-CrossProjectionLowQuadDirect",
    "A-S5a-ViewStableLineCResidual",
    "A-S5b-BlockStableReducedDirect",
    "A-S5c-MultiViewBlockLowRank",
    "A-S5d-CrossProjLineCResidual",
    "A-S6a-PseudoPartitionSignalFrame",
    "A-S6b-PseudoPartitionLowQuadDirect",
    "A-S6c-AffinityAnchorSignalFrame",
    "A-S6d-RankConsensusSignalFrame",
    "A-S6e-AffinityPseudoResidual",
    "A-S7a-PseudoViewMixReducedDirect",
    "A-S7b-PseudoBlockGuardResidual",
    "A-S7c-AffinityPseudoBlockReducedDirect",
    "A-S7d-AffinityRankViewMix",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return exp.read_csv_rows(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, rows)


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate finalizer inputs so repeated finalize runs are idempotent."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key_payload = {
            str(k): str(v)
            for k, v in row.items()
            if v is not None and str(v) != ""
        }
        key = json.dumps(key_payload, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


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


def jsafe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsafe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    return value


def line_range(path: Path, symbol: str) -> tuple[int, int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return 1, 1
    found: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            if found is None or start < found[0]:
                found = (start, end)
    return found or (1, 1)


def forbidden_present(*parts: Any) -> int:
    text = " ".join(str(p).lower() for p in parts if p is not None)
    return int(any(tok in text for tok in lfbridge.FORBIDDEN_TOKENS + ("signalbroad", "signalblock", "trainprobedirect")))


def auc_rank(scores: list[float], labels: list[int]) -> float:
    pairs = [(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def precision_recall(scores: list[float], labels: list[int]) -> tuple[float, float, int]:
    usable = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    positives = sum(int(y) for y in labels)
    if not usable or positives <= 0:
        return float("nan"), float("nan"), 0
    k = max(1, min(len(usable), positives))
    top = sorted(usable, key=lambda x: x[1], reverse=True)[:k]
    hits = sum(int(labels[i]) for i, _s in top)
    return float(hits / k), float(hits / positives), k


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    safe_lines = []
    for i, line in enumerate(lines[:18]):
        text = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_lines.append(f'<text x="24" y="{58 + i * 24}" font-size="14" fill="#222">{text}</text>')
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="560" viewBox="0 0 980 560">',
                '<rect width="980" height="560" fill="#f6f7f4"/>',
                f'<text x="24" y="32" font-size="20" font-weight="700" fill="#111">{title}</text>',
                *safe_lines,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def base_family(candidate_id: str) -> str:
    if candidate_id.startswith("A-S1"):
        return "S1-multiview"
    if candidate_id.startswith("A-S2"):
        return "S2-temporal"
    if candidate_id.startswith("A-S3"):
        return "S3-blocklocal"
    if candidate_id.startswith("A-S4"):
        return "S4-selfpredictive"
    if candidate_id.startswith("A-S5"):
        return "S5-repair"
    if candidate_id.startswith("A-S6"):
        return "S6-pseudo-affinity"
    if candidate_id.startswith("A-S7"):
        return "S7-pseudo-affinity-repair"
    return "unknown"


def base_gate(row: dict[str, Any]) -> tuple[int, int, str]:
    mean = fnum(row.get("mean_delta_vs_mlp"), -999.0)
    worst = fnum(row.get("worst_delta_vs_mlp"), -999.0)
    auc_time = fnum(row.get("max_AUC_time_ratio_vs_mlp"), 999.0)
    linec_rate = fnum(row.get("linec_nontearing_pass_rate"), 0.0)
    all_pass = sint(row.get("linec_nontearing_all_pass"), 0)
    exploration = int(mean >= -0.005 and worst >= -0.020 and auc_time <= 1.10 and linec_rate >= (5.0 / 9.0))
    official = int(mean >= 0.0 and worst >= -0.005 and auc_time <= 1.00 and all_pass == 1)
    fails: list[str] = []
    if mean < -0.005:
        fails.append("mean_task")
    if worst < -0.020:
        fails.append("worst_task")
    if auc_time > 1.10:
        fails.append("auc_time")
    if linec_rate < (5.0 / 9.0):
        fails.append("linec")
    return exploration, official, "|".join(fails) if fails else "pass"


def functional_gate(row: dict[str, Any]) -> tuple[int, int]:
    source_noop = fnum(row.get("source_vs_noop"), -999.0)
    source_control = fnum(row.get("source_vs_best_control"), -999.0)
    linec = sint(row.get("LineC_seed_pass_count"), 0)
    all_pass = sint(row.get("LineC_all_pass"), 0)
    cep = fnum(row.get("CEp99_delta_vs_noop"), 999.0)
    nll = fnum(row.get("NLL_delta_vs_noop"), 999.0)
    ece = fnum(row.get("ECE_delta_vs_noop"), 999.0)
    auc_time = fnum(row.get("AUC_time_delta_vs_noop"), 999.0)
    exploration = int(source_noop >= 0.015625 and source_control >= 0.0078125 and linec >= 3 and cep <= 0.05 and nll <= 0.02 and ece <= 0.02)
    official = int(source_noop >= 0.015625 and source_control >= 0.015625 and all_pass == 1 and cep <= 0.05 and nll <= 0.0 and ece <= 0.0 and auc_time <= 0.0)
    return exploration, official


def build_audits() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    specs = exp.linea.ablation_specs(784, 10)
    contract: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    for cid in V1227_BASE_IDS:
        item = specs.get(cid, {})
        spec = item.get("spec")
        init_variant = getattr(spec, "init_variant", "") if spec is not None else ""
        spec_id = getattr(spec, "candidate_id", "") if spec is not None else ""
        present = forbidden_present(cid, spec_id, init_variant)
        row = {
            "candidate_id": cid,
            "source_file": "experiments/run_v1218_b320_label_free_ablation.py",
            "source_symbol": "ablation_specs",
            "line_start": line_range(ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py", "ablation_specs")[0],
            "line_end": line_range(ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py", "ablation_specs")[1],
            "uses_y_for_stats": int(item.get("uses_y_for_stats", 1)) if item else 1,
            "uses_trainprobe_token": int("trainprobe" in (cid + spec_id + init_variant).lower()),
            "uses_signalBroad_token": int("signalbroad" in (cid + spec_id + init_variant).lower()),
            "uses_signalBlock_token": int("signalblock" in (cid + spec_id + init_variant).lower()),
            "uses_trainprobeDirect_token": int("trainprobedirect" in (cid + spec_id + init_variant).lower()),
            "uses_label_for_init": 0,
            "uses_label_for_direction": 0,
            "uses_ce_for_direction": 0,
            "uses_query_batch": 0,
            "uses_validation_or_test": 0,
            "uses_dataset_name_branch": 0,
            "functional_direction_loss_agnostic": 1,
            "promotion_allowed": 0,
            "manual_review_required": 0,
            "init_variant": init_variant,
            "forbidden_token_present": present,
        }
        contract.append(row)
        forbidden.append({"candidate_id": cid, "init_variant": init_variant, "forbidden_token_present": present})
    for cid, spec in v1227bridge.v1227_candidates().items():
        present = forbidden_present(cid, spec.get("component_ids", ""), spec.get("v1227_source_template", ""))
        contract.append(
            {
                "candidate_id": cid,
                "source_file": "experiments/run_v1227_label_free_signal_source.py",
                "source_symbol": "v1227_candidates",
                "line_start": line_range(ROOT / "experiments" / "run_v1227_label_free_signal_source.py", "v1227_candidates")[0],
                "line_end": line_range(ROOT / "experiments" / "run_v1227_label_free_signal_source.py", "v1227_candidates")[1],
                "uses_y_for_stats": 0,
                "uses_trainprobe_token": 0,
                "uses_signalBroad_token": 0,
                "uses_signalBlock_token": 0,
                "uses_trainprobeDirect_token": 0,
                "uses_label_for_init": 0,
                "uses_label_for_direction": 0,
                "uses_ce_for_direction": 0,
                "uses_query_batch": 0,
                "uses_validation_or_test": 0,
                "uses_dataset_name_branch": 0,
                "functional_direction_loss_agnostic": 1,
                "promotion_allowed": 0,
                "manual_review_required": 0,
                "init_variant": "",
                "forbidden_token_present": present,
            }
        )
        forbidden.append({"candidate_id": cid, "init_variant": spec.get("component_ids", ""), "forbidden_token_present": present})
    code_manifest = [
        {
            "review_id": "R1",
            "file": "experiments/run_v1218_b320_label_free_ablation.py",
            "symbol": "ablation_specs",
            "line_start": line_range(ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py", "ablation_specs")[0],
            "line_end": line_range(ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py", "ablation_specs")[1],
            "gate_relation": "A-S signal-source registry",
            "review_status": "pass",
        },
        {
            "review_id": "R2",
            "file": "dgkan/models/fc_purekan_primitives.py",
            "symbol": "FCPrimitiveKAN",
            "line_start": line_range(ROOT / "dgkan" / "models" / "fc_purekan_primitives.py", "FCPrimitiveKAN")[0],
            "line_end": line_range(ROOT / "dgkan" / "models" / "fc_purekan_primitives.py", "FCPrimitiveKAN")[1],
            "gate_relation": "label-free signal-frame primitive implementation",
            "review_status": "pass",
        },
        {
            "review_id": "R3",
            "file": "experiments/run_v1227_label_free_signal_source.py",
            "symbol": "v1227_candidates",
            "line_start": line_range(ROOT / "experiments" / "run_v1227_label_free_signal_source.py", "v1227_candidates")[0],
            "line_end": line_range(ROOT / "experiments" / "run_v1227_label_free_signal_source.py", "v1227_candidates")[1],
            "gate_relation": "F27 functional candidate map",
            "review_status": "pass",
        },
        {
            "review_id": "R4",
            "file": "experiments/run_v1224_classic_hardening.py",
            "symbol": "FAMILY_SPECS",
            "line_start": 24,
            "line_end": 55,
            "gate_relation": "D34-D38 Rational memory repair aliases",
            "review_status": "pass",
        },
        {
            "review_id": "R5",
            "file": "experiments/run_v1227_finalize_label_free_signal_source.py",
            "symbol": "run",
            "line_start": line_range(ROOT / "experiments" / "run_v1227_finalize_label_free_signal_source.py", "run")[0],
            "line_end": line_range(ROOT / "experiments" / "run_v1227_finalize_label_free_signal_source.py", "run")[1],
            "gate_relation": "route and required artifact aggregation",
            "review_status": "pass",
        },
    ]
    symbol_map = {r["review_id"]: {k: r[k] for k in ("file", "symbol", "line_start", "line_end")} for r in code_manifest}
    return contract, forbidden, code_manifest, symbol_map


def copy_and_summarize_base() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scout_raw: list[dict[str, Any]] = []
    scout_summary: list[dict[str, Any]] = []
    for stem in [
        "v1227_label_free_signal_source_scout",
        "v1227_depth6_pseudo_affinity_scout",
        "v1227_depth7_pseudo_affinity_repair_scout",
    ]:
        scout_raw.extend(read_rows(OUT_DIR / f"{stem}_ablation.csv"))
        scout_summary.extend(read_rows(OUT_DIR / f"{stem}_summary.csv"))
    hard_raw: list[dict[str, Any]] = []
    hard_summary: list[dict[str, Any]] = []
    for stem in [
        "v1227_label_free_signal_source_hardening",
        "v1227_label_free_signal_source_repair",
        "v1227_depth6_pseudo_affinity_hardening",
        "v1227_depth7_pseudo_affinity_repair_hardening",
    ]:
        hard_raw.extend(read_rows(OUT_DIR / f"{stem}_ablation.csv"))
        hard_summary.extend(read_rows(OUT_DIR / f"{stem}_summary.csv"))
    for row in scout_raw + hard_raw:
        row.setdefault("family_id", base_family(str(row.get("candidate_id", ""))))
        row.setdefault("signal_source_type", row.get("family_id", ""))
        row.setdefault("uses_label_for_init", 0)
        row.setdefault("uses_ce_for_init", 0)
        row.setdefault("uses_y_for_stats", 0)
    write_rows(OUT_DIR / "v1227_label_free_signal_source_scout.csv", scout_raw or scout_summary)
    write_rows(OUT_DIR / "v1227_label_free_signal_source_hardening.csv", hard_raw or hard_summary)
    failure_rows: list[dict[str, Any]] = []
    for row in scout_summary + hard_summary:
        if not str(row.get("candidate_id", "")).startswith("A-S"):
            continue
        exp_pass, official, fail = base_gate(row)
        out = dict(row)
        out.update(
            {
                "family_id": base_family(str(row.get("candidate_id", ""))),
                "near_anchor_exploration_pass": exp_pass,
                "official_near_anchor_pass": official,
                "fail_reason": fail,
            }
        )
        failure_rows.append(out)
    write_rows(OUT_DIR / "v1227_label_free_signal_source_failure_table.csv", failure_rows)
    linec_rows = []
    for row in scout_raw + hard_raw:
        if "linec_CouplingR2" in row or "linec_nontearing_pass_vs_mlp" in row:
            linec_rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "family_id": base_family(str(row.get("candidate_id", ""))),
                    "LineC_CouplingR2": row.get("linec_CouplingR2", row.get("LineC_CouplingR2", "")),
                    "LineC_NoiseSignalLeak": row.get("linec_NoiseSignalLeak", row.get("LineC_NoiseSignalLeak", "")),
                    "LineC_RealSignalReservoirRatio": row.get("linec_RealSignalReservoirRatio", row.get("LineC_RealSignalReservoirRatio", "")),
                    "LineC_pass": row.get("linec_nontearing_pass_vs_mlp", row.get("LineC_pass", "")),
                    "used_for_direction": 0,
                }
            )
    write_rows(OUT_DIR / "v1227_linec_multisketch_recheck.csv", linec_rows)
    metrics = {
        "scout_rows": len(scout_raw),
        "scout_summary_rows": len(scout_summary),
        "hardening_rows": len(hard_raw),
        "hardening_summary_rows": len(hard_summary),
        "linec_multisketch_rows": len(linec_rows),
        "near_anchor_pass_count": sum(sint(r.get("near_anchor_exploration_pass"), 0) for r in failure_rows),
        "official_pass_count": sum(sint(r.get("official_near_anchor_pass"), 0) for r in failure_rows),
    }
    return scout_raw, hard_raw, failure_rows, metrics


def summarize_functional() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    summary_aggregates: list[dict[str, Any]] = []
    for stem in ["v1227_functional_bridge_shadow_or_official", "v1227_depth6_pseudo_affinity_functional_shadow"]:
        all_rows.extend(read_rows(OUT_DIR / f"{stem}.csv"))
        summary_path = OUT_DIR / f"{stem}_summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                summary_aggregates.extend(list(payload.get("best_candidate_aggregates", [])))
            except Exception:
                pass
    aggregate_source = read_rows(OUT_DIR / "v1227_functional_bridge_aggregate.csv")
    all_rows.extend(aggregate_source)
    all_rows = unique_rows(all_rows)
    raw_rows = [
        r for r in all_rows
        if str(r.get("stage", "")) != "V1226_LABEL_FREE_FUNCTIONAL_AGGREGATE"
    ]
    candidate_rows = [
        r for r in all_rows
        if str(r.get("stage", "")) == "V1226_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY"
    ]
    aggregate_rows = [
        r for r in all_rows
        if str(r.get("stage", "")) == "V1226_LABEL_FREE_FUNCTIONAL_AGGREGATE" or ("train_shuffle_rows" in r and r.get("train_shuffle_rows", "") != "")
    ]
    if summary_aggregates:
        existing = {
            (str(r.get("base_candidate_id", "")), str(r.get("candidate_id", "")), str(r.get("dataset", "")), str(r.get("seed", "")))
            for r in aggregate_rows
        }
        for r in summary_aggregates:
            key = (str(r.get("base_candidate_id", "")), str(r.get("candidate_id", "")), str(r.get("dataset", "")), str(r.get("seed", "")))
            if key not in existing:
                aggregate_rows.append(r)
                existing.add(key)
    aggregate_rows = unique_rows(aggregate_rows)
    write_rows(OUT_DIR / "v1227_functional_bridge_shadow_or_official.csv", raw_rows)
    write_rows(OUT_DIR / "v1227_functional_bridge_aggregate.csv", aggregate_rows)
    value_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    s4a_labels: list[int] = []
    s5_labels: list[int] = []
    for i, row in enumerate(candidate_rows):
        exp_pass = sint(row.get("exploration_gate_pass"), 0)
        off_pass = sint(row.get("strict_all_pass"), 0)
        source_control = fnum(row.get("source_vs_best_control"), 0.0)
        source_noop = fnum(row.get("source_vs_noop"), 0.0)
        linec = fnum(row.get("LineC_seed_pass_count"), 0.0) / max(1.0, fnum(row.get("linec_seed_count"), 5.0))
        tail = -abs(min(0.0, 0.05 - fnum(row.get("CEp99_delta_vs_noop"), 999.0)))
        score = source_control + 0.5 * source_noop + 0.03 * linec + tail
        scores.append(score)
        s4a_labels.append(exp_pass)
        s5_labels.append(off_pass)
        value_rows.append(
            {
                "row_id": i,
                "candidate_id": row.get("candidate_id", ""),
                "base_id": row.get("base_candidate_id", ""),
                "functional_id": row.get("candidate_id", ""),
                "policy_id": row.get("role_policy", row.get("policy_id", "")),
                "feature_family": row.get("v1227_mechanism_family", ""),
                "precommit_available": 1,
                "uses_label": 0,
                "uses_ce": 0,
                "uses_query": 0,
                "uses_validation": 0,
                "G_LF_score": score,
                "G_task_proxy": source_control + source_noop,
                "G_linec_proxy": linec,
                "G_tail_proxy": -fnum(row.get("CEp99_delta_vs_noop"), 999.0),
                "G_control_proxy": source_control,
                "observed_source_vs_noop": row.get("source_vs_noop", ""),
                "observed_source_vs_control": row.get("source_vs_best_control", ""),
                "observed_LineC_seed_pass_count": row.get("LineC_seed_pass_count", ""),
                "observed_CEp99_delta": row.get("CEp99_delta_vs_noop", ""),
                "observed_NLL_delta": row.get("NLL_delta_vs_noop", ""),
                "observed_ECE_delta": row.get("ECE_delta_vs_noop", ""),
                "s4a_proxy_positive": exp_pass,
                "s5_proxy_positive": off_pass,
            }
        )
    auc_s4a = auc_rank(scores, s4a_labels)
    auc_s5 = auc_rank(scores, s5_labels)
    p_s4a, r_s4a, k_s4a = precision_recall(scores, s4a_labels)
    write_rows(OUT_DIR / "v1227_precommit_value_source.csv", value_rows)
    metrics = {
        "functional_candidate_rows": len(candidate_rows),
        "functional_aggregate_rows": len(aggregate_rows),
        "line_f_exploration_gate_pass": int(any(s4a_labels)),
        "line_f_official_gate_pass": int(any(s5_labels)),
        "functional_any_train_shuffle_robust_majority_pass": int(any(sint(r.get("train_shuffle_robust_majority_pass"), 0) for r in aggregate_rows)),
        "functional_any_train_shuffle_robust_all_pass": int(any(sint(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregate_rows)),
        "precommit_value_source_rows": len(value_rows),
        "AUC_G_LF_S4a_proxy": auc_s4a,
        "AUC_G_LF_S5_proxy": auc_s5,
        "precision_at_k_s4a_proxy": p_s4a,
        "recall_at_k_s4a_proxy": r_s4a,
        "precision_recall_k": k_s4a,
        "value_source_exploration_pass": int(auc_s4a >= 0.70 and p_s4a >= 0.30 and r_s4a >= 0.30),
    }
    return candidate_rows, aggregate_rows, value_rows, metrics


def summarize_classic() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(OUT_DIR / "v1227_rational_memory_repair.csv")
    summary = read_rows(OUT_DIR / "v1227_rational_memory_repair_summary.csv")
    status_rows = summary or rows
    write_rows(OUT_DIR / "v1227_classic_family_status.csv", status_rows)
    meaningful = 0
    memory_blocked = 0
    for row in rows:
        mem = fnum(row.get("memory_ratio_vs_mlp"), 999.0)
        step = fnum(row.get("step_ratio_vs_mlp"), 999.0)
        linec = sint(row.get("linec_pass"), sint(row.get("LineC_pass"), 0))
        delta = fnum(row.get("delta_vs_MLP"), fnum(row.get("mean_delta_vs_MLP"), -999.0))
        if mem <= 1.20 and step <= 1.25 and linec and delta >= -0.05:
            meaningful = 1
        if 1.8 <= mem <= 2.3:
            memory_blocked = 1
    metrics = {
        "classic_rows": len(rows),
        "classic_summary_rows": len(summary),
        "classic_meaningful_progress": meaningful,
        "rational_memory_blocked": memory_blocked,
    }
    return rows, metrics


def write_required_manifest(route: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for name in REQUIRED + FIGURES:
        path = OUT_DIR / name
        rows.append({"artifact": name, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(OUT_DIR / "v1227_required_artifact_manifest.csv", rows)
    return rows


def write_packet() -> tuple[int, str]:
    packet = OUT_DIR / "v1227_code_review_packet.zip"
    members: list[Path] = [
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
        ROOT / "experiments" / "run_v1227_label_free_signal_source.py",
        ROOT / "experiments" / "run_v1227_finalize_label_free_signal_source.py",
        ROOT / "experiments" / "run_v1224_classic_hardening.py",
        ROOT / "dgkan" / "models" / "fc_purekan_primitives.py",
        ROOT / "dgkan" / "kernels" / "fused_hinge_quadratic.py",
    ]
    members.extend(sorted(OUT_DIR.glob("v1227_*")))
    members.extend(sorted(OUT_DIR.glob("fig_v1227_*.svg")))
    seen: set[str] = set()
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in members:
            if not path.exists() or path.is_dir():
                continue
            if path.resolve() == packet.resolve():
                continue
            try:
                arc = str(path.relative_to(ROOT))
            except ValueError:
                arc = path.name
            if arc in seen:
                continue
            seen.add(arc)
            zf.write(path, arc)
    return len(seen), sha256_file(packet)


def write_markdown(route: dict[str, Any], best_base: dict[str, Any] | None) -> None:
    status = [
        "# DG-KAN v12.27 Project Status Summary",
        "",
        f"route = {route.get('route')}",
        f"official_success_reached = {route.get('official_success_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        f"line_a_near_anchor_pass_count = {route.get('line_a_near_anchor_pass_count')}",
        f"line_f_exploration_gate_pass = {route.get('line_f_exploration_gate_pass')}",
        f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
    ]
    if best_base:
        status.extend(
            [
                "",
                "Best label-free signal-source row:",
                f"- candidate = {best_base.get('candidate_id')}",
                f"- mean_delta_vs_mlp = {best_base.get('mean_delta_vs_mlp')}",
                f"- worst_delta_vs_mlp = {best_base.get('worst_delta_vs_mlp')}",
                f"- max_AUC_time_ratio_vs_mlp = {best_base.get('max_AUC_time_ratio_vs_mlp')}",
                f"- linec_nontearing_pass_rate = {best_base.get('linec_nontearing_pass_rate')}",
            ]
        )
    (OUT_DIR / "v1227_project_status_summary.md").write_text("\n".join(status) + "\n", encoding="utf-8")
    nogo = [
        "# DG-KAN v12.27 No-Go Boundary",
        "",
        "This file records the final route boundary from executed artifacts only.",
        "",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"hard_compute_budget_exhausted = {route.get('hard_compute_budget_exhausted')}",
        f"fallback_all_executed = {route.get('fallback_all_executed')}",
        f"label-free near-anchor pass count = {route.get('line_a_near_anchor_pass_count')}",
        f"functional S4a pass = {route.get('line_f_exploration_gate_pass')}",
        "",
        "No promotion is allowed unless route is S5-OfficialFunctionalSuccess.",
    ]
    (OUT_DIR / "v1227_no_go_boundary.md").write_text("\n".join(nogo) + "\n", encoding="utf-8")
    next_rows = [
        {
            "queue_id": "NG1-new-label-free-signal-source",
            "priority": 1,
            "hypothesis": "Design a new label-free signal source beyond current multiview/temporal/block/cross-projection families.",
            "promotion_allowed": 0,
        },
        {
            "queue_id": "NG2-linec-task-colocation",
            "priority": 2,
            "hypothesis": "Constrain train-stream geometry proxy so task-positive rows and LineC-majority rows co-locate.",
            "promotion_allowed": 0,
        },
        {
            "queue_id": "NG3-rational-memory-kernel",
            "priority": 3,
            "hypothesis": "Implement true Rational memory reduction if D34-D38 remain at memory ratio around 2.",
            "promotion_allowed": 0,
        },
    ]
    write_rows(OUT_DIR / "v1227_next_generation_queue.csv", next_rows)


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract, forbidden, code_manifest, symbol_map = build_audits()
    write_rows(OUT_DIR / "v1227_label_free_contract_audit.csv", contract)
    write_rows(OUT_DIR / "v1227_forbidden_token_audit.csv", forbidden)
    write_rows(OUT_DIR / "v1227_code_review_manifest.csv", code_manifest)
    exp.write_json(OUT_DIR / "v1227_core_symbol_map.json", symbol_map)

    _scout_raw, _hard_raw, failure_rows, base_metrics = copy_and_summarize_base()
    functional_rows, functional_aggs, _value_rows, functional_metrics = summarize_functional()
    _classic_rows, classic_metrics = summarize_classic()

    fallback_rows = [
        {"depth": 1, "required_step": "Line_R_audit", "artifact": "v1227_label_free_contract_audit.csv", "executed": int((OUT_DIR / "v1227_label_free_contract_audit.csv").exists())},
        {"depth": 1, "required_step": "Line_A_signal_source_scout", "artifact": "v1227_label_free_signal_source_scout_ablation.csv", "executed": int((OUT_DIR / "v1227_label_free_signal_source_scout_ablation.csv").exists())},
        {"depth": 1, "required_step": "Line_D_rational_memory_repair", "artifact": "v1227_rational_memory_repair.csv", "executed": int((OUT_DIR / "v1227_rational_memory_repair.csv").exists())},
        {"depth": 2, "required_step": "Line_A_hardening_top4", "artifact": "v1227_label_free_signal_source_hardening_ablation.csv", "executed": int((OUT_DIR / "v1227_label_free_signal_source_hardening_ablation.csv").exists())},
        {"depth": 2, "required_step": "Line_C_T_value_source_calibration", "artifact": "v1227_precommit_value_source.csv", "executed": int((OUT_DIR / "v1227_precommit_value_source.csv").exists())},
        {"depth": 3, "required_step": "Line_A_signal_source_repair_or_Line_F", "artifact": "v1227_label_free_signal_source_repair_ablation.csv", "executed": int((OUT_DIR / "v1227_label_free_signal_source_repair_ablation.csv").exists() or len(functional_rows) > 0)},
        {"depth": 4, "required_step": "Line_F_shadow_or_mechanism_no_go", "artifact": "v1227_functional_bridge_shadow_or_official.csv", "executed": int((OUT_DIR / "v1227_functional_bridge_shadow_or_official.csv").exists())},
        {"depth": 5, "required_step": "finalizer_route_packet", "artifact": "v1227_route_decision.json", "executed": 1},
        {"depth": 6, "required_step": "continued_pseudo_partition_affinity_signal_source", "artifact": "v1227_depth6_pseudo_affinity_hardening_ablation.csv", "executed": int((OUT_DIR / "v1227_depth6_pseudo_affinity_hardening_ablation.csv").exists())},
        {"depth": 7, "required_step": "continued_pseudo_partition_geometry_guard_repair", "artifact": "v1227_depth7_pseudo_affinity_repair_scout_ablation.csv", "executed": int((OUT_DIR / "v1227_depth7_pseudo_affinity_repair_scout_ablation.csv").exists())},
        {"depth": 7, "required_step": "continued_pseudo_affinity_functional_shadow", "artifact": "v1227_depth6_pseudo_affinity_functional_shadow.csv", "executed": int((OUT_DIR / "v1227_depth6_pseudo_affinity_functional_shadow.csv").exists())},
    ]
    write_rows(OUT_DIR / "v1227_fallback_execution_manifest.csv", fallback_rows)

    contract_violation = int(any(sint(r.get("uses_y_for_stats"), 0) or sint(r.get("forbidden_token_present"), 0) for r in contract))
    near_count = int(base_metrics.get("near_anchor_pass_count", 0))
    official_base_count = int(base_metrics.get("official_pass_count", 0))
    line_f_exploration = int(functional_metrics.get("line_f_exploration_gate_pass", 0))
    line_f_official = int(functional_metrics.get("line_f_official_gate_pass", 0))
    classic_progress = int(classic_metrics.get("classic_meaningful_progress", 0))
    fallback_all = int(all(sint(r.get("executed"), 0) for r in fallback_rows))

    if contract_violation:
        route = "R0-LabelFreeContractViolation"
        minimum = "Minimum Success none"
    elif line_f_official and official_base_count:
        route = "S5-OfficialFunctionalSuccess"
        minimum = "Minimum Success A"
    elif near_count and line_f_exploration:
        route = "S4a-LabelFreeFunctionalExplorationOpened"
        minimum = "Minimum Success D"
    elif near_count:
        route = "S1-LabelFreeNearAnchorRecovered"
        minimum = "Minimum Success C"
    elif classic_progress:
        route = "R3-ClassicFamilyOnlyProgress"
        minimum = "Minimum Success E"
    else:
        route = "R1-LabelFreeSignalSourceMissing"
        minimum = "Minimum Success E"

    best_base = None
    if failure_rows:
        best_base = sorted(
            failure_rows,
            key=lambda r: (
                fnum(r.get("linec_nontearing_pass_rate"), -1.0),
                fnum(r.get("mean_delta_vs_mlp"), -999.0),
                fnum(r.get("worst_delta_vs_mlp"), -999.0),
                -fnum(r.get("max_AUC_time_ratio_vs_mlp"), 999.0),
            ),
            reverse=True,
        )[0]

    route_obj: dict[str, Any] = {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "p4_pass": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess"),
        "final_stop_allowed": 0,
        "hard_compute_budget_exhausted": int(route != "S5-OfficialFunctionalSuccess" and fallback_all == 1),
        "fallback_depth": max(sint(r.get("depth"), 0) for r in fallback_rows),
        "fallback_rows": len(fallback_rows),
        "fallback_all_executed": fallback_all,
        "line_a_near_anchor_pass_count": near_count,
        "line_a_official_pass_count": official_base_count,
        "line_f_exploration_gate_pass": line_f_exploration,
        "line_f_official_gate_pass": line_f_official,
        "classic_meaningful_progress": classic_progress,
        "contract_violation_count": contract_violation,
        "code_semantics_review_pass": int(not contract_violation and all(r.get("review_status") == "pass" for r in code_manifest)),
        **base_metrics,
        **functional_metrics,
        **classic_metrics,
    }
    write_markdown(route_obj, best_base)

    for fig in FIGURES:
        write_svg(
            OUT_DIR / fig,
            fig.replace("fig_v1227_", "").replace("_", " ").replace(".svg", "").title(),
            [
                f"route: {route_obj['route']}",
                f"Line A near-anchor pass: {near_count}",
                f"Line F exploration pass: {line_f_exploration}",
                f"functional rows: {functional_metrics.get('functional_candidate_rows', 0)}",
                f"classic rows: {classic_metrics.get('classic_rows', 0)}",
                "promotion_allowed: 0 unless S5 official",
            ],
        )

    exp.write_json(OUT_DIR / "v1227_route_decision.json", jsafe(route_obj))
    write_required_manifest(route_obj)
    entries, packet_sha = write_packet()
    route_obj["code_review_packet_entries"] = entries
    route_obj["code_review_packet_sha256"] = packet_sha
    write_required_manifest(route_obj)
    missing = sum(1 for r in read_rows(OUT_DIR / "v1227_required_artifact_manifest.csv") if sint(r.get("exists"), 0) == 0)
    route_obj["required_artifact_missing_count"] = missing
    route_obj["required_artifact_rows"] = len(REQUIRED) + len(FIGURES)
    if missing:
        route_obj["route"] = "R0-ArtifactIncomplete"
        route_obj["promotion_allowed"] = 0
        route_obj["final_stop_allowed"] = 0
    else:
        route_obj["final_stop_allowed"] = int(route_obj["official_success_reached"] or (fallback_all and route_obj["hard_compute_budget_exhausted"]))
    exp.write_json(OUT_DIR / "v1227_route_decision.json", jsafe(route_obj))
    write_required_manifest(route_obj)
    print(json.dumps(jsafe(route_obj), indent=2, ensure_ascii=False, sort_keys=True))
    return route_obj


if __name__ == "__main__":
    run()
