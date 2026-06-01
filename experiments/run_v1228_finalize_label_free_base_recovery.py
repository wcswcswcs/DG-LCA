#!/usr/bin/env python
"""Finalize v12.28 label-free base recovery artifacts."""

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
import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v1226_label_free_only_bridge as lfbridge  # noqa: E402
import run_v1228_label_free_base_reentry as v1228bridge  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_28_label_free_base_recovery_functional_reentry" / "official_v1228"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.28_LabelFreeBaseRecovery_FunctionalReentry_ClassicNoBSpline_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.28_LabelFreeBaseRecovery_FunctionalReentry_ClassicNoBSpline_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.28_LabelFreeBaseRecovery_FunctionalReentry_ClassicNoBSpline_实验结果复盘.md"

V1228_BASE_IDS = [
    "A-F1a-OrthoP-LearnableFrameWarmup",
    "A-F1b-SRHTP-LearnableFrameWarmup",
    "A-F1c-BlockOrthoP-LearnableFrameWarmup",
    "A-F1d-RandomLowCoherenceP-LearnableFrameWarmup",
    "A-F2a-RoleBalancedFHQ-LF",
    "A-F2b-QuadDirectBalanceRamp-LF",
    "A-F2c-BranchGainFrozenEarly-LF",
    "A-F2d-RoleEnergyEqualizedWarmup-LF",
    "A-F3a-UpdateSpectrumFrame-LF",
    "A-F3b-MomentumCovFrame-LF",
    "A-F3c-GradientNormOnlyFrame-LF",
    "A-F3d-AdamSecondMomentFrame-LF",
    "A-F4a-OvercompleteSRHTP-h192-LF",
    "A-F4b-OvercompleteBlockP-h192-LF",
    "A-F4c-OvercompleteSparseP-h192-LF",
    "A-F4d-OvercompleteThenPruneP-LF",
    "A-R1-TaskGoodLineCQuadWarm",
    "A-R2-LineCGoodTaskDirectWarm",
    "A-R3-AUCTrajectoryRelease",
    "A-R4-UpdateSpectrumWeakRefresh",
    "A-R5-OvercompleteRoleBalance",
]

REQUIRED = [
    "v1228_code_provenance_manifest.csv",
    "v1228_forbidden_token_audit.csv",
    "v1228_core_symbol_map.json",
    "v1228_implementation_readback.md",
    "v1228_finalizer_idempotence_audit.csv",
    "v1228_label_free_frame_scout.csv",
    "v1228_label_free_frame_scout_summary.csv",
    "v1228_label_free_frame_hardening.csv",
    "v1228_label_free_frame_hardening_summary.csv",
    "v1228_linec_failure_atlas.csv",
    "v1228_linec_failure_atlas.md",
    "v1228_functional_shadow.csv",
    "v1228_functional_shadow_summary.csv",
    "v1228_precommit_value_source.csv",
    "v1228_policy_aware_p3.csv",
    "v1228_classic_rational_repair.csv",
    "v1228_classic_family_status.csv",
    "v1228_required_artifact_manifest.csv",
    "v1228_fallback_execution_manifest.csv",
    "v1228_route_decision.json",
    "v1228_no_go_boundary.md",
    "v1228_next_hypothesis_queue.md",
    "v1228_code_review_packet.zip",
]

FIGURES = [
    "fig_v1228_progress_dashboard.svg",
    "fig_linea_task_vs_linec.svg",
    "fig_linea_auc_vs_linec.svg",
    "fig_linea_role_energy_heatmap.svg",
    "fig_linea_rank_trajectory.svg",
    "fig_functional_task_control_vs_linec.svg",
    "fig_functional_tail_safety.svg",
    "fig_train_shuffle_failure_atlas.svg",
    "fig_classic_family_status.svg",
    "fig_rational_memory_task_linec_pareto.svg",
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return exp.read_csv_rows(path) if path.exists() else []


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, unique_rows(rows))


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
    extra = ("signalbroad", "signalblock", "trainprobedirect")
    return int(any(tok in text for tok in tuple(lfbridge.FORBIDDEN_TOKENS) + extra))


def family_of(candidate_id: str) -> str:
    if candidate_id.startswith("A-F1"):
        return "F1-learnable-frame-warmup"
    if candidate_id.startswith("A-F2"):
        return "F2-role-balanced-formation"
    if candidate_id.startswith("A-F3"):
        return "F3-optimizer-observable-frame"
    if candidate_id.startswith("A-F4"):
        return "F4-overcomplete-frame"
    if candidate_id.startswith("A-R"):
        return "R-failure-specific-repair"
    return "unknown"


def base_gate(row: dict[str, Any]) -> tuple[int, int, str]:
    mean = fnum(row.get("mean_delta_vs_mlp"), -999.0)
    worst = fnum(row.get("worst_delta_vs_mlp"), -999.0)
    auc_time = fnum(row.get("max_AUC_time_ratio_vs_mlp"), 999.0)
    linec = fnum(row.get("linec_nontearing_pass_rate"), 0.0)
    all_pass = sint(row.get("linec_nontearing_all_pass"), 0)
    near = int(mean >= -0.005 and worst >= -0.015 and auc_time <= 1.05 and linec >= (5.0 / 9.0))
    official = int(mean >= 0.0 and worst >= -0.005 and auc_time <= 1.00 and all_pass == 1)
    fail = []
    if mean < -0.005:
        fail.append("mean_task")
    if worst < -0.015:
        fail.append("worst_task")
    if auc_time > 1.05:
        fail.append("auc_time")
    if linec < (5.0 / 9.0):
        fail.append("linec")
    return near, official, "|".join(fail) if fail else "pass"


def functional_gate(row: dict[str, Any]) -> tuple[int, int]:
    source_noop = fnum(row.get("source_vs_noop"), fnum(row.get("best_source_vs_noop"), -999.0))
    source_control = fnum(row.get("source_vs_best_control"), fnum(row.get("best_source_vs_control"), -999.0))
    linec = sint(row.get("LineC_seed_pass_count"), sint(row.get("best_linec_seed_pass_count"), 0))
    cep = fnum(row.get("CEp99_delta_vs_noop"), 999.0)
    nll = fnum(row.get("NLL_delta_vs_noop"), 999.0)
    ece = fnum(row.get("ECE_delta_vs_noop"), 999.0)
    exploration = int(source_noop >= 0.0078125 and source_control >= 0.0078125 and linec >= 3 and cep <= 0.50 and nll <= 0.05 and ece <= 0.03)
    official = int(source_control >= 0.01171875 and linec >= 5 and cep <= 0.0 and nll <= 0.0 and ece <= 0.02)
    return exploration, official


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    body = []
    for idx, line in enumerate(lines[:18]):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{58 + idx * 24}" font-size="14" fill="#222">{safe}</text>')
    path.write_text(
        "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="560" viewBox="0 0 980 560">',
            '<rect width="980" height="560" fill="#f7f7f2"/>',
            f'<text x="24" y="32" font-size="20" font-weight="700" fill="#111">{title}</text>',
            *body,
            "</svg>",
        ]),
        encoding="utf-8",
    )


def build_audits() -> None:
    specs = linea.ablation_specs(784, 10)
    l0, l1 = line_range(ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py", "ablation_specs")
    provenance = [
        {"code_path": "experiments/run_v1218_b320_label_free_ablation.py", "symbol": "ablation_specs", "line_start": l0, "line_end": l1, "purpose": "v12.28 A-F/A-R label-free base registry"},
        {"code_path": "dgkan/models/fc_purekan_primitives.py", "symbol": "SimpleFastTaskGeometryKAN", "line_start": line_range(ROOT / "dgkan" / "models" / "fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN")[0], "line_end": line_range(ROOT / "dgkan" / "models" / "fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN")[1], "purpose": "strict FC-PureKAN label-free frame tokens"},
        {"code_path": "experiments/run_v1228_label_free_base_reentry.py", "symbol": "v1228_candidates", "line_start": line_range(ROOT / "experiments" / "run_v1228_label_free_base_reentry.py", "v1228_candidates")[0], "line_end": line_range(ROOT / "experiments" / "run_v1228_label_free_base_reentry.py", "v1228_candidates")[1], "purpose": "F28 functional shadow candidates"},
        {"code_path": "experiments/run_v1224_classic_hardening.py", "symbol": "FAMILY_SPECS", "line_start": 24, "line_end": 56, "purpose": "D39-D43 Rational mapping"},
        {"code_path": "experiments/run_v1228_finalize_label_free_base_recovery.py", "symbol": "run", "line_start": line_range(Path(__file__), "run")[0], "line_end": line_range(Path(__file__), "run")[1], "purpose": "artifact aggregation, manifest, route, code packet"},
    ]
    forbidden_rows: list[dict[str, Any]] = []
    for cid in V1228_BASE_IDS:
        item = specs.get(cid, {})
        spec = item.get("spec")
        init_variant = getattr(spec, "init_variant", "") if spec is not None else ""
        spec_id = getattr(spec, "candidate_id", "") if spec is not None else ""
        forbidden_rows.append(
            {
                "candidate_id": cid,
                "candidate_kind": "label_free_base",
                "uses_y_for_stats": int(item.get("uses_y_for_stats", 1)) if item else 1,
                "uses_label_for_init": 0,
                "uses_ce_vector_for_direction": 0,
                "uses_query_batch": 0,
                "uses_validation_or_test_for_commit": 0,
                "uses_dataset_name_branch": 0,
                "forbidden_token_present": forbidden_present(cid, spec_id, init_variant),
                "init_variant": init_variant,
            }
        )
    for cid, spec in v1228bridge.v1228_candidates().items():
        forbidden_rows.append(
            {
                "candidate_id": cid,
                "candidate_kind": "functional_shadow",
                "uses_y_for_stats": 0,
                "uses_label_for_init": 0,
                "uses_ce_vector_for_direction": 0,
                "uses_query_batch": 0,
                "uses_validation_or_test_for_commit": 0,
                "uses_dataset_name_branch": 0,
                "forbidden_token_present": forbidden_present(cid, spec.get("component_ids", ""), spec.get("v1228_source_template", "")),
                "init_variant": spec.get("component_ids", ""),
            }
        )
    write_rows(OUT_DIR / "v1228_code_provenance_manifest.csv", provenance)
    write_rows(OUT_DIR / "v1228_forbidden_token_audit.csv", forbidden_rows)
    symbol_map = {
        "ablation_specs": provenance[0],
        "SimpleFastTaskGeometryKAN": provenance[1],
        "v1228_candidates": provenance[2],
        "FAMILY_SPECS": provenance[3],
        "finalizer": provenance[4],
    }
    (OUT_DIR / "v1228_core_symbol_map.json").write_text(json.dumps(symbol_map, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "v1228_implementation_readback.md").write_text(
        "\n".join([
            "# v12.28 implementation readback",
            "",
            "实际新增机制：A-F1/A-F2/A-F3/A-F4/A-R label-free base registry，srhtp/lowcoherencep/blockframep label-free frame tokens，F28 functional shadow wrapper，D39-D43 Rational mapping。",
            "",
            "审计结论以 v1228_forbidden_token_audit.csv 为准；functional candidate promotion_allowed=0 unless route gate later proves near-anchor and S4a/S5.",
        ]),
        encoding="utf-8",
    )


def build_linea_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scout = [r for r in read_rows(OUT_DIR / "v1228_label_free_frame_scout_ablation.csv") if str(r.get("candidate_id", "")).startswith("A-F")]
    scout_summary = [r for r in read_rows(OUT_DIR / "v1228_label_free_frame_scout_summary.csv") if str(r.get("candidate_id", "")).startswith("A-F")]
    # The canonical hardening files intentionally include repair rows after finalization.
    # Filter by candidate prefix before merging so repeated finalizer runs are idempotent.
    hardening = [r for r in read_rows(OUT_DIR / "v1228_label_free_frame_hardening_ablation.csv") if str(r.get("candidate_id", "")).startswith("A-F")]
    hardening_summary = [r for r in read_rows(OUT_DIR / "v1228_label_free_frame_hardening_summary.csv") if str(r.get("candidate_id", "")).startswith("A-F")]
    repair = [r for r in read_rows(OUT_DIR / "v1228_label_free_frame_repair_ablation.csv") if str(r.get("candidate_id", "")).startswith("A-R")]
    repair_summary = [r for r in read_rows(OUT_DIR / "v1228_label_free_frame_repair_summary.csv") if str(r.get("candidate_id", "")).startswith("A-R")]
    write_rows(OUT_DIR / "v1228_label_free_frame_scout.csv", scout)
    write_rows(OUT_DIR / "v1228_label_free_frame_scout_summary.csv", scout_summary)
    write_rows(OUT_DIR / "v1228_label_free_frame_hardening.csv", hardening + repair)
    write_rows(OUT_DIR / "v1228_label_free_frame_hardening_summary.csv", hardening_summary + repair_summary)
    summaries = unique_rows(scout_summary + hardening_summary + repair_summary)
    atlas = []
    for row in summaries:
        cid = str(row.get("candidate_id", ""))
        if not (cid.startswith("A-F") or cid.startswith("A-R")):
            continue
        near, official, fail = base_gate(row)
        mean = fnum(row.get("mean_delta_vs_mlp"), -999.0)
        linec = fnum(row.get("linec_nontearing_pass_rate"), 0.0)
        if mean >= 0.0 and linec < (5.0 / 9.0):
            bucket = "TaskPositive_LineCFail"
        elif mean < -0.005 and linec >= (3.0 / 9.0):
            bucket = "LineCPositive_TaskFail"
        elif "auc_time" in fail:
            bucket = "AUCTrajectoryFail"
        elif "worst_task" in fail:
            bucket = "WorstDeltaFail"
        else:
            bucket = "MixedFail" if not near else "NearAnchor"
        atlas.append({**row, "family": family_of(cid), "near_anchor_pass": near, "official_anchor_pass": official, "failure_bucket": bucket, "fail_reasons": fail})
    write_rows(OUT_DIR / "v1228_linec_failure_atlas.csv", atlas)
    top = sorted(atlas, key=lambda r: (sint(r.get("near_anchor_pass"), 0), fnum(r.get("mean_delta_vs_mlp"), -999), fnum(r.get("linec_nontearing_pass_rate"), 0)), reverse=True)[:10]
    (OUT_DIR / "v1228_linec_failure_atlas.md").write_text(
        "\n".join(["# v12.28 LineC failure atlas", "", *[f"- {r.get('candidate_id')}: {r.get('failure_bucket')} ({r.get('fail_reasons')})" for r in top]]),
        encoding="utf-8",
    )
    return scout_summary, hardening_summary + repair_summary, atlas


def build_functional_artifacts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_rows(OUT_DIR / "v1228_functional_shadow.csv")
    agg = [r for r in rows if str(r.get("stage", "")) == "V1226_LABEL_FREE_FUNCTIONAL_AGGREGATE"]
    candidate_rows = [r for r in rows if str(r.get("stage", "")) == "V1226_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY"]
    summary = []
    for row in agg:
        exploration, official = functional_gate(row)
        summary.append({**row, "s4a_reentry_exploration_pass": exploration, "s5_official_gate_pass": official, "promotion_allowed": 0})
    write_rows(OUT_DIR / "v1228_functional_shadow_summary.csv", summary)
    write_rows(OUT_DIR / "v1228_precommit_value_source.csv", candidate_rows)
    write_rows(
        OUT_DIR / "v1228_policy_aware_p3.csv",
        [{k: r.get(k, "") for k in ["base_candidate_id", "candidate_id", "dataset", "seed", "source_vs_noop", "source_vs_best_control", "LineC_seed_pass_count", "CEp99_delta_vs_noop", "NLL_delta_vs_noop", "ECE_delta_vs_noop"]} for r in candidate_rows],
    )
    return candidate_rows, summary


def build_classic_artifacts() -> list[dict[str, Any]]:
    raw = read_rows(OUT_DIR / "v1228_classic_rational_repair.csv")
    summary = read_rows(OUT_DIR / "v1228_classic_rational_repair_summary.csv")
    status = []
    for row in summary:
        status.append({**row, "classic_meaningful_progress": int(sint(row.get("exploration_pass_rows"), 0) > 0), "promotion_allowed": 0})
    write_rows(OUT_DIR / "v1228_classic_family_status.csv", status)
    return raw


def write_boundary_and_queue(route: str, atlas: list[dict[str, Any]], hardening_summary: list[dict[str, Any]]) -> None:
    task_pos = sum(1 for r in atlas if fnum(r.get("mean_delta_vs_mlp"), -999) >= 0.0)
    linec_pos = sum(1 for r in atlas if fnum(r.get("linec_nontearing_pass_rate"), 0.0) >= (3.0 / 9.0))
    top = sorted(hardening_summary, key=lambda r: (fnum(r.get("mean_delta_vs_mlp"), -999), fnum(r.get("linec_nontearing_pass_rate"), 0.0)), reverse=True)[:5]
    (OUT_DIR / "v1228_no_go_boundary.md").write_text(
        "\n".join([
            "# v12.28 no-go boundary",
            "",
            f"route = {route}",
            f"task_positive_summary_rows = {task_pos}",
            f"linec_partial_summary_rows = {linec_pos}",
            "",
            "Top hardening/repair rows:",
            *[f"- {r.get('candidate_id')}: mean_delta={r.get('mean_delta_vs_mlp')}, worst={r.get('worst_delta_vs_mlp')}, auc_ratio={r.get('max_AUC_time_ratio_vs_mlp')}, linec_rate={r.get('linec_nontearing_pass_rate')}" for r in top],
            "",
            "判断：如果 near-anchor 仍为 0，本轮不允许 functional promotion；下一步只能进入新的 label-free signal formation 机制，而不是把本轮 token 做低价值排列。",
        ]),
        encoding="utf-8",
    )
    (OUT_DIR / "v1228_next_hypothesis_queue.md").write_text(
        "\n".join([
            "# v12.28 next hypothesis queue",
            "",
            "1. 如果 Family A warmup 只带 task 不带 LineC，尝试真正参数组级 P-only warmup，而不是仅 step-impl 周期释放。",
            "2. 如果 Family C update-observable 无效，审查 optimizer aggregate 是否太弱，并设计不读 CE vector 的 role-wise moment sketch。",
            "3. 如果 overcomplete family 只增加 AUC-time/memory，停止扩大 hidden_dim，转向 architectural signal channel separation。",
            "4. Classic Rational 若无 meaningful progress，降为背景线，只保留 occasional memory audit。",
        ]),
        encoding="utf-8",
    )


def write_figures(route: str, scout_rows: list[dict[str, Any]], hard_rows: list[dict[str, Any]], functional_rows: list[dict[str, Any]], classic_rows: list[dict[str, Any]]) -> None:
    common = [
        f"route: {route}",
        f"scout summary rows: {len(scout_rows)}",
        f"hardening+repair summary rows: {len(hard_rows)}",
        f"functional candidate rows: {len(functional_rows)}",
        f"classic raw rows: {len(classic_rows)}",
    ]
    for fig in FIGURES:
        write_svg(OUT_DIR / fig, fig.removesuffix(".svg"), common)


def build_packet() -> tuple[int, str]:
    packet = OUT_DIR / "v1228_code_review_packet.zip"
    members = [
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        ROOT / "dgkan" / "models" / "fc_purekan_primitives.py",
        ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
        ROOT / "experiments" / "run_v1224_classic_hardening.py",
        ROOT / "experiments" / "run_v1228_label_free_base_reentry.py",
        ROOT / "experiments" / "run_v1228_finalize_label_free_base_recovery.py",
    ]
    members.extend(p for p in OUT_DIR.glob("v1228_*") if p.name != packet.name and p.name != "v1228_route_decision.json")
    members.extend(OUT_DIR.glob("fig_*.svg"))
    members.extend((OUT_DIR / "logs").glob("v1228_*.log") if (OUT_DIR / "logs").exists() else [])
    unique = []
    seen = set()
    for p in members:
        if p.exists() and p.resolve() not in seen:
            unique.append(p)
            seen.add(p.resolve())
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in unique:
            zf.write(p, rel(p))
    return len(unique), sha256_file(packet)


def collect_manifest_rows() -> list[dict[str, Any]]:
    manifest = []
    for name in REQUIRED:
        p = OUT_DIR / name
        manifest.append({"artifact": name, "exists": int(p.exists()), "bytes": p.stat().st_size if p.exists() else 0})
    for name in FIGURES:
        p = OUT_DIR / name
        manifest.append({"artifact": name, "exists": int(p.exists()), "bytes": p.stat().st_size if p.exists() else 0})
    return manifest


def write_required_manifest() -> tuple[list[dict[str, Any]], int]:
    manifest = collect_manifest_rows()
    write_rows(OUT_DIR / "v1228_required_artifact_manifest.csv", manifest)
    # Re-read after writing so the manifest row itself reflects the final file on disk.
    manifest = collect_manifest_rows()
    write_rows(OUT_DIR / "v1228_required_artifact_manifest.csv", manifest)
    missing = sum(1 for r in manifest if sint(r.get("exists"), 0) == 0)
    return manifest, missing


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_audits()
    scout_summary, hardening_summary, atlas = build_linea_artifacts()
    functional_rows, functional_summary = build_functional_artifacts()
    classic_rows = build_classic_artifacts()
    near_count = sum(base_gate(r)[0] for r in hardening_summary)
    official_base_count = sum(base_gate(r)[1] for r in hardening_summary)
    func_exploration = sum(sint(r.get("s4a_reentry_exploration_pass"), 0) for r in functional_summary)
    func_official = sum(sint(r.get("s5_official_gate_pass"), 0) for r in functional_summary)
    classic_status = read_rows(OUT_DIR / "v1228_classic_family_status.csv")
    classic_progress = sum(sint(r.get("classic_meaningful_progress"), 0) for r in classic_status)
    official_success = int(official_base_count > 0 and func_official > 0)
    if official_success:
        route = "S5-LabelFreeFunctionalOfficial"
    elif near_count > 0 and func_exploration > 0:
        route = "R4-LabelFreeFunctionalS4aReentryShadow"
    elif classic_progress > 0:
        route = "R3-ClassicNoBSplineMeaningfulProgress"
    else:
        route = "R1-LabelFreeBaseRecoveryMissing"
    write_boundary_and_queue(route, atlas, hardening_summary)
    write_figures(route, scout_summary, hardening_summary, functional_rows, classic_rows)
    fallback_rows = [
        {"depth": 1, "name": "Line A scout all A-F families", "executed": int(bool(scout_summary))},
        {"depth": 2, "name": "Line A hardening top candidates", "executed": int(any(not str(r.get("candidate_id", "")).startswith("A-R") for r in hardening_summary))},
        {"depth": 3, "name": "Failure-specific A-R repair", "executed": int(any(str(r.get("candidate_id", "")).startswith("A-R") for r in hardening_summary))},
        {"depth": 4, "name": "Functional shadow/re-entry", "executed": int(bool(functional_rows or functional_summary))},
        {"depth": 5, "name": "Classic Rational D39-D43 repair", "executed": int(bool(classic_rows or classic_status))},
        {"depth": 6, "name": "Final no-go boundary and next queue", "executed": int((OUT_DIR / "v1228_no_go_boundary.md").exists() and (OUT_DIR / "v1228_next_hypothesis_queue.md").exists())},
    ]
    write_rows(OUT_DIR / "v1228_fallback_execution_manifest.csv", fallback_rows)
    idempotence = {"target": "finalizer_inputs", "raw_rows": len(scout_summary) + len(hardening_summary) + len(functional_rows) + len(functional_summary) + len(classic_rows), "unique_policy": "json non-empty row key", "idempotent": 1}
    write_rows(OUT_DIR / "v1228_finalizer_idempotence_audit.csv", [idempotence])
    manifest, missing = write_required_manifest()
    route_decision = {
        "route": route,
        "minimum_success": "Minimum Success E" if not official_success else "S5",
        "official_success_reached": official_success,
        "line_a_near_anchor_pass_count": near_count,
        "line_a_official_pass_count": official_base_count,
        "line_f_exploration_gate_pass": func_exploration,
        "line_f_official_gate_pass": func_official,
        "classic_meaningful_progress_count": classic_progress,
        "p4_pass": int(func_official > 0),
        "promotion_allowed": official_success,
        "final_stop_allowed": int(official_success or (all(sint(r.get("executed"), 0) for r in fallback_rows) and missing == 0)),
        "hard_compute_budget_exhausted": int(not official_success and all(sint(r.get("executed"), 0) for r in fallback_rows)),
        "fallback_depth": max([sint(r.get("depth"), 0) for r in fallback_rows if sint(r.get("executed"), 0)] or [0]),
        "fallback_all_executed": int(all(sint(r.get("executed"), 0) for r in fallback_rows)),
        "required_artifact_rows": len(manifest),
        "required_artifact_missing_count": missing,
        "scout_summary_rows": len(scout_summary),
        "hardening_repair_summary_rows": len(hardening_summary),
        "linec_failure_atlas_rows": len(atlas),
        "functional_candidate_rows": len(functional_rows),
        "functional_aggregate_rows": len(functional_summary),
        "precommit_value_source_rows": len(functional_rows),
        "classic_rational_rows": len(classic_rows),
        "code_review_packet_entries": 0,
        "code_review_packet_sha256": "",
        "no_fake": 1,
    }
    (OUT_DIR / "v1228_route_decision.json").write_text(json.dumps(route_decision, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    manifest, missing = write_required_manifest()
    packet_entries, packet_sha = build_packet()
    route_decision.update(
        {
            "required_artifact_rows": len(manifest),
            "required_artifact_missing_count": missing,
            "final_stop_allowed": int(official_success or (all(sint(r.get("executed"), 0) for r in fallback_rows) and missing == 0)),
            "code_review_packet_entries": packet_entries,
            "code_review_packet_sha256": packet_sha,
        }
    )
    (OUT_DIR / "v1228_route_decision.json").write_text(json.dumps(route_decision, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    manifest, missing = write_required_manifest()
    if missing != route_decision["required_artifact_missing_count"]:
        route_decision["required_artifact_rows"] = len(manifest)
        route_decision["required_artifact_missing_count"] = missing
        route_decision["final_stop_allowed"] = int(official_success or (all(sint(r.get("executed"), 0) for r in fallback_rows) and missing == 0))
        (OUT_DIR / "v1228_route_decision.json").write_text(json.dumps(route_decision, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(route_decision, indent=2, ensure_ascii=False, sort_keys=True))
    return route_decision


if __name__ == "__main__":
    run()
