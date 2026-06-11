#!/usr/bin/env python3
"""v22.03 final route, recap, execution log, and packets."""

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

from experiments.run_v22_03_common import (  # noqa: E402
    PYTHON,
    V2203_RECAP_DOC,
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
    write_text,
)


REQUIRED_ARTIFACTS = [
    "v22_03_code_truth_gate.csv",
    "v22_03_efficiency_full_loop_summary.csv",
    "v22_03_drat_drbf_active_repair.csv",
    "v22_03_drat_drbf_active_repair_summary.csv",
    "v22_03_drat_drbf_component_waterfall.csv",
    "v22_03_drat_drbf_repair_decision.json",
    "v22_03_drat_drbf_runner_integration_summary.csv",
    "v22_03_drat_drbf_runner_integration_decision.json",
    "v22_03_drat_drbf_limited_smoke_summary.csv",
    "v22_03_drat_drbf_limited_smoke_route.csv",
    "v22_03_drat_drbf_limited_smoke_route.json",
    "v22_03_drat_drbf_c4_smoke_summary.csv",
    "v22_03_drat_drbf_c4_smoke_route.csv",
    "v22_03_drat_drbf_c4_smoke_route.json",
    "v22_03_terminal_erosion_autopsy.csv",
    "v22_03_terminal_erosion_route.json",
    "v22_03_f145_f147_source_preserve_audit.csv",
    "v22_03_f145_f147_source_preserve_route.json",
    "v22_03_f145_f147_source_preserve_dataset_localization.csv",
    "v22_03_f145_f147_source_preserve_row_localization.csv",
    "v22_03_f148_f150_diffeomorphic_target_audit.csv",
    "v22_03_f148_f150_diffeomorphic_target_route.json",
    "v22_03_f148_f150_diffeomorphic_target_dataset_localization.csv",
    "v22_03_f148_f150_diffeomorphic_target_row_localization.csv",
    "v22_03_f151_f153_terminal_preserve_repair_audit.csv",
    "v22_03_f151_f153_terminal_preserve_repair_route.json",
    "v22_03_f151_f153_terminal_preserve_repair_dataset_localization.csv",
    "v22_03_f151_f153_terminal_preserve_repair_row_localization.csv",
    "v22_03_f154_f156_terminal_debt_lownds_dualmem_audit.csv",
    "v22_03_f154_f156_terminal_debt_lownds_dualmem_route.json",
    "v22_03_f154_f156_terminal_debt_lownds_dualmem_dataset_localization.csv",
    "v22_03_f154_f156_terminal_debt_lownds_dualmem_row_localization.csv",
    "v22_03_f157_f159_signal_estimator_audit.csv",
    "v22_03_f157_f159_signal_estimator_route.json",
    "v22_03_f157_f159_signal_estimator_dataset_localization.csv",
    "v22_03_f157_f159_signal_estimator_row_localization.csv",
    "v22_03_f160_f162_post_h4000_source_floor_audit.csv",
    "v22_03_f160_f162_post_h4000_source_floor_route.json",
    "v22_03_f160_f162_post_h4000_source_floor_dataset_localization.csv",
    "v22_03_f160_f162_post_h4000_source_floor_row_localization.csv",
    "v22_03_f163_f165_raw_guard_post_h4000_floor_audit.csv",
    "v22_03_f163_f165_raw_guard_post_h4000_floor_route.json",
    "v22_03_f163_f165_raw_guard_post_h4000_floor_dataset_localization.csv",
    "v22_03_f163_f165_raw_guard_post_h4000_floor_row_localization.csv",
    "v22_03_f166_f168_anti_erosion_audit.csv",
    "v22_03_f166_f168_anti_erosion_route.json",
    "v22_03_f166_f168_anti_erosion_dataset_localization.csv",
    "v22_03_f166_f168_anti_erosion_row_localization.csv",
    "v22_03_f169_f171_h3200_anchor_transport_audit.csv",
    "v22_03_f169_f171_h3200_anchor_transport_route.json",
    "v22_03_f169_f171_h3200_anchor_transport_dataset_localization.csv",
    "v22_03_f169_f171_h3200_anchor_transport_row_localization.csv",
    "v22_03_f172_f174_progress_carry_audit.csv",
    "v22_03_f172_f174_progress_carry_route.json",
    "v22_03_f172_f174_progress_carry_dataset_localization.csv",
    "v22_03_f172_f174_progress_carry_row_localization.csv",
    "v22_03_f175_f177_gentle_progress_audit.csv",
    "v22_03_f175_f177_gentle_progress_route.json",
    "v22_03_f175_f177_gentle_progress_dataset_localization.csv",
    "v22_03_f175_f177_gentle_progress_row_localization.csv",
    "v22_03_f178_f180_control_relative_catchup_audit.csv",
    "v22_03_f178_f180_control_relative_catchup_route.json",
    "v22_03_f178_f180_control_relative_catchup_dataset_localization.csv",
    "v22_03_f178_f180_control_relative_catchup_row_localization.csv",
    "v22_03_f181_f183_trajectory_adaptive_audit.csv",
    "v22_03_f181_f183_trajectory_adaptive_route.json",
    "v22_03_f181_f183_trajectory_adaptive_dataset_localization.csv",
    "v22_03_f181_f183_trajectory_adaptive_row_localization.csv",
    "v22_03_f184_f186_minimal_transport_audit.csv",
    "v22_03_f184_f186_minimal_transport_route.json",
    "v22_03_f184_f186_minimal_transport_dataset_localization.csv",
    "v22_03_f184_f186_minimal_transport_row_localization.csv",
    "v22_03_f187_f189_accept_memory_audit.csv",
    "v22_03_f187_f189_accept_memory_route.json",
    "v22_03_f187_f189_accept_memory_dataset_localization.csv",
    "v22_03_f187_f189_accept_memory_row_localization.csv",
    "v22_03_kan_source_channel_writer_matrix.csv",
    "v22_03_runnable_queue.csv",
    "v22_03_gpu_assignment_manifest.csv",
    "v22_03_gpu_utilization_timeline.csv",
    "v22_03_idle_violation.csv",
    "v22_03_deferred_items.csv",
    "v22_03_queue_drain_report.csv",
    "v22_03_command_journal.csv",
]


SOURCE_CONTINUATIONS = [
    {
        "key": "f145_f147_source_preserve",
        "label": "F145-F147 source-preservation oldshape full",
        "source_dir": "continuation_f145_f147_source_preserve_oldshape_full",
    },
    {
        "key": "f148_f150_diffeomorphic_target",
        "label": "F148-F150 diffeomorphic target oldshape full",
        "source_dir": "continuation_f148_f150_diffeomorphic_target_oldshape_full",
    },
    {
        "key": "f151_f153_terminal_preserve_repair",
        "label": "F151-F153 terminal-preserve repair oldshape full",
        "source_dir": "continuation_f151_f153_terminal_preserve_repair_oldshape_full",
    },
    {
        "key": "f154_f156_terminal_debt_lownds_dualmem",
        "label": "F154-F156 terminal debt/low-NDS/dual-memory oldshape full",
        "source_dir": "continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full",
    },
    {
        "key": "f157_f159_signal_estimator",
        "label": "F157-F159 signal-estimator terminal retention oldshape full",
        "source_dir": "continuation_f157_f159_signal_estimator_oldshape_full",
    },
    {
        "key": "f160_f162_post_h4000_source_floor",
        "label": "F160-F162 post-h4000 source-floor terminal retention oldshape full",
        "source_dir": "continuation_f160_f162_post_h4000_source_floor_oldshape_full",
    },
    {
        "key": "f163_f165_raw_guard_post_h4000_floor",
        "label": "F163-F165 raw-guard post-h4000 floor oldshape full",
        "source_dir": "continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full",
    },
    {
        "key": "f166_f168_anti_erosion",
        "label": "F166-F168 anti-erosion orthogonal/transport oldshape full",
        "source_dir": "continuation_f166_f168_anti_erosion_oldshape_full",
    },
    {
        "key": "f169_f171_h3200_anchor_transport",
        "label": "F169-F171 h3200-anchor transport oldshape full",
        "source_dir": "continuation_f169_f171_h3200_anchor_transport_oldshape_full",
    },
    {
        "key": "f172_f174_progress_carry",
        "label": "F172-F174 source-preserving progress-carry oldshape full",
        "source_dir": "continuation_f172_f174_progress_carry_oldshape_full",
    },
    {
        "key": "f175_f177_gentle_progress",
        "label": "F175-F177 raw-guard gentle progress oldshape full",
        "source_dir": "continuation_f175_f177_gentle_progress_oldshape_full",
    },
    {
        "key": "f178_f180_control_relative_catchup",
        "label": "F178-F180 control-relative terminal catch-up oldshape full",
        "source_dir": "continuation_f178_f180_control_relative_catchup_oldshape_full",
    },
    {
        "key": "f181_f183_trajectory_adaptive",
        "label": "F181-F183 trajectory-adaptive terminal retention oldshape full",
        "source_dir": "continuation_f181_f183_trajectory_adaptive_oldshape_full",
    },
    {
        "key": "f184_f186_minimal_transport",
        "label": "F184-F186 minimal terminal transport oldshape full",
        "source_dir": "continuation_f184_f186_minimal_transport_oldshape_full",
    },
    {
        "key": "f187_f189_accept_memory",
        "label": "F187-F189 terminal accept-memory oldshape full",
        "source_dir": "continuation_f187_f189_accept_memory_oldshape_full",
    },
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _first(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    return next((r for r in rows if str(r.get(key, "")) == value), {})


def write_queue_artifacts(out_dir: Path) -> None:
    journal = read_rows(out_dir / "v22_03_command_journal.csv")
    latest_by_command: dict[str, dict[str, Any]] = {}
    for row in journal:
        cmd = str(row.get("command", ""))
        if "run_v21_01_source_retention.py" in cmd:
            latest_by_command[cmd] = row
    queue_rows = []
    gpu_rows = []
    for row in latest_by_command.values():
        cmd = str(row.get("command", ""))
        queue_rows.append(
            {
                "command": cmd,
                "status": row.get("status", ""),
                "note": row.get("note", ""),
                "timestamp": row.get("timestamp", ""),
            }
        )
        for idx in range(4):
            if f"cuda:{idx}" in cmd:
                gpu_rows.append({"gpu": idx, "command": cmd, "status": row.get("status", ""), "timestamp": row.get("timestamp", "")})
    write_rows(out_dir / "v22_03_runnable_queue.csv", queue_rows)
    write_rows(out_dir / "v22_03_gpu_assignment_manifest.csv", gpu_rows)
    write_rows(
        out_dir / "v22_03_gpu_utilization_timeline.csv",
        [{"timestamp": now_sg(), "source": "not_sampled_by_runner", "note": "nvidia-smi was checked before launch; continuous utilization sampling was not collected"}],
    )
    write_rows(
        out_dir / "v22_03_idle_violation.csv",
        [{"checked": 1, "idle_violation": "", "blocker": "continuous_gpu_utilization_not_sampled"}],
    )
    write_rows(
        out_dir / "v22_03_deferred_items.csv",
        [
            {"item": "D-RAT/D-RBF functional smoke", "reason": "active micro-kernel repair found micro-near-E1 rows and limited same-kernel runner integration passed, but production official fused kernels are still missing"},
            {"item": "D-RAT/D-RBF existing-carrier source smoke", "reason": "existing PrimitiveKAN carrier path smoke ran on MNIST seed0 to h1600 and did not open h800 source; this is a negative readback, not a micro-kernel official fused proof"},
            {"item": "D-RAT/D-RBF C4 named FU smoke", "reason": "RAT-FU/RBF-FU named smoke ran as existing source-retention aliases; it remains a limited readback until production official fused RAT/RBF kernels are wired into the functional runner"},
            {"item": "v22.03 independent rerun confirmation", "reason": "only required if productive_h4800_group passes"},
            {"item": "exact Jacobian/Hessian diffeomorphic diagnostics", "reason": "C2 recorded train-only local logit displacement proxies; exact high-order diagnostics remain audit-only future work"},
        ],
    )
    write_rows(
        out_dir / "v22_03_queue_drain_report.csv",
        [{"queued_commands": len(queue_rows), "gpu_assigned_commands": len(gpu_rows), "drained": int(all(str(r.get("status")) != "started" for r in queue_rows))}],
    )


def route_from(out_dir: Path) -> dict[str, Any]:
    code = read_rows(out_dir / "v22_03_code_truth_gate.csv")
    eff = read_rows(out_dir / "v22_03_efficiency_full_loop_summary.csv")
    dr = read_json(out_dir / "v22_03_drat_drbf_repair_decision.json")
    eros = read_json(out_dir / "v22_03_terminal_erosion_route.json")
    src_routes = [read_json(out_dir / f"v22_03_{c['key']}_route.json") for c in SOURCE_CONTINUATIONS]
    src_routes = [r for r in src_routes if r]
    kan = read_json(out_dir / "v22_03_kan_source_channel_writer_decision.json")
    missing = [name for name in REQUIRED_ARTIFACTS if not (out_dir / name).exists()]
    s010 = int(bool(code) and all(int_flag(r.get("pass")) for r in code))
    dche = _first(eff, "carrier", "D-CHE")
    dfou = _first(eff, "carrier", "D-FOU")
    dche_closure = int_flag(dche.get("full_loop_official_closure"))
    dfou_closure = int_flag(dfou.get("full_loop_official_closure"))
    early = sum(int(r.get("candidate_early_chain", 0) or 0) for r in src_routes)
    h4800 = sum(int(r.get("candidate_h4800", 0) or 0) for r in src_routes)
    h3200 = sum(int(r.get("candidate_continuous_h3200", 0) or 0) for r in src_routes)
    erosion_groups = sum(int(r.get("terminal_erosion_groups", 0) or 0) for r in src_routes)
    best_route = max(src_routes, key=lambda r: finite_float(r.get("best_h4800_retention_ratio"), -1.0), default={})
    if missing:
        route = "R0-CodeOrArtifactIncomplete"
    elif not s010:
        route = "R0-CodeOrMetricInvalid"
    elif not (dche_closure and dfou_closure):
        route = "R1-EfficiencyOfficializationIncomplete"
    elif h4800:
        route = "R5-MLPS3RetainedGenericSource"
    elif eros.get("decision") == "TerminalErosionExplained":
        route = "R3-MLPTerminalErosionExplained"
    else:
        route = "R4-MLPTerminalErosionUnknown"
    return {
        "route": route,
        "S0_10_pass": s010,
        "D-CHE_full_loop_official_closure": dche_closure,
        "D-FOU_full_loop_official_closure": dfou_closure,
        "D-RAT_decision": (dr.get("D-RAT") or {}).get("decision", ""),
        "D-RBF_decision": (dr.get("D-RBF") or {}).get("decision", ""),
        "functional_early": early,
        "functional_h3200": h3200,
        "functional_h4800": h4800,
        "terminal_erosion_groups": erosion_groups,
        "best_source_continuation": best_route.get("continuation", ""),
        "best_source_candidate": best_route.get("best_v22_id", ""),
        "best_h4800": best_route.get("best_h4800", ""),
        "best_h4800_retention_ratio": best_route.get("best_h4800_retention_ratio", ""),
        "terminal_autopsy_decision": eros.get("decision", ""),
        "KANRoute": kan.get("decision", "KANSourceChannelMismatch"),
        "promotion_allowed": 0,
        "required_artifact_missing_count": len(missing),
        "missing_artifacts": ";".join(missing),
    }


def _continuation_rows(out_dir: Path, suffix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cont in SOURCE_CONTINUATIONS:
        for row in read_rows(out_dir / f"v22_03_{cont['key']}_{suffix}.csv"):
            row = dict(row)
            row.setdefault("continuation", cont["label"])
            rows.append(row)
    return rows


def _source_route_rows(out_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cont in SOURCE_CONTINUATIONS:
        row = read_json(out_dir / f"v22_03_{cont['key']}_route.json")
        if row:
            out.append(row)
    return out


def _candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if not str(r.get("v22_id", "")).startswith("CTRL")]


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _target_trace_summary(out_dir: Path) -> list[dict[str, Any]]:
    trace_path = out_dir / "continuation_f148_f150_diffeomorphic_target_oldshape_full" / "v21_01_source_retention_raw_traces.csv"
    rows = read_rows(trace_path)
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        v22_id = str(row.get("v21_id", ""))
        if not v22_id.startswith(("MLP-F148-", "MLP-F149-", "MLP-F150-")):
            continue
        if not any(str(row.get(k, "")) for k in ("operator_status", "diffeomorphic_target_family", "target_kind")):
            continue
        by_id.setdefault(v22_id, []).append(row)
    out: list[dict[str, Any]] = []
    numeric_fields = [
        "ActuationR2",
        "B2_transfer_gain",
        "B3_safety_gain",
        "jacobian_condition_mean",
        "fold_rate",
        "neighbor_order_flip_rate",
        "local_distance_distortion",
        "operator_gate_accept",
    ]
    for v22_id, group in sorted(by_id.items()):
        vals = {field: [finite_float(r.get(field)) for r in group] for field in numeric_fields}
        statuses = sorted({str(r.get("operator_status", "")) for r in group if str(r.get("operator_status", ""))})
        families = sorted({str(r.get("diffeomorphic_target_family", "")) for r in group if str(r.get("diffeomorphic_target_family", ""))})
        out.append(
            {
                "v22_id": v22_id,
                "target_attempt_rows": len(group),
                "operator_accept_rows": sum(int_flag(r.get("operator_gate_accept")) for r in group),
                "target_families": ";".join(families),
                "operator_statuses": ";".join(statuses),
                "mean_ActuationR2": _mean(vals["ActuationR2"]),
                "max_B2_transfer_gain": max([v for v in vals["B2_transfer_gain"] if v == v], default=""),
                "max_B3_safety_gain": max([v for v in vals["B3_safety_gain"] if v == v], default=""),
                "max_jacobian_condition_mean": max([v for v in vals["jacobian_condition_mean"] if v == v], default=""),
                "max_fold_rate": max([v for v in vals["fold_rate"] if v == v], default=""),
                "max_neighbor_order_flip_rate": max([v for v in vals["neighbor_order_flip_rate"] if v == v], default=""),
                "max_local_distance_distortion": max([v for v in vals["local_distance_distortion"] if v == v], default=""),
            }
        )
    write_rows(out_dir / "v22_03_f148_f150_diffeomorphic_target_trace_summary.csv", out)
    return out


def write_recap(out_dir: Path, route: dict[str, Any], zip_path: Path, bundle: Path) -> None:
    code = read_rows(out_dir / "v22_03_code_truth_gate.csv")
    eff = read_rows(out_dir / "v22_03_efficiency_full_loop_summary.csv")
    dr = read_rows(out_dir / "v22_03_drat_drbf_repair_decision.csv")
    runner = read_rows(out_dir / "v22_03_drat_drbf_runner_integration_summary.csv")
    limited_smoke = read_rows(out_dir / "v22_03_drat_drbf_limited_smoke_summary.csv")
    limited_smoke_route = read_rows(out_dir / "v22_03_drat_drbf_limited_smoke_route.csv")
    c4_smoke = read_rows(out_dir / "v22_03_drat_drbf_c4_smoke_summary.csv")
    c4_smoke_route = read_rows(out_dir / "v22_03_drat_drbf_c4_smoke_route.csv")
    eros = read_rows(out_dir / "v22_03_terminal_erosion_autopsy.csv")
    eros_classes = read_rows(out_dir / "v22_03_terminal_erosion_class_summary.csv")
    source_routes = _source_route_rows(out_dir)
    src = _continuation_rows(out_dir, "audit")
    dataset = _continuation_rows(out_dir, "dataset_localization")
    rows = _continuation_rows(out_dir, "row_localization")
    c1_src = read_rows(out_dir / "v22_03_f145_f147_source_preserve_audit.csv")
    c2_src = read_rows(out_dir / "v22_03_f148_f150_diffeomorphic_target_audit.csv")
    c3_src = read_rows(out_dir / "v22_03_f151_f153_terminal_preserve_repair_audit.csv")
    c4_src = read_rows(out_dir / "v22_03_f154_f156_terminal_debt_lownds_dualmem_audit.csv")
    c5_src = read_rows(out_dir / "v22_03_f157_f159_signal_estimator_audit.csv")
    c6_src = read_rows(out_dir / "v22_03_f160_f162_post_h4000_source_floor_audit.csv")
    c7_src = read_rows(out_dir / "v22_03_f163_f165_raw_guard_post_h4000_floor_audit.csv")
    c8_src = read_rows(out_dir / "v22_03_f166_f168_anti_erosion_audit.csv")
    c9_src = read_rows(out_dir / "v22_03_f169_f171_h3200_anchor_transport_audit.csv")
    c10_src = read_rows(out_dir / "v22_03_f172_f174_progress_carry_audit.csv")
    c11_src = read_rows(out_dir / "v22_03_f175_f177_gentle_progress_audit.csv")
    c12_src = read_rows(out_dir / "v22_03_f178_f180_control_relative_catchup_audit.csv")
    c13_src = read_rows(out_dir / "v22_03_f181_f183_trajectory_adaptive_audit.csv")
    c14_src = read_rows(out_dir / "v22_03_f184_f186_minimal_transport_audit.csv")
    c15_src = read_rows(out_dir / "v22_03_f187_f189_accept_memory_audit.csv")
    target_trace = _target_trace_summary(out_dir)
    kan = read_rows(out_dir / "v22_03_kan_source_channel_writer_matrix.csv")
    artifacts = []
    for path in sorted(out_dir.glob("v22_03_*")):
        if path.is_file():
            artifacts.append({"artifact": str(path), "exists": 1, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(out_dir / "v22_03_required_artifact_manifest.csv", artifacts)
    best_src = sorted(_candidate_rows(src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c1 = sorted(_candidate_rows(c1_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c2 = sorted(_candidate_rows(c2_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c3 = sorted(_candidate_rows(c3_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c4 = sorted(_candidate_rows(c4_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c5 = sorted(_candidate_rows(c5_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c6 = sorted(_candidate_rows(c6_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c7 = sorted(_candidate_rows(c7_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c8 = sorted(_candidate_rows(c8_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c9 = sorted(_candidate_rows(c9_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c10 = sorted(_candidate_rows(c10_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c11 = sorted(_candidate_rows(c11_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c12 = sorted(_candidate_rows(c12_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c13 = sorted(_candidate_rows(c13_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c14 = sorted(_candidate_rows(c14_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best_c15 = sorted(_candidate_rows(c15_src), key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)
    best = best_src[0] if best_src else {}
    c1_best = best_c1[0] if best_c1 else {}
    c2_best = best_c2[0] if best_c2 else {}
    c3_best = best_c3[0] if best_c3 else {}
    c4_best = best_c4[0] if best_c4 else {}
    c5_best = best_c5[0] if best_c5 else {}
    c6_best = best_c6[0] if best_c6 else {}
    c7_best = best_c7[0] if best_c7 else {}
    c8_best = best_c8[0] if best_c8 else {}
    c9_best = best_c9[0] if best_c9 else {}
    c10_best = best_c10[0] if best_c10 else {}
    c11_best = best_c11[0] if best_c11 else {}
    c12_best = best_c12[0] if best_c12 else {}
    c13_best = best_c13[0] if best_c13 else {}
    c14_best = best_c14[0] if best_c14 else {}
    c15_best = best_c15[0] if best_c15 else {}
    drat_repair = _first(dr, "carrier", "D-RAT")
    drbf_repair = _first(dr, "carrier", "D-RBF")
    drat_smoke = _first(limited_smoke_route, "carrier", "D-RAT")
    drbf_smoke = _first(limited_smoke_route, "carrier", "D-RBF")
    drat_c4_smoke = _first(c4_smoke_route, "carrier", "D-RAT")
    drbf_c4_smoke = _first(c4_smoke_route, "carrier", "D-RBF")
    text = [
        "# DG-KAN v22.03 TerminalRetention DiffeomorphicSourceChannel BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- S0.10 pass: {route.get('S0_10_pass')}",
        f"- D-CHE/D-FOU full-loop official closure: {route.get('D-CHE_full_loop_official_closure')} / {route.get('D-FOU_full_loop_official_closure')}",
        f"- D-RAT/D-RBF: {route.get('D-RAT_decision')} / {route.get('D-RBF_decision')}",
        f"- functional early / h3200 / h4800: {route.get('functional_early')} / {route.get('functional_h3200')} / {route.get('functional_h4800')}",
        f"- terminal erosion groups: {route.get('terminal_erosion_groups')}",
        f"- best source continuation: `{route.get('best_source_continuation')}`",
        f"- best source candidate: `{route.get('best_source_candidate')}`",
        f"- best h4800 / retention ratio: {route.get('best_h4800')} / {route.get('best_h4800_retention_ratio')}",
        f"- terminal autopsy decision: {route.get('terminal_autopsy_decision')}",
        f"- KANRoute: {route.get('KANRoute')}",
        f"- promotion_allowed: {route.get('promotion_allowed')}",
        f"- required artifact missing count: {route.get('required_artifact_missing_count')}",
        "",
        "## Part 1 Code Audit",
        "",
        md_table(code, ["check", "pass", "metric", "value", "blocker"], max_rows=20),
        "## Part 2 Basis Efficiency / D-RAT / D-RBF",
        "",
        md_table(eff, ["carrier", "S1_pass_rows", "full_loop_official_closure", "v22_03_same_kernel_functional_runner_proof", "v22_03_decision"], max_rows=20),
        md_table(dr, ["carrier", "profile_rows", "near_E1_rows", "micro_near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "gradcheck_pass_rows", "decision", "blocker"], max_rows=20),
        "### D-RAT / D-RBF Limited Same-Kernel Runner Integration",
        "",
        md_table(runner, ["carrier", "component_variant", "device", "runner_steps", "finite_loss_rows", "nonzero_grad_rows", "same_shape_rows", "loss_start", "loss_end", "loss_delta", "median_step_ms", "limited_runner_kernel_match", "official_fused_kernel_complete", "blocker"], max_rows=20),
        "### D-RAT / D-RBF Existing-Carrier Limited Functional Smoke",
        "",
        "- This smoke uses the existing `PrimitiveKAN` carrier path in `run_v21_01_source_retention.py`; it is not the active micro-kernel production fused path.",
        "- Scope: MNIST seed0, h1600, KSW1/KSW2/KSW9 plus matched controls. It is a blocker readback, not promotion evidence.",
        "",
        md_table(limited_smoke_route, ["carrier", "limited_smoke_scope", "candidate_rows", "candidate_h100_positive", "candidate_h400_positive", "candidate_h800_positive", "candidate_h1600_positive", "best_h800_v21_id", "best_h800", "best_h1600_for_best_h800", "best_positive_early_v21_id", "best_positive_early_h100", "best_positive_early_h400", "decision", "blocker"], max_rows=20),
        md_table(limited_smoke, ["carrier", "v21_id", "source_h100_mean", "source_h400_mean", "source_h800_mean", "source_h1600_mean", "source_h100_pass_count", "source_h400_pass_count", "source_h800_pass_count", "source_h1600_pass_count", "decision", "blocker", "uses_micro_kernel_path", "official_fused_kernel_complete", "promotion_allowed"], max_rows=20),
        "### D-RAT / D-RBF C4 Named FU Limited Smoke",
        "",
        "- This is the plan-named C4 smoke: RAT-FU1/FU2/FU3 and RBF-FU1/FU2/FU3 plus matched controls.",
        "- The FU names are existing source-retention aliases; D-RBF uses the available no-dense Triton RBF path, while D-RAT remains blocked by the absence of a production official fused rational path.",
        "",
        md_table(c4_smoke_route, ["carrier", "limited_smoke_scope", "candidate_rows", "candidate_h100_positive", "candidate_h400_positive", "candidate_h800_positive", "candidate_h1600_positive", "best_h800_v21_id", "best_h800", "best_h1600_for_best_h800", "best_positive_early_v21_id", "best_positive_early_h100", "best_positive_early_h400", "decision", "blocker"], max_rows=20),
        md_table(c4_smoke, ["carrier", "v21_id", "source_h100_mean", "source_h400_mean", "source_h800_mean", "source_h1600_mean", "source_h100_pass_count", "source_h400_pass_count", "source_h800_pass_count", "source_h1600_pass_count", "decision", "blocker", "uses_micro_kernel_path", "official_fused_kernel_complete", "promotion_allowed"], max_rows=24),
        "## Part 3 Terminal Erosion Autopsy",
        "",
        md_table(eros_classes, ["terminal_erosion_class", "groups", "fraction"], max_rows=20),
        md_table(eros, ["v22_id", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "terminal_erosion_class", "v22_03_blocker"], max_rows=24),
        "## Part 4 C1/C2/C1-Fallback Continuation Ledger",
        "",
        "- All rows below are rendered from fresh old-shape continuation CSV/JSON artifacts.",
        "- Productive h4800 gate is unchanged: mean h4800 >= 0.005, `R_4800/3200 >= 0.50`, and row-positive >= 7/9.",
        "",
        md_table(source_routes, ["continuation", "candidate_groups", "candidate_early_chain", "candidate_continuous_h3200", "candidate_h4800", "terminal_erosion_groups", "best_v22_id", "best_h4800", "best_h4800_retention_ratio", "decision"], max_rows=20),
        "### Best Candidate Ranking",
        "",
        md_table(best_src, ["continuation", "v22_id", "h100", "h400", "h800", "h1600", "h2400", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "source_chain_blocker"], max_rows=30),
        "## Part 5 C1 F145-F147 Source-Preservation Continuation",
        "",
        "- continuation source: `continuation_f145_f147_source_preserve_oldshape_full`",
        "- C1 repair family: source-preserve strong, source-preserve gentle, and information-volume guard around the F118-style early100 h800 slow-EMA path.",
        f"- best C1 candidate: `{c1_best.get('v22_id', '')}` h4800={c1_best.get('h4800', '')}, h4800_retention_ratio={c1_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c1_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker"], max_rows=20),
        "## Part 6 C2 F148-F150 Diffeomorphic Target Continuation",
        "",
        "- continuation source: `continuation_f148_f150_diffeomorphic_target_oldshape_full`",
        "- C2 repair family: low-NDS weak-stable target, information-volume split-consensus target, and low-rank readout transport target.",
        "- Diffeomorphic metrics are local train-stream logits displacement proxies recorded for audit; they are not promoted as exact Jacobian/Hessian proof.",
        f"- best C2 candidate: `{c2_best.get('v22_id', '')}` h4800={c2_best.get('h4800', '')}, h4800_retention_ratio={c2_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c2_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "### C2 Target-Gate Trace Evidence",
        "",
        md_table(target_trace, ["v22_id", "target_attempt_rows", "operator_accept_rows", "target_families", "operator_statuses", "mean_ActuationR2", "max_B2_transfer_gain", "max_B3_safety_gain", "max_jacobian_condition_mean", "max_fold_rate", "max_neighbor_order_flip_rate", "max_local_distance_distortion"], max_rows=20),
        "## Part 7 C1-Fallback F151-F153 Terminal-Preserve Repair",
        "",
        "- continuation source: `continuation_f151_f153_terminal_preserve_repair_oldshape_full`",
        "- Trigger: C1/C2 left h4800 positive but retention ratio below 0.50, so the plan directs increasing source preservation strength, trying h3600 terminal attach, and using row-orthogonal source updates rather than more ActuationR2-only target variants.",
        "- F151 increases terminal preservation strength without increasing h800 amplitude; F152 attaches source preservation at h3600; F153 tests a Nora-style row-orthogonal source update using train-stream source state only.",
        f"- best C1-fallback candidate: `{c3_best.get('v22_id', '')}` h4800={c3_best.get('h4800', '')}, h4800_retention_ratio={c3_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c3_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 8 C1-Fallback F154-F156 Debt / Low-NDS / Dual-Memory Repair",
        "",
        "- continuation source: `continuation_f154_f156_terminal_debt_lownds_dualmem_oldshape_full`",
        "- Trigger: F151-F153 worsened retention relative to the F118 near-miss, so this run tests the remaining C1 mechanisms rather than stronger preservation: debt-aware terminal preserve, low-NDS hidden/readout matrix-block source, and long-memory terminal source preservation.",
        f"- best C1-fallback-2 candidate: `{c4_best.get('v22_id', '')}` h4800={c4_best.get('h4800', '')}, h4800_retention_ratio={c4_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c4_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 9 F157-F159 Signal-Estimator Terminal Retention",
        "",
        "- continuation source: `continuation_f157_f159_signal_estimator_oldshape_full`",
        "- Trigger: F154-F156 still left terminal erosion below the 0.50 retention gate, so this run reopens PopRisk/SNR only as a train-stream signal estimator: terminal-retention predictor, split-consensus estimator, and signal/reservoir transport. It is not a direct parameter mask and does not use h4800 outcomes as a selector.",
        f"- best signal-estimator candidate: `{c5_best.get('v22_id', '')}` h4800={c5_best.get('h4800', '')}, h4800_retention_ratio={c5_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c5_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 10 F160-F162 Post-H4000 Source-Floor Terminal Retention",
        "",
        "- continuation source: `continuation_f160_f162_post_h4000_source_floor_oldshape_full`",
        "- Trigger: F157-F159 still left h4000->h4800 erosion below the 0.50 retention gate. This continuation tests source-floor, h4000-anchor floor, and decay-aware source floor using only train-stream source/support/corrupt telemetry; it is not a pure freeze/hold candidate and does not use h4800 outcomes as selector input.",
        f"- best post-h4000 source-floor candidate: `{c6_best.get('v22_id', '')}` h4800={c6_best.get('h4800', '')}, h4800_retention_ratio={c6_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c6_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 11 F163-F165 Raw-Guard Post-H4000 Source-Floor Retention",
        "",
        "- continuation source: `continuation_f163_f165_raw_guard_post_h4000_floor_oldshape_full`",
        "- Trigger: F160-F162 did not improve over F154 and remained far below the v22.02 F118 near miss. This continuation keeps the F118-style terminal raw guard before h4000, then switches only the post-h4000 terminal phase to source-floor, h4000-anchor floor, or decay-aware floor. It does not freeze/hold training and does not use h4800 outcomes as selector input.",
        f"- best raw-guard post-h4000 floor candidate: `{c7_best.get('v22_id', '')}` h4800={c7_best.get('h4800', '')}, h4800_retention_ratio={c7_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c7_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 12 F166-F168 Anti-Erosion Orthogonal / Reflection / Transport",
        "",
        "- continuation source: `continuation_f166_f168_anti_erosion_oldshape_full`",
        "- Trigger: F163-F165 still left post-h4000 erosion below the 0.50 retention gate. This continuation tests a different train-only hypothesis: late erosion is caused by train-stream gradient components pointing against the retained source channel, so the terminal route removes, reflects, or anchor-transports only the anti-source component after h4000. It does not use h4800 outcomes as selector input.",
        f"- best anti-erosion candidate: `{c8_best.get('v22_id', '')}` h4800={c8_best.get('h4800', '')}, h4800_retention_ratio={c8_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c8_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 13 F169-F171 H3200-Anchor Transport",
        "",
        "- continuation source: `continuation_f169_f171_h3200_anchor_transport_oldshape_full`",
        "- Trigger: F166-F168 showed the post-h4000 anti-source gradient component was mostly zero, so this continuation changes the hypothesis from anti-gradient removal to transporting the terminal route toward the train-stream h3200 source anchor. It still does not use h4800 outcomes as selector input.",
        f"- best h3200-anchor transport candidate: `{c9_best.get('v22_id', '')}` h4800={c9_best.get('h4800', '')}, h4800_retention_ratio={c9_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c9_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 14 F172-F174 Source-Preserving Progress Carry",
        "",
        "- continuation source: `continuation_f172_f174_progress_carry_oldshape_full`",
        "- Trigger: F154/F166/F169 decomposition showed the candidate terminal loss is nearly flat after h3200/h4000, while the matched `CTRL-SGD` best-control continues improving; the retained-source ratio therefore falls because the control baseline moves, not only because the source channel is actively erased. This continuation lets the terminal phase carry train-stream task progress while preserving the h3200/h4000 source anchor.",
        "- F172 uses h3200 progress carry, F173 uses h4000 progress carry, and F174 blends source-projected progress with the h3200 source anchor. None use h4800 outcomes as selector input.",
        f"- best progress-carry candidate: `{c10_best.get('v22_id', '')}` h4800={c10_best.get('h4800', '')}, h4800_retention_ratio={c10_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c10_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 15 F175-F177 Raw-Guard Gentle Progress",
        "",
        "- continuation source: `continuation_f175_f177_gentle_progress_oldshape_full`",
        "- Trigger: F172-F174 confirmed naive progress carry lowers source/validation while only slightly improving train loss. This continuation follows the plan fallback for h3200 source drop: attach later and weaker, keep raw-guard source geometry, and allow progress only through train-split lookahead-positive alt steps; non-alt terminal steps hold instead of running plain SGD.",
        "- F175 attaches gentle progress at h3600, F176 at h4000, and F177 at h4400 with projected progress. None use h4800 outcomes as selector input.",
        f"- best gentle-progress candidate: `{c11_best.get('v22_id', '')}` h4800={c11_best.get('h4800', '')}, h4800_retention_ratio={c11_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c11_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 16 F178-F180 Control-Relative Terminal Catch-Up",
        "",
        "- continuation source: `continuation_f178_f180_control_relative_catchup_oldshape_full`",
        "- Trigger: F172-F177 showed that source-preserving progress or late gentle progress still fails to catch the matched-control terminal improvement. This continuation tests a control-relative terminal objective using only train split A/B immediate loss gain and corrupt-label veto, not validation/test/future/h4800 outcome.",
        "- F178 uses SGD catch-up, F179 uses AdamW catch-up, and F180 uses a source-balanced catch-up direction. All three keep the unchanged productive h4800 gate.",
        f"- best control-relative candidate: `{c12_best.get('v22_id', '')}` h4800={c12_best.get('h4800', '')}, h4800_retention_ratio={c12_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c12_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 17 F181-F183 Trajectory-Adaptive Terminal Retention",
        "",
        "- continuation source: `continuation_f181_f183_trajectory_adaptive_oldshape_full`",
        "- Trigger: F154/F178 localization shows mixed failure modes: some rows lose too much between h1600 and h3200, while others keep continuous h3200 but fail the h4800/h3200 ratio. This continuation uses only reached train-stream source anchors (h1600/h2400/h3200/h4000), split A/B one-step gain, and corrupt-label veto to choose a mid-erosion bridge or post-h4000 ratio repair.",
        "- F181 uses trajectory-adaptive preserve, F182 emphasizes mid-erosion bridging before h3200, and F183 uses a two-phase mid bridge plus post-h4000 ratio repair. None read dataset name, validation/test/future, or h4800 outcome as selector input.",
        f"- best trajectory-adaptive candidate: `{c13_best.get('v22_id', '')}` h4800={c13_best.get('h4800', '')}, h4800_retention_ratio={c13_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c13_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 18 F184-F186 Minimal Terminal Transport / Hold",
        "",
        "- continuation source: `continuation_f184_f186_minimal_transport_oldshape_full`",
        "- Trigger: F181-F183 preserved early/h3200 but worsened post-h4000 retention relative to F154, suggesting terminal intervention was too disruptive rather than too weak. This continuation keeps the F154-style debt-aware preserve before h4000, then tests minimal train-stream post-h4000 transport.",
        "- F184 clips only the train-stream anti-source component, F185 holds post-h4000 unless debt or anti-source evidence requires a tiny clip, and F186 applies a very small h3200/h4000 anchor flow. None use dataset name, validation/test/future, or h4800 outcome as selector input.",
        f"- best minimal-transport candidate: `{c14_best.get('v22_id', '')}` h4800={c14_best.get('h4800', '')}, h4800_retention_ratio={c14_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c14_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 19 F187-F189 Terminal Accept-Memory",
        "",
        "- continuation source: `continuation_f187_f189_accept_memory_oldshape_full`",
        "- Trigger: F184-F186 fixed the runner wiring but the minimal post-h4000 transport worsened relative to F154, while F154 raw traces showed many successful train-stream terminal accepts before later noisy rejects. This continuation changes the retained-target theory: source observability may require persisting the last train-only accepted terminal transport instead of re-solving a noisy one-step target at every terminal horizon.",
        "- F187 replays the last accepted terminal transport if the current train split A/B and corrupt-label gate still permits it, F188 adds a debt-aware transport candidate, and F189 stores a source-progress transport memory. None read dataset name, validation/test/future, or h4800 outcome as selector input.",
        f"- best accept-memory candidate: `{c15_best.get('v22_id', '')}` h4800={c15_best.get('h4800', '')}, h4800_retention_ratio={c15_best.get('h4800_retention_ratio', '')}",
        "",
        md_table(c15_src, ["v22_id", "rows", "h800", "h3200", "h4000", "h4800", "h4800_retention_ratio", "row_h4800_positive_count", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_erosion_group", "terminal_erosion_class", "source_chain_blocker", "source_hidden_fraction_mean", "source_readout_fraction_mean", "source_norm_over_whitened_norm_mean", "source_projection_fold_proxy_mean", "diffeomorphic_audit_pass"], max_rows=20),
        "## Part 20 Dataset Localization",
        "",
        md_table(dataset, ["v22_id", "dataset", "rows", "mean_h800", "mean_h3200", "mean_h4000", "mean_h4800", "h4800_positive_rows", "early_rows", "continuous_rows", "h4800_rows", "blocker"], max_rows=36),
        "### Row Localization",
        "",
        md_table(rows, ["v22_id", "dataset", "seed", "source_h800", "source_h3200", "source_h4000", "source_h4800", "v22_03_early_source_chain", "v22_03_continuous_h3200_chain", "v22_03_productive_h4800_chain", "v22_03_terminal_erosion", "v22_03_source_chain_blocker"], max_rows=36),
        "## Part 21 KAN Source-Channel Writer",
        "",
        md_table(kan, ["carrier", "v22_id", "writer_family", "h800", "h3200", "h4800", "h4800_positive_rows", "KAN_FU_S3_v22_02", "v22_03_source_channel_writer_decision"], max_rows=24),
        "## 修改记录",
        "",
        "- 新增 `dgkan/fu/terminal_retention.py`：实现 v22.03 productive h4800 / terminal erosion gate 和 measured terminal-erosion taxonomy。",
        "- 新增 `dgkan/fu/diffeomorphic_target.py`：提供 source-channel information-volume / fold proxy audit helper；本轮只作为审计 helper，不把它当成功证据。",
        "- 扩展 `dgkan/fu/mechanisms.py`、`experiments/run_v21_common.py`、`experiments/run_v21_01_source_retention.py`、`experiments/run_v17_common.py`：新增 M170/F145 source-preserve-strong、M171/F146 source-preserve-gentle、M172/F147 info-volume guard，并接入 h4000 后 train-only source-preserving terminal route。",
        "- 继续按计划 C2 扩展 M173/F148、M174/F149、M175/F150 以及 target-only M176/M177/M178；在 `experiments/run_v17_common.py` 中记录 ActuationR2、B2/B3、local distortion、fold/order-flip 等 train-stream target-gate telemetry。",
        "- C2 仍未达成后，按计划 C1 fallback 新增 M179/F151 very-strong source preservation、M180/F152 h3600 terminal source preservation、M181/F153 Nora-style row-orthogonal source update；三者都保持 train-only gate，不使用 validation/test/future/dataset-name 分支。",
        "- F151-F153 仍未达成后，继续按计划 C1 剩余机制新增 M182/F154 debt-aware terminal preserve、M183/F155 low-NDS matrix-block source、M184/F156 dual-memory terminal preserve；不使用 h4800 outcome 训练 selector。",
        "- F154-F156 仍未达成后，按计划 3.1.1 允许重开 PopRisk/SNR 的受限语义，新增 M185/F157 terminal SNR predictor、M186/F158 split-consensus estimator、M187/F159 signal-reservoir transport；三者只把 train-stream signal/corrupt readback 作为 terminal gate/estimator，不作为直接参数 mask 或 h4800 outcome selector。",
        "- F157-F159 仍未达成后，基于 post-h4000 erosion 证据新增 M188/F160 terminal source floor、M189/F161 h4000-anchor source floor、M190/F162 decay-aware source floor；三者不做 pure freeze/hold，不读取 validation/test/future/h4800 outcome，只用 train-stream source/support/corrupt telemetry gate。",
        "- F160-F162 仍未改善后，回到 v22.02 最接近的 F118 raw-guard 证据链，新增 M191/F163 raw-guard source floor、M192/F164 raw-guard h4000-anchor floor、M193/F165 raw-guard decay-aware floor：h4000 前沿用 raw guard，h4000 后才切到 train-stream source floor。",
        "- F163-F165 仍未达到 productive h4800 后，新增 M194/F166 anti-erosion orthogonal、M195/F167 source-reflection guard、M196/F168 h4000 transport corrector：只在 h4000 后对 train-stream anti-source gradient component 做移除/反射/anchor transport，不读取 h4800 outcome。",
        "- F166-F168 仍未达到 productive h4800 且 anti-source component 基本为 0 后，新增 M197/F169 h3200-anchor transport、M198/F170 raw-guard+h3200-anchor transport、M199/F171 h3200-ratio reentry：用 train-stream h3200 source anchor 作为 post-h4000 transport target，不读取 h4800 outcome。",
        "- F169-F171 仍未达到 productive h4800 后，对 F154/F166/F169 做 terminal source/control decomposition，发现 h3200/h4000 之后 candidate 自身 loss 近乎持平，而 `CTRL-SGD` best-control 继续改善；据此新增 M200/F172 h3200 progress carry、M201/F173 h4000 progress carry、M202/F174 h3200 source-progress blend：在保留 source anchor 的同时允许 train-stream task-progress update 继续推进。",
        "- F172-F174 仍未达到 productive h4800，且 F172 只改善 train loss、不改善 validation/source 后，新增 M203/F175 raw-guard h3600 gentle progress、M204/F176 raw-guard h4000 gentle progress、M205/F177 raw-guard h4400 projected progress：按计划 fallback 将 terminal attach 变晚/变轻，非 alt terminal step 不再普通 SGD。",
        "- F175-F177 仍未达到 productive h4800 后，新增 M206/F178 control-relative SGD catch-up、M207/F179 control-relative AdamW catch-up、M208/F180 source-balanced catch-up：只用 train split A/B one-step gain 与 corrupt-label veto 判断 terminal update 是否可提交，不读取 h4800 outcome。",
        "- F178-F180 仍未达到 productive h4800 后，新增 M209/F181 trajectory-adaptive preserve、M210/F182 mid-erosion bridge、M211/F183 two-phase ratio repair：按已到达的 train-stream source anchor（h1600/h2400/h3200/h4000）选择 mid-erosion bridge 或 post-h4000 ratio repair，不读取 dataset-name、validation/test/future 或 h4800 outcome。",
        "- F181-F183 full 显示 trajectory-adaptive terminal route 比 F154 更弱后，新增 M212/F184 anti-source clip、M213/F185 debt-aware hold、M214/F186 anchor-flow-tiny：h4000 前沿用 F154-style debt-aware preserve，h4000 后只做极小 train-stream anti-source clipping/anchor flow 或 hold，不读取 h4800 outcome。",
        "- F184-F186 smoke 暴露两处 runner wiring 问题后，修复 M212/M213/M214 在 `experiments/run_v17_common.py` 的 source-retention 主 allowlist 与 early active-source mode 列表；两次 pre-wiring smoke 分别保留为 `continuation_f184_f186_minimal_transport_smoke_pre_front_wiring_fix` 与 `continuation_f184_f186_minimal_transport_smoke_pre_runner_wiring_fix`，不作为科学成功证据。",
        "- F184-F186 仍未达到 productive h4800 后，基于 F154/F181/F186 raw trace 诊断新增 M215/F187 accept-memory、M216/F188 accept-memory-debt、M217/F189 source-progress-memory：只持久化 train split A/B 与 corrupt-label gate 接受过的 terminal transport，h4000 后 replay 仍需当前 train-only gate 通过，不读取 dataset-name、validation/test/future 或 h4800 outcome。",
        "- 新增 v22.03 runner：code truth gate、efficiency readback、D-RAT/D-RBF active micro-kernel repair、terminal erosion autopsy、source-preservation aggregation、KAN readback、finalizer。",
        "- 将 `experiments/run_v22_03_drat_drbf_repair.py` 从 readback-only 扩展为 active micro-kernel benchmark：覆盖 D-RAT Horner/reciprocal/telemetry-free/low-degree rational 和 D-RBF local-k/no-dense/active-center/exp-approx/sparse-backward 方向；输出 component waterfall、gradcheck、finite-rate、memory/materialization 和 micro-near-E1 判定，但不把 micro-kernel 结果伪称为 official fused 或 functional runner 证据。",
        "- 新增 `experiments/run_v22_03_drat_drbf_runner_integration.py`：对 D-RAT/D-RBF 的 micro-near-E1 变体执行 limited same-kernel runner integration，验证 CUDA train loop 中有限 loss、非零梯度、shape 一致、无 CPU offload 和同 callable hash；该证据只移除 functional_runner_kernel_mismatch，不把 `official_fused_kernel_complete=0` 伪称为 promotion。",
        "- 新增 `experiments/run_v22_03_drat_drbf_limited_smoke_summary.py`：汇总 D-RAT/D-RBF existing-carrier limited functional smoke；该 smoke 只读 `run_v21_01_source_retention.py` 的现有 PrimitiveKAN carrier 路径，不把结果伪称为 active micro-kernel production runner 或 official fused kernel。",
        "- 扩展 `experiments/run_v21_common.py` / `experiments/run_v17_common.py`：注册计划指定的 RAT-FU1/FU2/FU3 与 RBF-FU1/FU2/FU3 C4 smoke aliases，并将 RBF22.03 compact/active-center repair label 映射到已有 `rbf_k*_triton_l3_matmul` no-dense path；D-RAT 仍不伪称 production official fused rational path 已接入。",
        "- 执行 F145-F147 smoke 后，运行 4GPU old-shape full：train_size=512、val_size=256、batch_size=64、steps=6400、MNIST/Fashion-MNIST/KMNIST x seed0/1/2、F145-F147 + matched controls。",
        "- F145-F147 没有达到 productive h4800 后，继续执行 F148-F150 h1600/h4200 smoke 和 4GPU old-shape full；所有数值来自落盘 `v21_01_source_retention_matrix.csv` 与 raw trace，不手写提升。",
        "- F148-F150 仍没有达到 productive h4800 后，继续执行 F151-F153 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F151-F153 仍没有达到 productive h4800 后，继续执行 F154-F156 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F154-F156 仍没有达到 productive h4800 后，继续执行 F157-F159 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F157-F159 仍没有达到 productive h4800 后，继续执行 F160-F162 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F160-F162 仍没有达到 productive h4800 后，继续执行 F163-F165 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F163-F165 仍没有达到 productive h4800 后，继续执行 F166-F168 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F166-F168 仍没有达到 productive h4800 后，继续执行 F169-F171 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F169-F171 仍没有达到 productive h4800 后，继续执行 F172-F174 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F172-F174 仍没有达到 productive h4800 后，继续执行 F175-F177 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F175-F177 仍没有达到 productive h4800 后，继续执行 F178-F180 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F178-F180 仍没有达到 productive h4800 后，继续执行 F181-F183 smoke 和 4GPU old-shape full；所有数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "- F181-F183 仍没有达到 productive h4800 且低于 F154 后，继续执行 F184-F186 smoke；前两次 smoke 暴露 wiring 缺口并已留档，修复后 smoke 恢复 early source chain，再执行 4GPU old-shape full；所有正式数值来自修复后的 fresh matrix 与 raw trace。",
        "- F184-F186 仍没有达到 productive h4800 后，继续执行 F187-F189 accept-memory smoke 和 4GPU old-shape full；所有正式数值来自该 continuation 的 fresh matrix 与 raw trace。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- v22.03 重新表述了 blocker：F118 之后的最好状态不是 h4800 变负的 terminal collapse，而是 h4800 仍为正但 retained ratio 未达 0.50 的 terminal erosion。",
        "- C0 autopsy 只基于 measured h3200/h4000/h4800 trajectory 与 row-positive 证据分类；非因果指标不会被写成机制证明。",
        "- F145-F147 是按计划 C1 的 source-preserving terminal route 做的真实修复尝试：不使用 validation/test/future，不使用 dataset-name branch，不修改 loss/sampler/class weight。",
        f"- C1 best 为 `{c1_best.get('v22_id', '')}`，h4800_retention_ratio={c1_best.get('h4800_retention_ratio', '')}；它保留 early/h3200 chain，但仍是 terminal erosion。",
        "- 因 C1 未达 productive h4800，继续按计划执行 C2 diffeomorphic/source-channel target reset；C2 target telemetry 显示 F148/F150 的 target gate 全部 reject，F149 仅 1/54 次 accept，且该 accept 没有转化为 productive h4800，所以 high local actuation/readout observability 不能写成 retention success。",
        f"- C2 best 为 `{c2_best.get('v22_id', '')}`，h4800_retention_ratio={c2_best.get('h4800_retention_ratio', '')}；它略高于 C1 best，但仍低于 0.50 gate，也低于 v22.02 F118 near-miss ratio=0.4905266741203586。",
        f"- C1 fallback best 为 `{c3_best.get('v22_id', '')}`，h4800_retention_ratio={c3_best.get('h4800_retention_ratio', '')}；它检验 preservation strength、h3600 attach 与 row-orthogonal source update 是否能修复 C1/C2 的 terminal erosion。",
        f"- C1 fallback-2 best 为 `{c4_best.get('v22_id', '')}`，h4800_retention_ratio={c4_best.get('h4800_retention_ratio', '')}；它检验 debt-aware、low-NDS matrix-block 与 dual-memory source state 是否能修复 post-h4000 erosion。",
        f"- Signal-estimator best 为 `{c5_best.get('v22_id', '')}`，h4800_retention_ratio={c5_best.get('h4800_retention_ratio', '')}；它检验 PopRisk/SNR 是否能作为 train-only terminal-retention estimator 改善 source transport，而不是恢复旧的 one-shot parameter selector。",
        f"- Post-h4000 source-floor best 为 `{c6_best.get('v22_id', '')}`，h4800_retention_ratio={c6_best.get('h4800_retention_ratio', '')}；它检验 terminal erosion 是否能被低幅度 source floor / h4000 anchor floor 修复，同时避免 pure freeze/hold 造成的假保留。",
        f"- Raw-guard post-h4000 floor best 为 `{c7_best.get('v22_id', '')}`，h4800_retention_ratio={c7_best.get('h4800_retention_ratio', '')}；它检验 F118-style raw guard 近失误点是否能通过 h4000 后 source-floor repair 跨过 productive gate。",
        f"- Anti-erosion best 为 `{c8_best.get('v22_id', '')}`，h4800_retention_ratio={c8_best.get('h4800_retention_ratio', '')}；它检验后 h4000 anti-source gradient 是否是主要 erosion source，而不是继续调 source floor 强度。",
        f"- H3200-anchor transport best 为 `{c9_best.get('v22_id', '')}`，h4800_retention_ratio={c9_best.get('h4800_retention_ratio', '')}；它检验 h3200 时刻的 train-stream source geometry 是否比 h4000 anchor 更适合作为 terminal transport target。",
        f"- Progress-carry best 为 `{c10_best.get('v22_id', '')}`，h4800_retention_ratio={c10_best.get('h4800_retention_ratio', '')}；它检验 source-retention route 是否需要同时携带 train-stream task progress，避免 candidate flat 而 matched control 继续改善导致 retained-ratio 被动下滑。",
        f"- Gentle-progress best 为 `{c11_best.get('v22_id', '')}`，h4800_retention_ratio={c11_best.get('h4800_retention_ratio', '')}；它检验 F172 的失败是否来自 terminal progress 过早/过强，以及非 alt SGD 是否会破坏 retained source。",
        f"- Control-relative catch-up best 为 `{c12_best.get('v22_id', '')}`，h4800_retention_ratio={c12_best.get('h4800_retention_ratio', '')}；它检验 terminal objective 是否应显式追赶 matched-control 的 train-stream task progress，同时用 corrupt-label veto 避免伪进展。",
        f"- Trajectory-adaptive best 为 `{c13_best.get('v22_id', '')}`，h4800_retention_ratio={c13_best.get('h4800_retention_ratio', '')}；它检验同一 continuation 内的 row-state 异质性是否需要 train-only horizon-anchor 分流，而不是单一后段 source/SGD 强度。",
        f"- Minimal-transport best 为 `{c14_best.get('v22_id', '')}`，h4800_retention_ratio={c14_best.get('h4800_retention_ratio', '')}；它检验 F181 失败是否来自 terminal route 过强/扰动过多，改为 h4000 后 minimal clip/hold/tiny anchor flow。",
        f"- Accept-memory best 为 `{c15_best.get('v22_id', '')}`，h4800_retention_ratio={c15_best.get('h4800_retention_ratio', '')}；它检验 F154 raw trace 暗示的 last-good accepted terminal transport 是否比每个 terminal horizon 重新求 noisy one-step target 更稳定。",
        "- F151-F189 已覆盖计划内的 C1/C2 fallback 与后续解释性修复：preservation strength、h3600 attach、row-orthogonal、debt-aware、low-NDS matrix-block、dual-memory、SNR estimator、source-floor、anti-erosion、h3200 transport、progress carry、gentle progress、control-relative catch-up、trajectory-adaptive horizon-anchor 分流、minimal anti-source clipping、post-h4000 hold/tiny anchor flow 与 train-only accepted terminal transport memory；若仍全部 productive_h4800_group=0，则继续同族小修缺少 plan-backed 因果依据，下一步需要新的 train-only retained-target/source-observability 定义，而不是再调 source amplitude、terminal hold 或 replay 强度。",
        f"- 本轮 overall best candidate 为 `{best.get('v22_id', '')}`，productive_h4800_group={best.get('productive_h4800_group', '')}；因此不能 promotion，也不能启动 independent confirmation。",
        f"- D-RAT active repair 结果为 `{drat_repair.get('decision', '')}`：micro_near_E1_rows={drat_repair.get('micro_near_E1_rows', '')}/{drat_repair.get('profile_rows', '')}，best_forward_ratio={drat_repair.get('best_forward_ratio', '')}，best_step_ratio={drat_repair.get('best_step_ratio', '')}；limited same-kernel runner 已通过，但 blocker 仍是 official_fused_missing。",
        f"- D-RBF active repair 结果为 `{drbf_repair.get('decision', '')}`：micro_near_E1_rows={drbf_repair.get('micro_near_E1_rows', '')}/{drbf_repair.get('profile_rows', '')}，best_forward_ratio={drbf_repair.get('best_forward_ratio', '')}，best_step_ratio={drbf_repair.get('best_step_ratio', '')}；active-center/local-k 方向降低了 materialization，limited same-kernel runner 已通过，但 blocker 仍是 official_fused_missing。",
        f"- D-RAT existing-carrier limited smoke 结果为 `{drat_smoke.get('decision', '')}`：best_h800_v21_id=`{drat_smoke.get('best_h800_v21_id', '')}`，best_h800={drat_smoke.get('best_h800', '')}，best_h1600={drat_smoke.get('best_h1600_for_best_h800', '')}；best positive early row=`{drat_smoke.get('best_positive_early_v21_id', '')}`，h100={drat_smoke.get('best_positive_early_h100', '')}，h400={drat_smoke.get('best_positive_early_h400', '')}。该 smoke 没有打开 h800 source，不能进入 functional promotion。",
        f"- D-RBF existing-carrier limited smoke 结果为 `{drbf_smoke.get('decision', '')}`：best_h800_v21_id=`{drbf_smoke.get('best_h800_v21_id', '')}`，best_h800={drbf_smoke.get('best_h800', '')}，best_h1600={drbf_smoke.get('best_h1600_for_best_h800', '')}；best positive early row=`{drbf_smoke.get('best_positive_early_v21_id', '')}`，h100={drbf_smoke.get('best_positive_early_h100', '')}，h400={drbf_smoke.get('best_positive_early_h400', '')}。这说明 micro-near-E1 进展尚未迁移到现有 KAN source writer 路径。",
        f"- C4 named D-RAT smoke 结果为 `{drat_c4_smoke.get('decision', '')}`：best_h800_v21_id=`{drat_c4_smoke.get('best_h800_v21_id', '')}`，best_h800={drat_c4_smoke.get('best_h800', '')}，best_h1600={drat_c4_smoke.get('best_h1600_for_best_h800', '')}；其 blocker 仍包含 official fused rational path 未接入 functional runner。",
        f"- C4 named D-RBF smoke 结果为 `{drbf_c4_smoke.get('decision', '')}`：best_h800_v21_id=`{drbf_c4_smoke.get('best_h800_v21_id', '')}`，best_h800={drbf_c4_smoke.get('best_h800', '')}，best_h1600={drbf_c4_smoke.get('best_h1600_for_best_h800', '')}；即使映射到已有 no-dense Triton RBF path，也没有达到 h800 source gate 时不能 promotion。",
        "- D-CHE/D-FOU efficiency 继续作为 official efficient carrier 证据；D-RAT/D-RBF active micro-kernel repair 和 limited runner integration 有进展，但 near_E1_rows official 仍为 0 且 production official fused kernel 仍缺失，所以不能做 functional smoke promotion。",
        "- KAN source-channel writer 本轮为 readback evidence；若 KAN_FU_S3 仍为 0，说明 MLP source-channel 形状尚未能可靠映射到 KAN writer。",
        "- 证据链的核心失败模式是：h4800 大多为正、row-positive 已经足够，但 h4000->h4800 持续衰减让 `R_4800/3200` 停在 0.42 左右；这不是 threshold tuning 能解决的 selector 问题，而是 retained target/source observability 仍不足。",
        "",
        "## Artifact Index",
        "",
        md_table(artifacts, ["artifact", "exists", "size_bytes", "sha256"], max_rows=180),
        "",
        f"- v22_03_code_review_packet.zip: size={zip_path.stat().st_size if zip_path.exists() else ''} sha256={sha256_file(zip_path) if zip_path.exists() else ''}",
        f"- v22_03_results_bundle.zip: size={bundle.stat().st_size if bundle.exists() else ''} sha256={sha256_file(bundle) if bundle.exists() else ''}",
        "",
    ]
    write_text(V2203_RECAP_DOC, "\n".join(text))


def clean_unzip_self_test(out_dir: Path, zip_path: Path) -> None:
    extract_dir = out_dir / "v22_03_packet_clean_unzip"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    code, log = run_cmd([PYTHON, "-m", "compileall", "-q", "."], cwd=extract_dir, timeout=300)
    (out_dir / "v22_03_clean_unzip_self_test.log").write_text(log, encoding="utf-8")
    write_rows(out_dir / "v22_03_clean_unzip_self_test.csv", [{"compileall_exit": code, "pass": int(code == 0)}])
    append_exec(out_dir, f"{PYTHON} -m compileall -q .  # inside extracted v22_03_code_review_packet.zip", status="completed" if code == 0 else "failed", note=f"clean unzip self-test exit={code}")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    write_queue_artifacts(out_dir)
    route = route_from(out_dir)
    write_json(out_dir / "v22_03_route_decision.json", route)
    write_rows(out_dir / "v22_03_route_decision.csv", [route])
    zip_path, bundle = build_packet(out_dir)
    clean_unzip_self_test(out_dir, zip_path)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_03_finalize.py --out-dir {out_dir}", status="completed", note=f"route={route['route']}")
    zip_path, bundle = build_packet(out_dir)
    write_recap(out_dir, route, zip_path, bundle)


if __name__ == "__main__":
    main()
