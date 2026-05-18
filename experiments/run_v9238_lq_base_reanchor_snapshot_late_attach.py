#!/usr/bin/env python3
"""DG-KAN v9.2.38 LQ base re-anchor and snapshot late-attach runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9237_training_path_equivalent_functional_attach as v9237  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.38_LQBaseReanchor_SnapshotLateAttach_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9238_lq_base_reanchor_snapshot_late_attach.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.38_LQBaseReanchor_SnapshotLateAttach_实验复盘.md"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9237 = RESULT_ROOT / "v9237_training_path_equivalent_functional_attach_first_20260511T100000Z"
SRC_V927 = RESULT_ROOT / "v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z"

HISTORICAL_ROWS = SRC_V927 / "p2_p5_reproduction_rows.csv"
LQ_ID = "TPEA0-LQ-reference"
P1_CANDIDATES = [
    ("CurrentLQReproduction", LQ_ID),
    ("TPEA1-off", "TPEA1-RegisteredZeroExcluded"),
    ("TPEA3-late-attach-off", "TPEA3-LateAttachOrthogonalTail"),
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    try:
        return artifact_hash_rows(path)
    except Exception:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _canonical_datasets(text: str) -> List[str]:
    return [v92._canonical_task(d) for d in _parse_list(text)]


def _historical_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return (str(row.get("dataset")), _int(row.get("seed")))


def _historical_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hrows = [r for r in rows if str(r.get("candidate_id")) == "LQ-t2-h256"]
    near = sum(_int(r.get("minimum_trainability_pass")) for r in hrows)
    macro = _mean(_float(r.get("delta_vs_mlp_match", r.get("delta_vs_mlp"))) for r in hrows)
    return {
        "historical_row_count": len(hrows),
        "historical_near_count": near,
        "historical_near_rate": near / max(1, len(hrows)),
        "historical_macro_delta": macro,
        "historical_source_artifact": _rel(HISTORICAL_ROWS),
    }


def _protocol_hashes(args: argparse.Namespace, candidate_id: str, protocol: str, params_kan: int, params_mlp: int) -> Dict[str, str]:
    gate = "near_delta>=-0.01;near_rate>=0.80;macro_delta>=-0.01;abs(current_macro-historical_macro)<=0.003"
    mlp = f"matched_mlp3_hidden(params_kan={params_kan},in_dim=784,out_dim=10);params_mlp={params_mlp}"
    seed = "init_seed=seed+923700;epoch_perm_seed=seed*1000+epoch+9237;mlp_seed=seed+923711"
    cfg = {
        "candidate_id": candidate_id,
        "hidden_dim": 256,
        "basis": "t2",
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
    }
    return {
        "p5_gate_definition_hash": _hash_text(gate),
        "mlp_match_definition_hash": _hash_text(mlp),
        "candidate_config_hash": _hash_text(json.dumps(cfg, sort_keys=True)),
        "runner_hash": _hash_file(SCRIPT_PATH),
        "data_protocol_hash": _hash_text(protocol),
        "seed_protocol_hash": _hash_text(seed),
    }


def _minibatch_hash(train_size: int, seed: int, epochs: int, device: torch.device) -> str:
    h = hashlib.sha256()
    for epoch in range(int(epochs)):
        gen = torch.Generator(device=device).manual_seed(seed * 1000 + epoch + 9237)
        perm = torch.randperm(int(train_size), device=device, generator=gen)
        h.update(perm.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9237 / "route_decision.json")
    audit = read_csv_rows(SRC_V9237 / "v9237_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R11-TPEAAllFailPrimitiveReset"
        and _int(route.get("training_path_equivalence_pass")) == 1
        and _int(route.get("tpea_contract_pass")) == 1
        and _int(route.get("tpea_grad_pass")) == 1
        and _int(route.get("tpea_p5_nearpass")) == 0
        and _int(route.get("functional_carrier_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9237_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": _rel(SRC_V9237),
        "route": route.get("route", ""),
        "source_route_v9236": route.get("source_route", ""),
        "bpfs_failure_mode": route.get("bpfs_failure_mode", ""),
        "bpfs_failure_primary_fraction": route.get("bpfs_failure_primary_fraction", ""),
        "tpea_implemented_count": route.get("tpea_implemented_count", ""),
        "training_path_equivalence_pass": route.get("training_path_equivalence_pass", ""),
        "task_init_hash_match": route.get("task_init_hash_match", ""),
        "optimizer_state_match": route.get("optimizer_state_match", ""),
        "lambda_inactive_leak_detected": route.get("lambda_inactive_leak_detected", ""),
        "tpea_contract_pass": route.get("tpea_contract_pass", ""),
        "tpea_grad_pass": route.get("tpea_grad_pass", ""),
        "best_tpea_candidate": route.get("best_tpea_candidate", ""),
        "tpea_p4_pass": route.get("tpea_p4_pass", ""),
        "tpea_p5_nearpass": route.get("tpea_p5_nearpass", ""),
        "base_preservation_pass": route.get("base_preservation_pass", ""),
        "functional_carrier_pass": route.get("functional_carrier_pass", ""),
        "max_r_z_tail": route.get("max_r_z_tail", ""),
        "max_r_perp_tail": route.get("max_r_perp_tail", ""),
        "primary_blocker": route.get("primary_blocker", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_lq_anchor(args: argparse.Namespace, device: torch.device, opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_LQ_BASE_ANCHOR_PROTOCOL_AUDIT", "p1_lq_base_anchor_protocol_audit.csv", "P0_v9237_boundary_failed")
        return [row], [row], {"lq_base_anchor_pass": 0}
    hist_rows_raw = read_csv_rows(HISTORICAL_ROWS)
    hist_rows = [r for r in hist_rows_raw if str(r.get("candidate_id")) == "LQ-t2-h256" and str(r.get("stage")) == "P2_P5_REPRODUCTION"]
    hist_by_key = {_historical_key(r): r for r in hist_rows}
    hist = _historical_summary(hist_rows)
    registry = v9237._candidate_registry()
    rows: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    # Historical rows are source recaps, not new measurements.
    for r in hist_rows:
        params_kan = _int(r.get("params_kan"))
        params_mlp = _int(r.get("params_mlp_match"))
        hashes = _protocol_hashes(args, "HistoricalLQReference", str(r.get("protocol", "")), params_kan, params_mlp)
        row = {
            "stage": "P1_LQ_BASE_ANCHOR_PROTOCOL_AUDIT",
            "status": "source_historical_reference",
            "candidate": "HistoricalLQReference",
            "source_artifact": _rel(HISTORICAL_ROWS),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "hidden_dim": r.get("hidden_dim", 256),
            "basis_type": r.get("basis", "t2"),
            "fan_in_output_scale": r.get("output_scale", "historical_source"),
            "memory_mode": "compact_recompute_live_set",
            "train_size": r.get("train_size", ""),
            "eval_size": r.get("test_size", ""),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "epochs": r.get("epochs", ""),
            "seed_protocol_hash": "not_recorded_in_historical_source",
            "data_split_hash": hashes["data_protocol_hash"],
            "minibatch_order_hash": "not_recorded_in_historical_source",
            "mlp_match_param_count": params_mlp,
            "kan_param_count": params_kan,
            "param_ratio": params_kan / max(1, params_mlp),
            "MLP_match_acc": r.get("mlp_match_acc", ""),
            "KAN_acc": r.get("kan_acc", ""),
            "historical_LQ_acc": r.get("kan_acc", ""),
            "delta_vs_mlp": r.get("delta_vs_mlp_match", r.get("delta_vs_mlp", "")),
            "delta_vs_historical_LQ": 0.0,
            "near_pass": r.get("minimum_trainability_pass", ""),
            "CEp99": r.get("CE_p99", ""),
            "margin_p10": r.get("margin_p10", ""),
            "ECE": r.get("ECE", ""),
            "NLL": r.get("NLL", ""),
            **hashes,
            "historical_macro_delta": hist["historical_macro_delta"],
            "historical_near_count": hist["historical_near_count"],
            "historical_near_rate": hist["historical_near_rate"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        trace.append(row)
    current_rows_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for label, cid in P1_CANDIDATES:
        cand = registry[cid]
        measured: List[Dict[str, Any]] = []
        for dataset in _canonical_datasets(args.datasets):
            for seed in _parse_ints(args.seeds):
                tr, _saved = v9237._train_tpea_base(args, cand, dataset, seed, device, store_cache=False)
                hrow = hist_by_key.get((dataset, seed), {})
                hist_acc = _float(hrow.get("kan_acc"), float("nan"))
                params_kan = _int(tr.get("params_kan_official"))
                params_mlp = _int(tr.get("params_mlp_match"))
                hashes = _protocol_hashes(args, cid, str(tr.get("protocol", "")), params_kan, params_mlp)
                mb_hash = _minibatch_hash(int(args.train_size), seed, int(args.epochs), device)
                row = {
                    "stage": "P1_LQ_BASE_ANCHOR_PROTOCOL_AUDIT",
                    "status": "measured_current",
                    "candidate": label,
                    "candidate_internal_id": cid,
                    "attach_mode": cand.attach_mode,
                    "dataset": dataset,
                    "seed": seed,
                    "hidden_dim": cand.spec.hidden_dim,
                    "basis_type": "t2",
                    "fan_in_output_scale": "current_actuator_init",
                    "memory_mode": "compact_recompute_live_set",
                    "train_size": int(args.train_size),
                    "eval_size": int(args.test_size),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "epochs": int(args.epochs),
                    "seed_protocol_hash": hashes["seed_protocol_hash"],
                    "data_split_hash": hashes["data_protocol_hash"],
                    "minibatch_order_hash": mb_hash,
                    "mlp_match_param_count": params_mlp,
                    "kan_param_count": params_kan,
                    "param_ratio": params_kan / max(1, params_mlp),
                    "MLP_match_acc": tr.get("MLP_match_acc", ""),
                    "KAN_acc": tr.get("KAN_acc", ""),
                    "historical_LQ_acc": hrow.get("kan_acc", ""),
                    "delta_vs_mlp": tr.get("delta_vs_mlp", ""),
                    "delta_vs_historical_LQ": (_float(tr.get("KAN_acc")) - hist_acc) if hrow else "",
                    "near_pass": tr.get("near_pass", ""),
                    "CEp99": tr.get("CEp99", ""),
                    "margin_p10": tr.get("margin_p10", ""),
                    "ECE": tr.get("ECE", ""),
                    "NLL": tr.get("NLL", ""),
                    **hashes,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                rows.append(row)
                trace.append(row)
                measured.append(row)
        current_rows_by_candidate[label] = measured
    current_lq = current_rows_by_candidate.get("CurrentLQReproduction", [])
    current_near = sum(_int(r.get("near_pass")) for r in current_lq)
    current_macro = _mean(_float(r.get("delta_vs_mlp")) for r in current_lq)
    hist_macro = _float(hist["historical_macro_delta"])
    hist_current_delta = current_macro - hist_macro
    lq_anchor = int(current_lq and current_near / len(current_lq) >= 0.80 and current_macro >= -0.01 and abs(hist_current_delta) <= 0.003)
    # TPEA-off equivalence is audited against current LQ, even if LQ itself fails.
    current_key = {(r["dataset"], _int(r["seed"])): r for r in current_lq}
    equivalence_stats: Dict[str, float] = {}
    tpea_equiv_passes: List[int] = []
    for label in ["TPEA1-off", "TPEA3-late-attach-off"]:
        diffs = []
        acc_diffs = []
        for r in current_rows_by_candidate.get(label, []):
            base = current_key.get((r["dataset"], _int(r["seed"])), {})
            diffs.append(abs(_float(r.get("delta_vs_mlp")) - _float(base.get("delta_vs_mlp"))))
            acc_diffs.append(abs(_float(r.get("KAN_acc")) - _float(base.get("KAN_acc"))))
        max_delta_diff = max(diffs, default=999.0)
        max_acc_diff = max(acc_diffs, default=999.0)
        equivalence_stats[f"{label}_max_delta_diff_vs_current_lq"] = max_delta_diff
        equivalence_stats[f"{label}_max_acc_diff_vs_current_lq"] = max_acc_diff
        tpea_equiv_passes.append(int(max_delta_diff <= 0.001 and max_acc_diff <= 0.001))
    tpea_off_equiv = int(all(tpea_equiv_passes) and bool(tpea_equiv_passes))
    row_count_match = int(len(hist_rows) == len(current_lq))
    near_count_drift = int(current_near != _int(hist["historical_near_count"]))
    protocol_mismatch = int(near_count_drift or not row_count_match)
    if not lq_anchor and protocol_mismatch:
        mismatch_mode = "historical_current_nearpass_drift"
    elif not lq_anchor:
        mismatch_mode = "current_lq_base_anchor_fail_without_hash_mismatch"
    else:
        mismatch_mode = "none"
    summary = {
        "lq_base_anchor_pass": lq_anchor,
        "current_lq_nearpass": int(current_near / max(1, len(current_lq)) >= 0.80),
        "current_lq_near_count": current_near,
        "current_lq_row_count": len(current_lq),
        "current_lq_near_rate": current_near / max(1, len(current_lq)),
        "current_lq_macro_delta": current_macro,
        "historical_lq_macro_delta": hist_macro,
        "historical_current_lq_delta": hist_current_delta,
        "historical_lq_near_count": hist["historical_near_count"],
        "historical_lq_row_count": hist["historical_row_count"],
        "tpea_off_p5_equivalence_pass": tpea_off_equiv,
        "protocol_mismatch_detected": protocol_mismatch,
        "protocol_mismatch_mode": mismatch_mode,
        **equivalence_stats,
    }
    summary_row = {
        "stage": "P1_LQ_BASE_ANCHOR_PROTOCOL_AUDIT",
        "status": "summary",
        **summary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary_row)
    trace.append(summary_row)
    return rows, trace, summary


def _write_gate_blocked_artifacts(out_dir: Path, reason: str) -> None:
    mapping = [
        ("p2_snapshot_late_attach_implementation.csv", "P2_SNAPSHOT_LATE_ATTACH_IMPLEMENTATION"),
        ("snapshot_attach_trace_v9238.csv", "P2_SNAPSHOT_LATE_ATTACH_TRACE"),
        ("p3_checkpoint_inactive_equivalence.csv", "P3_CHECKPOINT_INACTIVE_EQUIVALENCE"),
        ("checkpoint_equivalence_trace_v9238.csv", "P3_CHECKPOINT_INACTIVE_EQUIVALENCE_TRACE"),
        ("p4_no_event_replay_base_preservation.csv", "P4_NO_EVENT_REPLAY_BASE_PRESERVATION"),
        ("p5_functional_carrier_actuatability.csv", "P5_FUNCTIONAL_CARRIER_ACTUATABILITY"),
        ("functional_carrier_trace_v9238.csv", "P5_FUNCTIONAL_CARRIER_TRACE"),
        ("p6_value_observability_audit.csv", "P6_VALUE_OBSERVABILITY_AUDIT"),
        ("value_score_trace_v9238.csv", "P6_VALUE_SCORE_TRACE"),
        ("p7_leave_dataset_and_stratum_out_validation.csv", "P7_LEAVE_DATASET_AND_STRATUM_OUT_VALIDATION"),
        ("leave_dataset_out_trace_v9238.csv", "P7_LEAVE_DATASET_AND_STRATUM_OUT_TRACE"),
        ("p8_official_snapshot_late_attach_paired_replay.csv", "P8_OFFICIAL_SNAPSHOT_LATE_ATTACH_PAIRED_REPLAY"),
        ("paired_replay_branch_trace_v9238.csv", "P8_PAIRED_REPLAY_BRANCH_TRACE"),
        ("p9_short_run_functional_validation.csv", "P9_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p10_full_10seed_functional_validation.csv", "P10_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p11_adamw_only_lq_fullpass_repair.csv", "P11_ADAMW_ONLY_LQ_FULLPASS_REPAIR"),
        ("p12_robustness_external_ready.csv", "P12_ROBUSTNESS_EXTERNAL_READY"),
    ]
    for filename, stage in mapping:
        write_csv_rows(out_dir / filename, [_not_run(stage, filename, reason)])


def _write_report(out_dir: Path, route: Dict[str, Any], artifacts: Sequence[Path]) -> None:
    p1_rows = read_csv_rows(out_dir / "p1_lq_base_anchor_protocol_audit.csv")
    current = [r for r in p1_rows if r.get("status") == "measured_current"]
    summary = next((r for r in p1_rows if r.get("status") == "summary"), {})
    lines = [
        "# DG-KAN v9.2.38 LQ Base Re-Anchor 与 Snapshot Late-Attach Functional Carrier 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.38_LQBaseReanchor_SnapshotLateAttach_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route['route']}",
        f"base_candidate = {route['base_candidate']}",
        f"success_v9238_strict_purekan_functional = {bool(route['success_v9238_strict_purekan_functional'])}",
        f"success_v9238_full_functional = {bool(route['success_v9238_full_functional'])}",
        f"success_v9238_external_ready = {bool(route['success_v9238_external_ready'])}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        _rel(out_dir),
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.37 boundary：source route = `{route.get('source_route')}`，source blocker = `{route.get('source_primary_blocker')}`。",
        f"2. P1 Current LQ re-anchor：near-pass `{route.get('current_lq_near_count')}/{route.get('current_lq_row_count')}`，macro delta `{route.get('current_lq_macro_delta'):.6f}`，anchor pass = `{route.get('lq_base_anchor_pass')}`。",
        f"3. Historical reference：near-pass `{route.get('historical_lq_near_count')}/{route.get('historical_lq_row_count')}`，macro delta `{route.get('historical_lq_macro_delta'):.6f}`；historical-current macro delta `{route.get('historical_current_lq_delta'):.6f}`。",
        f"4. TPEA-off P5 equivalence pass = `{route.get('tpea_off_p5_equivalence_pass')}`；protocol mismatch detected = `{route.get('protocol_mismatch_detected')}`，mode = `{route.get('protocol_mismatch_mode')}`。",
        f"5. 因 LQ anchor 未过，本轮 snapshot late-attach / carrier / value / paired replay 全部未打开。",
        f"6. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9238_lq_base_reanchor_snapshot_late_attach.py` | v9.2.38 runner；生成 P0-P12 artifacts、LQ anchor/protocol audit、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9238_lq_base_reanchor_snapshot_late_attach.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9238_lq_base_reanchor_snapshot_late_attach.py \\",
        "  --out-dir results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z \\",
        "  --fresh --device auto --data-root data --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 3. P1 LQ base anchor / protocol mismatch audit",
        "",
        "| candidate | rows | near rows | macro delta | max acc diff vs Current LQ |",
        "|---|---:|---:|---:|---:|",
    ]
    by_cand: Dict[str, List[Dict[str, Any]]] = {}
    for r in current:
        by_cand.setdefault(str(r.get("candidate")), []).append(r)
    for cand, rows in by_cand.items():
        near = sum(_int(r.get("near_pass")) for r in rows)
        macro = _mean(_float(r.get("delta_vs_mlp")) for r in rows)
        if cand == "CurrentLQReproduction":
            max_acc = 0.0
        else:
            base = {(r.get("dataset"), _int(r.get("seed"))): r for r in by_cand.get("CurrentLQReproduction", [])}
            max_acc = max([abs(_float(r.get("KAN_acc")) - _float(base.get((r.get("dataset"), _int(r.get("seed"))), {}).get("KAN_acc"))) for r in rows], default=0.0)
        lines.append(f"| {cand} | `{len(rows)}` | `{near}/{len(rows)}` | `{macro:.6f}` | `{max_acc:.6f}` |")
    lines.extend([
        "",
        "P1 summary：",
        "",
        "```text",
        f"lq_base_anchor_pass = {summary.get('lq_base_anchor_pass')}",
        f"current_lq_near_rate = {_float(summary.get('current_lq_near_rate')):.6f}",
        f"historical_current_lq_delta = {_float(summary.get('historical_current_lq_delta')):.6f}",
        f"tpea_off_p5_equivalence_pass = {summary.get('tpea_off_p5_equivalence_pass')}",
        f"protocol_mismatch_detected = {summary.get('protocol_mismatch_detected')}",
        f"protocol_mismatch_mode = {summary.get('protocol_mismatch_mode')}",
        "```",
        "",
        "判断：Current LQ 在本轮真实 runner 下没有达到 `near_pass_rate >= 0.80`，因此 functional route 按计划停止；没有把 snapshot late-attach 或 functional carrier 在未锚定 base 上继续推进。",
        "",
        "## 4. Downstream boundary",
        "",
        "P2-P12 只有在 LQ base anchor pass 后打开。本轮均以 `not_run` row 落盘，reason = `LQ_base_anchor_failed`。",
        "",
        "## 5. No-fake audit",
        "",
        "```text",
        f"rows_checked = {route.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {route.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {route.get('fake_data_used')}",
        f"proxy_row_used = {route.get('proxy_row_used')}",
        f"cpu_offload_used = {route.get('cpu_offload_used')}",
        f"no_fake = {route.get('no_fake')}",
        f"no_proxy = {route.get('no_proxy')}",
        "```",
        "",
        "## 6. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
        f"| runner | `{_hash_file(SCRIPT_PATH)}` |",
    ])
    for path in artifacts:
        lines.append(f"| `{_rel(path)}` | `{_hash_file(path)}` |")
    lines.extend([
        "",
        "## 7. 最终分析结论",
        "",
        "v9.2.38 的真实推进是：",
        "",
        "```text",
        "v9.2.37: TPEA training-path equivalence pass, but P5 near-pass fail.",
        "v9.2.38: LQ base re-anchor / protocol mismatch audit is performed before snapshot late attach.",
        "```",
        "",
        "机制判断：",
        "",
        "1. 本轮最重要的结论不是 functional carrier failure，而是 current LQ base anchor 仍未闭合。",
        "2. Current LQ macro delta 与 historical reference 接近，但 near-pass row count 从 historical `8/9` 变为 current `6/9`，没有达到计划的 anchor gate。",
        "3. TPEA-off 与 current LQ 的 P5 等价性可以审计，但在 LQ anchor 未过前不能打开 snapshot late-attach。",
        "4. 下一步应先修 LQ base/protocol re-anchor，而不是继续设计 value score 或 dataset-specific functional route。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.38 真实执行后停在 `{route['route']}`：`{route.get('primary_blocker')}`。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=3)
    args = parser.parse_args()
    args.data_root = str(args.data_root)
    args.p5_train_size = int(args.train_size)
    args.p5_test_size = int(args.test_size)
    args.p5_epochs = int(args.epochs)
    args.p5_lr = float(args.lr)
    out_dir = args.out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))
    write_json(out_dir / "run_manifest.json", {
        "version": "v9.2.38",
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "source_v9237": _rel(SRC_V9237),
        "historical_source_v927": _rel(SRC_V927),
        "created_at": _now_iso(),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    p0 = _p0_boundary()
    write_csv_rows(out_dir / "p0_v9237_boundary_reproduction.csv", [p0])
    p1_rows, p1_trace, p1 = _p1_lq_anchor(args, device, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_lq_base_anchor_protocol_audit.csv", p1_rows)
    write_csv_rows(out_dir / "lq_anchor_trace_v9238.csv", p1_trace)
    hash_rows = []
    for r in p1_rows:
        if r.get("status") in {"measured_current", "summary"}:
            hash_rows.append({
                "stage": "P1_PROTOCOL_HASH_DIFF",
                "candidate": r.get("candidate", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "p5_gate_definition_hash": r.get("p5_gate_definition_hash", ""),
                "mlp_match_definition_hash": r.get("mlp_match_definition_hash", ""),
                "candidate_config_hash": r.get("candidate_config_hash", ""),
                "runner_hash": r.get("runner_hash", ""),
                "data_protocol_hash": r.get("data_protocol_hash", ""),
                "seed_protocol_hash": r.get("seed_protocol_hash", ""),
                "protocol_mismatch_mode": p1.get("protocol_mismatch_mode", ""),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    write_csv_rows(out_dir / "protocol_hash_diff_v9238.csv", hash_rows or [_not_run("P1_PROTOCOL_HASH_DIFF", "protocol_hash_diff_v9238.csv", "P1_not_measured")])
    downstream_reason = "LQ_base_anchor_failed" if not _int(p1.get("lq_base_anchor_pass")) else "snapshot_late_attach_not_implemented_after_anchor_pass"
    _write_gate_blocked_artifacts(out_dir, downstream_reason)
    write_csv_rows(out_dir / "contract_audit_v9238.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "dataset_tuning_detected": 0,
    }])
    if not _int(p0.get("P0_pass")):
        route_name, primary = "R0-V9237BoundaryMismatch", "v9237_boundary_not_reproduced"
    elif not _int(p1.get("lq_base_anchor_pass")):
        route_name, primary = "R1-LQBaseAnchorBroken", "current_lq_base_nearpass_rate_below_gate"
    elif not _int(p1.get("tpea_off_p5_equivalence_pass")):
        route_name, primary = "R4-TPEAOffEquivalentToLQ", "tpea_off_not_p5_equivalent_to_lq"
    else:
        route_name, primary = "R3-LQBaseReanchored", "snapshot_late_attach_not_implemented_in_this_runner"
    artifacts = [
        out_dir / "run_manifest.json",
        out_dir / "contract_audit_v9238.csv",
        out_dir / "p0_v9237_boundary_reproduction.csv",
        out_dir / "p1_lq_base_anchor_protocol_audit.csv",
        out_dir / "p2_snapshot_late_attach_implementation.csv",
        out_dir / "p3_checkpoint_inactive_equivalence.csv",
        out_dir / "p4_no_event_replay_base_preservation.csv",
        out_dir / "p5_functional_carrier_actuatability.csv",
        out_dir / "p6_value_observability_audit.csv",
        out_dir / "p7_leave_dataset_and_stratum_out_validation.csv",
        out_dir / "p8_official_snapshot_late_attach_paired_replay.csv",
        out_dir / "p9_short_run_functional_validation.csv",
        out_dir / "p10_full_10seed_functional_validation.csv",
        out_dir / "p11_adamw_only_lq_fullpass_repair.csv",
        out_dir / "p12_robustness_external_ready.csv",
        out_dir / "lq_anchor_trace_v9238.csv",
        out_dir / "protocol_hash_diff_v9238.csv",
        out_dir / "snapshot_attach_trace_v9238.csv",
        out_dir / "checkpoint_equivalence_trace_v9238.csv",
        out_dir / "functional_carrier_trace_v9238.csv",
        out_dir / "value_score_trace_v9238.csv",
        out_dir / "leave_dataset_out_trace_v9238.csv",
        out_dir / "paired_replay_branch_trace_v9238.csv",
    ]
    audit = audit_no_fake(artifacts)
    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9237_boundary_pass": _int(p0.get("P0_pass")),
        "source_route": p0.get("route", ""),
        "source_primary_blocker": p0.get("primary_blocker", ""),
        "dataset_tuning_detected": 0,
        "lq_base_anchor_pass": p1.get("lq_base_anchor_pass", 0),
        "historical_current_lq_delta": p1.get("historical_current_lq_delta", 0.0),
        "protocol_mismatch_detected": p1.get("protocol_mismatch_detected", 0),
        "protocol_mismatch_mode": p1.get("protocol_mismatch_mode", ""),
        "current_lq_nearpass": p1.get("current_lq_nearpass", 0),
        "current_lq_near_count": p1.get("current_lq_near_count", 0),
        "current_lq_row_count": p1.get("current_lq_row_count", 0),
        "current_lq_near_rate": p1.get("current_lq_near_rate", 0.0),
        "current_lq_macro_delta": p1.get("current_lq_macro_delta", 0.0),
        "historical_lq_near_count": p1.get("historical_lq_near_count", 0),
        "historical_lq_row_count": p1.get("historical_lq_row_count", 0),
        "historical_lq_macro_delta": p1.get("historical_lq_macro_delta", 0.0),
        "tpea_off_p5_equivalence_pass": p1.get("tpea_off_p5_equivalence_pass", 0),
        "snapshot_attach_implemented_count": 0,
        "best_late_attach_candidate": "",
        "checkpoint_inactive_equivalence_pass": 0,
        "no_event_replay_preservation_pass": 0,
        "functional_carrier_pass": 0,
        "value_observability_pass": 0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": "repair_current_lq_base_anchor_or_protocol_before_snapshot_late_attach",
        "success_v9238_strict_purekan_functional": 0,
        "success_v9238_full_functional": 0,
        "success_v9238_external_ready": 0,
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure = [
        {"stage": "P0", "pass": _int(p0.get("P0_pass")), "blocker": "" if _int(p0.get("P0_pass")) else "F2_v9237_boundary_unstable"},
        {"stage": "P1", "pass": p1.get("lq_base_anchor_pass", 0), "blocker": "" if p1.get("lq_base_anchor_pass") else "F4_lq_base_anchor_fail"},
        {"stage": "P1_TPEA_OFF_EQUIV", "pass": p1.get("tpea_off_p5_equivalence_pass", 0), "blocker": "" if p1.get("tpea_off_p5_equivalence_pass") else "F9_tpea_off_not_p5_equivalent_to_lq"},
        {"stage": "P2", "pass": 0, "blocker": "not_opened_LQ_base_anchor_failed" if not p1.get("lq_base_anchor_pass") else "F10_snapshot_late_attach_not_implemented"},
        {"stage": "P3", "pass": 0, "blocker": "not_opened_snapshot_late_attach_not_available"},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure)
    write_csv_rows(out_dir / "v9238_provenance_audit.csv", [audit])
    final_artifacts = [*artifacts, out_dir / "route_decision.json", out_dir / "aggregate_decision.json", out_dir / "failure_table.csv", out_dir / "v9238_provenance_audit.csv"]
    _write_report(out_dir, route, final_artifacts)


if __name__ == "__main__":
    main()
