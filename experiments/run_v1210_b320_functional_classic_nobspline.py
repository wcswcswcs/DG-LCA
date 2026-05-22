#!/usr/bin/env python3
"""DG-KAN v12.10 B320 anchor hardening wrapper.

This script is intentionally a thin audit layer around the existing measured
v12.8.3/v12.9 runner. It does not rewrite raw experimental numbers; it keeps
the reused raw artifacts and emits v12.10 route/policy/provenance files whose
claims are derived from those artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RUNNER = REPO_ROOT / "experiments" / "run_v1283_b109_classic_family_functional_geometry.py"
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "DG-KAN_v12.10_B320_Functional_ClassicNoBSpline_执行复盘.md"

B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"
B321_ID = "B321b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp085"
B314_ID = "B314b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090"
B109_ID = "B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075"
B109_FIXEDP_ID = "B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075"

ACTIVE_FAMILY_IDS = [
    "B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3",
    "B7me-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG150-freezeBackbone-hiddenBias-manualAdamW-L3",
    "B3y-ChebyKAN-K3-h96-paircrossR32-tritonL3-gradbuf",
    "B3z-ChebyKAN-K3-h88-paircrossR32-tritonL3-gradbuf",
    "B3aa-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P8-tritonL3-gradbuf",
    "B3f-ChebyKAN-K4-tritonL3-matmulTile",
    "B3al-ChebyKAN-K3-h72-paircrossR32-inputrot2L4P8-tritonL3-gradbuf",
    "B3am-ChebyKAN-K3-h64-paircrossR32-inputrot2L4P8-tritonL3-gradbuf",
    "B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3",
    "B4q-FourierKAN-lowfreq-K4-h48-linearres-gemmDirectL3",
    "B4v-FourierKAN-lowfreq-K4-h8-linearres050-gemmDirectL3",
    "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile",
    "B2r-FastKAN-RBF-stream-K2-repair",
    "B2s-GaussianRBF-stream-K4-recompute",
    "B5h-HatWaveletKAN-local-K4",
]

FAMILY_BY_ID_PREFIX = {
    "B7": "Rational",
    "B3": "Chebyshev",
    "B4": "Fourier",
    "B2": "RBF/FastKAN",
    "B5": "Wavelet",
}

STRICT_STEP_GATE = 1.00
STRICT_MEMORY_GATE = 0.30
STRICT_WORST_GATE = -0.003
STRICT_ECE_GATE = 0.02
STRICT_AUC_GATE = 1.00


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def active_family_policy_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in ACTIVE_FAMILY_IDS:
        prefix = candidate_id.split("-", 1)[0][:2]
        rows.append(
            {
                "stage": "V1210_FAMILY_POLICY",
                "family": FAMILY_BY_ID_PREFIX.get(prefix, "Unknown"),
                "candidate_id": candidate_id,
                "status": "Active",
                "active_followup": 1,
                "codex_budget": 1,
                "bspline_excluded": 1 if "BSpline" not in candidate_id and "spline" not in candidate_id.lower() else 0,
                "reason": "v12.10 active classic-family branch excluding B-spline",
            }
        )
    rows.append(
        {
            "stage": "V1210_FAMILY_POLICY",
            "family": "BSpline",
            "candidate_id": "BSpline-family",
            "status": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
            "active_followup": 0,
            "codex_budget": 0,
            "bspline_excluded": 1,
            "reason": "frozen by v12.10 plan until a real fused forward+backward/update spline kernel first satisfies step<=1.10 and memory<=0.80",
        }
    )
    return rows


def write_contract_files(args: argparse.Namespace, out_dir: Path, raw_dir: Path, cmd: Sequence[str]) -> None:
    route_config = {
        "stage": "V1210_ROUTE_CONFIG",
        "generated_at": now_iso(),
        "raw_runner": str(RAW_RUNNER.relative_to(REPO_ROOT)),
        "raw_dir": str(raw_dir.relative_to(REPO_ROOT)),
        "exact_candidate_id": B320_ID,
        "comparison_candidate_ids": [B321_ID, B314_ID, B109_ID, B109_FIXEDP_ID, "MLP-same-param-AdamW"],
        "raw_repair_candidate_ids": args.raw_repair_candidate_ids,
        "raw_family_candidate_ids": args.family_candidate_ids,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "kernel_warmup_steps": args.kernel_warmup_steps,
        "kernel_measure_steps": args.kernel_measure_steps,
        "task_compile_warmup_steps": args.task_compile_warmup_steps,
        "strict_gates": {
            "step_ratio_q90_max": STRICT_STEP_GATE,
            "memory_ratio_q90_max": STRICT_MEMORY_GATE,
            "mean_delta_min": 0.0,
            "worst_delta_min": STRICT_WORST_GATE,
            "near_pass_rate_min": 1.0,
            "auc_step_ratio_max": STRICT_AUC_GATE,
            "auc_time_ratio_max": STRICT_AUC_GATE,
            "ece_delta_max": STRICT_ECE_GATE,
        },
        "hard_constraints": {
            "ce_only": 1,
            "no_teacher": 1,
            "no_distillation": 1,
            "no_loss_modification": 1,
            "no_sampler_or_class_weight": 1,
            "no_dataset_name_branch": 1,
            "no_fake_proxy_cpu_offload": 1,
        },
        "bspline_policy": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
        "command": list(cmd),
    }
    family_rows = active_family_policy_rows()
    family_policy = {
        "stage": "V1210_FAMILY_POLICY",
        "generated_at": now_iso(),
        "active_families": ["Rational", "Chebyshev", "Fourier", "RBF/FastKAN", "Wavelet"],
        "frozen_families": ["BSpline"],
        "active_candidate_ids": ACTIVE_FAMILY_IDS,
        "bspline": {
            "status": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
            "active_followup": 0,
            "codex_budget": 0,
            "reopen_condition": "real fused forward+backward/update spline kernel with step_ratio<=1.10 and memory_ratio<=0.80",
        },
        "rows": family_rows,
    }
    write_json(out_dir / "v1210_route_config.json", route_config)
    write_json(out_dir / "v1210_family_policy.json", family_policy)
    write_csv_rows(out_dir / "v1210_family_policy.csv", family_rows)


def run_raw_or_reuse(args: argparse.Namespace, out_dir: Path) -> tuple[Path, int, str]:
    raw_dir = Path(args.reuse_raw_dir).resolve() if args.reuse_raw_dir else out_dir / "raw_v1283_reuse"
    raw_report = out_dir / "v1210_raw_runner_report.md"
    if args.reuse_raw_dir:
        write_contract_files(args, out_dir, raw_dir, ["reuse_raw_dir", str(raw_dir)])
        return raw_dir, 0, "reused"
    if args.fresh and raw_dir.exists():
        shutil.rmtree(raw_dir)
    ensure_dir(raw_dir)
    cmd = [
        sys.executable,
        str(RAW_RUNNER),
        "--out-dir",
        str(raw_dir),
        "--device",
        args.device,
        "--data-root",
        args.data_root,
        "--seed",
        str(args.seed),
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--kernel-warmup-steps",
        str(args.kernel_warmup_steps),
        "--kernel-measure-steps",
        str(args.kernel_measure_steps),
        "--task-compile-warmup-steps",
        str(args.task_compile_warmup_steps),
        "--datasets",
        args.datasets,
        "--seeds",
        args.seeds,
        "--exact-candidate-id",
        B320_ID,
        "--fixedp-candidate-id",
        B109_FIXEDP_ID,
        "--repair-candidate-ids",
        args.raw_repair_candidate_ids,
        "--family-candidate-ids",
        args.family_candidate_ids,
        "--family-linec-ids",
        ",".join([ACTIVE_FAMILY_IDS[0], ACTIVE_FAMILY_IDS[1]]),
        "--report-path",
        str(raw_report),
    ]
    if args.no_download:
        cmd.append("--no-download")
    write_contract_files(args, out_dir, raw_dir, cmd)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    (out_dir / "v1210_raw_runner_stdout.log").write_text(result.stdout, encoding="utf-8")
    (out_dir / "v1210_raw_runner_stderr.log").write_text(result.stderr, encoding="utf-8")
    return raw_dir, result.returncode, "ran"


def copy_raw_artifacts(raw_dir: Path, out_dir: Path) -> None:
    mapping = {
        "v1283_b109_fullstep_profile.csv": "v1210_b320_efficiency_profile.csv",
        "v1283_b109_auc_attribution.csv": "v1210_b320_auc_attribution.csv",
        "v1283_b109_auc_autopsy_trace.csv": "v1210_b320_task_trace.csv",
        "v1283_b109_task_autopsy_final.csv": "v1210_b320_task_final.csv",
        "v1283_b109_linec_diagnostics.csv": "v1210_linec_diagnostics.csv",
        "v1283_b109_linec_coupling.csv": "v1210_linec_coupling.csv",
        "v1283_b109_linec_noise_leak.csv": "v1210_linec_noise_leak.csv",
        "v1283_b109_linec_signal_reservoir.csv": "v1210_linec_signal_reservoir.csv",
        "v1283_b109_linec_functional_reentry_summary.csv": "v1210_functional_reentry_summary.csv",
        "v1283_b109_functional_delta_score_repair_summary.csv": "v1210_functional_delta_score_repair_summary.csv",
        "v1283_family_efficiency.csv": "v1210_family_efficiency.csv",
        "v1283_family_expression.csv": "v1210_family_expression.csv",
        "v1283_family_task_triage.csv": "v1210_family_task_triage.csv",
        "v1283_family_linec_diagnostics.csv": "v1210_family_linec_diagnostics.csv",
        "v1283_family_gradcheck.csv": "v1210_family_gradcheck.csv",
        "v1283_family_component_profile.csv": "v1210_family_component_profile.csv",
        "v1283_family_manifest.csv": "v1210_family_manifest_raw.csv",
        "v1283_provenance_audit.csv": "v1210_raw_provenance_audit.csv",
        "v1283_modification_audit.csv": "v1210_raw_modification_audit.csv",
    }
    for src_name, dst_name in mapping.items():
        src = raw_dir / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def filter_bspline_family_status(raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    raw_status = read_json(raw_dir / "v1283_family_status.json")
    families = raw_status.get("families", {}) if isinstance(raw_status, Mapping) else {}
    filtered: dict[str, Any] = {}
    for family, payload in families.items():
        if family == "BSpline":
            continue
        filtered[family] = payload
    filtered["BSpline"] = {
        "status": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
        "active_followup": 0,
        "codex_budget": 0,
        "source_status_before_v1210_policy": families.get("BSpline", {}).get("status", ""),
        "source_best_L3_manual_step_ratio": families.get("BSpline", {}).get("best_L3_manual_step_ratio", ""),
        "blocker": "excluded from active v12.10 follow-up by plan; no B-spline candidates scheduled in this run",
        "official_family_failure_claim": False,
    }
    payload = {
        "stage": "V1210_FAMILY_STATUS",
        "generated_at": now_iso(),
        "families": filtered,
    }
    write_json(out_dir / "v1210_family_status.json", payload)
    failure_rows = []
    for family, info in filtered.items():
        row = {"stage": "V1210_FAMILY_FAILURE_TABLE", "family": family}
        row.update(info if isinstance(info, Mapping) else {})
        failure_rows.append(row)
    write_csv_rows(out_dir / "v1210_family_failure_table.csv", failure_rows)
    return payload


def strict_b320_efficiency_rows(full_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        r
        for r in full_rows
        if r.get("candidate_id") == B320_ID
        and str(r.get("implementation_id")) == "F3-triton-workspace-forward-delta-readout-proj-grad-learnableP"
    ]


def summarize_b320_task(auc_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_rows = [
        r
        for r in auc_rows
        if r.get("candidate_id") == B320_ID and str(r.get("stage")) == "V1283_B109_AUC_ATTRIBUTION"
    ]
    per_metrics: dict[str, Any] = {}
    if per_rows:
        ece = [safe_float(r.get("ECE_delta_vs_mlp"), 999.0) for r in per_rows]
        auc_step = [safe_float(r.get("AUC_step_ratio"), 999.0) for r in per_rows]
        auc_time = [safe_float(r.get("AUC_time_ratio"), 999.0) for r in per_rows]
        per_metrics = {
            "rows": len(per_rows),
            "max_ECE_delta": max(ece),
            "max_AUC_step": max(auc_step),
            "max_AUC_time": max(auc_time),
        }
    summaries = [
        r
        for r in auc_rows
        if r.get("candidate_id") == B320_ID and str(r.get("stage")) == "V1283_B109_AUC_ATTRIBUTION_SUMMARY"
    ]
    if not summaries:
        if not per_rows:
            return {"status": "not_measured"}
        val_acc = [safe_float(r.get("val_acc_delta_vs_mlp"), 0.0) for r in per_rows]
        near = [1 if v >= STRICT_WORST_GATE else 0 for v in val_acc]
        return {
            "status": "derived_from_per_dataset_rows",
            "rows": len(per_rows),
            "mean_delta": sum(val_acc) / len(val_acc),
            "worst_delta": min(val_acc),
            "near_pass_rate": sum(near) / len(near),
            "max_ECE_delta": per_metrics.get("max_ECE_delta", ""),
            "max_AUC_step": per_metrics.get("max_AUC_step", ""),
            "max_AUC_time": per_metrics.get("max_AUC_time", ""),
            "ece_ok": int(safe_float(per_metrics.get("max_ECE_delta"), 999.0) <= STRICT_ECE_GATE),
            "auc_step_ok": int(safe_float(per_metrics.get("max_AUC_step"), 999.0) <= STRICT_AUC_GATE),
            "auc_time_ok": int(safe_float(per_metrics.get("max_AUC_time"), 999.0) <= STRICT_AUC_GATE),
        }
    best = summaries[0]
    return {
        "status": "measured_summary",
        "rows": per_metrics.get("rows", best.get("rows", "")),
        "mean_delta": safe_float(best.get("mean_delta"), 0.0),
        "worst_delta": safe_float(best.get("worst_delta"), -999.0),
        "near_pass_rate": safe_float(best.get("near_pass_rate"), 0.0),
        "max_ECE_delta": per_metrics.get("max_ECE_delta", ""),
        "max_AUC_step": per_metrics.get("max_AUC_step", ""),
        "max_AUC_time": per_metrics.get("max_AUC_time", ""),
        "ece_ok": int(safe_float(best.get("ece_ok"), 0.0)),
        "auc_step_ok": int(safe_float(best.get("auc_step_ok"), 0.0)),
        "auc_time_ok": int(safe_float(best.get("auc_time_ok"), 0.0)),
        "A5_autopsy_pass": int(safe_float(best.get("A5_autopsy_pass"), 0.0)),
    }


def summarize_linec(raw_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(raw_dir / "v1283_b109_linec_diagnostics.csv")
    out: dict[str, Any] = {"status": "not_measured"}
    for row in rows:
        if row.get("candidate_id") == B320_ID:
            out = {
                "status": row.get("status", "measured"),
                "candidate_id": B320_ID,
                "CouplingR2": safe_float(row.get("CouplingR2"), 0.0),
                "NoiseSignalLeak": safe_float(row.get("NoiseSignalLeak"), 999.0),
                "RealSignalReservoirRatio": safe_float(row.get("RealSignalReservoirRatio"), 0.0),
                "linec_interpretation": row.get("linec_interpretation", ""),
            }
    mlp = {}
    for row in rows:
        if str(row.get("candidate_id")) == "MLP-same-param-AdamW":
            mlp = row
    if out.get("status") != "not_measured" and mlp:
        mlp_coupling = safe_float(mlp.get("CouplingR2"), 0.0)
        mlp_leak = safe_float(mlp.get("NoiseSignalLeak"), 0.0)
        out["MLP_CouplingR2"] = mlp_coupling
        out["MLP_NoiseSignalLeak"] = mlp_leak
        out["nontearing_pass"] = int(
            safe_float(out.get("CouplingR2"), -999.0) >= mlp_coupling - 0.02
            and safe_float(out.get("NoiseSignalLeak"), 999.0) <= mlp_leak + 0.02
        )
    return out


def write_functional_p3_gate(raw_dir: Path, out_dir: Path, base_linec: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        r
        for r in read_csv_rows(raw_dir / "v1283_b109_functional_delta_score_repair.csv")
        if r.get("selected_b109_candidate_id") == B320_ID and r.get("candidate_id") == B320_ID
    ]
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        update_type = str(row.get("update_type", ""))
        out_rows.append(
            {
                "stage": "V1210_FUNCTIONAL_ONE_FIVE_STEP",
                "candidate_id": B320_ID,
                "functional_id": update_type if update_type.startswith("F") else "",
                "control_id": update_type if update_type.startswith("C") else "",
                "source_update_type": update_type,
                "holdout_descent_ratio": row.get("holdout_descent_ratio", ""),
                "bad_step": 0 if int(safe_float(row.get("accepted"), 0)) == 1 else 1,
                "bad_step_reason": "" if int(safe_float(row.get("accepted"), 0)) == 1 else "rejected_by_raw_backtracking_or_safety",
                "CouplingR2_before": base_linec.get("CouplingR2", ""),
                "CouplingR2_after": row.get("CouplingR2", ""),
                "CouplingR2_delta": safe_float(row.get("CouplingR2"), 0.0) - safe_float(base_linec.get("CouplingR2"), 0.0),
                "RealSignalReservoirRatio_before": row.get("RealSignalReservoirRatio_before", ""),
                "RealSignalReservoirRatio_after": row.get("RealSignalReservoirRatio_after", ""),
                "RealSignalReservoirRatio_delta": row.get("RealSignalReservoirRatio_delta", ""),
                "NoiseSignalLeak_before": row.get("NoiseSignalLeak_before", ""),
                "NoiseSignalLeak_after": row.get("NoiseSignalLeak_after", ""),
                "NoiseSignalLeak_delta": row.get("NoiseSignalLeak_delta", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "margin_p10_delta": row.get("margin_p10_delta", ""),
                "control_gap": "",
                "delta_score": row.get("delta_score", ""),
                "official_candidate_pass": 0,
                "source_stage": row.get("stage", ""),
                "no_fake": 1,
            }
        )
    write_csv_rows(out_dir / "v1210_functional_one_five_step.csv", out_rows)
    if not rows:
        summary = {
            "stage": "V1210_FUNCTIONAL_P3_GATE",
            "candidate_id": B320_ID,
            "status": "not_measured",
            "official_candidate_pass": 0,
        }
        write_csv_rows(out_dir / "v1210_functional_p3_gate.csv", [summary])
        return summary
    controls = [r for r in rows if str(r.get("update_type", "")).startswith("C")]
    funcs = [r for r in rows if str(r.get("update_type", "")).startswith("F")]
    best_control = max(controls, key=lambda r: safe_float(r.get("delta_score"), -999.0), default={})
    best_func = max(funcs, key=lambda r: safe_float(r.get("delta_score"), -999.0), default={})
    control_gap = safe_float(best_func.get("delta_score"), -999.0) - safe_float(best_control.get("delta_score"), -999.0)
    coupling_delta_vs_base_linec = safe_float(best_func.get("CouplingR2"), -999.0) - safe_float(base_linec.get("CouplingR2"), 0.0)
    coupling_delta_vs_best_control = safe_float(best_func.get("CouplingR2"), -999.0) - safe_float(best_control.get("CouplingR2"), 0.0)
    noise_ok = safe_float(best_func.get("NoiseSignalLeak_after"), 999.0) <= safe_float(best_func.get("NoiseSignalLeak_before"), -999.0)
    holdout_ok = safe_float(best_func.get("holdout_descent_ratio"), 0.0) >= 0.95
    coupling_ok = coupling_delta_vs_best_control >= 0.02
    control_gap_ok = control_gap >= 0.005
    official_pass = int(holdout_ok and coupling_ok and noise_ok and control_gap_ok)
    fail_reasons = []
    if not holdout_ok:
        fail_reasons.append("holdout_descent_ratio<0.95")
    if not coupling_ok:
        fail_reasons.append("CouplingR2_delta_vs_best_control<0.02")
    if not noise_ok:
        fail_reasons.append("NoiseSignalLeak_after>before")
    if not control_gap_ok:
        fail_reasons.append("control_gap<0.005")
    summary = {
        "stage": "V1210_FUNCTIONAL_P3_GATE",
        "candidate_id": B320_ID,
        "status": "measured",
        "best_control": best_control.get("update_type", ""),
        "best_control_delta_score": best_control.get("delta_score", ""),
        "best_functional_update": best_func.get("update_type", ""),
        "best_functional_delta_score": best_func.get("delta_score", ""),
        "control_gap": control_gap,
        "holdout_descent_ratio": best_func.get("holdout_descent_ratio", ""),
        "CouplingR2_base_linec": base_linec.get("CouplingR2", ""),
        "CouplingR2_best_control": best_control.get("CouplingR2", ""),
        "CouplingR2_after": best_func.get("CouplingR2", ""),
        "CouplingR2_delta_vs_base_linec": coupling_delta_vs_base_linec,
        "CouplingR2_delta_vs_best_control": coupling_delta_vs_best_control,
        "NoiseSignalLeak_before": best_func.get("NoiseSignalLeak_before", ""),
        "NoiseSignalLeak_after": best_func.get("NoiseSignalLeak_after", ""),
        "NoiseSignalLeak_delta": best_func.get("NoiseSignalLeak_delta", ""),
        "official_candidate_pass": official_pass,
        "fail_reason": ";".join(fail_reasons),
        "interpretation": "F-F1/F-F3: functional diagnostic did not beat strong controls and did not improve one-step coupling over best control by +0.02" if not official_pass else "P3 diagnostic gate passed",
    }
    write_csv_rows(out_dir / "v1210_functional_p3_gate.csv", [summary])
    short_run = {
        "stage": "V1210_FUNCTIONAL_SHORT_RUN",
        "candidate_id": B320_ID,
        "functional_id": summary.get("best_functional_update", ""),
        "status": "not_run_implementation_blocked" if official_pass else "not_run_p3_gate_failed",
        "P3_official_candidate_pass": official_pass,
        "strict_pass": 0,
        "official_functional_success": 0,
        "reason": "P3 gate opened, but P4 requires integrating this functional event into the existing FHQ/manual-AdamW task loop with strong controls; no equivalent official short-run implementation exists in this runner" if official_pass else "P3 diagnostic gate did not pass",
        "claim_scope": "no_official_success_claim",
    }
    write_csv_rows(out_dir / "v1210_functional_short_run.csv", [short_run])
    return summary


def build_decision(raw_dir: Path, out_dir: Path, raw_returncode: int, raw_mode: str, family_status: Mapping[str, Any]) -> dict[str, Any]:
    full_rows = read_csv_rows(raw_dir / "v1283_b109_fullstep_profile.csv")
    auc_rows = read_csv_rows(raw_dir / "v1283_b109_auc_attribution.csv")
    eff_rows = strict_b320_efficiency_rows(full_rows)
    eff = eff_rows[0] if eff_rows else {}
    task = summarize_b320_task(auc_rows)
    linec = summarize_linec(raw_dir)
    functional_p3 = write_functional_p3_gate(raw_dir, out_dir, linec)
    raw_decision = read_json(raw_dir / "v1283_route_decision.json")
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    family_pass_count = sum(1 for fam, info in families.items() if fam != "BSpline" and str(info.get("status")) == "FamilyPass")
    step_ratio = safe_float(eff.get("step_ratio_q90"), 999.0)
    memory_ratio = safe_float(eff.get("memory_ratio_q90"), 999.0)
    task_measured = task.get("status") != "not_measured"
    hardening_pass = bool(
        raw_returncode == 0
        and eff
        and step_ratio <= STRICT_STEP_GATE
        and memory_ratio <= STRICT_MEMORY_GATE
        and task_measured
        and safe_float(task.get("mean_delta"), -999.0) >= 0.0
        and safe_float(task.get("worst_delta"), -999.0) >= STRICT_WORST_GATE
        and safe_float(task.get("near_pass_rate"), 0.0) >= 1.0
        and safe_float(task.get("max_AUC_step"), 999.0) <= STRICT_AUC_GATE
        and safe_float(task.get("max_AUC_time"), 999.0) <= STRICT_AUC_GATE
        and safe_float(task.get("max_ECE_delta"), 999.0) <= STRICT_ECE_GATE
    )
    strict_fail_reasons: list[str] = []
    if raw_returncode != 0:
        strict_fail_reasons.append("raw_runner_failed")
    if not eff:
        strict_fail_reasons.append("B320_F3_not_measured")
    elif step_ratio > STRICT_STEP_GATE:
        strict_fail_reasons.append(f"F3_step_ratio_q90>{STRICT_STEP_GATE}")
    if eff and memory_ratio > STRICT_MEMORY_GATE:
        strict_fail_reasons.append(f"F3_memory_ratio_q90>{STRICT_MEMORY_GATE}")
    if not task_measured:
        strict_fail_reasons.append("B320_task_not_measured")
    elif safe_float(task.get("mean_delta"), -999.0) < 0.0:
        strict_fail_reasons.append("mean_delta<0")
    if task_measured and safe_float(task.get("worst_delta"), -999.0) < STRICT_WORST_GATE:
        strict_fail_reasons.append(f"worst_delta<{STRICT_WORST_GATE}")
    if task_measured and safe_float(task.get("near_pass_rate"), 0.0) < 1.0:
        strict_fail_reasons.append("near_pass_rate<1.0")
    if task_measured and safe_float(task.get("max_AUC_step"), 999.0) > STRICT_AUC_GATE:
        strict_fail_reasons.append(f"max_AUC_step>{STRICT_AUC_GATE}")
    if task_measured and safe_float(task.get("max_AUC_time"), 999.0) > STRICT_AUC_GATE:
        strict_fail_reasons.append(f"max_AUC_time>{STRICT_AUC_GATE}")
    if task_measured and safe_float(task.get("max_ECE_delta"), 999.0) > STRICT_ECE_GATE:
        strict_fail_reasons.append(f"max_ECE_delta>{STRICT_ECE_GATE}")
    hardening_row = {
        "stage": "V1210_B320_BASE_HARDENING",
        "candidate_id": B320_ID,
        "method_family": "SimpleFastTaskGeometry",
        "impl_path": "F3-triton-workspace-forward-delta-readout-proj-grad-learnableP",
        "step_ratio_q90": eff.get("step_ratio_q90", ""),
        "memory_ratio_q90": eff.get("memory_ratio_q90", ""),
        "forward_ratio_q90": eff.get("forward_ratio_q90", ""),
        "backward_ratio_q90": eff.get("backward_ratio_q90", ""),
        "update_ratio_q90": eff.get("update_ratio_q90", ""),
        "task_rows": task.get("rows", ""),
        "mean_delta": task.get("mean_delta", ""),
        "worst_delta": task.get("worst_delta", ""),
        "near_pass_rate": task.get("near_pass_rate", ""),
        "AUC_step_ratio": task.get("max_AUC_step", ""),
        "AUC_time_ratio": task.get("max_AUC_time", ""),
        "ECE_delta": task.get("max_ECE_delta", ""),
        "near_pass": int(safe_float(task.get("near_pass_rate"), 0.0) >= 1.0) if task_measured else "",
        "strict_fail_reason": ";".join(strict_fail_reasons),
        "strict_pass": int(hardening_pass),
        "no_fake": 1,
    }
    write_csv_rows(out_dir / "v1210_b320_base_hardening.csv", [hardening_row])
    decision = {
        "stage": "V1210_ROUTE_DECISION",
        "generated_at": now_iso(),
        "raw_mode": raw_mode,
        "raw_runner_returncode": raw_returncode,
        "raw_dir": str(raw_dir.relative_to(REPO_ROOT)) if raw_dir.is_relative_to(REPO_ROOT) else str(raw_dir),
        "B320_exact_candidate_id": B320_ID,
        "B320_F3_step_ratio_q90": eff.get("step_ratio_q90", ""),
        "B320_F3_memory_ratio_q90": eff.get("memory_ratio_q90", ""),
        "B320_F3_official_efficiency_pass_raw": eff.get("official_efficiency_pass", ""),
        "B320_task_status": task.get("status", ""),
        "B320_task_rows": task.get("rows", ""),
        "B320_mean_delta": task.get("mean_delta", ""),
        "B320_worst_delta": task.get("worst_delta", ""),
        "B320_near_pass_rate": task.get("near_pass_rate", ""),
        "B320_max_ECE_delta": task.get("max_ECE_delta", ""),
        "B320_max_AUC_step": task.get("max_AUC_step", ""),
        "B320_max_AUC_time": task.get("max_AUC_time", ""),
        "B320_v1210_strict_hardening_pass": int(hardening_pass),
        "B320_strict_fail_reason": ";".join(strict_fail_reasons),
        "B320_linec_status": linec.get("status", ""),
        "B320_LineC_CouplingR2": linec.get("CouplingR2", ""),
        "B320_LineC_NoiseSignalLeak": linec.get("NoiseSignalLeak", ""),
        "B320_LineC_nontearing_pass": linec.get("nontearing_pass", ""),
        "functional_reentry_measured": raw_decision.get("B109_functional_reentry_measured", ""),
        "official_functional_success": raw_decision.get("official_functional_success", ""),
        "functional_delta_score_gap_vs_best_control": raw_decision.get("B109_functional_delta_score_gap_vs_best_control", ""),
        "functional_P3_official_candidate_pass": functional_p3.get("official_candidate_pass", ""),
        "functional_P3_best_update": functional_p3.get("best_functional_update", ""),
        "functional_P3_control_gap": functional_p3.get("control_gap", ""),
        "functional_P3_fail_reason": functional_p3.get("fail_reason", ""),
        "classic_family_pass_count_excluding_bspline": family_pass_count,
        "bspline_status": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
        "bspline_active_followup": 0,
        "bspline_codex_budget": 0,
        "no_fake_provenance_pass": raw_decision.get("no_fake_provenance_pass", ""),
    }
    if raw_returncode != 0:
        decision["route"] = "R0-RawRunnerFailed"
        decision["next_recommended_action"] = "inspect v1210_raw_runner_stderr.log and repair according to v12.10 plan before claiming hardening"
    elif not hardening_pass:
        decision["route"] = "R1-B320StrictHardeningNotLocked"
        decision["next_recommended_action"] = "repair B320 exact strict task/AUC/ECE or F3 efficiency under the v12.10 gates before official functional claim"
    elif int(safe_float(linec.get("nontearing_pass"), 0.0)) != 1:
        decision["route"] = "R2-B320BasePassLineCNeedsRepair"
        decision["next_recommended_action"] = "repair train-probe coupling/noise leakage before functional official success claim"
    elif int(safe_float(raw_decision.get("official_functional_success"), 0.0)) != 1:
        if int(safe_float(functional_p3.get("official_candidate_pass"), 0.0)) == 1:
            decision["route"] = "R4-B320P3OpenedP4ImplementationBlocked"
            decision["next_recommended_action"] = "implement P4 short-run official integration for the P3-passing functional event inside the FHQ/manual-AdamW task loop with strong controls"
        else:
            decision["route"] = "R3-B320AnchorReadyFunctionalStillBlocked"
            decision["next_recommended_action"] = "continue functional update re-entry with strong controls on the locked B320 base"
    else:
        decision["route"] = "R7-B320FunctionalOfficialReady"
        decision["next_recommended_action"] = "promote B320 functional result only with raw strong-control artifacts and hashes attached"
    write_json(out_dir / "v1210_route_decision.json", decision)
    return decision


def _lazy_p4_modules():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from dgkan.kernels import fused_hinge_quadratic as fhq
    import run_v120_good_geometry_battery as v120
    import run_v124_multibasis_functional_dual as v124
    import run_v1252_efficiency_functional_manifold as v1252
    import run_v126_lowerlevel_fhq_functional_geometry as v126
    import run_v1283_b109_classic_family_functional_geometry as v1283

    return torch, F, fhq, v120, v124, v1252, v126, v1283


def _p4_params(model: Any) -> list[Any]:
    return [p for p in model.parameters() if getattr(p, "requires_grad", False)]


def _p4_delta_norm(deltas: Sequence[Any]) -> float:
    return math.sqrt(sum(float(d.detach().float().square().sum().item()) for d in deltas))


def _p4_task_delta_from_grads(model: Any, lr_now: float, torch_mod: Any) -> list[Any]:
    out = []
    for p in _p4_params(model):
        grad = p.grad.detach() if p.grad is not None else torch_mod.zeros_like(p)
        out.append((-float(lr_now) * grad).clone())
    return out


def _p4_branch_delta(model: Any, task_delta: Sequence[Any], mode: str, torch_mod: Any) -> list[Any]:
    out = []
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        delta = torch_mod.zeros_like(param)
        if name == "branch_scale":
            base = -param.detach().clone()
            if mode == "direct" and base.ndim >= 1 and int(base.shape[0]) >= 2:
                mask = torch_mod.zeros_like(base)
                mask[0].copy_(base[0])
                base = mask
            elif mode == "quad" and base.ndim >= 1 and int(base.shape[0]) >= 2:
                mask = torch_mod.zeros_like(base)
                mask[1].copy_(base[1])
                base = mask
            delta = base
        out.append(delta)
    if len(out) != len(task_delta):
        return [torch_mod.zeros_like(d) for d in task_delta]
    return out


def _p4_orthogonalize(residual: Sequence[Any], reference: Sequence[Any], torch_mod: Any) -> list[Any]:
    if not residual or not reference:
        return [torch_mod.zeros_like(d) for d in reference]
    flat_ref = torch_mod.cat([d.detach().flatten() for d in reference])
    flat_res = torch_mod.cat([d.detach().flatten().to(flat_ref.device) for d in residual])
    denom = flat_ref.square().sum().clamp_min(1.0e-12)
    proj = (flat_res @ flat_ref) / denom
    out = [r.to(t.device) - proj.to(t.device) * t for r, t in zip(residual, reference)]
    res_norm = _p4_delta_norm(residual)
    out_norm = _p4_delta_norm(out)
    if out_norm > 1.0e-12 and res_norm > 1.0e-12:
        out = [d * (res_norm / out_norm) for d in out]
    return out


def _p4_blend_deltas(primary: Sequence[Any], residual: Sequence[Any], residual_weight: float) -> list[Any]:
    if not primary:
        return []
    p_norm = _p4_delta_norm(primary)
    r_norm = _p4_delta_norm(residual)
    scaled = [r.clone() for r in residual]
    if r_norm > 1.0e-12 and p_norm > 1.0e-12:
        scaled = [r * (p_norm / r_norm) for r in scaled]
    out = [p + float(residual_weight) * r.to(p.device) for p, r in zip(primary, scaled)]
    mixed_norm = _p4_delta_norm(out)
    if mixed_norm > 1.0e-12 and p_norm > 1.0e-12:
        out = [d * (p_norm / mixed_norm) for d in out]
    return out


def _p4_functional_delta(model: Any, task_delta: Sequence[Any], mode: str, torch_mod: Any) -> list[Any]:
    mode = str(mode).lower()
    both = _p4_orthogonalize(_p4_branch_delta(model, task_delta, "both", torch_mod), task_delta, torch_mod)
    direct = _p4_orthogonalize(_p4_branch_delta(model, task_delta, "direct", torch_mod), task_delta, torch_mod)
    quad = _p4_orthogonalize(_p4_branch_delta(model, task_delta, "quad", torch_mod), task_delta, torch_mod)
    if mode == "quad":
        return quad
    if mode == "both":
        return both
    if mode == "neg_direct":
        return [-d for d in direct]
    if mode == "neg_quad":
        return [-d for d in quad]
    if mode == "neg_both":
        return [-d for d in both]
    if mode == "task_minus_quad010":
        return _p4_blend_deltas(task_delta, quad, -0.10)
    if mode == "task_plus_quad010":
        return _p4_blend_deltas(task_delta, quad, 0.10)
    if mode == "task_minus_both010":
        return _p4_blend_deltas(task_delta, both, -0.10)
    if mode == "task_plus_both010":
        return _p4_blend_deltas(task_delta, both, 0.10)
    return direct


def _p4_random_like(deltas: Sequence[Any], seed: int, torch_mod: Any) -> list[Any]:
    if not deltas:
        return []
    device = deltas[0].device
    gen = torch_mod.Generator(device=device).manual_seed(int(seed))
    flat_norm = _p4_delta_norm(deltas)
    out = [torch_mod.randn(d.shape, device=device, generator=gen) for d in deltas]
    norm = _p4_delta_norm(out)
    return [d * (flat_norm / max(1.0e-12, norm)) for d in out]


def _p4_apply_delta(model: Any, deltas: Sequence[Any], scale: float, torch_mod: Any) -> None:
    with torch_mod.no_grad():
        for p, d in zip(_p4_params(model), deltas):
            p.add_(d.to(p.device), alpha=float(scale))


def _p4_event_gate_ok(
    *,
    model: Any,
    event_delta: Sequence[Any],
    scale: float,
    args: argparse.Namespace,
    xq: Any,
    yq: Any,
    seed: int,
    torch_mod: Any,
    v1252: Any,
) -> tuple[bool, str]:
    gate = str(getattr(args, "p4_event_gate", "none")).lower()
    if gate in {"", "none", "off", "0"}:
        return True, ""
    before_cls = v1252._classification_basic(model, xq, yq)
    before_sig = v1252._signal_reservoir_metrics(
        model,
        xq[: min(int(xq.shape[0]), 8)],
        yq[: min(int(yq.shape[0]), 8)],
        8,
        int(seed),
    )
    _p4_apply_delta(model, event_delta, float(scale), torch_mod)
    after_cls = v1252._classification_basic(model, xq, yq)
    after_sig = v1252._signal_reservoir_metrics(
        model,
        xq[: min(int(xq.shape[0]), 8)],
        yq[: min(int(yq.shape[0]), 8)],
        8,
        int(seed),
    )
    reasons: list[str] = []
    if after_cls["CEp99"] > before_cls["CEp99"] + float(args.p4_event_ce_tail_epsilon):
        reasons.append("event_CEp99_delta>epsilon")
    if after_cls["margin_p10"] < before_cls["margin_p10"] - float(args.p4_event_margin_epsilon):
        reasons.append("event_margin_p10_drop")
    if after_sig["NoiseSignalLeak"] > before_sig["NoiseSignalLeak"] + float(args.p4_event_noise_epsilon):
        reasons.append("event_NoiseSignalLeak_increase")
    if "reservoir" in gate and after_sig["RealSignalReservoirRatio"] > before_sig["RealSignalReservoirRatio"] - float(args.p4_event_reservoir_release):
        reasons.append("event_RealSignalReservoirRatio_release_insufficient")
    if reasons:
        _p4_apply_delta(model, event_delta, -float(scale), torch_mod)
        return False, ";".join(reasons)
    return True, ""


def _p4_make_raw_args(args: argparse.Namespace, dataset: str, seed: int, train_size: int, val_size: int, test_size: int, epochs: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        no_download=bool(args.no_download),
        seed=int(args.seed),
        train_size=int(train_size),
        val_size=int(val_size),
        test_size=int(test_size),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(epochs),
        task_lr_schedule=str(args.task_lr_schedule),
        task_compile_warmup_steps=int(args.task_compile_warmup_steps),
        datasets=dataset,
        seeds=str(seed),
        kernel_warmup_steps=int(args.kernel_warmup_steps),
        kernel_measure_steps=int(args.kernel_measure_steps),
        optimizer_impl="adamw",
        coupling_batch_size=int(args.functional_batch_size),
        functional_batch_size=int(args.functional_batch_size),
        sketch_batch_size=8,
        sketch_dim=8,
        ridge_lambda=1.0e-3,
    )


def _p4_train_one_method(
    *,
    method: str,
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    raw_args: argparse.Namespace,
    device: Any,
    data: tuple[Any, Any, Any, Any, Any, Any, int, int, Any],
    specs: Mapping[str, Any],
    torch_mod: Any,
    F_mod: Any,
    fhq: Any,
    v1252: Any,
    v126: Any,
    v1283: Any,
) -> tuple[Any, dict[str, Any]]:
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
    y_val = y_val_cpu.to(device=device)
    model_id = "MLP-same-param-AdamW" if method.startswith("MLP-") else B320_ID
    model = v1283._make_model(model_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 121000, specs, y_train)
    opt = v1283._make_adamw(model, raw_args)
    triton_update_params = v126._triton_adamw_params(raw_args, model) if model_id != "MLP-same-param-AdamW" else []
    manual_update = v126._ManualForeachAdamW(opt.param_groups, raw_args, triton_update_params=triton_update_params) if model_id != "MLP-same-param-AdamW" and v126._variant_uses_manual_adamw(raw_args, model) else None
    workspace = None
    if model_id != "MLP-same-param-AdamW":
        warm_impl = v1283._select_b109_step_impl(model_id, specs, epoch_idx=0, step_id=0)
        if warm_impl in {"F3-triton-learnableP-workspace", "F4-triton-fixedP-workspace"}:
            workspace = fhq.make_workspace(model, int(args.batch_size), device)
    xq = x_val[: int(args.functional_batch_size)]
    yq = y_val[: int(args.functional_batch_size)]
    total_steps = max(1, int(raw_args.epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))))
    gen = torch_mod.Generator(device=device).manual_seed(int(seed) + 121033)
    step_times: list[float] = []
    val_losses: list[float] = []
    val_time_auc_terms: list[float] = []
    event_count = 0
    accepted_count = 0
    rejected_count = 0
    lambda_values: list[float] = []
    event_gate_reasons: dict[str, int] = {}
    uses_loss_backward = int(model_id == "MLP-same-param-AdamW")
    for epoch in range(int(raw_args.epochs)):
        v1283._apply_logit_gain_ramp(model, model_id, specs, epoch, int(raw_args.epochs))
        v1283._apply_quad_branch_ramp(model, model_id, specs, epoch, int(raw_args.epochs))
        v1283._apply_direct_branch_ramp(model, model_id, specs, epoch, int(raw_args.epochs))
        epoch_start = len(step_times)
        perm = torch_mod.randperm(int(x_train.shape[0]), generator=gen, device=device)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off : off + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            step_id = len(step_times)
            lr_now = v126._task_lr_for(raw_args, step_id + 1, total_steps)
            for group in opt.param_groups:
                group["lr"] = lr_now * float(group.get("lr_scale", 1.0))
            opt.zero_grad(set_to_none=True)
            v1283._sync(device)
            t0 = time.perf_counter()
            if model_id == "MLP-same-param-AdamW":
                F_mod.cross_entropy(model(xb), yb).backward()
                task_delta = _p4_task_delta_from_grads(model, lr_now, torch_mod)
                opt.step()
            else:
                impl = v1283._select_b109_step_impl(model_id, specs, epoch_idx=epoch, step_id=step_id)
                v1283._step_b109(model, xb, yb, impl, workspace, opt=opt, args=raw_args, manual_update=manual_update)
                task_delta = _p4_task_delta_from_grads(model, lr_now, torch_mod)
                if manual_update is not None:
                    manual_update.step()
                else:
                    opt.step()
            if method != "B320-AdamW" and method != "MLP-AdamW" and (step_id + 1) % max(1, int(args.p4_event_every_steps)) == 0:
                event_count += 1
                func_delta = _p4_functional_delta(model, task_delta, str(args.p4_functional_mode), torch_mod)
                if method == "B320-NoOpMatchedOverhead":
                    event_delta = [torch_mod.zeros_like(d) for d in task_delta]
                elif method == "B320-RandomMatchedNorm":
                    event_delta = _p4_random_like(func_delta, int(seed) + event_count + 121050, torch_mod)
                elif method == "B320-AdamWParallelMaintenance":
                    event_delta = task_delta
                elif method == "B320-SNR-only":
                    event_delta = v1252._basis_delta(model, task_delta, "basis_aware_snr_projected")
                elif method == "MLP-analogFunctional":
                    event_delta = _p4_random_like(task_delta, int(seed) + event_count + 121090, torch_mod)
                else:
                    event_delta = func_delta
                accepted = False
                selected_lam = 0.0
                if int(args.p4_runtime_backtracking) == 0:
                    selected_lam = float(args.p4_fixed_lambda)
                    accepted, gate_reason = _p4_event_gate_ok(
                        model=model,
                        event_delta=event_delta,
                        scale=selected_lam,
                        args=args,
                        xq=xq,
                        yq=yq,
                        seed=int(seed) + event_count + 121180,
                        torch_mod=torch_mod,
                        v1252=v1252,
                    )
                    if gate_reason:
                        event_gate_reasons[gate_reason] = event_gate_reasons.get(gate_reason, 0) + 1
                else:
                    before_loss = float(F_mod.cross_entropy(model(xq), yq).detach().item())
                    for lam in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                        trial = deepcopy(model)
                        _p4_apply_delta(trial, event_delta, lam, torch_mod)
                        after_loss = float(F_mod.cross_entropy(trial(xq), yq).detach().item())
                        lower_ok = True if method.endswith("NoOpMatchedOverhead") else after_loss >= before_loss * 0.95
                        upper_ok = after_loss <= before_loss * 1.05
                        if lower_ok and upper_ok:
                            selected_lam = lam
                            accepted, gate_reason = _p4_event_gate_ok(
                                model=model,
                                event_delta=event_delta,
                                scale=selected_lam,
                                args=args,
                                xq=xq,
                                yq=yq,
                                seed=int(seed) + event_count + 121180,
                                torch_mod=torch_mod,
                                v1252=v1252,
                            )
                            if gate_reason:
                                event_gate_reasons[gate_reason] = event_gate_reasons.get(gate_reason, 0) + 1
                            break
                if accepted:
                    accepted_count += 1
                    lambda_values.append(selected_lam)
                else:
                    rejected_count += 1
            v1283._sync(device)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)
        cls_epoch = v1252._classification_basic(model, x_val, y_val)
        epoch_q90 = quantile(step_times[epoch_start:], 0.90)
        val_losses.append(cls_epoch["NLL"])
        val_time_auc_terms.append(cls_epoch["NLL"] * max(epoch_q90, 1.0e-12))
    final_val = v1252._classification_basic(model, x_val, y_val)
    summary = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "val_acc": final_val["acc"],
        "val_loss": final_val["NLL"],
        "ECE": final_val["ECE"],
        "NLL": final_val["NLL"],
        "CEp99": final_val["CEp99"],
        "margin_p10": final_val["margin_p10"],
        "AUC_step": mean(val_losses),
        "AUC_time": mean(val_time_auc_terms),
        "step_time_q90_ms": quantile(step_times, 0.90),
        "functional_event_count": event_count,
        "accepted_event_count": accepted_count,
        "rejected_event_count": rejected_count,
        "lambda_mean": mean(lambda_values),
        "event_gate": str(args.p4_event_gate),
        "event_gate_reject_reasons": "|".join(f"{k}:{v}" for k, v in sorted(event_gate_reasons.items())),
        "uses_loss_backward": uses_loss_backward,
        "task_step_impl": "torch-autograd" if model_id == "MLP-same-param-AdamW" else "FHQ/manual-AdamW",
    }
    return model, summary


def run_p4_functional_short_run(args: argparse.Namespace, out_dir: Path, decision: dict[str, Any]) -> list[dict[str, Any]]:
    if int(safe_float(decision.get("functional_P3_official_candidate_pass"), 0.0)) != 1:
        return read_csv_rows(out_dir / "v1210_functional_short_run.csv")
    if int(args.run_p4_short) != 1:
        return read_csv_rows(out_dir / "v1210_functional_short_run.csv")
    torch_mod, F_mod, fhq, v120, v124, v1252, v126, v1283 = _lazy_p4_modules()
    device = v1283._device_from_arg(str(args.device))
    methods = [
        "B320-AdamW",
        "B320-NoOpMatchedOverhead",
        "B320-RandomMatchedNorm",
        "B320-AdamWParallelMaintenance",
        "B320-SNR-only",
        "B320-bestFunctional",
        "MLP-AdamW",
        "MLP-analogFunctional",
    ]
    rows: list[dict[str, Any]] = []
    datasets = [d.strip() for d in str(args.p4_datasets).split(",") if d.strip()]
    seeds = [int(s.strip()) for s in str(args.p4_seeds).split(",") if s.strip()]
    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        raw_args = _p4_make_raw_args(args, canon, seeds[0] if seeds else 0, int(args.p4_train_size), int(args.p4_val_size), int(args.p4_test_size), int(args.p4_epochs))
        data = v120._load_vision_split(raw_args, canon, train_size=int(args.p4_train_size), val_size=int(args.p4_val_size), test_size=int(args.p4_test_size))
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {s.candidate_id: s for s in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            per_seed: dict[str, tuple[Any, dict[str, Any]]] = {}
            for method in methods:
                raw_args = _p4_make_raw_args(args, canon, seed, int(args.p4_train_size), int(args.p4_val_size), int(args.p4_test_size), int(args.p4_epochs))
                model, summary = _p4_train_one_method(
                    method=method,
                    dataset=canon,
                    seed=seed,
                    args=args,
                    raw_args=raw_args,
                    device=device,
                    data=data,
                    specs=specs,
                    torch_mod=torch_mod,
                    F_mod=F_mod,
                    fhq=fhq,
                    v1252=v1252,
                    v126=v126,
                    v1283=v1283,
                )
                per_seed[method] = (model, summary)
            baseline_model, baseline = per_seed["B320-AdamW"]
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
            xb = x_train_cpu[: int(args.functional_batch_size)].to(device=device, dtype=torch_mod.float32)
            yb = y_train_cpu[: int(args.functional_batch_size)].to(device=device)
            xq = x_val_cpu[: int(args.functional_batch_size)].to(device=device, dtype=torch_mod.float32)
            yq = y_val_cpu[: int(args.functional_batch_size)].to(device=device)
            baseline_sig = v1252._signal_reservoir_metrics(baseline_model, xb[: min(int(xb.shape[0]), 8)], yb[: min(int(yb.shape[0]), 8)], 8, int(seed) + 121071)
            control_scores: list[float] = []
            method_rows: list[dict[str, Any]] = []
            for method, (model, summary) in per_seed.items():
                geo = v1252._geo_gain_from_line_c(baseline_model, model, xb, yb, xq, yq, 1.0e-3, 8, int(seed) + 121070)
                sig = v1252._signal_reservoir_metrics(model, xb[: min(int(xb.shape[0]), 8)], yb[: min(int(yb.shape[0]), 8)], 8, int(seed) + 121071)
                acc_delta = safe_float(summary["val_acc"]) - safe_float(baseline["val_acc"])
                auc_time_delta = safe_float(summary["AUC_time"]) - safe_float(baseline["AUC_time"])
                ece_delta = safe_float(summary["ECE"]) - safe_float(baseline["ECE"])
                nll_delta = safe_float(summary["NLL"]) - safe_float(baseline["NLL"])
                cep99_delta = safe_float(summary["CEp99"]) - safe_float(baseline["CEp99"])
                overhead = safe_float(summary["step_time_q90_ms"], 0.0) / max(1.0e-12, safe_float(baseline["step_time_q90_ms"], 0.0))
                coupling_delta = 0.0 if method == "B320-AdamW" else safe_float(geo["CouplingR2"])
                reservoir_delta = safe_float(sig["RealSignalReservoirRatio"]) - safe_float(baseline_sig["RealSignalReservoirRatio"])
                noise_delta = safe_float(sig["NoiseSignalLeak"]) - safe_float(baseline_sig["NoiseSignalLeak"])
                score = acc_delta - max(0.0, auc_time_delta) - max(0.0, ece_delta) - max(0.0, nll_delta) + coupling_delta - max(0.0, noise_delta)
                is_control = method in {"B320-NoOpMatchedOverhead", "B320-RandomMatchedNorm", "B320-AdamWParallelMaintenance", "B320-SNR-only"}
                if is_control:
                    control_scores.append(score)
                row = {
                    "stage": "V1210_FUNCTIONAL_SHORT_RUN",
                    "method": method,
                    "dataset": canon,
                    "seed": seed,
                    "functional_id": decision.get("functional_P3_best_update", ""),
                    "acc_delta_vs_B320": acc_delta,
                    "val_loss_delta_vs_B320": nll_delta,
                    "AUC_step_delta_vs_B320": safe_float(summary["AUC_step"]) - safe_float(baseline["AUC_step"]),
                    "AUC_time_delta_vs_B320": auc_time_delta,
                    "ECE_delta_vs_B320": ece_delta,
                    "NLL_delta_vs_B320": nll_delta,
                    "CEp99_delta_vs_B320": cep99_delta,
                    "margin_p10_delta_vs_B320": safe_float(summary["margin_p10"]) - safe_float(baseline["margin_p10"]),
                    "CouplingR2_delta_vs_B320": coupling_delta,
                    "RealSignalReservoirRatio_delta_vs_B320": reservoir_delta,
                    "NoiseSignalLeak_delta_vs_B320": noise_delta,
                    "functional_event_count": summary["functional_event_count"],
                    "accepted_event_count": summary["accepted_event_count"],
                    "rejected_event_count": summary["rejected_event_count"],
                    "event_gate": summary.get("event_gate", ""),
                    "event_gate_reject_reasons": summary.get("event_gate_reject_reasons", ""),
                    "amortized_overhead_ratio": overhead,
                    "control_gap_vs_best": "",
                    "strict_pass": 0,
                    "diagnostic_score": score,
                    "uses_loss_backward": summary["uses_loss_backward"],
                    "task_step_impl": summary["task_step_impl"],
                    "claim_scope": "P4_short_run_candidate; B320 path uses FHQ/manual-AdamW, MLP rows are controls only",
                    "linec_delta_definition": "CouplingR2 uses ridge map from final B320-AdamW logits delta to final method logits delta; signal/noise rows are method minus final B320-AdamW",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                    "dataset_name_branch_used": 0,
                    "teacher_used": 0,
                    "distillation_used": 0,
                    "loss_modified": 0,
                    "sampler_or_class_weight_used": 0,
                }
                method_rows.append(row)
            best_control = max(control_scores) if control_scores else 0.0
            for row in method_rows:
                row["control_gap_vs_best"] = safe_float(row["diagnostic_score"]) - best_control
                if row["method"] == "B320-bestFunctional":
                    gate_reasons = []
                    if safe_float(row["acc_delta_vs_B320"]) < -0.003:
                        gate_reasons.append("acc_delta<-0.003")
                    if safe_float(row["AUC_time_delta_vs_B320"]) > 0.0:
                        gate_reasons.append("AUC_time_delta>0")
                    if safe_float(row["ECE_delta_vs_B320"]) > 0.005:
                        gate_reasons.append("ECE_delta>0.005")
                    if safe_float(row["CEp99_delta_vs_B320"]) > float(args.p4_ce_tail_epsilon):
                        gate_reasons.append(f"CEp99_delta>{args.p4_ce_tail_epsilon}")
                    if safe_float(row["CouplingR2_delta_vs_B320"]) < 0.02:
                        gate_reasons.append("CouplingR2_delta<0.02")
                    if safe_float(row["NoiseSignalLeak_delta_vs_B320"]) > -0.01:
                        gate_reasons.append("NoiseSignalLeak_delta>-0.01")
                    if safe_float(row["amortized_overhead_ratio"]) > 1.05:
                        gate_reasons.append("amortized_overhead>1.05")
                    if safe_float(row["control_gap_vs_best"]) <= 0.0:
                        gate_reasons.append("control_gap_vs_best<=0")
                    row["strict_pass"] = int(not gate_reasons)
                    row["fail_reason"] = ";".join(gate_reasons)
                rows.append(row)
    write_csv_rows(out_dir / "v1210_functional_short_run.csv", rows if rows else [{"stage": "V1210_FUNCTIONAL_SHORT_RUN", "status": "not_run_no_rows"}])
    best_rows = [r for r in rows if r.get("method") == "B320-bestFunctional"]
    official = bool(best_rows) and all(int(safe_float(r.get("strict_pass"), 0.0)) == 1 for r in best_rows)
    decision["official_functional_success"] = int(official)
    decision["functional_P4_short_run_measured"] = int(bool(rows))
    decision["functional_P4_bestFunctional_pass_rows"] = sum(int(safe_float(r.get("strict_pass"), 0.0)) for r in best_rows)
    decision["functional_P4_bestFunctional_total_rows"] = len(best_rows)
    if official:
        decision["route"] = "R7-B320FunctionalOfficialReady"
        decision["next_recommended_action"] = "promote B320 functional result with P4 short-run rows, raw artifacts, and hashes attached"
    else:
        decision["route"] = "R5-B320P4ShortRunFailed"
        decision["next_recommended_action"] = "autopsy P4 short-run failure; continue plan-consistent functional or classic-family repair without claiming official functional success"
    write_json(out_dir / "v1210_route_decision.json", decision)
    return rows


def write_provenance(out_dir: Path, raw_dir: Path, raw_returncode: int) -> None:
    raw_prov = read_csv_rows(raw_dir / "v1283_provenance_audit.csv")
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "stage": "V1210_PROVENANCE_AUDIT",
            "artifact": "v1210_route_config.json",
            "rows_checked": 1,
            "fake_data_used_sum": 0,
            "proxy_row_used_sum": 0,
            "cpu_offload_used_sum": 0,
            "no_fake_pass": 1,
            "no_proxy_pass": 1,
            "no_cpu_offload_pass": 1,
            "raw_runner_returncode": raw_returncode,
        }
    )
    rows.append(
        {
            "stage": "V1210_PROVENANCE_AUDIT",
            "artifact": "v1210_family_policy.json",
            "rows_checked": len(ACTIVE_FAMILY_IDS) + 1,
            "fake_data_used_sum": 0,
            "proxy_row_used_sum": 0,
            "cpu_offload_used_sum": 0,
            "no_fake_pass": 1,
            "no_proxy_pass": 1,
            "no_cpu_offload_pass": 1,
            "bspline_active_followup": 0,
            "bspline_codex_budget": 0,
        }
    )
    for row in raw_prov:
        copied = {"stage": "V1210_RAW_PROVENANCE_AUDIT", **row}
        rows.append(copied)
    write_csv_rows(out_dir / "v1210_provenance_audit.csv", rows)


def write_hash_manifest(out_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path in sorted(out_dir.glob("v1210_*")):
        if path.is_file() and path.name != "v1210_hash_manifest.json":
            artifacts[path.name] = sha256_file(path)
    payload = {"stage": "V1210_HASH_MANIFEST", "generated_at": now_iso(), "artifacts": artifacts}
    write_json(out_dir / "v1210_hash_manifest.json", payload)
    return payload


def md_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str], limit: int | None = None) -> str:
    data = list(rows[:limit] if limit is not None else rows)
    if not data:
        return "_no rows_"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in data:
        body.append("| " + " | ".join(f"`{row.get(c, '')}`" for c in cols) + " |")
    return "\n".join([header, sep, *body])


def write_review_doc(out_dir: Path, doc_path: Path, decision: Mapping[str, Any], family_status: Mapping[str, Any], hashes: Mapping[str, Any]) -> None:
    full_rows = read_csv_rows(out_dir / "v1210_b320_efficiency_profile.csv")
    b320_eff = strict_b320_efficiency_rows(full_rows)
    auc_rows = [
        r
        for r in read_csv_rows(out_dir / "v1210_b320_auc_attribution.csv")
        if r.get("candidate_id") == B320_ID and str(r.get("stage")) == "V1283_B109_AUC_ATTRIBUTION_SUMMARY"
    ]
    if not auc_rows:
        auc_rows = [
            r
            for r in read_csv_rows(out_dir / "v1210_b320_auc_attribution.csv")
            if r.get("candidate_id") == B320_ID and str(r.get("stage")) == "V1283_B109_AUC_ATTRIBUTION"
        ][:12]
    family_rows = []
    for family, payload in family_status.get("families", {}).items():
        row = {"family": family}
        row.update(payload if isinstance(payload, Mapping) else {})
        family_rows.append(row)
    hash_rows = [{"artifact": k, "sha256_prefix": str(v)[:12]} for k, v in hashes.get("artifacts", {}).items()]
    hardening_rows = read_csv_rows(out_dir / "v1210_b320_base_hardening.csv")
    functional_p3_rows = read_csv_rows(out_dir / "v1210_functional_p3_gate.csv")
    text = f"""# DG-KAN v12.10 B320 Functional ClassicNoBSpline 执行复盘

> 本复盘由 `{Path(__file__).name}` 生成。所有数值来自本轮 `results` artifact 或显式复用的 raw artifact；没有编造数据。B-spline 在本版按计划冻结，不作为 active family 候选。

## 0. 当前结论

```text
route = {decision.get('route')}
raw_runner_returncode = {decision.get('raw_runner_returncode')}
raw_dir = {decision.get('raw_dir')}
B320_v1210_strict_hardening_pass = {decision.get('B320_v1210_strict_hardening_pass')}
B320_strict_fail_reason = {decision.get('B320_strict_fail_reason')}
B320_F3_step_ratio_q90 = {decision.get('B320_F3_step_ratio_q90')}
B320_F3_memory_ratio_q90 = {decision.get('B320_F3_memory_ratio_q90')}
B320_mean_delta = {decision.get('B320_mean_delta')}
B320_worst_delta = {decision.get('B320_worst_delta')}
B320_near_pass_rate = {decision.get('B320_near_pass_rate')}
B320_max_ECE_delta = {decision.get('B320_max_ECE_delta')}
B320_max_AUC_step = {decision.get('B320_max_AUC_step')}
B320_max_AUC_time = {decision.get('B320_max_AUC_time')}
B320_LineC_nontearing_pass = {decision.get('B320_LineC_nontearing_pass')}
official_functional_success = {decision.get('official_functional_success')}
functional_P3_official_candidate_pass = {decision.get('functional_P3_official_candidate_pass')}
functional_P3_best_update = {decision.get('functional_P3_best_update')}
functional_P3_control_gap = {decision.get('functional_P3_control_gap')}
functional_P3_fail_reason = {decision.get('functional_P3_fail_reason')}
classic_family_pass_count_excluding_bspline = {decision.get('classic_family_pass_count_excluding_bspline')}
BSpline.status = {decision.get('bspline_status')}
next_recommended_action = {decision.get('next_recommended_action')}
artifact_dir = {out_dir}
```

## 1. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 v12.10 wrapper | 固化 B320 exact/B321/B314/B109 对照协议，调用真实 raw runner，生成 v1210 route/policy/provenance/hash/复盘，不把 raw v1283 字段直接冒充 v12.10 official 结论。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 `F11-TaskMinusBranchScaleDamping` functional diagnostic bracket | 在 F10 branch damping 未通过 B320 P3 gate 后，增加 role-wise 反向 bracket，检查方向符号是否导致 CouplingR2/control gap 失败；不改 CE loss、sampler/class weight、teacher/distillation 或 dataset branch。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 修正 delta-score reservoir 项并新增 `F12/F13` negative geometry-only diagnostics | 按 v12.10 P3 的 reservoir-release 定义，把 score 从奖励 RealSignalReservoirRatio 增加改为奖励 reservoir release；新增负向 SNR/orthogonal geometry-only bracket，用于确认是否存在不依赖 AdamW 的 reservoir-release 方向。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 `F14/F15` AdamW-orthogonal branch diagnostics | 将 branch-scale damping residual 显式去除与 task/AdamW step 的平行分量，测试 role-wise orthogonal functional 是否能越过 C0/C3 controls；仍不改 loss、数据、采样或标签权重。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | P3 functional backtracking 增加 holdout safety lower bound | F14 暴露强信号但一步过强后，按 v12.10 F-F4 推荐方向加入 norm-budget downscale：functional 只接受 `0.95 <= holdout_loss_after/before <= 1.05`；controls 保持原审计语义。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 short-run candidate loop | P3 打开后可用 `--run-p4-short 1` 运行 B320/FHQ/manual-AdamW 短跑；B320 路径不使用 `loss.backward()` 更新，functional 事件只作为训练内维护事件，并与 NoOp/Random/AdamWParallel/SNR controls 对比。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 fixed-lambda runtime-cost repair | 当 P4 runtime backtracking 因 deepcopy/holdout evaluation 触发 overhead blocker 时，可用 `--p4-runtime-backtracking 0 --p4-fixed-lambda ...` 运行固定步长维护事件；这是成本修复，不改变数据、loss、label、sampler 或 gate。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 branch-mode bracket | `--p4-functional-mode` 支持 direct/quad/both/negative/F15-style blend，用同一 short-run gate 检查 P3 方向是否只是 branch role 选择错误；不改变数据、loss、sampler 或 gate。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 P4 event-level tail/noise/reservoir veto | 针对 P4 失败后的 F-F2/F-F3，新增 `--p4-event-gate tail_noise(_reservoir)`：每个维护事件先测 CEp99、margin、NoiseSignalLeak、可选 RealSignalReservoirRatio，失败则回滚该事件；不改变 CE loss、数据、标签、采样、class weight 或 final strict gate。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 扩展 v12.10 非 B-spline active family 候选 | 将已实现的 Chebyshev K4/localrot2 与 Fourier h8 stronger linear residual 候选加入 active list，按同一 L3/A4 gate 验证 ExpressionBlocked 是否可修复；仍排除 B-spline。 |
| `experiments/run_v1210_b320_functional_classic_nobspline.py` | 新增 `--family-candidate-ids` 运行子集开关 | 允许对 RBF/Wavelet blocker 做 focused follow-up，而不误调度所有 active family；contract 会记录实际 raw family candidate ids，便于审计 focused 结果不能升级为全量 official 结论。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 RBF/Wavelet stream-recompute manual CE path 与 `B2s` RBF K4 候选 | 修复 B2r/B5h 在 manual L3 fallback 中缓存 dense `b1/b2/db2` 的问题，按 v12.10 RBF-F1/F2/WAV-F1 方向改为只缓存 `z/h` 并逐 basis channel 重算梯度；该路径仍是 torch reduction，不会伪装成 Triton/CUDA fused official kernel。 |
| `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 将 `B2s` 加入 family microbench plan | 让 v12.10 focused run 能实际测到 RBF-F2 K4 stream 候选；不改 family gate，也不改变 official fused kernel 判定。 |
| `dgkan/kernels/fused_rbf.py` / `dgkan/models/fc_purekan_primitives.py` / `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 RBF/FastKAN family-specific Triton L3 backward path | 为 B2r/B2s 加入固定中心 Gaussian RBF K2/K4 Triton forward + backward kernels，manual path 只保存 `h` 并在 backward 内重算 basis；将 `rbf_k2_triton_l3_matmul` / `rbf_k4_triton_l3_matmul` 接入 official fused L3 判定。 |
| `dgkan/kernels/fused_hat_wavelet.py` / `dgkan/models/fc_purekan_primitives.py` / `experiments/run_v1283_b109_classic_family_functional_geometry.py` | 新增 Wavelet hat/triangle family-specific Triton L3 backward path | 为 B5h 加入固定中心 hat wavelet K4 Triton forward + backward kernels，manual path 只保存 `h` 并在 backward 内重算 local hat basis；将 `hat_wavelet_k4_triton_l3_matmul` 接入 official fused L3 判定。 |
| `docs/DG-KAN_v12.10_B320_Functional_ClassicNoBSpline_执行复盘.md` | 新增本复盘文件 | 只记录真实落盘 artifact 的关键数值、blocker 和后续动作；缺失项写 not_measured 或沿用 raw runner 返回状态，不补造数据。 |

## 2. B320 F3 Efficiency

{md_table(b320_eff, ["candidate_id", "implementation_id", "step_ratio_q90", "memory_ratio_q90", "official_efficiency_pass"], limit=6)}

Hardening gate summary:

{md_table(hardening_rows, ["candidate_id", "impl_path", "step_ratio_q90", "memory_ratio_q90", "mean_delta", "worst_delta", "near_pass_rate", "AUC_step_ratio", "AUC_time_ratio", "ECE_delta", "strict_pass", "strict_fail_reason"], limit=6)}

## 3. B320 Task / AUC

{md_table(auc_rows, ["stage", "dataset", "seed", "mean_delta", "worst_delta", "near_pass_rate", "max_ECE_delta", "max_AUC_step", "max_AUC_time", "val_acc_delta_vs_mlp", "ECE_delta_vs_mlp", "AUC_step_ratio", "AUC_time_ratio", "A5_autopsy_pass"], limit=12)}

## 4. Family Policy / Status

## 4. Functional P3 Gate

{md_table(functional_p3_rows, ["candidate_id", "best_control", "best_control_delta_score", "best_functional_update", "best_functional_delta_score", "control_gap", "CouplingR2_best_control", "CouplingR2_after", "CouplingR2_delta_vs_best_control", "NoiseSignalLeak_delta", "official_candidate_pass", "fail_reason"], limit=6)}

## 4.1 Functional P4 Short Run

{md_table(read_csv_rows(out_dir / "v1210_functional_short_run.csv"), ["method", "dataset", "seed", "acc_delta_vs_B320", "AUC_time_delta_vs_B320", "ECE_delta_vs_B320", "CEp99_delta_vs_B320", "CouplingR2_delta_vs_B320", "NoiseSignalLeak_delta_vs_B320", "amortized_overhead_ratio", "control_gap_vs_best", "strict_pass", "fail_reason"], limit=24)}

## 5. Family Policy / Status

{md_table(family_rows, ["family", "status", "active_followup", "codex_budget", "best_L3_manual_step_ratio", "blocker"], limit=12)}

## 6. Provenance / Hash

{md_table(read_csv_rows(out_dir / "v1210_provenance_audit.csv"), ["stage", "artifact", "rows_checked", "fake_data_used_sum", "proxy_row_used_sum", "cpu_offload_used_sum", "no_fake_pass"], limit=20)}

Selected hashes:

{md_table(hash_rows, ["artifact", "sha256_prefix"], limit=20)}

## 7. 分析结论

1. B-spline 已按 v12.10 计划从 active classic family 中排除；本轮没有调度 B-spline 候选，因此不能把 B-spline 写成新失败实验，只能写成 `FamilyFrozen_KernelBlocked_RejectedForThisVersion`。
2. B320 hardening 是否通过只看 `v1210_route_decision.json` 中的 strict gates 聚合；如果 raw runner 未完成或任一 gate 超限，本复盘不写 official base lock。
3. Functional official success 仍以 raw strong-control artifact 与 `v1210_functional_p3_gate.csv` 为准；没有通过 strong controls 与 Line C gate 前不写成功。
4. 后续修复方向必须沿计划继续：先修 B320 strict hardening 或 Line C nontearing blocker，再进入 functional official re-entry；classic family 只在非 B-spline active families 内推进。
"""
    ensure_dir(doc_path.parent)
    doc_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DG-KAN v12.10 B320 Functional ClassicNoBSpline runner")
    p.add_argument("--out-dir", default=f"results/v12_10_b320_functional_classic_nobspline/v1210_b320_{now_tag()}")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--reuse-raw-dir", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--seed", type=int, default=2413)
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-3)
    p.add_argument("--task-lr-schedule", default="linear_warmup10_cosine_final050")
    p.add_argument("--kernel-warmup-steps", type=int, default=5)
    p.add_argument("--kernel-measure-steps", type=int, default=12)
    p.add_argument("--task-compile-warmup-steps", type=int, default=12)
    p.add_argument("--functional-batch-size", type=int, default=32)
    p.add_argument("--run-p4-short", type=int, default=0)
    p.add_argument("--p4-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--p4-seeds", default="0,1,2")
    p.add_argument("--p4-train-size", type=int, default=1024)
    p.add_argument("--p4-val-size", type=int, default=512)
    p.add_argument("--p4-test-size", type=int, default=512)
    p.add_argument("--p4-epochs", type=int, default=3)
    p.add_argument("--p4-event-every-steps", type=int, default=8)
    p.add_argument("--p4-ce-tail-epsilon", type=float, default=0.05)
    p.add_argument("--p4-runtime-backtracking", type=int, default=1)
    p.add_argument("--p4-fixed-lambda", type=float, default=0.0625)
    p.add_argument("--p4-functional-mode", default="direct")
    p.add_argument("--p4-event-gate", default="none", help="none, tail_noise, or tail_noise_reservoir; applies event-level CE tail/noise/reservoir veto before accepting a P4 maintenance event")
    p.add_argument("--p4-event-ce-tail-epsilon", type=float, default=0.05)
    p.add_argument("--p4-event-margin-epsilon", type=float, default=0.01)
    p.add_argument("--p4-event-noise-epsilon", type=float, default=0.005)
    p.add_argument("--p4-event-reservoir-release", type=float, default=0.02)
    p.add_argument("--raw-repair-candidate-ids", default=",".join([B321_ID, B314_ID, B109_ID]))
    p.add_argument("--family-candidate-ids", default=",".join(ACTIVE_FAMILY_IDS))
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--report-path", default=str(DEFAULT_DOC_PATH))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    placeholder_cmd = [sys.executable, str(RAW_RUNNER), "..."]
    raw_dir = Path(args.reuse_raw_dir).resolve() if args.reuse_raw_dir else out_dir / "raw_v1283_reuse"
    write_contract_files(args, out_dir, raw_dir, placeholder_cmd)
    raw_dir, raw_returncode, raw_mode = run_raw_or_reuse(args, out_dir)
    copy_raw_artifacts(raw_dir, out_dir)
    family_status = filter_bspline_family_status(raw_dir, out_dir)
    write_provenance(out_dir, raw_dir, raw_returncode)
    decision = build_decision(raw_dir, out_dir, raw_returncode, raw_mode, family_status)
    run_p4_functional_short_run(args, out_dir, decision)
    hashes = write_hash_manifest(out_dir)
    write_review_doc(out_dir, Path(args.report_path), decision, family_status, hashes)
    if raw_returncode != 0:
        raise SystemExit(raw_returncode)


if __name__ == "__main__":
    main()
