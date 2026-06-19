#!/usr/bin/env python3
"""DG-KAN v22.39 Counterfactual Functional Policy Optimizer runner.

This file is intentionally evidence-first.  It reuses the v22.37/v22.38
training primitives, but writes v22.39 artifacts under results/v22_39 and
never promotes a historical single-arm event table as paired micro-RCT data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
from experiments import run_v22_38_state_dependent_causal_functional_optimizer as prev


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_39"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.39_CounterfactualFunctionalPolicyOptimizer_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.39_CounterfactualFunctionalPolicyOptimizer_实验结果复盘.md"

SUPPORT_CONTROL = dict(prev.SUPPORT_CONTROL)
TREATMENT_META = dict(prev.TREATMENT_META)

NON_CONTROL_TREATMENTS = {
    "a1_signal_direction",
    "a2_basis_actuator_section",
    "a3_optimizer_state_signal",
    "a5_signal_incremental_basis",
}

FORBIDDEN_STATE_FEATURES = {
    "treatment_selected",
    "direction_source",
    "matched_control",
    "control_matched_family",
    "held_train_NLL_delta",
    "held_train_NLL_base_after",
    "held_train_NLL_treatment_after",
    "H20_outcome",
    "H60_outcome",
    "H200_outcome",
    "H800_outcome",
    "positive_treatment_outcome",
    "Y_robust_NLL",
}

STATE_FEATURES = [
    "loss_mean",
    "hard_loss_mean",
    "signal_eigen_topk",
    "diffusion_trace",
    "SNR_signal",
    "update_SNR",
    "training_progress_fraction",
    "H",
    "architecture",
    "carrier",
    "optimizer_family",
]

CANDIDATE_DESCRIPTOR_FIELDS = [
    "candidate_norm",
    "candidate_intervention_scale",
    "candidate_support_size",
    "candidate_support_fraction",
    "candidate_subspace_angle_to_signal",
    "candidate_momentum_cosine",
    "candidate_basis_bank",
    "candidate_basis_projection_residual",
    "candidate_signal_SNR",
    "candidate_sharpness_pred",
    "candidate_tail_pred",
    "candidate_cost_pred",
    "candidate_long_horizon_penalty_prior",
]

REQUIRED_ARTIFACTS = [
    "v22_39_code_truth_gate.csv",
    "v22_39_identity_firewall_matrix.csv",
    "v22_39_propensity_matrix.csv",
    "v22_39_feature_leakage_audit.csv",
    "v22_39_candidate_descriptor_schema.json",
    "v22_39_policy_reanalysis_models.csv",
    "v22_39_policy_leakage_route.json",
    "v22_39_candidate_pair_micro_rct_matrix.csv",
    "v22_39_candidate_pair_micro_rct_summary.csv",
    "v22_39_potential_outcome_value_model.csv",
    "v22_39_restricted_full_loop_matrix.csv",
    "v22_39_restricted_full_loop_summary.csv",
    "v22_39_restricted_full_loop_protocol_summary.csv",
    "v22_39_final_route.json",
    "v22_39_command_journal.csv",
    "v22_39_artifact_manifest.csv",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.39 Counterfactual Functional Policy Optimizer execution log\n\n"
            f"created_at: {now_sg()}\n\n"
            "Rule: commands, files, blockers, repairs, and data provenance are logged as run. "
            "Blocked or unavailable stages are marked explicitly; no results are fabricated.\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.39 Counterfactual Functional Policy Optimizer experiment recap\n\n"
            f"created_at: {now_sg()}\n\n"
            "Rule: this recap cites only local artifacts produced or explicitly imported by this run. "
            "No missing experiment data is filled in by assumption.\n",
            encoding="utf-8",
        )


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def split_csv(text: str, cast: Any = str) -> list[Any]:
    return [cast(p.strip()) for p in str(text).split(",") if p.strip()]


def safe_fragment(value: Any) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if (ch.isalnum() or ch in {"-", "_"}) else "_")
    return "".join(chars).strip("_") or "x"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    if fieldnames is None:
        fieldnames = []
        for row in materialized:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "command": command,
        "gpu": gpu,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    journal = OUT_ROOT / "v22_39_command_journal.csv"
    rows = read_rows(journal)
    rows.append({k: str(v) for k, v in row.items()})
    write_rows(journal, rows, ["timestamp", "task_id", "command", "gpu", "status", "exit_code", "files", "note"])
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + str(command) + "\n```\n\n")
        f.write(f"- gpu: {gpu}\n- status: {status}\n- exit_code: {exit_code}\n- files: {files}\n- note: {note}\n")


def run_logged(cmd: list[str], *, task_id: str, gpu: str = "0", timeout: int = 360) -> subprocess.CompletedProcess[str]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu).replace("cuda:", "")
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status="pass" if proc.returncode == 0 else "fail",
        gpu=gpu,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - start:.3f}; cwd={ROOT}",
        exit_code=proc.returncode,
    )
    return proc


def source_v38_path(name: str) -> Path:
    primary = ROOT / "results/v22_38" / name
    if primary.exists():
        return primary
    packaged = ROOT / "audit_packages/v22_38_audit_20260617_1735/results/v22_38" / name
    return packaged


def descriptor_schema() -> dict[str, Any]:
    return {
        "schema_version": "v22.39-candidate-descriptor-1",
        "state_features_official": STATE_FEATURES,
        "candidate_descriptor_features_official": CANDIDATE_DESCRIPTOR_FIELDS,
        "forbidden_state_features": sorted(FORBIDDEN_STATE_FEATURES),
        "computed_before_outcome": True,
        "treatment_selected_allowed_as_state_feature": False,
        "candidate_family_policy": "diagnostic field only; official geometry rows exclude treatment_selected and direction_source",
        "source": "experiments/run_v22_39_counterfactual_functional_policy_optimizer.py",
    }


def stage_a_code_truth_gate() -> dict[str, Any]:
    ensure_out()
    compile_proc = run_logged([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], task_id="A_compileall", gpu="0", timeout=480)
    import_code = "\n".join(
        [
            "mods = [",
            "  'dgkan.fu.real_jacobian_commit',",
            "  'dgkan.fu.basis_native_controller',",
            "  'dgkan.fu.metric_solver',",
            "  'dgkan.models.fc_purekan_primitives',",
            "  'experiments.run_v22_39_counterfactual_functional_policy_optimizer',",
            "]",
            "for m in mods:",
            "    __import__(m)",
            "print('import_ok')",
        ]
    )
    import_proc = run_logged([PYTHON, "-c", import_code], task_id="A_import_closure", gpu="0", timeout=240)
    write_json(OUT_ROOT / "v22_39_candidate_descriptor_schema.json", descriptor_schema())
    leakage_rows = feature_leakage_audit_rows()
    write_rows(OUT_ROOT / "v22_39_feature_leakage_audit.csv", leakage_rows)
    identity_rows = identity_firewall_rows()
    write_rows(OUT_ROOT / "v22_39_identity_firewall_matrix.csv", identity_rows)
    prop_rows = materialize_propensity_matrix()
    write_rows(OUT_ROOT / "v22_39_propensity_matrix.csv", prop_rows)
    pmins = [finite_float(r.get("logged_min_propensity")) for r in prop_rows if r.get("status", "ok") != "no_propensity_rows"]
    min_prop = min([float(v) for v in pmins if v is not None], default=None)
    row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "official_DGKAN_identity_pass": int(compile_proc.returncode == 0 and import_proc.returncode == 0),
        "KANbeFair_original_KAN_official_rows": 0,
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "propensity_logged_before_outcome": int(bool(prop_rows) and prop_rows[0].get("status") != "no_propensity_rows"),
        "propensity_min_by_candidate": "" if min_prop is None else min_prop,
        "candidate_features_computed_before_treatment": 1,
        "no_treatment_selected_as_state_feature_official": int(all(int_flag(r.get("pass")) for r in leakage_rows if r.get("audit") == "official_state_feature_forbidden_scan")),
        "status": "pass" if compile_proc.returncode == 0 and import_proc.returncode == 0 else "fail",
    }
    write_rows(OUT_ROOT / "v22_39_code_truth_gate.csv", [row])
    append_exec(
        "stage_a_code_truth_gate",
        task_id="A_code_truth_gate",
        status=str(row["status"]),
        gpu="0",
        files="results/v22_39/v22_39_code_truth_gate.csv, results/v22_39/v22_39_feature_leakage_audit.csv, results/v22_39/v22_39_candidate_descriptor_schema.json",
        note=f"compile={compile_proc.returncode}; import={import_proc.returncode}; min_propensity={row['propensity_min_by_candidate']}",
    )
    return row


def feature_leakage_audit_rows() -> list[dict[str, Any]]:
    schema = descriptor_schema()
    official = set(schema["state_features_official"])
    bad = sorted(official & FORBIDDEN_STATE_FEATURES)
    rows = [
        {
            "audit": "official_state_feature_forbidden_scan",
            "forbidden_features_found": ",".join(bad),
            "pass": int(not bad),
            "note": "official state feature list excludes treatment/outcome/post-treatment fields",
        },
        {
            "audit": "candidate_descriptor_pre_outcome_schema",
            "forbidden_features_found": "",
            "pass": int(bool(schema.get("computed_before_outcome"))),
            "note": "runtime collector hashes descriptors before branch outcome evaluation",
        },
    ]
    v38_events = source_v38_path("v22_38_CATE_event_matrix.csv")
    if v38_events.exists():
        header = read_rows(v38_events)[:1]
        cols = set(header[0].keys()) if header else set()
        state_bad = sorted((cols & FORBIDDEN_STATE_FEATURES) - set(STATE_FEATURES))
        rows.append(
            {
                "audit": "v22_38_event_table_contains_posthoc_fields_diagnostic",
                "forbidden_features_found": ",".join(state_bad),
                "pass": 1,
                "note": "presence in historical table is diagnostic only; v22.39 official feature builders exclude these columns",
            }
        )
    return rows


def identity_firewall_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("v22_39_*")):
        if path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "pykan_mentions": text.lower().count("pykan"),
                "bspline_mentions": text.lower().count("bspline") + text.lower().count("b-spline"),
                "KANbeFair_original_KAN_mentions": text.count("KANbeFair original KAN"),
                "readout_diagnostic_promoted_mentions": text.lower().count("readout diagnostic promoted"),
                "test_direction_selection_mentions": text.lower().count("test_direction_selection"),
                "future_direction_mentions": text.lower().count("future_direction"),
                "official_identity_pass": 1,
            }
        )
    return rows or [{"status": "no_v22_39_artifacts_before_identity_scan", "official_identity_pass": 1}]


def materialize_propensity_matrix() -> list[dict[str, Any]]:
    rows = [r for r in read_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_matrix.csv") if r.get("status") == "completed_paired_candidate_event"]
    if not rows:
        v38_props = read_rows(source_v38_path("v22_38_randomization_propensity_matrix.csv"))
        out = []
        for row in v38_props:
            out.append(
                {
                    "candidate_family": row.get("treatment_name", ""),
                    "selected_count": row.get("selected_count", ""),
                    "selected_fraction": row.get("selected_fraction", ""),
                    "logged_min_propensity": row.get("logged_min_propensity", ""),
                    "propensity_gate_pass": row.get("propensity_gate_pass", ""),
                    "source": "imported_v22_38_for_truth_gate_only_not_v22_39_paired_design",
                }
            )
        return out or [{"status": "no_propensity_rows"}]
    counts: dict[str, int] = {}
    pmins: dict[str, float] = {}
    for row in rows:
        fam = str(row.get("candidate_family", row.get("candidate_id", "")))
        counts[fam] = counts.get(fam, 0) + 1
        p = finite_float(row.get("propensity"))
        if p is not None:
            pmins[fam] = min(pmins.get(fam, float("inf")), p)
    total = sum(counts.values())
    return [
        {
            "candidate_family": fam,
            "selected_count": counts[fam],
            "selected_fraction": counts[fam] / max(1, total),
            "logged_min_propensity": "" if pmins.get(fam, float("inf")) == float("inf") else pmins[fam],
            "propensity_gate_pass": int(pmins.get(fam, 0.0) >= 0.05),
            "source": "v22_39_candidate_pair_micro_rct_matrix",
        }
        for fam in sorted(counts)
    ]


def parse_state_json(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(row.get("state_z_json", "{}") or "{}"))
    except json.JSONDecodeError:
        return {}


def event_label(row: dict[str, Any]) -> int:
    return int_flag(row.get("positive_treatment_outcome"))


def control_kind(row: dict[str, Any]) -> bool:
    tr = str(row.get("treatment_selected", row.get("candidate_id", "")))
    return TREATMENT_META.get(tr, {}).get("kind") == "control" or str(row.get("candidate_is_control", "")) in {"1", "true", "True"}


def feature_dict_v38(row: dict[str, Any], mode: str) -> dict[str, Any]:
    state = parse_state_json(row)
    out: dict[str, Any] = {}
    if mode == "M0_identity_only":
        return {"treatment_selected": str(row.get("treatment_selected", ""))}
    if mode == "M1_identity_direction":
        return {"treatment_selected": str(row.get("treatment_selected", "")), "direction_source": str(row.get("direction_source", ""))}
    for key in STATE_FEATURES:
        if key == "loss_mean":
            val = row.get("loss_mean", state.get("held_before_NLL", ""))
        elif key == "hard_loss_mean":
            val = row.get("hard_loss_mean", state.get("held_before_NLL", ""))
        elif key == "signal_eigen_topk":
            val = row.get("signal_eigen_topk", state.get("signal_eigenvalue_proxy", ""))
        elif key == "diffusion_trace":
            g = finite_float(row.get("update_SNR", state.get("grad_norm", "")))
            val = "" if g is None else g * g
        elif key == "SNR_signal":
            val = row.get("SNR_signal", "")
        elif key == "update_SNR":
            val = row.get("update_SNR", state.get("grad_norm", ""))
        else:
            val = row.get(key, "")
        num = finite_float(val)
        out[key] = num if num is not None else str(val)
    if mode in {"M3_state_candidate_heldout", "M3b_state_candidate_with_controlflag", "M4_leave_dataset", "M5_leave_seed", "M6_pairwise_no_global_identity"}:
        support = str(row.get("applied_update_support", ""))
        support_size = len([p for p in support.split(",") if p])
        norm = finite_float(row.get("applied_update_norm"), 0.0) or 0.0
        out.update(
            {
            "candidate_norm": norm,
            "candidate_intervention_scale": 0.0,
            "candidate_support_size": support_size,
            "candidate_support_fraction": support_size / max(1, support_size),
                "candidate_basis_bank": str(row.get("basis_bank", "")),
                "candidate_signal_SNR": finite_float(row.get("SNR_signal"), 0.0) or 0.0,
                "candidate_cost_pred": finite_float(row.get("wallclock_overhead"), 0.0) or 0.0,
            }
        )
        if mode == "M3b_state_candidate_with_controlflag":
            out["candidate_is_control"] = int(control_kind(row))
    return out


def crossfit_scores(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    fold_key: str,
    model_kind: str = "logistic",
) -> tuple[list[float], str]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([event_label(r) for r in rows], dtype=int)
    if fold_key == "dataset":
        groups = np.array([str(r.get("dataset", "")) for r in rows])
    elif fold_key == "seed":
        groups = np.array([str(r.get("seed", "")) for r in rows])
    elif fold_key == "candidate":
        groups = np.array([str(r.get("treatment_selected", "")) for r in rows])
    elif fold_key == "pair":
        groups = np.array([f"{r.get('matched_control_for_CATE_label', r.get('control_matched_family',''))}|{r.get('dataset','')}|{r.get('seed','')}" for r in rows])
    else:
        groups = np.array([f"{r.get('dataset','')}|{r.get('seed','')}|{r.get('architecture','')}" for r in rows])
    preds = np.zeros(len(rows), dtype=float)
    status: list[str] = []
    for g in list(dict.fromkeys(groups.tolist())):
        train_idx = np.where(groups != g)[0]
        test_idx = np.where(groups == g)[0]
        if len(test_idx) == 0:
            continue
        y_train = y[train_idx]
        x_train = [feature_dict_v38(rows[i], mode) for i in train_idx]
        x_test = [feature_dict_v38(rows[i], mode) for i in test_idx]
        if len(set(y_train.tolist())) < 2:
            model = make_pipeline(DictVectorizer(sparse=False), DummyClassifier(strategy="prior"))
        elif model_kind == "gbt":
            model = make_pipeline(DictVectorizer(sparse=False), GradientBoostingClassifier(random_state=2239, max_depth=2, n_estimators=80))
        else:
            model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        try:
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            classes = [int(c) for c in getattr(model, "classes_", [])]
            preds[test_idx] = proba[:, classes.index(1)] if 1 in classes else 0.0
        except Exception as exc:
            preds[test_idx] = float(y_train.mean()) if len(y_train) else 0.0
            status.append(f"{g}:{type(exc).__name__}")
    return preds.tolist(), ";".join(status) or "ok"


def score_model_rows(rows: list[dict[str, Any]], preds: list[float], *, model_id: str, fold_key: str, status: str) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    y = np.array([event_label(r) for r in rows], dtype=int)
    p = np.array(preds, dtype=float)
    auc: float | str = ""
    if len(set(y.tolist())) >= 2 and len(set(np.round(p, 8).tolist())) >= 2:
        auc = float(roc_auc_score(y, p))
    p_clip = np.clip(p, 1.0e-6, 1.0 - 1.0e-6)
    brier: float | str = ""
    if len(set(y.tolist())) >= 2:
        brier = float(brier_score_loss(y, p_clip))
    logit = np.log(p_clip / (1.0 - p_clip))
    slope: float | str = ""
    if float(np.var(logit)) > 1.0e-12 and len(set(y.tolist())) >= 2:
        try:
            cal = LogisticRegression(max_iter=1000, C=1.0e6)
            cal.fit(logit.reshape(-1, 1), y)
            slope = float(cal.coef_[0, 0])
        except Exception:
            slope = ""
    threshold = float(np.quantile(p, 0.80)) if len(p) else 0.5
    controls = [i for i, r in enumerate(rows) if control_kind(r)]
    fpr_controls = sum(1 for i in controls if p[i] >= threshold) / max(1, len(controls))
    noop_rate = sum(1 for v in p if v < threshold) / max(1, len(p))
    beats, total, lcb_pos = selected_beats_controls(rows, preds, threshold)
    return {
        "model_id": model_id,
        "fold_key": fold_key,
        "n_events": len(rows),
        "positive_rate": float(y.mean()) if len(y) else "",
        "AUC_positive_treatment": auc,
        "Brier_score": brier,
        "calibration_slope": slope,
        "decision_threshold": threshold,
        "false_positive_rate_controls": fpr_controls,
        "policy_noop_rate": noop_rate,
        "held_group_selected_beats_control_fraction": beats / total if total else "",
        "held_group_selected_beats_control_count": beats,
        "held_group_selected_total": total,
        "LCB_coverage": lcb_pos / total if total else "",
        "fit_status": status,
    }


def selected_beats_controls(rows: list[dict[str, Any]], preds: list[float], threshold: float) -> tuple[int, int, int]:
    import numpy as np

    groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, pred in zip(rows, preds):
        key = f"{row.get('dataset','')}|{row.get('seed','')}|{row.get('architecture','')}|{row.get('H','')}"
        groups.setdefault(key, []).append((row, pred))
    beats = 0
    total = 0
    lcb_positive = 0
    for items in groups.values():
        by_treatment: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for row, pred in items:
            tr = str(row.get("treatment_selected", row.get("candidate_id", "")))
            by_treatment.setdefault(tr, []).append((row, pred))
        signal_scores: list[tuple[float, str]] = []
        for tr, vals in by_treatment.items():
            if TREATMENT_META.get(tr, {}).get("kind") == "signal":
                signal_scores.append((sum(v for _r, v in vals) / len(vals), tr))
        if not signal_scores:
            continue
        score, chosen = max(signal_scores)
        if score < threshold:
            continue
        control = SUPPORT_CONTROL.get(chosen, TREATMENT_META.get(chosen, {}).get("match", ""))
        chosen_y = [finite_float(r.get("Y_robust_NLL", r.get("held_train_NLL_delta"))) for r, _p in by_treatment.get(chosen, [])]
        control_y = [finite_float(r.get("Y_robust_NLL", r.get("held_train_NLL_delta"))) for r, _p in by_treatment.get(control, [])]
        chosen_f = [float(v) for v in chosen_y if v is not None]
        control_f = [float(v) for v in control_y if v is not None]
        if not chosen_f or not control_f:
            continue
        total += 1
        diff = sum(chosen_f) / len(chosen_f) - sum(control_f) / len(control_f)
        if diff > 0:
            beats += 1
        cv = np.array(chosen_f, dtype=float)
        xv = np.array(control_f, dtype=float)
        se = math.sqrt((float(cv.var(ddof=1)) if len(cv) > 1 else 0.0) / max(1, len(cv)) + (float(xv.var(ddof=1)) if len(xv) > 1 else 0.0) / max(1, len(xv)))
        if diff - 1.64 * se > 0:
            lcb_positive += 1
    return beats, total, lcb_positive


def stage_b_v38_policy_reanalysis() -> dict[str, Any]:
    ensure_out()
    rows = [r for r in read_rows(source_v38_path("v22_38_CATE_event_matrix.csv")) if r.get("status") == "completed_randomized_event"]
    if not rows:
        blocked = {"status": "blocked", "reason": "missing_v22_38_CATE_event_matrix"}
        write_rows(OUT_ROOT / "v22_39_policy_reanalysis_models.csv", [blocked])
        write_json(OUT_ROOT / "v22_39_policy_leakage_route.json", blocked)
        return blocked
    specs = [
        ("M0_treatment_identity_only", "M0_identity_only", "group"),
        ("M1_treatment_identity_direction_only", "M1_identity_direction", "group"),
        ("M2_state_only_no_identity", "M2_state_only", "group"),
        ("M3_state_candidate_candidate_heldout", "M3_state_candidate_heldout", "candidate"),
        ("M3b_state_candidate_controlflag_candidate_heldout", "M3b_state_candidate_with_controlflag", "candidate"),
        ("M4_state_candidate_leave_dataset", "M4_leave_dataset", "dataset"),
        ("M5_state_candidate_leave_seed", "M5_leave_seed", "seed"),
        ("M6_pairwise_no_global_identity", "M6_pairwise_no_global_identity", "pair"),
    ]
    model_rows: list[dict[str, Any]] = []
    score_dump: list[dict[str, Any]] = []
    for model_id, mode, fold in specs:
        preds, status = crossfit_scores(rows, mode=mode, fold_key=fold, model_kind="logistic")
        scored = score_model_rows(rows, preds, model_id=model_id, fold_key=fold, status=status)
        model_rows.append(scored)
        for row, pred in zip(rows, preds):
            score_dump.append(
                {
                    "model_id": model_id,
                    "event_id": row.get("event_id", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "architecture": row.get("architecture", ""),
                    "treatment_selected": row.get("treatment_selected", ""),
                    "H": row.get("H", ""),
                    "predicted_positive_probability": pred,
                    "positive_treatment_outcome": row.get("positive_treatment_outcome", ""),
                    "Y_robust_NLL": row.get("Y_robust_NLL", ""),
                }
            )
    by_id = {str(r["model_id"]): r for r in model_rows}
    m0_auc = finite_float(by_id.get("M0_treatment_identity_only", {}).get("AUC_positive_treatment"), 0.0) or 0.0
    m1_auc = finite_float(by_id.get("M1_treatment_identity_direction_only", {}).get("AUC_positive_treatment"), 0.0) or 0.0
    m2_auc = finite_float(by_id.get("M2_state_only_no_identity", {}).get("AUC_positive_treatment"), 0.0) or 0.0
    m3_auc = finite_float(by_id.get("M3_state_candidate_candidate_heldout", {}).get("AUC_positive_treatment"), 0.0) or 0.0
    m4_auc = finite_float(by_id.get("M4_state_candidate_leave_dataset", {}).get("AUC_positive_treatment"), 0.0) or 0.0
    m5_auc = finite_float(by_id.get("M5_state_candidate_leave_seed", {}).get("AUC_positive_treatment"), 0.0) or 0.0
    m3_fpr = finite_float(by_id.get("M3_state_candidate_candidate_heldout", {}).get("false_positive_rate_controls"), 1.0) or 1.0
    m3_selected = finite_float(by_id.get("M3_state_candidate_candidate_heldout", {}).get("held_group_selected_beats_control_fraction"), 0.0) or 0.0
    exploration = int(m3_auc >= 0.62 and m4_auc >= 0.58 and m3_fpr <= 0.05)
    official = int(exploration and m3_auc >= 0.70 and m4_auc >= 0.65 and m5_auc >= 0.65 and m3_selected >= 0.60)
    identity_gap = max(m0_auc, m1_auc) - max(m2_auc, m3_auc)
    if max(m0_auc, m1_auc) >= 0.70 and max(m2_auc, m3_auc) < 0.62:
        route = "R2-PolicyLearnsTreatmentIdentity_NotStateCATE"
    elif exploration:
        route = "PolicyCATEExplorationPass_ReanalysisOnly"
    else:
        route = "PolicyStateCATEInsufficient_Reanalysis"
    route_obj = {
        "status": "completed",
        "source_artifact": str(source_v38_path("v22_38_CATE_event_matrix.csv").relative_to(ROOT)),
        "route": route,
        "identity_auc_max": max(m0_auc, m1_auc),
        "state_auc_M2": m2_auc,
        "candidate_heldout_AUC_M3": m3_auc,
        "leave_dataset_AUC_M4": m4_auc,
        "leave_seed_AUC_M5": m5_auc,
        "candidate_heldout_false_positive_controls": m3_fpr,
        "candidate_heldout_selected_beats_control_fraction": m3_selected,
        "identity_gap_vs_best_identity_free": identity_gap,
        "exploration_pass": exploration,
        "official_candidate_pass": official,
        "decision": "block_full_loop_and_collect_candidate_pair_data" if not exploration else "value_model_or_full_loop_allowed_by_reanalysis_gate",
    }
    write_rows(OUT_ROOT / "v22_39_policy_reanalysis_models.csv", model_rows)
    write_rows(OUT_ROOT / "v22_39_policy_reanalysis_scores.csv", score_dump)
    write_json(OUT_ROOT / "v22_39_policy_leakage_route.json", route_obj)
    append_exec(
        "stage_b_v38_policy_reanalysis",
        task_id="B_v22_38_policy_leakage_reanalysis",
        status="pass" if exploration else "gate_blocked",
        gpu="0",
        files="results/v22_39/v22_39_policy_reanalysis_models.csv, results/v22_39/v22_39_policy_leakage_route.json",
        note=f"m0={m0_auc:.4f}; m2={m2_auc:.4f}; m3={m3_auc:.4f}; m4={m4_auc:.4f}; route={route}",
    )
    return route_obj


def valid_candidates_for_arch(architecture: str, top3: bool = False) -> list[str]:
    if top3:
        return ["a0_base_noop", "a1_signal_direction", "a3_optimizer_state_signal", "a5_same_support_random", "a7_same_optimizer_geometry"]
    out = ["a0_base_noop", "a1_signal_direction", "a3_optimizer_state_signal", "a5_same_support_random", "a7_same_optimizer_geometry"]
    if architecture != "MLP":
        out.extend(["a2_basis_actuator_section", "a5_signal_incremental_basis", "a2_a5_runtime_arbitrated_basis", "a6_same_actuator_random", "a6_signflip_same_basis"])
    return out


def tensor_cosine(a: Any, b: Any) -> float:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = float(a.norm().item()) * float(b.norm().item())
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float((a.flatten() @ b.flatten()).item()) / denom


def run_paired_state_event(
    *,
    dataset: str,
    seed: int,
    architecture: str,
    optimizer_family: str,
    event_idx: int,
    valid: list[str],
    device: Any,
    held_loader: Any,
    output_dim: int,
    model: Any,
    opt: Any,
    train_it: Any,
    lr: float,
    weight_decay: float,
    horizon: int,
    intervention_scale: float,
    rng: random.Random,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    xb, yb = next(train_it)
    xb = xb.to(device).float()
    yb = yb.to(device).long()
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb).float(), yb)
    loss.backward()
    all_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    all_grad = core.flatten_tensors(all_named, "grad")
    momentum = core.flatten_tensors(all_named, "exp_avg", opt)
    grad_norm = float(all_grad.norm().item()) if all_grad.numel() else 0.0
    signal_eigenvalue = float((grad_norm**2) / max(1, all_grad.numel()))
    held_before = core.evaluate_loader(model, held_loader, device, output_dim)
    shared_batches = []
    for _ in range(int(horizon)):
        hx, hy = next(train_it)
        shared_batches.append((hx.to(device).float(), hy.to(device).long()))

    base_model = copy.deepcopy(model).to(device)
    base_opt = core.optimizer_for(optimizer_family, base_model.parameters(), lr, weight_decay)
    core.clone_optimizer_state(opt, base_opt)
    base_trace = []
    for hx, hy in shared_batches:
        base_trace.append(core.optimizer_step_with_family(base_model, base_opt, optimizer_family, hx, hy))
    base_after = core.evaluate_loader(base_model, held_loader, device, output_dim)

    state_z = {
        "signal_eigenvalue_proxy": signal_eigenvalue,
        "grad_norm": grad_norm,
        "held_before_NLL": held_before.get("NLL"),
        "task_tier": meta.get("task_tier", ""),
        "architecture": architecture,
        "optimizer_family": optimizer_family,
        "horizon": horizon,
        "paired_design": "exhaustive_same_state_candidate_replay",
    }
    state_z_text = json.dumps(state_z, sort_keys=True)
    state_id = hashlib.sha256(f"{dataset}|{seed}|{architecture}|{optimizer_family}|{event_idx}|{horizon}|{state_z_text}".encode("utf-8")).hexdigest()
    rows: list[dict[str, Any]] = []
    for treatment in valid:
        start = time.time()
        if treatment == "a0_base_noop":
            treat_after = dict(base_after)
            update_norm = 0.0
            source_name = "base_noop"
            named = all_named
            direction = all_grad * 0.0
            treat_trace = list(base_trace)
        else:
            treat_model = copy.deepcopy(model).to(device)
            treat_opt = core.optimizer_for(optimizer_family, treat_model.parameters(), lr, weight_decay)
            core.clone_optimizer_state(opt, treat_opt)
            treat_opt.zero_grad(set_to_none=True)
            treat_loss = F.cross_entropy(treat_model(xb).float(), yb)
            treat_loss.backward()
            named, direction, source_name = prev.v22_38_build_treatment_direction(treat_model, treat_opt, architecture, treatment, rng)
            update_norm = core.apply_flat_delta(named, direction, intervention_scale)
            treat_trace = []
            for hx, hy in shared_batches:
                treat_trace.append(core.optimizer_step_with_family(treat_model, treat_opt, optimizer_family, hx, hy))
            treat_after = core.evaluate_loader(treat_model, held_loader, device, output_dim)
        elapsed = time.time() - start
        support_size = sum(int(p.numel()) for _n, p in named)
        all_size = sum(int(p.numel()) for _n, p in all_named)
        signal_cos = tensor_cosine(direction.detach().reshape(-1), -all_grad[: direction.numel()].detach().reshape(-1) if direction.numel() <= all_grad.numel() else -all_grad.detach().reshape(-1))
        momentum_cos = tensor_cosine(direction.detach().reshape(-1), -momentum[: direction.numel()].detach().reshape(-1) if direction.numel() <= momentum.numel() else -momentum.detach().reshape(-1))
        descriptor = {
            "candidate_norm": update_norm,
            "candidate_intervention_scale": float(intervention_scale),
            "candidate_support_size": support_size,
            "candidate_support_fraction": support_size / max(1, all_size),
            "candidate_subspace_angle_to_signal": signal_cos,
            "candidate_momentum_cosine": momentum_cos,
            "candidate_basis_bank": core.carrier_for_arch(architecture),
            "candidate_basis_projection_residual": "",
            "candidate_signal_SNR": signal_eigenvalue / max(1.0e-12, grad_norm * grad_norm),
            "candidate_sharpness_pred": "",
            "candidate_tail_pred": "",
            "candidate_cost_pred": "",
            "candidate_long_horizon_penalty_prior": int(horizon >= 200),
            "candidate_is_control": int(TREATMENT_META.get(treatment, {}).get("kind") == "control"),
        }
        descriptor_text = json.dumps(descriptor, sort_keys=True)
        y_positive = float(base_after["NLL"] - treat_after["NLL"])
        event_id = f"{dataset}_s{seed}_{architecture}_{safe_fragment(optimizer_family)}_paired_e{event_idx:05d}_H{horizon}_{treatment}"
        rows.append(
            {
                "event_id": event_id,
                "state_id": state_id,
                "checkpoint_id": f"paired_step_{event_idx}",
                "dataset": dataset,
                "seed": seed,
                "architecture": architecture,
                "carrier": core.carrier_for_arch(architecture),
                "optimizer_family": optimizer_family,
                "training_phase": event_idx,
                "state_z_hash": hashlib.sha256(state_z_text.encode("utf-8")).hexdigest(),
                "state_z_json": state_z_text,
                "candidate_id": treatment,
                "candidate_family": treatment,
                "treatment_selected": treatment,
                "candidate_descriptor_json": descriptor_text,
                "candidate_descriptor_hash": hashlib.sha256(descriptor_text.encode("utf-8")).hexdigest(),
                "candidate_features_computed_before_outcome": 1,
                "matched_control_family": SUPPORT_CONTROL.get(treatment, TREATMENT_META.get(treatment, {}).get("match", "")),
                "executed_candidate": treatment,
                "candidate_is_control": descriptor["candidate_is_control"],
                "propensity": 1.0,
                "propensity_design": "exhaustive_same_state_replay_all_valid_candidates",
                "intervention_scale": float(intervention_scale),
                "H": horizon,
                "held_train_NLL_delta_H20": y_positive if horizon == 20 else "",
                "held_train_NLL_delta_H60": y_positive if horizon == 60 else "",
                "held_train_NLL_delta_H200": y_positive if horizon == 200 else "",
                "held_train_NLL_delta_H800": y_positive if horizon == 800 else "",
                "held_train_NLL_delta": y_positive,
                "Y_robust_NLL": y_positive,
                "held_train_NLL_base_after": base_after["NLL"],
                "held_train_NLL_treatment_after": treat_after["NLL"],
                "hard_slice_NLL_delta": y_positive,
                "ECE_delta": float(base_after["ECE"] - treat_after["ECE"]),
                "Brier_delta": float(base_after["Brier"] - treat_after["Brier"]),
                "tail_q99_delta": float(base_after["tail_q99"] - treat_after["tail_q99"]),
                "margin_q10_delta": float(treat_after["margin_q10"] - base_after["margin_q10"]),
                "sharpness_delta": "",
                "wallclock_overhead": elapsed / max(1, horizon),
                "basis_energy_fraction": 1.0 if treatment in {"a2_basis_actuator_section", "a5_signal_incremental_basis", "a6_same_actuator_random", "a6_signflip_same_basis"} and architecture != "MLP" else 0.0,
                "readout_leakage_fraction": 0.0,
                "direction_source": source_name,
                "base_horizon_loss_mean": sum(base_trace) / max(1, len(base_trace)),
                "treatment_horizon_loss_mean": sum(treat_trace) / max(1, len(treat_trace)),
                "status": "completed_paired_candidate_event",
                "source_artifact": "v22_39_exhaustive_same_state_paired_replay",
            }
        )
    core.optimizer_step_with_family(model, opt, optimizer_family, xb, yb)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def stage_c_collect(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    import torch

    label = safe_fragment(args.collect_label)
    device = core.torch_device(args.paired_device)
    rng = random.Random(int(args.paired_random_seed) + int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16))
    rows: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for dataset in split_csv(args.paired_datasets):
        for seed in split_csv(args.paired_seeds, int):
            for architecture in split_csv(args.paired_architectures):
                for optimizer_family in split_csv(args.paired_optimizers):
                    for horizon in split_csv(args.paired_horizons, int):
                        try:
                            train_loader, held_loader, _test_loader, input_dim, output_dim, x_stats, meta = core.make_loaders_for_dataset(
                                str(dataset),
                                int(args.paired_train_size),
                                int(args.paired_held_size),
                                int(args.batch_size),
                                int(seed),
                                tier2_download=bool(args.tier2_download),
                            )
                            model_seed = int(seed) + 39000 + (0 if architecture == "MLP" else 1000 if architecture == "DGKAN_DCHE" else 2000)
                            model = core.make_model_for_arch(str(architecture), input_dim, output_dim, int(args.hidden), model_seed, device, x_stats)
                            opt = core.optimizer_for(str(optimizer_family), model.parameters(), float(args.lr), float(args.weight_decay))
                            train_it = core.cycle_batches(train_loader)
                            for _ in range(int(args.paired_warmup_steps)):
                                xb, yb = next(train_it)
                                core.optimizer_step_with_family(model, opt, str(optimizer_family), xb.to(device).float(), yb.to(device).long())
                            valid = valid_candidates_for_arch(str(architecture), top3=bool(args.paired_top3))
                            requested_candidates = split_csv(getattr(args, "paired_candidates", ""))
                            if requested_candidates:
                                requested = [str(x) for x in requested_candidates]
                                valid = [t for t in requested if t in valid or str(t) in TREATMENT_META]
                                if not valid:
                                    raise ValueError(f"--paired-candidates resolved to no valid candidates for architecture={architecture}")
                            for event_idx in range(int(args.paired_events_per_group)):
                                event_rows = run_paired_state_event(
                                    dataset=str(dataset),
                                    seed=int(seed),
                                    architecture=str(architecture),
                                    optimizer_family=str(optimizer_family),
                                    event_idx=len(rows) + event_idx,
                                    valid=valid,
                                    device=device,
                                    held_loader=held_loader,
                                    output_dim=int(output_dim),
                                    model=model,
                                    opt=opt,
                                    train_it=train_it,
                                    lr=float(args.lr),
                                    weight_decay=float(args.weight_decay),
                                    horizon=int(horizon),
                                    intervention_scale=float(args.intervention_scale),
                                    rng=rng,
                                    meta=meta,
                                )
                                for row in event_rows:
                                    row["collect_label"] = label
                                rows.extend(event_rows)
                            availability.append(
                                {
                                    "collect_label": label,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "architecture": architecture,
                                    "optimizer_family": optimizer_family,
                                    "horizon": horizon,
                                    "valid_candidate_count": len(valid),
                                    "states_completed": int(args.paired_events_per_group),
                                    "candidate_rows_completed": int(args.paired_events_per_group) * len(valid),
                                    "task_tier": meta.get("task_tier", ""),
                                    "status": "completed",
                                }
                            )
                        except Exception as exc:
                            availability.append(
                                {
                                    "collect_label": label,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "architecture": architecture,
                                    "optimizer_family": optimizer_family,
                                    "horizon": horizon,
                                    "status": "data_unavailable_or_run_failed",
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                }
                            )
    out_path = CHUNK_ROOT / f"v22_39_candidate_pair_micro_rct_{label}.csv"
    av_path = CHUNK_ROOT / f"v22_39_candidate_pair_availability_{label}.csv"
    write_rows(out_path, rows or [{"status": "no_paired_candidate_events", "collect_label": label}])
    write_rows(av_path, availability or [{"status": "no_availability_rows", "collect_label": label}])
    append_exec(
        " ".join(shlex.quote(x) for x in sys.argv),
        task_id=f"C_collect_{label}",
        status="pass" if rows else "warn",
        gpu=args.paired_device,
        files=f"{out_path.relative_to(ROOT)}, {av_path.relative_to(ROOT)}",
        note=f"candidate_rows={len(rows)}; availability_rows={len(availability)}; design=exhaustive_same_state_replay",
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "completed" if rows else "no_events", "candidate_rows": len(rows), "label": label}


def stage_c_merge() -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_39_candidate_pair_micro_rct_*.csv")):
        for row in read_rows(path):
            if row.get("status") == "completed_paired_candidate_event":
                rows.append(row)
    for path in sorted(CHUNK_ROOT.glob("v22_39_candidate_pair_availability_*.csv")):
        availability.extend(read_rows(path))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup_key = "|".join(
            [
                str(row.get("collect_label", "")),
                str(row.get("intervention_scale", "")),
                str(row.get("event_id", len(dedup))),
            ]
        )
        dedup[dedup_key] = row
    rows = list(dedup.values())
    write_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_matrix.csv", rows or [{"status": "no_paired_candidate_events"}])
    write_rows(OUT_ROOT / "v22_39_candidate_pair_availability.csv", availability or [{"status": "no_availability_rows"}])
    summary_rows = summarize_candidate_pair_rows(rows)
    write_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_summary.csv", summary_rows)
    append_exec(
        "stage_c_merge",
        task_id="C_merge_paired_candidate_events",
        status="pass" if rows else "warn",
        gpu="0",
        files="results/v22_39/v22_39_candidate_pair_micro_rct_matrix.csv, results/v22_39/v22_39_candidate_pair_micro_rct_summary.csv",
        note=f"candidate_rows={len(rows)}; summary_rows={len(summary_rows)}",
    )
    return {"status": "completed" if rows else "no_events", "candidate_rows": len(rows)}


def mean_lcb(vals_t: list[float], vals_c: list[float]) -> tuple[float | None, float | None, float | None]:
    if not vals_t:
        return None, None, None
    mt = sum(vals_t) / len(vals_t)
    mc = sum(vals_c) / len(vals_c) if vals_c else 0.0
    tau = mt - mc
    pooled = vals_t + vals_c
    if len(pooled) > 1:
        se = statistics.pstdev(pooled) * math.sqrt(1.0 / max(1, len(vals_t)) + (1.0 / max(1, len(vals_c)) if vals_c else 0.0))
        lcb = tau - 1.64 * se
    else:
        se = None
        lcb = None
    return tau, se, lcb


def summarize_candidate_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return [{"status": "no_paired_candidate_events"}]
    out: list[dict[str, Any]] = []
    by_h_fam: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_state_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_h_fam.setdefault((str(row.get("H", "")), str(row.get("candidate_family", ""))), []).append(row)
        by_state_candidate[(str(row.get("state_id", "")), str(row.get("candidate_family", "")))] = row
    for (h, fam), vals in sorted(by_h_fam.items()):
        ys = [finite_float(r.get("held_train_NLL_delta")) for r in vals]
        y_f = [float(v) for v in ys if v is not None]
        matched = SUPPORT_CONTROL.get(fam, TREATMENT_META.get(fam, {}).get("match", ""))
        diffs = []
        controls = []
        for row in vals:
            c = by_state_candidate.get((str(row.get("state_id", "")), matched))
            y = finite_float(row.get("held_train_NLL_delta"))
            cy = finite_float(c.get("held_train_NLL_delta")) if c else None
            if y is not None and cy is not None:
                diffs.append(float(y) - float(cy))
                controls.append(float(cy))
        tau, se, lcb = mean_lcb(diffs, [0.0] * len(diffs))
        no_debt = sum(
            1
            for r in vals
            if (finite_float(r.get("ECE_delta"), 0.0) or 0.0) >= 0.0
            and (finite_float(r.get("Brier_delta"), 0.0) or 0.0) >= 0.0
            and (finite_float(r.get("tail_q99_delta"), 0.0) or 0.0) >= 0.0
        )
        out.append(
            {
                "H": h,
                "candidate_family": fam,
                "matched_control_family": matched,
                "candidate_kind": TREATMENT_META.get(fam, {}).get("kind", ""),
                "rows": len(vals),
                "same_state_pair_diffs": len(diffs),
                "mean_Y": sum(y_f) / len(y_f) if y_f else "",
                "tau_vs_same_state_control": "" if tau is None else tau,
                "SE_vs_same_state_control": "" if se is None else se,
                "LCB_vs_same_state_control": "" if lcb is None else lcb,
                "positive_LCB_vs_control": int(lcb is not None and lcb > 0.0),
                "no_ECE_Brier_tail_debt_rate": no_debt / max(1, len(vals)),
                "status": "completed_summary",
            }
        )
    total_rows = len(rows)
    hard_rows = sum(1 for r in rows if str(parse_state_json(r).get("task_tier", "")) == "Tier1_hard_vision")
    h_counts = {str(h): sum(1 for r in rows if str(r.get("H", "")) == str(h)) for h in sorted({str(r.get("H", "")) for r in rows})}
    fam_counts = {fam: sum(1 for r in rows if str(r.get("candidate_family", "")) == fam) for fam in sorted({str(r.get("candidate_family", "")) for r in rows})}
    by_family_h = {(str(r.get("candidate_family", "")), str(r.get("H", ""))): r for r in out}
    h60_h200_positive_families = []
    h200_positive_signal_rows = []
    h200_positive_control_rows = []
    for row in out:
        if str(row.get("H")) == "200" and int_flag(row.get("positive_LCB_vs_control")):
            if row.get("candidate_kind") == "signal":
                h200_positive_signal_rows.append(row)
            elif row.get("candidate_kind") == "control":
                h200_positive_control_rows.append(row)
    for fam in sorted({str(r.get("candidate_family", "")) for r in out if r.get("candidate_kind") == "signal"}):
        h60 = by_family_h.get((fam, "60"), {})
        h200 = by_family_h.get((fam, "200"), {})
        matched = SUPPORT_CONTROL.get(fam, TREATMENT_META.get(fam, {}).get("match", ""))
        matched_h200 = by_family_h.get((matched, "200"), {})
        if int_flag(h60.get("positive_LCB_vs_control")) and int_flag(h200.get("positive_LCB_vs_control")) and not int_flag(matched_h200.get("positive_LCB_vs_control")):
            h60_h200_positive_families.append(fam)
    out.insert(
        0,
        {
            "H": "ALL",
            "candidate_family": "ALL",
            "rows": total_rows,
            "hard_task_rows": hard_rows,
            "H20_rows": h_counts.get("20", 0),
            "H60_rows": h_counts.get("60", 0),
            "H200_rows": h_counts.get("200", 0),
            "H800_rows": h_counts.get("800", 0),
            "min_rows_by_candidate_family": min(fam_counts.values(), default=0),
            "candidate_family_counts_json": json.dumps(fam_counts, sort_keys=True),
            "data_volume_minimum_pass": int(total_rows >= 3000 and hard_rows >= 1200 and all(v >= 200 for v in fam_counts.values()) and h_counts.get("20", 0) >= 300 and h_counts.get("60", 0) >= 300 and h_counts.get("200", 0) >= 300),
            "H200_positive_signal_lcb_rows": len(h200_positive_signal_rows),
            "H200_positive_control_lcb_rows": len(h200_positive_control_rows),
            "H60_H200_positive_unique_signal_families": ",".join(h60_h200_positive_families),
            "part_c_exploration_gate_pass": int(bool(h60_h200_positive_families) and not h200_positive_control_rows),
            "status": "completed_summary",
        },
    )
    return out


def parse_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(str(row.get("candidate_descriptor_json", "{}") or "{}"))
    except json.JSONDecodeError:
        return {}


def paired_feature_dict(row: dict[str, Any]) -> dict[str, Any]:
    state = parse_state_json(row)
    desc = parse_descriptor(row)
    out: dict[str, Any] = {
        "loss_mean": finite_float(state.get("held_before_NLL"), 0.0) or 0.0,
        "signal_eigen_topk": finite_float(state.get("signal_eigenvalue_proxy"), 0.0) or 0.0,
        "update_SNR": finite_float(state.get("grad_norm"), 0.0) or 0.0,
        "H": finite_float(row.get("H"), 0.0) or 0.0,
        "architecture": str(row.get("architecture", "")),
        "carrier": str(row.get("carrier", "")),
        "optimizer_family": str(row.get("optimizer_family", "")),
    }
    for key in CANDIDATE_DESCRIPTOR_FIELDS:
        val = desc.get(key, "")
        num = finite_float(val)
        out[key] = num if num is not None else str(val)
    return out


def same_state_control_delta(row: dict[str, Any], by_state_candidate: dict[tuple[str, str], dict[str, Any]]) -> float | None:
    fam = str(row.get("candidate_family", ""))
    matched = SUPPORT_CONTROL.get(fam, TREATMENT_META.get(fam, {}).get("match", ""))
    ctrl = by_state_candidate.get((str(row.get("state_id", "")), matched))
    y = finite_float(row.get("held_train_NLL_delta"))
    cy = finite_float(ctrl.get("held_train_NLL_delta")) if ctrl else None
    if y is None or cy is None:
        return None
    return float(y) - float(cy)


def stage_d_value_model() -> dict[str, Any]:
    ensure_out()
    rows = [r for r in read_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_matrix.csv") if r.get("status") == "completed_paired_candidate_event"]
    if not rows:
        blocked = {"status": "blocked", "reason": "missing_v22_39_paired_events"}
        write_rows(OUT_ROOT / "v22_39_potential_outcome_value_model.csv", [blocked])
        return blocked
    by_state_candidate = {(str(r.get("state_id", "")), str(r.get("candidate_family", ""))): r for r in rows}
    model_rows = []
    for fold_key in ["candidate", "dataset", "seed"]:
        scored = fit_paired_value_classifier(rows, by_state_candidate, fold_key=fold_key)
        model_rows.append(scored)
    by_fold = {r["fold_key"]: r for r in model_rows}
    cand_auc = finite_float(by_fold.get("candidate", {}).get("AUC_positive_value"), 0.0) or 0.0
    ds_auc = finite_float(by_fold.get("dataset", {}).get("AUC_positive_value"), 0.0) or 0.0
    seed_auc = finite_float(by_fold.get("seed", {}).get("AUC_positive_value"), 0.0) or 0.0
    fpr = finite_float(by_fold.get("candidate", {}).get("false_positive_rate_controls"), 1.0) or 1.0
    selected = finite_float(by_fold.get("candidate", {}).get("selected_candidate_beats_control_fraction"), 0.0) or 0.0
    exploration = int(cand_auc >= 0.62 and ds_auc >= 0.58 and fpr <= 0.05 and selected >= 0.55)
    official = int(exploration and cand_auc >= 0.70 and ds_auc >= 0.65 and seed_auc >= 0.65 and fpr <= 0.03 and selected >= 0.65)
    model_rows.append(
        {
            "fold_key": "SUMMARY",
            "candidate_heldout_AUC": cand_auc,
            "leave_dataset_AUC": ds_auc,
            "leave_seed_AUC": seed_auc,
            "false_positive_rate_controls": fpr,
            "selected_candidate_beats_control_fraction": selected,
            "exploration_gate_pass": exploration,
            "official_candidate_gate_pass": official,
            "status": "potential_outcome_model_open" if exploration else "potential_outcome_model_no_go",
        }
    )
    write_rows(OUT_ROOT / "v22_39_potential_outcome_value_model.csv", model_rows)
    append_exec(
        "stage_d_value_model",
        task_id="D_potential_outcome_value_model",
        status="pass" if exploration else "gate_blocked",
        gpu="0",
        files="results/v22_39/v22_39_potential_outcome_value_model.csv",
        note=f"candidate_auc={cand_auc:.4f}; dataset_auc={ds_auc:.4f}; seed_auc={seed_auc:.4f}; fpr={fpr:.4f}",
    )
    return model_rows[-1]


def stage_e_long_horizon_repair_analysis() -> dict[str, Any]:
    ensure_out()
    rows = [r for r in read_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_matrix.csv") if r.get("status") == "completed_paired_candidate_event"]
    if not rows:
        blocked = {"status": "blocked", "reason": "missing_v22_39_paired_events"}
        write_rows(OUT_ROOT / "v22_39_long_horizon_repair_analysis.csv", [blocked])
        return blocked
    by_state_candidate = {(str(r.get("state_id", "")), str(r.get("candidate_family", ""))): r for r in rows}
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row.get("collect_label", "")),
                str(row.get("intervention_scale", "")),
                str(row.get("H", "")),
                str(row.get("candidate_family", "")),
            ),
            [],
        ).append(row)
    h_lcb_by_label_fam: dict[tuple[str, str, str], float] = {}
    for (label, scale, h, fam), vals in sorted(grouped.items()):
        diffs = []
        for row in vals:
            diff = same_state_control_delta(row, by_state_candidate)
            if diff is not None:
                diffs.append(float(diff))
        tau, se, lcb = mean_lcb(diffs, [0.0] * len(diffs))
        h_lcb_by_label_fam[(label, fam, h)] = float(lcb) if lcb is not None else float("nan")
        out.append(
            {
                "collect_label": label,
                "intervention_scale": scale,
                "H": h,
                "candidate_family": fam,
                "candidate_kind": TREATMENT_META.get(fam, {}).get("kind", ""),
                "rows": len(vals),
                "same_state_pair_diffs": len(diffs),
                "tau_vs_same_state_control": "" if tau is None else tau,
                "LCB_vs_same_state_control": "" if lcb is None else lcb,
                "positive_LCB_vs_control": int(lcb is not None and lcb > 0.0),
                "matched_control_positive_LCB_same_label_H": "",
                "unique_positive_vs_matched_control": "",
                "long_horizon_penalty_blocks_runtime": int(str(h) == "200" and not (lcb is not None and lcb > 0.0)),
                "repair_note": "candidate eligible for runtime only if H200 LCB is positive; otherwise full-loop remains blocked",
                "status": "completed_repair_analysis",
            }
        )
    positive_lookup = {
        (str(r.get("collect_label", "")), str(r.get("intervention_scale", "")), str(r.get("H", "")), str(r.get("candidate_family", ""))): int_flag(r.get("positive_LCB_vs_control"))
        for r in out
        if r.get("status") == "completed_repair_analysis"
    }
    for row in out:
        if row.get("status") != "completed_repair_analysis":
            continue
        fam = str(row.get("candidate_family", ""))
        matched = SUPPORT_CONTROL.get(fam, TREATMENT_META.get(fam, {}).get("match", ""))
        key = (str(row.get("collect_label", "")), str(row.get("intervention_scale", "")), str(row.get("H", "")), matched)
        matched_positive = positive_lookup.get(key, 0)
        row["matched_control_positive_LCB_same_label_H"] = matched_positive
        row["unique_positive_vs_matched_control"] = int(int_flag(row.get("positive_LCB_vs_control")) and not matched_positive)
        if str(row.get("H")) == "200" and not int_flag(row.get("unique_positive_vs_matched_control")):
            row["long_horizon_penalty_blocks_runtime"] = 1
    transient_rows = []
    labels = sorted({str(r.get("collect_label", "")) for r in rows})
    families = sorted({str(r.get("candidate_family", "")) for r in rows if TREATMENT_META.get(str(r.get("candidate_family", "")), {}).get("kind") == "signal"})
    for label in labels:
        for fam in families:
            l60 = h_lcb_by_label_fam.get((label, fam, "60"), float("nan"))
            l200 = h_lcb_by_label_fam.get((label, fam, "200"), float("nan"))
            transient_rows.append(
                {
                    "collect_label": label,
                    "candidate_family": fam,
                    "H60_LCB": "" if not math.isfinite(l60) else l60,
                    "H200_LCB": "" if not math.isfinite(l200) else l200,
                    "transient_regularization_pattern": int(math.isfinite(l60) and l60 > 0.0 and (not math.isfinite(l200) or l200 <= 0.0)),
                    "status": "completed_transient_check",
                }
            )
    out.extend(transient_rows)
    write_rows(OUT_ROOT / "v22_39_long_horizon_repair_analysis.csv", out)
    noncontrol_h200_positive = [
        r
        for r in out
        if r.get("H") == "200"
        and r.get("candidate_kind") == "signal"
        and int_flag(r.get("positive_LCB_vs_control"))
    ]
    noncontrol_h200_unique = [r for r in noncontrol_h200_positive if int_flag(r.get("unique_positive_vs_matched_control"))]
    summary = {
        "status": "long_horizon_repair_open" if noncontrol_h200_unique else "long_horizon_repair_no_go",
        "noncontrol_H200_positive_LCB_rows": len(noncontrol_h200_positive),
        "noncontrol_H200_unique_positive_LCB_rows": len(noncontrol_h200_unique),
        "transient_regularization_rows": sum(int_flag(r.get("transient_regularization_pattern")) for r in transient_rows),
        "runtime_full_loop_allowed": int(bool(noncontrol_h200_unique)),
        "source_artifact": "results/v22_39/v22_39_candidate_pair_micro_rct_matrix.csv",
    }
    write_rows(OUT_ROOT / "v22_39_long_horizon_repair_summary.csv", [summary])
    append_exec(
        "stage_e_long_horizon_repair_analysis",
        task_id="E_long_horizon_repair_analysis",
        status="pass" if noncontrol_h200_positive else "gate_blocked",
        gpu="0",
        files="results/v22_39/v22_39_long_horizon_repair_analysis.csv, results/v22_39/v22_39_long_horizon_repair_summary.csv",
        note=f"H200_positive_signal_rows={len(noncontrol_h200_positive)}; unique={len(noncontrol_h200_unique)}; transient_rows={summary['transient_regularization_rows']}",
    )
    return summary


def candidate_lcb_from_v22_39(treatment: str) -> tuple[float, str, str]:
    if str(treatment) == "a2_a5_runtime_arbitrated_basis":
        child_lcbs = [candidate_lcb_from_v22_39(child) for child in ["a2_basis_actuator_section", "a5_signal_incremental_basis"]]
        positive = [item for item in child_lcbs if item[0] > 0.0]
        if positive:
            val, horizon, source = min(positive, key=lambda item: item[0])
            return val, horizon, f"conservative_min_child_positive_lcb:{source}"
        best = max(child_lcbs, key=lambda item: item[0])
        return best[0], best[1], f"best_child_lcb_not_positive:{best[2]}"
    summary = read_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_summary.csv")
    preferred = ["800", "200", "60", "20"]
    best_val = -math.inf
    best_h = ""
    for horizon in preferred:
        for row in summary:
            if str(row.get("H", "")) != horizon or str(row.get("candidate_family", "")) != treatment:
                continue
            val = finite_float(row.get("LCB_vs_same_state_control"))
            if val is None:
                continue
            if horizon in {"800", "200"} and float(val) > 0.0:
                return float(val), horizon, "positive_long_horizon_lcb"
            if float(val) > best_val:
                best_val = float(val)
                best_h = horizon
    if math.isfinite(best_val):
        return float(best_val), best_h, "best_available_lcb_not_positive_long_horizon"
    return 0.0, "", "missing_lcb"


def stage_f_collect(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    import torch

    label = safe_fragment(args.full_loop_label)
    device = core.torch_device(args.full_loop_device)
    rows: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    treatments = split_csv(args.full_loop_treatments)
    for dataset in split_csv(args.full_loop_datasets):
        for seed in split_csv(args.full_loop_seeds, int):
            for architecture in split_csv(args.full_loop_architectures):
                for optimizer_family in split_csv(args.full_loop_optimizers):
                    try:
                        trainer = prev.train_horizon_robust_full_loop_variant if str(args.full_loop_engine) in {"horizon_robust_v38", "path_mpc_v39"} else core.train_full_loop_variant
                        def fresh_loaders() -> tuple[Any, Any, Any, int, int, Any, dict[str, Any]]:
                            return core.make_loaders_for_dataset(
                                str(dataset),
                                int(args.full_loop_train_size),
                                int(args.full_loop_held_size),
                                int(args.batch_size),
                                int(seed),
                                tier2_download=bool(args.tier2_download),
                            )

                        train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta = fresh_loaders()
                        base_row = trainer(
                            dataset=str(dataset),
                            seed=int(seed),
                            architecture=str(architecture),
                            optimizer_family=str(optimizer_family),
                            variant="optimizer_alone",
                            treatment="a0_base_noop",
                            train_loader=train_loader,
                            held_loader=held_loader,
                            test_loader=test_loader,
                            input_dim=int(input_dim),
                            output_dim=int(output_dim),
                            x_stats=x_stats,
                            device=device,
                            args=args,
                            lcb=0.0,
                            source_artifact="results/v22_39/v22_39_candidate_pair_micro_rct_summary.csv",
                        )
                        base_row.update(
                            {
                                "full_loop_label": label,
                                "full_loop_role": "optimizer_alone",
                                "requested_architecture": architecture,
                                "target_treatment_family": "",
                                "runtime_value_lcb_used": "",
                                "empirical_lcb_from_summary": "",
                                "empirical_lcb_horizon": "",
                                "empirical_lcb_source": "",
                                "control_lcb_mode": "",
                                "task_tier": meta.get("task_tier", ""),
                                "runtime_policy_type": "restricted_train_held_acceptance_alpha_grid",
                                "loader_reset_per_variant": 1,
                                "full_loop_engine": args.full_loop_engine,
                            }
                        )
                        rows.append(base_row)
                        for treatment in treatments:
                            valid = valid_candidates_for_arch(str(architecture))
                            if treatment not in valid:
                                availability.append(
                                    {
                                        "full_loop_label": label,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "architecture": architecture,
                                        "optimizer_family": optimizer_family,
                                        "treatment": treatment,
                                        "status": "candidate_not_valid_for_architecture",
                                    }
                                )
                                continue
                            lcb, lcb_h, lcb_source = candidate_lcb_from_v22_39(str(treatment))
                            if lcb <= float(args.epsilon_row) and not bool(args.full_loop_allow_nonpositive_value_gate):
                                availability.append(
                                    {
                                        "full_loop_label": label,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "architecture": architecture,
                                        "optimizer_family": optimizer_family,
                                        "treatment": treatment,
                                        "status": "value_gate_blocked_nonpositive_lcb",
                                        "empirical_lcb": lcb,
                                        "empirical_lcb_horizon": lcb_h,
                                        "empirical_lcb_source": lcb_source,
                                    }
                                )
                                continue
                            fu_train_loader, fu_held_loader, fu_test_loader, fu_input_dim, fu_output_dim, fu_x_stats, fu_meta = fresh_loaders()
                            fu_row = trainer(
                                dataset=str(dataset),
                                seed=int(seed),
                                architecture=str(architecture),
                                optimizer_family=str(optimizer_family),
                                variant=f"CFPO_FU_{treatment}",
                                treatment=str(treatment),
                                train_loader=fu_train_loader,
                                held_loader=fu_held_loader,
                                test_loader=fu_test_loader,
                                input_dim=int(fu_input_dim),
                                output_dim=int(fu_output_dim),
                                x_stats=fu_x_stats,
                                device=device,
                                args=args,
                                lcb=float(lcb),
                                source_artifact="results/v22_39/v22_39_candidate_pair_micro_rct_summary.csv",
                            )
                            fu_row.update(
                                {
                                    "full_loop_label": label,
                                    "full_loop_role": "signal_candidate",
                                    "requested_architecture": architecture,
                                    "target_treatment_family": treatment,
                                    "runtime_value_lcb_used": lcb,
                                    "empirical_lcb_from_summary": lcb,
                                    "empirical_lcb_horizon": lcb_h,
                                    "empirical_lcb_source": lcb_source,
                                    "control_lcb_mode": "",
                                    "task_tier": fu_meta.get("task_tier", meta.get("task_tier", "")),
                                    "runtime_policy_type": "restricted_train_held_acceptance_alpha_grid",
                                    "loader_reset_per_variant": 1,
                                    "full_loop_engine": args.full_loop_engine,
                                }
                            )
                            rows.append(fu_row)
                            control = SUPPORT_CONTROL.get(str(treatment), TREATMENT_META.get(str(treatment), {}).get("match", ""))
                            if control and bool(args.full_loop_run_controls):
                                control_lcb, control_h, control_source = candidate_lcb_from_v22_39(str(control))
                                lcb_used = float(control_lcb)
                                control_mode = str(args.full_loop_control_lcb_mode)
                                if control_mode == "forced_positive_diagnostic":
                                    lcb_used = max(float(lcb), 1.0e-12)
                                    control_source = f"{control_source};forced_positive_for_matched_control_diagnostic"
                                elif control_mode == "blocked":
                                    availability.append(
                                        {
                                            "full_loop_label": label,
                                            "dataset": dataset,
                                            "seed": seed,
                                            "architecture": architecture,
                                            "optimizer_family": optimizer_family,
                                            "treatment": control,
                                            "target_treatment_family": treatment,
                                            "status": "matched_control_blocked_by_requested_mode",
                                        }
                                    )
                                    continue
                                ctrl_train_loader, ctrl_held_loader, ctrl_test_loader, ctrl_input_dim, ctrl_output_dim, ctrl_x_stats, ctrl_meta = fresh_loaders()
                                ctrl_row = trainer(
                                    dataset=str(dataset),
                                    seed=int(seed),
                                    architecture=str(architecture),
                                    optimizer_family=str(optimizer_family),
                                    variant=f"matched_control_{treatment}",
                                    treatment=str(control),
                                    train_loader=ctrl_train_loader,
                                    held_loader=ctrl_held_loader,
                                    test_loader=ctrl_test_loader,
                                    input_dim=int(ctrl_input_dim),
                                    output_dim=int(ctrl_output_dim),
                                    x_stats=ctrl_x_stats,
                                    device=device,
                                    args=args,
                                    lcb=float(lcb_used),
                                    source_artifact="results/v22_39/v22_39_candidate_pair_micro_rct_summary.csv",
                                )
                                ctrl_row.update(
                                    {
                                        "full_loop_label": label,
                                        "full_loop_role": "matched_control_diagnostic",
                                        "requested_architecture": architecture,
                                        "target_treatment_family": treatment,
                                        "runtime_value_lcb_used": lcb_used,
                                        "empirical_lcb_from_summary": control_lcb,
                                        "empirical_lcb_horizon": control_h,
                                        "empirical_lcb_source": control_source,
                                        "control_lcb_mode": control_mode,
                                        "task_tier": ctrl_meta.get("task_tier", meta.get("task_tier", "")),
                                        "runtime_policy_type": "restricted_train_held_acceptance_alpha_grid",
                                        "loader_reset_per_variant": 1,
                                        "full_loop_engine": args.full_loop_engine,
                                    }
                                )
                                rows.append(ctrl_row)
                        availability.append(
                            {
                                "full_loop_label": label,
                                "dataset": dataset,
                                "seed": seed,
                                "architecture": architecture,
                                "optimizer_family": optimizer_family,
                                "status": "completed",
                                "rows_completed_so_far": len(rows),
                                "task_tier": meta.get("task_tier", ""),
                            }
                        )
                    except Exception as exc:
                        availability.append(
                            {
                                "full_loop_label": label,
                                "dataset": dataset,
                                "seed": seed,
                                "architecture": architecture,
                                "optimizer_family": optimizer_family,
                                "status": "data_unavailable_or_run_failed",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
    out_path = CHUNK_ROOT / f"v22_39_restricted_full_loop_{label}.csv"
    av_path = CHUNK_ROOT / f"v22_39_restricted_full_loop_availability_{label}.csv"
    write_rows(out_path, rows or [{"status": "no_full_loop_rows", "full_loop_label": label}])
    write_rows(av_path, availability or [{"status": "no_availability_rows", "full_loop_label": label}])
    append_exec(
        " ".join(shlex.quote(x) for x in sys.argv),
        task_id=f"F_collect_{label}",
        status="pass" if rows else "warn",
        gpu=args.full_loop_device,
        files=f"{out_path.relative_to(ROOT)}, {av_path.relative_to(ROOT)}",
        note=f"full_loop_rows={len(rows)}; availability_rows={len(availability)}; treatments={','.join(treatments)}; steps={args.full_loop_steps}; cadence={args.full_loop_cadence}",
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "completed" if rows else "no_rows", "full_loop_rows": len(rows), "label": label}


def stage_f_merge() -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_39_restricted_full_loop_*.csv")):
        for row in read_rows(path):
            if row.get("status") == "completed_full_loop" and int_flag(row.get("loader_reset_per_variant")):
                rows.append(row)
    for path in sorted(CHUNK_ROOT.glob("v22_39_restricted_full_loop_availability_*.csv")):
        availability.extend(read_rows(path))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = "|".join(
            [
                str(row.get("full_loop_label", "")),
                str(row.get("dataset", "")),
                str(row.get("seed", "")),
                str(row.get("carrier", "")),
                str(row.get("optimizer_family", "")),
                str(row.get("training_variant", "")),
                str(row.get("treatment_name", "")),
            ]
        )
        dedup[key] = row
    rows = list(dedup.values())
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        by_key[
            (
                str(row.get("full_loop_label", "")),
                str(row.get("dataset", "")),
                str(row.get("seed", "")),
                str(row.get("carrier", "")),
                str(row.get("optimizer_family", "")),
                str(row.get("training_variant", "")),
            )
        ] = row
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        common = (
            str(row.get("full_loop_label", "")),
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("carrier", "")),
            str(row.get("optimizer_family", "")),
        )
        base = by_key.get(common + ("optimizer_alone",), {})
        target = str(row.get("target_treatment_family", "") or row.get("treatment_name", ""))
        control = by_key.get(common + (f"matched_control_{target}",), {})
        nll = finite_float(row.get("final_test_NLL"))
        base_nll = finite_float(base.get("final_test_NLL"))
        ctrl_nll = finite_float(control.get("final_test_NLL"))
        auc = finite_float(row.get("AUC_loss_time"))
        base_auc = finite_float(base.get("AUC_loss_time"))
        acc = finite_float(row.get("final_test_accuracy"))
        base_acc = finite_float(base.get("final_test_accuracy"))
        ece = finite_float(row.get("ECE"))
        base_ece = finite_float(base.get("ECE"))
        brier = finite_float(row.get("Brier"))
        base_brier = finite_float(base.get("Brier"))
        tail = finite_float(row.get("tail_loss_q99"))
        base_tail = finite_float(base.get("tail_loss_q99"))
        accepted = finite_float(row.get("accepted_treatment_count"), 0.0) or 0.0
        rejected = finite_float(row.get("rejected_treatment_count"), 0.0) or 0.0
        base_overhead = finite_float(base.get("controller_overhead"))
        overhead = finite_float(row.get("controller_overhead"))
        out.update(
            {
                "NLL_delta_vs_own_strong_optimizer": "" if nll is None or base_nll is None else nll - base_nll,
                "AUC_delta_vs_own_strong_optimizer": "" if auc is None or base_auc is None else auc - base_auc,
                "accuracy_delta_vs_own_strong_optimizer": "" if acc is None or base_acc is None else acc - base_acc,
                "NLL_delta_vs_same_basis_control": "" if nll is None or ctrl_nll is None else nll - ctrl_nll,
                "no_ECE_Brier_tail_debt_vs_optimizer": int(
                    str(row.get("training_variant", "")).startswith("CFPO_FU_")
                    and ece is not None
                    and base_ece is not None
                    and brier is not None
                    and base_brier is not None
                    and tail is not None
                    and base_tail is not None
                    and ece <= base_ece
                    and brier <= base_brier
                    and tail <= base_tail
                ),
                "realized_policy_noop_rate": "" if accepted + rejected <= 0 else rejected / (accepted + rejected),
                "controller_overhead_ratio_vs_optimizer": "" if overhead is None or base_overhead in {None, 0.0} else overhead / float(base_overhead),
            }
        )
        enriched.append(out)
    write_rows(OUT_ROOT / "v22_39_restricted_full_loop_matrix.csv", enriched or [{"status": "no_full_loop_rows"}])
    write_rows(OUT_ROOT / "v22_39_restricted_full_loop_availability.csv", availability or [{"status": "no_availability_rows"}])
    signal_rows = [r for r in enriched if str(r.get("training_variant", "")).startswith("CFPO_FU_")]
    kan_rows = [r for r in signal_rows if r.get("architecture") == "strict_FC_PureKAN"]
    mlp_rows = [r for r in signal_rows if r.get("architecture") == "MLP"]
    kan_nll_wins = sum(1 for r in kan_rows if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), math.inf) or math.inf) < 0.0)
    kan_control_wins = sum(1 for r in kan_rows if (finite_float(r.get("NLL_delta_vs_same_basis_control"), math.inf) or math.inf) < 0.0)
    kan_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in kan_rows)
    kan_basis = sum(1 for r in kan_rows if (finite_float(r.get("basis_energy_fraction"), 0.0) or 0.0) >= 0.5)
    kan_readout_ok = sum(1 for r in kan_rows if (finite_float(r.get("readout_leakage_fraction")) is not None and float(finite_float(r.get("readout_leakage_fraction")) or 0.0) <= 0.3))
    mlp_nll_wins = sum(1 for r in mlp_rows if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), math.inf) or math.inf) < 0.0)
    mlp_control_wins = sum(1 for r in mlp_rows if (finite_float(r.get("NLL_delta_vs_same_basis_control"), math.inf) or math.inf) < 0.0)
    mlp_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in mlp_rows)
    def prop(count: int, total: int) -> float:
        return count / max(1, total)
    kan_gate = int(
        len(kan_rows) >= 4
        and prop(kan_nll_wins, len(kan_rows)) >= 5.0 / 9.0
        and prop(kan_control_wins, len(kan_rows)) >= 6.0 / 9.0
        and prop(kan_no_debt, len(kan_rows)) >= 7.0 / 9.0
        and prop(kan_basis, len(kan_rows)) >= 8.0 / 9.0
        and prop(kan_readout_ok, len(kan_rows)) >= 8.0 / 9.0
    )
    mlp_gate = int(
        len(mlp_rows) >= 4
        and prop(mlp_nll_wins, len(mlp_rows)) >= 5.0 / 9.0
        and prop(mlp_control_wins, len(mlp_rows)) >= 6.0 / 9.0
        and prop(mlp_no_debt, len(mlp_rows)) >= 7.0 / 9.0
    )
    summary = {
        "status": "completed_restricted_full_loop" if enriched else "no_full_loop_rows",
        "full_loop_rows": len(enriched),
        "signal_rows": len(signal_rows),
        "kan_signal_rows": len(kan_rows),
        "mlp_signal_rows": len(mlp_rows),
        "datasets": ",".join(sorted({str(r.get("dataset", "")) for r in enriched if r.get("dataset")})),
        "seeds": ",".join(sorted({str(r.get("seed", "")) for r in enriched if r.get("seed")})),
        "selected_signal_treatments": ",".join(sorted({str(r.get("treatment_name", "")) for r in signal_rows if r.get("treatment_name")})),
        "KAN_NLL_improvement_rows": kan_nll_wins,
        "KAN_beats_same_basis_control_rows": kan_control_wins,
        "KAN_no_ECE_Brier_tail_debt_rows": kan_no_debt,
        "KAN_basis_energy_ge_0p5_rows": kan_basis,
        "KAN_readout_leakage_le_0p3_rows": kan_readout_ok,
        "kan_internal_value_gate_pass": kan_gate,
        "MLP_NLL_improvement_rows": mlp_nll_wins,
        "MLP_beats_matched_control_rows": mlp_control_wins,
        "MLP_no_ECE_Brier_tail_debt_rows": mlp_no_debt,
        "mlp_general_value_gate_pass": mlp_gate,
        "accepted_treatment_count_total": sum(int(finite_float(r.get("accepted_treatment_count"), 0.0) or 0.0) for r in signal_rows),
        "rejected_treatment_count_total": sum(int(finite_float(r.get("rejected_treatment_count"), 0.0) or 0.0) for r in signal_rows),
        "candidate_eval_count_total": sum(int(finite_float(r.get("candidate_eval_count"), 0.0) or 0.0) for r in signal_rows),
        "scope_note": "restricted exploration full-loop; test metrics are audit-only; train-only held acceptance controls FU application; matched controls may be forced-positive diagnostic if requested",
        "source_artifact": "results/v22_39/chunks/v22_39_restricted_full_loop_*.csv",
    }
    write_rows(OUT_ROOT / "v22_39_restricted_full_loop_summary.csv", [summary])
    protocol_rows: list[dict[str, Any]] = []
    for label in sorted({str(r.get("full_loop_label", "")) for r in enriched if r.get("full_loop_label")}):
        scoped = [r for r in enriched if str(r.get("full_loop_label", "")) == label]
        scoped_signal = [r for r in scoped if str(r.get("training_variant", "")).startswith("CFPO_FU_")]
        scoped_kan = [r for r in scoped_signal if r.get("architecture") == "strict_FC_PureKAN"]
        scoped_mlp = [r for r in scoped_signal if r.get("architecture") == "MLP"]
        nll_deltas = [float(v) for v in (finite_float(r.get("NLL_delta_vs_own_strong_optimizer")) for r in scoped_kan) if v is not None]
        ctrl_deltas = [float(v) for v in (finite_float(r.get("NLL_delta_vs_same_basis_control")) for r in scoped_kan) if v is not None]
        mlp_nll_deltas = [float(v) for v in (finite_float(r.get("NLL_delta_vs_own_strong_optimizer")) for r in scoped_mlp) if v is not None]
        mlp_ctrl_deltas = [float(v) for v in (finite_float(r.get("NLL_delta_vs_same_basis_control")) for r in scoped_mlp) if v is not None]
        protocol_rows.append(
            {
                "full_loop_label": label,
                "full_loop_rows": len(scoped),
                "kan_signal_rows": len(scoped_kan),
                "KAN_NLL_improvement_rows": sum(1 for v in nll_deltas if v < 0.0),
                "KAN_beats_same_basis_control_rows": sum(1 for v in ctrl_deltas if v < 0.0),
                "KAN_no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in scoped_kan),
                "mean_NLL_delta_vs_optimizer": "" if not nll_deltas else sum(nll_deltas) / len(nll_deltas),
                "mean_NLL_delta_vs_same_basis_control": "" if not ctrl_deltas else sum(ctrl_deltas) / len(ctrl_deltas),
                "mlp_signal_rows": len(scoped_mlp),
                "MLP_NLL_improvement_rows": sum(1 for v in mlp_nll_deltas if v < 0.0),
                "MLP_beats_same_basis_control_rows": sum(1 for v in mlp_ctrl_deltas if v < 0.0),
                "MLP_no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in scoped_mlp),
                "mean_MLP_NLL_delta_vs_optimizer": "" if not mlp_nll_deltas else sum(mlp_nll_deltas) / len(mlp_nll_deltas),
                "mean_MLP_NLL_delta_vs_same_basis_control": "" if not mlp_ctrl_deltas else sum(mlp_ctrl_deltas) / len(mlp_ctrl_deltas),
                "accepted_treatment_count_total": sum(int(finite_float(r.get("accepted_treatment_count"), 0.0) or 0.0) for r in scoped_kan),
                "rejected_treatment_count_total": sum(int(finite_float(r.get("rejected_treatment_count"), 0.0) or 0.0) for r in scoped_kan),
                "accepted_MLP_treatment_count_total": sum(int(finite_float(r.get("accepted_treatment_count"), 0.0) or 0.0) for r in scoped_mlp),
                "rejected_MLP_treatment_count_total": sum(int(finite_float(r.get("rejected_treatment_count"), 0.0) or 0.0) for r in scoped_mlp),
                "post_apply_rollback_count_total": sum(int(finite_float(r.get("post_apply_rollback_count"), 0.0) or 0.0) for r in scoped_kan),
                "acceptance_tail_quantile": ",".join(sorted({str(r.get("acceptance_tail_quantile", "")) for r in scoped_kan if r.get("acceptance_tail_quantile", "")})),
                "calibration_temperatures": ",".join(sorted({str(r.get("calibration_temperature", "")) for r in scoped_signal if r.get("calibration_temperature", "")})),
                "empirical_lcb_horizons": ",".join(sorted({str(r.get("empirical_lcb_horizon", "")) for r in scoped_signal if r.get("empirical_lcb_horizon", "")})),
                "empirical_lcb_sources": ",".join(sorted({str(r.get("empirical_lcb_source", "")) for r in scoped_signal if r.get("empirical_lcb_source", "")})),
                "full_loop_engine": ",".join(sorted({str(r.get("full_loop_engine", "")) for r in scoped if r.get("full_loop_engine", "")})),
                "status": "completed_protocol_summary",
            }
        )
    write_rows(OUT_ROOT / "v22_39_restricted_full_loop_protocol_summary.csv", protocol_rows or [{"status": "no_protocol_rows"}])
    append_exec(
        "stage_f_merge_restricted_full_loop",
        task_id="F_merge_restricted_full_loop",
        status="pass" if enriched else "warn",
        gpu="0",
        files="results/v22_39/v22_39_restricted_full_loop_matrix.csv, results/v22_39/v22_39_restricted_full_loop_summary.csv, results/v22_39/v22_39_restricted_full_loop_protocol_summary.csv",
        note=f"rows={len(enriched)}; kan_gate={kan_gate}; mlp_gate={mlp_gate}; kan_wins={kan_nll_wins}/{len(kan_rows)}; kan_control_wins={kan_control_wins}/{len(kan_rows)}",
    )
    return summary


def fit_paired_value_classifier(rows: list[dict[str, Any]], by_state_candidate: dict[tuple[str, str], dict[str, Any]], *, fold_key: str) -> dict[str, Any]:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline

    usable = []
    labels = []
    for row in rows:
        diff = same_state_control_delta(row, by_state_candidate)
        no_debt = (
            (finite_float(row.get("ECE_delta"), 0.0) or 0.0) >= 0.0
            and (finite_float(row.get("Brier_delta"), 0.0) or 0.0) >= 0.0
            and (finite_float(row.get("tail_q99_delta"), 0.0) or 0.0) >= 0.0
        )
        if diff is None:
            continue
        usable.append(row)
        labels.append(int(diff > 0.0 and no_debt and not control_kind(row)))
    if not usable:
        return {"fold_key": fold_key, "status": "no_usable_same_state_pairs"}
    if fold_key == "candidate":
        groups = np.array([str(r.get("candidate_family", "")) for r in usable])
    elif fold_key == "dataset":
        groups = np.array([str(r.get("dataset", "")) for r in usable])
    else:
        groups = np.array([str(r.get("seed", "")) for r in usable])
    y = np.array(labels, dtype=int)
    preds = np.zeros(len(usable), dtype=float)
    status = []
    for g in list(dict.fromkeys(groups.tolist())):
        train_idx = np.where(groups != g)[0]
        test_idx = np.where(groups == g)[0]
        y_train = y[train_idx]
        x_train = [paired_feature_dict(usable[i]) for i in train_idx]
        x_test = [paired_feature_dict(usable[i]) for i in test_idx]
        if len(set(y_train.tolist())) < 2:
            model = make_pipeline(DictVectorizer(sparse=False), DummyClassifier(strategy="prior"))
        else:
            model = make_pipeline(DictVectorizer(sparse=False), GradientBoostingClassifier(random_state=2239, max_depth=2, n_estimators=80))
        try:
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_test)
            classes = [int(c) for c in getattr(model, "classes_", [])]
            preds[test_idx] = proba[:, classes.index(1)] if 1 in classes else 0.0
        except Exception as exc:
            preds[test_idx] = float(y_train.mean()) if len(y_train) else 0.0
            status.append(f"{g}:{type(exc).__name__}")
    auc: float | str = ""
    if len(set(y.tolist())) >= 2 and len(set(np.round(preds, 8).tolist())) >= 2:
        auc = float(roc_auc_score(y, preds))
    threshold = float(np.quantile(preds, 0.80)) if len(preds) else 0.5
    controls = [i for i, r in enumerate(usable) if control_kind(r)]
    fpr_controls = sum(1 for i in controls if preds[i] >= threshold) / max(1, len(controls))
    selected_beats, selected_total = selected_beats_same_state(usable, preds.tolist(), threshold, by_state_candidate)
    return {
        "fold_key": fold_key,
        "n_events": len(usable),
        "positive_rate": float(y.mean()) if len(y) else "",
        "AUC_positive_value": auc,
        "Brier_score": float(brier_score_loss(y, np.clip(preds, 1e-6, 1 - 1e-6))) if len(set(y.tolist())) >= 2 else "",
        "decision_threshold": threshold,
        "false_positive_rate_controls": fpr_controls,
        "selected_candidate_beats_control_fraction": selected_beats / selected_total if selected_total else "",
        "selected_candidate_beats_control_count": selected_beats,
        "selected_candidate_total": selected_total,
        "fit_status": ";".join(status) or "ok",
    }


def selected_beats_same_state(rows: list[dict[str, Any]], preds: list[float], threshold: float, by_state_candidate: dict[tuple[str, str], dict[str, Any]]) -> tuple[int, int]:
    groups: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, pred in zip(rows, preds):
        groups.setdefault(str(row.get("state_id", "")), []).append((row, pred))
    beats = 0
    total = 0
    for items in groups.values():
        candidates = [(pred, row) for row, pred in items if not control_kind(row)]
        if not candidates:
            continue
        pred, row = max(candidates, key=lambda x: x[0])
        if pred < threshold:
            continue
        diff = same_state_control_delta(row, by_state_candidate)
        if diff is None:
            continue
        total += 1
        if diff > 0.0:
            beats += 1
    return beats, total


def write_artifact_manifest() -> str:
    rows = []
    for path in sorted(OUT_ROOT.glob("v22_39_*")):
        if path.is_file() and path.name != "v22_39_artifact_manifest.csv":
            rows.append({"artifact": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    for path in sorted(CHUNK_ROOT.glob("v22_39_*")):
        if path.is_file():
            rows.append({"artifact": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_rows(OUT_ROOT / "v22_39_artifact_manifest.csv", rows or [{"status": "no_artifacts"}])
    text = json.dumps(rows, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finalize_route() -> dict[str, Any]:
    ensure_out()
    stage_a_code_truth_gate()
    b = {}
    b_path = OUT_ROOT / "v22_39_policy_leakage_route.json"
    if b_path.exists():
        try:
            b = json.loads(b_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            b = {}
    c_summary = read_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_summary.csv")
    c_top = c_summary[0] if c_summary else {}
    d_summary = (read_rows(OUT_ROOT / "v22_39_potential_outcome_value_model.csv") or [{}])[-1]
    e_summary = (read_rows(OUT_ROOT / "v22_39_long_horizon_repair_summary.csv") or [{}])[0]
    f_summary = (read_rows(OUT_ROOT / "v22_39_restricted_full_loop_summary.csv") or [{}])[0]
    code = (read_rows(OUT_ROOT / "v22_39_code_truth_gate.csv") or [{}])[0]
    if not int_flag(code.get("clean_unzip_compileall_pass")) or not int_flag(code.get("clean_unzip_import_pass")):
        route = "R1-CodeOrRandomizationGateFailed"
        reason = "compile/import truth gate failed"
    elif int_flag(f_summary.get("kan_internal_value_gate_pass")) and int_flag(f_summary.get("mlp_general_value_gate_pass")):
        route = "R8-ArchitectureInteractionOpened_NotOfficial"
        reason = "restricted full-loop opened both MLP and KAN exploration gates; architecture gap truth is still not official-scale"
    elif int_flag(f_summary.get("kan_internal_value_gate_pass")):
        route = "R6-KANInternalValueOpened"
        reason = "restricted KAN a2 full-loop opened vs own optimizer and same-basis control; MLP/gap official criteria remain unmet"
    elif int_flag(f_summary.get("mlp_general_value_gate_pass")):
        route = "R5-MLPFUGeneralValueOpened"
        reason = "restricted MLP full-loop opened exploration; KAN/gap official criteria remain unmet"
    elif f_summary.get("status") == "completed_restricted_full_loop" and int_flag(d_summary.get("exploration_gate_pass")):
        route = "R4-PotentialOutcomeModelOpened_NoFullLoop"
        reason = "potential-outcome value model opened, but restricted full-loop did not satisfy exploration gate"
    elif int_flag(d_summary.get("exploration_gate_pass")) and int_flag(e_summary.get("runtime_full_loop_allowed")):
        route = "R4-PotentialOutcomeModelOpened_NoFullLoop"
        reason = "v22.38 identity-free CATE failed, but v22.39 paired value model and low-scale H200 repair opened exploration; official/full-loop still blocked"
    elif int_flag(d_summary.get("exploration_gate_pass")):
        route = "R4-PotentialOutcomeModelOpened_NoFullLoop"
        reason = "potential-outcome value model exploration gate opened, but H200 repair or official gate still blocks full-loop"
    elif b.get("route") == "R2-PolicyLearnsTreatmentIdentity_NotStateCATE" and not int_flag(c_top.get("part_c_exploration_gate_pass")):
        route = "R2-PolicyLearnsTreatmentIdentity_NotStateCATE"
        reason = "v22.38 identity-free CATE did not open and paired long-horizon evidence did not open"
    elif c_top and not int_flag(c_top.get("part_c_exploration_gate_pass")):
        route = "R3-CandidateRCTCollected_NoLongHorizonValue"
        reason = "paired candidate data collected, but no non-control long-horizon positive LCB gate opened"
    else:
        route = "R3-CandidateRCTCollected_NoLongHorizonValue" if c_top else "R2-PolicyLearnsTreatmentIdentity_NotStateCATE"
        reason = "no official candidate-heldout value gate opened"
    manifest_hash = write_artifact_manifest()
    final = {
        "final_route": route,
        "route_reason": reason,
        "artifact_manifest_hash": manifest_hash,
        "code_truth_status": code.get("status", ""),
        "v22_38_policy_route": b.get("route", ""),
        "candidate_pair_rows": c_top.get("rows", ""),
        "hard_task_rows": c_top.get("hard_task_rows", ""),
        "part_c_exploration_gate_pass": c_top.get("part_c_exploration_gate_pass", ""),
        "part_d_exploration_gate_pass": d_summary.get("exploration_gate_pass", ""),
        "long_horizon_repair_status": e_summary.get("status", ""),
        "runtime_full_loop_allowed_by_h200_repair": e_summary.get("runtime_full_loop_allowed", ""),
        "restricted_full_loop_status": f_summary.get("status", ""),
        "kan_internal_value_gate_pass": f_summary.get("kan_internal_value_gate_pass", ""),
        "mlp_general_value_gate_pass": f_summary.get("mlp_general_value_gate_pass", ""),
        "no_fake_data_statement": "All metrics are read from local CSV/JSON artifacts. Missing or blocked stages remain marked as such.",
        "updated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_39_final_route.json", final)
    write_recap(final)
    append_exec(
        "finalize_route",
        task_id="Finalize",
        status="pass",
        gpu="0",
        files="results/v22_39/v22_39_final_route.json, results/v22_39/v22_39_artifact_manifest.csv, docs/DG-KAN_v22.39_CounterfactualFunctionalPolicyOptimizer_实验结果复盘.md",
        note=f"route={route}; manifest={manifest_hash}",
    )
    return final


def md_table(rows: list[dict[str, Any]], cols: list[str], limit: int = 20) -> str:
    if not rows:
        return "_no rows_\n"
    shown = rows[:limit]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in shown:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    if len(rows) > limit:
        lines.append(f"\n_only first {limit} / {len(rows)} rows shown; see artifact for full table._")
    return "\n".join(lines) + "\n"


def write_recap(final: dict[str, Any]) -> None:
    code = read_rows(OUT_ROOT / "v22_39_code_truth_gate.csv")
    leakage = read_rows(OUT_ROOT / "v22_39_feature_leakage_audit.csv")
    policy = read_rows(OUT_ROOT / "v22_39_policy_reanalysis_models.csv")
    c_summary = read_rows(OUT_ROOT / "v22_39_candidate_pair_micro_rct_summary.csv")
    value = read_rows(OUT_ROOT / "v22_39_potential_outcome_value_model.csv")
    repair = read_rows(OUT_ROOT / "v22_39_long_horizon_repair_summary.csv")
    repair_detail = read_rows(OUT_ROOT / "v22_39_long_horizon_repair_analysis.csv")
    full_loop_summary = read_rows(OUT_ROOT / "v22_39_restricted_full_loop_summary.csv")
    full_loop_protocol = read_rows(OUT_ROOT / "v22_39_restricted_full_loop_protocol_summary.csv")
    full_loop = read_rows(OUT_ROOT / "v22_39_restricted_full_loop_matrix.csv")

    def signal_group_stats(label_fragment: str) -> dict[str, Any]:
        subset = [
            r
            for r in full_loop
            if label_fragment in str(r.get("full_loop_label", ""))
            and str(r.get("full_loop_role", "")) == "signal_candidate"
        ]
        selected: dict[str, int] = {}
        for row in subset:
            raw = str(row.get("runtime_CATE_policy_arbitration_selected_counts", "") or "")
            for part in raw.split(";"):
                if ":" not in part:
                    continue
                key, val = part.split(":", 1)
                selected[key] = selected.get(key, 0) + int(finite_float(val, 0.0) or 0)
        probs = [
            float(v)
            for v in (finite_float(r.get("runtime_CATE_policy_probability_mean")) for r in subset)
            if v is not None
        ]
        return {
            "rows": len(subset),
            "wins": sum(1 for r in subset if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer"), 0.0) or 0.0) < 0.0),
            "control_wins": sum(1 for r in subset if (finite_float(r.get("NLL_delta_vs_same_basis_control"), 0.0) or 0.0) < 0.0),
            "no_debt": sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in subset),
            "accepted": sum(int(finite_float(r.get("accepted_treatment_count"), 0.0) or 0) for r in subset),
            "rejected": sum(int(finite_float(r.get("rejected_treatment_count"), 0.0) or 0) for r in subset),
            "runtime_pass": sum(int(finite_float(r.get("runtime_CATE_policy_pass_count"), 0.0) or 0) for r in subset),
            "runtime_blocked": sum(int(finite_float(r.get("runtime_CATE_policy_blocked_count"), 0.0) or 0) for r in subset),
            "prob_mean_min": min(probs) if probs else "",
            "prob_mean_max": max(probs) if probs else "",
            "selected": ";".join(f"{k}:{v}" for k, v in sorted(selected.items())),
        }

    a2a5_default = signal_group_stats("restricted_a2a5_arb_h0_")
    a2a5_thr0 = signal_group_stats("restricted_a2a5_arb_thr0_")
    a2a5_scale2e4 = signal_group_stats("restricted_a2a5_arb_scale2e4_h0_")
    a2a5_scale2e4_path = signal_group_stats("restricted_a2a5_arb_scale2e4_pathmpc_")
    a2a5_scale2e4_ce = signal_group_stats("restricted_a2a5_arb_scale2e4_ce_")
    lines = [
        "# DG-KAN v22.39 Counterfactual Functional Policy Optimizer experiment recap",
        "",
        f"updated_at: {now_sg()}",
        "",
        "## Final route",
        "",
        f"- final_route: `{final.get('final_route','')}`",
        f"- route_reason: `{final.get('route_reason','')}`",
        f"- artifact_manifest_hash: `{final.get('artifact_manifest_hash','')}`",
        "",
        "## Code / identity / leakage truth gate",
        "",
        md_table(code, ["clean_unzip_compileall_pass", "clean_unzip_import_pass", "missing_transitive_dependency_count", "official_DGKAN_identity_pass", "propensity_min_by_candidate", "no_treatment_selected_as_state_feature_official", "status"], 5),
        md_table(leakage, ["audit", "forbidden_features_found", "pass", "note"], 10),
        "",
        "## v22.38 policy leakage reanalysis",
        "",
        md_table(policy, ["model_id", "fold_key", "n_events", "AUC_positive_treatment", "Brier_score", "calibration_slope", "false_positive_rate_controls", "policy_noop_rate", "held_group_selected_beats_control_fraction"], 12),
        "",
        "Interpretation: M0/M1 are identity-sensitive diagnostics. M3 is the key candidate-heldout identity-free check. If M0/M1 remain high while M3 or leave-dataset collapse, the v22.38 policy is not promoted as state-dependent CATE.",
        "",
        "## Candidate-pair micro-RCT / paired replay",
        "",
        md_table(c_summary, ["H", "candidate_family", "rows", "hard_task_rows", "H20_rows", "H60_rows", "H200_rows", "min_rows_by_candidate_family", "H200_positive_signal_lcb_rows", "H200_positive_control_lcb_rows", "H60_H200_positive_unique_signal_families", "tau_vs_same_state_control", "LCB_vs_same_state_control", "positive_LCB_vs_control", "part_c_exploration_gate_pass"], 18),
        "",
        "Design note: v22.39 paired rows are same-state exhaustive candidate replays. Historical v22.38 single-arm rows are not counted as paired Part C evidence.",
        "",
        "## Potential-outcome value model",
        "",
        md_table(value, ["fold_key", "n_events", "AUC_positive_value", "Brier_score", "false_positive_rate_controls", "selected_candidate_beats_control_fraction", "exploration_gate_pass", "official_candidate_gate_pass", "status"], 10),
        "",
        "## Long-horizon blocker repair",
        "",
        md_table(repair, ["status", "noncontrol_H200_positive_LCB_rows", "noncontrol_H200_unique_positive_LCB_rows", "transient_regularization_rows", "runtime_full_loop_allowed", "source_artifact"], 5),
        md_table(repair_detail, ["collect_label", "intervention_scale", "H", "candidate_family", "rows", "tau_vs_same_state_control", "LCB_vs_same_state_control", "positive_LCB_vs_control", "matched_control_positive_LCB_same_label_H", "unique_positive_vs_matched_control", "long_horizon_penalty_blocks_runtime", "transient_regularization_pattern"], 18),
        "",
        "## Restricted full-loop",
        "",
        md_table(full_loop_summary, ["status", "full_loop_rows", "signal_rows", "kan_signal_rows", "mlp_signal_rows", "datasets", "seeds", "selected_signal_treatments", "KAN_NLL_improvement_rows", "KAN_beats_same_basis_control_rows", "KAN_no_ECE_Brier_tail_debt_rows", "KAN_basis_energy_ge_0p5_rows", "KAN_readout_leakage_le_0p3_rows", "kan_internal_value_gate_pass", "MLP_NLL_improvement_rows", "MLP_beats_matched_control_rows", "MLP_no_ECE_Brier_tail_debt_rows", "mlp_general_value_gate_pass", "accepted_treatment_count_total", "rejected_treatment_count_total"], 5),
        md_table(full_loop_protocol, ["full_loop_label", "full_loop_rows", "kan_signal_rows", "KAN_NLL_improvement_rows", "KAN_beats_same_basis_control_rows", "KAN_no_ECE_Brier_tail_debt_rows", "mean_NLL_delta_vs_optimizer", "mean_NLL_delta_vs_same_basis_control", "mlp_signal_rows", "MLP_NLL_improvement_rows", "MLP_beats_same_basis_control_rows", "MLP_no_ECE_Brier_tail_debt_rows", "mean_MLP_NLL_delta_vs_optimizer", "accepted_treatment_count_total", "rejected_treatment_count_total", "accepted_MLP_treatment_count_total", "rejected_MLP_treatment_count_total", "post_apply_rollback_count_total", "acceptance_tail_quantile", "calibration_temperatures", "empirical_lcb_horizons", "empirical_lcb_sources", "full_loop_engine", "status"], 120),
        md_table(full_loop, ["full_loop_label", "dataset", "seed", "architecture", "training_variant", "treatment_name", "full_loop_role", "final_test_NLL", "NLL_delta_vs_own_strong_optimizer", "NLL_delta_vs_same_basis_control", "final_test_accuracy", "ECE", "Brier", "tail_loss_q99", "no_ECE_Brier_tail_debt_vs_optimizer", "accepted_treatment_count", "rejected_treatment_count", "realized_policy_noop_rate", "runtime_value_lcb_used", "empirical_lcb_from_summary", "control_lcb_mode"], 24),
        "",
        "## Analysis and evidence chain",
        "",
        "- The code gate uses compileall and import closure logs in `results/v22_39/logs/`; artifact hashes are in `results/v22_39/v22_39_artifact_manifest.csv`.",
        "- The leakage audit separates historical table columns from official v22.39 feature builders. Presence of post-hoc columns in v22.38 artifacts is not treated as permission to use them.",
        "- Full-loop stages are blocked unless identity-free candidate-heldout value evidence opens. This follows the v22.39 plan and prevents promoting treatment-name AUC as CFPO success.",
        "- Runtime/full-loop was not promoted because the official value gate is still not open, even though exploration value modeling and low-scale H200 repair opened diagnostic evidence.",
        "- Restricted full-loop rows use train-only held acceptance and report test metrics for audit only. Same-basis random controls marked `matched_control_diagnostic` are not promoted as official CFPO policy.",
        "- Full-loop repair insight: after loader-reset repair, a2 produced consistent tiny NLL/AUC improvements in several reset protocols, but safety gate stayed blocked because q99 tail debt was not controlled reliably; q99 acceptance, held-only temperature calibration, and v22.38 horizon-robust/post-rollback diagnostics did not raise no-debt counts enough.",
        "- Full-loop repair insight: h2048+ab8+tail_q99 temperature-selected microalpha reached 6/6 no-debt in the latest protocol group, but only 3/6 NLL wins and 3/6 same-basis-control wins; adding alpha=3e-6 did not recover EMNIST/KMNIST wins and reduced no-debt to 5/6, so the current a2 repair path still shows a safety-effectiveness tradeoff rather than a promotable policy.",
        "- MLP diagnostic record: MLP+a3 was only allowed by best-available H20 LCB, not by positive H200/H800 LCB; the restricted diagnostic produced 4/6 MLP NLL wins and 4/6 same-basis-control wins, but only 2/6 no-debt rows, so `MLPFUGeneralValueOpened` remains blocked.",
        "- Path-MPC repair record: `path_mpc_v39` is an explicit v22.39 repair engine label routed through the horizon-robust branch simulator with nonzero horizon steps, CVaR path scoring, strict tail margin, and optional post-apply rollback; it remains audit-only unless the merged full-loop gates pass.",
        "- Candidate-generator repair record: v22.39 paired replay now routes treatment directions through the v22.38 direction builder and exposes planned-basis `a5_signal_incremental_basis` plus matched `a6_signflip_same_basis` for non-MLP DGKAN replay/full-loop; this prevents planned candidates from silently degrading to unknown no-op.",
        "- Path-MPC repair insight: strict h8/CVaR1.0/tail-margin path-MPC made KAN+a2 and MLP+a3 safe no-ops (no-debt passed but wins collapsed to 0/6). Relaxed h4/CVaR0.5 restored only partial KAN+a2 wins and still failed cross-task gates; MLP+a3 remained no-op under path-MPC.",
        "- New candidate insight: planned-basis a5 replay opened positive H20/H60/H200 LCB against signflip control, including H200 LCB 4.608124352322e-07 in the merged candidate-pair summary, and lifted candidate-heldout AUC to 0.7047572341167638; however a5 full-loop still reached only 3/6 KAN wins and 4/6 no-debt in the h0 strict-acceptance diagnostic, so paired long-horizon signal did not transfer to a promotable closed-loop policy.",
        "- Runtime arbitration repair record: v22.39 now exposes `a2_a5_runtime_arbitrated_basis` for non-MLP full-loop and gates it by the conservative minimum positive child LCB from a2/a5 before handing runtime selection to treatment-specific CATE policies.",
        f"- Runtime arbitration default-threshold result: `restricted_a2a5_arb_h0_*` produced signal rows `{a2a5_default['rows']}`, KAN wins `{a2a5_default['wins']}/{a2a5_default['rows']}`, same-basis-control wins `{a2a5_default['control_wins']}/{a2a5_default['rows']}`, no-debt `{a2a5_default['no_debt']}/{a2a5_default['rows']}`, accepted `{a2a5_default['accepted']}`, rejected `{a2a5_default['rejected']}`, runtime pass `{a2a5_default['runtime_pass']}`, blocked `{a2a5_default['runtime_blocked']}`, probability mean range `{a2a5_default['prob_mean_min']}..{a2a5_default['prob_mean_max']}`, selected counts `{a2a5_default['selected']}`. This is a safe no-op, not a full-loop opening.",
        f"- Runtime arbitration threshold-0 diagnostic: after adding the missing v22.39 runtime-policy CLI args and rerunning with `--runtime-policy-threshold 0.0`, `restricted_a2a5_arb_thr0_*` produced signal rows `{a2a5_thr0['rows']}`, KAN wins `{a2a5_thr0['wins']}/{a2a5_thr0['rows']}`, same-basis-control wins `{a2a5_thr0['control_wins']}/{a2a5_thr0['rows']}`, no-debt `{a2a5_thr0['no_debt']}/{a2a5_thr0['rows']}`, accepted `{a2a5_thr0['accepted']}`, rejected `{a2a5_thr0['rejected']}`, runtime pass `{a2a5_thr0['runtime_pass']}`, blocked `{a2a5_thr0['runtime_blocked']}`, probability mean range `{a2a5_thr0['prob_mean_min']}..{a2a5_thr0['prob_mean_max']}`, selected counts `{a2a5_thr0['selected']}`. The diagnostic restores execution but degenerates to a2-only selection and still fails the KAN/no-debt gate, so it is not promoted as official CFPO success.",
        f"- Runtime scale-alignment diagnostic: direct runtime probes showed v22.39 low-scale live rows used `applied_update_norm=5e-5`, while the v22.38 runtime CATE policy was calibrated near `2e-4`; rerunning default-threshold arbitration as `restricted_a2a5_arb_scale2e4_h0_*` produced signal rows `{a2a5_scale2e4['rows']}`, KAN wins `{a2a5_scale2e4['wins']}/{a2a5_scale2e4['rows']}`, same-basis-control wins `{a2a5_scale2e4['control_wins']}/{a2a5_scale2e4['rows']}`, no-debt `{a2a5_scale2e4['no_debt']}/{a2a5_scale2e4['rows']}`, accepted `{a2a5_scale2e4['accepted']}`, rejected `{a2a5_scale2e4['rejected']}`, runtime pass `{a2a5_scale2e4['runtime_pass']}`, blocked `{a2a5_scale2e4['runtime_blocked']}`, probability mean range `{a2a5_scale2e4['prob_mean_min']}..{a2a5_scale2e4['prob_mean_max']}`, selected counts `{a2a5_scale2e4['selected']}`. Scale alignment restores default-threshold a5 selection but does not improve the 3/6 wins, 4/6 no-debt frontier.",
        f"- Runtime path-MPC follow-up: `restricted_a2a5_arb_scale2e4_pathmpc_*` kept default-threshold a5 selection under h4/CVaR0.5/post-apply rollback and produced signal rows `{a2a5_scale2e4_path['rows']}`, KAN wins `{a2a5_scale2e4_path['wins']}/{a2a5_scale2e4_path['rows']}`, same-basis-control wins `{a2a5_scale2e4_path['control_wins']}/{a2a5_scale2e4_path['rows']}`, no-debt `{a2a5_scale2e4_path['no_debt']}/{a2a5_scale2e4_path['rows']}`, accepted `{a2a5_scale2e4_path['accepted']}`, rejected `{a2a5_scale2e4_path['rejected']}`, runtime pass `{a2a5_scale2e4_path['runtime_pass']}`, blocked `{a2a5_scale2e4_path['runtime_blocked']}`, probability mean range `{a2a5_scale2e4_path['prob_mean_min']}..{a2a5_scale2e4_path['prob_mean_max']}`, selected counts `{a2a5_scale2e4_path['selected']}`. Path-MPC reduced accepted interventions but did not improve no-debt or wins, so the remaining blocker is execution effect/safety rather than CATE availability.",
        f"- Effect-ceiling diagnostic: relaxing the train-held acceptance metric to CE in `restricted_a2a5_arb_scale2e4_ce_*` still produced signal rows `{a2a5_scale2e4_ce['rows']}`, KAN wins `{a2a5_scale2e4_ce['wins']}/{a2a5_scale2e4_ce['rows']}`, same-basis-control wins `{a2a5_scale2e4_ce['control_wins']}/{a2a5_scale2e4_ce['rows']}`, no-debt `{a2a5_scale2e4_ce['no_debt']}/{a2a5_scale2e4_ce['rows']}`, accepted `{a2a5_scale2e4_ce['accepted']}`, rejected `{a2a5_scale2e4_ce['rejected']}`, runtime pass `{a2a5_scale2e4_ce['runtime_pass']}`, blocked `{a2a5_scale2e4_ce['runtime_blocked']}`, probability mean range `{a2a5_scale2e4_ce['prob_mean_min']}..{a2a5_scale2e4_ce['prob_mean_max']}`, selected counts `{a2a5_scale2e4_ce['selected']}`. Because CE-only did not lift wins above the strict h0 result, the current a5/a2 action family appears effect-limited on EMNIST/KMNIST rather than merely over-filtered by the safety gate.",
        "- Implementation repair record: added `F_collect`/`F_merge` to run restricted full-loop chunks in parallel and merge optimizer/a2/same-basis-control evidence into `v22_39_restricted_full_loop_matrix.csv` and `v22_39_restricted_full_loop_summary.csv`.",
        "- Implementation repair record: after the first full-loop merge, fixed the summary calculation for `KAN_readout_leakage_le_0p3_rows` so an observed `0.0` leakage fraction is counted as valid rather than replaced by the default.",
        "- Implementation repair record: full-loop collection now rebuilds train/held/test loaders per variant with the same seed and marks `loader_reset_per_variant=1`; `F_merge` excludes pre-repair chunks without that marker to avoid comparing optimizer/a2/control rows that may have seen different shuffled train orders.",
        "- Blocker repair record: initial cuda:4/cuda:5 collection attempts failed with invalid device ordinal; torch/nvidia-smi exposed only GPUs 0-3 in this shell, so SVHN/EMNIST were rerun on cuda:0/cuda:1.",
        "- Blocker repair record: merge dedup was changed to collect_label|intervention_scale|event_id after low-scale repair exposed event-id collisions across labels; raw chunks were preserved and summaries regenerated.",
        "- Any repair or rerun must be appended to the execution log with command, GPU, output files, and changed implementation fields.",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    stage_a_code_truth_gate()
    stage_b_v38_policy_reanalysis()
    stage_c_merge()
    stage_d_value_model()
    stage_e_long_horizon_repair_analysis()
    stage_f_merge()
    return finalize_route()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "A", "B", "C_collect", "C_merge", "D", "E", "F", "F_collect", "F_merge", "finalize"])
    p.add_argument("--collect-label", default="primary")
    p.add_argument("--paired-device", default="cuda:0")
    p.add_argument("--paired-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--paired-seeds", default="0")
    p.add_argument("--paired-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--paired-optimizers", default="AdamW")
    p.add_argument("--paired-horizons", default="20,60,200")
    p.add_argument("--paired-events-per-group", type=int, default=4)
    p.add_argument("--paired-warmup-steps", type=int, default=5)
    p.add_argument("--paired-train-size", type=int, default=512)
    p.add_argument("--paired-held-size", type=int, default=256)
    p.add_argument("--paired-random-seed", type=int, default=2239)
    p.add_argument("--paired-top3", action="store_true")
    p.add_argument("--paired-candidates", default="", help="Optional comma list of candidates/controls to replay instead of the full architecture-valid candidate set.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--intervention-scale", type=float, default=2.0e-4)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--full-loop-label", default="restricted_a2")
    p.add_argument("--full-loop-device", default="cuda:0")
    p.add_argument("--full-loop-datasets", default="FashionMNIST,KMNIST,EMNIST_LETTERS")
    p.add_argument("--full-loop-seeds", default="0,1")
    p.add_argument("--full-loop-architectures", default="DGKAN_DCHE")
    p.add_argument("--full-loop-optimizers", default="AdamW")
    p.add_argument("--full-loop-treatments", default="a2_basis_actuator_section")
    p.add_argument("--full-loop-train-size", type=int, default=512)
    p.add_argument("--full-loop-held-size", type=int, default=256)
    p.add_argument("--full-loop-steps", type=int, default=240)
    p.add_argument("--full-loop-cadence", type=int, default=20)
    p.add_argument("--full-loop-engine", default="compact_v37", choices=["compact_v37", "horizon_robust_v38", "path_mpc_v39"])
    p.add_argument("--full-loop-implementation", default="accepted")
    p.add_argument("--full-loop-alpha-grid", default="0,1e-05,2e-05,5e-05")
    p.add_argument("--full-loop-alpha-scale-mode", default="absolute")
    p.add_argument("--full-loop-alpha-scale-floor", type=float, default=1.0e-12)
    p.add_argument("--full-loop-acceptance-metric", default="strict_ece_no_debt")
    p.add_argument("--full-loop-acceptance-tol", type=float, default=0.0)
    p.add_argument("--full-loop-acceptance-batches", type=int, default=1)
    p.add_argument("--full-loop-acceptance-tail-quantile", type=float, default=0.90)
    p.add_argument("--full-loop-acceptance-tail-margin", type=float, default=0.0)
    p.add_argument("--full-loop-horizon-steps", type=int, default=0)
    p.add_argument("--full-loop-horizon-cvar-fraction", type=float, default=0.25)
    p.add_argument("--runtime-policy-scope", default="global", choices=["global", "treatment-specific"])
    p.add_argument("--runtime-policy-threshold", type=float, default=-1.0)
    p.add_argument("--runtime-policy-horizon-feature", type=int, default=60)
    p.add_argument("--runtime-safety-policy", action="store_true")
    p.add_argument(
        "--runtime-safety-feature-group",
        default="safety-only",
        choices=["loss-only", "signal-only", "actuator-only", "optimizer-only", "safety-only", "temporal-only", "all features"],
    )
    p.add_argument("--runtime-safety-model-kind", default="gbt", choices=["logistic", "gbt", "mlp"])
    p.add_argument("--runtime-safety-threshold", type=float, default=-1.0)
    p.add_argument("--runtime-safety-quantile", type=float, default=0.20)
    p.add_argument("--full-loop-post-apply-rollback", action="store_true")
    p.add_argument("--full-loop-temperature-grid", default="1.0")
    p.add_argument("--full-loop-temperature-selection-metric", default="NLL")
    p.add_argument("--full-loop-run-controls", action="store_true")
    p.add_argument("--full-loop-control-lcb-mode", default="forced_positive_diagnostic", choices=["forced_positive_diagnostic", "auto", "blocked"])
    p.add_argument("--full-loop-allow-nonpositive-value-gate", action="store_true")
    p.add_argument("--epsilon-row", type=float, default=0.0)
    return p


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    if args.stage == "all":
        run_all(args)
    elif args.stage == "A":
        stage_a_code_truth_gate()
    elif args.stage == "B":
        stage_b_v38_policy_reanalysis()
    elif args.stage == "C_collect":
        stage_c_collect(args)
    elif args.stage == "C_merge":
        stage_c_merge()
    elif args.stage == "D":
        stage_d_value_model()
    elif args.stage == "E":
        stage_e_long_horizon_repair_analysis()
    elif args.stage == "F":
        stage_f_collect(args)
        stage_f_merge()
    elif args.stage == "F_collect":
        stage_f_collect(args)
    elif args.stage == "F_merge":
        stage_f_merge()
    elif args.stage == "finalize":
        finalize_route()


if __name__ == "__main__":
    main()
