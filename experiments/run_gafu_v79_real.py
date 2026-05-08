#!/usr/bin/env python3
"""DG-KAN v7.9 code-first teacher-free autonomous advantage runner.

v7.9 keeps the audited real training/evaluation path from v7.6/v7.3, but moves
the official decision logic out of candidate-id guesses and into explicit
candidate metadata plus a measured strict-contract validator.

No rows in this runner are fake/proxy rows.  Unsupported stages are recorded as
``not_run`` or ``not_implemented`` and never counted as pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import run_gafu_v76_real as v76
import run_gafu_v78_real as v78
import run_gafu_v72_real as v72
from dgkan_core import parse_int_list, parse_str_list, save_json, write_csv
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.9_CodeFirst_TeacherFree_AutonomousAdvantage_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v79_real.py"
METRIC_UNAVAILABLE = v76.METRIC_UNAVAILABLE


@dataclass(frozen=True)
class CandidateMeta:
    candidate_id: str
    candidate_name: str
    model_family: str
    dense_cls: str
    hidden_dim: str
    basis_count: str
    depth: str
    init_policy: str
    optimizer_policy: str
    external_teacher_used: int
    self_teacher_used: int
    teacher_source: str
    teacher_logits_used: int
    teacher_forward_used: int
    teacher_checkpoint_path: str
    teacher_precompute_path: str
    teacher_artifact_hash: str
    official_eligible: int
    expected_manual_forward: int
    expected_manual_backward: int
    expected_manual_update: int
    expected_nonkan_count: int
    implementation_status: str = "implemented"
    route_role: str = "diagnostic"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else default


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash_file(path: Path) -> str:
    return v72._hash_file(path) if path.exists() else ""


def _base_meta_registry(args: argparse.Namespace) -> Dict[str, CandidateMeta]:
    hidden = str(getattr(args, "hidden_dim", "runtime_arg"))
    basis = str(getattr(args, "basis_count", "runtime_arg"))

    def meta(
        cid: str,
        name: str,
        family: str,
        *,
        external: int = 0,
        self_teacher: int = 0,
        teacher: str = "none",
        official: int = 1,
        manual: int = 1,
        dense_cls: str = "manual_purekan",
        init: str = "matched_seed",
        role: str = "official",
        status: str = "implemented",
    ) -> CandidateMeta:
        return CandidateMeta(
            candidate_id=cid,
            candidate_name=name,
            model_family=family,
            dense_cls=dense_cls,
            hidden_dim=hidden,
            basis_count=basis,
            depth="runtime_spec",
            init_policy=init,
            optimizer_policy="manual_fast_adamw_no_loss_backward" if manual else "torch_adamw",
            external_teacher_used=external,
            self_teacher_used=self_teacher,
            teacher_source=teacher,
            teacher_logits_used=int(external or self_teacher),
            teacher_forward_used=int(external),
            teacher_checkpoint_path="none",
            teacher_precompute_path="none",
            teacher_artifact_hash=METRIC_UNAVAILABLE if external else "none",
            official_eligible=official,
            expected_manual_forward=manual,
            expected_manual_backward=manual,
            expected_manual_update=manual,
            expected_nonkan_count=0 if manual else -1,
            implementation_status=status,
            route_role=role,
        )

    registry: Dict[str, CandidateMeta] = {
        "B0": meta("B0", "MLP-AdamW-reference", "mlp_reference", official=0, manual=0, dense_cls="mlp", role="baseline"),
        "B2": meta("B2", "MLP-C3-logit-distill-diagnostic", "mlp_external_teacher_diagnostic", external=1, teacher="C3", official=0, manual=0, dense_cls="mlp", role="diagnostic_external"),
        "M12": meta("M12", "M12-C3-logit-distill-diagnostic", "purekan_external_teacher_diagnostic", external=1, teacher="C3", official=0, role="diagnostic_external"),
        "M13": meta("M13", "M13-teacher-free-M5init-official", "purekan_teacher_free", init="M5init_no_external_teacher", role="official"),
        "M4": meta("M4", "A2S-fused-linear-silu-and-poly2-silu-head", "purekan_teacher_free_structural", init="matched_seed", role="official"),
        "M9": meta("M9", "A2S-fused-linear-silu-stack-packed-generic-head", "purekan_teacher_free_structural", init="matched_seed", role="official"),
        "M1": meta("M1", "A2S-fused-linear-silu-stack", "purekan_teacher_free_structural", init="matched_seed", role="official"),
        "M7": meta("M7", "A2S-packed-fused-linear-silu-stack", "purekan_teacher_free_structural", init="matched_seed", role="official"),
        "A2S": meta("A2S", "A2S-cached-linear-stack-poly2-silu-head", "purekan_teacher_free_structural", init="matched_seed", role="official"),
        "E1": meta("E1", "A2S-recompute-y-checkpoint-stack", "purekan_teacher_free_s1_diagnostic", init="matched_seed", role="official"),
        "Z1": meta("Z1", "A2S-cache-y-only-stack", "purekan_teacher_free_s1_diagnostic", init="matched_seed", role="official"),
        "U1": meta("U1", "A2S-cache-late-y-stack", "purekan_teacher_free_s1_diagnostic", init="matched_seed", role="official"),
    }
    for cid, family in {
        "SB0": "self_bootstrap_ema",
        "SB1": "self_bootstrap_delayed",
        "SB2": "self_bootstrap_snapshot",
        "SB3": "self_bootstrap_dual_view",
        "RR1": "representation_sparseinterp",
        "RR2": "representation_sparsespline",
        "RR3": "representation_dwm2lite",
        "RR5": "representation_prototype_head",
        "SYS1": "system_root_input_lifetime_trim",
    }.items():
        registry[cid] = meta(
            cid,
            f"{cid}-{family}",
            family,
            self_teacher=1 if cid.startswith("SB") else 0,
            official=1 if not cid.startswith("SYS") else 0,
            role="planned_no_external",
            status="not_implemented",
        )
    return registry


def _summary_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows}


def _macro_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows if str(row.get("dataset")) == "macro"}


def _grad_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for cid, items in grouped.items():
        rels = [_float(row.get("grad_relerr_max")) for row in items if _finite(row.get("grad_relerr_max"))]
        coss = [_float(row.get("grad_cos_min")) for row in items if _finite(row.get("grad_cos_min"))]
        out[cid] = {
            "grad_rows": len(items),
            "grad_pass_rows": sum(1 for row in items if _is_one(row.get("grad_pass"))),
            "grad_pass": int(bool(items) and all(_is_one(row.get("grad_pass")) for row in items)),
            "grad_relerr_max": max(rels) if rels else METRIC_UNAVAILABLE,
            "grad_cos_min": min(coss) if coss else METRIC_UNAVAILABLE,
        }
    return out


def _eff_counts(rows: Sequence[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    items = [r for r in rows if str(r.get("candidate_id")) == cid]
    mems = [_float(r.get("memory_ratio")) for r in items if _finite(r.get("memory_ratio"))]
    steps = [_float(r.get("step_ratio")) for r in items if _finite(r.get("step_ratio"))]
    return {
        "eff_rows": len(items),
        "memory_ratio_mean": _mean(mems),
        "memory_ratio_max": max(mems) if mems else METRIC_UNAVAILABLE,
        "step_ratio_mean": _mean(steps),
        "step_ratio_max": max(steps) if steps else METRIC_UNAVAILABLE,
        "s2_shape_count": sum(1 for r in items if _float(r.get("memory_ratio"), 99.0) <= 1.05 and _float(r.get("step_ratio"), 99.0) <= 1.50),
        "s1_shape_count": sum(1 for r in items if _float(r.get("memory_ratio"), 99.0) < 1.00 and _float(r.get("step_ratio"), 99.0) <= 1.35),
    }


def _time_auc_ratio(p7_rows: Sequence[Dict[str, Any]], candidate: str, baseline: str = "B0") -> Dict[str, Any]:
    c_step = _mean(row.get("val_loss_auc_step") for row in p7_rows if str(row.get("candidate_id")) == candidate)
    b_step = _mean(row.get("val_loss_auc_step") for row in p7_rows if str(row.get("candidate_id")) == baseline)
    c_time = _mean(row.get("val_loss_auc_time") for row in p7_rows if str(row.get("candidate_id")) == candidate)
    b_time = _mean(row.get("val_loss_auc_time") for row in p7_rows if str(row.get("candidate_id")) == baseline)
    return {
        "val_loss_auc_step": c_step if _finite(c_step) else METRIC_UNAVAILABLE,
        "val_loss_auc_time": c_time if _finite(c_time) else METRIC_UNAVAILABLE,
        "val_loss_auc_step_ratio_vs_B0": c_step / b_step if _finite(c_step) and _finite(b_step) and b_step else METRIC_UNAVAILABLE,
        "val_loss_auc_time_ratio_vs_B0": c_time / b_time if _finite(c_time) and _finite(b_time) and b_time else METRIC_UNAVAILABLE,
        "time_auc_pass": int(_finite(c_time) and _finite(b_time) and c_time <= b_time),
    }


def _seed_win_rate(task_rows: Sequence[Dict[str, Any]], candidate: str, baseline: str = "B0") -> Tuple[float, int, int]:
    by_seed: Dict[int, Dict[str, List[float]]] = {}
    for row in task_rows:
        cid = str(row.get("candidate_id"))
        if cid not in {candidate, baseline}:
            continue
        try:
            seed = int(row.get("seed"))
        except Exception:
            continue
        by_seed.setdefault(seed, {}).setdefault(cid, []).append(_float(row.get("val_acc")))
    wins = 0
    total = 0
    for data in by_seed.values():
        if candidate in data and baseline in data:
            total += 1
            wins += int(_mean(data[candidate]) > _mean(data[baseline]))
    return (wins / total if total else float("nan"), wins, total)


def _macro_gap_between(task_rows: Sequence[Dict[str, Any]], a: str, b: str) -> float:
    datasets = sorted({str(row.get("dataset")) for row in task_rows})
    gaps: List[float] = []
    for ds in datasets:
        av = _mean(row.get("val_acc") for row in task_rows if str(row.get("candidate_id")) == a and str(row.get("dataset")) == ds)
        bv = _mean(row.get("val_acc") for row in task_rows if str(row.get("candidate_id")) == b and str(row.get("dataset")) == ds)
        if _finite(av) and _finite(bv):
            gaps.append(av - bv)
    return _mean(gaps)


def _nll_between(task_rows: Sequence[Dict[str, Any]], a: str, b: str) -> float:
    av = _mean(row.get("NLL") for row in task_rows if str(row.get("candidate_id")) == a)
    bv = _mean(row.get("NLL") for row in task_rows if str(row.get("candidate_id")) == b)
    return av - bv if _finite(av) and _finite(bv) else float("nan")


def _first_task_row(task_rows: Sequence[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    return next((r for r in task_rows if str(r.get("candidate_id")) == cid), {})


def _write_candidate_registry(out_dir: Path, registry: Dict[str, CandidateMeta], measured_ids: Sequence[str]) -> List[Dict[str, Any]]:
    measured = set(measured_ids)
    rows: List[Dict[str, Any]] = []
    for cid in sorted(registry):
        row = asdict(registry[cid])
        row.update({
            "stage": "P1_CANDIDATE_METADATA",
            "measured_in_run": int(cid in measured),
            "can_enter_official_route": int(registry[cid].official_eligible and registry[cid].external_teacher_used == 0 and registry[cid].implementation_status == "implemented"),
            "can_enter_diagnostic_route": 1,
            "reason_if_not_official": "eligible" if registry[cid].official_eligible else ("external_teacher" if registry[cid].external_teacher_used else "diagnostic_or_not_implemented"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        rows.append(row)
    write_csv(out_dir / "candidate_registry.csv", rows)
    write_csv(out_dir / "p1_candidate_metadata_audit.csv", rows)
    return rows


def _write_contract_validator(
    out_dir: Path,
    registry: Dict[str, CandidateMeta],
    measured_ids: Sequence[str],
    task_summary: Sequence[Dict[str, Any]],
    task_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    by_summary = _summary_by_candidate(task_summary)
    rows: List[Dict[str, Any]] = []
    for cid in measured_ids:
        meta = registry.get(cid)
        summary = by_summary.get(cid, {})
        first = _first_task_row(task_rows, cid)
        expected_manual = meta.expected_manual_forward if meta else METRIC_UNAVAILABLE
        actual_forward = int(_is_one(summary.get("manual_forward"))) if summary else 0
        actual_backward = int(_is_one(summary.get("manual_backward"))) if summary else 0
        actual_update = int(_is_one(summary.get("manual_update"))) if summary else 0
        uses_loss_backward = int(_is_one(first.get("uses_torch_loss_backward"))) if first else 1
        uses_autograd_graph = int(_is_one(first.get("uses_torch_autograd_graph"))) if first else 1
        nonkan = _float(summary.get("non_kan_trainable_param_count"), float("nan"))
        edge_count = _float(first.get("edge_param_count"), float("nan"))
        kan_count = _float(first.get("kan_trainable_param_count"), float("nan"))
        head_is_kan = int(_is_one(summary.get("head_is_kan")))
        is_manual_candidate = bool(meta and meta.expected_manual_forward)
        strict = int(
            bool(meta)
            and is_manual_candidate
            and _finite(nonkan) and int(nonkan) == int(meta.expected_nonkan_count)
            and head_is_kan == 1
            and actual_forward == 1
            and actual_backward == 1
            and actual_update == 1
            and uses_loss_backward == 0
        )
        rows.append({
            "stage": "P0_CODE_CONTRACT",
            "candidate_id": cid,
            "candidate_name": summary.get("candidate_name", meta.candidate_name if meta else METRIC_UNAVAILABLE),
            "metadata_present": int(bool(meta)),
            "official_eligible": meta.official_eligible if meta else 0,
            "external_teacher_used": meta.external_teacher_used if meta else METRIC_UNAVAILABLE,
            "expected_manual_forward": expected_manual,
            "expected_manual_backward": meta.expected_manual_backward if meta else METRIC_UNAVAILABLE,
            "expected_manual_update": meta.expected_manual_update if meta else METRIC_UNAVAILABLE,
            "actual_manual_forward": actual_forward,
            "actual_manual_backward": actual_backward,
            "actual_manual_update": actual_update,
            "uses_loss_backward": uses_loss_backward,
            "uses_torch_autograd_graph": uses_autograd_graph,
            "nonKAN_param_count": int(nonkan) if _finite(nonkan) else METRIC_UNAVAILABLE,
            "KAN_edge_param_count": int(kan_count) if _finite(kan_count) else METRIC_UNAVAILABLE,
            "head_is_kan": head_is_kan,
            "input_kan_is_kan": METRIC_UNAVAILABLE,
            "output_kan_is_kan": head_is_kan,
            "edge_param_coverage": (edge_count / kan_count if _finite(edge_count) and _finite(kan_count) and kan_count else METRIC_UNAVAILABLE),
            "strict_pass": strict,
            "strict_fail_reason": "pass" if strict else (
                "non_manual_reference" if meta and not is_manual_candidate else
                "metadata_missing_or_contract_field_fail"
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "contract_validator.csv", rows)
    write_csv(out_dir / "p0_code_contract.csv", rows)
    measured_official = [r for r in rows if registry.get(str(r["candidate_id"]), None) and registry[str(r["candidate_id"])].official_eligible]
    contract_pass = int(bool(rows) and all(r["metadata_present"] for r in rows) and all(_is_one(r["strict_pass"]) for r in measured_official))
    return rows, contract_pass


def _write_teacher_leak_audit(out_dir: Path, registry: Dict[str, CandidateMeta], measured_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cid in measured_ids:
        meta = registry.get(cid)
        if not meta:
            rows.append({
                "stage": "P1_TEACHER_LEAK_AUDIT",
                "candidate_id": cid,
                "metadata_present": 0,
                "teacher_leak_detected": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
            continue
        leak = int(meta.official_eligible and (meta.external_teacher_used or meta.teacher_logits_used or meta.teacher_forward_used))
        rows.append({
            "stage": "P1_TEACHER_LEAK_AUDIT",
            "candidate_id": cid,
            "candidate_name": meta.candidate_name,
            "official_eligible": meta.official_eligible,
            "external_teacher_used": meta.external_teacher_used,
            "self_teacher_used": meta.self_teacher_used,
            "teacher_source": meta.teacher_source,
            "teacher_logits_used": meta.teacher_logits_used,
            "teacher_forward_used": meta.teacher_forward_used,
            "teacher_checkpoint_path": meta.teacher_checkpoint_path,
            "teacher_precompute_path": meta.teacher_precompute_path,
            "teacher_artifact_hash": meta.teacher_artifact_hash,
            "teacher_leak_detected": leak,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "teacher_leak_audit.csv", rows)
    return rows


def _macro_pass(macro: Dict[str, Any]) -> int:
    gap = _float(macro.get("mean_val_gap"))
    ci = _float(macro.get("bootstrap_ci95_low"))
    holm = _float(macro.get("holm_corrected_p"))
    test_gap = _float(macro.get("mean_test_gap"))
    ece = _float(macro.get("ECE_delta_macro", macro.get("ECE_delta")))
    nll = _float(macro.get("NLL_delta_macro", macro.get("NLL_delta")))
    return int(
        _finite(gap) and gap >= 0.0200
        and _finite(ci) and ci > 0
        and _finite(holm) and holm < 0.05
        and _finite(test_gap) and test_gap >= 0.015
        and _finite(ece) and ece <= 0.005
        and _finite(nll) and nll <= 0.01
    )


def _candidate_decision_row(
    cid: str,
    *,
    registry: Dict[str, CandidateMeta],
    task_by: Dict[str, Dict[str, Any]],
    macro_by: Dict[str, Dict[str, Any]],
    grad_by: Dict[str, Dict[str, Any]],
    strict_by: Dict[str, Dict[str, Any]],
    eff_rows: Sequence[Dict[str, Any]],
    p7_rows: Sequence[Dict[str, Any]],
    task_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    meta = registry.get(cid)
    task = task_by.get(cid, {})
    macro = macro_by.get(cid, {})
    grad = grad_by.get(cid, {})
    strict = strict_by.get(cid, {})
    eff = _eff_counts(eff_rows, cid)
    auc = _time_auc_ratio(p7_rows, cid)
    win_rate, wins, total = _seed_win_rate(task_rows, cid)
    full_s2 = int(eff["eff_rows"] > 0 and eff["s2_shape_count"] == eff["eff_rows"])
    full_s1 = int(eff["eff_rows"] > 0 and eff["s1_shape_count"] == eff["eff_rows"])
    macro_pass = _macro_pass(macro)
    official = int(bool(meta and meta.official_eligible and not meta.external_teacher_used and meta.implementation_status == "implemented"))
    return {
        "stage": "P16_ROUTE_INPUT",
        "candidate_id": cid,
        "candidate_name": task.get("candidate_name", meta.candidate_name if meta else METRIC_UNAVAILABLE),
        "model_family": meta.model_family if meta else METRIC_UNAVAILABLE,
        "external_teacher_used": meta.external_teacher_used if meta else METRIC_UNAVAILABLE,
        "self_teacher_used": meta.self_teacher_used if meta else METRIC_UNAVAILABLE,
        "official_eligible": official,
        "strict_pass": strict.get("strict_pass", 0),
        "grad_pass": grad.get("grad_pass", 0 if official else METRIC_UNAVAILABLE),
        "teacher_free_macro_pass": int(official and macro_pass),
        "macro_significant_pass": macro_pass,
        "fullgrid_s2_pass": full_s2,
        "fullgrid_s1_pass": full_s1,
        "time_auc_pass": auc["time_auc_pass"],
        "profiler_pass": 0,
        "val_acc_mean": task.get("val_acc_mean", METRIC_UNAVAILABLE),
        "test_acc_mean": task.get("test_acc_mean", METRIC_UNAVAILABLE),
        "macro_gap": macro.get("mean_val_gap", task.get("val_gap_vs_MLP_mean", METRIC_UNAVAILABLE)),
        "ci95_low": macro.get("bootstrap_ci95_low", METRIC_UNAVAILABLE),
        "holm_p": macro.get("holm_corrected_p", METRIC_UNAVAILABLE),
        "test_gap": macro.get("mean_test_gap", METRIC_UNAVAILABLE),
        "ECE_delta": macro.get("ECE_delta_macro", macro.get("ECE_delta", METRIC_UNAVAILABLE)),
        "NLL_delta": macro.get("NLL_delta_macro", macro.get("NLL_delta", METRIC_UNAVAILABLE)),
        "seed_win_rate": win_rate if _finite(win_rate) else METRIC_UNAVAILABLE,
        "seed_win_count": wins,
        "seed_count": total,
        "memory_ratio_mean": eff["memory_ratio_mean"],
        "memory_ratio_max": eff["memory_ratio_max"],
        "step_ratio_mean": eff["step_ratio_mean"],
        "step_ratio_max": eff["step_ratio_max"],
        "s2_shape_count": eff["s2_shape_count"],
        "s1_shape_count": eff["s1_shape_count"],
        "eff_shape_count": eff["eff_rows"],
        "grad_relerr_max": grad.get("grad_relerr_max", METRIC_UNAVAILABLE),
        "grad_cos_min": grad.get("grad_cos_min", METRIC_UNAVAILABLE),
        **auc,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _write_stage_artifacts(
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    task_rows: Sequence[Dict[str, Any]],
    task_by: Dict[str, Dict[str, Any]],
) -> None:
    official_rows = [r for r in rows if _is_one(r.get("official_eligible"))]
    write_csv(out_dir / "p2_teacher_free_baseline.csv", official_rows)
    write_csv(out_dir / "p8_teacher_free_10seed.csv", official_rows)
    write_csv(out_dir / "p16_route_decision.csv", rows)

    # P3 is not a substitute for repeated validation shards.  It records the
    # measured val/test discrepancy only and explicitly does not pass split audit.
    p3_rows: List[Dict[str, Any]] = []
    for row in official_rows:
        cid = str(row["candidate_id"])
        p3_rows.append({
            "stage": "P3_VALIDATION_SPLIT_AUDIT",
            "candidate_id": cid,
            "split_audit_status": "not_run",
            "measured_original_val_gap": row.get("macro_gap", METRIC_UNAVAILABLE),
            "measured_test_gap": row.get("test_gap", METRIC_UNAVAILABLE),
            "val_minus_test_gap": (_float(row.get("macro_gap")) - _float(row.get("test_gap"))) if _finite(row.get("macro_gap")) and _finite(row.get("test_gap")) else METRIC_UNAVAILABLE,
            "macro_gap_split_mean": METRIC_UNAVAILABLE,
            "CI95_low_split": METRIC_UNAVAILABLE,
            "Holm_p_split": METRIC_UNAVAILABLE,
            "validation_split_effect_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p3_validation_split_audit.csv", p3_rows)

    p4_rows: List[Dict[str, Any]] = []
    if "M12" in task_by and "M13" in task_by:
        p4_rows.append({
            "stage": "P4_TEACHER_GAP_ATTRIBUTION",
            "candidate_teacher_assisted": "M12",
            "candidate_teacher_free": "M13",
            "macro_gap_M12_minus_M13": _macro_gap_between(task_rows, "M12", "M13"),
            "NLL_M12_minus_M13": _nll_between(task_rows, "M12", "M13"),
            "feature_CKA_M13_M12": METRIC_UNAVAILABLE,
            "logit_KL_M12_M13": METRIC_UNAVAILABLE,
            "error_overlap_rate": METRIC_UNAVAILABLE,
            "hard_sample_gain": METRIC_UNAVAILABLE,
            "attribution_status": "partial_measured_metric_delta_only",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p4_teacher_gap_attribution.csv", p4_rows)

    empty_stages = {
        "p5_self_bootstrap.csv": ("P5_SELF_BOOTSTRAP", "not_implemented"),
        "p9_time_accounting.csv": ("P9_TIME_ACCOUNTING", "not_run"),
        "p10_phase_mapped_profiler.csv": ("P10_PHASE_MAPPED_PROFILER", "not_run"),
        "p12_fused_streaming_package.csv": ("P12_FUSED_STREAMING_PACKAGE", "not_implemented"),
        "p14_scaling_robustness.csv": ("P14_SCALING_ROBUSTNESS", "not_run"),
        "p15_geometry_pareto.csv": ("P15_GEOMETRY_PARETO", "not_run"),
    }
    for filename, (stage, status) in empty_stages.items():
        write_csv(out_dir / filename, [{
            "stage": stage,
            "stage_status": status,
            "pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }])

    repair_rows = []
    base = next((r for r in rows if str(r.get("candidate_id")) == "M13"), {})
    base_gap = _float(base.get("macro_gap"))
    for row in official_rows:
        cid = str(row.get("candidate_id"))
        if cid == "M13":
            continue
        gap = _float(row.get("macro_gap"))
        repair_rows.append({
            "stage": "P6_TEACHER_FREE_REPAIR",
            "candidate_id": cid,
            "repair_type": "measured_teacher_free_structural_or_runtime_candidate",
            "external_teacher_used": row.get("external_teacher_used"),
            "macro_gap": row.get("macro_gap"),
            "macro_delta_vs_M13": gap - base_gap if _finite(gap) and _finite(base_gap) else METRIC_UNAVAILABLE,
            "teacher_free_macro_pass": row.get("teacher_free_macro_pass"),
            "fullgrid_s2_pass": row.get("fullgrid_s2_pass"),
            "grad_pass": row.get("grad_pass"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p6_teacher_free_repair.csv", repair_rows or [{
        "stage": "P6_TEACHER_FREE_REPAIR",
        "stage_status": "not_run",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p7_representation_repair.csv", repair_rows or [{
        "stage": "P7_REPRESENTATION_REPAIR",
        "stage_status": "not_run",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])

    p11_rows = []
    for row in official_rows:
        cid = str(row.get("candidate_id"))
        p11_rows.append({
            "stage": "P11_S1_MEMORY_ATTRIBUTION",
            "candidate_id": cid,
            "memory_ratio_max": row.get("memory_ratio_max"),
            "step_ratio_max": row.get("step_ratio_max"),
            "s1_shape_count": row.get("s1_shape_count"),
            "s1_attribution_status": "uses_measured_p10_efficiency_sources",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p11_s1_memory_attribution.csv", p11_rows)
    write_csv(out_dir / "p13_fullgrid_profiler.csv", [{
        "stage": "P13_FULLGRID_PROFILER",
        "candidate_id": row.get("candidate_id"),
        "memory_ratio_max": row.get("memory_ratio_max"),
        "step_ratio_max": row.get("step_ratio_max"),
        "s2_shape_count": row.get("s2_shape_count"),
        "s1_shape_count": row.get("s1_shape_count"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    } for row in official_rows])


def _write_v79_postprocess(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    task_rows = _read_csv_rows(out_dir / "p9_task_gate.csv")
    task_summary = _read_csv_rows(out_dir / "p9_task_summary.csv")
    sig_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    p7_rows = _read_csv_rows(out_dir / "p7_time_auc_v76.csv")
    task_by = _summary_by_candidate(task_summary)
    macro_by = {str(row.get("candidate_id")): row for row in sig_rows if str(row.get("dataset")) == "macro"}
    grad_by = _grad_summary(grad_rows)
    measured_ids = sorted(task_by)
    registry = _base_meta_registry(args)
    registry_rows = _write_candidate_registry(out_dir, registry, measured_ids)
    contract_rows, code_contract_pass = _write_contract_validator(out_dir, registry, measured_ids, task_summary, task_rows)
    leak_rows = _write_teacher_leak_audit(out_dir, registry, measured_ids)
    strict_by = {str(row.get("candidate_id")): row for row in contract_rows}
    rows = [
        _candidate_decision_row(
            cid,
            registry=registry,
            task_by=task_by,
            macro_by=macro_by,
            grad_by=grad_by,
            strict_by=strict_by,
            eff_rows=eff_rows,
            p7_rows=p7_rows,
            task_rows=task_rows,
        )
        for cid in measured_ids
    ]
    _write_stage_artifacts(out_dir, rows, task_rows, task_by)

    official_rows = [r for r in rows if _is_one(r.get("official_eligible")) and _is_one(r.get("strict_pass"))]
    official_rows.sort(
        key=lambda r: (
            int(_is_one(r.get("teacher_free_macro_pass"))),
            int(_is_one(r.get("grad_pass"))),
            int(_is_one(r.get("fullgrid_s2_pass"))),
            _float(r.get("macro_gap"), -99.0),
        ),
        reverse=True,
    )
    best = official_rows[0] if official_rows else {}
    assisted = next((r for r in rows if str(r.get("candidate_id")) == "M12"), {})
    teacher_free_macro = int(_is_one(best.get("teacher_free_macro_pass")))
    s2 = int(_is_one(best.get("fullgrid_s2_pass")))
    s1 = int(_is_one(best.get("fullgrid_s1_pass")))
    time_auc = int(_is_one(best.get("time_auc_pass")))
    grad_pass = int(_is_one(best.get("grad_pass")))
    strict_pass = int(_is_one(best.get("strict_pass")))
    profiler_pass = 0
    if not code_contract_pass:
        route = "R8-CodeContractFail"
        primary = "code_contract_incomplete"
    elif teacher_free_macro and s1 and time_auc and profiler_pass:
        route = "R1-TeacherFreeFullSystemAdvantage"
        primary = "none"
    elif teacher_free_macro and s2:
        route = "R2-TeacherFreeS2QualityAdvantage"
        primary = "s1_or_time_auc_or_profiler_open"
    elif not best:
        route = "R7-NoReproduction"
        primary = "no_official_candidate_measured"
    elif not teacher_free_macro and _is_one(assisted.get("macro_significant_pass")):
        route = "R4-TeacherAssistedOnly"
        primary = "teacher_free_macro_gap_not_closed"
    else:
        route = "R5-TeacherFreeNearPass"
        primary = "teacher_free_macro_gap_not_closed"
    success_min = int(code_contract_pass and strict_pass and grad_pass and teacher_free_macro and s2)
    success_formal = int(success_min and s1 and time_auc and profiler_pass)
    route_json = {
        "route": route,
        "best_official_candidate_id": best.get("candidate_id", METRIC_UNAVAILABLE),
        "external_teacher_used": best.get("external_teacher_used", METRIC_UNAVAILABLE),
        "self_teacher_used": best.get("self_teacher_used", METRIC_UNAVAILABLE),
        "strict_pass": strict_pass,
        "grad_pass": grad_pass,
        "teacher_free_macro_pass": teacher_free_macro,
        "fullgrid_s2_pass": s2,
        "fullgrid_s1_pass": s1,
        "time_auc_pass": time_auc,
        "profiler_pass": profiler_pass,
        "code_contract_pass": code_contract_pass,
        "success_v79_minimum": success_min,
        "success_v79_formal": success_formal,
        "macro_gap": best.get("macro_gap", METRIC_UNAVAILABLE),
        "ci95_low": best.get("ci95_low", METRIC_UNAVAILABLE),
        "holm_p": best.get("holm_p", METRIC_UNAVAILABLE),
        "test_gap": best.get("test_gap", METRIC_UNAVAILABLE),
        "memory_ratio_max": best.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "step_ratio_max": best.get("step_ratio_max", METRIC_UNAVAILABLE),
        "diagnostic_m12_c3_macro_gap": assisted.get("macro_gap", METRIC_UNAVAILABLE),
        "diagnostic_m12_minus_best_gap": _macro_gap_between(task_rows, "M12", str(best.get("candidate_id"))) if assisted and best else METRIC_UNAVAILABLE,
        "primary_blocker": primary,
        "next_required_implementation": "self_bootstrap_or_representation_repair" if not teacher_free_macro else "s1_time_auc_profiler",
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "route_decision.json", route_json)
    save_json(out_dir / "aggregate_decision.json", route_json)
    save_json(out_dir / "v79_route_decision.json", route_json)

    failures = []
    if not code_contract_pass:
        failures.append("F1_code_contract_fail")
    if any(_is_one(r.get("teacher_leak_detected")) for r in leak_rows):
        failures.append("F3_teacher_leak_detected")
    if not teacher_free_macro:
        failures.append("F4_teacher_free_macro_fail")
    if best and not s2:
        failures.append("F9_s2_fail")
    if best and not s1:
        failures.append("F10_s1_fail")
    if best and not time_auc:
        failures.append("F11_time_auc_fail")
    if best and not grad_pass:
        failures.append("F12_grad_fail")
    if profiler_pass == 0:
        failures.append("F13_profiler_incomplete")
    write_csv(out_dir / "failure_table.csv", [{
        "failure_type": f,
        "count": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    } for f in failures])

    artifact_paths = [
        out_dir / "candidate_registry.csv",
        out_dir / "contract_validator.csv",
        out_dir / "teacher_leak_audit.csv",
        out_dir / "p0_code_contract.csv",
        out_dir / "p1_candidate_metadata_audit.csv",
        out_dir / "p2_teacher_free_baseline.csv",
        out_dir / "p3_validation_split_audit.csv",
        out_dir / "p4_teacher_gap_attribution.csv",
        out_dir / "p5_self_bootstrap.csv",
        out_dir / "p6_teacher_free_repair.csv",
        out_dir / "p7_representation_repair.csv",
        out_dir / "p8_teacher_free_10seed.csv",
        out_dir / "p9_time_accounting.csv",
        out_dir / "p10_phase_mapped_profiler.csv",
        out_dir / "p11_s1_memory_attribution.csv",
        out_dir / "p12_fused_streaming_package.csv",
        out_dir / "p13_fullgrid_profiler.csv",
        out_dir / "p14_scaling_robustness.csv",
        out_dir / "p15_geometry_pareto.csv",
        out_dir / "p16_route_decision.csv",
        out_dir / "failure_table.csv",
        out_dir / "route_decision.json",
        out_dir / "aggregate_decision.json",
        out_dir / "v79_route_decision.json",
        out_dir / "p9_task_summary.csv",
        out_dir / "p9_task_gate.csv",
        out_dir / "p9_task_trace.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p10_efficiency_profiler.csv",
        out_dir / "p10_efficiency_summary.csv",
        out_dir / "p7_time_auc_v76.csv",
        out_dir / "run_manifest.json",
        out_dir / "v76_manifest.json",
        out_dir / "v76_provenance_audit.csv",
    ]
    audit = v72._audit_fake_proxy([p for p in artifact_paths if p.exists()])
    write_csv(out_dir / "provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "v79_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "gate_config.json", {
        "teacher_free_macro_gap_min": 0.0200,
        "ci95_low_min": 0.0,
        "holm_p_max": 0.05,
        "test_gap_min": 0.015,
        "s2_memory_max": 1.05,
        "s2_step_max": 1.50,
        "s1_memory_lt": 1.00,
        "s1_step_max": 1.35,
    })
    save_json(out_dir / "v79_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v76_real.py",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v79_route": route_json,
        "v79_audit": audit,
        "registry_rows": len(registry_rows),
    })
    return route_json


def run(args: argparse.Namespace) -> None:
    v76.PLAN_PATH = PLAN_PATH
    v76.SCRIPT_PATH = SCRIPT_PATH
    v76.run(args)
    out_dir = Path(args.out_dir)
    _write_v79_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v76_manifest.json": "v79_reused_v76_manifest.json",
        "v76_route_decision.json": "v79_reused_v76_route_decision.json",
        "v76_provenance_audit.csv": "v79_reused_v76_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    return v76.parse_args()


if __name__ == "__main__":
    run(parse_args())
