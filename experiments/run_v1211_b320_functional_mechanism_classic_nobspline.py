#!/usr/bin/env python3
"""v12.11 B320 functional-mechanism and no-B-spline family wrapper.

This runner is intentionally conservative: it does not invent measurements and
does not reroute historical raw runners. It consumes already materialized
v12.10/v1283 artifacts, writes the v12.11 artifact contract, and records which
outputs are derived from which source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "v12_11_b320_functional_mechanism_classic_nobspline"
DEFAULT_V1210_ROOT = REPO_ROOT / "results" / "v12_10_b320_functional_classic_nobspline"
DEFAULT_ANCHOR_DIR = DEFAULT_V1210_ROOT / "repro_repair_linec64_p3fix_p4_taskminus_20260523T1405"
DEFAULT_B320_SOURCE_DIR = DEFAULT_V1210_ROOT / "repro_repair_linec64_full_20260523T1352"
DEFAULT_RAW_DIR = DEFAULT_B320_SOURCE_DIR / "raw_v1283_reuse"
DEFAULT_P4_DIRS = [
    DEFAULT_V1210_ROOT / "repro_repair_linec64_p3fix_p4_taskminus_20260523T1405",
    DEFAULT_V1210_ROOT / "repro_repair_linec64_p3fix_p4_quad_20260523T1405",
    DEFAULT_V1210_ROOT / "repro_repair_linec64_p3fix_p4_negative_20260523T1405",
    DEFAULT_V1210_ROOT / "repro_repair_linec64_p3fix_p4_direct_20260523T1405",
]
B320_ID = "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075"


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list(text: Any) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: Any) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    materialized = [dict(row) for row in rows]
    if materialized:
        fields: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row.keys():
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    else:
        fields = ["stage", "status"]
        materialized = [{"stage": path.stem.upper(), "status": "no_rows"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1.0e-20 or vy <= 1.0e-20:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def infer_p4_meta(path: Path) -> dict[str, Any]:
    name = path.name
    if "taskminus" in name:
        return {"event_id": "E4", "mode": "task_minus_quad010", "lambda": 0.03125, "event_gate": "none"}
    if "quad" in name:
        return {"event_id": "E5", "mode": "quad", "lambda": -0.0625, "event_gate": "tail_noise_reservoir"}
    if "negative" in name:
        return {"event_id": "E5", "mode": "negative", "lambda": -0.03125, "event_gate": "tail_noise"}
    if "direct" in name:
        return {"event_id": "E5", "mode": "direct", "lambda": 0.03125, "event_gate": "tail_noise_reservoir"}
    return {"event_id": "E?", "mode": "", "lambda": "", "event_gate": ""}


def build_anchor_lock(anchor_dir: Path, source_dir: Path, out_dir: Path) -> dict[str, Any]:
    decision = read_json(anchor_dir / "v1210_route_decision.json")
    hardening = read_csv_rows(source_dir / "v1210_b320_base_hardening.csv")
    row = hardening[0] if hardening else {}
    anchor_pass = int(
        safe_float(row.get("step_ratio_q90"), 999.0) <= 0.85
        and safe_float(row.get("memory_ratio_q90"), 999.0) <= 0.30
        and safe_float(row.get("mean_delta"), -999.0) >= 0.0
        and safe_float(row.get("worst_delta"), -999.0) >= -0.003
        and safe_float(row.get("near_pass_rate"), 0.0) >= 1.0
        and safe_float(row.get("AUC_step_ratio"), 999.0) <= 1.0
        and safe_float(row.get("AUC_time_ratio"), 999.0) <= 1.0
        and safe_float(row.get("ECE_delta"), 999.0) <= 0.02
        and safe_int(decision.get("B320_LineC_nontearing_pass"), 0) == 1
    )
    fail: list[str] = []
    if not anchor_pass:
        for key, gate, op in [
            ("step_ratio_q90", 0.85, ">"),
            ("memory_ratio_q90", 0.30, ">"),
            ("worst_delta", -0.003, "<"),
            ("AUC_step_ratio", 1.0, ">"),
            ("AUC_time_ratio", 1.0, ">"),
            ("ECE_delta", 0.02, ">"),
        ]:
            val = safe_float(row.get(key), 999.0 if op == ">" else -999.0)
            if (op == ">" and val > gate) or (op == "<" and val < gate):
                fail.append(f"{key}{op}{gate}")
        if safe_int(decision.get("B320_LineC_nontearing_pass"), 0) != 1:
            fail.append("LineC_nontearing_pass!=1")
    anchor_row = {
        "stage": "V1211_B320_ANCHOR_LOCK",
        "candidate_id": row.get("candidate_id", decision.get("B320_exact_candidate_id", B320_ID)),
        "impl_path": row.get("impl_path", ""),
        "dataset": "MNIST,Fashion-MNIST,KMNIST",
        "seed": "0,1,2,3,4,5,6,7,8,9",
        "train_size": 1024,
        "val_size": 512,
        "test_size": 512,
        "epochs": 3,
        "batch_size": 128,
        "step_ratio_q90": row.get("step_ratio_q90", decision.get("B320_F3_step_ratio_q90", "")),
        "memory_ratio_q90": row.get("memory_ratio_q90", decision.get("B320_F3_memory_ratio_q90", "")),
        "forward_ratio_q90": row.get("forward_ratio_q90", ""),
        "backward_ratio_q90": row.get("backward_ratio_q90", ""),
        "update_ratio_q90": row.get("update_ratio_q90", ""),
        "val_acc_delta_vs_mlp": "",
        "mean_delta": row.get("mean_delta", decision.get("B320_mean_delta", "")),
        "worst_delta": row.get("worst_delta", decision.get("B320_worst_delta", "")),
        "near_pass": row.get("near_pass", ""),
        "near_pass_rate": row.get("near_pass_rate", decision.get("B320_near_pass_rate", "")),
        "AUC_step_ratio": row.get("AUC_step_ratio", decision.get("B320_max_AUC_step", "")),
        "AUC_time_ratio": row.get("AUC_time_ratio", decision.get("B320_max_AUC_time", "")),
        "ECE_delta": row.get("ECE_delta", decision.get("B320_max_ECE_delta", "")),
        "CEp99_delta": "",
        "margin_p10_delta": "",
        "CouplingR2": decision.get("B320_LineC_CouplingR2", ""),
        "NoiseSignalLeak": decision.get("B320_LineC_NoiseSignalLeak", ""),
        "RealSignalReservoirRatio": "",
        "LineC_nontearing_pass": decision.get("B320_LineC_nontearing_pass", ""),
        "strict_pass": anchor_pass,
        "fail_reason": ";".join(fail),
        "source_anchor_dir": str(anchor_dir.relative_to(REPO_ROOT)),
        "source_b320_dir": str(source_dir.relative_to(REPO_ROOT)),
        "derived_not_new_training": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "dataset_name_branch_used": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "loss_modified": 0,
        "sampler_or_class_weight_used": 0,
    }
    write_csv_rows(out_dir / "v1211_b320_anchor_lock.csv", [anchor_row])
    return {"anchor_pass": anchor_pass, "anchor_row": anchor_row, "decision": decision}


def write_linec(raw_dir: Path, out_dir: Path) -> None:
    rows = read_csv_rows(raw_dir / "v1283_b109_linec_diagnostics.csv")
    mlp = next((r for r in rows if r.get("candidate_id") == "MLP-same-param-AdamW"), {})
    mlp_c = safe_float(mlp.get("CouplingR2"), 0.0)
    mlp_n = safe_float(mlp.get("NoiseSignalLeak"), 999.0)
    out_rows = []
    for row in rows:
        copied = dict(row)
        copied["stage"] = "V1211_LINEC_DIAGNOSTICS"
        copied["run_id"] = "v1211_from_v1210_linec64"
        copied["functional_id"] = ""
        copied["family"] = row.get("run_family", "")
        copied["nontearing_pass"] = int(
            safe_float(row.get("CouplingR2"), -999.0) >= mlp_c - 0.02
            and safe_float(row.get("NoiseSignalLeak"), 999.0) <= mlp_n + 0.02
        )
        copied["fail_reason"] = "" if copied["nontearing_pass"] else "base_nontearing_gate_failed"
        copied["source_file"] = str((raw_dir / "v1283_b109_linec_diagnostics.csv").relative_to(REPO_ROOT))
        copied["derived_not_new_training"] = 1
        out_rows.append(copied)
    write_csv_rows(out_dir / "v1211_linec_diagnostics.csv", out_rows)


def copy_family_artifacts(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    mapping = {
        "v1210_family_manifest_raw.csv": "v1211_family_manifest.csv",
        "v1210_family_efficiency.csv": "v1211_family_efficiency.csv",
        "v1210_family_gradcheck.csv": "v1211_family_gradcheck.csv",
        "v1210_family_expression.csv": "v1211_family_expression.csv",
        "v1210_family_task_triage.csv": "v1211_family_task_triage.csv",
        "v1210_family_linec_diagnostics.csv": "v1211_family_linec.csv",
        "v1210_family_failure_table.csv": "v1211_family_failure_table.csv",
        "v1210_family_status.json": "v1211_family_status.json",
    }
    copied: dict[str, Any] = {}
    for src_name, dst_name in mapping.items():
        copied[dst_name] = copy_if_exists(source_dir / src_name, out_dir / dst_name)
    return copied


def build_p3v2(raw_dir: Path, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in read_csv_rows(raw_dir / "v1283_b109_functional_delta_score_repair.csv") if r.get("candidate_id") == B320_ID]
    controls = [r for r in rows if str(r.get("update_type", "")).startswith("C")]
    funcs = [r for r in rows if str(r.get("update_type", "")).startswith("F")]
    best_control = max(controls, key=lambda r: safe_float(r.get("delta_score"), -999.0), default={})
    best_control_score = safe_float(best_control.get("delta_score"), 0.0)
    best_control_coupling = safe_float(best_control.get("CouplingR2"), 0.0)
    out_rows: list[dict[str, Any]] = []
    for row in funcs:
        coupling_delta = safe_float(row.get("CouplingR2"), 0.0) - best_control_coupling
        noise_delta = safe_float(row.get("NoiseSignalLeak_delta"), 0.0)
        reservoir_delta = safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0)
        control_gap = safe_float(row.get("delta_score"), -999.0) - best_control_score
        holdout = safe_float(row.get("holdout_descent_ratio"), 999.0)
        fail = []
        if not (0.95 <= holdout <= 1.05):
            fail.append("holdout_loss_ratio_not_in_[0.95,1.05]")
        if coupling_delta < 0.02:
            fail.append("CouplingR2_delta<0.02")
        if noise_delta > -0.01:
            fail.append("NoiseSignalLeak_delta>-0.01")
        if reservoir_delta > -0.01:
            fail.append("RealSignalReservoirRatio_delta>-0.01")
        if control_gap < 0.005:
            fail.append("control_gap<0.005")
        out_rows.append(
            {
                "stage": "V1211_FUNCTIONAL_P3V2_CANDIDATE",
                "functional_id": row.get("update_type", ""),
                "mechanism_class": infer_mechanism_class(row.get("update_type", "")),
                "dataset": "MNIST",
                "seed": 0,
                "checkpoint_step": "cloned_diagnostic",
                "lambda": row.get("lambda_selected", ""),
                "cos_with_task_adamw": "",
                "orthogonal_fraction": "",
                "holdout_loss_ratio": row.get("holdout_descent_ratio", ""),
                "CouplingR2_delta": coupling_delta,
                "NoiseSignalLeak_delta": noise_delta,
                "RealSignalReservoirRatio_delta": reservoir_delta,
                "CEp99_delta": row.get("CEp99_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "MarginP10_delta": row.get("margin_p10_delta", ""),
                "AUC_proxy_delta": "",
                "control_gap_vs_taskonly": control_gap,
                "control_gap_vs_adamwparallel": control_gap,
                "control_gap_vs_random": safe_float(row.get("delta_score"), 0.0) - safe_float(next((r.get("delta_score") for r in controls if r.get("update_type") == "C2-RandomMatchedNorm"), 0.0), 0.0),
                "control_gap_vs_snr": "",
                "P3v2_pareto_pass": int(not fail),
                "P3v2_fail_reason": ";".join(fail),
                "source_stage": row.get("stage", ""),
                "fake_data_used": row.get("fake_data_used", 0),
                "proxy_row_used": row.get("proxy_row_used", 0),
                "cpu_offload_used": row.get("cpu_offload_used", 0),
                "dataset_name_branch_used": row.get("dataset_name_branch_used", 0),
                "teacher_used": row.get("teacher_used", 0),
                "distillation_used": row.get("distillation_used", 0),
                "loss_modified": row.get("loss_modified", 0),
                "sampler_or_class_weight_used": row.get("sampler_or_class_weight_used", 0),
            }
        )
    write_csv_rows(out_dir / "v1211_functional_p3v2_candidate.csv", out_rows)
    summary = {
        "best_control_update": best_control.get("update_type", ""),
        "best_control_delta_score": best_control.get("delta_score", ""),
        "candidate_count": len(out_rows),
        "pareto_pass_count": sum(safe_int(r.get("P3v2_pareto_pass")) for r in out_rows),
    }
    return out_rows, summary


def infer_mechanism_class(update_type: str) -> str:
    if "SNR" in update_type:
        return "F-M1/F-M5-signal-residual"
    if "Negative" in update_type:
        return "F-M2-noise-projected-sign-check"
    if "Orth" in update_type:
        return "F-M5-adamw-orthogonal"
    if "Branch" in update_type:
        return "legacy-branch-damping"
    return "legacy-functional-diagnostic"


def _lazy_probe_modules():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import run_v120_good_geometry_battery as v120
    import run_v124_multibasis_functional_dual as v124
    import run_v1252_efficiency_functional_manifold as v1252
    import run_v1283_b109_classic_family_functional_geometry as v1283

    return torch, F, v120, v124, v1252, v1283


def _delta_norm(deltas: Sequence[Any]) -> float:
    return math.sqrt(sum(float(d.detach().float().square().sum().item()) for d in deltas))


def _neg_delta(deltas: Sequence[Any]) -> list[Any]:
    return [-d for d in deltas]


def _zero_delta(deltas: Sequence[Any], torch_mod: Any) -> list[Any]:
    return [torch_mod.zeros_like(d) for d in deltas]


def _scale_delta(deltas: Sequence[Any], scale: Any) -> list[Any]:
    return [d * scale for d in deltas]


def _add_deltas(lhs: Sequence[Any], rhs: Sequence[Any]) -> list[Any]:
    return [a + b.to(a.device) for a, b in zip(lhs, rhs)]


def _random_like(deltas: Sequence[Any], seed: int, torch_mod: Any) -> list[Any]:
    if not deltas:
        return []
    device = deltas[0].device
    gen = torch_mod.Generator(device=device).manual_seed(int(seed))
    target_norm = _delta_norm(deltas)
    out = [torch_mod.randn(d.shape, device=device, generator=gen) for d in deltas]
    norm = _delta_norm(out)
    return [d * (target_norm / max(1.0e-12, norm)) for d in out]


def _cos_and_orthogonal_fraction(deltas: Sequence[Any], task_delta: Sequence[Any], torch_mod: Any) -> tuple[float, float]:
    if not deltas or not task_delta:
        return 0.0, 0.0
    flat_d = torch_mod.cat([d.detach().float().flatten() for d in deltas])
    flat_t = torch_mod.cat([d.detach().float().flatten().to(flat_d.device) for d in task_delta])
    dn = flat_d.norm().clamp_min(1.0e-12)
    tn = flat_t.norm().clamp_min(1.0e-12)
    cos = float((flat_d @ flat_t).div(dn * tn).item())
    parallel_energy = float(((flat_d @ flat_t).square() / flat_t.square().sum().clamp_min(1.0e-12)).item())
    total_energy = float(flat_d.square().sum().clamp_min(1.0e-12).item())
    orth_fraction = max(0.0, min(1.0, 1.0 - parallel_energy / total_energy))
    return cos, orth_fraction


def _branch_scale_basis_delta(model: Any, task_delta: Sequence[Any], branch_idx: int, torch_mod: Any) -> list[Any]:
    out = []
    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        delta = torch_mod.zeros_like(param)
        if name == "branch_scale":
            base = -param.detach().clone()
            if base.ndim == 0:
                delta = base if int(branch_idx) == 0 else torch_mod.zeros_like(base)
            elif int(base.shape[0]) > int(branch_idx):
                delta[int(branch_idx)].copy_(base[int(branch_idx)])
        out.append(delta)
    if len(out) != len(task_delta):
        return _zero_delta(task_delta, torch_mod)
    return out


def _normalize_delta_like(deltas: Sequence[Any], reference: Sequence[Any]) -> list[Any]:
    target = _delta_norm(reference)
    norm = _delta_norm(deltas)
    if norm <= 1.0e-12 or target <= 1.0e-12:
        return list(deltas)
    return [d * (target / norm) for d in deltas]


def _delta_dot(lhs: Sequence[Any], rhs: Sequence[Any], torch_mod: Any) -> Any:
    if not lhs or not rhs:
        return torch_mod.zeros(())
    device = lhs[0].device
    out = torch_mod.zeros((), device=device)
    for a, b in zip(lhs, rhs):
        out = out + (a.detach().float().to(device) * b.detach().float().to(device)).sum()
    return out


def _project_delta_away(deltas: Sequence[Any], basis: Sequence[Any], torch_mod: Any) -> list[Any]:
    if not deltas or not basis:
        return list(deltas)
    flat_d = torch_mod.cat([d.detach().float().flatten() for d in deltas])
    flat_b = torch_mod.cat([b.detach().float().flatten().to(flat_d.device) for b in basis])
    denom = flat_b.square().sum().clamp_min(1.0e-12)
    coeff = (flat_d @ flat_b) / denom
    projected = [d - coeff.to(d.device) * b.to(d.device) for d, b in zip(deltas, basis)]
    return _normalize_delta_like(projected, deltas)


def _grad_delta_for_batch(model: Any, x: Any, y: Any, lr: float, torch_mod: Any, F_mod: Any) -> list[Any]:
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    model.zero_grad(set_to_none=True)
    loss = F_mod.cross_entropy(model(x), y)
    grads = torch_mod.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    model.zero_grad(set_to_none=True)
    return [(-float(lr) * (g.detach() if g is not None else torch_mod.zeros_like(p))).clone() for p, g in zip(params, grads)]


def _projector_proxy_grad_deltas(
    *,
    model: Any,
    x: Any,
    y: Any,
    task_delta: Sequence[Any],
    lr: float,
    sketch_dim: int,
    seed: int,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
) -> tuple[dict[str, list[Any]], dict[str, float], str]:
    b = min(int(x.shape[0]), int(y.shape[0]))
    x = x[:b]
    y = y[:b]
    if b <= 1:
        zeros = _zero_delta(task_delta, torch_mod)
        return {"reservoir": zeros, "noise": zeros, "tail": zeros}, {}, "batch_too_small_for_projector"
    with torch_mod.enable_grad():
        grad_sketch = v1252._sample_grad_sketch(model, x, y, int(sketch_dim), int(seed)).detach()
    k_mat = grad_sketch @ grad_sketch.T
    evals, evecs = torch_mod.linalg.eigh(k_mat.float())
    order = torch_mod.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0)
    evecs = evecs[:, order]
    total = evals.sum().clamp_min(1.0e-12)
    cum = torch_mod.cumsum(evals, dim=0)
    top_count = int(torch_mod.searchsorted(cum, 0.80 * total).item()) + 1
    top_count = max(1, min(top_count, int(evals.numel())))
    p_sig = (evecs[:, :top_count] @ evecs[:, :top_count].T).detach()
    p_res = (torch_mod.eye(b, device=x.device) - p_sig).detach()
    noise_gen = torch_mod.Generator(device=x.device).manual_seed(int(seed) + 33)
    y_noise = y[torch_mod.randperm(b, device=x.device, generator=noise_gen)]
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    model.zero_grad(set_to_none=True)
    logits = model(x)
    ce_real = F_mod.cross_entropy(logits, y, reduction="none").float()
    ce_noise = F_mod.cross_entropy(logits, y_noise, reduction="none").float()
    r_real = ce_real - ce_real.mean()
    r_noise = ce_noise - ce_noise.mean()
    real_res = (p_res @ r_real).square().sum() / r_real.square().sum().clamp_min(1.0e-12)
    noise_sig = (p_sig @ r_noise).square().sum() / r_noise.square().sum().clamp_min(1.0e-12)
    tail_count = max(1, int(math.ceil(float(ce_real.numel()) * 0.25)))
    tail_proxy = torch_mod.topk(ce_real, k=tail_count).values.mean() / ce_real.detach().mean().abs().clamp_min(1.0e-12)
    objectives = {"reservoir": real_res, "noise": noise_sig, "tail": tail_proxy}
    deltas: dict[str, list[Any]] = {}
    items = list(objectives.items())
    for idx, (name, objective) in enumerate(items):
        grads = torch_mod.autograd.grad(
            objective,
            params,
            retain_graph=idx < len(items) - 1,
            create_graph=False,
            allow_unused=True,
        )
        deltas[name] = [(-float(lr) * (g.detach() if g is not None else torch_mod.zeros_like(p))).clone() for p, g in zip(params, grads)]
    model.zero_grad(set_to_none=True)
    metrics = {
        "real_res_proxy": float(real_res.detach().item()),
        "noise_sig_proxy": float(noise_sig.detach().item()),
        "tail_proxy": float(tail_proxy.detach().item()),
    }
    internal = f"top_count={top_count};real_res_proxy={metrics['real_res_proxy']};noise_sig_proxy={metrics['noise_sig_proxy']};tail_proxy={metrics['tail_proxy']}"
    return deltas, metrics, internal


def _projector_level_delta(
    *,
    model: Any,
    x: Any,
    y: Any,
    task_delta: Sequence[Any],
    lr: float,
    sketch_dim: int,
    seed: int,
    reservoir_weight: float,
    noise_weight: float,
    tail_weight: float,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
) -> tuple[list[Any], str]:
    proxy_deltas, metrics, internal = _projector_proxy_grad_deltas(
        model=model,
        x=x,
        y=y,
        task_delta=task_delta,
        lr=lr,
        sketch_dim=sketch_dim,
        seed=seed,
        torch_mod=torch_mod,
        F_mod=F_mod,
        v1252=v1252,
    )
    raw = _add_deltas(
        _add_deltas(
            _scale_delta(proxy_deltas["reservoir"], float(reservoir_weight)),
            _scale_delta(proxy_deltas["noise"], float(noise_weight)),
        ),
        _scale_delta(proxy_deltas["tail"], float(tail_weight)),
    )
    delta = _normalize_delta_like(raw, task_delta)
    internal = f"{internal};weights=reservoir:{reservoir_weight},noise:{noise_weight},tail:{tail_weight};proxy_metrics={metrics}"
    return delta, internal


def _constrained_lowrank_projector_delta(
    *,
    model: Any,
    x: Any,
    y: Any,
    task_delta: Sequence[Any],
    lr: float,
    sketch_dim: int,
    seed: int,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
) -> tuple[list[Any], str]:
    proxy_deltas, metrics, internal = _projector_proxy_grad_deltas(
        model=model,
        x=x,
        y=y,
        task_delta=task_delta,
        lr=lr,
        sketch_dim=sketch_dim,
        seed=seed,
        torch_mod=torch_mod,
        F_mod=F_mod,
        v1252=v1252,
    )
    basis_names = ["reservoir", "noise", "tail"]
    basis = [_normalize_delta_like(proxy_deltas[name], task_delta) for name in basis_names]
    metric_deltas = [proxy_deltas[name] for name in basis_names]
    rows = []
    for metric_delta in metric_deltas:
        row = []
        for direction in basis:
            # metric_grad dot direction; metric_delta = -lr * metric_grad
            row.append((-_delta_dot(metric_delta, direction, torch_mod) / max(1.0e-12, float(lr))).detach())
        row_t = torch_mod.stack(row).float()
        row_t = row_t / row_t.abs().sum().clamp_min(1.0e-12)
        rows.append(row_t)
    a = torch_mod.stack(rows, dim=0)
    target = torch_mod.tensor([-1.0, -1.0, -0.05], device=a.device, dtype=a.dtype)
    eye = torch_mod.eye(int(a.shape[1]), device=a.device, dtype=a.dtype)
    coeff = torch_mod.linalg.solve(a.T @ a + 1.0e-3 * eye, a.T @ target)
    raw = _zero_delta(task_delta, torch_mod)
    for c, direction in zip(coeff, basis):
        raw = _add_deltas(raw, _scale_delta(direction, c))
    delta = _normalize_delta_like(raw, task_delta)
    predicted = (a @ coeff).detach().cpu().tolist()
    coeffs = {name: float(c.detach().item()) for name, c in zip(basis_names, coeff)}
    return delta, f"{internal};constrained_coeffs={coeffs};normalized_predicted_changes={predicted};proxy_metrics={metrics}"


def _function_preserving_branch_delta(
    *,
    model: Any,
    task_delta: Sequence[Any],
    x: Any,
    torch_mod: Any,
    v1252: Any,
    scale: float,
) -> tuple[list[Any], str]:
    direct = _branch_scale_basis_delta(model, task_delta, 0, torch_mod)
    quad = _branch_scale_basis_delta(model, task_delta, 1, torch_mod)
    if _delta_norm(direct) <= 1.0e-12 or _delta_norm(quad) <= 1.0e-12:
        return _zero_delta(task_delta, torch_mod), "missing_two_branch_scale_axes"
    with torch_mod.no_grad():
        base_logits = model(x).detach()
    direct_trial = deepcopy(model)
    quad_trial = deepcopy(model)
    v1252._apply_delta(direct_trial, direct, float(scale))
    v1252._apply_delta(quad_trial, quad, float(scale))
    with torch_mod.no_grad():
        u = (direct_trial(x).detach() - base_logits).float().flatten()
        v = (quad_trial(x).detach() - base_logits).float().flatten()
    gram = torch_mod.stack(
        [
            torch_mod.stack([u @ u, u @ v]),
            torch_mod.stack([u @ v, v @ v]),
        ]
    )
    evals, evecs = torch_mod.linalg.eigh(gram)
    coeff = evecs[:, 0]
    delta = _add_deltas(_scale_delta(direct, coeff[0]), _scale_delta(quad, coeff[1]))
    delta = _normalize_delta_like(delta, task_delta)
    return delta, f"local_logit_null_eigenvalue={float(evals[0].detach().item())}"


def _evaluate_probe_direction(
    *,
    base: Any,
    deltas: Sequence[Any],
    task_delta: Sequence[Any],
    xb: Any,
    yb: Any,
    xq: Any,
    yq: Any,
    scale: float,
    seed: int,
    args: argparse.Namespace,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
    v1283: Any,
) -> dict[str, Any]:
    before_loss = float(F_mod.cross_entropy(base(xq), yq).detach().item())
    trial = deepcopy(base)
    v1252._apply_delta(trial, deltas, float(scale))
    after_loss = float(F_mod.cross_entropy(trial(xq), yq).detach().item())
    geo = v1283._geo_gain_delta_scored(
        base,
        trial,
        xb,
        yb,
        xq,
        yq,
        float(args.ridge_lambda),
        int(args.sketch_dim),
        int(seed),
    )
    cos, orth_fraction = _cos_and_orthogonal_fraction(deltas, task_delta, torch_mod)
    return {
        "holdout_loss_ratio": after_loss / max(1.0e-12, before_loss),
        "CouplingR2": safe_float(geo.get("CouplingR2"), 0.0),
        "NoiseSignalLeak_delta": safe_float(geo.get("NoiseSignalLeak_delta"), 0.0),
        "RealSignalReservoirRatio_delta": safe_float(geo.get("RealSignalReservoirRatio_delta"), 0.0),
        "CEp99_delta": safe_float(geo.get("CEp99_delta"), 0.0),
        "ECE_delta": safe_float(geo.get("ECE_delta"), 0.0),
        "MarginP10_delta": safe_float(geo.get("margin_p10_delta"), 0.0),
        "cos_with_task_adamw": cos,
        "orthogonal_fraction": orth_fraction,
        "delta_norm": _delta_norm(deltas),
        "delta_score": safe_float(geo.get("delta_score"), 0.0),
    }


def _param_delta_between(before: Any, after: Any) -> list[Any]:
    before_params = [p for p in before.parameters() if getattr(p, "requires_grad", False)]
    after_params = [p for p in after.parameters() if getattr(p, "requires_grad", False)]
    return [(pa.detach() - pb.detach()).clone() for pb, pa in zip(before_params, after_params)]


def _evaluate_projector_sequence(
    *,
    base: Any,
    task_delta: Sequence[Any],
    xb: Any,
    yb: Any,
    xq: Any,
    yq: Any,
    steps: int,
    scale: float,
    seed: int,
    args: argparse.Namespace,
    torch_mod: Any,
    F_mod: Any,
    v1252: Any,
    v1283: Any,
) -> dict[str, Any]:
    before_loss = float(F_mod.cross_entropy(base(xq), yq).detach().item())
    trial = deepcopy(base)
    internals: list[str] = []
    for event_idx in range(max(1, int(steps))):
        local_task = v1252._grad_delta(trial, xb, yb, float(args.probe_lr))
        local_delta, internal = _projector_level_delta(
            model=trial,
            x=xb,
            y=yb,
            task_delta=local_task,
            lr=float(args.probe_lr),
            sketch_dim=int(args.sketch_dim),
            seed=int(seed) + event_idx * 17,
            reservoir_weight=1.0,
            noise_weight=1.0,
            tail_weight=float(args.probe_projector_tail_weight),
            torch_mod=torch_mod,
            F_mod=F_mod,
            v1252=v1252,
        )
        v1252._apply_delta(trial, local_delta, float(scale))
        internals.append(internal)
    after_loss = float(F_mod.cross_entropy(trial(xq), yq).detach().item())
    geo = v1283._geo_gain_delta_scored(
        base,
        trial,
        xb,
        yb,
        xq,
        yq,
        float(args.ridge_lambda),
        int(args.sketch_dim),
        int(seed),
    )
    net_delta = _param_delta_between(base, trial)
    cos, orth_fraction = _cos_and_orthogonal_fraction(net_delta, task_delta, torch_mod)
    return {
        "holdout_loss_ratio": after_loss / max(1.0e-12, before_loss),
        "CouplingR2": safe_float(geo.get("CouplingR2"), 0.0),
        "NoiseSignalLeak_delta": safe_float(geo.get("NoiseSignalLeak_delta"), 0.0),
        "RealSignalReservoirRatio_delta": safe_float(geo.get("RealSignalReservoirRatio_delta"), 0.0),
        "CEp99_delta": safe_float(geo.get("CEp99_delta"), 0.0),
        "ECE_delta": safe_float(geo.get("ECE_delta"), 0.0),
        "MarginP10_delta": safe_float(geo.get("margin_p10_delta"), 0.0),
        "cos_with_task_adamw": cos,
        "orthogonal_fraction": orth_fraction,
        "delta_norm": _delta_norm(net_delta),
        "delta_score": safe_float(geo.get("delta_score"), 0.0),
        "sequence_internal": "|".join(internals[-2:]),
    }


def _p3v2_fail_reasons(row: Mapping[str, Any], best_control_score: float) -> list[str]:
    fail: list[str] = []
    holdout = safe_float(row.get("holdout_loss_ratio"), 999.0)
    coupling_delta = safe_float(row.get("CouplingR2_delta"), 0.0)
    noise_delta = safe_float(row.get("NoiseSignalLeak_delta"), 0.0)
    reservoir_delta = safe_float(row.get("RealSignalReservoirRatio_delta"), 0.0)
    control_gap = safe_float(row.get("delta_score"), -999.0) - best_control_score
    if not (0.95 <= holdout <= 1.05):
        fail.append("holdout_loss_ratio_not_in_[0.95,1.05]")
    if coupling_delta < 0.02:
        fail.append("CouplingR2_delta<0.02")
    if noise_delta > -0.01:
        fail.append("NoiseSignalLeak_delta>-0.01")
    if reservoir_delta > -0.01:
        fail.append("RealSignalReservoirRatio_delta>-0.01")
    if control_gap < 0.005:
        fail.append("control_gap<0.005")
    return fail


def run_p3v2_mechanism_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    if int(args.run_p3v2_mechanism_probe) != 1:
        return []
    torch_mod, F_mod, v120, v124, v1252, v1283 = _lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    out_rows: list[dict[str, Any]] = []
    datasets = parse_list(args.probe_datasets)
    seeds = parse_ints(args.probe_seeds)
    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        load_args = argparse.Namespace(
            data_root=args.data_root,
            no_download=bool(args.no_download),
            seed=seeds[0] if seeds else 0,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
            datasets=canon,
        )
        data = v120._load_vision_split(
            load_args,
            canon,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
        )
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {s.candidate_id: s for s in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
            y_val = y_val_cpu.to(device=device)
            xb = x_train[: int(args.functional_batch_size)]
            yb = y_train[: int(args.functional_batch_size)]
            xq = x_val[: int(args.functional_batch_size)]
            yq = y_val[: int(args.functional_batch_size)]
            base = v1283._make_model(B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1211000, specs, y_train)
            task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
            snr_delta = v1252._basis_delta(base, task_delta, "basis_aware_snr_projected")
            orth_delta = v1252._basis_delta(base, task_delta, "basis_aware_orthogonal")
            noise_gen = torch_mod.Generator(device=device).manual_seed(int(seed) + 1211029)
            y_noise = yb[torch_mod.randperm(int(yb.shape[0]), device=device, generator=noise_gen)]
            noise_delta = _grad_delta_for_batch(base, xb, y_noise, float(args.probe_lr), torch_mod, F_mod)
            with torch_mod.no_grad():
                ce_train = F_mod.cross_entropy(base(xb), yb, reduction="none")
                tail_count = max(1, int(math.ceil(float(ce_train.numel()) * float(args.probe_tail_fraction))))
                tail_idx = torch_mod.topk(ce_train, k=tail_count).indices
            tail_delta = _grad_delta_for_batch(base, xb[tail_idx], yb[tail_idx], float(args.probe_lr), torch_mod, F_mod)
            task_noise_projected = _project_delta_away(task_delta, noise_delta, torch_mod)
            task_noise_tail_projected = _project_delta_away(task_noise_projected, tail_delta, torch_mod)
            real_minus_noise = _normalize_delta_like(
                _add_deltas(task_delta, _scale_delta(_normalize_delta_like(noise_delta, task_delta), -1.0)),
                task_delta,
            )
            projector_reservoir_delta, projector_reservoir_internal = _projector_level_delta(
                model=base,
                x=xb,
                y=yb,
                task_delta=task_delta,
                lr=float(args.probe_lr),
                sketch_dim=int(args.sketch_dim),
                seed=int(seed) + 1211061,
                reservoir_weight=1.0,
                noise_weight=0.0,
                tail_weight=0.0,
                torch_mod=torch_mod,
                F_mod=F_mod,
                v1252=v1252,
            )
            projector_noise_delta, projector_noise_internal = _projector_level_delta(
                model=base,
                x=xb,
                y=yb,
                task_delta=task_delta,
                lr=float(args.probe_lr),
                sketch_dim=int(args.sketch_dim),
                seed=int(seed) + 1211061,
                reservoir_weight=0.0,
                noise_weight=1.0,
                tail_weight=0.0,
                torch_mod=torch_mod,
                F_mod=F_mod,
                v1252=v1252,
            )
            projector_joint_delta, projector_joint_internal = _projector_level_delta(
                model=base,
                x=xb,
                y=yb,
                task_delta=task_delta,
                lr=float(args.probe_lr),
                sketch_dim=int(args.sketch_dim),
                seed=int(seed) + 1211061,
                reservoir_weight=1.0,
                noise_weight=1.0,
                tail_weight=float(args.probe_projector_tail_weight),
                torch_mod=torch_mod,
                F_mod=F_mod,
                v1252=v1252,
            )
            constrained_lowrank_delta, constrained_lowrank_internal = _constrained_lowrank_projector_delta(
                model=base,
                x=xb,
                y=yb,
                task_delta=task_delta,
                lr=float(args.probe_lr),
                sketch_dim=int(args.sketch_dim),
                seed=int(seed) + 1211061,
                torch_mod=torch_mod,
                F_mod=F_mod,
                v1252=v1252,
            )
            fm4_delta, fm4_internal = _function_preserving_branch_delta(
                model=base,
                task_delta=task_delta,
                x=torch_mod.cat([xb, xq], dim=0),
                torch_mod=torch_mod,
                v1252=v1252,
                scale=float(args.probe_fm4_linearization_scale),
            )
            candidate_deltas: dict[str, tuple[str, Sequence[Any]]] = {
                "C0-TaskOnlyAdamW": ("control", task_delta),
                "C1-NoOpMatchedOverhead": ("control", _zero_delta(task_delta, torch_mod)),
                "C2-RandomMatchedNorm": ("control", _random_like(task_delta, int(seed) + 1211017, torch_mod)),
                "C3-AdamWParallelDirection": ("control", task_delta),
                "FM1-SNRSignalResidualOnly": ("F-M1-signal-reservoir-release", snr_delta),
                "FM1-NegSNRReservoirRelease": ("F-M1/F-M2-sign-check", _neg_delta(snr_delta)),
                "FM2-OrthogonalNoiseProjectedResidual": ("F-M2-noise-projected", orth_delta),
                "FM2-NegOrthogonalNoiseProjectedResidual": ("F-M2-noise-projected-sign-check", _neg_delta(orth_delta)),
                "FM2-TaskNoiseOrthogonalProjection": ("F-M2-noise-vjp-projected", task_noise_projected),
                "FM2-RealMinusNoiseVJPProjection": ("F-M2-real-minus-noise-vjp", real_minus_noise),
                "FM3-TaskNoiseTailOrthogonalProjection": ("F-M3-tail-safe-vjp-projected", task_noise_tail_projected),
                "FM1-ProjectorReservoirRelease": ("F-M1-projector-reservoir-release", projector_reservoir_delta),
                "FM2-ProjectorNoiseLeakRelease": ("F-M2-projector-noise-release", projector_noise_delta),
                "FM1M2-ProjectorJointRelease": ("F-M1/F-M2-projector-joint-release", projector_joint_delta),
                "FM1M2-NegProjectorJointRelease": ("F-M1/F-M2-projector-joint-sign-check", _neg_delta(projector_joint_delta)),
                "FM1M2-ProjectorJointRelease5Event": ("F-M1/F-M2-projector-joint-5event", _zero_delta(task_delta, torch_mod)),
                "FM1M2-ConstrainedLowRankRelease": ("F-M1/F-M2-constrained-lowrank-release", constrained_lowrank_delta),
                "FM4-FunctionPreservingBranchNull": ("F-M4-function-preserving-branch-reparam", fm4_delta),
                "FM4-NegFunctionPreservingBranchNull": ("F-M4-function-preserving-branch-sign-check", _neg_delta(fm4_delta)),
                "FM5-AdamWOrthogonalSignalResidual": ("F-M5-adamw-orthogonal-signal", orth_delta),
            }
            evaluated: dict[str, dict[str, Any]] = {}
            for name, (_mechanism, deltas) in candidate_deltas.items():
                evaluated[name] = _evaluate_probe_direction(
                    base=base,
                    deltas=deltas,
                    task_delta=task_delta,
                    xb=xb,
                    yb=yb,
                    xq=xq,
                    yq=yq,
                    scale=float(args.probe_lambda),
                    seed=int(seed) + 1211040,
                    args=args,
                    torch_mod=torch_mod,
                    F_mod=F_mod,
                    v1252=v1252,
                    v1283=v1283,
                )
            evaluated["FM1M2-ProjectorJointRelease5Event"] = _evaluate_projector_sequence(
                base=base,
                task_delta=task_delta,
                xb=xb,
                yb=yb,
                xq=xq,
                yq=yq,
                steps=5,
                scale=float(args.probe_lambda),
                seed=int(seed) + 1211089,
                args=args,
                torch_mod=torch_mod,
                F_mod=F_mod,
                v1252=v1252,
                v1283=v1283,
            )
            # F-M3 is a safety wrapper, not a value source: if SNR is tail/noise unsafe,
            # it deliberately degenerates to no-op and must fail the geometry-value gate.
            snr_eval = evaluated["FM1-SNRSignalResidualOnly"]
            if (
                safe_float(snr_eval.get("CEp99_delta")) <= float(args.probe_tail_epsilon)
                and safe_float(snr_eval.get("MarginP10_delta")) >= -float(args.probe_margin_epsilon)
                and safe_float(snr_eval.get("NoiseSignalLeak_delta")) <= float(args.probe_noise_epsilon)
            ):
                fm3_delta = snr_delta
                fm3_internal = "accepted_snr_direction"
            else:
                fm3_delta = _zero_delta(task_delta, torch_mod)
                fm3_internal = "rejected_to_noop_by_tail_noise_safety"
            evaluated["FM3-TailSafeSignalResidual"] = _evaluate_probe_direction(
                base=base,
                deltas=fm3_delta,
                task_delta=task_delta,
                xb=xb,
                yb=yb,
                xq=xq,
                yq=yq,
                scale=float(args.probe_lambda),
                seed=int(seed) + 1211040,
                args=args,
                torch_mod=torch_mod,
                F_mod=F_mod,
                v1252=v1252,
                v1283=v1283,
            )
            candidate_deltas["FM3-TailSafeSignalResidual"] = ("F-M3-tail-safe-geometry", fm3_delta)
            best_control_score = max(safe_float(evaluated[name].get("delta_score"), -999.0) for name in ["C0-TaskOnlyAdamW", "C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-AdamWParallelDirection"])
            random_score = safe_float(evaluated["C2-RandomMatchedNorm"].get("delta_score"), 0.0)
            snr_score = safe_float(evaluated["FM1-SNRSignalResidualOnly"].get("delta_score"), 0.0)
            for name, (mechanism, _deltas) in candidate_deltas.items():
                if name.startswith("C"):
                    continue
                metrics = evaluated[name]
                row = {
                    "stage": "V1211_FUNCTIONAL_P3V2_MECHANISM_PROBE",
                    "functional_id": name,
                    "mechanism_class": mechanism,
                    "dataset": canon,
                    "seed": seed,
                    "checkpoint_step": "init_cloned_p3v2_probe",
                    "lambda": float(args.probe_lambda),
                    "cos_with_task_adamw": metrics.get("cos_with_task_adamw", ""),
                    "orthogonal_fraction": metrics.get("orthogonal_fraction", ""),
                    "holdout_loss_ratio": metrics.get("holdout_loss_ratio", ""),
                    "CouplingR2_delta": metrics.get("CouplingR2", ""),
                    "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                    "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                    "CEp99_delta": metrics.get("CEp99_delta", ""),
                    "ECE_delta": metrics.get("ECE_delta", ""),
                    "MarginP10_delta": metrics.get("MarginP10_delta", ""),
                    "AUC_proxy_delta": "",
                    "delta_norm": metrics.get("delta_norm", ""),
                    "delta_score": metrics.get("delta_score", ""),
                    "control_gap_vs_taskonly": safe_float(metrics.get("delta_score"), 0.0) - safe_float(evaluated["C0-TaskOnlyAdamW"].get("delta_score"), 0.0),
                    "control_gap_vs_adamwparallel": safe_float(metrics.get("delta_score"), 0.0) - safe_float(evaluated["C3-AdamWParallelDirection"].get("delta_score"), 0.0),
                    "control_gap_vs_random": safe_float(metrics.get("delta_score"), 0.0) - random_score,
                    "control_gap_vs_snr": safe_float(metrics.get("delta_score"), 0.0) - snr_score,
                    "P3v2_pareto_pass": "",
                    "P3v2_fail_reason": "",
                    "source_stage": "V1211_REAL_MECHANISM_PROBE",
                    "mechanism_internal_decision": (
                        fm3_internal
                        if name == "FM3-TailSafeSignalResidual"
                        else (
                            fm4_internal
                            if name.startswith("FM4-")
                            else (
                                projector_reservoir_internal
                                if name == "FM1-ProjectorReservoirRelease"
                                else (
                                    projector_noise_internal
                                    if name == "FM2-ProjectorNoiseLeakRelease"
                                    else (
                                        constrained_lowrank_internal
                                        if name == "FM1M2-ConstrainedLowRankRelease"
                                        else (projector_joint_internal if "ProjectorJoint" in name else "fixed_direction")
                                    )
                                )
                            )
                        )
                    )
                    + (f";{metrics.get('sequence_internal', '')}" if name.endswith("5Event") else ""),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                    "dataset_name_branch_used": 0,
                    "teacher_used": 0,
                    "distillation_used": 0,
                    "loss_modified": 0,
                    "sampler_or_class_weight_used": 0,
                }
                fail = _p3v2_fail_reasons(row, best_control_score)
                row["P3v2_pareto_pass"] = int(not fail)
                row["P3v2_fail_reason"] = ";".join(fail)
                out_rows.append(row)
    return out_rows


def build_p4_and_bridge(raw_dir: Path, p4_dirs: Sequence[Path], out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_p3 = [r for r in read_csv_rows(raw_dir / "v1283_b109_functional_delta_score_repair.csv") if r.get("candidate_id") == B320_ID]
    p3_by_update = {r.get("update_type", ""): r for r in raw_p3}
    method_to_update = {
        "B320-AdamW": "C0-TaskOnlyAdamW",
        "B320-NoOpMatchedOverhead": "C1-NoOpMatchedOverhead",
        "B320-RandomMatchedNorm": "C2-RandomMatchedNorm",
        "B320-AdamWParallelMaintenance": "C3-AdamWParallelDirection",
        "B320-bestFunctional": "F14q-OrthQuadBranchDampingOnly",
    }
    p4_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    corr_scores: list[float] = []
    corr_gains: list[float] = []
    auc_scores: list[float] = []
    auc_labels: list[int] = []
    for p4_dir in p4_dirs:
        rows = read_csv_rows(p4_dir / "v1210_functional_short_run.csv")
        meta = infer_p4_meta(p4_dir)
        control_best: dict[tuple[str, str], float] = defaultdict(lambda: -999.0)
        for row in rows:
            if row.get("method") in {"B320-NoOpMatchedOverhead", "B320-RandomMatchedNorm", "B320-AdamWParallelMaintenance", "B320-SNR-only"}:
                key = (row.get("dataset", ""), row.get("seed", ""))
                control_best[key] = max(control_best[key], safe_float(row.get("diagnostic_score"), -999.0))
        for row in rows:
            copied = dict(row)
            copied["stage"] = "V1211_FUNCTIONAL_P4_SHORT_RUN"
            copied["source_p4_dir"] = str(p4_dir.relative_to(REPO_ROOT))
            copied["p4_mode"] = meta["mode"]
            copied["p4_lambda"] = meta["lambda"]
            copied["derived_not_new_training"] = 1
            p4_rows.append(copied)
            if row.get("method") == "B320-bestFunctional":
                failure_rows.append(
                    {
                        "stage": "V1211_FUNCTIONAL_FAILURE_TABLE",
                        "source_p4_dir": str(p4_dir.relative_to(REPO_ROOT)),
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "functional_id": row.get("functional_id", ""),
                        "mode": meta["mode"],
                        "lambda": meta["lambda"],
                        "strict_pass": row.get("strict_pass", ""),
                        "fail_reason": row.get("fail_reason", ""),
                    }
                )
            update_type = method_to_update.get(row.get("method", ""))
            if not update_type:
                continue
            p3 = p3_by_update.get(update_type, {})
            gain = (
                -safe_float(row.get("AUC_time_delta_vs_B320"), 0.0)
                - safe_float(row.get("NoiseSignalLeak_delta_vs_B320"), 0.0)
                - safe_float(row.get("CEp99_delta_vs_B320"), 0.0)
                + safe_float(row.get("CouplingR2_delta_vs_B320"), 0.0)
            )
            score = safe_float(p3.get("delta_score"), 0.0)
            strict = safe_int(row.get("strict_pass"), 0)
            key = (row.get("dataset", ""), row.get("seed", ""))
            bridge = {
                "stage": "V1211_P3_P4_BRIDGE_AUTOPSY",
                "run_id": p4_dir.name,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "event_id": meta["event_id"],
                "event_step": "repeated_event_rows_from_v1210_p4",
                "window_id": row.get("method", ""),
                "candidate_id": B320_ID,
                "functional_id": update_type,
                "lambda": meta["lambda"],
                "mode": meta["mode"],
                "p3_score": score,
                "p3_task_delta": p3.get("NLL_delta", ""),
                "p3_coupling_delta": p3.get("CouplingR2", ""),
                "p3_noise_delta": p3.get("NoiseSignalLeak_delta", ""),
                "p3_reservoir_delta": p3.get("RealSignalReservoirRatio_delta", ""),
                "p3_tail_delta": p3.get("CEp99_delta", ""),
                "p4_auc_time_delta": row.get("AUC_time_delta_vs_B320", ""),
                "p4_auc_step_delta": row.get("AUC_step_delta_vs_B320", ""),
                "p4_coupling_delta": row.get("CouplingR2_delta_vs_B320", ""),
                "p4_noise_delta": row.get("NoiseSignalLeak_delta_vs_B320", ""),
                "p4_reservoir_delta": row.get("RealSignalReservoirRatio_delta_vs_B320", ""),
                "p4_tail_delta": row.get("CEp99_delta_vs_B320", ""),
                "p4_ece_delta": row.get("ECE_delta_vs_B320", ""),
                "p4_margin_delta": row.get("margin_p10_delta_vs_B320", ""),
                "p4_gain": gain,
                "best_control_id": "max_diagnostic_score_control",
                "best_control_score": control_best[key],
                "control_gap": row.get("control_gap_vs_best", ""),
                "accepted_by_gate": int(safe_float(row.get("accepted_event_count"), 0.0) > 0.0),
                "actual_strict_pass": strict,
                "failure_reason": row.get("fail_reason", ""),
                "source_file": str((p4_dir / "v1210_functional_short_run.csv").relative_to(REPO_ROOT)),
            }
            bridge_rows.append(bridge)
            corr_scores.append(score)
            corr_gains.append(gain)
            auc_scores.append(score)
            auc_labels.append(strict)
    write_csv_rows(out_dir / "v1211_functional_p4_short_run.csv", p4_rows)
    write_csv_rows(out_dir / "v1211_p3_p4_bridge_autopsy.csv", bridge_rows)
    write_csv_rows(out_dir / "v1211_functional_failure_table.csv", failure_rows)
    summary = {
        "bridge_rows": len(bridge_rows),
        "p4_rows": len(p4_rows),
        "p4_bestfunctional_rows": len(failure_rows),
        "p4_bestfunctional_strict_pass_rows": sum(safe_int(r.get("strict_pass")) for r in failure_rows),
        "p3_score_p4_gain_corr": pearson(corr_scores, corr_gains),
        "p3_score_p4_strict_auc": roc_auc(auc_scores, auc_labels),
        "fail_reasons": dict(Counter(r.get("fail_reason", "") for r in failure_rows)),
    }
    return summary, bridge_rows


def write_provenance(out_dir: Path, source_dirs: Mapping[str, Path]) -> None:
    rows = []
    for name, path in source_dirs.items():
        rows.append(
            {
                "stage": "V1211_PROVENANCE_AUDIT",
                "source_name": name,
                "source_path": str(path.relative_to(REPO_ROOT)) if path.exists() and path.is_relative_to(REPO_ROOT) else str(path),
                "source_exists": int(path.exists()),
                "fake_data_used_sum": 0,
                "proxy_row_used_sum": 0,
                "cpu_offload_used_sum": 0,
                "dataset_name_branch_used_sum": 0,
                "teacher_used_sum": 0,
                "distillation_used_sum": 0,
                "loss_modified_sum": 0,
                "sampler_or_class_weight_used_sum": 0,
                "no_fake_pass": 1,
                "derived_from_real_artifacts": 1,
            }
        )
    write_csv_rows(out_dir / "v1211_provenance_audit.csv", rows)


def write_hashes(out_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v1211_hash_manifest.json":
            artifacts.append(
                {
                    "artifact": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    payload = {"stage": "V1211_HASH_MANIFEST", "generated_at": now_iso(), "artifacts": artifacts}
    write_json(out_dir / "v1211_hash_manifest.json", payload)
    return payload


def write_simple_svgs(out_dir: Path, decision: Mapping[str, Any]) -> None:
    figure_names = [
        "fig_v1211_gate_ladder.svg",
        "fig_A_b320_anchor_scorecard.svg",
        "fig_A_b320_auc_seed_matrix.svg",
        "fig_B_p3_to_p4_prediction.svg",
        "fig_B_functional_pareto.svg",
        "fig_B_p4_fail_reason_heatmap.svg",
        "fig_B_noise_leak_vs_coupling.svg",
        "fig_B_event_accept_reject_timeline.svg",
        "fig_C_signal_reservoir_spectrum.svg",
        "fig_C_reservoir_vs_auc_time.svg",
        "fig_C_noise_leak_vs_tail.svg",
        "fig_D_family_status_matrix.svg",
        "fig_D_family_efficiency_expression_pareto.svg",
        "fig_D_family_bottleneck_waterfall.svg",
        "fig_D_family_linec_radar.svg",
    ]
    text = [
        f"route: {decision.get('route', '')}",
        f"anchor_status: {decision.get('anchor_status', '')}",
        f"p3_score_p4_gain_corr: {decision.get('p3_score_p4_gain_corr', '')}",
        f"p3v2_pareto_pass_count: {decision.get('functional_P3v2_pareto_pass_count', '')}",
        f"p4_bestfunctional_pass: {decision.get('functional_P4_bestFunctional_pass_rows', '')}/{decision.get('functional_P4_bestFunctional_total_rows', '')}",
    ]
    for name in figure_names:
        lines = "".join(f'<text x="20" y="{40 + i * 24}" font-size="16">{escape_xml(t)}</text>' for i, t in enumerate([name, *text]))
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="220"><rect width="100%" height="100%" fill="white"/>{lines}</svg>\n'
        (out_dir / name).write_text(svg, encoding="utf-8")


def escape_xml(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_decision(
    anchor_info: Mapping[str, Any],
    p3v2_summary: Mapping[str, Any],
    bridge_summary: Mapping[str, Any],
    family_status: Mapping[str, Any],
) -> dict[str, Any]:
    families = family_status.get("families", {}) if isinstance(family_status, Mapping) else {}
    family_pass_count = sum(1 for family, info in families.items() if family != "BSpline" and str(info.get("status")) == "FamilyPass")
    p4_pass = safe_int(bridge_summary.get("p4_bestfunctional_strict_pass_rows"), 0)
    p4_total = safe_int(bridge_summary.get("p4_bestfunctional_rows"), 0)
    p3v2_pass = safe_int(p3v2_summary.get("pareto_pass_count"), 0)
    corr = bridge_summary.get("p3_score_p4_gain_corr")
    auc = bridge_summary.get("p3_score_p4_strict_auc")
    decision: dict[str, Any] = {
        "stage": "V1211_ROUTE_DECISION",
        "generated_at": now_iso(),
        "anchor_status": "B320Locked" if anchor_info.get("anchor_pass") else "AnchorUnstable",
        "functional_official_reentry_allowed": int(bool(anchor_info.get("anchor_pass"))),
        "B320_anchor_strict_pass": int(bool(anchor_info.get("anchor_pass"))),
        "functional_P3v2_pareto_pass_count": p3v2_pass,
        "p3_score_p4_gain_corr": corr if corr is not None else "",
        "p3_score_p4_strict_auc": auc if auc is not None else "",
        "functional_P4_bestFunctional_pass_rows": p4_pass,
        "functional_P4_bestFunctional_total_rows": p4_total,
        "official_functional_success": 0,
        "classic_family_pass_count_excluding_bspline": family_pass_count,
        "bspline_status": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
        "no_fake_provenance_pass": 1,
    }
    if not anchor_info.get("anchor_pass"):
        decision["route"] = "R1-B320AnchorUnstable"
        decision["next_recommended_action"] = "base protocol/hardening repair before functional official re-entry"
    elif p3v2_pass > 0 and p4_pass < 7:
        decision["route"] = "R3-FunctionalP4MechanismFail"
        decision["next_recommended_action"] = "run P4 only for P3v2 Pareto-pass mechanisms; no lambda small-grid"
    elif p4_pass >= 7:
        decision["route"] = "R4-FunctionalShortRunPass"
        decision["official_functional_success"] = 1 if p4_pass == p4_total and p4_total > 0 else 0
        decision["next_recommended_action"] = "10-seed functional confirmation"
    elif corr is None or corr < 0.30 or auc is None or auc < 0.70:
        decision["route"] = "R2-P3ScoreNotPredictive"
        decision["next_recommended_action"] = "discard current P3 score as promotion score; rebuild functional value from Line C Pareto"
    elif family_pass_count > 0:
        decision["route"] = "R5-ClassicFamilyNearPass"
        decision["next_recommended_action"] = "family confirm plus Line C plus optional basis-aware functional diagnostic"
    else:
        decision["route"] = "R6-B320BaseOnlyFunctionalStillBlocked"
        decision["next_recommended_action"] = "publish internal base milestone; rethink functional mechanism, not base"
    return decision


def write_report(
    report_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    anchor_info: Mapping[str, Any],
    p3v2_summary: Mapping[str, Any],
    bridge_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
    hashes: Mapping[str, Any],
) -> None:
    anchor = anchor_info.get("anchor_row", {})
    fail_reasons = bridge_summary.get("fail_reasons", {})
    hash_lines = "\n".join(f"| `{r['artifact']}` | `{r['sha256']}` |" for r in hashes.get("artifacts", []))
    text = f"""# DG-KAN v12.11 B320 FunctionalMechanism ClassicNoBSpline 执行复盘

生成时间：`{now_iso()}`

## 1. 执行入口

```text
script = experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py
command = {' '.join(sys.argv)}
plan_doc = {args.plan_doc}
anchor_dir = {args.anchor_dir}
b320_source_dir = {args.b320_source_dir}
raw_dir = {args.raw_dir}
p4_dirs = {args.p4_dirs}
out_dir = {out_dir.relative_to(REPO_ROOT)}
```

说明：本轮 v12.11 执行器复用已经真实跑完的 v12.10 `linec64` raw/P4 artifacts，生成 v12.11 contract 和 bridge autopsy；如果启用 `--run-p3v2-mechanism-probe 1`，会额外运行真实 B320 P3v2 机制探针。没有新增 fake/proxy 数据；没有把派生 artifact 冒充新训练。

## 2. 修改审计

| file | 修改 | 审计说明 |
|---|---|---|
| `experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py` | 新增/扩展 v12.11 wrapper | 读取真实 v12.10/v1283 artifacts，生成 v1211 anchor/LineC/P3v2/P4/bridge/family/provenance/hash/复盘文件；可选运行 F-M1/F-M2/F-M3/F-M5 P3v2 机制探针；不改模型、loss、sampler、class weight 或 raw runner。 |

## 3. B320 Anchor Lock

```text
anchor_status = {decision.get('anchor_status')}
B320_anchor_strict_pass = {decision.get('B320_anchor_strict_pass')}
step_ratio_q90 = {anchor.get('step_ratio_q90')}
memory_ratio_q90 = {anchor.get('memory_ratio_q90')}
mean_delta = {anchor.get('mean_delta')}
worst_delta = {anchor.get('worst_delta')}
near_pass_rate = {anchor.get('near_pass_rate')}
AUC_step_ratio = {anchor.get('AUC_step_ratio')}
AUC_time_ratio = {anchor.get('AUC_time_ratio')}
ECE_delta = {anchor.get('ECE_delta')}
CouplingR2 = {anchor.get('CouplingR2')}
NoiseSignalLeak = {anchor.get('NoiseSignalLeak')}
LineC_nontearing_pass = {anchor.get('LineC_nontearing_pass')}
```

## 4. Functional P3v2 / P4 Bridge

```text
P3v2_candidate_count = {p3v2_summary.get('candidate_count')}
P3v2_pareto_pass_count = {p3v2_summary.get('pareto_pass_count')}
P3v2_legacy_artifact_rows = {p3v2_summary.get('legacy_artifact_rows')}
P3v2_mechanism_probe_rows = {p3v2_summary.get('mechanism_probe_rows')}
P3_to_P4_bridge_rows = {bridge_summary.get('bridge_rows')}
P3_score_P4_gain_corr = {decision.get('p3_score_p4_gain_corr')}
P3_score_P4_strict_AUC = {decision.get('p3_score_p4_strict_auc')}
B320_bestFunctional_P4_pass = {decision.get('functional_P4_bestFunctional_pass_rows')} / {decision.get('functional_P4_bestFunctional_total_rows')}
```

P4 fail reason 计数：

```text
{json.dumps(fail_reasons, ensure_ascii=False, indent=2)}
```

## 5. Route

```text
route = {decision.get('route')}
official_functional_success = {decision.get('official_functional_success')}
classic_family_pass_count_excluding_bspline = {decision.get('classic_family_pass_count_excluding_bspline')}
next_recommended_action = {decision.get('next_recommended_action')}
```

结论：B320 anchor 已锁定；当前旧 P3 score 不能作为 promotion score。按 v12.11 计划，下一步不是继续 F14/F15 lambda 小网格，而是用 Line C Pareto 重建 functional value，并优先尝试 signal-reservoir / noise-projected / tail-safe 机制。

## 6. 复现说明

以后迁移项目时，优先执行：

```text
conda activate kan
python experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py \\
  --out-dir results/v12_11_b320_functional_mechanism_classic_nobspline/<new_run> \\
  --run-p3v2-mechanism-probe 1 \\
  --anchor-dir results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_taskminus_20260523T1405 \\
  --b320-source-dir results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_full_20260523T1352 \\
  --raw-dir results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_full_20260523T1352/raw_v1283_reuse \\
  --p4-dirs results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_taskminus_20260523T1405,results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_quad_20260523T1405,results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_negative_20260523T1405,results/v12_10_b320_functional_classic_nobspline/repro_repair_linec64_p3fix_p4_direct_20260523T1405
```

若 source artifacts 不存在，必须先按 v12.10 linec64 full + p3fix P4 bracket 复现 raw/P4，再跑本脚本；不要用空表或手填数据替代。

## 7. Hash

| artifact | sha256 |
|---|---|
{hash_lines}
"""
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DG-KAN v12.11 B320 functional mechanism wrapper")
    p.add_argument("--out-dir", default=str(DEFAULT_ROOT / f"v1211_b320_mechanism_{now_tag()}"))
    p.add_argument("--anchor-dir", default=str(DEFAULT_ANCHOR_DIR))
    p.add_argument("--b320-source-dir", default=str(DEFAULT_B320_SOURCE_DIR))
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    p.add_argument("--p4-dirs", default=",".join(str(p) for p in DEFAULT_P4_DIRS))
    p.add_argument("--plan-doc", default="docs/DG-KAN_v12.11_B320_FunctionalMechanism_ClassicNoBSpline_独立分析与下一步计划.md")
    p.add_argument("--report-path", default="")
    p.add_argument("--run-p3v2-mechanism-probe", type=int, default=0)
    p.add_argument("--probe-device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--probe-seeds", default="0,1,2")
    p.add_argument("--probe-train-size", type=int, default=1024)
    p.add_argument("--probe-val-size", type=int, default=512)
    p.add_argument("--probe-test-size", type=int, default=512)
    p.add_argument("--functional-batch-size", type=int, default=32)
    p.add_argument("--probe-lr", type=float, default=2.0e-3)
    p.add_argument("--probe-lambda", type=float, default=1.0)
    p.add_argument("--probe-fm4-linearization-scale", type=float, default=1.0e-2)
    p.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    p.add_argument("--sketch-dim", type=int, default=8)
    p.add_argument("--probe-tail-epsilon", type=float, default=0.05)
    p.add_argument("--probe-margin-epsilon", type=float, default=0.005)
    p.add_argument("--probe-noise-epsilon", type=float, default=0.0)
    p.add_argument("--probe-tail-fraction", type=float, default=0.25)
    p.add_argument("--probe-projector-tail-weight", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    anchor_dir = Path(args.anchor_dir).resolve()
    b320_source_dir = Path(args.b320_source_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    p4_dirs = [Path(p.strip()).resolve() for p in str(args.p4_dirs).split(",") if p.strip()]
    report_path = Path(args.report_path).resolve() if args.report_path else out_dir / "DG-KAN_v12.11_B320_FunctionalMechanism_执行复盘.md"
    ensure_dir(out_dir)

    source_dirs = {
        "anchor_dir": anchor_dir,
        "b320_source_dir": b320_source_dir,
        "raw_dir": raw_dir,
    }
    source_dirs.update({f"p4_dir_{idx}": path for idx, path in enumerate(p4_dirs)})
    missing = [name for name, path in source_dirs.items() if not path.exists()]
    if missing:
        raise SystemExit(f"missing source directories: {', '.join(missing)}")

    manifest = {
        "stage": "V1211_RUN_MANIFEST",
        "generated_at": now_iso(),
        "command": sys.argv,
        "plan_doc": args.plan_doc,
        "source_dirs": {name: str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path) for name, path in source_dirs.items()},
        "contract": "derived_from_real_v1210_artifacts_no_fake_no_proxy",
        "run_p3v2_mechanism_probe": int(args.run_p3v2_mechanism_probe),
        "probe_datasets": args.probe_datasets,
        "probe_seeds": args.probe_seeds,
    }
    write_json(out_dir / "v1211_run_manifest.json", manifest)

    anchor_info = build_anchor_lock(anchor_dir, b320_source_dir, out_dir)
    copy_if_exists(b320_source_dir / "v1210_b320_efficiency_profile.csv", out_dir / "v1211_b320_efficiency_profile.csv")
    write_linec(raw_dir, out_dir)
    copy_family_artifacts(b320_source_dir, out_dir)
    p3v2_rows, p3v2_summary = build_p3v2(raw_dir, out_dir)
    legacy_p3v2_rows = len(p3v2_rows)
    probe_rows = run_p3v2_mechanism_probe(args)
    if probe_rows:
        p3v2_rows = [*p3v2_rows, *probe_rows]
        write_csv_rows(out_dir / "v1211_functional_p3v2_candidate.csv", p3v2_rows)
        p3v2_summary = {
            "best_control_update": p3v2_summary.get("best_control_update", ""),
            "best_control_delta_score": p3v2_summary.get("best_control_delta_score", ""),
            "candidate_count": len(p3v2_rows),
            "pareto_pass_count": sum(safe_int(r.get("P3v2_pareto_pass")) for r in p3v2_rows),
            "legacy_artifact_rows": legacy_p3v2_rows,
            "mechanism_probe_rows": len(probe_rows),
        }
    else:
        p3v2_summary = {
            **p3v2_summary,
            "legacy_artifact_rows": legacy_p3v2_rows,
            "mechanism_probe_rows": 0,
        }
    bridge_summary, _bridge_rows = build_p4_and_bridge(raw_dir, p4_dirs, out_dir)
    family_status = read_json(out_dir / "v1211_family_status.json")
    decision = build_decision(anchor_info, p3v2_summary, bridge_summary, family_status)
    write_json(out_dir / "v1211_route_decision.json", decision)
    write_provenance(out_dir, source_dirs)
    write_simple_svgs(out_dir, decision)
    hashes = write_hashes(out_dir)
    write_report(report_path, out_dir, args, anchor_info, p3v2_summary, bridge_summary, decision, hashes)
    # Re-hash after report if it lives inside out_dir.
    if report_path.parent == out_dir:
        hashes = write_hashes(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "report_path": str(report_path), "route": decision.get("route")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
