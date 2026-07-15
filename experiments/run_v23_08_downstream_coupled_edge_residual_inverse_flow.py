#!/usr/bin/env python3
"""DG-KAN v23.08 downstream-coupled edge residual inverse flow runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import math
import os
import py_compile
import re
import sys
import time
import tokenize
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_07_edge_function_residual_inverse_flow as v2307
from dgkan.fu import downstream_coupled_residual_inverse as dcerif
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevPopulationFlow


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.08_DownstreamCoupledEdgeResidualInverseFlow_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.08_DownstreamCoupledEdgeResidualInverseFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.08_DownstreamCoupledEdgeResidualInverseFlow_实验结果复盘.md"
HELPER = ROOT / "dgkan/fu/downstream_coupled_residual_inverse.py"
OUT_ROOT = Path(os.environ.get("V2308_OUT_ROOT", str(ROOT / "results/v23_08"))).resolve()

AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "external_product_feature_used": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "structure_transform_in_forward": 0,
}

PART_D_SCHEMES = [
    "D0_gradient_adjoint",
    "D1_local_EFRF",
    "D2_last_layer_exact_EFRF",
    "D3_penultimate_downstream_exact_or_CG",
    "D4_topdown_last_then_penultimate_refresh",
    "D5_hidden_downstream_sketch_rank8",
    "D6_hidden_downstream_sketch_rank16",
    "D7_random_projected_downstream_EFRF",
    "D8_shuffled_design_downstream_EFRF",
    "D9_same_compute_noop",
]
DOWNSTREAM_D_CANDIDATES = {
    "D3_penultimate_downstream_exact_or_CG",
    "D4_topdown_last_then_penultimate_refresh",
    "D5_hidden_downstream_sketch_rank8",
    "D6_hidden_downstream_sketch_rank16",
}

PART_E_SCHEMES = [
    "E0_FunctionalGram_one_step_baseline",
    "E1_v23_07_local_EFRF_one_step_C5_reference",
    "E2_last_layer_exact_EFRF_one_step",
    "E3_penultimate_downstream_EFRF_one_step",
    "E4_topdown_last_penultimate_one_step",
    "E5_hidden_downstream_sketch_one_step",
    "E6_random_projected_downstream_one_step",
    "E7_shuffled_design_downstream_one_step",
    "E8_same_compute_noop_one_step",
    "E9_topdown_last_penultimate_adjoint_local_one_step",
]
DOWNSTREAM_E_CANDIDATES = {
    "E3_penultimate_downstream_EFRF_one_step",
    "E4_topdown_last_penultimate_one_step",
    "E5_hidden_downstream_sketch_one_step",
}
LAST_LAYER_E_CANDIDATES = {"E2_last_layer_exact_EFRF_one_step"}

PART_F_SCHEMES = [
    "F0_FunctionalGram_baseline",
    "F1_BlockSNR_baseline",
    "F2_H10_known_structure_baseline",
    "F3_v23_07_local_EFRF_all_layer_GS_control",
    "F4_last_layer_only_EFRF",
    "F5_last_two_layers_topdown_EFRF",
    "F6_penultimate_downstream_CG_EFRF",
    "F7_topdown_last_to_first_refresh_EFRF",
    "F8_hidden_downstream_sketch_rank8_EFRF",
    "F9_hidden_downstream_sketch_rank16_EFRF",
    "F14_hidden_downstream_sketch_rank32_EFRF",
    "F10_hybrid_EFRF_every5_steps_FunctionalGram_between",
    "F15_hybrid_EFRF_every10_steps_FunctionalGram_between",
    "F16_hybrid_EFRF_every10_alignment_gated_FunctionalGram_between",
    "F17_hybrid_EFRF_every10_alignment_softscaled_FunctionalGram_between",
    "F18_hybrid_EFRF_every10_negative_alignment_softclip_FunctionalGram_between",
    "F19_hybrid_EFRF_every10_alignment_gated_tail99_guard_FunctionalGram_between",
    "F20_hybrid_EFRF_every10_alignment_gated_cumulative_tail99_guard_FunctionalGram_between",
    "F21_hybrid_EFRF_every10_alignment_gated_tail_weighted_residual_FunctionalGram_between",
    "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction",
    "F23_hybrid_EFRF_every20_steps_FunctionalGram_between",
    "F24_hybrid_EFRF_every20_alignment_gated_FunctionalGram_between",
    "F25_hybrid_EFRF_every30_steps_FunctionalGram_between",
    "F26_hybrid_EFRF_every30_alignment_gated_FunctionalGram_between",
    "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between",
    "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
    "F11_random_projected_residual_flow",
    "F12_shuffled_design_residual_flow",
    "F13_same_compute_noop",
]
F_CANDIDATE_SCHEMES = {
    "F4_last_layer_only_EFRF",
    "F5_last_two_layers_topdown_EFRF",
    "F6_penultimate_downstream_CG_EFRF",
    "F7_topdown_last_to_first_refresh_EFRF",
    "F8_hidden_downstream_sketch_rank8_EFRF",
    "F9_hidden_downstream_sketch_rank16_EFRF",
    "F14_hidden_downstream_sketch_rank32_EFRF",
    "F10_hybrid_EFRF_every5_steps_FunctionalGram_between",
    "F15_hybrid_EFRF_every10_steps_FunctionalGram_between",
    "F16_hybrid_EFRF_every10_alignment_gated_FunctionalGram_between",
    "F17_hybrid_EFRF_every10_alignment_softscaled_FunctionalGram_between",
    "F18_hybrid_EFRF_every10_negative_alignment_softclip_FunctionalGram_between",
    "F19_hybrid_EFRF_every10_alignment_gated_tail99_guard_FunctionalGram_between",
    "F20_hybrid_EFRF_every10_alignment_gated_cumulative_tail99_guard_FunctionalGram_between",
    "F21_hybrid_EFRF_every10_alignment_gated_tail_weighted_residual_FunctionalGram_between",
    "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction",
    "F23_hybrid_EFRF_every20_steps_FunctionalGram_between",
    "F24_hybrid_EFRF_every20_alignment_gated_FunctionalGram_between",
    "F25_hybrid_EFRF_every30_steps_FunctionalGram_between",
    "F26_hybrid_EFRF_every30_alignment_gated_FunctionalGram_between",
    "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between",
    "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
}

ADJOINT_LOCAL_HYBRID_SCHEMES = {
    "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between",
    "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
}

HYBRID_REFRESH_INTERVALS = {
    "F10_hybrid_EFRF_every5_steps_FunctionalGram_between": 5,
    "F15_hybrid_EFRF_every10_steps_FunctionalGram_between": 10,
    "F16_hybrid_EFRF_every10_alignment_gated_FunctionalGram_between": 10,
    "F17_hybrid_EFRF_every10_alignment_softscaled_FunctionalGram_between": 10,
    "F18_hybrid_EFRF_every10_negative_alignment_softclip_FunctionalGram_between": 10,
    "F19_hybrid_EFRF_every10_alignment_gated_tail99_guard_FunctionalGram_between": 10,
    "F20_hybrid_EFRF_every10_alignment_gated_cumulative_tail99_guard_FunctionalGram_between": 10,
    "F21_hybrid_EFRF_every10_alignment_gated_tail_weighted_residual_FunctionalGram_between": 10,
    "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction": 10,
    "F23_hybrid_EFRF_every20_steps_FunctionalGram_between": 20,
    "F24_hybrid_EFRF_every20_alignment_gated_FunctionalGram_between": 20,
    "F25_hybrid_EFRF_every30_steps_FunctionalGram_between": 30,
    "F26_hybrid_EFRF_every30_alignment_gated_FunctionalGram_between": 30,
    "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between": 20,
    "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between": 30,
}

ALIGNMENT_REFERENCE_SCHEMES = {
    "F16_hybrid_EFRF_every10_alignment_gated_FunctionalGram_between",
    "F17_hybrid_EFRF_every10_alignment_softscaled_FunctionalGram_between",
    "F18_hybrid_EFRF_every10_negative_alignment_softclip_FunctionalGram_between",
    "F19_hybrid_EFRF_every10_alignment_gated_tail99_guard_FunctionalGram_between",
    "F20_hybrid_EFRF_every10_alignment_gated_cumulative_tail99_guard_FunctionalGram_between",
    "F21_hybrid_EFRF_every10_alignment_gated_tail_weighted_residual_FunctionalGram_between",
    "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction",
    "F24_hybrid_EFRF_every20_alignment_gated_FunctionalGram_between",
    "F26_hybrid_EFRF_every30_alignment_gated_FunctionalGram_between",
}

PART_I_SCHEMES = [
    "I0_FunctionalGram_baseline",
    "I1_BlockSNR_baseline",
    "I2_H10_scalar_dual_if_applicable_diagnostic",
    "I3_DownstreamEFRF_best_positive_control_scheme",
    "I4_DownstreamEFRF_without_Wpop",
    "I5_random_projected_residual_flow",
    "I6_same_compute_noop",
]

PART_I_SCHEME_TO_F_SCHEME = {
    "I0_FunctionalGram_baseline": "F0_FunctionalGram_baseline",
    "I1_BlockSNR_baseline": "F1_BlockSNR_baseline",
    "I3_DownstreamEFRF_best_positive_control_scheme": "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
    "I4_DownstreamEFRF_without_Wpop": "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
    "I5_random_projected_residual_flow": "F11_random_projected_residual_flow",
    "I6_same_compute_noop": "F13_same_compute_noop",
}


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def command_text(argv: Iterable[str]) -> str:
    cmd = " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])
    env = os.environ.get("V2308_OUT_ROOT")
    return f"V2308_OUT_ROOT={env} {cmd}" if env else cmd


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def csv_items(text: Any) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def float_items(text: Any) -> list[float]:
    out: list[float] = []
    for item in csv_items(text):
        try:
            out.append(float(item))
        except Exception:
            pass
    return out or [1.0]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for item_idx, item in enumerate(items) if item_idx % count == idx]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**AUDIT_DEFAULTS, **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(enriched)
    return path


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    if not path.exists() or path.stat().st_size == 0:
        write_rows(path, enriched)
        return
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        keys = list(reader.fieldnames or [])
    for row in enriched:
        for key in row:
            if key not in keys:
                existing = read_rows(path)
                write_rows(path, existing + enriched)
                return
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writerows(enriched)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.08 Downstream-Coupled Edge Residual Inverse Flow 执行日志\n\n"
            "- 原则：不造假；不使用 held/test 诱导；每条命令和产物路径尽量可复现。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.08 Downstream-Coupled Edge Residual Inverse Flow 实验结果复盘\n\n"
            "- 原则：只记录真实 artifact 中的数据；失败也写证据链和修复尝试。\n",
            encoding="utf-8",
        )


def append_exec(title: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title} {status}\n\n")
        fh.write(f"- command: `{command}`\n")
        if gpu:
            fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, data: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def next_actions(part: str, route: str, blocker: str, allowed: list[str], required: list[str] | None = None) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part.upper(),
            "route": route,
            "dominant_blocker": blocker,
            "allowed_next_actions": allowed,
            "forbidden_next_actions": [
                "fabricate_data",
                "lower_gate_thresholds_to_force_pass",
                "runtime_winner_selection",
                "held_test_induction",
                "add_mlp_stem_or_readout",
                "add_new_edge_function_family",
            ],
            "required_artifacts_before_rerun": required or [],
            "max_repair_rounds": 3,
            "promotion_allowed_after_repair": False,
        },
    )


def write_failure_decomposition(part: str, payload: dict[str, Any]) -> Path:
    return write_json(OUT_ROOT / f"part_{part.lower()}_failure_decomposition.json", {"part": part.upper(), **payload})


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    denom = aa.norm() * bb.norm()
    if float(denom.detach().cpu().item()) <= 1.0e-12:
        return 0.0
    return float((aa @ bb).div(denom).detach().cpu().item())


def debt_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        "Brier_delta": after["brier"] - before["brier"],
        "ECE_delta": after["ece"] - before["ece"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "tail99_delta": after["tail99"] - before["tail99"],
        "margin10_delta": after["margin10"] - before["margin10"],
        "debt_delta": after["debt"] - before["debt"],
    }


def no_debt_ok(deltas: dict[str, float], budget: float) -> int:
    return int(
        deltas["Brier_delta"] <= float(budget)
        and deltas["ECE_delta"] <= float(budget)
        and deltas["tail95_delta"] <= float(budget)
        and deltas["tail99_delta"] <= float(budget)
        and deltas["margin10_delta"] >= -float(budget)
        and deltas["debt_delta"] <= float(budget)
    )


def strip_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}:
                out.append("\n" if token.type in {tokenize.NL, tokenize.NEWLINE} else " ")
            else:
                out.append(token.string)
    except tokenize.TokenError:
        return text
    return "".join(out)


def static_identity_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_selector": re.compile(r"\b(select_best|choose_best|winner_selector|runtime_winner)\b"),
        "held_test_induction": re.compile(r"\bheld_.*(select|choose|lambda|rank|gate)|test_.*(select|choose|lambda|rank|gate)\b", re.I),
        "new_edge_function": re.compile(r"\b(product_edge|external_product|convkan|bspline_new)\b", re.I),
        "mlp_stem_or_readout": re.compile(r"\bMatchedMLP\s*\(|MLPStem|MLPReadout\b"),
    }
    hits: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            hits.append({"file": rel(path), "check": "missing_file", "match": "missing"})
            continue
        code = strip_code(path)
        for check, pattern in patterns.items():
            match = pattern.search(code)
            if match:
                hits.append({"file": rel(path), "check": check, "match": match.group(0)})
    return int(not hits), hits


def eig_condition(x: torch.Tensor) -> tuple[float, float, float]:
    vals = torch.linalg.eigvalsh(dcerif.sym(x.to(dtype=torch.float64)))
    vals = vals[torch.isfinite(vals)]
    if int(vals.numel()) == 0:
        return 0.0, 0.0, float("inf")
    mn = float(vals.min().detach().cpu().item())
    mx = float(vals.max().detach().cpu().item())
    return mn, mx, float(mx / max(mn, 1.0e-12)) if mn > 0 else float("inf")


def output_residual(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)
    prob = torch.softmax(logits.float(), dim=1)
    onehot = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(device=logits.device, dtype=prob.dtype)
    residual = (onehot - prob).to(dtype=torch.float64) / float(max(1, int(x.shape[0])))
    key = str(target).lower()
    if key in {"norm", "norm_normalized"}:
        residual = residual / residual.square().mean().sqrt().clamp_min(1.0e-12)
    elif key in {"tail_weighted_norm", "tail_weighted"}:
        true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1).detach().to(dtype=torch.float64)
        weights = (1.0 - true_prob).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1.0e-12)
        residual = residual * weights.reshape(-1, 1)
        residual = residual / residual.square().mean().sqrt().clamp_min(1.0e-12)
    elif key in {"tail_quantile_norm", "tail_top5_norm"}:
        true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1).detach().to(dtype=torch.float64)
        risk = (1.0 - true_prob).clamp_min(0.0)
        threshold = torch.quantile(risk, 0.95)
        weights = (risk >= threshold).to(dtype=torch.float64)
        if float(weights.sum().detach().cpu().item()) < 1.0:
            weights = torch.zeros_like(risk, dtype=torch.float64)
            weights[int(torch.argmax(risk).detach().cpu().item())] = 1.0
        weights = weights / weights.mean().clamp_min(1.0e-12)
        residual = residual * weights.reshape(-1, 1)
        residual = residual / residual.square().mean().sqrt().clamp_min(1.0e-12)
    elif key == "trust_scaled":
        residual = residual / residual.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        residual = residual / math.sqrt(float(max(1, int(residual.shape[1]))))
    elif key in {"margin_weighted_norm", "margin_weighted"}:
        true_prob = prob.gather(1, y.long().reshape(-1, 1)).reshape(-1).detach().to(dtype=torch.float64)
        masked_prob = prob.detach().to(dtype=torch.float64).clone()
        masked_prob.scatter_(1, y.long().reshape(-1, 1), -1.0)
        other_prob = masked_prob.max(dim=1).values
        margin = true_prob - other_prob
        weights = (1.0 - margin).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1.0e-12)
        residual = residual * weights.reshape(-1, 1)
        residual = residual / residual.square().mean().sqrt().clamp_min(1.0e-12)
    elif key in {"uncertainty_weighted_norm", "entropy_weighted_norm", "entropy_weighted"}:
        stable_prob = prob.detach().to(dtype=torch.float64).clamp_min(1.0e-12)
        entropy = -(stable_prob * stable_prob.log()).sum(dim=1)
        entropy = entropy / math.log(float(max(2, int(stable_prob.shape[1]))))
        weights = entropy.clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1.0e-12)
        residual = residual * weights.reshape(-1, 1)
        residual = residual / residual.square().mean().sqrt().clamp_min(1.0e-12)
    return logits.detach(), residual.detach()


def model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def load_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({name: tensor.detach().clone() for name, tensor in state.items()})


def apply_delta(model: Any, layer_id: int, delta: torch.Tensor, alpha: float = 1.0) -> None:
    v2307.apply_coeff_deltas(model, {int(layer_id): delta}, float(alpha))


def actual_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return v2307.actual_metrics(model, x, y)


def load_part_i_real_bundle(dataset: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    name = str(dataset)
    low = name.lower().replace("_", "-")
    train_size = int(args.part_i_train_size)
    guard_size = int(args.part_i_guard_size)
    held_size = int(args.part_i_held_size)
    test_size = max(1, int(args.part_i_test_size))
    if low in {"mnist", "fashionmnist", "fashion-mnist", "fmnist", "kmnist"}:
        from experiments import dgkan_core as data_core

        loader_name = {"fashionmnist": "Fashion-MNIST", "fashion-mnist": "Fashion-MNIST", "fmnist": "Fashion-MNIST"}.get(low, name)
        bundle = data_core.load_vision_bundle(
            loader_name,
            data_root=Path(args.part_i_data_root),
            train_size=train_size,
            val_size=guard_size + held_size,
            test_size=test_size,
            seed=int(seed),
            download=False,
            allow_fake_data=False,
        )
        x_val = bundle.x_val.float()
        y_val = bundle.y_val.long()
        guard_n = min(guard_size, max(1, int(x_val.shape[0]) - 1))
        held_n = min(held_size, max(1, int(x_val.shape[0]) - guard_n))
        return {
            "dataset_family": "visual",
            "source_kind": f"experiments.dgkan_core.load_vision_bundle:{loader_name}",
            "dataset_loader_name": loader_name,
            "input_dim": int(bundle.input_dim),
            "num_classes": int(bundle.num_classes),
            "x_train": bundle.x_train.float().to(device),
            "y_train": bundle.y_train.long().to(device),
            "x_guard": x_val[:guard_n].to(device),
            "y_guard": y_val[:guard_n].to(device),
            "x_held": x_val[guard_n : guard_n + held_n].to(device),
            "y_held": y_val[guard_n : guard_n + held_n].to(device),
            "train_size_actual": int(bundle.x_train.shape[0]),
            "guard_size_actual": int(guard_n),
            "held_size_actual": int(held_n),
            "used_fake_data": int(bool(bundle.used_fake_data)),
        }
    if low == "wine":
        from sklearn.datasets import load_wine

        data = load_wine()
        x_np = data.data.astype("float32")
        y_np = data.target.astype("int64")
        rng = np.random.default_rng(int(seed))
        idx = rng.permutation(len(y_np))
        requested = train_size + guard_size + held_size + test_size
        need = min(len(idx), requested)
        idx = idx[:need]
        if need < train_size + guard_size + held_size:
            train_n = max(1, int(round(0.55 * need)))
            guard_n = max(1, int(round(0.225 * need)))
            held_n = max(1, need - train_n - guard_n)
            while train_n + guard_n + held_n > need and held_n > 1:
                held_n -= 1
        else:
            train_n, guard_n, held_n = train_size, guard_size, held_size
        train_idx = idx[:train_n]
        guard_idx = idx[train_n : train_n + guard_n]
        held_idx = idx[train_n + guard_n : train_n + guard_n + held_n]
        mu = x_np[train_idx].mean(axis=0, keepdims=True)
        sigma = x_np[train_idx].std(axis=0, keepdims=True)
        sigma[sigma < 1.0e-6] = 1.0
        x_np = (x_np - mu) / sigma
        return {
            "dataset_family": "tabular",
            "source_kind": "sklearn.datasets.load_wine",
            "dataset_loader_name": "Wine",
            "input_dim": int(x_np.shape[1]),
            "num_classes": int(np.max(y_np) + 1),
            "x_train": torch.tensor(x_np[train_idx], dtype=torch.float32, device=device),
            "y_train": torch.tensor(y_np[train_idx], dtype=torch.long, device=device),
            "x_guard": torch.tensor(x_np[guard_idx], dtype=torch.float32, device=device),
            "y_guard": torch.tensor(y_np[guard_idx], dtype=torch.long, device=device),
            "x_held": torch.tensor(x_np[held_idx], dtype=torch.float32, device=device),
            "y_held": torch.tensor(y_np[held_idx], dtype=torch.long, device=device),
            "train_size_actual": int(train_n),
            "guard_size_actual": int(guard_n),
            "held_size_actual": int(held_n),
            "used_fake_data": 0,
        }
    raise ValueError(f"unknown Part I dataset {dataset!r}; expected MNIST/FashionMNIST/KMNIST/Wine")


def part_i_f_args(args: argparse.Namespace, *, residual_target_override: str | None = None) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.part_f_steps = int(args.part_i_steps)
    out.part_f_lambda = float(args.part_i_lambda)
    out.part_f_residual_target = str(residual_target_override or args.part_i_residual_target)
    out.part_f_alphas = str(args.part_i_alphas)
    out.part_f_delta_scale = float(args.part_i_delta_scale)
    out.part_f_delta_sign = float(args.part_i_delta_sign)
    out.part_f_trust_mode = str(args.part_i_trust_mode)
    out.part_f_trust_tolerance = float(args.part_i_trust_tolerance)
    out.part_f_local_reference_target = str(args.part_i_local_reference_target)
    out.part_f_functionalgram_lr = float(args.part_i_functionalgram_lr)
    out.part_f_adam_lr = float(args.part_i_adam_lr)
    out.part_f_solver = str(args.part_i_solver)
    out.part_f_sketch_rank = int(args.part_i_sketch_rank)
    out.part_f_activation_drift_cap = float(args.part_i_activation_drift_cap)
    out.part_f_alignment_min = float(args.part_i_alignment_min)
    out.part_f_alignment_soft_floor = float(args.part_i_alignment_soft_floor)
    out.part_f_alignment_soft_ceiling = float(args.part_i_alignment_soft_ceiling)
    out.part_f_alignment_soft_min_scale = float(args.part_i_alignment_soft_min_scale)
    out.part_f_alignment_soft_power = float(args.part_i_alignment_soft_power)
    out.part_f_tail99_step_budget = float(args.part_i_tail99_step_budget)
    out.part_f_tail99_cumulative_budget = float(args.part_i_tail99_cumulative_budget)
    out.part_f_tail_correction_scales = str(args.part_f_tail_correction_scales)
    out.part_f_tail_correction_target = str(args.part_f_tail_correction_target)
    out.part_f_tail_correction_layer_group = str(args.part_f_tail_correction_layer_group)
    out.part_f_tail_correction_transport_min = float(args.part_f_tail_correction_transport_min)
    out.part_f_model_seed_scheme_key = str(args.part_i_model_seed_scheme_key)
    out.part_f_checkpoint = str(args.part_i_checkpoint)
    out.checkpoint_steps = int(args.part_i_checkpoint_steps)
    out.checkpoint_lr = float(args.part_i_checkpoint_lr)
    out.batch_size = int(args.part_i_batch_size)
    out.part_d_solver = str(args.part_i_solver)
    return out


def fit_metrics(design_or_op: torch.Tensor | dcerif.DownstreamOperator, residual: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    pred = design_or_op.apply_H(delta) if isinstance(design_or_op, dcerif.DownstreamOperator) else design_or_op @ delta
    rr = residual.to(device=pred.device, dtype=torch.float64)
    before = rr.square().mean().clamp_min(1.0e-12)
    after = (rr - pred.to(dtype=torch.float64)).square().mean()
    denom = pred.norm() * rr.norm()
    cosine = float(((pred.reshape(-1) @ rr.reshape(-1)) / denom.clamp_min(1.0e-12)).detach().cpu().item())
    return {
        "fit_before": float(before.detach().cpu().item()),
        "fit_after": float(after.detach().cpu().item()),
        "fit_improvement_ratio": float(((before - after) / before).detach().cpu().item()),
        "output_alignment_cosine": cosine,
    }


def solve_local_hidden(model: Any, acts: list[torch.Tensor], hidden_residual: torch.Tensor, layer_id: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    phi = model.layer_phi(acts[int(layer_id)], int(layer_id))
    gram = v2307.expand_edge_gram(model, int(layer_id), str(args.basis_key), 0.0, args, device=device)
    return v2307.solve_normal(phi, hidden_residual.to(device=device, dtype=torch.float64), gram, float(args.part_d_lambda), None)


def downstream_adjoint_hidden_target(
    model: Any,
    activation_after_layer: torch.Tensor,
    start_layer_id: int,
    output_residual: torch.Tensor,
) -> torch.Tensor:
    h = activation_after_layer.detach().clone().requires_grad_(True)
    out = dcerif.downstream_forward_from_activation(model, h, int(start_layer_id))
    residual = output_residual.to(device=out.device, dtype=out.dtype)
    score = (out * residual).sum()
    grad = torch.autograd.grad(score, h, retain_graph=False, create_graph=False)[0]
    return grad.detach().to(dtype=torch.float64)


def solve_downstream(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    layer_id: int,
    target: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    residual_override: torch.Tensor | None = None,
    random_project: bool = False,
    shuffle_design: bool = False,
    sketch_rank: int = 0,
    seed: int = 0,
) -> tuple[torch.Tensor, dcerif.DownstreamOperator, torch.Tensor, dict[str, Any]]:
    logits, acts = model.forward_with_activations(x)
    _base_logits, residual = output_residual(model, x, y, target)
    if residual_override is not None:
        residual = residual_override.to(device=device, dtype=torch.float64)
    operator = dcerif.make_downstream_operator(model, acts, int(layer_id))
    rr = residual
    gen = torch.Generator(device=device).manual_seed(2308000 + int(seed))
    if random_project:
        flat = rr.reshape(-1, 1)
        q, _ = torch.linalg.qr(torch.randn((int(flat.shape[0]), min(int(flat.shape[0]), max(1, int(flat.shape[0]) // 4))), device=device, dtype=torch.float64, generator=gen))
        rr = (q @ (q.T @ flat)).reshape_as(rr)
    gram = v2307.expand_edge_gram(model, int(layer_id), str(args.basis_key), 0.0, args, device=device)
    if sketch_rank > 0:
        h, q = dcerif.sketch_downstream_operator(operator, int(sketch_rank), seed=int(seed))
        r = q.T @ rr.reshape(-1, 1)
        metric = dcerif.expanded_edge_metric(gram, operator.layer_output_dim)
        normal = h.T @ h + float(args.part_d_lambda) * metric
        rhs = h.T @ r
        if shuffle_design:
            perm = torch.randperm(int(h.shape[0]), generator=gen, device=device)
            normal = h[perm].T @ h[perm] + float(args.part_d_lambda) * metric
        flat, diag = dcerif.solve_ridge_normal_exact(normal, rhs)
        diag["cg_iterations"] = 0.0
        diag["rank_H"] = float(torch.linalg.matrix_rank(h, tol=1.0e-8).detach().cpu().item())
        _mn, _mx, cond = eig_condition(normal)
        diag["condition_number_after_ridge"] = cond
        delta = flat.reshape(operator.basis_dim, operator.layer_output_dim)
    else:
        if shuffle_design:
            h = operator.explicit_matrix()
            perm = torch.randperm(int(h.shape[0]), generator=gen, device=device)
            metric = dcerif.expanded_edge_metric(gram, operator.layer_output_dim)
            normal = h[perm].T @ h[perm] + float(args.part_d_lambda) * metric
            rhs = h.T @ rr.reshape(-1, 1)
            flat, diag = dcerif.solve_ridge_normal_exact(normal, rhs)
            diag["rank_H"] = float(torch.linalg.matrix_rank(h, tol=1.0e-8).detach().cpu().item())
            _mn, _mx, cond = eig_condition(normal)
            diag["condition_number_after_ridge"] = cond
            delta = flat.reshape(operator.basis_dim, operator.layer_output_dim)
        else:
            delta, diag = dcerif.solve_downstream_ridge_cg(operator, rr, gram, float(args.part_d_lambda), solver=str(args.part_d_solver), tol=float(args.cg_tol), max_iter=int(args.cg_max_iter))
    return delta, operator, rr, diag


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    device = device_from_args(args)
    compile_pass = 1
    compile_errors = []
    for path in [RUNNER, HELPER]:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_pass = 0
            compile_errors.append({"file": rel(path), "error": repr(exc)})
    try:
        importlib.import_module("dgkan.fu.downstream_coupled_residual_inverse")
        import_pass = 1
    except Exception as exc:
        import_pass = 0
        compile_errors.append({"file": "import", "error": repr(exc)})
    static_scan_pass, static_hits = static_identity_scan([RUNNER, HELPER])
    gen = torch.Generator(device=device).manual_seed(23080001)
    a = torch.randn((16, 16), device=device, dtype=torch.float64, generator=gen)
    metric = a.T @ a + 0.1 * torch.eye(16, device=device, dtype=torch.float64)
    coeff = torch.randn((16, 3), device=device, dtype=torch.float64, generator=gen)
    white, chol = dcerif.whiten_edge_coeff(metric, coeff)
    unwhite = dcerif.unwhiten_edge_coeff(chol, white)
    whiten_err = float((unwhite - coeff).norm().div(coeff.norm().clamp_min(1.0e-12)).detach().cpu().item())
    normal = metric + 0.5 * torch.eye(16, device=device, dtype=torch.float64)
    rhs = torch.randn((16, 2), device=device, dtype=torch.float64, generator=gen)
    _sol, exact_diag = dcerif.solve_ridge_normal_exact(normal, rhs)
    _cg_sol, cg_diag = dcerif.solve_ridge_normal_cg(normal, rhs, tol=1.0e-7, max_iter=512)
    x, y, _xg, _yg = v2307.visual_data("local_patch_interaction", 0, args, device)
    model = v2307.make_model(str(args.basis_key), 2, int(x.shape[1]), int(y.max().item()) + 1, 230800, args, device, basis_input_gain=0.25)
    logits, acts = model.forward_with_activations(x[:16])
    op = dcerif.make_downstream_operator(model, acts, 0)
    v = torch.randn((op.basis_dim, op.layer_output_dim), device=device, dtype=torch.float64, generator=gen)
    yy = torch.randn((op.sample_count, op.output_dim), device=device, dtype=torch.float64, generator=gen)
    lhs = float((op.apply_H(v) * yy).sum().detach().cpu().item())
    rhs_adj = float((v * op.apply_HT(yy)).sum().detach().cpu().item())
    adj_err = abs(lhs - rhs_adj) / max(abs(lhs), abs(rhs_adj), 1.0e-12)
    shape_ok = int(tuple(op.apply_H(v).shape) == (op.sample_count, op.output_dim) and tuple(op.apply_HT(yy).shape) == (op.basis_dim, op.layer_output_dim))
    state = model_state(model)
    before = torch.cat([p.detach().reshape(-1).to(dtype=torch.float64) for p in model.parameters()])
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    load_model_state(model, state)
    after = torch.cat([p.detach().reshape(-1).to(dtype=torch.float64) for p in model.parameters()])
    restore_err = float((after - before).norm().detach().cpu().item())
    base_state = model_state(model)
    delta = torch.randn_like(model.coeffs[-1].detach()).to(dtype=model.coeffs[-1].dtype) * 0.001
    with torch.no_grad():
        model.coeffs[-1].add_(delta)
    load_model_state(model, base_state)
    rollback_err = float((model.coeffs[-1].detach() - base_state["coeffs.1"]).norm().detach().cpu().item())
    gate = int(
        compile_pass
        and import_pass
        and static_scan_pass
        and whiten_err <= 1.0e-6
        and fval(exact_diag.get("solve_residual")) <= 1.0e-5
        and fval(cg_diag.get("cg_residual")) <= 1.0e-4
        and adj_err <= 1.0e-4
        and shape_ok
        and restore_err <= 1.0e-7
    )
    blocker = "none" if gate else "solver_or_identity_audit_failed"
    summary = {
        "part": "A",
        "gate_pass": gate,
        "route": "PartAImplementationIdentityPass" if gate else "A_FailedIdentity",
        "dominant_blocker": blocker,
        "compile_pass": compile_pass,
        "import_pass": import_pass,
        "static_scan_pass": static_scan_pass,
        "static_scan_hits": static_hits,
        "compile_errors": compile_errors,
        "whiten_unwhiten_error": whiten_err,
        "exact_solve_residual": fval(exact_diag.get("solve_residual")),
        "cg_solve_residual": fval(cg_diag.get("cg_residual")),
        "cg_iterations": fval(cg_diag.get("cg_iterations")),
        "H_HT_adjoint_error": adj_err,
        "H_shape_consistency": shape_ok,
        "snapshot_restore_error": restore_err,
        "finite_step_rollback_error": rollback_err,
        "helper_module": rel(HELPER),
    }
    write_json(OUT_ROOT / "part_a_summary.json", summary)
    if not gate:
        next_actions("a", summary["route"], blocker, ["fix VJP/JVP adjoint, ridge, CG, rollback, or static scan false positive"], [rel(OUT_ROOT / "part_a_summary.json")])
    append_exec("Part A", command_text(sys.argv), "done", files=rel(OUT_ROOT / "part_a_summary.json"), gpu=str(device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part A implementation identity and solver audit", summary)
    return summary


def get_nested(path: Path, *keys: str, default: Any = "missing") -> Any:
    data = read_json(path)
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    vroot = ROOT / str(args.v23_07_root)
    pb = read_json(vroot / "part_b_frozen_residual_summary.json")
    pc = read_json(vroot / "part_c_summary.json")
    pd = read_json(vroot / "part_d_summary.json")
    final = read_json(vroot / "final_route.json")
    repair = read_json(ROOT / "results/v23_07_d_repair_lastlayer_train768_seed5/part_d_summary.json")
    pb_groups = pb.get("passing_scheme_groups") or []
    best_b = pb_groups[0] if pb_groups else {}
    pc_groups = pc.get("passing_scheme_groups") or []
    c5 = next((g for g in pc_groups if str(g.get("scheme")) == "C5_EFRF_Wpop_gauss_seidel"), pc_groups[0] if pc_groups else {})
    candidate_groups = pd.get("candidate_groups") or []
    all_layer_c2 = median(g.get("C2_coverage_improvement_median") for g in candidate_groups)
    repair_groups = read_rows(ROOT / "results/v23_07_d_repair_lastlayer_train768_seed5/part_d_full_positive_control_group_summary.csv")
    c1_repair = next((r for r in repair_groups if r.get("scheme") == "C1_EFRF_L2_last_layer"), {})
    fg_repair = next((r for r in repair_groups if r.get("scheme") == "D1_FunctionalGram_AdamW"), {})
    missing = []
    for name, data in [("part_b", pb), ("part_c", pc), ("part_d", pd), ("final", final)]:
        if not data:
            missing.append(name)
    summary = {
        "part": "B",
        "gate_pass": 1,
        "route": "PartBHistoryLockPass" if not missing else "PartBHistoryLockPassWithMissing",
        "dominant_blocker": "none" if not missing else "missing_v23_07_artifact_values",
        "v23_07_root": rel(vroot),
        "missing_artifacts": missing,
        "v23_07_part_b_frozen_pass": pb.get("gate_pass", "missing"),
        "v23_07_frozen_best_scheme": best_b.get("scheme", "missing"),
        "v23_07_EFRF_minus_gradient_guard_fit": best_b.get("EFRF_minus_gradient_guard_fit_median", "missing"),
        "v23_07_EFRF_minus_random_projected_guard_fit": best_b.get("EFRF_minus_random_projected_guard_fit_median", "missing"),
        "v23_07_one_step_pass": pc.get("gate_pass", "missing"),
        "v23_07_one_step_best_scheme": c5.get("scheme", "missing"),
        "v23_07_C5_actual_C2_coverage_delta": c5.get("actual_C2_coverage_delta_median", "missing"),
        "v23_07_C5_random_projected_gap": c5.get("random_projected_gap_median", "missing"),
        "v23_07_formal_part_d_route": pd.get("route", "missing"),
        "v23_07_all_layer_C2_median": all_layer_c2 if candidate_groups else "missing",
        "v23_07_last_layer_repair_C2_median": c1_repair.get("C2_coverage_improvement_median", "missing"),
        "v23_07_last_layer_vs_FunctionalGram_gap": (
            fval(c1_repair.get("C2_coverage_improvement_median")) - fval(fg_repair.get("C2_coverage_improvement_median"))
            if c1_repair and fg_repair
            else "missing"
        ),
        "v23_07_final_promotion_allowed": final.get("promotion_allowed", "missing"),
        "history_interpretation": [
            "EFRF local residual inverse is locally valid.",
            "Naive all-layer local repeated EFRF is not validated.",
            "The next hypothesis is downstream-coupled residual inverse, not observer search.",
        ],
        "downstream_official_allowed": int(not missing),
    }
    write_json(OUT_ROOT / "part_b_history_lock_summary.json", summary)
    write_json(OUT_ROOT / "history_interpretation.json", {"part": "B", "statements": summary["history_interpretation"], "missing_artifacts": missing})
    append_exec("Part B", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'part_b_history_lock_summary.json')}; {rel(OUT_ROOT / 'history_interpretation.json')}", gpu=str(args.device), note=f"missing={missing}")
    append_recap("Part B v23.07 evidence lock", summary)
    return summary


def toy_row(name: str, phi: torch.Tensor, j: torch.Tensor, r: torch.Tensor, lam: float, seed: int) -> dict[str, Any]:
    device = phi.device
    p = int(phi.shape[1])
    g = torch.eye(p, device=device, dtype=torch.float64)
    local_normal = phi.T @ phi + float(lam) * g
    local_rhs = phi.T @ (j.T @ r)
    local, ldiag = dcerif.solve_ridge_normal_exact(local_normal, local_rhs)
    h = j @ phi
    d_normal = h.T @ h + float(lam) * g
    d_rhs = h.T @ r
    down, ddiag = dcerif.solve_ridge_normal_exact(d_normal, d_rhs)
    grad = phi.T @ (j.T @ r)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn((int(r.shape[0]), min(int(r.shape[0]), max(1, int(r.shape[0]) // 3))), device=device, dtype=torch.float64, generator=gen))
    rrand = q @ (q.T @ r)
    rand, rdiag = dcerif.solve_ridge_normal_exact(d_normal, h.T @ rrand)
    before = float(r.norm().detach().cpu().item())
    local_after = float((r - h @ local).norm().detach().cpu().item())
    down_after = float((r - h @ down).norm().detach().cpu().item())
    grad_after = float((r - h @ (0.05 * grad)).norm().detach().cpu().item())
    rand_after = float((r - h @ rand).norm().detach().cpu().item())
    local_gain = before - local_after
    down_gain = before - down_after
    rand_gain = before - rand_after
    _mn_h, _mx_h, cond_h = eig_condition(d_normal)
    _mn_p, _mx_p, cond_p = eig_condition(local_normal)
    lhs = float(((h @ down) * r).sum().detach().cpu().item())
    rhs = float((down * (h.T @ r)).sum().detach().cpu().item())
    adj = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-12)
    return {
        "toy": name,
        "output_residual_norm_before": before,
        "local_solve_output_residual_norm_after": local_after,
        "downstream_solve_output_residual_norm_after": down_after,
        "gradient_output_residual_norm_after": grad_after,
        "random_projected_output_residual_norm_after": rand_after,
        "local_vs_downstream_gain_gap": down_gain - local_gain,
        "solve_residual": ddiag.get("solve_residual"),
        "normal_equation_residual": ddiag.get("solve_residual"),
        "condition_number_HtWH": cond_h,
        "condition_number_PhiTPhi": cond_p,
        "rank_H": int(torch.linalg.matrix_rank(h, tol=1.0e-8).detach().cpu().item()),
        "rank_Phi": int(torch.linalg.matrix_rank(phi, tol=1.0e-8).detach().cpu().item()),
        "G_norm_delta": float((down.T @ g @ down).clamp_min(0.0).sqrt().detach().cpu().item()),
        "source_guard_gap": 0.0,
        "H_HT_adjoint_error": adj,
        "downstream_gain": down_gain,
        "random_gain": rand_gain,
    }


def part_c(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    device = device_from_args(args)
    gen = torch.Generator(device=device).manual_seed(230801)
    rows: list[dict[str, Any]] = []
    phi = torch.randn((48, 14), device=device, dtype=torch.float64, generator=gen)
    j = torch.randn((18, 48), device=device, dtype=torch.float64, generator=gen)
    r = torch.randn((18, 1), device=device, dtype=torch.float64, generator=gen)
    rows.append(toy_row("linear_downstream_mixing", phi, j, r, float(args.part_c_lambda), 1))
    x, y, _xg, _yg = v2307.visual_data("rotation_sensitive", 1, args, device)
    model = v2307.make_model(str(args.basis_key), 2, int(x.shape[1]), int(y.max().item()) + 1, 230802, args, device, basis_input_gain=0.25)
    logits, acts = model.forward_with_activations(x[:32])
    op = dcerif.make_downstream_operator(model, acts, 0)
    h = op.explicit_matrix()
    phi2 = torch.eye(int(h.shape[1]), device=device, dtype=torch.float64)
    j2 = h
    _log, rr = output_residual(model, x[:32], y[:32], "norm")
    rows.append(toy_row("nonlinear_tiny_purekan_downstream", phi2, j2, rr.reshape(-1, 1), float(args.part_c_lambda), 2))
    base = torch.randn((36, 10), device=device, dtype=torch.float64, generator=gen)
    phi3 = base.clone()
    phi3[:, 5:] = phi3[:, :5]
    j3 = torch.randn((20, 36), device=device, dtype=torch.float64, generator=gen)
    r3 = torch.randn((20, 1), device=device, dtype=torch.float64, generator=gen)
    rows.append(toy_row("rank_deficient_activation", phi3, j3, r3, max(float(args.part_c_lambda), 1.0e-1), 3))
    matrix = write_rows(OUT_ROOT / "part_c_toy_compositional_inverse_matrix.csv", rows)
    pass_rows = []
    for row in rows:
        before = fval(row.get("output_residual_norm_before"))
        local_after = fval(row.get("local_solve_output_residual_norm_after"))
        down_after = fval(row.get("downstream_solve_output_residual_norm_after"))
        grad_after = fval(row.get("gradient_output_residual_norm_after"))
        down_gain = fval(row.get("downstream_gain"))
        rand_gain = fval(row.get("random_gain"))
        random_not_close = int(rand_gain < 0.80 * down_gain) if down_gain > 0 else 0
        passed = int(
            down_after <= 0.75 * local_after
            and down_after <= 0.75 * grad_after
            and fval(row.get("local_vs_downstream_gain_gap")) >= 0.10
            and random_not_close
            and math.isfinite(fval(row.get("condition_number_HtWH"), float("inf")))
        )
        row["toy_gate_pass"] = passed
        pass_rows.append(passed)
    write_rows(matrix, rows)
    gate = int(sum(pass_rows) >= 2)
    blocker = "none" if gate else "downstream_not_better_than_local_or_random_close"
    summary = {
        "part": "C",
        "gate_pass": gate,
        "route": "PartCToyCompositionalInversePass" if gate else "C_DownstreamSolverToyFailed",
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "passing_toy_families": sum(pass_rows),
        "matrix": rel(matrix),
        "toy_gate_passes": pass_rows,
    }
    write_json(OUT_ROOT / "part_c_summary.json", summary)
    if not gate:
        next_actions("c", summary["route"], blocker, ["construct non-identity downstream mixing", "increase ridge for rank-deficient toy", "inspect random target alignment"], [rel(matrix)])
    append_exec("Part C", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_c_summary.json')}", gpu=str(device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part C toy compositional inverse", summary)
    return summary


def d_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, int, str]]:
    return [
        (scheme, checkpoint, task, seed, target)
        for scheme in csv_items(args.part_d_schemes)
        for checkpoint in csv_items(args.part_d_checkpoints)
        for task in csv_items(args.part_d_tasks)
        for seed in range(int(args.part_d_seed_count))
        for target in csv_items(args.part_d_residual_targets)
    ]


def run_d_row(job: tuple[str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, checkpoint, task, seed, target = job
    start = time.time()
    try:
        x, y, xg, yg = v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 23081000 + int(seed) * 1009 + sum(ord(c) for c in scheme + checkpoint + task)
        model = v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        ckpt = v2307.train_checkpoint(model, x, y, checkpoint, int(seed), args)
        logits_s, acts_s = model.forward_with_activations(x)
        logits_g, acts_g = model.forward_with_activations(xg)
        _ls, rs = output_residual(model, x, y, target)
        _lg, rg = output_residual(model, xg, yg, target)
        last = len(model.coeffs) - 1
        pen = max(0, last - 1)
        deltas: dict[int, torch.Tensor] = {}
        diag: dict[str, Any] = {"solve_residual": 0.0, "condition_number_after_ridge": 0.0, "rank_H": 0}
        source_op: dcerif.DownstreamOperator | None = None
        guard_op: dcerif.DownstreamOperator | None = None
        residual_for_fit = rs
        layer_group = "none"
        solver_type = "exact"
        sketch_rank = 0
        if scheme == "D9_same_compute_noop":
            layer_group = "noop"
        elif scheme == "D0_gradient_adjoint":
            layer_group = "penultimate_layer"
            delta, source_op, residual_for_fit, d0 = solve_downstream(model, x, y, pen, target, args, device, seed=int(seed))
            delta = source_op.apply_HT(residual_for_fit)
            deltas[pen] = 0.05 * delta
            diag.update(d0)
        elif scheme == "D1_local_EFRF":
            layer_group = "all_layers_local_control"
            _acts, hidden_resids, _logits, _loss = v2307.residual_bundle(model, x, y, "norm" if str(target).startswith("norm") else "act")
            delta, ldiag = solve_local_hidden(model, _acts, hidden_resids[pen], pen, args, device)
            deltas[pen] = delta
            source_op = dcerif.make_downstream_operator(model, acts_s, pen)
            residual_for_fit = rs
            diag.update(ldiag)
            diag["rank_H"] = float(torch.linalg.matrix_rank(source_op.explicit_matrix(), tol=1.0e-8).detach().cpu().item())
        elif scheme == "D2_last_layer_exact_EFRF":
            layer_group = "last_layer"
            delta, source_op, residual_for_fit, diag = solve_downstream(model, x, y, last, target, args, device, seed=int(seed))
            deltas[last] = delta
        elif scheme == "D3_penultimate_downstream_exact_or_CG":
            layer_group = "penultimate_layer"
            solver_type = str(args.part_d_solver)
            delta, source_op, residual_for_fit, diag = solve_downstream(model, x, y, pen, target, args, device, seed=int(seed))
            deltas[pen] = delta
        elif scheme == "D4_topdown_last_then_penultimate_refresh":
            layer_group = "top_down_last_then_penultimate"
            delta_last, op_last, rr, diag_last = solve_downstream(model, x, y, last, target, args, device, seed=int(seed))
            pred_last = op_last.apply_H(delta_last)
            leftover = rr - pred_last.detach()
            delta_pen, source_op, residual_for_fit, diag_pen = solve_downstream(model, x, y, pen, target, args, device, residual_override=leftover, seed=int(seed) + 17)
            deltas[last] = delta_last
            deltas[pen] = delta_pen
            residual_for_fit = rr
            diag = {**diag_last, "penultimate_solve_residual": diag_pen.get("solve_residual"), "rank_H": diag_pen.get("rank_H", 0)}
        elif scheme in {"D5_hidden_downstream_sketch_rank8", "D6_hidden_downstream_sketch_rank16"}:
            layer_group = "all_layers_downstream_sketch_diagnostic"
            sketch_rank = 8 if "rank8" in scheme else 16
            delta, source_op, residual_for_fit, diag = solve_downstream(model, x, y, pen, target, args, device, sketch_rank=sketch_rank, seed=int(seed))
            deltas[pen] = delta
        elif scheme == "D7_random_projected_downstream_EFRF":
            layer_group = "penultimate_random_projected"
            delta, source_op, residual_for_fit, diag = solve_downstream(model, x, y, pen, target, args, device, random_project=True, seed=int(seed))
            deltas[pen] = delta
        elif scheme == "D8_shuffled_design_downstream_EFRF":
            layer_group = "penultimate_shuffled_design"
            delta, source_op, residual_for_fit, diag = solve_downstream(model, x, y, pen, target, args, device, shuffle_design=True, seed=int(seed))
            deltas[pen] = delta
        else:
            raise ValueError(f"unknown scheme {scheme}")
        if source_op is None:
            source_fit = {"fit_before": float(rs.square().mean().detach().cpu().item()), "fit_after": float(rs.square().mean().detach().cpu().item()), "fit_improvement_ratio": 0.0, "output_alignment_cosine": 0.0}
            guard_fit = {"fit_before": float(rg.square().mean().detach().cpu().item()), "fit_after": float(rg.square().mean().detach().cpu().item()), "fit_improvement_ratio": 0.0, "output_alignment_cosine": 0.0}
            pred_source = torch.zeros_like(rs)
            pred_guard = torch.zeros_like(rg)
        else:
            pred_source = torch.zeros_like(rs)
            pred_guard = torch.zeros_like(rg)
            for layer_id, delta in deltas.items():
                op_s = dcerif.make_downstream_operator(model, acts_s, int(layer_id))
                op_g = dcerif.make_downstream_operator(model, acts_g, int(layer_id))
                pred_source = pred_source + op_s.apply_H(delta)
                pred_guard = pred_guard + op_g.apply_H(delta)
            before_s = rs.square().mean().clamp_min(1.0e-12)
            after_s = (rs - pred_source).square().mean()
            source_fit = {
                "fit_before": float(before_s.detach().cpu().item()),
                "fit_after": float(after_s.detach().cpu().item()),
                "fit_improvement_ratio": float(((before_s - after_s) / before_s).detach().cpu().item()),
                "output_alignment_cosine": float(((pred_source.reshape(-1) @ rs.reshape(-1)) / (pred_source.norm() * rs.norm()).clamp_min(1.0e-12)).detach().cpu().item()),
            }
            before_g = rg.square().mean().clamp_min(1.0e-12)
            after_g = (rg - pred_guard).square().mean()
            guard_fit = {
                "fit_before": float(before_g.detach().cpu().item()),
                "fit_after": float(after_g.detach().cpu().item()),
                "fit_improvement_ratio": float(((before_g - after_g) / before_g).detach().cpu().item()),
                "output_alignment_cosine": float(((pred_guard.reshape(-1) @ rg.reshape(-1)) / (pred_guard.norm() * rg.norm()).clamp_min(1.0e-12)).detach().cpu().item()),
            }
        state = model_state(model)
        for lid, delta in deltas.items():
            apply_delta(model, int(lid), delta, 1.0)
        logits_after = model(x).detach().to(dtype=torch.float64)
        output_after = float((rs - (logits_after - logits_s.detach()).to(dtype=torch.float64)).norm().detach().cpu().item())
        load_model_state(model, state)
        param_norm = math.sqrt(sum(float(d.square().sum().detach().cpu().item()) for d in deltas.values())) if deltas else 0.0
        g_norm = param_norm
        rank_phi = 0
        if source_op is not None:
            rank_phi = int(torch.linalg.matrix_rank(source_op.phi, tol=1.0e-8).detach().cpu().item())
        return {
            "status": "ok",
            "part": "D",
            "scheme": scheme,
            "candidate_downstream": int(scheme in DOWNSTREAM_D_CANDIDATES),
            "checkpoint_name": checkpoint,
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "layer_group": layer_group,
            "basis_key": str(args.basis_key),
            "depth": int(args.depth),
            "task": task,
            "seed": int(seed),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "residual_target_type": target,
            "lambda": float(args.part_d_lambda),
            "solver_type": solver_type,
            "sketch_rank": int(sketch_rank),
            "cg_iterations": fval(diag.get("cg_iterations")),
            "source_residual_fit_before": source_fit["fit_before"],
            "source_residual_fit_after": source_fit["fit_after"],
            "guard_residual_fit_before": guard_fit["fit_before"],
            "guard_residual_fit_after": guard_fit["fit_after"],
            "source_fit_improvement_ratio": source_fit["fit_improvement_ratio"],
            "guard_fit_improvement_ratio": guard_fit["fit_improvement_ratio"],
            "output_alignment_cosine": guard_fit["output_alignment_cosine"],
            "output_residual_norm_after_actual_forward": output_after,
            "G_edge_norm_delta": g_norm,
            "parameter_norm_delta": param_norm,
            "condition_number_after_ridge": fval(diag.get("condition_number_after_ridge"), float("inf")),
            "rank_H": fval(diag.get("rank_H")),
            "rank_Phi": rank_phi,
            "design_rank_deficient_flag": int(rank_phi == 0),
            "solve_residual": fval(diag.get("solve_residual")),
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {"status": "error", "part": "D", "scheme": scheme, "checkpoint_name": checkpoint, "task": task, "seed": int(seed), "residual_target_type": target, "error_message": repr(exc), "wall_time_s": time.time() - start}


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    pb = read_json(OUT_ROOT / "part_b_history_lock_summary.json")
    pc = read_json(OUT_ROOT / "part_c_summary.json")
    if not ival(pa.get("gate_pass")):
        return blocked_summary("D", "D_BlockedByPartA", str(pa.get("dominant_blocker", "part_a_missing_or_failed")))
    if not ival(pb.get("gate_pass")):
        return blocked_summary("D", "D_BlockedByPartB", str(pb.get("dominant_blocker", "part_b_missing_or_failed")))
    if not ival(pc.get("gate_pass")):
        return blocked_summary("D", "D_BlockedByPartC", str(pc.get("dominant_blocker", "part_c_missing_or_failed")))
    device = device_from_args(args)
    jobs = shard_items(d_jobs(args), args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"part_d_frozen_downstream_residual_matrix{suffix}.csv"
    rows: list[dict[str, Any]] = []
    existing = read_rows(matrix) if int(args.part_d_resume) else []
    done = {(r.get("scheme"), r.get("checkpoint_name"), r.get("task"), r.get("seed"), r.get("residual_target_type")) for r in existing if r.get("status") == "ok"}
    for job in jobs:
        key = (job[0], job[1], job[2], str(job[3]), job[4])
        if key in done:
            continue
        row = run_d_row(job, args, device)
        rows.append(row)
        if int(args.part_d_flush_every) and len(rows) % int(args.part_d_flush_every) == 0:
            append_rows(matrix, rows)
            print(json.dumps({"part": "D", "rows_written": len(read_rows(matrix)), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
            rows = []
    if rows:
        append_rows(matrix, rows)
    all_rows = read_rows(matrix)
    summary = {
        "part": "D",
        "gate_pass": 0,
        "route": "PartDShardOnly" if suffix else "PartDNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(all_rows),
        "ok_rows": sum(1 for r in all_rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in all_rows if r.get("status") != "ok"),
        "matrix": rel(matrix),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_d_resume),
    }
    write_json(OUT_ROOT / f"part_d_summary{suffix}.json", summary)
    append_exec("Part D", command_text(sys.argv), "done", files=rel(matrix), gpu=str(device), note=f"rows={len(all_rows)}")
    append_recap("Part D frozen downstream residual shard", summary)
    return summary


def part_d_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_d_frozen_downstream_residual_matrix_shard*_of_*.csv"))
    rows = [r for path in shard_paths for r in read_rows(path)] if shard_paths else read_rows(OUT_ROOT / "part_d_frozen_downstream_residual_matrix.csv")
    matrix = write_rows(OUT_ROOT / "part_d_frozen_downstream_residual_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    fg: dict[tuple[str, str, str, str], float] = {}
    local: dict[tuple[str, str, str, str], float] = {}
    rand: dict[tuple[str, str, str, str], float] = {}
    shuf: dict[tuple[str, str, str, str], float] = {}
    grad: dict[tuple[str, str, str, str], float] = {}
    for r in ok:
        key = (r.get("checkpoint_name", ""), r.get("task", ""), r.get("seed", ""), r.get("residual_target_type", ""))
        val = fval(r.get("guard_fit_improvement_ratio"))
        if r.get("scheme") == "D1_local_EFRF":
            local[key] = val
        elif r.get("scheme") == "D7_random_projected_downstream_EFRF":
            rand[key] = val
        elif r.get("scheme") == "D8_shuffled_design_downstream_EFRF":
            shuf[key] = val
        elif r.get("scheme") == "D0_gradient_adjoint":
            grad[key] = val
    enriched_ok: list[dict[str, Any]] = []
    all_enriched: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "ok":
            all_enriched.append(dict(r))
            continue
        key = (r.get("checkpoint_name", ""), r.get("task", ""), r.get("seed", ""), r.get("residual_target_type", ""))
        e = dict(r)
        e["EFRF_minus_gradient_guard_fit"] = fval(r.get("guard_fit_improvement_ratio")) - grad.get(key, 0.0)
        e["EFRF_minus_local_guard_fit"] = fval(r.get("guard_fit_improvement_ratio")) - local.get(key, 0.0)
        e["EFRF_minus_random_projected_guard_fit"] = fval(r.get("guard_fit_improvement_ratio")) - rand.get(key, 0.0)
        e["EFRF_minus_shuffled_design_guard_fit"] = fval(r.get("guard_fit_improvement_ratio")) - shuf.get(key, 0.0)
        e["source_guard_fit_gap"] = fval(r.get("source_fit_improvement_ratio")) - fval(r.get("guard_fit_improvement_ratio"))
        enriched_ok.append(e)
        all_enriched.append(e)
    matrix = write_rows(OUT_ROOT / "part_d_frozen_downstream_residual_matrix.csv", all_enriched)
    groups: list[dict[str, Any]] = []
    group_keys = sorted({(r.get("scheme"), r.get("checkpoint_name"), r.get("residual_target_type"), r.get("layer_group")) for r in enriched_ok})
    for key in group_keys:
        group = [r for r in enriched_ok if (r.get("scheme"), r.get("checkpoint_name"), r.get("residual_target_type"), r.get("layer_group")) == key]
        scheme = str(key[0])
        candidate = scheme in DOWNSTREAM_D_CANDIDATES
        guard = median(r.get("guard_fit_improvement_ratio") for r in group)
        grad_gap = median(r.get("EFRF_minus_gradient_guard_fit") for r in group)
        local_gap = median(r.get("EFRF_minus_local_guard_fit") for r in group)
        rand_gap = median(r.get("EFRF_minus_random_projected_guard_fit") for r in group)
        shuf_gap = median(r.get("EFRF_minus_shuffled_design_guard_fit") for r in group)
        sg_gap = median(r.get("source_guard_fit_gap") for r in group)
        cond = median(r.get("condition_number_after_ridge") for r in group)
        solve = median(r.get("solve_residual") for r in group)
        rank_h = median(r.get("rank_H") for r in group)
        stable = int(cond <= 1.0e6 or solve <= 1.0e-3)
        pass_gate = int(
            candidate
            and guard >= 0.15
            and grad_gap >= 0.10
            and local_gap >= 0.05
            and rand_gap >= 0.10
            and shuf_gap >= 0.10
            and sg_gap <= 0.20
            and stable
            and rank_h > 0
        )
        groups.append(
            {
                "scheme": key[0],
                "checkpoint_name": key[1],
                "residual_target_type": key[2],
                "layer_group": key[3],
                "rows": len(group),
                "candidate_downstream": int(candidate),
                "guard_fit_improvement_ratio_median": guard,
                "source_fit_improvement_ratio_median": median(r.get("source_fit_improvement_ratio") for r in group),
                "EFRF_minus_gradient_guard_fit_median": grad_gap,
                "EFRF_minus_local_guard_fit_median": local_gap,
                "EFRF_minus_random_projected_guard_fit_median": rand_gap,
                "EFRF_minus_shuffled_design_guard_fit_median": shuf_gap,
                "source_guard_fit_gap_median": sg_gap,
                "condition_number_after_ridge_median": cond,
                "solve_residual_median": solve,
                "rank_H_median": rank_h,
                "rank_Phi_median": median(r.get("rank_Phi") for r in group),
                "wall_time_s_median": median(r.get("wall_time_s") for r in group),
                "scheme_gate_pass": pass_gate,
            }
        )
    group_csv = write_rows(OUT_ROOT / "part_d_frozen_downstream_residual_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("scheme_gate_pass")) == 1]
    last_groups = [g for g in groups if str(g.get("scheme")) == "D2_last_layer_exact_EFRF"]
    last_ok = any(fval(g.get("guard_fit_improvement_ratio_median")) >= 0.15 and fval(g.get("EFRF_minus_random_projected_guard_fit_median")) >= 0.10 for g in last_groups)
    candidate_groups = [g for g in groups if ival(g.get("candidate_downstream")) == 1]
    if pass_groups:
        gate, route, blocker = 1, "PartDFrozenDownstreamResidualPass", "none"
    else:
        gate = 0
        if not last_ok:
            route, blocker = "LastLayerEFRFFailed", "last_layer_sanity_failed"
        elif any(fval(g.get("EFRF_minus_random_projected_guard_fit_median")) < 0.10 for g in candidate_groups):
            route, blocker = "D_RandomProjectedAssimilation", "random_projected_gap_low"
        elif any(fval(g.get("EFRF_minus_local_guard_fit_median")) < 0.05 for g in candidate_groups):
            route, blocker = "D_FrozenDownstreamResidualFailed", "downstream_not_better_than_local"
        else:
            route, blocker = "D_FrozenDownstreamResidualFailed", "no_fixed_downstream_scheme_passed"
    summary = {
        "part": "D",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
        "passing_scheme_groups": pass_groups,
        "last_layer_sanity_pass": int(last_ok),
        "candidate_groups": candidate_groups,
    }
    write_json(OUT_ROOT / "part_d_summary.json", summary)
    failed_candidate_groups = []
    for group in candidate_groups:
        reasons = []
        if fval(group.get("guard_fit_improvement_ratio_median")) < 0.15:
            reasons.append("guard_fit_low")
        if fval(group.get("EFRF_minus_gradient_guard_fit_median")) < 0.10:
            reasons.append("gradient_gap_low")
        if fval(group.get("EFRF_minus_local_guard_fit_median")) < 0.05:
            reasons.append("local_gap_low")
        if fval(group.get("EFRF_minus_random_projected_guard_fit_median")) < 0.10:
            reasons.append("random_projected_gap_low")
        if fval(group.get("EFRF_minus_shuffled_design_guard_fit_median")) < 0.10:
            reasons.append("shuffled_design_gap_low")
        if fval(group.get("source_guard_fit_gap_median")) > 0.20:
            reasons.append("source_guard_gap_high")
        if fval(group.get("condition_number_after_ridge_median"), float("inf")) > 1.0e6 and fval(group.get("solve_residual_median"), float("inf")) > 1.0e-3:
            reasons.append("solver_unstable")
        if fval(group.get("rank_H_median")) <= 0:
            reasons.append("rank_H_zero")
        if reasons:
            failed_candidate_groups.append({"scheme": group.get("scheme"), "checkpoint_name": group.get("checkpoint_name"), "residual_target_type": group.get("residual_target_type"), "reasons": reasons})
    failure_path = write_failure_decomposition(
        "d",
        {
            "gate_pass": gate,
            "route": route,
            "dominant_blocker": blocker,
            "passing_scheme_count": len(pass_groups),
            "passing_scheme_groups": pass_groups,
            "failed_candidate_groups": failed_candidate_groups,
            "last_layer_sanity_pass": int(last_ok),
            "error_rows": len(rows) - len(ok),
        },
    )
    allowed = [] if gate else ["check residual target shape/sign", "increase source/guard size", "increase ridge or sketch rank", "inspect H/H^T VJP/JVP", "add same-compute/shuffled controls before promotion"]
    next_actions("d", route, blocker, allowed, [rel(matrix), rel(group_csv), rel(failure_path)])
    append_exec("Part D merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_d_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part D frozen downstream residual merge", summary)
    return summary


def e_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, int, str]]:
    return [
        (scheme, checkpoint, task, seed, target)
        for scheme in csv_items(args.part_e_schemes)
        for checkpoint in csv_items(args.part_e_checkpoints)
        for task in csv_items(args.part_e_tasks)
        for seed in range(int(args.part_e_seed_count))
        for target in csv_items(args.part_e_residual_targets)
    ]


def scaled_deltas(deltas: dict[int, torch.Tensor], args: argparse.Namespace) -> dict[int, torch.Tensor]:
    scale = float(args.part_e_delta_scale) * float(args.part_e_delta_sign)
    return {int(layer): delta.to(dtype=torch.float64) * scale for layer, delta in deltas.items()}


def apply_deltas(model: Any, deltas: dict[int, torch.Tensor], alpha: float) -> None:
    for layer_id, delta in deltas.items():
        apply_delta(model, int(layer_id), delta, float(alpha))


def add_deltas(left: dict[int, torch.Tensor], right: dict[int, torch.Tensor], right_scale: float) -> dict[int, torch.Tensor]:
    out = {int(layer): delta.detach().clone().to(dtype=torch.float64) for layer, delta in left.items()}
    for layer, delta in right.items():
        key = int(layer)
        term = delta.to(dtype=torch.float64) * float(right_scale)
        out[key] = out.get(key, torch.zeros_like(term)) + term
    return out


def predicted_guard_delta(model: Any, xg: torch.Tensor, deltas: dict[int, torch.Tensor]) -> torch.Tensor:
    logits, acts = model.forward_with_activations(xg)
    pred = torch.zeros_like(logits.detach(), dtype=torch.float64)
    for layer_id, delta in deltas.items():
        op = dcerif.make_downstream_operator(model, acts, int(layer_id))
        pred = pred + op.apply_H(delta.to(device=logits.device, dtype=torch.float64))
    return pred.detach()


def e_delta_candidate(
    scheme: str,
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    target: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
    task: str,
    checkpoint: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    eargs = argparse.Namespace(**vars(args))
    eargs.part_d_lambda = float(args.part_e_lambda)
    last = len(model.coeffs) - 1
    pen = max(0, last - 1)
    diag: dict[str, Any] = {
        "layer_group": "none",
        "solver_type": "none",
        "sketch_rank": 0,
        "solve_residual_max": 0.0,
        "condition_number_after_ridge_max": 0.0,
        "rank_H_max": 0.0,
        "selected_layers": "",
        "predicted_guard_fit_mean": 0.0,
        "predicted_guard_loss_delta_sum": 0.0,
        "same_compute_candidate_built": 0,
    }
    if scheme == "E8_same_compute_noop_one_step":
        delta, _op, _rr, d0 = solve_downstream(model, x, y, pen, target, eargs, device, seed=int(seed))
        diag.update(
            {
                "layer_group": "same_compute_noop",
                "solver_type": str(args.part_d_solver),
                "selected_layers": str(pen),
                "same_compute_candidate_built": 1,
                "solve_residual_max": fval(d0.get("solve_residual")),
                "condition_number_after_ridge_max": fval(d0.get("condition_number_after_ridge")),
                "rank_H_max": fval(d0.get("rank_H")),
                "parameter_norm_delta": float(delta.norm().detach().cpu().item()),
            }
        )
        return {}, diag
    if scheme == "E1_v23_07_local_EFRF_one_step_C5_reference":
        dargs = argparse.Namespace(**vars(args))
        dargs.part_c_lambda = float(args.part_e_lambda)
        dargs.part_c_alphas = str(args.part_e_alphas)
        dargs.part_c_trust_loss_tolerance = float(args.part_e_trust_tolerance)
        deltas, local_diag = v2307.part_c_scheme_delta(
            model,
            x,
            y,
            xg,
            yg,
            str(args.part_e_local_reference_target),
            "C5_EFRF_Wpop_gauss_seidel",
            dargs,
            device,
            seed=int(seed),
            task=task,
            checkpoint=checkpoint,
        )
        diag.update(
            {
                "layer_group": "v23_07_local_all_layer_GS_reference",
                "solver_type": "v23_07_local_exact",
                "selected_layers": local_diag.get("selected_layers", ""),
                "solve_residual_max": fval(local_diag.get("solve_residual_max")),
                "condition_number_after_ridge_max": fval(local_diag.get("condition_number_after_ridge_max")),
                "predicted_guard_fit_mean": fval(local_diag.get("predicted_guard_fit_mean")),
                "predicted_guard_loss_delta_sum": fval(local_diag.get("predicted_guard_loss_delta_sum")),
            }
        )
        return {int(k): v for k, v in deltas.items()}, diag
    if scheme == "E2_last_layer_exact_EFRF_one_step":
        delta, _op, _rr, d0 = solve_downstream(model, x, y, last, target, eargs, device, seed=int(seed))
        deltas = {last: delta}
        diag.update({"layer_group": "last_layer", "solver_type": str(args.part_d_solver), "selected_layers": str(last), "solve_residual_max": fval(d0.get("solve_residual")), "condition_number_after_ridge_max": fval(d0.get("condition_number_after_ridge")), "rank_H_max": fval(d0.get("rank_H"))})
        return deltas, diag
    if scheme == "E3_penultimate_downstream_EFRF_one_step":
        delta, _op, _rr, d0 = solve_downstream(model, x, y, pen, target, eargs, device, seed=int(seed))
        deltas = {pen: delta}
        diag.update({"layer_group": "penultimate_layer", "solver_type": str(args.part_d_solver), "selected_layers": str(pen), "solve_residual_max": fval(d0.get("solve_residual")), "condition_number_after_ridge_max": fval(d0.get("condition_number_after_ridge")), "rank_H_max": fval(d0.get("rank_H"))})
        return deltas, diag
    if scheme == "E4_topdown_last_penultimate_one_step":
        delta_last, op_last, rr, dlast = solve_downstream(model, x, y, last, target, eargs, device, seed=int(seed))
        leftover = rr - op_last.apply_H(delta_last).detach()
        delta_pen, _op_pen, _rpen, dpen = solve_downstream(model, x, y, pen, target, eargs, device, residual_override=leftover, seed=int(seed) + 17)
        deltas = {last: delta_last, pen: delta_pen}
        diag.update(
            {
                "layer_group": "topdown_last_penultimate",
                "solver_type": str(args.part_d_solver),
                "selected_layers": f"{last},{pen}",
                "solve_residual_max": max(fval(dlast.get("solve_residual")), fval(dpen.get("solve_residual"))),
                "condition_number_after_ridge_max": max(fval(dlast.get("condition_number_after_ridge")), fval(dpen.get("condition_number_after_ridge"))),
                "rank_H_max": max(fval(dlast.get("rank_H")), fval(dpen.get("rank_H"))),
            }
        )
        return deltas, diag
    if scheme == "E9_topdown_last_penultimate_adjoint_local_one_step":
        _logits, acts = model.forward_with_activations(x)
        delta_last, op_last, rr, dlast = solve_downstream(model, x, y, last, target, eargs, device, seed=int(seed))
        leftover = rr - op_last.apply_H(delta_last).detach()
        hidden_target = downstream_adjoint_hidden_target(model, acts[pen + 1], pen + 1, leftover)
        delta_pen, dpen = solve_local_hidden(model, acts, hidden_target, pen, eargs, device)
        deltas = {last: delta_last, pen: delta_pen}
        diag.update(
            {
                "layer_group": "topdown_last_penultimate_adjoint_local",
                "solver_type": "last_exact_plus_adjoint_local",
                "selected_layers": f"{last},{pen}",
                "solve_residual_max": max(fval(dlast.get("solve_residual")), fval(dpen.get("solve_residual"))),
                "condition_number_after_ridge_max": max(fval(dlast.get("condition_number_after_ridge")), fval(dpen.get("condition_number_after_ridge"))),
                "rank_H_max": fval(dlast.get("rank_H")),
                "adjoint_hidden_target_norm": float(hidden_target.norm().detach().cpu().item()),
            }
        )
        return deltas, diag
    if scheme == "E5_hidden_downstream_sketch_one_step":
        rank = int(args.part_e_sketch_rank)
        delta, _op, _rr, d0 = solve_downstream(model, x, y, pen, target, eargs, device, sketch_rank=rank, seed=int(seed))
        deltas = {pen: delta}
        diag.update({"layer_group": "penultimate_sketch", "solver_type": "sketch_exact", "sketch_rank": rank, "selected_layers": str(pen), "solve_residual_max": fval(d0.get("solve_residual")), "condition_number_after_ridge_max": fval(d0.get("condition_number_after_ridge")), "rank_H_max": fval(d0.get("rank_H"))})
        return deltas, diag
    if scheme == "E6_random_projected_downstream_one_step":
        delta, _op, _rr, d0 = solve_downstream(model, x, y, pen, target, eargs, device, random_project=True, seed=int(seed))
        deltas = {pen: delta}
        diag.update({"layer_group": "penultimate_random_projected", "solver_type": str(args.part_d_solver), "selected_layers": str(pen), "solve_residual_max": fval(d0.get("solve_residual")), "condition_number_after_ridge_max": fval(d0.get("condition_number_after_ridge")), "rank_H_max": fval(d0.get("rank_H"))})
        return deltas, diag
    if scheme == "E7_shuffled_design_downstream_one_step":
        delta, _op, _rr, d0 = solve_downstream(model, x, y, pen, target, eargs, device, shuffle_design=True, seed=int(seed))
        deltas = {pen: delta}
        diag.update({"layer_group": "penultimate_shuffled_design", "solver_type": str(args.part_d_solver), "selected_layers": str(pen), "solve_residual_max": fval(d0.get("solve_residual")), "condition_number_after_ridge_max": fval(d0.get("condition_number_after_ridge")), "rank_H_max": fval(d0.get("rank_H"))})
        return deltas, diag
    raise ValueError(f"unknown Part E scheme {scheme}")


def trust_accept_e(source_before: dict[str, float], source_after: dict[str, float], guard_before: dict[str, float], guard_after: dict[str, float], args: argparse.Namespace) -> tuple[bool, str]:
    tol = float(args.part_e_trust_tolerance)
    mode = str(args.part_e_trust_mode).lower()
    if mode == "coverage_floor":
        ok = source_after["coverage"] >= source_before["coverage"] - tol and guard_after["coverage"] >= guard_before["coverage"] - tol
        return bool(ok), "coverage_floor_trust_reject"
    if mode in {"pareto", "loss_and_coverage", "loss_coverage"}:
        ok = (
            source_after["loss"] <= source_before["loss"] + tol
            and guard_after["loss"] <= guard_before["loss"] + tol
            and source_after["coverage"] >= source_before["coverage"] - tol
            and guard_after["coverage"] >= guard_before["coverage"] - tol
        )
        return bool(ok), "pareto_trust_reject"
    ok = source_after["loss"] <= source_before["loss"] + tol and guard_after["loss"] <= guard_before["loss"] + tol
    return bool(ok), "loss_trust_reject"


def tail99_step_accept(guard_before: dict[str, float], guard_after: dict[str, float], args: argparse.Namespace) -> bool:
    budget = float(getattr(args, "part_f_tail99_step_budget", -1.0))
    if budget < 0.0:
        return True
    return bool(guard_after["tail99"] <= guard_before["tail99"] + budget)


def tail99_cumulative_accept(guard_reference: dict[str, float] | None, guard_after: dict[str, float], args: argparse.Namespace) -> bool:
    budget = float(getattr(args, "part_f_tail99_cumulative_budget", -1.0))
    if budget < 0.0 or guard_reference is None:
        return True
    return bool(guard_after["tail99"] <= guard_reference["tail99"] + budget)


def choose_alpha_for_e_deltas(
    model: Any,
    base_state: dict[str, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    reject_reason = "no_alpha_candidates"
    if not deltas:
        return 0.0, 1, "same_compute_noop", source_before, guard_before
    for alpha in float_items(args.part_e_alphas):
        load_model_state(model, base_state)
        apply_deltas(model, deltas, float(alpha))
        source_after = actual_metrics(model, x, y)
        guard_after = actual_metrics(model, xg, yg)
        accepted, reject_reason = trust_accept_e(source_before, source_after, guard_before, guard_after, args)
        if accepted:
            return float(alpha), 1, "accepted", source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before


def max_activation_drift_to_base(model: Any, xg: torch.Tensor, base_acts: list[torch.Tensor]) -> float:
    _logits, acts = model.forward_with_activations(xg)
    drifts = [
        float((after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)).norm().div(before.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12)).detach().cpu().item())
        for before, after in zip(base_acts[1:], acts[1:])
    ]
    return max(drifts or [0.0])


def choose_alpha_for_f_deltas(
    model: Any,
    base_state: dict[str, torch.Tensor],
    deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    base_acts: list[torch.Tensor],
    tail99_reference: dict[str, float] | None = None,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    reject_reason = "no_alpha_candidates"
    if not deltas:
        return 0.0, 1, "same_compute_noop", source_before, guard_before
    drift_cap = float(getattr(args, "part_f_activation_drift_cap", -1.0))
    for alpha in float_items(args.part_e_alphas):
        load_model_state(model, base_state)
        apply_deltas(model, deltas, float(alpha))
        source_after = actual_metrics(model, x, y)
        guard_after = actual_metrics(model, xg, yg)
        accepted, reject_reason = trust_accept_e(source_before, source_after, guard_before, guard_after, args)
        if accepted and not tail99_step_accept(guard_before, guard_after, args):
            accepted, reject_reason = False, "tail99_step_trust_reject"
        if accepted and not tail99_cumulative_accept(tail99_reference, guard_after, args):
            accepted, reject_reason = False, "tail99_cumulative_trust_reject"
        if accepted and drift_cap > 0.0:
            drift = max_activation_drift_to_base(model, xg, base_acts)
            if drift > drift_cap:
                accepted, reject_reason = False, "activation_drift_cap_reject"
        if accepted:
            return float(alpha), 1, "accepted", source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before


def choose_tail_corrected_f_deltas(
    model: Any,
    base_state: dict[str, torch.Tensor],
    main_deltas: dict[int, torch.Tensor],
    tail_deltas: dict[int, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    base_acts: list[torch.Tensor],
    *,
    tail99_reference: dict[str, float] | None = None,
    beta_records: list[float] | None = None,
    transport_alignment_records: list[float] | None = None,
) -> tuple[float, int, str, dict[str, float], dict[str, float], dict[int, torch.Tensor], float]:
    reject_reason = "no_tail_correction_variant_accepted"
    min_align = float(getattr(args, "part_f_tail_correction_transport_min", 0.80))
    main_pred = predicted_guard_delta(model, xg, main_deltas)
    for beta in float_items(args.part_f_tail_correction_scales):
        deltas = add_deltas(main_deltas, tail_deltas, float(beta))
        combined_pred = predicted_guard_delta(model, xg, deltas)
        align = cosine_flat(combined_pred, main_pred)
        if transport_alignment_records is not None:
            transport_alignment_records.append(float(align))
        if align < min_align:
            reject_reason = "tail_correction_transport_alignment_reject"
            continue
        alpha, accepted, reason, source_after, guard_after = choose_alpha_for_f_deltas(
            model,
            base_state,
            deltas,
            source_before,
            guard_before,
            x,
            y,
            xg,
            yg,
            args,
            base_acts,
            tail99_reference=tail99_reference,
        )
        reject_reason = reason
        if accepted:
            if beta_records is not None:
                beta_records.append(float(beta))
            return alpha, 1, "accepted", source_after, guard_after, deltas, float(beta)
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before, main_deltas, 0.0


def choose_alpha_functionalgram_e(
    model: Any,
    base_state: dict[str, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    reject_reason = "no_alpha_candidates"
    for alpha in float_items(args.part_e_alphas):
        load_model_state(model, base_state)
        opt = EdgeSobolevPopulationFlow(
            list(model.coeffs),
            lr=float(alpha) * float(args.part_e_functionalgram_lr),
            weight_decay=float(args.weight_decay),
            sobolev_exponent=0.0,
            edge_metric_type="functional_gram",
            edge_weight_normalization="trace",
            edge_weight_ridge=float(args.functional_gram_ridge),
            functional_gram_quadrature_points=int(args.quadrature_points),
            use_population_gate=False,
        )
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y.long())
        loss.backward()
        opt.step()
        source_after = actual_metrics(model, x, y)
        guard_after = actual_metrics(model, xg, yg)
        accepted, reject_reason = trust_accept_e(source_before, source_after, guard_before, guard_after, args)
        if accepted:
            return float(alpha), 1, "accepted", source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before


def run_e_row(job: tuple[str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, checkpoint, task, seed, target = job
    start = time.time()
    try:
        x, y, xg, yg = v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 23082000 + int(seed) * 1009 + sum(ord(c) for c in scheme + checkpoint + task)
        model = v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        ckpt = v2307.train_checkpoint(model, x, y, checkpoint, int(seed), args)
        base_state = model_state(model)
        source_before = actual_metrics(model, x, y)
        guard_before = actual_metrics(model, xg, yg)
        logits_g_before, acts_g_before = model.forward_with_activations(xg)
        _logits_s_before, acts_s_before = model.forward_with_activations(x)
        _ls, rs = output_residual(model, x, y, target)
        _lg, rg = output_residual(model, xg, yg, target)
        diag: dict[str, Any] = {"layer_group": "optimizer", "solver_type": "optimizer", "sketch_rank": 0}
        raw_deltas: dict[int, torch.Tensor] = {}
        if scheme == "E0_FunctionalGram_one_step_baseline":
            alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_functionalgram_e(model, base_state, source_before, guard_before, x, y, xg, yg, args)
            selected_deltas: dict[int, torch.Tensor] = {}
        else:
            raw_deltas, diag = e_delta_candidate(scheme, model, x, y, xg, yg, target, args, device, seed=int(seed), task=task, checkpoint=checkpoint)
            selected_deltas = scaled_deltas(raw_deltas, args)
            alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_for_e_deltas(model, base_state, selected_deltas, source_before, guard_before, x, y, xg, yg, args)
        load_model_state(model, base_state)
        pred_before = torch.zeros_like(rg)
        for layer_id, delta in selected_deltas.items():
            op_before = dcerif.make_downstream_operator(model, acts_g_before, int(layer_id))
            pred_before = pred_before + op_before.apply_H(delta * float(alpha))
        apply_deltas(model, selected_deltas, alpha)
        source_after = actual_metrics(model, x, y)
        guard_after = actual_metrics(model, xg, yg)
        logits_g_after, acts_g_after = model.forward_with_activations(xg)
        actual_output_delta = (logits_g_after.detach() - logits_g_before.detach()).to(dtype=torch.float64)
        activation_drifts = []
        for before, after in zip(acts_g_before[1:], acts_g_after[1:]):
            activation_drifts.append(float((after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)).norm().div(before.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12)).detach().cpu().item()))
        pred_after = torch.zeros_like(rg)
        for layer_id, delta in selected_deltas.items():
            op_after = dcerif.make_downstream_operator(model, acts_g_after, int(layer_id))
            step_delta = delta * float(alpha)
            pred_after = pred_after + op_after.apply_H(step_delta)
        j_drift = 0.0 if not selected_deltas else 1.0 - cosine_flat(pred_before, pred_after)
        source_guard_gap = (source_after["coverage"] - source_before["coverage"]) - (guard_after["coverage"] - guard_before["coverage"])
        param_norm = math.sqrt(sum(float((delta * float(alpha)).square().sum().detach().cpu().item()) for delta in selected_deltas.values())) if selected_deltas else 0.0
        debt = debt_deltas(guard_before, guard_after)
        row = {
            "part": "E",
            "status": "ok",
            "scheme": scheme,
            "candidate_downstream": int(scheme in DOWNSTREAM_E_CANDIDATES),
            "candidate_last_layer": int(scheme in LAST_LAYER_E_CANDIDATES),
            "checkpoint_name": checkpoint,
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "basis_key": str(args.basis_key),
            "depth": int(args.depth),
            "task": task,
            "seed": int(seed),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "residual_target_type": target,
            "local_reference_target": str(args.part_e_local_reference_target),
            "lambda": float(args.part_e_lambda),
            "delta_scale": float(args.part_e_delta_scale),
            "delta_sign": float(args.part_e_delta_sign),
            "trust_mode": str(args.part_e_trust_mode),
            "alpha_selected": float(alpha),
            "finite_step_accepted": int(accepted),
            "finite_step_accept_rate": float(accepted),
            "finite_step_scale_mean": float(alpha),
            "finite_step_skip_count": int(not accepted),
            "finite_step_reject_reasons": json.dumps({str(reject_reason): 1}, sort_keys=True),
            "reject_reason": reject_reason,
            "source_C2_coverage_before": source_before["coverage"],
            "source_C2_coverage_after": source_after["coverage"],
            "guard_C2_coverage_before": guard_before["coverage"],
            "guard_C2_coverage_after": guard_after["coverage"],
            "actual_C2_coverage_delta": guard_after["coverage"] - guard_before["coverage"],
            "actual_C2_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
            "local_patch_coverage_delta": guard_after["coverage"] - guard_before["coverage"] if task == "local_patch_interaction" else 0.0,
            "rotation_coverage_delta": guard_after["coverage"] - guard_before["coverage"] if task == "rotation_sensitive" else 0.0,
            "source_guard_c2_gap": source_guard_gap,
            "source_loss_delta": source_after["loss"] - source_before["loss"],
            "guard_loss_delta": guard_after["loss"] - guard_before["loss"],
            "output_residual_alignment_after_step": cosine_flat(actual_output_delta, rg),
            "activation_drift_layerwise": json.dumps(activation_drifts),
            "activation_drift_median": float(np.median(activation_drifts)) if activation_drifts else 0.0,
            "activation_drift_max": max(activation_drifts or [0.0]),
            "J_downstream_drift_estimate": j_drift,
            "G_edge_norm_step": param_norm,
            "parameter_norm_step": param_norm,
            "solver_overhead_ratio": 0.0,
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= 0.0),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "F5_no_debt_pass": no_debt_ok(debt, float(args.no_debt_budget)),
            "wall_time_s": time.time() - start,
            **diag,
        }
        return row
    except Exception as exc:
        return {"part": "E", "status": "error", "scheme": scheme, "checkpoint_name": checkpoint, "task": task, "seed": int(seed), "residual_target_type": target, "error_message": repr(exc), "wall_time_s": time.time() - start}


def part_e(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pd = read_json(OUT_ROOT / "part_d_summary.json")
    if not ival(pd.get("gate_pass")):
        return blocked_summary("E", "E_BlockedByPartD", str(pd.get("dominant_blocker", "part_d_missing_or_failed")))
    device = device_from_args(args)
    jobs = shard_items(e_jobs(args), args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"part_e_one_step_actual_matrix{suffix}.csv"
    rows: list[dict[str, Any]] = []
    existing = read_rows(matrix) if int(args.part_e_resume) else []
    done = {(r.get("scheme"), r.get("checkpoint_name"), r.get("task"), r.get("seed"), r.get("residual_target_type"), r.get("delta_scale"), r.get("delta_sign"), r.get("trust_mode")) for r in existing if r.get("status") == "ok"}
    for job in jobs:
        key = (job[0], job[1], job[2], str(job[3]), job[4], str(float(args.part_e_delta_scale)), str(float(args.part_e_delta_sign)), str(args.part_e_trust_mode))
        if key in done:
            continue
        rows.append(run_e_row(job, args, device))
        if int(args.part_e_flush_every) and len(rows) % int(args.part_e_flush_every) == 0:
            append_rows(matrix, rows)
            print(json.dumps({"part": "E", "rows_written": len(read_rows(matrix)), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
            rows = []
    if rows:
        append_rows(matrix, rows)
    all_rows = read_rows(matrix)
    summary = {
        "part": "E",
        "gate_pass": 0,
        "route": "PartEShardOnly" if suffix else "PartENeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(all_rows),
        "ok_rows": sum(1 for r in all_rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in all_rows if r.get("status") != "ok"),
        "matrix": rel(matrix),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_e_resume),
    }
    write_json(OUT_ROOT / f"part_e_summary{suffix}.json", summary)
    append_exec("Part E", command_text(sys.argv), "done", files=rel(matrix), gpu=str(device), note=f"rows={len(all_rows)}")
    append_recap("Part E one-step actual shard", summary)
    return summary


def enrich_part_e_controls(rows: list[dict[str, Any]]) -> None:
    base_keys = ["checkpoint_name", "task", "seed", "residual_target_type", "delta_scale", "delta_sign", "trust_mode"]
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(k, "") for k in base_keys)
        by_key.setdefault(key, {})[str(row.get("scheme"))] = row
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(k, "") for k in base_keys)
        controls = by_key.get(key, {})
        fg = controls.get("E0_FunctionalGram_one_step_baseline", {})
        local = controls.get("E1_v23_07_local_EFRF_one_step_C5_reference", {})
        rand = controls.get("E6_random_projected_downstream_one_step", {})
        shuf = controls.get("E7_shuffled_design_downstream_one_step", {})
        noop = controls.get("E8_same_compute_noop_one_step", {})
        row["functionalgram_C2_coverage_delta"] = fg.get("actual_C2_coverage_delta", "")
        row["v23_07_local_EFRF_C2_coverage_delta"] = local.get("actual_C2_coverage_delta", "")
        row["random_projected_C2_coverage_delta"] = rand.get("actual_C2_coverage_delta", "")
        row["shuffled_design_C2_coverage_delta"] = shuf.get("actual_C2_coverage_delta", "")
        row["same_compute_noop_C2_coverage_delta"] = noop.get("actual_C2_coverage_delta", "")
        row["functionalgram_coverage_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(fg.get("actual_C2_coverage_delta"))
        row["v23_07_local_EFRF_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(local.get("actual_C2_coverage_delta"))
        row["random_projected_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(rand.get("actual_C2_coverage_delta"))
        row["shuffled_design_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(shuf.get("actual_C2_coverage_delta"))
        row["same_compute_noop_gap"] = fval(row.get("actual_C2_coverage_delta")) - fval(noop.get("actual_C2_coverage_delta"))


def part_e_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_e_one_step_actual_matrix_shard*_of_*.csv"))
    rows = [r for path in shard_paths for r in read_rows(path)] if shard_paths else read_rows(OUT_ROOT / "part_e_one_step_actual_matrix.csv")
    enrich_part_e_controls(rows)
    matrix = write_rows(OUT_ROOT / "part_e_one_step_actual_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("task"), r.get("checkpoint_name"), r.get("residual_target_type"), r.get("delta_scale"), r.get("delta_sign"), r.get("trust_mode")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("task"), r.get("checkpoint_name"), r.get("residual_target_type"), r.get("delta_scale"), r.get("delta_sign"), r.get("trust_mode")) == key]
        task_summaries.append(
            {
                "scheme": key[0],
                "task": key[1],
                "checkpoint_name": key[2],
                "residual_target_type": key[3],
                "delta_scale": key[4],
                "delta_sign": key[5],
                "trust_mode": key[6],
                "rows": len(group),
                "actual_C2_coverage_delta_median": median(r.get("actual_C2_coverage_delta") for r in group),
                "actual_C2_accuracy_delta_median": median(r.get("actual_C2_accuracy_delta") for r in group),
                "random_projected_gap_median": median(r.get("random_projected_gap") for r in group),
                "shuffled_design_gap_median": median(r.get("shuffled_design_gap") for r in group),
            }
        )
    group_keys = sorted({(r.get("scheme"), r.get("checkpoint_name"), r.get("residual_target_type"), r.get("delta_scale"), r.get("delta_sign"), r.get("trust_mode")) for r in ok})
    groups: list[dict[str, Any]] = []
    for key in group_keys:
        group = [r for r in ok if (r.get("scheme"), r.get("checkpoint_name"), r.get("residual_target_type"), r.get("delta_scale"), r.get("delta_sign"), r.get("trust_mode")) == key]
        scheme = str(key[0])
        candidate = scheme in DOWNSTREAM_E_CANDIDATES
        last_candidate = scheme in LAST_LAYER_E_CANDIDATES
        local_patch = median(r.get("actual_C2_coverage_delta") for r in group if r.get("task") == "local_patch_interaction")
        rotation = median(r.get("actual_C2_coverage_delta") for r in group if r.get("task") == "rotation_sensitive")
        accept_rate = mean(r.get("finite_step_accept_rate") for r in group)
        scale_mean = mean(r.get("finite_step_scale_mean") for r in group)
        skip_count = sum(ival(r.get("finite_step_skip_count")) for r in group)
        fg_gap = median(r.get("functionalgram_coverage_gap") for r in group)
        rand_gap = median(r.get("random_projected_gap") for r in group)
        shuf_gap = median(r.get("shuffled_design_gap") for r in group)
        cov = median(r.get("actual_C2_coverage_delta") for r in group)
        acc = median(r.get("actual_C2_accuracy_delta") for r in group)
        pass_gate = int(
            candidate
            and cov >= 0.15
            and acc >= 0.05
            and fg_gap >= 0.05
            and rand_gap >= 0.10
            and shuf_gap >= 0.10
            and accept_rate >= 0.70
            and scale_mean >= 0.70
            and skip_count <= 5
            and not (local_patch < 0.0 and rotation < 0.0)
        )
        last_pass = int(
            last_candidate
            and cov >= 0.15
            and acc >= 0.05
            and fg_gap >= 0.05
            and rand_gap >= 0.10
            and shuf_gap >= 0.10
            and accept_rate >= 0.70
            and scale_mean >= 0.70
            and skip_count <= 5
        )
        groups.append(
            {
                "scheme": key[0],
                "checkpoint_name": key[1],
                "residual_target_type": key[2],
                "delta_scale": key[3],
                "delta_sign": key[4],
                "trust_mode": key[5],
                "rows": len(group),
                "candidate_downstream": int(candidate),
                "candidate_last_layer": int(last_candidate),
                "actual_C2_coverage_delta_median": cov,
                "actual_C2_accuracy_delta_median": acc,
                "local_patch_coverage_delta_median": local_patch,
                "rotation_coverage_delta_median": rotation,
                "functionalgram_coverage_gap_median": fg_gap,
                "v23_07_local_EFRF_gap_median": median(r.get("v23_07_local_EFRF_gap") for r in group),
                "random_projected_gap_median": rand_gap,
                "shuffled_design_gap_median": shuf_gap,
                "same_compute_noop_gap_median": median(r.get("same_compute_noop_gap") for r in group),
                "finite_step_accept_rate": accept_rate,
                "finite_step_scale_mean": scale_mean,
                "finite_step_skip_count": skip_count,
                "source_guard_c2_gap_median": median(r.get("source_guard_c2_gap") for r in group),
                "output_residual_alignment_after_step_median": median(r.get("output_residual_alignment_after_step") for r in group),
                "activation_drift_median": median(r.get("activation_drift_median") for r in group),
                "activation_drift_max_median": median(r.get("activation_drift_max") for r in group),
                "J_downstream_drift_estimate_median": median(r.get("J_downstream_drift_estimate") for r in group),
                "solver_overhead_ratio_median": median(r.get("solver_overhead_ratio") for r in group),
                "wall_time_s_median": median(r.get("wall_time_s") for r in group),
                "scheme_gate_pass": pass_gate,
                "last_layer_only_gate_pass": last_pass,
            }
        )
    task_csv = write_rows(OUT_ROOT / "part_e_one_step_actual_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_e_one_step_actual_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("scheme_gate_pass")) == 1]
    last_pass_groups = [g for g in groups if ival(g.get("last_layer_only_gate_pass")) == 1]
    candidate_groups = [g for g in groups if ival(g.get("candidate_downstream")) == 1]
    if pass_groups:
        gate, route, blocker = 1, "PartEOneStepDownstreamActualPass", "none"
    elif last_pass_groups:
        gate, route, blocker = 1, "LastLayerOnlySignal", "hidden_downstream_one_step_not_passed"
    else:
        gate = 0
        if any(fval(g.get("finite_step_accept_rate")) < 0.70 or fval(g.get("finite_step_scale_mean")) < 0.70 for g in candidate_groups):
            route, blocker = "E_OneStepDownstreamActualFailed", "finite_step_trust_low"
        elif any(fval(g.get("actual_C2_coverage_delta_median")) < 0.15 or fval(g.get("actual_C2_accuracy_delta_median")) < 0.05 for g in candidate_groups):
            route, blocker = "E_OneStepDownstreamActualFailed", "actual_c2_delta_low"
        elif any(fval(g.get("random_projected_gap_median")) < 0.10 or fval(g.get("shuffled_design_gap_median")) < 0.10 for g in candidate_groups):
            route, blocker = "E_OneStepDownstreamActualFailed", "random_or_shuffled_gap_low"
        else:
            route, blocker = "E_OneStepDownstreamActualFailed", "no_fixed_downstream_scheme_passed"
    failed_candidate_groups = []
    for group in candidate_groups:
        reasons = []
        if fval(group.get("actual_C2_coverage_delta_median")) < 0.15:
            reasons.append("actual_C2_coverage_delta_low")
        if fval(group.get("actual_C2_accuracy_delta_median")) < 0.05:
            reasons.append("actual_C2_accuracy_delta_low")
        if fval(group.get("functionalgram_coverage_gap_median")) < 0.05:
            reasons.append("functionalgram_gap_low")
        if fval(group.get("random_projected_gap_median")) < 0.10:
            reasons.append("random_projected_gap_low")
        if fval(group.get("shuffled_design_gap_median")) < 0.10:
            reasons.append("shuffled_design_gap_low")
        if fval(group.get("finite_step_accept_rate")) < 0.70 or fval(group.get("finite_step_scale_mean")) < 0.70:
            reasons.append("finite_step_trust_low")
        if reasons:
            failed_candidate_groups.append({"scheme": group.get("scheme"), "checkpoint_name": group.get("checkpoint_name"), "reasons": reasons})
    summary = {
        "part": "E",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "passing_scheme_groups": pass_groups,
        "last_layer_only_passing_groups": last_pass_groups,
        "candidate_groups": candidate_groups,
        "matrix": rel(matrix),
        "task_summary": rel(task_csv),
        "group_summary": rel(group_csv),
    }
    write_json(OUT_ROOT / "part_e_summary.json", summary)
    failure_path = write_failure_decomposition(
        "e",
        {
            "gate_pass": gate,
            "route": route,
            "dominant_blocker": blocker,
            "passing_scheme_count": len(pass_groups),
            "last_layer_only_passing_count": len(last_pass_groups),
            "failed_candidate_groups": failed_candidate_groups,
            "error_rows": len(rows) - len(ok),
        },
    )
    allowed = [] if gate else ["check sign convention with fixed rerun", "try smaller candidate norm/delta scale", "switch to objective-aligned coverage trust", "verify output residual target and unchanged Part D frozen fit"]
    next_actions("e", route, blocker, allowed, [rel(matrix), rel(task_csv), rel(group_csv), rel(failure_path)])
    append_exec("Part E merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_e_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part E one-step actual merge", summary)
    return summary


def f_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [(scheme, task, seed) for scheme in csv_items(args.part_f_schemes) for task in csv_items(args.part_f_tasks) for seed in range(int(args.part_f_seed_count))]


def f_e_args(args: argparse.Namespace, *, sketch_rank: int | None = None) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.part_e_lambda = float(args.part_f_lambda)
    out.part_e_alphas = str(args.part_f_alphas)
    out.part_e_delta_scale = float(args.part_f_delta_scale)
    out.part_e_delta_sign = float(args.part_f_delta_sign)
    out.part_e_trust_mode = str(args.part_f_trust_mode)
    out.part_e_trust_tolerance = float(args.part_f_trust_tolerance)
    out.part_e_local_reference_target = str(args.part_f_local_reference_target)
    out.part_e_functionalgram_lr = float(args.part_f_functionalgram_lr)
    out.part_e_sketch_rank = int(args.part_f_sketch_rank if sketch_rank is None else sketch_rank)
    out.part_d_solver = str(args.part_f_solver)
    return out


def choose_alpha_edge_optimizer_f(
    model: Any,
    base_state: dict[str, torch.Tensor],
    source_before: dict[str, float],
    guard_before: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    args: argparse.Namespace,
    *,
    use_population_gate: bool,
    adamw: bool = False,
    base_acts: list[torch.Tensor] | None = None,
    guard_logits_before: torch.Tensor | None = None,
    alignment_reference: torch.Tensor | None = None,
    alignment_records: list[float] | None = None,
    alignment_soft_scale_records: list[float] | None = None,
    tail99_reference: dict[str, float] | None = None,
) -> tuple[float, int, str, dict[str, float], dict[str, float]]:
    reject_reason = "no_alpha_candidates"
    eargs = f_e_args(args)
    drift_cap = float(getattr(args, "part_f_activation_drift_cap", -1.0))
    align_min = float(getattr(args, "part_f_alignment_min", -2.0))
    align_soft_floor = float(getattr(args, "part_f_alignment_soft_floor", -2.0))
    align_soft_ceiling = float(getattr(args, "part_f_alignment_soft_ceiling", 1.0))
    align_soft_min_scale = min(1.0, max(0.0, float(getattr(args, "part_f_alignment_soft_min_scale", 0.25))))
    align_soft_power = max(0.05, float(getattr(args, "part_f_alignment_soft_power", 0.5)))

    def optimizer_step(alpha_value: float) -> tuple[dict[str, float], dict[str, float]]:
        load_model_state(model, base_state)
        if adamw:
            opt = torch.optim.AdamW(model.parameters(), lr=float(alpha_value) * float(args.part_f_adam_lr), weight_decay=float(args.weight_decay))
        else:
            opt = EdgeSobolevPopulationFlow(
                list(model.coeffs),
                lr=float(alpha_value) * float(args.part_f_functionalgram_lr),
                weight_decay=float(args.weight_decay),
                sobolev_exponent=0.0,
                edge_metric_type="functional_gram",
                edge_weight_normalization="trace",
                edge_weight_ridge=float(args.functional_gram_ridge),
                functional_gram_quadrature_points=int(args.quadrature_points),
                use_population_gate=bool(use_population_gate),
                gate_family="block",
                strict_edge_params=True,
            )
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y.long())
        loss.backward()
        opt.step()
        return actual_metrics(model, x, y), actual_metrics(model, xg, yg)

    for alpha in float_items(args.part_f_alphas):
        effective_alpha = float(alpha)
        source_after, guard_after = optimizer_step(effective_alpha)
        accepted, reject_reason = trust_accept_e(source_before, source_after, guard_before, guard_after, eargs)
        if accepted and alignment_reference is not None and guard_logits_before is not None and (align_min > -1.5 or align_soft_floor > -1.5):
            guard_logits_after = model(xg).detach()
            guard_logits_before_device = guard_logits_before.to(device=guard_logits_after.device)
            align = cosine_flat(guard_logits_after - guard_logits_before_device, alignment_reference)
            soft_scale = 1.0
            if align_soft_floor > -1.5:
                denom = max(1.0e-12, align_soft_ceiling - align_soft_floor)
                cone_coordinate = min(1.0, max(0.0, (float(align) - align_soft_floor) / denom))
                soft_scale = max(align_soft_min_scale, min(1.0, cone_coordinate ** align_soft_power))
                if soft_scale < 0.999999:
                    effective_alpha = float(alpha) * soft_scale
                    source_after, guard_after = optimizer_step(effective_alpha)
                    accepted, reject_reason = trust_accept_e(source_before, source_after, guard_before, guard_after, eargs)
                    if accepted:
                        guard_logits_after = model(xg).detach()
                        align = cosine_flat(guard_logits_after - guard_logits_before_device, alignment_reference)
            if alignment_records is not None:
                alignment_records.append(float(align))
            if alignment_soft_scale_records is not None:
                alignment_soft_scale_records.append(float(soft_scale))
            if accepted and align_min > -1.5 and align < align_min:
                accepted, reject_reason = False, "alignment_gate_reject_optimizer"
        if accepted and not tail99_step_accept(guard_before, guard_after, args):
            accepted, reject_reason = False, "tail99_step_trust_reject_optimizer"
        if accepted and not tail99_cumulative_accept(tail99_reference, guard_after, args):
            accepted, reject_reason = False, "tail99_cumulative_trust_reject_optimizer"
        if accepted and base_acts is not None and drift_cap > 0.0:
            drift = max_activation_drift_to_base(model, xg, base_acts)
            if drift > drift_cap:
                accepted, reject_reason = False, "activation_drift_cap_reject_optimizer"
        if accepted:
            return float(effective_alpha), 1, "accepted", source_after, guard_after
    load_model_state(model, base_state)
    return 0.0, 0, reject_reason, source_before, guard_before


def f_scheme_to_e_scheme(scheme: str) -> tuple[str, int | None]:
    if scheme == "F4_last_layer_only_EFRF":
        return "E2_last_layer_exact_EFRF_one_step", None
    if scheme in {"F5_last_two_layers_topdown_EFRF", "F7_topdown_last_to_first_refresh_EFRF"}:
        return "E4_topdown_last_penultimate_one_step", None
    if scheme == "F6_penultimate_downstream_CG_EFRF":
        return "E3_penultimate_downstream_EFRF_one_step", None
    if scheme == "F8_hidden_downstream_sketch_rank8_EFRF":
        return "E5_hidden_downstream_sketch_one_step", 8
    if scheme == "F9_hidden_downstream_sketch_rank16_EFRF":
        return "E5_hidden_downstream_sketch_one_step", 16
    if scheme == "F14_hidden_downstream_sketch_rank32_EFRF":
        return "E5_hidden_downstream_sketch_one_step", 32
    if scheme == "F11_random_projected_residual_flow":
        return "E6_random_projected_downstream_one_step", None
    if scheme == "F12_shuffled_design_residual_flow":
        return "E7_shuffled_design_downstream_one_step", None
    if scheme == "F13_same_compute_noop":
        return "E8_same_compute_noop_one_step", None
    if scheme == "F3_v23_07_local_EFRF_all_layer_GS_control":
        return "E1_v23_07_local_EFRF_one_step_C5_reference", None
    if scheme in ADJOINT_LOCAL_HYBRID_SCHEMES:
        return "E9_topdown_last_penultimate_adjoint_local_one_step", None
    if scheme in HYBRID_REFRESH_INTERVALS:
        return "E4_topdown_last_penultimate_one_step", None
    raise ValueError(f"no E mapping for {scheme}")


def run_f_row(job: tuple[str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, task, seed = job
    start = time.time()
    try:
        x, y, xg, yg = v2307.visual_data(task, int(seed), args, device)
        classes = int(max(y.max(), yg.max()).detach().cpu().item()) + 1
        model_seed_scheme_key = str(args.part_f_model_seed_scheme_key or scheme)
        model_seed = 23083000 + int(seed) * 1009 + sum(ord(c) for c in model_seed_scheme_key + task)
        model = v2307.make_model(str(args.basis_key), int(args.depth), int(x.shape[1]), classes, model_seed, args, device, basis_input_gain=float(args.part_d_basis_input_gain))
        ckpt = v2307.train_checkpoint(model, x, y, str(args.part_f_checkpoint), int(seed), args)
        train_before = actual_metrics(model, x, y)
        guard_before = actual_metrics(model, xg, yg)
        _logits0, acts0 = model.forward_with_activations(xg)
        finite_accept = 0
        finite_skip = 0
        finite_scales: list[float] = []
        reject_reasons: dict[str, int] = {}
        activation_drift_curve: list[float] = []
        solve_residuals: list[float] = []
        conds: list[float] = []
        cg_iters: list[float] = []
        step_gains: list[float] = []
        between_alignments: list[float] = []
        between_alignment_soft_scales: list[float] = []
        tail_correction_betas: list[float] = []
        tail_correction_transport_alignments: list[float] = []
        latest_topdown_guard_delta: torch.Tensor | None = None
        for step in range(1, int(args.part_f_steps) + 1):
            source_before = actual_metrics(model, x, y)
            step_guard_before = actual_metrics(model, xg, yg)
            guard_logits_step_before = model(xg).detach()
            base_state = model_state(model)
            if scheme == "F13_same_compute_noop":
                e_scheme, rank = f_scheme_to_e_scheme(scheme)
                _deltas, diag = e_delta_candidate(e_scheme, model, x, y, xg, yg, str(args.part_f_residual_target), f_e_args(args, sketch_rank=rank), device, seed=int(seed) + step, task=task, checkpoint=str(args.part_f_checkpoint))
                alpha, accepted, reject_reason, source_after, guard_after = 0.0, 1, "same_compute_noop", source_before, step_guard_before
            elif scheme == "F0_FunctionalGram_baseline":
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, args, use_population_gate=False)
            elif scheme == "F1_BlockSNR_baseline":
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, args, use_population_gate=True)
            elif scheme == "F2_H10_known_structure_baseline":
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, args, use_population_gate=False, adamw=True)
            elif scheme in HYBRID_REFRESH_INTERVALS and (step % int(HYBRID_REFRESH_INTERVALS[scheme])) != 1:
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(
                    model,
                    base_state,
                    source_before,
                    step_guard_before,
                    x,
                    y,
                    xg,
                    yg,
                    args,
                    use_population_gate=False,
                    base_acts=acts0,
                    guard_logits_before=guard_logits_step_before,
                    alignment_reference=latest_topdown_guard_delta if scheme in ALIGNMENT_REFERENCE_SCHEMES else None,
                    alignment_records=between_alignments,
                    alignment_soft_scale_records=between_alignment_soft_scales,
                    tail99_reference=guard_before,
                )
            else:
                e_scheme, rank = f_scheme_to_e_scheme(scheme)
                eargs = f_e_args(args, sketch_rank=rank)
                raw_deltas, diag = e_delta_candidate(e_scheme, model, x, y, xg, yg, str(args.part_f_residual_target), eargs, device, seed=int(seed) + step, task=task, checkpoint=str(args.part_f_checkpoint))
                deltas = scaled_deltas(raw_deltas, eargs)
                if scheme == "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction":
                    tail_group = str(args.part_f_tail_correction_layer_group).lower()
                    tail_scheme = "E2_last_layer_exact_EFRF_one_step" if tail_group == "last_layer" else e_scheme
                    tail_eargs = f_e_args(args)
                    tail_raw, tail_diag = e_delta_candidate(tail_scheme, model, x, y, xg, yg, str(args.part_f_tail_correction_target), tail_eargs, device, seed=int(seed) + step + 9100, task=task, checkpoint=str(args.part_f_checkpoint))
                    tail_deltas = scaled_deltas(tail_raw, tail_eargs)
                    alpha, accepted, reject_reason, source_after, guard_after, deltas, _beta = choose_tail_corrected_f_deltas(
                        model,
                        base_state,
                        deltas,
                        tail_deltas,
                        source_before,
                        step_guard_before,
                        x,
                        y,
                        xg,
                        yg,
                        eargs,
                        acts0,
                        tail99_reference=guard_before,
                        beta_records=tail_correction_betas,
                        transport_alignment_records=tail_correction_transport_alignments,
                    )
                    diag["tail_solve_residual_max"] = fval(tail_diag.get("solve_residual_max"))
                    diag["tail_condition_number_after_ridge_max"] = fval(tail_diag.get("condition_number_after_ridge_max"))
                else:
                    alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_for_f_deltas(model, base_state, deltas, source_before, step_guard_before, x, y, xg, yg, eargs, acts0, tail99_reference=guard_before)
                if scheme in ALIGNMENT_REFERENCE_SCHEMES and accepted:
                    guard_logits_after = model(xg).detach()
                    delta_logits = guard_logits_after - guard_logits_step_before.to(device=guard_logits_after.device)
                    if float(delta_logits.norm().detach().cpu().item()) > 1.0e-12:
                        latest_topdown_guard_delta = delta_logits.detach()
            finite_accept += int(accepted)
            finite_skip += int(not accepted)
            finite_scales.append(float(alpha))
            reject_reasons[str(reject_reason)] = reject_reasons.get(str(reject_reason), 0) + 1
            solve_residuals.append(fval(diag.get("solve_residual_max")))
            conds.append(fval(diag.get("condition_number_after_ridge_max")))
            cg_iters.append(fval(diag.get("cg_iterations")))
            step_gains.append(guard_after["coverage"] - step_guard_before["coverage"])
            _logits_t, acts_t = model.forward_with_activations(xg)
            drifts = [
                float((after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)).norm().div(before.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12)).detach().cpu().item())
                for before, after in zip(acts0[1:], acts_t[1:])
            ]
            activation_drift_curve.append(max(drifts or [0.0]))
        train_after = actual_metrics(model, x, y)
        guard_after = actual_metrics(model, xg, yg)
        debt = debt_deltas(guard_before, guard_after)
        if scheme in HYBRID_REFRESH_INTERVALS:
            interval = int(HYBRID_REFRESH_INTERVALS[scheme])
            if scheme == "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction":
                cadence = f"every_{interval}_steps_transport_preserving_tail_correction_refresh"
            elif scheme in ADJOINT_LOCAL_HYBRID_SCHEMES:
                cadence = f"every_{interval}_steps_adjoint_local_refresh"
            elif scheme in ALIGNMENT_REFERENCE_SCHEMES:
                cadence = f"every_{interval}_steps_alignment_gated_refresh"
            else:
                cadence = f"every_{interval}_steps_refresh"
        else:
            cadence = "topdown_refresh_after_each_layer" if scheme in {"F5_last_two_layers_topdown_EFRF", "F7_topdown_last_to_first_refresh_EFRF"} else "every_step_refresh"
        return {
            "part": "F",
            "status": "ok",
            "scheme": scheme,
            "model_seed_scheme_key": model_seed_scheme_key,
            "candidate_downstream": int(scheme in F_CANDIDATE_SCHEMES),
            "basis_key": str(args.basis_key),
            "depth": int(args.depth),
            "task": task,
            "seed": int(seed),
            "train_size": int(args.train_size),
            "guard_size": int(args.guard_size),
            "train_steps": int(args.part_f_steps),
            "residual_refresh_cadence": cadence,
            "solver_type": str(args.part_f_solver),
            "lambda": float(args.part_f_lambda),
            "sketch_rank": 8 if "rank8" in scheme else (16 if "rank16" in scheme else (32 if "rank32" in scheme else 0)),
            "residual_target": str(args.part_f_residual_target),
            "activation_drift_cap": float(args.part_f_activation_drift_cap),
            "alignment_min": float(args.part_f_alignment_min),
            "alignment_soft_floor": float(args.part_f_alignment_soft_floor),
            "alignment_soft_ceiling": float(args.part_f_alignment_soft_ceiling),
            "alignment_soft_min_scale": float(args.part_f_alignment_soft_min_scale),
            "alignment_soft_power": float(args.part_f_alignment_soft_power),
            "tail99_step_budget": float(args.part_f_tail99_step_budget),
            "tail99_cumulative_budget": float(args.part_f_tail99_cumulative_budget),
            "tail_correction_scales": str(args.part_f_tail_correction_scales),
            "tail_correction_target": str(args.part_f_tail_correction_target),
            "tail_correction_layer_group": str(args.part_f_tail_correction_layer_group),
            "tail_correction_transport_min": float(args.part_f_tail_correction_transport_min),
            "functionalgram_lr": float(args.part_f_functionalgram_lr),
            "checkpoint_name": str(args.part_f_checkpoint),
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "C2_accuracy_improvement": guard_after["accuracy"] - guard_before["accuracy"],
            "C2_coverage_improvement": guard_after["coverage"] - guard_before["coverage"],
            "local_patch_coverage_improvement": guard_after["coverage"] - guard_before["coverage"] if task == "local_patch_interaction" else 0.0,
            "rotation_coverage_improvement": guard_after["coverage"] - guard_before["coverage"] if task == "rotation_sensitive" else 0.0,
            "guard_nll_delta": guard_after["loss"] - guard_before["loss"],
            "train_nll_delta": train_after["loss"] - train_before["loss"],
            "source_guard_c2_gap": (train_after["coverage"] - train_before["coverage"]) - (guard_after["coverage"] - guard_before["coverage"]),
            "F5_no_debt_pass": no_debt_ok(debt, float(args.no_debt_budget)),
            "Brier_delta": debt["Brier_delta"],
            "ECE_delta": debt["ECE_delta"],
            "tail95_delta": debt["tail95_delta"],
            "tail99_delta": debt["tail99_delta"],
            "margin10_delta_correct_sign": int(debt["margin10_delta"] >= 0.0),
            "component_non_positive": int(debt["Brier_delta"] <= 0.0 and debt["ECE_delta"] <= 0.0 and debt["tail95_delta"] <= 0.0 and debt["tail99_delta"] <= 0.0),
            "finite_step_accept_count": finite_accept,
            "finite_step_accept_rate": finite_accept / max(1, int(args.part_f_steps)),
            "finite_step_scale_mean": mean(finite_scales),
            "finite_step_skip_count": finite_skip,
            "finite_step_reject_reasons": json.dumps(reject_reasons, sort_keys=True),
            "between_step_alignment_median": median(between_alignments),
            "between_step_alignment_min": min(between_alignments) if between_alignments else 0.0,
            "between_step_alignment_count": len(between_alignments),
            "between_step_alignment_soft_scale_median": median(between_alignment_soft_scales),
            "between_step_alignment_soft_scale_count": len(between_alignment_soft_scales),
            "tail_correction_beta_median": median(tail_correction_betas),
            "tail_correction_beta_count": len(tail_correction_betas),
            "tail_correction_transport_alignment_median": median(tail_correction_transport_alignments),
            "tail_correction_transport_alignment_min": min(tail_correction_transport_alignments) if tail_correction_transport_alignments else 0.0,
            "tail_correction_transport_alignment_count": len(tail_correction_transport_alignments),
            "activation_drift_median": median(activation_drift_curve),
            "activation_drift_max": max(activation_drift_curve or [0.0]),
            "J_downstream_drift_sketch": 0.0,
            "residual_fit_retention_curve": json.dumps(step_gains),
            "one_step_gain_retention_curve": json.dumps(step_gains),
            "solver_overhead_ratio": 0.0,
            "cg_iteration_median": median(cg_iters),
            "condition_number_median": median(conds),
            "solve_residual_max": max(solve_residuals or [0.0]),
            "held_test_usage": 0,
            "used_fake_data_rows": 0,
            "runtime_selector_used": 0,
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {"part": "F", "status": "error", "scheme": scheme, "task": task, "seed": int(seed), "error_message": repr(exc), "wall_time_s": time.time() - start}


def part_f(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pe = read_json(OUT_ROOT / "part_e_summary.json")
    if not ival(pe.get("gate_pass")):
        return blocked_summary("F", "F_BlockedByPartE", str(pe.get("dominant_blocker", "part_e_missing_or_failed")))
    device = device_from_args(args)
    jobs = shard_items(f_jobs(args), args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"part_f_multistep_positive_control_matrix{suffix}.csv"
    rows: list[dict[str, Any]] = []
    existing = read_rows(matrix) if int(args.part_f_resume) else []
    done = {
        (
            r.get("scheme"),
            r.get("task"),
            r.get("seed"),
            r.get("train_steps"),
            r.get("residual_target"),
            r.get("lambda"),
            r.get("activation_drift_cap"),
            r.get("functionalgram_lr"),
            r.get("alignment_min"),
            r.get("alignment_soft_floor"),
            r.get("alignment_soft_ceiling"),
            r.get("alignment_soft_min_scale"),
            r.get("alignment_soft_power"),
            r.get("tail99_step_budget"),
            r.get("tail99_cumulative_budget"),
            r.get("model_seed_scheme_key"),
            r.get("tail_correction_scales"),
            r.get("tail_correction_target"),
            r.get("tail_correction_layer_group"),
            r.get("tail_correction_transport_min"),
        )
        for r in existing
        if r.get("status") == "ok"
    }
    for job in jobs:
        key = (
            job[0],
            job[1],
            str(job[2]),
            str(int(args.part_f_steps)),
            str(args.part_f_residual_target),
            str(float(args.part_f_lambda)),
            str(float(args.part_f_activation_drift_cap)),
            str(float(args.part_f_functionalgram_lr)),
            str(float(args.part_f_alignment_min)),
            str(float(args.part_f_alignment_soft_floor)),
            str(float(args.part_f_alignment_soft_ceiling)),
            str(float(args.part_f_alignment_soft_min_scale)),
            str(float(args.part_f_alignment_soft_power)),
            str(float(args.part_f_tail99_step_budget)),
            str(float(args.part_f_tail99_cumulative_budget)),
            str(args.part_f_model_seed_scheme_key or job[0]),
            str(args.part_f_tail_correction_scales),
            str(args.part_f_tail_correction_target),
            str(args.part_f_tail_correction_layer_group),
            str(float(args.part_f_tail_correction_transport_min)),
        )
        if key in done:
            continue
        rows.append(run_f_row(job, args, device))
        if int(args.part_f_flush_every) and len(rows) % int(args.part_f_flush_every) == 0:
            append_rows(matrix, rows)
            print(json.dumps({"part": "F", "rows_written": len(read_rows(matrix)), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
            rows = []
    if rows:
        append_rows(matrix, rows)
    all_rows = read_rows(matrix)
    summary = {
        "part": "F",
        "gate_pass": 0,
        "route": "PartFShardOnly" if suffix else "PartFNeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(all_rows),
        "ok_rows": sum(1 for r in all_rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in all_rows if r.get("status") != "ok"),
        "matrix": rel(matrix),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_f_resume),
    }
    write_json(OUT_ROOT / f"part_f_summary{suffix}.json", summary)
    append_exec("Part F", command_text(sys.argv), "done", files=rel(matrix), gpu=str(device), note=f"rows={len(all_rows)}")
    append_recap("Part F multistep shard", summary)
    return summary


def part_f_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_f_multistep_positive_control_matrix_shard*_of_*.csv"))
    main_path = OUT_ROOT / "part_f_multistep_positive_control_matrix.csv"
    raw_rows = [r for path in shard_paths for r in read_rows(path)]
    raw_rows.extend(read_rows(main_path))
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            str(row.get("scheme") or ""),
            str(row.get("task") or ""),
            str(row.get("seed") or ""),
            str(row.get("train_steps") or ""),
            str(row.get("residual_target") or ""),
            str(row.get("lambda") or ""),
            str(row.get("checkpoint_name") or ""),
            str(row.get("train_size") or ""),
            str(row.get("guard_size") or ""),
            str(row.get("activation_drift_cap") or ""),
            str(row.get("functionalgram_lr") or ""),
            str(row.get("alignment_min") or ""),
            str(row.get("alignment_soft_floor") or ""),
            str(row.get("alignment_soft_ceiling") or ""),
            str(row.get("alignment_soft_min_scale") or ""),
            str(row.get("alignment_soft_power") or ""),
            str(row.get("tail99_step_budget") or ""),
            str(row.get("tail99_cumulative_budget") or ""),
            str(row.get("model_seed_scheme_key") or ""),
            str(row.get("tail_correction_scales") or ""),
            str(row.get("tail_correction_target") or ""),
            str(row.get("tail_correction_layer_group") or ""),
            str(row.get("tail_correction_transport_min") or ""),
        )
        dedup[key] = row
    rows = list(dedup.values())
    matrix = write_rows(OUT_ROOT / "part_f_multistep_positive_control_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("task")) == key]
        task_summaries.append({"scheme": key[0], "task": key[1], "rows": len(group), "C2_coverage_improvement_median": median(r.get("C2_coverage_improvement") for r in group), "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group)})
    task_by = {(t["scheme"], t["task"]): t for t in task_summaries}

    def task_cov(scheme: str, task: str) -> float:
        return fval(task_by.get((scheme, task), {}).get("C2_coverage_improvement_median"))

    def scheme_cov(scheme: str) -> float:
        return median(t.get("C2_coverage_improvement_median") for t in task_summaries if t.get("scheme") == scheme)

    fg_cov = scheme_cov("F0_FunctionalGram_baseline")
    bs_cov = scheme_cov("F1_BlockSNR_baseline")
    h10_cov = scheme_cov("F2_H10_known_structure_baseline")
    rand_cov = scheme_cov("F11_random_projected_residual_flow")
    shuf_cov = scheme_cov("F12_shuffled_design_residual_flow")
    noop_cov = scheme_cov("F13_same_compute_noop")
    fg_wall = median(r.get("wall_time_s") for r in ok if r.get("scheme") == "F0_FunctionalGram_baseline")
    official_shape = int(int(args.part_f_seed_count) >= 15 and int(args.part_f_steps) >= 80)
    groups: list[dict[str, Any]] = []
    for scheme in sorted({r.get("scheme") for r in ok}):
        group = [r for r in ok if r.get("scheme") == scheme]
        cov = scheme_cov(str(scheme))
        local_cov = task_cov(str(scheme), "local_patch_interaction")
        rotation_cov = task_cov(str(scheme), "rotation_sensitive")
        wall = median(r.get("wall_time_s") for r in group)
        overhead = wall / max(fg_wall, 1.0e-12) if fg_wall > 0 else 99.0
        random_gap = cov - rand_cov
        shuffled_gap = cov - shuf_cov
        pass_gate = int(
            official_shape
            and str(scheme) in F_CANDIDATE_SCHEMES
            and cov >= fg_cov + 0.03
            and cov >= 0.08
            and local_cov >= 0.02
            and rotation_cov >= 0.02
            and random_gap >= 0.05
            and shuffled_gap >= 0.05
            and sum(ival(r.get("F5_no_debt_pass")) for r in group) >= 12
            and sum(ival(r.get("component_non_positive")) for r in group) >= 12
            and median(r.get("finite_step_accept_rate") for r in group) >= 0.20
            and median(r.get("finite_step_scale_mean") for r in group) >= 0.05
            and median(r.get("finite_step_skip_count") for r in group) <= 56
            and overhead <= 5.0
        )
        groups.append(
            {
                "scheme": scheme,
                "rows": len(group),
                "candidate_downstream": int(str(scheme) in F_CANDIDATE_SCHEMES),
                "model_seed_scheme_key": str(group[0].get("model_seed_scheme_key", "")),
                "residual_target": str(group[0].get("residual_target", "")),
                "official_shape": official_shape,
                "C2_coverage_improvement_median": cov,
                "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
                "local_patch_coverage_improvement_median": local_cov,
                "rotation_coverage_improvement_median": rotation_cov,
                "beats_FunctionalGram_gap": cov - fg_cov,
                "beats_BlockSNR_gap": cov - bs_cov,
                "beats_H10_gap_or_gap_to_H10": cov - h10_cov,
                "random_projected_gap": random_gap,
                "shuffled_design_gap": shuffled_gap,
                "same_compute_gap": cov - noop_cov,
                "F5_no_debt_count": sum(ival(r.get("F5_no_debt_pass")) for r in group),
                "component_non_positive_rows": sum(ival(r.get("component_non_positive")) for r in group),
                "finite_step_accept_rate_median": median(r.get("finite_step_accept_rate") for r in group),
                "finite_step_scale_mean_median": median(r.get("finite_step_scale_mean") for r in group),
                "finite_step_skip_count_median": median(r.get("finite_step_skip_count") for r in group),
                "between_step_alignment_median": median(r.get("between_step_alignment_median") for r in group),
                "between_step_alignment_min_median": median(r.get("between_step_alignment_min") for r in group),
                "between_step_alignment_count_median": median(r.get("between_step_alignment_count") for r in group),
                "between_step_alignment_soft_scale_median": median(r.get("between_step_alignment_soft_scale_median") for r in group),
                "between_step_alignment_soft_scale_count_median": median(r.get("between_step_alignment_soft_scale_count") for r in group),
                "alignment_min": median(r.get("alignment_min") for r in group),
                "alignment_soft_floor": median(r.get("alignment_soft_floor") for r in group),
                "alignment_soft_ceiling": median(r.get("alignment_soft_ceiling") for r in group),
                "alignment_soft_min_scale": median(r.get("alignment_soft_min_scale") for r in group),
                "alignment_soft_power": median(r.get("alignment_soft_power") for r in group),
                "tail99_step_budget": median(r.get("tail99_step_budget") for r in group),
                "tail99_cumulative_budget": median(r.get("tail99_cumulative_budget") for r in group),
                "tail_correction_scales": str(group[0].get("tail_correction_scales", "")),
                "tail_correction_target": str(group[0].get("tail_correction_target", "")),
                "tail_correction_layer_group": str(group[0].get("tail_correction_layer_group", "")),
                "tail_correction_transport_min": median(r.get("tail_correction_transport_min") for r in group),
                "tail_correction_beta_median": median(r.get("tail_correction_beta_median") for r in group),
                "tail_correction_beta_count_median": median(r.get("tail_correction_beta_count") for r in group),
                "tail_correction_transport_alignment_median": median(r.get("tail_correction_transport_alignment_median") for r in group),
                "tail_correction_transport_alignment_min_median": median(r.get("tail_correction_transport_alignment_min") for r in group),
                "tail_correction_transport_alignment_count_median": median(r.get("tail_correction_transport_alignment_count") for r in group),
                "functionalgram_lr": median(r.get("functionalgram_lr") for r in group),
                "activation_drift_median": median(r.get("activation_drift_median") for r in group),
                "activation_drift_max_median": median(r.get("activation_drift_max") for r in group),
                "solver_overhead_ratio": overhead,
                "wall_time_s_median": wall,
                "cg_iteration_median": median(r.get("cg_iteration_median") for r in group),
                "condition_number_median": median(r.get("condition_number_median") for r in group),
                "held_test_usage": max(ival(r.get("held_test_usage")) for r in group),
                "used_fake_data_rows": max(ival(r.get("used_fake_data_rows")) for r in group),
                "runtime_selector_used": max(ival(r.get("runtime_selector_used")) for r in group),
                "scheme_gate_pass": pass_gate,
            }
        )
    task_csv = write_rows(OUT_ROOT / "part_f_multistep_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_f_multistep_group_summary.csv", groups)
    pass_groups = [g for g in groups if ival(g.get("scheme_gate_pass")) == 1]
    candidate_groups = [g for g in groups if ival(g.get("candidate_downstream")) == 1]
    if pass_groups:
        route, blocker, gate = "PartFPositiveControlPass", "none", 1
    elif not official_shape:
        route, blocker, gate = "PartFDiagnosticOnly_FormalNotRun", "formal_seed_or_step_count_not_met", 0
    elif any(fval(g.get("solver_overhead_ratio")) > 5.0 for g in candidate_groups):
        route, blocker, gate = "MechanismDiagnosticOnly_OverheadTooHigh", "overhead_ratio_high", 0
    elif any(fval(g.get("random_projected_gap")) < 0.05 or fval(g.get("shuffled_design_gap")) < 0.05 for g in candidate_groups):
        route, blocker, gate = "F_PositiveControlFailed", "random_or_shuffled_gap_low", 0
    else:
        route, blocker, gate = "F_PositiveControlFailed", "no_fixed_scheme_passed", 0
    summary = {"part": "F", "gate_pass": gate, "route": route, "dominant_blocker": blocker, "official_shape": official_shape, "row_count": len(rows), "ok_rows": len(ok), "error_rows": len(rows) - len(ok), "passing_scheme_groups": pass_groups, "candidate_groups": candidate_groups, "matrix": rel(matrix), "task_summary": rel(task_csv), "group_summary": rel(group_csv)}
    write_json(OUT_ROOT / "part_f_summary.json", summary)
    failure_path = write_failure_decomposition("f", {"gate_pass": gate, "route": route, "dominant_blocker": blocker, "official_shape": official_shape, "candidate_groups": candidate_groups, "error_rows": len(rows) - len(ok)})
    allowed = [] if gate else ["run 15-seed/80-step formal only if diagnostic mechanism and overhead justify it", "run Part G drift audit", "try shorter refresh cadence or candidate norm clipping if one-step signal fails to accumulate", "enter Part H scaling if exact overhead dominates"]
    next_actions("f", route, blocker, allowed, [rel(matrix), rel(task_csv), rel(group_csv), rel(failure_path)])
    append_exec("Part F merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_f_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; official_shape={official_shape}")
    append_recap("Part F multistep merge", summary)
    return summary


def part_g(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pf = read_json(OUT_ROOT / "part_f_summary.json")
    if not pf:
        return blocked_summary("G", "G_BlockedByPartF", "part_f_missing")
    groups = read_rows(OUT_ROOT / "part_f_multistep_group_summary.csv")
    candidates = [g for g in groups if ival(g.get("candidate_downstream")) == 1]
    best = max(candidates, key=lambda g: fval(g.get("C2_coverage_improvement_median")), default={})
    blockers: list[str] = []
    if fval(best.get("C2_coverage_improvement_median")) < 0.08:
        blockers.append("one_step_signal_not_accumulated")
    if fval(best.get("activation_drift_max_median")) > 0.50:
        blockers.append("activation_drift_high")
    if fval(best.get("finite_step_accept_rate_median")) < 0.20 or fval(best.get("finite_step_scale_mean_median")) < 0.05:
        blockers.append("trust_scale_kills_step")
    if fval(best.get("random_projected_gap")) < 0.05 or fval(best.get("shuffled_design_gap")) < 0.05:
        blockers.append("random_control_assimilates")
    if fval(best.get("solver_overhead_ratio")) > 5.0:
        blockers.append("solver_overhead_high")
    if not blockers:
        blockers.append(str(pf.get("dominant_blocker", "no_blocker_classified")))
    summary = {
        "part": "G",
        "gate_pass": 1,
        "route": "G_AccumulationExplained",
        "dominant_blocker": blockers[0],
        "all_blockers": blockers,
        "part_f_route": pf.get("route"),
        "best_candidate_group": best,
        "matrix": rel(OUT_ROOT / "part_f_multistep_positive_control_matrix.csv"),
        "group_summary": rel(OUT_ROOT / "part_f_multistep_group_summary.csv"),
        "interpretation": "Part G explains Part F outcome from recorded drift, trust, random-control and overhead diagnostics; it does not promote a method.",
    }
    write_json(OUT_ROOT / "part_g_summary.json", summary)
    next_actions("g", summary["route"], summary["dominant_blocker"], ["if activation drift dominates, try topdown refresh/candidate clipping; if overhead dominates, run Part H scaling; do not enter real-task without Part F/H pass"], [summary["matrix"], summary["group_summary"]])
    append_exec("Part G", command_text(sys.argv), "done", files=rel(OUT_ROOT / "part_g_summary.json"), gpu=str(args.device), note=f"blocker={summary['dominant_blocker']}")
    append_recap("Part G accumulation audit", summary)
    return summary


def part_h(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    group_path = OUT_ROOT / "part_f_multistep_group_summary.csv"
    groups = read_rows(group_path)
    if not groups:
        return blocked_summary("H", "H_BlockedByPartF", "part_f_group_summary_missing")
    by_scheme = {str(g.get("scheme")): g for g in groups}
    exact = by_scheme.get(str(args.part_h_exact_reference_scheme), {})
    if not exact:
        return blocked_summary("H", "H_BlockedByExactReference", "exact_reference_scheme_missing")
    exact_cov = fval(exact.get("C2_coverage_improvement_median"))
    exact_random = fval(exact.get("random_projected_gap"))
    exact_f5 = max(1.0, fval(exact.get("F5_no_debt_count")))
    rows: list[dict[str, Any]] = []
    mapping = {
        "H3_output_sketch_rank8": "F8_hidden_downstream_sketch_rank8_EFRF",
        "H4_output_sketch_rank16": "F9_hidden_downstream_sketch_rank16_EFRF",
        "H5_output_sketch_rank32": "F14_hidden_downstream_sketch_rank32_EFRF",
        "H7_penultimate_exact_less_topdown": "F6_penultimate_downstream_CG_EFRF",
        "H8_EFRF_every5_steps": "F10_hybrid_EFRF_every5_steps_FunctionalGram_between",
        "H9_EFRF_every10_steps": "F15_hybrid_EFRF_every10_steps_FunctionalGram_between",
        "H10_EFRF_every10_alignment_gated": "F16_hybrid_EFRF_every10_alignment_gated_FunctionalGram_between",
        "H13_EFRF_every10_alignment_softscaled": "F17_hybrid_EFRF_every10_alignment_softscaled_FunctionalGram_between",
        "H14_EFRF_every10_negative_alignment_softclip": "F18_hybrid_EFRF_every10_negative_alignment_softclip_FunctionalGram_between",
        "H15_EFRF_every10_alignment_gated_tail99_guard": "F19_hybrid_EFRF_every10_alignment_gated_tail99_guard_FunctionalGram_between",
        "H16_EFRF_every10_alignment_gated_cumulative_tail99_guard": "F20_hybrid_EFRF_every10_alignment_gated_cumulative_tail99_guard_FunctionalGram_between",
        "H17_EFRF_every10_alignment_gated_tail_weighted_residual": "F21_hybrid_EFRF_every10_alignment_gated_tail_weighted_residual_FunctionalGram_between",
        "H18_EFRF_every10_transport_preserving_tail_correction": "F22_hybrid_EFRF_every10_alignment_gated_transport_preserving_tail_correction",
        "H19_EFRF_every20_steps": "F23_hybrid_EFRF_every20_steps_FunctionalGram_between",
        "H20_EFRF_every20_alignment_gated": "F24_hybrid_EFRF_every20_alignment_gated_FunctionalGram_between",
        "H21_EFRF_every30_steps": "F25_hybrid_EFRF_every30_steps_FunctionalGram_between",
        "H22_EFRF_every30_alignment_gated": "F26_hybrid_EFRF_every30_alignment_gated_FunctionalGram_between",
        "H23_EFRF_every20_adjoint_local": "F27_hybrid_EFRF_every20_adjoint_local_FunctionalGram_between",
        "H24_EFRF_every30_adjoint_local": "F28_hybrid_EFRF_every30_adjoint_local_FunctionalGram_between",
        "H11_random_projected_control": "F11_random_projected_residual_flow",
        "H12_shuffled_design_control": "F12_shuffled_design_residual_flow",
    }
    for h_scheme, f_scheme in mapping.items():
        g = by_scheme.get(f_scheme, {})
        if not g:
            rows.append({"part": "H", "status": "missing", "h_scheme": h_scheme, "source_f_scheme": f_scheme})
            continue
        cov_ret = fval(g.get("C2_coverage_improvement_median")) / max(exact_cov, 1.0e-12)
        random_ret = fval(g.get("random_projected_gap")) / max(exact_random, 1.0e-12)
        f5_ret = fval(g.get("F5_no_debt_count")) / exact_f5
        overhead = fval(g.get("solver_overhead_ratio"), float("inf"))
        pass_gate = int(cov_ret >= 0.85 and random_ret >= 0.85 and f5_ret >= 0.85 and overhead <= 3.0)
        rows.append(
            {
                "part": "H",
                "status": "ok",
                "h_scheme": h_scheme,
                "source_f_scheme": f_scheme,
                "exact_reference_scheme": args.part_h_exact_reference_scheme,
                "C2_coverage_retention_vs_exact": cov_ret,
                "random_gap_retention": random_ret,
                "F5_retention": f5_ret,
                "solver_time_ratio": overhead,
                "memory_peak_ratio": "not_measured",
                "cg_iterations": g.get("cg_iteration_median", ""),
                "normal_matrix_build_time": "not_measured",
                "sketch_rank": g.get("sketch_rank", ""),
                "approximation_error_to_exact_on_probe": 1.0 - cov_ret,
                "condition_number_after_preconditioner": g.get("condition_number_median", ""),
                "scheme_gate_pass": pass_gate,
            }
        )
    matrix = write_rows(OUT_ROOT / "part_h_solver_scaling_matrix.csv", rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    pass_rows = [r for r in ok if ival(r.get("scheme_gate_pass")) == 1]
    if pass_rows:
        gate, route, blocker = 1, "PartHScalingPass", "none"
    elif any(fval(r.get("C2_coverage_retention_vs_exact")) >= 0.85 and fval(r.get("random_gap_retention")) >= 0.85 and fval(r.get("solver_time_ratio")) > 3.0 for r in ok):
        gate, route, blocker = 0, "MechanismPass_ScalingBlocked", "overhead_ratio_high"
    else:
        gate, route, blocker = 0, "H_ScalingBlocked", "retention_or_overhead_failed"
    summary = {
        "part": "H",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "passing_solver_rows": pass_rows,
        "matrix": rel(matrix),
        "source_group_summary": rel(group_path),
        "exact_reference": exact,
    }
    write_json(OUT_ROOT / "part_h_summary.json", summary)
    failure_path = write_failure_decomposition("h", {"gate_pass": gate, "route": route, "dominant_blocker": blocker, "rows": rows})
    next_actions("h", route, blocker, [] if gate else ["implement cheaper topdown approximation: cached/EMA normal matrix, output sketch rank16/32, warm-start CG, or hybrid cadence with drift cap"], [rel(matrix), rel(failure_path)])
    append_exec("Part H", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_h_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part H solver scaling audit", summary)
    return summary


def part_i_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [(scheme, dataset, seed) for scheme in csv_items(args.part_i_schemes) for dataset in csv_items(args.part_i_datasets) for seed in range(int(args.part_i_seed_count))]


def part_i_scheme_config(i_scheme: str, args: argparse.Namespace) -> dict[str, Any]:
    if i_scheme == "I2_H10_scalar_dual_if_applicable_diagnostic":
        return {
            "status": "skipped",
            "skip_reason": "not_applicable: no fixed real-task H10 scalar dual observer is defined in v23.08; row is recorded explicitly instead of substituting a different method.",
        }
    f_scheme = PART_I_SCHEME_TO_F_SCHEME.get(i_scheme)
    if not f_scheme:
        raise ValueError(f"unknown Part I scheme {i_scheme!r}")
    residual_target = str(args.part_i_residual_target)
    if i_scheme == "I4_DownstreamEFRF_without_Wpop":
        residual_target = str(args.part_i_without_wpop_residual_target)
    return {"status": "ok", "f_scheme": f_scheme, "residual_target": residual_target}


def run_part_i_row(job: tuple[str, str, int], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    i_scheme, dataset, seed = job
    start = time.time()
    try:
        cfg = part_i_scheme_config(str(i_scheme), args)
        if cfg.get("status") == "skipped":
            return {
                "part": "I",
                "status": "skipped",
                "i_scheme": str(i_scheme),
                "scheme": "",
                "dataset": str(dataset),
                "seed": int(seed),
                "skip_reason": str(cfg.get("skip_reason")),
                "held_test_usage": 0,
                "used_fake_data_rows": 0,
                "runtime_selector_used": 0,
                "wall_time_s": time.time() - start,
            }
        f_scheme = str(cfg["f_scheme"])
        residual_target = str(cfg["residual_target"])
        bundle = load_part_i_real_bundle(str(dataset), int(seed), args, device)
        x = bundle["x_train"]
        y = bundle["y_train"]
        xg = bundle["x_guard"]
        yg = bundle["y_guard"]
        xh = bundle["x_held"]
        yh = bundle["y_held"]
        fargs = part_i_f_args(args, residual_target_override=residual_target)
        model_seed_key = str(args.part_i_model_seed_scheme_key or "part_i_common_init")
        model_seed = 23084000 + int(seed) * 1009 + sum(ord(c) for c in model_seed_key + str(dataset))
        model = v2307.make_model(
            str(args.part_i_basis),
            int(args.part_i_depth),
            int(bundle["input_dim"]),
            int(bundle["num_classes"]),
            model_seed,
            fargs,
            device,
            basis_input_gain=float(args.part_i_basis_input_gain),
        )
        ckpt = v2307.train_checkpoint(model, x, y, str(args.part_i_checkpoint), int(seed), fargs)
        train_before = actual_metrics(model, x, y)
        guard_before = actual_metrics(model, xg, yg)
        held_before = actual_metrics(model, xh, yh)
        _logits0, acts0 = model.forward_with_activations(xg)
        finite_accept = 0
        finite_skip = 0
        finite_scales: list[float] = []
        reject_reasons: dict[str, int] = {}
        solve_residuals: list[float] = []
        conds: list[float] = []
        cg_iters: list[float] = []
        activation_drift_curve: list[float] = []
        same_compute_candidate_count = 0
        for step in range(1, int(args.part_i_steps) + 1):
            source_before = actual_metrics(model, x, y)
            step_guard_before = actual_metrics(model, xg, yg)
            base_state = model_state(model)
            control_refresh = (step % max(1, int(args.part_i_control_refresh_interval))) == 1
            if str(i_scheme) == "I6_same_compute_noop" and not control_refresh:
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = 0.0, 1, "same_compute_noop_between_refresh", source_before, step_guard_before
            elif str(i_scheme) == "I5_random_projected_residual_flow" and not control_refresh:
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, fargs, use_population_gate=False)
            elif f_scheme == "F13_same_compute_noop":
                e_scheme, rank = f_scheme_to_e_scheme(f_scheme)
                _deltas, diag = e_delta_candidate(e_scheme, model, x, y, xg, yg, residual_target, f_e_args(fargs, sketch_rank=rank), device, seed=int(seed) + step, task=str(dataset), checkpoint=str(args.part_i_checkpoint))
                alpha, accepted, reject_reason, source_after, guard_after = 0.0, 1, "same_compute_noop", source_before, step_guard_before
                same_compute_candidate_count += ival(diag.get("same_compute_candidate_built"))
            elif f_scheme == "F0_FunctionalGram_baseline":
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, fargs, use_population_gate=False)
            elif f_scheme == "F1_BlockSNR_baseline":
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, fargs, use_population_gate=True)
            elif f_scheme in HYBRID_REFRESH_INTERVALS and (step % int(HYBRID_REFRESH_INTERVALS[f_scheme])) != 1:
                diag = {"solve_residual_max": 0.0, "condition_number_after_ridge_max": 0.0, "cg_iterations": 0.0}
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_edge_optimizer_f(model, base_state, source_before, step_guard_before, x, y, xg, yg, fargs, use_population_gate=False, base_acts=acts0, tail99_reference=guard_before)
            else:
                e_scheme, rank = f_scheme_to_e_scheme(f_scheme)
                eargs = f_e_args(fargs, sketch_rank=rank)
                raw_deltas, diag = e_delta_candidate(e_scheme, model, x, y, xg, yg, residual_target, eargs, device, seed=int(seed) + step, task=str(dataset), checkpoint=str(args.part_i_checkpoint))
                deltas = scaled_deltas(raw_deltas, eargs)
                alpha, accepted, reject_reason, source_after, guard_after = choose_alpha_for_f_deltas(model, base_state, deltas, source_before, step_guard_before, x, y, xg, yg, eargs, acts0, tail99_reference=guard_before)
            finite_accept += int(accepted)
            finite_skip += int(not accepted)
            finite_scales.append(float(alpha))
            reject_reasons[str(reject_reason)] = reject_reasons.get(str(reject_reason), 0) + 1
            solve_residuals.append(fval(diag.get("solve_residual_max")))
            conds.append(fval(diag.get("condition_number_after_ridge_max")))
            cg_iters.append(fval(diag.get("cg_iterations")))
            _logits_t, acts_t = model.forward_with_activations(xg)
            drifts = [
                float((after.detach().to(dtype=torch.float64) - before.detach().to(dtype=torch.float64)).norm().div(before.detach().to(dtype=torch.float64).norm().clamp_min(1.0e-12)).detach().cpu().item())
                for before, after in zip(acts0[1:], acts_t[1:])
            ]
            activation_drift_curve.append(max(drifts or [0.0]))
        train_after = actual_metrics(model, x, y)
        guard_after = actual_metrics(model, xg, yg)
        held_after = actual_metrics(model, xh, yh)
        held_debt = debt_deltas(held_before, held_after)
        return {
            "part": "I",
            "status": "ok",
            "i_scheme": str(i_scheme),
            "scheme": f_scheme,
            "dataset": str(dataset),
            "dataset_family": str(bundle["dataset_family"]),
            "seed": int(seed),
            "basis_key": str(args.part_i_basis),
            "depth": int(args.part_i_depth),
            "input_dim": int(bundle["input_dim"]),
            "output_dim": int(bundle["num_classes"]),
            "source_kind": str(bundle["source_kind"]),
            "dataset_loader_name": str(bundle["dataset_loader_name"]),
            "train_size": int(bundle["train_size_actual"]),
            "guard_size": int(bundle["guard_size_actual"]),
            "held_size": int(bundle["held_size_actual"]),
            "train_steps": int(args.part_i_steps),
            "residual_target": residual_target,
            "part_i_control_refresh_interval": int(args.part_i_control_refresh_interval),
            "checkpoint_name": str(args.part_i_checkpoint),
            "checkpoint_optimizer": ckpt.get("checkpoint_optimizer", ""),
            "model_seed_scheme_key": model_seed_key,
            "candidate_downstream": int(f_scheme in F_CANDIDATE_SCHEMES),
            "held_nll_initial": held_before["loss"],
            "held_nll_final": held_after["loss"],
            "held_nll_delta": held_after["loss"] - held_before["loss"],
            "held_accuracy_initial": held_before["accuracy"],
            "held_accuracy_final": held_after["accuracy"],
            "held_accuracy_improvement": held_after["accuracy"] - held_before["accuracy"],
            "held_coverage_initial": held_before["coverage"],
            "held_coverage_final": held_after["coverage"],
            "held_coverage_improvement": held_after["coverage"] - held_before["coverage"],
            "train_loss_delta": train_after["loss"] - train_before["loss"],
            "guard_loss_delta": guard_after["loss"] - guard_before["loss"],
            "real_no_debt": no_debt_ok(held_debt, float(args.no_debt_budget)),
            "Brier_delta": held_debt["Brier_delta"],
            "ECE_delta": held_debt["ECE_delta"],
            "tail95_delta": held_debt["tail95_delta"],
            "tail99_delta": held_debt["tail99_delta"],
            "margin10_delta": held_debt["margin10_delta"],
            "margin10_delta_correct_sign": int(held_debt["margin10_delta"] >= -float(args.no_debt_budget)),
            "accept_rate": finite_accept / max(1, int(args.part_i_steps)),
            "scale_mean": mean(finite_scales),
            "skip_count": finite_skip,
            "finite_step_accept_count": finite_accept,
            "finite_step_skip_count": finite_skip,
            "finite_step_reject_reasons": json.dumps(reject_reasons, sort_keys=True),
            "activation_drift_median": median(activation_drift_curve),
            "activation_drift_max": max(activation_drift_curve or [0.0]),
            "solve_residual_max": max(solve_residuals or [0.0]),
            "condition_number_median": median(conds),
            "cg_iteration_median": median(cg_iters),
            "same_compute_candidate_count": same_compute_candidate_count,
            "beats_FunctionalGram_NLL": "",
            "beats_FunctionalGram_coverage": "",
            "random_projected_gap": "",
            "overhead_ratio": "",
            "held_test_usage": 0,
            "used_fake_data_rows": int(bundle["used_fake_data"]),
            "runtime_selector_used": 0,
            "wall_time_s": time.time() - start,
        }
    except Exception as exc:
        return {
            "part": "I",
            "status": "error",
            "i_scheme": str(i_scheme),
            "dataset": str(dataset),
            "seed": int(seed),
            "error_message": repr(exc),
            "held_test_usage": 0,
            "used_fake_data_rows": 0,
            "runtime_selector_used": 0,
            "wall_time_s": time.time() - start,
        }


def part_i(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pf = read_json(OUT_ROOT / "part_f_summary.json")
    ph = read_json(OUT_ROOT / "part_h_summary.json")
    if not ival(pf.get("gate_pass")) and not bool(int(args.part_i_allow_without_prereq)):
        return blocked_summary("I", "I_BlockedByPartF", "part_f_not_passed")
    if not ival(ph.get("gate_pass")) and not bool(int(args.part_i_allow_without_prereq)):
        return blocked_summary("I", "I_BlockedByPartH", "part_h_not_passed")
    device = device_from_args(args)
    jobs = shard_items(part_i_jobs(args), args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = OUT_ROOT / f"part_i_real_task_preflight_matrix{suffix}.csv"
    rows: list[dict[str, Any]] = []
    existing = read_rows(matrix) if int(args.part_i_resume) else []
    done = {(r.get("i_scheme"), r.get("dataset"), r.get("seed"), r.get("train_steps"), r.get("residual_target")) for r in existing if r.get("status") in {"ok", "skipped"}}
    for job in jobs:
        cfg = part_i_scheme_config(job[0], args)
        residual_target = str(cfg.get("residual_target", "not_applicable"))
        key = (job[0], job[1], str(job[2]), str(int(args.part_i_steps)), residual_target)
        if key in done:
            continue
        rows.append(run_part_i_row(job, args, device))
        if int(args.part_i_flush_every) and len(rows) % int(args.part_i_flush_every) == 0:
            append_rows(matrix, rows)
            print(json.dumps({"part": "I", "rows_written": len(read_rows(matrix)), "shard": suffix or "single", "total_jobs_in_shard": len(jobs)}, sort_keys=True), flush=True)
            rows = []
    if rows:
        append_rows(matrix, rows)
    all_rows = read_rows(matrix)
    summary = {
        "part": "I",
        "gate_pass": 0,
        "route": "PartIShardOnly" if suffix else "PartINeedsMerge",
        "dominant_blocker": "merge_required",
        "row_count": len(all_rows),
        "ok_rows": sum(1 for r in all_rows if r.get("status") == "ok"),
        "skipped_rows": sum(1 for r in all_rows if r.get("status") == "skipped"),
        "error_rows": sum(1 for r in all_rows if r.get("status") == "error"),
        "matrix": rel(matrix),
        "total_jobs_in_shard": len(jobs),
        "resume_used": int(args.part_i_resume),
    }
    write_json(OUT_ROOT / f"part_i_summary{suffix}.json", summary)
    append_exec("Part I", command_text(sys.argv), "done", files=rel(matrix), gpu=str(device), note=f"rows={len(all_rows)}")
    append_recap("Part I real-task preflight shard", summary)
    return summary


def part_i_merge(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    shard_paths = sorted(OUT_ROOT.glob("part_i_real_task_preflight_matrix_shard*_of_*.csv"))
    main_path = OUT_ROOT / "part_i_real_task_preflight_matrix.csv"
    rows = [r for path in shard_paths for r in read_rows(path)]
    rows.extend(read_rows(main_path))
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("i_scheme") or ""),
            str(row.get("dataset") or ""),
            str(row.get("seed") or ""),
            str(row.get("train_steps") or ""),
            str(row.get("residual_target") or ""),
        )
        dedup[key] = row
    rows = list(dedup.values())
    ok = [r for r in rows if r.get("status") == "ok"]
    skipped = [r for r in rows if r.get("status") == "skipped"]
    errors = [r for r in rows if r.get("status") == "error"]
    fg_by_key = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I0_FunctionalGram_baseline"}
    rand_by_key = {(r.get("dataset"), r.get("seed")): r for r in ok if r.get("i_scheme") == "I5_random_projected_residual_flow"}
    fg_wall_by_dataset: dict[str, float] = {}
    for dataset in sorted({str(r.get("dataset")) for r in ok}):
        fg_wall_by_dataset[dataset] = median(r.get("wall_time_s") for r in ok if r.get("dataset") == dataset and r.get("i_scheme") == "I0_FunctionalGram_baseline")
    for row in ok:
        fg = fg_by_key.get((row.get("dataset"), row.get("seed")))
        rand = rand_by_key.get((row.get("dataset"), row.get("seed")))
        if fg:
            row["beats_FunctionalGram_NLL"] = fval(fg.get("held_nll_delta")) - fval(row.get("held_nll_delta"))
            row["beats_FunctionalGram_coverage"] = fval(row.get("held_coverage_improvement")) - fval(fg.get("held_coverage_improvement"))
        if rand:
            row["random_projected_gap"] = fval(row.get("held_coverage_improvement")) - fval(rand.get("held_coverage_improvement"))
        fg_wall = fg_wall_by_dataset.get(str(row.get("dataset")), 0.0)
        row["overhead_ratio"] = fval(row.get("wall_time_s")) / max(fg_wall, 1.0e-12) if fg_wall > 0 else 99.0
    matrix = write_rows(OUT_ROOT / "part_i_real_task_preflight_matrix.csv", rows)
    groups: list[dict[str, Any]] = []
    for key in sorted({(r.get("i_scheme"), r.get("dataset")) for r in ok}):
        group = [r for r in ok if (r.get("i_scheme"), r.get("dataset")) == key]
        groups.append(
            {
                "i_scheme": key[0],
                "scheme": str(group[0].get("scheme", "")),
                "dataset": key[1],
                "dataset_family": str(group[0].get("dataset_family", "")),
                "rows": len(group),
                "held_nll_delta_median": median(r.get("held_nll_delta") for r in group),
                "held_accuracy_improvement_median": median(r.get("held_accuracy_improvement") for r in group),
                "held_coverage_improvement_median": median(r.get("held_coverage_improvement") for r in group),
                "train_loss_delta_median": median(r.get("train_loss_delta") for r in group),
                "guard_loss_delta_median": median(r.get("guard_loss_delta") for r in group),
                "real_no_debt_count": sum(ival(r.get("real_no_debt")) for r in group),
                "Brier_delta_median": median(r.get("Brier_delta") for r in group),
                "ECE_delta_median": median(r.get("ECE_delta") for r in group),
                "tail95_delta_median": median(r.get("tail95_delta") for r in group),
                "tail99_delta_median": median(r.get("tail99_delta") for r in group),
                "margin10_delta_correct_sign_count": sum(ival(r.get("margin10_delta_correct_sign")) for r in group),
                "beats_FunctionalGram_NLL_median": median(r.get("beats_FunctionalGram_NLL") for r in group),
                "beats_FunctionalGram_coverage_median": median(r.get("beats_FunctionalGram_coverage") for r in group),
                "random_projected_gap_median": median(r.get("random_projected_gap") for r in group),
                "accept_rate_median": median(r.get("accept_rate") for r in group),
                "scale_mean_median": median(r.get("scale_mean") for r in group),
                "skip_count_median": median(r.get("skip_count") for r in group),
                "activation_drift_max_median": median(r.get("activation_drift_max") for r in group),
                "overhead_ratio_median": median(r.get("overhead_ratio") for r in group),
                "wall_time_s_median": median(r.get("wall_time_s") for r in group),
                "held_test_usage": max(ival(r.get("held_test_usage")) for r in group),
                "used_fake_data_rows": max(ival(r.get("used_fake_data_rows")) for r in group),
                "runtime_selector_used": max(ival(r.get("runtime_selector_used")) for r in group),
            }
        )
    group_csv = write_rows(OUT_ROOT / "part_i_real_task_group_summary.csv", groups)
    by_group = {(str(g.get("i_scheme")), str(g.get("dataset"))): g for g in groups}
    visual_datasets = [d for d in ["MNIST", "FashionMNIST", "KMNIST"] if d in csv_items(args.part_i_datasets)]
    visual_pass_details: list[dict[str, Any]] = []
    for dataset in visual_datasets:
        official = by_group.get((str(args.part_i_official_scheme), dataset), {})
        fg = by_group.get(("I0_FunctionalGram_baseline", dataset), {})
        passed = int(
            bool(official and fg)
            and fval(official.get("held_nll_delta_median"), 999.0) <= fval(fg.get("held_nll_delta_median"), 999.0) + float(args.part_i_nll_tolerance)
            and fval(official.get("held_coverage_improvement_median"), -999.0) >= fval(fg.get("held_coverage_improvement_median"), 999.0) - float(args.part_i_coverage_tolerance)
            and fval(official.get("random_projected_gap_median"), -999.0) > 0.0
        )
        visual_pass_details.append({"dataset": dataset, "passed": passed, "official": official, "functionalgram": fg})
    visual_pass_count = sum(ival(d.get("passed")) for d in visual_pass_details)
    wine_official = by_group.get((str(args.part_i_official_scheme), "Wine"), {})
    wine_fg = by_group.get(("I0_FunctionalGram_baseline", "Wine"), {})
    wine_ok = int(
        "Wine" not in csv_items(args.part_i_datasets)
        or (
            bool(wine_official and wine_fg)
            and fval(wine_official.get("held_nll_delta_median"), 999.0) <= fval(wine_fg.get("held_nll_delta_median"), 999.0) + float(args.part_i_wine_nll_tolerance)
            and fval(wine_official.get("held_coverage_improvement_median"), -999.0) >= fval(wine_fg.get("held_coverage_improvement_median"), 999.0) - float(args.part_i_wine_coverage_tolerance)
        )
    )
    audit_clean = int(not any(ival(r.get("held_test_usage")) or ival(r.get("used_fake_data_rows")) or ival(r.get("runtime_selector_used")) for r in ok))
    gate = int(not errors and audit_clean and visual_pass_count >= 2 and wine_ok)
    if errors:
        blocker = "part_i_job_errors"
    elif not audit_clean:
        blocker = "audit_flags_failed"
    elif visual_pass_count < 2:
        blocker = "visual_transfer_not_enough"
    elif not wine_ok:
        blocker = "wine_catastrophic_degradation_vs_functionalgram"
    else:
        blocker = "none"
    route = "LimitedRealTaskPreflightPass_NotOfficial" if gate else "LimitedRealTaskPreflightFailed"
    summary = {
        "part": "I",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": len(ok),
        "skipped_rows": len(skipped),
        "error_rows": len(errors),
        "official_scheme": str(args.part_i_official_scheme),
        "visual_pass_count": visual_pass_count,
        "visual_pass_details": visual_pass_details,
        "wine_ok": wine_ok,
        "skipped_diagnostics": skipped,
        "errors": errors,
        "audit_clean": audit_clean,
        "matrix": rel(matrix),
        "group_summary": rel(group_csv),
        "promotion_allowed": 0,
        "note": "Part I is a limited real-task preflight only; it is not an official real-task promotion gate.",
    }
    write_json(OUT_ROOT / "part_i_summary.json", summary)
    write_json(OUT_ROOT / "part_i_real_task_preflight_summary.json", summary)
    next_actions(
        "i",
        route,
        blocker,
        [] if gate else ["if visual transfer fails, inspect whether real residual target should use held-free tail weighting or feature-space local targets", "if Wine degrades, separate tabular residual geometry from visual residual geometry before scaling"],
        [rel(matrix), rel(group_csv), rel(OUT_ROOT / "part_i_summary.json")],
    )
    append_exec("Part I merge", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_i_summary.json')}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}; visual_pass_count={visual_pass_count}")
    append_recap("Part I limited real-task preflight merge", summary)
    return summary


def blocked_summary(part: str, route: str, blocker: str) -> dict[str, Any]:
    summary = {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker}
    write_json(OUT_ROOT / f"part_{part.lower()}_summary.json", summary)
    next_actions(part, route, blocker, ["restore prerequisite artifact and rerun"], [])
    append_recap(f"Part {part} blocked", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    pa = read_json(OUT_ROOT / "part_a_summary.json")
    pb = read_json(OUT_ROOT / "part_b_history_lock_summary.json")
    pc = read_json(OUT_ROOT / "part_c_summary.json")
    pd = read_json(OUT_ROOT / "part_d_summary.json")
    pe = read_json(OUT_ROOT / "part_e_summary.json")
    pf = read_json(OUT_ROOT / "part_f_summary.json")
    pg = read_json(OUT_ROOT / "part_g_summary.json")
    ph = read_json(OUT_ROOT / "part_h_summary.json")
    pi = read_json(OUT_ROOT / "part_i_summary.json")
    if not ival(pa.get("gate_pass")):
        route, blocker = "A_FailedIdentity", str(pa.get("dominant_blocker", "part_a_failed"))
    elif not ival(pb.get("gate_pass")):
        route, blocker = "B_HistoryMissing", str(pb.get("dominant_blocker", "part_b_failed"))
    elif not ival(pc.get("gate_pass")):
        route, blocker = "C_DownstreamSolverToyFailed", str(pc.get("dominant_blocker", "part_c_failed"))
    elif pd and not ival(pd.get("gate_pass")):
        route, blocker = str(pd.get("route", "D_FrozenDownstreamResidualFailed")), str(pd.get("dominant_blocker", "part_d_failed"))
    elif pe and not ival(pe.get("gate_pass")):
        route, blocker = str(pe.get("route", "E_OneStepDownstreamActualFailed")), str(pe.get("dominant_blocker", "part_e_failed"))
    elif pg and ival(pg.get("gate_pass")) and pf and not ival(pf.get("gate_pass")):
        route, blocker = str(pf.get("route", "F_PositiveControlFailed")), str(pg.get("dominant_blocker", "part_g_explained"))
    elif pf and not ival(pf.get("gate_pass")):
        route, blocker = str(pf.get("route", "F_PositiveControlFailed")), str(pf.get("dominant_blocker", "part_f_failed"))
    elif pf and ival(pf.get("gate_pass")) and ph and not ival(ph.get("gate_pass")):
        route, blocker = "F_PositiveControlPassScalingBlocked", str(ph.get("dominant_blocker", "part_h_failed"))
    elif pf and ival(pf.get("gate_pass")) and ph and ival(ph.get("gate_pass")) and pi and ival(pi.get("gate_pass")):
        route, blocker = str(pi.get("route", "LimitedRealTaskPreflightPass_NotOfficial")), "none"
    elif pf and ival(pf.get("gate_pass")) and ph and ival(ph.get("gate_pass")) and pi and not ival(pi.get("gate_pass")):
        route, blocker = "F_PositiveControlPass_RealTaskPreflightFailed", str(pi.get("dominant_blocker", "part_i_failed"))
    elif pf and ival(pf.get("gate_pass")) and ph and ival(ph.get("gate_pass")):
        route, blocker = "F_PositiveControlPass_PartIAllowedNotRun", "part_i_not_executed"
    elif pf and ival(pf.get("gate_pass")):
        route, blocker = "F_PositiveControlPass_PartHRequired", "part_h_not_executed"
    elif pe and ival(pe.get("gate_pass")):
        route, blocker = "E_OneStepDownstreamActualPass_PartFRequired", "part_f_not_executed"
    elif pd and ival(pd.get("gate_pass")):
        route, blocker = "D_FrozenDownstreamResidualPass_PartERequired", "part_e_not_executed"
    else:
        route, blocker = "D_FrozenDownstreamResidualRequired", "part_d_not_executed"
    summary = {
        "final_route": route,
        "dominant_blocker": blocker,
        "part_a_gate_pass": ival(pa.get("gate_pass")),
        "part_b_gate_pass": ival(pb.get("gate_pass")),
        "part_c_gate_pass": ival(pc.get("gate_pass")),
        "part_d_gate_pass": ival(pd.get("gate_pass")),
        "part_e_gate_pass": ival(pe.get("gate_pass")),
        "part_f_gate_pass": ival(pf.get("gate_pass")),
        "part_g_gate_pass": ival(pg.get("gate_pass")),
        "part_h_gate_pass": ival(ph.get("gate_pass")),
        "part_i_gate_pass": ival(pi.get("gate_pass")),
        "promotion_allowed": 0,
        "evidence": {
            "part_a": rel(OUT_ROOT / "part_a_summary.json"),
            "part_b": rel(OUT_ROOT / "part_b_history_lock_summary.json"),
            "part_c": rel(OUT_ROOT / "part_c_summary.json"),
            "part_d": rel(OUT_ROOT / "part_d_summary.json"),
            "part_e": rel(OUT_ROOT / "part_e_summary.json"),
            "part_f": rel(OUT_ROOT / "part_f_summary.json"),
            "part_g": rel(OUT_ROOT / "part_g_summary.json"),
            "part_h": rel(OUT_ROOT / "part_h_summary.json"),
            "part_i": rel(OUT_ROOT / "part_i_summary.json"),
        },
    }
    write_json(OUT_ROOT / "final_route.json", summary)
    append_exec("Finalize", command_text(sys.argv), "done", files=rel(OUT_ROOT / "final_route.json"), gpu=str(args.device), note=f"route={route}; blocker={blocker}")
    append_recap("Final route decision", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--basis-key", default="dche_k9")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--train-size", type=int, default=768)
    p.add_argument("--guard-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--checkpoint-steps", type=int, default=20)
    p.add_argument("--checkpoint-lr", type=float, default=0.02)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--functional-gram-ridge", type=float, default=1.0e-6)
    p.add_argument("--population-beta", type=float, default=1.0)
    p.add_argument("--population-weight-mode", default="sample_coherence")
    p.add_argument("--cg-tol", type=float, default=1.0e-6)
    p.add_argument("--cg-max-iter", type=int, default=512)
    p.add_argument("--v23-07-root", default="results/v23_07_rank_repair_train768")
    p.add_argument("--part-c-lambda", type=float, default=1.0e-2)
    p.add_argument("--part-d-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-d-seed-count", type=int, default=3)
    p.add_argument("--part-d-checkpoints", default="functionalgram_partial,h10_isomorphic_partial,random_init")
    p.add_argument("--part-d-residual-targets", default="norm")
    p.add_argument("--part-d-schemes", default=",".join(PART_D_SCHEMES))
    p.add_argument("--part-d-lambda", type=float, default=1.0)
    p.add_argument("--part-d-solver", default="exact")
    p.add_argument("--part-d-basis-input-gain", type=float, default=0.25)
    p.add_argument("--part-d-flush-every", type=int, default=1)
    p.add_argument("--part-d-resume", type=int, default=1)
    p.add_argument("--part-e-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-e-seed-count", type=int, default=3)
    p.add_argument("--part-e-checkpoints", default="h10_isomorphic_partial")
    p.add_argument("--part-e-residual-targets", default="trust_scaled")
    p.add_argument("--part-e-schemes", default=",".join(PART_E_SCHEMES))
    p.add_argument("--part-e-lambda", type=float, default=1.0)
    p.add_argument("--part-e-alphas", default="1,0.5,0.25,0.125,0.0625")
    p.add_argument("--part-e-delta-scale", type=float, default=1.0)
    p.add_argument("--part-e-delta-sign", type=float, default=1.0)
    p.add_argument("--part-e-trust-mode", default="loss")
    p.add_argument("--part-e-trust-tolerance", type=float, default=0.0)
    p.add_argument("--part-e-local-reference-target", default="norm")
    p.add_argument("--part-e-functionalgram-lr", type=float, default=0.02)
    p.add_argument("--part-e-sketch-rank", type=int, default=16)
    p.add_argument("--part-e-flush-every", type=int, default=1)
    p.add_argument("--part-e-resume", type=int, default=1)
    p.add_argument("--part-f-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-f-seed-count", type=int, default=3)
    p.add_argument("--part-f-steps", type=int, default=10)
    p.add_argument("--part-f-schemes", default=",".join(PART_F_SCHEMES))
    p.add_argument("--part-f-checkpoint", default="h10_isomorphic_partial")
    p.add_argument("--part-f-residual-target", default="norm")
    p.add_argument("--part-f-lambda", type=float, default=10.0)
    p.add_argument("--part-f-alphas", default="1,0.5,0.25,0.125")
    p.add_argument("--part-f-delta-scale", type=float, default=1.0)
    p.add_argument("--part-f-delta-sign", type=float, default=1.0)
    p.add_argument("--part-f-trust-mode", default="coverage_floor")
    p.add_argument("--part-f-trust-tolerance", type=float, default=0.0)
    p.add_argument("--part-f-local-reference-target", default="norm")
    p.add_argument("--part-f-functionalgram-lr", type=float, default=0.02)
    p.add_argument("--part-f-adam-lr", type=float, default=0.02)
    p.add_argument("--part-f-solver", default="exact")
    p.add_argument("--part-f-sketch-rank", type=int, default=16)
    p.add_argument("--part-f-activation-drift-cap", type=float, default=-1.0)
    p.add_argument("--part-f-alignment-min", type=float, default=-2.0)
    p.add_argument("--part-f-alignment-soft-floor", type=float, default=-2.0)
    p.add_argument("--part-f-alignment-soft-ceiling", type=float, default=1.0)
    p.add_argument("--part-f-alignment-soft-min-scale", type=float, default=0.25)
    p.add_argument("--part-f-alignment-soft-power", type=float, default=0.5)
    p.add_argument("--part-f-tail99-step-budget", type=float, default=-1.0)
    p.add_argument("--part-f-tail99-cumulative-budget", type=float, default=-1.0)
    p.add_argument("--part-f-model-seed-scheme-key", default="")
    p.add_argument("--part-f-tail-correction-scales", default="0.25,0.125,0.0625,0")
    p.add_argument("--part-f-tail-correction-target", default="tail_weighted_norm")
    p.add_argument("--part-f-tail-correction-layer-group", default="topdown")
    p.add_argument("--part-f-tail-correction-transport-min", type=float, default=0.90)
    p.add_argument("--part-f-flush-every", type=int, default=1)
    p.add_argument("--part-f-resume", type=int, default=1)
    p.add_argument("--part-h-exact-reference-scheme", default="F5_last_two_layers_topdown_EFRF")
    p.add_argument("--part-i-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--part-i-schemes", default=",".join(PART_I_SCHEMES))
    p.add_argument("--part-i-official-scheme", default="I3_DownstreamEFRF_best_positive_control_scheme")
    p.add_argument("--part-i-data-root", default=str(ROOT / "data"))
    p.add_argument("--part-i-seed-count", type=int, default=1)
    p.add_argument("--part-i-train-size", type=int, default=128)
    p.add_argument("--part-i-guard-size", type=int, default=64)
    p.add_argument("--part-i-held-size", type=int, default=64)
    p.add_argument("--part-i-test-size", type=int, default=1)
    p.add_argument("--part-i-steps", type=int, default=20)
    p.add_argument("--part-i-batch-size", type=int, default=0)
    p.add_argument("--part-i-basis", default="dche_k9")
    p.add_argument("--part-i-depth", type=int, default=3)
    p.add_argument("--part-i-basis-input-gain", type=float, default=1.0)
    p.add_argument("--part-i-checkpoint", default="h10_isomorphic_partial")
    p.add_argument("--part-i-checkpoint-steps", type=int, default=20)
    p.add_argument("--part-i-checkpoint-lr", type=float, default=0.02)
    p.add_argument("--part-i-residual-target", default="norm")
    p.add_argument("--part-i-without-wpop-residual-target", default="raw")
    p.add_argument("--part-i-lambda", type=float, default=10.0)
    p.add_argument("--part-i-alphas", default="1,0.5,0.25,0.125,0.0625,0.03125")
    p.add_argument("--part-i-delta-scale", type=float, default=1.0)
    p.add_argument("--part-i-delta-sign", type=float, default=1.0)
    p.add_argument("--part-i-trust-mode", default="coverage_floor")
    p.add_argument("--part-i-trust-tolerance", type=float, default=0.0)
    p.add_argument("--part-i-local-reference-target", default="norm")
    p.add_argument("--part-i-functionalgram-lr", type=float, default=0.0075)
    p.add_argument("--part-i-adam-lr", type=float, default=0.02)
    p.add_argument("--part-i-solver", default="exact")
    p.add_argument("--part-i-sketch-rank", type=int, default=16)
    p.add_argument("--part-i-control-refresh-interval", type=int, default=30)
    p.add_argument("--part-i-activation-drift-cap", type=float, default=-1.0)
    p.add_argument("--part-i-alignment-min", type=float, default=-2.0)
    p.add_argument("--part-i-alignment-soft-floor", type=float, default=-2.0)
    p.add_argument("--part-i-alignment-soft-ceiling", type=float, default=1.0)
    p.add_argument("--part-i-alignment-soft-min-scale", type=float, default=0.25)
    p.add_argument("--part-i-alignment-soft-power", type=float, default=0.5)
    p.add_argument("--part-i-tail99-step-budget", type=float, default=-1.0)
    p.add_argument("--part-i-tail99-cumulative-budget", type=float, default=-1.0)
    p.add_argument("--part-i-model-seed-scheme-key", default="part_i_common_init")
    p.add_argument("--part-i-nll-tolerance", type=float, default=0.0)
    p.add_argument("--part-i-coverage-tolerance", type=float, default=0.0)
    p.add_argument("--part-i-wine-nll-tolerance", type=float, default=0.05)
    p.add_argument("--part-i-wine-coverage-tolerance", type=float, default=0.05)
    p.add_argument("--part-i-allow-without-prereq", type=int, default=0)
    p.add_argument("--part-i-flush-every", type=int, default=1)
    p.add_argument("--part-i-resume", type=int, default=1)
    p.add_argument("--no-debt-budget", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode).lower()
    if mode in {"a", "part-a"}:
        return part_a(args)
    if mode in {"b", "part-b"}:
        return part_b(args)
    if mode in {"c", "part-c"}:
        return part_c(args)
    if mode in {"d", "part-d"}:
        return part_d(args)
    if mode in {"d-merge", "part-d-merge"}:
        return part_d_merge(args)
    if mode in {"e", "part-e"}:
        return part_e(args)
    if mode in {"e-merge", "part-e-merge"}:
        return part_e_merge(args)
    if mode in {"f", "part-f"}:
        return part_f(args)
    if mode in {"f-merge", "part-f-merge"}:
        return part_f_merge(args)
    if mode in {"g", "part-g"}:
        return part_g(args)
    if mode in {"h", "part-h"}:
        return part_h(args)
    if mode in {"i", "part-i"}:
        return part_i(args)
    if mode in {"i-merge", "part-i-merge"}:
        return part_i_merge(args)
    if mode == "finalize":
        return finalize(args)
    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
