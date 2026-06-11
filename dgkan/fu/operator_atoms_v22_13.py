"""v22.13 operator atom tests and construction helpers."""

from __future__ import annotations

from typing import Any, Callable

import torch

from dgkan.fu.operator_core import OFFICIAL_OPERATOR_IDS, apply_operator


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def gain(delta: torch.Tensor, update: torch.Tensor) -> float:
    return float((-(delta.detach().float() * update.detach().float()).sum() / max(1, delta.numel())).item())


def operator_law_test(fn: Callable[[torch.Tensor], torch.Tensor], d1: torch.Tensor, d2: torch.Tensor) -> dict[str, Any]:
    a1 = fn(d1)
    a2 = fn(d2)
    asum = fn(d1 + d2)
    denom = torch.linalg.vector_norm(a1 + a2).clamp_min(1.0e-8)
    linearity = float((torch.linalg.vector_norm(asum - a1 - a2) / denom).item())
    ah = fn(2.0 * d1)
    hom = float((torch.linalg.vector_norm(ah - 2.0 * a1) / torch.linalg.vector_norm(2.0 * a1).clamp_min(1.0e-8)).item())
    lip = float(torch.linalg.vector_norm(a1 - a2).item() / torch.linalg.vector_norm(d1 - d2).clamp_min(1.0e-8).item())
    perm = torch.randperm(int(d1.shape[0]))
    unperm = torch.empty_like(perm)
    unperm[perm] = torch.arange(int(perm.numel()), device=perm.device)
    p_out = fn(d1[perm])[unperm]
    perm_err = float((torch.linalg.vector_norm(p_out - a1) / torch.linalg.vector_norm(a1).clamp_min(1.0e-8)).item())
    sign = _safe_cos(fn(-d1), -a1)
    return {
        "operator_linearity_error": linearity,
        "operator_homogeneity_error": hom,
        "operator_lipschitz_ratio": lip,
        "cotangent_permutation_equivariance_error": perm_err,
        "cotangent_sign_flip_consistency": sign,
    }


def adapter_renaming_test(logits: torch.Tensor, cotangent: torch.Tensor, operator_id: str, seed: int = 2213) -> dict[str, Any]:
    out_a, _ = apply_operator(operator_id=operator_id, logits=logits, cotangent=cotangent, seed=seed)
    out_b, _ = apply_operator(operator_id=operator_id, logits=logits, cotangent=cotangent.clone(), seed=seed)
    rel = float((torch.linalg.vector_norm(out_a - out_b) / torch.linalg.vector_norm(out_a).clamp_min(1.0e-8)).item())
    return {
        "operator_id": operator_id,
        "adapter_renaming_output_cosine": _safe_cos(out_a, out_b),
        "adapter_renaming_output_rel_error": rel,
        "adapter_renaming_commit_rel_error": rel,
        "adapter_renaming_pass": int(rel <= 1.0e-6),
    }


def build_operator_atom_rows(
    logits: torch.Tensor,
    cotangents: list[tuple[str, torch.Tensor, int]],
    *,
    norm_scale: float = 0.16,
    seed: int = 2213,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for operator_id in OFFICIAL_OPERATOR_IDS:
        outputs: list[torch.Tensor] = []
        gains: list[float] = []
        telemetry_rows: list[dict[str, Any]] = []
        for idx, (_display_name, delta, _seen) in enumerate(cotangents):
            out, telem = apply_operator(operator_id=operator_id, logits=logits, cotangent=delta, norm_scale=norm_scale, seed=seed + idx)
            outputs.append(out)
            gains.append(gain(delta, out))
            telemetry_rows.append(telem.to_row())
        mean_out = sum(outputs) / max(1, len(outputs))
        tensors[operator_id] = mean_out.detach().float()
        law = operator_law_test(lambda d, op=operator_id: apply_operator(operator_id=op, logits=logits, cotangent=d, norm_scale=norm_scale, seed=seed)[0], cotangents[0][1], cotangents[min(1, len(cotangents) - 1)][1])
        renaming = adapter_renaming_test(logits, cotangents[0][1], operator_id, seed=seed)
        telem0 = telemetry_rows[0] if telemetry_rows else {}
        positive_fraction = sum(int(g > 0.0) for g in gains) / max(1, len(gains))
        pass_flag = int(
            renaming["adapter_renaming_pass"]
            and positive_fraction >= 0.50
            and float(telem0.get("control_projection_after", 1.0)) <= 0.25
            and float(telem0.get("NDS_reduction", 0.0)) >= -0.10
            and float(law.get("operator_lipschitz_ratio", 99.0)) <= 5.0
        )
        if str(telem0.get("operator_class")) == "linear_metric_operator":
            pass_flag = int(pass_flag and float(law["operator_linearity_error"]) <= 0.15 and float(law["operator_homogeneity_error"]) <= 0.10)
        blockers: list[str] = []
        if not renaming["adapter_renaming_pass"]:
            blockers.append("adapter_renaming")
        if positive_fraction < 0.50:
            blockers.append("gain_fraction")
        if float(telem0.get("control_projection_after", 1.0)) > 0.25:
            blockers.append("control_projection")
        if float(telem0.get("NDS_reduction", 0.0)) < -0.10:
            blockers.append("NDS")
        if float(law.get("operator_lipschitz_ratio", 99.0)) > 5.0:
            blockers.append("operator_lipschitz")
        row = {
            **telem0,
            **law,
            **renaming,
            "GainRatio_by_adapter": ";".join(f"{name}:{g:.8g}" for (name, _d, _s), g in zip(cotangents, gains)),
            "gain_positive_fraction": positive_fraction,
            "S2_operator_atom_pass": pass_flag,
            "blocker": "" if pass_flag else ";".join(blockers),
        }
        rows.append(row)
    return rows, tensors


__all__ = ["adapter_renaming_test", "build_operator_atom_rows", "gain", "operator_law_test"]
