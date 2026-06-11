#!/usr/bin/env python3
"""v21 function-space target contrast diagnostics.

This implements the plan 13.5 checks for high-ActuationR2/source-fail cases:
train-source target vs random matched target, sign-flipped target, corrupted
target, B1-only vs B1+B2 target, and lower-rank target projection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.core import flat_grad, flat_params, load_flat_params  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, resolve_device  # noqa: E402
from experiments.run_v21_common import PYTHON, append_exec, ensure_out, finite_float, write_rows  # noqa: E402


TARGET_SPECS = [
    ("T1-loss-cotangent-B1B2", "loss", "b1b2", 0),
    ("T1b-loss-cotangent-B1-only", "loss", "b1", 0),
    ("T1c-loss-cotangent-B1B2-rank8", "loss", "b1b2", 8),
    ("T5-random-matched-target", "random", "b1b2", 0),
    ("T6-sign-flipped-target", "sign_flip", "b1b2", 0),
    ("T7-corrupted-label-target", "corrupt", "b1b2", 0),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--carriers", default="D-CHE,D-FOU")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=4800)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--alt-period", type=int, default=50)
    p.add_argument("--source-warmup-steps", type=int, default=0)
    p.add_argument("--basis-repair-variant", default="CHE-R4-k3-gradbuf-triton")
    p.add_argument("--basis-repair-variant-fou", default="FOU-R4-k4-triton-no-materialize")
    p.add_argument("--run-label", default="v21_target_contrast")
    return p


def one_hot_like(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(logits)
    target.scatter_(1, y.reshape(-1, 1), 1.0)
    return target


def find_readout_param(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter | None]:
    for pname, param in model.named_parameters():
        if "w2" in pname.lower() and param.ndim == 3:
            return pname, param
    return "", None


def split_batch(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    split1 = max(1, n // 3)
    split2 = max(split1 + 1, (2 * n) // 3)
    xb1, yb1 = x[:split1], y[:split1]
    xb2, yb2 = x[split1:split2], y[split1:split2]
    xb3, yb3 = x[split2:], y[split2:]
    if int(xb3.shape[0]) == 0:
        xb3, yb3 = xb2, yb2
    return xb1, yb1, xb2, yb2, xb3, yb3


def target_from_logits(kind: str, logits: torch.Tensor, labels: torch.Tensor, *, generator: torch.Generator, classes: int, reference: torch.Tensor | None = None) -> torch.Tensor:
    loss_target = one_hot_like(logits, labels) - torch.softmax(logits, dim=1)
    if kind == "loss":
        return loss_target
    if kind == "sign_flip":
        return -loss_target
    if kind == "corrupt":
        corrupt = (labels + 1) % max(2, int(classes))
        return one_hot_like(logits, corrupt) - torch.softmax(logits, dim=1)
    if kind == "random":
        noise = torch.randn(loss_target.shape, device=logits.device, dtype=logits.dtype, generator=generator)
        ref = reference if reference is not None else loss_target
        return noise * (torch.linalg.vector_norm(ref).clamp_min(1.0e-12) / torch.linalg.vector_norm(noise).clamp_min(1.0e-12))
    raise ValueError(f"unknown target kind: {kind}")


def solve_delta(feats: torch.Tensor, target: torch.Tensor, rank_limit: int) -> torch.Tensor:
    fit_feats = feats
    selected = None
    if int(rank_limit) > 0 and feats.shape[1] > int(rank_limit):
        norms = torch.linalg.vector_norm(feats.detach().float(), dim=0)
        selected = torch.topk(norms, k=int(rank_limit)).indices
        fit_feats = feats[:, selected]
    gram = fit_feats.T @ fit_feats + 1.0e-3 * torch.eye(fit_feats.shape[1], device=feats.device, dtype=fit_feats.dtype)
    rhs = fit_feats.T @ target.to(device=feats.device, dtype=fit_feats.dtype)
    try:
        delta_small = torch.linalg.solve(gram, rhs)
    except Exception:
        delta_small = torch.linalg.lstsq(gram, rhs).solution
    if selected is None:
        return delta_small
    out = torch.zeros(feats.shape[1], target.shape[1], device=feats.device, dtype=fit_feats.dtype)
    out[selected] = delta_small
    return out


def measure_target(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, target_id: str, target_kind: str, fit_scope: str, rank_limit: int, seed: int) -> dict[str, Any]:
    device = x.device
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    w2_name, w2_param = find_readout_param(model)
    g_ref = flat_grad(model, device)
    if w2_param is None or not hasattr(model, "frozen_readout_features") or int(x.shape[0]) < 3:
        return {"target_id": target_id, "operator_status": "exact_readout_unavailable"}
    xb1, yb1, xb2, yb2, xb3, yb3 = split_batch(x, y)
    base_params = flat_params(model).detach().to(device=device)
    gen = torch.Generator(device=device).manual_seed(771_000 + int(seed) + sum(ord(c) for c in target_id))

    with torch.no_grad():
        load_flat_params(model, base_params)
        base1 = model(xb1).detach().float()
        base2 = model(xb2).detach().float()
        base3 = model(xb3).detach().float()
        feats1 = model.frozen_readout_features(xb1).detach().float()
        feats2 = model.frozen_readout_features(xb2).detach().float()
        hdim = int(w2_param.shape[0])
        classes = int(w2_param.shape[1])
        kval = int(w2_param.shape[2])
        n_readout = hdim * kval
        feats1 = feats1[:, :n_readout] / float(max(1, hdim)) ** 0.5
        feats2 = feats2[:, :n_readout] / float(max(1, hdim)) ** 0.5
        ref1 = one_hot_like(base1, yb1) - torch.softmax(base1, dim=1)
        ref2 = one_hot_like(base2, yb2) - torch.softmax(base2, dim=1)
        target1 = target_from_logits(target_kind, base1, yb1, generator=gen, classes=int(args.classes), reference=ref1)
        target2 = target_from_logits(target_kind, base2, yb2, generator=gen, classes=int(args.classes), reference=ref2)
        if fit_scope == "b1":
            fit_feats = feats1
            fit_target = target1
        else:
            fit_feats = torch.cat([feats1, feats2], dim=0)
            fit_target = torch.cat([target1, target2], dim=0)
        measure_target_flat = torch.cat([target1.reshape(-1), target2.reshape(-1)], dim=0)
        delta_matrix = solve_delta(fit_feats, fit_target, rank_limit)
        delta_w2 = delta_matrix.reshape(hdim, kval, classes).permute(0, 2, 1).contiguous()
        b1_before = F.cross_entropy(base1, yb1)
        b2_before = F.cross_entropy(base2, yb2)
        b3_before = F.cross_entropy(base3, yb3)
        for pname, param in params:
            if pname == w2_name:
                param.add_(delta_w2.to(device=param.device, dtype=param.dtype))
                break
        after1 = model(xb1).detach().float()
        after2 = model(xb2).detach().float()
        after3 = model(xb3).detach().float()
        b1_after = F.cross_entropy(after1, yb1)
        b2_after = F.cross_entropy(after2, yb2)
        b3_after = F.cross_entropy(after3, yb3)
        actual = torch.cat([(after1 - base1).reshape(-1), (after2 - base2).reshape(-1)], dim=0)
        residual = measure_target_flat.to(dtype=actual.dtype) - actual
        centered = measure_target_flat.to(dtype=actual.dtype) - measure_target_flat.to(dtype=actual.dtype).mean()
        denom = torch.sum(centered.square()).clamp_min(1.0e-12)
        actuation_r2 = 1.0 - torch.sum(residual.square()) / denom
        cos_denom = torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(measure_target_flat.to(dtype=actual.dtype))
        actuation_cos = (actual @ measure_target_flat.to(dtype=actual.dtype) / cos_denom.clamp_min(1.0e-12)).clamp(-1.0, 1.0)
        load_flat_params(model, base_params)
    return {
        "target_id": target_id,
        "target_kind": target_kind,
        "fit_scope": fit_scope,
        "rank_limit": int(rank_limit),
        "operator_status": "measured",
        "operator_rank": int(delta_matrix.shape[0]),
        "ActuationR2": float(actuation_r2.item()),
        "ActuationCosine": float(actuation_cos.item()),
        "projection_residual_norm": float(torch.linalg.vector_norm(residual).item() / torch.linalg.vector_norm(measure_target_flat).clamp_min(1.0e-12).item()),
        "B1_gain": float((b1_before - b1_after).item()),
        "B2_transfer_gain": float((b2_before - b2_after).item()),
        "B3_safety_gain": float((b3_before - b3_after).item()),
        "function_displacement_norm": float(torch.linalg.vector_norm(actual).item()),
        "target_norm": float(torch.linalg.vector_norm(measure_target_flat).item()),
        "parameter_delta_norm": float(torch.linalg.vector_norm(delta_w2).item()),
        "grad_ref_norm": float(torch.linalg.vector_norm(g_ref.detach()).item()) if g_ref.numel() else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("carrier")), str(row.get("basis_repair_variant")), str(row.get("target_id"))), []).append(row)
    summary: list[dict[str, Any]] = []
    by_carrier_random: dict[tuple[str, str], float] = {}
    for (carrier, variant, target_id), group in sorted(groups.items()):
        def mean(key: str) -> float:
            vals = [finite_float(r.get(key)) for r in group]
            vals = [v for v in vals if v == v]
            return sum(vals) / len(vals) if vals else float("nan")

        item = {
            "carrier": carrier,
            "basis_repair_variant": variant,
            "target_id": target_id,
            "rows": len(group),
            "target_kind": group[0].get("target_kind", "") if group else "",
            "fit_scope": group[0].get("fit_scope", "") if group else "",
            "rank_limit": group[0].get("rank_limit", "") if group else "",
            "ActuationR2_mean": mean("ActuationR2"),
            "B1_gain_mean": mean("B1_gain"),
            "B2_transfer_gain_mean": mean("B2_transfer_gain"),
            "B3_safety_gain_mean": mean("B3_safety_gain"),
            "projection_residual_norm_mean": mean("projection_residual_norm"),
        }
        if target_id == "T5-random-matched-target":
            by_carrier_random[(carrier, variant)] = finite_float(item["B2_transfer_gain_mean"], 0.0)
        summary.append(item)
    for item in summary:
        random_b2 = by_carrier_random.get((str(item.get("carrier")), str(item.get("basis_repair_variant"))), float("nan"))
        b2 = finite_float(item.get("B2_transfer_gain_mean"))
        item["random_target_B2_gain_mean"] = random_b2 if random_b2 == random_b2 else ""
        item["B2_gain_minus_random"] = b2 - random_b2 if b2 == b2 and random_b2 == random_b2 else ""
        item["target_success_like"] = int(
            str(item.get("target_id")) == "T1-loss-cotangent-B1B2"
            and finite_float(item.get("ActuationR2_mean"), -1.0) >= 0.50
            and b2 == b2
            and random_b2 == random_b2
            and b2 > random_b2 + 0.005
        )
    decisions = []
    for (carrier, variant), random_b2 in sorted(by_carrier_random.items()):
        carrier_rows = [r for r in summary if str(r.get("carrier")) == carrier and str(r.get("basis_repair_variant")) == variant]
        loss = next((r for r in carrier_rows if r.get("target_id") == "T1-loss-cotangent-B1B2"), {})
        sign = next((r for r in carrier_rows if r.get("target_id") == "T6-sign-flipped-target"), {})
        corrupt = next((r for r in carrier_rows if r.get("target_id") == "T7-corrupted-label-target"), {})
        b1only = next((r for r in carrier_rows if r.get("target_id") == "T1b-loss-cotangent-B1-only"), {})
        rank8 = next((r for r in carrier_rows if r.get("target_id") == "T1c-loss-cotangent-B1B2-rank8"), {})
        loss_b2 = finite_float(loss.get("B2_transfer_gain_mean"))
        decisions.append(
            {
                "carrier": carrier,
                "basis_repair_variant": variant,
                "loss_target_ActuationR2_mean": loss.get("ActuationR2_mean", ""),
                "loss_target_B2_gain_mean": loss.get("B2_transfer_gain_mean", ""),
                "random_target_B2_gain_mean": random_b2,
                "sign_flipped_B2_gain_mean": sign.get("B2_transfer_gain_mean", ""),
                "corrupted_target_B2_gain_mean": corrupt.get("B2_transfer_gain_mean", ""),
                "B1_only_B2_gain_mean": b1only.get("B2_transfer_gain_mean", ""),
                "rank8_B2_gain_mean": rank8.get("B2_transfer_gain_mean", ""),
                "loss_beats_random_by_0p005": int(loss_b2 == loss_b2 and loss_b2 > random_b2 + 0.005),
                "target_observable_success": int(any(int(r.get("target_success_like", 0)) for r in carrier_rows)),
                "target_observable_no_go": int(not any(int(r.get("target_success_like", 0)) for r in carrier_rows)),
                "promotion_allowed": 0,
            }
        )
    return summary, decisions


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = resolve_device(args.device)
    carriers = [x.strip() for x in str(args.carriers).split(",") if x.strip()]
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    total = len(carriers) * len(datasets) * len(seeds) * len(TARGET_SPECS)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_function_space_target_contrast.py --out-dir {out_dir} --device {args.device}", status="started", note=f"rows_planned={total}")
    rows: list[dict[str, Any]] = []
    for carrier in carriers:
        repair = str(args.basis_repair_variant_fou) if carrier == "D-FOU" else str(args.basis_repair_variant)
        for dataset in datasets:
            for seed in seeds:
                started = time.perf_counter()
                try:
                    x_train, y_train, _x_val, _y_val = load_dataset(dataset, Path(args.data_root), int(args.train_size), int(args.val_size), int(seed), device, int(args.input_size))
                    local = deepcopy(args)
                    local.basis_repair_variant = repair
                    model_seed = 221_000 + int(seed) + 97 * len(rows)
                    model = carrier_model(carrier, x_train, model_seed, local, device)
                    x_batch = x_train[: min(int(args.batch_size), int(x_train.shape[0]))]
                    y_batch = y_train[: min(int(args.batch_size), int(y_train.shape[0]))]
                    for target_id, kind, fit_scope, rank_limit in TARGET_SPECS:
                        item = measure_target(model, x_batch, y_batch, local, target_id, kind, fit_scope, rank_limit, model_seed)
                        rows.append(
                            {
                                "run_label": str(args.run_label),
                                "carrier": carrier,
                                "basis_repair_variant": repair,
                                "dataset": dataset,
                                "seed": seed,
                                **item,
                                "runtime_sec_group": time.perf_counter() - started,
                                "execution_status": "measured",
                                "blocker": "",
                                "promotion_allowed": 0,
                            }
                        )
                except Exception as exc:
                    for target_id, kind, fit_scope, rank_limit in TARGET_SPECS:
                        rows.append(
                            {
                                "run_label": str(args.run_label),
                                "carrier": carrier,
                                "basis_repair_variant": repair,
                                "dataset": dataset,
                                "seed": seed,
                                "target_id": target_id,
                                "target_kind": kind,
                                "fit_scope": fit_scope,
                                "rank_limit": rank_limit,
                                "runtime_sec_group": time.perf_counter() - started,
                                "execution_status": f"blocked:{type(exc).__name__}",
                                "blocker": str(exc)[:500],
                                "promotion_allowed": 0,
                            }
                        )
    summary, decisions = summarize(rows)
    write_rows(out_dir / "v21_function_space_target_contrast_matrix.csv", rows)
    write_rows(out_dir / "v21_function_space_target_contrast_summary.csv", summary)
    write_rows(out_dir / "v21_function_space_target_contrast_decision.csv", decisions)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_function_space_target_contrast.py --out-dir {out_dir}", status="completed", note=f"rows={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
