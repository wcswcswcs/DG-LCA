#!/usr/bin/env python
"""v12.24 train-stream/precommit compensation bridge.

The script deliberately avoids the v12.23 query-batch compensation path.  It
uses train-stream micro-batches to build direct-readout compensation deltas,
then audits source/noop/control P4 behavior and multi-sketch LineC stability.
All results are audit artifacts; promotion is decided by the v12.24 aggregator.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m
import experiments.run_v1223_p4_trainable_role_scan as role_scan


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def auc_error_time(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    total = max(points[-1][0] - points[0][0], 1.0e-9)
    area = 0.0
    for (t0, a0), (t1, a1) in zip(points[:-1], points[1:]):
        area += max(t1 - t0, 0.0) * (((1.0 - a0) + (1.0 - a1)) / 2.0)
    return float(area / total)


def direct_delta(model: torch.nn.Module, x_ref: torch.Tensor, base_logits: torch.Tensor, ridge: float = 1.0e-3) -> tuple[torch.Tensor | None, float]:
    direct = getattr(model, "direct_readout", None)
    if not torch.is_tensor(direct) or not hasattr(model, "_direct_logits_and_features"):
        return None, float("nan")
    with torch.no_grad():
        logits_before = model(x_ref).detach()
        pre = float((logits_before - base_logits).abs().max().item())
        _dl, feats = model._direct_logits_and_features(x_ref)  # type: ignore[attr-defined]
        f = feats.detach().float()
        target = (base_logits - logits_before).detach().float()
        gram = f @ f.transpose(0, 1)
        eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
        coef = torch.linalg.solve(gram + float(ridge) * eye, target)
        scale = math.sqrt(max(1, int(getattr(model, "input_dim", f.shape[1]))))
        delta = (f.transpose(0, 1) @ coef) * float(scale)
    if tuple(delta.shape) != tuple(direct.shape):
        return None, pre
    return delta.detach(), pre


def combine_deltas(deltas: list[torch.Tensor], mode: str) -> torch.Tensor | None:
    if not deltas:
        return None
    stack = torch.stack([d.float() for d in deltas], dim=0)
    if mode == "ema":
        weights = torch.tensor([0.5 ** (len(deltas) - i - 1) for i in range(len(deltas))], device=stack.device, dtype=stack.dtype)
        weights = weights / weights.sum().clamp_min(1.0e-12)
        return (stack * weights.view(-1, *([1] * (stack.dim() - 1)))).sum(dim=0)
    if mode == "median":
        return stack.median(dim=0).values
    if mode == "trimmed_mean" and len(deltas) >= 3:
        sorted_stack, _ = stack.sort(dim=0)
        return sorted_stack[1:-1].mean(dim=0)
    return stack.mean(dim=0)


def apply_train_stream_compensation(
    model: torch.nn.Module,
    refs: list[tuple[torch.Tensor, torch.Tensor]],
    mode: str,
) -> dict[str, Any]:
    direct = getattr(model, "direct_readout", None)
    if not torch.is_tensor(direct):
        return {
            "direct_logit_compensation_applied": 0,
            "direct_logit_compensation_mode": mode,
            "direct_logit_compensation_pre_drift": "",
            "direct_logit_compensation_post_drift": "",
            "direct_logit_compensation_norm": "",
        }
    deltas: list[torch.Tensor] = []
    pres: list[float] = []
    for x_ref, base_logits in refs:
        delta, pre = direct_delta(model, x_ref, base_logits)
        pres.append(pre)
        if delta is not None:
            deltas.append(delta)
    combined = combine_deltas(deltas, mode)
    if combined is None:
        return {
            "direct_logit_compensation_applied": 0,
            "direct_logit_compensation_mode": mode,
            "direct_logit_compensation_pre_drift": max([p for p in pres if math.isfinite(p)] or [float("nan")]),
            "direct_logit_compensation_post_drift": "",
            "direct_logit_compensation_norm": "",
        }
    with torch.no_grad():
        direct.add_(combined.to(device=direct.device, dtype=direct.dtype))
        posts = [float((model(x_ref).detach() - base_logits).abs().max().item()) for x_ref, base_logits in refs]
    return {
        "direct_logit_compensation_applied": 1,
        "direct_logit_compensation_mode": mode,
        "direct_logit_compensation_pre_drift": max([p for p in pres if math.isfinite(p)] or [float("nan")]),
        "direct_logit_compensation_post_drift": max(posts) if posts else "",
        "direct_logit_compensation_norm": float(combined.norm().item()),
    }


def train_role(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    role_policy: str,
    label_smoothing: float = 0.0,
) -> tuple[dict[str, Any], float, float]:
    params = [p for n, p in model.named_parameters() if p.requires_grad and role_scan.include_param(n, role_policy)]
    points = [(0.0, fnum(exp.v1252._classification_basic(model, x_val, y_val).get("acc")))]
    if not params:
        return exp.v1252._classification_basic(model, x_val, y_val), float("nan"), float("nan")
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed))
    times: list[float] = []
    elapsed = 0.0
    for _epoch in range(int(epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(batch_size)):
            idx = perm[off: off + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx], label_smoothing=float(label_smoothing))
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1000.0
            elapsed += dt
            times.append(dt)
        points.append((elapsed, fnum(exp.v1252._classification_basic(model, x_val, y_val).get("acc"))))
    q90 = float(torch.tensor(times).quantile(0.90).item()) if times else float("nan")
    return exp.v1252._classification_basic(model, x_val, y_val), q90, auc_error_time(points)


def build_train_refs(base: torch.nn.Module, x_train: torch.Tensor, batch: int, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    refs: list[tuple[torch.Tensor, torch.Tensor]] = []
    batch = max(1, min(int(batch), int(x_train.shape[0])))
    with torch.no_grad():
        for i in range(int(count)):
            start = i * batch
            if start >= int(x_train.shape[0]):
                break
            x_ref = x_train[start:min(start + batch, int(x_train.shape[0]))]
            refs.append((x_ref, base(x_ref).detach()))
    return refs


def linec_seed_summary(models: dict[str, torch.nn.Module], xb, yb, xq, yq, seed: int, linec_seeds: list[int], sketch_dim: int) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    pass_count = 0
    for base_seed in linec_seeds:
        metrics: dict[str, dict[str, Any]] = {}
        for idx, (kind, model) in enumerate(models.items()):
            try:
                metrics[kind] = exp.v1221._linec_metrics(model, xb, yb, xq, yq, int(seed) + int(base_seed) + idx, int(sketch_dim))
            except Exception as exc:
                metrics[kind] = {"error": f"{type(exc).__name__}: {exc}"}
        src = metrics.get("source", {})
        noop = metrics.get("noop", {})
        c_pass = int(fnum(src.get("CouplingR2")) >= fnum(noop.get("CouplingR2")))
        n_pass = int(fnum(src.get("NoiseSignalLeak")) <= fnum(noop.get("NoiseSignalLeak")))
        r_pass = int(fnum(src.get("RealSignalReservoirRatio")) <= fnum(noop.get("RealSignalReservoirRatio")))
        one_pass = int(c_pass and n_pass and r_pass)
        pass_count += one_pass
        rows.append(
            {
                "stage": "V1224_BRIDGE_MULTISKETCH_SEED",
                "linec_seed_base": base_seed,
                "linec_seed_pass": one_pass,
                "source_CouplingR2": fnum(src.get("CouplingR2")),
                "noop_CouplingR2": fnum(noop.get("CouplingR2")),
                "source_NoiseSignalLeak": fnum(src.get("NoiseSignalLeak")),
                "noop_NoiseSignalLeak": fnum(noop.get("NoiseSignalLeak")),
                "source_RealSignalReservoirRatio": fnum(src.get("RealSignalReservoirRatio")),
                "noop_RealSignalReservoirRatio": fnum(noop.get("RealSignalReservoirRatio")),
            }
        )
    return rows, pass_count, int(pass_count == len(linec_seeds) and len(linec_seeds) > 0)


CANDIDATES = {
    "I28-TrainStreamEMACompensation": ("I26-TrainDirectLogitCompensatedQuadRelease", "TrainDirectLogitCompensatedRandomControl", "ema"),
    "I29-TrainProbeMedianCompensation": ("I26-TrainDirectLogitCompensatedQuadRelease", "TrainDirectLogitCompensatedRandomControl", "median"),
    "I29-TrainProbeTrimmedMeanCompensation": ("I26-TrainDirectLogitCompensatedQuadRelease", "TrainDirectLogitCompensatedRandomControl", "trimmed_mean"),
    "I31-NullLogitCompensatedShadowRelease": ("I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl", "median"),
}


def run_candidate(args: argparse.Namespace, source: dict[str, Any], candidate_name: str, train_seed_base: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_actuator, substrate_control, comp_mode = CANDIDATES[candidate_name]
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, _base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    refs = build_train_refs(base, x_train, int(args.compensation_batch), int(args.ensemble_count))
    budget = float(source["norm_budget"])
    signed = float(source["signed_direction"])
    specs = [
        ("noop", "NoOpMatchedOverhead", 0.0, 0),
        ("source", source_actuator, budget, 1),
        ("same_compensation_control", substrate_control, budget, 1),
        ("adamw_parallel_control", "AdamWParallelDirection", budget, 0),
        ("snr_only_control", "SNR-only", budget, 0),
        ("random_norm_control", "RandomMatchedNorm", budget, 0),
    ]
    rows: list[dict[str, Any]] = []
    finals: dict[str, dict[str, Any]] = {}
    models: dict[str, torch.nn.Module] = {}
    for idx, (kind, actuator, bgt, apply_comp) in enumerate(specs):
        trial = copy.deepcopy(base).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(bgt) * 100000) + len(actuator) * 17 + (1 if signed > 0 else 2))
        role = exp.apply_v1223_actuator(trial, actuator, float(bgt), float(signed), gen, opt_delta)
        comp = {
            "direct_logit_compensation_applied": 0,
            "direct_logit_compensation_mode": "",
            "direct_logit_compensation_pre_drift": "",
            "direct_logit_compensation_post_drift": "",
            "direct_logit_compensation_norm": "",
        }
        if apply_comp:
            comp = apply_train_stream_compensation(trial, refs, comp_mode)
        init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        final_eval, q90, auc = train_role(
            trial,
            x_train,
            y_train,
            x_val,
            y_val,
            int(args.batch_size),
            int(args.epochs),
            float(args.lr),
            float(args.weight_decay),
            int(seed) + int(train_seed_base) + idx + len(candidate_name),
            device,
            str(args.role_policy),
            float(args.label_smoothing),
        )
        row = {
            "stage": "V1224_TRAIN_STREAM_BRIDGE_VARIANT",
            "candidate_name": candidate_name,
            "variant_kind": kind,
            "actuator_id": actuator,
            "role": role,
            "dataset": dataset,
            "seed": seed,
            "train_seed_base": int(train_seed_base),
            "norm_budget": bgt,
            "signed_direction": signed,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "role_policy": str(args.role_policy),
            "label_smoothing": float(args.label_smoothing),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "init_acc": init_eval.get("acc", ""),
            "init_NLL": init_eval.get("NLL", ""),
            "init_ECE": init_eval.get("ECE", ""),
            "init_CEp99": init_eval.get("CEp99", ""),
            "final_acc": final_eval.get("acc", ""),
            "final_NLL": final_eval.get("NLL", ""),
            "final_ECE": final_eval.get("ECE", ""),
            "final_CEp99": final_eval.get("CEp99", ""),
            "step_time_q90_ms": q90,
            "auc_error_time": auc,
            "uses_label": int(actuator == "AdamWParallelDirection"),
            "uses_ce_vector": int(actuator == "AdamWParallelDirection"),
            "uses_query_batch": 0,
            "uses_train_batch": int(apply_comp),
            "precommit_available": int(actuator != "AdamWParallelDirection"),
            "loss_agnostic_direction": int(actuator != "AdamWParallelDirection"),
            "promotion_allowed": 0,
            "no_fake": 1,
            **comp,
        }
        rows.append(row)
        finals[kind] = row
        models[kind] = trial
    linec_rows, linec_pass_count, linec_all_pass = linec_seed_summary(
        {"noop": models["noop"], "source": models["source"]},
        xb,
        yb,
        xq,
        yq,
        int(seed),
        parse_ints(args.linec_seeds),
        int(args.linec_sketch_dim),
    )
    for linec_row in linec_rows:
        rows.append(
            {
                **linec_row,
                "candidate_name": candidate_name,
                "dataset": dataset,
                "seed": seed,
                "train_seed_base": int(train_seed_base),
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
    src = finals["source"]
    noop = finals["noop"]
    controls = [finals[k] for k in ["same_compensation_control", "adamw_parallel_control", "snr_only_control", "random_norm_control"]]
    src_acc = fnum(src.get("final_acc"))
    noop_acc = fnum(noop.get("final_acc"))
    control_acc = max(fnum(c.get("final_acc")) for c in controls)
    task_pass = int(
        math.isfinite(src_acc)
        and src_acc >= max(noop_acc, control_acc) + 0.005
        and fnum(src.get("final_NLL")) <= fnum(noop.get("final_NLL"))
        and fnum(src.get("final_CEp99")) <= fnum(noop.get("final_CEp99")) + 0.05
        and fnum(src.get("final_ECE")) <= fnum(noop.get("final_ECE")) + 0.02
        and fnum(src.get("auc_error_time")) <= fnum(noop.get("auc_error_time"))
        and fnum(src.get("step_time_q90_ms")) <= fnum(noop.get("step_time_q90_ms")) * 1.05
    )
    linec_majority_pass = int(linec_pass_count >= math.ceil(len(parse_ints(args.linec_seeds)) / 2.0))
    summary = {
        "stage": "V1224_TRAIN_STREAM_BRIDGE_SUMMARY",
        "candidate_name": candidate_name,
        "source_actuator": source_actuator,
        "same_compensation_control": substrate_control,
        "direct_logit_compensation_mode": comp_mode,
        "dataset": dataset,
        "seed": seed,
        "train_seed_base": int(train_seed_base),
        "source_final_acc": src_acc,
        "noop_final_acc": noop_acc,
        "best_control_final_acc": control_acc,
        "source_vs_noop_acc_delta": src_acc - noop_acc,
        "source_vs_best_control_acc_delta": src_acc - control_acc,
        "task_gate_pass": task_pass,
        "linec_seed_count": len(parse_ints(args.linec_seeds)),
        "linec_seed_pass_count": linec_pass_count,
        "linec_majority_pass": linec_majority_pass,
        "linec_all_pass": linec_all_pass,
        "strict_majority_pass": int(task_pass and linec_majority_pass),
        "strict_all_pass": int(task_pass and linec_all_pass),
        "uses_query_batch": 0,
        "uses_train_batch": 1,
        "precommit_available": 1,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    rows.append(summary)
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-out-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--role-policy", default="quad_only")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--compensation-batch", type=int, default=32)
    ap.add_argument("--ensemble-count", type=int, default=8)
    ap.add_argument("--train-seed-bases", default="12240400,12241400,12242400")
    ap.add_argument("--linec-seeds", default="12239500,12240600,12241600,12242600,12243600")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--candidates", default="I28-TrainStreamEMACompensation,I29-TrainProbeMedianCompensation,I29-TrainProbeTrimmedMeanCompensation,I31-NullLogitCompensatedShadowRelease")
    ap.add_argument("--artifact-prefix", default="v1224_train_stream_functional_bridge")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    exp.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.24 train-stream bridge requires CUDA")
    torch.cuda.set_device(device)

    source = p4m.pick_source(Path(args.source_out_dir))
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in parse_csv(args.candidates):
        if candidate not in CANDIDATES:
            raise ValueError(f"unknown candidate {candidate}; available={sorted(CANDIDATES)}")
        for train_seed_base in parse_ints(args.train_seed_bases):
            rr, ss = run_candidate(args, source, candidate, train_seed_base, device)
            rows.extend(rr)
            summaries.append(ss)
            torch.cuda.empty_cache()

    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_candidate.setdefault(str(summary["candidate_name"]), []).append(summary)
    aggregate_rows: list[dict[str, Any]] = []
    for candidate, group in sorted(by_candidate.items()):
        train_shuffle_strict_majority = sum(exp.safe_int(r.get("strict_majority_pass"), 0) for r in group)
        train_shuffle_strict_all = sum(exp.safe_int(r.get("strict_all_pass"), 0) for r in group)
        aggregate_rows.append(
            {
                "stage": "V1224_TRAIN_STREAM_BRIDGE_CANDIDATE_AGGREGATE",
                "candidate_name": candidate,
                "train_shuffle_rows": len(group),
                "train_shuffle_strict_majority_pass_count": train_shuffle_strict_majority,
                "train_shuffle_strict_all_pass_count": train_shuffle_strict_all,
                "train_shuffle_robust_majority_pass": int(train_shuffle_strict_majority >= 2),
                "train_shuffle_robust_all_pass": int(train_shuffle_strict_all >= 2),
                "best_source_vs_noop_acc_delta": max(fnum(r.get("source_vs_noop_acc_delta"), -999.0) for r in group),
                "best_source_vs_control_acc_delta": max(fnum(r.get("source_vs_best_control_acc_delta"), -999.0) for r in group),
                "uses_query_batch": 0,
                "uses_train_batch": 1,
                "precommit_available": 1,
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
    rows.extend(aggregate_rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, rows)
    result = {
        "stage": "V1224_TRAIN_STREAM_BRIDGE_AGGREGATE",
        "artifact_csv": exp.rel(csv_path),
        "source_out_dir": exp.rel(args.source_out_dir),
        "source_dataset": source.get("dataset", ""),
        "source_seed": source.get("seed", ""),
        "source_actuator": source.get("actuator_id", ""),
        "source_norm_budget": source.get("norm_budget", ""),
        "source_signed_direction": source.get("signed_direction", ""),
        "candidate_rows": len(summaries),
        "aggregate_rows": len(aggregate_rows),
        "any_strict_majority_pass": int(any(exp.safe_int(r.get("strict_majority_pass"), 0) for r in summaries)),
        "any_strict_all_pass": int(any(exp.safe_int(r.get("strict_all_pass"), 0) for r in summaries)),
        "any_train_shuffle_robust_majority_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0) for r in aggregate_rows)),
        "any_train_shuffle_robust_all_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregate_rows)),
        "best_candidate_aggregates": sorted(
            aggregate_rows,
            key=lambda r: (
                exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0),
                exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0),
                fnum(r.get("best_source_vs_noop_acc_delta"), -999.0),
                fnum(r.get("best_source_vs_control_acc_delta"), -999.0),
            ),
            reverse=True,
        ),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(json_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
