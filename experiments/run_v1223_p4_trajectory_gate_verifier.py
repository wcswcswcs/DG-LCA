#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
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


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except Exception:
        return default


def auc_error_time(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    pts = sorted(points, key=lambda p: p[0])
    total = max(pts[-1][0] - pts[0][0], 1.0e-9)
    area = 0.0
    for (t0, a0), (t1, a1) in zip(pts[:-1], pts[1:]):
        area += max(t1 - t0, 0.0) * (((1.0 - a0) + (1.0 - a1)) / 2.0)
    return float(area / total)


def train_role_trajectory(model, x_train, y_train, x_val, y_val, batch_size, epochs, lr, weight_decay, seed, device, role_policy):
    params = [p for n, p in model.named_parameters() if p.requires_grad and role_scan.include_param(n, role_policy)]
    if not params:
        return [], float("nan")
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed))
    step_times: list[float] = []
    rows: list[dict[str, Any]] = []
    cumulative_ms = 0.0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        epoch_ms = 0.0
        for off in range(0, int(x_train.shape[0]), int(batch_size)):
            idx = perm[off: off + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1000.0
            step_times.append(float(dt))
            epoch_ms += float(dt)
        cumulative_ms += epoch_ms
        ev = exp.v1252._classification_basic(model, x_val, y_val)
        rows.append({
            "epoch": epoch,
            "elapsed_train_ms": cumulative_ms,
            "epoch_train_ms": epoch_ms,
            "acc": ev.get("acc", ""),
            "NLL": ev.get("NLL", ""),
            "ECE": ev.get("ECE", ""),
            "CEp99": ev.get("CEp99", ""),
        })
    q90 = float(torch.tensor(step_times).quantile(0.90).item()) if step_times else float("nan")
    return rows, q90


def run_candidate(args, source: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    comp_batch = min(int(getattr(args, "compensation_batch", 0) or xb.shape[0]), int(x_train.shape[0]))
    x_comp_train = x_train[:comp_batch]
    with torch.no_grad():
        base_train_logits = base(x_comp_train).detach()
    budget = float(source["norm_budget"])
    signed = float(source["signed_direction"])
    actuator = str(getattr(args, "source_actuator_override", "") or source["actuator_id"])
    control_actuator = str(getattr(args, "control_actuator_override", "") or "DirectLogitCompensatedRandomControl")
    variants = [
        ("NoOpMatchedOverhead", "noop_control", 0.0, signed),
        (actuator, "p3_source_actuator", budget, signed),
        (control_actuator, "direct_logit_compensated_control", budget, signed),
    ]
    all_rows: list[dict[str, Any]] = []
    finals: dict[str, dict[str, Any]] = {}
    for idx, (aid, kind, bgt, sgn) in enumerate(variants):
        trial = copy.deepcopy(base).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(bgt) * 100000) + len(aid) * 17 + (1 if sgn > 0 else 2))
        role = exp.apply_v1223_actuator(trial, aid, float(bgt), float(sgn), gen, opt_delta)
        direct = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
        if aid in {"I24-DirectLogitCompensatedQuadRelease", "I25-DirectLogitCompensatedShadowRelease", "DirectLogitCompensatedRandomControl", "I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
            if str(getattr(args, "compensation_reference", "query")) == "train" or aid in {"I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
                direct = exp.apply_direct_readout_logit_compensation(trial, x_comp_train, base_train_logits)
            else:
                direct = exp.apply_direct_readout_logit_compensation(trial, xq, base_query_logits)
        init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        traj, q90 = train_role_trajectory(trial, x_train, y_train, x_val, y_val, int(args.batch_size), int(args.epochs), float(args.lr), float(args.weight_decay), seed + int(args.train_seed_base) + idx + len(args.role_policy), device, args.role_policy)
        try:
            linec = exp.v1221._linec_metrics(trial, xb, yb, xq, yq, seed + int(args.linec_seed_base) + idx, int(args.linec_sketch_dim))
        except Exception as exc:
            linec = {"error": f"{type(exc).__name__}: {exc}"}
        points = [(0.0, fnum(init_eval.get("acc")))] + [(fnum(r["elapsed_train_ms"]), fnum(r["acc"])) for r in traj]
        final_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        final = {
            "stage": "V1223_FUNCTIONAL_P4_TRAJECTORY_GATE_VARIANT",
            "variant_kind": kind,
            "role_policy": args.role_policy,
            "actuator_id": aid,
            "source_actuator": actuator,
            "compensation_reference": str(getattr(args, "compensation_reference", "query")),
            "compensation_batch": comp_batch if str(getattr(args, "compensation_reference", "query")) == "train" or aid in {"I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"} else int(xq.shape[0]),
            "dataset": dataset,
            "seed": seed,
            "norm_budget": bgt,
            "signed_direction": sgn,
            "role": role,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "init_acc": init_eval.get("acc", ""),
            "final_acc": final_eval.get("acc", ""),
            "final_NLL": final_eval.get("NLL", ""),
            "final_ECE": final_eval.get("ECE", ""),
            "final_CEp99": final_eval.get("CEp99", ""),
            "step_time_q90_ms": q90,
            "auc_error_time": auc_error_time(points),
            **direct,
            **{f"linec_{k}": v for k, v in linec.items()},
        }
        finals[kind] = final
        all_rows.append(final)
        for tr in traj:
            all_rows.append({
                "stage": "V1223_FUNCTIONAL_P4_TRAJECTORY_GATE_EPOCH",
                "variant_kind": kind,
                "role_policy": args.role_policy,
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                **tr,
            })
    src = finals["p3_source_actuator"]
    noop = finals["noop_control"]
    ctrl = finals["direct_logit_compensated_control"]
    src_acc, noop_acc, ctrl_acc = fnum(src.get("final_acc")), fnum(noop.get("final_acc")), fnum(ctrl.get("final_acc"))
    src_nll, noop_nll = fnum(src.get("final_NLL")), fnum(noop.get("final_NLL"))
    src_cep, noop_cep = fnum(src.get("final_CEp99")), fnum(noop.get("final_CEp99"))
    src_ece, noop_ece = fnum(src.get("final_ECE")), fnum(noop.get("final_ECE"))
    src_q90, noop_q90 = fnum(src.get("step_time_q90_ms")), fnum(noop.get("step_time_q90_ms"))
    src_auc, noop_auc = fnum(src.get("auc_error_time")), fnum(noop.get("auc_error_time"))
    src_cpl, noop_cpl = fnum(src.get("linec_CouplingR2")), fnum(noop.get("linec_CouplingR2"))
    src_noise, noop_noise = fnum(src.get("linec_NoiseSignalLeak")), fnum(noop.get("linec_NoiseSignalLeak"))
    src_res, noop_res = fnum(src.get("linec_RealSignalReservoirRatio")), fnum(noop.get("linec_RealSignalReservoirRatio"))
    strict_gate = int(
        math.isfinite(src_acc) and math.isfinite(noop_acc) and math.isfinite(ctrl_acc)
        and src_acc >= max(noop_acc, ctrl_acc) + 0.005
        and (not math.isfinite(src_nll) or not math.isfinite(noop_nll) or src_nll <= noop_nll)
        and (not math.isfinite(src_cep) or not math.isfinite(noop_cep) or src_cep <= noop_cep + 0.05)
        and (not math.isfinite(src_ece) or not math.isfinite(noop_ece) or src_ece <= noop_ece + 0.02)
        and (not math.isfinite(src_q90) or not math.isfinite(noop_q90) or src_q90 <= noop_q90 * 1.05)
        and (not math.isfinite(src_auc) or not math.isfinite(noop_auc) or src_auc <= noop_auc)
        and (not math.isfinite(src_cpl) or not math.isfinite(noop_cpl) or src_cpl >= noop_cpl)
        and (not math.isfinite(src_noise) or not math.isfinite(noop_noise) or src_noise <= noop_noise)
        and (not math.isfinite(src_res) or not math.isfinite(noop_res) or src_res <= noop_res)
    )
    summary = {
        "stage": "V1223_FUNCTIONAL_P4_TRAJECTORY_GATE_SUMMARY",
        "artifact_prefix": args.artifact_prefix,
        "source_actuator": actuator,
        "compensation_reference": str(getattr(args, "compensation_reference", "query")),
        "compensation_batch": comp_batch,
        "control_actuator": control_actuator,
        "source_dataset": dataset,
        "source_seed": seed,
        "source_norm_budget": source.get("norm_budget", ""),
        "source_signed_direction": source.get("signed_direction", ""),
        "role_policy": args.role_policy,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "source_final_acc": src_acc,
        "noop_final_acc": noop_acc,
        "control_final_acc": ctrl_acc,
        "source_vs_noop_acc_delta": src_acc - noop_acc,
        "source_vs_control_acc_delta": src_acc - ctrl_acc,
        "source_NLL": src_nll,
        "noop_NLL": noop_nll,
        "source_CEp99": src_cep,
        "noop_CEp99": noop_cep,
        "source_ECE": src_ece,
        "noop_ECE": noop_ece,
        "source_step_time_q90_ms": src_q90,
        "noop_step_time_q90_ms": noop_q90,
        "step_time_ratio_source_vs_noop": src_q90 / noop_q90 if math.isfinite(src_q90) and math.isfinite(noop_q90) and noop_q90 else "",
        "source_auc_error_time": src_auc,
        "noop_auc_error_time": noop_auc,
        "auc_error_time_ratio_source_vs_noop": src_auc / noop_auc if math.isfinite(src_auc) and math.isfinite(noop_auc) and noop_auc else "",
        "source_CouplingR2": src_cpl,
        "noop_CouplingR2": noop_cpl,
        "source_NoiseSignalLeak": src_noise,
        "noop_NoiseSignalLeak": noop_noise,
        "source_RealSignalReservoirRatio": src_res,
        "noop_RealSignalReservoirRatio": noop_res,
        "strict_trajectory_gate_pass": strict_gate,
        "promotion_allowed": 0,
        "status": "strict_gate_pass_audit_only_route_not_mutated" if strict_gate else "strict_gate_fail_audit_only",
        "no_fake": 1,
    }
    all_rows.append(summary)
    return all_rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--role-policy", default="quad_only")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--artifact-prefix", default="v1223_functional_p4_trajectory_gate")
    ap.add_argument("--train-seed-base", type=int, default=12239400)
    ap.add_argument("--linec-seed-base", type=int, default=12239500)
    ap.add_argument("--source-actuator-override", default="")
    ap.add_argument("--control-actuator-override", default="")
    ap.add_argument("--compensation-reference", choices=["query", "train"], default="query")
    ap.add_argument("--compensation-batch", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("trajectory gate verifier requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(out_dir)
    rows, summary = run_candidate(args, source, device)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary["artifact_csv"] = exp.rel(csv_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    try:
        state = json.loads((out_dir / "v1223_route_decision.json").read_text(encoding="utf-8"))
    except Exception:
        state = {}
    state.update({
        "p4_trajectory_gate_last_summary": exp.rel(json_path),
        "p4_trajectory_gate_last_strict_pass": int(summary["strict_trajectory_gate_pass"]),
        "p4_trajectory_gate_promotion_allowed": 0,
        "no_fake": 1,
    })
    (out_dir / "v1223_route_decision.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
