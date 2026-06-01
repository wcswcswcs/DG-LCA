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


def include_param(name: str, role_policy: str) -> bool:
    return role_scan.include_param(name, role_policy)


def build_variant(base, aid: str, budget: float, signed: float, seed: int, device, opt_delta, x_ref, base_ref_logits):
    trial = copy.deepcopy(base).to(device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(aid) * 17 + (1 if signed > 0 else 2))
    role = exp.apply_v1223_actuator(trial, aid, float(budget), float(signed), gen, opt_delta)
    direct = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
    if aid in {"I24-DirectLogitCompensatedQuadRelease", "I25-DirectLogitCompensatedShadowRelease", "DirectLogitCompensatedRandomControl", "I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
        direct = exp.apply_direct_readout_logit_compensation(trial, x_ref, base_ref_logits)
    return trial, role, direct


def auc_error_time(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    total = max(points[-1][0] - points[0][0], 1.0e-9)
    area = 0.0
    for (t0, a0), (t1, a1) in zip(points[:-1], points[1:]):
        area += max(t1 - t0, 0.0) * (((1.0 - a0) + (1.0 - a1)) / 2.0)
    return float(area / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--role-policy", default="quad_only")
    ap.add_argument("--source-actuator-override", default="")
    ap.add_argument("--control-actuator-override", default="")
    ap.add_argument("--compensation-reference", choices=["query", "train"], default="query")
    ap.add_argument("--compensation-batch", type=int, default=0)
    ap.add_argument("--train-seed-base", type=int, default=12239400)
    ap.add_argument("--linec-seed-base", type=int, default=12239500)
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--artifact-prefix", default="v1223_functional_p4_reservoir_veto_checkpoint")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("reservoir-veto checkpoint scan requires CUDA")
    torch.cuda.set_device(device)

    source = p4m.pick_source(out_dir)
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    comp_batch = min(int(args.compensation_batch or xb.shape[0]), int(x_train.shape[0]))
    if args.compensation_reference == "train":
        x_ref = x_train[:comp_batch]
        with torch.no_grad():
            base_ref_logits = base(x_ref).detach()
    else:
        x_ref = xq
        base_ref_logits = base_query_logits
        comp_batch = int(xq.shape[0])
    budget = float(source["norm_budget"])
    signed = float(source["signed_direction"])
    source_actuator = str(args.source_actuator_override or source["actuator_id"])
    control_actuator = str(args.control_actuator_override or ("TrainDirectLogitCompensatedRandomControl" if args.compensation_reference == "train" else "DirectLogitCompensatedRandomControl"))
    variant_specs = [
        ("noop_control", "NoOpMatchedOverhead", 0.0),
        ("p3_source_actuator", source_actuator, budget),
        ("direct_logit_compensated_control", control_actuator, budget),
    ]
    variants: dict[str, dict[str, Any]] = {}
    for idx, (kind, aid, bgt) in enumerate(variant_specs):
        model, role, direct = build_variant(base, aid, bgt, signed, seed, device, opt_delta, x_ref, base_ref_logits)
        params = [p for n, p in model.named_parameters() if p.requires_grad and include_param(n, args.role_policy)]
        opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay)) if params else None
        init_eval = exp.v1252._classification_basic(model, x_val, y_val)
        variants[kind] = {"model": model, "optimizer": opt, "aid": aid, "role": role, "direct": direct, "idx": idx, "points": [(0.0, fnum(init_eval.get("acc")))], "elapsed_ms": 0.0, "step_times": []}
    rows: list[dict[str, Any]] = []
    gen_by_kind = {kind: torch.Generator(device=device).manual_seed(int(seed) + int(args.train_seed_base) + int(v["idx"]) + len(args.role_policy)) for kind, v in variants.items()}
    for epoch in range(1, int(args.epochs) + 1):
        for kind, v in variants.items():
            model = v["model"]
            opt = v["optimizer"]
            model.train()
            epoch_ms = 0.0
            if opt is not None:
                perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_by_kind[kind])
                for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
                    idxs = perm[off: off + int(args.batch_size)]
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    loss = F.cross_entropy(model(x_train[idxs]), y_train[idxs])
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize(device)
                    dt = (time.perf_counter() - t0) * 1000.0
                    epoch_ms += float(dt)
                    v["step_times"].append(float(dt))
            v["elapsed_ms"] += epoch_ms
            ev = exp.v1252._classification_basic(model, x_val, y_val)
            try:
                linec = exp.v1221._linec_metrics(model, xb, yb, xq, yq, int(seed) + int(args.linec_seed_base) + int(v["idx"]) + epoch * 100, int(args.linec_sketch_dim))
            except Exception as exc:
                linec = {"error": f"{type(exc).__name__}: {exc}"}
            v["points"].append((float(v["elapsed_ms"]), fnum(ev.get("acc"))))
            rows.append({
                "stage": "V1223_FUNCTIONAL_P4_RESERVOIR_VETO_EPOCH",
                "variant_kind": kind,
                "epoch": epoch,
                "role_policy": args.role_policy,
                "actuator_id": v["aid"],
                "role": v["role"],
                "source_dataset": dataset,
                "source_seed": seed,
                "source_norm_budget": source.get("norm_budget", ""),
                "source_signed_direction": source.get("signed_direction", ""),
                "compensation_reference": args.compensation_reference,
                "compensation_batch": comp_batch,
                "final_acc": ev.get("acc", ""),
                "final_NLL": ev.get("NLL", ""),
                "final_ECE": ev.get("ECE", ""),
                "final_CEp99": ev.get("CEp99", ""),
                "elapsed_train_ms": v["elapsed_ms"],
                "epoch_train_ms": epoch_ms,
                **v["direct"],
                **{f"linec_{k}": val for k, val in linec.items()},
            })
    summary_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        erows = {r["variant_kind"]: r for r in rows if r.get("epoch") == epoch}
        if not {"noop_control", "p3_source_actuator", "direct_logit_compensated_control"}.issubset(erows):
            continue
        src, noop, ctrl = erows["p3_source_actuator"], erows["noop_control"], erows["direct_logit_compensated_control"]
        src_acc, noop_acc, ctrl_acc = fnum(src.get("final_acc")), fnum(noop.get("final_acc")), fnum(ctrl.get("final_acc"))
        src_nll, noop_nll = fnum(src.get("final_NLL")), fnum(noop.get("final_NLL"))
        src_cep, noop_cep = fnum(src.get("final_CEp99")), fnum(noop.get("final_CEp99"))
        src_ece, noop_ece = fnum(src.get("final_ECE")), fnum(noop.get("final_ECE"))
        src_cpl, noop_cpl = fnum(src.get("linec_CouplingR2")), fnum(noop.get("linec_CouplingR2"))
        src_noise, noop_noise = fnum(src.get("linec_NoiseSignalLeak")), fnum(noop.get("linec_NoiseSignalLeak"))
        src_res, noop_res = fnum(src.get("linec_RealSignalReservoirRatio")), fnum(noop.get("linec_RealSignalReservoirRatio"))
        pass_gate = int(math.isfinite(src_acc) and math.isfinite(noop_acc) and math.isfinite(ctrl_acc) and src_acc >= max(noop_acc, ctrl_acc) + 0.005 and src_nll <= noop_nll and src_cep <= noop_cep + 0.05 and src_ece <= noop_ece + 0.02 and src_cpl >= noop_cpl and src_noise <= noop_noise and src_res <= noop_res)
        summary_rows.append({
            "stage": "V1223_FUNCTIONAL_P4_RESERVOIR_VETO_CHECKPOINT_SUMMARY",
            "epoch": epoch,
            "p4_reservoir_veto_pass": pass_gate,
            "promotion_allowed": 0,
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
            "source_CouplingR2": src_cpl,
            "noop_CouplingR2": noop_cpl,
            "source_NoiseSignalLeak": src_noise,
            "noop_NoiseSignalLeak": noop_noise,
            "source_RealSignalReservoirRatio": src_res,
            "noop_RealSignalReservoirRatio": noop_res,
            "status": "pass_audit_only_checkpoint_not_promoted" if pass_gate else "executed_no_pass",
            "no_fake": 1,
        })
    rows.extend(summary_rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, rows)
    best = sorted(summary_rows, key=lambda r: (exp.safe_int(r.get("p4_reservoir_veto_pass"), 0), exp.safe_float(r.get("source_vs_noop_acc_delta"), -999.0), -max(0.0, exp.safe_float(r.get("source_RealSignalReservoirRatio"), 999.0) - exp.safe_float(r.get("noop_RealSignalReservoirRatio"), 999.0))), reverse=True)[:10]
    result = {
        "stage": "V1223_FUNCTIONAL_P4_RESERVOIR_VETO_CHECKPOINT_SCAN",
        "artifact_csv": exp.rel(csv_path),
        "source_actuator": source_actuator,
        "control_actuator": control_actuator,
        "source_dataset": dataset,
        "source_seed": seed,
        "source_norm_budget": source.get("norm_budget", ""),
        "source_signed_direction": source.get("signed_direction", ""),
        "role_policy": args.role_policy,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "summary_rows": len(summary_rows),
        "any_reservoir_veto_pass": int(any(exp.safe_int(r.get("p4_reservoir_veto_pass"), 0) for r in summary_rows)),
        "best_summaries": best,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(json_path, result)
    state_path = out_dir / "v1223_route_decision.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"p4_reservoir_veto_checkpoint_summary": exp.rel(json_path), "p4_reservoir_veto_checkpoint_any_pass": result["any_reservoir_veto_pass"], "p4_reservoir_veto_promotion_allowed": 0, "no_fake": 1})
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
