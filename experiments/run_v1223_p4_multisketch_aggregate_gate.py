#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, csv, json, math, sys, time
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


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def auc_error_time(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    total = max(points[-1][0] - points[0][0], 1.0e-9)
    area = 0.0
    for (t0, a0), (t1, a1) in zip(points[:-1], points[1:]):
        area += max(t1 - t0, 0.0) * (((1.0 - a0) + (1.0 - a1)) / 2.0)
    return float(area / total)


def train_role(model, x_train, y_train, x_val, y_val, batch_size, epochs, lr, wd, seed, device, role_policy):
    params = [p for n, p in model.named_parameters() if p.requires_grad and role_scan.include_param(n, role_policy)]
    if not params:
        ev = exp.v1252._classification_basic(model, x_val, y_val)
        return ev, float("nan"), float("nan")
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(wd))
    gen = torch.Generator(device=device).manual_seed(int(seed))
    times: list[float] = []
    elapsed = 0.0
    points = [(0.0, fnum(exp.v1252._classification_basic(model, x_val, y_val).get("acc")))]
    for _epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(batch_size)):
            idx = perm[off:off + int(batch_size)]
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
            elapsed += float(dt)
            times.append(float(dt))
        points.append((elapsed, fnum(exp.v1252._classification_basic(model, x_val, y_val).get("acc"))))
    ev = exp.v1252._classification_basic(model, x_val, y_val)
    q90 = float(torch.tensor(times).quantile(0.90).item()) if times else float("nan")
    return ev, q90, auc_error_time(points)


def apply_actuator_and_comp(args, trial, aid, budget, signed, seed, device, opt_delta, xq, base_query_logits, x_train, base, role_idx):
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(aid) * 17 + (1 if signed > 0 else 2))
    role = exp.apply_v1223_actuator(trial, aid, float(budget), float(signed), gen, opt_delta)
    direct = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
    if aid in {"I24-DirectLogitCompensatedQuadRelease", "I25-DirectLogitCompensatedShadowRelease", "DirectLogitCompensatedRandomControl", "I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
        if args.compensation_reference == "train" or aid in {"I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
            b = min(int(args.compensation_batch or args.linec_batch), int(x_train.shape[0]))
            x_ref = x_train[:b]
            with torch.no_grad():
                base_ref = base(x_ref).detach()
            direct = exp.apply_direct_readout_logit_compensation(trial, x_ref, base_ref)
        else:
            direct = exp.apply_direct_readout_logit_compensation(trial, xq, base_query_logits)
    return role, direct


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
    ap.add_argument("--train-seed-base", type=int, default=12239400)
    ap.add_argument("--linec-seeds", default="12239500,12240600,12241600,12242600,12243600")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--source-actuator-override", default="")
    ap.add_argument("--control-actuator-override", default="")
    ap.add_argument("--compensation-reference", choices=["query", "train"], default="query")
    ap.add_argument("--compensation-batch", type=int, default=0)
    ap.add_argument("--artifact-prefix", default="v1223_p4_multisketch_aggregate_gate")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("multi-sketch aggregate gate requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(out_dir)
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    budget = float(source["norm_budget"])
    signed = float(source["signed_direction"])
    source_actuator = str(args.source_actuator_override or source["actuator_id"])
    control_actuator = str(args.control_actuator_override or ("TrainDirectLogitCompensatedRandomControl" if args.compensation_reference == "train" else "DirectLogitCompensatedRandomControl"))
    specs = [("noop_control", "NoOpMatchedOverhead", 0.0), ("p3_source_actuator", source_actuator, budget), ("direct_logit_compensated_control", control_actuator, budget)]
    rows: list[dict[str, Any]] = []
    finals: dict[str, dict[str, Any]] = {}
    models: dict[str, torch.nn.Module] = {}
    for idx, (kind, aid, bgt) in enumerate(specs):
        trial = copy.deepcopy(base).to(device)
        role, direct = apply_actuator_and_comp(args, trial, aid, bgt, signed, seed, device, opt_delta, xq, base_query_logits, x_train, base, idx)
        ev, q90, auc = train_role(trial, x_train, y_train, x_val, y_val, int(args.batch_size), int(args.epochs), float(args.lr), float(args.weight_decay), int(seed) + int(args.train_seed_base) + idx + len(args.role_policy), device, args.role_policy)
        final = {"stage": "V1223_P4_MULTISKETCH_VARIANT", "variant_kind": kind, "actuator_id": aid, "role": role, "source_dataset": dataset, "source_seed": seed, "norm_budget": bgt, "signed_direction": signed, "epochs": int(args.epochs), "lr": float(args.lr), "weight_decay": float(args.weight_decay), "final_acc": ev.get("acc", ""), "final_NLL": ev.get("NLL", ""), "final_ECE": ev.get("ECE", ""), "final_CEp99": ev.get("CEp99", ""), "step_time_q90_ms": q90, "auc_error_time": auc, "compensation_reference": args.compensation_reference, **direct}
        finals[kind] = final
        models[kind] = trial
        rows.append(final)
    linec_seeds = parse_ints(args.linec_seeds)
    seed_passes = []
    linec_by_seed: list[dict[str, Any]] = []
    for lseed in linec_seeds:
        metrics = {}
        for idx, (kind, _aid, _bgt) in enumerate(specs):
            try:
                metrics[kind] = exp.v1221._linec_metrics(models[kind], xb, yb, xq, yq, int(seed) + int(lseed) + idx, int(args.linec_sketch_dim))
            except Exception as exc:
                metrics[kind] = {"error": f"{type(exc).__name__}: {exc}"}
            row = {"stage": "V1223_P4_MULTISKETCH_LINEC", "linec_seed_base": lseed, "variant_kind": kind, **{f"linec_{k}": v for k, v in metrics[kind].items()}}
            rows.append(row)
        src_m, noop_m = metrics["p3_source_actuator"], metrics["noop_control"]
        c_pass = int(fnum(src_m.get("CouplingR2")) >= fnum(noop_m.get("CouplingR2")))
        n_pass = int(fnum(src_m.get("NoiseSignalLeak")) <= fnum(noop_m.get("NoiseSignalLeak")))
        r_pass = int(fnum(src_m.get("RealSignalReservoirRatio")) <= fnum(noop_m.get("RealSignalReservoirRatio")))
        one = {"stage": "V1223_P4_MULTISKETCH_SEED_SUMMARY", "linec_seed_base": lseed, "linec_seed_pass": int(c_pass and n_pass and r_pass), "coupling_pass": c_pass, "noise_pass": n_pass, "reservoir_pass": r_pass, "source_CouplingR2": fnum(src_m.get("CouplingR2")), "noop_CouplingR2": fnum(noop_m.get("CouplingR2")), "source_NoiseSignalLeak": fnum(src_m.get("NoiseSignalLeak")), "noop_NoiseSignalLeak": fnum(noop_m.get("NoiseSignalLeak")), "source_RealSignalReservoirRatio": fnum(src_m.get("RealSignalReservoirRatio")), "noop_RealSignalReservoirRatio": fnum(noop_m.get("RealSignalReservoirRatio"))}
        seed_passes.append(one)
        linec_by_seed.append(one)
        rows.append(one)
    src, noop, ctrl = finals["p3_source_actuator"], finals["noop_control"], finals["direct_logit_compensated_control"]
    src_acc, noop_acc, ctrl_acc = fnum(src.get("final_acc")), fnum(noop.get("final_acc")), fnum(ctrl.get("final_acc"))
    src_nll, noop_nll = fnum(src.get("final_NLL")), fnum(noop.get("final_NLL"))
    src_cep, noop_cep = fnum(src.get("final_CEp99")), fnum(noop.get("final_CEp99"))
    src_ece, noop_ece = fnum(src.get("final_ECE")), fnum(noop.get("final_ECE"))
    src_auc, noop_auc = fnum(src.get("auc_error_time")), fnum(noop.get("auc_error_time"))
    src_time, noop_time = fnum(src.get("step_time_q90_ms")), fnum(noop.get("step_time_q90_ms"))
    task_pass = int(src_acc >= max(noop_acc, ctrl_acc) + 0.005 and src_nll <= noop_nll and src_cep <= noop_cep + 0.05 and src_ece <= noop_ece + 0.02 and src_auc <= noop_auc and src_time <= noop_time * 1.05)
    linec_pass_count = sum(int(r["linec_seed_pass"]) for r in seed_passes)
    all_linec = int(linec_pass_count == len(seed_passes) and len(seed_passes) > 0)
    majority_linec = int(linec_pass_count >= math.ceil(len(seed_passes) / 2.0) and len(seed_passes) > 0)
    strict_all = int(task_pass and all_linec)
    strict_majority = int(task_pass and majority_linec)
    summary = {"stage": "V1223_P4_MULTISKETCH_AGGREGATE_SUMMARY", "artifact_prefix": args.artifact_prefix, "source_actuator": source_actuator, "control_actuator": control_actuator, "source_dataset": dataset, "source_seed": seed, "source_norm_budget": source.get("norm_budget", ""), "source_signed_direction": source.get("signed_direction", ""), "epochs": int(args.epochs), "lr": float(args.lr), "role_policy": args.role_policy, "linec_seed_count": len(seed_passes), "linec_seed_pass_count": linec_pass_count, "task_gate_pass": task_pass, "strict_all_sketch_pass": strict_all, "strict_majority_sketch_pass": strict_majority, "promotion_allowed": 0, "source_final_acc": src_acc, "noop_final_acc": noop_acc, "control_final_acc": ctrl_acc, "source_vs_noop_acc_delta": src_acc - noop_acc, "source_vs_control_acc_delta": src_acc - ctrl_acc, "source_NLL": src_nll, "noop_NLL": noop_nll, "source_CEp99": src_cep, "noop_CEp99": noop_cep, "source_ECE": src_ece, "noop_ECE": noop_ece, "source_auc_error_time": src_auc, "noop_auc_error_time": noop_auc, "status": "strict_all_pass_audit_only_not_promoted" if strict_all else "executed_no_official_pass", "no_fake": 1, "seed_summaries": linec_by_seed}
    rows.append(summary)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, rows)
    summary["artifact_csv"] = exp.rel(csv_path)
    exp.write_json(json_path, summary)
    state_path = out_dir / "v1223_route_decision.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"p4_multisketch_last_summary": exp.rel(json_path), "p4_multisketch_last_strict_all_pass": strict_all, "p4_multisketch_last_strict_majority_pass": strict_majority, "p4_multisketch_promotion_allowed": 0, "official_success_reached": 0, "p4_pass": 0, "no_fake": 1})
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
