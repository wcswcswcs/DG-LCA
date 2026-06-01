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


def entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    return -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(dim=-1)


def unlabeled_proxy_metrics(model: torch.nn.Module, x_probe: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_probe)
        probs = F.softmax(logits, dim=-1)
        top2 = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
        usage = probs.mean(dim=0)
        log_classes = math.log(max(int(probs.shape[-1]), 2))
        return {
            "proxy_mean_entropy": float(entropy_from_probs(probs).mean().item()),
            "proxy_usage_entropy_norm": float((entropy_from_probs(usage).item() / log_classes) if log_classes > 0 else 0.0),
            "proxy_margin_mean": float(margin.mean().item()),
            "proxy_logit_norm_mean": float(logits.norm(dim=-1).mean().item()),
            "proxy_logit_std": float(logits.std().item()),
            "proxy_top_class_share": float(probs.argmax(dim=-1).bincount(minlength=probs.shape[-1]).float().max().item() / max(1, probs.shape[0])),
        }


def proxy_score(src: dict[str, float], noop: dict[str, float], ctrl: dict[str, float], args) -> dict[str, float | int]:
    # Precommit score: no labels, no CE/NLL, no LineC/audit target.  It rewards source confidence gain only when
    # class usage is not collapsing and logit norm stays close to NoOp; matched control is used only as an unlabeled guard.
    margin_gain = fnum(src.get("proxy_margin_mean")) - fnum(noop.get("proxy_margin_mean"))
    usage_delta = fnum(src.get("proxy_usage_entropy_norm")) - fnum(noop.get("proxy_usage_entropy_norm"))
    ctrl_usage_delta = fnum(src.get("proxy_usage_entropy_norm")) - fnum(ctrl.get("proxy_usage_entropy_norm"))
    src_norm = fnum(src.get("proxy_logit_norm_mean"))
    noop_norm = max(fnum(noop.get("proxy_logit_norm_mean")), 1.0e-9)
    norm_ratio = src_norm / noop_norm
    top_share = fnum(src.get("proxy_top_class_share"))
    score = margin_gain + 0.25 * usage_delta + 0.10 * ctrl_usage_delta - 0.05 * abs(norm_ratio - 1.0) - 0.05 * max(0.0, top_share - float(args.max_top_class_share))
    passed = int(
        math.isfinite(score)
        and margin_gain >= float(args.min_margin_gain)
        and usage_delta >= float(args.min_usage_delta)
        and ctrl_usage_delta >= float(args.min_control_usage_delta)
        and norm_ratio <= float(args.max_logit_norm_ratio)
        and top_share <= float(args.max_top_class_share)
    )
    return {
        "precommit_proxy_score": float(score),
        "proxy_margin_gain_vs_noop": float(margin_gain),
        "proxy_usage_delta_vs_noop": float(usage_delta),
        "proxy_usage_delta_vs_control": float(ctrl_usage_delta),
        "proxy_logit_norm_ratio_vs_noop": float(norm_ratio),
        "precommit_proxy_pass": passed,
    }


def auc_error_time(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    total = max(points[-1][0] - points[0][0], 1.0e-9)
    area = 0.0
    for (t0, a0), (t1, a1) in zip(points[:-1], points[1:]):
        area += max(t1 - t0, 0.0) * (((1.0 - a0) + (1.0 - a1)) / 2.0)
    return float(area / total)


def apply_actuator(args, trial, aid, budget, signed, seed, device, opt_delta, xq, base_query_logits, x_train, base):
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(aid) * 17 + (1 if signed > 0 else 2))
    role = exp.apply_v1223_actuator(trial, aid, float(budget), float(signed), gen, opt_delta)
    direct = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
    if aid in {"I24-DirectLogitCompensatedQuadRelease", "I25-DirectLogitCompensatedShadowRelease", "DirectLogitCompensatedRandomControl", "I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
        if args.compensation_reference == "train" or aid in {"I26-TrainDirectLogitCompensatedQuadRelease", "I27-TrainDirectLogitCompensatedShadowRelease", "TrainDirectLogitCompensatedRandomControl"}:
            b = min(int(args.compensation_batch or args.linec_batch), int(x_train.shape[0]))
            x_ref = x_train[:b]
            with torch.no_grad():
                base_ref_logits = base(x_ref).detach()
            direct = exp.apply_direct_readout_logit_compensation(trial, x_ref, base_ref_logits)
        else:
            direct = exp.apply_direct_readout_logit_compensation(trial, xq, base_query_logits)
    return role, direct


def build_variants(args, source, base, x_train, xq, base_query_logits, opt_delta, seed, device):
    budget = float(source["norm_budget"])
    signed = float(source["signed_direction"])
    source_actuator = str(args.source_actuator_override or source["actuator_id"])
    control_actuator = str(args.control_actuator_override or ("TrainDirectLogitCompensatedRandomControl" if args.compensation_reference == "train" else "DirectLogitCompensatedRandomControl"))
    specs = [("noop_control", "NoOpMatchedOverhead", 0.0), ("p3_source_actuator", source_actuator, budget), ("direct_logit_compensated_control", control_actuator, budget)]
    variants = {}
    for idx, (kind, aid, bgt) in enumerate(specs):
        model = copy.deepcopy(base).to(device)
        role, direct = apply_actuator(args, model, aid, bgt, signed, seed, device, opt_delta, xq, base_query_logits, x_train, base)
        params = [p for n, p in model.named_parameters() if p.requires_grad and role_scan.include_param(n, args.role_policy)]
        opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay)) if params else None
        variants[kind] = {"idx": idx, "kind": kind, "aid": aid, "budget": bgt, "signed": signed, "model": model, "optimizer": opt, "role": role, "direct": direct, "points": [], "elapsed_ms": 0.0, "step_times": []}
    return variants, source_actuator, control_actuator


def train_one_epoch(args, v, x_train, y_train, gen, device):
    model = v["model"]
    opt = v["optimizer"]
    epoch_ms = 0.0
    model.train()
    if opt is None:
        return epoch_ms
    perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
    for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
        idx = perm[off:off + int(args.batch_size)]
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
        epoch_ms += float(dt)
        v["step_times"].append(float(dt))
    return epoch_ms


def run_proxy_pass(args, source, base, x_train, y_train, xq, base_query_logits, opt_delta, seed, device):
    variants, source_actuator, control_actuator = build_variants(args, source, base, x_train, xq, base_query_logits, opt_delta, seed, device)
    gens = {kind: torch.Generator(device=device).manual_seed(int(seed) + int(args.train_seed_base) + int(v["idx"]) + len(args.role_policy)) for kind, v in variants.items()}
    b = min(int(args.proxy_batch), int(x_train.shape[0]))
    x_probe = x_train[:b]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        proxy_by = {}
        for kind, v in variants.items():
            epoch_ms = train_one_epoch(args, v, x_train, y_train, gens[kind], device)
            v["elapsed_ms"] += float(epoch_ms)
            pm = unlabeled_proxy_metrics(v["model"], x_probe)
            proxy_by[kind] = pm
            rows.append({"stage": "V1223_P4_PRECOMMIT_PROXY_CHECKPOINT_EPOCH", "epoch": epoch, "variant_kind": kind, "actuator_id": v["aid"], "role_policy": args.role_policy, "elapsed_train_ms": v["elapsed_ms"], "epoch_train_ms": epoch_ms, "proxy_uses_labels": 0, "proxy_uses_ce_or_loss": 0, "proxy_uses_linec_or_audit_target": 0, **pm})
        ps = proxy_score(proxy_by["p3_source_actuator"], proxy_by["noop_control"], proxy_by["direct_logit_compensated_control"], args)
        summary = {"stage": "V1223_P4_PRECOMMIT_PROXY_CHECKPOINT_SUMMARY", "epoch": epoch, "promotion_allowed": 0, "selection_scope": "pre_audit_unlabeled_train_probe", **ps}
        summaries.append(summary)
        rows.append(summary)
    eligible = [r for r in summaries if int(r["precommit_proxy_pass"]) == 1]
    if int(args.min_selection_epoch) > 1:
        eligible_after = [r for r in eligible if int(r.get("epoch", 0)) >= int(args.min_selection_epoch)]
        all_after = [r for r in summaries if int(r.get("epoch", 0)) >= int(args.min_selection_epoch)]
    else:
        eligible_after, all_after = eligible, summaries
    pool = eligible_after if eligible_after else (eligible if eligible else (all_after if all_after else summaries))
    if args.selection_strategy == "latest_eligible":
        selected = sorted(pool, key=lambda r: (int(r.get("epoch", 0)), fnum(r.get("precommit_proxy_score"))), reverse=True)[0]
    elif args.selection_strategy == "best_after_warmup":
        selected = sorted(pool, key=lambda r: (fnum(r.get("precommit_proxy_score")), fnum(r.get("proxy_margin_gain_vs_noop")), -int(r.get("epoch", 0))), reverse=True)[0]
    else:
        selected = sorted(eligible if eligible else summaries, key=lambda r: (fnum(r.get("precommit_proxy_score")), fnum(r.get("proxy_margin_gain_vs_noop")), -int(r.get("epoch", 0))), reverse=True)[0]
    selected_epoch = int(selected["epoch"])
    selection = {"selected_epoch": selected_epoch, "selection_strategy": args.selection_strategy, "min_selection_epoch": int(args.min_selection_epoch), "selection_had_proxy_pass": int(bool(eligible)), "selected_precommit_proxy_score": fnum(selected.get("precommit_proxy_score")), "selected_proxy_margin_gain_vs_noop": fnum(selected.get("proxy_margin_gain_vs_noop")), "selected_proxy_usage_delta_vs_noop": fnum(selected.get("proxy_usage_delta_vs_noop")), "selected_proxy_logit_norm_ratio_vs_noop": fnum(selected.get("proxy_logit_norm_ratio_vs_noop")), "precommit_proxy_pass_epoch_count": len(eligible), "source_actuator": source_actuator, "control_actuator": control_actuator}
    rows.append({"stage": "V1223_P4_PRECOMMIT_PROXY_SELECTION", **selection, "promotion_allowed": 0, "no_fake": 1})
    return rows, selection


def train_to_epoch(args, source, base, x_train, y_train, x_val, y_val, xq, base_query_logits, opt_delta, seed, device, selected_epoch):
    variants, source_actuator, control_actuator = build_variants(args, source, base, x_train, xq, base_query_logits, opt_delta, seed, device)
    gens = {kind: torch.Generator(device=device).manual_seed(int(seed) + int(args.train_seed_base) + int(v["idx"]) + len(args.role_policy)) for kind, v in variants.items()}
    for epoch in range(1, int(selected_epoch) + 1):
        for kind, v in variants.items():
            init_acc = fnum(exp.v1252._classification_basic(v["model"], x_val, y_val).get("acc")) if not v["points"] else v["points"][-1][1]
            if not v["points"]:
                v["points"].append((float(v["elapsed_ms"]), init_acc))
            dt = train_one_epoch(args, v, x_train, y_train, gens[kind], device)
            v["elapsed_ms"] += float(dt)
            ev = exp.v1252._classification_basic(v["model"], x_val, y_val)
            v["points"].append((float(v["elapsed_ms"]), fnum(ev.get("acc"))))
    finals = {}
    rows = []
    for kind, v in variants.items():
        ev = exp.v1252._classification_basic(v["model"], x_val, y_val)
        q90 = float(torch.tensor(v["step_times"]).quantile(0.90).item()) if v["step_times"] else float("nan")
        final = {"stage": "V1223_P4_PRECOMMIT_PROXY_SELECTED_AUDIT_VARIANT", "variant_kind": kind, "actuator_id": v["aid"], "role": v["role"], "selected_epoch": int(selected_epoch), "role_policy": args.role_policy, "final_acc": ev.get("acc", ""), "final_NLL": ev.get("NLL", ""), "final_ECE": ev.get("ECE", ""), "final_CEp99": ev.get("CEp99", ""), "auc_error_time": auc_error_time(v["points"]), "step_time_q90_ms": q90, **v["direct"]}
        finals[kind] = final
        rows.append(final)
    return variants, finals, rows, source_actuator, control_actuator


def audit_selected(args, variants, finals, xb, yb, xq, yq, seed):
    rows = []
    seed_summaries = []
    for lseed in parse_ints(args.linec_seeds):
        metrics = {}
        for kind, v in variants.items():
            try:
                metrics[kind] = exp.v1221._linec_metrics(v["model"], xb, yb, xq, yq, int(seed) + int(lseed) + int(v["idx"]), int(args.linec_sketch_dim))
            except Exception as exc:
                metrics[kind] = {"error": f"{type(exc).__name__}: {exc}"}
            rows.append({"stage": "V1223_P4_PRECOMMIT_PROXY_SELECTED_AUDIT_LINEC", "linec_seed_base": int(lseed), "variant_kind": kind, **{f"linec_{k}": val for k, val in metrics[kind].items()}})
        src, noop = metrics["p3_source_actuator"], metrics["noop_control"]
        one = {"stage": "V1223_P4_PRECOMMIT_PROXY_SELECTED_AUDIT_SEED", "linec_seed_base": int(lseed), "coupling_pass": int(fnum(src.get("CouplingR2")) >= fnum(noop.get("CouplingR2"))), "noise_pass": int(fnum(src.get("NoiseSignalLeak")) <= fnum(noop.get("NoiseSignalLeak"))), "reservoir_pass": int(fnum(src.get("RealSignalReservoirRatio")) <= fnum(noop.get("RealSignalReservoirRatio"))), "source_CouplingR2": fnum(src.get("CouplingR2")), "noop_CouplingR2": fnum(noop.get("CouplingR2")), "source_NoiseSignalLeak": fnum(src.get("NoiseSignalLeak")), "noop_NoiseSignalLeak": fnum(noop.get("NoiseSignalLeak")), "source_RealSignalReservoirRatio": fnum(src.get("RealSignalReservoirRatio")), "noop_RealSignalReservoirRatio": fnum(noop.get("RealSignalReservoirRatio"))}
        one["linec_seed_pass"] = int(one["coupling_pass"] and one["noise_pass"] and one["reservoir_pass"])
        seed_summaries.append(one)
        rows.append(one)
    src, noop, ctrl = finals["p3_source_actuator"], finals["noop_control"], finals["direct_logit_compensated_control"]
    src_acc, noop_acc, ctrl_acc = fnum(src.get("final_acc")), fnum(noop.get("final_acc")), fnum(ctrl.get("final_acc"))
    task_pass = int(src_acc >= max(noop_acc, ctrl_acc) + 0.005 and fnum(src.get("final_NLL")) <= fnum(noop.get("final_NLL")) and fnum(src.get("final_CEp99")) <= fnum(noop.get("final_CEp99")) + 0.05 and fnum(src.get("final_ECE")) <= fnum(noop.get("final_ECE")) + 0.02 and fnum(src.get("auc_error_time")) <= fnum(noop.get("auc_error_time")) and fnum(src.get("step_time_q90_ms")) <= fnum(noop.get("step_time_q90_ms")) * 1.05)
    pass_count = sum(int(r["linec_seed_pass"]) for r in seed_summaries)
    all_linec = int(pass_count == len(seed_summaries) and len(seed_summaries) > 0)
    majority_linec = int(pass_count >= math.ceil(len(seed_summaries) / 2.0) and len(seed_summaries) > 0)
    return rows, seed_summaries, {"task_gate_pass": task_pass, "linec_seed_count": len(seed_summaries), "linec_seed_pass_count": pass_count, "strict_all_sketch_pass": int(task_pass and all_linec), "strict_majority_sketch_pass": int(task_pass and majority_linec), "source_final_acc": src_acc, "noop_final_acc": noop_acc, "control_final_acc": ctrl_acc, "source_vs_noop_acc_delta": src_acc - noop_acc, "source_vs_control_acc_delta": src_acc - ctrl_acc, "source_NLL": fnum(src.get("final_NLL")), "noop_NLL": fnum(noop.get("final_NLL")), "source_CEp99": fnum(src.get("final_CEp99")), "noop_CEp99": fnum(noop.get("final_CEp99")), "source_ECE": fnum(src.get("final_ECE")), "noop_ECE": fnum(noop.get("final_ECE")), "seed_summaries": seed_summaries}


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
    ap.add_argument("--train-seed-base", type=int, default=12239400)
    ap.add_argument("--linec-seeds", default="12239500,12240600,12241600,12242600,12243600")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--source-actuator-override", default="")
    ap.add_argument("--control-actuator-override", default="")
    ap.add_argument("--compensation-reference", choices=["query", "train"], default="query")
    ap.add_argument("--compensation-batch", type=int, default=0)
    ap.add_argument("--proxy-batch", type=int, default=128)
    ap.add_argument("--min-margin-gain", type=float, default=0.0)
    ap.add_argument("--min-usage-delta", type=float, default=-0.03)
    ap.add_argument("--min-control-usage-delta", type=float, default=-0.04)
    ap.add_argument("--max-logit-norm-ratio", type=float, default=1.25)
    ap.add_argument("--max-top-class-share", type=float, default=0.65)
    ap.add_argument("--selection-strategy", choices=["best_score", "latest_eligible", "best_after_warmup"], default="best_score")
    ap.add_argument("--min-selection-epoch", type=int, default=1)
    ap.add_argument("--artifact-prefix", default="v1223_p4_precommit_proxy_checkpoint")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("precommit proxy checkpoint requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(out_dir)
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    rows, selection = run_proxy_pass(args, source, base, x_train, y_train, xq, base_query_logits, opt_delta, seed, device)
    # Rebuild deterministic context for audit after precommit epoch selection; selected_epoch was chosen without labels/audit targets.
    base2, x_train2, y_train2, x_val2, y_val2, xb2, yb2, xq2, yq2, base_query_logits2, opt_delta2, dataset2, seed2 = role_scan.build_context(args, source, device)
    variants, finals, audit_variant_rows, source_actuator, control_actuator = train_to_epoch(args, source, base2, x_train2, y_train2, x_val2, y_val2, xq2, base_query_logits2, opt_delta2, seed2, device, int(selection["selected_epoch"]))
    rows.extend(audit_variant_rows)
    linec_rows, seed_summaries, audit = audit_selected(args, variants, finals, xb2, yb2, xq2, yq2, seed2)
    rows.extend(linec_rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    summary = {"stage": "V1223_P4_PRECOMMIT_PROXY_CHECKPOINT_RESULT", "artifact_prefix": args.artifact_prefix, "artifact_csv": exp.rel(csv_path), "source_actuator": source_actuator, "control_actuator": control_actuator, "source_dataset": dataset2, "source_seed": seed2, "source_norm_budget": source.get("norm_budget", ""), "source_signed_direction": source.get("signed_direction", ""), "epochs_scanned": int(args.epochs), "selected_epoch": int(selection["selected_epoch"]), "selection_strategy": selection.get("selection_strategy", ""), "min_selection_epoch": int(selection.get("min_selection_epoch", 1)), "selection_had_proxy_pass": int(selection["selection_had_proxy_pass"]), "precommit_proxy_pass_epoch_count": int(selection["precommit_proxy_pass_epoch_count"]), "selected_precommit_proxy_score": fnum(selection["selected_precommit_proxy_score"]), "proxy_uses_labels": 0, "proxy_uses_ce_or_loss": 0, "proxy_uses_linec_or_audit_target": 0, "promotion_allowed": 0, "status": "precommit_proxy_selected_checkpoint_audit_all_pass_not_promoted" if int(audit["strict_all_sketch_pass"]) else "executed_no_official_pass", "no_fake": 1, **audit}
    rows.append(summary)
    exp.write_csv_rows(csv_path, rows)
    exp.write_json(json_path, summary)
    state_path = out_dir / "v1223_route_decision.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"p4_precommit_proxy_checkpoint_summary": exp.rel(json_path), "p4_precommit_proxy_strict_all_pass": int(audit["strict_all_sketch_pass"]), "p4_precommit_proxy_strict_majority_pass": int(audit["strict_majority_sketch_pass"]), "p4_precommit_proxy_promotion_allowed": 0, "official_success_reached": 0, "p4_pass": 0, "no_fake": 1})
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
