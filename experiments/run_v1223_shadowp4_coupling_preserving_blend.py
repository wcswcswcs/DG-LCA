#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, json, math, sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def make_base(args, device):
    dataset = exp.v120._canonical_dataset(str(args.dataset))
    seed = int(args.seed)
    load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=seed)
    data = exp.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    spec = exp.linea.ablation_specs(int(input_dim), int(output_dim))["A1-noYForStats"]["spec"]
    base = exp.linea.make_model("A1-noYForStats", spec, int(input_dim), int(output_dim), x_train, y_train, device, seed + 1223000, 0)
    b = min(int(args.linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
    xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
    with torch.no_grad():
        base_logits = base(xq).detach()
    try:
        opt_updated = exp.v1252._take_adamw_window(base, xb, yb, 2.0e-3, 1.0e-3)
        opt_delta = exp.v1221._copy_state_delta(base, opt_updated)
    except Exception:
        opt_delta = {}
    base_metrics = exp.v1221._linec_metrics(base, xb, yb, xq, yq, seed + 12239100, int(args.linec_sketch_dim))
    base_eval = exp.v1252._classification_basic(base, x_val, y_val)
    refs = {
        "base_p": getattr(base, "quad_proj", torch.empty(0, device=device)).detach().clone() if hasattr(base, "quad_proj") else torch.empty(0, device=device),
        "base_direct": getattr(base, "direct_readout", torch.empty(0, device=device)).detach().clone() if hasattr(base, "direct_readout") else torch.empty(0, device=device),
        "base_branch": getattr(base, "branch_scale", torch.empty(0, device=device)).detach().clone() if hasattr(base, "branch_scale") else torch.empty(0, device=device),
        "base_gain": getattr(base, "logit_gain", torch.empty(0, device=device)).detach().clone() if hasattr(base, "logit_gain") else torch.empty(0, device=device),
    }
    return dataset, seed, base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_logits, opt_delta, base_metrics, base_eval, refs


def apply_blend(model, p3_budget, shadow_budget, seed, opt_delta, xq, base_logits, device):
    gen1 = torch.Generator(device=device).manual_seed(seed + int(abs(p3_budget) * 1000000) + 281)
    exp.apply_v1223_actuator(model, "I24-DirectLogitCompensatedQuadRelease", float(p3_budget), 1.0, gen1, opt_delta)
    gen2 = torch.Generator(device=device).manual_seed(seed + int(abs(shadow_budget) * 1000000) + 282)
    exp.apply_v1223_actuator(model, "I24-DirectLogitCompensatedQuadRelease", float(shadow_budget), -1.0, gen2, opt_delta)
    return exp.apply_direct_readout_logit_compensation(model, xq, base_logits)


def apply_control(model, total_budget, seed, opt_delta, xq, base_logits, device):
    gen = torch.Generator(device=device).manual_seed(seed + int(abs(total_budget) * 1000000) + 383)
    exp.apply_v1223_actuator(model, "DirectLogitCompensatedRandomControl", float(total_budget), 1.0, gen, opt_delta)
    return exp.apply_direct_readout_logit_compensation(model, xq, base_logits)


def audit_metrics(model, base_logits, xb, yb, xq, yq, x_val, y_val, base_metrics, base_eval, refs, seed, sketch_dim, role="quad_proj"):
    with torch.no_grad():
        logits = model(xq).detach()
    p_now = getattr(model, "quad_proj", torch.empty(0, device=logits.device)).detach()
    base_p = refs["base_p"]
    angle = exp.v1221._projector_angle_deg(base_p, p_now) if int(base_p.numel()) and int(p_now.numel()) == int(base_p.numel()) else 0.0
    drift = float((logits - base_logits).abs().max().item())
    sketch_delta = float((logits - base_logits).float().norm().div(math.sqrt(max(1, int(logits.numel())))).item())
    metrics = exp.v1221._linec_metrics(model, xb, yb, xq, yq, seed + 12239200, int(sketch_dim))
    eval_now = exp.v1252._classification_basic(model, x_val, y_val)
    direct_delta = exp.tensor_delta_norm(refs["base_direct"], getattr(model, "direct_readout", None))
    branch_delta = exp.tensor_delta_norm(refs["base_branch"], getattr(model, "branch_scale", None))
    gain_delta = exp.tensor_delta_norm(refs["base_gain"], getattr(model, "logit_gain", None))
    n_delta = metrics["NoiseSignalLeak"] - base_metrics["NoiseSignalLeak"]
    r_delta = metrics["RealSignalReservoirRatio"] - base_metrics["RealSignalReservoirRatio"]
    c_delta = metrics["CouplingR2"] - base_metrics["CouplingR2"]
    return {
        "projector_angle_deg": angle,
        "logit_max_abs_drift": drift,
        "sketch_delta_fro": sketch_delta,
        "direct_readout_shift_norm": direct_delta,
        "branch_scale_shift_norm": branch_delta,
        "logit_gain_shift_norm": gain_delta,
        "role_safe_movement_pass": exp.role_safe_pass(role, sketch_delta, angle, drift, direct_delta, branch_delta, gain_delta),
        "CouplingR2_delta": c_delta,
        "NoiseSignalLeak_delta_audit": n_delta,
        "RealSignalReservoirRatio_delta_audit": r_delta,
        "CEp99_delta": exp.safe_float(eval_now.get("CEp99"), 0.0) - exp.safe_float(base_eval.get("CEp99"), 0.0),
        "ECE_delta": exp.safe_float(eval_now.get("ECE"), 0.0) - exp.safe_float(base_eval.get("ECE"), 0.0),
    }


def train_variant(base, kind, p3_budget, shadow_budget, total_budget, seed, args, opt_delta, x_train, y_train, x_val, y_val, xq, base_logits, device):
    trial = copy.deepcopy(base).to(device)
    if kind == "source_blend":
        direct = apply_blend(trial, p3_budget, shadow_budget, seed, opt_delta, xq, base_logits, device)
    elif kind == "random_control":
        direct = apply_control(trial, total_budget, seed, opt_delta, xq, base_logits, device)
    else:
        direct = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
    init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
    q90 = p4m.train_short(trial, x_train, y_train, int(args.batch_size), int(args.epochs), float(args.lr), float(args.weight_decay), seed + 12239300 + len(kind), device)
    final_eval = exp.v1252._classification_basic(trial, x_val, y_val)
    return {"variant_kind": kind, "init_acc": init_eval.get("acc", ""), "init_NLL": init_eval.get("NLL", ""), "init_ECE": init_eval.get("ECE", ""), "init_CEp99": init_eval.get("CEp99", ""), "final_acc": final_eval.get("acc", ""), "final_NLL": final_eval.get("NLL", ""), "final_ECE": final_eval.get("ECE", ""), "final_CEp99": final_eval.get("CEp99", ""), "step_time_q90_ms": q90, **direct}


def p4_pass_from(rows):
    by = {r["variant_kind"]: r for r in rows}
    src, noop, ctrl = by["source_blend"], by["noop_control"], by["random_control"]
    src_acc = exp.safe_float(src.get("final_acc"), float("nan"))
    noop_acc = exp.safe_float(noop.get("final_acc"), float("nan"))
    ctrl_acc = exp.safe_float(ctrl.get("final_acc"), float("nan"))
    src_nll = exp.safe_float(src.get("final_NLL"), float("nan"))
    noop_nll = exp.safe_float(noop.get("final_NLL"), float("nan"))
    src_cep99 = exp.safe_float(src.get("final_CEp99"), float("nan"))
    noop_cep99 = exp.safe_float(noop.get("final_CEp99"), float("nan"))
    passed = int(math.isfinite(src_acc) and math.isfinite(noop_acc) and math.isfinite(ctrl_acc) and src_acc >= max(noop_acc, ctrl_acc) + 0.005 and (not math.isfinite(src_nll) or not math.isfinite(noop_nll) or src_nll <= noop_nll) and (not math.isfinite(src_cep99) or not math.isfinite(noop_cep99) or src_cep99 <= noop_cep99 + 0.05))
    return passed, src_acc, noop_acc, ctrl_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="KMNIST")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--p3-budgets", default="0.018,0.021,0.024,0.027,0.03")
    ap.add_argument("--shadow-budgets", default="0.006,0.012,0.018,0.024,0.03")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.0e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--artifact-prefix", default="v1223_shadowp4_coupling_preserving_blend")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("blend diagnostic requires CUDA")
    torch.cuda.set_device(device)
    dataset, seed, base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_logits, opt_delta, base_metrics, base_eval, refs = make_base(args, device)
    rows = []
    summaries = []
    for p3_budget in parse_floats(args.p3_budgets):
        for shadow_budget in parse_floats(args.shadow_budgets):
            total_budget = math.sqrt(float(p3_budget) ** 2 + float(shadow_budget) ** 2)
            source = copy.deepcopy(base).to(device)
            direct = apply_blend(source, p3_budget, shadow_budget, seed, opt_delta, xq, base_logits, device)
            source_metrics = audit_metrics(source, base_logits, xb, yb, xq, yq, x_val, y_val, base_metrics, base_eval, refs, seed, int(args.linec_sketch_dim), "quad_proj")
            control = copy.deepcopy(base).to(device)
            control_direct = apply_control(control, total_budget, seed, opt_delta, xq, base_logits, device)
            control_metrics = audit_metrics(control, base_logits, xb, yb, xq, yq, x_val, y_val, base_metrics, base_eval, refs, seed + 17, int(args.linec_sketch_dim), "quad_proj_control")
            source_score = -source_metrics["NoiseSignalLeak_delta_audit"] - source_metrics["RealSignalReservoirRatio_delta_audit"]
            control_score = -control_metrics["NoiseSignalLeak_delta_audit"] - control_metrics["RealSignalReservoirRatio_delta_audit"]
            control_gap = source_score - control_score
            p3_pass = int(source_metrics["role_safe_movement_pass"] == 1 and source_metrics["NoiseSignalLeak_delta_audit"] <= -0.01 and source_metrics["RealSignalReservoirRatio_delta_audit"] <= -0.01 and source_metrics["CouplingR2_delta"] >= 0.02 and control_gap >= 0.005 and source_metrics["CEp99_delta"] <= 0.05 and source_metrics["ECE_delta"] <= 0.02)
            train_rows = []
            for kind in ["noop_control", "source_blend", "random_control"]:
                tr = train_variant(base, kind, p3_budget, shadow_budget, total_budget, seed, args, opt_delta, x_train, y_train, x_val, y_val, xq, base_logits, device)
                tr.update({"stage": "V1223_SHADOWP4_COUPLING_BLEND_P4", "dataset": dataset, "seed": seed, "p3_budget": p3_budget, "shadow_budget": shadow_budget, "total_control_budget": total_budget})
                train_rows.append(tr)
                rows.append(tr)
            p4_pass, src_acc, noop_acc, ctrl_acc = p4_pass_from(train_rows)
            row = {"stage": "V1223_SHADOWP4_COUPLING_BLEND_SUMMARY", "dataset": dataset, "seed": seed, "p3_budget": p3_budget, "shadow_budget": shadow_budget, "total_control_budget": total_budget, **source_metrics, "source_direct_logit_compensation_norm": direct.get("direct_logit_compensation_norm", ""), "control_direct_logit_compensation_norm": control_direct.get("direct_logit_compensation_norm", ""), "control_gap": control_gap, "p3_pass": p3_pass, "source_final_acc": src_acc, "noop_final_acc": noop_acc, "control_final_acc": ctrl_acc, "source_vs_noop_acc_delta": src_acc - noop_acc if math.isfinite(src_acc) and math.isfinite(noop_acc) else "", "source_vs_control_acc_delta": src_acc - ctrl_acc if math.isfinite(src_acc) and math.isfinite(ctrl_acc) else "", "p4_pass": p4_pass, "both_p3_p4_pass": int(p3_pass and p4_pass), "promotion_allowed": 0, "no_fake": 1}
            summaries.append(row)
            rows.append(row)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, rows)
    summaries.sort(key=lambda r: (int(r.get("both_p3_p4_pass", 0)), int(r.get("p3_pass", 0)), int(r.get("p4_pass", 0)), exp.safe_float(r.get("source_vs_control_acc_delta"), -999.0), exp.safe_float(r.get("control_gap"), -999.0)), reverse=True)
    result = {"stage": "V1223_SHADOWP4_COUPLING_BLEND_AGGREGATE", "artifact_csv": exp.rel(csv_path), "summary_rows": len(summaries), "any_both_p3_p4_pass": int(any(int(r.get("both_p3_p4_pass", 0)) for r in summaries)), "any_p3_pass": int(any(int(r.get("p3_pass", 0)) for r in summaries)), "any_p4_pass": int(any(int(r.get("p4_pass", 0)) for r in summaries)), "best_summaries": summaries[:10], "promotion_allowed": 0, "no_fake": 1}
    exp.write_json(json_path, result)
    state_path = out_dir / "v1223_route_decision.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.update({"shadowp4_coupling_blend_rows": len(rows), "shadowp4_coupling_blend_summary": exp.rel(json_path), "shadowp4_coupling_blend_any_both_pass": result["any_both_p3_p4_pass"]})
        zip_path = exp.package_zip(out_dir)
        state["code_review_packet"] = exp.rel(zip_path)
        state["code_review_packet_sha256"] = exp.sha256_file(zip_path)
        state.update(exp.write_required_manifest(out_dir))
        exp.write_hash_manifest(out_dir)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        result["code_review_packet_sha256"] = state["code_review_packet_sha256"]
        result["required_artifact_missing_count"] = state.get("required_artifact_missing_count")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
