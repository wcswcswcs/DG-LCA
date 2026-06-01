#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, json, math, sys
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


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def eval_logits(logits: torch.Tensor, y: torch.Tensor, temperature: float) -> dict[str, float]:
    z = logits / max(float(temperature), 1.0e-6)
    ce = F.cross_entropy(z, y, reduction="none")
    probs = torch.softmax(z, dim=-1)
    conf, pred = probs.max(dim=-1)
    acc = (pred == y).float()
    ece = torch.zeros((), device=z.device)
    for lo, hi in zip(torch.linspace(0, 1, 11, device=z.device)[:-1], torch.linspace(0, 1, 11, device=z.device)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece = ece + mask.float().mean() * (conf[mask].mean() - acc[mask].mean()).abs()
    return {"acc": float(acc.mean().item()), "NLL": float(ce.mean().item()), "ECE": float(ece.item()), "CEp99": float(torch.quantile(ce.float(), 0.99).item())}


def train_variants(args, source, ctx, device, role_policy: str, epochs: int, lr: float, wd: float):
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_query_logits, opt_delta, dataset, seed = ctx
    budget = float(source["norm_budget"])
    signed = float(source["signed_direction"])
    actuator = str(source["actuator_id"])
    variants = [("NoOpMatchedOverhead", "noop_control", 0.0, signed), (actuator, "p3_source_actuator", budget, signed), ("DirectLogitCompensatedRandomControl", "direct_logit_compensated_control", budget, signed)]
    trained = []
    for idx, (aid, kind, bgt, sgn) in enumerate(variants):
        trial = copy.deepcopy(base).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(bgt) * 100000) + len(aid) * 17 + (1 if sgn > 0 else 2))
        role = exp.apply_v1223_actuator(trial, aid, float(bgt), float(sgn), gen, opt_delta)
        direct = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
        if aid in {"I24-DirectLogitCompensatedQuadRelease", "I25-DirectLogitCompensatedShadowRelease", "DirectLogitCompensatedRandomControl"}:
            direct = exp.apply_direct_readout_logit_compensation(trial, xq, base_query_logits)
        role_scan.train_short_role(trial, x_train, y_train, int(args.batch_size), int(epochs), float(lr), float(wd), seed + 12239600 + idx + len(role_policy), device, role_policy)
        with torch.no_grad():
            logits = trial(x_val).detach()
        trained.append({"kind": kind, "aid": aid, "role": role, "logits": logits, "direct": direct})
    return trained, x_val, y_val, dataset, seed


def p4_from_metrics(metrics_by_kind: dict[str, dict[str, float]]) -> tuple[int, float, float, float]:
    src = metrics_by_kind["p3_source_actuator"]
    noop = metrics_by_kind["noop_control"]
    ctrl = metrics_by_kind["direct_logit_compensated_control"]
    src_acc, noop_acc, ctrl_acc = src["acc"], noop["acc"], ctrl["acc"]
    p4_pass = int(math.isfinite(src_acc) and math.isfinite(noop_acc) and math.isfinite(ctrl_acc) and src_acc >= max(noop_acc, ctrl_acc) + 0.005 and src["NLL"] <= noop["NLL"] and src["CEp99"] <= noop["CEp99"] + 0.05)
    return p4_pass, src_acc, noop_acc, ctrl_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs-list", default="5")
    ap.add_argument("--lrs", default="0.001")
    ap.add_argument("--weight-decays", default="0.001")
    ap.add_argument("--role-policies", default="direct_only,direct_gain,direct_branch_gain,freeze_quad")
    ap.add_argument("--temperatures", default="0.75,1.0,1.25,1.5,2.0,3.0,4.0")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--artifact-prefix", default="v1223_functional_p4_temperature_calibration_scan")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("temperature calibration scan requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(out_dir)
    ctx = role_scan.build_context(args, source, device)
    rows: list[dict[str, Any]] = []
    for role_policy in parse_csv(args.role_policies):
        for epochs in parse_ints(args.epochs_list):
            for lr in parse_floats(args.lrs):
                for wd in parse_floats(args.weight_decays):
                    trained, _x_val, y_val, dataset, seed = train_variants(args, source, ctx, device, role_policy, epochs, lr, wd)
                    for temp in parse_floats(args.temperatures):
                        metrics_by_kind = {item["kind"]: eval_logits(item["logits"], y_val, temp) for item in trained}
                        for item in trained:
                            m = metrics_by_kind[item["kind"]]
                            rows.append({"stage": "V1223_FUNCTIONAL_P4_TEMPERATURE_CALIBRATION", "variant_kind": item["kind"], "role_policy": role_policy, "temperature": temp, "epochs": int(epochs), "lr": float(lr), "weight_decay": float(wd), "actuator_id": item["aid"], "role": item["role"], "dataset": dataset, "seed": seed, "final_acc": m["acc"], "final_NLL": m["NLL"], "final_ECE": m["ECE"], "final_CEp99": m["CEp99"], **item["direct"]})
                        p4_pass, src_acc, noop_acc, ctrl_acc = p4_from_metrics(metrics_by_kind)
                        rows.append({"stage": "V1223_FUNCTIONAL_P4_TEMPERATURE_CALIBRATION_SUMMARY", "role_policy": role_policy, "temperature": temp, "epochs": int(epochs), "lr": float(lr), "weight_decay": float(wd), "p4_pass": p4_pass, "promotion_allowed": 0, "source_final_acc": src_acc, "noop_final_acc": noop_acc, "control_final_acc": ctrl_acc, "source_vs_noop_acc_delta": src_acc - noop_acc, "source_vs_control_acc_delta": src_acc - ctrl_acc, "source_NLL": metrics_by_kind["p3_source_actuator"]["NLL"], "noop_NLL": metrics_by_kind["noop_control"]["NLL"], "source_CEp99": metrics_by_kind["p3_source_actuator"]["CEp99"], "noop_CEp99": metrics_by_kind["noop_control"]["CEp99"], "status": "pass_audit_only_no_s5_route" if p4_pass else "executed_no_pass", "no_fake": 1})
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, rows)
    summaries = [r for r in rows if r.get("stage") == "V1223_FUNCTIONAL_P4_TEMPERATURE_CALIBRATION_SUMMARY"]
    summaries.sort(key=lambda r: (exp.safe_int(r.get("p4_pass"), 0), exp.safe_float(r.get("source_vs_noop_acc_delta"), -999.0), -exp.safe_float(r.get("source_NLL"), 999.0)), reverse=True)
    result = {"stage": "V1223_FUNCTIONAL_P4_TEMPERATURE_CALIBRATION_SCAN", "artifact_csv": exp.rel(csv_path), "source_actuator": source.get("actuator_id", ""), "source_dataset": source.get("dataset", ""), "source_seed": source.get("seed", ""), "source_norm_budget": source.get("norm_budget", ""), "source_signed_direction": source.get("signed_direction", ""), "summary_rows": len(summaries), "any_p4_pass": int(any(exp.safe_int(r.get("p4_pass"), 0) for r in summaries)), "best_summaries": summaries[:10], "promotion_allowed": 0, "no_fake": 1}
    exp.write_json(json_path, result)
    state_path = out_dir / "v1223_route_decision.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.update({"p4_temperature_calibration_scan_rows": len(rows), "p4_temperature_calibration_scan_any_pass": result["any_p4_pass"], "p4_temperature_calibration_scan_summary": exp.rel(json_path)})
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
