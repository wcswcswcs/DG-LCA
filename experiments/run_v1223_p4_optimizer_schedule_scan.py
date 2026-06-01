#!/usr/bin/env python
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def build_context(args, source, device):
    dataset = exp.v120._canonical_dataset(str(source["dataset"]))
    seed = int(float(source["seed"]))
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
        base_query_logits = base(xq).detach()
        base_train_logits = base(xb).detach()
    try:
        opt_updated = exp.v1252._take_adamw_window(base, xb, yb, 2.0e-3, 1.0e-3)
        opt_delta = exp.v1221._copy_state_delta(base, opt_updated)
    except Exception:
        opt_delta = {}
    return base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_train_logits, base_query_logits, opt_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs-list", default="3,5")
    ap.add_argument("--lrs", default="0.0005,0.001,0.002,0.004,0.008")
    ap.add_argument("--weight-decays", default="0,0.001")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--modes", default="per_variant_query")
    ap.add_argument("--artifact-prefix", default="v1223_functional_p4_optimizer_schedule_scan")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("optimizer schedule scan requires CUDA")
    torch.cuda.set_device(device)
    source = p4m.pick_source(out_dir)
    ctx = build_context(args, source, device)
    all_rows = []
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    for epochs in parse_ints(args.epochs_list):
        for lr in parse_floats(args.lrs):
            for wd in parse_floats(args.weight_decays):
                run_args = SimpleNamespace(**vars(args))
                run_args.epochs = int(epochs)
                run_args.lr = float(lr)
                run_args.weight_decay = float(wd)
                for mode in modes:
                    rows = p4m.run_mode(run_args, mode, ctx[0], source, *ctx[1:], device)
                    for row in rows:
                        row["schedule_epochs"] = int(epochs)
                        row["schedule_lr"] = float(lr)
                        row["schedule_weight_decay"] = float(wd)
                        row["source_dataset"] = source.get("dataset", "")
                        row["source_seed"] = source.get("seed", "")
                        row["source_norm_budget"] = source.get("norm_budget", "")
                        row["source_signed_direction"] = source.get("signed_direction", "")
                        row["source_CouplingR2_delta"] = source.get("CouplingR2_delta", "")
                        row["source_control_gap"] = source.get("control_gap", "")
                    all_rows.extend(rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    json_path = out_dir / f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path, all_rows)
    summaries = [r for r in all_rows if r.get("stage") == "V1223_FUNCTIONAL_P4_COMPENSATION_MODE_SUMMARY"]
    summaries.sort(key=lambda r: (exp.safe_int(r.get("p4_pass"), 0), exp.safe_float(r.get("source_vs_noop_acc_delta"), -999.0), exp.safe_float(r.get("source_vs_control_acc_delta"), -999.0)), reverse=True)
    result = {
        "stage": "V1223_FUNCTIONAL_P4_OPTIMIZER_SCHEDULE_SCAN",
        "artifact_csv": exp.rel(csv_path),
        "source_actuator": source.get("actuator_id", ""),
        "source_dataset": source.get("dataset", ""),
        "source_seed": source.get("seed", ""),
        "source_norm_budget": source.get("norm_budget", ""),
        "source_signed_direction": source.get("signed_direction", ""),
        "summary_rows": len(summaries),
        "any_p4_pass": int(any(exp.safe_int(r.get("p4_pass"), 0) for r in summaries)),
        "best_summaries": summaries[:10],
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(json_path, result)
    state_path = out_dir / "v1223_route_decision.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.update({
            "p4_optimizer_schedule_scan_rows": len(all_rows),
            "p4_optimizer_schedule_scan_any_pass": result["any_p4_pass"],
            "p4_optimizer_schedule_scan_summary": exp.rel(json_path),
        })
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
