#!/usr/bin/env python3
"""v12.15 continuation P3 test from the fused primitive actuator evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import run_v1215_b320locked_lossagnostic_signal_estimator_primitive_instrumentation as v15
from experiments import run_v1215_continuation_actuator_budget_repair as cont


DEFAULT_ROOT = REPO_ROOT / "results" / "v12_15_b320locked_lossagnostic_signal_estimator_primitive_instrumentation"
EPS = 1.0e-12


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_control_updates(prev: Any, torch_mod: Any, model: Any, task_delta: Sequence[Any], seed: int) -> dict[str, dict[str, Any]]:
    random_delta = prev._random_like(task_delta, int(seed) + 17, torch_mod)
    snr_delta, snr_role = v15.filter_delta_by_role(model, task_delta, task_delta, "quad")
    return {
        "U0-NoOp": {
            "deltas": v15.zero_delta(task_delta),
            "control_id": "noop",
            "loss_agnostic_direction": 1,
            "label_used_for_direction": 0,
            "ce_vector_used_for_direction": 0,
        },
        "U1-RandomMatchedNorm": {
            "deltas": v15.normalize_like(random_delta, task_delta),
            "control_id": "random_matched_norm",
            "loss_agnostic_direction": 1,
            "label_used_for_direction": 0,
            "ce_vector_used_for_direction": 0,
        },
        "U2-TaskOnlyAdamW-Audit": {
            "deltas": task_delta,
            "control_id": "task_adamw_audit",
            "loss_agnostic_direction": 0,
            "label_used_for_direction": 1,
            "ce_vector_used_for_direction": 0,
        },
        "U3-SNRGatedAdamWResidual-Audit": {
            "deltas": snr_delta,
            "control_id": "snr_task_residual_audit",
            "loss_agnostic_direction": 0,
            "label_used_for_direction": 1,
            "ce_vector_used_for_direction": 0,
            **snr_role,
        },
    }


def candidate_delta(torch_mod: Any, model: Any, task_delta: Sequence[Any], seed_base: int, repair_id: str) -> tuple[list[Any], dict[str, Any]]:
    if repair_id != "K6-FusedQuadSignBudgetCap":
        raise ValueError(f"Unsupported continuation repair_id: {repair_id}")
    deltas, stats = cont.random_role_param_delta(
        torch_mod,
        model,
        task_delta,
        "quad",
        int(seed_base) + v15.stable_seed(repair_id),
        "sign",
    )
    return deltas, {
        "repair_id": repair_id,
        "candidate_id": "B15-E-FusedQuadSignGateOpenActuator",
        "candidate_family": "B15-E",
        "source_raw_direction": "primitive_param_space",
        "selected_basis_ids": repair_id,
        "basis_update_count": 1,
        "loss_agnostic_direction": 1,
        "ce_vector_used_for_direction": 0,
        "label_used_for_direction": 0,
        "permuted_label_used_for_direction": 0,
        "validation_used_for_commit": 0,
        "dataset_name_used_for_commit": 0,
        **stats,
    }


def summarize_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("candidate_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for candidate_id, items in sorted(groups.items()):
        fail_counter: Counter[str] = Counter()
        for item in items:
            for reason in str(item.get("fail_reason", "")).split(";"):
                if reason:
                    fail_counter[reason] += 1
        values = lambda key: [v15.safe_float(item.get(key)) for item in items]
        out.append(
            {
                "stage": "V1215_CONTINUATION_P3_SUMMARY",
                "candidate_id": candidate_id,
                "candidate_family": items[0].get("candidate_family", ""),
                "rows": len(items),
                "p3_pass_rows": sum(v15.safe_int(item.get("p3_row_pass")) for item in items),
                "dataset_seed_pass_count": sum(v15.safe_int(item.get("p3_row_pass")) for item in items),
                "expected_dataset_seed_count": len(items),
                "strong_promotion": int(len(items) >= 9 and all(v15.safe_int(item.get("p3_row_pass")) for item in items)),
                "weak_promotion": int(len(items) >= 9 and sum(v15.safe_int(item.get("p3_row_pass")) for item in items) >= 6),
                "mean_CouplingR2_delta": float(np.mean(values("CouplingR2_delta"))) if items else 0.0,
                "mean_NoiseSignalLeak_delta": float(np.mean(values("NoiseSignalLeak_delta"))) if items else 0.0,
                "mean_RealSignalReservoirRatio_delta": float(np.mean(values("RealSignalReservoirRatio_delta"))) if items else 0.0,
                "best_noise_delta": min(values("NoiseSignalLeak_delta") or [0.0]),
                "best_reservoir_delta": min(values("RealSignalReservoirRatio_delta") or [0.0]),
                "mean_control_gap": float(np.mean(values("control_gap"))) if items else 0.0,
                "best_control_gap": max(values("control_gap") or [0.0]),
                "max_sketch_delta_fro": max(values("sketch_delta_fro") or [0.0]),
                "max_projector_angle_deg": max([max(v15.safe_float(item.get("signal_projector_angle")), v15.safe_float(item.get("reservoir_projector_angle"))) for item in items] or [0.0]),
                "max_logit_max_abs_drift": max(values("logit_max_abs_drift") or [0.0]),
                "fail_reason_counts": json.dumps(dict(sorted(fail_counter.items())), sort_keys=True),
            }
        )
    return out


def route_payload(out_dir: Path, rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    pass_rows = sum(v15.safe_int(row.get("p3_row_pass")) for row in rows)
    fail_counter: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("fail_reason", "")).split(";"):
            if reason:
                fail_counter[reason] += 1
    max_sketch = max([v15.safe_float(row.get("sketch_delta_fro")) for row in rows] or [0.0])
    max_angle = max([max(v15.safe_float(row.get("signal_projector_angle")), v15.safe_float(row.get("reservoir_projector_angle"))) for row in rows] or [0.0])
    best_noise = min([v15.safe_float(row.get("NoiseSignalLeak_delta")) for row in rows] or [0.0])
    best_res = min([v15.safe_float(row.get("RealSignalReservoirRatio_delta")) for row in rows] or [0.0])
    if pass_rows > 0:
        route = "R4-ContinuationP3PartialSurvivorFound"
        next_action = "run P4 short-run only if promotion summary opens strong or weak promotion"
    else:
        route = "R4-ContinuationFusedActuatorP3Failed"
        next_action = "objective is still misaligned; redesign explicit noise/reservoir spectral target before more P3 attempts"
    return {
        "stage": "V1215_CONTINUATION_P3_ROUTE",
        "generated_at": now_iso(),
        "run_id": out_dir.name,
        "route": route,
        "candidate_rows": len(rows),
        "p3_pass_rows": pass_rows,
        "p4_open": int(any(v15.safe_int(row.get("strong_promotion")) or v15.safe_int(row.get("weak_promotion")) for row in summary_rows)),
        "official_p3_claim": int(pass_rows > 0),
        "max_sketch_delta_fro": max_sketch,
        "max_projector_angle_deg": max_angle,
        "best_noise_delta": best_noise,
        "best_reservoir_delta": best_res,
        "fail_reason_counts": dict(sorted(fail_counter.items())),
        "repair_id": args.repair_id,
        "candidate_mode": args.candidate_mode,
        "budget_multiplier": float(args.budget_multiplier),
        "selector_probe_budget_multiplier": float(args.selector_probe_budget_multiplier),
        "target_logit_drift_train": float(args.target_logit_drift),
        "no_fake_proxy_cpu": 1,
        "next_recommended_action": next_action,
    }


def run_main(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    base_mod = v15.v1213()
    prev = base_mod.load_v1211_runner()
    torch_mod, F_mod, v120, v124, v1252, v1283 = prev._lazy_probe_modules()
    device = v1283._device_from_arg(str(args.probe_device))
    if str(device).startswith("cuda"):
        torch_mod.cuda.set_device(device.index if device.index is not None else 0)

    datasets = v15.parse_list(args.probe_datasets)
    seeds = v15.parse_ints(args.probe_seeds)
    windows = [v15.safe_int(item) for item in v15.parse_list(args.windows)]
    batch_size = int(args.functional_batch_size)
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        canon = v120._canonical_dataset(dataset)
        load_args = argparse.Namespace(
            data_root=args.data_root,
            no_download=bool(args.no_download),
            seed=seeds[0] if seeds else 0,
            train_size=int(args.probe_train_size),
            val_size=int(args.probe_val_size),
            test_size=int(args.probe_test_size),
            datasets=canon,
        )
        data = v120._load_vision_split(load_args, canon, train_size=int(args.probe_train_size), val_size=int(args.probe_val_size), test_size=int(args.probe_test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu = data[0], data[1], data[2], data[3]
        input_dim = int(data[6])
        output_dim = int(data[7])
        _, budget = v124._param_budget(input_dim, output_dim)
        specs = {spec.candidate_id: spec for spec in v1283.prim.primitive_specs(budget, input_dim, output_dim)}
        for seed in seeds:
            x_train = x_train_cpu.to(device=device, dtype=torch_mod.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch_mod.float32)
            y_val = y_val_cpu.to(device=device)
            base = v1283._make_model(v15.B320_ID, input_dim, output_dim, x_train, device, int(seed) + 1215500, specs, y_train)
            xq = x_val[:batch_size]
            yq = y_val[:batch_size]
            x_safe = x_train[: min(int(x_train.shape[0]), max(batch_size, int(args.safety_batch_factor) * batch_size))]
            for split_id in range(int(args.probe_splits)):
                start = (split_id * batch_size) % max(1, int(x_train.shape[0]) - batch_size + 1)
                xb = x_train[start : start + batch_size]
                yb = y_train[start : start + batch_size]
                task_delta = v1252._grad_delta(base, xb, yb, float(args.probe_lr))
                seed_base = v15.stable_seed(canon, seed, split_id, "v1215_continuation")
                for window in windows:
                    controls = make_control_updates(prev, torch_mod, base, task_delta, seed_base)
                    control_scores: list[float] = []
                    for control_id, control in controls.items():
                        metrics = v15.evaluate_direction_full(
                            prev,
                            torch_mod,
                            F_mod,
                            v1252,
                            v1283,
                            base,
                            v15.scale_delta(control["deltas"], float(window)),
                            task_delta,
                            xb,
                            yb,
                            xq,
                            yq,
                            seed_base + v15.stable_seed(control_id, window),
                            args,
                        )
                        control_scores.append(v15.safe_float(metrics.get("delta_score"), -999.0))
                        control_rows.append(
                            {
                                "stage": "V1215_CONTINUATION_P3_CONTROL",
                                "run_id": out_dir.name,
                                "control_id": control_id,
                                "dataset": canon,
                                "seed": seed,
                                "split_id": split_id,
                                "window": window,
                                "delta_score": metrics.get("delta_score", ""),
                                "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                                "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                                "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                                "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                            }
                        )
                    base_delta, stats = candidate_delta(torch_mod, base, task_delta, seed_base, str(args.repair_id))
                    selected_sign = 1.0
                    selector_score = ""
                    selector_probe_noise = ""
                    selector_probe_reservoir = ""
                    selector_probe_coupling = ""
                    if str(args.candidate_mode) == "bidirectional_noise_reservoir_selector":
                        best_probe: tuple[float, float, Mapping[str, Any]] | None = None
                        for sign in [1.0, -1.0]:
                            probe_requested = v15.scale_delta(base_delta, sign * float(window) * float(args.selector_probe_budget_multiplier))
                            probe_capped, probe_requested_drift, probe_cap_scale, probe_capped_drift = cont.safety_cap_delta(
                                v1252,
                                torch_mod,
                                base,
                                probe_requested,
                                x_safe,
                                float(args.target_logit_drift),
                            )
                            probe_metrics = v15.evaluate_direction_full(
                                prev,
                                torch_mod,
                                F_mod,
                                v1252,
                                v1283,
                                base,
                                probe_capped,
                                task_delta,
                                xb,
                                yb,
                                xq,
                                yq,
                                seed_base + v15.stable_seed(stats["candidate_id"], "selector", sign, window),
                                args,
                            )
                            noise = v15.safe_float(probe_metrics.get("NoiseSignalLeak_delta"))
                            reservoir = v15.safe_float(probe_metrics.get("RealSignalReservoirRatio_delta"))
                            coupling = v15.safe_float(probe_metrics.get("CouplingR2_delta"))
                            logit = v15.safe_float(probe_metrics.get("logit_max_abs_drift"))
                            cep99 = v15.safe_float(probe_metrics.get("CEp99_delta_audit"))
                            safety_penalty = max(0.0, logit - 0.05) + max(0.0, cep99 - 0.05)
                            score = -noise - reservoir + 0.10 * coupling - 10.0 * safety_penalty
                            selector_rows.append(
                                {
                                    "stage": "V1215_CONTINUATION_P3_SELECTOR_PROBE",
                                    "run_id": out_dir.name,
                                    "candidate_id": stats["candidate_id"],
                                    "repair_id": stats["repair_id"],
                                    "dataset": canon,
                                    "seed": seed,
                                    "split_id": split_id,
                                    "window": window,
                                    "selector_sign": sign,
                                    "selector_probe_budget_multiplier": float(args.selector_probe_budget_multiplier),
                                    "requested_logit_drift_train": probe_requested_drift,
                                    "safety_cap_scale": probe_cap_scale,
                                    "capped_logit_drift_train": probe_capped_drift,
                                    "selector_score": score,
                                    "CouplingR2_delta": coupling,
                                    "NoiseSignalLeak_delta": noise,
                                    "RealSignalReservoirRatio_delta": reservoir,
                                    "logit_max_abs_drift": logit,
                                    "CEp99_delta_audit": cep99,
                                }
                            )
                            item = (score, sign, probe_metrics)
                            if best_probe is None or item[0] > best_probe[0]:
                                best_probe = item
                        if best_probe is not None:
                            selector_score, selected_sign, selected_metrics = best_probe
                            selector_probe_noise = selected_metrics.get("NoiseSignalLeak_delta", "")
                            selector_probe_reservoir = selected_metrics.get("RealSignalReservoirRatio_delta", "")
                            selector_probe_coupling = selected_metrics.get("CouplingR2_delta", "")

                    requested = v15.scale_delta(base_delta, selected_sign * float(window) * float(args.budget_multiplier))
                    capped, requested_drift, cap_scale, capped_drift = cont.safety_cap_delta(
                        v1252,
                        torch_mod,
                        base,
                        requested,
                        x_safe,
                        float(args.target_logit_drift),
                    )
                    metrics = v15.evaluate_direction_full(
                        prev,
                        torch_mod,
                        F_mod,
                        v1252,
                        v1283,
                        base,
                        capped,
                        task_delta,
                        xb,
                        yb,
                        xq,
                        yq,
                        seed_base + v15.stable_seed(stats["candidate_id"], window, args.budget_multiplier),
                        args,
                    )
                    best_control = max(control_scores) if control_scores else 0.0
                    row = {
                        "stage": "V1215_CONTINUATION_P3_CANDIDATE",
                        "run_id": out_dir.name,
                        "candidate_id": stats["candidate_id"],
                        "candidate_family": stats["candidate_family"],
                        "repair_id": stats["repair_id"],
                        "dataset": canon,
                        "seed": seed,
                        "split_id": split_id,
                        "window": window,
                        "batch_size": batch_size,
                        "budget_multiplier": float(args.budget_multiplier),
                        "candidate_mode": args.candidate_mode,
                        "selected_sign": selected_sign,
                        "selector_score": selector_score,
                        "selector_probe_budget_multiplier": float(args.selector_probe_budget_multiplier),
                        "selector_probe_NoiseSignalLeak_delta": selector_probe_noise,
                        "selector_probe_RealSignalReservoirRatio_delta": selector_probe_reservoir,
                        "selector_probe_CouplingR2_delta": selector_probe_coupling,
                        "target_logit_drift_train": float(args.target_logit_drift),
                        "requested_logit_drift_train": requested_drift,
                        "safety_cap_scale": cap_scale,
                        "capped_logit_drift_train": capped_drift,
                        "loss_agnostic_direction": stats.get("loss_agnostic_direction", 1),
                        "ce_vector_used_for_direction": stats.get("ce_vector_used_for_direction", 0),
                        "label_used_for_direction": stats.get("label_used_for_direction", 0),
                        "permuted_label_used_for_direction": stats.get("permuted_label_used_for_direction", 0),
                        "validation_used_for_commit": stats.get("validation_used_for_commit", 0),
                        "dataset_name_used_for_commit": stats.get("dataset_name_used_for_commit", 0),
                        "selected_basis_ids": stats.get("selected_basis_ids", ""),
                        "basis_update_count": stats.get("basis_update_count", ""),
                        "primitive_param_mode": stats.get("primitive_param_mode", ""),
                        "primitive_param_role": stats.get("primitive_param_role", ""),
                        "primitive_param_active_tensors": stats.get("primitive_param_active_tensors", ""),
                        "primitive_param_active_params": stats.get("primitive_param_active_params", ""),
                        "CouplingR2_delta": metrics.get("CouplingR2_delta", ""),
                        "NoiseSignalLeak_delta": metrics.get("NoiseSignalLeak_delta", ""),
                        "RealSignalReservoirRatio_delta": metrics.get("RealSignalReservoirRatio_delta", ""),
                        "control_gap": v15.safe_float(metrics.get("delta_score"), 0.0) - best_control,
                        "delta_score": metrics.get("delta_score", ""),
                        "logit_max_abs_drift": metrics.get("logit_max_abs_drift", ""),
                        "CEp99_delta_audit": metrics.get("CEp99_delta_audit", ""),
                        "ECE_delta_audit": metrics.get("ECE_delta_audit", ""),
                        "holdout_loss_ratio_audit": metrics.get("holdout_loss_ratio_audit", ""),
                        "sketch_delta_fro": metrics.get("sketch_delta_fro", ""),
                        "sketch_delta_op": metrics.get("sketch_delta_op", ""),
                        "signal_projector_angle": metrics.get("signal_projector_angle", ""),
                        "reservoir_projector_angle": metrics.get("reservoir_projector_angle", ""),
                    }
                    fail = v15.p3_fail(row)
                    row["p3_row_pass"] = int(not fail)
                    row["fail_reason"] = ";".join(fail)
                    rows.append(row)

    summary_rows = summarize_candidate_rows(rows)
    route = route_payload(out_dir, rows, summary_rows, args)
    if selector_rows:
        v15.write_csv_rows(out_dir / "v1215_continuation_p3_selector_probes.csv", selector_rows)
    v15.write_csv_rows(out_dir / "v1215_continuation_p3_controls.csv", control_rows)
    v15.write_csv_rows(out_dir / "v1215_continuation_p3_candidates.csv", rows)
    v15.write_csv_rows(out_dir / "v1215_continuation_p3_summary.csv", summary_rows)
    with (out_dir / "v1215_continuation_p3_route_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(route, handle, indent=2, sort_keys=True)
    manifest = {
        name: v15.sha256_file(out_dir / name)
        for name in [
            "v1215_continuation_p3_controls.csv",
            "v1215_continuation_p3_selector_probes.csv",
            "v1215_continuation_p3_candidates.csv",
            "v1215_continuation_p3_summary.csv",
            "v1215_continuation_p3_route_decision.json",
        ]
        if (out_dir / name).exists()
    }
    with (out_dir / "v1215_continuation_p3_hash_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return route


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--probe-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--probe-seeds", default="0,1,2")
    parser.add_argument("--probe-splits", type=int, default=1)
    parser.add_argument("--windows", default="5")
    parser.add_argument("--functional-batch-size", type=int, default=32)
    parser.add_argument("--probe-train-size", type=int, default=512)
    parser.add_argument("--probe-val-size", type=int, default=256)
    parser.add_argument("--probe-test-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--sketch-dim", type=int, default=12)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--budget-multiplier", type=float, default=8.0)
    parser.add_argument("--candidate-mode", default="fixed_gate_open_actuator", choices=["fixed_gate_open_actuator", "bidirectional_noise_reservoir_selector"])
    parser.add_argument("--selector-probe-budget-multiplier", type=float, default=2.0)
    parser.add_argument("--target-logit-drift", type=float, default=0.045)
    parser.add_argument("--safety-batch-factor", type=int, default=2)
    parser.add_argument("--repair-id", default="K6-FusedQuadSignBudgetCap")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_ROOT / "continuation_p3_fused_quad_sign_3x3_b32_w5"))
    parser.add_argument("--fresh", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route = run_main(args, out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
