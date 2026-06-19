#!/usr/bin/env python3
"""Part I v22.16 strict task proof with separated accuracy/NLL/AUC gates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.run_v22_16_common import (  # noqa: E402
    DATASETS,
    PYTHON,
    append_exec,
    apply_gradient_guidance,
    count_parameters,
    device_from_arg,
    ensure_out,
    evaluate,
    init_docs,
    int_flag,
    loader_for,
    make_model,
    read_json,
    write_json,
    write_rows,
)


VARIANTS = [
    "MLP+AdamW",
    "MLP+SGD",
    "MLP+AdaptiveFU",
    "KAN+AdamW",
    "KAN+AdaptiveFU-readout-diagnostic",
    "KAN+AdaptiveFU-basis-official",
    "KAN+AdaptiveFU-source-manifold-official",
    "KAN+AdaptiveFU-coupled-basis-readout-diagnostic",
    "KAN+AdaptiveFU-basis-manifold-official",
]
OFFICIAL = {"KAN+AdaptiveFU-basis-official", "KAN+AdaptiveFU-source-manifold-official", "KAN+AdaptiveFU-basis-manifold-official"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--download", action="store_true")
    p.add_argument("--force-readback", action="store_true")
    return p


def _selector_mode(variant: str) -> tuple[str, str]:
    if "readout" in variant:
        return "readout", "predictive"
    if "basis-manifold" in variant or "source-manifold" in variant:
        return "basis", "manifold"
    if "basis-official" in variant:
        return "basis", "predictive"
    if "AdaptiveFU" in variant:
        return "all", "predictive"
    return "all", "none"


def _pending_rows(blocker: str, datasets_arg: str, seeds_arg: str) -> list[dict[str, Any]]:
    return [
        {"dataset": dataset, "seed": int(seed), "variant": variant, "status": "not_run", "blocker": blocker}
        for dataset in [d.strip() for d in datasets_arg.split(",") if d.strip()]
        for seed in [s.strip() for s in seeds_arg.split(",") if s.strip()]
        for variant in VARIANTS
    ]


def _mechanism_gates(out_dir: Path) -> dict[str, int]:
    s0 = read_json(out_dir / "v22_16_code_truth_route.json")
    traj = read_json(out_dir / "v22_16_real_trajectory_route.json")
    mlp = read_json(out_dir / "v22_16_mlp_adaptive_route.json")
    geom = read_json(out_dir / "v22_16_loss_geometry_route.json")
    kan = read_json(out_dir / "v22_16_kan_basis_route.json")
    eff = read_json(out_dir / "v22_16_efficiency_route.json")
    sm = read_json(out_dir / "v22_16_source_manifold_route.json")
    return {
        "S0_truth_pass": int_flag(s0.get("S0_pass")),
        "real_trajectory_pass": int_flag(traj.get("real_trajectory_logging_pass")),
        "real_MLP_adaptive_pass": int_flag(mlp.get("C1_mlp_adaptive_pass")),
        "real_risk_prediction_pass": int_flag(mlp.get("C2_risk_prediction_pass")),
        "loss_geometry_pointwise_or_pairwise_pass": int(int_flag(geom.get("pointwise_pass")) or int_flag(geom.get("ranking_pairwise_exploration_pass"))),
        "real_KAN_basis_exploration_pass": int_flag(kan.get("KAN_basis_exploration_pass")),
        "real_efficiency_exploration_pass": int_flag(eff.get("real_adaptive_efficiency_exploration_pass")),
        "controls_fail": int(int_flag(mlp.get("controls_fail")) and int_flag(geom.get("controls_fail")) and int_flag(kan.get("controls_fail")) and int_flag(sm.get("controls_fail", 1))),
    }


def _train_one(dataset: str, seed: int, variant: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    train_loader = loader_for(dataset, True, args.train_size, args.batch_size, seed, download=args.download)
    test_loader = loader_for(dataset, False, args.test_size, args.batch_size, seed, download=args.download, shuffle=False)
    first_x, _ = next(iter(train_loader))
    model = make_model(variant, first_x, args.hidden, seed + 82216, device).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt: torch.optim.Optimizer
    if "SGD" in variant:
        opt = torch.optim.SGD(params, lr=5.0e-2, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=2.0e-3 if variant.startswith("MLP") else 2.5e-3, weight_decay=1.0e-4)
    selector, mode = _selector_mode(variant)
    source_state: torch.Tensor | None = None
    losses: list[float] = []
    source_func_vals: list[float] = []
    source_loss_vals: list[float] = []
    train_iter = iter(train_loader)
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(xb).float()
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        diag: dict[str, Any] = {}
        if "AdaptiveFU" in variant:
            source_state, diag = apply_gradient_guidance(model, source_state, selector=selector, mode=mode, step=step, seed=seed)
            source_func_vals.append(float(diag.get("source_retention", 0.0)))
            source_loss_vals.append(float(diag.get("source_loss_proxy", 0.0)))
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        losses.append(float(loss.detach().item()))
    elapsed = time.perf_counter() - started
    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    memory = 0.0
    if device.type == "cuda":
        try:
            memory = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        except RuntimeError:
            memory = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "status": "completed",
        "param_count": count_parameters(model),
        "train_steps": args.steps,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "final_train_loss": train_metrics["loss"],
        "final_test_loss_readback": test_metrics["loss"],
        "final_train_accuracy": train_metrics["accuracy"],
        "final_test_accuracy_readback": test_metrics["accuracy"],
        "NLL_delta_vs_MLP": "",
        "accuracy_delta_vs_MLP": "",
        "AUC_loss_step": sum(losses),
        "AUC_loss_time": sum(losses),
        "AUC_loss_time_ratio_vs_best_control": "",
        "time_to_train_loss_threshold": "",
        "time_to_accuracy_threshold": "",
        "ECE_delta_vs_MLP": "",
        "Brier_delta_vs_MLP": "",
        "tail_loss_q95": test_metrics["tail_loss_q95"],
        "tail_loss_q99": test_metrics["tail_loss_q99"],
        "calibration_debt": test_metrics["ECE"],
        "forgetting_after_shift": "",
        "source_func_task_h4800": sum(source_func_vals) / max(1, len(source_func_vals)) if source_func_vals else "",
        "source_loss_task_h4800": sum(source_loss_vals) / max(1, len(source_loss_vals)) if source_loss_vals else "",
        "source_decay_rate_task": "",
        "basis_channel_energy_fraction_task": "",
        "readout_channel_energy_fraction_task": "",
        "full_loop_step_ratio_vs_mlp": "",
        "memory_ratio_vs_mlp": "",
        "full_loop_step_ms": elapsed * 1000.0 / max(1, int(args.steps)),
        "memory_peak_mb": memory,
        "diagnostic_only": int("diagnostic" in variant or "readout" in variant or "coupled" in variant),
        "official_basis_claim_allowed": int(variant in OFFICIAL),
        "uses_validation_test_future_query_for_direction": 0,
    }


def _add_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            groups.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)
    functional_rows = accuracy_rows = nll_rows = auc_rows = no_debt_rows = 0
    for group in groups.values():
        mlp = next((r for r in group if r["variant"] == "MLP+AdamW"), None)
        kan_base = next((r for r in group if r["variant"] == "KAN+AdamW"), None)
        controls = [r for r in group if r["variant"] in {"MLP+AdamW", "MLP+SGD", "KAN+AdamW"}]
        best_control_auc = min(float(r["AUC_loss_time"]) for r in controls) if controls else 1.0
        official = [r for r in group if r["variant"] in OFFICIAL and not int(r.get("diagnostic_only", 0))]
        best = min(official, key=lambda r: float(r["final_train_loss"]), default=None)
        for row in group:
            if mlp:
                row["NLL_delta_vs_MLP"] = float(row["final_test_loss_readback"]) - float(mlp["final_test_loss_readback"])
                row["accuracy_delta_vs_MLP"] = float(row["final_test_accuracy_readback"]) - float(mlp["final_test_accuracy_readback"])
                row["ECE_delta_vs_MLP"] = float(row["calibration_debt"]) - float(mlp["calibration_debt"])
                row["Brier_delta_vs_MLP"] = ""
                row["param_ratio_vs_MLP"] = float(row["param_count"]) / max(1.0, float(mlp["param_count"]))
                row["full_loop_step_ratio_vs_mlp"] = float(row["full_loop_step_ms"]) / max(1.0e-9, float(mlp["full_loop_step_ms"]))
                row["memory_ratio_vs_mlp"] = float(row["memory_peak_mb"]) / max(1.0e-9, float(mlp["memory_peak_mb"])) if float(mlp["memory_peak_mb"]) > 0 else ""
            row["AUC_loss_time_ratio_vs_best_control"] = float(row["AUC_loss_time"]) / max(1.0e-9, best_control_auc)
        if best and kan_base and (float(best["AUC_loss_time"]) <= float(kan_base["AUC_loss_time"]) or float(best["final_train_loss"]) <= float(kan_base["final_train_loss"])):
            functional_rows += 1
        if best and mlp:
            if float(best["final_test_accuracy_readback"]) >= float(mlp["final_test_accuracy_readback"]):
                accuracy_rows += 1
            if float(best["final_test_loss_readback"]) <= float(mlp["final_test_loss_readback"]):
                nll_rows += 1
            if float(best["AUC_loss_time_ratio_vs_best_control"]) <= 1.0:
                auc_rows += 1
            if float(best["calibration_debt"]) <= float(mlp["calibration_debt"]) + 0.10 and float(best["tail_loss_q99"]) <= float(mlp["tail_loss_q99"]) + 1.0:
                no_debt_rows += 1
    return {
        "task_groups": len(groups),
        "functional_value_vs_KANAdamW_rows": functional_rows,
        "accuracy_superiority_rows": accuracy_rows,
        "nll_superiority_rows": nll_rows,
        "auc_superiority_rows": auc_rows,
        "no_debt_rows": no_debt_rows,
        "functional_value_vs_KANAdamW_pass": int(functional_rows >= 8),
        "accuracy_superiority_pass": int(accuracy_rows >= 7),
        "nll_superiority_pass": int(nll_rows >= 7),
        "auc_superiority_pass": int(auc_rows >= 7),
        "full_superiority_pass": int(functional_rows >= 8 and accuracy_rows >= 7 and nll_rows >= 7 and auc_rows >= 7 and no_debt_rows >= 7),
    }


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = (
        f"{PYTHON} experiments/run_v22_16_strict_task_proof.py --device {args.device} --datasets {args.datasets} "
        f"--seeds {args.seeds} --train-size {args.train_size} --test-size {args.test_size} --steps {args.steps} "
        f"--batch-size {args.batch_size} --hidden {args.hidden} --out-dir {out_dir}"
        + (" --download" if args.download else "")
        + (" --force-readback" if args.force_readback else "")
    )
    gates = _mechanism_gates(out_dir)
    mechanism_ready = int(all(gates.values()))
    rows: list[dict[str, Any]]
    blockers: list[str] = []
    if not mechanism_ready and not args.force_readback:
        blocker = "mechanism_gate_not_passed:" + ";".join(k for k, v in gates.items() if not v)
        rows = _pending_rows(blocker, args.datasets, args.seeds)
        summary = {"task_groups": 0, "functional_value_vs_KANAdamW_pass": 0, "accuracy_superiority_pass": 0, "nll_superiority_pass": 0, "auc_superiority_pass": 0, "full_superiority_pass": 0}
        route_name = "TaskProofDeferredByMechanismGate"
    else:
        rows = []
        device = device_from_arg(args.device)
        for dataset in [d.strip() for d in str(args.datasets).split(",") if d.strip()]:
            for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
                for variant in VARIANTS:
                    try:
                        rows.append(_train_one(dataset, seed, variant, args, device))
                    except Exception as exc:
                        blockers.append(f"{dataset}:{seed}:{variant}:{repr(exc)}")
                        rows.append({"dataset": dataset, "seed": seed, "variant": variant, "status": "blocked", "blocker": repr(exc)})
        summary = _add_comparisons(rows)
        if blockers:
            route_name = "TaskProofPartialOrBlocked"
        elif int_flag(summary.get("full_superiority_pass")):
            route_name = "TaskProofFullSuperiorityPass"
        elif int_flag(summary.get("nll_superiority_pass")):
            route_name = "TaskProofNLLSuperiorityOnly"
        elif int_flag(summary.get("functional_value_vs_KANAdamW_pass")):
            route_name = "TaskProofFunctionalValueOnly_MLPNotBeaten"
        else:
            route_name = "TaskProofCompleted_GatesFailed"
    write_rows(out_dir / "v22_16_strict_task_eval_matrix.csv", rows)
    write_rows(out_dir / "v22_16_convergence_speed_matrix.csv", rows)
    write_rows(out_dir / "v22_16_forgetting_readback_matrix.csv", rows)
    write_rows(out_dir / "v22_16_expression_metrics_matrix.csv", rows)
    write_rows(out_dir / "v22_16_calibration_debt_matrix.csv", rows)
    write_rows(out_dir / "v22_16_task_accuracy_gate.csv", [{"accuracy_superiority_rows": summary.get("accuracy_superiority_rows", 0), "accuracy_superiority_pass": summary.get("accuracy_superiority_pass", 0)}])
    write_rows(out_dir / "v22_16_task_nll_gate.csv", [{"nll_superiority_rows": summary.get("nll_superiority_rows", 0), "nll_superiority_pass": summary.get("nll_superiority_pass", 0)}])
    write_rows(out_dir / "v22_16_task_auc_gate.csv", [{"auc_superiority_rows": summary.get("auc_superiority_rows", 0), "auc_superiority_pass": summary.get("auc_superiority_pass", 0)}])
    route = {"route": route_name, "mechanism_ready_for_task": mechanism_ready, **gates, **summary, "blocker": ";".join(blockers[:20])}
    write_json(out_dir / "v22_16_task_proof_route.json", route)
    append_exec(out_dir, command, status="completed" if rows and not blockers else "blocked", gpu=args.device if mechanism_ready or args.force_readback else "n/a", task_id="I-strict-task-proof", files="v22_16_strict_task_eval_matrix.csv; v22_16_task_accuracy_gate.csv; v22_16_task_nll_gate.csv; v22_16_task_auc_gate.csv", note=f"route={route_name} mechanism_ready={mechanism_ready} rows={len(rows)}")


if __name__ == "__main__":
    main()
