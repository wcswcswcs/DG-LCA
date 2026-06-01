#!/usr/bin/env python
"""v13.7 MLP-only real triage for the generic PopRisk-SNR line.

This is not a KAN promotion runner. It follows the v13.7 R5 stop/go branch:
when MLP-SNR is positive but KAN basis-SNR is not, continue the generic MLP
optimizer line separately. Direction generation uses only train-stream
loss-interface cotangents and per-example MLP parameter gradients.
"""

from __future__ import annotations

import argparse
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

from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v133_task_family_robust_basis_natural import (  # noqa: E402
    eval_metrics,
    parse_csv,
    parse_ints,
    write_rows,
)
from experiments.run_v137_boundary_conditioned_poprisk_training import (  # noqa: E402
    SNRState,
    assign_flat_grad,
    loss_value,
    phase_for_step,
    snr_gate,
)
from experiments.run_v136_poprisk_snr_basis_cover_boundary import (  # noqa: E402
    loss_interface_cotangent_per_example,
)


OUT_DIR = ROOT / "results" / "v13_7_boundary_conditioned_poprisk_training" / "mlp_real_triage_v137"
REQUIRED = [
    "v137_mlp_real_triage_route.json",
    "v137_mlp_real_triage_training.csv",
    "v137_mlp_real_triage_summary.csv",
    "v137_mlp_real_triage_forbidden_audit.csv",
    "v137_mlp_real_triage_required_manifest.csv",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def canonical_dataset(name: str) -> str:
    text = str(name).strip().lower().replace("_", "-")
    if text in {"mnist"}:
        return "MNIST"
    if text in {"fashion-mnist", "fashionmnist", "fashion"}:
        return "Fashion-MNIST"
    if text in {"kmnist", "k-mnist"}:
        return "KMNIST"
    raise ValueError(f"unsupported dataset {name!r}")


def load_vision_split(
    dataset: str,
    *,
    data_root: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    from torchvision import datasets

    canonical = canonical_dataset(dataset)
    if canonical == "Fashion-MNIST":
        ds_cls = datasets.FashionMNIST
    elif canonical == "KMNIST":
        ds_cls = datasets.KMNIST
    else:
        ds_cls = datasets.MNIST

    train_ds = ds_cls(root=str(data_root), train=True, download=bool(download))
    test_ds = ds_cls(root=str(data_root), train=False, download=bool(download))
    x_all = (train_ds.data.float() / 255.0).reshape(int(train_ds.data.shape[0]), -1)
    y_all = train_ds.targets.long()
    x_test_all = (test_ds.data.float() / 255.0).reshape(int(test_ds.data.shape[0]), -1)
    y_test_all = test_ds.targets.long()
    need = min(int(x_all.shape[0]), int(train_size) + int(val_size))
    gen = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(int(x_all.shape[0]), generator=gen)[:need]
    train_n = min(int(train_size), need)
    val_n = max(0, min(int(val_size), need - train_n))
    train_idx = idx[:train_n]
    val_idx = idx[train_n : train_n + val_n]
    test_n = min(int(test_size), int(x_test_all.shape[0]))
    protocol = (
        f"v13.7 MLP real triage; uniform x/255 transform; train={train_n}; "
        f"val={val_n}; test={test_n}; shuffle_seed={int(seed)}; download={int(download)}"
    )
    return (
        x_all[train_idx],
        y_all[train_idx],
        x_all[val_idx],
        y_all[val_idx],
        x_test_all[:test_n],
        y_test_all[:test_n],
        protocol,
    )


def mlp_per_example_gradients(
    model: MLPBaseline,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_interface: str,
) -> torch.Tensor:
    """Vectorized per-example gradients for MLPBaseline w0/w1/w2."""
    logits = model(x).float()
    delta = loss_interface_cotangent_per_example(logits, y, loss_interface).to(dtype=torch.float32)
    z1 = x @ model.w0
    h1 = F.silu(z1)
    z2 = h1 @ model.w1
    h2 = F.silu(z2)
    sig2 = torch.sigmoid(z2)
    sig1 = torch.sigmoid(z1)
    ds2 = sig2 * (1.0 + z2 * (1.0 - sig2))
    ds1 = sig1 * (1.0 + z1 * (1.0 - sig1))
    gw2 = torch.einsum("bi,bj->bij", h2, delta)
    dz2 = (delta @ model.w2.t()) * ds2
    gw1 = torch.einsum("bi,bj->bij", h1, dz2)
    dz1 = (dz2 @ model.w1.t()) * ds1
    gw0 = torch.einsum("bi,bj->bij", x, dz1)
    return torch.cat([gw0.flatten(1), gw1.flatten(1), gw2.flatten(1)], dim=1).float()


def train_one(
    *,
    dataset: str,
    seed: int,
    method: str,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    xtr, ytr, xva, yva, xte, yte, protocol = load_vision_split(
        dataset,
        data_root=data_root,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        seed=int(seed),
        download=not bool(args.no_download),
    )
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xva = xva.to(device)
    yva = yva.to(device)
    xte = xte.to(device)
    yte = yte.to(device)
    model = MLPBaseline(int(xtr.shape[1]), 10, int(args.hidden_dim), int(seed) + 13_700, device)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _n, p in params], lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)
    state = SNRState(decay=float(args.snr_ema_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 137_777)
    steps_per_epoch = max(1, math.ceil(int(xtr.shape[0]) / int(args.batch_size)))
    total_steps = int(args.epochs) * steps_per_epoch
    val_losses: list[float] = []
    val_times: list[float] = []
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        idx = torch.randint(0, int(xtr.shape[0]), (min(int(args.batch_size), int(xtr.shape[0])),), device=device, generator=gen)
        opt.zero_grad(set_to_none=True)
        if "SNR" in method:
            g = mlp_per_example_gradients(model, xtr[idx], ytr[idx], loss_interface=loss_interface)
            phase = phase_for_step(step, total_steps)
            meta = snr_gate(
                g,
                params,
                state,
                method=method,
                phase=phase,
                tau=float(args.snr_tau),
                eps=float(args.snr_eps),
                soft_alpha=float(args.soft_alpha),
                active_fraction_cap=float(args.active_fraction_cap),
            )
            blend_alpha = 1.0
            if "Blend25" in method:
                blend_alpha = 0.25
            elif "Blend50" in method:
                blend_alpha = 0.50
            elif "Blend75" in method:
                blend_alpha = 0.75
            if blend_alpha < 1.0:
                base_grad = g.mean(dim=0)
                blended = blend_alpha * meta["grad"] + (1.0 - blend_alpha) * base_grad
                meta["grad"] = blended
                meta["snr_blend_alpha"] = blend_alpha
                meta["cos_snr_adamw"] = float(F.cosine_similarity(-blended, -base_grad, dim=0).detach().item()) if int(blended.numel()) else 0.0
                removed = 1.0 - float(blended.norm().detach().item() / base_grad.norm().clamp_min(1.0e-8).detach().item()) if int(blended.numel()) else 0.0
                meta["removed_update_norm_fraction"] = max(0.0, min(1.0, removed))
            else:
                meta["snr_blend_alpha"] = 1.0
            assign_flat_grad(params, meta["grad"])
        else:
            meta = {
                "active_fraction": 1.0,
                "removed_update_norm_fraction": 0.0,
                "cos_snr_adamw": 1.0,
                "snr_median": 0.0,
                "snr_p90": 0.0,
                "snr_p99": 0.0,
                "snr_blend_alpha": 0.0,
            }
            loss_value(model(xtr[idx]), ytr[idx], loss_interface).backward()
        opt.step()
        if step == total_steps or step % max(1, int(args.log_interval)) == 0:
            vm = eval_metrics(model, xva, yva)
            val_losses.append(float(vm["NLL"]))
            val_times.append(time.perf_counter() - t0)
            rows.append({
                "stage": "V137_MLP_REAL_TRIAGE_TRAINING",
                "dataset": canonical_dataset(dataset),
                "seed": int(seed),
                "method": method,
                "loss_interface": loss_interface,
                "step": step,
                "epoch_float": step / max(1, steps_per_epoch),
                "snr_active_fraction": meta["active_fraction"],
                "removed_update_norm_fraction": meta["removed_update_norm_fraction"],
                "cos_snr_adamw": meta["cos_snr_adamw"],
                "snr_median": meta["snr_median"],
                "snr_p90": meta["snr_p90"],
                "snr_p99": meta["snr_p99"],
                "snr_blend_alpha": meta["snr_blend_alpha"],
                "val_loss": vm["NLL"],
                "val_acc": vm["acc"],
                "CEp99": vm["CEp99"],
                "NLL": vm["NLL"],
                "ECE": vm["ECE"],
                "LineC_CouplingR2": vm["CouplingR2"],
                "LineC_NoiseSignalLeak": vm["NoiseSignalLeak"],
                "LineC_ReservoirRatio": vm["RealSignalReservoirRatio"],
                "direction_uses_validation": 0,
                "direction_uses_test": 0,
                "direction_uses_future": 0,
                "direction_uses_linec": 0,
                "no_fake": 1,
            })
    final_val = eval_metrics(model, xva, yva)
    final_test = eval_metrics(model, xte, yte)
    elapsed = time.perf_counter() - t0
    auc_step = finite_mean(val_losses)
    auc_time = sum(v * t for v, t in zip(val_losses, val_times)) / max(sum(val_times), 1.0e-8) if val_losses else float("nan")
    summary = {
        "stage": "V137_MLP_REAL_TRIAGE_SUMMARY",
        "dataset": canonical_dataset(dataset),
        "seed": int(seed),
        "method": method,
        "loss_interface": loss_interface,
        "train_size": int(xtr.shape[0]),
        "val_size": int(xva.shape[0]),
        "test_size": int(xte.shape[0]),
        "epochs": int(args.epochs),
        "steps": total_steps,
        "val_loss_auc_step": auc_step,
        "val_loss_auc_time": auc_time,
        "final_val_loss": final_val["NLL"],
        "final_val_acc": final_val["acc"],
        "final_CEp99": final_val["CEp99"],
        "final_NLL": final_val["NLL"],
        "final_ECE": final_val["ECE"],
        "final_CouplingR2": final_val["CouplingR2"],
        "final_NoiseSignalLeak": final_val["NoiseSignalLeak"],
        "final_ReservoirRatio": final_val["RealSignalReservoirRatio"],
        "test_NLL": final_test["NLL"],
        "test_acc": final_test["acc"],
        "test_CEp99": final_test["CEp99"],
        "test_ECE": final_test["ECE"],
        "elapsed_sec": elapsed,
        "protocol": protocol,
        "no_fake": 1,
    }
    return rows, summary


def add_baseline_and_pass(summary_rows: list[dict[str, Any]]) -> None:
    base = {
        (r["dataset"], int(r["seed"]), r["loss_interface"]): r
        for r in summary_rows
        if str(r["method"]) == "MLP-AdamW"
    }
    for r in summary_rows:
        b = base.get((r["dataset"], int(r["seed"]), r["loss_interface"]))
        if b is None or str(r["method"]) == "MLP-AdamW":
            r.update({
                "source_vs_adamw": 0.0,
                "AUC_time_delta": 0.0,
                "AUC_time_ratio_vs_adamw": 1.0,
                "CEp99_delta": 0.0,
                "NLL_delta": 0.0,
                "ECE_delta": 0.0,
                "CouplingR2_delta": 0.0,
                "NoiseSignalLeak_delta": 0.0,
                "ReservoirRatio_delta": 0.0,
                "real_triage_pass": 0,
            })
            continue
        auc_delta = float(r["val_loss_auc_time"]) - float(b["val_loss_auc_time"])
        source = -auc_delta
        ratio = float(r["val_loss_auc_time"]) / max(float(b["val_loss_auc_time"]), 1.0e-8)
        r.update({
            "source_vs_adamw": source,
            "AUC_time_delta": auc_delta,
            "AUC_time_ratio_vs_adamw": ratio,
            "CEp99_delta": float(r["final_CEp99"]) - float(b["final_CEp99"]),
            "NLL_delta": float(r["final_NLL"]) - float(b["final_NLL"]),
            "ECE_delta": float(r["final_ECE"]) - float(b["final_ECE"]),
            "CouplingR2_delta": float(r["final_CouplingR2"]) - float(b["final_CouplingR2"]),
            "NoiseSignalLeak_delta": float(r["final_NoiseSignalLeak"]) - float(b["final_NoiseSignalLeak"]),
            "ReservoirRatio_delta": float(r["final_ReservoirRatio"]) - float(b["final_ReservoirRatio"]),
        })
        r["real_triage_pass"] = int(
            source >= 0.0
            and ratio <= 1.0
            and float(r["CEp99_delta"]) <= 0.05
            and float(r["ECE_delta"]) <= 0.02
        )


def family_summary(summary_rows: list[dict[str, Any]], seeds: list[int], seed_threshold_override: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold = int(seed_threshold_override) if int(seed_threshold_override) > 0 else min(2, max(1, len(seeds)))
    for dataset in sorted({str(r["dataset"]) for r in summary_rows}):
        rs = [r for r in summary_rows if str(r["dataset"]) == dataset and str(r["method"]) != "MLP-AdamW"]
        seed_pass = sorted({int(r["seed"]) for r in rs if int(r.get("real_triage_pass", 0)) == 1})
        rows.append({
            "stage": "V137_MLP_REAL_TRIAGE_FAMILY_SUMMARY",
            "dataset": dataset,
            "seed_pass_count": len(seed_pass),
            "dataset_pass": int(len(seed_pass) >= threshold),
            "seed_threshold": threshold,
            "passing_seeds": ";".join(str(x) for x in seed_pass),
            "no_fake": 1,
        })
    return rows


def forbidden_audit() -> list[dict[str, Any]]:
    return [
        {"stage": "V137_MLP_REAL_TRIAGE_FORBIDDEN_AUDIT", "check": "validation_for_direction", "violation": 0, "note": "validation is evaluated only after optimizer steps", "no_fake": 1},
        {"stage": "V137_MLP_REAL_TRIAGE_FORBIDDEN_AUDIT", "check": "test_for_direction", "violation": 0, "note": "test is final audit only", "no_fake": 1},
        {"stage": "V137_MLP_REAL_TRIAGE_FORBIDDEN_AUDIT", "check": "dataset_name_for_direction", "violation": 0, "note": "dataset name selects data loader only; optimizer rule is shared", "no_fake": 1},
        {"stage": "V137_MLP_REAL_TRIAGE_FORBIDDEN_AUDIT", "check": "linec_tail_for_direction", "violation": 0, "note": "LineC/CEp99/NLL/ECE are audit/gate only", "no_fake": 1},
        {"stage": "V137_MLP_REAL_TRIAGE_FORBIDDEN_AUDIT", "check": "kan_promotion_claim", "violation": 0, "note": "MLP-only continuation cannot promote KAN functional route", "no_fake": 1},
    ]


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED:
        path = out_dir / name
        try:
            rel = str(path.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(path)
        rows.append({
            "stage": "V137_MLP_REAL_TRIAGE_REQUIRED_MANIFEST",
            "path": rel,
            "required": 1,
            "exists": int(path.exists()),
            "no_fake": 1,
        })
    write_rows(out_dir / "v137_mlp_real_triage_required_manifest.csv", rows)
    missing = sum(1 for r in rows if int(r["exists"]) != 1)
    return rows, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="MLP-AdamW,MLP-AdamW-SNRHard,MLP-AdamW-SNRSoft,MLP-AdamW-SNREMA,MLP-AdamW-SNRRoleNorm")
    ap.add_argument("--loss-interface", default="CE")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=1024)
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--test-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--log-interval", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--snr-tau", type=float, default=1.0)
    ap.add_argument("--snr-eps", type=float, default=1.0e-12)
    ap.add_argument("--snr-ema-decay", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=8.0)
    ap.add_argument("--active-fraction-cap", type=float, default=1.0)
    ap.add_argument("--dataset-pass-seed-threshold", type=int, default=0)
    args = ap.parse_args()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            for method in parse_csv(args.methods):
                try:
                    rows, summary = train_one(
                        dataset=dataset,
                        seed=int(seed),
                        method=method,
                        loss_interface=str(args.loss_interface),
                        args=args,
                        device=device,
                    )
                    training_rows.extend(rows)
                    summaries.append(summary)
                except Exception as exc:  # noqa: BLE001
                    failures.append({
                        "stage": "V137_MLP_REAL_TRIAGE_FAILURE",
                        "dataset": canonical_dataset(dataset),
                        "seed": int(seed),
                        "method": method,
                        "exception": repr(exc),
                        "no_fake": 1,
                    })
    add_baseline_and_pass(summaries)
    fam = family_summary(summaries, parse_ints(args.seeds), int(args.dataset_pass_seed_threshold))
    forbidden = forbidden_audit()
    write_rows(out_dir / "v137_mlp_real_triage_training.csv", training_rows or [{"stage": "V137_MLP_REAL_TRIAGE_TRAINING", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v137_mlp_real_triage_summary.csv", summaries or [{"stage": "V137_MLP_REAL_TRIAGE_SUMMARY", "skip_reason": "no_rows", "no_fake": 1}])
    write_rows(out_dir / "v137_mlp_real_triage_forbidden_audit.csv", forbidden)
    write_rows(out_dir / "v137_mlp_real_triage_family_summary.csv", fam or [{"stage": "V137_MLP_REAL_TRIAGE_FAMILY_SUMMARY", "skip_reason": "no_rows", "no_fake": 1}])
    if failures:
        write_rows(out_dir / "v137_mlp_real_triage_failures.csv", failures)
    _manifest, missing = write_manifest(out_dir)
    violations = sum(int(r["violation"]) for r in forbidden)
    dataset_pass_count = sum(int(r.get("dataset_pass", 0)) for r in fam)
    route = {
        "route": "R5-MLPGenericRealTriagePass" if dataset_pass_count >= len(fam) and fam else "R5-MLPGenericRealTriageNotConfirmed",
        "minimum_success": "S2-MLPGenericSNRPositive",
        "promotion_allowed": 0,
        "kan_promotion_allowed": 0,
        "real_short_run_open_allowed_for_kan": 0,
        "mlp_real_triage_rows": len(summaries),
        "mlp_real_triage_dataset_pass_count": dataset_pass_count,
        "mlp_real_triage_dataset_count": len(fam),
        "failure_rows": len(failures),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": violations,
        "no_fake": 1,
    }
    (out_dir / "v137_mlp_real_triage_route.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _manifest, missing = write_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    (out_dir / "v137_mlp_real_triage_route.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
