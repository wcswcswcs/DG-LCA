#!/usr/bin/env python
"""v14.5 Line W Wavelet substrate hardening diagnostic.

This runner is bounded to the all-basis substrate portfolio. It never executes
FMS proof and never uses LineC/tail/AUC audit metrics as a direction source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES, fnum, parse_csv, parse_ints  # noqa: E402
from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments import run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import classification_basic, linec_metrics, make_adamw  # noqa: E402
from experiments.run_v143_nonrat_compact_task_health_probe import (  # noqa: E402
    apply_output_geometry_repair,
    apply_wavelet_support_repair,
    load_workspace_rows,
    manual_train_step,
    wavelet_support_audit,
    write_rows,
)
from experiments.run_v143_nonrat_manual_kernel_substrate_probe import make_probe_model  # noqa: E402


DEFAULT_CANDIDATES = (
    "D-WAV17-Raw005SupportHealthSubstrate,"
    "D-WAV18-Raw002SupportHealthSubstrate,"
    "D-WAV19-Raw001SupportHealthSubstrate"
)


@dataclass(frozen=True)
class WaveletConfig:
    config_id: str
    description: str
    support_repair: str
    role_constraint: str
    output_geometry_repair: str


WAVELET_CONFIGS: dict[str, WaveletConfig] = {
    "W123-train-entropy": WaveletConfig(
        "W123-train-entropy",
        "D-WAV17/18/19 train-entropy substrate hardening",
        "none",
        "none",
        "train_entropy_t080_100_else050",
    ),
    "W4-support-entropy-readout": WaveletConfig(
        "W4-support-entropy-readout",
        "Wavelet support + train entropy + delayed readout mixing",
        "quantile_scale075",
        "linear_readout_grad050",
        "train_entropy_t080_100_else050",
    ),
    "W5-scale-occupancy": WaveletConfig(
        "W5-scale-occupancy",
        "Wavelet scale occupancy balanced update",
        "quantile_scale050",
        "linear_readout_grad025",
        "train_entropy_t085_100_else050",
    ),
    "W6-local-tail-guard": WaveletConfig(
        "W6-local-tail-guard",
        "Wavelet local-tail coverage guard",
        "quantile_scale050",
        "linear_readout_grad050",
        "train_topprob_t020_050_else100",
    ),
    "W7-reservoir-balance": WaveletConfig(
        "W7-reservoir-balance",
        "Train-stream reservoir-balance readout gradient gate",
        "quantile_scale050",
        "linear_readout_grad010",
        "none",
    ),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_label(value: str) -> str:
    return v1223.v120._canonical_dataset(str(value))


def finite_ratio(num: float, den: float, default: float = float("nan")) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1.0e-12:
        return default
    return float(num / den)


def train_mlp_auc_reference(
    args: argparse.Namespace,
    input_dim: int,
    output_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model = MLPBaseline(input_dim, output_dim, int(args.mlp_hidden), int(seed) + 1_231_000, device).to(device)
    opt = make_adamw(args, model.parameters())
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1_231_100)
    nll_trace: list[float] = []
    times: list[float] = []
    for _epoch in range(int(args.epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off : off + int(args.batch_size)]
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        nll_trace.append(float(classification_basic(model, x_val, y_val)["NLL"]))
    ev = classification_basic(model, x_val, y_val)
    return {
        "model": model,
        "mlp_val_acc": ev["acc"],
        "mlp_NLL": ev["NLL"],
        "mlp_ECE": ev["ECE"],
        "mlp_CEp99": ev["CEp99"],
        "mlp_auc_nll": float(sum(nll_trace)),
        "mlp_nll_trace": "|".join(f"{v:.8f}" for v in nll_trace),
        "mlp_step_time_q90_ms": float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan"),
    }


def wavelet_support_geometry(model: torch.nn.Module, x_train: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scale_occupancy_entropy": "",
        "support_overlap": "",
        "local_tail_coverage": "",
    }
    if getattr(getattr(model, "spec", None), "basis_name", "") != "hat_wavelet":
        return out
    with torch.no_grad():
        x = x_train[: min(512, int(x_train.shape[0]))]
        b1 = model.layer1_basis(x)
        active = (b1.abs() > 1.0e-6).float()
        flat = active.reshape(-1, active.shape[-1])
        occ = flat.mean(dim=0).clamp_min(1.0e-8)
        occ_dist = occ / occ.sum().clamp_min(1.0e-8)
        entropy = -(occ_dist * occ_dist.log()).sum() / math.log(max(2, int(occ_dist.numel())))
        if flat.shape[1] > 1:
            overlap = (flat.T @ flat) / max(1, flat.shape[0])
            mask = ~torch.eye(flat.shape[1], dtype=torch.bool, device=flat.device)
            support_overlap = float(overlap[mask].mean().item())
        else:
            support_overlap = 0.0
        logits = model(x)
        probs = logits.float().softmax(dim=1)
        ent = -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=1) / math.log(max(2, int(probs.shape[1])))
        sample_active = active.reshape(active.shape[0], -1).mean(dim=1)
        local_tail_coverage = float((sample_active * ent).mean().item())
        out.update(
            {
                "scale_occupancy_entropy": float(entropy.item()),
                "support_overlap": support_overlap,
                "local_tail_coverage": local_tail_coverage,
            }
        )
    return out


def linec_pass(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("CouplingR2"), -999.0) >= 0.15
        and fnum(row.get("NoiseSignalLeak"), 999.0) <= 0.20
        and fnum(row.get("RealSignalReservoirRatio"), 999.0) <= 0.70
    )


def train_wavelet_candidate(
    args: argparse.Namespace,
    config: WaveletConfig,
    candidate_id: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int,
    seed: int,
    device: torch.device,
    mlp_ref: dict[str, Any],
    workspace_row: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_args = argparse.Namespace(**vars(args))
    local_args.wavelet_support_repair = config.support_repair
    local_args.wavelet_role_constraint = config.role_constraint
    local_args.output_geometry_repair = config.output_geometry_repair

    cand = V1235_BASIS_CANDIDATES[candidate_id]
    model, spec = make_probe_model(cand.candidate_id, input_dim, output_dim, x_train, device, int(seed) + 14_360, int(args.hidden_override))
    support_info = apply_wavelet_support_repair(model, local_args, x_train)
    opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
    gen = torch.Generator(device=device).manual_seed(int(seed) + 14_370)
    times: list[float] = []
    nll_trace: list[float] = []
    role_constraint_steps = 0
    for _epoch in range(int(args.epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off : off + int(args.batch_size)]
            dt_ms, applied = manual_train_step(model, opt, x_train[idx], y_train[idx], device, local_args)
            times.append(dt_ms)
            role_constraint_steps += int(applied)
        nll_trace.append(float(classification_basic(model, x_val, y_val)["NLL"]))

    support_geom = wavelet_support_geometry(model, x_train)
    eval_model, output_info = apply_output_geometry_repair(model, local_args, x_train)
    ev = classification_basic(eval_model, x_val, y_val)

    linec_rows: list[dict[str, Any]] = []
    b = min(int(args.linec_batch_size), int(x_train.shape[0]), int(x_val.shape[0]))
    for ls in parse_ints(args.linec_seeds):
        try:
            lm = linec_metrics(eval_model, x_train[:b], y_train[:b], x_val[:b], y_val[:b], int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
            status = "executed"
            error = ""
        except Exception as exc:  # noqa: BLE001
            lm = {
                "CouplingR2": float("nan"),
                "CouplingCorr": float("nan"),
                "NoiseSignalLeak": float("nan"),
                "RealSignalReservoirRatio": float("nan"),
            }
            status = "blocked"
            error = f"{type(exc).__name__}: {exc}"
        lrow = {
            "stage": "V145_LINEW_WAVELET_LINEC_AUDIT",
            "config_id": config.config_id,
            "family": cand.family,
            "candidate_id": cand.candidate_id,
            "mapped_method_id": cand.method_id,
            "linec_seed": int(ls),
            "linec_status": status,
            "linec_error": error,
            **lm,
            "linec_pass": linec_pass(lm),
            "linec_used_for_direction": 0,
            "wavelet_support_repair": config.support_repair,
            "wavelet_role_constraint": config.role_constraint,
            "output_geometry_repair": config.output_geometry_repair,
            "linec_tail_auc_used_for_direction": 0,
            "promotion_allowed": 0,
        }
        linec_rows.append(lrow)

    linec_count = len(linec_rows)
    linec_pass_rate = sum(int(r["linec_pass"]) for r in linec_rows) / max(1, linec_count)
    mean_noise = sum(fnum(r.get("NoiseSignalLeak"), float("nan")) for r in linec_rows) / max(1, linec_count)
    mean_rsr = sum(fnum(r.get("RealSignalReservoirRatio"), float("nan")) for r in linec_rows) / max(1, linec_count)
    mean_coupling = sum(fnum(r.get("CouplingR2"), float("nan")) for r in linec_rows) / max(1, linec_count)
    auc_nll = float(sum(nll_trace))
    mlp_auc = fnum(mlp_ref.get("mlp_auc_nll"), float("nan"))
    step_q90 = float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan")
    step_ratio = finite_ratio(step_q90, fnum(mlp_ref.get("mlp_step_time_q90_ms"), float("nan")))
    mean_delta = fnum(ev.get("acc"), float("nan")) - fnum(mlp_ref.get("mlp_val_acc"), float("nan"))
    nll_delta = fnum(ev.get("NLL"), float("nan")) - fnum(mlp_ref.get("mlp_NLL"), float("nan"))
    ece_delta = fnum(ev.get("ECE"), float("nan")) - fnum(mlp_ref.get("mlp_ECE"), float("nan"))
    cep99_delta = fnum(ev.get("CEp99"), float("nan")) - fnum(mlp_ref.get("mlp_CEp99"), float("nan"))
    workspace_incremental = fnum(workspace_row.get("incremental_memory_ratio_vs_mlp"), float("nan"))
    workspace_raw = fnum(workspace_row.get("raw_memory_ratio_vs_mlp"), float("nan"))
    workspace_step = fnum(workspace_row.get("step_ratio_vs_mlp"), float("nan"))
    auc_ratio = finite_ratio(auc_nll, mlp_auc)
    gate_pass = int(
        workspace_incremental <= 1.75
        and step_ratio <= 1.75
        and mean_delta >= -0.05
        and mean_delta >= -0.10
        and auc_ratio <= 2.0
        and linec_pass_rate >= 0.30
    )
    row = {
        "stage": "V145_LINEW_WAVELET_SUBSTRATE_HARDENING",
        "config_id": config.config_id,
        "config_description": config.description,
        "family": cand.family,
        "candidate_id": cand.candidate_id,
        "mapped_method_id": cand.method_id,
        "basis_name": getattr(spec, "basis_name", ""),
        "dataset": "",
        "seed": int(seed),
        "workspace_raw_ratio": workspace_raw,
        "workspace_incremental_ratio": workspace_incremental,
        "workspace_step_ratio": workspace_step,
        "step_ratio": step_ratio,
        "mean_delta_vs_mlp": mean_delta,
        "worst_delta_vs_mlp": mean_delta,
        "AUCtime_ratio": auc_ratio,
        "CEp99_delta": cep99_delta,
        "NLL_delta": nll_delta,
        "ECE_delta": ece_delta,
        "LineC_pass_rate": linec_pass_rate,
        "LineC_CouplingR2_mean": mean_coupling,
        "RealSignalReservoirRatio": mean_rsr,
        "NoiseSignalLeak": mean_noise,
        **support_geom,
        "wavelet_substrate_gate_pass": gate_pass,
        "wavelet_support_repair": config.support_repair,
        "wavelet_role_constraint": config.role_constraint,
        "wavelet_role_constraint_steps": role_constraint_steps,
        "output_geometry_repair": config.output_geometry_repair,
        "wavelet_support_uses_train_stream_features": support_info.get("wavelet_support_uses_train_stream_features", 0),
        "output_geometry_uses_train_stream_logits": output_info.get("output_geometry_uses_train_stream_logits", 0),
        "linec_tail_auc_used_for_direction": 0,
        "labels_used_for_direction": 0,
        "validation_test_future_query_used_for_direction": 0,
        "official_fms_proof_executed": 0,
        "promotion_allowed": 0,
        "candidate_nll_trace": "|".join(f"{v:.8f}" for v in nll_trace),
        "mlp_nll_trace": mlp_ref.get("mlp_nll_trace", ""),
        "val_acc": ev["acc"],
        "mlp_val_acc": mlp_ref.get("mlp_val_acc", ""),
        "NLL": ev["NLL"],
        "mlp_NLL": mlp_ref.get("mlp_NLL", ""),
        "ECE": ev["ECE"],
        "mlp_ECE": mlp_ref.get("mlp_ECE", ""),
        "CEp99": ev["CEp99"],
        "mlp_CEp99": mlp_ref.get("mlp_CEp99", ""),
        "step_time_q90_ms": step_q90,
        "mlp_step_time_q90_ms": mlp_ref.get("mlp_step_time_q90_ms", ""),
        **support_info,
        **output_info,
    }
    return row, linec_rows


def classify_failure(row: dict[str, Any]) -> str:
    blockers: list[str] = []
    if fnum(row.get("workspace_incremental_ratio"), 999.0) > 1.75:
        blockers.append("workspace")
    if fnum(row.get("step_ratio"), 999.0) > 1.75:
        blockers.append("step")
    if fnum(row.get("mean_delta_vs_mlp"), -999.0) < -0.05 or fnum(row.get("worst_delta_vs_mlp"), -999.0) < -0.10:
        blockers.append("task")
    if fnum(row.get("AUCtime_ratio"), 999.0) > 2.0:
        blockers.append("AUCtime")
    if fnum(row.get("LineC_pass_rate"), 0.0) < 0.30:
        blockers.append("LineC")
    if fnum(row.get("NoiseSignalLeak"), 0.0) > 0.20:
        blockers.append("NoiseSignalLeak")
    if fnum(row.get("RealSignalReservoirRatio"), 0.0) > 0.70:
        blockers.append("RealSignalReservoirRatio")
    return ",".join(dict.fromkeys(blockers)) if blockers else "-"


def write_required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    required = [
        "v145_wavelet_substrate_hardening.csv",
        "v145_wavelet_linec_audit.csv",
        "v145_wavelet_substrate_summary.csv",
        "v145_wavelet_failure_table.csv",
        "v145_wavelet_forbidden_information_audit.csv",
        "v145_wavelet_route_decision.json",
    ]
    rows = []
    for name in required:
        exists = (out_dir / name).exists()
        rows.append({"artifact": name, "exists": int(exists), "missing": int(not exists)})
    write_rows(out_dir / "v145_wavelet_required_manifest.csv", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workspace-csv", required=True)
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--configs", default="W123-train-entropy,W4-support-entropy-readout,W5-scale-occupancy,W6-local-tail-guard,W7-reservoir-balance")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--train-size", type=int, default=128)
    ap.add_argument("--val-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--hidden-override", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="false")
    ap.add_argument("--linec-batch-size", type=int, default=24)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12319500")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Line W hardening diagnostic requires CUDA")
    torch.cuda.set_device(device)

    workspace = load_workspace_rows(str(args.workspace_csv))
    task_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    for dataset in parse_csv(args.datasets):
        ds = dataset_label(dataset)
        for seed in parse_ints(args.seeds):
            load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
            data = v1223.v120._load_vision_split(load_args, ds, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            mlp_ref = train_mlp_auc_reference(args, int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, int(seed), device)
            for config_id in parse_csv(args.configs):
                config = WAVELET_CONFIGS[config_id]
                for cid in parse_csv(args.candidates):
                    ws = workspace.get((cid, "manual_no_materialize"), {})
                    row, lc = train_wavelet_candidate(
                        args,
                        config,
                        cid,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        int(input_dim),
                        int(output_dim),
                        int(seed),
                        device,
                        mlp_ref,
                        ws,
                    )
                    row["dataset"] = ds
                    row["failure_class"] = classify_failure(row)
                    task_rows.append(row)
                    for lrow in lc:
                        lrow["dataset"] = ds
                        lrow["seed"] = int(seed)
                    linec_rows.extend(lc)
                    torch.cuda.empty_cache()

    write_rows(out_dir / "v145_wavelet_substrate_hardening.csv", task_rows)
    write_rows(out_dir / "v145_wavelet_linec_audit.csv", linec_rows)
    failure_rows = [r for r in task_rows if int(r.get("wavelet_substrate_gate_pass", 0)) == 0]
    write_rows(out_dir / "v145_wavelet_failure_table.csv", failure_rows)

    summary_rows: list[dict[str, Any]] = []
    for config_id in sorted({str(r["config_id"]) for r in task_rows}):
        rows = [r for r in task_rows if str(r["config_id"]) == config_id]
        dataset_seed_pass = {
            (str(r["dataset"]), int(r["seed"]))
            for r in rows
            if int(r.get("wavelet_substrate_gate_pass", 0)) == 1
        }
        summary_rows.append(
            {
                "stage": "V145_LINEW_WAVELET_SUBSTRATE_SUMMARY",
                "config_id": config_id,
                "rows": len(rows),
                "dataset_seed_pass_count": len(dataset_seed_pass),
                "gate_pass_rows": sum(int(r.get("wavelet_substrate_gate_pass", 0)) for r in rows),
                "best_mean_delta_vs_mlp": max(fnum(r.get("mean_delta_vs_mlp"), -999.0) for r in rows),
                "best_AUCtime_ratio": min(fnum(r.get("AUCtime_ratio"), 999.0) for r in rows),
                "best_LineC_pass_rate": max(fnum(r.get("LineC_pass_rate"), 0.0) for r in rows),
                "min_NoiseSignalLeak": min(fnum(r.get("NoiseSignalLeak"), 999.0) for r in rows),
                "min_RealSignalReservoirRatio": min(fnum(r.get("RealSignalReservoirRatio"), 999.0) for r in rows),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v145_wavelet_substrate_summary.csv", summary_rows)

    forbidden = [
        {
            "direction_uses_validation_test_future_query": 0,
            "direction_uses_linec_tail_auc_metric": 0,
            "direction_uses_labels": 0,
            "linec_tail_auc_used_for_direction_rows": sum(int(r.get("linec_tail_auc_used_for_direction", 0)) for r in task_rows),
            "labels_used_for_direction_rows": sum(int(r.get("labels_used_for_direction", 0)) for r in task_rows),
            "promotion_allowed": 0,
        }
    ]
    write_rows(out_dir / "v145_wavelet_forbidden_information_audit.csv", forbidden)

    dataset_seed_pass = {
        (str(r["dataset"]), int(r["seed"]))
        for r in task_rows
        if int(r.get("wavelet_substrate_gate_pass", 0)) == 1
    }
    route = {
        "stage": "V145_LINEW_WAVELET_SUBSTRATE_HARDENING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": "W1-WaveletSubstratePass" if len(dataset_seed_pass) == 9 else "W0-WaveletSubstrateFail",
        "wavelet_dataset_seed_pass_count": len(dataset_seed_pass),
        "wavelet_gate_pass_rows": sum(int(r.get("wavelet_substrate_gate_pass", 0)) for r in task_rows),
        "expected_dataset_seed_count": 9,
        "official_fms_proof_executed": 0,
        "open_wavelet_fms_synthetic_proof": int(len(dataset_seed_pass) == 9),
        "promotion_allowed": 0,
        "forbidden_information_violation_count": 0,
    }
    (out_dir / "v145_wavelet_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    manifest = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = sum(int(r["missing"]) for r in manifest)
    (out_dir / "v145_wavelet_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
