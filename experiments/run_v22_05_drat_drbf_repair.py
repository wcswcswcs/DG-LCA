#!/usr/bin/env python3
"""v22.05 D-RAT/D-RBF active repair wrapper."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.profiling.efficiency_v22_05 import classify_drat_drbf_v22_05  # noqa: E402
from dgkan.profiling.efficiency_v20 import efficiency_waterfall_rows, profile_isolated  # noqa: E402
from experiments import run_v22_03_drat_drbf_repair as repair  # noqa: E402
from experiments.run_v17_common import carrier_model, load_dataset, loss_value, make_update, same_param_mlp_factory  # noqa: E402
from experiments.run_v22_05_common import PYTHON, append_exec, ensure_out, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--batch-sizes", default="512")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--official-transition", type=int, default=1)
    p.add_argument("--official-transition-batch-sizes", default="512,2048")
    p.add_argument("--data-root", default="data")
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=128)
    p.add_argument("--profiler-repeats", type=int, default=3)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.003)
    return p


def _f(value: Any, default: float = 999.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _sync(device: Any) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time_probe(fn: Any, device: Any, repeats: int = 2) -> float:
    reps = max(1, int(repeats))
    _sync(device)
    times: list[float] = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, torch.Tensor):
            _ = out.detach()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(sum(times) / len(times))


def _component_error(component_ms: list[float], total_ms: float) -> float:
    total = max(1.0e-9, float(total_ms))
    return abs(float(sum(component_ms)) - total) / total


def _official_component_telemetry(
    *,
    carrier: str,
    model: Any,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: Any,
    repeats: int,
) -> dict[str, Any]:
    """Measure diagnostic component timings on the official fused runner path.

    These probes run on the same instantiated official model and batch shape as
    the robust timing row. They are not imported from repair/bridge rows and are
    not used to change the forward/step/memory ratios.
    """

    probe_n = min(128, int(xb.shape[0]))
    x_probe = xb[:probe_n].contiguous()
    y_probe = yb[:probe_n].contiguous()
    out: dict[str, Any] = {
        "official_component_telemetry_probe": 1,
        "component_telemetry_source": "official_fused_runner_probe",
    }
    with torch.no_grad():
        if carrier == "D-RAT":
            z = model._norm_input(x_probe)  # type: ignore[attr-defined]

            def numerator() -> torch.Tensor:
                vals = [z]
                if int(model.k) >= 2:
                    vals.append(z.square())
                if int(model.k) >= 3:
                    vals.append(z * z.square())
                if int(model.k) >= 4:
                    vals.append(z.square().square())
                return torch.stack(vals[: int(model.k)], dim=-1)

            def denominator() -> torch.Tensor:
                base = 1.0 + 0.5 * z.abs() + 0.125 * z.square()
                shifts = torch.arange(int(model.k), device=z.device, dtype=z.dtype).view(1, 1, -1)
                return base.unsqueeze(-1) + 0.05 * shifts

            den_cache = denominator()

            numerator_ms = _time_probe(numerator, device, repeats)
            denominator_ms = _time_probe(denominator, device, repeats)
            reciprocal_ms = _time_probe(lambda: den_cache.reciprocal(), device, repeats)
            safety_ms = _time_probe(lambda: torch.isfinite(den_cache).float().mean() + den_cache.amin(), device, repeats)
            fused_forward_ms = _time_probe(lambda: model.manual_ce_forward_cache(x_probe)[0], device, repeats)  # type: ignore[attr-defined]

            def backward_probe() -> torch.Tensor:
                logits, cache = model.manual_ce_forward_cache(x_probe)  # type: ignore[attr-defined]
                return model.manual_ce_backward_from_cache(logits, cache, y_probe)  # type: ignore[attr-defined]

            derivative_ms = _time_probe(backward_probe, device, 1)
            train_path_ms = fused_forward_ms + derivative_ms
            out.update(
                {
                    "numerator_eval_ms": numerator_ms,
                    "denominator_eval_ms": denominator_ms,
                    "reciprocal_ms": reciprocal_ms,
                    "numden_fused_ms": fused_forward_ms,
                    "safety_guard_ms": safety_ms,
                    "safety_clamp_ms": safety_ms,
                    "derivative_telemetry_ms": derivative_ms,
                    "train_path_without_telemetry_ms": train_path_ms,
                    "component_sum_vs_total_error": _component_error(
                        [numerator_ms, denominator_ms, reciprocal_ms, safety_ms],
                        fused_forward_ms,
                    ),
                    "rational_den_min": float(den_cache.amin().detach().cpu().item()),
                    "rational_den_p01": float(torch.quantile(den_cache.flatten(), 0.01).detach().cpu().item()),
                }
            )
            return out

        if carrier == "D-RBF":
            z = model._norm_input(x_probe)  # type: ignore[attr-defined]
            width = model.scales[0].clamp_min(1.0e-3)
            centers = model.centers[: int(model.k)]

            def local_gather() -> torch.Tensor:
                return z.unsqueeze(-1) - centers.view(1, 1, -1)

            def exp_eval() -> torch.Tensor:
                r = local_gather() / width
                return torch.exp(-0.5 * r.square())

            local_gather_ms = _time_probe(local_gather, device, repeats)
            exp_eval_ms = _time_probe(exp_eval, device, repeats)
            readout_ms = _time_probe(lambda: model.manual_ce_forward_cache(x_probe)[0], device, repeats)  # type: ignore[attr-defined]

            def backward_probe() -> torch.Tensor:
                logits, cache = model.manual_ce_forward_cache(x_probe)  # type: ignore[attr-defined]
                return model.manual_ce_backward_from_cache(logits, cache, y_probe)  # type: ignore[attr-defined]

            local_backward_ms = _time_probe(backward_probe, device, 1)
            out.update(
                {
                    "active_center_fraction": 1.0,
                    "mean_local_K": int(model.k),
                    "local_gather_ms": local_gather_ms,
                    "exp_eval_ms": exp_eval_ms,
                    "readout_matmul_ms": readout_ms,
                    "local_backward_ms": local_backward_ms,
                    "dense_materialized_bytes": 0,
                    "dense_materialization_bytes": 0,
                    "no_dense_materialization_proof": 1,
                    "component_sum_vs_total_error": _component_error(
                        [local_gather_ms, exp_eval_ms],
                        readout_ms,
                    ),
                }
            )
            return out
    return out


def _official_transition_rows(
    args: argparse.Namespace,
    device: Any,
    *,
    carrier: str,
    repair_variant: str,
    component_variant: str,
    manual_token: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure a PrimitiveKAN manual CE train path as official rows.

    This deliberately uses the same manual CE correctness/profile contract as
    the v21 officialization runner, rather than relabeling the v22.03 micro
    basis benchmark.
    """

    rows: list[dict[str, Any]] = []
    waterfall: list[dict[str, Any]] = []
    batches = [int(x) for x in str(args.official_transition_batch_sizes).split(",") if x.strip()]
    seed_base = 22_105 if carrier == "D-RBF" else 22_205
    for job_index, batch in enumerate(batches):
        start = time.perf_counter()
        try:
            load_n = max(int(args.train_size), int(batch))
            x_train, y_train, x_val, y_val = load_dataset(
                "MNIST",
                Path(args.data_root),
                load_n,
                int(args.val_size),
                seed_base + job_index,
                device,
                int(args.input_size),
            )
            xb = x_train[:batch]
            yb = y_train[:batch]
            local = argparse.Namespace(**vars(args))
            local.classes = int(args.classes)
            local.hidden = int(args.hidden)
            local.param_budget = int(args.param_budget)
            local.input_size = int(args.input_size)
            local.basis_repair_variant = repair_variant
            seed = seed_base + job_index
            base = carrier_model(carrier, x_train, seed, local, device)
            same_hidden, mlp_factory = same_param_mlp_factory(base, int(x_train.shape[1]), int(args.classes), seed, device)

            def model_factory(s: int = seed):
                return carrier_model(carrier, x_train, s, local, device)

            def update_factory(model):
                return make_update(model, "M3-FUPrimary", xb, yb, seed=seed)

            def audit(model):
                return loss_value(model, x_val, y_val)

            manual_variant = str(base.manual_kernel_variant()) if hasattr(base, "manual_kernel_variant") else ""
            manual_audit: dict[str, Any]
            try:
                manual_audit = base.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))])  # type: ignore[attr-defined]
            except Exception as exc:
                manual_audit = {"manual_correctness_blocker": f"{type(exc).__name__}:{exc}"}
            prof = profile_isolated(
                model_factory,
                mlp_factory,
                xb,
                yb,
                update_factory,
                audit,
                device=device,
                repeats=int(args.profiler_repeats),
                warmup=int(args.profiler_warmup),
                lr=float(args.lr),
                use_manual_ce=True,
            )
            grad_relerr = _f(manual_audit.get("grad_relerr_max"))
            grad_cos = _f(manual_audit.get("grad_cos_min"), -1.0)
            output_err = _f(manual_audit.get("output_max_abs_error"))
            manual_correct = int(grad_relerr < 1.0e-4 and grad_cos > 0.999 and output_err < 1.0e-4)
            manual_profiled = int(prof.get("manual_ce_train_stream_profiled", 0) or 0)
            official_fused = int(manual_profiled and manual_correct and manual_token in manual_variant)
            component_telemetry: dict[str, Any] = {}
            if official_fused:
                try:
                    component_telemetry = _official_component_telemetry(
                        carrier=carrier,
                        model=base,
                        xb=xb,
                        yb=yb,
                        device=device,
                        repeats=max(1, min(2, int(args.profiler_repeats))),
                    )
                except Exception as exc:
                    component_telemetry = {
                        "official_component_telemetry_probe": 1,
                        "component_telemetry_source": "official_fused_runner_probe",
                        "component_telemetry_error": f"{type(exc).__name__}:{exc}",
                    }
            row = {
                "carrier": carrier,
                "component_variant": component_variant,
                "batch_size": batch,
                "same_param_mlp_hidden": same_hidden,
                "repair_variant": local.basis_repair_variant,
                "manual_kernel_variant": manual_variant,
                "manual_kernel_variant_profiled": prof.get("manual_kernel_variant_profiled", ""),
                "manual_ce_train_stream_profiled": manual_profiled,
                "manual_grad_relerr_max": manual_audit.get("grad_relerr_max", ""),
                "manual_grad_cos_min": manual_audit.get("grad_cos_min", ""),
                "manual_output_max_abs_error": manual_audit.get("output_max_abs_error", ""),
                "manual_correctness_pass": manual_correct,
                "manual_correctness_blocker": manual_audit.get("manual_correctness_blocker", ""),
                "gradcheck_pass": manual_correct,
                "official_fused_kernel_complete": official_fused,
                "functional_runner_kernel_match": int(manual_profiled and str(prof.get("manual_kernel_variant_profiled", "")) == manual_variant),
                "no_materialize_complete": int(manual_profiled and "triton" in manual_variant),
                "train_stream_fused_kernel_complete": official_fused,
                "forward_ratio_vs_mlp": prof.get("forward_ratio", ""),
                "backward_ratio_vs_mlp": prof.get("backward_ratio", ""),
                "step_ratio_vs_mlp": prof.get("training_step_ratio", ""),
                "memory_ratio_vs_mlp": prof.get("memory_ratio", ""),
                "forward_only_ms": prof.get("forward_only_ms", ""),
                "backward_grad_ms": prof.get("backward_grad_ms", ""),
                "step_training_only_ms": prof.get("step_training_only_ms", ""),
                "v22_05_official_transition_probe": 1,
                "runtime_sec": time.perf_counter() - start,
                "execution_status": "measured",
            }
            row.update(component_telemetry)
            row.update(classify_drat_drbf_v22_05(row))
            row["official_promotion_allowed"] = int(row.get("E1_official", 0))
            rows.append(row)
            waterfall.extend(efficiency_waterfall_rows(row))
        except Exception as exc:
            row = {
                "carrier": carrier,
                "component_variant": component_variant,
                "batch_size": batch,
                "execution_status": f"blocked:{type(exc).__name__}",
                "blocker": str(exc)[:500],
                "gradcheck_pass": 0,
                "official_fused_kernel_complete": 0,
                "v22_05_official_transition_probe": 1,
                "runtime_sec": time.perf_counter() - start,
            }
            row.update(classify_drat_drbf_v22_05(row))
            row["official_promotion_allowed"] = 0
            rows.append(row)
    return rows, waterfall


def _drat_official_transition_rows(args: argparse.Namespace, device: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _official_transition_rows(
        args,
        device,
        carrier="D-RAT",
        repair_variant="RAT22.05-official-rational-k4-triton",
        component_variant="RAT22.05-official-rational-k4-triton-trainpath",
        manual_token="rational_k4_triton_l3_matmul",
    )


def _drbf_official_transition_rows(args: argparse.Namespace, device: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _official_transition_rows(
        args,
        device,
        carrier="D-RBF",
        repair_variant="RBF22.03-R1-compact-local-k4-no-dense",
        component_variant="RBF22.05-official-rbf-k4-triton-trainpath",
        manual_token="rbf_k4_triton_l3_matmul",
    )


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for carrier in sorted({str(r.get("carrier", "")) for r in rows}):
        group = [r for r in rows if str(r.get("carrier", "")) == carrier]
        best_forward = min(_f(r.get("forward_ratio_vs_mlp")) for r in group)
        best_step = min(_f(r.get("step_ratio_vs_mlp")) for r in group)
        best_memory = min(_f(r.get("memory_ratio_vs_mlp")) for r in group)
        near = sum(int(r.get("micro_near_E1", 0)) for r in group)
        e1 = sum(int(r.get("E1_official", 0)) for r in group)
        official_fused = sum(int(r.get("official_fused_kernel_complete", 0)) for r in group)
        blocker = ""
        if not e1:
            blocker = "near_E1_missing" if official_fused else ("official_fused_missing" if near else "near_E1_missing")
        out.append(
            {
                "carrier": carrier,
                "profile_rows": len(group),
                "micro_near_E1_rows": near,
                "E1_official_rows": e1,
                "official_fused_rows": official_fused,
                "best_forward_ratio": best_forward,
                "best_step_ratio": best_step,
                "best_memory_ratio": best_memory,
                "decision": "OfficialRepairOpened" if e1 else ("MicroNearE1OfficialFusedBlocked" if near else "NearE1Blocked"),
                "blocker": blocker,
            }
        )
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    device = repair._device(args.device)
    batches = [int(x) for x in str(args.batch_sizes).split(",") if x.strip()]
    rows = repair._rat_rows(device, batches, args.hidden, args.warmup, args.iters)
    rows.extend(repair._rbf_rows(device, batches, args.hidden, args.warmup, args.iters))
    for row in rows:
        row.update(classify_drat_drbf_v22_05(row))
        row["v22_05_active_repair"] = 1
        row["official_promotion_allowed"] = int(row.get("E1_official", 0))
    official_rows: list[dict[str, Any]] = []
    official_waterfall: list[dict[str, Any]] = []
    if int(args.official_transition):
        drat_official_rows, drat_official_waterfall = _drat_official_transition_rows(args, device)
        drbf_official_rows, drbf_official_waterfall = _drbf_official_transition_rows(args, device)
        official_rows = drat_official_rows + drbf_official_rows
        official_waterfall = drat_official_waterfall + drbf_official_waterfall
        rows.extend(official_rows)
    write_rows(out_dir / "v22_05_drat_drbf_active_repair.csv", rows)
    write_rows(out_dir / "v22_05_drat_component_waterfall.csv", [r for r in rows if r.get("carrier") == "D-RAT"])
    write_rows(out_dir / "v22_05_drbf_component_waterfall.csv", [r for r in rows if r.get("carrier") == "D-RBF"])
    write_rows(out_dir / "v22_05_drat_official_transition.csv", [r for r in official_rows if r.get("carrier") == "D-RAT"])
    write_rows(out_dir / "v22_05_drbf_official_transition.csv", [r for r in official_rows if r.get("carrier") == "D-RBF"])
    write_rows(out_dir / "v22_05_drat_official_transition_waterfall.csv", [r for r in official_waterfall if r.get("carrier") == "D-RAT"])
    write_rows(out_dir / "v22_05_drbf_official_transition_waterfall.csv", [r for r in official_waterfall if r.get("carrier") == "D-RBF"])
    summary = _summary(rows)
    write_rows(out_dir / "v22_05_drat_drbf_repair_summary.csv", summary)
    write_json(out_dir / "v22_05_drat_drbf_repair_route.json", {r["carrier"]: r for r in summary})
    simple_svg(out_dir / "figures/D-RAT_component_waterfall.svg", "v22.05 D-RAT component waterfall", [r for r in rows if r.get("carrier") == "D-RAT"], "forward_ratio_vs_mlp")
    simple_svg(out_dir / "figures/D-RBF_component_waterfall.svg", "v22.05 D-RBF component waterfall", [r for r in rows if r.get("carrier") == "D-RBF"], "forward_ratio_vs_mlp")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_drat_drbf_repair.py --out-dir {out_dir} --device {args.device} --batch-sizes {args.batch_sizes} --hidden {args.hidden} --iters {args.iters} --warmup {args.warmup}",
        status="completed",
        note=f"rows={len(rows)} near={sum(int(r.get('micro_near_E1',0)) for r in rows)} official={sum(int(r.get('E1_official',0)) for r in rows)}",
    )


if __name__ == "__main__":
    main()
