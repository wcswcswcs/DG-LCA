#!/usr/bin/env python3
"""v12.11 environment verification: reproduce v12.10 family L3 fused-kernel
step_ratio_q90 / memory_ratio_q90 numbers WITHOUT modifying any historical
runner code.

Strategy
--------
The v12.10 reproduction blocker is that ``run_v1283_b109_classic_family_functional_geometry.py``
references several optimizer helpers on ``run_v126_lowerlevel_fhq_functional_geometry.py``
(``_ManualForeachAdamW``, ``_variant_uses_*``, ``_triton_adamw_params``, ``_quad_proj_group_hparams``)
that do not exist in the current v126 source. Those helpers are ONLY needed
inside the B109 task loop (``run_b109_auc_autopsy``). The family efficiency
measurement (``run_family_microbench``) does NOT need them — it only calls
``v126._measure_step`` (which exists) and ``_time_call``.

So we drive the family efficiency path directly here, with no changes to the
historical runner files. We only reuse public attributes of the historical
modules:

- ``run_v126_lowerlevel_fhq_functional_geometry._measure_step``
- ``run_v126_lowerlevel_fhq_functional_geometry._make_adamw``
- ``run_v1283_b109_classic_family_functional_geometry._make_model``
- ``run_v1283_b109_classic_family_functional_geometry._specs_for``

Ground truth numbers we are checking against (from
``docs/DG-KAN_v12.10_B320_Functional_ClassicNoBSpline_执行复盘.md`` sections 12-13
and 9) are reported alongside the freshly measured values so any deviation is
auditable. We do not write any official artifact and we do not claim official
family success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
import run_v126_lowerlevel_fhq_functional_geometry as v126  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


# v12.10 ground-truth (focused MNIST seed0 epochs1 train_size=512 protocol).
# These are the numbers documented as already measured in the v12.10 复盘.
GROUND_TRUTH = [
    {
        "family": "Chebyshev",
        "candidate_id": "B3f-ChebyKAN-K4-tritonL3-matmulTile",
        "v1210_l3_step_ratio_q90": 0.6521527773287539,
        "v1210_l3_memory_ratio_q90": 1.0047619047619047,
        "v1210_l3_official_fused_kernel": 1,
        "v1210_source": "执行复盘 §9 (Chebyshev extended focused run)",
    },
    {
        "family": "Fourier",
        "candidate_id": "B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile",
        "v1210_l3_step_ratio_q90": 0.7572990222904608,
        "v1210_l3_memory_ratio_q90": 0.12428571428571429,
        "v1210_l3_official_fused_kernel": 1,
        "v1210_source": "执行复盘 §9 (Fourier extended focused run)",
    },
    {
        "family": "RBF",
        "candidate_id": "B2r-FastKAN-RBF-stream-K2-repair",
        "v1210_l3_step_ratio_q90": 0.7103699836137742,
        "v1210_l3_memory_ratio_q90": 0.9990476190476191,
        "v1210_l3_official_fused_kernel": 1,
        "v1210_source": "执行复盘 §12 (RBF/FastKAN fused L3 kernel)",
    },
    {
        "family": "RBF",
        "candidate_id": "B2s-GaussianRBF-stream-K4-recompute",
        "v1210_l3_step_ratio_q90": 0.6920597147480739,
        "v1210_l3_memory_ratio_q90": 1.0047619047619047,
        "v1210_l3_official_fused_kernel": 1,
        "v1210_source": "执行复盘 §12 (RBF/FastKAN fused L3 kernel)",
    },
    {
        "family": "Wavelet",
        "candidate_id": "B5h-HatWaveletKAN-local-K4",
        "v1210_l3_step_ratio_q90": 0.648993247415022,
        "v1210_l3_memory_ratio_q90": 1.0047619047619047,
        "v1210_l3_official_fused_kernel": 1,
        "v1210_source": "执行复盘 §13 (Wavelet hat fused L3 kernel)",
    },
]


def now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v12.11 environment/family-efficiency reproduction check"
    )
    p.add_argument("--out-dir", default=f"results/v12_11_b320_functional_mechanism_classic_nobspline/v1211_env_repro_{now_tag()}")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--kernel-warmup-steps", type=int, default=5)
    p.add_argument("--kernel-measure-steps", type=int, default=20)
    p.add_argument("--task-compile-warmup-steps", type=int, default=0)
    p.add_argument("--task-lr-schedule", default="linear_warmup10_cosine_final075")
    p.add_argument("--datasets", default="MNIST")
    p.add_argument("--seeds", default="0")
    p.add_argument("--no-download", action="store_true", default=True)
    p.add_argument("--task-timing-warmup-epochs", type=int, default=0)
    p.add_argument(
        "--candidates",
        default=",".join(r["candidate_id"] for r in GROUND_TRUTH),
        help="comma-separated candidate ids to verify",
    )
    return p.parse_args()


def setup_device(arg: str) -> torch.device:
    if arg in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_float(value, default=float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(args.device)
    print(f"[v1211-verify] device={device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"[v1211-verify] gpu0={prop.name} sm={prop.major}.{prop.minor} mem={prop.total_memory/1e9:.1f}GB")

    data = v120._load_vision_split(
        args,
        "MNIST",
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
    )
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    print(f"[v1211-verify] data ok: input_dim={input_dim} output_dim={output_dim} train={x_train.shape}")

    _, budget = v124._param_budget(int(input_dim), int(output_dim))
    specs = {s.candidate_id: s for s in prim.primitive_specs(budget, int(input_dim), int(output_dim))}
    print(f"[v1211-verify] total specs available: {len(specs)}  param_budget={budget}")

    xb = x_train[: int(args.batch_size)].contiguous()
    yb = y_train[: int(args.batch_size)].contiguous()

    # Reference MLP timing (this is the denominator of step_ratio / memory_ratio).
    print("[v1211-verify] measuring MLP-same-param-AdamW baseline ...")
    mlp = v124._make_model("MLP-same-param-AdamW", int(input_dim), int(output_dim), x_train, device, int(args.seed) + 8001, specs.get("MLP-same-param-AdamW"), budget)
    if device.type == "cuda":
        torch.cuda.synchronize()
    mlp_m = v126._measure_step(args, mlp, "MLP-same-param-AdamW", xb, yb, device, "autograd")
    print(f"[v1211-verify] MLP step_q90_ms={mlp_m['step_q90_ms']:.4f} mem_peak_mb={mlp_m['memory_peak_mb']:.2f}")

    candidate_ids = [c.strip() for c in str(args.candidates).split(",") if c.strip()]
    gt_by_id = {r["candidate_id"]: r for r in GROUND_TRUTH}

    rows = []
    for cid in candidate_ids:
        gt = gt_by_id.get(cid, {})
        print(f"\n[v1211-verify] === measuring {cid} ===")
        spec = specs.get(cid)
        if spec is None:
            print(f"  SKIP: spec missing for {cid}")
            rows.append({
                "candidate_id": cid,
                "family": gt.get("family", "?"),
                "status": "spec_missing",
            })
            continue
        try:
            t0 = time.time()
            model = v124._make_model(cid, int(input_dim), int(output_dim), x_train, device, int(args.seed) + 9100 + len(rows), spec, budget)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            manual_variant = model.manual_kernel_variant() if hasattr(model, "manual_kernel_variant") else "no_manual_variant"
            l1_m = v126._measure_step(args, model, cid, xb, yb, device, "autograd")
            l3_m = None
            l3_failure = ""
            try:
                manual_model = v124._make_model(cid, int(input_dim), int(output_dim), x_train, device, int(args.seed) + 9300 + len(rows), spec, budget)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                l3_m = v126._measure_step(args, manual_model, cid, xb, yb, device, "manual")
            except Exception as exc:
                l3_failure = f"{type(exc).__name__}:{exc}"
            elapsed = time.time() - t0
            l1_step_ratio = _safe_float(l1_m.get("step_q90_ms")) / max(1e-9, _safe_float(mlp_m.get("step_q90_ms"), 1.0))
            l1_mem_ratio = _safe_float(l1_m.get("memory_peak_mb")) / max(1e-9, _safe_float(mlp_m.get("memory_peak_mb"), 1.0))
            l3_step_ratio = _safe_float(l3_m.get("step_q90_ms")) / max(1e-9, _safe_float(mlp_m.get("step_q90_ms"), 1.0)) if l3_m else float("nan")
            l3_mem_ratio = _safe_float(l3_m.get("memory_peak_mb")) / max(1e-9, _safe_float(mlp_m.get("memory_peak_mb"), 1.0)) if l3_m else float("nan")
            row = {
                "candidate_id": cid,
                "family": gt.get("family", "?"),
                "manual_kernel_variant": manual_variant,
                "elapsed_s": round(elapsed, 2),
                "l1_step_q90_ms": l1_m.get("step_q90_ms"),
                "l1_forward_q90_ms": l1_m.get("forward_q90_ms"),
                "l1_backward_q90_ms": l1_m.get("backward_q90_ms"),
                "l1_update_q90_ms": l1_m.get("update_q90_ms"),
                "l1_memory_peak_mb": l1_m.get("memory_peak_mb"),
                "l1_step_ratio_q90": l1_step_ratio,
                "l1_memory_ratio_q90": l1_mem_ratio,
                "l3_step_q90_ms": l3_m.get("step_q90_ms") if l3_m else "",
                "l3_forward_q90_ms": l3_m.get("forward_q90_ms") if l3_m else "",
                "l3_backward_q90_ms": l3_m.get("backward_q90_ms") if l3_m else "",
                "l3_update_q90_ms": l3_m.get("update_q90_ms") if l3_m else "",
                "l3_memory_peak_mb": l3_m.get("memory_peak_mb") if l3_m else "",
                "l3_step_ratio_q90": l3_step_ratio,
                "l3_memory_ratio_q90": l3_mem_ratio,
                "l3_failure": l3_failure,
                "v1210_l3_step_ratio_q90": gt.get("v1210_l3_step_ratio_q90"),
                "v1210_l3_memory_ratio_q90": gt.get("v1210_l3_memory_ratio_q90"),
                "v1210_source": gt.get("v1210_source", ""),
            }
            if not math.isnan(l3_step_ratio) and gt.get("v1210_l3_step_ratio_q90") is not None:
                gt_step = float(gt["v1210_l3_step_ratio_q90"])
                gt_mem = float(gt["v1210_l3_memory_ratio_q90"])
                row["l3_step_ratio_rel_diff_pct"] = round(100.0 * (l3_step_ratio - gt_step) / gt_step, 2)
                row["l3_memory_ratio_rel_diff_pct"] = round(100.0 * (l3_mem_ratio - gt_mem) / max(1e-9, gt_mem), 2)
            rows.append(row)
            print(f"  L1 step={l1_m['step_q90_ms']:.4f}ms ratio={l1_step_ratio:.4f}  mem={l1_m['memory_peak_mb']:.2f}MB ratio={l1_mem_ratio:.4f}")
            if l3_m:
                print(f"  L3 step={l3_m['step_q90_ms']:.4f}ms ratio={l3_step_ratio:.4f}  mem={l3_m['memory_peak_mb']:.2f}MB ratio={l3_mem_ratio:.4f}  variant={manual_variant}")
                if gt:
                    print(f"  v1210_ref l3_step_ratio={gt['v1210_l3_step_ratio_q90']:.4f}  rel_diff={row.get('l3_step_ratio_rel_diff_pct')}%")
            else:
                print(f"  L3 FAILED: {l3_failure}")
            del model
            if l3_m:
                del manual_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({
                "candidate_id": cid,
                "family": gt.get("family", "?"),
                "status": "failed",
                "failure": f"{type(exc).__name__}:{exc}",
            })
            print(f"  FAILED: {type(exc).__name__}:{exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # write csv
    out_csv = out_dir / "v1211_env_repro_family_efficiency.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    out_json = out_dir / "v1211_env_repro_summary.json"
    summary = {
        "stamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "mlp_step_q90_ms": mlp_m.get("step_q90_ms"),
        "mlp_memory_peak_mb": mlp_m.get("memory_peak_mb"),
        "row_count": len(rows),
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[v1211-verify] wrote {out_csv}")
    print(f"[v1211-verify] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
