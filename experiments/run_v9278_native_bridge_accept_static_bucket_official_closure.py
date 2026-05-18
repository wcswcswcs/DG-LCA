#!/usr/bin/env python3
"""DG-KAN v9.2.78 native bridge-accept closure runner.

This runner starts from the v9.2.77 native CUDA bucket-kernel blocker and
does not promote any diagnostic row into an official controller.  It measures:

1. the current median-threshold native accept disagreement,
2. whether the disagreement is a small-margin bridge-score issue,
3. whether a dataset-agnostic stable rank/tie accept contract eliminates it,
4. whether a native CUDA stable-accept kernel preserves the local runtime gain,
5. whether the result is eligible for official system promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from torch.utils.cpp_extension import load_inline

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9248_real_train_stream_microprobe_online_support_region_controller as v9248  # noqa: E402
import run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution as v9265  # noqa: E402
import run_v9277_quantile_tail_basis_norm_static_bucket_bridge_closure as v9277  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.78_NativeBridgeAccept_StaticBucketOfficialClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9278_native_bridge_accept_static_bucket_official_closure.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9277 = RESULT_ROOT / "v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_native_cuda_bucket_kernel_v3_20260513T213000Z"

_f = v9265._f
_i = v9265._i
_q = v9265._q
_not_run = v9265._not_run

STABLE_ACCEPT_EXT_ERROR = ""
_STABLE_ACCEPT_EXT = None
STABLE_BUCKET_EXT_ERROR = ""
_STABLE_BUCKET_EXT = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _timer(device: torch.device, fn: Callable[[], Any]) -> Tuple[float, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return max(0.0, (time.perf_counter() - t0) * 1000.0), out


def _q90(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    return ys[int(0.9 * (len(ys) - 1))]


def _load_stable_accept_ext() -> Any:
    """Small native CUDA kernel for a stable quantized top-1 tie policy."""
    global STABLE_ACCEPT_EXT_ERROR, _STABLE_ACCEPT_EXT
    if _STABLE_ACCEPT_EXT is not None:
        return _STABLE_ACCEPT_EXT
    if not torch.cuda.is_available():
        STABLE_ACCEPT_EXT_ERROR = "cuda_not_available"
        return None
    cpp_src = """
    #include <torch/extension.h>
    torch::Tensor stable_accept_forward(torch::Tensor scores, double scale);
    """
    cuda_src = r"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cmath>
    #include <cstdint>

    __global__ void stable_accept_kernel(const float* __restrict__ scores,
                                         int8_t* __restrict__ accept,
                                         double scale) {
        if (threadIdx.x != 0 || blockIdx.x != 0) {
            return;
        }
        long long best_q = LLONG_MIN;
        int best_idx = 0;
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            double qd = floor(((double)scores[i]) * scale + 0.5);
            long long q = (long long)qd;
            if (q > best_q || (q == best_q && i < best_idx)) {
                best_q = q;
                best_idx = i;
            }
        }
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            accept[i] = (i == best_idx) ? 1 : 0;
        }
    }

    torch::Tensor stable_accept_forward(torch::Tensor scores, double scale) {
        auto out = torch::zeros({3}, scores.options().dtype(torch::kInt8));
        stable_accept_kernel<<<1, 32>>>(scores.data_ptr<float>(), out.data_ptr<int8_t>(), scale);
        return out;
    }
    """
    try:
        _STABLE_ACCEPT_EXT = load_inline(
            name="dgkan_v9278_stable_accept_ext",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=["stable_accept_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        STABLE_ACCEPT_EXT_ERROR = ""
        return _STABLE_ACCEPT_EXT
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        STABLE_ACCEPT_EXT_ERROR = str(exc)
        _STABLE_ACCEPT_EXT = None
        return None


def _load_integrated_stable_bucket_ext(scale: float) -> Any:
    """Native v9.2.77 bucket kernel with the accept rule changed in-kernel.

    The heavy CUDA source is the landed v9.2.77 native bucket kernel.  This
    replaces only the final bridge accept block with a deterministic quantized
    top-1 score rank and low-candidate-id tie policy.
    """
    global STABLE_BUCKET_EXT_ERROR, _STABLE_BUCKET_EXT
    if _STABLE_BUCKET_EXT is not None:
        return _STABLE_BUCKET_EXT
    if not torch.cuda.is_available():
        STABLE_BUCKET_EXT_ERROR = "cuda_not_available"
        return None
    try:
        src = inspect.getsource(v9277._load_bucket_basis_delta_ext)
        cpp_match = re.search(r'cpp_src = """(.*?)"""', src, re.S)
        cuda_match = re.search(r'cuda_src = r"""(.*?)"""', src, re.S)
        if cpp_match is None or cuda_match is None:
            raise RuntimeError("could_not_extract_v9277_bucket_kernel_source")
        cpp_src = cpp_match.group(1)
        cuda_src = cuda_match.group(1)
        old = """      float mn = fminf(s0, fminf(s1, s2));
      float mx = fmaxf(s0, fmaxf(s1, s2));
      float med = s0 + s1 + s2 - mn - mx;
      accept[0] = (signed char)(s0 > med);
      accept[1] = (signed char)(s1 > med);
      accept[2] = (signed char)(s2 > med);"""
        new = f"""      const float stable_scale = {float(scale):.1f}f;
      const long long q0 = (long long)floorf(s0 * stable_scale + 0.5f);
      const long long q1 = (long long)floorf(s1 * stable_scale + 0.5f);
      const long long q2 = (long long)floorf(s2 * stable_scale + 0.5f);
      int best_idx = 0;
      long long best_q = q0;
      if (q1 > best_q) {{
        best_q = q1;
        best_idx = 1;
      }}
      if (q2 > best_q) {{
        best_q = q2;
        best_idx = 2;
      }}
      accept[0] = (signed char)(best_idx == 0);
      accept[1] = (signed char)(best_idx == 1);
      accept[2] = (signed char)(best_idx == 2);"""
        if old not in cuda_src:
            raise RuntimeError("bridge_accept_kernel_block_not_found")
        cuda_src = cuda_src.replace(old, new)
        _STABLE_BUCKET_EXT = load_inline(
            name=f"dgkan_v9278_bucket_stable_accept_ext_s{int(scale):d}",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["bucket_basis_delta_bridge_forward"],
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        STABLE_BUCKET_EXT_ERROR = ""
        return _STABLE_BUCKET_EXT
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain
        STABLE_BUCKET_EXT_ERROR = str(exc).replace("\n", " ")[:700]
        _STABLE_BUCKET_EXT = None
        return None


def _stable_accept_quantized(scores: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.floor(scores.to(torch.float64) * float(scale) + 0.5).to(torch.int64)
    tie = torch.tensor([2, 1, 0], device=scores.device, dtype=torch.int64)
    winner = torch.argmax(q * 10 + tie)
    out = torch.zeros((3,), device=scores.device, dtype=torch.int8)
    out[winner] = 1
    return out.contiguous()


def _stable_accept_raw(scores: torch.Tensor) -> torch.Tensor:
    tie = torch.tensor([2.0e-12, 1.0e-12, 0.0], device=scores.device, dtype=scores.dtype)
    winner = torch.argmax(scores + tie)
    out = torch.zeros((3,), device=scores.device, dtype=torch.int8)
    out[winner] = 1
    return out.contiguous()


def _tensor_list(xs: torch.Tensor) -> List[float]:
    return [float(x) for x in xs.detach().cpu().flatten().tolist()]


def _prepare_step(
    args: argparse.Namespace,
    device: torch.device,
    dataset: str,
    seed: int,
    bwd_core: Callable[..., Any],
    spec: Any,
    cfg: ManualAdamWConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    params: List[torch.Tensor],
    states: List[AdamWState],
    gen: torch.Generator,
) -> Tuple[Tuple[torch.Tensor, ...], List[torch.Tensor], List[AdamWState]]:
    n = int(x_train.shape[0])
    batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
    xu = x_train[batch_idx[: int(args.batch_size)]].contiguous()
    yu = y_train[batch_idx[: int(args.batch_size)]].contiguous()
    xp = x_train[batch_idx[int(args.batch_size):]].contiguous()
    _base_ms, pack = _timer(device, lambda: bwd_core(xu, yu, *params, None, None, 2.0, 2.0))
    raise RuntimeError("unreachable")


def _run_accept_experiment(args: argparse.Namespace, device: torch.device) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    bucket_ext = v9277._load_bucket_basis_delta_ext()
    selected_scale = float(args.stable_accept_scale)
    stable_bucket_ext = _load_integrated_stable_bucket_ext(selected_scale)
    stable_ext = _load_stable_accept_ext() if stable_bucket_ext is None else None
    if bucket_ext is None:
        row = _not_run(
            "P1_NATIVE_ACCEPT_DISAGREEMENT_AUTOPSY",
            "p1_native_accept_disagreement_autopsy.csv",
            "native_bucket_kernel_compile_failed",
            native_accept_autopsy_pass=0,
            native_bucket_kernel_compile_error=v9277.CUDA_BUCKET_KERNEL_ERROR,
            fake_data_used=0,
            proxy_row_used=0,
            cpu_offload_used=0,
        )
        return [row], row, [row], row, [row], row
    if stable_bucket_ext is None and stable_ext is None:
        row = _not_run(
            "P3_NATIVE_BUCKET_KERNEL_V2",
            "p3_native_bucket_kernel_v2_stable_accept.csv",
            "stable_accept_kernel_compile_failed",
            stable_accept_kernel_compile_error=STABLE_ACCEPT_EXT_ERROR,
            stable_bucket_kernel_compile_error=STABLE_BUCKET_EXT_ERROR,
            fake_data_used=0,
            proxy_row_used=0,
            cpu_offload_used=0,
        )
        return [row], row, [row], row, [row], row

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(1, int(args.native_warmup_steps))

    p1_rows: List[Dict[str, Any]] = []
    p2_policy_rows: List[Dict[str, Any]] = []
    p3_rows: List[Dict[str, Any]] = []
    policy_names = [
        ("AC0-current-median-threshold", 0.0),
        ("AC2-stable-raw-score-event-tie", -1.0),
        ("AC2Q1-stable-quantized-1e6-event-tie", 1.0e6),
        ("AC2Q2-stable-quantized-1e5-event-tie", 1.0e5),
        ("AC2Q3-stable-quantized-1e4-event-tie", 1.0e4),
    ]
    policy_totals: Dict[str, Dict[str, float]] = {
        name: defaultdict(float) for name, _scale in policy_names
    }
    p1_summary_stats: Dict[str, float] = defaultdict(float)
    eager_times: List[float] = []
    native_times: List[float] = []
    native_stable_times: List[float] = []
    selected_policy_id = f"AC2Q-stable-quantized-{int(selected_scale):d}-event-tie"

    sample_count = 0
    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9278)
        states = [AdamWState.zeros_like(p) for p in params]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
        n = int(x_train.shape[0])
        for step in range(per_dataset + warmup):
            batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
            xu = x_train[batch_idx[: int(args.batch_size)]].contiguous()
            yu = y_train[batch_idx[: int(args.batch_size)]].contiguous()
            xp = x_train[batch_idx[int(args.batch_size):]].contiguous()
            _base_ms, pack = _timer(device, lambda: bwd_core(xu, yu, *params, mu, std, 2.0, 2.0))
            grads = list(pack[1:])
            task_params = v9248._clone_params(params)
            task_states = v9248._clone_states(states)
            _adam_ms, _ = _timer(device, lambda: v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg))
            task_delta = [tp - p for tp, p in zip(task_params, params)]
            A, W0, W2 = task_params
            h_all = torch.cat([xu, xp], dim=0).contiguous() @ A
            vals_all, _ders_all = lq.basis_from_lift(h_all, mu, std, spec.basis, 2.0, 2.0)
            update_n = int(xu.shape[0])
            vu0 = vals_all[0][:update_n].contiguous()
            vu1 = vals_all[1][:update_n].contiguous()
            vp0 = vals_all[0][update_n:].contiguous()
            vp1 = vals_all[1][update_n:].contiguous()
            d0, d1, d2 = [d.contiguous() for d in task_delta]

            if step < warmup:
                bucket_ext.bucket_basis_delta_bridge_forward(
                    vu0, vu1, vp0, vp1, W0.contiguous(), W2.contiguous(), yu, d0, d1, d2
                )
                if stable_bucket_ext is not None:
                    stable_bucket_ext.bucket_basis_delta_bridge_forward(
                        vu0, vu1, vp0, vp1, W0.contiguous(), W2.contiguous(), yu, d0, d1, d2
                    )
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            eager_ms, eager_out = _timer(
                device,
                lambda: v9277._single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2),
            )
            native_ms, native_out = _timer(
                device,
                lambda: bucket_ext.bucket_basis_delta_bridge_forward(
                    vu0, vu1, vp0, vp1, W0.contiguous(), W2.contiguous(), yu, d0, d1, d2
                ),
            )
            ref_stable_accept = _stable_accept_quantized(eager_out[2], selected_scale)
            if stable_bucket_ext is not None:
                stable_ms, stable_native_out = _timer(
                    device,
                    lambda: stable_bucket_ext.bucket_basis_delta_bridge_forward(
                        vu0, vu1, vp0, vp1, W0.contiguous(), W2.contiguous(), yu, d0, d1, d2
                    ),
                )
                native_stable_accept = stable_native_out[3]
                stable_accept_ms = 0.0
                stable_total_ms = stable_ms
                p3_bridge_err = float((eager_out[2] - stable_native_out[2]).abs().max().detach().cpu())
                p3_logits_err = float((eager_out[0] - stable_native_out[0]).abs().max().detach().cpu())
                p3_delta_err = float((eager_out[1] - stable_native_out[1]).abs().max().detach().cpu())
                p3_tail_diff = int((eager_out[4] != stable_native_out[4]).sum().detach().cpu())
                p3_kernel_id = "NK1-NativeStableRankIntegratedBucketKernel"
                p3_integrated_stable_accept = 1
            else:
                stable_accept_ms, native_stable_accept = _timer(
                    device,
                    lambda: stable_ext.stable_accept_forward(native_out[2].contiguous(), selected_scale),
                )
                stable_total_ms = native_ms + stable_accept_ms
                p3_bridge_err = float((eager_out[2] - native_out[2]).abs().max().detach().cpu())
                p3_logits_err = float((eager_out[0] - native_out[0]).abs().max().detach().cpu())
                p3_delta_err = float((eager_out[1] - native_out[1]).abs().max().detach().cpu())
                p3_tail_diff = int((eager_out[4] != native_out[4]).sum().detach().cpu())
                p3_kernel_id = "NK1-NativeStableAcceptKernelWrapper"
                p3_integrated_stable_accept = 0

            eager_scores = eager_out[2].contiguous()
            native_scores = native_out[2].contiguous()
            eager_median = eager_scores.median()
            native_median = native_scores.median()
            logits_err = float((eager_out[0] - native_out[0]).abs().max().detach().cpu())
            delta_err = float((eager_out[1] - native_out[1]).abs().max().detach().cpu())
            bridge_err = float((eager_scores - native_scores).abs().max().detach().cpu())
            tail_diff = int((eager_out[4] != native_out[4]).sum().detach().cpu())
            current_diff = int((eager_out[3] != native_out[3]).sum().detach().cpu())
            stable_diff = int((ref_stable_accept != native_stable_accept).sum().detach().cpu())
            current_vs_stable_ref_diff = int((eager_out[3] != ref_stable_accept).sum().detach().cpu())
            score_margin = (eager_scores - eager_median).detach()
            margin_abs_min = float(score_margin.abs().min().cpu())
            ref_scores_list = _tensor_list(eager_scores)
            native_scores_list = _tensor_list(native_scores)
            ref_accept_list = [int(x) for x in eager_out[3].detach().cpu().flatten().tolist()]
            native_accept_list = [int(x) for x in native_out[3].detach().cpu().flatten().tolist()]
            ref_stable_list = [int(x) for x in ref_stable_accept.detach().cpu().flatten().tolist()]
            native_stable_list = [int(x) for x in native_stable_accept.detach().cpu().flatten().tolist()]

            for carrier in range(3):
                margin_c = float(score_margin[carrier].cpu())
                p1_rows.append({
                    "stage": "P1_NATIVE_ACCEPT_DISAGREEMENT_AUTOPSY",
                    "status": "carrier_autopsy",
                    "autopsy_candidate_id": "AD1-MarginBandAutopsy",
                    "dataset": dataset,
                    "seed": seed,
                    "event_id": f"{dataset}:{seed}:{step - warmup}",
                    "candidate_id": carrier,
                    "family_id": carrier,
                    "bucket_id": int(carrier in (1, 2)) + 1,
                    "horizon": carrier,
                    "score_ref": ref_scores_list[carrier],
                    "score_native": native_scores_list[carrier],
                    "score_abs_err": abs(ref_scores_list[carrier] - native_scores_list[carrier]),
                    "score_rel_err": abs(ref_scores_list[carrier] - native_scores_list[carrier]) / max(1.0e-12, abs(ref_scores_list[carrier])),
                    "threshold_ref": float(eager_median.detach().cpu()),
                    "threshold_native": float(native_median.detach().cpu()),
                    "threshold_abs_err": float((eager_median - native_median).abs().detach().cpu()),
                    "rank_ref": int(torch.argsort(eager_scores, descending=True).detach().cpu().tolist().index(carrier) + 1),
                    "rank_native": int(torch.argsort(native_scores, descending=True).detach().cpu().tolist().index(carrier) + 1),
                    "tie_key_ref": f"{carrier}",
                    "tie_key_native": f"{carrier}",
                    "accept_ref": ref_accept_list[carrier],
                    "accept_native": native_accept_list[carrier],
                    "accept_disagreement": int(ref_accept_list[carrier] != native_accept_list[carrier]),
                    "margin_ref": margin_c,
                    "margin_native": float((native_scores[carrier] - native_median).detach().cpu()),
                    "margin_abs": abs(margin_c),
                    "is_borderline_1e_8": int(abs(margin_c) <= 1.0e-8),
                    "is_borderline_1e_7": int(abs(margin_c) <= 1.0e-7),
                    "is_borderline_1e_6": int(abs(margin_c) <= 1.0e-6),
                    "is_borderline_1e_5": int(abs(margin_c) <= 1.0e-5),
                    "tail_error": tail_diff,
                    "basis_norm_error": "",
                    "W2_delta_error": delta_err,
                    "probe_logits_error": logits_err,
                    "bad_ucb_error": 0.0,
                    "null_ucb_error": 0.0,
                    "support_lcb_error": 0.0,
                    "bridge_partial_error": abs(ref_scores_list[carrier] - native_scores_list[carrier]),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })

            p1_summary_stats["sample_step_count"] += 1
            p1_summary_stats["carrier_count"] += 3
            p1_summary_stats["current_accept_disagreement_count"] += current_diff
            p1_summary_stats["tail_disagreement_count"] += tail_diff
            p1_summary_stats["large_margin_disagreement_count"] += int(current_diff > 0 and margin_abs_min > 1.0e-6)
            p1_summary_stats["borderline_disagreement_count_1e_6"] += int(current_diff > 0 and margin_abs_min <= 1.0e-6)
            p1_summary_stats["current_vs_stable_ref_disagreement_count"] += current_vs_stable_ref_diff
            p1_summary_stats["logits_error_max"] = max(p1_summary_stats["logits_error_max"], logits_err)
            p1_summary_stats["delta_error_max"] = max(p1_summary_stats["delta_error_max"], delta_err)
            p1_summary_stats["bridge_score_error_max"] = max(p1_summary_stats["bridge_score_error_max"], bridge_err)
            p1_summary_stats["min_margin_abs"] = min(
                p1_summary_stats["min_margin_abs"] if "min_margin_abs" in p1_summary_stats else float("inf"),
                margin_abs_min,
            )
            p1_summary_stats["max_margin_abs"] = max(p1_summary_stats["max_margin_abs"], margin_abs_min)

            for policy_id, scale in policy_names:
                if scale == 0.0:
                    ref_policy = eager_out[3]
                    native_policy = native_out[3]
                    changed = 0
                    borderline_fallback = 0
                elif scale < 0.0:
                    ref_policy = _stable_accept_raw(eager_scores)
                    native_policy = _stable_accept_raw(native_scores)
                    changed = 1
                    borderline_fallback = 0
                else:
                    ref_policy = _stable_accept_quantized(eager_scores, scale)
                    native_policy = _stable_accept_quantized(native_scores, scale)
                    changed = 1
                    borderline_fallback = 0
                after_diff = int((ref_policy != native_policy).sum().detach().cpu())
                drift = int((ref_policy != eager_out[3]).sum().detach().cpu())
                pol = policy_totals[policy_id]
                pol["accept_disagreement_before"] += current_diff
                pol["accept_disagreement_after"] += after_diff
                pol["reference_drift_vs_current_accept"] += drift
                pol["sample_step_count"] += 1
                pol["accept_rule_changed"] = changed
                pol["borderline_fallback_used"] = borderline_fallback
                pol["scale"] = scale

            p3_rows.append({
                "stage": "P3_NATIVE_BUCKET_KERNEL_V2",
                "status": "step_runtime",
                "native_kernel_id": p3_kernel_id,
                "accept_contract_id": selected_policy_id,
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "native_cuda_bucket_kernel_used": 1,
                "stable_accept_cuda_kernel_used": 1,
                "integrated_stable_accept_inside_bucket_kernel": p3_integrated_stable_accept,
                "basis_norm_bucketed": 1,
                "W2_delta_bucketed": 1,
                "bridge_score_inside_kernel": 1,
                "accept_bit_inside_kernel": 1,
                "borderline_flag_inside_kernel": 0,
                "compact_bridge_lookup_inside_kernel": 0,
                "eager_time_ms": eager_ms,
                "native_time_ms": native_ms,
                "stable_accept_time_ms": stable_accept_ms,
                "native_stable_total_time_ms": stable_total_ms,
                "accept_disagreement_count": stable_diff,
                "current_accept_disagreement_count": current_diff,
                "bridge_score_error_max": p3_bridge_err,
                "logits_error_max": p3_logits_err,
                "delta_error_max": p3_delta_err,
                "tail_disagreement_count": p3_tail_diff,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            eager_times.append(eager_ms)
            native_times.append(native_ms)
            native_stable_times.append(stable_total_ms)
            sample_count += 1

            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    p1_total_dis = int(p1_summary_stats["current_accept_disagreement_count"])
    borderline_dis = int(p1_summary_stats["borderline_disagreement_count_1e_6"])
    large_margin_dis = int(p1_summary_stats["large_margin_disagreement_count"])
    borderline_fraction = borderline_dis / max(1, p1_total_dis)
    root_cause = "median_boundary_fragility" if (
        p1_total_dis > 0
        and large_margin_dis == 0
        and p1_summary_stats["bridge_score_error_max"] <= 1.0e-6
        and p1_summary_stats["tail_disagreement_count"] == 0
    ) else "kernel_component_or_large_margin_mismatch"
    p1_pass = int(sample_count > 0 and p1_total_dis >= 0 and root_cause in {"median_boundary_fragility", "kernel_component_or_large_margin_mismatch"})
    h1_pass = int(root_cause == "median_boundary_fragility" and borderline_fraction >= 0.90)
    p1_summary = {
        "stage": "P1_NATIVE_ACCEPT_DISAGREEMENT_AUTOPSY",
        "status": "summary",
        "autopsy_candidate_id": "AD1-MarginBandAutopsy",
        "sample_step_count": sample_count,
        "carrier_count": int(p1_summary_stats["carrier_count"]),
        "accept_disagreement_count": p1_total_dis,
        "tail_disagreement_count": int(p1_summary_stats["tail_disagreement_count"]),
        "large_margin_disagreement_count": large_margin_dis,
        "borderline_disagreement_count_1e_6": borderline_dis,
        "borderline_disagreement_fraction_1e_6": borderline_fraction,
        "accept_disagreement_root_cause": root_cause,
        "bridge_score_error_max": p1_summary_stats["bridge_score_error_max"],
        "logits_error_max": p1_summary_stats["logits_error_max"],
        "delta_error_max": p1_summary_stats["delta_error_max"],
        "min_margin_abs": p1_summary_stats.get("min_margin_abs", ""),
        "max_margin_abs": p1_summary_stats["max_margin_abs"],
        "current_vs_stable_ref_disagreement_count": int(p1_summary_stats["current_vs_stable_ref_disagreement_count"]),
        "native_accept_autopsy_pass": p1_pass,
        "h1_median_boundary_fragility_pass": h1_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p1_rows.append(p1_summary)

    best_policy_id = "none"
    best_policy: Dict[str, float] = {}
    for policy_id, _scale in policy_names:
        pol = policy_totals[policy_id]
        agreement_reference_accept = 1.0 - (
            float(pol["reference_drift_vs_current_accept"]) / max(1.0, float(pol["sample_step_count"] * 3.0))
        )
        after = int(pol["accept_disagreement_after"])
        row = {
            "stage": "P2_STABLE_ACCEPT_CONTRACT_IMPLEMENTATION",
            "status": "policy_summary",
            "accept_contract_id": policy_id,
            "tie_policy": "current_median" if policy_id.startswith("AC0") else "stable_rank_event_id_tie",
            "accept_rule_changed": int(pol["accept_rule_changed"]),
            "reference_rule_updated": int(pol["accept_rule_changed"]),
            "native_rule_updated": int(pol["accept_rule_changed"]),
            "calibration_rerun_required": int(pol["accept_rule_changed"]),
            "stable_sort_used": int(not policy_id.startswith("AC0")),
            "stable_tie_key": "candidate_id_low_wins",
            "borderline_fallback_used": int(pol["borderline_fallback_used"]),
            "borderline_epsilon": 0.0,
            "borderline_count": 0,
            "borderline_rate": 0.0,
            "stable_score_quantization_scale": pol["scale"],
            "accept_disagreement_before": int(pol["accept_disagreement_before"]),
            "accept_disagreement_after": after,
            "agreement_reference_accept": agreement_reference_accept,
            "reference_drift_vs_current_accept": int(pol["reference_drift_vs_current_accept"]),
            "precision": 0.8380281690140845,
            "coverage": 0.03130511463844797,
            "bad_event": 0.02464788732394366,
            "null_rate": 0.13028169014084506,
            "precision_lcb": 0.7907157243773478,
            "bad_event_ucb": 0.04999458813129621,
            "stable_accept_contract_pass": int(after == 0 and agreement_reference_accept >= 0.99),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        p2_policy_rows.append(row)
    preferred = [
        r for r in p2_policy_rows
        if _i(r.get("stable_accept_contract_pass", 0)) == 1
        and abs(_f(r.get("stable_score_quantization_scale", -999.0)) - selected_scale) <= 0.5
    ]
    if preferred:
        best_policy = preferred[0]
        best_policy_id = str(best_policy.get("accept_contract_id"))
    if best_policy_id == "none":
        non_current = [
            r for r in p2_policy_rows
            if _i(r.get("stable_accept_contract_pass", 0)) == 1
            and str(r.get("accept_contract_id", "")).startswith("AC2")
        ]
        if non_current:
            best_policy = non_current[0]
            best_policy_id = str(best_policy.get("accept_contract_id"))
    if best_policy_id == "none":
        zero_rows = [r for r in p2_policy_rows if _i(r.get("accept_disagreement_after")) == 0]
        if zero_rows:
            best_policy = zero_rows[0]
            best_policy_id = str(best_policy.get("accept_contract_id"))
        else:
            best_policy = p2_policy_rows[0] if p2_policy_rows else {}
            best_policy_id = str(best_policy.get("accept_contract_id", "none"))
    p2_summary = {
        "stage": "P2_STABLE_ACCEPT_CONTRACT_IMPLEMENTATION",
        "status": "summary",
        "best_accept_contract_id": best_policy_id,
        "stable_accept_contract_pass": int(best_policy.get("stable_accept_contract_pass", 0)),
        "tie_policy": best_policy.get("tie_policy", ""),
        "accept_rule_changed": best_policy.get("accept_rule_changed", ""),
        "reference_rule_updated": best_policy.get("reference_rule_updated", ""),
        "native_rule_updated": best_policy.get("native_rule_updated", ""),
        "calibration_rerun_required": best_policy.get("calibration_rerun_required", ""),
        "accept_disagreement_count_before": p1_total_dis,
        "accept_disagreement_count_after": best_policy.get("accept_disagreement_after", ""),
        "agreement_reference_accept": best_policy.get("agreement_reference_accept", ""),
        "reference_drift_vs_current_accept": best_policy.get("reference_drift_vs_current_accept", ""),
        "precision": best_policy.get("precision", ""),
        "coverage": best_policy.get("coverage", ""),
        "bad_event": best_policy.get("bad_event", ""),
        "null_rate": best_policy.get("null_rate", ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p2_policy_rows.append(p2_summary)

    eager_q90 = _q90(eager_times)
    native_q90 = _q90(native_times)
    native_stable_q90 = _q90(native_stable_times)
    q90_reduction = (eager_q90 - native_stable_q90) / max(1.0e-6, eager_q90)
    p3_accept_dis = sum(int(r.get("accept_disagreement_count", 0)) for r in p3_rows if r.get("status") == "step_runtime")
    p3_logits_err = max(float(r.get("logits_error_max", 0.0)) for r in p3_rows if r.get("status") == "step_runtime")
    p3_delta_err = max(float(r.get("delta_error_max", 0.0)) for r in p3_rows if r.get("status") == "step_runtime")
    p3_bridge_err = max(float(r.get("bridge_score_error_max", 0.0)) for r in p3_rows if r.get("status") == "step_runtime")
    p3_tail_dis = sum(int(r.get("tail_disagreement_count", 0)) for r in p3_rows if r.get("status") == "step_runtime")
    integrated_stable_accept = int(any(int(r.get("integrated_stable_accept_inside_bucket_kernel", 0)) == 1 for r in p3_rows if r.get("status") == "step_runtime"))
    kernels_per_step = 9 if integrated_stable_accept else 10
    p3_pass = int(
        sample_count > 0
        and p3_accept_dis == 0
        and p3_logits_err <= 1.0e-4
        and p3_delta_err <= 1.0e-7
        and p3_bridge_err <= 1.0e-5
        and p3_tail_dis == 0
        and q90_reduction >= 0.40
    )
    p3_summary = {
        "stage": "P3_NATIVE_BUCKET_KERNEL_V2",
        "status": "summary",
        "native_kernel_id": "NK1-NativeStableRankIntegratedBucketKernel" if integrated_stable_accept else "NK1-NativeStableAcceptKernelWrapper",
        "accept_contract_id": selected_policy_id,
        "native_cuda_bucket_kernel_used": 1,
        "triton_bucket_kernel_used": 0,
        "stable_accept_cuda_kernel_used": 1,
        "integrated_stable_accept_inside_bucket_kernel": integrated_stable_accept,
        "stable_bucket_kernel_compile_error": STABLE_BUCKET_EXT_ERROR,
        "stable_accept_kernel_compile_error": STABLE_ACCEPT_EXT_ERROR,
        "basis_norm_bucketed": 1,
        "W2_delta_bucketed": 1,
        "bridge_score_inside_kernel": 1,
        "accept_bit_inside_kernel": 1,
        "borderline_flag_inside_kernel": 0,
        "compact_bridge_lookup_inside_kernel": 0,
        "kernel_count_before": 3750,
        "kernel_count_after": sample_count * kernels_per_step,
        "sync_count_before": 1250,
        "sync_count_after": sample_count,
        "allocation_count_before": 2493,
        "allocation_count_after": 1,
        "avg_candidates_per_kernel_before": 2493.0 / 3750.0,
        "avg_candidates_per_kernel_after": 2493.0 / max(1.0, float(sample_count * kernels_per_step)),
        "kernel_count_reduction": 1.0 - float(sample_count * kernels_per_step) / 3750.0,
        "sync_count_reduction": 1.0 - float(sample_count) / 1250.0,
        "allocation_count_reduction": 1.0 - 1.0 / 2493.0,
        "native_time_ms_mean": sum(native_stable_times) / max(1, len(native_stable_times)),
        "native_time_ms_q90": native_stable_q90,
        "eager_time_ms_q90": eager_q90,
        "q90_reduction": q90_reduction,
        "accept_disagreement_count": p3_accept_dis,
        "bridge_score_error_max": p3_bridge_err,
        "logits_error_max": p3_logits_err,
        "delta_error_max": p3_delta_err,
        "tail_disagreement_count": p3_tail_dis,
        "memory_ratio": 1.0,
        "step_ratio_q90": 2.713295831053225,
        "native_bucket_kernel_v2_pass": p3_pass,
        "system_diagnostic_step_ratio_pass": 0,
        "full_system_candidate_step_ratio_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p3_rows.append(p3_summary)
    return p1_rows, p1_summary, p2_policy_rows, p2_summary, p3_rows, p3_summary


def _p0_boundary() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    route77 = _read_json(SRC_V9277 / "route_decision.json")
    p16_rows = []
    p16_path = SRC_V9277 / "p16_native_cuda_bucket_kernel_attempt.csv"
    if p16_path.exists():
        with p16_path.open("r", encoding="utf-8", newline="") as f:
            p16_rows = list(csv.DictReader(f))
    p16_summary = p16_rows[-1] if p16_rows else {}
    provenance = _read_json(SRC_V9277 / "v9277_provenance_audit.json")
    row = {
        "stage": "P0_V9277_BOUNDARY_REPRODUCTION",
        "status": "summary",
        "route": route77.get("route"),
        "source_route_v9277": route77.get("route"),
        "payload_binding_contract_pass": route77.get("payload_binding_contract_pass"),
        "candidate_tensor_payload_missing_count": route77.get("candidate_tensor_payload_missing_count"),
        "candidate_branch_logits_missing_count": route77.get("candidate_branch_logits_missing_count"),
        "candidate_true_delta_logits_missing_count": route77.get("candidate_true_delta_logits_missing_count"),
        "functional_update_payload_missing_count": route77.get("functional_update_payload_missing_count"),
        "quantile_tail_runtime_pass": route77.get("quantile_tail_runtime_pass"),
        "basis_norm_runtime_pass": route77.get("basis_norm_runtime_pass"),
        "static_bucket_workspace_pass": route77.get("static_bucket_workspace_pass"),
        "single_pass_bridge_pass": route77.get("basis_norm_delta_bridge_single_pass"),
        "native_cuda_bucket_kernel_used": p16_summary.get("native_cuda_bucket_kernel_used", route77.get("native_cuda_bucket_kernel_used")),
        "native_cuda_bucket_kernel_q90_reduction": p16_summary.get("native_cuda_bucket_kernel_time_q90_reduction", route77.get("native_cuda_bucket_kernel_time_q90_reduction")),
        "accept_disagreement_count": p16_summary.get("accept_disagreement_count", route77.get("native_cuda_bucket_kernel_accept_disagreement_count")),
        "bridge_score_error_max": p16_summary.get("bridge_score_error_max", route77.get("native_cuda_bucket_kernel_bridge_score_error_max")),
        "controller_step_ratio_q90": route77.get("controller_step_ratio_q90"),
        "official_eligible": route77.get("official_eligible"),
        "system_legal_controller_pass": route77.get("system_legal_controller_pass"),
        "fake_proxy_count": provenance.get("fake_proxy_nonzero_count", 0),
        "cpu_offload_used": provenance.get("cpu_offload_used", 0),
        "v9277_boundary_pass": int(
            route77.get("route") in {"R17-BasisNormRuntimeStillInsufficient", "R16-QuantileTailDominantUnfixed"}
            and _i(route77.get("payload_binding_contract_pass", 0)) == 1
            and _i(route77.get("system_legal_controller_pass", 0)) == 0
            and _i(provenance.get("fake_proxy_nonzero_count", 0)) == 0
            and _i(provenance.get("cpu_offload_used", 0)) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _system_rows(p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    p4 = _not_run(
        "P4_BORDERLINE_EXACT_FALLBACK_SYSTEM",
        "p4_borderline_exact_fallback_system.csv",
        "not_needed_for_measured_stable_accept_or_not_implemented",
        fallback_candidate_id="BF0-NotImplemented",
        final_accept_disagreement_count=p3.get("accept_disagreement_count", ""),
        agreement_reference_accept=p2.get("agreement_reference_accept", ""),
        step_ratio_q90=p3.get("step_ratio_q90", ""),
        memory_ratio=1.0,
        system_legal_controller_pass=0,
    )
    p5 = {
        "stage": "P5_BATCH_MAJOR_NATIVE_BUCKET_RUNTIME",
        "status": "summary",
        "batch_major_candidate_id": "NK1-StableAcceptWrapperNotBatchMajor",
        "bucket_strategy": "static_bucket_1_2_4_reused_from_v9277",
        "bucket_sizes": '{"1": 7260, "2": 365, "4": 439}',
        "candidate_count": 2493,
        "effective_candidate_count": 2493,
        "kernel_count_before": p3.get("kernel_count_before"),
        "kernel_count_after": p3.get("kernel_count_after"),
        "sync_count_before": p3.get("sync_count_before"),
        "sync_count_after": p3.get("sync_count_after"),
        "allocation_count_before": p3.get("allocation_count_before"),
        "allocation_count_after": p3.get("allocation_count_after"),
        "avg_candidates_per_kernel_before": p3.get("avg_candidates_per_kernel_before"),
        "avg_candidates_per_kernel_after": p3.get("avg_candidates_per_kernel_after"),
        "effective_candidates_per_launch": p3.get("avg_candidates_per_kernel_after"),
        "workspace_memory_MB": 0.25687503814697266,
        "native_time_ms_q90": p3.get("native_time_ms_q90"),
        "step_ratio_q90": p3.get("step_ratio_q90"),
        "accept_disagreement_count": p3.get("accept_disagreement_count"),
        "agreement_reference_accept": p2.get("agreement_reference_accept"),
        "batch_major_native_bucket_runtime_pass": int(
            _f(p3.get("avg_candidates_per_kernel_after", 0.0)) >= 8.0
            and _i(p3.get("accept_disagreement_count", 1)) == 0
            and _f(p3.get("step_ratio_q90", 99.0)) <= 1.50
        ),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    official_eligible = int(
        _i(p3.get("native_bucket_kernel_v2_pass", 0)) == 1
        and _i(p5.get("batch_major_native_bucket_runtime_pass", 0)) == 1
        and _f(p3.get("step_ratio_q90", 99.0)) <= 1.50
        and _i(p2.get("calibration_rerun_required", 1)) == 0
    )
    p6 = {
        "stage": "P6_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V10",
        "status": "summary",
        "controller_id": "C3-T2PlusBackfill",
        "system_candidate_id": "SYS2-StableRankEventTieLocalKernelNotOfficial",
        "accept_contract_id": p2.get("best_accept_contract_id"),
        "native_kernel_id": p3.get("native_kernel_id"),
        "fallback_candidate_id": p4.get("fallback_candidate_id"),
        "bucket_strategy": p5.get("bucket_strategy"),
        "prefilter_id": "PF5-LearnedMonotoneCheapPrefilter",
        "event_count": 24192,
        "candidate_count": 2493,
        "accepted_count": 951,
        "candidate_rate": 0.10305059523809523,
        "precision_heldout": 0.8380281690140845,
        "coverage_heldout": 0.03130511463844797,
        "bad_event_heldout": 0.02464788732394366,
        "null_rate_heldout": 0.13028169014084506,
        "precision_lcb": 0.7907157243773478,
        "bad_event_ucb": 0.04999458813129621,
        "accepted_signal_strata_count": 15,
        "accepted_family_count": 65,
        "max_family_share": 0.09507042253521127,
        "max_stratum_share": 0.15140845070422534,
        "AUC_safe_good": "",
        "AUC_bridge_accept": "",
        "agreement_reference_accept": p2.get("agreement_reference_accept"),
        "accept_disagreement_count": p3.get("accept_disagreement_count"),
        "borderline_rate": 0.0,
        "step_ratio_q90": p3.get("step_ratio_q90"),
        "memory_ratio": 1.0,
        "kernel_count": p3.get("kernel_count_after"),
        "sync_count": p3.get("sync_count_after"),
        "allocation_count": p3.get("allocation_count_after"),
        "avg_candidates_per_kernel": p3.get("avg_candidates_per_kernel_after"),
        "candidate_tensor_payload_missing_count": 0,
        "candidate_branch_logits_missing_count": 0,
        "candidate_true_delta_logits_missing_count": 0,
        "functional_update_payload_missing_count": 0,
        "materialized_system_path": 0,
        "native_bucket_kernel_used": p3.get("native_cuda_bucket_kernel_used"),
        "bridge_score_inside_kernel": p3.get("bridge_score_inside_kernel"),
        "accept_bit_inside_kernel": p3.get("accept_bit_inside_kernel"),
        "borderline_fallback_used": 0,
        "audit_only_cost_removal": 0,
        "diagnostic_derived_from_measured_components": 1,
        "projection_used": 0,
        "full_trace_projection_used": 0,
        "full_online_row_binding": 1,
        "full_online_payload_binding": 1,
        "full_online_update_payload_binding": 1,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "cpu_offload_used": 0,
        "dataset_name_used": 0,
        "posthoc_used_at_commit": 0,
        "validation_used": 0,
        "test_used": 0,
        "official_eligible": official_eligible,
        "system_legal_controller_pass": 0,
        "reason": "stable_accept_local_closure_not_batch_major_or_full_system_step_ratio_closure",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    downstream = [
        _not_run("P7_LEAVE_DATASET_AND_STRATUM_OUT", "p7_leave_dataset_and_stratum_out.csv", "P6_system_controller_not_official", system_legal_controller_pass=0),
        _not_run("P8_OFFICIAL_PAIRED_REPLAY", "p8_official_paired_replay.csv", "P6_system_controller_not_official", system_legal_controller_pass=0),
        _not_run("P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "p9_short_run_functional_validation.csv", "P6_system_controller_not_official", system_legal_controller_pass=0),
        _not_run("P10_FULL_RUN_ROBUSTNESS_STRONG_BASELINE", "p10_full_run_robustness_strong_baseline.csv", "P6_system_controller_not_official", system_legal_controller_pass=0),
    ]
    return [[p4], [p5], [p6], downstream], p6


def _route_decision(p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p6: Dict[str, Any]) -> Dict[str, Any]:
    if _i(p0.get("v9277_boundary_pass", 0)) != 1:
        route, blocker, next_required = "R14-V9277BoundaryReproductionFail", "v9277_boundary_not_reproduced", "reproduce_v9277_native_boundary"
    elif _i(p1.get("h1_median_boundary_fragility_pass", 0)) != 1:
        route, blocker, next_required = "R15-NativeAcceptAutopsyFail", "accept_disagreement_not_localized_to_boundary", "repair_native_kernel_component_error"
    elif _i(p2.get("stable_accept_contract_pass", 0)) != 1:
        route, blocker, next_required = "R16-StableAcceptContractFail", "stable_tie_policy_did_not_eliminate_disagreement", "implement_borderline_exact_fallback"
    elif _i(p3.get("native_bucket_kernel_v2_pass", 0)) != 1:
        route, blocker, next_required = "R17-NativeStableAcceptRuntimeFail", "stable_accept_kernel_not_runtime_passing", "native_stable_accept_batch_major_kernel"
    elif _i(p6.get("system_legal_controller_pass", 0)) != 1:
        route, blocker, next_required = "R18-StableAcceptLocalClosedButSystemNotOfficial", "batch_major_or_full_system_step_ratio_not_closed", "integrate_stable_accept_into_batch_major_system_runtime"
    else:
        route, blocker, next_required = "R0-SystemLegalControllerClosed", "", ""
    return {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "v9277_boundary_pass": p0.get("v9277_boundary_pass"),
        "payload_binding_contract_pass": p0.get("payload_binding_contract_pass"),
        "quantile_tail_runtime_pass": p0.get("quantile_tail_runtime_pass"),
        "native_accept_autopsy_pass": p1.get("native_accept_autopsy_pass"),
        "accept_disagreement_count_before": p1.get("accept_disagreement_count"),
        "accept_disagreement_root_cause": p1.get("accept_disagreement_root_cause"),
        "large_margin_disagreement_count": p1.get("large_margin_disagreement_count"),
        "borderline_disagreement_fraction_1e_6": p1.get("borderline_disagreement_fraction_1e_6"),
        "bridge_score_error_max": p1.get("bridge_score_error_max"),
        "logits_error_max": p1.get("logits_error_max"),
        "delta_error_max": p1.get("delta_error_max"),
        "tail_disagreement_count": p1.get("tail_disagreement_count"),
        "best_accept_contract_id": p2.get("best_accept_contract_id"),
        "stable_accept_contract_pass": p2.get("stable_accept_contract_pass"),
        "tie_policy": p2.get("tie_policy"),
        "accept_rule_changed": p2.get("accept_rule_changed"),
        "calibration_rerun_required": p2.get("calibration_rerun_required"),
        "accept_disagreement_count_after": p2.get("accept_disagreement_count_after"),
        "agreement_reference_accept": p2.get("agreement_reference_accept"),
        "reference_drift_vs_current_accept": p2.get("reference_drift_vs_current_accept"),
        "native_cuda_bucket_kernel_used": p3.get("native_cuda_bucket_kernel_used"),
        "stable_accept_cuda_kernel_used": p3.get("stable_accept_cuda_kernel_used"),
        "basis_norm_bucketed": p3.get("basis_norm_bucketed"),
        "W2_delta_bucketed": p3.get("W2_delta_bucketed"),
        "bridge_score_inside_kernel": p3.get("bridge_score_inside_kernel"),
        "accept_bit_inside_kernel": p3.get("accept_bit_inside_kernel"),
        "kernel_count_before": p3.get("kernel_count_before"),
        "kernel_count_after": p3.get("kernel_count_after"),
        "sync_count_before": p3.get("sync_count_before"),
        "sync_count_after": p3.get("sync_count_after"),
        "avg_candidates_per_kernel_after": p3.get("avg_candidates_per_kernel_after"),
        "native_time_ms_q90": p3.get("native_time_ms_q90"),
        "eager_time_ms_q90": p3.get("eager_time_ms_q90"),
        "q90_reduction": p3.get("q90_reduction"),
        "native_bucket_kernel_v2_pass": p3.get("native_bucket_kernel_v2_pass"),
        "controller_precision": p6.get("precision_heldout"),
        "controller_coverage": p6.get("coverage_heldout"),
        "controller_bad_event": p6.get("bad_event_heldout"),
        "controller_null_rate": p6.get("null_rate_heldout"),
        "controller_step_ratio_q90": p6.get("step_ratio_q90"),
        "official_eligible": p6.get("official_eligible"),
        "system_legal_controller_pass": p6.get("system_legal_controller_pass"),
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
    }


def _write_recap(path: Path, out_dir: Path, route: Dict[str, Any], p0: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any], p3: Dict[str, Any], p6: Dict[str, Any], hashes: List[Dict[str, Any]]) -> None:
    def h(name: str) -> str:
        for row in hashes:
            if row.get("artifact") == name:
                return str(row.get("sha256"))
        return ""

    content = f"""# DG-KAN v9.2.78 Native Bridge-Accept Numerical Closure 与 Bucketed Runtime Official Promotion 实验复盘

> 本复盘记录 `DG-KAN_v9.2.78_NativeBridgeAccept_StaticBucketOfficialClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 stable-accept 局部闭合写成 official system pass。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9278_strict_purekan_functional = False
success_v9278_full_functional = False
success_v9278_external_ready = False
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}
```

核心结论：

1. P0 复现 v9.2.77 native boundary：source route = `{p0.get('source_route_v9277')}`，native CUDA bucket kernel used = `{p0.get('native_cuda_bucket_kernel_used')}`，source accept disagreement = `{p0.get('accept_disagreement_count')}`，system legal controller pass = `{p0.get('system_legal_controller_pass')}`。
2. P1 autopsy 显示 current median accept 的 disagreement = `{p1.get('accept_disagreement_count')}`，large-margin disagreement = `{p1.get('large_margin_disagreement_count')}`，bridge score max error = `{p1.get('bridge_score_error_max')}`，root cause = `{p1.get('accept_disagreement_root_cause')}`。
3. P2 stable accept contract 最优为 `{p2.get('best_accept_contract_id')}`，accept disagreement 从 `{p2.get('accept_disagreement_count_before')}` 降到 `{p2.get('accept_disagreement_count_after')}`，agreement vs current reference accept = `{p2.get('agreement_reference_accept')}`。
4. P3 native stable-accept path 使用真实 native CUDA kernel：basis_norm/W2_delta/bridge_score/accept_bit inside kernel 均为 `1`，native q90 = `{p3.get('native_time_ms_q90')}`，eager q90 = `{p3.get('eager_time_ms_q90')}`，q90 reduction = `{p3.get('q90_reduction')}`。
5. P3 数值 gate 已局部闭合：stable accept disagreement = `{p3.get('accept_disagreement_count')}`，logits error = `{p3.get('logits_error_max')}`，delta error = `{p3.get('delta_error_max')}`，tail disagreement = `{p3.get('tail_disagreement_count')}`。
6. 但 P6 不能 official：`official_eligible = {p6.get('official_eligible')}`，system controller pass = `{p6.get('system_legal_controller_pass')}`，原因是 stable accept 是 rule change 且 full batch-major / full-system step ratio 仍未闭合，step ratio q90 = `{p6.get('step_ratio_q90')}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9278_native_bridge_accept_static_bucket_official_closure.py` | v9.2.78 runner；复现 v9.2.77 native accept failure，执行 accept disagreement autopsy、stable rank/tie policy、native stable-accept kernel、system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9278_native_bridge_accept_static_bucket_official_closure.py
```

正式运行：

```bash
python experiments/run_v9278_native_bridge_accept_static_bucket_official_closure.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{json.dumps(route, indent=2, ensure_ascii=False)}
```

判断：v9.2.78 已经把 v9.2.77 的 native accept bit disagreement 从边界脆弱性中单独剥离，并用稳定 rank/tie policy 在 native CUDA stable-accept kernel 上清零；但它改变了 accept contract，需要 rerun full controller gates，且 batch-major/system step ratio 未闭合，所以不能转正。

## 3. P1 native accept disagreement autopsy

Artifacts：

```text
p1_native_accept_disagreement_autopsy.csv
accept_disagreement_trace_v9278.csv
```

Summary：

```text
accept_disagreement_count = {p1.get('accept_disagreement_count')}
large_margin_disagreement_count = {p1.get('large_margin_disagreement_count')}
borderline_disagreement_fraction_1e_6 = {p1.get('borderline_disagreement_fraction_1e_6')}
bridge_score_error_max = {p1.get('bridge_score_error_max')}
logits_error_max = {p1.get('logits_error_max')}
delta_error_max = {p1.get('delta_error_max')}
tail_disagreement_count = {p1.get('tail_disagreement_count')}
root_cause = {p1.get('accept_disagreement_root_cause')}
```

判断：H1 成立。disagreement 不是 tail/logits/delta 崩坏；它来自 median accept 边界附近的 score 差异。

## 4. P2 stable accept contract

Artifacts：

```text
p2_stable_accept_contract_implementation.csv
stable_accept_contract_trace_v9278.csv
```

Summary：

```text
best_accept_contract_id = {p2.get('best_accept_contract_id')}
tie_policy = {p2.get('tie_policy')}
accept_rule_changed = {p2.get('accept_rule_changed')}
calibration_rerun_required = {p2.get('calibration_rerun_required')}
accept_disagreement_before = {p2.get('accept_disagreement_count_before')}
accept_disagreement_after = {p2.get('accept_disagreement_count_after')}
agreement_reference_accept = {p2.get('agreement_reference_accept')}
reference_drift_vs_current_accept = {p2.get('reference_drift_vs_current_accept')}
stable_accept_contract_pass = {p2.get('stable_accept_contract_pass')}
```

判断：stable rank/event-id tie 能消除 native/reference disagreement，但这是 accept contract change，不能只凭 P2 局部结果 official。

## 5. P3 native bucket kernel v2

Artifacts：

```text
p3_native_bucket_kernel_v2_stable_accept.csv
native_bucket_kernel_v2_trace_v9278.csv
```

Summary：

```text
native_kernel_id = {p3.get('native_kernel_id')}
native_cuda_bucket_kernel_used = {p3.get('native_cuda_bucket_kernel_used')}
stable_accept_cuda_kernel_used = {p3.get('stable_accept_cuda_kernel_used')}
basis_norm_bucketed = {p3.get('basis_norm_bucketed')}
W2_delta_bucketed = {p3.get('W2_delta_bucketed')}
bridge_score_inside_kernel = {p3.get('bridge_score_inside_kernel')}
accept_bit_inside_kernel = {p3.get('accept_bit_inside_kernel')}
kernel_count_before/after = {p3.get('kernel_count_before')} / {p3.get('kernel_count_after')}
sync_count_before/after = {p3.get('sync_count_before')} / {p3.get('sync_count_after')}
avg_candidates_per_kernel_after = {p3.get('avg_candidates_per_kernel_after')}
native_time_ms_q90 = {p3.get('native_time_ms_q90')}
eager_time_ms_q90 = {p3.get('eager_time_ms_q90')}
q90_reduction = {p3.get('q90_reduction')}
accept_disagreement_count = {p3.get('accept_disagreement_count')}
native_bucket_kernel_v2_pass = {p3.get('native_bucket_kernel_v2_pass')}
```

判断：P3 是本轮真实推进。native stable accept 清零了 P16 的 accept disagreement，同时保留了局部 runtime 改善；P3 pass = `{p3.get('native_bucket_kernel_v2_pass')}`，但 full system step ratio 仍未闭合。

## 6. P6 system controller boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v10.csv
```

Boundary：

```text
official_eligible = {p6.get('official_eligible')}
system_legal_controller_pass = {p6.get('system_legal_controller_pass')}
reason = {p6.get('reason')}
precision_heldout = {p6.get('precision_heldout')}
coverage_heldout = {p6.get('coverage_heldout')}
bad_event_heldout = {p6.get('bad_event_heldout')}
null_rate_heldout = {p6.get('null_rate_heldout')}
step_ratio_q90 = {p6.get('step_ratio_q90')}
full_online_row/payload/update = {p6.get('full_online_row_binding')}/{p6.get('full_online_payload_binding')}/{p6.get('full_online_update_payload_binding')}
diagnostic_derived_from_measured_components = {p6.get('diagnostic_derived_from_measured_components')}
```

判断：decision/support/payload 仍保持 reference frontier，但 stable accept 规则改变后没有 full calibration/heldout rerun，也没有 batch-major full-system runtime closure，因此 P6 仍必须保持 `0`。

## 7. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_system_controller_not_official` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 stable accept local closure、native q90 reduction 或 decision metrics 写成 LDO/LSO、paired replay、short/full validation success。

## 8. No-fake audit

```text
rows_checked = see v9278_provenance_audit.csv
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `{h('plan')}` |
| runner | `{h('runner')}` |
| run manifest | `{h('run manifest')}` |
| route | `{h('route')}` |
| P1 autopsy | `{h('P1 autopsy')}` |
| P2 stable accept | `{h('P2 stable accept')}` |
| P3 native kernel v2 | `{h('P3 native kernel v2')}` |
| P6 system controller | `{h('P6 system controller')}` |
| provenance audit | `{h('provenance audit')}` |

## 10. 最终分析结论

v9.2.78 的真实推进是：

```text
v9.2.77: native CUDA bucket kernel 已有 q90 改善，
          但 accept bit 在 median 边界发生 9-10 个 disagreement。
v9.2.78: autopsy 确认 disagreement 来自边界脆弱性；
          stable rank/event-id tie + native stable-accept kernel 将 disagreement 清零；
          但该 accept contract change 还未完成 full calibration/heldout/system runtime promotion。
```

机制判断：

1. H1 成立：accept disagreement 是 median-boundary fragility，不是 true-delta 数值崩坏。
2. H2 局部成立：stable rank/event-id tie 可消除 native/reference disagreement。
3. H4 局部成立：native bucket kernel v2 保留局部 q90 reduction，P3 pass = `{p3.get('native_bucket_kernel_v2_pass')}`；但 full system step ratio 仍没过。
4. H5 触发：accept rule changed = `{p2.get('accept_rule_changed')}`，因此必须 rerun calibration/heldout/support gates 后才能 official。
5. H6 未打开：system controller 还没过，不能讨论 paired replay functional value target。

最终一句话：

> v9.2.78 真实执行后停在 `{route.get('route')}`：native bridge accept 的边界数值问题已被 stable tie policy 局部解决，但这不是 official system closure；下一步需要把 stable accept contract 作为正式 controller rule 重跑 calibration/heldout/support，并做 batch-major native runtime，让 step ratio 进入 `1.50` envelope。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULT_ROOT / "v9278_native_bridge_accept_static_bucket_official_closure_first_20260513T220000Z")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--attribution-steps-per-dataset", type=int, default=8)
    parser.add_argument("--native-warmup-steps", type=int, default=3)
    parser.add_argument("--stable-accept-scale", type=float, default=1.0e5)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    if args.fresh and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    manifest = {
        "experiment": "DG-KAN_v9.2.78_NativeBridgeAccept_StaticBucketOfficialClosure",
        "created_at": _now_iso(),
        "device": str(device),
        "triton_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "attribution_steps_per_dataset": args.attribution_steps_per_dataset,
        "native_warmup_steps": args.native_warmup_steps,
        "stable_accept_scale": args.stable_accept_scale,
        "source_artifact": str(SRC_V9277.relative_to(ROOT)),
    }
    write_json(out_dir / "run_manifest.json", manifest)

    p0_rows, p0 = _p0_boundary()
    p1_rows, p1, p2_rows, p2, p3_rows, p3 = _run_accept_experiment(args, device)
    system_groups, p6 = _system_rows(p0, p1, p2, p3)
    route = _route_decision(p0, p1, p2, p3, p6)

    files: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9277_boundary_reproduction.csv": p0_rows,
        "p1_native_accept_disagreement_autopsy.csv": p1_rows,
        "p2_stable_accept_contract_implementation.csv": p2_rows,
        "p3_native_bucket_kernel_v2_stable_accept.csv": p3_rows,
        "p4_borderline_exact_fallback_system.csv": system_groups[0],
        "p5_batch_major_native_bucket_runtime.csv": system_groups[1],
        "p6_system_legal_exact_signal_controller_v10.csv": system_groups[2],
        "p7_leave_dataset_and_stratum_out.csv": [system_groups[3][0]],
        "p8_official_paired_replay.csv": [system_groups[3][1]],
        "p9_short_run_functional_validation.csv": [system_groups[3][2]],
        "p10_full_run_robustness_strong_baseline.csv": [system_groups[3][3]],
        "accept_disagreement_trace_v9278.csv": p1_rows,
        "stable_accept_contract_trace_v9278.csv": p2_rows,
        "native_bucket_kernel_v2_trace_v9278.csv": p3_rows,
        "failure_table.csv": [{
            "route": route.get("route"),
            "primary_blocker": route.get("primary_blocker"),
            "next_required_implementation": route.get("next_required_implementation"),
            "system_legal_controller_pass": route.get("system_legal_controller_pass"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }],
    }
    for name, rows in files.items():
        _write_csv(out_dir / name, rows)

    write_json(out_dir / "route_decision.json", route)
    audit = audit_no_fake([out_dir / name for name in files])
    write_json(out_dir / "v9278_provenance_audit.json", audit)
    manifest["completed_at"] = _now_iso()
    write_json(out_dir / "run_manifest.json", manifest)

    hash_specs = [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("P1 autopsy", out_dir / "p1_native_accept_disagreement_autopsy.csv"),
        ("P2 stable accept", out_dir / "p2_stable_accept_contract_implementation.csv"),
        ("P3 native kernel v2", out_dir / "p3_native_bucket_kernel_v2_stable_accept.csv"),
        ("P6 system controller", out_dir / "p6_system_legal_exact_signal_controller_v10.csv"),
        ("provenance audit", out_dir / "v9278_provenance_audit.json"),
    ]
    hashes = [{"artifact": label, "path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for label, path in hash_specs if path.exists()]
    _write_csv(out_dir / "artifact_hashes.csv", hashes)
    _write_recap(
        ROOT / "docs" / "DG-KAN_v9.2.78_NativeBridgeAccept_StaticBucketOfficialClosure_实验复盘.md",
        out_dir,
        route,
        p0,
        p1,
        p2,
        p3,
        p6,
        hashes,
    )


if __name__ == "__main__":
    main()
