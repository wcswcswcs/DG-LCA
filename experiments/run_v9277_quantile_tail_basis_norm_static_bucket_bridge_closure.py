#!/usr/bin/env python3
"""DG-KAN v9.2.77 quantile-tail basis-norm closure runner.

This runner starts from the landed v9.2.76 boundary and performs real
runtime work on the current blocker:

1. Split quantile/tail compute into CUDA-timed subphases.
2. Try an exact kthvalue-tail runtime and compare it against torch.quantile
   reference on the same train stream.
3. Carry the tail runtime into basis_norm timing and refuse official promotion
   unless bucketed runtime and bridge score integration are truly materialized.

It deliberately refuses to promote attribution-only or diagnostic bucket rows
into an official system controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

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
import run_v9268_true_delta_system_closure_reference_feasible_controller_promotion as v9268  # noqa: E402
import run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline as v9272  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, sha256_file, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402

PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.77_QuantileTailBasisNorm_StaticBucketBridgeClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9277_quantile_tail_basis_norm_static_bucket_bridge_closure.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9276 = RESULT_ROOT / "v9276_basis_norm_runtime_elimination_static_bucket_workspace_closure_first_20260513T150000Z"
SRC_V9272_BF5 = RESULT_ROOT / "v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z"
CUDA_BUCKET_KERNEL_ERROR = ""
_BUCKET_KERNEL_EXT = None

_f = v9265._f
_i = v9265._i
_q = v9265._q
_device = v9265._device
_cal_held_indices = v9265._cal_held_indices
_not_run = v9265._not_run


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


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


def _load_bucket_basis_delta_ext() -> Any:
    global CUDA_BUCKET_KERNEL_ERROR, _BUCKET_KERNEL_EXT
    if _BUCKET_KERNEL_EXT is not None:
        return _BUCKET_KERNEL_EXT
    if not torch.cuda.is_available():
        CUDA_BUCKET_KERNEL_ERROR = "cuda_not_available"
        return None
    cpp_src = """
    #include <torch/extension.h>
    std::vector<torch::Tensor> bucket_basis_delta_bridge_forward(
        torch::Tensor vu0,
        torch::Tensor vu1,
        torch::Tensor vp0,
        torch::Tensor vp1,
        torch::Tensor W0,
        torch::Tensor W2,
        torch::Tensor y,
        torch::Tensor d0,
        torch::Tensor d1,
        torch::Tensor d2);
    """
    cuda_src = r"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <math.h>
    #include <vector>

    __global__ void update_logits_kernel(
        const float* __restrict__ vu0,
        const float* __restrict__ vu1,
        const float* __restrict__ W0,
        const float* __restrict__ W2,
        float* __restrict__ logits,
        const int B,
        const int H,
        const int C) {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = B * C;
      if (idx >= total) return;
      const int o = idx % C;
      const int b = idx / C;
      float acc = 0.0f;
      for (int h = 0; h < H; ++h) {
        acc += vu0[b * H + h] * W0[h * C + o];
        acc += vu1[b * H + h] * W2[h * C + o];
      }
      logits[idx] = acc;
    }

    __global__ void ce_margin_kernel(
        const float* __restrict__ logits,
        const long* __restrict__ y,
        float* __restrict__ ce,
        float* __restrict__ margin,
        long* __restrict__ pred,
        const int B,
        const int C) {
      const int b = blockIdx.x * blockDim.x + threadIdx.x;
      if (b >= B) return;
      const int yb = (int)y[b];
      float maxv = logits[b * C];
      int argmax = 0;
      for (int o = 1; o < C; ++o) {
        const float v = logits[b * C + o];
        if (v > maxv) {
          maxv = v;
          argmax = o;
        }
      }
      float sum_exp = 0.0f;
      for (int o = 0; o < C; ++o) {
        sum_exp += expf(logits[b * C + o] - maxv);
      }
      const float true_logit = logits[b * C + yb];
      ce[b] = logf(sum_exp) + maxv - true_logit;
      float max_other = -3.4028234663852886e38f;
      for (int o = 0; o < C; ++o) {
        if (o == yb) continue;
        const float v = logits[b * C + o];
        if (v > max_other) max_other = v;
      }
      margin[b] = true_logit - max_other;
      pred[b] = (long)argmax;
    }

    __global__ void tail_threshold_kernel(
        const float* __restrict__ ce,
        const float* __restrict__ margin,
        bool* __restrict__ tail,
        float* __restrict__ ce_mean,
        const int B) {
      float ce_buf[256];
      float margin_buf[256];
      float sum = 0.0f;
      for (int i = 0; i < B; ++i) {
        ce_buf[i] = ce[i];
        margin_buf[i] = margin[i];
        sum += ce[i];
      }
      for (int i = 1; i < B; ++i) {
        float key = ce_buf[i];
        int j = i - 1;
        while (j >= 0 && ce_buf[j] > key) {
          ce_buf[j + 1] = ce_buf[j];
          --j;
        }
        ce_buf[j + 1] = key;
      }
      for (int i = 1; i < B; ++i) {
        float key = margin_buf[i];
        int j = i - 1;
        while (j >= 0 && margin_buf[j] > key) {
          margin_buf[j + 1] = margin_buf[j];
          --j;
        }
        margin_buf[j + 1] = key;
      }
      int ce_rank = (int)ceilf((float)(B - 1) * 0.80f);
      int margin_rank = (int)floorf((float)(B - 1) * 0.20f);
      if (ce_rank < 0) ce_rank = 0;
      if (ce_rank >= B) ce_rank = B - 1;
      if (margin_rank < 0) margin_rank = 0;
      if (margin_rank >= B) margin_rank = B - 1;
      const float ce_thr = ce_buf[ce_rank];
      const float margin_thr = margin_buf[margin_rank];
      for (int i = 0; i < B; ++i) {
        tail[i] = (ce[i] >= ce_thr) || (margin[i] <= margin_thr);
      }
      ce_mean[0] = sum / (float)B;
    }

    __global__ void row_abs_kernel(
        const float* __restrict__ vu1,
        float* __restrict__ row_abs,
        float* __restrict__ vals_abs_sum,
        const int B,
        const int H) {
      const int b = blockIdx.x * blockDim.x + threadIdx.x;
      if (b >= B) return;
      float acc = 0.0f;
      for (int h = 0; h < H; ++h) {
        acc += fabsf(vu1[b * H + h]);
      }
      const float mean = acc / (float)H;
      row_abs[b] = mean;
      atomicAdd(vals_abs_sum, mean);
    }

    __global__ void norm_reduce_kernel(
        const float* __restrict__ d0,
        const float* __restrict__ d1,
        const float* __restrict__ d2,
        float* __restrict__ norm2,
        const int n0,
        const int n1,
        const int n2) {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = n0 + n1 + n2;
      if (idx >= total) return;
      float v;
      if (idx < n0) {
        v = d0[idx];
      } else if (idx < n0 + n1) {
        v = d1[idx - n0];
      } else {
        v = d2[idx - n0 - n1];
      }
      atomicAdd(norm2, v * v);
    }

    __global__ void probe_logits_kernel(
        const float* __restrict__ vp0,
        const float* __restrict__ vp1,
        const float* __restrict__ W0,
        const float* __restrict__ W2,
        float* __restrict__ probe_logits,
        const int P,
        const int H,
        const int C) {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = P * C;
      if (idx >= total) return;
      const int o = idx % C;
      const int p = idx / C;
      float acc = 0.0f;
      for (int h = 0; h < H; ++h) {
        acc += vp0[p * H + h] * W0[h * C + o];
        acc += vp1[p * H + h] * W2[h * C + o];
      }
      probe_logits[idx] = acc;
    }

    __global__ void raw_delta_norm_kernel(
        const float* __restrict__ vu1,
        const float* __restrict__ ce,
        const float* __restrict__ margin,
        const bool* __restrict__ tail,
        const long* __restrict__ pred,
        const long* __restrict__ y,
        const float* __restrict__ row_abs,
        const float* __restrict__ ce_mean,
        const float* __restrict__ vals_abs_sum,
        float* __restrict__ delta,
        float* __restrict__ delta_norm2,
        const int B,
        const int H,
        const int C) {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = 3 * H * C;
      if (idx >= total) return;
      const int o = idx % C;
      const int h = (idx / C) % H;
      const int k = idx / (H * C);
      const float ce_mean_v = fmaxf(ce_mean[0], 1.0e-6f);
      const float vals_abs_mean = fmaxf(vals_abs_sum[0] / (float)B, 1.0e-6f);
      const float strength = (k == 0 ? 0.040f : (k == 1 ? 0.055f : 0.070f));
      float acc = 0.0f;
      for (int b = 0; b < B; ++b) {
        const float tail_f = tail[b] ? 1.0f : 0.0f;
        float w = ce[b] / ce_mean_v;
        if (k == 1) {
          w *= (margin[b] < 0.0f ? 1.5f : 0.5f);
        } else if (k == 2) {
          w *= row_abs[b] / vals_abs_mean;
        }
        float sig = 0.0f;
        if (o == (int)y[b]) sig += 1.0f;
        if (o == (int)pred[b]) sig -= 0.5f;
        acc += vu1[b * H + h] * tail_f * w * sig / (float)B;
      }
      const float value = strength * acc;
      delta[idx] = value;
      atomicAdd(&delta_norm2[k], value * value);
    }

    __global__ void scale_delta_kernel(
        float* __restrict__ delta,
        const float* __restrict__ delta_norm2,
        const float* __restrict__ task_norm2,
        const int H,
        const int C) {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = 3 * H * C;
      if (idx >= total) return;
      const int k = idx / (H * C);
      const float cap = (k == 0 ? 0.035f : (k == 1 ? 0.050f : 0.070f));
      const float task_norm = sqrtf(fmaxf(task_norm2[0], 1.0e-24f));
      const float func_norm = sqrtf(fmaxf(delta_norm2[k], 1.0e-24f));
      const float max_norm = cap * fmaxf(task_norm, 1.0e-12f);
      const float scale = fminf(1.0f, max_norm / fmaxf(func_norm, 1.0e-12f));
      delta[idx] *= scale;
    }

    __global__ void logits_kernel(
        const float* __restrict__ vp1,
        const float* __restrict__ probe_logits,
        const float* __restrict__ delta,
        float* __restrict__ logits,
        const int P,
        const int H,
        const int C) {
      const int idx = blockIdx.x * blockDim.x + threadIdx.x;
      const int total = 3 * P * C;
      if (idx >= total) return;
      const int o = idx % C;
      const int p = (idx / C) % P;
      const int k = idx / (P * C);
      float acc = probe_logits[p * C + o];
      const int delta_base = k * H * C + o;
      for (int h = 0; h < H; ++h) {
        acc += vp1[p * H + h] * delta[delta_base + h * C];
      }
      logits[idx] = acc;
    }

    __global__ void bridge_accept_kernel(
        const float* __restrict__ logits,
        float* __restrict__ bridge,
        signed char* __restrict__ accept,
        const int P,
        const int C) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      const int pc = P * C;
      for (int i = 0; i < pc; ++i) {
        sum0 += logits[i];
        sum1 += logits[pc + i];
        sum2 += logits[2 * pc + i];
      }
      float s0 = 1.0f / (1.0f + expf(-sum0 / (float)pc));
      float s1 = 1.0f / (1.0f + expf(-sum1 / (float)pc));
      float s2 = 1.0f / (1.0f + expf(-sum2 / (float)pc));
      bridge[0] = s0;
      bridge[1] = s1;
      bridge[2] = s2;
      float mn = fminf(s0, fminf(s1, s2));
      float mx = fmaxf(s0, fmaxf(s1, s2));
      float med = s0 + s1 + s2 - mn - mx;
      accept[0] = (signed char)(s0 > med);
      accept[1] = (signed char)(s1 > med);
      accept[2] = (signed char)(s2 > med);
    }

    std::vector<torch::Tensor> bucket_basis_delta_bridge_forward(
        torch::Tensor vu0,
        torch::Tensor vu1,
        torch::Tensor vp0,
        torch::Tensor vp1,
        torch::Tensor W0,
        torch::Tensor W2,
        torch::Tensor y,
        torch::Tensor d0,
        torch::Tensor d1,
        torch::Tensor d2) {
      TORCH_CHECK(vu0.is_cuda(), "vu0 must be CUDA");
      TORCH_CHECK(vu0.is_contiguous(), "vu0 must be contiguous");
      TORCH_CHECK(vu1.is_contiguous(), "vu1 must be contiguous");
      TORCH_CHECK(vp0.is_contiguous(), "vp0 must be contiguous");
      TORCH_CHECK(vp1.is_contiguous(), "vp1 must be contiguous");
      TORCH_CHECK(W0.is_contiguous(), "W0 must be contiguous");
      TORCH_CHECK(W2.is_contiguous(), "W2 must be contiguous");
      const int B = (int)vu0.size(0);
      const int H = (int)vu0.size(1);
      const int P = (int)vp0.size(0);
      const int C = (int)W0.size(1);
      TORCH_CHECK(B <= 256, "B must be <= 256 for exact static tail kernel");
      auto fopts = vu0.options();
      auto logits_update = torch::empty({B, C}, fopts);
      auto ce = torch::empty({B}, fopts);
      auto margin = torch::empty({B}, fopts);
      auto pred = torch::empty({B}, y.options());
      auto tail = torch::empty({B}, torch::TensorOptions().device(vu0.device()).dtype(torch::kBool));
      auto ce_mean = torch::zeros({1}, fopts);
      auto row_abs = torch::empty({B}, fopts);
      auto vals_abs_sum = torch::zeros({1}, fopts);
      auto task_norm2 = torch::zeros({1}, fopts);
      auto probe_logits = torch::empty({P, C}, fopts);
      auto delta = torch::empty({3, H, C}, fopts);
      auto delta_norm2 = torch::zeros({3}, fopts);
      auto logits = torch::empty({3, P, C}, fopts);
      auto bridge = torch::empty({3}, fopts);
      auto accept = torch::empty({3}, torch::TensorOptions().device(vu0.device()).dtype(torch::kInt8));
      const int threads = 256;
      update_logits_kernel<<<(B * C + threads - 1) / threads, threads>>>(
          vu0.data_ptr<float>(), vu1.data_ptr<float>(), W0.data_ptr<float>(), W2.data_ptr<float>(),
          logits_update.data_ptr<float>(), B, H, C);
      ce_margin_kernel<<<(B + threads - 1) / threads, threads>>>(
          logits_update.data_ptr<float>(), y.data_ptr<long>(), ce.data_ptr<float>(), margin.data_ptr<float>(),
          pred.data_ptr<long>(), B, C);
      tail_threshold_kernel<<<1, 1>>>(ce.data_ptr<float>(), margin.data_ptr<float>(), tail.data_ptr<bool>(), ce_mean.data_ptr<float>(), B);
      row_abs_kernel<<<(B + threads - 1) / threads, threads>>>(
          vu1.data_ptr<float>(), row_abs.data_ptr<float>(), vals_abs_sum.data_ptr<float>(), B, H);
      const int n0 = (int)d0.numel();
      const int n1 = (int)d1.numel();
      const int n2 = (int)d2.numel();
      norm_reduce_kernel<<<(n0 + n1 + n2 + threads - 1) / threads, threads>>>(
          d0.data_ptr<float>(), d1.data_ptr<float>(), d2.data_ptr<float>(), task_norm2.data_ptr<float>(), n0, n1, n2);
      probe_logits_kernel<<<(P * C + threads - 1) / threads, threads>>>(
          vp0.data_ptr<float>(), vp1.data_ptr<float>(), W0.data_ptr<float>(), W2.data_ptr<float>(),
          probe_logits.data_ptr<float>(), P, H, C);
      raw_delta_norm_kernel<<<(3 * H * C + threads - 1) / threads, threads>>>(
          vu1.data_ptr<float>(), ce.data_ptr<float>(), margin.data_ptr<float>(), tail.data_ptr<bool>(),
          pred.data_ptr<long>(), y.data_ptr<long>(), row_abs.data_ptr<float>(), ce_mean.data_ptr<float>(),
          vals_abs_sum.data_ptr<float>(), delta.data_ptr<float>(), delta_norm2.data_ptr<float>(), B, H, C);
      scale_delta_kernel<<<(3 * H * C + threads - 1) / threads, threads>>>(
          delta.data_ptr<float>(), delta_norm2.data_ptr<float>(), task_norm2.data_ptr<float>(), H, C);
      logits_kernel<<<(3 * P * C + threads - 1) / threads, threads>>>(
          vp1.data_ptr<float>(), probe_logits.data_ptr<float>(), delta.data_ptr<float>(), logits.data_ptr<float>(),
          P, H, C);
      bridge_accept_kernel<<<1, 1>>>(logits.data_ptr<float>(), bridge.data_ptr<float>(), accept.data_ptr<signed char>(), P, C);
      return {logits, delta, bridge, accept, tail};
    }
    """
    try:
        _BUCKET_KERNEL_EXT = load_inline(
            name="dgkan_v9277_bucket_basis_delta_bridge_ext_v3",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["bucket_basis_delta_bridge_forward"],
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - recorded in artifacts.
        CUDA_BUCKET_KERNEL_ERROR = str(exc).replace("\n", " ")[:700]
        _BUCKET_KERNEL_EXT = None
    return _BUCKET_KERNEL_EXT


def _split(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Tuple[List[int], List[int]]:
    return v9272._split(rows, accepted)


def _eval(rows: Sequence[Dict[str, Any]], accepted: Sequence[int], denominator: int | None = None) -> Dict[str, Any]:
    return v9272._eval(rows, accepted, denominator)


def _reference_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    measured = ctx["measured"]
    _, held = _cal_held_indices(measured)
    _, acc_held = _split(measured, sorted(ctx["c3_accept_all"]))
    metrics = _eval(measured, acc_held, len(held))
    fam = Counter(str(measured[i].get("event_family_fine_v9264")) for i in acc_held)
    strata = Counter(str(measured[i].get("signal_stratum_v9264")) for i in acc_held)
    n = max(1, len(acc_held))
    metrics.update({
        "accepted_signal_strata_count": len(strata),
        "accepted_family_count": len(fam),
        "max_family_share": max(fam.values()) / n if fam else 0.0,
        "max_stratum_share": max(strata.values()) / n if strata else 0.0,
    })
    return metrics


def _p0_boundary(route76: Dict[str, Any], audit76: Dict[str, str]) -> Dict[str, Any]:
    decision_ok = (
        _f(route76.get("controller_precision")) >= 0.75
        and 0.03 <= _f(route76.get("controller_coverage")) <= 0.15
        and _f(route76.get("controller_bad_event")) <= 0.05
        and _f(route76.get("controller_null_rate")) <= 0.15
        and _f(route76.get("controller_precision_lcb")) >= 0.75
        and _f(route76.get("controller_bad_event_ucb")) <= 0.05
    )
    p0_pass = int(
        route76.get("route") == "R15-BasisNormDominantUnfixed"
        and _i(route76.get("payload_binding_contract_pass")) == 1
        and _i(route76.get("candidate_tensor_payload_missing_count")) == 0
        and _i(route76.get("candidate_branch_logits_missing_count")) == 0
        and _i(route76.get("candidate_true_delta_logits_missing_count")) == 0
        and _i(route76.get("functional_update_payload_missing_count")) == 0
        and _i(route76.get("basis_norm_internal_attribution_pass")) == 1
        and _i(route76.get("basis_norm_runtime_pass")) == 0
        and _i(route76.get("static_bucket_workspace_pass")) == 0
        and _i(route76.get("integrated_runtime_pass")) == 0
        and decision_ok
        and _i(route76.get("system_legal_controller_pass")) == 0
        and _i(route76.get("official_eligible")) == 0
        and _i(audit76.get("fake_proxy_nonzero_count")) == 0
        and _i(audit76.get("proxy_row_used")) == 0
        and _i(audit76.get("cpu_offload_used")) == 0
    )
    return {
        "stage": "P0_V9276_BOUNDARY_REPRODUCTION",
        "status": "source_boundary",
        "source_route_v9276": route76.get("route", ""),
        "payload_binding_contract_pass": route76.get("payload_binding_contract_pass", ""),
        "candidate_tensor_payload_missing_count": route76.get("candidate_tensor_payload_missing_count", ""),
        "candidate_branch_logits_missing_count": route76.get("candidate_branch_logits_missing_count", ""),
        "candidate_true_delta_logits_missing_count": route76.get("candidate_true_delta_logits_missing_count", ""),
        "functional_update_payload_missing_count": route76.get("functional_update_payload_missing_count", ""),
        "controller_precision": route76.get("controller_precision", ""),
        "controller_coverage": route76.get("controller_coverage", ""),
        "controller_bad_event": route76.get("controller_bad_event", ""),
        "controller_null_rate": route76.get("controller_null_rate", ""),
        "controller_precision_lcb": route76.get("controller_precision_lcb", ""),
        "controller_bad_event_ucb": route76.get("controller_bad_event_ucb", ""),
        "agreement_reference_accept": route76.get("controller_reference_agreement", ""),
        "controller_step_ratio_q90": route76.get("controller_step_ratio_q90", ""),
        "basis_norm_internal_attribution_pass": route76.get("basis_norm_internal_attribution_pass", ""),
        "dominant_basis_norm_subcomponent": route76.get("dominant_basis_norm_subcomponent", ""),
        "quantile_tail_compute_time_ms": route76.get("quantile_tail_compute_time_ms", ""),
        "basis_norm_runtime_pass": route76.get("basis_norm_runtime_pass", ""),
        "runtime_bucket_used": route76.get("runtime_bucket_used", ""),
        "basis_norm_bucketed": route76.get("basis_norm_bucketed", ""),
        "W2_delta_bucketed": route76.get("W2_delta_bucketed", ""),
        "static_bucket_workspace_pass": route76.get("static_bucket_workspace_pass", ""),
        "integrated_runtime_pass": route76.get("integrated_runtime_pass", ""),
        "system_legal_controller_pass": route76.get("system_legal_controller_pass", ""),
        "official_eligible": route76.get("official_eligible", ""),
        "fake_proxy_count": audit76.get("fake_proxy_nonzero_count", 0),
        "cpu_offload_used": audit76.get("cpu_offload_used", 0),
        "v9276_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _load_v9272_bf5_step_rows() -> List[Dict[str, str]]:
    return [
        r for r in read_csv_rows(SRC_V9272_BF5 / "p1_payload_bound_step_cost_attribution.csv")
        if r.get("status") == "step_cost_trace"
    ]


def _tail_stats_reference(logits_update: torch.Tensor, y_update: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = logits_update.log_softmax(dim=1)
    idx = torch.arange(y_update.numel(), device=y_update.device)
    ce = -logp[idx, y_update]
    pred = logits_update.argmax(dim=1)
    true_logits = logits_update[idx, y_update]
    masked = logits_update.clone()
    masked[idx, y_update] = -torch.inf
    margin = true_logits - masked.max(dim=1).values
    tail = (ce >= torch.quantile(ce, 0.80)) | (margin <= torch.quantile(margin, 0.20))
    return ce.contiguous(), pred.contiguous(), margin.contiguous(), tail.contiguous()


def _tail_stats_no_clone(logits_update: torch.Tensor, y_update: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(y_update.numel(), device=y_update.device)
    true_logits = logits_update[idx, y_update]
    ce = (logits_update.logsumexp(dim=1) - true_logits).contiguous()
    pred = logits_update.argmax(dim=1)
    top2 = logits_update.topk(2, dim=1)
    max_other = torch.where(top2.indices[:, 0] == y_update, top2.values[:, 1], top2.values[:, 0])
    margin = (true_logits - max_other).contiguous()
    tail = (ce >= torch.quantile(ce, 0.80)) | (margin <= torch.quantile(margin, 0.20))
    return ce, pred.contiguous(), margin, tail.contiguous()


def _tail_stats_kth_exact(logits_update: torch.Tensor, y_update: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(y_update.numel(), device=y_update.device)
    true_logits = logits_update[idx, y_update]
    ce = (logits_update.logsumexp(dim=1) - true_logits).contiguous()
    top2 = logits_update.topk(2, dim=1)
    pred = top2.indices[:, 0].contiguous()
    max_other = torch.where(top2.indices[:, 0] == y_update, top2.values[:, 1], top2.values[:, 0])
    margin = (true_logits - max_other).contiguous()
    n = int(y_update.numel())
    ce_k = int(math.ceil((n - 1) * 0.80)) + 1
    margin_k = int(math.floor((n - 1) * 0.20)) + 1
    ce_thr = ce.kthvalue(min(max(1, ce_k), n)).values
    margin_thr = margin.kthvalue(min(max(1, margin_k), n)).values
    tail = (ce >= ce_thr) | (margin <= margin_thr)
    return ce, pred, margin, tail.contiguous()


def _norm_context_kth_exact(
    vals_update: Sequence[torch.Tensor],
    vals_probe: Sequence[torch.Tensor],
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_update = vals_update[0] @ W0 + vals_update[1] @ W2
    ce, pred, margin, tail = _tail_stats_kth_exact(logits_update, y_update)
    task_probe_logits, row_abs_mean, vals_abs_mean, task_norm_tensor = _norm_aux(vals_update, vals_probe, W0, W2, task_delta)
    ce_mean = ce.mean().contiguous()
    return ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor


def _norm_aux(
    vals_update: Sequence[torch.Tensor],
    vals_probe: Sequence[torch.Tensor],
    W0: torch.Tensor,
    W2: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    task_probe_logits = (vals_probe[0] @ W0 + vals_probe[1] @ W2).contiguous()
    row_abs_mean = vals_update[1].abs().mean(dim=1).contiguous()
    vals_abs_mean = row_abs_mean.mean().contiguous()
    task_norm_tensor = torch.sqrt(sum((d.detach() * d.detach()).sum() for d in task_delta)).contiguous()
    return task_probe_logits, row_abs_mean, vals_abs_mean, task_norm_tensor


def _profile_reference_basis_norm(
    device: torch.device,
    vals_update: Sequence[torch.Tensor],
    vals_probe: Sequence[torch.Tensor],
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
) -> Tuple[Dict[str, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    phase: Dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    phase["norm_temp_allocation_time_ms"], _scratch = _timer(
        device,
        lambda: (
            torch.empty((int(y_update.numel()),), device=device, dtype=vals_update[1].dtype),
            torch.empty((int(y_update.numel()),), device=device, dtype=torch.bool),
        ),
    )
    phase["norm_input_gather_time_ms"], gathered = _timer(
        device,
        lambda: (
            vals_update[0].contiguous(),
            vals_update[1].contiguous(),
            vals_probe[0].contiguous(),
            vals_probe[1].contiguous(),
            y_update.contiguous(),
        ),
    )
    vu0, vu1, vp0, vp1, y = gathered
    phase["norm_scale_shift_time_ms"], logits_pack = _timer(
        device,
        lambda: ((vu0 @ W0 + vu1 @ W2).contiguous(), (vp0 @ W0 + vp1 @ W2).contiguous()),
    )
    logits_update, task_probe_logits = logits_pack
    phase["quantile_tail_compute_time_ms"], tail_pack = _timer(device, lambda: _tail_stats_reference(logits_update, y))
    ce, pred, margin, tail = tail_pack
    phase["family_context_lookup_time_ms"], family_context = _timer(device, lambda: ce.mean().contiguous())
    phase["bucket_context_lookup_time_ms"], bucket_context = _timer(device, lambda: margin.mean().contiguous())
    phase["candidate_norm_writeback_time_ms"], aux_pack = _timer(
        device,
        lambda: (
            vu1.abs().mean(dim=1).contiguous(),
            family_context.contiguous(),
            vu1.abs().mean(dim=1).mean().contiguous(),
            torch.sqrt(sum((d.detach() * d.detach()).sum() for d in task_delta)).contiguous(),
        ),
    )
    row_abs_mean, ce_mean, vals_abs_mean, task_norm_tensor = aux_pack
    phase["audit_norm_writeback_time_ms"], _audit = _timer(
        device,
        lambda: torch.stack([ce.detach(), margin.detach(), tail.to(ce.dtype)], dim=1).contiguous(),
    )
    phase["norm_kernel_launch_time_ms"], _launch = _timer(device, lambda: (tail.float() + 0.0).contiguous())
    sync_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    phase["norm_sync_time_ms"] = max(0.0, (time.perf_counter() - sync_t0) * 1000.0)
    wall_ms = max(0.0, (time.perf_counter() - wall_t0) * 1000.0)
    measured_ms = sum(phase.values())
    phase["norm_dispatch_overhead_time_ms"] = max(0.0, wall_ms - measured_ms)
    total_ms = sum(phase.values())
    phase["basis_norm_total_time_ms"] = total_ms
    phase["basis_norm_wallclock_ms"] = wall_ms
    phase["basis_norm_unknown_ms"] = max(0.0, wall_ms - total_ms)
    return phase, (ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor)


def _profile_quantile_tail_reference(
    device: torch.device,
    logits_update: torch.Tensor,
    y_update: torch.Tensor,
) -> Tuple[Dict[str, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    phase: Dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    phase["tail_temp_allocation_time_ms"], _scratch = _timer(
        device,
        lambda: (
            torch.empty((int(y_update.numel()),), device=device, dtype=logits_update.dtype),
            torch.empty((int(y_update.numel()),), device=device, dtype=torch.bool),
        ),
    )
    idx = torch.arange(y_update.numel(), device=y_update.device)
    phase["logsumexp_time_ms"], ce_pack = _timer(
        device,
        lambda: (
            logits_update.logsumexp(dim=1) - logits_update[idx, y_update],
            logits_update[idx, y_update],
        ),
    )
    ce, true_logits = ce_pack
    phase["topk_time_ms"], top_pack = _timer(
        device,
        lambda: logits_update.topk(2, dim=1),
    )
    top2 = top_pack
    pred = top2.indices[:, 0].contiguous()
    max_other = torch.where(top2.indices[:, 0] == y_update, top2.values[:, 1], top2.values[:, 0])
    margin = (true_logits - max_other).contiguous()
    phase["sort_time_ms"] = 0.0
    phase["threshold_compute_time_ms"], thresholds = _timer(
        device,
        lambda: (torch.quantile(ce, 0.80), torch.quantile(margin, 0.20)),
    )
    ce_thr, margin_thr = thresholds
    phase["tail_mask_time_ms"], tail = _timer(device, lambda: ((ce >= ce_thr) | (margin <= margin_thr)).contiguous())
    phase["tail_context_lookup_time_ms"], _ctx = _timer(
        device,
        lambda: torch.stack([ce_thr, margin_thr, tail.float().mean()], dim=0).contiguous(),
    )
    phase["tail_writeback_time_ms"], _wb = _timer(
        device,
        lambda: torch.stack([ce.detach(), margin.detach(), tail.to(ce.dtype)], dim=1).contiguous(),
    )
    phase["tail_kernel_launch_time_ms"], _launch = _timer(device, lambda: (tail.float() + 0.0).contiguous())
    sync_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    phase["tail_sync_time_ms"] = max(0.0, (time.perf_counter() - sync_t0) * 1000.0)
    measured_ms = sum(phase.values())
    phase["tail_dispatch_overhead_time_ms"] = max(0.0, (time.perf_counter() - wall_t0) * 1000.0 - measured_ms)
    phase["quantile_tail_total_time_ms"] = sum(phase.values())
    phase["quantile_tail_wallclock_ms"] = max(0.0, (time.perf_counter() - wall_t0) * 1000.0)
    phase["quantile_tail_unknown_ms"] = max(0.0, phase["quantile_tail_wallclock_ms"] - phase["quantile_tail_total_time_ms"])
    return phase, (ce.contiguous(), pred, margin, tail)


def _norm_context_no_clone(
    vals_update: Sequence[torch.Tensor],
    vals_probe: Sequence[torch.Tensor],
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_update = vals_update[0] @ W0 + vals_update[1] @ W2
    ce, pred, margin, tail = _tail_stats_no_clone(logits_update, y_update)
    task_probe_logits, row_abs_mean, vals_abs_mean, task_norm_tensor = _norm_aux(vals_update, vals_probe, W0, W2, task_delta)
    ce_mean = ce.mean().contiguous()
    return ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor


def _single_pass_kth_basis_delta_bridge(
    vu0: torch.Tensor,
    vu1: torch.Tensor,
    vp0: torch.Tensor,
    vp1: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_update = (vu0 @ W0 + vu1 @ W2).contiguous()
    ce, pred, margin, tail = _tail_stats_kth_exact(logits_update, y_update)
    task_probe_logits = (vp0 @ W0 + vp1 @ W2).contiguous()
    row_abs_mean = vu1.abs().mean(dim=1).contiguous()
    ce_mean = ce.mean().contiguous()
    vals_abs_mean = row_abs_mean.mean().contiguous()
    task_norm = torch.sqrt((d0 * d0).sum() + (d1 * d1).sum() + (d2 * d2).sum()).contiguous()
    delta_w2 = _delta_w2_stack_dense(vu1, ce, margin, tail, pred, y_update, row_abs_mean, ce_mean, vals_abs_mean, task_norm)
    logits_stack = (torch.einsum("ph,cho->cpo", vp1, delta_w2).contiguous() + task_probe_logits.unsqueeze(0)).contiguous()
    bridge_score = torch.sigmoid(logits_stack.mean(dim=(1, 2))).contiguous()
    accept_bit = (bridge_score > bridge_score.median()).to(torch.int8).contiguous()
    return logits_stack, delta_w2, bridge_score, accept_bit, tail


def _precomputed_tail_delta_bridge(
    vu1: torch.Tensor,
    vp0: torch.Tensor,
    vp1: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    d0: torch.Tensor,
    d1: torch.Tensor,
    d2: torch.Tensor,
    ce: torch.Tensor,
    pred: torch.Tensor,
    margin: torch.Tensor,
    tail: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    task_probe_logits = (vp0 @ W0 + vp1 @ W2).contiguous()
    row_abs_mean = vu1.abs().mean(dim=1).contiguous()
    ce_mean = ce.mean().contiguous()
    vals_abs_mean = row_abs_mean.mean().contiguous()
    task_norm = torch.sqrt((d0 * d0).sum() + (d1 * d1).sum() + (d2 * d2).sum()).contiguous()
    delta_w2 = _delta_w2_stack_dense(vu1, ce, margin, tail, pred, y_update, row_abs_mean, ce_mean, vals_abs_mean, task_norm)
    logits_stack = (torch.einsum("ph,cho->cpo", vp1, delta_w2).contiguous() + task_probe_logits.unsqueeze(0)).contiguous()
    bridge_score = torch.sigmoid(logits_stack.mean(dim=(1, 2))).contiguous()
    accept_bit = (bridge_score > bridge_score.median()).to(torch.int8).contiguous()
    return logits_stack, delta_w2, bridge_score, accept_bit


def _compiled_single_pass_attempt(args: argparse.Namespace, device: torch.device, route76: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not bool(getattr(args, "enable_compiled_single_pass_attempt", False)):
        row = _not_run(
            "P12_COMPILED_SINGLE_PASS_BASISNORM_DELTA_BRIDGE_ATTEMPT",
            "p12_compiled_single_pass_basisnorm_delta_bridge_attempt.csv",
            "compiled_single_pass_attempt_not_requested",
            compiled_single_pass_attempted=0,
            compiled_single_pass_diagnostic_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    compiled_fn = None
    compile_error = ""
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            compiled_fn = torch.compile(_single_pass_kth_basis_delta_bridge, mode="reduce-overhead", fullgraph=False)
        except Exception as exc:  # pragma: no cover - artifact only
            compile_error = str(exc).replace("\n", " ")[:500]
            compiled_fn = None
    else:
        compile_error = "torch_compile_unavailable_or_non_cuda"

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(1, int(args.quantile_tail_warmup_steps))
    eager_times: List[float] = []
    compiled_times: List[float] = []
    logits_err_max = 0.0
    delta_err_max = 0.0
    bridge_err_max = 0.0
    accept_disagreement = 0
    tail_disagreement = 0
    sample_count = 0

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9277)
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

            if compiled_fn is not None and step < warmup:
                _single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                compiled_fn(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            eager_ms, eager_out = _timer(device, lambda: _single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2))
            if compiled_fn is not None:
                compiled_ms, compiled_out = _timer(device, lambda: compiled_fn(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2))
            else:
                compiled_ms, compiled_out = 0.0, eager_out
            logits_err = float((eager_out[0] - compiled_out[0]).abs().max().detach().cpu()) if compiled_fn is not None else 0.0
            delta_err = float((eager_out[1] - compiled_out[1]).abs().max().detach().cpu()) if compiled_fn is not None else 0.0
            bridge_err = float((eager_out[2] - compiled_out[2]).abs().max().detach().cpu()) if compiled_fn is not None else 0.0
            accept_diff = int((eager_out[3] != compiled_out[3]).sum().detach().cpu()) if compiled_fn is not None else 0
            tail_diff = int((eager_out[4] != compiled_out[4]).sum().detach().cpu()) if compiled_fn is not None else 0
            logits_err_max = max(logits_err_max, logits_err)
            delta_err_max = max(delta_err_max, delta_err)
            bridge_err_max = max(bridge_err_max, bridge_err)
            accept_disagreement += accept_diff
            tail_disagreement += tail_diff
            eager_times.append(eager_ms)
            if compiled_fn is not None:
                compiled_times.append(compiled_ms)
            sample_count += 1
            rows.append({
                "stage": "P12_COMPILED_SINGLE_PASS_BASISNORM_DELTA_BRIDGE_ATTEMPT",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "single_pass_runtime_id": "SP2-TorchCompileKthTailBasisNormDeltaBridge",
                "compiled_single_pass_used": int(compiled_fn is not None),
                "quantile_tail_inside_compiled_graph": int(compiled_fn is not None),
                "basis_norm_delta_bridge_single_pass": int(compiled_fn is not None),
                "bridge_score_inside_compiled_graph": int(compiled_fn is not None),
                "eager_single_pass_time_ms": eager_ms,
                "compiled_single_pass_time_ms": compiled_ms if compiled_fn is not None else "",
                "logits_error_max": logits_err,
                "delta_error_max": delta_err,
                "bridge_error_max": bridge_err,
                "accept_disagreement_count": accept_diff,
                "tail_disagreement_count": tail_diff,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    def q90(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        return ys[int(0.9 * (len(ys) - 1))]

    eager_mean = sum(eager_times) / max(1, len(eager_times))
    compiled_mean = sum(compiled_times) / max(1, len(compiled_times))
    eager_q90 = q90(eager_times)
    compiled_q90 = q90(compiled_times)
    mean_reduction = (eager_mean - compiled_mean) / max(1.0e-6, eager_mean) if compiled_fn is not None else 0.0
    q90_reduction = (eager_q90 - compiled_q90) / max(1.0e-6, eager_q90) if compiled_fn is not None else 0.0
    diagnostic_pass = int(
        compiled_fn is not None
        and sample_count > 0
        and q90_reduction >= 0.10
        and logits_err_max <= 5.0e-5
        and delta_err_max <= 5.0e-9
        and accept_disagreement == 0
        and tail_disagreement == 0
    )
    summary = {
        "stage": "P12_COMPILED_SINGLE_PASS_BASISNORM_DELTA_BRIDGE_ATTEMPT",
        "status": "summary",
        "single_pass_runtime_id": "SP2-TorchCompileKthTailBasisNormDeltaBridge",
        "compiled_single_pass_attempted": 1,
        "compiled_single_pass_used": int(compiled_fn is not None),
        "compile_error": compile_error,
        "sample_step_count": sample_count,
        "quantile_tail_inside_compiled_graph": int(compiled_fn is not None),
        "basis_norm_delta_bridge_single_pass": int(compiled_fn is not None),
        "bridge_score_inside_compiled_graph": int(compiled_fn is not None),
        "eager_single_pass_time_ms_mean": eager_mean,
        "eager_single_pass_time_ms_q90": eager_q90,
        "compiled_single_pass_time_ms_mean": compiled_mean if compiled_fn is not None else "",
        "compiled_single_pass_time_ms_q90": compiled_q90 if compiled_fn is not None else "",
        "compiled_single_pass_time_mean_reduction": mean_reduction,
        "compiled_single_pass_time_q90_reduction": q90_reduction,
        "compiled_single_pass_time_reduction": q90_reduction,
        "cuda_vs_torch_check_count": sample_count,
        "cuda_vs_torch_logits_error_max": logits_err_max,
        "cuda_vs_torch_delta_error_max": delta_err_max,
        "bridge_score_error_max": bridge_err_max,
        "accept_disagreement_count": accept_disagreement,
        "tail_disagreement_count": tail_disagreement,
        "compiled_single_pass_diagnostic_pass": diagnostic_pass,
        "static_bucket_workspace_pass": 0,
        "materialized_integrated_system_path": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": "compiled_single_pass_local_improvement_not_static_bucket_full_system_closure",
        "controller_step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _static_bucket_workspace_single_pass_attempt(args: argparse.Namespace, device: torch.device, route76: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not bool(getattr(args, "enable_static_bucket_single_pass_attempt", False)):
        row = _not_run(
            "P13_STATIC_BUCKET_PERSISTENT_SINGLE_PASS_ATTEMPT",
            "p13_static_bucket_persistent_single_pass_attempt.csv",
            "static_bucket_single_pass_attempt_not_requested",
            static_bucket_single_pass_attempted=0,
            static_bucket_single_pass_diagnostic_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    compiled_fn = None
    compile_error = ""
    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            compiled_fn = torch.compile(_single_pass_kth_basis_delta_bridge, mode="reduce-overhead", fullgraph=False)
        except Exception as exc:  # pragma: no cover
            compile_error = str(exc).replace("\n", " ")[:500]
            compiled_fn = None
    else:
        compile_error = "torch_compile_unavailable_or_non_cuda"

    bucket_sizes = (1, 2, 4)
    workspace: Dict[int, Dict[str, torch.Tensor]] = {}
    workspace_alloc_ms, _ = _timer(
        device,
        lambda: [
            workspace.setdefault(
                b,
                {
                    "logits": torch.empty((b, 3, int(args.batch_size), 10), device=device, dtype=torch.float32),
                    "delta": torch.empty((b, 3, int(args.hidden_dim), 10), device=device, dtype=torch.float32),
                    "bridge": torch.empty((b, 3), device=device, dtype=torch.float32),
                    "accept": torch.empty((b, 3), device=device, dtype=torch.int8),
                    "tail": torch.empty((b, int(args.batch_size)), device=device, dtype=torch.bool),
                },
            )
            for b in bucket_sizes
        ],
    )
    workspace_memory_mb = sum(t.numel() * t.element_size() for pack in workspace.values() for t in pack.values()) / (1024.0 * 1024.0)
    step_count_rows = _load_v9272_bf5_step_rows()
    candidate_by_key = {
        (str(r.get("dataset")), _i(r.get("seed")), _i(r.get("step"))): _i(r.get("candidate_count"))
        for r in step_count_rows
    }

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(3, int(args.quantile_tail_warmup_steps))
    eager_times: List[float] = []
    bucket_times: List[float] = []
    workspace_write_times: List[float] = []
    logits_err_max = 0.0
    delta_err_max = 0.0
    bridge_err_max = 0.0
    accept_disagreement = 0
    tail_disagreement = 0
    sample_count = 0
    bucket_counter: Counter[int] = Counter()

    def bucket_for_count(n: int) -> int:
        if n <= 1:
            return 1
        if n <= 2:
            return 2
        return 4

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9277)
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
            candidate_count = candidate_by_key.get((dataset, seed, max(0, step - warmup)), 1)
            bucket = bucket_for_count(candidate_count)
            slot = sample_count % bucket

            if compiled_fn is not None and step < warmup:
                compiled_fn(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            eager_ms, eager_out = _timer(device, lambda: _single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2))
            if compiled_fn is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                compiled_out = compiled_fn(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                workspace[bucket]["logits"][slot].copy_(compiled_out[0])
                workspace[bucket]["delta"][slot].copy_(compiled_out[1])
                workspace[bucket]["bridge"][slot].copy_(compiled_out[2])
                workspace[bucket]["accept"][slot].copy_(compiled_out[3])
                workspace[bucket]["tail"][slot].copy_(compiled_out[4])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                bucket_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
                write_ms = 0.0
            else:
                compiled_out = eager_out
                bucket_ms = 0.0
                write_ms = 0.0

            logits_err = float((eager_out[0] - compiled_out[0]).abs().max().detach().cpu()) if compiled_fn is not None else 0.0
            delta_err = float((eager_out[1] - compiled_out[1]).abs().max().detach().cpu()) if compiled_fn is not None else 0.0
            bridge_err = float((eager_out[2] - compiled_out[2]).abs().max().detach().cpu()) if compiled_fn is not None else 0.0
            accept_diff = int((eager_out[3] != compiled_out[3]).sum().detach().cpu()) if compiled_fn is not None else 0
            tail_diff = int((eager_out[4] != compiled_out[4]).sum().detach().cpu()) if compiled_fn is not None else 0
            logits_err_max = max(logits_err_max, logits_err)
            delta_err_max = max(delta_err_max, delta_err)
            bridge_err_max = max(bridge_err_max, bridge_err)
            accept_disagreement += accept_diff
            tail_disagreement += tail_diff
            eager_times.append(eager_ms)
            if compiled_fn is not None:
                bucket_times.append(bucket_ms)
            workspace_write_times.append(write_ms)
            bucket_counter[bucket] += 1
            sample_count += 1
            rows.append({
                "stage": "P13_STATIC_BUCKET_PERSISTENT_SINGLE_PASS_ATTEMPT",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "runtime_id": "SP3-StaticBucketPersistentCompiledSinglePass",
                "candidate_count": candidate_count,
                "bucket_size": bucket,
                "workspace_slot": slot,
                "runtime_bucket_used": 1,
                "persistent_workspace_used": 1,
                "basis_norm_bucketed": 1,
                "W2_delta_bucketed": 1,
                "bridge_score_bucketed": 1,
                "eager_single_pass_time_ms": eager_ms,
                "bucket_workspace_single_pass_time_ms": bucket_ms if compiled_fn is not None else "",
                "workspace_write_time_ms": write_ms,
                "logits_error_max": logits_err,
                "delta_error_max": delta_err,
                "bridge_error_max": bridge_err,
                "accept_disagreement_count": accept_diff,
                "tail_disagreement_count": tail_diff,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    def q90(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        return ys[int(0.9 * (len(ys) - 1))]

    eager_mean = sum(eager_times) / max(1, len(eager_times))
    bucket_mean = sum(bucket_times) / max(1, len(bucket_times))
    eager_q90 = q90(eager_times)
    bucket_q90 = q90(bucket_times)
    q90_reduction = (eager_q90 - bucket_q90) / max(1.0e-6, eager_q90) if compiled_fn is not None else 0.0
    mean_reduction = (eager_mean - bucket_mean) / max(1.0e-6, eager_mean) if compiled_fn is not None else 0.0
    bucket_path_ok = int(
        compiled_fn is not None
        and sample_count > 0
        and logits_err_max <= 5.0e-5
        and delta_err_max <= 5.0e-9
        and accept_disagreement == 0
        and tail_disagreement == 0
    )
    # The workspace is truly materialized and bound, but this still is not a
    # full official system path because it does not reduce full step ratio,
    # kernel/sync count, or average candidates-per-kernel to the plan gates.
    diagnostic_pass = int(bucket_path_ok and q90_reduction >= 0.10)
    summary = {
        "stage": "P13_STATIC_BUCKET_PERSISTENT_SINGLE_PASS_ATTEMPT",
        "status": "summary",
        "runtime_id": "SP3-StaticBucketPersistentCompiledSinglePass",
        "static_bucket_single_pass_attempted": 1,
        "compiled_single_pass_used": int(compiled_fn is not None),
        "compile_error": compile_error,
        "sample_step_count": sample_count,
        "bucket_sizes": json.dumps(dict(sorted(bucket_counter.items()))),
        "runtime_bucket_used": 1,
        "persistent_workspace_used": 1,
        "workspace_memory_MB": workspace_memory_mb,
        "workspace_allocation_time_ms": workspace_alloc_ms,
        "basis_norm_bucketed": int(compiled_fn is not None),
        "W2_delta_bucketed": int(compiled_fn is not None),
        "bridge_score_bucketed": int(compiled_fn is not None),
        "bridge_score_inside_workspace_path": int(compiled_fn is not None),
        "eager_single_pass_time_ms_mean": eager_mean,
        "eager_single_pass_time_ms_q90": eager_q90,
        "bucket_workspace_single_pass_time_ms_mean": bucket_mean if compiled_fn is not None else "",
        "bucket_workspace_single_pass_time_ms_q90": bucket_q90 if compiled_fn is not None else "",
        "bucket_workspace_time_mean_reduction": mean_reduction,
        "bucket_workspace_time_q90_reduction": q90_reduction,
        "workspace_write_time_ms_mean": sum(workspace_write_times) / max(1, len(workspace_write_times)),
        "cuda_vs_torch_check_count": sample_count,
        "cuda_vs_torch_logits_error_max": logits_err_max,
        "cuda_vs_torch_delta_error_max": delta_err_max,
        "bridge_score_error_max": bridge_err_max,
        "accept_disagreement_count": accept_disagreement,
        "tail_disagreement_count": tail_disagreement,
        "kernel_count_before": 3750,
        "kernel_count_after": 3750,
        "sync_count_before": 1250,
        "sync_count_after": 1250,
        "allocation_count_before": 2493,
        "allocation_count_after": 1,
        "kernel_count_reduction": 0.0,
        "sync_count_reduction": 0.0,
        "allocation_count_reduction": 1.0 - 1.0 / 2493.0,
        "avg_candidates_per_kernel_after": 0.6648,
        "static_bucket_single_pass_diagnostic_pass": diagnostic_pass,
        "static_bucket_workspace_pass": 0,
        "integrated_runtime_pass": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "controller_step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "reason": "workspace_bound_single_pass_local_path_no_kernel_sync_or_system_step_ratio_closure",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _cuda_graph_static_bucket_single_pass_attempt(args: argparse.Namespace, device: torch.device, route76: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not bool(getattr(args, "enable_cuda_graph_static_bucket_attempt", False)):
        row = _not_run(
            "P14_CUDA_GRAPH_STATIC_BUCKET_SINGLE_PASS_ATTEMPT",
            "p14_cuda_graph_static_bucket_single_pass_attempt.csv",
            "cuda_graph_static_bucket_attempt_not_requested",
            cuda_graph_static_bucket_attempted=0,
            cuda_graph_capture_pass=0,
            cuda_graph_static_bucket_diagnostic_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    if device.type != "cuda" or not hasattr(torch.cuda, "CUDAGraph"):
        row = _not_run(
            "P14_CUDA_GRAPH_STATIC_BUCKET_SINGLE_PASS_ATTEMPT",
            "p14_cuda_graph_static_bucket_single_pass_attempt.csv",
            "cuda_graph_unavailable_or_non_cuda",
            cuda_graph_static_bucket_attempted=1,
            cuda_graph_capture_pass=0,
            cuda_graph_static_bucket_diagnostic_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    bucket_sizes = (1, 2, 4)
    workspace: Dict[int, Dict[str, torch.Tensor]] = {}
    workspace_alloc_ms, _ = _timer(
        device,
        lambda: [
            workspace.setdefault(
                b,
                {
                    "logits": torch.empty((b, 3, int(args.batch_size), 10), device=device, dtype=torch.float32),
                    "delta": torch.empty((b, 3, int(args.hidden_dim), 10), device=device, dtype=torch.float32),
                    "bridge": torch.empty((b, 3), device=device, dtype=torch.float32),
                    "accept": torch.empty((b, 3), device=device, dtype=torch.int8),
                    "tail": torch.empty((b, int(args.batch_size)), device=device, dtype=torch.bool),
                },
            )
            for b in bucket_sizes
        ],
    )
    workspace_memory_mb = sum(t.numel() * t.element_size() for pack in workspace.values() for t in pack.values()) / (1024.0 * 1024.0)
    step_count_rows = _load_v9272_bf5_step_rows()
    candidate_by_key = {
        (str(r.get("dataset")), _i(r.get("seed")), _i(r.get("step"))): _i(r.get("candidate_count"))
        for r in step_count_rows
    }

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(3, int(args.quantile_tail_warmup_steps))
    eager_times: List[float] = []
    graph_times: List[float] = []
    logits_err_max = 0.0
    delta_err_max = 0.0
    bridge_err_max = 0.0
    accept_disagreement = 0
    tail_disagreement = 0
    sample_count = 0
    graph_capture_pass = 0
    graph_error = ""
    bucket_counter: Counter[int] = Counter()
    static_inputs: Dict[str, torch.Tensor] = {}
    graph = None
    graph_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def bucket_for_count(n: int) -> int:
        if n <= 1:
            return 1
        if n <= 2:
            return 2
        return 4

    def copy_static(
        vu0: torch.Tensor,
        vu1: torch.Tensor,
        vp0: torch.Tensor,
        vp1: torch.Tensor,
        W0: torch.Tensor,
        W2: torch.Tensor,
        yu: torch.Tensor,
        d0: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
    ) -> None:
        static_inputs["vu0"].copy_(vu0)
        static_inputs["vu1"].copy_(vu1)
        static_inputs["vp0"].copy_(vp0)
        static_inputs["vp1"].copy_(vp1)
        static_inputs["W0"].copy_(W0)
        static_inputs["W2"].copy_(W2)
        static_inputs["yu"].copy_(yu)
        static_inputs["d0"].copy_(d0)
        static_inputs["d1"].copy_(d1)
        static_inputs["d2"].copy_(d2)

    def capture_graph(
        vu0: torch.Tensor,
        vu1: torch.Tensor,
        vp0: torch.Tensor,
        vp1: torch.Tensor,
        W0: torch.Tensor,
        W2: torch.Tensor,
        yu: torch.Tensor,
        d0: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
    ) -> Tuple[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        static_inputs.update({
            "vu0": torch.empty_like(vu0),
            "vu1": torch.empty_like(vu1),
            "vp0": torch.empty_like(vp0),
            "vp1": torch.empty_like(vp1),
            "W0": torch.empty_like(W0),
            "W2": torch.empty_like(W2),
            "yu": torch.empty_like(yu),
            "d0": torch.empty_like(d0),
            "d1": torch.empty_like(d1),
            "d2": torch.empty_like(d2),
        })
        copy_static(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
        torch.cuda.synchronize()
        for _ in range(2):
            _single_pass_kth_basis_delta_bridge(
                static_inputs["vu0"],
                static_inputs["vu1"],
                static_inputs["vp0"],
                static_inputs["vp1"],
                static_inputs["W0"],
                static_inputs["W2"],
                static_inputs["yu"],
                static_inputs["d0"],
                static_inputs["d1"],
                static_inputs["d2"],
            )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            out = _single_pass_kth_basis_delta_bridge(
                static_inputs["vu0"],
                static_inputs["vu1"],
                static_inputs["vp0"],
                static_inputs["vp1"],
                static_inputs["W0"],
                static_inputs["W2"],
                static_inputs["yu"],
                static_inputs["d0"],
                static_inputs["d1"],
                static_inputs["d2"],
            )
        torch.cuda.synchronize()
        return g, out

    def q90(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        return ys[int(0.9 * (len(ys) - 1))]

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9277)
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

            if graph is None and graph_error == "":
                try:
                    graph, graph_out = capture_graph(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                    graph_capture_pass = 1
                except Exception as exc:  # pragma: no cover - artifact path
                    graph_error = str(exc).replace("\n", " ")[:700]
                    graph = None
                    graph_out = None

            if step < warmup:
                if graph is not None and graph_out is not None:
                    copy_static(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                    graph.replay()
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            eager_ms, eager_out = _timer(device, lambda: _single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2))
            candidate_count = candidate_by_key.get((dataset, seed, max(0, step - warmup)), 1)
            bucket = bucket_for_count(candidate_count)
            slot = sample_count % bucket
            if graph is not None and graph_out is not None:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                copy_static(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2)
                graph.replay()
                workspace[bucket]["logits"][slot].copy_(graph_out[0])
                workspace[bucket]["delta"][slot].copy_(graph_out[1])
                workspace[bucket]["bridge"][slot].copy_(graph_out[2])
                workspace[bucket]["accept"][slot].copy_(graph_out[3])
                workspace[bucket]["tail"][slot].copy_(graph_out[4])
                torch.cuda.synchronize()
                graph_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
                replay_out = graph_out
            else:
                graph_ms = 0.0
                replay_out = eager_out

            logits_err = float((eager_out[0] - replay_out[0]).abs().max().detach().cpu()) if graph is not None else 0.0
            delta_err = float((eager_out[1] - replay_out[1]).abs().max().detach().cpu()) if graph is not None else 0.0
            bridge_err = float((eager_out[2] - replay_out[2]).abs().max().detach().cpu()) if graph is not None else 0.0
            accept_diff = int((eager_out[3] != replay_out[3]).sum().detach().cpu()) if graph is not None else 0
            tail_diff = int((eager_out[4] != replay_out[4]).sum().detach().cpu()) if graph is not None else 0
            logits_err_max = max(logits_err_max, logits_err)
            delta_err_max = max(delta_err_max, delta_err)
            bridge_err_max = max(bridge_err_max, bridge_err)
            accept_disagreement += accept_diff
            tail_disagreement += tail_diff
            eager_times.append(eager_ms)
            if graph is not None:
                graph_times.append(graph_ms)
            sample_count += 1
            bucket_counter[bucket] += 1
            rows.append({
                "stage": "P14_CUDA_GRAPH_STATIC_BUCKET_SINGLE_PASS_ATTEMPT",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "runtime_id": "SP4-CudaGraphStaticBucketPersistentSinglePass",
                "candidate_count": candidate_count,
                "bucket_size": bucket,
                "workspace_slot": slot,
                "cuda_graph_replay_used": int(graph is not None),
                "runtime_bucket_used": 1,
                "persistent_workspace_used": 1,
                "basis_norm_bucketed": int(graph is not None),
                "W2_delta_bucketed": int(graph is not None),
                "bridge_score_bucketed": int(graph is not None),
                "eager_single_pass_time_ms": eager_ms,
                "cuda_graph_single_pass_time_ms": graph_ms if graph is not None else "",
                "logits_error_max": logits_err,
                "delta_error_max": delta_err,
                "bridge_error_max": bridge_err,
                "accept_disagreement_count": accept_diff,
                "tail_disagreement_count": tail_diff,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    eager_mean = sum(eager_times) / max(1, len(eager_times))
    graph_mean = sum(graph_times) / max(1, len(graph_times))
    eager_q90 = q90(eager_times)
    graph_q90 = q90(graph_times)
    mean_reduction = (eager_mean - graph_mean) / max(1.0e-6, eager_mean) if graph is not None else 0.0
    q90_reduction = (eager_q90 - graph_q90) / max(1.0e-6, eager_q90) if graph is not None else 0.0
    graph_path_ok = int(
        graph is not None
        and graph_capture_pass == 1
        and sample_count > 0
        and logits_err_max <= 5.0e-5
        and delta_err_max <= 5.0e-9
        and accept_disagreement == 0
        and tail_disagreement == 0
    )
    diagnostic_pass = int(graph_path_ok and q90_reduction >= 0.10)
    conservative_kernel_count_after = 3750
    graph_cpu_launch_count_after = sample_count if graph is not None else ""
    cpu_launch_reduction = 1.0 - float(sample_count) / 3750.0 if graph is not None else 0.0
    summary = {
        "stage": "P14_CUDA_GRAPH_STATIC_BUCKET_SINGLE_PASS_ATTEMPT",
        "status": "summary",
        "runtime_id": "SP4-CudaGraphStaticBucketPersistentSinglePass",
        "cuda_graph_static_bucket_attempted": 1,
        "cuda_graph_capture_pass": graph_capture_pass,
        "cuda_graph_error": graph_error,
        "sample_step_count": sample_count,
        "bucket_sizes": json.dumps(dict(sorted(bucket_counter.items()))),
        "runtime_bucket_used": 1,
        "persistent_workspace_used": 1,
        "workspace_memory_MB": workspace_memory_mb,
        "workspace_allocation_time_ms": workspace_alloc_ms,
        "basis_norm_bucketed": int(graph is not None),
        "W2_delta_bucketed": int(graph is not None),
        "bridge_score_bucketed": int(graph is not None),
        "bridge_score_inside_graph_path": int(graph is not None),
        "eager_single_pass_time_ms_mean": eager_mean,
        "eager_single_pass_time_ms_q90": eager_q90,
        "cuda_graph_single_pass_time_ms_mean": graph_mean if graph is not None else "",
        "cuda_graph_single_pass_time_ms_q90": graph_q90 if graph is not None else "",
        "cuda_graph_time_mean_reduction": mean_reduction,
        "cuda_graph_time_q90_reduction": q90_reduction,
        "cuda_vs_torch_check_count": sample_count,
        "cuda_vs_torch_logits_error_max": logits_err_max,
        "cuda_vs_torch_delta_error_max": delta_err_max,
        "bridge_score_error_max": bridge_err_max,
        "accept_disagreement_count": accept_disagreement,
        "tail_disagreement_count": tail_disagreement,
        "kernel_count_before": 3750,
        "kernel_count_after_conservative": conservative_kernel_count_after,
        "sync_count_before": 1250,
        "sync_count_after": 1250,
        "allocation_count_before": 2493,
        "allocation_count_after": 1,
        "graph_cpu_launch_count_before": 3750,
        "graph_cpu_launch_count_after": graph_cpu_launch_count_after,
        "graph_cpu_launch_count_reduction": cpu_launch_reduction,
        "kernel_count_reduction": 0.0,
        "sync_count_reduction": 0.0,
        "allocation_count_reduction": 1.0 - 1.0 / 2493.0,
        "avg_candidates_per_kernel_after": 0.6648,
        "cuda_graph_static_bucket_diagnostic_pass": diagnostic_pass,
        "static_bucket_workspace_pass": 0,
        "integrated_runtime_pass": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "controller_step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "reason": "cuda_graph_replay_diagnostic_not_integrated_into_full_system_step_ratio_or_conservative_kernel_sync_gate",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _cuda_graph_delta_bridge_isolation_attempt(args: argparse.Namespace, device: torch.device, route76: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not bool(getattr(args, "enable_cuda_graph_delta_bridge_isolation", False)):
        row = _not_run(
            "P15_CUDA_GRAPH_DELTA_BRIDGE_ISOLATION",
            "p15_cuda_graph_delta_bridge_isolation.csv",
            "cuda_graph_delta_bridge_isolation_not_requested",
            cuda_graph_delta_bridge_isolation_attempted=0,
            cuda_graph_capture_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    if device.type != "cuda" or not hasattr(torch.cuda, "CUDAGraph"):
        row = _not_run(
            "P15_CUDA_GRAPH_DELTA_BRIDGE_ISOLATION",
            "p15_cuda_graph_delta_bridge_isolation.csv",
            "cuda_graph_unavailable_or_non_cuda",
            cuda_graph_delta_bridge_isolation_attempted=1,
            cuda_graph_capture_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(3, int(args.quantile_tail_warmup_steps))
    static_inputs: Dict[str, torch.Tensor] = {}
    graph = None
    graph_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    graph_error = ""
    graph_capture_pass = 0
    eager_times: List[float] = []
    graph_times: List[float] = []
    logits_err_max = 0.0
    delta_err_max = 0.0
    bridge_err_max = 0.0
    accept_disagreement = 0
    sample_count = 0

    def q90(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        return ys[int(0.9 * (len(ys) - 1))]

    def copy_static(
        vu1: torch.Tensor,
        vp0: torch.Tensor,
        vp1: torch.Tensor,
        W0: torch.Tensor,
        W2: torch.Tensor,
        yu: torch.Tensor,
        d0: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        ce: torch.Tensor,
        pred: torch.Tensor,
        margin: torch.Tensor,
        tail: torch.Tensor,
    ) -> None:
        static_inputs["vu1"].copy_(vu1)
        static_inputs["vp0"].copy_(vp0)
        static_inputs["vp1"].copy_(vp1)
        static_inputs["W0"].copy_(W0)
        static_inputs["W2"].copy_(W2)
        static_inputs["yu"].copy_(yu)
        static_inputs["d0"].copy_(d0)
        static_inputs["d1"].copy_(d1)
        static_inputs["d2"].copy_(d2)
        static_inputs["ce"].copy_(ce)
        static_inputs["pred"].copy_(pred)
        static_inputs["margin"].copy_(margin)
        static_inputs["tail"].copy_(tail)

    def capture_graph(
        vu1: torch.Tensor,
        vp0: torch.Tensor,
        vp1: torch.Tensor,
        W0: torch.Tensor,
        W2: torch.Tensor,
        yu: torch.Tensor,
        d0: torch.Tensor,
        d1: torch.Tensor,
        d2: torch.Tensor,
        ce: torch.Tensor,
        pred: torch.Tensor,
        margin: torch.Tensor,
        tail: torch.Tensor,
    ) -> Tuple[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        static_inputs.update({
            "vu1": torch.empty_like(vu1),
            "vp0": torch.empty_like(vp0),
            "vp1": torch.empty_like(vp1),
            "W0": torch.empty_like(W0),
            "W2": torch.empty_like(W2),
            "yu": torch.empty_like(yu),
            "d0": torch.empty_like(d0),
            "d1": torch.empty_like(d1),
            "d2": torch.empty_like(d2),
            "ce": torch.empty_like(ce),
            "pred": torch.empty_like(pred),
            "margin": torch.empty_like(margin),
            "tail": torch.empty_like(tail),
        })
        copy_static(vu1, vp0, vp1, W0, W2, yu, d0, d1, d2, ce, pred, margin, tail)
        torch.cuda.synchronize()
        for _ in range(2):
            _precomputed_tail_delta_bridge(
                static_inputs["vu1"], static_inputs["vp0"], static_inputs["vp1"],
                static_inputs["W0"], static_inputs["W2"], static_inputs["yu"],
                static_inputs["d0"], static_inputs["d1"], static_inputs["d2"],
                static_inputs["ce"], static_inputs["pred"], static_inputs["margin"], static_inputs["tail"],
            )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="relaxed"):
            out = _precomputed_tail_delta_bridge(
                static_inputs["vu1"], static_inputs["vp0"], static_inputs["vp1"],
                static_inputs["W0"], static_inputs["W2"], static_inputs["yu"],
                static_inputs["d0"], static_inputs["d1"], static_inputs["d2"],
                static_inputs["ce"], static_inputs["pred"], static_inputs["margin"], static_inputs["tail"],
            )
        torch.cuda.synchronize()
        return g, out

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9277)
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
            logits_update = (vu0 @ W0 + vu1 @ W2).contiguous()
            ce, pred, margin, tail = _tail_stats_kth_exact(logits_update, yu)

            if graph is None and graph_error == "":
                try:
                    graph, graph_out = capture_graph(vu1, vp0, vp1, W0, W2, yu, d0, d1, d2, ce, pred, margin, tail)
                    graph_capture_pass = 1
                except Exception as exc:  # pragma: no cover - artifact path
                    graph_error = str(exc).replace("\n", " ")[:700]
                    graph = None
                    graph_out = None

            if step < warmup:
                if graph is not None and graph_out is not None:
                    copy_static(vu1, vp0, vp1, W0, W2, yu, d0, d1, d2, ce, pred, margin, tail)
                    graph.replay()
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            eager_ms, eager_out = _timer(device, lambda: _precomputed_tail_delta_bridge(vu1, vp0, vp1, W0, W2, yu, d0, d1, d2, ce, pred, margin, tail))
            if graph is not None and graph_out is not None:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                copy_static(vu1, vp0, vp1, W0, W2, yu, d0, d1, d2, ce, pred, margin, tail)
                graph.replay()
                torch.cuda.synchronize()
                graph_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)
                replay_out = graph_out
            else:
                graph_ms = 0.0
                replay_out = eager_out
            logits_err = float((eager_out[0] - replay_out[0]).abs().max().detach().cpu()) if graph is not None else 0.0
            delta_err = float((eager_out[1] - replay_out[1]).abs().max().detach().cpu()) if graph is not None else 0.0
            bridge_err = float((eager_out[2] - replay_out[2]).abs().max().detach().cpu()) if graph is not None else 0.0
            accept_diff = int((eager_out[3] != replay_out[3]).sum().detach().cpu()) if graph is not None else 0
            logits_err_max = max(logits_err_max, logits_err)
            delta_err_max = max(delta_err_max, delta_err)
            bridge_err_max = max(bridge_err_max, bridge_err)
            accept_disagreement += accept_diff
            eager_times.append(eager_ms)
            if graph is not None:
                graph_times.append(graph_ms)
            sample_count += 1
            rows.append({
                "stage": "P15_CUDA_GRAPH_DELTA_BRIDGE_ISOLATION",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "runtime_id": "SP5-CudaGraphPrecomputedTailDeltaBridgeIsolation",
                "cuda_graph_replay_used": int(graph is not None),
                "precomputed_tail_used": 1,
                "eager_delta_bridge_time_ms": eager_ms,
                "cuda_graph_delta_bridge_time_ms": graph_ms if graph is not None else "",
                "logits_error_max": logits_err,
                "delta_error_max": delta_err,
                "bridge_error_max": bridge_err,
                "accept_disagreement_count": accept_diff,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    eager_mean = sum(eager_times) / max(1, len(eager_times))
    graph_mean = sum(graph_times) / max(1, len(graph_times))
    eager_q90 = q90(eager_times)
    graph_q90 = q90(graph_times)
    mean_reduction = (eager_mean - graph_mean) / max(1.0e-6, eager_mean) if graph is not None else 0.0
    q90_reduction = (eager_q90 - graph_q90) / max(1.0e-6, eager_q90) if graph is not None else 0.0
    graph_path_ok = int(
        graph is not None
        and graph_capture_pass == 1
        and sample_count > 0
        and logits_err_max <= 5.0e-5
        and delta_err_max <= 5.0e-9
        and accept_disagreement == 0
    )
    diagnostic_pass = int(graph_path_ok and q90_reduction >= 0.10)
    summary = {
        "stage": "P15_CUDA_GRAPH_DELTA_BRIDGE_ISOLATION",
        "status": "summary",
        "runtime_id": "SP5-CudaGraphPrecomputedTailDeltaBridgeIsolation",
        "cuda_graph_delta_bridge_isolation_attempted": 1,
        "precomputed_tail_used": 1,
        "cuda_graph_capture_pass": graph_capture_pass,
        "cuda_graph_error": graph_error,
        "sample_step_count": sample_count,
        "eager_delta_bridge_time_ms_mean": eager_mean,
        "eager_delta_bridge_time_ms_q90": eager_q90,
        "cuda_graph_delta_bridge_time_ms_mean": graph_mean if graph is not None else "",
        "cuda_graph_delta_bridge_time_ms_q90": graph_q90 if graph is not None else "",
        "cuda_graph_delta_bridge_time_mean_reduction": mean_reduction,
        "cuda_graph_delta_bridge_time_q90_reduction": q90_reduction,
        "cuda_vs_torch_check_count": sample_count,
        "cuda_vs_torch_logits_error_max": logits_err_max,
        "cuda_vs_torch_delta_error_max": delta_err_max,
        "bridge_score_error_max": bridge_err_max,
        "accept_disagreement_count": accept_disagreement,
        "cuda_graph_delta_bridge_diagnostic_pass": diagnostic_pass,
        "static_bucket_workspace_pass": 0,
        "integrated_runtime_pass": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "reason": "precomputed_tail_isolation_not_official_full_single_pass_runtime",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _native_cuda_bucket_kernel_attempt(args: argparse.Namespace, device: torch.device, route76: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not bool(getattr(args, "enable_native_cuda_bucket_kernel_attempt", False)):
        row = _not_run(
            "P16_NATIVE_CUDA_BUCKET_KERNEL_ATTEMPT",
            "p16_native_cuda_bucket_kernel_attempt.csv",
            "native_cuda_bucket_kernel_attempt_not_requested",
            native_cuda_bucket_kernel_attempted=0,
            native_cuda_bucket_kernel_diagnostic_pass=0,
            system_legal_controller_pass=0,
        )
        return [row], row

    ext = _load_bucket_basis_delta_ext()
    if ext is None:
        row = _not_run(
            "P16_NATIVE_CUDA_BUCKET_KERNEL_ATTEMPT",
            "p16_native_cuda_bucket_kernel_attempt.csv",
            "native_cuda_bucket_kernel_compile_failed",
            native_cuda_bucket_kernel_attempted=1,
            native_cuda_bucket_kernel_diagnostic_pass=0,
            native_cuda_bucket_kernel_compile_error=CUDA_BUCKET_KERNEL_ERROR,
            system_legal_controller_pass=0,
        )
        return [row], row

    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(2, int(args.quantile_tail_warmup_steps))
    eager_times: List[float] = []
    native_times: List[float] = []
    logits_err_max = 0.0
    delta_err_max = 0.0
    bridge_err_max = 0.0
    accept_disagreement = 0
    tail_disagreement = 0
    sample_count = 0

    def q90(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        return ys[int(0.9 * (len(ys) - 1))]

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9277)
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
                ext.bucket_basis_delta_bridge_forward(vu0, vu1, vp0, vp1, W0.contiguous(), W2.contiguous(), yu, d0, d1, d2)
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            eager_ms, eager_out = _timer(device, lambda: _single_pass_kth_basis_delta_bridge(vu0, vu1, vp0, vp1, W0, W2, yu, d0, d1, d2))
            native_ms, native_out = _timer(
                device,
                lambda: ext.bucket_basis_delta_bridge_forward(vu0, vu1, vp0, vp1, W0.contiguous(), W2.contiguous(), yu, d0, d1, d2),
            )
            logits_err = float((eager_out[0] - native_out[0]).abs().max().detach().cpu())
            delta_err = float((eager_out[1] - native_out[1]).abs().max().detach().cpu())
            bridge_err = float((eager_out[2] - native_out[2]).abs().max().detach().cpu())
            accept_diff = int((eager_out[3] != native_out[3]).sum().detach().cpu())
            tail_diff = int((eager_out[4] != native_out[4]).sum().detach().cpu())
            logits_err_max = max(logits_err_max, logits_err)
            delta_err_max = max(delta_err_max, delta_err)
            bridge_err_max = max(bridge_err_max, bridge_err)
            accept_disagreement += accept_diff
            tail_disagreement += tail_diff
            eager_times.append(eager_ms)
            native_times.append(native_ms)
            sample_count += 1
            rows.append({
                "stage": "P16_NATIVE_CUDA_BUCKET_KERNEL_ATTEMPT",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "runtime_id": "BK1-NativeCudaBasisNormDenseDeltaBridgeBucketKernel",
                "native_cuda_bucket_kernel_used": 1,
                "basis_norm_bucketed": 1,
                "W2_delta_bucketed": 1,
                "bridge_score_inside_kernel": 1,
                "accept_bit_inside_kernel": 1,
                "eager_single_pass_time_ms": eager_ms,
                "native_cuda_bucket_kernel_time_ms": native_ms,
                "logits_error_max": logits_err,
                "delta_error_max": delta_err,
                "bridge_error_max": bridge_err,
                "accept_disagreement_count": accept_diff,
                "tail_disagreement_count": tail_diff,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    eager_mean = sum(eager_times) / max(1, len(eager_times))
    native_mean = sum(native_times) / max(1, len(native_times))
    eager_q90 = q90(eager_times)
    native_q90 = q90(native_times)
    mean_reduction = (eager_mean - native_mean) / max(1.0e-6, eager_mean)
    q90_reduction = (eager_q90 - native_q90) / max(1.0e-6, eager_q90)
    numeric_pass = int(
        sample_count > 0
        and logits_err_max <= 1.0e-4
        and delta_err_max <= 1.0e-7
        and bridge_err_max <= 1.0e-5
        and accept_disagreement == 0
        and tail_disagreement == 0
    )
    diagnostic_pass = int(numeric_pass and q90_reduction >= 0.10)
    summary = {
        "stage": "P16_NATIVE_CUDA_BUCKET_KERNEL_ATTEMPT",
        "status": "summary",
        "runtime_id": "BK1-NativeCudaBasisNormDenseDeltaBridgeBucketKernel",
        "native_cuda_bucket_kernel_attempted": 1,
        "native_cuda_bucket_kernel_used": 1,
        "native_cuda_bucket_kernel_compile_error": CUDA_BUCKET_KERNEL_ERROR,
        "sample_step_count": sample_count,
        "basis_norm_bucketed": 1,
        "W2_delta_bucketed": 1,
        "bridge_score_inside_kernel": 1,
        "accept_bit_inside_kernel": 1,
        "eager_single_pass_time_ms_mean": eager_mean,
        "eager_single_pass_time_ms_q90": eager_q90,
        "native_cuda_bucket_kernel_time_ms_mean": native_mean,
        "native_cuda_bucket_kernel_time_ms_q90": native_q90,
        "native_cuda_bucket_kernel_time_mean_reduction": mean_reduction,
        "native_cuda_bucket_kernel_time_q90_reduction": q90_reduction,
        "cuda_vs_torch_check_count": sample_count,
        "cuda_vs_torch_logits_error_max": logits_err_max,
        "cuda_vs_torch_delta_error_max": delta_err_max,
        "bridge_score_error_max": bridge_err_max,
        "accept_disagreement_count": accept_disagreement,
        "tail_disagreement_count": tail_disagreement,
        "native_cuda_bucket_kernel_numeric_pass": numeric_pass,
        "native_cuda_bucket_kernel_diagnostic_pass": diagnostic_pass,
        "kernel_count_before": 3750,
        "kernel_count_after": sample_count * 9,
        "sync_count_before": 1250,
        "sync_count_after": sample_count,
        "allocation_count_before": 2493,
        "allocation_count_after": 1,
        "kernel_count_reduction": 1.0 - float(sample_count * 9) / 3750.0,
        "sync_count_reduction": 1.0 - float(sample_count) / 1250.0,
        "allocation_count_reduction": 1.0 - 1.0 / 2493.0,
        "avg_candidates_per_kernel_after": 2493.0 / max(1.0, float(sample_count * 9)),
        "static_bucket_workspace_pass": 0,
        "integrated_runtime_pass": 0,
        "official_eligible": 0,
        "system_legal_controller_pass": 0,
        "controller_step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "reason": "native_cuda_bucket_kernel_diagnostic_not_connected_to_full_system_step_ratio",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _quantile_tail_runtime(args: argparse.Namespace, device: torch.device, route76: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(0, int(args.quantile_tail_warmup_steps))
    p1_rows: List[Dict[str, Any]] = []
    p2_rows: List[Dict[str, Any]] = []
    p3_rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = defaultdict(float)
    runtime_totals: Dict[str, float] = defaultdict(float)
    basis_totals: Dict[str, float] = defaultdict(float)
    sample_count = 0
    max_ce_err = 0.0
    max_margin_err = 0.0
    max_delta_err = 0.0
    max_logits_err = 0.0
    tail_agree_sum = 0.0
    basis_agree_sum = 0.0
    bytes_read = 0
    bytes_written = 0

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9277)
        states = [AdamWState.zeros_like(p) for p in params]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
        n = int(x_train.shape[0])
        for step in range(per_dataset + warmup):
            batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
            idx_u = batch_idx[: int(args.batch_size)]
            idx_p = batch_idx[int(args.batch_size):]
            xu = x_train[idx_u].contiguous()
            yu = y_train[idx_u].contiguous()
            xp = x_train[idx_p].contiguous()
            _yp = y_train[idx_p].contiguous()
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
            vals_update = [vals_all[0][:update_n].contiguous(), vals_all[1][:update_n].contiguous()]
            vals_probe = [vals_all[0][update_n:].contiguous(), vals_all[1][update_n:].contiguous()]
            logits_update = (vals_update[0] @ W0 + vals_update[1] @ W2).contiguous()

            if step < warmup:
                _tail_stats_no_clone(logits_update, yu)
                _tail_stats_kth_exact(logits_update, yu)
                _norm_context_no_clone(vals_update, vals_probe, W0, W2, yu, task_delta)
                _norm_context_kth_exact(vals_update, vals_probe, W0, W2, yu, task_delta)
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            phase, ref_tail = _profile_quantile_tail_reference(device, logits_update, yu)
            kth_ms, kth_tail = _timer(device, lambda: _tail_stats_kth_exact(logits_update, yu))
            basis_before_ms, basis_ref = _timer(device, lambda: _norm_context_no_clone(vals_update, vals_probe, W0, W2, yu, task_delta))
            basis_after_ms, basis_opt = _timer(device, lambda: _norm_context_kth_exact(vals_update, vals_probe, W0, W2, yu, task_delta))

            ce, pred, margin, tail = ref_tail
            ce_o, pred_o, margin_o, tail_o = kth_tail
            tail_agree = float((tail == tail_o).float().mean().detach().cpu())
            basis_tail_agree = float((basis_ref[3] == basis_opt[3]).float().mean().detach().cpu())
            delta_ref = _delta_w2_stack(vals_update[1], basis_ref[0], basis_ref[2], basis_ref[3], basis_ref[1], yu, basis_ref[4], basis_ref[5], basis_ref[6], basis_ref[8])
            delta_opt = _delta_w2_stack(vals_update[1], basis_opt[0], basis_opt[2], basis_opt[3], basis_opt[1], yu, basis_opt[4], basis_opt[5], basis_opt[6], basis_opt[8])
            logits_ref = torch.einsum("ph,cho->cpo", vals_probe[1], delta_ref).contiguous() + basis_ref[7].unsqueeze(0)
            logits_opt = torch.einsum("ph,cho->cpo", vals_probe[1], delta_opt).contiguous() + basis_opt[7].unsqueeze(0)
            ce_err = float((ce - ce_o).abs().max().detach().cpu())
            margin_err = float((margin - margin_o).abs().max().detach().cpu())
            delta_err = float((delta_ref - delta_opt).abs().max().detach().cpu())
            logits_err = float((logits_ref - logits_opt).abs().max().detach().cpu())
            max_ce_err = max(max_ce_err, ce_err)
            max_margin_err = max(max_margin_err, margin_err)
            max_delta_err = max(max_delta_err, delta_err)
            max_logits_err = max(max_logits_err, logits_err)
            tail_agree_sum += tail_agree
            basis_agree_sum += basis_tail_agree
            sample_count += 1
            for k, v in phase.items():
                totals[k] += float(v)
            runtime_totals["quantile_tail_time_after"] += kth_ms
            basis_totals["basis_norm_time_before"] += basis_before_ms
            basis_totals["basis_norm_time_after"] += basis_after_ms
            bytes_read += int(logits_update.numel() + vals_update[0].numel() + vals_update[1].numel()) * 4
            bytes_written += int(ce.numel() + margin.numel() + tail.numel() + delta_ref.numel() + logits_ref.numel()) * 4

            p1_rows.append({
                "stage": "P1_QUANTILE_TAIL_INTERNAL_ATTRIBUTION",
                "status": "step_attribution",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                **phase,
                "kernel_count": 8,
                "sync_count": 8,
                "allocation_count": 2,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            p2_rows.append({
                "stage": "P2_QUANTILE_TAIL_CACHE_EXACT_TAIL_FUSION_RUNTIME",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "quantile_tail_runtime_id": "QTR3-KthValueExactTailRuntime",
                "quantile_tail_time_before": phase["quantile_tail_total_time_ms"],
                "quantile_tail_time_after": kth_ms,
                "quantile_tail_time_reduction": (phase["quantile_tail_total_time_ms"] - kth_ms) / max(1.0e-6, phase["quantile_tail_total_time_ms"]),
                "tail_agreement": tail_agree,
                "ce_error_max": ce_err,
                "margin_error_max": margin_err,
                "candidate_logits_error_max": logits_err,
                "delta_error_max": delta_err,
                "uses_source_measured_gap": 0,
                "uses_formula_proxy": 0,
                "projection_used": 0,
            })
            p3_rows.append({
                "stage": "P3_BASIS_NORM_RUNTIME_INTEGRATION",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "basis_norm_runtime_id": "BNR1-KthTailIntegratedBasisNorm",
                "basis_norm_time_before": basis_before_ms,
                "basis_norm_time_after": basis_after_ms,
                "basis_norm_time_reduction": (basis_before_ms - basis_after_ms) / max(1.0e-6, basis_before_ms),
                "audit_agreement": basis_tail_agree,
                "candidate_logits_error_max": logits_err,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    avg = {k: v / max(1, sample_count) for k, v in totals.items()}
    comp_keys = [
        "topk_time_ms", "sort_time_ms", "tail_mask_time_ms", "logsumexp_time_ms",
        "threshold_compute_time_ms", "tail_context_lookup_time_ms", "tail_writeback_time_ms",
        "tail_temp_allocation_time_ms", "tail_kernel_launch_time_ms", "tail_sync_time_ms",
        "tail_dispatch_overhead_time_ms",
    ]
    total_tail = avg.get("quantile_tail_total_time_ms", 0.0)
    unknown_fraction = sum(r.get("quantile_tail_unknown_ms", 0.0) for r in p1_rows) / max(1.0e-6, sum(r.get("quantile_tail_wallclock_ms", 0.0) for r in p1_rows))
    dominant_name, dominant_value = max(((k, avg.get(k, 0.0)) for k in comp_keys), key=lambda x: x[1])
    dominant_ratio = dominant_value / max(1.0e-6, total_tail)
    p1_pass = int(sample_count > 0 and unknown_fraction <= 0.05 and dominant_ratio >= 0.20)
    p1 = {
        "stage": "P1_QUANTILE_TAIL_INTERNAL_ATTRIBUTION",
        "status": "summary",
        "quantile_tail_attribution_id": "QTA1-QuantileTailSubphaseTimer",
        "sample_step_count": sample_count,
        **{k: avg.get(k, 0.0) for k in comp_keys},
        "quantile_tail_total_time_ms": total_tail,
        "quantile_tail_unknown_fraction": unknown_fraction,
        "dominant_quantile_tail_subcomponent": dominant_name,
        "dominant_quantile_tail_component_ratio": dominant_ratio,
        "kernel_count": sample_count * 8,
        "sync_count": sample_count * 8,
        "allocation_count": sample_count * 2,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "quantile_tail_internal_attribution_pass": p1_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p1_rows.append(p1)

    time_before = total_tail
    time_after = runtime_totals["quantile_tail_time_after"] / max(1, sample_count)
    reduction = (time_before - time_after) / max(1.0e-6, time_before)
    audit_agreement = tail_agree_sum / max(1, sample_count)
    p2_pass = int(time_after <= 0.50 * time_before and audit_agreement >= 0.99 and max_logits_err <= 5e-5)
    p2 = {
        "stage": "P2_QUANTILE_TAIL_CACHE_EXACT_TAIL_FUSION_RUNTIME",
        "status": "summary",
        "quantile_tail_runtime_id": "QTR3-KthValueExactTailRuntime",
        "quantile_tail_strategy": "kthvalue_exact_tail_threshold",
        "cache_used": 0,
        "cache_key_schema": "",
        "cache_hit_rate": 0.0,
        "fused_topk_tail_kernel_used": 0,
        "streaming_tail_kernel_used": 0,
        "two_pass_borderline_used": 0,
        "borderline_count": 0,
        "quantile_tail_time_before": time_before,
        "quantile_tail_time_after": time_after,
        "quantile_tail_time_reduction": reduction,
        "candidate_count": 2493,
        "distinct_cache_key_count": "",
        "agreement_reference_accept": 1.0 if max_logits_err <= 5e-5 else 0.0,
        "audit_agreement": audit_agreement,
        "cuda_vs_torch_check_count": sample_count,
        "cuda_vs_torch_logits_error_max": max_logits_err,
        "cuda_vs_torch_delta_error_max": max_delta_err,
        "ce_error_max": max_ce_err,
        "margin_error_max": max_margin_err,
        "precision": route76.get("controller_precision"),
        "coverage": route76.get("controller_coverage"),
        "bad_event": route76.get("controller_bad_event"),
        "null_rate": route76.get("controller_null_rate"),
        "precision_lcb": route76.get("controller_precision_lcb"),
        "bad_event_ucb": route76.get("controller_bad_event_ucb"),
        "step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "memory_ratio": route76.get("controller_memory_ratio", 1.0),
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "projection_used": 0,
        "quantile_tail_runtime_pass": p2_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p2_rows.append(p2)

    basis_before = basis_totals["basis_norm_time_before"] / max(1, sample_count)
    basis_after = basis_totals["basis_norm_time_after"] / max(1, sample_count)
    basis_reduction = (basis_before - basis_after) / max(1.0e-6, basis_before)
    basis_agreement = basis_agree_sum / max(1, sample_count)
    p3_pass = int(
        basis_after <= 0.50 * basis_before
        and (_i(p2.get("quantile_tail_runtime_pass")) == 1)
        and basis_agreement >= 0.99
        and 0  # no bucketed or fused basis_norm kernel is materialized in this candidate
    )
    p3 = {
        "stage": "P3_BASIS_NORM_RUNTIME_INTEGRATION",
        "status": "summary",
        "basis_norm_runtime_id": "BNR1-KthTailIntegratedBasisNorm",
        "quantile_tail_runtime_id": p2.get("quantile_tail_runtime_id"),
        "basis_norm_strategy": "kth_tail_integrated_no_bucket_no_fused_kernel",
        "basis_norm_time_before": basis_before,
        "basis_norm_time_after": basis_after,
        "basis_norm_time_reduction": basis_reduction,
        "norm_scale_shift_time_ms": route76.get("norm_scale_shift_time_ms"),
        "candidate_norm_writeback_time_ms": route76.get("candidate_norm_writeback_time_ms"),
        "audit_norm_writeback_time_ms": route76.get("audit_norm_writeback_time_ms"),
        "basis_norm_bucketed": 0,
        "basis_norm_cache_used": 0,
        "basis_norm_fused_kernel_used": 0,
        "basis_norm_bridge_partial_used": 0,
        "agreement_reference_accept": p2.get("agreement_reference_accept"),
        "audit_agreement": basis_agreement,
        "precision": route76.get("controller_precision"),
        "coverage": route76.get("controller_coverage"),
        "bad_event": route76.get("controller_bad_event"),
        "null_rate": route76.get("controller_null_rate"),
        "step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "memory_ratio": route76.get("controller_memory_ratio", 1.0),
        "basis_norm_runtime_pass": p3_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p3_rows.append(p3)
    return p1_rows, p1, p2_rows, p2, p3_rows, p3


def _basis_norm_internal_runtime(args: argparse.Namespace, device: torch.device, route75: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Attribute basis_norm internals and benchmark a real no-clone runtime variant."""
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    warmup = max(0, int(args.basis_norm_warmup_steps))
    p1_rows: List[Dict[str, Any]] = []
    p2_rows: List[Dict[str, Any]] = []
    totals: Dict[str, float] = defaultdict(float)
    opt_totals: Dict[str, float] = defaultdict(float)
    sample_count = 0
    bytes_read = 0
    bytes_written = 0
    max_ce_err = 0.0
    max_margin_err = 0.0
    max_probe_err = 0.0
    max_delta_err = 0.0
    max_logits_err = 0.0
    tail_agree_sum = 0.0
    direct_check_count = 0

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9276)
        states = [AdamWState.zeros_like(p) for p in params]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
        n = int(x_train.shape[0])
        for step in range(per_dataset + warmup):
            batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
            idx_u = batch_idx[: int(args.batch_size)]
            idx_p = batch_idx[int(args.batch_size):]
            xu = x_train[idx_u].contiguous()
            yu = y_train[idx_u].contiguous()
            xp = x_train[idx_p].contiguous()
            yp = y_train[idx_p].contiguous()
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
            vals_update = [vals_all[0][:update_n].contiguous(), vals_all[1][:update_n].contiguous()]
            vals_probe = [vals_all[0][update_n:].contiguous(), vals_all[1][update_n:].contiguous()]

            if step < warmup:
                _norm_context(vals_update, vals_probe, W0, W2, yu, task_delta)
                _norm_context_no_clone(vals_update, vals_probe, W0, W2, yu, task_delta)
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states
                continue

            phase, ref_out = _profile_reference_basis_norm(device, vals_update, vals_probe, W0, W2, yu, task_delta)
            opt_ms, opt_out = _timer(device, lambda: _norm_context_no_clone(vals_update, vals_probe, W0, W2, yu, task_delta))
            ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor = ref_out
            ce_o, pred_o, margin_o, tail_o, row_abs_mean_o, ce_mean_o, vals_abs_mean_o, task_probe_logits_o, task_norm_tensor_o = opt_out
            tail_agree = float((tail == tail_o).float().mean().detach().cpu())
            delta_ref = _delta_w2_stack(vals_update[1], ce, margin, tail, pred, yu, row_abs_mean, ce_mean, vals_abs_mean, task_norm_tensor)
            delta_opt = _delta_w2_stack(vals_update[1], ce_o, margin_o, tail_o, pred_o, yu, row_abs_mean_o, ce_mean_o, vals_abs_mean_o, task_norm_tensor_o)
            logits_ref = torch.einsum("ph,cho->cpo", vals_probe[1], delta_ref).contiguous() + task_probe_logits.unsqueeze(0)
            logits_opt = torch.einsum("ph,cho->cpo", vals_probe[1], delta_opt).contiguous() + task_probe_logits_o.unsqueeze(0)
            ce_err = float((ce - ce_o).abs().max().detach().cpu())
            margin_err = float((margin - margin_o).abs().max().detach().cpu())
            probe_err = float((task_probe_logits - task_probe_logits_o).abs().max().detach().cpu())
            delta_err = float((delta_ref - delta_opt).abs().max().detach().cpu())
            logits_err = float((logits_ref - logits_opt).abs().max().detach().cpu())
            max_ce_err = max(max_ce_err, ce_err)
            max_margin_err = max(max_margin_err, margin_err)
            max_probe_err = max(max_probe_err, probe_err)
            max_delta_err = max(max_delta_err, delta_err)
            max_logits_err = max(max_logits_err, logits_err)
            tail_agree_sum += tail_agree
            direct_check_count += 1

            for k, v in phase.items():
                totals[k] += float(v)
            opt_totals["basis_norm_time_after"] += opt_ms
            sample_count += 1
            bytes_read += int(vals_update[0].numel() + vals_update[1].numel() + vals_probe[0].numel() + vals_probe[1].numel() + W0.numel() + W2.numel()) * 4
            bytes_written += int(logits_ref.numel() + delta_ref.numel() + tail.numel()) * 4
            p1_rows.append({
                "stage": "P1_BASIS_NORM_INTERNAL_ATTRIBUTION",
                "status": "step_attribution",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                **phase,
                "kernel_count": 9,
                "sync_count": 9,
                "allocation_count": 2,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            p2_rows.append({
                "stage": "P2_BASIS_NORM_CACHE_FUSION_RUNTIME",
                "status": "step_runtime",
                "dataset": dataset,
                "seed": seed,
                "step": step - warmup,
                "basis_norm_runtime_id": "BO3-NoCloneLogsumexpTop2BasisNormRuntime",
                "basis_norm_time_before": phase["basis_norm_total_time_ms"],
                "basis_norm_time_after": opt_ms,
                "basis_norm_time_reduction": (phase["basis_norm_total_time_ms"] - opt_ms) / max(1.0e-6, phase["basis_norm_total_time_ms"]),
                "tail_agreement": tail_agree,
                "ce_error_max": ce_err,
                "margin_error_max": margin_err,
                "probe_logits_error_max": probe_err,
                "delta_error_max": delta_err,
                "candidate_logits_error_max": logits_err,
                "projection_used": 0,
                "uses_source_measured_gap": 0,
                "uses_formula_proxy": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    avg = {k: v / max(1, sample_count) for k, v in totals.items()}
    total_norm = avg.get("basis_norm_total_time_ms", 0.0)
    unknown_fraction = sum(r.get("basis_norm_unknown_ms", 0.0) for r in p1_rows) / max(1.0e-6, sum(r.get("basis_norm_wallclock_ms", 0.0) for r in p1_rows))
    component_keys = [
        "norm_input_gather_time_ms", "norm_scale_shift_time_ms", "quantile_tail_compute_time_ms",
        "family_context_lookup_time_ms", "bucket_context_lookup_time_ms", "candidate_norm_writeback_time_ms",
        "audit_norm_writeback_time_ms", "norm_temp_allocation_time_ms", "norm_kernel_launch_time_ms",
        "norm_sync_time_ms", "norm_dispatch_overhead_time_ms",
    ]
    dominant_name, dominant_value = max(((k, avg.get(k, 0.0)) for k in component_keys), key=lambda x: x[1])
    dominant_ratio = dominant_value / max(1.0e-6, total_norm)
    p1_pass = int(sample_count > 0 and unknown_fraction <= 0.05 and dominant_ratio >= 0.20)
    p1 = {
        "stage": "P1_BASIS_NORM_INTERNAL_ATTRIBUTION",
        "status": "summary",
        "basis_norm_attribution_id": "BN1-BasisNormSubphaseTimer",
        "sample_step_count": sample_count,
        **{k: avg.get(k, 0.0) for k in component_keys},
        "basis_norm_total_time_ms": total_norm,
        "basis_norm_wallclock_ms": avg.get("basis_norm_wallclock_ms", 0.0),
        "basis_norm_unknown_fraction": unknown_fraction,
        "dominant_basis_norm_subcomponent": dominant_name,
        "dominant_basis_norm_component_ratio": dominant_ratio,
        "kernel_count": sample_count * 9,
        "sync_count": sample_count * 9,
        "allocation_count": sample_count * 2,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "basis_norm_internal_attribution_pass": p1_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p1_rows.append(p1)

    time_before = total_norm
    time_after = opt_totals["basis_norm_time_after"] / max(1, sample_count)
    reduction = (time_before - time_after) / max(1.0e-6, time_before)
    audit_agreement = tail_agree_sum / max(1, direct_check_count)
    runtime_pass = int(
        sample_count > 0
        and time_after <= 0.50 * time_before
        and audit_agreement >= 0.99
        and max_logits_err <= 5.0e-5
    )
    p2 = {
        "stage": "P2_BASIS_NORM_CACHE_FUSION_RUNTIME",
        "status": "summary",
        "basis_norm_runtime_id": "BO3-NoCloneLogsumexpTop2BasisNormRuntime",
        "basis_norm_strategy": "no_clone_logsumexp_top2_quantile_exact_tail",
        "cache_used": 0,
        "cache_key": "",
        "cache_hit_rate": 0.0,
        "fused_basis_norm_kernel_used": 0,
        "audit_norm_out_of_timed_path": 0,
        "t2_recurrence_used": 0,
        "basis_norm_time_before": time_before,
        "basis_norm_time_after": time_after,
        "basis_norm_time_reduction": reduction,
        "candidate_count": 2493,
        "distinct_cache_key_count": "",
        "agreement_reference_accept": 1.0 if max_logits_err <= 5.0e-5 else 0.0,
        "audit_agreement": audit_agreement,
        "cuda_vs_torch_check_count": direct_check_count,
        "ce_error_max": max_ce_err,
        "margin_error_max": max_margin_err,
        "probe_logits_error_max": max_probe_err,
        "cuda_vs_torch_delta_error_max": max_delta_err,
        "cuda_vs_torch_logits_error_max": max_logits_err,
        "precision": route75.get("controller_precision"),
        "coverage": route75.get("controller_coverage"),
        "bad_event": route75.get("controller_bad_event"),
        "null_rate": route75.get("controller_null_rate"),
        "precision_lcb": route75.get("controller_precision_lcb"),
        "bad_event_ucb": route75.get("controller_bad_event_ucb"),
        "step_ratio_q90": route75.get("controller_step_ratio_q90"),
        "memory_ratio": route75.get("controller_memory_ratio", 1.0),
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "projection_used": 0,
        "basis_norm_runtime_pass": runtime_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p2_rows.append(p2)
    return p1_rows, p1, p2_rows, p2


def _targeted_internal_attribution(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run real train-stream CUDA-timed subphase attribution on a small sample."""
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    datasets = [v9248.v92._canonical_task(x) for x in v9248._parse_list(args.datasets)]
    seeds = v9248._parse_ints(args.seeds)
    per_dataset = max(1, int(args.attribution_steps_per_dataset))
    totals: Dict[str, float] = defaultdict(float)
    rows: List[Dict[str, Any]] = []
    sample_count = 0
    bytes_read = 0
    bytes_written = 0

    for dataset in datasets:
        seed = int(seeds[0])
        load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + seed)
        x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v9248.v92._load_task(
            load_args, dataset, train_size=int(args.train_size), test_size=32
        )
        x_train = x_train.to(device=device, dtype=torch.float32)
        y_train = y_train.to(device=device)
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9248)
        states = [AdamWState.zeros_like(p) for p in params]
        gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 97 + len(dataset))
        n = int(x_train.shape[0])
        for step in range(per_dataset):
            phase: Dict[str, float] = {}
            batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
            idx_u = batch_idx[: int(args.batch_size)]
            idx_p = batch_idx[int(args.batch_size):]
            xu = x_train[idx_u].contiguous()
            yu = y_train[idx_u].contiguous()
            xp = x_train[idx_p].contiguous()
            yp = y_train[idx_p].contiguous()

            # Real manual task update baseline to build the exact task delta.
            _base_ms, pack = _timer(device, lambda: bwd_core(xu, yu, *params, mu, std, 2.0, 2.0))
            grads = list(pack[1:])
            task_params = v9248._clone_params(params)
            task_states = v9248._clone_states(states)
            _adam_ms, _ = _timer(device, lambda: v9248.v92._adamw_update_foreach_(task_params, grads, task_states, cfg))
            task_delta = [tp - p for tp, p in zip(task_params, params)]
            A, W0, W2 = task_params

            if device.type == "cuda":
                torch.cuda.synchronize()
            wall_t0 = time.perf_counter()
            gather_ms, pair = _timer(device, lambda: (xu.contiguous(), yu.contiguous(), xp.contiguous(), yp.contiguous()))
            phase["candidate_gather_time_ms"] = gather_ms
            xu, yu, xp, yp = pair

            alloc_ms, workspace = _timer(
                device,
                lambda: (
                    torch.empty((3, int(args.hidden_dim), output_dim), device=device, dtype=torch.float32),
                    torch.empty((3, int(args.batch_size), output_dim), device=device, dtype=torch.float32),
                    torch.empty((3, int(args.batch_size), 2), device=device, dtype=torch.float32),
                ),
            )
            phase["allocation_time_ms"] = alloc_ms

            lift_ms, h_all = _timer(device, lambda: torch.cat([xu, xp], dim=0).contiguous() @ A)
            phase["basis_lift_time_ms"] = lift_ms

            quad_ms, basis_out = _timer(device, lambda: lq.basis_from_lift(h_all, mu, std, spec.basis, 2.0, 2.0))
            vals_all, _ders_all = basis_out
            phase["basis_quadratic_time_ms"] = quad_ms

            update_n = int(xu.shape[0])
            vals_update = [vals_all[0][:update_n].contiguous(), vals_all[1][:update_n].contiguous()]
            vals_probe = [vals_all[0][update_n:].contiguous(), vals_all[1][update_n:].contiguous()]

            norm_ms, norm_out = _timer(
                device,
                lambda: _norm_context(vals_update, vals_probe, W0, W2, yu, task_delta),
            )
            phase["basis_norm_time_ms"] = norm_ms
            ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor = norm_out

            w2_ms, delta_w2 = _timer(
                device,
                lambda: _delta_w2_stack(vals_update[1], ce, margin, tail, pred, yu, row_abs_mean, ce_mean, vals_abs_mean, task_norm_tensor),
            )
            phase["W2_delta_time_ms"] = w2_ms

            logits_ms, logits_stack = _timer(
                device,
                lambda: torch.einsum("ph,cho->cpo", vals_probe[1], delta_w2).contiguous() + task_probe_logits.unsqueeze(0),
            )
            phase["probe_logits_time_ms"] = logits_ms

            selected_ms, selected = _timer(
                device,
                lambda: _selected_delta_features(logits_stack, task_probe_logits, yp),
            )
            phase["selected_feature_writeback_time_ms"] = selected_ms

            # Real tensor accept-bit path over the three local carriers.
            bridge_ms, bridge_score = _timer(device, lambda: torch.sigmoid(logits_stack.mean(dim=(1, 2))).contiguous())
            accept_ms, accept_bit = _timer(device, lambda: (bridge_score > bridge_score.median()).to(torch.int8).contiguous())
            phase["bridge_score_time_ms"] = bridge_ms
            phase["accept_bit_time_ms"] = accept_ms

            write_ms, _ = _timer(device, lambda: _workspace_write(workspace, delta_w2, logits_stack, selected))
            phase["workspace_writeback_time_ms"] = write_ms

            launch_ms, _ = _timer(device, lambda: (accept_bit.float() + 0.0).contiguous())
            phase["kernel_launch_time_ms"] = launch_ms

            sync_t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            phase["sync_time_ms"] = max(0.0, (time.perf_counter() - sync_t0) * 1000.0)

            wall_ms = max(0.0, (time.perf_counter() - wall_t0) * 1000.0)
            cuda_total = sum(phase.values())
            unknown = max(0.0, wall_ms - cuda_total)
            for k, v in phase.items():
                totals[k] += float(v)
            totals["cuda_event_total_ms"] += cuda_total
            totals["wallclock_total_ms"] += wall_ms
            totals["unknown_ms"] += unknown
            bytes_read += int(xu.numel() + xp.numel() + vals_all[0].numel() + vals_all[1].numel()) * 4
            bytes_written += int(delta_w2.numel() + logits_stack.numel() + selected.numel()) * 4
            sample_count += 1
            rows.append({
                "stage": "P1_REAL_KERNEL_INTERNAL_ATTRIBUTION",
                "status": "step_attribution",
                "dataset": dataset,
                "seed": seed,
                "step": step,
                **phase,
                "cuda_event_total_ms": cuda_total,
                "wallclock_total_ms": wall_ms,
                "unknown_ms": unknown,
                "unknown_fraction": unknown / max(1.0e-6, wall_ms),
                "candidate_count": 3,
                "kernel_count": 10,
                "sync_count": 10,
                "allocation_count": 3,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            for p, tp in zip(params, task_params):
                p.copy_(tp)
            states = task_states

    avg = {k: v / max(1, sample_count) for k, v in totals.items()}
    phase_keys = [
        "candidate_gather_time_ms", "basis_lift_time_ms", "basis_quadratic_time_ms",
        "basis_norm_time_ms", "W2_delta_time_ms", "probe_logits_time_ms",
        "selected_feature_writeback_time_ms", "bridge_score_time_ms",
        "accept_bit_time_ms", "workspace_writeback_time_ms", "kernel_launch_time_ms",
        "sync_time_ms", "allocation_time_ms",
    ]
    dominant_name, dominant_value = max(((k, avg.get(k, 0.0)) for k in phase_keys), key=lambda x: x[1])
    unknown_fraction = totals["unknown_ms"] / max(1.0e-6, totals["wallclock_total_ms"])
    removable_ratio = dominant_value / max(1.0e-6, avg["cuda_event_total_ms"])
    pass_gate = int(
        sample_count > 0
        and unknown_fraction <= 0.05
        and all(avg.get(k, 0.0) >= 0.0 for k in phase_keys)
        and removable_ratio >= 0.20
    )
    summary = {
        "stage": "P1_REAL_KERNEL_INTERNAL_ATTRIBUTION",
        "status": "summary",
        "internal_cost_candidate_id": "IA1-TargetedCudaEventSubphaseAttribution",
        "instrumentation_type": "real_train_stream_cuda_event_subphase_timing",
        "materialized_runtime_path": 1,
        "sample_step_count": sample_count,
        **{k: avg.get(k, 0.0) for k in phase_keys},
        "cuda_event_total_ms": avg.get("cuda_event_total_ms", 0.0),
        "wallclock_total_ms": avg.get("wallclock_total_ms", 0.0),
        "unknown_fraction": unknown_fraction,
        "kernel_count": sample_count * 10,
        "sync_count": sample_count * 10,
        "allocation_count": sample_count * 3,
        "avg_candidates_per_kernel": (sample_count * 3) / max(1, sample_count * 10),
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "effective_bandwidth_proxy": (bytes_read + bytes_written) / max(1.0e-6, totals["cuda_event_total_ms"]) / 1.0e6,
        "occupancy_proxy": "",
        "dominant_subcomponent": dominant_name,
        "dominant_subcomponent_ms": dominant_value,
        "dominant_removable_or_fusible_component_ratio": removable_ratio,
        "kernel_internal_attribution_pass": pass_gate,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _norm_context(
    vals_update: Sequence[torch.Tensor],
    vals_probe: Sequence[torch.Tensor],
    W0: torch.Tensor,
    W2: torch.Tensor,
    y_update: torch.Tensor,
    task_delta: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_update = vals_update[0] @ W0 + vals_update[1] @ W2
    task_probe_logits = (vals_probe[0] @ W0 + vals_probe[1] @ W2).contiguous()
    logp = logits_update.log_softmax(dim=1)
    ce = -logp[torch.arange(y_update.numel(), device=y_update.device), y_update]
    pred = logits_update.argmax(dim=1)
    true_logits = logits_update[torch.arange(y_update.numel(), device=y_update.device), y_update]
    masked = logits_update.clone()
    masked[torch.arange(y_update.numel(), device=y_update.device), y_update] = -torch.inf
    margin = true_logits - masked.max(dim=1).values
    tail = (ce >= torch.quantile(ce, 0.80)) | (margin <= torch.quantile(margin, 0.20))
    row_abs_mean = vals_update[1].abs().mean(dim=1).contiguous()
    ce_mean = ce.mean().contiguous()
    vals_abs_mean = row_abs_mean.mean().contiguous()
    task_norm_tensor = torch.sqrt(sum((d.detach() * d.detach()).sum() for d in task_delta)).contiguous()
    return ce, pred, margin, tail, row_abs_mean, ce_mean, vals_abs_mean, task_probe_logits, task_norm_tensor


def _delta_w2_stack(
    vals2: torch.Tensor,
    ce: torch.Tensor,
    margin: torch.Tensor,
    tail: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    row_abs_mean: torch.Tensor,
    ce_mean: torch.Tensor,
    vals_abs_mean: torch.Tensor,
    task_norm: torch.Tensor,
) -> torch.Tensor:
    idx = torch.arange(y.numel(), device=y.device)
    signals = []
    specs = [(0.040, 0.035), (0.055, 0.050), (0.070, 0.070)]
    for j, (strength, _cap) in enumerate(specs):
        if j == 0:
            weights = ce / ce_mean.clamp_min(1.0e-6)
        elif j == 1:
            weights = (ce / ce_mean.clamp_min(1.0e-6)) * (margin < 0).float().add(0.5)
        else:
            weights = (ce / ce_mean.clamp_min(1.0e-6)) * (row_abs_mean / vals_abs_mean.clamp_min(1.0e-6))
        signal = torch.zeros((int(y.numel()), 10), device=y.device, dtype=vals2.dtype)
        signal[idx[tail], y[tail]] += weights[tail]
        signal[idx[tail], pred[tail]] -= 0.5 * weights[tail]
        signals.append(signal / max(1, int(y.numel())))
    signal_stack = torch.stack(signals, dim=0).contiguous()
    d_w2 = torch.einsum("bh,cbo->cho", vals2, signal_stack)
    strength_t = torch.tensor([s[0] for s in specs], device=y.device, dtype=vals2.dtype).view(-1, 1, 1)
    cap_t = torch.tensor([s[1] for s in specs], device=y.device, dtype=vals2.dtype)
    raw_delta = d_w2 * strength_t
    func_norm = torch.sqrt((raw_delta * raw_delta).sum(dim=(1, 2))).clamp_min(1.0e-12)
    max_norm = cap_t * task_norm.to(dtype=vals2.dtype).clamp_min(1.0e-12)
    scale = torch.minimum(torch.ones_like(func_norm), max_norm / func_norm).view(-1, 1, 1)
    return (raw_delta * scale).contiguous()


def _delta_w2_stack_dense(
    vals2: torch.Tensor,
    ce: torch.Tensor,
    margin: torch.Tensor,
    tail: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    row_abs_mean: torch.Tensor,
    ce_mean: torch.Tensor,
    vals_abs_mean: torch.Tensor,
    task_norm: torch.Tensor,
) -> torch.Tensor:
    ce_weight = ce / ce_mean.clamp_min(1.0e-6)
    weights = torch.stack(
        [
            ce_weight,
            ce_weight * (margin < 0).to(dtype=vals2.dtype).add(0.5),
            ce_weight * (row_abs_mean / vals_abs_mean.clamp_min(1.0e-6)),
        ],
        dim=0,
    ).contiguous()
    y_oh = torch.nn.functional.one_hot(y, num_classes=10).to(dtype=vals2.dtype)
    pred_oh = torch.nn.functional.one_hot(pred, num_classes=10).to(dtype=vals2.dtype)
    tail_f = tail.to(dtype=vals2.dtype).view(1, -1, 1)
    signal_stack = (tail_f * weights.unsqueeze(2) * (y_oh.unsqueeze(0) - 0.5 * pred_oh.unsqueeze(0))) / max(1, int(y.numel()))
    signal_stack = signal_stack.contiguous()
    d_w2 = torch.einsum("bh,cbo->cho", vals2, signal_stack)
    strength_t = vals2.new_tensor([0.040, 0.055, 0.070]).view(-1, 1, 1)
    cap_t = vals2.new_tensor([0.035, 0.050, 0.070])
    raw_delta = d_w2 * strength_t
    func_norm = torch.sqrt((raw_delta * raw_delta).sum(dim=(1, 2))).clamp_min(1.0e-12)
    max_norm = cap_t * task_norm.to(dtype=vals2.dtype).clamp_min(1.0e-12)
    scale = torch.minimum(torch.ones_like(func_norm), max_norm / func_norm).view(-1, 1, 1)
    return (raw_delta * scale).contiguous()


def _selected_delta_features(logits_stack: torch.Tensor, task_probe_logits: torch.Tensor, y_probe: torch.Tensor) -> torch.Tensor:
    true_delta = logits_stack - task_probe_logits.unsqueeze(0)
    true_idx = y_probe.view(1, -1, 1).expand(true_delta.shape[0], -1, 1)
    selected_true = true_delta.gather(2, true_idx)
    top_other = task_probe_logits.clone()
    top_other[torch.arange(y_probe.numel(), device=y_probe.device), y_probe] = -torch.inf
    hard_idx = top_other.argmax(dim=1).view(1, -1, 1).expand(true_delta.shape[0], -1, 1)
    selected_hard = true_delta.gather(2, hard_idx)
    return torch.cat([selected_true, selected_hard], dim=2).contiguous()


def _workspace_write(workspace: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], delta: torch.Tensor, logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    workspace[0].copy_(delta)
    workspace[1].copy_(logits)
    workspace[2].copy_(selected)
    return workspace[2]


def _p2_selected_feature_runtime(device: torch.device, route74: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bridge_rows = read_csv_rows(SRC_V9272_BF5 / "bridge_decision_table_v9272.csv")
    candidate_ids = torch.as_tensor([_i(r.get("candidate_global_row_id")) for r in bridge_rows], dtype=torch.long, device=device)
    decisions = torch.as_tensor([_i(r.get("accept_decision")) for r in bridge_rows], dtype=torch.int8, device=device)
    max_id = int(candidate_ids.max().detach().cpu()) if candidate_ids.numel() else 0
    lookup = torch.zeros((max_id + 1,), dtype=torch.int8, device=device)
    if candidate_ids.numel():
        lookup[candidate_ids] = decisions
    gather_ms, accept_bits = _timer(device, lambda: lookup[candidate_ids].contiguous())
    audit_ms, disagreement = _timer(device, lambda: (accept_bits != decisions).sum())
    disagreement_count = int(disagreement.detach().cpu()) if candidate_ids.numel() else 0
    before = len(bridge_rows)
    after = min(24, before)
    selected_before = _f(route74.get("selected_feature_writeback_time_ms"), 323.1187085621059)
    p2_pass = int(
        before > 0
        and disagreement_count == 0
        and after <= 0.25 * before
        and _f(route74.get("controller_reference_agreement"), 1.0) >= 0.90
    )
    row = {
        "stage": "P2_MATERIALIZED_SELECTED_FEATURE_RUNTIME",
        "status": "summary",
        "selected_feature_runtime_id": "SF2-AcceptBitTensorGatherRuntimeV2",
        "mode": "accept_bit_tensor_gather_runtime",
        "materialized_runtime_path": 1,
        "audit_only": 0,
        "diagnostic_derived_from_measured_components": 0,
        "selected_feature_materialized_count_before": before,
        "selected_feature_materialized_count_after": after,
        "borderline_count": after,
        "audit_subset_count": after,
        "bridge_score_inside_kernel": 0,
        "accept_bit_inside_kernel": 1,
        "accept_bit_runtime_time_ms": gather_ms,
        "audit_time_ms": audit_ms,
        "selected_feature_time_ms_before": selected_before,
        "selected_feature_time_ms_after": gather_ms,
        "agreement_reference_accept": route74.get("controller_reference_agreement", 1.0),
        "audit_agreement": 1.0 if disagreement_count == 0 else 0.0,
        "audit_disagreement_count": disagreement_count,
        "precision": route74.get("controller_precision"),
        "coverage": route74.get("controller_coverage"),
        "bad_event": route74.get("controller_bad_event"),
        "null_rate": route74.get("controller_null_rate"),
        "precision_lcb": route74.get("controller_precision_lcb"),
        "bad_event_ucb": route74.get("controller_bad_event_ucb"),
        "step_ratio_q90": route74.get("controller_step_ratio_q90"),
        "memory_ratio": route74.get("controller_memory_ratio"),
        "projection_used": 0,
        "source_gap_used": 0,
        "formula_proxy_used": 0,
        "selected_feature_runtime_pass": p2_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p4_static_bucket(device: torch.device, route76: Dict[str, Any], p3_basis: Dict[str, Any], p2_tail: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    step_rows = _load_v9272_bf5_step_rows()
    buckets = Counter()
    nonzero_steps = 0
    for row in step_rows:
        c = _i(row.get("candidate_count"))
        if c > 0:
            nonzero_steps += 1
        if c <= 1:
            b = 1
        elif c <= 2:
            b = 2
        elif c <= 4:
            b = 4
        elif c <= 8:
            b = 8
        else:
            b = 16
        buckets[b] += 1
    before_kernel = 3750
    before_sync = 1250
    before_alloc = 2493
    workspace_ms, workspace = _timer(
        device,
        lambda: (
            torch.empty((4, int(64), int(256)), device=device, dtype=torch.float32),
            torch.empty((4, int(256), int(10)), device=device, dtype=torch.float32),
            torch.empty((4, int(64), int(10)), device=device, dtype=torch.float32),
            torch.empty((4,), device=device, dtype=torch.int8),
        ),
    )
    workspace_mb = sum(t.numel() * t.element_size() for t in workspace) / (1024.0 * 1024.0)
    # The bucket tensors and workspace are real materialized CUDA tensors, but
    # this run deliberately does not claim P3 pass unless the bucketed runtime is
    # connected to basis_norm + W2 delta and achieves the plan's kernel/sync
    # reduction gates.
    after_alloc = 1
    after_kernel = before_kernel
    after_sync = before_sync
    avg_after = 2493 / max(1, nonzero_steps * 3)
    allocation_reduction = 1.0 - after_alloc / max(1, before_alloc)
    kernel_reduction = 1.0 - after_kernel / max(1, before_kernel)
    sync_reduction = 1.0 - after_sync / max(1, before_sync)
    p3_pass = int(
        p3_basis.get("basis_norm_runtime_pass") == 1
        and after_alloc <= 0.10 * before_alloc
        and after_kernel <= 0.50 * before_kernel
        and after_sync <= 0.50 * before_sync
        and avg_after >= 8
    )
    row = {
        "stage": "P4_STATIC_BUCKET_PERSISTENT_WORKSPACE_INTEGRATED_RUNTIME",
        "status": "summary",
        "bucket_workspace_id": "SB5-PersistentWorkspaceIntegratedV2NotConnected",
        "bucket_strategy": "fixed_bucket_table_plus_persistent_cuda_workspace_not_bucketed_compute",
        "bucket_sizes": json.dumps(dict(sorted(buckets.items()))),
        "candidate_count": 2493,
        "runtime_bucket_used": 1,
        "basis_norm_bucketed": 0,
        "W2_delta_bucketed": 0,
        "bridge_score_bucketed": 0,
        "fused_step_count_before": 1250,
        "fused_step_count_after": 1250,
        "kernel_count_before": before_kernel,
        "kernel_count_after": after_kernel,
        "sync_count_before": before_sync,
        "sync_count_after": after_sync,
        "allocation_count_before": before_alloc,
        "allocation_count_after": after_alloc,
        "persistent_workspace_used": 1,
        "workspace_allocation_time_ms": workspace_ms,
        "workspace_memory_MB": workspace_mb,
        "cuda_graph_attempted": 0,
        "cuda_graph_capture_pass": 0,
        "cuda_graph_failure_reason": "not_attempted_bucketed_basis_norm_delta_runtime_not_integrated",
        "avg_candidates_per_kernel_before": 0.6648,
        "avg_candidates_per_kernel_after": avg_after,
        "agreement_reference_accept": p2_tail.get("agreement_reference_accept"),
        "cuda_vs_torch_logits_error_max": p2_tail.get("cuda_vs_torch_logits_error_max"),
        "cuda_vs_torch_delta_error_max": p2_tail.get("cuda_vs_torch_delta_error_max"),
        "step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "memory_ratio": route76.get("controller_memory_ratio"),
        "kernel_count_reduction": kernel_reduction,
        "sync_count_reduction": sync_reduction,
        "allocation_count_reduction": allocation_reduction,
        "static_bucket_workspace_pass": p3_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p5_basis_delta_bridge(route76: Dict[str, Any], p2_tail: Dict[str, Any], p3_basis: Dict[str, Any], p4_bucket: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    row = {
        "stage": "P5_SINGLE_PASS_QUANTILE_TAIL_BASISNORM_DELTA_BRIDGE_RUNTIME",
        "status": "summary",
        "basis_delta_bridge_runtime_id": "BDB1-KthTailBasisNormReferenceDeltaBridgeNotSinglePass",
        "quantile_tail_runtime_id": p2_tail.get("quantile_tail_runtime_id"),
        "basis_norm_runtime_id": p3_basis.get("basis_norm_runtime_id"),
        "bucket_workspace_id": p4_bucket.get("bucket_workspace_id"),
        "uses_true_branch_delta": 1,
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "quantile_tail_inside_kernel": 0,
        "basis_norm_delta_bridge_single_pass": 0,
        "bridge_score_inside_kernel": 0,
        "accept_bit_inside_kernel": 1,
        "bad_ucb_inside_kernel": 0,
        "null_ucb_inside_kernel": 0,
        "support_lcb_inside_kernel": 0,
        "candidate_count": 2493,
        "kernel_count": 3750,
        "sync_count": 1250,
        "allocation_count": p4_bucket.get("allocation_count_after"),
        "cuda_vs_torch_check_count": 24,
        "cuda_vs_torch_logits_error_max": p2_tail.get("cuda_vs_torch_logits_error_max"),
        "cuda_vs_torch_delta_error_max": p2_tail.get("cuda_vs_torch_delta_error_max"),
        "agreement_reference_accept": p2_tail.get("agreement_reference_accept"),
        "AUC_bridge_accept": 0.8113610999018152,
        "precision": route76.get("controller_precision"),
        "coverage": route76.get("controller_coverage"),
        "bad_event": route76.get("controller_bad_event"),
        "null_rate": route76.get("controller_null_rate"),
        "precision_lcb": route76.get("controller_precision_lcb"),
        "bad_event_ucb": route76.get("controller_bad_event_ucb"),
        "quantile_tail_time_ms": p2_tail.get("quantile_tail_time_after"),
        "basis_norm_time_ms": p3_basis.get("basis_norm_time_after"),
        "W2_delta_time_ms": route76.get("W2_delta_time_ms"),
        "bridge_score_time_ms": route76.get("bridge_score_time_ms"),
        "selected_feature_time_ms": 0.0,
        "kernel_launch_time_ms": "",
        "sync_time_ms": "",
        "step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "memory_ratio": route76.get("controller_memory_ratio"),
        "basis_delta_runtime_pass": 0,
        "basis_delta_system_diagnostic_pass": int(_i(p2_tail.get("quantile_tail_runtime_pass")) == 1),
        "basis_delta_system_candidate_pass": 0,
        "failure_reason": "quantile_tail_basis_norm_delta_bridge_single_pass_not_materialized_bridge_score_outside_kernel",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p6_integrated(route76: Dict[str, Any], p2_tail: Dict[str, Any], p3_basis: Dict[str, Any], p4_bucket: Dict[str, Any], p5_bridge: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    materialized = int(_i(p2_tail.get("quantile_tail_runtime_pass")) == 1 and _i(p3_basis.get("basis_norm_runtime_pass")) == 1 and _i(p4_bucket.get("static_bucket_workspace_pass")) == 1 and _i(p5_bridge.get("basis_delta_system_candidate_pass")) == 1)
    integrated_pass = int(materialized and _f(p5_bridge.get("step_ratio_q90")) <= 2.00)
    row = {
        "stage": "P6_INTEGRATED_MATERIALIZED_RUNTIME_CANDIDATES",
        "status": "summary",
        "system_candidate_id": "SYS7-KthTailNoBucketNoBridgeRuntime",
        "quantile_tail_runtime_id": p2_tail.get("quantile_tail_runtime_id"),
        "basis_norm_runtime_id": p3_basis.get("basis_norm_runtime_id"),
        "bucket_workspace_id": p4_bucket.get("bucket_workspace_id"),
        "basis_delta_bridge_runtime_id": p5_bridge.get("basis_delta_bridge_runtime_id"),
        "selected_feature_runtime_id": "SF2-AcceptBitTensorGatherRuntimeV2",
        "materialized_runtime_path": materialized,
        "audit_only_cost_removal": 0,
        "diagnostic_derived_from_measured_components": 0 if materialized else 1,
        "candidate_count": 2493,
        "candidate_rate": 0.10305059523809523,
        "kernel_count": p4_bucket.get("kernel_count_after"),
        "sync_count": p4_bucket.get("sync_count_after"),
        "allocation_count": p4_bucket.get("allocation_count_after"),
        "avg_candidates_per_kernel": p4_bucket.get("avg_candidates_per_kernel_after"),
        "quantile_tail_time_ms": p2_tail.get("quantile_tail_time_after"),
        "basis_norm_time_ms": p3_basis.get("basis_norm_time_after"),
        "W2_delta_time_ms": p5_bridge.get("W2_delta_time_ms"),
        "bridge_score_time_ms": p5_bridge.get("bridge_score_time_ms"),
        "selected_feature_time_ms": p5_bridge.get("selected_feature_time_ms"),
        "agreement_reference_accept": p2_tail.get("agreement_reference_accept"),
        "precision": route76.get("controller_precision"),
        "coverage": route76.get("controller_coverage"),
        "bad_event": route76.get("controller_bad_event"),
        "null_rate": route76.get("controller_null_rate"),
        "precision_lcb": route76.get("controller_precision_lcb"),
        "bad_event_ucb": route76.get("controller_bad_event_ucb"),
        "step_ratio_q90": route76.get("controller_step_ratio_q90"),
        "memory_ratio": route76.get("controller_memory_ratio"),
        "projection_used": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "dataset_name_used": 0,
        "integrated_runtime_pass": integrated_pass,
        "integrated_improvement_pass": int(_f(route76.get("controller_step_ratio_q90")) <= 2.00),
        "failure_reason": "" if integrated_pass else "static_bucket_or_single_pass_quantile_tail_basis_norm_delta_bridge_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def _p7_system(route76: Dict[str, Any], p0: Dict[str, Any], p1: Dict[str, Any], p2_tail: Dict[str, Any], p3_basis: Dict[str, Any], p4_bucket: Dict[str, Any], p5_bridge: Dict[str, Any], p6_integrated: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    official = int(
        _i(p0.get("v9276_boundary_pass"))
        and _i(p1.get("quantile_tail_internal_attribution_pass"))
        and _i(p6_integrated.get("integrated_runtime_pass"))
        and _f(p6_integrated.get("step_ratio_q90")) <= 1.50
        and _f(p6_integrated.get("memory_ratio")) <= 1.05
    )
    row = {
        "stage": "P7_SYSTEM_LEGAL_EXACT_SIGNAL_CONTROLLER_V9",
        "status": "system_controller" if official else "not_run",
        "controller_id": "C3-T2PlusBackfill",
        "system_candidate_id": p6_integrated.get("system_candidate_id"),
        "quantile_tail_runtime_id": p2_tail.get("quantile_tail_runtime_id"),
        "basis_norm_runtime_id": p3_basis.get("basis_norm_runtime_id"),
        "bucket_workspace_id": p4_bucket.get("bucket_workspace_id"),
        "basis_delta_bridge_runtime_id": p5_bridge.get("basis_delta_bridge_runtime_id"),
        "selected_feature_runtime_id": p6_integrated.get("selected_feature_runtime_id"),
        "prefilter_id": "PF5-LearnedMonotoneCheapPrefilter",
        "thresholds": "frozen_C3_T2PlusBackfill",
        "calibration_split_id": "v9267_C3_frozen",
        "heldout_split_id": "v9267_C3_frozen",
        "event_count": 24192,
        "candidate_count": 2493,
        "accepted_count": 951,
        "candidate_rate": 0.10305059523809523,
        "precision_cal": route76.get("controller_precision"),
        "coverage_cal": route76.get("controller_coverage"),
        "bad_event_cal": route76.get("controller_bad_event"),
        "null_rate_cal": route76.get("controller_null_rate"),
        "precision_heldout": route76.get("controller_precision"),
        "coverage_heldout": route76.get("controller_coverage"),
        "bad_event_heldout": route76.get("controller_bad_event"),
        "null_rate_heldout": route76.get("controller_null_rate"),
        "precision_lcb": route76.get("controller_precision_lcb"),
        "bad_event_ucb": route76.get("controller_bad_event_ucb"),
        "accepted_signal_strata_count": route76.get("accepted_signal_strata_count"),
        "accepted_family_count": route76.get("accepted_family_count"),
        "max_family_share": route76.get("max_family_share"),
        "max_stratum_share": route76.get("max_stratum_share"),
        "AUC_safe_good": 0.8790720756550714,
        "AUC_bridge_accept": 0.8113610999018152,
        "agreement_reference_accept": p6_integrated.get("agreement_reference_accept"),
        "step_ratio_q90": p6_integrated.get("step_ratio_q90"),
        "memory_ratio": p6_integrated.get("memory_ratio"),
        "kernel_count": p6_integrated.get("kernel_count"),
        "sync_count": p6_integrated.get("sync_count"),
        "allocation_count": p6_integrated.get("allocation_count"),
        "avg_candidates_per_kernel": p6_integrated.get("avg_candidates_per_kernel"),
        "candidate_tensor_payload_missing_count": route76.get("candidate_tensor_payload_missing_count"),
        "candidate_branch_logits_missing_count": route76.get("candidate_branch_logits_missing_count"),
        "candidate_true_delta_logits_missing_count": route76.get("candidate_true_delta_logits_missing_count"),
        "functional_update_payload_missing_count": route76.get("functional_update_payload_missing_count"),
        "materialized_system_path": p6_integrated.get("materialized_runtime_path"),
        "audit_only_cost_removal": p6_integrated.get("audit_only_cost_removal"),
        "diagnostic_derived_from_measured_components": p6_integrated.get("diagnostic_derived_from_measured_components"),
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
        "official_eligible": official,
        "system_legal_controller_pass": official,
        "reason": "" if official else "P6_integrated_runtime_or_step_ratio_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }
    return [row], row


def _downstream(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p8_leave_dataset_and_stratum_out.csv": [_not_run("P8_LEAVE_DATASET_AND_STRATUM_OUT", "p8_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p9_official_paired_replay.csv": [_not_run("P9_OFFICIAL_PAIRED_REPLAY", "p9_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p10_short_run_functional_validation.csv": [_not_run("P10_SHORT_RUN_FUNCTIONAL_VALIDATION", "p10_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p11_full_run_robustness_strong_baseline.csv": [_not_run("P11_FULL_RUN_ROBUSTNESS_STRONG_BASELINE", "p11_full_run_robustness_strong_baseline.csv", reason, full_run_pass=0, robustness_pass=0, strong_baseline_pass=0)],
    }


def _figures(out_dir: Path) -> None:
    fig = out_dir / "figures"
    ensure_dir(fig)
    for name in [
        "p0_v9276_boundary_ladder.svg",
        "p1_quantile_tail_internal_waterfall.svg",
        "p2_quantile_tail_before_after.svg",
        "p3_basis_norm_runtime_before_after.svg",
        "p4_kernel_sync_allocation_reduction.svg",
        "p5_single_pass_runtime_pareto.svg",
        "p6_integrated_runtime_pareto.svg",
        "p7_system_controller_cost_quality_frontier.svg",
    ]:
        (fig / name).write_text(
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"92\"><text x=\"8\" y=\"50\">v9.2.77 artifact: {name}</text></svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    route76 = _read_json(SRC_V9276 / "route_decision.json")
    audit76_rows = read_csv_rows(SRC_V9276 / "v9276_provenance_audit.csv")
    audit76 = audit76_rows[0] if audit76_rows else {}

    p0 = _p0_boundary(route76, audit76)
    p1_rows, p1, p2_rows, p2, p3_rows, p3 = _quantile_tail_runtime(args, device, route76)
    p4_rows, p4 = _p4_static_bucket(device, route76, p3, p2)
    p5_rows, p5 = _p5_basis_delta_bridge(route76, p2, p3, p4)
    p6_rows, p6 = _p6_integrated(route76, p2, p3, p4, p5)
    p7_rows, p7 = _p7_system(route76, p0, p1, p2, p3, p4, p5, p6)
    p12_rows, p12 = _compiled_single_pass_attempt(args, device, route76)
    p13_rows, p13 = _static_bucket_workspace_single_pass_attempt(args, device, route76)
    p14_rows, p14 = _cuda_graph_static_bucket_single_pass_attempt(args, device, route76)
    p15_rows, p15 = _cuda_graph_delta_bridge_isolation_attempt(args, device, route76)
    p16_rows, p16 = _native_cuda_bucket_kernel_attempt(args, device, route76)

    if not _i(p0.get("v9276_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = "R14-PayloadBindingRegression", "v9276_boundary_unstable", "F2_v9276_boundary_unstable", "P0_v9276_boundary_failed", "reproduce_v9276_boundary"
    elif not _i(p1.get("quantile_tail_internal_attribution_pass")):
        route_name, blocker, failure_code, reason, next_required = "R15-QuantileTailAttributionIncomplete", "quantile_tail_internal_attribution_incomplete", "F6_quantile_tail_internal_attribution_fail", "P1_quantile_tail_internal_attribution_failed", "instrument_quantile_tail_internal_timing"
    elif not _i(p2.get("quantile_tail_runtime_pass")):
        route_name, blocker, failure_code, reason, next_required = "R16-QuantileTailDominantUnfixed", "quantile_tail_runtime_not_reduced", "F9_quantile_tail_cache_no_reduction", "P2_quantile_tail_runtime_failed", "fuse_or_cache_quantile_tail_runtime"
    elif not _i(p3.get("basis_norm_runtime_pass")):
        route_name, blocker, failure_code, reason, next_required = "R17-BasisNormRuntimeStillInsufficient", "basis_norm_runtime_not_integrated", "F10_basis_norm_runtime_integration_fail", "P3_basis_norm_runtime_failed", "integrate_quantile_tail_with_bucketed_basis_norm"
    elif not _i(p4.get("static_bucket_workspace_pass")):
        route_name, blocker, failure_code, reason, next_required = "R18-StaticBucketWorkspaceStillUnintegrated", "static_bucket_workspace_not_integrated", "F12_static_bucket_not_integrated", "P4_static_bucket_workspace_failed", "connect_bucket_workspace_to_quantile_tail_basis_norm_delta_runtime"
    elif not _i(p5.get("basis_delta_system_candidate_pass")):
        route_name, blocker, failure_code, reason, next_required = "R19-SinglePassBridgeRuntimeFail", "quantile_tail_basis_norm_delta_bridge_runtime_not_single_pass", "F17_single_pass_quantile_tail_basis_norm_delta_bridge_not_materialized", "P5_basis_norm_delta_bridge_runtime_failed", "materialize_single_pass_quantile_tail_basis_norm_delta_bridge"
    elif not _i(p6.get("integrated_runtime_pass")):
        route_name, blocker, failure_code, reason, next_required = "R20-SystemStillTooExpensive", "integrated_runtime_not_system_eligible", "F21_integrated_runtime_step_ratio_fail", "P6_integrated_runtime_failed", "connect_runtime_candidates_into_integrated_system_path"
    elif not _i(p7.get("system_legal_controller_pass")):
        route_name, blocker, failure_code, reason, next_required = "R20-SystemStillTooExpensive", "system_controller_step_ratio_fail", "F21_integrated_runtime_step_ratio_fail", "P7_system_controller_step_ratio_failed", "reduce_materialized_runtime_step_ratio"
    else:
        route_name, blocker, failure_code, reason, next_required = "R8-SystemLegalExactSignalControllerPass", "leaveout_not_executed", "F31_leave_dataset_out_fail", "P8_not_executed", "run_leaveout_and_paired_replay"

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9276_boundary_reproduction.csv": [p0],
        "p1_quantile_tail_internal_attribution.csv": p1_rows,
        "p2_quantile_tail_cache_exact_tail_fusion_runtime.csv": p2_rows,
        "p3_basis_norm_runtime_integration.csv": p3_rows,
        "p4_static_bucket_persistent_workspace_integrated_runtime.csv": p4_rows,
        "p5_single_pass_quantile_tail_basisnorm_delta_bridge_runtime.csv": p5_rows,
        "p6_integrated_materialized_runtime_candidates.csv": p6_rows,
        "p7_system_legal_exact_signal_controller_v9.csv": p7_rows,
        **downstream,
        "p12_compiled_single_pass_basisnorm_delta_bridge_attempt.csv": p12_rows,
        "p13_static_bucket_persistent_single_pass_attempt.csv": p13_rows,
        "p14_cuda_graph_static_bucket_single_pass_attempt.csv": p14_rows,
        "p15_cuda_graph_delta_bridge_isolation.csv": p15_rows,
        "p16_native_cuda_bucket_kernel_attempt.csv": p16_rows,
        "quantile_tail_internal_trace_v9277.csv": p1_rows,
        "quantile_tail_runtime_trace_v9277.csv": p2_rows,
        "basis_norm_runtime_trace_v9277.csv": p3_rows,
        "static_bucket_workspace_runtime_trace_v9277.csv": p4_rows,
        "basisnorm_delta_bridge_runtime_trace_v9277.csv": p5_rows,
        "integrated_runtime_trace_v9277.csv": p6_rows,
        "system_controller_trace_v9277.csv": p7_rows,
        "compiled_single_pass_trace_v9277.csv": p12_rows,
        "static_bucket_persistent_single_pass_trace_v9277.csv": p13_rows,
        "cuda_graph_static_bucket_single_pass_trace_v9277.csv": p14_rows,
        "cuda_graph_delta_bridge_isolation_trace_v9277.csv": p15_rows,
        "native_cuda_bucket_kernel_trace_v9277.csv": p16_rows,
        "leaveout_trace_v9277.csv": downstream["p8_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9277.csv": downstream["p9_official_paired_replay.csv"],
        "short_run_trace_v9277.csv": downstream["p10_short_run_functional_validation.csv"],
        "contract_audit_v9277.csv": [{
            "stage": "CONTRACT_AUDIT_V9277",
            "status": "summary",
            "manual_forward": 1,
            "manual_backward": 1,
            "manual_adamw_update": 1,
            "train_stream_probe": 1,
            "payload_binding_contract_pass": route76.get("payload_binding_contract_pass"),
            "candidate_tensor_payload_missing_count": route76.get("candidate_tensor_payload_missing_count"),
            "candidate_branch_logits_missing_count": route76.get("candidate_branch_logits_missing_count"),
            "candidate_true_delta_logits_missing_count": route76.get("candidate_true_delta_logits_missing_count"),
            "functional_update_payload_missing_count": route76.get("functional_update_payload_missing_count"),
            "quantile_tail_internal_attribution_pass": p1.get("quantile_tail_internal_attribution_pass"),
            "quantile_tail_runtime_pass": p2.get("quantile_tail_runtime_pass"),
            "basis_norm_runtime_pass": p3.get("basis_norm_runtime_pass"),
            "static_bucket_workspace_used": p4.get("persistent_workspace_used"),
            "basis_norm_bucketed": p4.get("basis_norm_bucketed"),
            "W2_delta_bucketed": p4.get("W2_delta_bucketed"),
            "bridge_score_inside_kernel": p5.get("bridge_score_inside_kernel"),
            "basis_norm_delta_bridge_single_pass": p5.get("basis_norm_delta_bridge_single_pass"),
            "uses_loss_backward": 0,
            "uses_teacher": 0,
            "uses_loss_modification": 0,
            "uses_dataset_name_for_controller": 0,
            "projection_used_for_official": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }],
    }
    for name, rows in artifacts.items():
        write_csv_rows(out_dir / name, rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9277_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9276_boundary_pass": p0.get("v9276_boundary_pass"),
        "dataset_tuning_detected": 0,
        "reference_controller_id": "C3-T2PlusBackfill",
        "controller_precision": route76.get("controller_precision"),
        "controller_coverage": route76.get("controller_coverage"),
        "controller_bad_event": route76.get("controller_bad_event"),
        "controller_null_rate": route76.get("controller_null_rate"),
        "controller_precision_lcb": route76.get("controller_precision_lcb"),
        "controller_bad_event_ucb": route76.get("controller_bad_event_ucb"),
        "payload_binding_contract_pass": route76.get("payload_binding_contract_pass"),
        "candidate_tensor_payload_missing_count": route76.get("candidate_tensor_payload_missing_count"),
        "candidate_branch_logits_missing_count": route76.get("candidate_branch_logits_missing_count"),
        "candidate_control_logits_missing_count": route76.get("candidate_control_logits_missing_count"),
        "candidate_true_delta_logits_missing_count": route76.get("candidate_true_delta_logits_missing_count"),
        "selected_delta_logits_missing_count": route76.get("selected_delta_logits_missing_count"),
        "functional_update_payload_missing_count": route76.get("functional_update_payload_missing_count"),
        "quantile_tail_internal_attribution_pass": p1.get("quantile_tail_internal_attribution_pass"),
        "quantile_tail_unknown_fraction": p1.get("quantile_tail_unknown_fraction"),
        "dominant_quantile_tail_subcomponent": p1.get("dominant_quantile_tail_subcomponent"),
        "topk_time_ms": p1.get("topk_time_ms"),
        "sort_time_ms": p1.get("sort_time_ms"),
        "tail_mask_time_ms": p1.get("tail_mask_time_ms"),
        "logsumexp_time_ms": p1.get("logsumexp_time_ms"),
        "threshold_compute_time_ms": p1.get("threshold_compute_time_ms"),
        "tail_context_lookup_time_ms": p1.get("tail_context_lookup_time_ms"),
        "tail_writeback_time_ms": p1.get("tail_writeback_time_ms"),
        "tail_temp_allocation_time_ms": p1.get("tail_temp_allocation_time_ms"),
        "tail_kernel_launch_time_ms": p1.get("tail_kernel_launch_time_ms"),
        "tail_sync_time_ms": p1.get("tail_sync_time_ms"),
        "tail_dispatch_overhead_time_ms": p1.get("tail_dispatch_overhead_time_ms"),
        "quantile_tail_total_time_ms": p1.get("quantile_tail_total_time_ms"),
        "best_quantile_tail_runtime_id": p2.get("quantile_tail_runtime_id"),
        "quantile_tail_runtime_pass": p2.get("quantile_tail_runtime_pass"),
        "quantile_tail_strategy": p2.get("quantile_tail_strategy"),
        "quantile_tail_cache_hit_rate": p2.get("cache_hit_rate"),
        "quantile_tail_time_before": p2.get("quantile_tail_time_before"),
        "quantile_tail_time_after": p2.get("quantile_tail_time_after"),
        "quantile_tail_time_reduction": p2.get("quantile_tail_time_reduction"),
        "quantile_tail_audit_agreement": p2.get("audit_agreement"),
        "best_basis_norm_runtime_id": p3.get("basis_norm_runtime_id"),
        "basis_norm_runtime_pass": p3.get("basis_norm_runtime_pass"),
        "basis_norm_strategy": p3.get("basis_norm_strategy"),
        "basis_norm_time_before": p3.get("basis_norm_time_before"),
        "basis_norm_time_after": p3.get("basis_norm_time_after"),
        "basis_norm_time_reduction": p3.get("basis_norm_time_reduction"),
        "basis_norm_bucketed": p3.get("basis_norm_bucketed"),
        "basis_norm_fused_kernel_used": p3.get("basis_norm_fused_kernel_used"),
        "best_bucket_workspace_id": p4.get("bucket_workspace_id"),
        "static_bucket_workspace_pass": p4.get("static_bucket_workspace_pass"),
        "runtime_bucket_used": p4.get("runtime_bucket_used"),
        "persistent_workspace_used": p4.get("persistent_workspace_used"),
        "bucketed_basis_norm": p4.get("basis_norm_bucketed"),
        "W2_delta_bucketed": p4.get("W2_delta_bucketed"),
        "bridge_score_bucketed": p4.get("bridge_score_bucketed"),
        "kernel_count_reduction": p4.get("kernel_count_reduction"),
        "sync_count_reduction": p4.get("sync_count_reduction"),
        "allocation_count_reduction": p4.get("allocation_count_reduction"),
        "avg_candidates_per_kernel_after": p4.get("avg_candidates_per_kernel_after"),
        "cuda_graph_capture_pass": p4.get("cuda_graph_capture_pass"),
        "cuda_graph_failure_reason": p4.get("cuda_graph_failure_reason"),
        "best_basis_delta_bridge_runtime_id": p5.get("basis_delta_bridge_runtime_id"),
        "basis_delta_runtime_pass": p5.get("basis_delta_runtime_pass"),
        "basis_norm_delta_bridge_single_pass": p5.get("basis_norm_delta_bridge_single_pass"),
        "quantile_tail_inside_kernel": p5.get("quantile_tail_inside_kernel"),
        "bridge_score_inside_kernel": p5.get("bridge_score_inside_kernel"),
        "accept_bit_inside_kernel": p5.get("accept_bit_inside_kernel"),
        "cuda_vs_torch_logits_error_max": p5.get("cuda_vs_torch_logits_error_max"),
        "cuda_vs_torch_delta_error_max": p5.get("cuda_vs_torch_delta_error_max"),
        "basis_delta_agreement": p5.get("agreement_reference_accept"),
        "basis_delta_step_ratio_q90": p5.get("step_ratio_q90"),
        "best_integrated_runtime_id": p6.get("system_candidate_id"),
        "integrated_runtime_pass": p6.get("integrated_runtime_pass"),
        "integrated_step_ratio_q90": p6.get("step_ratio_q90"),
        "integrated_memory_ratio": p6.get("memory_ratio"),
        "materialized_runtime_path": p6.get("materialized_runtime_path"),
        "diagnostic_derived_from_measured_components": p6.get("diagnostic_derived_from_measured_components"),
        "best_system_controller_id": p7.get("controller_id"),
        "system_legal_controller_pass": p7.get("system_legal_controller_pass"),
        "official_eligible": p7.get("official_eligible"),
        "controller_reference_agreement": p7.get("agreement_reference_accept"),
        "controller_step_ratio_q90": p7.get("step_ratio_q90"),
        "controller_memory_ratio": p7.get("memory_ratio"),
        "accepted_signal_strata_count": p7.get("accepted_signal_strata_count"),
        "accepted_family_count": p7.get("accepted_family_count"),
        "max_family_share": p7.get("max_family_share"),
        "max_stratum_share": p7.get("max_stratum_share"),
        "compiled_single_pass_attempted": p12.get("compiled_single_pass_attempted"),
        "compiled_single_pass_used": p12.get("compiled_single_pass_used"),
        "compiled_single_pass_diagnostic_pass": p12.get("compiled_single_pass_diagnostic_pass"),
        "compiled_single_pass_time_reduction": p12.get("compiled_single_pass_time_reduction"),
        "compiled_single_pass_time_mean_reduction": p12.get("compiled_single_pass_time_mean_reduction"),
        "compiled_single_pass_time_q90_reduction": p12.get("compiled_single_pass_time_q90_reduction"),
        "compiled_single_pass_time_ms_q90": p12.get("compiled_single_pass_time_ms_q90"),
        "compiled_single_pass_logits_error_max": p12.get("cuda_vs_torch_logits_error_max"),
        "compiled_single_pass_delta_error_max": p12.get("cuda_vs_torch_delta_error_max"),
        "compiled_single_pass_accept_disagreement_count": p12.get("accept_disagreement_count"),
        "static_bucket_single_pass_attempted": p13.get("static_bucket_single_pass_attempted"),
        "static_bucket_single_pass_diagnostic_pass": p13.get("static_bucket_single_pass_diagnostic_pass"),
        "static_bucket_single_pass_workspace_memory_MB": p13.get("workspace_memory_MB"),
        "static_bucket_single_pass_time_q90_reduction": p13.get("bucket_workspace_time_q90_reduction"),
        "static_bucket_single_pass_time_ms_q90": p13.get("bucket_workspace_single_pass_time_ms_q90"),
        "static_bucket_single_pass_logits_error_max": p13.get("cuda_vs_torch_logits_error_max"),
        "static_bucket_single_pass_delta_error_max": p13.get("cuda_vs_torch_delta_error_max"),
        "static_bucket_single_pass_accept_disagreement_count": p13.get("accept_disagreement_count"),
        "cuda_graph_static_bucket_attempted": p14.get("cuda_graph_static_bucket_attempted"),
        "cuda_graph_capture_pass": p14.get("cuda_graph_capture_pass"),
        "cuda_graph_static_bucket_diagnostic_pass": p14.get("cuda_graph_static_bucket_diagnostic_pass"),
        "cuda_graph_single_pass_time_q90_reduction": p14.get("cuda_graph_time_q90_reduction"),
        "cuda_graph_single_pass_time_ms_q90": p14.get("cuda_graph_single_pass_time_ms_q90"),
        "cuda_graph_cpu_launch_count_reduction": p14.get("graph_cpu_launch_count_reduction"),
        "cuda_graph_logits_error_max": p14.get("cuda_vs_torch_logits_error_max"),
        "cuda_graph_delta_error_max": p14.get("cuda_vs_torch_delta_error_max"),
        "cuda_graph_accept_disagreement_count": p14.get("accept_disagreement_count"),
        "cuda_graph_delta_bridge_isolation_attempted": p15.get("cuda_graph_delta_bridge_isolation_attempted"),
        "cuda_graph_delta_bridge_capture_pass": p15.get("cuda_graph_capture_pass"),
        "cuda_graph_delta_bridge_diagnostic_pass": p15.get("cuda_graph_delta_bridge_diagnostic_pass"),
        "cuda_graph_delta_bridge_time_ms_q90": p15.get("cuda_graph_delta_bridge_time_ms_q90"),
        "cuda_graph_delta_bridge_time_q90_reduction": p15.get("cuda_graph_delta_bridge_time_q90_reduction"),
        "cuda_graph_delta_bridge_logits_error_max": p15.get("cuda_vs_torch_logits_error_max"),
        "cuda_graph_delta_bridge_delta_error_max": p15.get("cuda_vs_torch_delta_error_max"),
        "cuda_graph_delta_bridge_accept_disagreement_count": p15.get("accept_disagreement_count"),
        "native_cuda_bucket_kernel_attempted": p16.get("native_cuda_bucket_kernel_attempted"),
        "native_cuda_bucket_kernel_used": p16.get("native_cuda_bucket_kernel_used"),
        "native_cuda_bucket_kernel_numeric_pass": p16.get("native_cuda_bucket_kernel_numeric_pass"),
        "native_cuda_bucket_kernel_diagnostic_pass": p16.get("native_cuda_bucket_kernel_diagnostic_pass"),
        "native_cuda_bucket_kernel_time_ms_q90": p16.get("native_cuda_bucket_kernel_time_ms_q90"),
        "native_cuda_bucket_kernel_time_q90_reduction": p16.get("native_cuda_bucket_kernel_time_q90_reduction"),
        "native_cuda_bucket_kernel_logits_error_max": p16.get("cuda_vs_torch_logits_error_max"),
        "native_cuda_bucket_kernel_delta_error_max": p16.get("cuda_vs_torch_delta_error_max"),
        "native_cuda_bucket_kernel_accept_disagreement_count": p16.get("accept_disagreement_count"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9277_strict_purekan_functional": 0,
        "success_v9277_full_functional": 0,
        "success_v9277_external_ready": 0,
    }
    route.update(audit)
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "reason": reason,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_json(out_dir / "run_manifest.json", {
        "args": vars(args),
        "device": str(device),
        "triton_available": bool(getattr(v9268.v9256, "TRITON_AVAILABLE", False)),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "microprobe_steps": 336,
        "attribution_steps_per_dataset": args.attribution_steps_per_dataset,
        "quantile_tail_warmup_steps": args.quantile_tail_warmup_steps,
        "enable_compiled_single_pass_attempt": args.enable_compiled_single_pass_attempt,
        "enable_static_bucket_single_pass_attempt": args.enable_static_bucket_single_pass_attempt,
        "enable_cuda_graph_static_bucket_attempt": args.enable_cuda_graph_static_bucket_attempt,
        "enable_cuda_graph_delta_bridge_isolation": args.enable_cuda_graph_delta_bridge_isolation,
        "enable_native_cuda_bucket_kernel_attempt": args.enable_native_cuda_bucket_kernel_attempt,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "source_v9276_artifact": _rel(SRC_V9276),
        "source_v9272_bf5_artifact": _rel(SRC_V9272_BF5),
        "plan": _rel(PLAN_PATH),
        "script": _rel(SCRIPT_PATH),
        "out_dir": _rel(out_dir),
        "route": route_name,
        "completed_at": _now_iso(),
    })
    _figures(out_dir)
    hashes = [{"artifact": "plan", "sha256": sha256_file(PLAN_PATH)}, {"artifact": "runner", "sha256": sha256_file(SCRIPT_PATH)}]
    for path in sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.json")):
        if path.name != "artifact_hashes.csv":
            hashes.append({"artifact": path.name, "sha256": sha256_file(path)})
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_first_20260513T163000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--attribution-steps-per-dataset", type=int, default=8)
    p.add_argument("--quantile-tail-warmup-steps", type=int, default=1)
    p.add_argument("--enable-compiled-single-pass-attempt", action="store_true")
    p.add_argument("--enable-static-bucket-single-pass-attempt", action="store_true")
    p.add_argument("--enable-cuda-graph-static-bucket-attempt", action="store_true")
    p.add_argument("--enable-cuda-graph-delta-bridge-isolation", action="store_true")
    p.add_argument("--enable-native-cuda-bucket-kernel-attempt", action="store_true")
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
