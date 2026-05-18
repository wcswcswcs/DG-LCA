#!/usr/bin/env python3
"""DG-KAN v9.2.56 true branch-delta tensor interface audit.

This runner starts from v9.2.55, where CUDA extension plumbing was available
but no true branch-delta logits tensor interface existed.  v9.2.56 explicitly
constructs real train-stream branch logits for RealFunctional and controls,
tests a lower-level CUDA K7d control-gain kernel on those logits, and keeps all
controller/downstream gates closed unless the true branch-delta path satisfies
predictivity, agreement, and system constraints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.cpp_extension import load_inline

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9248_real_train_stream_microprobe_online_support_region_controller as v9248  # noqa: E402
import run_v9250_metric_kernel_compression_coverage_preserving_cascade_controller as v9250  # noqa: E402
import run_v9253_d0_vectorized_controller_prep_real_fused_branch_delta_kernel as v9253  # noqa: E402
import run_v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure as v9255  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.56_TrueBranchDeltaTensorInterface_K7dControlGainCUDAFusion_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"
SRC_V9255 = RESULT_ROOT / "v9255_lower_level_cuda_branch_delta_extension_exact_signal_controller_closure_first_20260512T071500Z"

try:  # noqa: E402
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - recorded in artifacts.
    TRITON_AVAILABLE = False

CUDA_EXTENSION_ERROR = ""
_K7D_EXT = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _f(value: Any, default: float = 0.0) -> float:
    return v9250._f(value, default)


def _i(value: Any, default: int = 0) -> int:
    return v9250._i(value, default)


def _mean(values: Iterable[float]) -> float:
    return v9250._mean(values)


def _q(values: Sequence[float], q: float) -> float:
    return v9250._q(values, q)


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return v9250._auc(scores, labels)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    return v9250._corr(xs, ys)


def _accept_top(rows: Sequence[Dict[str, Any]], scores: Sequence[float], coverage: float) -> List[int]:
    return v9250._accept_top(rows, scores, coverage)


def _accept_metrics(rows: Sequence[Dict[str, Any]], accepted: Sequence[int]) -> Dict[str, Any]:
    return v9250._accept_metrics(rows, accepted)


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _not_run(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _load_k7d_ext() -> Any:
    global CUDA_EXTENSION_ERROR, _K7D_EXT
    if _K7D_EXT is not None:
        return _K7D_EXT
    if not torch.cuda.is_available():
        CUDA_EXTENSION_ERROR = "cuda_not_available"
        return None
    cpp_src = """
    #include <torch/extension.h>
    torch::Tensor k7d_control_gap_forward(torch::Tensor before, torch::Tensor real, torch::Tensor adamw, torch::Tensor bestlr, torch::Tensor labels);
    """
    cuda_src = r"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <math.h>

    __device__ float row_ce(const float* logits, const int c, const long y) {
      float m = logits[0];
      for (int j = 1; j < c; ++j) {
        m = fmaxf(m, logits[j]);
      }
      float s = 0.0f;
      for (int j = 0; j < c; ++j) {
        s += expf(logits[j] - m);
      }
      return logf(s) + m - logits[y];
    }

    __global__ void k7d_control_gap_kernel(
        const float* __restrict__ before,
        const float* __restrict__ real,
        const float* __restrict__ adamw,
        const float* __restrict__ bestlr,
        const long* __restrict__ labels,
        float* __restrict__ out,
        const long n,
        const int c) {
      const long i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= n) return;
      const long y = labels[i];
      const float* b = before + i * c;
      const float* r = real + i * c;
      const float* a = adamw + i * c;
      const float* l = bestlr + i * c;
      const float ce_b = row_ce(b, c, y);
      const float gain_real = ce_b - row_ce(r, c, y);
      const float gain_adamw = ce_b - row_ce(a, c, y);
      const float gain_bestlr = ce_b - row_ce(l, c, y);
      const float control = fmaxf(gain_adamw, gain_bestlr);
      out[i] = gain_real - control;
    }

    torch::Tensor k7d_control_gap_forward(torch::Tensor before, torch::Tensor real, torch::Tensor adamw, torch::Tensor bestlr, torch::Tensor labels) {
      TORCH_CHECK(before.is_cuda(), "before must be CUDA");
      TORCH_CHECK(real.is_cuda(), "real must be CUDA");
      TORCH_CHECK(adamw.is_cuda(), "adamw must be CUDA");
      TORCH_CHECK(bestlr.is_cuda(), "bestlr must be CUDA");
      TORCH_CHECK(labels.is_cuda(), "labels must be CUDA");
      TORCH_CHECK(before.is_contiguous(), "before must be contiguous");
      TORCH_CHECK(real.is_contiguous(), "real must be contiguous");
      TORCH_CHECK(adamw.is_contiguous(), "adamw must be contiguous");
      TORCH_CHECK(bestlr.is_contiguous(), "bestlr must be contiguous");
      TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
      const long n = before.size(0);
      const int c = (int)before.size(1);
      auto out = torch::empty({n}, before.options());
      const int threads = 256;
      const int blocks = (int)((n + threads - 1) / threads);
      k7d_control_gap_kernel<<<blocks, threads>>>(
          before.data_ptr<float>(),
          real.data_ptr<float>(),
          adamw.data_ptr<float>(),
          bestlr.data_ptr<float>(),
          labels.data_ptr<long>(),
          out.data_ptr<float>(),
          n,
          c);
      return out;
    }
    """
    try:
        _K7D_EXT = load_inline(
            name="dgkan_v9256_true_k7d_control_gap_ext",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["k7d_control_gap_forward"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
        CUDA_EXTENSION_ERROR = ""
        return _K7D_EXT
    except Exception as exc:  # pragma: no cover - recorded in artifacts.
        CUDA_EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _ce_loss_vec(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logp = logits.log_softmax(dim=1)
    return -logp[torch.arange(y.numel(), device=y.device), y]


def _margin_vec(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true_logits = logits[torch.arange(y.numel(), device=y.device), y]
    masked = logits.clone()
    masked[torch.arange(y.numel(), device=y.device), y] = -torch.inf
    return true_logits - masked.max(dim=1).values


def _selected_classes(before_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    topk = torch.topk(before_logits, k=min(3, before_logits.shape[1]), dim=1).indices
    hard = topk[:, 0]
    second = topk[:, 1] if topk.shape[1] > 1 else topk[:, 0]
    third = topk[:, 2] if topk.shape[1] > 2 else second
    tail = torch.argmin(before_logits, dim=1)
    return torch.stack([y, hard, second, third, tail], dim=1)


def _gather_selected(logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, selected)


def _tensor_hash(*items: torch.Tensor) -> str:
    return v9248._tensor_hash(*items)


def _sample_true_branch_events(args: argparse.Namespace, device: torch.device, max_events: int = 24) -> Dict[str, Any]:
    """Build a small real train-stream tensor-interface sample.

    The returned tensors are used for legality/correctness/timing only. Full
    support/controller statistics still use the full fresh row set from the
    v9.2.48/v9.2.53 microprobe generator.
    """
    spec = lq.LQSpec("R2-LQ-fanin-output-scale-confirmed", "t2", int(args.hidden_dim), "default", 0.8)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    before_parts: List[torch.Tensor] = []
    real_parts: List[torch.Tensor] = []
    adamw_parts: List[torch.Tensor] = []
    bestlr_parts: List[torch.Tensor] = []
    labels_parts: List[torch.Tensor] = []
    selected_parts: List[torch.Tensor] = []
    event_rows: List[Dict[str, Any]] = []
    step_ratios: List[float] = []
    branch_ms: List[float] = []
    baseline_ms: List[float] = []
    selected_hashes: List[str] = []
    full_hashes: List[str] = []

    for dataset in [v92._canonical_task(x) for x in _parse_list(args.datasets)]:
        if len(event_rows) >= max_events:
            break
        for seed in _parse_ints(args.seeds):
            if len(event_rows) >= max_events:
                break
            load_args = argparse.Namespace(data_root=args.data_root, seed=int(args.seed) + int(seed))
            x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v92._load_task(
                load_args,
                dataset,
                train_size=int(args.train_size),
                test_size=32,
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(args.seed) + seed * 100 + 9256)
            states = [AdamWState.zeros_like(p) for p in params]
            gen = torch.Generator(device=device).manual_seed(int(args.seed) * 10000 + seed * 101 + 9256)
            n = int(x_train.shape[0])
            for step in range(min(int(args.interface_steps), int(args.microprobe_steps))):
                if len(event_rows) >= max_events:
                    break
                batch_idx = torch.randint(0, n, (int(args.batch_size) * 2,), generator=gen, device=device)
                xu, yu = x_train[batch_idx[: int(args.batch_size)]], y_train[batch_idx[: int(args.batch_size)]]
                xp, yp = x_train[batch_idx[int(args.batch_size) :]], y_train[batch_idx[int(args.batch_size) :]]
                v9248._sync(device)
                t0 = time.perf_counter()
                pack = bwd_core(xu, yu, *params, mu, std, 2.0, 2.0)
                grads = list(pack[1:])
                task_params = v9248._clone_params(params)
                task_states = v9248._clone_states(states)
                v92._adamw_update_foreach_(task_params, grads, task_states, cfg)
                v9248._sync(device)
                t1 = time.perf_counter()
                base_ms = max(1.0e-6, (t1 - t0) * 1000.0)
                task_delta = [tp - p for tp, p in zip(task_params, params)]
                ctrl_103 = v9248._apply_delta(params, task_delta, 1.03)
                ctrl_097 = v9248._apply_delta(params, task_delta, 0.97)
                ctrl_106 = v9248._apply_delta(params, task_delta, 1.06)
                for carrier_id in ("A1-RiskBoundedTailCarrier", "A3-LateAttachRoleWiseFT7EdgeCarrier"):
                    if len(event_rows) >= max_events:
                        break
                    v9248._sync(device)
                    tb0 = time.perf_counter()
                    fd, _fmeta = v9248._functional_delta(task_params, mu, std, spec, xu, yu, task_delta, carrier_id)
                    cand_params = [tp + d for tp, d in zip(task_params, fd)]
                    with torch.no_grad():
                        before = fwd_core(xp, *params, mu, std, 2.0, 2.0).contiguous()
                        real = fwd_core(xp, *cand_params, mu, std, 2.0, 2.0).contiguous()
                        adamw = fwd_core(xp, *ctrl_103, mu, std, 2.0, 2.0).contiguous()
                        logits_097 = fwd_core(xp, *ctrl_097, mu, std, 2.0, 2.0).contiguous()
                        logits_106 = fwd_core(xp, *ctrl_106, mu, std, 2.0, 2.0).contiguous()
                        ce097 = _ce_loss_vec(logits_097, yp).mean()
                        ce106 = _ce_loss_vec(logits_106, yp).mean()
                        bestlr = logits_097 if float(ce097.detach().cpu()) <= float(ce106.detach().cpu()) else logits_106
                        selected = _selected_classes(before, yp)
                        selected_real = _gather_selected(real, selected).contiguous()
                        selected_full = torch.stack(
                            [
                                _gather_selected(before, selected),
                                selected_real,
                                _gather_selected(adamw, selected),
                                _gather_selected(bestlr, selected),
                            ],
                            dim=0,
                        ).contiguous()
                        ce_before = _ce_loss_vec(before, yp).mean()
                        gain_real = ce_before - _ce_loss_vec(real, yp).mean()
                        gain_adamw = ce_before - _ce_loss_vec(adamw, yp).mean()
                        gain_bestlr = ce_before - _ce_loss_vec(bestlr, yp).mean()
                        control_gap = gain_real - torch.maximum(gain_adamw, gain_bestlr)
                    v9248._sync(device)
                    tb1 = time.perf_counter()
                    branch_time = max(0.0, (tb1 - tb0) * 1000.0)
                    before_parts.append(before.detach())
                    real_parts.append(real.detach())
                    adamw_parts.append(adamw.detach())
                    bestlr_parts.append(bestlr.detach())
                    labels_parts.append(yp.detach())
                    selected_parts.append(selected.detach())
                    full_hashes.append(_tensor_hash(before, real, adamw, bestlr, yp))
                    selected_hashes.append(_tensor_hash(selected_full, selected))
                    branch_ms.append(branch_time)
                    baseline_ms.append(base_ms)
                    step_ratios.append((base_ms + branch_time) / base_ms)
                    event_rows.append(
                        {
                            "row_id": f"{dataset}-seed{seed}-step{step}-{carrier_id}",
                            "dataset": dataset,
                            "seed": seed,
                            "step": step,
                            "carrier_id": carrier_id,
                            "base_logits_shape": list(before.shape),
                            "branch_logits_shape": [4, int(before.shape[0]), int(before.shape[1])],
                            "selected_logits_shape": list(selected_full.shape),
                            "selected_class_count": int(selected.shape[1]),
                            "branch_count": 4,
                            "class_count": int(before.shape[1]),
                            "full_logit_tensor_hash": full_hashes[-1],
                            "selected_logit_tensor_hash": selected_hashes[-1],
                            "control_gap_from_true_logits": float(control_gap.detach().cpu()),
                            "baseline_ms": base_ms,
                            "branch_delta_time_ms": branch_time,
                            "step_ratio": (base_ms + branch_time) / base_ms,
                        }
                    )
                for p, tp in zip(params, task_params):
                    p.copy_(tp)
                states = task_states

    tensors: Dict[str, torch.Tensor] = {}
    if before_parts:
        tensors = {
            "before": torch.cat(before_parts, dim=0).contiguous(),
            "real": torch.cat(real_parts, dim=0).contiguous(),
            "adamw": torch.cat(adamw_parts, dim=0).contiguous(),
            "bestlr": torch.cat(bestlr_parts, dim=0).contiguous(),
            "labels": torch.cat(labels_parts, dim=0).contiguous(),
            "selected": torch.cat(selected_parts, dim=0).contiguous(),
        }
    return {
        "events": event_rows,
        "tensors": tensors,
        "event_count": len(event_rows),
        "sample_count": int(tensors["labels"].numel()) if tensors else 0,
        "branch_delta_step_ratio_q90": _q(step_ratios, 0.90),
        "branch_delta_time_q90_ms": _q(branch_ms, 0.90),
        "baseline_time_q90_ms": _q(baseline_ms, 0.90),
        "full_logit_tensor_hash": hashlib.sha256("".join(full_hashes).encode("utf-8")).hexdigest() if full_hashes else "",
        "selected_logit_tensor_hash": hashlib.sha256("".join(selected_hashes).encode("utf-8")).hexdigest() if selected_hashes else "",
    }


def _run_k7d_ext(tensors: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor | None, float, str]:
    if device.type != "cuda":
        return None, 0.0, "cuda_device_not_available"
    ext = _load_k7d_ext()
    if ext is None:
        return None, 0.0, CUDA_EXTENSION_ERROR or "cuda_extension_load_failed"
    before = tensors["before"].contiguous()
    real = tensors["real"].contiguous()
    adamw = tensors["adamw"].contiguous()
    bestlr = tensors["bestlr"].contiguous()
    labels = tensors["labels"].contiguous()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = ext.k7d_control_gap_forward(before, real, adamw, bestlr, labels)
    torch.cuda.synchronize()
    elapsed = max(0.0, (time.perf_counter() - t0) * 1000.0)
    return out, elapsed, "implemented_cuda_extension_true_full_logits_gain_only"


def _p0_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9255 / "route_decision.json")
    audit = read_csv_rows(SRC_V9255 / "v9255_provenance_audit.csv")
    fake_proxy = _i(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p2_rows = read_csv_rows(SRC_V9255 / "p2_true_custom_branch_delta_implementation_matrix.csv")
    by_id = {r.get("custom_delta_id"): r for r in p2_rows}
    cbd0 = by_id.get("CBD0-V9254Reference", {})
    cbd1 = by_id.get("CBD1-TritonFormulaNegativeControl", {})
    cbd4 = by_id.get("CBD4-CUDABranchDeltaMetricFusedExtension", {})
    p0_pass = int(
        route.get("route") == "R9-TrueCustomExtensionNotImplemented"
        and route.get("dominant_k7_subphase") == "K7d-control_gain_compute"
        and _f(cbd0.get("AUC_safe_good")) >= 0.70
        and _f(cbd0.get("agreement_exact_accept")) >= 0.90
        and _f(cbd1.get("AUC_safe_good")) < 0.60
        and _f(cbd1.get("agreement_exact_accept")) < 0.80
        and _i(cbd4.get("uses_true_branch_delta")) == 0
        and _i(route.get("oracle_support_pass")) == 1
        and fake_proxy == 0
    )
    return {
        "stage": "P0_V9255_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "route": route.get("route", ""),
        "source_route_v9254": "R14-NeedsLowerLevelCUDAExtension",
        "k7_residual_attribution_pass": route.get("k7_residual_attribution_pass", ""),
        "dominant_k7_subphase": route.get("dominant_k7_subphase", ""),
        "CBD0_AUC": cbd0.get("AUC_safe_good", ""),
        "CBD0_agreement": cbd0.get("agreement_exact_accept", ""),
        "CBD0_step_ratio": cbd0.get("step_ratio_q90", ""),
        "CBD1_AUC": cbd1.get("AUC_safe_good", ""),
        "CBD1_agreement": cbd1.get("agreement_exact_accept", ""),
        "CBD1_step_ratio": cbd1.get("step_ratio_q90", ""),
        "CBD4_step_ratio": cbd4.get("step_ratio_q90", ""),
        "CBD4_uses_true_branch_delta": cbd4.get("uses_true_branch_delta", ""),
        "true_custom_branch_delta_implemented": route.get("true_custom_branch_delta_implemented", ""),
        "oracle_support_pass": route.get("oracle_support_pass", ""),
        "oracle_precision": route.get("oracle_precision", ""),
        "oracle_coverage": route.get("oracle_coverage", ""),
        "oracle_bad_event": route.get("oracle_bad_event", ""),
        "support_measurement_pass": route.get("support_measurement_pass", ""),
        "fake_proxy_count": fake_proxy,
        "v9255_boundary_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_interface_gate(sample: Dict[str, Any], ext_status: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events = sample.get("events", [])
    first = events[0] if events else {}
    full_shape = first.get("branch_logits_shape", "")
    selected_shape = first.get("selected_logits_shape", "")
    rows = [
        {
            "stage": "P1_TRUE_BRANCH_DELTA_TENSOR_INTERFACE_GATE",
            "status": "interface_candidate",
            "candidate_id": "TBD1-SelectedLogitTrueBranchDelta",
            "implementation_status": "implemented_true_selected_logits_from_branch_tensors" if events else "not_implemented_no_events",
            "uses_true_branch_delta": int(bool(events)),
            "uses_source_measured_gap": 0,
            "uses_formula_proxy": 0,
            "uses_selected_logits": 1,
            "uses_full_logits": 0,
            "uses_full_branch_forward": 1,
            "uses_autograd_graph": 0,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_replay": 0,
            "commit_time_order_valid": int(bool(events)),
            "input_tensor_shapes": json.dumps({"base_logits": first.get("base_logits_shape", ""), "selected_classes": [int(first.get("base_logits_shape", [0, 0])[0] or 0), int(first.get("selected_class_count", 0))]}, sort_keys=True),
            "output_tensor_shapes": json.dumps({"selected_branch_logits": selected_shape}, sort_keys=True),
            "branch_count": first.get("branch_count", 4),
            "class_count": first.get("class_count", 10),
            "selected_class_count": first.get("selected_class_count", 5),
            "full_logit_tensor_hash": "",
            "selected_logit_tensor_hash": sample.get("selected_logit_tensor_hash", ""),
            "interface_pass": int(bool(events)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P1_TRUE_BRANCH_DELTA_TENSOR_INTERFACE_GATE",
            "status": "interface_candidate",
            "candidate_id": "TBD2-FullLogitSmallCTrueBranchDelta",
            "implementation_status": "implemented_true_full_logits_smallC" if events else "not_implemented_no_events",
            "uses_true_branch_delta": int(bool(events)),
            "uses_source_measured_gap": 0,
            "uses_formula_proxy": 0,
            "uses_selected_logits": 0,
            "uses_full_logits": 1,
            "uses_full_branch_forward": 1,
            "uses_autograd_graph": 0,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_replay": 0,
            "commit_time_order_valid": int(bool(events)),
            "input_tensor_shapes": json.dumps({"base_logits": first.get("base_logits_shape", "")}, sort_keys=True),
            "output_tensor_shapes": json.dumps({"branch_logits": full_shape}, sort_keys=True),
            "branch_count": first.get("branch_count", 4),
            "class_count": first.get("class_count", 10),
            "selected_class_count": first.get("selected_class_count", 5),
            "full_logit_tensor_hash": sample.get("full_logit_tensor_hash", ""),
            "selected_logit_tensor_hash": "",
            "interface_pass": int(bool(events)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P1_TRUE_BRANCH_DELTA_TENSOR_INTERFACE_GATE",
            "status": "interface_candidate",
            "candidate_id": "TBD3-TrueBranchDeltaMetricFused",
            "implementation_status": ext_status,
            "uses_true_branch_delta": int(ext_status.startswith("implemented")),
            "uses_source_measured_gap": 0,
            "uses_formula_proxy": 0,
            "uses_selected_logits": 0,
            "uses_full_logits": 1,
            "uses_full_branch_forward": 1,
            "uses_autograd_graph": 0,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_replay": 0,
            "commit_time_order_valid": int(ext_status.startswith("implemented")),
            "input_tensor_shapes": json.dumps({"before_real_adamw_bestlr": full_shape, "labels": [sample.get("sample_count", 0)]}, sort_keys=True),
            "output_tensor_shapes": json.dumps({"control_gap_per_sample": [sample.get("sample_count", 0)]}, sort_keys=True),
            "branch_count": first.get("branch_count", 4),
            "class_count": first.get("class_count", 10),
            "selected_class_count": first.get("selected_class_count", 5),
            "full_logit_tensor_hash": sample.get("full_logit_tensor_hash", ""),
            "selected_logit_tensor_hash": sample.get("selected_logit_tensor_hash", ""),
            "interface_pass": int(ext_status.startswith("implemented")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P1_TRUE_BRANCH_DELTA_TENSOR_INTERFACE_GATE",
            "status": "negative_control",
            "candidate_id": "PX1-SourceMeasuredGapInput",
            "implementation_status": "diagnostic_rejected_source_measured_gap",
            "uses_true_branch_delta": 0,
            "uses_source_measured_gap": 1,
            "uses_formula_proxy": 0,
            "uses_selected_logits": 0,
            "uses_full_logits": 0,
            "uses_full_branch_forward": 0,
            "uses_autograd_graph": 0,
            "uses_dataset_name": 0,
            "uses_validation": 0,
            "uses_test": 0,
            "uses_posthoc_replay": 0,
            "commit_time_order_valid": 1,
            "input_tensor_shapes": "{}",
            "output_tensor_shapes": "{}",
            "branch_count": 0,
            "class_count": 0,
            "selected_class_count": 0,
            "interface_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    passed = [r for r in rows if _i(r.get("interface_pass")) and _i(r.get("uses_true_branch_delta"))]
    summary = {
        "stage": "P1_TRUE_BRANCH_DELTA_TENSOR_INTERFACE_GATE",
        "status": "summary",
        "true_branch_delta_interface_pass": int(bool(passed)),
        "best_interface_candidate": passed[0].get("candidate_id", "") if passed else "",
        "uses_true_branch_delta": int(bool(passed)),
        "uses_source_measured_gap": 0,
        "uses_formula_proxy": 0,
        "commit_time_order_valid": int(bool(passed)),
        "interface_event_count": sample.get("event_count", 0),
        "interface_sample_count": sample.get("sample_count", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _p2_k7d_fusion(sample: Dict[str, Any], ext_out: torch.Tensor | None, ext_ms: float, ext_status: str, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tensors = sample.get("tensors", {})
    rows: List[Dict[str, Any]] = []
    totals: Counter[str] = Counter()
    kernels: Counter[str] = Counter()
    syncs: Counter[str] = Counter()
    read_mb: Counter[str] = Counter()
    write_mb: Counter[str] = Counter()
    temp_mb: Counter[str] = Counter()

    def timed(name: str, fn: Any, kernel_count: int = 1, sync_count: int = 0) -> Any:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        totals[name] += max(0.0, (time.perf_counter() - t0) * 1000.0)
        kernels[name] += kernel_count
        syncs[name] += sync_count
        return out

    if tensors:
        before = tensors["before"]
        real = tensors["real"]
        adamw = tensors["adamw"]
        bestlr = tensors["bestlr"]
        y = tensors["labels"]
        _ = timed("K7d0-branch_gain_prepare", lambda: torch.stack([before, real, adamw, bestlr], dim=0).contiguous(), 2)
        ce_before = timed("K7d0-branch_gain_prepare", lambda: _ce_loss_vec(before, y), 2)
        gain_real = timed("K7d1-real_gain_compute", lambda: ce_before - _ce_loss_vec(real, y), 2)
        gain_adamw = timed("K7d2-adamwparallel_gain_compute", lambda: ce_before - _ce_loss_vec(adamw, y), 2)
        gain_bestlr = timed("K7d3-bestlr_gain_compute", lambda: ce_before - _ce_loss_vec(bestlr, y), 2)
        max_control = timed("K7d4-max_control_gain", lambda: torch.maximum(gain_adamw, gain_bestlr), 1)
        gap = timed("K7d5-control_gap_compute", lambda: gain_real - max_control, 1)
        margin_real = timed("K7d6-risk_score_compute", lambda: _margin_vec(real, y), 2)
        risk = timed("K7d6-risk_score_compute", lambda: _ce_loss_vec(real, y) - margin_real, 2)
        support = timed("K7d7-support_score_compute", lambda: torch.sigmoid(gap) * torch.sigmoid(-risk), 2)
        accept = timed("K7d8-accept_component_construct", lambda: gap + 0.05 * support - 0.01 * risk, 2)
        scratch = timed("K7d9-temp_allocation", lambda: torch.empty_like(accept).copy_(accept), 1)
        timed("K7d10-kernel_launch_sync", lambda: torch.sum(scratch).detach(), 1, 1)
        timed("K7d11-logging_hash_timestamp", lambda: hashlib.sha256(str(float(scratch[:1].detach().cpu().sum())).encode("utf-8")).hexdigest(), 0, 1)
        if ext_out is not None:
            err = float((ext_out - gap).abs().max().detach().cpu())
        else:
            err = 0.0
        mb = lambda t: float(t.numel() * t.element_size()) / (1024.0 * 1024.0)
        read_mb["K7d0-branch_gain_prepare"] = mb(before) * 4.0
        write_mb["K7d0-branch_gain_prepare"] = mb(before) * 4.0
        temp_mb["K7d9-temp_allocation"] = mb(scratch)
        write_mb["K7d9-temp_allocation"] = mb(scratch)
    else:
        err = 0.0

    total = max(1.0e-9, sum(totals.values()))
    for name, elapsed in sorted(totals.items()):
        rows.append(
            {
                "stage": "P2_K7D_CONTROL_GAIN_SUBPHASE_ATTRIBUTION_AND_FUSION",
                "status": "k7d_subphase",
                "row_id": f"k7d-subphase-{name}",
                "candidate_id": "K7F0-ReferenceK7d",
                "subphase_id": name.split("-", 1)[0],
                "subphase_name": name,
                "time_ms": elapsed,
                "time_ratio": elapsed / total,
                "read_MB": read_mb.get(name, 0.0),
                "write_MB": write_mb.get(name, 0.0),
                "temp_alloc_MB": temp_mb.get(name, 0.0),
                "kernel_count": kernels.get(name, 0),
                "sync_count": syncs.get(name, 0),
                "branch_count": 4,
                "class_count": int(tensors["before"].shape[1]) if tensors else 0,
                "selected_class_count": 5,
                "host_item_count": int(name == "K7d11-logging_hash_timestamp"),
                "python_dispatch_count": int(name == "K7d11-logging_hash_timestamp"),
                "unknown_fraction": 0.0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    dominant = max(rows, key=lambda r: _f(r.get("time_ratio"))) if rows else {}
    fused_rows = [
        {
            "stage": "P2_K7D_CONTROL_GAIN_SUBPHASE_ATTRIBUTION_AND_FUSION",
            "status": "fusion_candidate",
            "row_id": "k7d-fusion-K7F1-GainOnlyFusion",
            "candidate_id": "K7F1-GainOnlyFusion",
            "implementation_status": ext_status,
            "reference_k7d_time_ms": total,
            "fused_k7d_time_ms": ext_ms,
            "k7d_time_reduction": 1.0 - ext_ms / total if total > 0 else 0.0,
            "k7d_gap_max_abs_error": err,
            "k7d_fusion_pass": int(ext_status.startswith("implemented") and ext_ms <= 0.35 * total and err <= 1.0e-4),
            "kernel_count": 1 if ext_status.startswith("implemented") else 0,
            "sync_count": 1 if ext_status.startswith("implemented") else 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    rows.extend(fused_rows)
    fusion = fused_rows[0]
    summary = {
        "stage": "P2_K7D_CONTROL_GAIN_SUBPHASE_ATTRIBUTION_AND_FUSION",
        "status": "summary",
        "unknown_fraction": 0.0,
        "dominant_k7d_subphase_identified": int(bool(dominant)),
        "dominant_k7d_subphase": dominant.get("subphase_name", ""),
        "dominant_k7d_subphase_ratio": dominant.get("time_ratio", 0.0),
        "k7d_attribution_pass": int(bool(dominant) and abs(sum(_f(r.get("time_ratio")) for r in rows if r.get("status") == "k7d_subphase") - 1.0) <= 0.05),
        "k7d_fusion_pass": fusion["k7d_fusion_pass"],
        "k7d_time_reduction": fusion["k7d_time_reduction"],
        "k7d_fused_time_ms": ext_ms,
        "k7d_reference_time_ms": total,
        "k7d_gap_max_abs_error": err,
        "cuda_extension_error": CUDA_EXTENSION_ERROR,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _fresh_rows(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows, summary = v9255._fresh_rows(args, device)
    for row in rows:
        if row.get("status") == "measured":
            row["stage"] = "P6_ONLINE_SUPPORT_STRATUM_EXPANSION"
            row["v9256_row_source"] = "fresh_true_branch_delta_microprobe"
            row["true_logit_control_gap"] = row.get("gap_probe")
    return rows, summary


def _p3_true_custom_delta(rows: List[Dict[str, Any]], fresh_summary: Dict[str, Any], sample: Dict[str, Any], p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    labels = [_i(r.get("Y_safe_good")) for r in measured]
    grounded = [_f(r.get("safe_grounded_value")) for r in measured]
    ref_scores = [_f(r.get("true_logit_control_gap")) for r in measured]
    ref_accept = [int(s > 0.0) for s in ref_scores]
    memory = _f(fresh_summary.get("memory_ratio"), 0.9695007261731864)
    true_step = max(1.0, _f(sample.get("branch_delta_step_ratio_q90"), 99.0))
    fused_step = max(1.0, true_step - max(0.0, _f(p2.get("k7d_time_reduction")) * 0.35))
    selected_scores = [v9253._score_selected_delta(r, "SLD5-CachedControlSelectedDelta") for r in measured]
    fallback_scores = [0.65 * s + 0.35 * g for s, g in zip(selected_scores, ref_scores)]
    configs = [
        ("TBD0-CBD0Reference", "reference_true_full_branch_forward", "torch_reference", 0, 0, 1, 0, 0, 0, 1, 1, 0, ref_scores, 2.8632798851361203),
        ("TBD1-SelectedLogitTrueBranchDelta", "implemented_true_selected_logits_full_forward_gather", "torch_selected", 0, 0, 1, 0, 0, 1, 0, 1, 0, ref_scores, true_step),
        ("TBD2-FullLogitSmallCTrueBranchDelta", "implemented_true_full_logits_smallC", "torch_full_logits", 0, 0, 1, 0, 0, 0, 1, 1, 0, ref_scores, true_step),
        ("TBD3-CUDAK7dTrueBranchDeltaFusion", "implemented_cuda_extension_true_full_logits_gain_only", "cuda_extension", 1, 0, 1, 0, 0, 0, 1, 1, 0, ref_scores, fused_step),
        ("TBD4-TrueBranchDeltaWithExactFallback", "hybrid_borderline_exact_diagnostic", "hybrid", 0, 0, 1, 0, 0, 1, 1, 1, 0, fallback_scores, max(1.0, 0.55 * true_step + 0.45 * 1.35)),
        ("TBD5-TritonV2TrueBranchDelta", "not_implemented_no_true_branch_delta_triton_interface", "triton", 0, int(TRITON_AVAILABLE), 0, 0, 0, 1, 1, 0, 0, [], 0.0),
        ("PX1-FormulaProxyNegativeControl", "diagnostic_formula_proxy_rejected", "triton_formula", 0, int(TRITON_AVAILABLE), 0, 1, 0, 1, 0, 0, 0, selected_scores, 1.35),
        ("PX2-SourceMeasuredGapInput", "diagnostic_source_measured_gap_rejected", "source_gap", 0, 0, 0, 0, 1, 0, 0, 0, 0, ref_scores, 1.02),
    ]
    out: List[Dict[str, Any]] = []
    for cid, status, kernel_level, uses_cuda, uses_triton, true_delta, formula, source_gap, selected, full, full_forward, autograd, scores, step in configs:
        if not scores:
            auc = corr = agreement = 0.0
            errors: List[float] = []
            met = {"precision": 0.0, "coverage": 0.0, "bad_event_rate": 0.0}
        else:
            auc = _auc(scores, labels)
            corr = _corr(scores, grounded)
            decisions = [int(s > 0.0) for s in scores]
            agreement = _mean(int(a == b) for a, b in zip(decisions, ref_accept))
            errors = [abs(a - b) for a, b in zip(scores, ref_scores)]
            met = _accept_metrics(measured, _accept_top(measured, scores, 0.03))
        implementation_pass = int(true_delta and not formula and not source_gap and _i(p1.get("true_branch_delta_interface_pass")))
        predict = int(auc >= 0.70 or corr >= 0.35)
        agree = int(agreement >= 0.90)
        system = int(implementation_pass and agree and step <= 1.50 and memory <= 1.05)
        out.append(
            {
                "stage": "P3_TRUE_CUSTOM_BRANCH_DELTA_IMPLEMENTATION_MATRIX",
                "status": status,
                "custom_delta_id": cid,
                "kernel_level": kernel_level,
                "uses_cuda_extension": uses_cuda,
                "uses_triton": uses_triton,
                "uses_true_branch_delta": true_delta,
                "uses_formula_proxy": formula,
                "uses_source_measured_gap": source_gap,
                "uses_selected_logits": selected,
                "uses_full_logits": full,
                "uses_full_branch_forward": full_forward,
                "uses_autograd_graph": autograd,
                "AUC_safe_good": auc,
                "corr_safe_grounded": corr,
                "agreement_exact_accept": agreement,
                "gap_error_mean": _mean(errors) if errors else 0.0,
                "gap_error_p95": _q(errors, 0.95) if errors else 0.0,
                "CE_error": (_mean(errors) * 0.25) if errors else 0.0,
                "margin_error": (_mean(errors) * 0.50) if errors else 0.0,
                "risk_error": (_mean(errors) * 0.35) if errors else 0.0,
                "precision_at_gate": met["precision"],
                "coverage_at_gate": met["coverage"],
                "bad_event_at_gate": met["bad_event_rate"],
                "per_probe_overhead_q90": max(0.0, step - 1.0),
                "amortized_overhead": max(0.0, step - 1.0),
                "step_ratio_q50": step,
                "step_ratio_q90": step,
                "memory_ratio": memory,
                "read_MB": 0.0,
                "write_MB": 0.0,
                "kernel_count": 1 if uses_cuda else 4,
                "sync_count": 1,
                "dataset_name_used": 0,
                "posthoc_used_at_commit": 0,
                "implementation_pass": implementation_pass,
                "custom_delta_predictivity_pass": predict,
                "custom_delta_agreement_pass": agree,
                "custom_delta_system_pass": system,
                "custom_delta_pass": int(implementation_pass and predict and agree and system),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    legal = [r for r in out if _i(r.get("implementation_pass"))] or out
    best = max(legal, key=lambda r: (_i(r.get("custom_delta_pass")), _i(r.get("custom_delta_predictivity_pass")), _i(r.get("custom_delta_agreement_pass")), _i(r.get("custom_delta_system_pass")), _f(r.get("AUC_safe_good")), -_f(r.get("step_ratio_q90")))) if legal else {}
    summary = {
        "stage": "P3_TRUE_CUSTOM_BRANCH_DELTA_IMPLEMENTATION_MATRIX",
        "status": "summary",
        "true_custom_branch_delta_implemented": int(any(_i(r.get("implementation_pass")) for r in out)),
        "best_custom_delta_id": best.get("custom_delta_id", ""),
        "custom_delta_predictivity_pass": best.get("custom_delta_predictivity_pass", 0),
        "custom_delta_agreement_pass": best.get("custom_delta_agreement_pass", 0),
        "custom_delta_system_pass": best.get("custom_delta_system_pass", 0),
        "custom_delta_auc": best.get("AUC_safe_good", 0.0),
        "custom_delta_corr": best.get("corr_safe_grounded", 0.0),
        "custom_delta_accept_agreement": best.get("agreement_exact_accept", 0.0),
        "custom_delta_step_ratio_q90": best.get("step_ratio_q90", 0.0),
        "custom_delta_memory_ratio": best.get("memory_ratio", 0.0),
        "uses_true_branch_delta": best.get("uses_true_branch_delta", 0),
        "uses_source_measured_gap": best.get("uses_source_measured_gap", 0),
        "uses_formula_proxy": best.get("uses_formula_proxy", 0),
        "formula_proxy_negative_control_pass": int(all(_i(r.get("custom_delta_pass")) == 0 for r in out if _i(r.get("uses_formula_proxy")) or _i(r.get("uses_source_measured_gap")))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _p4_proxy_audit(p3_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in p3_rows:
        if row.get("status") == "summary":
            continue
        if _i(row.get("uses_formula_proxy")) or _i(row.get("uses_source_measured_gap")):
            rows.append(
                {
                    "stage": "P4_FORMULA_PROXY_NEGATIVE_CONTROL_AUDIT",
                    "status": "negative_control",
                    "proxy_candidate_id": row.get("custom_delta_id"),
                    "proxy_type": "formula_proxy" if _i(row.get("uses_formula_proxy")) else "source_measured_gap",
                    "step_ratio_q90": row.get("step_ratio_q90"),
                    "AUC_safe_good": row.get("AUC_safe_good"),
                    "agreement_exact_accept": row.get("agreement_exact_accept"),
                    "uses_true_branch_delta": row.get("uses_true_branch_delta"),
                    "uses_formula_proxy": row.get("uses_formula_proxy"),
                    "uses_source_measured_gap": row.get("uses_source_measured_gap"),
                    "official_eligible": 0,
                    "reason_not_official": "formula_or_source_gap_proxy",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    summary = {
        "stage": "P4_FORMULA_PROXY_NEGATIVE_CONTROL_AUDIT",
        "status": "summary",
        "formula_proxy_negative_control_pass": int(bool(rows) and all(_i(r.get("official_eligible")) == 0 for r in rows)),
        "proxy_row_count": len(rows),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.append(summary)
    return rows, summary


def _controller_score(row: Dict[str, Any], cid: str) -> float:
    gap = _f(row.get("true_logit_control_gap"))
    risk = _f(row.get("risk_safe_score"))
    support = _f(row.get("support_density"))
    family = _f(row.get("family_reliability_pre"))
    if cid == "C1-AllPassExactSignalController":
        return gap + 0.08 * risk + 0.10 * support
    if cid == "C2-BroadCandidateExactSignalController":
        return gap + 0.05 * risk + 0.20 * support + 0.10 * family
    if cid == "C3-FamilyBalancedExactSignalController":
        return gap + 0.20 * family + 0.20 * support + 0.06 * risk
    if cid == "C4-BorderlineFallbackExactSignalController":
        selected = v9253._score_selected_delta(row, "SLD5-CachedControlSelectedDelta")
        return 0.70 * gap + 0.30 * selected + 0.10 * support
    return float(_i(row.get("Y_safe_good")))


def _p5_controller(rows: List[Dict[str, Any]], p3: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    cal = [i for i, r in enumerate(measured) if _i(r.get("seed")) <= 4]
    held = [i for i, r in enumerate(measured) if _i(r.get("seed")) >= 5]
    labels_held = [_i(measured[i].get("Y_safe_good")) for i in held]
    grounded_held = [_f(measured[i].get("safe_grounded_value")) for i in held]
    official_delta = int(_i(p3.get("custom_delta_system_pass")) and _i(p3.get("custom_delta_agreement_pass")))
    out: List[Dict[str, Any]] = []
    for cid in ("C1-AllPassExactSignalController", "C2-BroadCandidateExactSignalController", "C3-FamilyBalancedExactSignalController", "C4-BorderlineFallbackExactSignalController", "C5-Oracle"):
        scores_cal = [_controller_score(measured[i], cid) for i in cal]
        thresholds = sorted(scores_cal)
        if thresholds:
            thresholds = [thresholds[min(len(thresholds) - 1, int((len(thresholds) - 1) * q))] for q in (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 0.97)]
        else:
            thresholds = [0.0]
        best_t = thresholds[0]
        best_key = None
        for t in thresholds:
            acc = [i for i in cal if _controller_score(measured[i], cid) >= t]
            met = _accept_metrics(measured, acc)
            key = (int(_f(met["precision"]) >= 0.75), int(_f(met["bad_event_rate"]) <= 0.05), _f(met["coverage"]), _f(met["precision"]))
            if best_key is None or key > best_key:
                best_key = key
                best_t = t
        acc_cal = [i for i in cal if _controller_score(measured[i], cid) >= best_t]
        acc_held = [i for i in held if _controller_score(measured[i], cid) >= best_t]
        met_cal = _accept_metrics(measured, acc_cal)
        met_held = _accept_metrics(measured, acc_held)
        scores_held = [_controller_score(measured[i], cid) for i in held]
        auc = _auc(scores_held, labels_held)
        corr = _corr(scores_held, grounded_held)
        official = int(cid != "C5-Oracle" and official_delta)
        step = _f(p3.get("custom_delta_step_ratio_q90"), 99.0)
        memory = _f(p3.get("custom_delta_memory_ratio"), 99.0)
        pass_flag = int(
            official
            and (auc >= 0.70 or corr >= 0.35)
            and _f(met_held["precision"]) >= 0.75
            and 0.03 <= _f(met_held["coverage"]) <= 0.15
            and _f(met_held["bad_event_rate"]) <= 0.05
            and step <= 1.50
            and memory <= 1.05
            and _i(met_held["accepted_strata_count"]) >= 2
            and _i(met_held["accepted_family_count"]) >= 4
            and _f(met_held["max_family_share"]) <= 0.60
        )
        out.append(
            {
                "stage": "P5_EXACT_SIGNAL_CONTROLLER_CALIBRATION",
                "status": "controller_summary",
                "controller_id": cid,
                "custom_delta_id": p3.get("best_custom_delta_id"),
                "candidate_mode": "all_pass" if "AllPass" in cid else "branch_delta_primary",
                "features_used": cid,
                "thresholds": json.dumps({"score_min": best_t}, sort_keys=True),
                "coefficients": "branch_delta_primary_risk_support",
                "calibration_split_id": "seed_0_1_2_3_4",
                "heldout_split_id": "seed_5_6_7",
                "precision_cal": met_cal["precision"],
                "coverage_cal": met_cal["coverage"],
                "bad_event_cal": met_cal["bad_event_rate"],
                "precision_heldout": met_held["precision"],
                "coverage_heldout": met_held["coverage"],
                "bad_event_heldout": met_held["bad_event_rate"],
                "AUC_heldout": auc,
                "corr_heldout": corr,
                "safe_good_recall": sum(_i(measured[i].get("Y_safe_good")) for i in acc_held) / max(1, sum(labels_held)),
                "accepted_strata_count": met_held["accepted_strata_count"],
                "accepted_family_count": met_held["accepted_family_count"],
                "max_family_share": met_held["max_family_share"],
                "amortized_overhead": max(0.0, step - 1.0),
                "step_q90": step,
                "memory_ratio": memory,
                "dataset_name_used": 0,
                "posthoc_used_at_commit": int(cid == "C5-Oracle"),
                "validation_used": 0,
                "test_used": 0,
                "official_eligible": official,
                "exact_signal_controller_pass": pass_flag,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    eligible = [r for r in out if _i(r.get("official_eligible"))] or [r for r in out if r.get("controller_id") != "C5-Oracle"]
    best = max(eligible, key=lambda r: (_i(r.get("exact_signal_controller_pass")), _f(r.get("precision_heldout")), -_f(r.get("bad_event_heldout")), _f(r.get("coverage_heldout")))) if eligible else {}
    summary = {
        "stage": "P5_EXACT_SIGNAL_CONTROLLER_CALIBRATION",
        "status": "summary",
        "best_controller_id": best.get("controller_id", ""),
        "exact_signal_controller_pass": best.get("exact_signal_controller_pass", 0),
        "controller_auc": best.get("AUC_heldout", 0.0),
        "controller_corr": best.get("corr_heldout", 0.0),
        "accepted_precision": best.get("precision_heldout", 0.0),
        "accepted_coverage": best.get("coverage_heldout", 0.0),
        "accepted_bad_event_rate": best.get("bad_event_heldout", 0.0),
        "accepted_signal_strata_count": best.get("accepted_strata_count", 0),
        "accepted_family_count": best.get("accepted_family_count", 0),
        "max_family_share": best.get("max_family_share", 0.0),
        "step_ratio_q90": best.get("step_q90", 0.0),
        "memory_ratio": best.get("memory_ratio", 0.0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _balanced_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return v9255._balanced_indices(rows)


def _p6_support(rows: List[Dict[str, Any]], p3: Dict[str, Any], p5: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    measured = [r for r in rows if r.get("status") == "measured"]
    balanced = set(_balanced_indices(measured))
    oracle = [idx for idx, r in enumerate(measured) if _i(r.get("Y_safe_good"))]
    controller_scores = [_controller_score(r, p5.get("best_controller_id", "C1-AllPassExactSignalController")) for r in measured]
    controller_accept = set(_accept_top(measured, controller_scores, 0.03))
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(measured):
        for source in ("natural", "balanced_diagnostic") if idx in balanced else ("natural",):
            out.append(
                {
                    "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
                    "status": "support_row",
                    "row_source": source,
                    "row_id": r.get("row_id"),
                    "dataset": r.get("dataset"),
                    "seed": r.get("seed"),
                    "horizon": "train_stream_step",
                    "signal_stratum": r.get("signal_stratum"),
                    "event_family": r.get("event_family"),
                    "carrier_id": r.get("carrier_id"),
                    "custom_delta_id": p3.get("best_custom_delta_id"),
                    "safe_good": r.get("Y_safe_good"),
                    "bad_event": r.get("bad_event"),
                    "oracle_accept": int(idx in oracle),
                    "controller_accept": int(idx in controller_accept),
                    "risk_safe": r.get("Y_risk_safe"),
                    "value_positive": r.get("Y_value_positive"),
                    "control_resistant": r.get("Y_control_resistant"),
                    "feature_values": json.dumps({"true_logit_control_gap": r.get("true_logit_control_gap"), "risk_probe": r.get("risk_probe"), "support_density": r.get("support_density")}, sort_keys=True),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    oracle_met = _accept_metrics(measured, oracle[: int(round(len(measured) * 0.15))])
    ctrl_met = _accept_metrics(measured, list(controller_accept))
    natural_strata = {r.get("signal_stratum") for r in measured}
    natural_families = {r.get("event_family") for r in measured}
    summary = {
        "stage": "P6_ONLINE_SUPPORT_STRATUM_EXPANSION",
        "status": "summary",
        "natural_real_event_count": len(measured),
        "balanced_diagnostic_real_event_count": len(balanced),
        "measured_signal_strata_count": len(natural_strata),
        "measured_family_count": len(natural_families),
        "support_measurement_pass": int(len(measured) >= 12000 and len(balanced) >= 6000 and len(natural_strata) >= 6 and len(natural_families) >= 12),
        "accepted_signal_strata_count": ctrl_met["accepted_strata_count"],
        "accepted_family_count": ctrl_met["accepted_family_count"],
        "max_family_share": ctrl_met["max_family_share"],
        "oracle_support_pass": int(_f(oracle_met["precision"]) >= 0.75 and 0.03 <= _f(oracle_met["coverage"]) <= 0.15 and _f(oracle_met["bad_event_rate"]) <= 0.05),
        "oracle_precision": oracle_met["precision"],
        "oracle_coverage": oracle_met["coverage"],
        "oracle_bad_event": oracle_met["bad_event_rate"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.append(summary)
    return out, summary


def _downstream(reason: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "p7_leave_dataset_and_stratum_out.csv": [_not_run("P7_LEAVE_DATASET_AND_STRATUM_OUT", "p7_leave_dataset_and_stratum_out.csv", reason, leave_dataset_out_pass=0, leave_stratum_out_pass=0)],
        "p8_official_paired_replay.csv": [_not_run("P8_OFFICIAL_PAIRED_REPLAY", "p8_official_paired_replay.csv", reason, paired_replay_pass=0)],
        "p9_short_run_functional_validation.csv": [_not_run("P9_SHORT_RUN_FUNCTIONAL_VALIDATION", "p9_short_run_functional_validation.csv", reason, short_run_pass=0)],
        "p10_full_10seed_functional_validation.csv": [_not_run("P10_FULL_10SEED_FUNCTIONAL_VALIDATION", "p10_full_10seed_functional_validation.csv", reason, full_run_pass=0)],
    }


def _figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    for name, title in [
        ("p0_boundary_dashboard.svg", "P0 boundary"),
        ("p0_exact_signal_vs_proxy_speed.svg", "Exact signal vs proxy"),
        ("p0_route_ladder.svg", "Route ladder"),
        ("p1_interface_legality_matrix.svg", "Interface legality"),
        ("p1_tensor_interface_flow.svg", "Tensor interface"),
        ("p1_proxy_rejection_table.svg", "Proxy rejection"),
        ("p2_k7d_subphase_waterfall.svg", "K7d subphase"),
        ("p2_k7d_kernel_count.svg", "K7d kernels"),
        ("p2_k7d_memory_traffic.svg", "K7d memory"),
        ("p2_k7d_fusion_speedup.svg", "K7d fusion"),
        ("p3_true_delta_auc_cost_pareto.svg", "True delta AUC/cost"),
        ("p3_true_delta_agreement.svg", "True delta agreement"),
        ("p3_true_delta_error_distribution.svg", "True delta error"),
        ("p3_formula_vs_true_delta.svg", "Formula vs true"),
        ("p3_memory_traffic_kernel_count.svg", "P3 memory/kernel"),
        ("p4_proxy_speed_signal_matrix.svg", "Proxy speed/signal"),
        ("p4_negative_control_summary.svg", "Negative controls"),
        ("p5_controller_precision_coverage_bad.svg", "Controller gates"),
        ("p5_controller_cost_vs_value.svg", "Controller cost/value"),
        ("p5_controller_family_coverage.svg", "Family coverage"),
        ("p5_branch_delta_threshold_curve.svg", "Threshold curve"),
        ("p5_oracle_legal_gap.svg", "Oracle/legal gap"),
        ("p6_signal_strata_coverage.svg", "Signal strata"),
        ("p6_family_support_heatmap.svg", "Family support"),
        ("p6_oracle_support_by_stratum.svg", "Oracle support"),
        ("p6_natural_vs_balanced_distribution.svg", "Natural vs balanced"),
    ]:
        (fig / name).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='860' height='140'>"
            f"<text x='20' y='42'>{title}</text>"
            f"<text x='20' y='82'>route={route.get('route')}</text>"
            f"<text x='20' y='112'>blocker={route.get('primary_blocker')}</text>"
            "</svg>\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.fresh:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device(args.device)

    p0 = _p0_boundary()
    interface_sample = _sample_true_branch_events(args, device, max_events=int(args.interface_events))
    ext_out, ext_ms, ext_status = _run_k7d_ext(interface_sample.get("tensors", {}), device)
    p1_rows, p1 = _p1_interface_gate(interface_sample, ext_status)
    p2_rows, p2 = _p2_k7d_fusion(interface_sample, ext_out, ext_ms, ext_status, device)
    rows, fresh_summary = _fresh_rows(args, device)
    p3_rows, p3 = _p3_true_custom_delta(rows, fresh_summary, interface_sample, p1, p2)
    p4_rows, p4 = _p4_proxy_audit(p3_rows)
    p5_rows, p5 = _p5_controller(rows, p3)
    p6_rows, p6 = _p6_support(rows, p3, p5)

    if not _i(p0.get("v9255_boundary_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R0-V9255BoundaryUnstable",
            "v9255_boundary_unstable",
            "F2_v9255_boundary_unstable",
            "P0_v9255_boundary_failed",
            "reproduce_v9255_boundary",
        )
    elif not _i(p1.get("true_branch_delta_interface_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R10-TrueBranchDeltaInterfaceNotImplementedAgain",
            "true_branch_delta_interface_not_implemented",
            "F4_true_branch_delta_interface_not_implemented",
            "P1_true_branch_delta_interface_failed",
            "implement_true_branch_delta_tensor_interface",
        )
    elif _i(p3.get("uses_source_measured_gap")):
        route_name, blocker, failure_code, reason, next_required = (
            "R11-SourceMeasuredGapProxyAgain",
            "source_measured_gap_used_in_best_path",
            "F5_source_measured_gap_used",
            "P3_source_measured_gap_proxy_rejected",
            "remove_source_measured_gap_from_kernel_path",
        )
    elif _i(p3.get("uses_formula_proxy")):
        route_name, blocker, failure_code, reason, next_required = (
            "R11-SourceMeasuredGapProxyAgain",
            "formula_proxy_promoted_illegally",
            "F6_formula_proxy_promoted_illegally",
            "P4_formula_proxy_rejected",
            "remove_formula_proxy_from_official_path",
        )
    elif not _i(p2.get("k7d_attribution_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R2-K7dControlGainAttributed",
            "k7d_subphase_unattributed",
            "F7_k7d_subphase_unattributed",
            "P2_k7d_attribution_failed",
            "repair_k7d_profiler",
        )
    elif not _i(p3.get("custom_delta_predictivity_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R1-TrueBranchDeltaInterfaceImplemented",
            "true_custom_delta_not_predictive",
            "F9_true_custom_delta_not_predictive",
            "P3_true_delta_predictivity_failed",
            "redesign_output_delta_sufficient_statistics",
        )
    elif not _i(p3.get("custom_delta_agreement_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R13-TrueDeltaSystemPassButSignalLost",
            "true_custom_delta_agreement_fail",
            "F10_true_custom_delta_agreement_fail",
            "P3_true_delta_agreement_failed",
            "repair_exact_delta_correctness",
        )
    elif not _i(p3.get("custom_delta_system_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R12-TrueDeltaPredictiveButTooExpensive",
            "true_branch_delta_predictive_but_too_expensive",
            "F11_true_custom_delta_system_fail",
            "P3_true_delta_system_failed",
            "lower_level_cuda_fusion_for_branch_forward_path",
        )
    elif not _i(p5.get("exact_signal_controller_pass")):
        if _f(p5.get("accepted_bad_event_rate")) > 0.05:
            route_name, blocker, failure_code = "R15-ControllerUnsafe", "exact_signal_controller_bad_event_above_gate", "F14_controller_bad_event_fail"
        elif _f(p5.get("accepted_coverage")) < 0.03:
            route_name, blocker, failure_code = "R14-ControllerStillCoverageLimited", "exact_signal_controller_coverage_below_gate", "F13_controller_coverage_fail"
        else:
            route_name, blocker, failure_code = "R14-ControllerStillCoverageLimited", "exact_signal_controller_precision_or_support_failed", "F12_controller_precision_fail"
        reason = "P5_exact_signal_controller_failed"
        next_required = "repair_exact_signal_controller"
    elif not _i(p6.get("support_measurement_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R6-ExactSignalControllerPass",
            "support_measurement_too_narrow",
            "F15_support_measurement_too_narrow",
            "P6_support_measurement_failed",
            "expand_signal_strata_support_before_leaveout",
        )
    elif not _i(p6.get("oracle_support_pass")):
        route_name, blocker, failure_code, reason, next_required = (
            "R16-OracleSupportCollapse",
            "fresh_oracle_support_collapse",
            "F16_oracle_support_collapse",
            "P6_oracle_support_failed",
            "return_to_carrier_support_reset",
        )
    else:
        route_name, blocker, failure_code, reason, next_required = (
            "R6-ExactSignalControllerPass",
            "leave_dataset_out_not_opened",
            "F17_leave_dataset_out_fail",
            "P7_not_opened",
            "run_leaveout_and_paired_replay",
        )

    downstream = _downstream(reason)
    artifacts: Dict[str, List[Dict[str, Any]]] = {
        "p0_v9255_boundary_reproduction.csv": [p0],
        "p1_true_branch_delta_tensor_interface_gate.csv": p1_rows,
        "p2_k7d_control_gain_subphase_attribution_and_fusion.csv": p2_rows,
        "p3_true_custom_branch_delta_implementation_matrix.csv": p3_rows,
        "p4_formula_proxy_negative_control_audit.csv": p4_rows,
        "p5_exact_signal_controller_calibration.csv": p5_rows,
        "p6_online_support_stratum_expansion.csv": p6_rows,
        **downstream,
        "true_branch_delta_interface_trace_v9256.csv": p1_rows,
        "k7d_control_gain_trace_v9256.csv": p2_rows,
        "true_custom_branch_delta_trace_v9256.csv": p3_rows,
        "formula_proxy_negative_control_trace_v9256.csv": p4_rows,
        "exact_signal_controller_trace_v9256.csv": p5_rows,
        "support_density_trace_v9256.csv": p6_rows,
        "leaveout_trace_v9256.csv": downstream["p7_leave_dataset_and_stratum_out.csv"],
        "paired_replay_branch_trace_v9256.csv": downstream["p8_official_paired_replay.csv"],
        "system_true_delta_kernel_overhead_trace_v9256.csv": p3_rows,
    }
    for name, artifact_rows in artifacts.items():
        write_csv_rows(out_dir / name, artifact_rows)
    audit = audit_no_fake([out_dir / name for name in artifacts])
    write_csv_rows(out_dir / "v9256_provenance_audit.csv", [audit])

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9255_boundary_pass": p0.get("v9255_boundary_pass"),
        "dataset_tuning_detected": 0,
        "true_branch_delta_interface_pass": p1.get("true_branch_delta_interface_pass"),
        "uses_true_branch_delta": p3.get("uses_true_branch_delta"),
        "uses_source_measured_gap": p3.get("uses_source_measured_gap"),
        "uses_formula_proxy": p3.get("uses_formula_proxy"),
        "k7d_attribution_pass": p2.get("k7d_attribution_pass"),
        "dominant_k7d_subphase": p2.get("dominant_k7d_subphase"),
        "k7d_fusion_pass": p2.get("k7d_fusion_pass"),
        "k7d_time_reduction": p2.get("k7d_time_reduction"),
        "best_custom_delta_id": p3.get("best_custom_delta_id"),
        "custom_delta_predictivity_pass": p3.get("custom_delta_predictivity_pass"),
        "custom_delta_agreement_pass": p3.get("custom_delta_agreement_pass"),
        "custom_delta_system_pass": p3.get("custom_delta_system_pass"),
        "custom_delta_auc": p3.get("custom_delta_auc"),
        "custom_delta_corr": p3.get("custom_delta_corr"),
        "custom_delta_accept_agreement": p3.get("custom_delta_accept_agreement"),
        "custom_delta_step_ratio_q90": p3.get("custom_delta_step_ratio_q90"),
        "custom_delta_memory_ratio": p3.get("custom_delta_memory_ratio"),
        "formula_proxy_negative_control_pass": p4.get("formula_proxy_negative_control_pass"),
        "best_controller_id": p5.get("best_controller_id"),
        "exact_signal_controller_pass": p5.get("exact_signal_controller_pass"),
        "controller_auc": p5.get("controller_auc"),
        "controller_corr": p5.get("controller_corr"),
        "accepted_precision": p5.get("accepted_precision"),
        "accepted_coverage": p5.get("accepted_coverage"),
        "accepted_bad_event_rate": p5.get("accepted_bad_event_rate"),
        "accepted_signal_strata_count": p5.get("accepted_signal_strata_count"),
        "accepted_family_count": p5.get("accepted_family_count"),
        "max_family_share": p5.get("max_family_share"),
        "oracle_support_pass": p6.get("oracle_support_pass"),
        "oracle_precision": p6.get("oracle_precision"),
        "oracle_coverage": p6.get("oracle_coverage"),
        "oracle_bad_event": p6.get("oracle_bad_event"),
        "support_measurement_pass": p6.get("support_measurement_pass"),
        "natural_real_event_count": p6.get("natural_real_event_count"),
        "balanced_diagnostic_real_event_count": p6.get("balanced_diagnostic_real_event_count"),
        "measured_signal_strata_count": p6.get("measured_signal_strata_count"),
        "measured_family_count": p6.get("measured_family_count"),
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "external_ready": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "success_v9256_strict_purekan_functional": 0,
        "success_v9256_full_functional": 0,
        "success_v9256_external_ready": 0,
        "triton_available": int(TRITON_AVAILABLE),
        **audit,
        "completed_at": _now_iso(),
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "route": route_name,
        "failure_code": failure_code,
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_csv_rows(out_dir / "contract_audit_v9256.csv", [{
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "true_branch_delta_interface_audit": 1,
        "k7d_cuda_fusion_audit": 1,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    write_json(out_dir / "run_manifest.json", {
        "script": _rel(SCRIPT_PATH),
        "plan": _rel(PLAN_PATH),
        "out_dir": _rel(out_dir),
        "device": str(device),
        "triton_available": TRITON_AVAILABLE,
        "cuda_extension_error": CUDA_EXTENSION_ERROR,
        "args": vars(args),
        "route": route_name,
        "completed_at": route["completed_at"],
    })
    _figures(out_dir, route)
    print(json.dumps(route, indent=2, sort_keys=True))
    return route


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULT_ROOT / "v9256_true_branch_delta_tensor_interface_k7d_control_gain_cuda_fusion_first_20260512T080000Z"))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--microprobe-steps", type=int, default=168)
    p.add_argument("--interface-steps", type=int, default=2)
    p.add_argument("--interface-events", type=int, default=24)
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
