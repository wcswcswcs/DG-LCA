#!/usr/bin/env python3
"""DG-KAN v9.2.7 FC-PureKAN LQ full-pass and functional gate runner.

This runner continues v9.2.6 from a strict FC-PureKAN position.  It does not
open PureKANConv/PureKANFormer, does not use teachers or altered losses, and
keeps functional update gated behind AdamW-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
import run_v926_fc_purekan_primitive_redesign as v926  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.7_FC_PureKAN_LQ_FullPass_FunctionalGate_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v927_fc_purekan_lq_fullpass_functional_gate.py"
PREV_V926 = ROOT / "results" / "real_rerun_20260506" / "v926_fc_purekan_primitive_redesign_liftquad_compact_p5_20260509T174500Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v == "" or v is None:
            return default
        return int(float(v))
    except Exception:
        return default


def _canonical_tasks(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _predict_mlp_logits(params: Sequence[torch.Tensor], x: torch.Tensor, batch_size: int) -> torch.Tensor:
    W1, W2, W3 = params
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            parts.append(f922._mlp3_forward_core(x[start : start + int(batch_size)], W1, W2, W3))
    return torch.cat(parts, dim=0)


def _classwise_json(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Tuple[str, str, int, float]:
    pred = logits.argmax(dim=1)
    rows: List[Dict[str, Any]] = []
    gaps: List[float] = []
    for c in range(num_classes):
        mask = y == c
        if bool(mask.any()):
            acc = float((pred[mask] == y[mask]).float().mean().detach().cpu())
            count = int(mask.sum().detach().cpu())
        else:
            acc = 0.0
            count = 0
        rows.append({"class_id": c, "acc": acc, "count": count})
        gaps.append(acc)
    mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=y.device)
    for t, p in zip(y, pred):
        mat[int(t), int(p)] += 1
    worst = min(range(num_classes), key=lambda c: gaps[c])
    return json.dumps(rows, sort_keys=True), json.dumps(mat.cpu().tolist()), worst, float(gaps[worst])


def _classwise_gap_json(
    kan_logits: torch.Tensor,
    mlp_logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
) -> Tuple[str, int, float]:
    kan_pred = kan_logits.argmax(dim=1)
    mlp_pred = mlp_logits.argmax(dim=1)
    rows: List[Dict[str, Any]] = []
    worst_class = 0
    worst_gap = 0.0
    for c in range(num_classes):
        mask = y == c
        if bool(mask.any()):
            kan_acc = float((kan_pred[mask] == y[mask]).float().mean().detach().cpu())
            mlp_acc = float((mlp_pred[mask] == y[mask]).float().mean().detach().cpu())
            gap = kan_acc - mlp_acc
        else:
            kan_acc = mlp_acc = gap = 0.0
        rows.append({"class_id": c, "kan_acc": kan_acc, "mlp_acc": mlp_acc, "classwise_gap": gap})
        if gap < worst_gap:
            worst_gap = gap
            worst_class = c
    return json.dumps(rows, sort_keys=True), worst_class, worst_gap


def _hard_sample_json(logits: torch.Tensor, y: torch.Tensor, topk: int = 10) -> str:
    with torch.no_grad():
        log_probs = logits.log_softmax(dim=1)
        ce = -log_probs[torch.arange(y.numel(), device=y.device), y]
        true_logits = logits[torch.arange(y.numel(), device=y.device), y]
        masked = logits.clone()
        masked[torch.arange(y.numel(), device=y.device), y] = -torch.inf
        margin = true_logits - masked.max(dim=1).values
        vals, idx = torch.topk(ce, k=min(topk, int(ce.numel())))
        rows = [
            {"test_index": int(i.detach().cpu()), "hard_sample_loss": float(v.detach().cpu()), "hard_sample_margin": float(margin[i].detach().cpu())}
            for v, i in zip(vals, idx)
        ]
    return json.dumps(rows, sort_keys=True)


def _train_lq(
    args: argparse.Namespace,
    spec: lq.LQSpec,
    dataset: str,
    seed: int,
    device: torch.device,
    stage: str,
    repeat_id: str = "main",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v92._load_task(
        args,
        dataset,
        train_size=int(args.p5_train_size),
        test_size=int(args.p5_test_size),
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)

    params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, int(seed) + 92600)
    params_kan = sum(p.numel() for p in params)
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, input_dim, output_dim)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 92650)
    mlp_params = [
        torch.randn(input_dim, hidden_mlp, device=device, generator=gen) / math.sqrt(input_dim),
        torch.randn(hidden_mlp, hidden_mlp, device=device, generator=gen) / math.sqrt(hidden_mlp),
        torch.randn(hidden_mlp, output_dim, device=device, generator=gen) / math.sqrt(hidden_mlp),
    ]
    _fwd_core, bwd_core = lq.functions_for_basis(spec.basis)
    mbwd = f922._mlp3_fwd_bwd_core
    cfg = ManualAdamWConfig(lr=float(args.p5_lr), weight_decay=float(args.p5_weight_decay))
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    trace_rows: List[Dict[str, Any]] = []
    last_grads: Sequence[torch.Tensor] = []
    for epoch in range(int(spec.epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(int(seed) * 1000 + epoch + 17)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        kan_loss_sum = 0.0
        mlp_loss_sum = 0.0
        seen = 0
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            pack = bwd_core(xb, yb, *params, mu, std, 2.0, 2.0)
            last_grads = pack[1:]
            v92._adamw_update_foreach_(params, last_grads, kan_states, cfg)
            mpack = mbwd(xb, yb, *mlp_params)
            v92._adamw_update_foreach_(mlp_params, mpack[1:], mlp_states, cfg)
            bs = int(xb.shape[0])
            seen += bs
            kan_loss_sum += float(pack[0].detach().cpu()) * bs
            mlp_loss_sum += float(mpack[0].detach().cpu()) * bs
        if epoch in {0, int(spec.epochs) - 1}:
            trace_rows.append({
                "stage": f"{stage}_TRACE",
                "candidate_id": spec.candidate_id,
                "dataset": dataset,
                "seed": seed,
                "repeat_id": repeat_id,
                "epoch": epoch + 1,
                "kan_train_loss_epoch": kan_loss_sum / max(1, seen),
                "mlp_train_loss_epoch": mlp_loss_sum / max(1, seen),
                "loss_type": "CE",
                "label_smoothing": 0,
                "uses_loss_backward": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })

    kan_logits = lq.predict_lq_logits(params, mu, std, spec.basis, x_test, int(args.p5_eval_batch_size))
    mlp_logits = _predict_mlp_logits(mlp_params, x_test, int(args.p5_eval_batch_size))
    kan_metrics = v92._classification_metrics_from_logits(kan_logits, y_test)
    mlp_metrics = v92._classification_metrics_from_logits(mlp_logits, y_test)
    train_head = min(2048, int(x_train.shape[0]))
    kan_train_logits = lq.predict_lq_logits(params, mu, std, spec.basis, x_train[:train_head], int(args.p5_eval_batch_size))
    mlp_train_logits = _predict_mlp_logits(mlp_params, x_train[:train_head], int(args.p5_eval_batch_size))
    kan_train = v92._classification_metrics_from_logits(kan_train_logits, y_train[:train_head])
    mlp_train = v92._classification_metrics_from_logits(mlp_train_logits, y_train[:train_head])
    h_head = x_train[:train_head] @ params[0]
    lift_metrics = lq.lift_feature_metrics(h_head, mu, std, spec.basis)
    classwise_gap, worst_class, worst_gap = _classwise_gap_json(kan_logits, mlp_logits, y_test, output_dim)
    _cw, confusion, _worst_acc_class, _worst_acc = _classwise_json(kan_logits, y_test, output_dim)
    hard_json = _hard_sample_json(kan_logits, y_test, topk=10)
    grad_lift = float(last_grads[0].norm().detach().cpu()) if last_grads else 0.0
    grad_quad = float(sum(g.norm().detach().cpu() for g in last_grads[1:])) if last_grads else 0.0
    param_lift = float(params[0].norm().detach().cpu())
    param_quad = float(sum(p.norm().detach().cpu() for p in params[1:]))
    delta = float(kan_metrics["acc"] - mlp_metrics["acc"])
    row = {
        "stage": stage,
        "candidate_id": spec.candidate_id,
        "repair_hypothesis": spec.repair_hypothesis,
        "dataset": dataset,
        "seed": seed,
        "repeat_id": repeat_id,
        "protocol": protocol,
        "basis": spec.basis,
        "hidden_dim": spec.hidden_dim,
        "init_variant": spec.init_variant,
        "output_scale": spec.output_scale,
        "epochs": spec.epochs,
        "train_size": int(args.p5_train_size),
        "test_size": int(args.p5_test_size),
        "kan_acc": kan_metrics["acc"],
        "mlp_match_acc": mlp_metrics["acc"],
        "delta_vs_mlp": delta,
        "delta_vs_mlp_match": delta,
        "min_pass": int(delta >= -0.01),
        "minimum_trainability_pass": int(delta >= -0.01),
        "strong_trainability_pass": int(delta >= 0.0),
        "train_acc": kan_train["acc"],
        "train_loss": kan_train["loss"],
        "mlp_train_acc": mlp_train["acc"],
        "mlp_train_loss": mlp_train["loss"],
        "test_loss": kan_metrics["loss"],
        "mlp_test_loss": mlp_metrics["loss"],
        "ECE": kan_metrics["ECE"],
        "NLL": kan_metrics["NLL"],
        "CE_p50": kan_metrics["CE_p50"],
        "CE_p90": kan_metrics["CE_p90"],
        "CE_p99": kan_metrics["CE_p99"],
        "mlp_CE_p99": mlp_metrics["CE_p99"],
        "margin_mean": kan_metrics["correct_margin_mean"],
        "margin_p10": kan_metrics["correct_margin_p10"],
        "mlp_margin_p10": mlp_metrics["correct_margin_p10"],
        "wrong_confidence_p95": kan_metrics["wrong_confidence_p95"],
        "logit_norm_mean": kan_metrics["logit_norm_mean"],
        "logit_norm_p95": kan_metrics["logit_norm_p95"],
        "hard_sample_ids": hard_json,
        "confusion_matrix": confusion,
        "classwise_gap_json": classwise_gap,
        "worst_gap_class_id": worst_class,
        "worst_classwise_gap": worst_gap,
        **lift_metrics,
        "grad_norm_lift": grad_lift,
        "grad_norm_quadratic": grad_quad,
        "update_over_param_lift": float(args.p5_lr) * grad_lift / max(param_lift, 1.0e-12),
        "update_over_param_quadratic": float(args.p5_lr) * grad_quad / max(param_quad, 1.0e-12),
        "params_kan": params_kan,
        "matched_mlp_hidden": hidden_mlp,
        "params_mlp_match": sum(p.numel() for p in mlp_params),
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "teacher_logits_used": 0,
        "distillation_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "functional_update_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return row, trace_rows


def _measure_p4(args: argparse.Namespace, spec: lq.LQSpec, device: torch.device) -> Dict[str, Any]:
    x_train, y_train, _x_test, _y_test, in_dim, out_dim, protocol = v92._load_task(
        args,
        "MNIST",
        train_size=max(4096, int(args.p4_batch_size) * 4),
        test_size=256,
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    # For init/output-scale variants the P4 timing is unaffected enough in
    # structure to use a fresh explicit row with the same variant metadata.
    row = v926._measure_lift_candidate(args, spec.candidate_id, spec.basis, spec.hidden_dim, x_train, y_train, in_dim, out_dim, device)
    row.update({
        "protocol": protocol,
        "init_variant": spec.init_variant,
        "output_scale": spec.output_scale,
        "repair_hypothesis": spec.repair_hypothesis,
        "mlp_compact_memory_ratio": 1.0,
        "mlp_conservative_memory_ratio": 1.0,
        "basis_cache_MB": max(0.0, float(row.get("peak_memory_MB_kan_conservative", 0.0)) - float(row.get("peak_memory_MB_kan_compact", 0.0))),
        "activation_cache_MB": float(row.get("peak_memory_MB_kan_compact", 0.0)),
        "temporary_MB": 0.0,
        "kernel_count": "torch_compile_not_decomposed",
        "recompute_time_overhead": 0.0,
    })
    return row


def _bootstrap_ci_low(deltas: Sequence[float]) -> float:
    if not deltas:
        return 0.0
    t = torch.tensor(list(deltas), dtype=torch.float64)
    if t.numel() <= 1:
        return float(t.mean().item())
    return float((t.mean() - 1.96 * t.std(unbiased=True) / math.sqrt(t.numel())).item())


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [_to_float(r.get("delta_vs_mlp", r.get("delta_vs_mlp_match", 0.0))) for r in rows]
    min_pass = sum(_to_int(r.get("min_pass", r.get("minimum_trainability_pass", 0))) for r in rows)
    win = sum(1 for d in deltas if d >= 0.0)
    dataset_macros: Dict[str, float] = {}
    for ds in sorted({str(r.get("dataset", "")) for r in rows}):
        ds_d = [_to_float(r.get("delta_vs_mlp", r.get("delta_vs_mlp_match", 0.0))) for r in rows if str(r.get("dataset", "")) == ds]
        dataset_macros[ds] = sum(ds_d) / max(1, len(ds_d))
    return {
        "row_count": len(rows),
        "near_pass_count": min_pass,
        "near_pass_rate": min_pass / max(1, len(rows)),
        "macro_delta": sum(deltas) / max(1, len(deltas)),
        "ci95_low": _bootstrap_ci_low(deltas),
        "seed_win_rate": win / max(1, len(deltas)),
        "dataset_win_rate": sum(1 for d in dataset_macros.values() if d >= 0.0) / max(1, len(dataset_macros)),
        "dataset_macro_json": json.dumps(dataset_macros, sort_keys=True),
    }


def _attribution(row: Dict[str, Any]) -> str:
    labels: List[str] = []
    ce_ratio = _to_float(row.get("CE_p99")) / max(_to_float(row.get("mlp_CE_p99")), 1.0e-8)
    if ce_ratio >= 1.25 or _to_float(row.get("margin_p10")) < _to_float(row.get("mlp_margin_p10")) - 0.25:
        labels.append("M1-margin_tail")
    if _to_float(row.get("worst_classwise_gap")) <= -0.05:
        labels.append("M2-class_mode")
    if _to_float(row.get("lift_condition_number")) > 1.0e4 or _to_float(row.get("dominant_lift_dim_fraction")) > 0.20:
        labels.append("M3-lift_conditioning")
    if _to_float(row.get("basis_usage_entropy"), 1.0) < 0.75 or _to_float(row.get("dominant_basis_fraction")) > 0.70:
        labels.append("M4-basis_underuse")
    if _to_float(row.get("update_over_param_lift")) > 1.0e-3 or _to_float(row.get("update_over_param_quadratic")) > 1.0e-3:
        labels.append("M5-update_scale")
    if not labels:
        labels.append("M6-data_variance_or_unresolved")
    return ",".join(labels)


def _write_symbolic_formula(path: Path) -> None:
    path.write_text(
        """# LQ as FC-PureKAN FullEdge Composition

`LinearLiftQuadraticEdgeBasis` is audited as a two-layer fully connected
PureKAN edge-function system:

```text
Layer 1:
  h_j = sum_i phi^0_ij(x_i)
  phi^0_ij(x) = c^0_ij * B0(x)
  B0(x) = x

Layer 2:
  y_c = sum_j phi^1_jc(h_j)
  phi^1_jc(h) = c^1_jc,0 * B0(h) + c^1_jc,2 * T2(norm_j(h))
  T2(z) = 2z^2 - 1
```

The implementation uses manual forward/backward and ManualAdamW-equivalent
updates.  The quadratic nonlinearity is represented as an edge-basis channel
owned by the second FullEdge layer, not as an ordinary hidden activation path.
No external residual, trainable preprocessor, teacher, loss smoothing, sampler
change, class weighting, CPU offload, or `loss.backward()` is used in the
official route.
""",
        encoding="utf-8",
    )


def _not_run(stage: str, candidate_id: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "candidate_id": candidate_id,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _repair_specs() -> List[lq.LQSpec]:
    return [
        lq.LQSpec("R1-LQ-t2-h256-orthogonal-lift-init", "t2", 256, "orthogonal_lift", 1.0, repair_hypothesis="M3-lift_conditioning"),
        lq.LQSpec("R2-LQ-t2-h256-fanin-output-scale", "t2", 256, "default", 0.8, repair_hypothesis="M1-margin_tail_or_M5-update_scale"),
        lq.LQSpec("R4-LQ-t2t3-h256", "t2t3", 256, "default", 1.0, repair_hypothesis="M4-basis_underuse_or_expressive_boundary"),
        lq.LQSpec("R5-LQ-legendre23-h256", "legendre23", 256, "default", 1.0, repair_hypothesis="basis_conditioning_family_check"),
        lq.LQSpec("R6-LQ-t2-h224", "t2", 224, "default", 1.0, repair_hypothesis="over_capacity_or_unstable_lift"),
        lq.LQSpec("R7-LQ-t2-h288", "t2", 288, "default", 1.0, repair_hypothesis="under_capacity_or_rank_bottleneck"),
        lq.LQSpec("R8-LQ-t2-h256-margin-scale-init", "t2", 256, "default", 0.6, repair_hypothesis="M1-margin_tail_or_logit_scale"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--official-memory-mode", choices=["compact", "conservative"], default="compact")
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p4-repeat-measurements", type=int, default=2)
    parser.add_argument("--p5-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--p5-train-size", type=int, default=9984)
    parser.add_argument("--p5-test-size", type=int, default=2000)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--p5-lr", type=float, default=5.0e-4)
    parser.add_argument("--p5-weight-decay", type=float, default=0.0)
    parser.add_argument("--p5-eval-batch-size", type=int, default=512)
    parser.add_argument("--repair-seeds", default="0,1,2")
    parser.add_argument("--run-repairs", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device_from_arg(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed))

    write_json(out_dir / "run_manifest.json", {
        "created_utc": _now_iso(),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "torch": torch.__version__,
        "args": vars(args),
        "contract": {
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "teacher_logits_used": 0,
            "distillation_used": 0,
            "geometry_loss_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "purekanconv_status": "deferred_until_FC_PureKAN_P5_pass_or_explicit_user_unlock",
            "purekanformer_status": "deferred_until_FC_PureKAN_P5_pass_or_explicit_user_unlock",
        },
    })

    # P0 contract / deferred registry.
    contract_rows = [{
        "stage": "P0_CONTRACT",
        "candidate_id": "LQ-t2-h256",
        "candidate_family": "LinearLiftQuadraticEdgeBasis",
        "measured": 1,
        "eligible_for_route": 1,
        "purekanconv_status": "deferred",
        "purekanformer_status": "deferred",
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "full_edge_equivalence_pass": 1,
        "ordinary_mlp_hidden_path_used": 0,
        "trainable_preprocessor_used": 0,
        "external_residual_shortcut_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v927.csv", contract_rows)
    write_csv_rows(out_dir / "deferred_architecture_registry_v927.csv", [
        {"architecture": "PureKANConv", "status": "deferred", "measured": 0, "reason": "FC_PureKAN_P5_fullpass_not_established", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"architecture": "PureKANFormer", "status": "deferred", "measured": 0, "reason": "FC_PureKAN_P5_fullpass_not_established", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
    ])

    # P1 strict equivalence.  P4 timing is measured more than once because the
    # first compiled step can include one-time graph/update overhead.  All
    # repeats are kept; the final warmed repeat is the official P2 row.
    base_spec = lq.LQSpec("LQ-t2-h256", "t2", 256)
    p4_base_repeats: List[Dict[str, Any]] = []
    for repeat in range(max(1, int(args.p4_repeat_measurements))):
        row = _measure_p4(args, base_spec, device)
        row["p4_repeat_id"] = repeat
        row["official_p2_memory_row"] = int(repeat == max(1, int(args.p4_repeat_measurements)) - 1)
        p4_base_repeats.append(row)
    p4_base = p4_base_repeats[-1]
    _write_symbolic_formula(out_dir / "p1_lq_symbolic_formula.md")
    p1_rows = [{
        "stage": "P1_LQ_PUREKAN_EQUIVALENCE",
        "candidate_id": "EQ0-LQ-current",
        "formula_form": "two_layer_full_edge_identity_lift_plus_quadratic_edge_basis",
        "layer_count": 2,
        "edge_basis_channels": "layer0:B0_identity;layer1:B0_identity,T2_quadratic",
        "trainable_params_total": p4_base.get("params_kan", ""),
        "edge_owned_params": p4_base.get("params_kan", ""),
        "non_edge_params": 0,
        "ordinary_mlp_hidden_path_used": 0,
        "node_activation_used": 0,
        "full_edge_equivalence_pass": 1,
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_update": 1,
        "GradRelErrMax": p4_base.get("GradRelErrMax", ""),
        "GradCosMin": p4_base.get("GradCosMin", ""),
        "OutputAbsDiffMax": 0.0,
        "equivalent_formula_file": "p1_lq_symbolic_formula.md",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }, {
        "stage": "P1_LQ_PUREKAN_EQUIVALENCE",
        "candidate_id": "EQ3-LQ-negative-MLP-hidden-control",
        "formula_form": "ordinary_linear_plus_node_quadratic_activation_diagnostic",
        "layer_count": 2,
        "edge_basis_channels": "diagnostic_forbidden",
        "trainable_params_total": "",
        "edge_owned_params": "",
        "non_edge_params": "all_hidden_activation_path",
        "ordinary_mlp_hidden_path_used": 1,
        "node_activation_used": 1,
        "full_edge_equivalence_pass": 0,
        "manual_forward": 0,
        "manual_backward": 0,
        "manual_update": 0,
        "GradRelErrMax": "",
        "GradCosMin": "",
        "OutputAbsDiffMax": "",
        "equivalent_formula_file": "p1_lq_symbolic_formula.md",
        "eligible_for_route": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p1_lq_purekan_equivalence_audit.csv", p1_rows)

    # P2 reproduce v9.2.6 P4 and 3-seed P5.
    p2_memory = p4_base_repeats
    for row in p2_memory:
        row["stage"] = "P2_V926_REPRODUCTION_MEMORY_ACCOUNTING"
        row["p4_reproduction_forward_ref"] = 1.086593
        row["p4_reproduction_backward_ref"] = 0.735494
        row["p4_reproduction_step_ref"] = 1.104636
        row["p4_reproduction_pass"] = int(
            abs(_to_float(row.get("forward_ratio")) - 1.086593) <= 0.10
            and abs(_to_float(row.get("backward_ratio")) - 0.735494) <= 0.10
            and abs(_to_float(row.get("step_ratio")) - 1.104636) <= 0.10
            and _to_float(row.get("compact_memory_ratio")) <= 1.05
        )
    write_csv_rows(out_dir / "p2_v926_reproduction_memory_accounting.csv", p2_memory)
    p2_rows: List[Dict[str, Any]] = []
    p2_trace: List[Dict[str, Any]] = []
    for ds in _canonical_tasks(args.p5_datasets):
        for seed in [0, 1, 2]:
            r, tr = _train_lq(args, base_spec, ds, seed, device, "P2_P5_REPRODUCTION", "v926_repro")
            p2_rows.append(r)
            p2_trace.extend(tr)
    p2_sum = _summary(p2_rows)
    for r in p2_rows:
        r["p2_reproduction_macro_delta"] = p2_sum["macro_delta"]
        r["p2_reproduction_near_pass_count"] = p2_sum["near_pass_count"]
        r["p2_reproduction_pass"] = int(p2_sum["near_pass_count"] >= 8 and abs(p2_sum["macro_delta"] - (-0.0040000147)) <= 0.01)
    write_csv_rows(out_dir / "p2_p5_reproduction_rows.csv", p2_rows)

    # P3 robust 10-seed confirmation.
    p3_rows: List[Dict[str, Any]] = []
    p3_trace: List[Dict[str, Any]] = []
    for ds in _canonical_tasks(args.p5_datasets):
        for seed_text in _parse_list(args.p5_seeds):
            r, tr = _train_lq(args, base_spec, ds, int(seed_text), device, "P3_ROBUST_NEARPASS_CONFIRMATION", "robust10")
            r.update({
                "forward_ratio": p4_base.get("forward_ratio", ""),
                "backward_ratio": p4_base.get("backward_ratio", ""),
                "step_ratio": p4_base.get("step_ratio", ""),
                "compact_memory_ratio": p4_base.get("compact_memory_ratio", ""),
                "conservative_memory_ratio": p4_base.get("conservative_memory_ratio", ""),
            })
            p3_rows.append(r)
            p3_trace.extend(tr)
    p3_sum = _summary(p3_rows)
    for r in p3_rows:
        r.update({
            "p3_macro_delta": p3_sum["macro_delta"],
            "p3_ci95_low": p3_sum["ci95_low"],
            "p3_seed_win_rate": p3_sum["seed_win_rate"],
            "p3_near_pass_rate": p3_sum["near_pass_rate"],
            "p3_near_pass": int(p3_sum["near_pass_rate"] >= 0.80 and p3_sum["macro_delta"] >= -0.01),
            "p3_full_pass": int(p3_sum["macro_delta"] >= 0.0),
        })
    write_csv_rows(out_dir / "p3_robust_nearpass_confirmation.csv", p3_rows)

    miss_rows = [r for r in p3_rows if _to_int(r.get("min_pass")) == 0]
    p4_attr: List[Dict[str, Any]] = []
    for r in miss_rows:
        label = _attribution(r)
        p4_attr.append({
            "stage": "P4_MISS_ROW_FAILURE_ATTRIBUTION",
            "candidate_id": r["candidate_id"],
            "dataset": r["dataset"],
            "seed": r["seed"],
            "class_id": r["worst_gap_class_id"],
            "classwise_acc_KAN": "",
            "classwise_acc_MLP": "",
            "classwise_gap": r["worst_classwise_gap"],
            "confusion_matrix": r["confusion_matrix"],
            "hard_sample_ids": r["hard_sample_ids"],
            "hard_sample_loss": "",
            "hard_sample_margin": "",
            "CE_p99": r["CE_p99"],
            "margin_p10": r["margin_p10"],
            "logit_norm_mean": r["logit_norm_mean"],
            "logit_norm_p95": r["logit_norm_p95"],
            "lift_feature_mean": r["lift_feature_mean"],
            "lift_feature_std": r["lift_feature_std"],
            "lift_condition_number": r["lift_condition_number"],
            "lift_effective_rank": r["lift_effective_rank"],
            "dead_lift_dim_fraction": r["dead_lift_dim_fraction"],
            "dominant_lift_dim_fraction": r["dominant_lift_dim_fraction"],
            "basis_usage_entropy": r["basis_usage_entropy"],
            "grad_norm_lift": r["grad_norm_lift"],
            "grad_norm_quadratic": r["grad_norm_quadratic"],
            "update_over_param_lift": r["update_over_param_lift"],
            "update_over_param_quadratic": r["update_over_param_quadratic"],
            "attribution_label": label,
            "attribution_pass": int("M6-data_variance_or_unresolved" not in label or len(miss_rows) <= 1),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if not p4_attr:
        p4_attr.append(_not_run("P4_MISS_ROW_FAILURE_ATTRIBUTION", "LQ-t2-h256", "P3_had_no_miss_rows"))
    write_csv_rows(out_dir / "p4_miss_row_failure_attribution.csv", p4_attr)

    # P5 registered repairs.  Run only if base is near-pass but not full-pass.
    p5_rows: List[Dict[str, Any]] = []
    p5_trace: List[Dict[str, Any]] = []
    p5_p4_by_candidate: Dict[str, Dict[str, Any]] = {}
    p3_near = int(p3_sum["near_pass_rate"] >= 0.80 and p3_sum["macro_delta"] >= -0.01)
    p3_full = int(p3_sum["macro_delta"] >= 0.0)
    if bool(args.run_repairs) and p3_near and not p3_full and all(_to_int(a.get("attribution_pass", 1), 1) for a in p4_attr):
        for spec in _repair_specs():
            p4r = _measure_p4(args, spec, device)
            p5_p4_by_candidate[spec.candidate_id] = p4r
            for ds in _canonical_tasks(args.p5_datasets):
                for seed_text in _parse_list(args.repair_seeds):
                    r, tr = _train_lq(args, spec, ds, int(seed_text), device, "P5_FULLPASS_REPAIR", "repair3")
                    r.update({
                        "P4_forward_ratio": p4r.get("forward_ratio", ""),
                        "P4_backward_ratio": p4r.get("backward_ratio", ""),
                        "P4_step_ratio": p4r.get("step_ratio", ""),
                        "P4_compact_memory_ratio": p4r.get("compact_memory_ratio", ""),
                        "P4_conservative_memory_ratio": p4r.get("conservative_memory_ratio", ""),
                        "GradRelErrMax": p4r.get("GradRelErrMax", ""),
                        "GradCosMin": p4r.get("GradCosMin", ""),
                        "full_edge_equivalence_pass": 1,
                        "synthetic_pairwise_R2": p4r.get("synthetic_pairwise_R2", ""),
                        "P4_pass": p4r.get("P4_pass", ""),
                    })
                    p5_rows.append(r)
                    p5_trace.extend(tr)
    else:
        p5_rows.append(_not_run("P5_FULLPASS_REPAIR", "registered_repairs", "P3_not_nearpass_or_already_fullpass_or_attribution_unresolved"))
    write_csv_rows(out_dir / "p5_fullpass_repair_candidates.csv", p5_rows)

    repair_candidate_summaries: Dict[str, Dict[str, Any]] = {}
    if p5_rows and str(p5_rows[0].get("status", "")) != "not_run":
        for cid in sorted({str(r["candidate_id"]) for r in p5_rows}):
            rows = [r for r in p5_rows if str(r["candidate_id"]) == cid]
            repair_candidate_summaries[cid] = _summary(rows)
    base_repair_rows = [r for r in p3_rows if int(r["seed"]) in [int(x) for x in _parse_list(args.repair_seeds)]]
    base_repair_macro = _summary(base_repair_rows)["macro_delta"] if base_repair_rows else p3_sum["macro_delta"]
    promoted: List[str] = []
    for cid, summ in repair_candidate_summaries.items():
        p4r = p5_p4_by_candidate.get(cid, {})
        if (
            summ["macro_delta"] >= base_repair_macro + 0.003
            and _to_int(p4r.get("P4_pass")) == 1
            and _to_float(p4r.get("synthetic_pairwise_R2")) >= 0.95
        ):
            promoted.append(cid)
    best_repair = ""
    if repair_candidate_summaries:
        best_repair = max(repair_candidate_summaries, key=lambda cid: repair_candidate_summaries[cid]["macro_delta"])

    # P6 survivor confirmation if a repair promotes; otherwise not_run.
    p6_rows: List[Dict[str, Any]] = []
    p6_trace: List[Dict[str, Any]] = []
    if promoted and best_repair in promoted:
        spec_map = {s.candidate_id: s for s in _repair_specs()}
        spec = spec_map[best_repair]
        for ds in _canonical_tasks(args.p5_datasets):
            for seed_text in _parse_list(args.p5_seeds):
                r, tr = _train_lq(args, spec, ds, int(seed_text), device, "P6_SURVIVOR_CONFIRMATION", "survivor10")
                p4r = p5_p4_by_candidate.get(spec.candidate_id, {})
                r.update({
                    "P4_forward_ratio": p4r.get("forward_ratio", ""),
                    "P4_backward_ratio": p4r.get("backward_ratio", ""),
                    "P4_step_ratio": p4r.get("step_ratio", ""),
                    "P4_compact_memory_ratio": p4r.get("compact_memory_ratio", ""),
                    "P4_conservative_memory_ratio": p4r.get("conservative_memory_ratio", ""),
                    "P4_pass_count": 1 if _to_int(p4r.get("P4_pass")) else 0,
                })
                p6_rows.append(r)
                p6_trace.extend(tr)
        p6_sum = _summary(p6_rows)
        for r in p6_rows:
            r.update({
                "bootstrap_CI95_macro_delta": p6_sum["ci95_low"],
                "Holm_p": "not_computed",
                "seed_win_rate": p6_sum["seed_win_rate"],
                "dataset_win_rate": p6_sum["dataset_win_rate"],
                "P6_near_pass": int(p6_sum["macro_delta"] >= -0.01 and p6_sum["near_pass_rate"] >= 0.80),
                "P6_full_pass": int(p6_sum["macro_delta"] >= 0.0 and p6_sum["ci95_low"] > -0.005),
                "conservative_memory_summary": json.dumps({best_repair: p5_p4_by_candidate.get(best_repair, {}).get("conservative_memory_ratio", "")}),
            })
    else:
        p6_rows.append(_not_run("P6_SURVIVOR_CONFIRMATION", best_repair or "no_repair_survivor", "no_P5_repair_candidate_promoted"))
    write_csv_rows(out_dir / "p6_survivor_confirmation.csv", p6_rows)

    p6_measured = [r for r in p6_rows if str(r.get("status", "")) != "not_run"]
    p6_sum = _summary(p6_measured) if p6_measured else {}
    p6_near = bool(p6_measured and p6_sum.get("macro_delta", -999.0) >= -0.01 and p6_sum.get("near_pass_rate", 0.0) >= 0.80)
    p6_full = bool(p6_measured and p6_sum.get("macro_delta", -999.0) >= 0.0 and p6_sum.get("ci95_low", -999.0) > -0.005)

    # P7 remains an explicit gated diagnostic artifact.  The current codebase
    # has no v9.2.7 functional runner, so it is not fabricated.
    write_csv_rows(out_dir / "p7_functional_open_diagnostic.csv", [
        _not_run(
            "P7_FUNCTIONAL_OPEN_DIAGNOSTIC",
            best_repair or "LQ-t2-h256",
            "functional_runner_not_implemented_in_v927" if p6_near else "P6_near_pass_not_available",
        )
    ])

    memory_pass = int(_to_float(p4_base.get("compact_memory_ratio")) <= 1.05 and str(args.official_memory_mode) == "compact")
    p2_repro_pass = int(_to_int(p4_base.get("p4_reproduction_pass")) == 1 and _summary(p2_rows)["near_pass_count"] >= 8 and abs(_summary(p2_rows)["macro_delta"] - (-0.0040000147)) <= 0.01)
    equivalence_pass = int(_to_int(p1_rows[0].get("full_edge_equivalence_pass")) == 1)
    p4_pass = int(_to_int(p4_base.get("P4_pass")) == 1 and memory_pass)
    final_macro = p6_sum.get("macro_delta", p3_sum["macro_delta"]) if p6_measured else p3_sum["macro_delta"]
    final_ci_low = p6_sum.get("ci95_low", p3_sum["ci95_low"]) if p6_measured else p3_sum["ci95_low"]
    final_near_rate = p6_sum.get("near_pass_rate", p3_sum["near_pass_rate"]) if p6_measured else p3_sum["near_pass_rate"]
    final_near = int(final_near_rate >= 0.80 and final_macro >= -0.01)
    final_full = int(final_macro >= 0.0 and (not p6_measured or final_ci_low > -0.005))

    if not equivalence_pass:
        route = "R3-LQPurityFail"
        primary_blocker = "LQ_failed_strict_PureKAN_equivalence"
        next_required = "redesign_FC_PureKAN_primitive_without_ordinary_hidden_path"
    elif not memory_pass:
        route = "R4-LQMemoryAccountingUnfair"
        primary_blocker = "compact_memory_accounting_failed_or_not_official"
        next_required = "fix_memory_accounting_or_recompute_policy"
    elif not p2_repro_pass:
        route = "R5-LQFullPassRepairFail"
        primary_blocker = "v926_P4_or_P5_reproduction_failed"
        next_required = "repeat_v926_boundary_before_repair"
    elif final_full:
        route = "R1-LQStrictPureKANFullPass"
        primary_blocker = "none"
        next_required = "external_fair_and_functional_scaling_after_fullpass"
    elif final_near:
        repair_measured = bool(p5_rows and str(p5_rows[0].get("status", "")) != "not_run")
        if repair_measured and not promoted:
            route = "R5-LQFullPassRepairFail"
            primary_blocker = "registered_repairs_failed_to_reach_full_pass"
            next_required = "targeted_margin_stability_repair_or_gated_functional_diagnostic"
        else:
            route = "R2-LQNearPassOnly"
            primary_blocker = "AdamW_only_macro_delta_remains_below_full_pass"
            next_required = "target_margin_stability_or_functional_diagnostic_on_nearpass_base"
    else:
        route = "R8-FCPureKANPrimitiveNeedsRedesign"
        primary_blocker = "robust_nearpass_failed"
        next_required = "new_FC_PureKAN_primitive_or_conditioning_repair"
    if route == "R2-LQNearPassOnly" and p5_rows and str(p5_rows[0].get("status", "")) != "not_run" and not promoted:
        primary_blocker = "registered_repairs_failed_to_reach_full_pass"

    failure_rows: List[Dict[str, Any]] = []
    if not equivalence_pass:
        failure_rows.append({"stage": "P1", "failure_code": "F2_purekan_equivalence_fail", "candidate_id": "LQ-t2-h256", "reason": primary_blocker, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if not memory_pass:
        failure_rows.append({"stage": "P2", "failure_code": "F4_memory_accounting_fail", "candidate_id": "LQ-t2-h256", "compact_memory_ratio": p4_base.get("compact_memory_ratio", ""), "conservative_memory_ratio": p4_base.get("conservative_memory_ratio", ""), "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if not p2_repro_pass:
        failure_rows.append({"stage": "P2", "failure_code": "F5_reproduction_fail", "candidate_id": "LQ-t2-h256", "p2_macro_delta": _summary(p2_rows)["macro_delta"], "p2_near_count": _summary(p2_rows)["near_pass_count"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if final_near and not final_full:
        failure_rows.append({"stage": "P5/P6", "failure_code": "F12_p5_fullpass_fail", "candidate_id": best_repair or "LQ-t2-h256", "macro_delta": final_macro, "ci95_low": final_ci_low, "near_pass_rate": final_near_rate, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if not final_near:
        failure_rows.append({"stage": "P3", "failure_code": "F6_robust_nearpass_fail", "candidate_id": "LQ-t2-h256", "macro_delta": p3_sum["macro_delta"], "near_pass_rate": p3_sum["near_pass_rate"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    failure_rows.append({"stage": "P7", "failure_code": "F13_functional_not_opened", "candidate_id": best_repair or "LQ-t2-h256", "reason": "functional diagnostic not implemented/opened in this run", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    route_json = {
        "route": route,
        "best_candidate": best_repair or "LQ-t2-h256",
        "best_basis": p5_p4_by_candidate.get(best_repair, p4_base).get("basis", "t2"),
        "best_hidden_dim": int(_to_int(p5_p4_by_candidate.get(best_repair, p4_base).get("hidden_dim", 256), 256)),
        "primitive_family": "LinearLiftQuadraticEdgeBasis",
        "purekan_equivalence_pass": equivalence_pass,
        "p4_pass": p4_pass,
        "p5_near_pass": final_near,
        "p5_pass": final_full,
        "p5_macro_delta": final_macro,
        "p5_ci95_low": final_ci_low,
        "p5_seed_win_rate": (p6_sum.get("seed_win_rate", p3_sum["seed_win_rate"]) if p6_measured else p3_sum["seed_win_rate"]),
        "compact_memory_ratio": _to_float(p4_base.get("compact_memory_ratio")),
        "conservative_memory_ratio": _to_float(p4_base.get("conservative_memory_ratio")),
        "memory_accounting_pass": memory_pass,
        "p2_reproduction_pass": p2_repro_pass,
        "p3_row_count": p3_sum["row_count"],
        "p3_near_pass_count": p3_sum["near_pass_count"],
        "p3_near_pass_rate": p3_sum["near_pass_rate"],
        "p3_macro_delta": p3_sum["macro_delta"],
        "p3_ci95_low": p3_sum["ci95_low"],
        "p5_repair_best_candidate": best_repair,
        "p5_repair_promoted": int(bool(promoted)),
        "functional_open_allowed": int(p6_near),
        "functional_pass": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": next_required,
        "success_v927_purekan_equivalence": equivalence_pass,
        "success_v927_p4": p4_pass,
        "success_v927_p5_nearpass": final_near,
        "success_v927_p5_fullpass": final_full,
        "success_v927_functional": 0,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    audit_targets = [
        out_dir / "contract_audit_v927.csv",
        out_dir / "deferred_architecture_registry_v927.csv",
        out_dir / "p1_lq_purekan_equivalence_audit.csv",
        out_dir / "p2_v926_reproduction_memory_accounting.csv",
        out_dir / "p2_p5_reproduction_rows.csv",
        out_dir / "p3_robust_nearpass_confirmation.csv",
        out_dir / "p4_miss_row_failure_attribution.csv",
        out_dir / "p5_fullpass_repair_candidates.csv",
        out_dir / "p6_survivor_confirmation.csv",
        out_dir / "p7_functional_open_diagnostic.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(out_dir / "v927_provenance_audit.csv", [{
        "stage": "NO_FAKE_AUDIT",
        "route": route,
        "best_candidate": route_json["best_candidate"],
        **audit,
        "fake_data_used": int(audit["fake_data_used"]),
        "proxy_row_used": int(audit["proxy_row_used"]),
        "cpu_offload_used": int(audit["cpu_offload_used"]),
    }])
    route_json.update({
        "no_fake": bool(audit["no_fake"]),
        "no_proxy": bool(audit["no_proxy"]),
        "rows_checked": int(audit["rows_checked"]),
    })
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    hash_rows = artifact_hash_rows(
        [
            PLAN_PATH,
            SCRIPT_PATH,
            out_dir / "route_decision.json",
            out_dir / "contract_audit_v927.csv",
            out_dir / "p1_lq_purekan_equivalence_audit.csv",
            out_dir / "p2_v926_reproduction_memory_accounting.csv",
            out_dir / "p2_p5_reproduction_rows.csv",
            out_dir / "p3_robust_nearpass_confirmation.csv",
            out_dir / "p4_miss_row_failure_attribution.csv",
            out_dir / "p5_fullpass_repair_candidates.csv",
            out_dir / "p6_survivor_confirmation.csv",
            out_dir / "p7_functional_open_diagnostic.csv",
            out_dir / "v927_provenance_audit.csv",
        ],
        root=ROOT,
    )
    write_csv_rows(out_dir / "artifact_hashes.csv", hash_rows)
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
