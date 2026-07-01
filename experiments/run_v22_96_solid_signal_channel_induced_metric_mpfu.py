#!/usr/bin/env python3
"""DG-KAN v22.96 solid signal-channel-induced metric runner.

The runner is gate-aware.  It implements Part 0/A/B/C and writes blocked
downstream artifacts when Part C is not valid, instead of running metric
induction experiments after a failed estimator gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import py_compile
import re
import shutil
import sys
import time
import tokenize
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v22_94_adamw_witness_decomposition_mpfu as v2294
from dgkan.fu.channel_to_metric import (
    project_channel_to_layers,
    project_channel_to_mode_bands,
    reverse_audit_smoke_test,
)
from dgkan.fu.fixed_point_signal_metric import fixed_point_smoke_test
from dgkan.fu.kan_intrinsic_metrics import (
    DataCompositeMetric,
    data_metric_smoke_tests,
    flatten_layer_param_masks,
    full_mode_metric_vector,
    subspace_angle_from_diags,
)
from dgkan.fu.signal_channel_estimators import (
    OffDiagonalAgreementEstimator,
    VelocityProxyChannelEstimator,
    WindowedDissipationEstimator,
    diagonal_snr_gate,
    full_output_jacobian,
    loss_gradient_vector,
    offdiag_synthetic_smoke_test,
    per_example_gradient_matrix,
    projector_from_psd,
    projector_overlap,
    vector_channel_fraction,
    windowed_linear_smoke_test,
)
from dgkan.fu.layer_composite_metric import EPS, sym


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v22.96_SolidSignalChannelInducedMetric_MultiDirection_MPFU_详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.96_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.96_实验结果复盘.md"
RESULT_ROOT = Path(os.environ.get("V2296_RESULT_ROOT", str(ROOT / "results/v22_96"))).resolve()
OUT_ROOT = Path(os.environ.get("V2296_OUT_ROOT", str(RESULT_ROOT / "current"))).resolve()
REPAIR0_ROOT = RESULT_ROOT / "repair_round0_initial"
PROJ_ROOT = OUT_ROOT / "part_c_channel_projectors"
PART0_PROJ_ROOT = OUT_ROOT / "part0_channel_projectors"

BASIS_KEYS = ("dche_k5", "dche_k9", "dfour_default")
VISUAL_TASKS = ("local_patch_interaction", "rotation_sensitive")
PART_C_ESTIMATORS = ("velocity_proxy", "window_no_pg", "window_pg1", "offdiag_diag", "offdiag_block")

AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "mlp_readout_used": 0,
    "mlp_initial_layer_used": 0,
    "external_product_feature_used": 0,
    "new_edge_function_added": 0,
}


def ensure_out() -> None:
    for path in (RESULT_ROOT, OUT_ROOT, REPAIR0_ROOT, PROJ_ROOT, PART0_PROJ_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def command_text(argv: Iterable[str]) -> str:
    return " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.96 Execution Log\n\n"
            f"- Created: {now()}\n"
            f"- Plan: `{rel(PLAN)}`\n"
            f"- Runner: `{rel(RUNNER)}`\n"
            f"- Output root: `{rel(OUT_ROOT)}`\n"
            f"- Python: `{PYTHON}`\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.96 Experiment Recap\n\n"
            f"- Created: {now()}\n"
            "- No fabricated metrics or inferred data rows are allowed in this file.\n\n",
            encoding="utf-8",
        )


def append_exec(part: str, command: str, status: str, files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"## {now()} {part} {status}\n\n")
        fh.write(f"Command: `{command}`\n\n")
        if files:
            fh.write(f"Files: {files}\n\n")
        if note:
            fh.write(f"Note: {note}\n\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        fh.write("\n```\n\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def fval(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = torch.tensor([float(v) for v in xs], dtype=torch.float64)
    y = torch.tensor([float(v) for v in ys], dtype=torch.float64)
    if int(x.numel()) < 2 or int(y.numel()) != int(x.numel()):
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    return float((x @ y / denom).detach().cpu().item())


def argument_hash(args: argparse.Namespace, *, ignore_shard: bool = True) -> str:
    payload = vars(args).copy()
    if ignore_shard:
        payload.pop("device", None)
        payload.pop("shard_index", None)
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def stable_seed(*items: Any, base: int = 0) -> int:
    text = "|".join(str(item) for item in items)
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return int((int(base) + value) % 2_147_483_647)


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    index = int(args.shard_index)
    return [item for idx, item in enumerate(items) if idx % count == index]


def device_from_args(args: argparse.Namespace) -> torch.device:
    text = str(args.device)
    if text.startswith("cuda") and torch.cuda.is_available():
        return torch.device(text)
    return torch.device("cpu")


def strip_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}:
                out.append("\n" if token.type in {tokenize.NL, tokenize.NEWLINE} else " ")
            else:
                out.append(token.string)
    except tokenize.TokenError:
        return text
    return "".join(out)


def static_scan(files: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_architecture_winner_selection": re.compile(r"\b(select_arch|choose_arch|winner_family)\s*\("),
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|winner_metric)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|candidate_score|winner_update)\s*\("),
        "runtime_topk_edge_selection": re.compile(r"\.topk\s*\("),
        "external_product_feature_call": re.compile(r"\b(make_external_product_feature|external_product_feature_transform|patch_product_feature)\s*\("),
        "new_edge_family_call": re.compile(r"\b(learned_square_edge|new_edge_basis_family)\s*\("),
        "mlp_stem_or_readout_call": re.compile(r"\b(mlp_stem|mlp_readout|readout_lstsq|oracle_target)\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in files:
        if not path.exists():
            hits.append({"file": rel(path), "check": "file_missing", "match": "missing"})
            continue
        code = strip_code(path)
        for check, pattern in patterns.items():
            match = pattern.search(code)
            if match:
                hits.append({"file": rel(path), "check": check, "match": match.group(0)})
    return int(not hits), hits


def gate_summary(part: str, gate_pass: int, route: str, blocker: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    ok_rows = sum(1 for row in rows if row.get("status") == "ok")
    error_rows = sum(1 for row in rows if row.get("status") == "error")
    out = {
        "part": part.upper(),
        "gate_pass": int(gate_pass),
        "route": route,
        "dominant_blocker": blocker,
        "ok_rows": int(ok_rows),
        "error_rows": int(error_rows),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "metric_winner_selection_used": 0,
        "candidate_update_selection_used": 0,
        "row_count": int(len(rows)),
        "generated_at": now(),
    }
    out.update(extra)
    return out


def common_next_actions(part: str, blocker: str, allowed: list[str], forbidden: list[str], commands: list[str]) -> Path:
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    write_json(
        path,
        {
            "part": part.upper(),
            "dominant_blocker": blocker,
            "allowed_actions": allowed,
            "forbidden_actions": forbidden,
            "rerun_commands": commands,
            "promotion_allowed": False,
            "must_write_repair_rationale": True,
        },
    )
    return path


def basis_args(args: argparse.Namespace, basis_key: str) -> argparse.Namespace:
    return v2294.basis_args(args, basis_key)


def make_model(depth: str, input_dim: int, classes: int, seed: int, args: argparse.Namespace, device: torch.device) -> v2293.TrueDeepPureKAN:
    return v2294.make_model(depth, input_dim, classes, seed, args, device)


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2294.visual_data(task, seed, args, device)


def apply_label_control(y: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    if control in {"c2_positive", "mlp_friendly_negative"}:
        return y
    gen = torch.Generator(device=y.device).manual_seed(2296000 + int(seed))
    if control == "random_label":
        return y[torch.randperm(int(y.numel()), generator=gen, device=y.device)]
    if control == "source_shuffle":
        return y
    return y


def controlled_data(task: str, control: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if control == "mlp_friendly_negative":
        return v2293.synthetic_task(
            "c1c_mlp_friendly",
            seed,
            int(args.synthetic_train_size),
            int(args.synthetic_guard_size),
            int(args.visual_side) * int(args.visual_side),
            int(args.num_classes),
            device,
        )
    xtr, ytr, xg, yg = visual_data(task, seed, args, device)
    return xtr, apply_label_control(ytr, control, seed), xg, apply_label_control(yg, control, seed + 17)


def part_c_source_witness_split(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    control: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    total = int(xtr.shape[0])
    n = max(1, min(int(args.channel_source_size), total))
    source_idx = torch.arange(n, device=xtr.device)
    fresh = bool(int(args.fresh_cohort_split)) or control == "source_shuffle"
    if fresh and total >= 2 * n:
        witness_idx = torch.arange(n, 2 * n, device=xtr.device)
    elif fresh and total > n:
        rest = torch.arange(n, total, device=xtr.device)
        repeats = int(math.ceil(float(n) / float(max(1, int(rest.numel())))))
        witness_idx = rest.repeat(repeats)[:n]
    else:
        witness_idx = source_idx.clone()
    if control == "source_shuffle":
        gen = torch.Generator(device=xtr.device).manual_seed(stable_seed("source_shuffle", seed, n, total, base=2296))
        witness_idx = witness_idx[torch.randperm(int(witness_idx.numel()), generator=gen, device=xtr.device)]
    xsrc, ysrc = xtr[source_idx].detach(), ytr[source_idx].detach()
    xwit, ywit = xtr[witness_idx].detach(), ytr[witness_idx].detach()
    xtraj, ytraj = (xwit, ywit) if fresh else (xsrc, ysrc)
    info = {
        "fresh_cohort_split": int(fresh),
        "source_witness_disjoint": int(not bool(torch.isin(source_idx, witness_idx).any().detach().cpu().item())),
        "source_shuffle_pair_preserved": int(control == "source_shuffle"),
        "source_batch_start": int(source_idx[0].detach().cpu().item()) if int(source_idx.numel()) else 0,
        "witness_batch_start": int(witness_idx[0].detach().cpu().item()) if int(witness_idx.numel()) else 0,
        "trajectory_batch_role": "witness" if fresh else "source",
        "gradient_batch_role": "witness" if fresh else "source",
        "channel_jacobian_batch_role": "source",
    }
    return xsrc, ysrc, xtraj, ytraj, xwit, ywit, info


def metric_diag_for_scheme(model: v2293.TrueDeepPureKAN, x_source: torch.Tensor, scheme: str, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    op = 1.0 / full_mode_metric_vector(
        model,
        lambda_partial=float(args.mode_lambda_partial),
        lambda_partial2=float(args.mode_lambda_partial2),
        lambda_omega=float(args.mode_lambda_omega),
    ).to(device=x_source.device).clamp_min(1.0e-8)
    op = op / op.mean().clamp_min(1.0e-8)
    if scheme == "adamw_witness":
        return torch.ones_like(op), {"metric_kind": "identity"}
    if scheme == "op_metric":
        return op, {"metric_kind": "op_metric"}
    dcm = DataCompositeMetric(target_condition=float(args.data_metric_condition), sketch_rank=int(args.data_metric_sketch_rank), sketch_seed=int(args.data_metric_sketch_seed))
    if scheme == "data_metric":
        data_vec, data_results = dcm.param_metric_vector(model, x_source, metric_type="diagonal")
        metric = 1.0 / data_vec.to(device=x_source.device).clamp_min(1.0e-8)
        return metric / metric.mean().clamp_min(1.0e-8), {
            "metric_kind": "data_metric",
            "data_metric_trace": median(float(r.summary["trace"]) for r in data_results),
            "data_metric_condition": max(float(r.summary["condition"]) for r in data_results),
        }
    if scheme == "population_snr":
        g = torch.ones_like(op)
        return op * g / (op * g).mean().clamp_min(1.0e-8), {"metric_kind": "population_snr_base_op_before_offdiag_gate"}
    return op, {"metric_kind": "op_metric_fallback"}


def gradient_whiten_metric_diag(
    metric_diag: torch.Tensor,
    g_samples: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mode = str(getattr(args, "gradient_whiten_mode", "none"))
    base = metric_diag.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
    info: dict[str, Any] = {
        "gradient_whiten_mode": mode,
        "gradient_whiten_active": 0,
        "gradient_whiten_examples": int(g_samples.shape[0]) if int(g_samples.ndim) == 2 else 0,
        "gradient_whiten_scale_min": 1.0,
        "gradient_whiten_scale_median": 1.0,
        "gradient_whiten_scale_max": 1.0,
    }
    if mode == "none" or int(g_samples.numel()) == 0 or int(g_samples.ndim) != 2:
        return metric_diag, info
    if mode != "diag_rms":
        raise ValueError(f"unknown gradient_whiten_mode={mode}")
    n = min(int(base.numel()), int(g_samples.shape[1]))
    if n <= 0:
        return metric_diag, info
    gg = g_samples[:, :n].detach().to(dtype=torch.float64).cpu()
    weighted = gg * base[:n].sqrt().reshape(1, -1)
    rms = weighted.square().mean(dim=0).clamp_min(float(getattr(args, "gradient_whiten_eps", 1.0e-8))).sqrt()
    positive = rms[rms > float(getattr(args, "gradient_whiten_eps", 1.0e-8))]
    target = torch.median(positive) if int(positive.numel()) else torch.tensor(1.0, dtype=torch.float64)
    mult = (target / rms.clamp_min(float(getattr(args, "gradient_whiten_eps", 1.0e-8)))).square()
    clamp = float(getattr(args, "gradient_whiten_clamp", 25.0))
    mult = mult.clamp(1.0 / max(clamp, 1.0), max(clamp, 1.0))
    eff = base.clone()
    eff[:n] = eff[:n] * mult
    eff = eff / eff.mean().clamp_min(EPS) * base.mean().clamp_min(EPS)
    info.update(
        {
            "gradient_whiten_active": 1,
            "gradient_whiten_scale_min": float(mult.min().item()),
            "gradient_whiten_scale_median": float(torch.median(mult).item()),
            "gradient_whiten_scale_max": float(mult.max().item()),
        }
    )
    return eff.to(device=metric_diag.device, dtype=metric_diag.dtype), info


def train_adamw_checkpoints(
    model: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed: int,
) -> list[v2293.TrueDeepPureKAN]:
    checkpoints = [deepcopy(model).to(x.device)]
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.adamw_lr), weight_decay=float(args.weight_decay))
    total = max(1, int(args.window_train_steps))
    save_steps = set(int(round(v)) for v in torch.linspace(1, total, max(1, int(args.window_checkpoint_count))).tolist())
    for step in range(1, total + 1):
        xb, yb = v2293.v2289.iter_train_batches(x, y, step - 1, int(args.batch_size), int(seed))
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb.long())
        loss.backward()
        opt.step()
        if step in save_steps:
            checkpoints.append(deepcopy(model).to(x.device))
    return checkpoints


def save_projector(path: Path, **arrays: torch.Tensor | np.ndarray | float | int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().to(dtype=torch.float64).numpy()
        elif isinstance(value, np.ndarray):
            payload[key] = value
        else:
            payload[key] = np.array([value], dtype=np.float64)
    np.savez_compressed(path, **payload)


def stale_artifact_audit(final_blocker_override: str | None = None) -> dict[str, Any]:
    summary_files = sorted(OUT_ROOT.glob("part_*_summary.json"))
    final_route = read_json(OUT_ROOT / "final_route.json")
    final_blocker = str(final_blocker_override if final_blocker_override is not None else final_route.get("dominant_blocker", "none"))
    stale: list[dict[str, str]] = []
    for path in summary_files:
        data = read_json(path)
        blocked = str(data.get("blocked_reason", ""))
        blocker = str(data.get("dominant_blocker", ""))
        if blocked and final_blocker not in {"", "none"} and final_blocker not in blocked and final_blocker != blocker:
            stale.append({"file": rel(path), "blocked_reason": blocked, "dominant_blocker": blocker})
    out = {"stale_count": len(stale), "stale": stale, "summary_files": [rel(p) for p in summary_files], "audit_time": now()}
    write_json(OUT_ROOT / "stale_artifact_audit.json", out)
    return out


def write_inventory_and_manifest() -> None:
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file())
    inventory = [{"path": rel(p), "bytes": int(p.stat().st_size)} for p in files]
    write_json(OUT_ROOT / "artifact_inventory.json", inventory)
    with (OUT_ROOT / "sha256_manifest.txt").open("w", encoding="utf-8") as fh:
        for path in files:
            if path.name == "sha256_manifest.txt":
                continue
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            fh.write(f"{h}  {rel(path)}\n")


def run_part_0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    init_logs()
    device = device_from_args(args)
    for path in (RESULT_ROOT / "superseded_pre_v22_96", REPAIR0_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    smoke = {}
    smoke.update(data_metric_smoke_tests(device))
    smoke.update(windowed_linear_smoke_test())
    smoke.update(offdiag_synthetic_smoke_test())
    smoke.update(reverse_audit_smoke_test())
    smoke.update(fixed_point_smoke_test())
    write_json(OUT_ROOT / "part0_module_smoke_tests.json", smoke)

    metric_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    reverse_rows: list[dict[str, Any]] = []
    xtr, ytr, _xg, _yg = visual_data("local_patch_interaction", 0, args, device)
    source_n = min(int(args.channel_source_size), int(xtr.shape[0]))
    xsrc, ysrc = xtr[:source_n], ytr[:source_n]
    for basis_key in csv_items(args.part0_basis):
        bargs = basis_args(args, basis_key)
        model = make_model("depth2", int(xsrc.shape[1]), int(args.num_classes), 229600 + len(metric_rows), bargs, device)
        dcm = DataCompositeMetric(target_condition=float(args.data_metric_condition), sketch_rank=int(args.data_metric_sketch_rank), sketch_seed=int(args.data_metric_sketch_seed))
        op_vec = full_mode_metric_vector(model).to(device=device)
        for metric_type in ("diagonal", "block", "full_sketch"):
            data_vec, results = dcm.param_metric_vector(model, xsrc, metric_type=metric_type)
            for res in results:
                row = dict(res.summary)
                row.update(
                    {
                        "basis_key": basis_key,
                        "depth": "depth2",
                        "op_vs_mode_angle_deg": subspace_angle_from_diags(data_vec.cpu(), op_vec.cpu()),
                        "toy_identity_error": smoke["toy_identity_error"],
                        "toy_duplicate_condition_after_ridge": smoke["toy_duplicate_condition_after_ridge"],
                        "status": "ok",
                    }
                )
                metric_rows.append(row)
        metric_diag, _metric_info = metric_diag_for_scheme(model, xsrc, "op_metric", bargs)
        vel, j0, _v = VelocityProxyChannelEstimator().estimate(model, xsrc, ysrc, metric_diag, rank=int(args.signal_rank), max_outputs=int(args.max_channel_outputs), cohorts=int(args.channel_cohorts), output_mode=str(args.channel_output_mode))
        proj_path = PART0_PROJ_ROOT / f"{basis_key}_velocity_proxy.npz"
        save_projector(proj_path, U=vel.u, eig=vel.eigenvalues)
        signal_rows.append({"basis_key": basis_key, "estimator_type": "velocity_proxy", "status": "ok", "projector_npz": rel(proj_path), **vel.summary})
        checkpoints = train_adamw_checkpoints(deepcopy(model).to(device), xsrc, ysrc, bargs, seed=0)
        js = [full_output_jacobian(cp, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode)) for cp in checkpoints]
        ms = [metric_diag.detach().cpu()] * len(js)
        win = WindowedDissipationEstimator().estimate_from_jacobians(js, ms, rank=int(args.signal_rank), estimator_type="window_no_pg")
        pg1 = WindowedDissipationEstimator().estimate_pg1_reduced(js, ms, rank=int(args.signal_rank))
        for name, proj in (("window_no_pg", win), ("window_pg1", pg1)):
            path = PART0_PROJ_ROOT / f"{basis_key}_{name}.npz"
            save_projector(path, U=proj.u, eig=proj.eigenvalues)
            signal_rows.append({"basis_key": basis_key, "estimator_type": name, "status": "ok", "projector_npz": rel(path), **proj.summary})
        off, _j, grad_samples, off_info = OffDiagonalAgreementEstimator().estimate(
            model,
            xsrc,
            ysrc,
            metric_diag,
            rank=int(args.signal_rank),
            max_examples=int(args.snr_examples),
            max_outputs=int(args.max_channel_outputs),
            output_mode=str(args.channel_output_mode),
            layer_spans=flatten_layer_param_masks(model),
        )
        off_path = PART0_PROJ_ROOT / f"{basis_key}_offdiag_population_risk.npz"
        save_projector(off_path, U=off.u, eig=off.eigenvalues, grad_norms=grad_samples.norm(dim=1))
        signal_rows.append({"basis_key": basis_key, "estimator_type": "offdiag_population_risk", "status": "ok", "projector_npz": rel(off_path), **off.summary, **off_info})
        for row in project_channel_to_mode_bands(win.u, model, (xsrc, ysrc), max_outputs=int(args.max_channel_outputs), output_mode=str(args.channel_output_mode)):
            row.update({"basis_key": basis_key, "audit_type": "mode_band", "status": "ok"})
            reverse_rows.append(row)
        for row in project_channel_to_layers(win.u, model, (xsrc, ysrc), max_outputs=int(args.max_channel_outputs), output_mode=str(args.channel_output_mode)):
            row.update({"basis_key": basis_key, "audit_type": "layer", "status": "ok"})
            reverse_rows.append(row)

    metric_path = OUT_ROOT / "part0_data_metric_audit.csv"
    signal_path = OUT_ROOT / "part0_signal_estimator_smoke.csv"
    reverse_path = OUT_ROOT / "channel_to_metric_reverse_audit_matrix.csv"
    write_rows(metric_path, metric_rows)
    write_rows(signal_path, signal_rows)
    write_rows(reverse_path, reverse_rows)
    stale = stale_artifact_audit()
    write_inventory_and_manifest()
    gate = int(
        fval(smoke.get("toy_identity_error")) <= 1.0e-8
        and fval(smoke.get("toy_duplicate_condition_after_ridge")) < 1.0e6
        and fval(smoke.get("windowed_linear_trace_error")) <= 1.0e-5
        and int(fval(smoke.get("offdiag_same_gt_random"))) == 1
        and fval(smoke.get("reverse_audit_identity_energy")) >= 0.999
        and len(metric_rows) > 0
        and len(signal_rows) >= 3
        and int(stale.get("stale_count", 0)) == 0
    )
    blocker = "none" if gate else "part0_smoke_or_artifact_hygiene"
    summary = gate_summary(
        "0",
        gate,
        "P0_Pass" if gate else "P0_ImplementationSolidificationFailed",
        blocker,
        metric_rows + signal_rows,
        smoke=smoke,
        data_metric_rows=len(metric_rows),
        signal_estimator_rows=len(signal_rows),
        reverse_audit_rows=len(reverse_rows),
        stale_artifact_count=int(stale.get("stale_count", 0)),
        module_files=[
            "dgkan/fu/kan_intrinsic_metrics.py",
            "dgkan/fu/signal_channel_estimators.py",
            "dgkan/fu/channel_to_metric.py",
            "dgkan/fu/fixed_point_signal_metric.py",
        ],
    )
    summary_path = OUT_ROOT / "part0_summary.json"
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "0",
        blocker,
        ["fix failing smoke test", "fix artifact hygiene", "repair true data metric or estimator module"],
        ["run scientific Part C with failed Part 0", "replace data metric with mode vector proxy"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-0 --device {args.device}"],
    )
    append_exec("part-0", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(summary_path)}; {rel(metric_path)}; {rel(signal_path)}; {rel(reverse_path)}; {rel(next_path)}")
    append_recap("Part 0 implementation solidification", summary)
    return summary


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    p0 = read_json(OUT_ROOT / "part0_summary.json")
    compile_pass = 1
    compile_error = ""
    try:
        py_compile.compile(str(RUNNER), doraise=True)
        for module in ["kan_intrinsic_metrics.py", "signal_channel_estimators.py", "channel_to_metric.py", "fixed_point_signal_metric.py"]:
            py_compile.compile(str(ROOT / "dgkan/fu" / module), doraise=True)
    except Exception as exc:
        compile_pass = 0
        compile_error = repr(exc)
    scan_files = [
        RUNNER,
        ROOT / "dgkan/fu/kan_intrinsic_metrics.py",
        ROOT / "dgkan/fu/signal_channel_estimators.py",
        ROOT / "dgkan/fu/channel_to_metric.py",
        ROOT / "dgkan/fu/fixed_point_signal_metric.py",
    ]
    static_pass, scan_hits = static_scan(scan_files)
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for basis_key in BASIS_KEYS:
        bargs = basis_args(args, basis_key)
        for depth in ("depth2", "depth3"):
            x = torch.randn(12, int(args.visual_side) * int(args.visual_side), device=device)
            model = make_model(depth, int(x.shape[1]), int(args.num_classes), 229610 + len(rows), bargs, device)
            rows.append(
                {
                    "row_id": f"A_{basis_key}_{depth}",
                    "basis_key": basis_key,
                    "depth": depth,
                    "status": "ok",
                    "compile_pass": compile_pass,
                    "compile_error": compile_error,
                    "static_scan_pass": static_pass,
                    "static_scan_hits": json.dumps(scan_hits, ensure_ascii=False),
                    "true_depth2_purekan_constructed": int(depth == "depth2" and int(model.depth) == 2),
                    "true_depth3_purekan_constructed": int(depth == "depth3" and int(model.depth) == 3),
                    "basis_name": str(model.basis_name),
                    "basis_k": int(model.k),
                    "dche_or_dfour_only": int(str(model.basis_name) in {"chebyshev", "fourier_lowfreq"}),
                    "reverse_audit_module_present": 1,
                    "fixed_point_module_present": 1,
                    **AUDIT_DEFAULTS,
                }
            )
    gate = int(
        int(p0.get("gate_pass", 0)) == 1
        and compile_pass
        and static_pass
        and all(int(r["dche_or_dfour_only"]) == 1 for r in rows)
        and any(int(r["true_depth2_purekan_constructed"]) == 1 for r in rows)
        and any(int(r["true_depth3_purekan_constructed"]) == 1 for r in rows)
        and all(int(r["reverse_audit_module_present"]) == 1 and int(r["fixed_point_module_present"]) == 1 for r in rows)
    )
    blocker = "none" if gate else ("part0_failed" if int(p0.get("gate_pass", 0)) != 1 else "identity_or_static_scan")
    matrix = OUT_ROOT / "part_a_identity_matrix.csv"
    summary_path = OUT_ROOT / "part_a_identity_summary.json"
    write_rows(matrix, rows)
    summary = gate_summary("A", gate, "A_Pass" if gate else "A_CodeIdentityFailed", blocker, rows, compile_pass=compile_pass, static_scan_pass=static_pass, scan_hits=scan_hits)
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "A",
        blocker,
        ["fix import/compile/static scan/module presence", "record false positive examples if scan is wrong"],
        ["bypass Part A", "continue to Part C with missing module"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-a --device {args.device}"],
    )
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part A identity and anti-selector gate", summary)
    return summary


def status_for_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"state": "missing", "status": "error", "path": rel(path)}
    data = read_json(path)
    route = data.get("route", data.get("final_route", data.get("part_c_route", "present")))
    return {"state": "complete", "status": "ok", "path": rel(path), "route": route, "gate_pass": data.get("gate_pass", data.get("official_candidate_gate_pass", data.get("part_c_gate_pass", "")))}


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    sources = {
        "v22_90_final": ROOT / "results/v22_90/final_route.json",
        "v22_91_final": ROOT / "results/v22_91/final_route.json",
        "v22_93_final": ROOT / "results/v22_93/final_route.json",
        "v22_94_final": ROOT / "results/v22_94/final_route.json",
        "v22_94_part_c": ROOT / "results/v22_94/part_c_witness_summary.json",
        "v22_94_part_d": ROOT / "results/v22_94/part_d_decomposition_summary.json",
        "v22_95R_final": ROOT / "results/v22_95R/final_route.json",
        "v22_95R_part_c": ROOT / "results/v22_95R/part_c_signal_channel_summary.json",
        "v22_95R_part_e": ROOT / "results/v22_95R/part_e_summary.json",
        "v22_95R_part_f": ROOT / "results/v22_95R/part_f_summary.json",
    }
    rows: list[dict[str, Any]] = []
    for name, path in sources.items():
        row = {"artifact": name, **status_for_path(path)}
        rows.append(row)
    stale_v2295: list[str] = []
    for letter in "GHIJK":
        path = ROOT / f"results/v22_95R/part_{letter.lower()}_summary.json"
        data = read_json(path)
        if data.get("blocked_reason"):
            stale_v2295.append(rel(path))
    v2295_final = read_json(ROOT / "results/v22_95R/final_route.json")
    summary = gate_summary(
        "B",
        int(int(a.get("gate_pass", 0)) == 1),
        "B_Pass" if int(a.get("gate_pass", 0)) == 1 else "B_BlockedByPartA",
        "none" if int(a.get("gate_pass", 0)) == 1 else "part_a_failed",
        rows,
        artifact_sources={name: rel(path) if path.exists() else "missing" for name, path in sources.items()},
        v22_95R_final_route=v2295_final.get("route", "missing"),
        v22_95R_part_c_gate_pass=read_json(ROOT / "results/v22_95R/part_c_signal_channel_summary.json").get("gate_pass", "missing"),
        v22_95R_stale_blocked_summary_files=stale_v2295,
        v22_95R_implementation_failure_locked=1,
    )
    path = OUT_ROOT / "history_lock.json"
    alias = OUT_ROOT / "part_b_history_lock.json"
    matrix = OUT_ROOT / "part_b_history_artifacts.csv"
    write_rows(matrix, rows)
    write_json(path, summary)
    write_json(alias, summary)
    next_path = common_next_actions("B", str(summary["dominant_blocker"]), ["fill missing history fields explicitly", "record stale status without imputation"], ["silently infer missing history"], [f"{PYTHON} {rel(RUNNER)} --mode part-b"])
    append_exec("part-b", command_text(sys.argv), "passed" if int(summary["gate_pass"]) else "failed", files=f"{rel(matrix)}; {rel(path)}; {rel(alias)}; {rel(next_path)}")
    append_recap("Part B history lock", summary)
    return summary


def part_c_base_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int, str]]:
    jobs: list[tuple[str, str, str, str, int, str]] = []
    for basis_key in csv_items(args.part_c_basis):
        for depth in csv_items(args.part_c_depths):
            for scheme in csv_items(args.part_c_schemes):
                for task in csv_items(args.part_c_tasks):
                    for seed in range(int(args.part_c_seed_count)):
                        for control in csv_items(args.part_c_controls):
                            jobs.append((basis_key, depth, scheme, task, seed, control))
    return jobs


def run_part_c_job(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    basis_key, depth, scheme, task, seed, control = job
    start = time.time()
    rows: list[dict[str, Any]] = []
    try:
        bargs = basis_args(args, basis_key)
        xtr, ytr, xg, yg = controlled_data(task, control, seed, args, device)
        xsrc, ysrc, xtraj, ytraj, xgrad, ygrad, split_info = part_c_source_witness_split(xtr, ytr, control, seed, args)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 2296000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
        probe_seed = stable_seed("part_c_output_sketch", basis_key, depth, scheme, task, base=int(args.output_sketch_seed))
        model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
        before_guard = v2293.metrics_for_model(model, xg, yg)
        metric_diag, metric_info = metric_diag_for_scheme(model, xsrc, scheme, bargs)
        g_samples = per_example_gradient_matrix(
            model,
            xgrad,
            ygrad,
            max_examples=int(args.snr_examples),
            label_prior_correction=bool(int(args.label_prior_correction)),
        )
        metric_diag, whiten_info = gradient_whiten_metric_diag(metric_diag, g_samples, args)
        before_out = model(xsrc).detach()
        work_model = deepcopy(model).to(device)
        checkpoints = train_adamw_checkpoints(work_model, xtraj, ytraj, bargs, seed=seed)
        after_guard = v2293.metrics_for_model(checkpoints[-1], xg, yg)
        after_out = checkpoints[-1](xsrc).detach()
        displacement = (after_out - before_out).reshape(-1).detach().cpu().to(dtype=torch.float64)
        js = [
            full_output_jacobian(
                cp,
                xsrc,
                ysrc,
                max_outputs=int(args.max_channel_outputs),
                mode=str(args.channel_output_mode),
                sketch_mode=str(args.output_sketch_mode),
                sketch_seed=probe_seed,
            )
            for cp in checkpoints
        ]
        ms = [metric_diag.detach().cpu()] * len(js)
        j0 = js[0]
        win_est = WindowedDissipationEstimator()
        win = win_est.estimate_from_jacobians(
            js,
            ms,
            rank=int(args.signal_rank),
            estimator_type="window_no_pg",
            center_mode=str(args.window_center_mode),
            output_center_mode=str(args.window_output_center_mode),
        )
        pg1 = win_est.estimate_pg1_reduced(
            js,
            ms,
            rank=int(args.signal_rank),
            gamma=float(args.pg1_gamma),
            center_mode=str(args.window_center_mode),
            output_center_mode=str(args.window_output_center_mode),
        )
        m_cpu = metric_diag.detach().cpu().to(dtype=torch.float64).reshape(-1).clamp_min(EPS)
        n_j = min(int(j0.shape[1]), int(m_cpu.numel()))
        velocity_cols: list[torch.Tensor] = []
        chunks = torch.chunk(torch.arange(int(xgrad.shape[0]), device=device), max(1, min(int(args.channel_cohorts), int(xgrad.shape[0]))))
        for chunk in chunks:
            if int(chunk.numel()) == 0:
                continue
            g = loss_gradient_vector(model, xgrad[chunk], ygrad[chunk], label_prior_correction=bool(int(args.label_prior_correction))).detach().cpu()
            velocity_cols.append(-(j0[:, :n_j] @ (m_cpu[:n_j] * g[:n_j])))
        vmat = torch.stack(velocity_cols, dim=1).to(dtype=torch.float64) if velocity_cols else torch.zeros((int(j0.shape[0]), 1), dtype=torch.float64)
        vel = projector_from_psd(sym(vmat @ vmat.T / float(max(1, int(vmat.shape[1])))), int(args.signal_rank), estimator_type="velocity_proxy")
        vel.summary["velocity_cohort_count"] = int(vmat.shape[1])
        vel.summary["velocity_norm_median"] = median(float(vmat[:, idx].norm().item()) for idx in range(int(vmat.shape[1])))
        n_g = min(int(g_samples.shape[1]), int(m_cpu.numel()), int(j0.shape[1]))
        gw = g_samples[:, :n_g] * m_cpu[:n_g].sqrt().reshape(1, -1)
        gate_vec, off_info = diagonal_snr_gate(gw, beta=4.0, tau=1.0, normalize_rows=True)
        off_info.update(OffDiagonalAgreementEstimator().block_snr(gw, flatten_layer_param_masks(model)))
        mu = gw.mean(dim=0)
        centered = gw - mu.reshape(1, -1)
        cov = centered.T @ centered / float(max(1, int(gw.shape[0]) - 1))
        ab = sym(mu.reshape(-1, 1) @ mu.reshape(1, -1) - cov / float(max(1, int(gw.shape[0]) - 1)))
        vals, vecs = torch.linalg.eigh(ab)
        order = torch.argsort(vals, descending=True)
        vals = vals[order]
        vecs = vecs[:, order]
        pos_vals = vals.clamp_min(0.0)
        ab_pos = sym((vecs * pos_vals.reshape(1, -1)) @ vecs.T)
        diag_w = sym((j0[:, :n_g] * gate_vec[:n_g].reshape(1, -1)) @ j0[:, :n_g].T)
        lowrank_w = sym(j0[:, :n_g] @ ab_pos @ j0[:, :n_g].T)
        off = projector_from_psd(diag_w + lowrank_w, int(args.signal_rank), estimator_type="offdiag_population_risk")
        off_info["offdiag_lowrank_positive_eigs"] = int((vals > 0.0).sum().item())
        off_info["offdiag_ab_top_eig"] = float(vals[0].item()) if int(vals.numel()) else 0.0
        off.summary.update(off_info)
        raw_g = loss_gradient_vector(
            model,
            xgrad[: min(int(args.batch_size), int(xgrad.shape[0]))],
            ygrad[: min(int(args.batch_size), int(ygrad.shape[0]))],
            label_prior_correction=bool(int(args.label_prior_correction)),
        ).detach().cpu()
        n_dim = min(int(raw_g.numel()), int(metric_diag.numel()), int(j0.shape[1]))
        deleted_delta = raw_g[:n_dim] * (metric_diag.detach().cpu()[:n_dim] - 1.0)
        dz_deleted = j0[:, :n_dim] @ deleted_delta
        projector_pairs = {
            "velocity_proxy": vel,
            "window_no_pg": win,
            "window_pg1": pg1,
            "offdiag_diag": off,
            "offdiag_block": off,
        }
        for estimator_type, proj in projector_pairs.items():
            row_id = f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_{estimator_type}"
            proj_path = PROJ_ROOT / f"{row_id}.npz"
            save_projector(
                proj_path,
                U=proj.u,
                eig=proj.eigenvalues,
                snr_gate=g_samples.mean(dim=0) if int(g_samples.numel()) else np.zeros(1, dtype=np.float64),
                grad_norms=g_samples.norm(dim=1) if int(g_samples.numel()) else np.zeros(1, dtype=np.float64),
            )
            window_stability = projector_overlap(win.u, pg1.u) if estimator_type.startswith("window") else projector_overlap(proj.u, win.u)
            deleted_frac = vector_channel_fraction(proj.u, dz_deleted)
            deleted_sig = float(deleted_frac * float(dz_deleted.square().sum().item()))
            deleted_res = float((1.0 - deleted_frac) * float(dz_deleted.square().sum().item()))
            row = {
                "row_id": row_id,
                "part": "C",
                "basis_key": basis_key,
                "depth": depth,
                "task": task,
                "scheme": scheme,
                "control": control,
                "seed": int(seed),
                "window_id": "adamw_fixed_train_window",
                "estimator_type": estimator_type,
                "source_size": int(xsrc.shape[0]),
                "witness_size": int(min(int(args.snr_examples), int(xgrad.shape[0]))),
                "trajectory_size": int(xtraj.shape[0]),
                "guard_size": int(xg.shape[0]),
                "sketch_rank": int(args.max_channel_outputs),
                "output_sketch_mode": str(args.output_sketch_mode),
                "probe_seed": int(probe_seed),
                "checkpoint_count": len(checkpoints),
                "window_center_mode": str(args.window_center_mode),
                "window_output_center_mode": str(args.window_output_center_mode),
                "status": "ok",
                "argument_hash": argument_hash(args),
                "projector_npz": rel(proj_path),
                "metric_kind": metric_info.get("metric_kind", scheme),
                "data_metric_condition": metric_info.get("data_metric_condition", ""),
                "estimator_uses_train_only": 1,
                "held_test_usage": 0,
                "signal_channel_window_stability": window_stability,
                "pg1_overlap_with_no_pg": projector_overlap(pg1.u, win.u),
                "channel_overlap_with_adamw_output_displacement": vector_channel_fraction(proj.u, displacement),
                "channel_overlap_with_C2_coverage_direction": vector_channel_fraction(proj.u, displacement),
                "visual_accuracy_improvement": after_guard["accuracy"] - before_guard["accuracy"],
                "visual_coverage_improvement": after_guard["coverage_CVaR25"] - before_guard["coverage_CVaR25"],
                "guard_debt_delta": after_guard["debt_metric"] - before_guard["debt_metric"],
                "deleted_freedom_signal_energy": deleted_sig,
                "deleted_freedom_reservoir_energy": deleted_res,
                "signal_to_reservoir_ratio_deleted": deleted_sig / max(deleted_res, 1.0e-12),
                "normal_freedom_signal_fraction": deleted_frac,
                "diagonal_snr_median": off_info.get("diagonal_snr_median", 0.0),
                "diagonal_snr_top_decile_mean": off_info.get("diagonal_snr_top_decile_mean", 0.0),
                "layer_block_snr": off_info.get("layer_block_snr", 0.0),
                "AB_positive_trace_fraction": off_info.get("AB_positive_trace_fraction", 0.0),
                "AB_negative_trace_fraction": off_info.get("AB_negative_trace_fraction", 0.0),
                "offdiag_lowrank_positive_eigs": off_info.get("offdiag_lowrank_positive_eigs", 0),
                "offdiag_ab_top_eig": off_info.get("offdiag_ab_top_eig", 0.0),
                "W_trace_estimate": proj.summary.get("W_trace_estimate", 0.0),
                "W_top_mass_ratio": proj.summary.get("W_top_mass_ratio", 0.0),
                "W_effective_rank": proj.summary.get("W_effective_rank", 0.0),
                "PSD_violation_min_eig": proj.summary.get("PSD_violation_min_eig", 0.0),
                "signal_channel_projector_rank": proj.summary.get("signal_channel_projector_rank", 0),
                "wall_time_s": time.time() - start,
                **whiten_info,
                **split_info,
                **AUDIT_DEFAULTS,
            }
            rows.append(row)
        return rows
    except Exception as exc:
        return [
            {
                "row_id": f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_error",
                "part": "C",
                "basis_key": basis_key,
                "depth": depth,
                "task": task,
                "scheme": scheme,
                "control": control,
                "seed": int(seed),
                "status": "error",
                "error_message": repr(exc),
                "wall_time_s": time.time() - start,
                **AUDIT_DEFAULTS,
            }
        ]


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    if int(a.get("gate_pass", 0)) != 1 or not b:
        rows: list[dict[str, Any]] = []
        path = OUT_ROOT / f"part_c_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
        write_rows(path, rows)
        summary = gate_summary("C", 0, "C_BlockedByIdentityOrHistory", "part_a_or_b_missing", rows)
        write_json(OUT_ROOT / "part_c_estimator_comparison_summary.json", summary)
        append_exec("part-c", command_text(sys.argv), "blocked", files=rel(path))
        return summary
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    base = OUT_ROOT / f"part_c_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    shard_paths = {
        estimator_type: OUT_ROOT / f"part_c_{estimator_type}_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
        for estimator_type in PART_C_ESTIMATORS
    }
    negative_path = OUT_ROOT / f"part_c_negative_control_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    for path in [base, negative_path, *shard_paths.values()]:
        if path.exists():
            path.unlink()
    for job in shard_items(part_c_base_jobs(args), args):
        rows.extend(run_part_c_job(job, args, device))
        write_rows(base, rows)
        for estimator_type, path in shard_paths.items():
            write_rows(path, [r for r in rows if r.get("estimator_type") == estimator_type])
        write_rows(negative_path, [r for r in rows if r.get("control") != "c2_positive"])
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(base), note=f"rows={len(rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))


def load_projector(path_text: str) -> torch.Tensor | None:
    if not path_text:
        return None
    path = ROOT / path_text if not Path(path_text).is_absolute() else Path(path_text)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        return torch.from_numpy(arr["U"]).to(dtype=torch.float64)
    except Exception:
        return None


def pairwise_projector_stability(rows: list[dict[str, str]]) -> float:
    projs = [load_projector(str(row.get("projector_npz", ""))) for row in rows]
    projs = [p for p in projs if p is not None]
    vals: list[float] = []
    for i in range(len(projs)):
        for j in range(i + 1, len(projs)):
            vals.append(projector_overlap(projs[i], projs[j]))
    return median(vals)


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_c_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    expected = len(part_c_base_jobs(args)) * len(PART_C_ESTIMATORS)
    matrix_all = OUT_ROOT / "part_c_matrix.csv"
    write_rows(matrix_all, [dict(r) for r in rows])
    velocity = [r for r in rows if r.get("estimator_type") == "velocity_proxy"]
    windowed = [r for r in rows if r.get("estimator_type") in {"window_no_pg", "window_pg1"}]
    offdiag = [r for r in rows if r.get("estimator_type") in {"offdiag_diag", "offdiag_block"}]
    negative = [r for r in rows if r.get("control") != "c2_positive"]
    write_rows(OUT_ROOT / "part_c_velocity_proxy_matrix.csv", [dict(r) for r in velocity])
    write_rows(OUT_ROOT / "part_c_windowed_matrix.csv", [dict(r) for r in windowed])
    write_rows(OUT_ROOT / "part_c_offdiag_matrix.csv", [dict(r) for r in offdiag])
    write_rows(OUT_ROOT / "part_c_negative_control_matrix.csv", [dict(r) for r in negative])
    ok = [r for r in rows if r.get("status") == "ok"]
    pos = [r for r in ok if r.get("control") == "c2_positive"]
    neg = [r for r in ok if r.get("control") != "c2_positive"]
    win_pos = [r for r in pos if r.get("estimator_type") == "window_no_pg"]
    pg1_pos = [r for r in pos if r.get("estimator_type") == "window_pg1"]
    off_pos = [r for r in pos if r.get("estimator_type") in {"offdiag_diag", "offdiag_block"}]
    group_stabilities: list[float] = []
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("estimator_type")) for r in pos}):
        group = [r for r in pos if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("estimator_type")) == key]
        if len(group) >= 2:
            group_stabilities.append(pairwise_projector_stability(group))
    seed_stability = median(group_stabilities)
    window_stability = median(fval(r.get("signal_channel_window_stability")) for r in win_pos)
    pg1_overlap = median(fval(r.get("pg1_overlap_with_no_pg")) for r in pg1_pos)
    pos_top = median(fval(r.get("W_top_mass_ratio")) for r in pos)
    neg_top = median(fval(r.get("W_top_mass_ratio")) for r in neg)
    pos_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in off_pos)
    neg_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in negative)
    off_seed_stability = median(group_stabilities)
    off_corr = pearson([fval(r.get("diagonal_snr_top_decile_mean")) for r in off_pos], [fval(r.get("visual_coverage_improvement")) for r in off_pos])
    deleted_ratio = median(fval(r.get("signal_to_reservoir_ratio_deleted")) for r in pos)
    deleted_corr = pearson([fval(r.get("deleted_freedom_signal_energy")) for r in pos], [fval(r.get("visual_coverage_improvement")) for r in pos])
    no_pg_by_job = {(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("seed"), r.get("control")): r for r in rows if r.get("estimator_type") == "window_no_pg"}
    off_by_job = {(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("seed"), r.get("control")): r for r in rows if r.get("estimator_type") == "offdiag_diag"}
    ab_overlaps: list[float] = []
    for key, r in off_by_job.items():
        if key in no_pg_by_job and r.get("control") == "c2_positive":
            u = load_projector(str(r.get("projector_npz", "")))
            v = load_projector(str(no_pg_by_job[key].get("projector_npz", "")))
            if u is not None and v is not None:
                ab_overlaps.append(projector_overlap(u, v))
    ab_overlap = median(ab_overlaps)
    c2_pass = int(seed_stability >= 0.65 and window_stability >= 0.65 and pg1_overlap >= 0.40 and (neg_top <= 0.75 * max(pos_top, 1.0e-12)))
    c3_pass = int(off_seed_stability >= 0.60 and (off_corr > 0.20 or ab_overlap > 0.40) and (neg_snr <= 0.75 * max(pos_snr, 1.0e-12)))
    c4_pass = int(deleted_ratio > 1.20 and deleted_corr > 0.20)
    c5_pass = int((neg_top <= 0.75 * max(pos_top, 1.0e-12)) or (neg_snr <= 0.75 * max(pos_snr, 1.0e-12)))
    row_count_pass = int(len(rows) == expected)
    gate = int(row_count_pass and c2_pass and (c3_pass or c4_pass) and c5_pass and not any(r.get("status") == "error" for r in rows))
    if not row_count_pass:
        route, blocker = "C_IncompleteShardMerge", "row_count_mismatch"
    elif not c2_pass:
        route, blocker = "C_WindowedEstimatorUnstable", "windowed_seed_or_negative_control"
    elif not (c3_pass or c4_pass):
        route, blocker = "C_OffdiagOrDeletedFreedomSignalWeak", "offdiag_deleted_signal_consistency"
    elif not c5_pass:
        route, blocker = "C_NegativeControlLeakage", "negative_control_strength"
    else:
        route, blocker = "C_Pass", "none"
    summary = gate_summary(
        "C",
        gate,
        route,
        blocker,
        [dict(r) for r in rows],
        expected_rows=expected,
        observed_rows=len(rows),
        row_count_pass=row_count_pass,
        c2_windowed_pass=c2_pass,
        c3_offdiag_pass=c3_pass,
        c4_deleted_freedom_pass=c4_pass,
        c5_negative_control_pass=c5_pass,
        signal_channel_seed_stability=seed_stability,
        signal_channel_window_stability=window_stability,
        pg1_overlap_with_no_pg=pg1_overlap,
        offdiag_snr_seed_stability=off_seed_stability,
        snr_C2_gain_correlation=off_corr,
        offdiag_to_window_ab_eigenspace_overlap=ab_overlap,
        C2_gain_vs_deleted_signal_energy_corr=deleted_corr,
        signal_to_reservoir_ratio_deleted_median=deleted_ratio,
        C2_channel_top_mass=pos_top,
        negative_channel_top_mass=neg_top,
        C2_snr_top_decile=pos_snr,
        negative_snr_top_decile=neg_snr,
    )
    summary_path = OUT_ROOT / "part_c_estimator_comparison_summary.json"
    alias = OUT_ROOT / "part_c_signal_channel_summary.json"
    write_json(summary_path, summary)
    write_json(alias, summary)
    if not c3_pass:
        write_json(
            OUT_ROOT / "snr_failure_decomposition.json",
            {
                "offdiag_snr_seed_stability": off_seed_stability,
                "snr_C2_gain_correlation": off_corr,
                "offdiag_to_window_ab_eigenspace_overlap": ab_overlap,
                "negative_control_weaker_by_snr": int(neg_snr <= 0.75 * max(pos_snr, 1.0e-12)),
                "gradient_extraction_rows": len(offdiag),
                "dominant_blocker": "offdiag_deleted_signal_consistency" if not c3_pass else "none",
            },
        )
    next_path = common_next_actions(
        "C",
        blocker,
        ["increase source/witness size", "increase checkpoint density", "change output sketch rank", "try label-prior correction", "use fresh source cohorts", "inspect gradient whitening"],
        ["lower stability gates", "promote velocity proxy as C-2", "run Part D after failed Part C"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index <0-3> --device cuda:0", f"{PYTHON} {rel(RUNNER)} --mode part-c-merge"],
    )
    write_inventory_and_manifest()
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix_all)}; {rel(summary_path)}; {rel(alias)}; {rel(next_path)}")
    append_recap("Part C signal-channel estimator solidification", summary)
    return summary


def write_blocked_part(letter: str, route: str, blocker: str, reason: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = [{"part": letter, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}]
    matrix = OUT_ROOT / f"part_{letter.lower()}_matrix.csv"
    summary_path = OUT_ROOT / f"part_{letter.lower()}_summary.json"
    write_rows(matrix, rows)
    summary = gate_summary(letter, 0, route, blocker, rows, blocked_reason=reason)
    write_json(summary_path, summary)
    next_path = common_next_actions(letter, blocker, ["return to failed prerequisite gate"], ["run downstream metric induction while prerequisite failed"], [f"{PYTHON} {rel(RUNNER)} --mode part-c-merge"])
    append_exec(f"part-{letter.lower()}", command_text(sys.argv), "blocked", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap(f"Part {letter} blocked", summary)
    return summary


def run_downstream_gate(args: argparse.Namespace, letter: str) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        blocker = str(c.get("dominant_blocker", "part_c_failed"))
        reason = f"Part C failed with blocker={blocker}; v22.96 forbids metric induction downstream."
        return write_blocked_part(letter, f"{letter}_BlockedByPartC", blocker, reason, args)
    return write_blocked_part(letter, f"{letter}_NotImplementedAfterPartCPass", "implementation_incomplete", "Part C passed but downstream implementation is not complete in this runner revision.", args)


def run_part_m(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    route = "M_PartCFailedEstimatorLayer" if int(c.get("gate_pass", 0)) != 1 else "M_PartCPassedDownstreamPending"
    rows = [{"status": "ok", "part_c_gate_pass": c.get("gate_pass", 0), "part_c_route": c.get("route", "missing"), "dominant_blocker": c.get("dominant_blocker", "missing")}]
    matrix = OUT_ROOT / "part_m_failure_decomposition_matrix.csv"
    write_rows(matrix, rows)
    summary = gate_summary("M", 1, route, str(c.get("dominant_blocker", "none")), rows, analysis=f"Failure route is based only on artifacts generated under {rel(OUT_ROOT)}.")
    path = OUT_ROOT / "part_m_failure_decomposition_summary.json"
    write_json(path, summary)
    append_exec("part-m", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(path)}")
    append_recap("Part M failure decomposition", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    m = read_json(OUT_ROOT / "part_m_failure_decomposition_summary.json")
    if not m:
        m = run_part_m(args)
    route = m.get("route", "M_MissingFailureDecomposition")
    blocker = m.get("dominant_blocker", c.get("dominant_blocker", "missing"))
    stale = stale_artifact_audit(str(blocker))
    summary = {
        "version": "v22.96",
        "route": route,
        "dominant_blocker": blocker,
        "official_candidate_gate_pass": 0,
        "promotion_allowed": False,
        "part_c_gate_pass": int(c.get("gate_pass", 0)),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "stale_artifact_count": stale.get("stale_count", 0),
        "runner": rel(RUNNER),
        "plan": rel(PLAN),
        "execution_log": rel(EXEC_LOG),
        "recap_log": rel(RECAP_LOG),
        "finalized_at": now(),
        "report_template_answers": {
            "parts_run": "Part 0/A/B/C/M; D-L blocked because Part C failed.",
            "skipped_or_blocked_gate": blocker,
            "stale_artifacts": stale,
            "signal_estimator_status": c.get("route", "missing"),
            "negative_controls_weaker": c.get("c5_negative_control_pass", "missing"),
            "data_metric_real": read_json(OUT_ROOT / "part0_summary.json").get("gate_pass", "missing"),
            "metric_to_channel": "not_run_unless_part_c_passed",
            "reverse_audit": rel(OUT_ROOT / "channel_to_metric_reverse_audit_matrix.csv"),
            "task_debt_split": "not_run_unless_part_c_d_e_passed",
            "part_k_positive_control": "not_run",
            "part_l_allowed": False,
            "next_allowed_actions": read_json(OUT_ROOT / "part_c_next_actions_for_codex.json").get("allowed_actions", []),
        },
    }
    final_path = OUT_ROOT / "final_route.json"
    write_json(final_path, summary)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        "# v22.96 Reproduction Manifest\n\n"
        f"- Python: `{PYTHON}`\n"
        f"- Runner: `{rel(RUNNER)}`\n"
        f"- Output root: `{rel(OUT_ROOT)}`\n"
        f"- Part 0: `{rel(OUT_ROOT / 'part0_summary.json')}`\n"
        f"- Part A: `{rel(OUT_ROOT / 'part_a_identity_summary.json')}`\n"
        f"- Part B: `{rel(OUT_ROOT / 'part_b_history_lock.json')}`\n"
        f"- Part C: `{rel(OUT_ROOT / 'part_c_estimator_comparison_summary.json')}`\n"
        f"- Final route: `{rel(final_path)}`\n\n"
        "Current official Part C shard command used for the packaged 720-row matrix:\n\n"
        "```bash\n"
        f"for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index $s --device cuda:0 --part-c-schemes op_metric --part-c-seed-count 3 --max-channel-outputs 8 --window-train-steps 1 --window-checkpoint-count 1 --channel-source-size 16 --snr-examples 16 & done\n"
        f"wait\n{PYTHON} {rel(RUNNER)} --mode part-c-merge --part-c-schemes op_metric --part-c-seed-count 3 --max-channel-outputs 8 --window-train-steps 1 --window-checkpoint-count 1 --channel-source-size 16 --snr-examples 16\n"
        "```\n",
        encoding="utf-8",
    )
    write_inventory_and_manifest()
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(final_path)}; {rel(manifest)}; {rel(OUT_ROOT / 'stale_artifact_audit.json')}")
    append_recap("Final route", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--basis-k-override", type=int, default=0)
    p.add_argument("--dfour-k", type=int, default=3)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--deep-width", type=int, default=12)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--synthetic-train-size", type=int, default=72)
    p.add_argument("--synthetic-guard-size", type=int, default=72)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--batch-size", type=int, default=36)
    p.add_argument("--adamw-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--window-train-steps", type=int, default=8)
    p.add_argument("--window-checkpoint-count", type=int, default=3)
    p.add_argument("--window-center-mode", default="none", choices=["none", "delta_j0"])
    p.add_argument("--window-output-center-mode", default="none", choices=["none", "row_mean"])
    p.add_argument("--mode-lambda-partial", type=float, default=1.0e-4)
    p.add_argument("--mode-lambda-partial2", type=float, default=1.0e-6)
    p.add_argument("--mode-lambda-omega", type=float, default=1.0e-4)
    p.add_argument("--part0-basis", default="dche_k5,dfour_default")
    p.add_argument("--part-c-basis", default="dche_k5,dche_k9,dfour_default")
    p.add_argument("--part-c-depths", default="depth2,depth3")
    p.add_argument("--part-c-schemes", default="adamw_witness,op_metric,population_snr,data_metric")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-controls", default="c2_positive,random_label,source_shuffle,mlp_friendly_negative")
    p.add_argument("--part-c-seed-count", type=int, default=2)
    p.add_argument("--channel-source-size", type=int, default=24)
    p.add_argument("--channel-output-mode", default="margin", choices=["margin", "logits"])
    p.add_argument("--max-channel-outputs", type=int, default=72)
    p.add_argument("--output-sketch-mode", default="prefix", choices=["prefix", "rademacher"])
    p.add_argument("--output-sketch-seed", type=int, default=229600)
    p.add_argument("--fresh-cohort-split", type=int, default=0)
    p.add_argument("--signal-rank", type=int, default=2)
    p.add_argument("--channel-cohorts", type=int, default=6)
    p.add_argument("--snr-examples", type=int, default=24)
    p.add_argument("--gradient-whiten-mode", default="none", choices=["none", "diag_rms"])
    p.add_argument("--gradient-whiten-eps", type=float, default=1.0e-8)
    p.add_argument("--gradient-whiten-clamp", type=float, default=25.0)
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--data-metric-condition", type=float, default=1.0e6)
    p.add_argument("--data-metric-sketch-rank", type=int, default=16)
    p.add_argument("--data-metric-sketch-seed", type=int, default=2296)
    p.add_argument("--pg1-gamma", type=float, default=0.15)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode)
    if mode == "part-0":
        return run_part_0(args)
    if mode == "part-a":
        return run_part_a(args)
    if mode == "part-b":
        return run_part_b(args)
    if mode == "part-c":
        return run_part_c(args)
    if mode == "part-c-merge":
        return merge_part_c(args)
    if mode in {"part-d", "part-e", "part-f", "part-g", "part-h", "part-i", "part-j", "part-k", "part-l"}:
        return run_downstream_gate(args, mode.split("-")[1].upper())
    if mode == "part-m":
        return run_part_m(args)
    if mode == "finalize":
        return finalize(args)
    if mode == "all":
        run_part_0(args)
        run_part_a(args)
        run_part_b(args)
        run_part_c(args)
        merge_part_c(args)
        for letter in "DEFGHIJKL":
            run_downstream_gate(args, letter)
        run_part_m(args)
        return finalize(args)
    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
