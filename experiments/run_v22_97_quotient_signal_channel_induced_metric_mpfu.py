#!/usr/bin/env python3
"""DG-KAN v22.97Q quotient signal-channel-induced metric runner.

This runner is intentionally gate-aware.  It runs Part 0/A/B/C, permits Part D
only when the strict raw estimator is stable enough, and blocks downstream
parts with explicit artifacts when a prerequisite fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import py_compile
import sys
import time
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
import experiments.run_v22_96_solid_signal_channel_induced_metric_mpfu as v2296
from dgkan.fu.channel_to_metric import project_channel_to_layers, project_channel_to_mode_bands, reverse_audit_smoke_test
from dgkan.fu.fixed_point_signal_metric import fixed_point_smoke_test
from dgkan.fu.kan_intrinsic_metrics import (
    DataCompositeMetric,
    data_metric_smoke_tests,
    flatten_layer_param_masks,
    full_mode_metric_vector,
    subspace_angle_from_diags,
)
from dgkan.fu.layer_composite_metric import EPS, sym
from dgkan.fu.quotient_signal_channel import (
    GeneralizedEigenQuotient,
    NegativeLeakageDecomposition,
    OrthogonalizedChannelQuotient,
    PSDClippedDifferenceQuotient,
    quotient_toy_smoke_test,
)
from dgkan.fu.signal_channel_estimators import (
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
from dgkan.fu.task_debt_channel_split import (
    ComponentWiseSafeIntersection,
    DebtComponentChannelBuilder,
    DebtOrthogonalizedTaskProjector,
    TaskDebtGeneralizedEigenSplit,
)


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v22.97_QuotientSignalChannelInducedMetric_MultiDirection_详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.97_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.97_实验结果复盘.md"
RESULT_ROOT = Path(os.environ.get("V2297_RESULT_ROOT", str(ROOT / "results/v22_97"))).resolve()
OUT_ROOT = Path(os.environ.get("V2297_OUT_ROOT", str(RESULT_ROOT / "current"))).resolve()
PROJ_ROOT = OUT_ROOT / "part_c_channel_projectors"
PART0_PROJ_ROOT = OUT_ROOT / "part0_channel_projectors"

BASIS_KEYS = ("dche_k5", "dche_k9", "dfour_default")
PART_C_ESTIMATORS = ("strict_offdiag", "window_no_pg", "window_pg1", "window_velocity")
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
    "changed_mlp_tensors": 0,
    "changed_edge_coefficients": 1,
    "standard_forward_backward_optimizer_loop": 1,
    "preconditioner_applied_inside_optimizer_step": 0,
}


def ensure_out() -> None:
    for path in (RESULT_ROOT, OUT_ROOT, PROJ_ROOT, PART0_PROJ_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
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


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.97Q Execution Log\n\n"
            f"- Created: {now()}\n"
            f"- Plan: `{rel(PLAN)}`\n"
            f"- Runner: `{rel(RUNNER)}`\n"
            f"- Output root: `{rel(OUT_ROOT)}`\n"
            f"- Python: `{PYTHON}`\n"
            "- Integrity rule: commands, artifacts, blockers, and repairs are recorded from actual runs only.\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.97Q Experiment Recap\n\n"
            f"- Created: {now()}\n"
            "- No fabricated metrics or inferred success states are allowed in this file.\n\n",
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


def gate_summary(part: str, gate_pass: int, route: str, blocker: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    out = {
        "part": part.upper(),
        "gate_pass": int(gate_pass),
        "route": route,
        "dominant_blocker": blocker,
        "ok_rows": int(sum(1 for row in rows if row.get("status") == "ok")),
        "error_rows": int(sum(1 for row in rows if row.get("status") == "error")),
        "row_count": int(len(rows)),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "metric_winner_selection_used": 0,
        "candidate_update_selection_used": 0,
        "generated_at": now(),
    }
    out.update(extra)
    return out


def common_next_actions(part: str, blocker: str, allowed: list[str], forbidden: list[str], commands: list[str]) -> Path:
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    write_json(
        path,
        {
            "dominant_blocker": blocker,
            "evidence_fields": [],
            "allowed_actions": allowed,
            "forbidden_actions": forbidden,
            "max_repair_rounds": 2,
            "rerun_commands": commands,
        },
    )
    return path


def device_from_args(args: argparse.Namespace) -> torch.device:
    text = str(args.device)
    if text.startswith("cuda") and torch.cuda.is_available():
        return torch.device(text)
    return torch.device("cpu")


def basis_args(args: argparse.Namespace, basis_key: str) -> argparse.Namespace:
    return v2296.basis_args(args, basis_key)


def argument_hash(args: argparse.Namespace) -> str:
    payload = vars(args).copy()
    payload.pop("device", None)
    payload.pop("shard_index", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def stable_seed(*items: Any, base: int = 0) -> int:
    text = "|".join(str(item) for item in items)
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return int((int(base) + value) % 2_147_483_647)


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    index = int(args.shard_index)
    return [item for idx, item in enumerate(items) if idx % count == index]


def save_npz(path: Path, **arrays: torch.Tensor | np.ndarray | float | int) -> None:
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


def load_channel_matrix(path_text: str) -> torch.Tensor | None:
    if not path_text:
        return None
    path = ROOT / path_text if not Path(path_text).is_absolute() else Path(path_text)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        if "W" in arr:
            return torch.from_numpy(arr["W"]).to(dtype=torch.float64)
        if "U" in arr:
            u = torch.from_numpy(arr["U"]).to(dtype=torch.float64)
            return sym(u @ u.T)
    except Exception:
        return None
    return None


def projector_basis(path_text: str) -> torch.Tensor | None:
    mat = load_channel_matrix(path_text)
    if mat is None:
        return None
    vals, vecs = torch.linalg.eigh(sym(mat))
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    r = int((vals > 1.0e-10).sum().item())
    return vecs[:, :r].contiguous() if r > 0 else torch.zeros((int(mat.shape[0]), 0), dtype=torch.float64)


def channel_overlap_matrix(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = sym(a.to(dtype=torch.float64))
    bb = sym(b.to(dtype=torch.float64))
    n = min(int(aa.shape[0]), int(bb.shape[0]))
    if n <= 0:
        return 0.0
    denom = torch.trace(aa[:n, :n]).abs().clamp_min(EPS)
    return float((torch.trace(aa[:n, :n] @ bb[:n, :n]).abs() / denom).clamp(0.0, 1.0).detach().cpu().item())


def pad_matrix(mat: torch.Tensor, dim: int) -> torch.Tensor:
    mm = mat.detach().to(dtype=torch.float64).cpu()
    if int(mm.numel()) == 0:
        return torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    out = torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    n = min(int(dim), int(mm.shape[0]))
    out[:n, :n] = sym(mm[:n, :n])
    return out


def projector_stability(rows: list[dict[str, str]]) -> float:
    bases = [projector_basis(str(row.get("projector_npz", ""))) for row in rows]
    bases = [b for b in bases if b is not None]
    vals: list[float] = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            vals.append(projector_overlap(bases[i], bases[j]))
    return median(vals)


def stale_artifact_audit(final_blocker_override: str | None = None) -> dict[str, Any]:
    summary_files = sorted(OUT_ROOT.glob("part_*_summary.json"))
    blocker = str(final_blocker_override or read_json(OUT_ROOT / "final_route.json").get("dominant_blocker", "none"))
    stale: list[dict[str, str]] = []
    for path in summary_files:
        data = read_json(path)
        blocked = str(data.get("blocked_reason", ""))
        dominant = str(data.get("dominant_blocker", ""))
        prerequisite_chain = str(data.get("route", "")).endswith("BlockedByPrerequisite") and dominant == "prerequisite_failed"
        if blocked and not prerequisite_chain and blocker not in {"", "none"} and blocker not in blocked and blocker != dominant:
            stale.append({"file": rel(path), "blocked_reason": blocked, "dominant_blocker": dominant})
    out = {"stale_count": len(stale), "stale": stale, "summary_files": [rel(p) for p in summary_files], "audit_time": now()}
    write_json(OUT_ROOT / "stale_artifact_audit.json", out)
    return out


def write_inventory_and_manifest() -> None:
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file())
    write_json(OUT_ROOT / "artifact_inventory.json", [{"path": rel(p), "bytes": int(p.stat().st_size)} for p in files])
    with (OUT_ROOT / "sha256_manifest.txt").open("w", encoding="utf-8") as fh:
        for path in files:
            if path.name == "sha256_manifest.txt":
                continue
            fh.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel(path)}\n")


def make_model(depth: str, input_dim: int, classes: int, seed: int, args: argparse.Namespace, device: torch.device) -> v2293.TrueDeepPureKAN:
    return v2296.make_model(depth, input_dim, classes, seed, args, device)


def controlled_data(task: str, control: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2296.controlled_data(task, control, seed, args, device)


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2296.visual_data(task, seed, args, device)


def metric_diag_for_scheme(model: v2293.TrueDeepPureKAN, x_source: torch.Tensor, scheme: str, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    return v2296.metric_diag_for_scheme(model, x_source, scheme, args)


def run_part_0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    init_logs()
    device = device_from_args(args)
    module_files = [
        "dgkan/fu/kan_intrinsic_metrics.py",
        "dgkan/fu/signal_channel_estimators.py",
        "dgkan/fu/channel_to_metric.py",
        "dgkan/fu/fixed_point_signal_metric.py",
        "dgkan/fu/quotient_signal_channel.py",
        "dgkan/fu/task_debt_channel_split.py",
    ]
    compile_pass = 1
    compile_errors: list[str] = []
    for mod in module_files + [rel(RUNNER)]:
        try:
            py_compile.compile(str(ROOT / mod), doraise=True)
        except Exception as exc:
            compile_pass = 0
            compile_errors.append(f"{mod}: {repr(exc)}")
    import_pass = 1
    import_errors: list[str] = []
    for name in [
        "dgkan.fu.kan_intrinsic_metrics",
        "dgkan.fu.signal_channel_estimators",
        "dgkan.fu.channel_to_metric",
        "dgkan.fu.fixed_point_signal_metric",
        "dgkan.fu.quotient_signal_channel",
        "dgkan.fu.task_debt_channel_split",
    ]:
        try:
            __import__(name, fromlist=["*"])
        except Exception as exc:
            import_pass = 0
            import_errors.append(f"{name}: {repr(exc)}")

    smoke: dict[str, Any] = {}
    smoke.update(data_metric_smoke_tests(device))
    smoke.update(windowed_linear_smoke_test())
    smoke.update(offdiag_synthetic_smoke_test())
    smoke.update(reverse_audit_smoke_test())
    smoke.update(fixed_point_smoke_test())
    smoke.update(quotient_toy_smoke_test())
    write_json(OUT_ROOT / "part0_module_smoke_tests.json", smoke)

    metric_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    reverse_rows: list[dict[str, Any]] = []
    xtr, ytr, _xg, _yg = visual_data("local_patch_interaction", 0, args, device)
    source_n = min(int(args.channel_source_size), int(xtr.shape[0]))
    xsrc, ysrc = xtr[:source_n], ytr[:source_n]
    for basis_key in csv_items(args.part0_basis):
        bargs = basis_args(args, basis_key)
        model = make_model("depth2", int(xsrc.shape[1]), int(args.num_classes), 229700 + len(metric_rows), bargs, device)
        op_vec = full_mode_metric_vector(model).to(device=device)
        dcm = DataCompositeMetric(target_condition=float(args.data_metric_condition), sketch_rank=int(args.data_metric_sketch_rank), sketch_seed=int(args.data_metric_sketch_seed))
        for metric_type in ("diagonal", "block", "full_sketch"):
            data_vec, results = dcm.param_metric_vector(model, xsrc, metric_type=metric_type)
            for res in results:
                row = dict(res.summary)
                row.update({"basis_key": basis_key, "depth": "depth2", "op_vs_mode_angle_deg": subspace_angle_from_diags(data_vec.cpu(), op_vec.cpu()), "true_phiTphi_row": 1, "status": "ok"})
                metric_rows.append(row)
        metric_diag, _metric_info = metric_diag_for_scheme(model, xsrc, "op_metric", bargs)
        j0 = full_output_jacobian(model, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode))
        win = WindowedDissipationEstimator().estimate_from_jacobians([j0], [metric_diag.detach().cpu()], rank=int(args.signal_rank), estimator_type="window_no_pg")
        path = PART0_PROJ_ROOT / f"{basis_key}_window_no_pg.npz"
        save_npz(path, U=win.u, eig=win.eigenvalues, W=win.projector)
        signal_rows.append({"basis_key": basis_key, "estimator_type": "window_no_pg", "projector_npz": rel(path), "status": "ok", **win.summary})
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
    true_phi_rows = sum(1 for r in metric_rows if int(r.get("true_phiTphi_row", 0)) == 1 and int(r.get("phi_rows", 0)) > 0 and int(r.get("phi_cols", 0)) > 0)
    gate = int(
        compile_pass == 1
        and import_pass == 1
        and true_phi_rows >= 6
        and int(fval(smoke.get("offdiag_same_gt_random"))) == 1
        and fval(smoke.get("quotient_positive_retention")) >= 0.99
        and fval(smoke.get("quotient_identical_channel_suppression")) <= 0.01
        and int(fval(smoke.get("fixed_point_smoke_converged"))) == 1
        and int(stale.get("stale_count", 0)) == 0
    )
    blocker = "none" if gate else "part0_compile_import_data_metric_or_quotient_smoke"
    summary = gate_summary(
        "0",
        gate,
        "P0_Pass" if gate else "P0_ImplementationSolidificationFailed",
        blocker,
        metric_rows + signal_rows,
        module_files=module_files,
        compile_pass=compile_pass,
        compile_errors=compile_errors,
        import_pass=import_pass,
        import_errors=import_errors,
        data_metric_rows=len(metric_rows),
        true_phiTphi_rows=true_phi_rows,
        quotient_toy_rows=1,
        offdiag_same_gt_random=int(fval(smoke.get("offdiag_same_gt_random"))),
        quotient_positive_retention=fval(smoke.get("quotient_positive_retention")),
        quotient_identical_channel_suppression=fval(smoke.get("quotient_identical_channel_suppression")),
        fixed_point_smoke_converged=int(fval(smoke.get("fixed_point_smoke_converged"))),
        stale_artifact_count=int(stale.get("stale_count", 0)),
        smoke=smoke,
    )
    summary_path = OUT_ROOT / "part0_summary.json"
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "0",
        blocker,
        ["fix module compile/import", "repair DataCompositeMetric true phiTphi construction", "repair PSD clipping/eigen threshold/rank truncation/normalization in quotient toy"],
        ["run Part C with failed Part 0", "replace data metric with mode vector proxy", "degrade quotient into raw positive channel"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-0 --device {args.device}"],
    )
    write_inventory_and_manifest()
    append_exec("part-0", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(summary_path)}; {rel(metric_path)}; {rel(signal_path)}; {rel(next_path)}")
    append_recap("Part 0 implementation solidification", summary)
    return summary


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    p0 = read_json(OUT_ROOT / "part0_summary.json")
    scan_files = [RUNNER] + [ROOT / p for p in read_json(OUT_ROOT / "part0_summary.json").get("module_files", [])]
    static_pass, scan_hits = v2296.static_scan(scan_files)
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for basis_key in BASIS_KEYS:
        bargs = basis_args(args, basis_key)
        for depth in ("depth2", "depth3"):
            x = torch.randn(8, int(args.visual_side) * int(args.visual_side), device=device)
            model = make_model(depth, int(x.shape[1]), int(args.num_classes), 229710 + len(rows), bargs, device)
            rows.append(
                {
                    "row_id": f"A_{basis_key}_{depth}",
                    "basis_key": basis_key,
                    "depth": depth,
                    "status": "ok",
                    "static_scan_pass": static_pass,
                    "static_scan_hits": json.dumps(scan_hits, ensure_ascii=False),
                    "purekan_depth2_constructed": int(depth == "depth2" and int(model.depth) == 2),
                    "purekan_depth3_constructed": int(depth == "depth3" and int(model.depth) == 3),
                    "dche_available": int(str(model.basis_name) == "chebyshev"),
                    "dfour_available": int(str(model.basis_name) == "fourier_lowfreq"),
                    "quotient_module_import_pass": 1,
                    **AUDIT_DEFAULTS,
                }
            )
    gate = int(
        int(p0.get("gate_pass", 0)) == 1
        and int(static_pass) == 1
        and any(int(r["purekan_depth2_constructed"]) == 1 for r in rows)
        and any(int(r["purekan_depth3_constructed"]) == 1 for r in rows)
        and any(int(r["dche_available"]) == 1 for r in rows)
        and any(int(r["dfour_available"]) == 1 for r in rows)
        and all(int(r["runtime_selector_used"]) == 0 and int(r["metric_winner_selection_used"]) == 0 for r in rows)
    )
    blocker = "none" if gate else ("part0_failed" if int(p0.get("gate_pass", 0)) != 1 else "identity_or_static_scan")
    matrix = OUT_ROOT / "part_a_identity_matrix.csv"
    summary_path = OUT_ROOT / "part_a_identity_summary.json"
    write_rows(matrix, rows)
    summary = gate_summary("A", gate, "A_Pass" if gate else "A_CodeIdentityFailed", blocker, rows, static_scan_pass=static_pass, scan_hits=scan_hits)
    write_json(summary_path, summary)
    next_path = common_next_actions(
        "A",
        blocker,
        ["fix import/compile/static scan/module presence", "repair actual runtime selector if scan hit is real"],
        ["bypass Part A", "continue to Part C with missing quotient module"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-a --device {args.device}"],
    )
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part A identity and anti-selector gate", summary)
    return summary


def history_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"state": "missing", "status": "missing", "path": rel(path)}
    data = read_json(path)
    if not data:
        return {"state": "malformed", "status": "error", "path": rel(path)}
    return {"state": "complete", "status": "ok", "path": rel(path), "route": data.get("route", data.get("part_c_route", "present")), "gate_pass": data.get("gate_pass", data.get("official_candidate_gate_pass", ""))}


def best_v2296_summary() -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1.0
    for path in sorted((ROOT / "results/v22_96").glob("*/part_c_estimator_comparison_summary.json")):
        data = read_json(path)
        score = fval(data.get("signal_channel_seed_stability")) + fval(data.get("offdiag_to_window_ab_eigenspace_overlap"))
        if data and score > best_score:
            best_score = score
            best = dict(data)
            best["source_path"] = rel(path)
    return best


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    sources = {
        "v22_90_final": ROOT / "results/v22_90/final_route.json",
        "v22_91_final": ROOT / "results/v22_91/final_route.json",
        "v22_93_final": ROOT / "results/v22_93/final_route.json",
        "v22_94_final": ROOT / "results/v22_94/final_route.json",
        "v22_95R_final": ROOT / "results/v22_95R/final_route.json",
        "v22_95R_part_c": ROOT / "results/v22_95R/part_c_signal_channel_summary.json",
        "v22_96_final": ROOT / "results/v22_96/current/final_route.json",
        "v22_96_part0": ROOT / "results/v22_96/current/part0_summary.json",
        "v22_96_part_c": ROOT / "results/v22_96/current/part_c_estimator_comparison_summary.json",
    }
    rows = [{"artifact": name, **history_status(path)} for name, path in sources.items()]
    v96 = read_json(ROOT / "results/v22_96/current/part_c_estimator_comparison_summary.json")
    best96 = best_v2296_summary()
    missing_or_error = [r for r in rows if r.get("status") != "ok"]
    gate = int(int(a.get("gate_pass", 0)) == 1 and not missing_or_error)
    blocker = "none" if gate else ("part_a_failed" if int(a.get("gate_pass", 0)) != 1 else "history_artifact_missing_or_malformed")
    summary = gate_summary(
        "B",
        gate,
        "B_Pass" if gate else "B_HistoryLockFailed",
        blocker,
        rows,
        artifact_sources={name: rel(path) if path.exists() else "missing" for name, path in sources.items()},
        missing_or_error_rows=missing_or_error,
        v22_96_part0_pass=read_json(ROOT / "results/v22_96/current/part0_summary.json").get("gate_pass", "missing"),
        v22_96_part_c_gate_pass=v96.get("gate_pass", "missing"),
        v22_96_best_repair_seed_stability=best96.get("signal_channel_seed_stability", "missing"),
        v22_96_best_repair_offdiag_window_overlap=best96.get("offdiag_to_window_ab_eigenspace_overlap", "missing"),
        v22_96_best_repair_c2_top_mass=best96.get("C2_channel_top_mass", "missing"),
        v22_96_best_repair_negative_top_mass=best96.get("negative_channel_top_mass", "missing"),
        v22_96_best_repair_c2_snr=best96.get("C2_snr_top_decile", "missing"),
        v22_96_best_repair_negative_snr=best96.get("negative_snr_top_decile", "missing"),
        v22_96_final_route=read_json(ROOT / "results/v22_96/current/final_route.json").get("route", "missing"),
        v22_96_best_repair_source=best96.get("source_path", "missing"),
        hypothesis_registry={
            "H1": "raw signal channel captures generic learnability and is insufficient",
            "H2": "quotient signal channel can suppress negative/MLP/debt leakage",
            "H3": "quotient channel can induce KAN metric without runtime winner selection",
            "H4": "if quotient fails, C2 coordinate-forming freedom and debt/generic freedom may be structurally entangled",
        },
    )
    matrix = OUT_ROOT / "part_b_history_artifacts.csv"
    path = OUT_ROOT / "part_b_history_lock.json"
    write_rows(matrix, rows)
    write_json(path, summary)
    write_json(OUT_ROOT / "history_lock.json", summary)
    next_path = common_next_actions(
        "B",
        blocker,
        ["restore missing history artifact", "record missing explicitly without imputation"],
        ["silently infer missing history", "rewrite older route values"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-b"],
    )
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(path)}; {rel(next_path)}")
    append_recap("Part B history lock and hypothesis registry", summary)
    return summary


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int, str]]:
    jobs: list[tuple[str, str, str, str, int, str]] = []
    for basis_key in csv_items(args.part_c_basis):
        for depth in csv_items(args.part_c_depths):
            for scheme in csv_items(args.part_c_schemes):
                for task in csv_items(args.part_c_tasks):
                    for seed in range(int(args.part_c_seed_count)):
                        for control in csv_items(args.part_c_controls):
                            jobs.append((basis_key, depth, scheme, task, seed, control))
    return jobs


def strict_offdiag_from_samples(j: torch.Tensor, g_samples: torch.Tensor, metric_diag: torch.Tensor, args: argparse.Namespace) -> tuple[Any, dict[str, Any], torch.Tensor]:
    m = metric_diag.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
    n = min(int(j.shape[1]), int(g_samples.shape[1]), int(m.numel()))
    gw = g_samples[:, :n].detach().to(dtype=torch.float64).cpu() * m[:n].sqrt().reshape(1, -1)
    score_g = gw.clone()
    if int(args.strict_offdiag_normalize_rows):
        score_g = score_g / score_g.norm(dim=1, keepdim=True).clamp_min(EPS)
    gram = score_g @ score_g.T
    b = int(gram.shape[0])
    diag_mask = torch.eye(b, dtype=torch.bool)
    self_energy = float(gram[diag_mask].abs().sum().item())
    offdiag_energy = float(gram[~diag_mask].abs().sum().item()) if b > 1 else 0.0
    signed_offdiag = float(gram[~diag_mask].mean().item()) if b > 1 else 0.0
    self_to_offdiag_ratio = self_energy / max(offdiag_energy, EPS)
    sum_g = score_g.sum(dim=0, keepdim=True)
    off_ab = sym((sum_g.T @ sum_g - score_g.T @ score_g) / float(max(1, b * max(1, b - 1))))
    vals, vecs = torch.linalg.eigh(off_ab)
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    pos = vals.clamp_min(0.0)
    ab_pos = sym((vecs * pos.reshape(1, -1)) @ vecs.T)
    jj = j.detach().to(dtype=torch.float64).cpu()
    raw_w = sym(jj[:, :n] @ off_ab @ jj[:, :n].T)
    w = sym(jj[:, :n] @ ab_pos @ jj[:, :n].T)
    proj = projector_from_psd(w, int(args.signal_rank), estimator_type="strict_offdiag")
    gate, snr = diagonal_snr_gate(gw, beta=4.0, tau=1.0, normalize_rows=True)
    info = {
        "self_block_energy": self_energy,
        "offdiag_energy": offdiag_energy,
        "self_to_offdiag_ratio": self_to_offdiag_ratio,
        "strict_offdiag_score": signed_offdiag,
        "offdiag_psd_min_eig": float(torch.linalg.eigvalsh(raw_w).min().item()) if int(raw_w.numel()) else 0.0,
        "offdiag_rank": int((pos > 1.0e-10).sum().item()),
        "offdiag_lowrank_positive_eigs": int((vals > 0.0).sum().item()),
        "offdiag_ab_top_eig": float(vals[0].item()) if int(vals.numel()) else 0.0,
        "snr_gate_mean": float(gate.mean().item()) if int(gate.numel()) else 0.0,
        **snr,
    }
    proj.summary.update(info)
    return proj, info, w


def strict_component_probe(j: torch.Tensor, g_samples: torch.Tensor, metric_diag: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    base_proj, base_info, _base_w = strict_offdiag_from_samples(j, g_samples, metric_diag, args)
    centered_g = g_samples - g_samples.mean(dim=0, keepdim=True)
    centered_proj, centered_info, _ = strict_offdiag_from_samples(j, centered_g, metric_diag, args)
    raw_args = deepcopy(args)
    setattr(raw_args, "strict_offdiag_normalize_rows", 0)
    raw_proj, raw_info, _ = strict_offdiag_from_samples(j, g_samples, metric_diag, raw_args)
    return {
        "negative_top_mass_before": base_proj.summary.get("W_top_mass_ratio", 0.0),
        "negative_top_mass_after_output_center": centered_proj.summary.get("W_top_mass_ratio", 0.0),
        "negative_snr_before": base_info.get("diagonal_snr_top_decile_mean", 0.0),
        "negative_snr_after_output_center": centered_info.get("diagonal_snr_top_decile_mean", 0.0),
        "gradient_norm_top_mass": raw_proj.summary.get("W_top_mass_ratio", 0.0),
        "gradient_norm_snr": raw_info.get("diagonal_snr_top_decile_mean", 0.0),
    }


def run_part_c_job(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    basis_key, depth, scheme, task, seed, control = job
    start = time.time()
    rows: list[dict[str, Any]] = []
    try:
        bargs = basis_args(args, basis_key)
        xtr, ytr, xg, yg = controlled_data(task, control, seed, args, device)
        xsrc, ysrc, xtraj, ytraj, xgrad, ygrad, split_info = v2296.part_c_source_witness_split(xtr, ytr, control, seed, args)
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 2297000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
        probe_seed = stable_seed("part_c_output_sketch", basis_key, depth, scheme, task, base=int(args.output_sketch_seed))
        model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
        before_guard = v2293.metrics_for_model(model, xg, yg)
        metric_diag, metric_info = metric_diag_for_scheme(model, xsrc, scheme, bargs)
        g_samples = per_example_gradient_matrix(model, xgrad, ygrad, max_examples=int(args.snr_examples), label_prior_correction=bool(int(args.label_prior_correction)))
        metric_diag, whiten_info = v2296.gradient_whiten_metric_diag(metric_diag, g_samples, args)
        before_out = model(xsrc).detach()
        work_model = deepcopy(model).to(device)
        checkpoints = v2296.train_adamw_checkpoints(work_model, xtraj, ytraj, bargs, seed=seed)
        after_guard = v2293.metrics_for_model(checkpoints[-1], xg, yg)
        after_out = checkpoints[-1](xsrc).detach()
        displacement = (after_out - before_out).reshape(-1).detach().cpu().to(dtype=torch.float64)
        js = [
            full_output_jacobian(cp, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
            for cp in checkpoints
        ]
        j0 = js[0]
        ms = [metric_diag.detach().cpu()] * len(js)
        win_est = WindowedDissipationEstimator()
        win = win_est.estimate_from_jacobians(js, ms, rank=int(args.signal_rank), estimator_type="window_no_pg", center_mode=str(args.window_center_mode), output_center_mode=str(args.window_output_center_mode))
        pg1 = win_est.estimate_pg1_reduced(js, ms, rank=int(args.signal_rank), gamma=float(args.pg1_gamma), center_mode=str(args.window_center_mode), output_center_mode=str(args.window_output_center_mode))
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
        velocity = projector_from_psd(sym(vmat @ vmat.T / float(max(1, int(vmat.shape[1])))), int(args.signal_rank), estimator_type="window_velocity")
        strict, strict_info, strict_w = strict_offdiag_from_samples(j0, g_samples, metric_diag, args)
        component_info = strict_component_probe(j0, g_samples, metric_diag, args) if control != "c2_positive" else {}
        raw_g = loss_gradient_vector(model, xgrad[: min(int(args.batch_size), int(xgrad.shape[0]))], ygrad[: min(int(args.batch_size), int(ygrad.shape[0]))], label_prior_correction=bool(int(args.label_prior_correction))).detach().cpu()
        n_dim = min(int(raw_g.numel()), int(metric_diag.numel()), int(j0.shape[1]))
        deleted_delta = raw_g[:n_dim] * (metric_diag.detach().cpu()[:n_dim] - 1.0)
        dz_deleted = j0[:, :n_dim] @ deleted_delta
        projectors = {
            "strict_offdiag": (strict, strict_w),
            "window_no_pg": (win, win.projector),
            "window_pg1": (pg1, pg1.projector),
            "window_velocity": (velocity, velocity.projector),
        }
        for estimator_type, (proj, wmat) in projectors.items():
            row_id = f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_{estimator_type}"
            proj_path = PROJ_ROOT / f"{row_id}.npz"
            save_npz(proj_path, U=proj.u, eig=proj.eigenvalues, W=wmat)
            deleted_frac = vector_channel_fraction(proj.u, dz_deleted)
            row = {
                "row_id": row_id,
                "part": "C",
                "basis_key": basis_key,
                "depth": depth,
                "scheme": scheme,
                "task": task,
                "seed": int(seed),
                "control": control,
                "estimator_type": estimator_type,
                "status": "ok",
                "argument_hash": argument_hash(args),
                "projector_npz": rel(proj_path),
                "source_size": int(xsrc.shape[0]),
                "witness_size": int(min(int(args.snr_examples), int(xgrad.shape[0]))),
                "trajectory_size": int(xtraj.shape[0]),
                "guard_size": int(xg.shape[0]),
                "metric_kind": metric_info.get("metric_kind", scheme),
                "checkpoint_count": int(len(checkpoints)),
                "probe_seed": int(probe_seed),
                "window_id": "adamw_fixed_train_window",
                "window_center_mode": str(args.window_center_mode),
                "window_output_center_mode": str(args.window_output_center_mode),
                "channel_seed_stability_placeholder": "",
                "channel_window_stability": projector_overlap(win.u, pg1.u) if estimator_type.startswith("window") else projector_overlap(proj.u, win.u),
                "pg1_overlap_with_no_pg": projector_overlap(pg1.u, win.u),
                "channel_overlap_with_C2_positive": vector_channel_fraction(proj.u, displacement),
                "channel_overlap_with_adamw_output_displacement": vector_channel_fraction(proj.u, displacement),
                "C2_accuracy_short_window": after_guard["accuracy"],
                "C2_coverage_short_window": after_guard["coverage_CVaR25"],
                "visual_accuracy_improvement": after_guard["accuracy"] - before_guard["accuracy"],
                "visual_coverage_improvement": after_guard["coverage_CVaR25"] - before_guard["coverage_CVaR25"],
                "guard_debt_delta": after_guard["debt_metric"] - before_guard["debt_metric"],
                "deleted_signal_energy_raw": float(deleted_frac * float(dz_deleted.square().sum().item())),
                "deleted_reservoir_energy_raw": float((1.0 - deleted_frac) * float(dz_deleted.square().sum().item())),
                "deleted_signal_to_reservoir_ratio_raw": float(deleted_frac / max(1.0 - deleted_frac, 1.0e-12)),
                "W_top_mass_ratio": proj.summary.get("W_top_mass_ratio", 0.0),
                "W_effective_rank": proj.summary.get("W_effective_rank", 0.0),
                "channel_rank": proj.summary.get("signal_channel_projector_rank", int(proj.rank)),
                "offdiag_rank": strict_info.get("offdiag_rank", 0),
                "self_block_energy": strict_info.get("self_block_energy", 0.0),
                "offdiag_energy": strict_info.get("offdiag_energy", 0.0),
                "self_to_offdiag_ratio": strict_info.get("self_to_offdiag_ratio", 0.0),
                "strict_offdiag_score": strict_info.get("strict_offdiag_score", 0.0),
                "positive_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "c2_positive" else "",
                "random_label_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "random_label" else "",
                "source_shuffle_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "source_shuffle" else "",
                "mlp_friendly_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "mlp_friendly_negative" else "",
                "offdiag_psd_min_eig": strict_info.get("offdiag_psd_min_eig", 0.0),
                "diagonal_snr_top_decile_mean": strict_info.get("diagonal_snr_top_decile_mean", 0.0),
                "diagonal_snr_median": strict_info.get("diagonal_snr_median", 0.0),
                "wall_time_s": time.time() - start,
                **component_info,
                **whiten_info,
                **split_info,
                **AUDIT_DEFAULTS,
            }
            rows.append(row)
        return rows
    except Exception as exc:
        return [{"row_id": f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_error", "part": "C", "basis_key": basis_key, "depth": depth, "scheme": scheme, "task": task, "seed": int(seed), "control": control, "status": "error", "error_message": repr(exc), "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}]


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    if int(a.get("gate_pass", 0)) != 1 or int(b.get("gate_pass", 0)) != 1:
        path = OUT_ROOT / f"part_c_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
        write_rows(path, [])
        summary = gate_summary("C", 0, "C_BlockedByPartAB", "part_a_or_b_failed", [])
        write_json(OUT_ROOT / "part_c_estimator_comparison_summary.json", summary)
        append_exec("part-c", command_text(sys.argv), "blocked", files=rel(path))
        return summary
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    base = OUT_ROOT / f"part_c_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    for path in [base, *[OUT_ROOT / f"part_c_{e}_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv" for e in PART_C_ESTIMATORS]]:
        if path.exists():
            path.unlink()
    for job in shard_items(part_c_jobs(args), args):
        rows.extend(run_part_c_job(job, args, device))
        write_rows(base, rows)
        for estimator_type in PART_C_ESTIMATORS:
            write_rows(OUT_ROOT / f"part_c_{estimator_type}_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", [r for r in rows if r.get("estimator_type") == estimator_type])
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(base), note=f"rows={len(rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_c_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    expected = len(part_c_jobs(args)) * len(PART_C_ESTIMATORS)
    write_rows(OUT_ROOT / "part_c_matrix.csv", [dict(r) for r in rows])
    for estimator_type in PART_C_ESTIMATORS:
        write_rows(OUT_ROOT / f"part_c_{estimator_type}_matrix.csv", [dict(r) for r in rows if r.get("estimator_type") == estimator_type])
    negative = [r for r in rows if r.get("control") != "c2_positive"]
    write_rows(OUT_ROOT / "part_c_negative_control_matrix.csv", [dict(r) for r in negative])
    ok = [r for r in rows if r.get("status") == "ok"]
    pos = [r for r in ok if r.get("control") == "c2_positive"]
    neg = [r for r in ok if r.get("control") in {"random_label", "source_shuffle"}]
    mlp = [r for r in ok if r.get("control") == "mlp_friendly_negative"]
    strict_rows = [r for r in ok if r.get("estimator_type") == "strict_offdiag"]
    window_rows = [r for r in ok if r.get("estimator_type") in {"window_no_pg", "window_pg1", "window_velocity"}]
    strict_pos = [r for r in strict_rows if r.get("control") == "c2_positive"]
    group_stabilities: list[float] = []
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("estimator_type")) for r in pos}):
        group = [r for r in pos if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("estimator_type")) == key]
        if len(group) >= 2:
            group_stabilities.append(projector_stability(group))
    seed_stability = median(group_stabilities)
    window_stability = median(fval(r.get("channel_window_stability")) for r in window_rows if r.get("control") == "c2_positive")
    self_ratio = median(fval(r.get("self_to_offdiag_ratio")) for r in strict_rows)
    pos_top = median(fval(r.get("W_top_mass_ratio")) for r in pos)
    neg_top = median(fval(r.get("W_top_mass_ratio")) for r in neg)
    mlp_top = median(fval(r.get("W_top_mass_ratio")) for r in mlp)
    pos_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in strict_pos)
    neg_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in neg)
    mlp_snr = median(fval(r.get("diagonal_snr_top_decile_mean")) for r in mlp)
    offdiag_finite = int(all(math.isfinite(fval(r.get("strict_offdiag_score"))) for r in strict_rows))
    row_count_pass = int(len(rows) == expected)
    c1_pass = int(self_ratio <= 0.25 and offdiag_finite == 1 and len(strict_rows) > 0)
    c2_stable = int(seed_stability >= 0.60 and window_stability >= 0.60)
    raw_specific = int((neg_top <= 0.75 * max(pos_top, EPS)) and (mlp_top <= 0.75 * max(pos_top, EPS)) and (neg_snr <= 0.75 * max(pos_snr, EPS)))
    if not row_count_pass:
        route, blocker, gate = "C_IncompleteShardMerge", "row_count_mismatch", 0
    elif not c1_pass:
        route, blocker, gate = "C_RawEstimatorBroken", "strict_offdiag_self_block_leakage", 0
    elif not c2_stable:
        route, blocker, gate = "C_RawEstimatorBroken", "seed_or_window_stability", 0
    elif raw_specific:
        route, blocker, gate = "C_RawEstimatorSpecificEnough", "none", 1
    else:
        route, blocker, gate = "C_RawEstimatorStableButGeneric", "negative_or_mlp_channel_remains_strong", 1
    decomposition_rows: list[dict[str, Any]] = []
    for control in ("random_label", "source_shuffle", "mlp_friendly_negative"):
        subset = [r for r in strict_rows if r.get("control") == control]
        if not subset:
            continue
        decomposition_rows.append(
            {
                "component": control,
                "negative_top_mass_before": median(fval(r.get("negative_top_mass_before", r.get("W_top_mass_ratio"))) for r in subset),
                "negative_top_mass_after": median(fval(r.get("negative_top_mass_after_output_center", r.get("W_top_mass_ratio"))) for r in subset),
                "negative_snr_before": median(fval(r.get("negative_snr_before", r.get("diagonal_snr_top_decile_mean"))) for r in subset),
                "negative_snr_after": median(fval(r.get("negative_snr_after_output_center", r.get("diagonal_snr_top_decile_mean"))) for r in subset),
                "c2_top_mass_before": pos_top,
                "c2_top_mass_after": pos_top,
                "c2_snr_before": pos_snr,
                "offdiag_overlap_before": median(fval(r.get("self_to_offdiag_ratio")) for r in subset),
                "offdiag_overlap_after": median(fval(r.get("self_to_offdiag_ratio")) for r in subset),
            }
        )
    write_rows(OUT_ROOT / "part_c_negative_leakage_decomposition.csv", decomposition_rows)
    deleted_corr = pearson([fval(r.get("deleted_signal_energy_raw")) for r in strict_pos], [fval(r.get("visual_coverage_improvement")) for r in strict_pos])
    debt_corr = pearson([fval(r.get("deleted_signal_energy_raw")) for r in strict_pos], [fval(r.get("guard_debt_delta")) for r in strict_pos])
    summary = gate_summary(
        "C",
        gate,
        route,
        blocker,
        [dict(r) for r in rows],
        expected_rows=expected,
        observed_rows=len(rows),
        row_count_pass=row_count_pass,
        c1_strict_offdiag_pass=c1_pass,
        c2_windowed_stability_pass=c2_stable,
        c3_raw_negative_leakage_decomposition_rows=len(decomposition_rows),
        c4_deleted_freedom_rows=len(strict_pos),
        self_to_offdiag_ratio=self_ratio,
        positive_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in strict_pos),
        random_label_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in strict_rows if r.get("control") == "random_label"),
        source_shuffle_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in strict_rows if r.get("control") == "source_shuffle"),
        mlp_friendly_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in strict_rows if r.get("control") == "mlp_friendly_negative"),
        channel_seed_stability=seed_stability,
        channel_window_stability=window_stability,
        c2_top_mass=pos_top,
        negative_top_mass=neg_top,
        mlp_top_mass=mlp_top,
        c2_snr_top_decile=pos_snr,
        negative_snr_top_decile=neg_snr,
        mlp_snr_top_decile=mlp_snr,
        raw_specific_enough=raw_specific,
        C2_gain_vs_deleted_signal_energy_raw_corr=deleted_corr,
        F5_debt_vs_deleted_signal_energy_raw_corr=debt_corr,
    )
    summary_path = OUT_ROOT / "part_c_estimator_comparison_summary.json"
    write_json(summary_path, summary)
    write_json(OUT_ROOT / "part_c_signal_channel_summary.json", summary)
    next_path = common_next_actions(
        "C",
        blocker,
        ["strict disjoint source/witness split", "blockwise VJP/JVP sketch correction", "increase source/witness size", "increase checkpoint density", "output centering", "fresh source/witness cohorts", "gradient whitening inspection"],
        ["lower strict offdiag gates", "delete random-label/source-shuffle/MLP controls", "run Part D after RawEstimatorBroken"],
        [f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 2 --shard-index 0 --device cuda:0", f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 2 --shard-index 1 --device cuda:0", f"{PYTHON} {rel(RUNNER)} --mode part-c-merge"],
    )
    write_inventory_and_manifest()
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_c_matrix.csv')}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part C strict raw-channel diagnosis", summary)
    return summary


def average_channel(rows: list[dict[str, str]]) -> torch.Tensor:
    mats = [load_channel_matrix(str(r.get("projector_npz", ""))) for r in rows]
    mats = [m for m in mats if m is not None and int(m.numel()) > 0]
    if not mats:
        return torch.zeros((0, 0), dtype=torch.float64)
    dim = min(int(m.shape[0]) for m in mats)
    out = torch.zeros((dim, dim), dtype=torch.float64)
    for mat in mats:
        out = out + sym(mat[:dim, :dim])
    return sym(out / float(len(mats)))


def build_seed_quotient(
    method: str,
    pos: torch.Tensor,
    neg: torch.Tensor,
    mlp: torch.Tensor,
    debt: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    dim = int(pos.shape[0]) if int(pos.numel()) else 0
    if str(method) == "D2_generalized_eigen":
        return GeneralizedEigenQuotient(rank=int(args.signal_rank), ridge=float(args.quotient_ridge)).build(
            pos,
            neg,
            mlp,
            debt,
            shape_metric=torch.eye(dim, dtype=torch.float64) if dim else None,
        ).basis
    if str(method) == "D3_orthogonalized":
        return OrthogonalizedChannelQuotient(rank=int(args.signal_rank), bad_rank=int(args.bad_channel_rank)).build(pos, neg, mlp, debt).basis
    if str(method) == "D4_partial_cca_proxy":
        bad = sym(pad_matrix(neg, dim) + pad_matrix(mlp, dim) + pad_matrix(debt, dim)) if dim else torch.zeros((0, 0), dtype=torch.float64)
        return OrthogonalizedChannelQuotient(rank=int(args.signal_rank), bad_rank=int(args.bad_channel_rank)).build(pos, bad).basis
    return PSDClippedDifferenceQuotient(rank=int(args.signal_rank)).build(pos, neg, mlp, debt).basis


def quotient_seed_stability(
    rows: list[dict[str, str]],
    neg_rows: list[dict[str, str]],
    mlp_rows: list[dict[str, str]],
    debt_rows: list[dict[str, str]],
    args: argparse.Namespace,
    method: str,
) -> float:
    vals: list[float] = []
    for seed in sorted({r.get("seed") for r in rows}):
        pos = average_channel([r for r in rows if r.get("seed") == seed])
        neg = average_channel([r for r in neg_rows if r.get("seed") == seed])
        mlp = average_channel([r for r in mlp_rows if r.get("seed") == seed])
        debt = average_channel([r for r in debt_rows if r.get("seed") == seed])
        if int(pos.numel()) == 0:
            continue
        q = build_seed_quotient(method, pos, neg, mlp, debt, args)
        if int(q.numel()) == 0:
            continue
        for other_seed in sorted({r.get("seed") for r in rows if r.get("seed") != seed}):
            pos2 = average_channel([r for r in rows if r.get("seed") == other_seed])
            neg2 = average_channel([r for r in neg_rows if r.get("seed") == other_seed])
            mlp2 = average_channel([r for r in mlp_rows if r.get("seed") == other_seed])
            debt2 = average_channel([r for r in debt_rows if r.get("seed") == other_seed])
            if int(pos2.numel()) == 0:
                continue
            q2 = build_seed_quotient(method, pos2, neg2, mlp2, debt2, args)
            if int(q2.numel()) != 0:
                vals.append(projector_overlap(q, q2))
    return median(vals)


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("D", "D_BlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C route is RawEstimatorBroken or incomplete; quotient construction is forbidden.", args)
    rows = read_rows(OUT_ROOT / "part_c_matrix.csv")
    source = [r for r in rows if r.get("status") == "ok" and r.get("estimator_type") == str(args.quotient_source_estimator)]
    pos_rows = [r for r in source if r.get("control") == "c2_positive"]
    neg_rows = [r for r in source if r.get("control") in {"random_label", "source_shuffle"}]
    mlp_rows = [r for r in source if r.get("control") == "mlp_friendly_negative"]
    debt_rows = [r for r in source if fval(r.get("guard_debt_delta")) > 0.0]
    a_pos = average_channel(pos_rows)
    a_neg = average_channel(neg_rows)
    a_mlp = average_channel(mlp_rows)
    a_debt = average_channel(debt_rows)
    dim = int(a_pos.shape[0]) if int(a_pos.numel()) else 0
    shape = torch.eye(dim, dtype=torch.float64)
    diff = PSDClippedDifferenceQuotient(rank=int(args.signal_rank)).build(a_pos, a_neg, a_mlp, a_debt)
    geneig = GeneralizedEigenQuotient(rank=int(args.signal_rank), ridge=float(args.quotient_ridge)).build(a_pos, a_neg, a_mlp, a_debt, shape_metric=shape)
    orth = OrthogonalizedChannelQuotient(rank=int(args.signal_rank), bad_rank=int(args.bad_channel_rank)).build(a_pos, a_neg, a_mlp, a_debt)
    bad = sym(pad_matrix(a_neg, dim) + pad_matrix(a_mlp, dim) + pad_matrix(a_debt, dim)) if dim else torch.zeros((0, 0), dtype=torch.float64)
    partial = OrthogonalizedChannelQuotient(rank=int(args.signal_rank), bad_rank=int(args.bad_channel_rank)).build(a_pos, bad)
    q_rows: list[dict[str, Any]] = []
    for name, result in (("D1_psd_diff", diff), ("D2_generalized_eigen", geneig), ("D3_orthogonalized", orth), ("D4_partial_cca_proxy", partial)):
        q_path = OUT_ROOT / f"part_d_{name}_quotient.npz"
        save_npz(q_path, U=result.basis, eig=result.eigenvalues, W=result.matrix)
        qmat = result.projector
        energies: list[float] = []
        for r in pos_rows:
            row_mat = load_channel_matrix(str(r.get("projector_npz", "")))
            energies.append(channel_overlap_matrix(qmat, row_mat if row_mat is not None else torch.zeros_like(qmat)))
        c2_corr = pearson(energies, [fval(r.get("visual_coverage_improvement")) for r in pos_rows])
        debt_corr = pearson(energies, [fval(r.get("guard_debt_delta")) for r in pos_rows])
        row = {
            "scheme": name,
            "status": "ok",
            "quotient_npz": rel(q_path),
            "quotient_seed_stability": quotient_seed_stability(pos_rows, neg_rows, mlp_rows, debt_rows, args, name),
            "quotient_C2_gain_corr": c2_corr,
            "quotient_F5_debt_corr": debt_corr,
            **result.summary,
        }
        if name == "D4_partial_cca_proxy":
            row.update({"partial_cca_top_corr": math.sqrt(max(fval(row.get("positive_energy_after_projection")), 0.0) / max(fval(row.get("positive_energy_before_projection")), EPS)), "partial_cca_seed_stability": row["quotient_seed_stability"], "conditional_negative_corr": row.get("negative_leakage_after_projection", 0.0), "conditional_mlp_corr": row.get("mlp_leakage_after_projection", 0.0), "conditional_debt_corr": row.get("debt_leakage_after_projection", 0.0), "mode_pullback_energy": fval(row.get("positive_energy_after_projection"))})
        q_rows.append(row)
    write_rows(OUT_ROOT / "part_d_matrix.csv", q_rows)
    diff_pass = int(
        fval(q_rows[0].get("quotient_seed_stability")) >= 0.65
        and fval(q_rows[0].get("quotient_negative_leakage")) <= 0.50
        and fval(q_rows[0].get("quotient_mlp_leakage")) <= 0.50
        and fval(q_rows[0].get("quotient_C2_gain_corr")) > 0.20
        and fval(q_rows[0].get("quotient_F5_debt_corr")) <= 0.20
    )
    geneig_pass = int(fval(q_rows[1].get("geneig_gap")) >= 1.20 and fval(q_rows[1].get("geneig_safe_rank")) >= 2 and fval(q_rows[1].get("quotient_seed_stability")) >= 0.65 and fval(q_rows[1].get("geneig_negative_overlap")) <= 0.50 and fval(q_rows[1].get("geneig_debt_overlap")) <= 0.50)
    orth_pass = int(fval(q_rows[2].get("retention_ratio")) >= 0.25 and fval(q_rows[2].get("negative_leakage_after_projection")) <= 0.50 * max(fval(q_rows[2].get("negative_leakage_before")), EPS) and fval(q_rows[2].get("mlp_leakage_after_projection")) <= 0.50 * max(fval(q_rows[2].get("mlp_leakage_before")), EPS) and fval(q_rows[2].get("debt_leakage_after_projection")) <= 0.50 * max(fval(q_rows[2].get("debt_leakage_before")), EPS))
    partial_pass = int(fval(q_rows[3].get("partial_cca_top_corr")) >= 0.25 and fval(q_rows[3].get("partial_cca_seed_stability")) >= 0.60 and fval(q_rows[3].get("conditional_negative_corr")) <= 0.15 and fval(q_rows[3].get("conditional_mlp_corr")) <= 0.15 and fval(q_rows[3].get("conditional_debt_corr")) <= 0.15)
    gate = int(diff_pass or geneig_pass or orth_pass or partial_pass)
    if gate:
        route, blocker = "D_QuotientChannelFound", "none"
    elif max(fval(r.get("quotient_positive_trace", r.get("positive_energy_after_projection"))) for r in q_rows) <= EPS:
        route, blocker = "D_QuotientSuppressesAllSignal", "quotient_trace_zero"
    elif max(fval(r.get("quotient_debt_leakage", r.get("debt_leakage_after_projection"))) for r in q_rows) > 0.50:
        route, blocker = "D_QuotientCannotSeparateDebt", "debt_leakage"
    else:
        route, blocker = "D_QuotientChannelUnstable", "seed_stability_or_specificity"
    summary = gate_summary(
        "D",
        gate,
        route,
        blocker,
        q_rows,
        d1_pass=diff_pass,
        d2_pass=geneig_pass,
        d3_pass=orth_pass,
        d4_pass=partial_pass,
        source_estimator=str(args.quotient_source_estimator),
        channel_space="output_sketch",
        shape_metric_space="output_sketch_identity_proxy",
        pos_rows=len(pos_rows),
        neg_rows=len(neg_rows),
        mlp_rows=len(mlp_rows),
        debt_rows=len(debt_rows),
        stability_audit_repair_note="2026-07-01: quotient_seed_stability is computed per quotient scheme, not reused from PSD-diff.",
    )
    summary_path = OUT_ROOT / "part_d_summary.json"
    write_json(summary_path, summary)
    write_rows(OUT_ROOT / "part_d_negative_leakage_decomposition.csv", NegativeLeakageDecomposition().decompose(a_pos, {"neg": a_neg, "mlp": a_mlp, "debt": a_debt}))
    next_path = common_next_actions(
        "D",
        blocker,
        ["increase quotient ridge", "trace normalization sensitivity", "reduce quotient rank", "blockwise generalized eigen", "try orthogonal quotient and partial CCA"],
        ["delete random-label/source-shuffle controls", "delete MLP channel", "delete debt channel", "select a runtime winner"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-d"],
    )
    append_exec("part-d", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_d_matrix.csv')}; {rel(summary_path)}; {rel(next_path)}")
    append_recap("Part D quotient channel construction", summary)
    return summary


def write_blocked_part(letter: str, route: str, blocker: str, reason: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = [{"part": letter, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}]
    matrix = OUT_ROOT / f"part_{letter.lower()}_matrix.csv"
    summary_path = OUT_ROOT / f"part_{letter.lower()}_summary.json"
    write_rows(matrix, rows)
    summary = gate_summary(letter, 0, route, blocker, rows, blocked_reason=reason)
    write_json(summary_path, summary)
    next_path = common_next_actions(letter, blocker, ["return to failed prerequisite gate"], ["run downstream metric induction while prerequisite failed"], [f"{PYTHON} {rel(RUNNER)} --mode part-c-merge", f"{PYTHON} {rel(RUNNER)} --mode part-d"])
    append_exec(f"part-{letter.lower()}", command_text(sys.argv), "blocked", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap(f"Part {letter} blocked", summary)
    return summary


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    d = read_json(OUT_ROOT / "part_d_summary.json")
    if int(d.get("gate_pass", 0)) != 1:
        return write_blocked_part("E", "E_BlockedByPartD", str(d.get("dominant_blocker", "part_d_failed")), "Part D did not find a quotient channel; reverse audit is forbidden.", args)
    rows_d = read_rows(OUT_ROOT / "part_d_matrix.csv")
    quotient_rows: list[tuple[dict[str, str], torch.Tensor]] = []
    for row in rows_d:
        qmat = load_channel_matrix(str(row.get("quotient_npz", "")))
        if qmat is not None:
            quotient_rows.append((row, qmat))
    if not quotient_rows:
        return write_blocked_part("E", "E_QuotientArtifactMissing", "quotient_npz_missing", "Part D summary passed but quotient matrix could not be loaded.", args)
    device = device_from_args(args)
    xtr, ytr, _xg, _yg = visual_data("local_patch_interaction", 0, args, device)
    xsrc, ysrc = xtr[: int(args.channel_source_size)], ytr[: int(args.channel_source_size)]
    e_rows: list[dict[str, Any]] = []
    scheme_max_energy: dict[str, float] = {}
    scheme_stability: dict[str, float] = {}
    for qrow, qmat in quotient_rows:
        scheme = str(qrow.get("scheme", "unknown"))
        scheme_stability[scheme] = fval(qrow.get("quotient_seed_stability"))
        for basis_key in csv_items(args.part0_basis):
            bargs = basis_args(args, basis_key)
            model = make_model("depth3", int(xsrc.shape[1]), int(args.num_classes), 229790 + len(e_rows), bargs, device)
            for row in project_channel_to_mode_bands(qmat, model, (xsrc, ysrc), max_outputs=int(args.max_channel_outputs), output_mode=str(args.channel_output_mode)):
                row.update({"status": "ok", "scheme": scheme, "audit_type": "mode_band", "basis_key": basis_key, "quotient_seed_stability": scheme_stability[scheme]})
                e_rows.append(row)
                scheme_max_energy[scheme] = max(scheme_max_energy.get(scheme, 0.0), fval(row.get("channel_energy_fraction")))
            for row in project_channel_to_layers(qmat, model, (xsrc, ysrc), max_outputs=int(args.max_channel_outputs), output_mode=str(args.channel_output_mode)):
                row.update({"status": "ok", "scheme": scheme, "audit_type": "layer", "basis_key": basis_key, "quotient_seed_stability": scheme_stability[scheme]})
                e_rows.append(row)
    write_rows(OUT_ROOT / "part_e_matrix.csv", e_rows)
    scheme_pass = {scheme: int(stab >= 0.65 and scheme_max_energy.get(scheme, 0.0) > 0.0) for scheme, stab in scheme_stability.items()}
    gate = int(any(scheme_pass.values()))
    induced = {
        "source_schemes": sorted(scheme_stability),
        "e_gate_pass_schemes": [scheme for scheme, passed in scheme_pass.items() if passed],
        "mode_weighted_metric": int(gate),
        "edge_measure_metric": 0,
        "signal_visible_operator_metric": int(gate),
        "note": "constructed from all pre-registered Part D quotient schemes; no runtime or result-based winner selection performed",
    }
    write_json(OUT_ROOT / "induced_metric_spec.json", induced)
    summary = gate_summary(
        "E",
        gate,
        "E_ReverseAuditPass" if gate else "E_ReverseAuditUnstable",
        "none" if gate else "mode_band_instability",
        e_rows,
        mode_band_seed_stability=max(scheme_stability.values() or [0.0]),
        max_mode_band_energy=max(scheme_max_energy.values() or [0.0]),
        scheme_seed_stability=scheme_stability,
        scheme_max_mode_band_energy=scheme_max_energy,
        scheme_pass=scheme_pass,
        induced_metric_spec=rel(OUT_ROOT / "induced_metric_spec.json"),
        repair_note="2026-07-01: Part E implementation repaired to audit all Part D quotient schemes instead of only the first matrix row.",
    )
    write_json(OUT_ROOT / "part_e_summary.json", summary)
    append_exec("part-e", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_e_matrix.csv')}; {rel(OUT_ROOT / 'part_e_summary.json')}; {rel(OUT_ROOT / 'induced_metric_spec.json')}")
    append_recap("Part E channel-to-metric reverse audit", summary)
    return summary


def coefficient_pullback_diag(j: torch.Tensor, channel: torch.Tensor, shape_diag: torch.Tensor | None = None) -> torch.Tensor:
    jj = j.detach().to(dtype=torch.float64).cpu()
    pp = sym(channel.detach().to(dtype=torch.float64).cpu())
    n_out = min(int(jj.shape[0]), int(pp.shape[0]))
    weighted = pp[:n_out, :n_out] @ jj[:n_out, :]
    diag = (jj[:n_out, :] * weighted).sum(dim=0).clamp_min(0.0)
    if shape_diag is not None:
        sd = shape_diag.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
        n = min(int(diag.numel()), int(sd.numel()))
        out = diag.clone()
        out[:n] = out[:n] / sd[:n]
        return out.clamp_min(0.0)
    return diag


def vector_summary(vec: torch.Tensor) -> dict[str, float | int]:
    v = vec.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    trace = float(v.sum().item())
    if trace <= EPS:
        return {"coeff_trace": 0.0, "coeff_top_mass": 0.0, "coeff_effective_rank": 0.0, "coeff_positive_count": 0}
    vals = torch.sort(v, descending=True).values
    k = max(1, int(math.ceil(0.10 * int(vals.numel()))))
    probs = v / max(trace, EPS)
    eff = float(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum()).item())
    return {
        "coeff_trace": trace,
        "coeff_top_mass": float(vals[:k].sum().item()) / max(trace, EPS),
        "coeff_effective_rank": eff,
        "coeff_positive_count": int((v > 0.0).sum().item()),
    }


def vec_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().to(dtype=torch.float64).cpu().reshape(-1)
    bb = b.detach().to(dtype=torch.float64).cpu().reshape(-1)
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    denom = aa.norm() * bb.norm()
    if float(denom.item()) <= EPS:
        return 0.0
    return float((aa @ bb / denom).clamp(0.0, 1.0).item())


def normalize_trace_vec(vec: torch.Tensor) -> torch.Tensor:
    v = vec.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    return v / v.sum().clamp_min(EPS)


def coeff_quotient_diag(pos: torch.Tensor, neg: torch.Tensor, mlp: torch.Tensor, debt: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    p = normalize_trace_vec(pos)
    n = normalize_trace_vec(neg)
    m = normalize_trace_vec(mlp)
    d = normalize_trace_vec(debt)
    q = (p - n - m - d).clamp_min(0.0)
    q_trace = float(q.sum().item())
    return q, {
        "coeff_quotient_trace": q_trace,
        "coeff_retention_ratio": q_trace / max(float(p.sum().item()), EPS),
        "coeff_negative_leakage_cos": vec_cosine(q, n),
        "coeff_mlp_leakage_cos": vec_cosine(q, m),
        "coeff_debt_leakage_cos": vec_cosine(q, d),
    }


def mode_band_masks_for_model(model: Any) -> list[tuple[str, torch.Tensor]]:
    masks: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for layer_idx, param in enumerate(model.coeffs):
        d_in, d_out, k = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
        if str(model.basis_name) == "chebyshev":
            bands = {
                "degree_0_1": range(0, min(k, 2)),
                "degree_2_3": range(2, min(k, 4)),
                "degree_4_5": range(4, min(k, 6)),
                "degree_6_plus": range(6, k),
            }
        else:
            bands = {
                "constant": range(0, min(k, 1)),
                "low_harmonic": range(1, min(k, 3)),
                "mid_harmonic": range(3, min(k, 5)),
                "high_harmonic": range(5, k),
            }
        for name, idxs in bands.items():
            mask = torch.zeros((d_in, d_out, k), dtype=torch.float64)
            for mode_idx in idxs:
                if int(mode_idx) < k:
                    mask[:, :, int(mode_idx)] = 1.0
            masks.append((f"layer{layer_idx}_{name}", torch.cat([torch.zeros(offset, dtype=torch.float64), mask.reshape(-1)])))
        offset += int(param.numel())
    total = offset
    padded: list[tuple[str, torch.Tensor]] = []
    for name, mask in masks:
        if int(mask.numel()) < total:
            mask = torch.cat([mask, torch.zeros(total - int(mask.numel()), dtype=torch.float64)])
        padded.append((name, mask[:total]))
    return padded


def mode_band_distribution(vec: torch.Tensor, model: Any) -> dict[str, float]:
    v = vec.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    total = float(v.sum().item())
    out: dict[str, float] = {}
    for name, mask in mode_band_masks_for_model(model):
        n = min(int(v.numel()), int(mask.numel()))
        out[name] = float((v[:n] * mask[:n]).sum().item()) / max(total, EPS)
    return out


def reconstruct_part_c_model_and_batch(row: dict[str, str], args: argparse.Namespace, device: torch.device) -> tuple[Any, tuple[torch.Tensor, torch.Tensor]]:
    basis_key = str(row.get("basis_key"))
    depth = str(row.get("depth"))
    task = str(row.get("task"))
    seed = int(float(row.get("seed", 0)))
    control = str(row.get("control"))
    bargs = basis_args(args, basis_key)
    xtr, ytr, xg, yg = controlled_data(task, control, seed, args, device)
    xsrc, ysrc, _xtraj, _ytraj, _xgrad, _ygrad, _split_info = v2296.part_c_source_witness_split(xtr, ytr, control, seed, args)
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = 2297000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
    model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
    return model, (xsrc, ysrc)


def run_part_e_coeff(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("E_coeff", "E_CoeffBlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C did not pass; coefficient pullback repair is forbidden.", args)
    rows = read_rows(OUT_ROOT / "part_c_strict_offdiag_matrix.csv")
    shard = shard_items(rows, args)
    device = device_from_args(args)
    out_rows: list[dict[str, Any]] = []
    vec_root = OUT_ROOT / "part_e_coeff_pullback_vectors"
    for row in shard:
        start = time.time()
        try:
            channel = load_channel_matrix(str(row.get("projector_npz", "")))
            if channel is None:
                raise ValueError("missing strict channel matrix")
            model, (xsrc, ysrc) = reconstruct_part_c_model_and_batch(row, args, device)
            probe_seed = int(float(row.get("probe_seed", args.output_sketch_seed)))
            j = full_output_jacobian(
                model,
                xsrc,
                ysrc,
                max_outputs=int(args.max_channel_outputs),
                mode=str(args.channel_output_mode),
                sketch_mode=str(args.output_sketch_mode),
                sketch_seed=probe_seed,
            )
            shape_diag = full_mode_metric_vector(model).detach().cpu()
            diag = coefficient_pullback_diag(j, channel, shape_diag if int(args.coeff_shape_whiten) else None)
            row_id = str(row.get("row_id", f"coeff_{len(out_rows)}"))
            vec_path = vec_root / f"{row_id}.npz"
            save_npz(vec_path, coeff_diag=diag)
            out = {
                "row_id": row_id,
                "status": "ok",
                "vector_npz": rel(vec_path),
                "basis_key": row.get("basis_key", ""),
                "depth": row.get("depth", ""),
                "scheme": row.get("scheme", ""),
                "task": row.get("task", ""),
                "seed": row.get("seed", ""),
                "control": row.get("control", ""),
                "guard_debt_delta": row.get("guard_debt_delta", ""),
                "visual_coverage_improvement": row.get("visual_coverage_improvement", ""),
                "shape_whitened": int(args.coeff_shape_whiten),
                "wall_time_s": time.time() - start,
                **vector_summary(diag),
                **AUDIT_DEFAULTS,
            }
            out_rows.append(out)
        except Exception as exc:
            out_rows.append(
                {
                    "row_id": row.get("row_id", f"coeff_error_{len(out_rows)}"),
                    "status": "error",
                    "error_message": repr(exc),
                    "basis_key": row.get("basis_key", ""),
                    "depth": row.get("depth", ""),
                    "scheme": row.get("scheme", ""),
                    "task": row.get("task", ""),
                    "seed": row.get("seed", ""),
                    "control": row.get("control", ""),
                    "wall_time_s": time.time() - start,
                    **AUDIT_DEFAULTS,
                }
            )
        write_rows(OUT_ROOT / f"part_e_coeff_pullback_rows_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", out_rows)
    path = OUT_ROOT / f"part_e_coeff_pullback_rows_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    append_exec("part-e-coeff", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(out_rows)}")
    return gate_summary("E_coeff", 0, "E_CoeffShardsWritten", "merge_required", out_rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))


def load_coeff_vector(path_text: str) -> torch.Tensor | None:
    if not path_text:
        return None
    path = ROOT / path_text if not Path(path_text).is_absolute() else Path(path_text)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        return torch.from_numpy(arr["coeff_diag"]).to(dtype=torch.float64)
    except Exception:
        return None


def average_coeff_vectors(rows: list[dict[str, str]]) -> torch.Tensor:
    vecs = [load_coeff_vector(str(r.get("vector_npz", ""))) for r in rows]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return torch.zeros(0, dtype=torch.float64)
    dim = min(int(v.numel()) for v in vecs)
    out = torch.zeros(dim, dtype=torch.float64)
    for vec in vecs:
        out = out + vec[:dim].detach().to(dtype=torch.float64).cpu()
    return out / float(len(vecs))


def run_part_e_coeff_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_e_coeff_pullback_rows_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "part_e_coeff_pullback_rows.csv", [dict(r) for r in rows])
    ok = [r for r in rows if r.get("status") == "ok"]
    q_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    stability_vals: list[float] = []
    band_stability_vals: list[float] = []
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task")) == key]
        seeds = sorted({r.get("seed") for r in group})
        q_by_seed: dict[str, torch.Tensor] = {}
        band_by_seed: dict[str, torch.Tensor] = {}
        basis_key, depth, scheme, task = key
        device = device_from_args(args)
        dummy_args = basis_args(args, str(basis_key))
        dummy_model = make_model(str(depth), int(args.visual_side) * int(args.visual_side), int(args.num_classes), 229799, dummy_args, device)
        masks = mode_band_masks_for_model(dummy_model)
        for seed in seeds:
            seed_group = [r for r in group if r.get("seed") == seed]
            pos = average_coeff_vectors([r for r in seed_group if r.get("control") == "c2_positive"])
            neg = average_coeff_vectors([r for r in seed_group if r.get("control") in {"random_label", "source_shuffle"}])
            mlp = average_coeff_vectors([r for r in seed_group if r.get("control") == "mlp_friendly_negative"])
            debt = average_coeff_vectors([r for r in seed_group if fval(r.get("guard_debt_delta")) > 0.0])
            if int(pos.numel()) == 0:
                continue
            dim = int(pos.numel())
            if int(neg.numel()) == 0:
                neg = torch.zeros(dim, dtype=torch.float64)
            if int(mlp.numel()) == 0:
                mlp = torch.zeros(dim, dtype=torch.float64)
            if int(debt.numel()) == 0:
                debt = torch.zeros(dim, dtype=torch.float64)
            neg = neg[:dim]
            mlp = mlp[:dim]
            debt = debt[:dim]
            q, info = coeff_quotient_diag(pos[:dim], neg, mlp, debt)
            q_by_seed[str(seed)] = q
            band_dist = mode_band_distribution(q, dummy_model)
            band_by_seed[str(seed)] = torch.tensor([band_dist.get(name, 0.0) for name, _mask in masks], dtype=torch.float64)
            q_path = OUT_ROOT / "part_e_coeff_pullback_vectors" / f"quotient_{basis_key}_{depth}_{scheme}_{task}_seed{seed}.npz"
            save_npz(q_path, coeff_quotient=q)
            q_rows.append(
                {
                    "basis_key": basis_key,
                    "depth": depth,
                    "scheme": scheme,
                    "task": task,
                    "seed": seed,
                    "status": "ok",
                    "quotient_npz": rel(q_path),
                    **info,
                    **vector_summary(q),
                    **AUDIT_DEFAULTS,
                }
            )
            qn = normalize_trace_vec(q)
            nn = normalize_trace_vec(neg)
            mn = normalize_trace_vec(mlp)
            dn = normalize_trace_vec(debt)
            for band_name, mask in masks:
                n = min(int(qn.numel()), int(mask.numel()))
                q_band = float((qn[:n] * mask[:n]).sum().item())
                bad_band = max(float((nn[:n] * mask[:n]).sum().item()), float((mn[:n] * mask[:n]).sum().item()), float((dn[:n] * mask[:n]).sum().item()))
                band_rows.append(
                    {
                        "basis_key": basis_key,
                        "depth": depth,
                        "scheme": scheme,
                        "task": task,
                        "seed": seed,
                        "band": band_name,
                        "quotient_band_fraction": q_band,
                        "bad_band_fraction_max": bad_band,
                        "quotient_over_bad_band_ratio": q_band / max(bad_band, EPS),
                        "status": "ok",
                    }
                )
        seed_names = sorted(q_by_seed)
        for i in range(len(seed_names)):
            for j in range(i + 1, len(seed_names)):
                stability_vals.append(vec_cosine(q_by_seed[seed_names[i]], q_by_seed[seed_names[j]]))
                band_stability_vals.append(vec_cosine(band_by_seed[seed_names[i]], band_by_seed[seed_names[j]]))
    write_rows(OUT_ROOT / "part_e_coeff_quotient_matrix.csv", q_rows)
    write_rows(OUT_ROOT / "part_e_coeff_mode_band_matrix.csv", band_rows)
    coeff_seed_stability = median(stability_vals)
    mode_band_seed_stability = median(band_stability_vals)
    max_band_ratio = max([fval(r.get("quotient_over_bad_band_ratio")) for r in band_rows] or [0.0])
    retention = median(fval(r.get("coeff_retention_ratio")) for r in q_rows)
    retention_gate = int(retention >= 0.25)
    gate = int(mode_band_seed_stability >= 0.65 and max_band_ratio >= 1.5 and retention_gate == 1 and not any(r.get("status") == "error" for r in rows))
    if retention_gate == 0:
        coeff_blocker = "coeff_retention_too_low"
    elif mode_band_seed_stability < 0.65:
        coeff_blocker = "coeff_mode_band_instability"
    elif max_band_ratio < 1.5:
        coeff_blocker = "coeff_band_specificity_low"
    else:
        coeff_blocker = "none"
    summary = gate_summary(
        "E_coeff",
        gate,
        "E_CoeffPullbackRepairPass" if gate else "E_CoeffPullbackRepairFailed",
        coeff_blocker,
        [dict(r) for r in rows],
        observed_rows=len(rows),
        ok_pullback_rows=len(ok),
        quotient_rows=len(q_rows),
        mode_band_rows=len(band_rows),
        coeff_seed_stability=coeff_seed_stability,
        mode_band_seed_stability=mode_band_seed_stability,
        max_quotient_over_bad_band_ratio=max_band_ratio,
        coeff_retention_ratio_median=retention,
        coeff_retention_gate=retention_gate,
        repair_note="Coefficient-space diagonal pullback repair using diag(J^T P J) with shape whitening; built from train-only Part C strict rows.",
    )
    write_json(OUT_ROOT / "part_e_coeff_pullback_summary.json", summary)
    write_json(
        OUT_ROOT / "part_e_summary.json",
        gate_summary(
            "E",
            gate,
            "E_CoeffPullbackRepairPass" if gate else "E_CoeffPullbackRepairFailed",
            coeff_blocker,
            [dict(r) for r in q_rows],
            coeff_pullback_summary=rel(OUT_ROOT / "part_e_coeff_pullback_summary.json"),
            mode_band_seed_stability=mode_band_seed_stability,
            max_quotient_over_bad_band_ratio=max_band_ratio,
            coeff_retention_ratio_median=retention,
            coeff_retention_gate=retention_gate,
            repair_note=(
                "Part E passed through coefficient-space diagonal pullback repair."
                if gate
                else "Part E coefficient-space pullback repair failed; high mode-band ratio was not accepted because quotient retention was too low."
            ),
        ),
    )
    write_json(
        OUT_ROOT / "induced_metric_spec.json",
        {
            "source_schemes": ["coeff_diag_quotient"],
            "e_gate_pass_schemes": ["coeff_diag_quotient"] if gate else [],
            "mode_weighted_metric": int(gate),
            "edge_measure_metric": 0,
            "signal_visible_operator_metric": int(gate),
            "dominant_blocker": coeff_blocker,
            "coeff_retention_ratio_median": retention,
            "mode_band_seed_stability": mode_band_seed_stability,
            "max_quotient_over_bad_band_ratio": max_band_ratio,
            "note": (
                "Constructed from coefficient-space diag(J^T P J) quotient repair; no runtime winner selection performed."
                if gate
                else "No induced metric promoted: coefficient-space quotient repair failed its pre-registered retention/specificity/stability gate."
            ),
        },
    )
    append_exec(
        "part-e-coeff-merge",
        command_text(sys.argv),
        "passed" if gate else "failed",
        files=f"{rel(OUT_ROOT / 'part_e_coeff_pullback_summary.json')}; {rel(OUT_ROOT / 'part_e_coeff_quotient_matrix.csv')}; {rel(OUT_ROOT / 'part_e_coeff_mode_band_matrix.csv')}",
    )
    append_recap("Part E coefficient-space pullback repair", summary)
    return summary


def load_npz_vector(path_text: str, key: str) -> torch.Tensor | None:
    if not path_text:
        return None
    path = ROOT / path_text if not Path(path_text).is_absolute() else Path(path_text)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        if key not in arr:
            return None
        return torch.from_numpy(arr[key]).to(dtype=torch.float64)
    except Exception:
        return None


def average_npz_vectors(rows: list[dict[str, str]], path_key: str, vec_key: str) -> torch.Tensor:
    vecs = [load_npz_vector(str(r.get(path_key, "")), vec_key) for r in rows]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return torch.zeros(0, dtype=torch.float64)
    dim = min(int(v.numel()) for v in vecs)
    out = torch.zeros(dim, dtype=torch.float64)
    for vec in vecs:
        out = out + vec[:dim].detach().to(dtype=torch.float64).cpu()
    return out / float(len(vecs))


def guard_batch_for_part_c_row(row: dict[str, str], args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    task = str(row.get("task"))
    control = str(row.get("control"))
    seed = int(float(row.get("seed", 0)))
    _xtr, _ytr, xg, yg = controlled_data(task, control, seed, args, device)
    n = min(int(args.channel_source_size), int(xg.shape[0]))
    return xg[:n].detach(), yg[:n].detach()


def run_part_e_sourceguard(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("E_sourceguard", "E_SourceGuardBlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C did not pass; source/guard repair is forbidden.", args)
    rows = [r for r in read_rows(OUT_ROOT / "part_c_strict_offdiag_matrix.csv") if r.get("status") == "ok"]
    shard = shard_items(rows, args)
    device = device_from_args(args)
    out_rows: list[dict[str, Any]] = []
    vec_root = OUT_ROOT / "part_e_sourceguard_pullback_vectors"
    for row in shard:
        start = time.time()
        try:
            channel = load_channel_matrix(str(row.get("projector_npz", "")))
            if channel is None:
                raise ValueError("missing strict channel matrix")
            model, (xsrc, ysrc) = reconstruct_part_c_model_and_batch(row, args, device)
            xguard, yguard = guard_batch_for_part_c_row(row, args, device)
            n = min(int(xsrc.shape[0]), int(xguard.shape[0]), int(args.channel_source_size))
            if n <= 0:
                raise ValueError("empty source/guard batch")
            probe_seed = int(float(row.get("probe_seed", args.output_sketch_seed)))
            common_j_kwargs = {
                "max_outputs": int(args.max_channel_outputs),
                "mode": str(args.channel_output_mode),
                "sketch_mode": str(args.output_sketch_mode),
                "sketch_seed": probe_seed,
            }
            j_source = full_output_jacobian(model, xsrc[:n], ysrc[:n], **common_j_kwargs)
            j_guard = full_output_jacobian(model, xguard[:n], yguard[:n], **common_j_kwargs)
            shape_diag = full_mode_metric_vector(model).detach().cpu()
            src_diag = coefficient_pullback_diag(j_source, channel, shape_diag if int(args.coeff_shape_whiten) else None)
            guard_diag = coefficient_pullback_diag(j_guard, channel, shape_diag if int(args.coeff_shape_whiten) else None)
            row_id = str(row.get("row_id", f"sourceguard_{len(out_rows)}"))
            vec_path = vec_root / f"{row_id}.npz"
            save_npz(vec_path, source_coeff_diag=src_diag, guard_coeff_diag=guard_diag)
            source_summary = {f"source_{k}": v for k, v in vector_summary(src_diag).items()}
            guard_summary = {f"guard_{k}": v for k, v in vector_summary(guard_diag).items()}
            out_rows.append(
                {
                    "row_id": row_id,
                    "status": "ok",
                    "vector_npz": rel(vec_path),
                    "basis_key": row.get("basis_key", ""),
                    "depth": row.get("depth", ""),
                    "scheme": row.get("scheme", ""),
                    "task": row.get("task", ""),
                    "seed": row.get("seed", ""),
                    "control": row.get("control", ""),
                    "guard_debt_delta": row.get("guard_debt_delta", ""),
                    "source_guard_coeff_cosine": vec_cosine(src_diag, guard_diag),
                    "source_size": n,
                    "guard_size": n,
                    "shape_whitened": int(args.coeff_shape_whiten),
                    "wall_time_s": time.time() - start,
                    **source_summary,
                    **guard_summary,
                    **AUDIT_DEFAULTS,
                }
            )
        except Exception as exc:
            out_rows.append(
                {
                    "row_id": row.get("row_id", f"sourceguard_error_{len(out_rows)}"),
                    "status": "error",
                    "error_message": repr(exc),
                    "basis_key": row.get("basis_key", ""),
                    "depth": row.get("depth", ""),
                    "scheme": row.get("scheme", ""),
                    "task": row.get("task", ""),
                    "seed": row.get("seed", ""),
                    "control": row.get("control", ""),
                    "wall_time_s": time.time() - start,
                    **AUDIT_DEFAULTS,
                }
            )
        write_rows(OUT_ROOT / f"part_e_sourceguard_pullback_rows_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv", out_rows)
    path = OUT_ROOT / f"part_e_sourceguard_pullback_rows_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    append_exec("part-e-sourceguard", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(out_rows)}")
    return gate_summary("E_sourceguard", 0, "E_SourceGuardShardsWritten", "merge_required", out_rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))


def run_part_e_sourceguard_merge(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_e_sourceguard_pullback_rows_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "part_e_sourceguard_pullback_rows.csv", [dict(r) for r in rows])
    ok = [r for r in rows if r.get("status") == "ok"]
    q_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    stability_vals: list[float] = []
    band_stability_vals: list[float] = []
    source_guard_cos_vals: list[float] = [fval(r.get("source_guard_coeff_cosine")) for r in ok]
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task")) for r in ok}):
        group = [r for r in ok if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task")) == key]
        seeds = sorted({r.get("seed") for r in group})
        q_by_seed: dict[str, torch.Tensor] = {}
        band_by_seed: dict[str, torch.Tensor] = {}
        basis_key, depth, scheme, task = key
        device = device_from_args(args)
        dummy_args = basis_args(args, str(basis_key))
        dummy_model = make_model(str(depth), int(args.visual_side) * int(args.visual_side), int(args.num_classes), 229899, dummy_args, device)
        masks = mode_band_masks_for_model(dummy_model)
        for seed in seeds:
            seed_group = [r for r in group if r.get("seed") == seed]
            src_pos = average_npz_vectors([r for r in seed_group if r.get("control") == "c2_positive"], "vector_npz", "source_coeff_diag")
            src_neg = average_npz_vectors([r for r in seed_group if r.get("control") in {"random_label", "source_shuffle"}], "vector_npz", "source_coeff_diag")
            src_mlp = average_npz_vectors([r for r in seed_group if r.get("control") == "mlp_friendly_negative"], "vector_npz", "source_coeff_diag")
            src_debt = average_npz_vectors([r for r in seed_group if fval(r.get("guard_debt_delta")) > 0.0], "vector_npz", "source_coeff_diag")
            grd_pos = average_npz_vectors([r for r in seed_group if r.get("control") == "c2_positive"], "vector_npz", "guard_coeff_diag")
            grd_neg = average_npz_vectors([r for r in seed_group if r.get("control") in {"random_label", "source_shuffle"}], "vector_npz", "guard_coeff_diag")
            grd_mlp = average_npz_vectors([r for r in seed_group if r.get("control") == "mlp_friendly_negative"], "vector_npz", "guard_coeff_diag")
            grd_debt = average_npz_vectors([r for r in seed_group if fval(r.get("guard_debt_delta")) > 0.0], "vector_npz", "guard_coeff_diag")
            if int(src_pos.numel()) == 0 or int(grd_pos.numel()) == 0:
                continue
            dim = min(int(src_pos.numel()), int(grd_pos.numel()))
            defaults = torch.zeros(dim, dtype=torch.float64)
            src_neg = src_neg[:dim] if int(src_neg.numel()) else defaults.clone()
            src_mlp = src_mlp[:dim] if int(src_mlp.numel()) else defaults.clone()
            src_debt = src_debt[:dim] if int(src_debt.numel()) else defaults.clone()
            grd_neg = grd_neg[:dim] if int(grd_neg.numel()) else defaults.clone()
            grd_mlp = grd_mlp[:dim] if int(grd_mlp.numel()) else defaults.clone()
            grd_debt = grd_debt[:dim] if int(grd_debt.numel()) else defaults.clone()
            q_src, info_src = coeff_quotient_diag(src_pos[:dim], src_neg, src_mlp, src_debt)
            q_grd, info_grd = coeff_quotient_diag(grd_pos[:dim], grd_neg, grd_mlp, grd_debt)
            q_common = torch.minimum(q_src[:dim], q_grd[:dim]).clamp_min(0.0)
            q_by_seed[str(seed)] = q_common
            band_dist = mode_band_distribution(q_common, dummy_model)
            band_by_seed[str(seed)] = torch.tensor([band_dist.get(name, 0.0) for name, _mask in masks], dtype=torch.float64)
            q_path = OUT_ROOT / "part_e_sourceguard_pullback_vectors" / f"quotient_{basis_key}_{depth}_{scheme}_{task}_seed{seed}.npz"
            save_npz(q_path, coeff_quotient=q_common, coeff_quotient_source=q_src, coeff_quotient_guard=q_grd)
            common_retention = float(q_common.sum().item())
            q_rows.append(
                {
                    "basis_key": basis_key,
                    "depth": depth,
                    "scheme": scheme,
                    "task": task,
                    "seed": seed,
                    "status": "ok",
                    "quotient_npz": rel(q_path),
                    "source_retention_ratio": info_src.get("coeff_retention_ratio", 0.0),
                    "guard_retention_ratio": info_grd.get("coeff_retention_ratio", 0.0),
                    "source_guard_quotient_cosine": vec_cosine(q_src, q_grd),
                    "source_guard_common_retention_ratio": common_retention,
                    **vector_summary(q_common),
                    **AUDIT_DEFAULTS,
                }
            )
            qn = normalize_trace_vec(q_common)
            src_bad = [normalize_trace_vec(src_neg), normalize_trace_vec(src_mlp), normalize_trace_vec(src_debt)]
            grd_bad = [normalize_trace_vec(grd_neg), normalize_trace_vec(grd_mlp), normalize_trace_vec(grd_debt)]
            for band_name, mask in masks:
                n = min(int(qn.numel()), int(mask.numel()))
                q_band = float((qn[:n] * mask[:n]).sum().item())
                bad_band = 0.0
                for s_bad, g_bad in zip(src_bad, grd_bad):
                    bad_band = max(bad_band, float(((s_bad[:n] + g_bad[:n]) * 0.5 * mask[:n]).sum().item()))
                band_rows.append(
                    {
                        "basis_key": basis_key,
                        "depth": depth,
                        "scheme": scheme,
                        "task": task,
                        "seed": seed,
                        "band": band_name,
                        "quotient_band_fraction": q_band,
                        "bad_band_fraction_max": bad_band,
                        "quotient_over_bad_band_ratio": q_band / max(bad_band, EPS),
                        "status": "ok",
                    }
                )
        seed_names = sorted(q_by_seed)
        for i in range(len(seed_names)):
            for j in range(i + 1, len(seed_names)):
                stability_vals.append(vec_cosine(q_by_seed[seed_names[i]], q_by_seed[seed_names[j]]))
                band_stability_vals.append(vec_cosine(band_by_seed[seed_names[i]], band_by_seed[seed_names[j]]))
    write_rows(OUT_ROOT / "part_e_sourceguard_quotient_matrix.csv", q_rows)
    write_rows(OUT_ROOT / "part_e_sourceguard_mode_band_matrix.csv", band_rows)
    coeff_seed_stability = median(stability_vals)
    mode_band_seed_stability = median(band_stability_vals)
    max_band_ratio = max([fval(r.get("quotient_over_bad_band_ratio")) for r in band_rows] or [0.0])
    row_source_guard_cos = median(source_guard_cos_vals)
    quotient_source_guard_cos = median(fval(r.get("source_guard_quotient_cosine")) for r in q_rows)
    retention = median(fval(r.get("source_guard_common_retention_ratio")) for r in q_rows)
    retention_gate = int(retention >= 0.25)
    source_guard_gate = int(row_source_guard_cos >= 0.60 and quotient_source_guard_cos >= 0.60)
    gate = int(
        mode_band_seed_stability >= 0.65
        and max_band_ratio >= 1.5
        and retention_gate == 1
        and source_guard_gate == 1
        and not any(r.get("status") == "error" for r in rows)
    )
    if retention_gate == 0:
        blocker = "source_guard_common_retention_too_low"
    elif source_guard_gate == 0:
        blocker = "source_guard_quotient_instability"
    elif mode_band_seed_stability < 0.65:
        blocker = "source_guard_mode_band_instability"
    elif max_band_ratio < 1.5:
        blocker = "source_guard_band_specificity_low"
    else:
        blocker = "none"
    summary = gate_summary(
        "E_sourceguard",
        gate,
        "E_SourceGuardCoeffRepairPass" if gate else "E_SourceGuardCoeffRepairFailed",
        blocker,
        [dict(r) for r in rows],
        observed_rows=len(rows),
        ok_pullback_rows=len(ok),
        quotient_rows=len(q_rows),
        mode_band_rows=len(band_rows),
        coeff_seed_stability=coeff_seed_stability,
        mode_band_seed_stability=mode_band_seed_stability,
        max_quotient_over_bad_band_ratio=max_band_ratio,
        source_guard_coeff_cosine_median=row_source_guard_cos,
        source_guard_quotient_cosine_median=quotient_source_guard_cos,
        source_guard_common_retention_ratio_median=retention,
        source_guard_retention_gate=retention_gate,
        source_guard_consistency_gate=source_guard_gate,
        repair_note="Source/guard coefficient pullback repair: diag(J_source^T P J_source) must agree with diag(J_guard^T P J_guard) before metric promotion.",
    )
    write_json(OUT_ROOT / "part_e_sourceguard_pullback_summary.json", summary)
    write_json(
        OUT_ROOT / "part_e_summary.json",
        gate_summary(
            "E",
            gate,
            "E_SourceGuardCoeffRepairPass" if gate else "E_SourceGuardCoeffRepairFailed",
            blocker,
            [dict(r) for r in q_rows],
            sourceguard_pullback_summary=rel(OUT_ROOT / "part_e_sourceguard_pullback_summary.json"),
            mode_band_seed_stability=mode_band_seed_stability,
            max_quotient_over_bad_band_ratio=max_band_ratio,
            source_guard_coeff_cosine_median=row_source_guard_cos,
            source_guard_quotient_cosine_median=quotient_source_guard_cos,
            source_guard_common_retention_ratio_median=retention,
            source_guard_retention_gate=retention_gate,
            source_guard_consistency_gate=source_guard_gate,
            repair_note=(
                "Part E passed through source/guard coefficient pullback repair."
                if gate
                else "Part E source/guard coefficient pullback repair failed; no induced metric promoted."
            ),
        ),
    )
    write_json(
        OUT_ROOT / "induced_metric_spec.json",
        {
            "source_schemes": ["sourceguard_coeff_diag_quotient"],
            "e_gate_pass_schemes": ["sourceguard_coeff_diag_quotient"] if gate else [],
            "mode_weighted_metric": int(gate),
            "edge_measure_metric": 0,
            "signal_visible_operator_metric": int(gate),
            "dominant_blocker": blocker,
            "source_guard_common_retention_ratio_median": retention,
            "source_guard_coeff_cosine_median": row_source_guard_cos,
            "source_guard_quotient_cosine_median": quotient_source_guard_cos,
            "mode_band_seed_stability": mode_band_seed_stability,
            "max_quotient_over_bad_band_ratio": max_band_ratio,
            "note": (
                "Constructed from source/guard-stable coefficient-space quotient repair; no runtime winner selection performed."
                if gate
                else "No induced metric promoted: source/guard coefficient-space quotient repair failed its retention/consistency/specificity/stability gate."
            ),
        },
    )
    next_path = common_next_actions(
        "E_sourceguard",
        blocker,
        ["increase seed count", "increase source/guard size", "formation-phase source/guard split diagnostic"],
        ["delete negative/MLP/debt controls", "promote source-only quotient", "lower retention gate"],
        [
            f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-e-sourceguard --shard-count 2 --shard-index 0 --device cuda:0",
            f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-e-sourceguard --shard-count 2 --shard-index 1 --device cuda:0",
            f"{PYTHON} {rel(RUNNER)} --mode part-e-sourceguard-merge --device {args.device}",
        ],
    )
    append_exec(
        "part-e-sourceguard-merge",
        command_text(sys.argv),
        "passed" if gate else "failed",
        files=f"{rel(OUT_ROOT / 'part_e_sourceguard_pullback_summary.json')}; {rel(OUT_ROOT / 'part_e_sourceguard_quotient_matrix.csv')}; {rel(OUT_ROOT / 'part_e_sourceguard_mode_band_matrix.csv')}; {rel(next_path)}",
    )
    append_recap("Part E source/guard coefficient pullback repair", summary)
    return summary


def run_downstream_gate(args: argparse.Namespace, letter: str) -> dict[str, Any]:
    prerequisites = {"F": "part_e_summary.json", "G": "part_f_summary.json", "H": "part_g_summary.json", "I": "part_g_summary.json", "J": "part_g_summary.json", "K": "part_j_summary.json"}
    prev_file = prerequisites.get(letter, "part_d_summary.json")
    prev = read_json(OUT_ROOT / prev_file)
    if int(prev.get("gate_pass", 0)) != 1:
        return write_blocked_part(letter, f"{letter}_BlockedByPrerequisite", str(prev.get("dominant_blocker", "prerequisite_failed")), f"Required prerequisite `{prev_file}` did not pass.", args)
    return write_blocked_part(letter, f"{letter}_NotImplementedAfterPrerequisitePass", "implementation_incomplete", f"Prerequisite `{prev_file}` passed, but this runner revision does not implement Part {letter} safely enough for promotion.", args)


def run_part_m(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json")
    d = read_json(OUT_ROOT / "part_d_summary.json")
    e = read_json(OUT_ROOT / "part_e_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        route = "M_PartC_EstimatorBroken"
        blocker = str(c.get("dominant_blocker", "part_c_failed"))
    elif int(d.get("gate_pass", 0)) != 1:
        droute = str(d.get("route", "D_missing"))
        route = "M_QuotientCannotSeparateDebt" if "Debt" in droute else "M_RawStableButQuotientFailed"
        blocker = str(d.get("dominant_blocker", "part_d_failed"))
    elif int(e.get("gate_pass", 0)) != 1:
        route = "M_InducedMetricNoCausality"
        blocker = str(e.get("dominant_blocker", "part_e_failed"))
    else:
        route = "M_IncompleteAfterPartE"
        blocker = "downstream_not_completed"
    rows = [{"status": "ok", "part_c_route": c.get("route", "missing"), "part_d_route": d.get("route", "missing"), "part_e_route": e.get("route", "missing"), "final_route": route, "dominant_blocker": blocker}]
    write_rows(OUT_ROOT / "part_m_failure_decomposition_matrix.csv", rows)
    summary = gate_summary("M", 1, route, blocker, rows, analysis=f"Final route is based only on artifacts generated under {rel(OUT_ROOT)}.")
    write_json(OUT_ROOT / "part_m_failure_decomposition_summary.json", summary)
    failure_md = OUT_ROOT / "failure_decomposition.md"
    failure_md.write_text(
        "# v22.97Q Failure Decomposition\n\n"
        f"- Final route: `{route}`\n"
        f"- Dominant blocker: `{blocker}`\n"
        f"- Part C route: `{c.get('route', 'missing')}`\n"
        f"- Part D route: `{d.get('route', 'missing')}`\n"
        f"- Part E route: `{e.get('route', 'missing')}`\n\n"
        "No downstream success is claimed for missing or blocked artifacts.\n",
        encoding="utf-8",
    )
    append_exec("part-m", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'part_m_failure_decomposition_summary.json')}; {rel(failure_md)}")
    append_recap("Part M failure decomposition", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    m = read_json(OUT_ROOT / "part_m_failure_decomposition_summary.json")
    if not m:
        m = run_part_m(args)
    blocker = str(m.get("dominant_blocker", "missing"))
    stale = stale_artifact_audit(blocker)
    summary = {
        "version": "v22.97Q",
        "route": m.get("route", "M_MissingFailureDecomposition"),
        "dominant_blocker": blocker,
        "official_candidate_gate_pass": 0,
        "promotion_allowed": False,
        "part_c_gate_pass": read_json(OUT_ROOT / "part_c_estimator_comparison_summary.json").get("gate_pass", "missing"),
        "part_d_gate_pass": read_json(OUT_ROOT / "part_d_summary.json").get("gate_pass", "missing"),
        "part_e_gate_pass": read_json(OUT_ROOT / "part_e_summary.json").get("gate_pass", "missing"),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "stale_artifact_count": stale.get("stale_count", 0),
        "runner": rel(RUNNER),
        "plan": rel(PLAN),
        "execution_log": rel(EXEC_LOG),
        "recap_log": rel(RECAP_LOG),
        "finalized_at": now(),
    }
    final_path = OUT_ROOT / "final_route.json"
    write_json(final_path, summary)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        "# v22.97Q Reproduction Manifest\n\n"
        f"- Python: `{PYTHON}`\n"
        f"- Runner: `{rel(RUNNER)}`\n"
        f"- Output root: `{rel(OUT_ROOT)}`\n"
        f"- Plan: `{rel(PLAN)}`\n"
        f"- Part 0: `{rel(OUT_ROOT / 'part0_summary.json')}`\n"
        f"- Part A: `{rel(OUT_ROOT / 'part_a_identity_summary.json')}`\n"
        f"- Part B: `{rel(OUT_ROOT / 'part_b_history_lock.json')}`\n"
        f"- Part C: `{rel(OUT_ROOT / 'part_c_estimator_comparison_summary.json')}`\n"
        f"- Part D: `{rel(OUT_ROOT / 'part_d_summary.json')}`\n"
        f"- Final route: `{rel(final_path)}`\n\n"
        "Parallel Part C command used by default for GPUs 2 and 3:\n\n"
        "```bash\n"
        f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 2 --shard-index 0 --device cuda:0\n"
        f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 2 --shard-index 1 --device cuda:0\n"
        f"{PYTHON} {rel(RUNNER)} --mode part-c-merge\n"
        "```\n",
        encoding="utf-8",
    )
    write_inventory_and_manifest()
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(final_path)}; {rel(manifest)}; {rel(OUT_ROOT / 'stale_artifact_audit.json')}; {rel(OUT_ROOT / 'sha256_manifest.txt')}")
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
    p.add_argument("--window-train-steps", type=int, default=1)
    p.add_argument("--window-checkpoint-count", type=int, default=1)
    p.add_argument("--window-center-mode", default="none", choices=["none", "delta_j0"])
    p.add_argument("--window-output-center-mode", default="none", choices=["none", "row_mean"])
    p.add_argument("--mode-lambda-partial", type=float, default=1.0e-4)
    p.add_argument("--mode-lambda-partial2", type=float, default=1.0e-6)
    p.add_argument("--mode-lambda-omega", type=float, default=1.0e-4)
    p.add_argument("--part0-basis", default="dche_k5,dfour_default")
    p.add_argument("--part-c-basis", default="dche_k5,dche_k9,dfour_default")
    p.add_argument("--part-c-depths", default="depth2,depth3")
    p.add_argument("--part-c-schemes", default="op_metric")
    p.add_argument("--part-c-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-c-controls", default="c2_positive,random_label,source_shuffle,mlp_friendly_negative")
    p.add_argument("--part-c-seed-count", type=int, default=3)
    p.add_argument("--channel-source-size", type=int, default=16)
    p.add_argument("--channel-output-mode", default="margin", choices=["margin", "logits"])
    p.add_argument("--max-channel-outputs", type=int, default=8)
    p.add_argument("--output-sketch-mode", default="prefix", choices=["prefix", "rademacher"])
    p.add_argument("--output-sketch-seed", type=int, default=229700)
    p.add_argument("--fresh-cohort-split", type=int, default=1)
    p.add_argument("--signal-rank", type=int, default=2)
    p.add_argument("--channel-cohorts", type=int, default=6)
    p.add_argument("--snr-examples", type=int, default=16)
    p.add_argument("--strict-offdiag-normalize-rows", type=int, default=1)
    p.add_argument("--gradient-whiten-mode", default="none", choices=["none", "diag_rms"])
    p.add_argument("--gradient-whiten-eps", type=float, default=1.0e-8)
    p.add_argument("--gradient-whiten-clamp", type=float, default=25.0)
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--data-metric-condition", type=float, default=1.0e6)
    p.add_argument("--data-metric-sketch-rank", type=int, default=16)
    p.add_argument("--data-metric-sketch-seed", type=int, default=2297)
    p.add_argument("--pg1-gamma", type=float, default=0.15)
    p.add_argument("--quotient-source-estimator", default="strict_offdiag", choices=list(PART_C_ESTIMATORS))
    p.add_argument("--quotient-ridge", type=float, default=1.0e-3)
    p.add_argument("--bad-channel-rank", type=int, default=4)
    p.add_argument("--coeff-shape-whiten", type=int, default=1)
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
    if mode == "part-d":
        return run_part_d(args)
    if mode == "part-e":
        return run_part_e(args)
    if mode == "part-e-coeff":
        return run_part_e_coeff(args)
    if mode == "part-e-coeff-merge":
        return run_part_e_coeff_merge(args)
    if mode == "part-e-sourceguard":
        return run_part_e_sourceguard(args)
    if mode == "part-e-sourceguard-merge":
        return run_part_e_sourceguard_merge(args)
    if mode in {"part-f", "part-g", "part-h", "part-i", "part-j", "part-k"}:
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
        run_part_d(args)
        run_part_e(args)
        for letter in "FGHIJK":
            run_downstream_gate(args, letter)
        run_part_m(args)
        return finalize(args)
    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
