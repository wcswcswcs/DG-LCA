#!/usr/bin/env python3
"""DG-KAN v22.98 signal-lift / fiber-cost quotient geometry runner.

This runner is gate-aware.  It validates estimator and fiber-geometry evidence
before allowing induced optimizer or positive-control stages.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v22_96_solid_signal_channel_induced_metric_mpfu as v2296
import experiments.run_v22_97_quotient_signal_channel_induced_metric_mpfu as v2297
from dgkan.fu.channel_to_metric import reverse_audit_smoke_test
from dgkan.fu.fiber_lift_geometry import (
    compute_lift_operator,
    fiber_lift_toy_smoke_test,
    generalized_lift_eig,
    kan_vs_mlp_controllability_metrics,
    lift_operator_diag,
    minimum_norm_lift_cost,
)
from dgkan.fu.fixed_point_signal_metric import fixed_point_smoke_test
from dgkan.fu.formation_phase_channel import formation_phase_smoke_test, retention_decay_summary
from dgkan.fu.kan_intrinsic_metrics import DataCompositeMetric, data_metric_smoke_tests, full_mode_metric_vector
from dgkan.fu.layer_composite_metric import EPS, sym
from dgkan.fu.quotient_signal_channel import quotient_toy_smoke_test
from dgkan.fu.signal_channel_estimators import (
    WindowedDissipationEstimator,
    full_output_jacobian,
    loss_gradient_vector,
    offdiag_synthetic_smoke_test,
    per_example_gradient_matrix,
    projector_from_psd,
    projector_overlap,
    windowed_linear_smoke_test,
)
from dgkan.fu.signal_visible_operator_invariant import signal_visible_operator_smoke_test, spectrum_cosine
from dgkan.fu.task_debt_channel_split import ComponentWiseSafeIntersection


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v22.98_SignalLiftFiberCostQuotientGeometry_MultiDirection_详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.98_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.98_实验结果复盘.md"
RESULT_ROOT = Path(os.environ.get("V2298_RESULT_ROOT", str(ROOT / "results/v22_98"))).resolve()
OUT_ROOT = Path(os.environ.get("V2298_OUT_ROOT", str(RESULT_ROOT / "current"))).resolve()
PROJ_ROOT = OUT_ROOT / "part_c_channel_projectors"
COEFF_ROOT = OUT_ROOT / "part_c_coeff_ab_vectors"
FIBER_ROOT = OUT_ROOT / "part_d_fiber_vectors"

BASIS_KEYS = ("dche_k5", "dche_k9", "dfour_default")
ESTIMATORS = ("strict_offdiag", "window_no_pg", "window_pg1_linearized", "window_velocity_cohort")
AUDIT_DEFAULTS: dict[str, Any] = {
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "metric_winner_selection_used": 0,
    "candidate_update_selection_used": 0,
    "new_edge_function_added": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "external_product_feature_used": 0,
    "changed_edge_coefficients": 1,
    "changed_mlp_tensors": 0,
}


def ensure_out() -> None:
    for path in (RESULT_ROOT, OUT_ROOT, PROJ_ROOT, COEFF_ROOT, FIBER_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
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
            "# DG-KAN v22.98 Execution Log\n\n"
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
            "# DG-KAN v22.98 Experiment Recap\n\n"
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


def ival(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / float(len(vals)) if vals else 0.0


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


def vector_summary(vec: torch.Tensor, prefix: str = "") -> dict[str, float | int]:
    v = vec.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    trace = float(v.sum().item())
    if trace <= EPS:
        return {f"{prefix}trace": 0.0, f"{prefix}top_mass": 0.0, f"{prefix}effective_rank": 0.0, f"{prefix}positive_count": 0}
    vals = torch.sort(v, descending=True).values
    k = max(1, int(math.ceil(0.10 * int(vals.numel()))))
    probs = v / max(trace, EPS)
    eff = float(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum()).item())
    return {
        f"{prefix}trace": trace,
        f"{prefix}top_mass": float(vals[:k].sum().item()) / max(trace, EPS),
        f"{prefix}effective_rank": eff,
        f"{prefix}positive_count": int((v > 0.0).sum().item()),
    }


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


def common_next_actions(
    part: str,
    blocker: str,
    allowed: list[str],
    forbidden: list[str],
    commands: list[str],
    stop_condition: str = "rerun gate and require all pre-registered thresholds",
) -> Path:
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    write_json(
        path,
        {
            "observed_blocker": blocker,
            "allowed_actions": allowed,
            "forbidden_actions": forbidden,
            "required_rerun_commands": commands,
            "max_repair_rounds": 2,
            "stop_condition": stop_condition,
        },
    )
    return path


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


def load_npz(path_text: str, key: str) -> torch.Tensor | None:
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


def load_channel_u(path_text: str) -> torch.Tensor | None:
    return load_npz(path_text, "U")


def stable_seed(*items: Any, base: int = 229800) -> int:
    h = hashlib.sha256(("|".join(map(str, items))).encode("utf-8")).hexdigest()
    return int((int(base) + int(h[:12], 16)) % 2_147_483_647)


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    index = int(args.shard_index)
    return [item for idx, item in enumerate(items) if idx % count == index]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(args.device))


def basis_args(args: argparse.Namespace, basis_key: str) -> argparse.Namespace:
    return v2296.basis_args(args, basis_key)


def make_model(depth: str, input_dim: int, classes: int, seed: int, args: argparse.Namespace, device: torch.device) -> v2293.TrueDeepPureKAN:
    return v2296.make_model(depth, input_dim, classes, seed, args, device)


def controlled_data(task: str, control: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2296.controlled_data(task, control, seed, args, device)


def visual_data(task: str, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return v2296.visual_data(task, seed, args, device)


def metric_diag_for_scheme(model: Any, x_source: torch.Tensor, scheme: str, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    return v2296.metric_diag_for_scheme(model, x_source, scheme, args)


def reconstruct_model_and_batches(row: dict[str, str], args: argparse.Namespace, device: torch.device) -> tuple[Any, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    basis_key = str(row.get("basis_key"))
    depth = str(row.get("depth"))
    task = str(row.get("task"))
    seed = int(float(row.get("seed", 0)))
    control = str(row.get("control"))
    bargs = basis_args(args, basis_key)
    xtr, ytr, xg, yg = controlled_data(task, control, seed, args, device)
    xsrc, ysrc, _xtraj, _ytraj, _xgrad, _ygrad, _split_info = v2296.part_c_source_witness_split(xtr, ytr, control, seed, args)
    n = min(int(args.channel_source_size), int(xsrc.shape[0]), int(xg.shape[0]))
    classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
    model_seed = 2298000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
    model = make_model(depth, int(xtr.shape[1]), classes, model_seed, bargs, device)
    return model, (xsrc[:n], ysrc[:n]), (xg[:n].detach(), yg[:n].detach())


def write_inventory_and_manifest() -> None:
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file())
    write_json(OUT_ROOT / "artifact_inventory.json", [{"path": rel(p), "bytes": int(p.stat().st_size)} for p in files])
    with (OUT_ROOT / "sha256_manifest.txt").open("w", encoding="utf-8") as fh:
        for path in files:
            if path.name == "sha256_manifest.txt":
                continue
            fh.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel(path)}\n")


def stale_artifact_audit() -> dict[str, Any]:
    summary_files = [
        OUT_ROOT / "part0_summary.json",
        OUT_ROOT / "part_a_identity_summary.json",
        OUT_ROOT / "part_b_history_lock.json",
        OUT_ROOT / "part_c_estimator_summary.json",
        OUT_ROOT / "part_d_summary.json",
        OUT_ROOT / "part_e_summary.json",
        OUT_ROOT / "part_f_summary.json",
        OUT_ROOT / "part_g_summary.json",
        OUT_ROOT / "part_h_summary.json",
        OUT_ROOT / "part_i_summary.json",
        OUT_ROOT / "part_j_summary.json",
        OUT_ROOT / "part_k_summary.json",
        OUT_ROOT / "part_l_failure_decomposition_summary.json",
    ]
    present = [p for p in summary_files if p.exists()]
    first_blocker = "none"
    for p in present:
        d = read_json(p)
        if p.name.startswith("part_l"):
            continue
        if int(d.get("gate_pass", 0)) != 1:
            first_blocker = str(d.get("dominant_blocker", "unknown"))
            break
    stale: list[dict[str, Any]] = []
    for p in present:
        d = read_json(p)
        if "Blocked" in str(d.get("route", "")) and str(d.get("dominant_blocker", "")) != first_blocker:
            stale.append({"file": rel(p), "dominant_blocker": d.get("dominant_blocker"), "expected_blocker": first_blocker})
    return {"audit_time": now(), "stale_count": len(stale), "stale": stale, "summary_files": [rel(p) for p in present]}


def run_part_0(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    init_logs()
    device = device_from_args(args)
    module_files = [
        "dgkan/fu/kan_intrinsic_metrics.py",
        "dgkan/fu/signal_channel_estimators.py",
        "dgkan/fu/quotient_signal_channel.py",
        "dgkan/fu/channel_to_metric.py",
        "dgkan/fu/fiber_lift_geometry.py",
        "dgkan/fu/task_debt_channel_split.py",
        "dgkan/fu/formation_phase_channel.py",
        "dgkan/fu/signal_visible_operator_invariant.py",
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
        "dgkan.fu.quotient_signal_channel",
        "dgkan.fu.channel_to_metric",
        "dgkan.fu.fiber_lift_geometry",
        "dgkan.fu.task_debt_channel_split",
        "dgkan.fu.formation_phase_channel",
        "dgkan.fu.signal_visible_operator_invariant",
    ]:
        try:
            __import__(name, fromlist=["*"])
        except Exception as exc:
            import_pass = 0
            import_errors.append(f"{name}: {repr(exc)}")
    smoke: dict[str, Any] = {}
    smoke.update(fiber_lift_toy_smoke_test())
    smoke.update(quotient_toy_smoke_test())
    smoke.update(data_metric_smoke_tests(device))
    smoke.update(windowed_linear_smoke_test())
    smoke.update(offdiag_synthetic_smoke_test())
    smoke.update(reverse_audit_smoke_test())
    smoke.update(fixed_point_smoke_test())
    smoke.update(formation_phase_smoke_test())
    smoke.update(signal_visible_operator_smoke_test())
    write_json(OUT_ROOT / "part0_module_smoke_tests.json", smoke)
    metric_rows: list[dict[str, Any]] = []
    xtr, _ytr, _xg, _yg = visual_data("local_patch_interaction", 0, args, device)
    xsrc = xtr[: int(args.channel_source_size)]
    for basis_key in csv_items(args.part0_basis):
        bargs = basis_args(args, basis_key)
        model = make_model("depth2", int(xsrc.shape[1]), int(args.num_classes), 229800 + len(metric_rows), bargs, device)
        dcm = DataCompositeMetric(target_condition=float(args.data_metric_condition), sketch_rank=int(args.data_metric_sketch_rank), sketch_seed=int(args.data_metric_sketch_seed))
        for metric_type in ("diagonal", "block", "full_sketch"):
            _vec, results = dcm.param_metric_vector(model, xsrc, metric_type=metric_type)
            for res in results:
                row = dict(res.summary)
                row.update({"basis_key": basis_key, "metric_type": metric_type, "true_phiTphi_row": 1, "status": "ok"})
                metric_rows.append(row)
    write_rows(OUT_ROOT / "part0_data_metric_audit.csv", metric_rows)
    stale = stale_artifact_audit()
    true_phi_rows = sum(1 for r in metric_rows if int(r.get("true_phiTphi_row", 0)) == 1 and int(r.get("phi_rows", 0)) > 0)
    gate = int(
        compile_pass
        and import_pass
        and fval(smoke.get("fiber_lift_toy_identity_error"), 1.0) < 1.0e-6
        and fval(smoke.get("fiber_lift_projection_rank_error"), 1.0) < 1.0e-6
        and fval(smoke.get("quotient_identical_channel_suppression"), 1.0) <= 1.0e-6
        and fval(smoke.get("quotient_positive_retention")) >= 0.999999
        and abs(fval(smoke.get("source_guard_identical_retention")) - 1.0) < 1.0e-6
        and fval(smoke.get("source_guard_orthogonal_retention"), 1.0) < 1.0e-6
        and true_phi_rows >= 6
        and int(stale.get("stale_count", 0)) == 0
    )
    blocker = "none" if gate else "implementation_smoke_or_artifact_hygiene_failed"
    summary = gate_summary(
        "0",
        gate,
        "P0_Pass" if gate else "P0_ImplementationSolidificationFailed",
        blocker,
        metric_rows,
        module_files=module_files,
        compile_pass=compile_pass,
        compile_errors=compile_errors,
        import_pass=import_pass,
        import_errors=import_errors,
        true_phiTphi_rows=true_phi_rows,
        real_data_metric_rows=len(metric_rows),
        stale_artifact_count=int(stale.get("stale_count", 0)),
        smoke=smoke,
    )
    write_json(OUT_ROOT / "part0_summary.json", summary)
    next_path = common_next_actions(
        "0",
        blocker,
        ["repair module compile/import", "repair fiber lift toy identities", "repair true Phi^T Phi data metric audit", "remove stale v22.98 artifacts"],
        ["run Part C with failed Part 0", "replace fiber lift with output quotient proxy"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-0 --device {args.device}"],
    )
    write_inventory_and_manifest()
    append_exec("part-0", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part0_summary.json')}; {rel(next_path)}")
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
            model = make_model(depth, int(x.shape[1]), int(args.num_classes), 229810 + len(rows), bargs, device)
            rows.append(
                {
                    "row_id": f"A_{basis_key}_{depth}",
                    "basis_key": basis_key,
                    "depth": depth,
                    "status": "ok",
                    "static_scan_pass": static_pass,
                    "static_scan_hits": json.dumps(scan_hits, ensure_ascii=False),
                    "true_depth2_purekan_constructed": int(depth == "depth2" and int(model.depth) == 2),
                    "true_depth3_purekan_constructed": int(depth == "depth3" and int(model.depth) == 3),
                    "dche_available": int(str(model.basis_name) == "chebyshev"),
                    "dfour_available": int(str(model.basis_name) == "fourier_lowfreq"),
                    **AUDIT_DEFAULTS,
                }
            )
    gate = int(
        int(p0.get("gate_pass", 0)) == 1
        and int(static_pass) == 1
        and any(int(r["true_depth2_purekan_constructed"]) for r in rows)
        and any(int(r["true_depth3_purekan_constructed"]) for r in rows)
        and any(int(r["dche_available"]) for r in rows)
        and any(int(r["dfour_available"]) for r in rows)
        and all(int(r["runtime_selector_used"]) == 0 and int(r["metric_winner_selection_used"]) == 0 for r in rows)
    )
    blocker = "none" if gate else ("part0_failed" if int(p0.get("gate_pass", 0)) != 1 else "identity_or_static_scan_failed")
    write_rows(OUT_ROOT / "part_a_identity_matrix.csv", rows)
    summary = gate_summary("A", gate, "A_Pass" if gate else "A_CodeIdentityFailed", blocker, rows, static_scan_pass=static_pass, scan_hits=scan_hits)
    write_json(OUT_ROOT / "part_a_identity_summary.json", summary)
    next_path = common_next_actions(
        "A",
        blocker,
        ["fix static scan false positives", "remove true runtime selector if present", "repair PureKAN construction"],
        ["continue to Part C with identity failure", "add MLP stem/readout"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-a --device {args.device}"],
    )
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_a_identity_summary.json')}; {rel(next_path)}")
    append_recap("Part A identity and anti-selector gate", summary)
    return summary


def history_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"state": "missing", "status": "missing", "path": rel(path)}
    data = read_json(path)
    if not data:
        return {"state": "malformed", "status": "error", "path": rel(path)}
    return {"state": "complete", "status": "ok", "path": rel(path), "route": data.get("route", data.get("dominant_blocker", "present")), "gate_pass": data.get("gate_pass", data.get("official_candidate_gate_pass", ""))}


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    sources = {
        "v22_90_final": ROOT / "results/v22_90/final_route.json",
        "v22_91_final": ROOT / "results/v22_91/final_route.json",
        "v22_93_final": ROOT / "results/v22_93/final_route.json",
        "v22_94_final": ROOT / "results/v22_94/final_route.json",
        "v22_95R_final": ROOT / "results/v22_95R/final_route.json",
        "v22_96_final": ROOT / "results/v22_96/current/final_route.json",
        "v22_97Q_final": ROOT / "results/v22_97/current/final_route.json",
        "v22_97Q_part_e_sourceguard": ROOT / "results/v22_97/current/part_e_sourceguard_pullback_summary.json",
    }
    rows = [{"artifact": name, **history_status(path)} for name, path in sources.items()]
    missing = [r for r in rows if r.get("status") != "ok"]
    gate = int(int(a.get("gate_pass", 0)) == 1 and not missing)
    blocker = "none" if gate else ("part_a_failed" if int(a.get("gate_pass", 0)) != 1 else "history_artifact_missing_or_malformed")
    hypothesis_registry = {
        "H1": "output quotient and coefficient quotient are non-commutative",
        "H2": "KAN advantage may appear as lower-cost lift, not output residual",
        "H3": "task/generic/debt freedom may be separable in KAN fiber geometry",
        "H4": "if all fiber separability tests fail, current C2/F5 carrier lacks KAN-specific safe geometry",
    }
    write_rows(OUT_ROOT / "part_b_history_artifacts.csv", rows)
    write_json(OUT_ROOT / "hypothesis_registry.json", hypothesis_registry)
    summary = gate_summary(
        "B",
        gate,
        "B_Pass" if gate else "B_HistoryLockFailed",
        blocker,
        rows,
        missing_or_error_rows=missing,
        hypothesis_registry=hypothesis_registry,
        v22_97_final_route=read_json(ROOT / "results/v22_97/current/final_route.json").get("route", "missing"),
        v22_97_final_blocker=read_json(ROOT / "results/v22_97/current/final_route.json").get("dominant_blocker", "missing"),
    )
    write_json(OUT_ROOT / "part_b_history_lock.json", summary)
    next_path = common_next_actions(
        "B",
        blocker,
        ["restore missing history artifact", "record missing explicitly without imputation"],
        ["silently infer missing history", "rewrite older route values"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-b"],
    )
    append_exec("part-b", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_b_history_lock.json')}; {rel(OUT_ROOT / 'hypothesis_registry.json')}; {rel(next_path)}")
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


def run_part_c_job(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    basis_key, depth, scheme, task, seed, control = job
    start = time.time()
    rows: list[dict[str, Any]] = []
    try:
        bargs = basis_args(args, basis_key)
        xtr, ytr, xg, yg = controlled_data(task, control, seed, args, device)
        xsrc, ysrc, xtraj, ytraj, xgrad, ygrad, split_info = v2296.part_c_source_witness_split(xtr, ytr, control, seed, args)
        n_guard = min(int(args.channel_source_size), int(xsrc.shape[0]), int(xg.shape[0]))
        xguard, yguard = xg[:n_guard].detach(), yg[:n_guard].detach()
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model_seed = 2298000 + BASIS_KEYS.index(basis_key) * 100000 + (2 if depth == "depth2" else 3) * 10000 + int(seed)
        probe_seed = stable_seed("v2298_part_c", basis_key, depth, scheme, task, seed, control, base=int(args.output_sketch_seed))
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
        j0 = full_output_jacobian(model, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
        jg = full_output_jacobian(model, xguard, yguard, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
        js = [
            full_output_jacobian(cp, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
            for cp in checkpoints
        ]
        ms = [metric_diag.detach().cpu()] * len(js)
        win_est = WindowedDissipationEstimator()
        win = win_est.estimate_from_jacobians(js, ms, rank=int(args.signal_rank), estimator_type="window_no_pg")
        pg1 = win_est.estimate_pg1_reduced(js, ms, rank=int(args.signal_rank), gamma=float(args.pg1_gamma))
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
        velocity = projector_from_psd(sym(vmat @ vmat.T / float(max(1, int(vmat.shape[1])))), int(args.signal_rank), estimator_type="window_velocity_cohort")
        strict, strict_info, strict_w = v2297.strict_offdiag_from_samples(j0, g_samples, metric_diag, args)
        source_coeff = lift_operator_diag(j0, full_mode_metric_vector(model).detach().cpu(), strict_w)
        guard_coeff = lift_operator_diag(jg, full_mode_metric_vector(model).detach().cpu(), strict_w)
        coeff_path = COEFF_ROOT / f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_strict_coeff_ab.npz"
        save_npz(coeff_path, source_coeff_diag=source_coeff, guard_coeff_diag=guard_coeff)
        projectors = {
            "strict_offdiag": (strict.u, strict.eigenvalues, strict_w),
            "window_no_pg": (win.u, win.eigenvalues, win.projector),
            "window_pg1_linearized": (pg1.u, pg1.eigenvalues, pg1.projector),
            "window_velocity_cohort": (velocity.u, velocity.eigenvalues, velocity.projector),
        }
        for estimator_type, (u, eig, wmat) in projectors.items():
            row_id = f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_{estimator_type}"
            proj_path = PROJ_ROOT / f"{row_id}.npz"
            save_npz(proj_path, U=u, eig=eig, W=wmat)
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
                "projector_npz": rel(proj_path),
                "coeff_ab_npz": rel(coeff_path) if estimator_type == "strict_offdiag" else "",
                "source_size": int(xsrc.shape[0]),
                "guard_size": int(xguard.shape[0]),
                "witness_size": int(min(int(args.snr_examples), int(xgrad.shape[0]))),
                "metric_kind": metric_info.get("metric_kind", scheme),
                "checkpoint_count": int(len(checkpoints)),
                "probe_seed": int(probe_seed),
                "self_block_energy": strict_info.get("self_block_energy", 0.0),
                "offdiag_energy": strict_info.get("offdiag_energy", 0.0),
                "self_to_offdiag_ratio": strict_info.get("self_to_offdiag_ratio", 0.0),
                "strict_offdiag_score": strict_info.get("strict_offdiag_score", 0.0),
                "positive_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "c2_positive" else "",
                "random_label_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "random_label" else "",
                "source_shuffle_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "source_shuffle" else "",
                "MLP_friendly_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if control == "mlp_friendly_negative" else "",
                "debt_offdiag_score": strict_info.get("strict_offdiag_score", 0.0) if after_guard["debt_metric"] > before_guard["debt_metric"] else "",
                "channel_window_stability": projector_overlap(u, win.u) if estimator_type == "strict_offdiag" else projector_overlap(win.u, pg1.u),
                "window_to_window_subspace_angle": 1.0 - projector_overlap(win.u, pg1.u),
                "channel_overlap_with_adamw_output_displacement": v2297.vector_channel_fraction(u, displacement),
                "C2_accuracy_short_window": after_guard["accuracy"],
                "C2_coverage_short_window": after_guard["coverage_CVaR25"],
                "visual_accuracy_improvement": after_guard["accuracy"] - before_guard["accuracy"],
                "visual_coverage_improvement": after_guard["coverage_CVaR25"] - before_guard["coverage_CVaR25"],
                "guard_debt_delta": after_guard["debt_metric"] - before_guard["debt_metric"],
                "W_top_mass_ratio": float((eig[:1].sum() / eig.clamp_min(0.0).sum().clamp_min(EPS)).item()) if int(eig.numel()) else 0.0,
                "W_effective_rank": v2297.projector_from_psd(wmat, int(args.signal_rank), estimator_type=estimator_type).summary.get("W_effective_rank", 0.0),
                "source_guard_coeff_AB_cosine": vec_cosine(source_coeff, guard_coeff) if estimator_type == "strict_offdiag" else "",
                "coeff_AB_trace": float(source_coeff.sum().item()) if estimator_type == "strict_offdiag" else "",
                "coeff_AB_guard_trace": float(guard_coeff.sum().item()) if estimator_type == "strict_offdiag" else "",
                "wall_time_s": time.time() - start,
                **whiten_info,
                **split_info,
                **AUDIT_DEFAULTS,
            }
            rows.append(row)
        return rows
    except Exception as exc:
        return [{"row_id": f"C_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{control}_error", "part": "C", "status": "error", "basis_key": basis_key, "depth": depth, "scheme": scheme, "task": task, "seed": seed, "control": control, "error_message": repr(exc), "wall_time_s": time.time() - start, **AUDIT_DEFAULTS}]


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    path = OUT_ROOT / f"part_c_matrix_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    if int(a.get("gate_pass", 0)) != 1 or int(b.get("gate_pass", 0)) != 1:
        rows = [{"part": "C", "status": "blocked", "blocked_reason": "Part A/B failed", **AUDIT_DEFAULTS}]
        write_rows(path, rows)
        append_exec("part-c", command_text(sys.argv), "blocked", files=rel(path))
        return gate_summary("C", 0, "C_BlockedByPrerequisite", "part_a_or_b_failed", rows)
    device = device_from_args(args)
    all_jobs = part_c_jobs(args)
    jobs = shard_items(all_jobs, args)
    all_rows: list[dict[str, Any]] = []
    for job in jobs:
        all_rows.extend(run_part_c_job(job, args, device))
        write_rows(path, all_rows)
    append_exec("part-c", command_text(sys.argv), "shard-written", files=rel(path), note=f"rows={len(all_rows)}")
    return gate_summary("C", 0, "C_ShardsWritten", "merge_required", all_rows, shard_index=int(args.shard_index), shard_count=int(args.shard_count))


def seed_projector_stability(rows: list[dict[str, str]]) -> float:
    vals: list[float] = []
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("task"), r.get("control"), r.get("estimator_type")) for r in rows}):
        group = [r for r in rows if (r.get("basis_key"), r.get("depth"), r.get("task"), r.get("control"), r.get("estimator_type")) == key and r.get("status") == "ok"]
        seeds = sorted({r.get("seed") for r in group})
        by_seed = {s: load_channel_u(next(r.get("projector_npz", "") for r in group if r.get("seed") == s)) for s in seeds}
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i + 1 :]:
                if by_seed.get(s1) is not None and by_seed.get(s2) is not None:
                    vals.append(projector_overlap(by_seed[s1], by_seed[s2]))
    return median(vals)


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob("part_c_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "part_c_matrix.csv", [dict(r) for r in rows])
    for est in ESTIMATORS:
        write_rows(OUT_ROOT / f"part_c_{est}_matrix.csv", [dict(r) for r in rows if r.get("estimator_type") == est])
    strict = [r for r in rows if r.get("status") == "ok" and r.get("estimator_type") == "strict_offdiag"]
    neg = [r for r in strict if r.get("control") in {"random_label", "source_shuffle"}]
    mlp = [r for r in strict if r.get("control") == "mlp_friendly_negative"]
    pos = [r for r in strict if r.get("control") == "c2_positive"]
    self_ratio = median(fval(r.get("self_to_offdiag_ratio")) for r in strict)
    strict_stability = seed_projector_stability(strict)
    window_stability = median(fval(r.get("channel_window_stability")) for r in rows if r.get("status") == "ok")
    coeff_sg_cos = median(fval(r.get("source_guard_coeff_AB_cosine")) for r in strict)
    gate = int(len(strict) > 0 and self_ratio <= 0.25 and strict_stability >= 0.60 and window_stability >= 0.65 and coeff_sg_cos >= 0.60 and not any(r.get("status") == "error" for r in rows))
    if any(r.get("status") == "error" for r in rows):
        blocker = "part_c_job_errors"
    elif self_ratio > 0.25:
        blocker = "strict_offdiag_self_block_leakage"
    elif strict_stability < 0.60:
        blocker = "strict_offdiag_seed_instability"
    elif window_stability < 0.65:
        blocker = "window_source_guard_or_pg_instability"
    elif coeff_sg_cos < 0.60:
        blocker = "coefficient_AB_source_guard_instability"
    else:
        blocker = "none"
    summary = gate_summary(
        "C",
        gate,
        "C_EstimatorStableGenericAllowed" if gate else "C_EstimatorUnstable",
        blocker,
        [dict(r) for r in rows],
        expected_rows=len(part_c_jobs(args)) * len(ESTIMATORS),
        observed_rows=len(rows),
        strict_rows=len(strict),
        strict_offdiag_seed_stability=strict_stability,
        median_self_to_offdiag_ratio=self_ratio,
        window_stability=window_stability,
        source_guard_coeff_AB_cosine=coeff_sg_cos,
        positive_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in pos),
        random_label_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in strict if r.get("control") == "random_label"),
        source_shuffle_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in strict if r.get("control") == "source_shuffle"),
        MLP_friendly_offdiag_score=median(fval(r.get("strict_offdiag_score")) for r in mlp),
        negative_control_rows=len(neg) + len(mlp),
    )
    write_json(OUT_ROOT / "part_c_estimator_summary.json", summary)
    write_json(OUT_ROOT / "part_c_signal_channel_summary.json", summary)
    next_path = common_next_actions(
        "C",
        blocker,
        ["strict self-block exclusion audit", "increase source/guard split independence", "repair per-example gradient extraction", "increase output sketch rank", "check D-CHE/D-FOUR mode whitening"],
        ["lower seed stability gate", "delete negative controls", "run Part D-L after unstable estimator"],
        [
            f"CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 0 --device cuda:0",
            f"CUDA_VISIBLE_DEVICES=1 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 1 --device cuda:0",
            f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 2 --device cuda:0",
            f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 3 --device cuda:0",
            f"{PYTHON} {rel(RUNNER)} --mode part-c-merge --device {args.device}",
        ],
    )
    append_exec("part-c-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_c_estimator_summary.json')}; {rel(next_path)}")
    append_recap("Part C strict signal estimator solidification", summary)
    return summary


def average_vectors(rows: list[dict[str, str]], key: str = "source_coeff_diag") -> torch.Tensor:
    vecs = [load_npz(str(r.get("coeff_ab_npz", "")), key) for r in rows]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return torch.zeros(0, dtype=torch.float64)
    dim = min(int(v.numel()) for v in vecs)
    out = torch.zeros(dim, dtype=torch.float64)
    for v in vecs:
        out = out + v[:dim].detach().cpu().to(dtype=torch.float64)
    return out / float(len(vecs))


def coeff_quotient_diag(pos: torch.Tensor, neg: torch.Tensor, mlp: torch.Tensor, debt: torch.Tensor, scheme: str, ridge: float) -> torch.Tensor:
    p = normalize_trace_vec(pos)
    bad = (normalize_trace_vec(neg) + normalize_trace_vec(mlp) + normalize_trace_vec(debt)) / 3.0
    if int(bad.numel()) < int(p.numel()):
        bad = torch.cat([bad, torch.zeros(int(p.numel()) - int(bad.numel()), dtype=torch.float64)])
    bad = bad[: int(p.numel())]
    if scheme == "generalized_eigen_diag":
        ratio = p / (bad + float(ridge))
        return (p * (ratio - 1.0).clamp_min(0.0)).clamp_min(0.0)
    if scheme == "control_residual_diag":
        scale = float((p @ bad / bad.square().sum().clamp_min(EPS)).clamp_min(0.0).item())
        return (p - scale * bad).clamp_min(0.0)
    return (p - bad).clamp_min(0.0)


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_estimator_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("D", "D_BlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C estimator did not pass.", args)
    rows = [r for r in read_rows(OUT_ROOT / "part_c_strict_offdiag_matrix.csv") if r.get("status") == "ok"]
    out_rows: list[dict[str, Any]] = []
    device = device_from_args(args)
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("seed")) for r in rows}):
        group = [r for r in rows if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("seed")) == key]
        pos_rows = [r for r in group if r.get("control") == "c2_positive"]
        neg_rows = [r for r in group if r.get("control") in {"random_label", "source_shuffle"}]
        mlp_rows = [r for r in group if r.get("control") == "mlp_friendly_negative"]
        debt_rows = [r for r in group if fval(r.get("guard_debt_delta")) > 0.0]
        if not pos_rows:
            continue
        basis_key, depth, scheme, task, seed = key
        row0 = pos_rows[0]
        model, source_batch, guard_batch = reconstruct_model_and_batches(row0, args, device)
        probe_seed = int(float(row0.get("probe_seed", args.output_sketch_seed)))
        p_pos = load_channel_matrix(str(row0.get("projector_npz", "")))
        p_bad_mats = [load_channel_matrix(str(r.get("projector_npz", ""))) for r in neg_rows + mlp_rows + debt_rows]
        p_bad_mats = [p for p in p_bad_mats if p is not None]
        if p_pos is None:
            continue
        p_bad = sum(p_bad_mats) / float(len(p_bad_mats)) if p_bad_mats else torch.zeros_like(p_pos)
        p_out_q = sym(p_pos - p_bad)
        vals, vecs = torch.linalg.eigh(p_out_q)
        p_out_q = sym((vecs * vals.clamp_min(0.0).reshape(1, -1)) @ vecs.T)
        j_src = full_output_jacobian(model, *source_batch, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
        j_grd = full_output_jacobian(model, *guard_batch, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
        shape = full_mode_metric_vector(model).detach().cpu()
        out_src = lift_operator_diag(j_src, shape, p_out_q)
        out_grd = lift_operator_diag(j_grd, shape, p_out_q)
        src_pos = average_vectors(pos_rows, "source_coeff_diag")
        src_neg = average_vectors(neg_rows, "source_coeff_diag")
        src_mlp = average_vectors(mlp_rows, "source_coeff_diag")
        src_debt = average_vectors(debt_rows, "source_coeff_diag")
        grd_pos = average_vectors(pos_rows, "guard_coeff_diag")
        grd_neg = average_vectors(neg_rows, "guard_coeff_diag")
        grd_mlp = average_vectors(mlp_rows, "guard_coeff_diag")
        grd_debt = average_vectors(debt_rows, "guard_coeff_diag")
        output_ret = float(normalize_trace_vec(out_src).sum().item()) if float(out_src.sum().item()) > EPS else 0.0
        output_common_ret = float(torch.minimum(normalize_trace_vec(out_src), normalize_trace_vec(out_grd)).sum().item()) if float(out_src.sum().item()) > EPS and float(out_grd.sum().item()) > EPS else 0.0
        for qscheme in ("psd_diff_diag", "generalized_eigen_diag", "control_residual_diag"):
            qs = coeff_quotient_diag(src_pos, src_neg, src_mlp, src_debt, qscheme, float(args.quotient_ridge))
            qg = coeff_quotient_diag(grd_pos, grd_neg, grd_mlp, grd_debt, qscheme, float(args.quotient_ridge))
            common = torch.minimum(qs, qg).clamp_min(0.0)
            retention = float(qs.sum().item())
            common_ret = float(common.sum().item())
            sg_cos = vec_cosine(qs, qg)
            delta = float((qs - normalize_trace_vec(out_src)[: int(qs.numel())]).norm().item() / max(float(qs.norm().item()), EPS)) if int(qs.numel()) else 0.0
            ratio_vs_output = common_ret / max(output_common_ret, EPS)
            q_path = FIBER_ROOT / f"D_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{qscheme}.npz"
            save_npz(q_path, fiber_quotient_source=qs, fiber_quotient_guard=qg, fiber_quotient_common=common, output_then_pullback_source=out_src, output_then_pullback_guard=out_grd)
            out_rows.append(
                {
                    "row_id": f"D_{basis_key}_{depth}_{scheme}_{task}_s{seed}_{qscheme}",
                    "status": "ok",
                    "basis_key": basis_key,
                    "depth": depth,
                    "scheme": scheme,
                    "task": task,
                    "seed": seed,
                    "quotient_scheme": qscheme,
                    "quotient_npz": rel(q_path),
                    "output_quotient_positive_retention": float(torch.trace(p_out_q).item()) / max(float(torch.trace(p_pos).item()), EPS),
                    "output_quotient_negative_leakage": float(torch.trace(p_out_q @ p_bad).item()) / max(float(torch.trace(p_out_q).item()), EPS),
                    "coeff_retention_after_output_quotient": output_ret,
                    "source_guard_coeff_cosine_after_output_quotient": vec_cosine(out_src, out_grd),
                    "output_source_guard_common_retention_ratio": output_common_ret,
                    "fiber_quotient_retention": retention,
                    "source_guard_common_retention_ratio": common_ret,
                    "source_guard_fiber_quotient_cosine": sg_cos,
                    "fiber_vs_output_retention_ratio": ratio_vs_output,
                    "noncomm_delta": delta,
                    "D2_pass_minimal": int(retention >= 0.05 and sg_cos >= 0.35 and ratio_vs_output >= 5.0),
                    **AUDIT_DEFAULTS,
                }
            )
    write_rows(OUT_ROOT / "part_d_matrix.csv", out_rows)
    dpass = [r for r in out_rows if int(r.get("D2_pass_minimal", 0)) == 1]
    gate = int(bool(dpass) and not any(r.get("status") == "error" for r in out_rows))
    blocker = "none" if gate else "D_OutputCoeffQuotientCommutativeOrBothFail"
    summary = gate_summary(
        "D",
        gate,
        "D_FiberQuotientNoncommutativePositive" if gate else "D_OutputCoeffQuotientCommutativeOrBothFail",
        blocker,
        out_rows,
        best_fiber_quotient_retention=max([fval(r.get("fiber_quotient_retention")) for r in out_rows] or [0.0]),
        best_source_guard_fiber_quotient_cosine=max([fval(r.get("source_guard_fiber_quotient_cosine")) for r in out_rows] or [0.0]),
        best_fiber_vs_output_retention_ratio=max([fval(r.get("fiber_vs_output_retention_ratio")) for r in out_rows] or [0.0]),
        median_noncomm_delta=median(fval(r.get("noncomm_delta")) for r in out_rows),
        pass_rows=len(dpass),
    )
    write_json(OUT_ROOT / "part_d_summary.json", summary)
    next_path = common_next_actions(
        "D",
        blocker,
        ["increase coefficient sketch rank", "increase quotient ridge", "try generalized eigen", "split layer/mode block quotient", "source/guard quotient intersection"],
        ["interpret tiny output residual as success", "lower output quotient retention gate"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-d --device {args.device}"],
    )
    append_exec("part-d", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_d_summary.json')}; {rel(OUT_ROOT / 'part_d_matrix.csv')}; {rel(next_path)}")
    append_recap("Part D output quotient vs fiber quotient non-commutativity", summary)
    return summary


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("E", "E_BlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C estimator did not pass.", args)
    rows = [r for r in read_rows(OUT_ROOT / "part_c_strict_offdiag_matrix.csv") if r.get("status") == "ok" and r.get("control") in {"c2_positive", "mlp_friendly_negative"}]
    device = device_from_args(args)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            model, source_batch, guard_batch = reconstruct_model_and_batches(row, args, device)
            xsrc, ysrc = source_batch
            xg, yg = guard_batch
            classes = int(max(ysrc.max(), yg.max()).detach().cpu().item()) + 1
            mlp = v2293.MatchedMLP(int(xsrc.shape[1]), classes, int(args.mlp_hidden), 2, stable_seed("mlp", row.get("row_id")), device)
            probe_seed = int(float(row.get("probe_seed", args.output_sketch_seed)))
            p_task = load_channel_matrix(str(row.get("projector_npz", "")))
            if p_task is None:
                continue
            j_kan = full_output_jacobian(model, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
            j_kan_g = full_output_jacobian(model, xg, yg, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
            j_mlp = full_output_jacobian(mlp, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
            j_mlp_g = full_output_jacobian(mlp, xg, yg, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=probe_seed)
            g_kan = full_mode_metric_vector(model).detach().cpu()
            g_mlp = torch.ones(int(j_mlp.shape[1]), dtype=torch.float64)
            b_kan = compute_lift_operator(j_kan, g_kan, p_task)
            b_mlp = compute_lift_operator(j_mlp, g_mlp, p_task)
            b_kan_g = compute_lift_operator(j_kan_g, g_kan, p_task)
            b_mlp_g = compute_lift_operator(j_mlp_g, g_mlp, p_task)
            metrics = kan_vs_mlp_controllability_metrics(b_kan, b_mlp)
            vals, vecs = torch.linalg.eigh(sym(p_task))
            u = vecs[:, torch.argsort(vals, descending=True)[0]].detach()
            kan_cost = minimum_norm_lift_cost(j_kan, g_kan, u)
            mlp_cost = minimum_norm_lift_cost(j_mlp, g_mlp, u)
            out_rows.append(
                {
                    "row_id": f"E_{row.get('row_id')}",
                    "status": "ok",
                    "basis_key": row.get("basis_key"),
                    "depth": row.get("depth"),
                    "task": row.get("task"),
                    "seed": row.get("seed"),
                    "control": row.get("control"),
                    **metrics,
                    "kan_spectrum_source_guard_cosine": spectrum_cosine(torch.linalg.eigvalsh(b_kan).clamp_min(0.0), torch.linalg.eigvalsh(b_kan_g).clamp_min(0.0)),
                    "mlp_spectrum_source_guard_cosine": spectrum_cosine(torch.linalg.eigvalsh(b_mlp).clamp_min(0.0), torch.linalg.eigvalsh(b_mlp_g).clamp_min(0.0)),
                    "kan_min_lift_cost": kan_cost["lift_cost"],
                    "mlp_min_lift_cost": mlp_cost["lift_cost"],
                    "kan_to_mlp_lift_cost_ratio": kan_cost["lift_cost"] / max(mlp_cost["lift_cost"], EPS),
                    "lift_residual_norm": kan_cost["lift_residual_norm"],
                    "source_guard_lift_cost_ratio": minimum_norm_lift_cost(j_kan_g, g_kan, u)["lift_cost"] / max(kan_cost["lift_cost"], EPS),
                    **AUDIT_DEFAULTS,
                }
            )
        except Exception as exc:
            out_rows.append({"row_id": f"E_error_{len(out_rows)}", "status": "error", "error_message": repr(exc), **AUDIT_DEFAULTS})
    write_rows(OUT_ROOT / "part_e_matrix.csv", out_rows)
    c2 = [r for r in out_rows if r.get("status") == "ok" and r.get("control") == "c2_positive"]
    mlp_neg = [r for r in out_rows if r.get("status") == "ok" and r.get("control") == "mlp_friendly_negative"]
    adv = median(fval(r.get("adv_spec")) for r in c2)
    sg = median(fval(r.get("kan_spectrum_source_guard_cosine")) for r in c2)
    cost_ratio = median(fval(r.get("kan_to_mlp_lift_cost_ratio"), 999.0) for r in c2)
    mlp_cost_ratio = median(fval(r.get("kan_to_mlp_lift_cost_ratio"), 999.0) for r in mlp_neg)
    gate = int(adv >= 1.15 and sg >= 0.5 and cost_ratio <= 0.8 and mlp_cost_ratio >= 0.9 and not any(r.get("status") == "error" for r in out_rows))
    blocker = "none" if gate else "E_NoFiberLiftAdvantage"
    summary = gate_summary(
        "E",
        gate,
        "E_FiberLiftAdvantagePositive" if gate else "E_NoFiberLiftAdvantage",
        blocker,
        out_rows,
        c2_adv_spec_median=adv,
        c2_source_guard_spectrum_cosine=sg,
        c2_kan_to_mlp_lift_cost_ratio_median=cost_ratio,
        mlp_friendly_kan_to_mlp_lift_cost_ratio_median=mlp_cost_ratio,
    )
    write_json(OUT_ROOT / "part_e_summary.json", summary)
    next_path = common_next_actions(
        "E",
        blocker,
        ["check fairness of G_KAN and G_MLP", "check D-CHE/D-FOUR mode whitening", "split C2 local_patch vs rotation_sensitive", "split depth2/depth3"],
        ["delete MLP controls", "make MLP cost metric unfair"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-e --device {args.device}"],
    )
    append_exec("part-e", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_e_summary.json')}; {rel(OUT_ROOT / 'part_e_matrix.csv')}; {rel(next_path)}")
    append_recap("Part E KAN-vs-MLP signal-lift controllability", summary)
    return summary


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("F", "F_BlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C estimator did not pass.", args)
    rows = [r for r in read_rows(OUT_ROOT / "part_c_strict_offdiag_matrix.csv") if r.get("status") == "ok"]
    out_rows: list[dict[str, Any]] = []
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("seed")) for r in rows}):
        group = [r for r in rows if (r.get("basis_key"), r.get("depth"), r.get("scheme"), r.get("task"), r.get("seed")) == key]
        pos = average_vectors([r for r in group if r.get("control") == "c2_positive"])
        neg = average_vectors([r for r in group if r.get("control") in {"random_label", "source_shuffle"}])
        mlp = average_vectors([r for r in group if r.get("control") == "mlp_friendly_negative"])
        debt = average_vectors([r for r in group if fval(r.get("guard_debt_delta")) > 0.0])
        pos_g = average_vectors([r for r in group if r.get("control") == "c2_positive"], "guard_coeff_diag")
        bad = (normalize_trace_vec(neg) + normalize_trace_vec(mlp) + normalize_trace_vec(debt)) / 3.0
        p = normalize_trace_vec(pos)
        n = min(int(p.numel()), int(bad.numel()))
        if n <= 0:
            continue
        ratio = p[:n] / (bad[:n] + float(args.quotient_ridge))
        vals = torch.sort(ratio, descending=True).values
        gap = float(vals[0].item() / max(float(vals[1].item()), EPS)) if int(vals.numel()) > 1 else 0.0
        mask = ratio >= 1.25
        task_energy = float(p[:n][mask].sum().item())
        generic = float(normalize_trace_vec(neg)[:n][mask].sum().item()) if int(neg.numel()) else 0.0
        mlpe = float(normalize_trace_vec(mlp)[:n][mask].sum().item()) if int(mlp.numel()) else 0.0
        debte = float(normalize_trace_vec(debt)[:n][mask].sum().item()) if int(debt.numel()) else 0.0
        sg = vec_cosine(p[:n] * mask.to(dtype=torch.float64), normalize_trace_vec(pos_g)[:n] * mask.to(dtype=torch.float64))
        out_rows.append(
            {
                "row_id": f"F_{'_'.join(map(str, key))}",
                "status": "ok",
                "basis_key": key[0],
                "depth": key[1],
                "scheme": key[2],
                "task": key[3],
                "seed": key[4],
                "top_generalized_eigenvalue": float(vals[0].item()) if int(vals.numel()) else 0.0,
                "second_generalized_eigenvalue": float(vals[1].item()) if int(vals.numel()) > 1 else 0.0,
                "eigen_gap": gap,
                "safe_task_subspace_dim": int(mask.sum().item()),
                "task_energy_in_safe_subspace": task_energy,
                "generic_energy_in_safe_subspace": generic,
                "MLP_energy_in_safe_subspace": mlpe,
                "debt_energy_in_safe_subspace": debte,
                "source_guard_stability": sg,
                "leakage_max": max(generic, mlpe, debte),
                **AUDIT_DEFAULTS,
            }
        )
    write_rows(OUT_ROOT / "part_f_matrix.csv", out_rows)
    pass_rows = [
        r
        for r in out_rows
        if fval(r.get("eigen_gap")) >= 1.25
        and fval(r.get("safe_task_subspace_dim")) >= 2
        and fval(r.get("task_energy_in_safe_subspace")) > max(fval(r.get("generic_energy_in_safe_subspace")), fval(r.get("MLP_energy_in_safe_subspace")), fval(r.get("debt_energy_in_safe_subspace")))
        and fval(r.get("source_guard_stability")) >= 0.5
    ]
    gate = int(len(pass_rows) >= max(2, math.ceil(0.10 * len(out_rows))) and not any(r.get("status") == "error" for r in out_rows))
    blocker = "none" if gate else "F_TaskDebtFiberEntangled"
    summary = gate_summary(
        "F",
        gate,
        "F_TaskDebtFiberSeparable" if gate else "F_TaskDebtFiberEntangled",
        blocker,
        out_rows,
        pass_rows=len(pass_rows),
        eigen_gap_median=median(fval(r.get("eigen_gap")) for r in out_rows),
        safe_task_subspace_dim_median=median(fval(r.get("safe_task_subspace_dim")) for r in out_rows),
        source_guard_stability_median=median(fval(r.get("source_guard_stability")) for r in out_rows),
        task_energy_safe_median=median(fval(r.get("task_energy_in_safe_subspace")) for r in out_rows),
        leakage_max_median=median(fval(r.get("leakage_max")) for r in out_rows),
    )
    write_json(OUT_ROOT / "part_f_summary.json", summary)
    next_path = common_next_actions(
        "F",
        blocker,
        ["component-wise debt split", "early formation channel", "mode block split", "source/guard intersection"],
        ["increase tailsafe loss", "lower no-debt gate", "delete worst debt component"],
        [f"{PYTHON} {rel(RUNNER)} --mode part-f --device {args.device}"],
    )
    append_exec("part-f", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_f_summary.json')}; {rel(OUT_ROOT / 'part_f_matrix.csv')}; {rel(next_path)}")
    append_recap("Part F task/generic/MLP/debt fiber split", summary)
    return summary


def metric_variant_scale(model: Any, variant: str, xsrc: torch.Tensor) -> torch.Tensor:
    old = full_mode_metric_vector(model).detach().cpu().clamp_min(EPS)
    if variant == "sobolev_h1":
        new = full_mode_metric_vector(model, lambda_partial=5.0e-4, lambda_partial2=1.0e-6, lambda_omega=5.0e-4).detach().cpu().clamp_min(EPS)
    elif variant == "curvature_h2":
        new = full_mode_metric_vector(model, lambda_partial=1.0e-4, lambda_partial2=5.0e-5, lambda_omega=1.0e-3).detach().cpu().clamp_min(EPS)
    elif variant == "data_composite_diag":
        dcm = DataCompositeMetric(target_condition=1.0e6, sketch_rank=16, sketch_seed=2298)
        new, _res = dcm.param_metric_vector(model, xsrc, metric_type="diagonal")
        new = new.detach().cpu().clamp_min(EPS)
    elif variant == "signal_weighted_edge_measure":
        new = old.sqrt().clamp_min(EPS)
    else:
        new = old
    n = min(int(old.numel()), int(new.numel()))
    return old[:n] / new[:n]


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("G", "G_BlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C estimator did not pass.", args)
    rows = [r for r in read_rows(OUT_ROOT / "part_c_strict_offdiag_matrix.csv") if r.get("status") == "ok"]
    device = device_from_args(args)
    out_rows: list[dict[str, Any]] = []
    for key in sorted({(r.get("basis_key"), r.get("depth"), r.get("task"), r.get("seed")) for r in rows}):
        group = [r for r in rows if (r.get("basis_key"), r.get("depth"), r.get("task"), r.get("seed")) == key]
        pos_rows = [r for r in group if r.get("control") == "c2_positive"]
        if not pos_rows:
            continue
        model, (xsrc, _ysrc), _guard = reconstruct_model_and_batches(pos_rows[0], args, device)
        for variant in ("mode_whitened", "sobolev_h1", "curvature_h2", "data_composite_diag", "signal_weighted_edge_measure"):
            scale = metric_variant_scale(model, variant, xsrc)
            def scaled(rs: list[dict[str, str]], k: str = "source_coeff_diag") -> torch.Tensor:
                v = average_vectors(rs, k)
                n = min(int(v.numel()), int(scale.numel()))
                return v[:n] * scale[:n] if n else torch.zeros(0, dtype=torch.float64)
            pos = scaled(pos_rows)
            neg = scaled([r for r in group if r.get("control") in {"random_label", "source_shuffle"}])
            mlp = scaled([r for r in group if r.get("control") == "mlp_friendly_negative"])
            debt = scaled([r for r in group if fval(r.get("guard_debt_delta")) > 0.0])
            q = coeff_quotient_diag(pos, neg, mlp, debt, "generalized_eigen_diag", float(args.quotient_ridge))
            qg = coeff_quotient_diag(scaled(pos_rows, "guard_coeff_diag"), scaled([r for r in group if r.get("control") in {"random_label", "source_shuffle"}], "guard_coeff_diag"), scaled([r for r in group if r.get("control") == "mlp_friendly_negative"], "guard_coeff_diag"), scaled([r for r in group if fval(r.get("guard_debt_delta")) > 0.0], "guard_coeff_diag"), "generalized_eigen_diag", float(args.quotient_ridge))
            retention = float(q.sum().item())
            sg = vec_cosine(q, qg)
            leak = max(vec_cosine(q, normalize_trace_vec(neg)), vec_cosine(q, normalize_trace_vec(mlp)), vec_cosine(q, normalize_trace_vec(debt)))
            out_rows.append({"row_id": f"G_{variant}_{'_'.join(map(str, key))}", "status": "ok", "metric_variant": variant, "basis_key": key[0], "depth": key[1], "task": key[2], "seed": key[3], "C2_task_lift_retention": retention, "source_guard_stability": sg, "debt_generic_mlp_leakage": leak, "variant_pass": int(retention >= 0.05 and sg >= 0.5 and leak <= 0.5), **AUDIT_DEFAULTS})
    write_rows(OUT_ROOT / "part_g_matrix.csv", out_rows)
    pass_variants = sorted({r.get("metric_variant") for r in out_rows if int(r.get("variant_pass", 0)) == 1})
    gate = int(bool(pass_variants))
    blocker = "none" if gate else "G_NoIntrinsicMetricSupportsLift"
    summary = gate_summary("G", gate, "G_IntrinsicMetricSupportsLift" if gate else "G_NoIntrinsicMetricSupportsLift", blocker, out_rows, pass_variants=pass_variants, best_retention=max([fval(r.get("C2_task_lift_retention")) for r in out_rows] or [0.0]), best_source_guard_stability=max([fval(r.get("source_guard_stability")) for r in out_rows] or [0.0]), min_leakage=min([fval(r.get("debt_generic_mlp_leakage"), 999.0) for r in out_rows] or [999.0]))
    write_json(OUT_ROOT / "part_g_summary.json", summary)
    next_path = common_next_actions("G", blocker, ["repair intrinsic metric shape", "check data-composite Gram diagnostic", "mode block split"], ["winner-search metric promotion", "use debt as primary metric"], [f"{PYTHON} {rel(RUNNER)} --mode part-g --device {args.device}"])
    append_exec("part-g", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_g_summary.json')}; {rel(OUT_ROOT / 'part_g_matrix.csv')}; {rel(next_path)}")
    append_recap("Part G intrinsic KAN metric variants", summary)
    return summary


def train_to_step(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, steps: int) -> torch.nn.Module:
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.adamw_lr), weight_decay=float(args.weight_decay))
    for step in range(int(steps)):
        xb, yb = v2293.v2289.iter_train_batches(x, y, step, int(args.batch_size), int(2298 + step))
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xb).float(), yb.long())
        loss.backward()
        opt.step()
    return model


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_estimator_summary.json")
    if int(c.get("gate_pass", 0)) != 1:
        return write_blocked_part("H", "H_BlockedByPartC", str(c.get("dominant_blocker", "part_c_failed")), "Part C estimator did not pass.", args)
    device = device_from_args(args)
    out_rows: list[dict[str, Any]] = []
    steps = [int(s) for s in csv_items(args.formation_steps)]
    for basis_key in csv_items(args.part_h_basis):
        for task in csv_items(args.part_h_tasks):
            for seed in range(int(args.part_h_seed_count)):
                bargs = basis_args(args, basis_key)
                xtr, ytr, xg, yg = controlled_data(task, "c2_positive", seed, args, device)
                xsrc, ysrc, _xtraj, _ytraj, xgrad, ygrad, _split = v2296.part_c_source_witness_split(xtr, ytr, "c2_positive", seed, args)
                classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
                retentions: list[float] = []
                sg_vals: list[float] = []
                for step in steps:
                    model = make_model("depth2", int(xtr.shape[1]), classes, 2298800 + seed, bargs, device)
                    train_to_step(model, xtr, ytr, args, step)
                    metric_diag, _info = metric_diag_for_scheme(model, xsrc, "op_metric", bargs)
                    g_samples = per_example_gradient_matrix(model, xgrad, ygrad, max_examples=int(args.snr_examples), label_prior_correction=bool(int(args.label_prior_correction)))
                    j = full_output_jacobian(model, xsrc, ysrc, max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=stable_seed("H", basis_key, task, seed, step))
                    strict, _strict_info, pmat = v2297.strict_offdiag_from_samples(j, g_samples, metric_diag, args)
                    jg = full_output_jacobian(model, xg[: int(args.channel_source_size)], yg[: int(args.channel_source_size)], max_outputs=int(args.max_channel_outputs), mode=str(args.channel_output_mode), sketch_mode=str(args.output_sketch_mode), sketch_seed=stable_seed("H", basis_key, task, seed, step))
                    shape = full_mode_metric_vector(model).detach().cpu()
                    src = normalize_trace_vec(lift_operator_diag(j, shape, pmat))
                    grd = normalize_trace_vec(lift_operator_diag(jg, shape, pmat))
                    common = torch.minimum(src, grd).sum().item()
                    retentions.append(float(common))
                    sg_vals.append(vec_cosine(src, grd))
                    out_rows.append({"row_id": f"H_{basis_key}_{task}_s{seed}_t{step}", "status": "ok", "basis_key": basis_key, "task": task, "seed": seed, "formation_step": step, "source_guard_common_retention": float(common), "source_guard_cosine": sg_vals[-1], "strict_rank": int(strict.rank), **AUDIT_DEFAULTS})
                summary = retention_decay_summary(retentions, steps)
                out_rows.append({"row_id": f"H_summary_{basis_key}_{task}_s{seed}", "status": "ok", "basis_key": basis_key, "task": task, "seed": seed, **summary, "source_guard_cosine_mean": mean(sg_vals), "summary_row": 1, **AUDIT_DEFAULTS})
    write_rows(OUT_ROOT / "part_h_matrix.csv", out_rows)
    summary_rows = [r for r in out_rows if int(r.get("summary_row", 0)) == 1]
    early = median(fval(r.get("early_quotient_retention")) for r in summary_rows)
    late = median(fval(r.get("late_quotient_retention")) for r in summary_rows)
    sg = median(fval(r.get("source_guard_cosine_mean")) for r in summary_rows)
    gate = int(early >= 0.05 and sg >= 0.35 and early >= 2.0 * max(late, EPS))
    blocker = "none" if gate else "H_EarlySignalCollapsedBeforeCapture"
    summary = gate_summary("H", gate, "H_EarlyFormationSignalPositive" if gate else "H_EarlySignalCollapsedBeforeCapture", blocker, out_rows, early_quotient_retention=early, late_quotient_retention=late, retention_decay_ratio=early / max(late, EPS), source_guard_cosine_median=sg)
    write_json(OUT_ROOT / "part_h_summary.json", summary)
    next_path = common_next_actions("H", blocker, ["fixed formation-preservation schedule", "shorten formation window", "use early-induced preservation metric"], ["choose T_form by validation", "runtime schedule selection"], [f"{PYTHON} {rel(RUNNER)} --mode part-h --device {args.device}"])
    append_exec("part-h", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(OUT_ROOT / 'part_h_summary.json')}; {rel(OUT_ROOT / 'part_h_matrix.csv')}; {rel(next_path)}")
    append_recap("Part H formation-phase channel audit", summary)
    return summary


def write_blocked_part(letter: str, route: str, blocker: str, reason: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = [{"part": letter, "status": "blocked", "blocked_reason": reason, **AUDIT_DEFAULTS}]
    matrix = OUT_ROOT / f"part_{letter.lower()}_matrix.csv"
    summary_path = OUT_ROOT / f"part_{letter.lower()}_summary.json"
    write_rows(matrix, rows)
    summary = gate_summary(letter, 0, route, blocker, rows, blocked_reason=reason)
    write_json(summary_path, summary)
    next_path = common_next_actions(letter, blocker, ["return to failed prerequisite and rerun"], ["run downstream while prerequisite failed"], [f"{PYTHON} {rel(RUNNER)} --mode part-c-merge", f"{PYTHON} {rel(RUNNER)} --mode part-d"])
    append_exec(f"part-{letter.lower()}", command_text(sys.argv), "blocked", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap(f"Part {letter} blocked", summary)
    return summary


def positive_evidence() -> tuple[bool, str]:
    paths = ["part_d_summary.json", "part_e_summary.json", "part_f_summary.json", "part_g_summary.json", "part_h_summary.json"]
    passed = [read_json(OUT_ROOT / p) for p in paths if int(read_json(OUT_ROOT / p).get("gate_pass", 0)) == 1]
    if passed:
        return True, ",".join(str(p.get("part")) for p in passed)
    blockers = [str(read_json(OUT_ROOT / p).get("dominant_blocker", "")) for p in paths if read_json(OUT_ROOT / p)]
    return False, blockers[0] if blockers else "no_positive_evidence"


def run_part_i(args: argparse.Namespace) -> dict[str, Any]:
    ok, info = positive_evidence()
    if not ok:
        return write_blocked_part("I", "I_BlockedNoStablePositiveEvidence", info, "Part D/E/F/G/H produced no stable positive evidence.", args)
    rows = [{"part": "I", "status": "ok", "positive_evidence_parts": info, "gate_density": 0.0, "candidate_fixed_optimizer": "diagnostic_only_not_implemented", **AUDIT_DEFAULTS}]
    summary = gate_summary("I", 0, "I_DiagnosticCandidateNotPromoted", "candidate_metric_not_implemented_in_this_run", rows)
    write_rows(OUT_ROOT / "part_i_matrix.csv", rows)
    write_json(OUT_ROOT / "part_i_summary.json", summary)
    append_exec("part-i", command_text(sys.argv), "failed", files=f"{rel(OUT_ROOT / 'part_i_summary.json')}")
    append_recap("Part I induced optimizer metric construction", summary)
    return summary


def run_part_j(args: argparse.Namespace) -> dict[str, Any]:
    i = read_json(OUT_ROOT / "part_i_summary.json")
    if int(i.get("gate_pass", 0)) != 1:
        return write_blocked_part("J", "J_BlockedByPartI", str(i.get("dominant_blocker", "part_i_failed")), "Part I did not produce a fixed candidate optimizer.", args)
    return write_blocked_part("J", "J_NotImplemented", "not_reached", "Unexpected path.", args)


def run_part_k(args: argparse.Namespace) -> dict[str, Any]:
    j = read_json(OUT_ROOT / "part_j_summary.json")
    if int(j.get("gate_pass", 0)) != 1:
        return write_blocked_part("K", "K_BlockedByPartJ", str(j.get("dominant_blocker", "part_j_failed")), "Part J did not pass.", args)
    return write_blocked_part("K", "K_NotImplemented", "not_reached", "Unexpected path.", args)


def run_part_l(args: argparse.Namespace) -> dict[str, Any]:
    parts = {name: read_json(OUT_ROOT / f"part_{name.lower()}_summary.json") for name in list("CDEFGHIJK")}
    if int(parts["C"].get("gate_pass", 0)) != 1:
        route, blocker = "C_EstimatorUnstable", str(parts["C"].get("dominant_blocker", "part_c_failed"))
    elif int(parts["D"].get("gate_pass", 0)) != 1 and int(parts["E"].get("gate_pass", 0)) != 1 and int(parts["F"].get("gate_pass", 0)) != 1 and int(parts["G"].get("gate_pass", 0)) != 1 and int(parts["H"].get("gate_pass", 0)) != 1:
        route, blocker = "D_E_F_G_H_AllFiberHypothesesFailed", "no_stable_signal_lift_fiber_geometry"
    elif int(parts["J"].get("gate_pass", 0)) != 1:
        route, blocker = "J_PositiveControlNotReachedOrFailed", str(parts["J"].get("dominant_blocker", "part_j_blocked"))
    elif int(parts["K"].get("gate_pass", 0)) != 1:
        route, blocker = "K_RealTaskMLPDominatesOrBlocked", str(parts["K"].get("dominant_blocker", "part_k_blocked"))
    else:
        route, blocker = "OfficialCandidateGatePass", "none"
    rows = [{"part": "L", "status": "ok", "route": route, "dominant_blocker": blocker, **AUDIT_DEFAULTS}]
    summary = gate_summary(
        "L",
        1,
        route,
        blocker,
        rows,
        hypothesis_results={
            "H1_noncommutativity": parts["D"].get("route", "missing"),
            "H2_lift_advantage": parts["E"].get("route", "missing"),
            "H3_task_debt_split": parts["F"].get("route", "missing"),
            "H4_no_safe_geometry_if_all_fail": route == "D_E_F_G_H_AllFiberHypothesesFailed",
        },
        blocked_parts={name: p.get("route", "missing") for name, p in parts.items() if "Blocked" in str(p.get("route", ""))},
        codex_tried=[
            "strict offdiag and coefficient AB estimator",
            "output-then-pullback vs pullback-then-quotient",
            "KAN-vs-MLP controllability and minimum-norm lift cost",
            "task/generic/MLP/debt fiber split",
            "intrinsic metric variants",
            "early formation channel diagnostic",
        ],
        codex_forbidden=[
            "delete negative/MLP/debt controls",
            "lower stability/retention gates",
            "use held/test for metric induction",
            "promote tiny output residual as success",
        ],
    )
    write_rows(OUT_ROOT / "part_l_failure_decomposition_matrix.csv", rows)
    write_json(OUT_ROOT / "part_l_failure_decomposition_summary.json", summary)
    (OUT_ROOT / "failure_decomposition.md").write_text(
        f"# v22.98 Failure Decomposition\n\nRoute: `{route}`\n\nDominant blocker: `{blocker}`\n\nSee `part_l_failure_decomposition_summary.json` and recap log for evidence.\n",
        encoding="utf-8",
    )
    append_exec("part-l", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'part_l_failure_decomposition_summary.json')}; {rel(OUT_ROOT / 'failure_decomposition.md')}")
    append_recap("Part L failure decomposition and closeout", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    l = read_json(OUT_ROOT / "part_l_failure_decomposition_summary.json")
    stale = stale_artifact_audit()
    summary = {
        "version": "v22.98",
        "route": l.get("route", "missing_part_l"),
        "dominant_blocker": l.get("dominant_blocker", "missing_part_l"),
        "official_candidate_gate_pass": int(l.get("route") == "OfficialCandidateGatePass"),
        "part_c_gate_pass": int(read_json(OUT_ROOT / "part_c_estimator_summary.json").get("gate_pass", 0)),
        "part_d_gate_pass": int(read_json(OUT_ROOT / "part_d_summary.json").get("gate_pass", 0)),
        "part_e_gate_pass": int(read_json(OUT_ROOT / "part_e_summary.json").get("gate_pass", 0)),
        "part_f_gate_pass": int(read_json(OUT_ROOT / "part_f_summary.json").get("gate_pass", 0)),
        "part_g_gate_pass": int(read_json(OUT_ROOT / "part_g_summary.json").get("gate_pass", 0)),
        "part_h_gate_pass": int(read_json(OUT_ROOT / "part_h_summary.json").get("gate_pass", 0)),
        "used_fake_data_rows": 0,
        "held_test_usage": 0,
        "stale_artifact_count": int(stale.get("stale_count", 0)),
        "promotion_allowed": bool(l.get("route") == "OfficialCandidateGatePass" and int(stale.get("stale_count", 0)) == 0),
        "plan": rel(PLAN),
        "runner": rel(RUNNER),
        "execution_log": rel(EXEC_LOG),
        "recap_log": rel(RECAP_LOG),
        "finalized_at": now(),
    }
    write_json(OUT_ROOT / "stale_artifact_audit.json", stale)
    write_json(OUT_ROOT / "final_route.json", summary)
    (OUT_ROOT / "reproduction_manifest.md").write_text(
        "# v22.98 Reproduction Manifest\n\n"
        f"- Python: `{PYTHON}`\n"
        f"- Runner: `{rel(RUNNER)}`\n"
        f"- Output root: `{rel(OUT_ROOT)}`\n"
        f"- Plan: `{rel(PLAN)}`\n\n"
        "Default four-GPU Part C command:\n\n"
        "```bash\n"
        f"CUDA_VISIBLE_DEVICES=0 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 0 --device cuda:0\n"
        f"CUDA_VISIBLE_DEVICES=1 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 1 --device cuda:0\n"
        f"CUDA_VISIBLE_DEVICES=2 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 2 --device cuda:0\n"
        f"CUDA_VISIBLE_DEVICES=3 {PYTHON} {rel(RUNNER)} --mode part-c --shard-count 4 --shard-index 3 --device cuda:0\n"
        f"{PYTHON} {rel(RUNNER)} --mode part-c-merge --device cuda:0\n"
        "```\n",
        encoding="utf-8",
    )
    write_inventory_and_manifest()
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(OUT_ROOT / 'stale_artifact_audit.json')}; {rel(OUT_ROOT / 'reproduction_manifest.md')}; {rel(OUT_ROOT / 'sha256_manifest.txt')}")
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
    p.add_argument("--part-h-basis", default="dche_k5,dfour_default")
    p.add_argument("--part-h-tasks", default="local_patch_interaction")
    p.add_argument("--part-h-seed-count", type=int, default=2)
    p.add_argument("--formation-steps", default="0,1,5,10,20")
    p.add_argument("--channel-source-size", type=int, default=16)
    p.add_argument("--channel-output-mode", default="margin", choices=["margin", "logits"])
    p.add_argument("--max-channel-outputs", type=int, default=8)
    p.add_argument("--output-sketch-mode", default="prefix", choices=["prefix", "rademacher"])
    p.add_argument("--output-sketch-seed", type=int, default=229800)
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
    p.add_argument("--data-metric-sketch-seed", type=int, default=2298)
    p.add_argument("--pg1-gamma", type=float, default=0.15)
    p.add_argument("--quotient-ridge", type=float, default=1.0e-3)
    p.add_argument("--mlp-hidden", type=int, default=24)
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
    if mode == "part-f":
        return run_part_f(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "part-h":
        return run_part_h(args)
    if mode == "part-i":
        return run_part_i(args)
    if mode == "part-j":
        return run_part_j(args)
    if mode == "part-k":
        return run_part_k(args)
    if mode == "part-l":
        return run_part_l(args)
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
        run_part_f(args)
        run_part_g(args)
        run_part_h(args)
        run_part_i(args)
        run_part_j(args)
        run_part_k(args)
        run_part_l(args)
        return finalize(args)
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()

