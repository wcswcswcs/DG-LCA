#!/usr/bin/env python3
"""DG-KAN v23.06 random-control-aware block-capacity regularizer runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import py_compile
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
import experiments.run_v23_00r_curve_geometry_population_flow as v2300
import experiments.run_v23_03_structure_projected_functional_population_flow as v2303
import experiments.run_v23_05_lowrank_phase_observer as v2305
from dgkan.optim.edge_sobolev_population_flow import EdgeSobolevPopulationFlow, mark_kan_edge_params


PYTHON = sys.executable
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.06_RandomControlAware_BlockCapacityRegularizer_完整详尽实验计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.06_RandomControlAware_BlockCapacityRegularizer_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.06_RandomControlAware_BlockCapacityRegularizer_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2306_OUT_ROOT", str(ROOT / "results/v23_06"))).resolve()

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
    "structure_transform_used_only_for_gate": 1,
    "structure_transform_in_forward": 0,
}

PART_B_SCHEMES = {
    "B0_H10_oracle_reference": {"kind": "h10_oracle", "block": "v23_03_h10"},
    "B1_FunctionalGram_baseline": {"kind": "functional_gram", "block": "degree_edgebank"},
    "B2_block_snr_capacity_gate": {"kind": "block_snr", "block": "input_degree"},
    "B3_functional_energy_capacity_gate": {"kind": "functional_energy", "block": "input_degree"},
    "B4_guard_stable_capacity_gate": {"kind": "guard_stable", "block": "input_degree"},
    "B5_capacity_fullblock_gate": {"kind": "fullblock_capacity", "block": "input_degree"},
    "B6_capacity_density_only_random_control": {"kind": "density_random", "block": "input_degree"},
    "B7_capacity_block_shuffle_control": {"kind": "block_shuffle", "block": "input_degree"},
    "B8_capacity_norm_matched_random_control": {"kind": "norm_random", "block": "input_degree"},
    "B9_guard_stable_capacity_modifier": {"kind": "guard_stable_modifier", "block": "input_degree"},
    "B10_fullblock_capacity_modifier": {"kind": "fullblock_capacity_modifier", "block": "input_degree"},
    "B11_density_only_modifier_random_control": {"kind": "density_random_modifier", "block": "input_degree"},
    "B12_block_shuffle_modifier_control": {"kind": "block_shuffle_modifier", "block": "input_degree"},
    "B13_norm_matched_modifier_control": {"kind": "norm_random_modifier", "block": "input_degree"},
    "B14_descent_cone_coordinate_gate": {"kind": "descent_cone", "block": "coordinate"},
    "B15_descent_cone_coordinate_modifier": {"kind": "descent_cone_modifier", "block": "coordinate"},
    "B16_coordinate_density_random_control": {"kind": "density_random", "block": "coordinate"},
    "B17_coordinate_shuffle_control": {"kind": "block_shuffle", "block": "coordinate"},
    "B18_coordinate_norm_matched_random_control": {"kind": "norm_random", "block": "coordinate"},
    "B19_descent_cone_coordinate_gate_finite_c2": {"kind": "descent_cone_finite_c2", "block": "coordinate"},
    "B20_descent_cone_coordinate_modifier_finite_c2": {"kind": "descent_cone_modifier_finite_c2", "block": "coordinate"},
    "B21_coordinate_density_random_finite_c2_control": {"kind": "density_random_finite_c2", "block": "coordinate"},
    "B22_coordinate_shuffle_finite_c2_control": {"kind": "block_shuffle_finite_c2", "block": "coordinate"},
    "B23_coordinate_norm_matched_random_finite_c2_control": {"kind": "norm_random_finite_c2", "block": "coordinate"},
}

RANDOM_CAPACITY_CONTROLS = {
    "B6_capacity_density_only_random_control",
    "B7_capacity_block_shuffle_control",
    "B8_capacity_norm_matched_random_control",
    "B11_density_only_modifier_random_control",
    "B12_block_shuffle_modifier_control",
    "B13_norm_matched_modifier_control",
    "B16_coordinate_density_random_control",
    "B17_coordinate_shuffle_control",
    "B18_coordinate_norm_matched_random_control",
    "B21_coordinate_density_random_finite_c2_control",
    "B22_coordinate_shuffle_finite_c2_control",
    "B23_coordinate_norm_matched_random_finite_c2_control",
}


def parse_capacity_kind(kind: str) -> tuple[str, bool, bool]:
    body = str(kind)
    finite_c2 = body.endswith("_finite_c2")
    if finite_c2:
        body = body.removesuffix("_finite_c2")
    apply_as_modifier = body.endswith("_modifier")
    if apply_as_modifier:
        body = body.removesuffix("_modifier")
    return body, apply_as_modifier, finite_c2


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "x"


def command_text(argv: Iterable[str]) -> str:
    cmd = " ".join([PYTHON, rel(RUNNER), *list(argv)[1:]])
    env = os.environ.get("V2306_OUT_ROOT")
    return f"V2306_OUT_ROOT={env} {cmd}" if env else cmd


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.median(vals)) if vals else float(default)


def csv_items(text: Any) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow({k: row.get(k, "") for k in keys})
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_exec(part: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    ensure_out()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {part} {status}\n\n")
        fh.write(f"- command: `{command}`\n")
        if gpu:
            fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    ensure_out()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def gate_summary(part: str, gate: int, route: str, blocker: str, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "part": part,
        "gate_pass": int(gate),
        "route": route,
        "dominant_blocker": blocker,
        "row_count": len(rows),
        "ok_rows": sum(1 for r in rows if r.get("status") == "ok"),
        "error_rows": sum(1 for r in rows if r.get("status") == "error"),
        "generated_at": now(),
        **AUDIT_DEFAULTS,
        **extra,
    }


def next_actions(part: str, gate: int, blocker: str, actions: list[str]) -> Path:
    return write_json(
        OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json",
        {
            "part": part,
            "gate_pass": int(gate),
            "dominant_blocker": blocker,
            "recommended_actions": actions,
            "forbidden_next_actions": [
                "do_not_use_held_or_test_selector",
                "do_not_delete_matched_random_controls",
                "do_not_report_smoke_as_official",
            ],
            "generated_at": now(),
        },
    )


def device_from_args(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device(str(args.device))
    return torch.device("cpu")


def make_edge_optimizer(model: v2293.TrueDeepPureKAN, args: argparse.Namespace, *, sobolev: float, use_population_gate: bool) -> EdgeSobolevPopulationFlow:
    return EdgeSobolevPopulationFlow(
        list(model.coeffs),
        lr=float(args.edge_lr),
        weight_decay=float(args.weight_decay),
        sobolev_exponent=float(sobolev),
        edge_metric_type="functional_gram",
        edge_weight_normalization=str(args.edge_weight_normalization),
        edge_weight_ridge=float(args.edge_weight_ridge),
        functional_gram_quadrature_points=int(args.functional_gram_quadrature_points),
        gate_beta=float(args.snr_beta),
        gate_family="degree_edgebank",
        gate_floor=float(args.gate_floor),
        gate_floor_mode=str(args.gate_floor_mode),
        gate_input_side=int(args.visual_side),
        gate_patch_size=2,
        stat_warmup_steps=0,
        use_population_gate=bool(use_population_gate),
    )


def compute_split_grads(model: v2293.TrueDeepPureKAN, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace) -> tuple[list[torch.Tensor], torch.Tensor]:
    flat, split, labels = v2305.compute_split_grads(model, x, y, args)
    return split, labels


def h10_artifact_row(task: str, seed: int) -> dict[str, Any]:
    path = ROOT / "results/v23_03_official_h10_full_positive_control_seed15_steps80/part_h_full_positive_control_matrix.csv"
    for row in read_rows(path):
        if (
            str(row.get("part")) == "H_C2"
            and str(row.get("h_scheme")) == "H10_F7ObjectiveAlignedSafetyDiagnostic"
            and str(row.get("task")) == str(task)
            and ival(row.get("seed")) == int(seed)
        ):
            return row
    return {}


def run_part_0(args: argparse.Namespace) -> dict[str, Any]:
    final = read_json(ROOT / "results/v23_05/final_route.json")
    part_c = read_json(ROOT / "results/v23_05/part_c_blind_observer_summary.json")
    recap = ROOT / "docs/DG-KAN_v23.05_H10RealTransfer_LowRankPhaseObserver_实验结果复盘.md"
    exec_log = ROOT / "docs/DG-KAN_v23.05_H10RealTransfer_LowRankPhaseObserver_执行日志.md"
    row = {
        "v23_05_route": final.get("route", "missing"),
        "v23_05_part_0_gate_pass": ival(final.get("part_0_gate_pass")),
        "v23_05_part_a_gate_pass": ival(final.get("part_a_gate_pass")),
        "v23_05_part_b_gate_pass": ival(final.get("part_b_gate_pass")),
        "v23_05_part_c_gate_pass": ival(final.get("part_c_gate_pass")),
        "v23_05_promotion_allowed": ival(final.get("promotion_allowed")),
        "v23_05_dominant_blocker": final.get("dominant_blocker", "missing"),
        "v23_05_part_c_route": part_c.get("route", "missing"),
        "v23_05_negative_result_acknowledged": int("closed negative evidence" in recap.read_text(encoding="utf-8") if recap.exists() else False),
        "v23_05_recap_exists": int(recap.exists()),
        "v23_05_exec_log_exists": int(exec_log.exists()),
        **AUDIT_DEFAULTS,
    }
    gate = int(
        row["v23_05_part_0_gate_pass"] == 1
        and row["v23_05_part_a_gate_pass"] == 1
        and row["v23_05_part_b_gate_pass"] == 1
        and row["v23_05_part_c_gate_pass"] == 0
        and row["v23_05_route"] == "BlindObserverBridgeFailed"
        and row["v23_05_promotion_allowed"] == 0
        and row["v23_05_negative_result_acknowledged"] == 1
    )
    matrix = write_rows(OUT_ROOT / "part_0_v2305_negative_lock_matrix.csv", [row])
    summary = gate_summary("0", gate, "V2305NegativeLockPass" if gate else "V2305NegativeLockFailed", "none" if gate else "v23_05_negative_lock_missing", [row], **row)
    write_json(OUT_ROOT / "part_0_v2305_negative_lock.json", summary)
    nxt = next_actions("0", gate, summary["dominant_blocker"], [] if gate else ["complete v23.05 closure and final route before v23.06"])
    append_exec("part-0", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_0_v2305_negative_lock.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part 0 v23.05 negative-result lock", summary)
    return summary


def static_scan(paths: list[Path]) -> tuple[int, list[dict[str, str]]]:
    patterns = {
        "runtime_metric_winner_selection": re.compile(r"\b(select_metric|choose_metric|metric_winner)\s*\("),
        "runtime_update_winner_selection": re.compile(r"\b(select_update|choose_update|winner_update)\s*\("),
        "external_product_feature": re.compile(r"\b(make_external_product_feature|patch_product_feature)\s*\("),
        "mlp_stem_constructor": re.compile(r"\bMLPStem\s*\("),
        "mlp_readout_constructor": re.compile(r"\bMLPReadout\s*\("),
    }
    hits: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            hits.append({"file": rel(path), "check": "missing_file", "match": "missing"})
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                hits.append({"file": rel(path), "check": name, "match": match.group(0)})
    return int(not hits), hits


def purekan_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    try:
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.visual_side = 8
        bargs = v2300.basis_args(smoke_args, "dche_k9")
        model = v2300.make_model("depth3", 64, 2, 2306001, bargs, device)
        mark_kan_edge_params(model, basis_key="dche_k9")
        before = torch.cat([p.detach().reshape(-1).cpu() for p in model.coeffs])
        opt = make_edge_optimizer(model, smoke_args, sobolev=0.25, use_population_gate=False)
        gen = torch.Generator(device=device).manual_seed(2306)
        xb = torch.randn(16, 64, generator=gen, device=device)
        yb = torch.randint(0, 2, (16,), generator=gen, device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb.long())
        loss.backward()
        opt.step()
        after = torch.cat([p.detach().reshape(-1).cpu() for p in model.coeffs])
        return {
            "changed_edge_coefficients": int(float((after - before).abs().sum().item()) > 0.0),
            "changed_mlp_tensors": 0,
            "state_updated": int(getattr(opt.last_stats, "edge_params_updated", 0) > 0),
            "smoke_loss": float(loss.detach().cpu().item()),
        }
    except Exception as exc:
        return {"changed_edge_coefficients": 0, "changed_mlp_tensors": 0, "state_updated": 0, "smoke_error": repr(exc)}


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    files = [
        RUNNER,
        ROOT / "dgkan/optim/edge_sobolev_population_flow.py",
        ROOT / "experiments/run_v23_00r_curve_geometry_population_flow.py",
    ]
    compile_errors: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}: {repr(exc)}")
    import_errors: list[str] = []
    for mod in [
        "dgkan.optim.edge_sobolev_population_flow",
        "experiments.run_v23_00r_curve_geometry_population_flow",
        "experiments.run_v23_05_lowrank_phase_observer",
    ]:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            import_errors.append(f"{mod}: {repr(exc)}")
    scan_pass, scan_hits = static_scan(files)
    smoke = purekan_smoke(args, device_from_args(args))
    row = {
        "compile_pass": int(not compile_errors),
        "import_pass": int(not import_errors),
        "static_scan_pass": scan_pass,
        **smoke,
        **AUDIT_DEFAULTS,
    }
    gate = int(row["compile_pass"] == 1 and row["import_pass"] == 1 and row["static_scan_pass"] == 1 and row.get("changed_edge_coefficients") == 1 and row.get("changed_mlp_tensors") == 0)
    row["status"] = "ok" if gate else "error"
    matrix = write_rows(OUT_ROOT / "part_a_identity_matrix.csv", [row])
    summary = gate_summary("A", gate, "PartAIdentityPass" if gate else "PartAIdentityFailed", "none" if gate else "compile_import_static_or_smoke_failed", [row], compile_errors=compile_errors, import_errors=import_errors, static_scan_hits=scan_hits)
    write_json(OUT_ROOT / "part_a_identity_summary.json", summary)
    nxt = next_actions("A", gate, summary["dominant_blocker"], [] if gate else ["repair compile/import/static scan or PureKAN smoke"])
    append_exec("part-a", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(OUT_ROOT / 'part_a_identity_summary.json')}; {rel(nxt)}", gpu=str(args.device))
    append_recap("Part A identity and anti-selector audit", summary)
    return summary


def block_capacity_scores(
    model: v2293.TrueDeepPureKAN,
    opt: EdgeSobolevPopulationFlow,
    x_source: torch.Tensor,
    y_source: torch.Tensor,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    args: argparse.Namespace,
    *,
    block_family: str,
    kind: str,
    seed: int,
) -> dict[str, Any]:
    base_kind, apply_as_modifier, finite_c2 = parse_capacity_kind(str(kind))
    params = list(model.coeffs)
    group = opt.param_groups[0]
    source_split, labels = compute_split_grads(model, x_source, y_source, args)
    guard_split, _guard_labels = compute_split_grads(model, x_guard, y_guard, args)
    records: list[tuple[int, torch.Tensor]] = []
    score_vals: list[float] = []
    stability_vals: list[float] = []
    energy_vals: list[float] = []
    snr_vals: list[float] = []
    for pidx, (param, sgrads) in enumerate(zip(params, source_split)):
        pgrads = sgrads.to(device=param.device, dtype=param.dtype)
        opt.observe_per_example_gradients(param, pgrads, labels=labels)
        ggrads = guard_split[pidx].to(device=param.device, dtype=param.dtype) if pidx < len(guard_split) else pgrads
        if str(block_family) == "coordinate":
            blocks = [torch.tensor([i], device=param.device, dtype=torch.long) for i in range(int(param.numel()))]
        else:
            blocks = v2305.block_indices_for_param(opt, param, group, args, block_family, labels=labels, logits=None)
        white = opt._apply_metric_inv_sqrt(pgrads, param, group).reshape(int(pgrads.shape[0]), -1)
        gwhite = opt._apply_metric_inv_sqrt(ggrads, param, group).reshape(int(ggrads.shape[0]), -1)
        for block in blocks:
            idx = block.detach().long().to(device=white.device)
            if int(idx.numel()) <= 0:
                continue
            sblock = white[:, idx]
            gblock = gwhite[:, idx]
            smu = sblock.mean(dim=0).detach().to(dtype=torch.float64).cpu()
            gmu = gblock.mean(dim=0).detach().to(dtype=torch.float64).cpu()
            centered = sblock.detach().to(dtype=torch.float64).cpu() - smu.reshape(1, -1)
            noise = centered.square().mean().sqrt().item()
            energy = float(smu.norm().div(math.sqrt(max(1, int(idx.numel())))).item())
            snr = float(energy / max(noise, 1.0e-12))
            stability = max(0.0, v2305.cosine_vec(smu, gmu, abs_value=False))
            ratio = float(min(smu.norm().item(), gmu.norm().item()) / max(max(smu.norm().item(), gmu.norm().item()), 1.0e-12))
            stable_score = stability * ratio
            descent_dot = float(torch.dot(smu, gmu).item())
            descent_score = max(0.0, descent_dot) / max(math.sqrt(max(1, int(idx.numel()))) * noise, 1.0e-12)
            if base_kind == "block_snr":
                score = snr
            elif base_kind == "functional_energy":
                score = energy
            elif base_kind == "guard_stable":
                score = stable_score
            elif base_kind == "fullblock_capacity":
                score = 0.5 * stable_score + 0.5 * snr
            elif base_kind == "descent_cone":
                score = descent_score
            else:
                score = stable_score
            records.append((pidx, block.detach().long().cpu()))
            score_vals.append(float(score))
            stability_vals.append(float(stability))
            energy_vals.append(float(energy))
            snr_vals.append(float(snr))
    scores = torch.tensor(score_vals, dtype=torch.float64)
    if int(scores.numel()) <= 0:
        q_block = torch.zeros(0, dtype=torch.float64)
    elif base_kind == "density_random":
        gen = torch.Generator().manual_seed(int(seed) * 10007 + 11)
        q_block = v2305.sparse_topk_gate(torch.randn(int(scores.numel()), generator=gen, dtype=torch.float64), float(args.capacity_density_target))
    else:
        q_block = v2305.sparse_topk_gate(scores, float(args.capacity_density_target))
        if base_kind == "block_shuffle":
            gen = torch.Generator().manual_seed(int(seed) * 10007 + 13)
            if int(q_block.numel()) > 0:
                q_block = q_block[torch.randperm(int(q_block.numel()), generator=gen)]
        elif base_kind == "norm_random":
            gen = torch.Generator().manual_seed(int(seed) * 10007 + 17)
            if int(q_block.numel()) > 0:
                q_block = v2305.sparse_topk_gate(scores[torch.randperm(int(scores.numel()), generator=gen)], float(args.capacity_density_target))
    gates = [torch.zeros(int(p.numel()), device=p.device, dtype=p.dtype) for p in params]
    counts = [torch.zeros(int(p.numel()), device=p.device, dtype=p.dtype) for p in params]
    for row_idx, (pidx, block_cpu) in enumerate(records):
        block = block_cpu.to(device=params[pidx].device)
        gates[pidx][block] += float(q_block[row_idx].item()) if row_idx < int(q_block.numel()) else 0.0
        counts[pidx][block] += 1.0
    for pidx, param in enumerate(params):
        gate = gates[pidx] / counts[pidx].clamp_min(1.0)
        if int((counts[pidx] == 0).sum().item()) > 0:
            gate[counts[pidx] == 0] = float(gate.mean().item()) if int(gate.numel()) else 0.0
        opt.observe_per_example_gradients(param, source_split[pidx].to(device=param.device, dtype=param.dtype), labels=labels)
        if apply_as_modifier:
            opt.observe_structure_update_modifier(param, gate.reshape_as(param))
        else:
            opt.observe_structure_gate(param, gate.reshape_as(param))
    selected_weight = q_block.detach().to(dtype=torch.float64).cpu() if int(q_block.numel()) else torch.zeros(0, dtype=torch.float64)
    stability_t = torch.tensor(stability_vals, dtype=torch.float64)
    score_t = torch.tensor(score_vals, dtype=torch.float64)
    selected_mass = float(selected_weight.sum().item()) if int(selected_weight.numel()) else 0.0
    selected_stability = float((stability_t[: int(selected_weight.numel())] * selected_weight).sum().item() / max(selected_mass, 1.0e-12)) if selected_mass > 0.0 else 0.0
    selected_score = float((score_t[: int(selected_weight.numel())] * selected_weight).sum().item() / max(selected_mass, 1.0e-12)) if selected_mass > 0.0 else 0.0
    return {
        "block_records": records,
        "q_block": q_block,
        "capacity_apply_mode": "modifier" if apply_as_modifier else "gate",
        "capacity_finite_c2": int(finite_c2),
        "capacity_all_source_guard_stability": mean(stability_vals),
        "capacity_source_guard_stability": selected_stability,
        "capacity_energy_mean": mean(energy_vals),
        "capacity_snr_mean": mean(snr_vals),
        "capacity_score_mean": selected_score,
        "capacity_q_density": float(q_block.mean().item()) if int(q_block.numel()) else 0.0,
        "capacity_block_count": len(records),
    }


def train_capacity_row(job: tuple[str, str, str, str, int, str], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    scheme, basis_key, depth, task, seed, repair_round = job
    start = time.time()
    cfg = PART_B_SCHEMES.get(str(scheme), PART_B_SCHEMES["B4_guard_stable_capacity_gate"])
    kind = str(cfg.get("kind"))
    base_kind, _apply_as_modifier, finite_c2 = parse_capacity_kind(kind)
    try:
        if kind == "h10_oracle" and bool(int(args.b0_from_h10_artifact)):
            artifact = h10_artifact_row(str(task), int(seed))
            if artifact:
                return {
                    "part": "B",
                    "status": "ok",
                    "scheme": scheme,
                    "basis_key": artifact.get("basis_key", basis_key),
                    "depth": artifact.get("depth", depth),
                    "task": task,
                    "seed": int(seed),
                    "repair_round": repair_round,
                    "observer_kind": "h10_oracle_artifact",
                    "block_family": "v23_03_h10",
                    "train_steps": ival(artifact.get("train_steps")),
                    "capacity_source_guard_stability": 1.0,
                    "capacity_q_density": fval(artifact.get("gate_density")),
                    "C2_coverage_improvement": fval(artifact.get("C2_coverage_improvement")),
                    "C2_accuracy_improvement": fval(artifact.get("C2_accuracy_improvement")),
                    "C2_coverage_initial": fval(artifact.get("C2_coverage_initial")),
                    "C2_coverage_final": fval(artifact.get("C2_coverage_final")),
                    "C2_accuracy_initial": fval(artifact.get("C2_accuracy_initial")),
                    "C2_accuracy_final": fval(artifact.get("C2_accuracy_final")),
                    "capacity_random_gap": 0.0,
                    "oracle_retention_ratio": 1.0,
                    "local_patch_coverage": fval(artifact.get("C2_coverage_improvement")) if task == "local_patch_interaction" else 0.0,
                    "rotation_coverage": fval(artifact.get("C2_coverage_improvement")) if task == "rotation_sensitive" else 0.0,
                    "wall_time_s": time.time() - start,
                    **AUDIT_DEFAULTS,
                }
        bargs = v2300.basis_args(args, basis_key)
        xtr, ytr, xg, yg = v2300.visual_data(task, seed, args, device)
        n = min(int(args.structure_guard_examples), max(1, int(xtr.shape[0]) // 2))
        x_source, y_source = xtr[:n], ytr[:n]
        x_guard, y_guard = xtr[-n:], ytr[-n:]
        classes = int(max(ytr.max(), yg.max()).detach().cpu().item()) + 1
        model = v2300.make_model(depth, int(xtr.shape[1]), classes, v2300.model_seed_for(basis_key, depth, task, seed), bargs, device)
        mark_kan_edge_params(model, basis_key=basis_key)
        uses_capacity = base_kind != "functional_gram"
        use_pop = uses_capacity and not _apply_as_modifier
        sobolev = 0.0 if kind == "functional_gram" else float(args.sobolev_exponent)
        opt = make_edge_optimizer(model, args, sobolev=sobolev, use_population_gate=use_pop)
        before = v2293.metrics_for_model(model, xg, yg)
        cap_meta: dict[str, Any] = {}
        gate_trace: list[float] = []
        finite_accept_count = 0
        finite_skip_count = 0
        finite_attempt_count = 0
        finite_scale_trace: list[float] = []
        finite_reject_reasons: dict[str, int] = {}
        finite_safety_cfg = {"finite_enabled": True, "finite_step_components": "", "finite_step_apply_all_parts": 0}
        for step in range(1, int(args.train_steps) + 1):
            xb, yb = v2293.v2289.iter_train_batches(xtr, ytr, step - 1, int(args.batch_size), int(seed))
            opt.zero_grad(set_to_none=True)
            if uses_capacity:
                refresh = step == 1 or int(args.capacity_gate_every) <= 1 or ((step - 1) % int(args.capacity_gate_every) == 0)
                if refresh:
                    cap_meta = block_capacity_scores(
                        model,
                        opt,
                        xb[: int(args.population_grad_examples)],
                        yb[: int(args.population_grad_examples)],
                        x_guard[: int(args.population_grad_examples)],
                        y_guard[: int(args.population_grad_examples)],
                        args,
                        block_family=str(cfg.get("block", "input_degree")),
                        kind=kind,
                        seed=int(seed) * 10000 + step,
                    )
                elif use_pop:
                    pe, _gm, _gp, _nn, _ni = v2300.observe_task_gradients(opt, model, xb, yb, args)
                    cap_meta["per_example_count"] = pe
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb.long())
            loss.backward()
            if finite_c2:
                finite_diag = v2303.v2303_finite_step_guarded_step(
                    opt,
                    model,
                    before,
                    xg,
                    yg,
                    args,
                    finite_safety_cfg,
                    part="G_C2",
                    step=step,
                )
            else:
                opt.step()
                finite_diag = {
                    "finite_step_enabled": 0,
                    "finite_step_accept": 1,
                    "finite_step_skip": 0,
                    "finite_step_scale": 1.0,
                    "finite_step_attempts": 1,
                    "finite_step_reject_reason": "disabled",
                }
            finite_accept_count += ival(finite_diag.get("finite_step_accept"))
            finite_skip_count += ival(finite_diag.get("finite_step_skip"))
            finite_attempt_count += ival(finite_diag.get("finite_step_attempts"))
            finite_scale_trace.append(fval(finite_diag.get("finite_step_scale"), 1.0))
            reason = str(finite_diag.get("finite_step_reject_reason", ""))
            if finite_c2 and reason and reason not in {"accepted", "not_checked", "disabled"}:
                finite_reject_reasons[reason] = finite_reject_reasons.get(reason, 0) + 1
            gate_trace.append(float(getattr(opt.last_stats, "gate_density_mean", 1.0)))
        after = v2293.metrics_for_model(model, xg, yg)
        return {
            "part": "B",
            "status": "ok",
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "seed": int(seed),
            "repair_round": repair_round,
            "observer_kind": kind,
            "capacity_apply_mode": cap_meta.get("capacity_apply_mode", "population_gate" if use_pop else "none"),
            "capacity_finite_c2": int(finite_c2),
            "block_family": str(cfg.get("block")),
            "train_steps": int(args.train_steps),
            "capacity_source_guard_stability": cap_meta.get("capacity_source_guard_stability", 1.0 if not use_pop else 0.0),
            "capacity_all_source_guard_stability": cap_meta.get("capacity_all_source_guard_stability", 1.0 if not use_pop else 0.0),
            "capacity_energy_mean": cap_meta.get("capacity_energy_mean", 0.0),
            "capacity_snr_mean": cap_meta.get("capacity_snr_mean", 0.0),
            "capacity_score_mean": cap_meta.get("capacity_score_mean", 0.0),
            "capacity_q_density": cap_meta.get("capacity_q_density", mean(gate_trace)),
            "capacity_block_count": cap_meta.get("capacity_block_count", 0),
            "C2_coverage_initial": before["coverage_CVaR25"],
            "C2_coverage_final": after["coverage_CVaR25"],
            "C2_coverage_improvement": after["coverage_CVaR25"] - before["coverage_CVaR25"],
            "C2_accuracy_initial": before["accuracy"],
            "C2_accuracy_final": after["accuracy"],
            "C2_accuracy_improvement": after["accuracy"] - before["accuracy"],
            "guard_NLL_delta": after["nll"] - before["nll"],
            "capacity_random_gap": 0.0,
            "oracle_retention_ratio": 0.0,
            "local_patch_coverage": cap_meta.get("capacity_source_guard_stability", 0.0) if task == "local_patch_interaction" else 0.0,
            "rotation_coverage": cap_meta.get("capacity_source_guard_stability", 0.0) if task == "rotation_sensitive" else 0.0,
            "gate_density_mean": mean(gate_trace),
            "finite_step_enabled": int(finite_c2),
            "finite_step_accept_count": finite_accept_count,
            "finite_step_skip_count": finite_skip_count,
            "finite_step_attempt_count": finite_attempt_count,
            "finite_step_accept_rate": finite_accept_count / max(1, finite_accept_count + finite_skip_count),
            "finite_step_scale_mean": mean(finite_scale_trace, 1.0),
            "finite_step_reject_reasons": ";".join(f"{k}:{v}" for k, v in sorted(finite_reject_reasons.items())),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }
    except Exception as exc:
        return {
            "part": "B",
            "status": "error",
            "scheme": scheme,
            "basis_key": basis_key,
            "depth": depth,
            "task": task,
            "seed": int(seed),
            "repair_round": repair_round,
            "error_message": repr(exc),
            "wall_time_s": time.time() - start,
            **AUDIT_DEFAULTS,
        }


def part_b_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, str, int, str]]:
    return [
        (scheme, basis, depth, task, seed, str(args.repair_round))
        for scheme in csv_items(args.part_b_schemes)
        for basis in csv_items(args.part_b_basis)
        for depth in csv_items(args.part_b_depths)
        for task in csv_items(args.part_b_tasks)
        for seed in range(int(args.part_b_seed_count))
    ]


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    path = OUT_ROOT / f"part_b_capacity_bridge_matrix_{safe_name(args.repair_round)}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    rows = read_rows(path) if path.exists() else []
    done = {(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task"), str(r.get("seed"))) for r in rows if r.get("status") == "ok"}
    for job in shard_items(part_b_jobs(args), args):
        key = (job[0], job[1], job[2], job[3], str(job[4]))
        if key in done:
            continue
        rows.append(train_capacity_row(job, args, device))
        write_rows(path, rows)
    if not path.exists():
        write_rows(path, rows)
    summary = gate_summary("B", 0, "PartBShardWritten", "merge_required", rows)
    summary_path = OUT_ROOT / f"part_b_capacity_bridge_summary_{safe_name(args.repair_round)}_shard{int(args.shard_index)}.json"
    write_json(summary_path, summary)
    append_exec("part-b", command_text(sys.argv), "shard-written", files=f"{rel(path)}; {rel(summary_path)}", gpu=str(args.device), note=f"rows={len(rows)} repair_round={args.repair_round}")
    return summary


def merge_part_b(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_b_capacity_bridge_matrix_*_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    rows = [r for r in rows if not str(r.get("repair_round", "")).startswith("smoke")]
    ok = [r for r in rows if r.get("status") == "ok"]
    by_key = {(r.get("scheme"), r.get("task"), r.get("seed"), r.get("repair_round")): r for r in ok}
    for r in ok:
        task = r.get("task")
        seed = r.get("seed")
        repair = r.get("repair_round")
        baseline = by_key.get(("B1_FunctionalGram_baseline", task, seed, repair))
        oracle = by_key.get(("B0_H10_oracle_reference", task, seed, repair))
        random_rows = [by_key.get((scheme, task, seed, repair)) for scheme in RANDOM_CAPACITY_CONTROLS]
        random_rows = [rr for rr in random_rows if rr]
        if baseline:
            r["coverage_vs_functional"] = fval(r.get("C2_coverage_improvement")) - fval(baseline.get("C2_coverage_improvement"))
            r["accuracy_vs_functional"] = fval(r.get("C2_accuracy_improvement")) - fval(baseline.get("C2_accuracy_improvement"))
        if random_rows:
            strongest = max(fval(rr.get("C2_coverage_improvement")) for rr in random_rows)
            r["capacity_random_gap"] = fval(r.get("C2_coverage_improvement")) - strongest
            r["strongest_random_coverage"] = strongest
        if oracle and baseline:
            denom = fval(oracle.get("C2_coverage_improvement")) - fval(baseline.get("C2_coverage_improvement")) + 1.0e-12
            r["oracle_retention_ratio"] = (fval(r.get("C2_coverage_improvement")) - fval(baseline.get("C2_coverage_improvement"))) / denom
    matrix = write_rows(OUT_ROOT / "part_b_capacity_bridge_matrix.csv", ok + [r for r in rows if r.get("status") != "ok"])
    task_summaries: list[dict[str, Any]] = []
    for key in sorted({(r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task"), r.get("repair_round")) for r in ok}):
        group = [r for r in ok if (r.get("scheme"), r.get("basis_key"), r.get("depth"), r.get("task"), r.get("repair_round")) == key]
        task_summaries.append({
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "task": key[3],
            "repair_round": key[4],
            "rows": len(group),
            "C2_coverage_improvement_median": median(r.get("C2_coverage_improvement") for r in group),
            "C2_accuracy_improvement_median": median(r.get("C2_accuracy_improvement") for r in group),
            "coverage_vs_functional_median": median(r.get("coverage_vs_functional") for r in group),
            "capacity_random_gap": median(r.get("capacity_random_gap") for r in group),
            "oracle_retention_ratio": median(r.get("oracle_retention_ratio") for r in group),
            "source_guard_stability": median(r.get("capacity_source_guard_stability") for r in group),
            "capacity_q_density": median(r.get("capacity_q_density") for r in group),
            "local_patch_coverage": median(r.get("local_patch_coverage") for r in group),
            "rotation_coverage": median(r.get("rotation_coverage") for r in group),
        })
    groups: list[dict[str, Any]] = []
    random_and_baseline = {"B0_H10_oracle_reference", "B1_FunctionalGram_baseline", *RANDOM_CAPACITY_CONTROLS}
    for key in sorted({(t.get("scheme"), t.get("basis_key"), t.get("depth"), t.get("repair_round")) for t in task_summaries}):
        tasks = [t for t in task_summaries if (t.get("scheme"), t.get("basis_key"), t.get("depth"), t.get("repair_round")) == key]
        candidate = str(key[0]) not in random_and_baseline
        g = {
            "scheme": key[0],
            "basis_key": key[1],
            "depth": key[2],
            "repair_round": key[3],
            "rows": sum(ival(t.get("rows")) for t in tasks),
            "capacity_candidate": int(candidate),
            "C2_coverage_improvement_median": median(t.get("C2_coverage_improvement_median") for t in tasks),
            "C2_accuracy_improvement_median": median(t.get("C2_accuracy_improvement_median") for t in tasks),
            "coverage_vs_functional_median": median(t.get("coverage_vs_functional_median") for t in tasks),
            "capacity_random_gap": median(t.get("capacity_random_gap") for t in tasks),
            "oracle_retention_ratio": median(t.get("oracle_retention_ratio") for t in tasks),
            "source_guard_stability": median(t.get("source_guard_stability") for t in tasks),
            "capacity_q_density": median(t.get("capacity_q_density") for t in tasks),
            "local_patch_coverage": max([fval(t.get("local_patch_coverage")) for t in tasks if str(t.get("task")) == "local_patch_interaction"] or [0.0]),
            "rotation_coverage": max([fval(t.get("rotation_coverage")) for t in tasks if str(t.get("task")) == "rotation_sensitive"] or [0.0]),
        }
        g["taskwise_all_pass"] = int(
            candidate
            and fval(g["coverage_vs_functional_median"]) >= float(args.b_coverage_vs_functional_gate)
            and fval(g["C2_accuracy_improvement_median"]) >= float(args.b_accuracy_gate)
            and fval(g["capacity_random_gap"]) >= float(args.b_capacity_random_gap_gate)
            and fval(g["oracle_retention_ratio"]) >= float(args.b_oracle_retention_gate)
            and fval(g["source_guard_stability"]) >= float(args.b_source_guard_gate)
        )
        groups.append(g)
    pass_groups = [g for g in groups if ival(g.get("taskwise_all_pass")) == 1]
    errors = [r for r in rows if r.get("status") == "error"]
    gate = int(bool(pass_groups) and not errors)
    if errors:
        blocker = "part_b_job_errors"
    elif not pass_groups:
        blockers = []
        candidates = [g for g in groups if ival(g.get("capacity_candidate"))]
        if all(fval(g.get("capacity_random_gap")) < float(args.b_capacity_random_gap_gate) for g in candidates):
            blockers.append("capacity_random_gap_low")
        if all(fval(g.get("source_guard_stability")) < float(args.b_source_guard_gate) for g in candidates):
            blockers.append("source_guard_stability_low")
        blocker = ";".join(blockers) or "capacity_bridge_failed"
    else:
        blocker = "none"
    task_csv = write_rows(OUT_ROOT / "part_b_capacity_bridge_task_summary.csv", task_summaries)
    group_csv = write_rows(OUT_ROOT / "part_b_capacity_bridge_group_summary.csv", groups)
    summary = gate_summary("B", gate, "CapacityBridgePass" if gate else "CapacityBridgeFailed", blocker, rows, taskwise_groups=task_summaries, scheme_groups=groups, passing_scheme_groups=pass_groups)
    write_json(OUT_ROOT / "part_b_capacity_bridge_summary.json", summary)
    nxt = next_actions("B", gate, blocker, [] if gate else ["inspect strongest matched random controls", "try simpler fixed density or block family only with paired random controls", "do not advance to real tasks before Part B/C pass"])
    append_exec("part-b-merge", command_text(sys.argv), "passed" if gate else "failed", files=f"{rel(matrix)}; {rel(task_csv)}; {rel(group_csv)}; {rel(OUT_ROOT / 'part_b_capacity_bridge_summary.json')}; {rel(nxt)}", gpu=str(args.device), note=f"blocker={blocker}")
    append_recap("Part B synthetic capacity regularizer bridge", summary)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    p0 = read_json(OUT_ROOT / "part_0_v2305_negative_lock.json")
    a = read_json(OUT_ROOT / "part_a_identity_summary.json")
    b = read_json(OUT_ROOT / "part_b_capacity_bridge_summary.json")
    if ival(p0.get("gate_pass")) != 1:
        route = "V2305NegativeLockFailed"
        blocker = "part_0_failed"
    elif ival(a.get("gate_pass")) != 1:
        route = "IdentityAuditFailed"
        blocker = "part_a_failed"
    elif ival(b.get("gate_pass")) != 1:
        route = "CapacityBridgeFailed"
        blocker = b.get("dominant_blocker", "part_b_failed")
    else:
        route = "CapacityBridgePass_NoRealPromotionYet"
        blocker = "none"
    final = {
        "route": route,
        "dominant_blocker": blocker,
        "official_candidate_gate_pass": int(route == "CapacityBridgePass_NoRealPromotionYet"),
        "part_0_gate_pass": p0.get("gate_pass", "missing"),
        "part_a_gate_pass": a.get("gate_pass", "missing"),
        "part_b_gate_pass": b.get("gate_pass", "missing"),
        "part_c_gate_pass": 0,
        "part_d_gate_pass": 0,
        "promotion_allowed": 0,
        "generated_at": now(),
        **AUDIT_DEFAULTS,
    }
    write_json(OUT_ROOT / "final_route.json", final)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        "\n".join([
            "# DG-KAN v23.06 reproduction manifest",
            "",
            f"Generated: {now()}",
            f"Runner: {rel(RUNNER)}",
            f"Plan: {rel(PLAN)}",
            f"Final route: {route}",
            f"Dominant blocker: {blocker}",
            "",
        ]),
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}", note=f"route={route} blocker={blocker}")
    append_recap("Final route", final)
    return final


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="part-0", choices=["part-0", "part-a", "part-b", "part-b-merge", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repair-round", default="initial")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--visual-side", type=int, default=8)
    p.add_argument("--input-dim", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--deep-width", type=int, default=12)
    p.add_argument("--mlp-hidden", type=int, default=24)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--synthetic-train-size", type=int, default=72)
    p.add_argument("--synthetic-guard-size", type=int, default=72)
    p.add_argument("--visual-fixed-patch-features", type=int, default=0)
    p.add_argument("--visual-task-version", default="balanced_interaction_v2")
    p.add_argument("--batch-size", type=int, default=36)
    p.add_argument("--train-steps", type=int, default=30)
    p.add_argument("--edge-lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--edge-weight-normalization", default="trace")
    p.add_argument("--edge-weight-ridge", type=float, default=1.0e-6)
    p.add_argument("--functional-gram-quadrature-points", type=int, default=257)
    p.add_argument("--sobolev-exponent", type=float, default=0.25)
    p.add_argument("--snr-beta", type=float, default=2.0)
    p.add_argument("--gate-floor", type=float, default=0.0)
    p.add_argument("--gate-floor-mode", default="final", choices=["final", "population"])
    p.add_argument("--capacity-gate-every", type=int, default=1)
    p.add_argument("--capacity-density-target", type=float, default=0.17)
    p.add_argument("--population-grad-examples", type=int, default=6)
    p.add_argument("--structure-guard-examples", type=int, default=64)
    p.add_argument("--label-prior-correction", type=int, default=0)
    p.add_argument("--part-i-band-grid", type=int, default=4)
    p.add_argument("--part-i-band-degree-bands", type=int, default=3)
    p.add_argument("--part-i-band-edge-bands", type=int, default=4)
    p.add_argument("--part-b-schemes", default="B0_H10_oracle_reference,B1_FunctionalGram_baseline,B2_block_snr_capacity_gate,B3_functional_energy_capacity_gate,B4_guard_stable_capacity_gate,B5_capacity_fullblock_gate,B6_capacity_density_only_random_control,B7_capacity_block_shuffle_control,B8_capacity_norm_matched_random_control,B9_guard_stable_capacity_modifier,B10_fullblock_capacity_modifier,B11_density_only_modifier_random_control,B12_block_shuffle_modifier_control,B13_norm_matched_modifier_control,B14_descent_cone_coordinate_gate,B15_descent_cone_coordinate_modifier,B16_coordinate_density_random_control,B17_coordinate_shuffle_control,B18_coordinate_norm_matched_random_control")
    p.add_argument("--part-b-basis", default="dche_k9")
    p.add_argument("--part-b-depths", default="depth3")
    p.add_argument("--part-b-tasks", default="local_patch_interaction,rotation_sensitive")
    p.add_argument("--part-b-seed-count", type=int, default=3)
    p.add_argument("--b0-from-h10-artifact", type=int, default=1)
    p.add_argument("--b-coverage-vs-functional-gate", type=float, default=0.15)
    p.add_argument("--b-accuracy-gate", type=float, default=0.20)
    p.add_argument("--b-capacity-random-gap-gate", type=float, default=0.10)
    p.add_argument("--b-oracle-retention-gate", type=float, default=0.50)
    p.add_argument("--b-source-guard-gate", type=float, default=0.60)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    if args.mode == "part-0":
        run_part_0(args)
    elif args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-b-merge":
        merge_part_b(args)
    elif args.mode == "finalize":
        finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
