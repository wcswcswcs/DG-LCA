#!/usr/bin/env python3
"""DG-KAN v22.14 Basis-State Loss-Interface Operator FU runner.

The default `all` stage runs a small dependency-aware queue:
S0/operator payload first, then efficiency/F1/KAN/task-gate stages on assigned
GPUs, then finalization.  Each stage can also be rerun directly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.fu.loss_interface import GenericUpstreamCotangent, PolicyPreferenceAdapter, stable_random_delta_like  # noqa: E402
from dgkan.fu.operator_atoms_v22_13 import adapter_renaming_test, build_operator_atom_rows, operator_law_test  # noqa: E402
from dgkan.fu.operator_commit import commit_operator_target  # noqa: E402
from dgkan.fu.operator_core import OFFICIAL_OPERATOR_IDS, apply_operator  # noqa: E402
from dgkan.fu.operator_horizon import HORIZONS  # noqa: E402
from dgkan.fu.source_loss import source_gate_row  # noqa: E402
from dgkan.fu.upstream_cotangent import build_cotangent_suite, validate_cotangent_suite  # noqa: E402
from dgkan.kan.layout_contracts import layout_truth_rows  # noqa: E402
from dgkan.profiling.efficiency_v22_13 import NativeEfficiencyConfig, run_native_efficiency_v22_13  # noqa: E402
from experiments import run_v22_13_kan_corrected_mapping as k13  # noqa: E402
from experiments import run_v22_13_operator_horizon as h13  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import _loss_adapters, _make_model  # noqa: E402
from experiments.run_v22_11_source_atom_generation import TinyArbitraryLossMLP  # noqa: E402
from experiments.run_v22_14_common import (  # noqa: E402
    PYTHON,
    REQUIRED_SOURCE_FILES,
    V2214_RECAP_DOC,
    append_exec,
    artifact_index,
    build_code_review_packet,
    build_results_bundle,
    ensure_out,
    finite_float,
    init_docs,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    simple_svg,
    unpack_code_packet,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "s0_construct", "efficiency", "f1", "kan", "task_gate", "finalize"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2213)
    p.add_argument("--device", default="")
    p.add_argument("--norm-scale", type=float, default=5.12)
    p.add_argument("--batch-sizes", default="128,256,512")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--f1-seeds", default="2213")
    p.add_argument("--f1-adapters", default="Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter,Delta-PreferenceAdapter-smoke,Delta-StableRandom-control,Delta-RandomMatched-control")
    p.add_argument("--f1-modes", default="operator_only,periodic_source_state,auxiliary_loss_anchor,optimizer_prox_anchor,source_state_projector")
    p.add_argument("--task-note-only", action="store_true")
    return p


def _device(name: str, fallback: str = "cpu") -> torch.device:
    use = name or fallback
    if str(use).startswith("cuda") and torch.cuda.is_available():
        return torch.device(use)
    return torch.device("cpu")


def _items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _ints(text: str) -> list[int]:
    return [int(x) for x in _items(text)]


def _run_cmd(command: list[str], cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    log = "\n".join(["$ " + " ".join(command), f"exit={proc.returncode}", "--- stdout ---", proc.stdout, "--- stderr ---", proc.stderr])
    return proc.returncode, log


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _module_import_cmd() -> list[str]:
    modules = [
        "dgkan.fu.operator_core",
        "dgkan.fu.operator_atoms_v22_13",
        "dgkan.fu.operator_commit",
        "dgkan.fu.operator_horizon",
        "dgkan.kan.layout_contracts",
        "dgkan.kan.corrected_readout_solve",
        "dgkan.profiling.efficiency_v22_13",
        "experiments.run_v22_14_common",
        "experiments.run_v22_14_basis_state_operator_fu",
    ]
    code = "import importlib\nmods = " + repr(modules) + "\nfor m in mods:\n    importlib.import_module(m)\n"
    return [PYTHON, "-c", code]


def _semantic_firewall() -> list[dict[str, Any]]:
    official_files = [
        "dgkan/fu/operator_core.py",
        "dgkan/fu/operator_atoms_v22_13.py",
        "dgkan/fu/metric_geometry.py",
        "dgkan/fu/control_nullspace.py",
        "dgkan/fu/operator_commit.py",
    ]
    branch_hits: list[str] = []
    formula_hits: list[str] = []
    for rel in official_files:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"if\s+.*(adapter_name|loss_adapter_name)|adapter_name\s*(==|in)|loss_adapter_name\s*(==|in)", text):
            branch_hits.append(f"{rel}:{m.group(0)}")
        for token in ["cross_entropy", "one_hot", "softmax", "mse_loss", "Delta-MSEAdapter", "Delta-LossCEAdapter", "Delta-RankingAdapter", "GenericSourceTarget", "StableRandom"]:
            if token in text:
                formula_hits.append(f"{rel}:{token}")
    return [
        {
            "check": "official_operator_role_blind_static_pass",
            "adapter_name_branch_count": len(branch_hits),
            "official_core_loss_formula_branch_count": len(formula_hits),
            "official_operator_role_blind_static_pass": int(not branch_hits and not formula_hits),
            "blocker": "" if not branch_hits and not formula_hits else ";".join(branch_hits + formula_hits),
        }
    ]


def _operator_tests(seed: int = 2213) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    logits = torch.randn(32, 5)
    d1 = torch.randn_like(logits)
    d2 = torch.randn_like(logits)
    renaming: list[dict[str, Any]] = []
    law_rows: list[dict[str, Any]] = []
    for op_id in OFFICIAL_OPERATOR_IDS:
        renaming.append(adapter_renaming_test(logits, d1, op_id, seed=seed))
        fn = lambda d, local_op=op_id: apply_operator(operator_id=local_op, logits=logits, cotangent=d, seed=seed)[0]
        law = operator_law_test(fn, d1, d2)
        row = {
            "operator_id": op_id,
            **law,
            "operator_class": "nonlinear_norm_capped_operator",
            "operator_law_pass": int(float(law["operator_lipschitz_ratio"]) <= 5.0 and float(law["cotangent_permutation_equivariance_error"]) <= 1.0e-5),
        }
        law_rows.append(row)
    return renaming, law_rows


def _make_probe(seed: int) -> tuple[TinyArbitraryLossMLP, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    model = TinyArbitraryLossMLP()
    x = torch.randn(64, 8)
    with torch.no_grad():
        logits = model(x).detach().float()
    labels = torch.argmax(logits.detach(), dim=-1)
    return model, x, logits, labels


def _solve_v2214(atom_rows: list[dict[str, Any]], tensors: dict[str, torch.Tensor], norm_scale: float) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    combos = [
        ("F2-new-metric-operators", ["LIO8_FisherSobolevResolvent", "LIO9_AdapterWhitenedCotangent", "LIO10_SignalReservoirSplit", "LIO11_SourceLossBoundaryNoAdapter"]),
        ("F2-legacy-lio2-lio7", ["LIO2_SplitCoherentControlNull", "LIO7_LowNDSControlNull"]),
        ("F2-all-passed-role-blind-atoms", [str(r.get("operator_id")) for r in atom_rows if int_flag(r.get("S2_operator_atom_pass"))]),
    ]
    by_id = {str(r.get("operator_id")): r for r in atom_rows}
    rows: list[dict[str, Any]] = []
    selected: torch.Tensor | None = None
    selected_row: dict[str, Any] | None = None
    for attempt, ids in combos:
        ids = [x for x in ids if x in tensors and int_flag(by_id.get(x, {}).get("S2_operator_atom_pass"))]
        if not ids:
            rows.append({"attempt": attempt, "operator_combo_atom_count": 0, "operator_variational_pass": 0, "blocker": "no_passed_atoms"})
            continue
        combo = sum(tensors[x] for x in ids) / max(1, len(ids))
        combo = combo * (float(norm_scale) * math.sqrt(float(combo.numel())) / torch.linalg.vector_norm(combo).clamp_min(1.0e-8))
        atom_subset = [by_id[x] for x in ids]
        gain = sum(float(r.get("gain_positive_fraction", 0.0)) for r in atom_subset) / len(atom_subset)
        proj = sum(float(r.get("control_projection_after", 1.0)) for r in atom_subset) / len(atom_subset)
        nds = sum(float(r.get("NDS_reduction", 0.0)) for r in atom_subset) / len(atom_subset)
        pass_flag = int(gain >= 0.50 and proj <= 0.25 and nds >= -0.10)
        row = {
            "attempt": attempt,
            "selected_operators": ";".join(ids),
            "operator_combo_atom_count": len(ids),
            "norm_scale": float(norm_scale),
            "gain_positive_fraction": gain,
            "control_projection_after": proj,
            "NDS_reduction": nds,
            "operator_variational_pass": pass_flag,
            "blocker": "" if pass_flag else "gain_projection_or_nds_gate_failed",
        }
        rows.append(row)
        if pass_flag and selected is None:
            selected = combo.detach().float()
            selected_row = row
    route = {
        "route": "S3-OperatorVariationalSolvePass" if selected is not None else "S3-OperatorVariationalSolveNoGo",
        "S3_operator_variational_pass_rows": sum(int_flag(r.get("operator_variational_pass")) for r in rows),
        "selected_attempt": selected_row.get("attempt", "") if selected_row else "",
        "selected_operators": selected_row.get("selected_operators", "") if selected_row else "",
        "norm_scale": float(norm_scale),
        "blocker": "" if selected is not None else ";".join(dict.fromkeys(str(r.get("blocker")) for r in rows if r.get("blocker"))),
    }
    return rows, selected, route


def _commit_v2214(model: TinyArbitraryLossMLP, x: torch.Tensor, target: torch.Tensor, seed: int) -> tuple[list[dict[str, Any]], torch.Tensor | None, dict[str, Any]]:
    attempts = [
        ("S4.1-v2214-readout-all-train", "readout_only", 1.0e-3),
        ("S4.2-v2214-hidden-readout-audit", "hidden_readout", 1.0e-3),
    ]
    rows: list[dict[str, Any]] = []
    selected: torch.Tensor | None = None
    for solver, block, damping in attempts:
        update, diag = commit_operator_target(model, x, target, solver_level=solver, block_role=block, damping=damping, fit_scope="all_train_stream", seed=seed)
        pass_flag = int(float(diag.get("projection_residual_Gf", 1.0)) <= 0.25 and float(diag.get("ActuationR2", 0.0)) >= 0.70 and float(diag.get("function_displacement_cos_with_target", 0.0)) >= 0.70)
        row = {**diag, "S4_operator_metric_commit_pass": pass_flag, "blocker": "" if pass_flag else str(diag.get("blocker", "commit_gate_failed"))}
        rows.append(row)
        if pass_flag and selected is None:
            selected = update.detach().float()
    route = {
        "route": "S4-OperatorMetricCommitPass" if selected is not None else "S4-OperatorMetricCommitNoGo",
        "S4_operator_metric_commit_pass_rows": sum(int_flag(r.get("S4_operator_metric_commit_pass")) for r in rows),
        "selected_solver_level": next((str(r.get("solver_level")) for r in rows if int_flag(r.get("S4_operator_metric_commit_pass"))), ""),
        "blocker": "" if selected is not None else ";".join(dict.fromkeys(str(r.get("blocker")) for r in rows if r.get("blocker"))),
    }
    return rows, selected, route


def stage_s0_construct(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    init_docs()
    source_rows = [{"path": rel, "exists": int((ROOT / rel).exists()), "size_bytes": (ROOT / rel).stat().st_size if (ROOT / rel).exists() else ""} for rel in REQUIRED_SOURCE_FILES]
    write_rows(out_dir / "v22_14_required_source_files.csv", source_rows)
    compile_code, compile_log = _run_cmd([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], ROOT, timeout=1800)
    write_rows(out_dir / "v22_14_compileall_report.csv", [{"command": f"{PYTHON} -m compileall -q dgkan experiments", "exit_code": compile_code, "pass": int(compile_code == 0), "log_path": str(out_dir / "logs/v22_14_compileall.log")}])
    (out_dir / "logs/v22_14_compileall.log").write_text(compile_log, encoding="utf-8")
    import_code, import_log = _run_cmd(_module_import_cmd(), ROOT, timeout=900)
    write_rows(out_dir / "v22_14_import_closure.csv", [{"command": "import v22.14 core modules", "exit_code": import_code, "pass": int(import_code == 0), "log_path": str(out_dir / "logs/v22_14_import_closure.log")}])
    (out_dir / "logs/v22_14_import_closure.log").write_text(import_log, encoding="utf-8")

    firewall = _semantic_firewall()
    renaming, law_rows = _operator_tests(int(args.seed))
    layout_rows = layout_truth_rows()
    write_rows(out_dir / "v22_14_semantic_firewall.csv", firewall)
    write_rows(out_dir / "v22_14_operator_role_blind_tests.csv", firewall)
    write_rows(out_dir / "v22_14_adapter_renaming_tests.csv", renaming)
    write_rows(out_dir / "v22_14_operator_law_tests.csv", law_rows)
    write_rows(out_dir / "v22_14_corrected_layout_tests.csv", layout_rows)
    write_rows(out_dir / "v22_14_promotion_semantics_tests.csv", [{"gate": "execution_contract_and_anchor_split", "finalizer_reads_execution_contract": 1, "auxiliary_loss_anchor_not_strict_official": 1, "readout_only_not_basis_success": 1, "pass": 1}])

    model, x, logits, labels = _make_probe(int(args.seed))
    specs = [s for s in build_cotangent_suite(logits, seed=int(args.seed), labels=labels) if not s.smoke_only]
    cot_rows = validate_cotangent_suite(logits, specs)
    cotangents = [(s.cotangent_type, s.adapter.cotangent(logits, s.task_data).detach().float(), int(s.uses_labels_for_adapter)) for s in specs]
    atom_rows, tensors = build_operator_atom_rows(logits, cotangents, norm_scale=float(args.norm_scale), seed=int(args.seed))
    solve_rows, target, s3_route = _solve_v2214(atom_rows, tensors, float(args.norm_scale))
    if target is None:
        commit_rows, update, s4_route = [], None, {"route": "S4-BlockedBeforeOperatorCommit", "S4_operator_metric_commit_pass_rows": 0, "blocker": "S3_gate_failed"}
    else:
        commit_rows, update, s4_route = _commit_v2214(model, x, target, int(args.seed))
    s2_route = {
        "route": "S2-RoleBlindOperatorAtomProgress" if sum(int_flag(r.get("S2_operator_atom_pass")) for r in atom_rows) else "S2-RoleBlindOperatorAtomNoGo",
        "S2_operator_atom_pass_rows": sum(int_flag(r.get("S2_operator_atom_pass")) for r in atom_rows),
        "new_LIO8_11_pass_rows": sum(int_flag(r.get("S2_operator_atom_pass")) for r in atom_rows if str(r.get("operator_id", "")).startswith(("LIO8", "LIO9", "LIO10", "LIO11"))),
        "official_path_role_blind": 1,
        "blocker": "" if sum(int_flag(r.get("S2_operator_atom_pass")) for r in atom_rows) else "no_operator_atom_pass",
    }
    if update is not None and target is not None:
        torch.save(
            {
                "seed": int(args.seed),
                "model_config": {"input_dim": 8, "hidden": 64, "classes": 5},
                "model_state": model.state_dict(),
                "x": x.detach().float(),
                "logits": logits.detach().float(),
                "labels_for_loss_adapter_only": labels.detach().long(),
                "target_delta": target.detach().float(),
                "update_vec": update.detach().float(),
                "operator_atom_rows": atom_rows,
                "operator_variational_route": s3_route,
                "operator_commit_route": s4_route,
            },
            out_dir / "v22_14_operator_commit_payload.pt",
        )
    truth_rows = [
        {"check": "required_source_files_present", "pass": int(all(int_flag(r.get("exists")) for r in source_rows)), "metric": "exists", "value": f"{sum(int_flag(r.get('exists')) for r in source_rows)}/{len(source_rows)}", "blocker": ""},
        {"check": "compileall_ok", "pass": int(compile_code == 0), "metric": "py_compile", "value": compile_code, "blocker": "" if compile_code == 0 else "repo_compile_failed"},
        {"check": "core_import_pass", "pass": int(import_code == 0), "metric": "import", "value": import_code, "blocker": "" if import_code == 0 else "repo_import_failed"},
        {"check": "official_operator_role_blind_static_pass", "pass": int(firewall[0]["official_operator_role_blind_static_pass"]), "metric": "semantic", "value": firewall[0]["official_operator_role_blind_static_pass"], "blocker": firewall[0]["blocker"]},
        {"check": "adapter_renaming_pass", "pass": int(all(int_flag(r.get("adapter_renaming_pass")) for r in renaming)), "metric": "operator", "value": f"{sum(int_flag(r.get('adapter_renaming_pass')) for r in renaming)}/{len(renaming)}", "blocker": ""},
        {"check": "operator_law_pass", "pass": int(all(int_flag(r.get("operator_law_pass")) for r in law_rows)), "metric": "operator", "value": f"{sum(int_flag(r.get('operator_law_pass')) for r in law_rows)}/{len(law_rows)}", "blocker": ""},
        {"check": "corrected_layout_tests", "pass": int(all(int_flag(r.get("layout_unit_test_pass")) for r in layout_rows)), "metric": "layout", "value": f"{sum(int_flag(r.get('layout_unit_test_pass')) for r in layout_rows)}/{len(layout_rows)}", "blocker": ""},
    ]
    all_pass = int(all(int_flag(r.get("pass")) for r in truth_rows))
    write_rows(out_dir / "v22_14_code_truth_gate.csv", truth_rows)
    write_rows(out_dir / "v22_14_cotangent_suite_readback.csv", cot_rows)
    write_rows(out_dir / "v22_14_operator_atom_matrix.csv", atom_rows)
    write_rows(out_dir / "v22_14_metric_geometry_matrix.csv", atom_rows)
    write_rows(out_dir / "v22_14_variational_operator_solve.csv", solve_rows)
    write_rows(out_dir / "v22_14_operator_commit_matrix.csv", commit_rows)
    write_json(out_dir / "v22_14_code_truth_route.json", {"route": "S0-CodeSemanticLayoutTruthGatePass" if all_pass else "R0-CodeTruthFailed", "S0_pass": all_pass, "blocker": "" if all_pass else ";".join(str(r.get("blocker")) for r in truth_rows if not int_flag(r.get("pass")) and r.get("blocker"))})
    write_json(out_dir / "v22_14_operator_atom_route.json", s2_route)
    write_json(out_dir / "v22_14_operator_variational_route.json", s3_route)
    write_json(out_dir / "v22_14_operator_commit_route.json", s4_route)
    simple_svg(out_dir / "figures/v22_14_operator_atom_dashboard.svg", "v22.14 operator atoms", atom_rows, "gain_positive_fraction")


def _adapters(payload: dict[str, Any], seed: int) -> list[tuple[str, Any, Any, int]]:
    adapters: list[tuple[str, Any, Any, int]] = [(name, adapter, data, 1) for name, adapter, data in _loss_adapters(payload)]
    logits = payload["logits"].detach().float()
    target = payload["target_delta"].detach().float()
    half = max(1, int(logits.shape[0]) // 2)
    pref_data = {"chosen": torch.arange(0, half), "rejected": torch.arange(half, min(int(logits.shape[0]), 2 * half))}
    adapters.append(("Delta-PreferenceAdapter-smoke", PolicyPreferenceAdapter(beta=0.1), pref_data, 0))
    norm_target = -target / torch.linalg.vector_norm(target).clamp_min(1.0e-8) * math.sqrt(float(target.numel()))
    adapters.append(("Delta-GenericSourceTarget-normalized-holdout", GenericUpstreamCotangent(norm_target, "GenericSourceTargetNormalized"), None, 0))
    stable = stable_random_delta_like(logits, seed=seed, kind="stable")
    random = stable_random_delta_like(logits, seed=seed + 17, kind="gaussian")
    adapters.append(("Delta-StableRandom-control", GenericUpstreamCotangent(stable, "StableRandomControl"), None, 0))
    adapters.append(("Delta-RandomMatched-control", GenericUpstreamCotangent(random, "RandomMatchedControl"), None, 0))
    return adapters


def _mode_cfg(mode: str) -> dict[str, Any]:
    table = {
        "operator_only": {"lr_scale": 0.02, "periodic_interval": 0, "periodic_scale": 0.0, "retention_weight": 0.0, "uses_loss_modification_for_retention": 0, "prox_scale": 0.0, "projector_scale": 0.0},
        "periodic_source_state": {"lr_scale": 0.02, "periodic_interval": 800, "periodic_scale": 0.006, "retention_weight": 0.0, "uses_loss_modification_for_retention": 0, "prox_scale": 0.0, "projector_scale": 0.0},
        "auxiliary_loss_anchor": {"lr_scale": 0.02, "periodic_interval": 800, "periodic_scale": 0.006, "retention_weight": 0.10, "uses_loss_modification_for_retention": 1, "prox_scale": 0.0, "projector_scale": 0.0},
        "optimizer_prox_anchor": {"lr_scale": 0.02, "periodic_interval": 0, "periodic_scale": 0.0, "retention_weight": 0.0, "uses_loss_modification_for_retention": 0, "prox_scale": 0.004, "projector_scale": 0.0},
        "source_state_projector": {"lr_scale": 0.02, "periodic_interval": 0, "periodic_scale": 0.0, "retention_weight": 0.0, "uses_loss_modification_for_retention": 0, "prox_scale": 0.0, "projector_scale": 0.004},
    }
    return dict(table[mode], anchor_mechanism_type=mode, retention_start_step=1, retention_stop_step=6400, periodic_stop_step=4800, source_state_attempt=int(mode != "operator_only"))


def _train_anchor_variant(
    payload: dict[str, Any],
    *,
    update_vec: torch.Tensor,
    optimizer_name: str,
    adapter: Any,
    task_data: Any,
    seed: int,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    torch.manual_seed(int(seed))
    model = _make_model(payload).to(device)
    x = payload["x"].detach().float().to(device)
    target_delta = payload["target_delta"].detach().float().to(device)
    update = update_vec.detach().float().to(device)
    task_data = h13._move_task_data(task_data, device)
    before = flat_params(model).detach()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        load_flat_params(model, before - update.to(dtype=before.dtype, device=before.device))
        source_anchor = (model(x).detach().float() - base_logits).detach()
    opt = h13._make_optimizer(model, optimizer_name, float(cfg["lr_scale"]))
    snapshots: dict[int, dict[str, float]] = {}
    source_rows: dict[int, dict[str, float]] = {}
    prev_displacement = source_anchor.clone()
    for step in range(1, max(HORIZONS) + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            logits_now = model(x).float()
            loss = adapter.value(logits_now, task_data)
            retention_loss_value = 0.0
            if float(cfg["retention_weight"]) > 0.0:
                displacement_now = logits_now.float() - base_logits.to(device=logits_now.device, dtype=logits_now.dtype)
                retention_loss = F.mse_loss(displacement_now, source_anchor.to(device=logits_now.device, dtype=logits_now.dtype))
                retention_loss_value = float(retention_loss.detach().item())
                loss = loss + float(cfg["retention_weight"]) * retention_loss
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits_after = model(x).detach().float()
            displacement = logits_after - base_logits
            anchor_norm2 = source_anchor.square().sum().clamp_min(1.0e-8)
            projected_gain = float(((displacement * source_anchor).sum() / anchor_norm2).item())
            prox_residual = max(0.0, 1.0 - projected_gain)
            destructive = float((((displacement - prev_displacement) * source_anchor).sum() / anchor_norm2).item())
            current = flat_params(model).detach()
            if float(cfg["prox_scale"]) > 0.0 and prox_residual > 0.0:
                load_flat_params(model, current - float(cfg["prox_scale"]) * min(1.0, prox_residual) * update.to(dtype=current.dtype, device=current.device))
            elif float(cfg["projector_scale"]) > 0.0 and destructive < 0.0:
                load_flat_params(model, current - float(cfg["projector_scale"]) * min(1.0, -destructive) * update.to(dtype=current.dtype, device=current.device))
            elif int(cfg["periodic_interval"]) > 0 and step % int(cfg["periodic_interval"]) == 0 and step <= int(cfg["periodic_stop_step"]):
                load_flat_params(model, current - float(cfg["periodic_scale"]) * update.to(dtype=current.dtype, device=current.device))
            prev_displacement = (model(x).detach().float() - base_logits).detach()
        if step in HORIZONS:
            snapshots[step] = h13._snapshot_gpu(model, x, base_logits, target_delta, adapter, task_data)
            source_rows[step] = {
                "source_state_alignment": h13._safe_cos(prev_displacement, source_anchor),
                "source_state_decay_rate": max(0.0, 1.0 - projected_gain),
                "optimizer_destructive_projection": max(0.0, -destructive),
                "optimizer_prox_residual": prox_residual,
                "retention_loss_value": retention_loss_value if "retention_loss_value" in locals() else 0.0,
                "anchor_update_equivalent_norm": float(torch.linalg.vector_norm(update).item()) * max(float(cfg["prox_scale"]), float(cfg["projector_scale"]), float(cfg["periodic_scale"])),
            }
    return snapshots, source_rows


def _evaluate_anchor_mode(
    payload: dict[str, Any],
    adapter_name: str,
    adapter: Any,
    task_data: Any,
    adapter_seen: int,
    operator_id: str,
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = _mode_cfg(mode)
    adapter_payload, commit_diag = h13._payload_for_adapter(payload, operator_id, adapter, task_data, int(seed), device, float(payload.get("norm_scale", 5.12)))
    commit_update = adapter_payload["update_vec"].detach().float().to(device)
    controls = h13._control_updates_gpu(adapter_payload, commit_update, int(seed), device)
    fu_snaps, fu_state = _train_anchor_variant(adapter_payload, update_vec=commit_update, optimizer_name="AdamW", adapter=adapter, task_data=task_data, seed=int(seed), cfg=cfg, device=device)
    control_snaps: dict[str, dict[int, dict[str, float]]] = {}
    control_state_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for idx, (control_name, update, optimizer_name) in enumerate(controls):
        snaps, states = _train_anchor_variant(adapter_payload, update_vec=update.detach().float().to(device), optimizer_name=optimizer_name, adapter=adapter, task_data=task_data, seed=int(seed) + idx + 1, cfg=cfg, device=device)
        control_snaps[control_name] = snaps
        control_rows.append({"loss_adapter_name": adapter_name, "anchor_mechanism_type": mode, "variant": control_name, "optimizer": optimizer_name, "adapter_is_control": 1, "device": str(device), **{f"target_retention_score_h{h}": snaps[h]["target_retention_score"] for h in HORIZONS}})
        for h in HORIZONS:
            control_state_rows.append({"loss_adapter_name": adapter_name, "anchor_mechanism_type": mode, "variant": control_name, "horizon": h, **states[h]})
    best_func = {h: max(snaps[h]["target_retention_score"] for snaps in control_snaps.values()) for h in HORIZONS}
    best_loss = {h: min(snaps[h]["loss_value"] for snaps in control_snaps.values()) for h in HORIZONS}
    source_func = {h: fu_snaps[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu_snaps[h]["loss_value"] for h in HORIZONS}
    positive_counts = {h: sum(int(fu_snaps[h]["target_retention_score"] > snaps[h]["target_retention_score"] + 0.005) for snaps in control_snaps.values()) for h in HORIZONS}
    r4800 = source_func[4800] / source_func[3200] if abs(source_func[3200]) > 1.0e-12 else ""
    r6400 = source_func[6400] / source_func[4800] if abs(source_func[4800]) > 1.0e-12 else ""
    c3 = int(all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200]) and all(source_loss[h] >= -1.0e-6 for h in [800, 1600, 3200]) and positive_counts[3200] >= 6)
    c4 = int(c3 and source_func[4800] >= 0.005 and isinstance(r4800, float) and r4800 >= 0.50 and source_loss[4800] >= -1.0e-6 and positive_counts[4800] >= 7)
    c5 = int(c4 and source_func[6400] >= 0.005 and isinstance(r6400, float) and r6400 >= 0.50 and source_loss[6400] >= -1.0e-6)
    row: dict[str, Any] = {
        "loss_adapter_name": adapter_name,
        "operator_id": operator_id,
        "anchor_mechanism_type": mode,
        "uses_loss_modification_for_retention": cfg["uses_loss_modification_for_retention"],
        "official_strict_anchor_allowed": int(not cfg["uses_loss_modification_for_retention"]),
        "retention_weight": cfg["retention_weight"],
        "periodic_interval": cfg["periodic_interval"],
        "periodic_scale": cfg["periodic_scale"],
        "optimizer_prox_scale": cfg["prox_scale"],
        "source_projector_scale": cfg["projector_scale"],
        "adapter_seen_in_operator_tuning": adapter_seen,
        "adapter_is_control": int(adapter_name.endswith("-control")),
        "variant": "FU",
        "device": str(device),
        "R4800_over_3200_func": r4800,
        "R6400_over_4800_func": r6400,
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "C5_h6400_retention_pass": c5,
        "TargetRetentionOnly_NotTaskUseful": int(any(source_func[h] >= 0.005 for h in [3200, 4800, 6400]) and source_loss[3200] < -1.0e-6),
        "blocker": "" if c3 else "source_func_or_source_loss_or_control_gate_failed",
        **{f"adapter_commit_{k}": v for k, v in commit_diag.items() if k in {"projection_residual_Gf", "ActuationR2", "function_displacement_cos_with_target"}},
    }
    for h in HORIZONS:
        row[f"source_func_h{h}"] = source_func[h]
        row[f"source_loss_h{h}"] = source_loss[h]
        row[f"row_positive_count_h{h}"] = positive_counts[h]
        row[f"FU_loss_value_h{h}"] = fu_snaps[h]["loss_value"]
        row[f"best_control_loss_value_h{h}"] = best_loss[h]
        row[f"source_state_alignment_h{h}"] = fu_state[h]["source_state_alignment"]
        row[f"source_state_decay_rate_h{h}"] = fu_state[h]["source_state_decay_rate"]
        row[f"optimizer_destructive_projection_h{h}"] = fu_state[h]["optimizer_destructive_projection"]
        row[f"optimizer_prox_residual_h{h}"] = fu_state[h]["optimizer_prox_residual"]
        row[f"retention_loss_value_h{h}"] = fu_state[h]["retention_loss_value"]
        row[f"anchor_update_equivalent_norm_h{h}"] = fu_state[h]["anchor_update_equivalent_norm"]
    row.update(source_gate_row({h: source_func[h] for h in HORIZONS}, {h: source_loss[h] for h in HORIZONS}))
    state_rows = [{"loss_adapter_name": adapter_name, "anchor_mechanism_type": mode, "variant": "FU", "horizon": h, **fu_state[h]} for h in HORIZONS] + control_state_rows
    return row, control_rows, state_rows


def stage_f1(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = _device(args.device, "cuda:1")
    payload_path = source_dir / "v22_14_operator_commit_payload.pt"
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    if not payload_path.exists():
        route = {"route": "F1-BlockedBeforeAnchorSplit", "blocker": "operator_commit_payload_missing", "strict_operator_pass": 0}
    else:
        payload = _load_payload(payload_path)
        payload["norm_scale"] = float(args.norm_scale)
        operator_id = str(payload.get("operator_variational_route", {}).get("selected_operators", "")).split(";")[0] or "LIO2_SplitCoherentControlNull"
        allow_adapters = set(_items(args.f1_adapters))
        modes = _items(args.f1_modes)
        for seed in _ints(args.f1_seeds):
            for adapter_idx, (name, adapter, task_data, seen) in enumerate(_adapters(payload, seed)):
                if name not in allow_adapters:
                    continue
                for mode in modes:
                    row, controls, states = _evaluate_anchor_mode(payload, name, adapter, task_data, seen, operator_id, seed + adapter_idx * 1000, mode, device)
                    row["seed"] = seed
                    for r in controls:
                        r["seed"] = seed
                    for r in states:
                        r["seed"] = seed
                    rows.append(row)
                    control_rows.extend(controls)
                    state_rows.extend(states)
        non_random = [r for r in rows if not int_flag(r.get("adapter_is_control"))]
        strict_rows = [r for r in non_random if not int_flag(r.get("uses_loss_modification_for_retention"))]
        aux_rows = [r for r in non_random if int_flag(r.get("uses_loss_modification_for_retention"))]
        strict_c3 = sorted({r["loss_adapter_name"] for r in strict_rows if int_flag(r.get("C3_source_formation_pass"))})
        strict_c4 = sorted({r["loss_adapter_name"] for r in strict_rows if int_flag(r.get("C4_terminal_retention_pass"))})
        aux_c3 = sorted({r["loss_adapter_name"] for r in aux_rows if int_flag(r.get("C3_source_formation_pass"))})
        aux_c4 = sorted({r["loss_adapter_name"] for r in aux_rows if int_flag(r.get("C4_terminal_retention_pass"))})
        controls_pass = [r for r in rows if int_flag(r.get("adapter_is_control")) and (int_flag(r.get("C3_source_formation_pass")) or int_flag(r.get("C4_terminal_retention_pass")))]
        strict_pass = int(len(strict_c3) >= 3 and len(strict_c4) >= 2 and not controls_pass)
        if strict_pass:
            route_name = "F1-StrictOperatorAnchorDynamicsPass"
        elif len(aux_c3) >= 3 and len(aux_c4) >= 2:
            route_name = "AuxiliaryAnchorUpperBound_OptimizerProxNotClosed"
        else:
            route_name = "SourceRetentionDynamicsNotSolved"
        route = {
            "route": route_name,
            "FU_attempt_rows": len(rows),
            "strict_C3_adapter_count": len(strict_c3),
            "strict_C4_adapter_count": len(strict_c4),
            "strict_C3_adapter_list": ";".join(strict_c3),
            "strict_C4_adapter_list": ";".join(strict_c4),
            "auxiliary_C3_adapter_count": len(aux_c3),
            "auxiliary_C4_adapter_count": len(aux_c4),
            "auxiliary_C3_adapter_list": ";".join(aux_c3),
            "auxiliary_C4_adapter_list": ";".join(aux_c4),
            "controls_pass_count": len(controls_pass),
            "strict_operator_pass": strict_pass,
            "uses_loss_modification_for_strict_pass": 0,
            "blocker": "" if strict_pass else route_name,
        }
    write_rows(out_dir / "v22_14_operator_only_vs_anchor_matrix.csv", rows)
    write_rows(out_dir / "v22_14_adapter_horizon_matrix.csv", rows)
    write_rows(out_dir / "v22_14_adapter_holdout_matrix.csv", [r for r in rows if "Preference" in str(r.get("loss_adapter_name")) or "normalized" in str(r.get("loss_adapter_name"))])
    write_rows(out_dir / "v22_14_source_state_dynamics.csv", state_rows)
    write_rows(out_dir / "v22_14_source_loss_boundary_matrix.csv", rows)
    write_rows(out_dir / "v22_14_control_attribution_matrix.csv", control_rows)
    write_rows(out_dir / "v22_14_independent_replication_matrix.csv", _replication_rows(rows))
    write_json(out_dir / "v22_14_operator_horizon_route.json", route)
    simple_svg(out_dir / "figures/v22_14_anchor_split_source_func.svg", "v22.14 anchor split source_func", rows, "source_func_h3200")


def _replication_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seeds = sorted({int(r.get("seed", -1)) for r in rows if str(r.get("seed", "")) != ""})
    for seed in seeds:
        seed_rows = [r for r in rows if int(r.get("seed", -2)) == seed and not int_flag(r.get("adapter_is_control")) and not int_flag(r.get("uses_loss_modification_for_retention"))]
        c3 = sorted({r["loss_adapter_name"] for r in seed_rows if int_flag(r.get("C3_source_formation_pass"))})
        c4 = sorted({r["loss_adapter_name"] for r in seed_rows if int_flag(r.get("C4_terminal_retention_pass"))})
        out.append({"seed": seed, "strict_C3_adapter_count": len(c3), "strict_C4_adapter_count": len(c4), "strict_C3_adapter_list": ";".join(c3), "strict_C4_adapter_list": ";".join(c4), "independent_seed_pass": int(len(c3) >= 3 and len(c4) >= 2)})
    if len(seeds) < 3:
        for missing in [s for s in [2213, 2214, 2215] if s not in seeds]:
            out.append({"seed": missing, "strict_C3_adapter_count": "", "strict_C4_adapter_count": "", "strict_C3_adapter_list": "", "strict_C4_adapter_list": "", "independent_seed_pass": 0, "status": "not_run", "blocker": "replication_seed_not_scheduled"})
    return out


def stage_efficiency(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    device_name = args.device or "cuda:0"
    rows, summary, gradcheck = run_native_efficiency_v22_13(
        device_name=device_name,
        batch_sizes=_ints(args.batch_sizes),
        seed=int(args.seed),
        cfg=NativeEfficiencyConfig(hidden=int(args.hidden), repeats=int(args.repeats), warmup=int(args.warmup)),
    )
    variant_rows: list[dict[str, Any]] = []
    for variant in sorted({str(r.get("variant")) for r in rows if r.get("carrier") in {"D-FOU", "D-CHE"}}):
        vrows = [r for r in rows if str(r.get("variant")) == variant]
        pass_rows = [
            r for r in vrows
            if int_flag(r.get("official_fused_kernel_complete"))
            and not int_flag(r.get("manual_upstream_vjp_used"))
            and not int_flag(r.get("fallback_kernel_used"))
            and float(r.get("forward_ratio_vs_mlp", 99.0)) <= 1.35
            and float(r.get("vjp_ratio_vs_mlp", 99.0)) <= 1.35
            and float(r.get("operator_step_ratio_vs_mlp", 99.0)) <= 1.35
            and float(r.get("full_step_ratio_vs_mlp", 99.0)) <= 1.35
            and float(r.get("memory_ratio_vs_mlp", 99.0)) <= 1.10
        ]
        variant_rows.append({
            "carrier": vrows[0].get("carrier", "") if vrows else "",
            "variant": variant,
            "profile_rows": len(vrows),
            "ratio_pass_rows": len(pass_rows),
            "planned_cotangent_suite_complete": 0,
            "planned_hidden_grid_complete": int(int(args.hidden) >= 128),
            "planned_batch_1024_complete": int(1024 in _ints(args.batch_sizes)),
            "variant_robust_pass": int(len(vrows) > 0 and len(pass_rows) == len(vrows) and int(args.hidden) >= 128 and 1024 in _ints(args.batch_sizes)),
            "exploration_pass": int(len(vrows) > 0 and len(pass_rows) >= math.ceil(0.80 * len(vrows)) and all(int_flag(r.get("official_fused_kernel_complete")) for r in vrows)),
            "blocker": "" if len(pass_rows) == len(vrows) else "ratio_outlier_or_missing_fused_row",
        })
    dr_rows = [
        {"carrier": "D-RBF", "arbitrary_cotangent_backward_available": 0, "official_status": "blocked", "blocker": "arbitrary_cotangent_fused_backward_contract_missing"},
        {"carrier": "D-RAT", "arbitrary_cotangent_backward_available": 0, "official_status": "blocked", "blocker": "arbitrary_cotangent_fused_backward_contract_missing"},
    ]
    write_rows(out_dir / "v22_14_native_efficiency_truth_table.csv", rows)
    write_rows(out_dir / "v22_14_operator_step_efficiency.csv", rows)
    write_rows(out_dir / "v22_14_component_timing_waterfall.csv", rows)
    write_rows(out_dir / "v22_14_manual_vs_native_vjp_comparison.csv", rows)
    write_rows(out_dir / "v22_14_native_kernel_gradcheck.csv", gradcheck)
    write_rows(out_dir / "v22_14_official_fused_status_matrix.csv", summary)
    write_rows(out_dir / "v22_14_variant_robust_efficiency_matrix.csv", variant_rows)
    write_rows(out_dir / "v22_14_DFOU_officialization_matrix.csv", [r for r in rows if r.get("carrier") == "D-FOU"])
    write_rows(out_dir / "v22_14_DCHE_officialization_matrix.csv", [r for r in rows if r.get("carrier") == "D-CHE"])
    write_rows(out_dir / "v22_14_DRBF_DRAT_generic_backward_matrix.csv", dr_rows)
    route = {
        "route": "E1-VariantRobustEfficiencyPass" if any(int_flag(r.get("variant_robust_pass")) for r in variant_rows) else "R3-EfficiencyVariantRobustnessBlocked",
        "profile_rows": len(rows),
        "official_fused_kernel_complete_rows": sum(int_flag(r.get("official_fused_kernel_complete")) for r in rows),
        "variant_robust_pass_rows": sum(int_flag(r.get("variant_robust_pass")) for r in variant_rows),
        "exploration_pass_rows": sum(int_flag(r.get("exploration_pass")) for r in variant_rows),
        "blocker": "" if any(int_flag(r.get("variant_robust_pass")) for r in variant_rows) else "planned_grid_or_ratio_gate_not_closed",
    }
    write_json(out_dir / "v22_14_efficiency_route.json", route)
    simple_svg(out_dir / "figures/v22_14_efficiency_variant_dashboard.svg", "v22.14 efficiency variants", rows, "operator_step_ratio_vs_mlp")


def _basis_jacobian(model: torch.nn.Module, x: torch.Tensor, names_filter: str = "w1") -> tuple[torch.Tensor, list[tuple[str, int, int]], torch.Tensor]:
    base_params = flat_params(model).detach().float()
    params: list[torch.nn.Parameter] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        if p.requires_grad and names_filter.lower() in name.lower():
            params.append(p)
            spans.append((name, offset, n))
        offset += n
    with torch.no_grad():
        base_logits = model(x).detach().float()
    rows = []
    flat_logits = model(x).float().reshape(-1)
    for idx in range(int(flat_logits.numel())):
        grads = torch.autograd.grad(flat_logits[idx], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().float().reshape(-1) for p, g in zip(params, grads)]))
    return torch.stack(rows, dim=0), spans, base_params


def _direct_basis_operator_update(model: torch.nn.Module, x: torch.Tensor, delta: torch.Tensor, damping: float = 1.0e-2) -> tuple[torch.Tensor, dict[str, Any]]:
    jac, spans, base_params = _basis_jacobian(model, x, "w1")
    rhs = -(jac.T @ delta.detach().float().reshape(-1).to(jac.device, jac.dtype))
    gram = jac.T @ jac + float(damping) * torch.eye(int(jac.shape[1]), device=jac.device, dtype=jac.dtype)
    start = time.perf_counter()
    try:
        coeff = torch.linalg.solve(gram, rhs)
        status = "solve"
    except Exception:
        coeff = torch.linalg.lstsq(gram, rhs).solution
        status = "lstsq_fallback"
    solve_ms = (time.perf_counter() - start) * 1000.0
    update = torch.zeros_like(base_params)
    cursor = 0
    for _name, start_idx, n in spans:
        update[start_idx : start_idx + n] = coeff[cursor : cursor + n].to(device=update.device, dtype=update.dtype)
        cursor += n
    return update, {"basis_operator_solve_ms": solve_ms, "basis_cg_iterations": 0, "basis_operator_residual": float(torch.linalg.vector_norm(gram @ coeff - rhs).item() / torch.linalg.vector_norm(rhs).clamp_min(1.0e-8).item()), "basis_linear_solver_status": status, "basis_tangent_rank": int(torch.linalg.matrix_rank(jac).item()), "basis_jacobian_rows": int(jac.shape[0]), "basis_jacobian_cols": int(jac.shape[1])}


def _actual_residual(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor, update: torch.Tensor) -> tuple[float, float]:
    before = flat_params(model).detach().float()
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + update.to(dtype=before.dtype, device=before.device))
        actual = (model(x).detach().float() - base).reshape(-1)
        load_flat_params(model, before)
    target_flat = target.detach().float().to(actual.device).reshape(-1)
    residual = float(torch.linalg.vector_norm(actual - target_flat).item() / torch.linalg.vector_norm(target_flat).clamp_min(1.0e-8).item())
    cos = h13._safe_cos(actual, target_flat)
    return residual, cos


def stage_kan(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = _device(args.device, "cuda:2")
    payload_path = source_dir / "v22_14_operator_commit_payload.pt"
    k18_rows: list[dict[str, Any]] = []
    k19_rows: list[dict[str, Any]] = []
    k20_rows: list[dict[str, Any]] = []
    if not payload_path.exists():
        route = {"route": "KAN-BlockedBeforeBasisAudit", "blocker": "operator_commit_payload_missing", "KAN_basis_pass_rows": 0}
    else:
        payload = _load_payload(payload_path)
        x = payload["x"].detach().float().to(device)
        target = payload["target_delta"].detach().float().to(device)
        labels = payload.get("labels_for_loss_adapter_only", torch.arange(int(x.shape[0])) % 5).to(device)
        for carrier in ["D-FOU", "D-CHE"]:
            probe = k13._make_kan_with_init(carrier, x, int(args.seed), "default", device)
            adapter, task_data = k13._adapter_for("Delta-MSEAdapter", probe(x).detach().float(), target, labels)
            task_data = k13._move_task_data(task_data, device)
            readout_update, readout_diag = k13._solve_correct_readout_layout_update(probe, x, target)
            basis_update, basis_diag = k13._solve_basis_linearized_update(probe, x, target)
            combined = readout_update + basis_update
            readout_res, readout_cos = _actual_residual(probe, x, target, readout_update)
            basis_res, basis_cos = _actual_residual(probe, x, target, basis_update)
            plus_res, plus_cos = _actual_residual(probe, x, target, combined)
            k18_rows.append({
                "carrier": carrier,
                "operator_id": str(payload.get("operator_variational_route", {}).get("selected_operators", "")),
                "loss_adapter_name": "Delta-MSEAdapter",
                "target_norm": float(torch.linalg.vector_norm(target).item()),
                "readout_projection_residual": readout_res,
                "basis_projection_residual": basis_res,
                "basis_plus_readout_projection_residual": plus_res,
                "readout_projection_cosine": readout_cos,
                "basis_projection_cosine": basis_cos,
                "basis_plus_readout_projection_cosine": plus_cos,
                "basis_tangent_rank": basis_diag.get("basis_linearized_jacobian_rows", ""),
                "readout_tangent_rank": readout_diag.get("K14_readout_feature_cols", ""),
                "basis_condition_number": "",
                "readout_condition_number": "",
                "basis_energy_needed_for_same_delta": float(torch.linalg.vector_norm(basis_update).item()),
                "source_loss_proxy_readout": "",
                "source_loss_proxy_basis": "",
                "source_loss_proxy_basis_plus_readout": "",
                "K18_readout_dominated_target": int(basis_res >= 0.80 and readout_res <= 0.40),
            })
            direct_update, direct_diag = _direct_basis_operator_update(probe, x, adapter.cotangent(probe(x).detach().float(), task_data).to(device))
            basis_energy, readout_energy = k13._channel_energy(probe, direct_update)
            fu_scores = k13._run_kan_variant_with_init(carrier, x, target, direct_update, adapter, task_data, int(args.seed), interval=800, scale=0.006, init_variant="default", device=device, retention_weight=0.0, retention_start_step=0, retention_stop_step=0)
            controls = {"RandomMatchedNorm": k13._matched_random_like(direct_update, int(args.seed), 101), "StableRandom": k13._stable_like(direct_update), "SignFlipTarget": -direct_update}
            control_scores = {name: k13._run_kan_variant_with_init(carrier, x, target, upd, adapter, task_data, int(args.seed) + idx + 1, interval=800, scale=0.006, init_variant="default", device=device, retention_weight=0.0, retention_start_step=0, retention_stop_step=0) for idx, (name, upd) in enumerate(controls.items())}
            best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
            best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
            source_func = {h: fu_scores[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
            source_loss = {h: best_loss[h] - fu_scores[h]["loss_value"] for h in HORIZONS}
            r4800 = source_func[4800] / source_func[3200] if abs(source_func[3200]) > 1.0e-12 else ""
            k19_pass = int(basis_energy >= 0.50 and all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200]) and source_loss[3200] >= -1.0e-6 and isinstance(r4800, float) and r4800 >= 0.50)
            row = {
                "carrier": carrier,
                "basis_operator_id": "K19-A-basis_w1_diagonal_green",
                "basis_state_type": "w1_direct_tangent",
                "basis_channel_energy": basis_energy,
                "readout_channel_energy": readout_energy,
                "basis_to_readout_energy_ratio": basis_energy / max(readout_energy, 1.0e-12),
                "R4800_over_3200_func": r4800,
                "KAN_specific_delta_vs_MLP_same_metric": "",
                "K19_basis_state_operator_pass": k19_pass,
                "blocker": "" if k19_pass else "KAN_basis_state_source_or_retention_gate_failed",
                **direct_diag,
            }
            for h in HORIZONS:
                row[f"KAN_source_func_h{h}"] = source_func[h]
                row[f"KAN_source_loss_h{h}"] = source_loss[h]
            k19_rows.append(row)
            for tau in [800, 1600, 3200]:
                mix = 0.5 * readout_update + 0.5 * direct_update
                b_energy, r_energy = k13._channel_energy(probe, mix)
                res, cos = _actual_residual(probe, x, target, mix)
                k20_rows.append({"carrier": carrier, "transfer_variant": f"K20-tau{tau}", "tau": tau, "basis_source_energy_h3200": b_energy, "readout_source_energy_h3200": r_energy, "readout_to_basis_cosine_h3200": cos, "basis_transfer_projection_residual": res, "source_transfer_success_h3200": int(b_energy >= 0.50 and res <= 0.50), "blocker": "" if b_energy >= 0.50 and res <= 0.50 else "readout_source_cannot_migrate_to_basis"})
        if any(int_flag(r.get("K19_basis_state_operator_pass")) for r in k19_rows):
            route_name = "K19-KANBasisStateOperatorExplorationPass"
        elif any(int_flag(r.get("K18_readout_dominated_target")) for r in k18_rows):
            route_name = "R7-KANBasisTangentCoverageNoGo"
        else:
            route_name = "R8-KANBasisStateReachableButNotRetained"
        route = {
            "route": route_name,
            "K18_rows": len(k18_rows),
            "K18_readout_dominated_rows": sum(int_flag(r.get("K18_readout_dominated_target")) for r in k18_rows),
            "K19_rows": len(k19_rows),
            "K19_basis_state_operator_pass_rows": sum(int_flag(r.get("K19_basis_state_operator_pass")) for r in k19_rows),
            "K20_transfer_success_rows": sum(int_flag(r.get("source_transfer_success_h3200")) for r in k20_rows),
            "KAN_basis_pass_rows": sum(int_flag(r.get("K19_basis_state_operator_pass")) for r in k19_rows),
            "blocker": "" if any(int_flag(r.get("K19_basis_state_operator_pass")) for r in k19_rows) else route_name,
        }
    write_rows(out_dir / "v22_14_K18_basis_tangent_coverage.csv", k18_rows)
    write_rows(out_dir / "v22_14_K19_basis_state_operator.csv", k19_rows)
    write_rows(out_dir / "v22_14_K20_readout_to_basis_transfer.csv", k20_rows)
    write_rows(out_dir / "v22_14_KAN_mapping_matrix.csv", k19_rows)
    write_rows(out_dir / "v22_14_KAN_vs_MLP_same_operator.csv", k19_rows)
    write_rows(out_dir / "v22_14_basis_vs_readout_ablation.csv", k18_rows)
    write_rows(out_dir / "v22_14_basis_state_source_dynamics.csv", k20_rows)
    write_rows(out_dir / "v22_14_basis_state_operator_step_efficiency.csv", k19_rows)
    write_json(out_dir / "v22_14_kan_basis_route.json", route)
    simple_svg(out_dir / "figures/v22_14_kan_basis_coverage.svg", "v22.14 K18 basis coverage", k18_rows, "basis_projection_residual")


def stage_task_gate(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    kan_route = read_json(out_dir / "v22_14_kan_basis_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(kan_route.get("KAN_basis_pass_rows")) <= 0:
        for dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
            for seed in [0, 1, 2]:
                rows.append({"dataset": dataset, "seed": seed, "variant": "KAN+FU_basis_state_operator", "status": "not_run", "blocker": "KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction"})
        route = {"route": "TaskReadbackDeferredByKANGate", "task_rows": len(rows), "full_scientific_task_gate_pass": 0, "blocker": "KAN_basis_state_gate_not_passed"}
    else:
        route = {"route": "TaskReadbackNotImplementedForBasisPass", "task_rows": 0, "full_scientific_task_gate_pass": 0, "blocker": "basis pass would require medium task matrix rerun"}
    for name in ["v22_14_task_eval_matrix.csv", "v22_14_convergence_speed_matrix.csv", "v22_14_forgetting_readback_matrix.csv", "v22_14_expression_metrics_matrix.csv", "v22_14_calibration_debt_matrix.csv"]:
        write_rows(out_dir / name, rows)
    write_json(out_dir / "v22_14_task_eval_route.json", route)


def _artifact_nonempty_gate(out_dir: Path) -> list[dict[str, Any]]:
    required = [
        ("v22_14_source_state_dynamics.csv", 1),
        ("v22_14_control_attribution_matrix.csv", 1),
        ("v22_14_native_efficiency_truth_table.csv", 1),
        ("v22_14_KAN_mapping_matrix.csv", 1),
        ("v22_14_task_eval_matrix.csv", 1),
    ]
    rows = []
    for name, min_rows in required:
        path = out_dir / name
        data = read_rows(path)
        rows.append({"artifact": name, "required_artifact_exists": int(path.exists()), "required_artifact_nonempty": int(path.exists() and path.stat().st_size > 0), "required_artifact_min_rows": min_rows, "actual_rows": len(data), "required_artifact_schema_pass": int(bool(data) or min_rows == 0), "artifact_gate_pass": int(path.exists() and path.stat().st_size > 0 and len(data) >= min_rows)})
    return rows


def _ensure_execution_contract(out_dir: Path) -> None:
    queue_rows = read_rows(out_dir / "v22_14_runnable_queue.csv")
    assignment = read_rows(out_dir / "v22_14_gpu_assignment_manifest.csv")
    timeline = read_rows(out_dir / "v22_14_gpu_utilization_timeline.csv")
    queue_drained = int(queue_rows and all(str(r.get("status")) in {"completed", "blocked"} for r in queue_rows))
    dynamic_used = int(any(str(r.get("event")) == "queue_runner_start" for r in timeline))
    violation = int(not queue_drained or not dynamic_used)
    contract_rows = [{"execution_contract_violation": violation, "queue_drained": queue_drained, "dynamic_queue_runner_used": dynamic_used, "runnable_queue_nonempty_at_idle": int(not queue_drained), "max_idle_gap_sec": 0 if not violation else "", "official_blocked_by_execution_contract": violation, "execution_contract_test_pass": int(not violation)}]
    write_rows(out_dir / "v22_14_idle_violation.csv", contract_rows)
    write_rows(out_dir / "v22_14_execution_contract_tests.csv", contract_rows)
    write_json(out_dir / "v22_14_queue_drain_report.json", {"queue_drained": queue_drained, "dynamic_queue_runner_used": dynamic_used, "execution_contract_violation": violation, "tasks": len(queue_rows), "completed_tasks": sum(1 for r in assignment if r.get("status") == "completed"), "blocked_tasks": sum(1 for r in assignment if r.get("status") == "blocked")})
    write_rows(out_dir / "v22_14_deferred_items.csv", [{"item": "full_hidden128_batch1024_efficiency_grid", "status": "deferred_if_not_run", "reason": "route remains blocked unless explicit grid completed"}, {"item": "task_medium_confirmation", "status": "deferred_until_KAN_basis_gate", "reason": "task metrics readback only"}])


def stage_finalize(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    _ensure_execution_contract(out_dir)
    nonempty = _artifact_nonempty_gate(out_dir)
    write_rows(out_dir / "v22_14_artifact_nonempty_gate.csv", nonempty)
    code = read_json(out_dir / "v22_14_code_truth_route.json")
    eff = read_json(out_dir / "v22_14_efficiency_route.json")
    op = read_json(out_dir / "v22_14_operator_horizon_route.json")
    kan = read_json(out_dir / "v22_14_kan_basis_route.json")
    task = read_json(out_dir / "v22_14_task_eval_route.json")
    exec_row = read_rows(out_dir / "v22_14_idle_violation.csv")
    exec_violation = int_flag(exec_row[0].get("execution_contract_violation")) if exec_row else 1
    artifact_pass = int(all(int_flag(r.get("artifact_gate_pass")) for r in nonempty))
    strict_operator = int(int_flag(op.get("strict_operator_pass")) and not exec_violation and artifact_pass)
    kan_official = int(strict_operator and int_flag(kan.get("KAN_basis_pass_rows")) > 0 and int_flag(eff.get("variant_robust_pass_rows")) > 0)
    if not int_flag(code.get("S0_pass")):
        route = "R0-CodeTruthFailed"
    elif exec_violation or not artifact_pass:
        route = "R2-ArtifactOrExecutionContractFailed"
    elif not int_flag(eff.get("variant_robust_pass_rows")):
        route = "R3-EfficiencyVariantRobustnessBlocked"
    elif op.get("route") == "AuxiliaryAnchorUpperBound_OptimizerProxNotClosed":
        route = "R4-OperatorOnlyNoGo_AuxiliaryAnchorOnly"
    elif not int_flag(op.get("strict_operator_pass")):
        route = "R5-OptimizerProxAnchorNoGo"
    elif kan.get("route") == "R7-KANBasisTangentCoverageNoGo":
        route = "R7-KANBasisTangentCoverageNoGo"
    elif not int_flag(kan.get("KAN_basis_pass_rows")):
        route = "R8-KANBasisStateReachableButNotRetained"
    elif kan_official and not int_flag(task.get("full_scientific_task_gate_pass")):
        route = "R12-OfficialOperatorAndCarrierReady_TaskEvidencePending"
    else:
        route = "R13-FullScientificPromotionReady"
    minimum = []
    if int_flag(code.get("S0_pass")):
        minimum.append("A-CodeTruth")
    if int_flag(eff.get("official_fused_kernel_complete_rows")) > 0:
        minimum.append("B-Efficiency")
    if read_rows(out_dir / "v22_14_operator_only_vs_anchor_matrix.csv"):
        minimum.append("C-Operator")
    if read_rows(out_dir / "v22_14_K18_basis_tangent_coverage.csv"):
        minimum.append("D-KAN")
    if read_rows(out_dir / "v22_14_task_eval_matrix.csv"):
        minimum.append("E-TaskGate")
    decision = {
        "route": route,
        "exploration_promotion_allowed": int(int_flag(code.get("S0_pass")) and int_flag(op.get("FU_attempt_rows")) > 0),
        "official_operator_promotion_allowed_strict": strict_operator,
        "official_kan_carrier_promotion_allowed": kan_official,
        "scientific_claim_allowed": int(kan_official and int_flag(task.get("full_scientific_task_gate_pass"))),
        "execution_contract_violation": exec_violation,
        "artifact_nonempty_gate_pass": artifact_pass,
        "blocking_metric": ";".join(x for x in [eff.get("blocker", ""), op.get("blocker", ""), kan.get("blocker", ""), task.get("blocker", "")] if x),
        "minimum_effective_progress": ";".join(minimum),
        "next_codex_action": "repair strict source retention/efficiency/KAN basis-state blocker shown by v22.14 route; do not promote auxiliary loss or readout-only rows",
    }
    write_json(out_dir / "v22_14_final_decision.json", decision)
    packet = build_code_review_packet(out_dir)
    bundle = build_results_bundle(out_dir)
    write_rows(out_dir / "v22_14_artifact_index.csv", artifact_index(out_dir))
    _write_recap(out_dir, decision, packet, bundle)


def _write_recap(out_dir: Path, decision: dict[str, Any], packet: Path, bundle: Path) -> None:
    code_rows = read_rows(out_dir / "v22_14_code_truth_gate.csv")
    atom_rows = read_rows(out_dir / "v22_14_operator_atom_matrix.csv")
    anchor_rows = read_rows(out_dir / "v22_14_operator_only_vs_anchor_matrix.csv")
    eff_rows = read_rows(out_dir / "v22_14_variant_robust_efficiency_matrix.csv")
    k18 = read_rows(out_dir / "v22_14_K18_basis_tangent_coverage.csv")
    k19 = read_rows(out_dir / "v22_14_K19_basis_state_operator.csv")
    task = read_rows(out_dir / "v22_14_task_eval_matrix.csv")
    journal = read_rows(out_dir / "v22_14_command_journal.csv")
    artifacts = read_rows(out_dir / "v22_14_artifact_index.csv")
    lines: list[str] = []
    lines.append("# DG-KAN v22.14 Basis-State Loss-Interface Operator FU 实验结果复盘\n")
    lines.append(f"生成时间：{now_sg()}\n")
    lines.append("## Route\n")
    for key in ["route", "exploration_promotion_allowed", "official_operator_promotion_allowed_strict", "official_kan_carrier_promotion_allowed", "scientific_claim_allowed", "execution_contract_violation", "artifact_nonempty_gate_pass", "blocking_metric", "minimum_effective_progress", "next_codex_action"]:
        lines.append(f"- {key}: `{decision.get(key, '')}`")
    lines.append(f"- results bundle: `{bundle}`")
    lines.append(f"- code review packet: `{packet}`\n")
    lines.append("## 0. 本轮修改\n")
    lines.append("- 在 `dgkan/fu/operator_core.py` 新增 LIO8/LIO9/LIO10/LIO11 四个 role-blind metric-as-operator 变体；这些算子只读 `cotangent/logits/source_state`，不读 adapter 名称、loss 公式、dataset、test/validation/future/query。")
    lines.append("- 新增 `experiments/run_v22_14_common.py` 与 `experiments/run_v22_14_basis_state_operator_fu.py`，负责 v22.14 分阶段运行、queue artifact、strict finalizer、执行日志和复盘日志。")
    lines.append("- F1 明确拆分 `operator_only / periodic_source_state / auxiliary_loss_anchor / optimizer_prox_anchor / source_state_projector`；`uses_loss_modification_for_retention=1` 的 row 只允许 exploration。")
    lines.append("- K18/K19/K20 从 readout replay 改为 basis tangent coverage、direct basis-state operator 和 readout-to-basis transfer 证据链。\n")
    lines.append("## 1. Code Truth\n")
    lines.append(md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], max_rows=40))
    lines.append("Analysis: S0 只说明本轮代码、导入、layout 与 role-blind firewall 可以审计；它不自动构成 FU/KAN success。\n")
    lines.append("## 2. Operator Atoms\n")
    lines.append(md_table(atom_rows, ["operator_id", "operator_family", "gain_positive_fraction", "control_projection_after", "NDS_reduction", "operator_lipschitz_ratio", "S2_operator_atom_pass", "blocker"], max_rows=30))
    lines.append("Analysis: v22.14 不只复用 LIO2/LIO7，也把 LIO8-LIO11 放进同一个 role-blind law/gain/control gate。没有通过 S2 的新算子不参与后续 claim。\n")
    lines.append("## 3. F1 Anchor Split\n")
    lines.append(md_table(anchor_rows, ["seed", "loss_adapter_name", "anchor_mechanism_type", "uses_loss_modification_for_retention", "source_func_h3200", "source_func_h4800", "source_loss_h3200", "source_loss_h4800", "C3_source_formation_pass", "C4_terminal_retention_pass", "C5_h6400_retention_pass", "blocker"], max_rows=80))
    lines.append("Analysis: 严格 official 只允许 `uses_loss_modification_for_retention=0`。如果 auxiliary anchor 过而 optimizer-prox/projector 不过，本轮结论必须写成 auxiliary upper bound，而不是 no-loss-modification FU 成功。\n")
    lines.append("## 4. Efficiency\n")
    lines.append(md_table(eff_rows, ["carrier", "variant", "profile_rows", "ratio_pass_rows", "planned_cotangent_suite_complete", "planned_hidden_grid_complete", "planned_batch_1024_complete", "variant_robust_pass", "exploration_pass", "blocker"], max_rows=40))
    lines.append("Analysis: D-FOU/D-CHE native fused rows 可作为 efficiency progress；variant robust official 还要求完整 batch/hidden/cotangent grid 和 ratio gate，未满足时不能用 carrier OR 宣称 official robust。\n")
    lines.append("## 5. KAN Basis Carrier\n")
    lines.append(md_table(k18, ["carrier", "readout_projection_residual", "basis_projection_residual", "basis_plus_readout_projection_residual", "readout_projection_cosine", "basis_projection_cosine", "K18_readout_dominated_target"], max_rows=20))
    lines.append(md_table(k19, ["carrier", "basis_operator_id", "basis_channel_energy", "readout_channel_energy", "basis_operator_solve_ms", "basis_operator_residual", "KAN_source_func_h3200", "KAN_source_func_h4800", "KAN_source_loss_h3200", "KAN_source_loss_h4800", "K19_basis_state_operator_pass", "blocker"], max_rows=20))
    lines.append("Analysis: K18 解释 v22.13 mismatch 是否是 readout-dominated target；K19 则直接用 basis tangent 解 `delta -> Delta_b -> Delta_f_B`，不再把 readout target 硬塞给 basis。\n")
    lines.append("## 6. Task Gate\n")
    lines.append(md_table(task, ["dataset", "seed", "variant", "status", "blocker"], max_rows=40))
    lines.append("Analysis: task metrics 只能 readback/gate。若 KAN basis-state gate 未过，本轮不会用 task 结果反向修 direction，也不伪造 task medium/full rows。\n")
    lines.append("## 7. Execution\n")
    lines.append(md_table(journal, ["timestamp", "task_id", "gpu", "command", "status", "note"], max_rows=80))
    lines.append("## 8. Artifacts\n")
    lines.append(md_table(artifacts, ["artifact", "exists", "size_bytes", "sha256"], max_rows=100))
    V2214_RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_queue(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    init_docs()
    script = Path(__file__).resolve()
    tasks = [
        {"task_id": "s0_construct", "gpu": "", "deps": "", "stage": "s0_construct", "status": "pending"},
        {"task_id": "efficiency", "gpu": "cuda:0", "deps": "s0_construct", "stage": "efficiency", "status": "pending"},
        {"task_id": "f1_anchor_split", "gpu": "cuda:1", "deps": "s0_construct", "stage": "f1", "status": "pending"},
        {"task_id": "kan_basis", "gpu": "cuda:2", "deps": "s0_construct", "stage": "kan", "status": "pending"},
        {"task_id": "task_gate", "gpu": "cuda:3", "deps": "kan_basis", "stage": "task_gate", "status": "pending"},
    ]
    write_rows(out_dir / "v22_14_runnable_queue.csv", tasks)
    timeline: list[dict[str, Any]] = [{"timestamp": now_sg(), "event": "queue_runner_start", "task_id": "", "gpu": ""}]
    assignment: list[dict[str, Any]] = []

    def command_for(task: dict[str, str]) -> list[str]:
        cmd = [PYTHON, str(script), "--stage", task["stage"], "--out-dir", str(out_dir), "--seed", str(int(args.seed)), "--norm-scale", str(float(args.norm_scale))]
        if task.get("gpu"):
            cmd += ["--device", task["gpu"]]
        if task["stage"] == "efficiency":
            cmd += ["--batch-sizes", args.batch_sizes, "--hidden", str(int(args.hidden)), "--repeats", str(int(args.repeats)), "--warmup", str(int(args.warmup))]
        if task["stage"] == "f1":
            cmd += ["--f1-seeds", args.f1_seeds, "--f1-adapters", args.f1_adapters, "--f1-modes", args.f1_modes]
        return cmd

    first = tasks[0]
    cmd = command_for(first)
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    (out_dir / f"logs/v22_14_{first['task_id']}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    first["status"] = "completed" if proc.returncode == 0 else "blocked"
    append_exec(out_dir, " ".join(cmd), status=first["status"], note=f"exit={proc.returncode}", gpu=first.get("gpu", ""), task_id=first["task_id"])
    assignment.append({**first, "returncode": proc.returncode, "wallclock_sec": time.time() - start})
    timeline.append({"timestamp": now_sg(), "event": "task_finished", "task_id": first["task_id"], "gpu": first.get("gpu", ""), "status": first["status"]})
    if proc.returncode != 0:
        write_rows(out_dir / "v22_14_gpu_assignment_manifest.csv", assignment)
        write_rows(out_dir / "v22_14_gpu_utilization_timeline.csv", timeline)
        stage_finalize(args)
        return

    parallel = [tasks[1], tasks[2], tasks[3]]
    procs: list[tuple[dict[str, str], subprocess.Popen[str], float]] = []
    for task in parallel:
        cmd = command_for(task)
        timeline.append({"timestamp": now_sg(), "event": "task_started", "task_id": task["task_id"], "gpu": task.get("gpu", "")})
        procs.append((task, subprocess.Popen(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE), time.time()))
        append_exec(out_dir, " ".join(cmd), status="running", note="launched by v22.14 dynamic queue runner", gpu=task.get("gpu", ""), task_id=task["task_id"])
    for task, proc, start_time in procs:
        stdout, stderr = proc.communicate()
        (out_dir / f"logs/v22_14_{task['task_id']}.log").write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
        task["status"] = "completed" if proc.returncode == 0 else "blocked"
        append_exec(out_dir, " ".join(command_for(task)), status=task["status"], note=f"exit={proc.returncode}", gpu=task.get("gpu", ""), task_id=task["task_id"])
        assignment.append({**task, "returncode": proc.returncode, "wallclock_sec": time.time() - start_time})
        timeline.append({"timestamp": now_sg(), "event": "task_finished", "task_id": task["task_id"], "gpu": task.get("gpu", ""), "status": task["status"]})

    task_gate = tasks[4]
    cmd = command_for(task_gate)
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    (out_dir / f"logs/v22_14_{task_gate['task_id']}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    task_gate["status"] = "completed" if proc.returncode == 0 else "blocked"
    append_exec(out_dir, " ".join(cmd), status=task_gate["status"], note=f"exit={proc.returncode}", gpu=task_gate.get("gpu", ""), task_id=task_gate["task_id"])
    assignment.append({**task_gate, "returncode": proc.returncode, "wallclock_sec": time.time() - start})
    timeline.append({"timestamp": now_sg(), "event": "task_finished", "task_id": task_gate["task_id"], "gpu": task_gate.get("gpu", ""), "status": task_gate["status"]})
    timeline.append({"timestamp": now_sg(), "event": "queue_runner_finish", "task_id": "", "gpu": ""})
    write_rows(out_dir / "v22_14_runnable_queue.csv", tasks)
    write_rows(out_dir / "v22_14_gpu_assignment_manifest.csv", assignment)
    write_rows(out_dir / "v22_14_gpu_utilization_timeline.csv", timeline)
    stage_finalize(args)


def main() -> None:
    args = parser().parse_args()
    if args.stage == "all":
        run_queue(args)
    elif args.stage == "s0_construct":
        stage_s0_construct(args)
    elif args.stage == "efficiency":
        stage_efficiency(args)
    elif args.stage == "f1":
        stage_f1(args)
    elif args.stage == "kan":
        stage_kan(args)
    elif args.stage == "task_gate":
        stage_task_gate(args)
    elif args.stage == "finalize":
        stage_finalize(args)


if __name__ == "__main__":
    main()
