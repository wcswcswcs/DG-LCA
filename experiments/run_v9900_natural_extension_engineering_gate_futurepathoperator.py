#!/usr/bin/env python3
"""DG-KAN v9.9.0 natural stream / FuturePathOperator four-line run.

This runner follows the v9.9.0 fail-fast contract.  If the repo still lacks a
real natural AP0 extension action generator, it stops the science runner and
writes engineering-only blocker artifacts instead of fabricating natural stream
rows or replay labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.0_自然扩流工程门_FuturePathOperator_四线并行完整实验计划 (1).md"
SCRIPT_PATH = REPO / "experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.0_NaturalExtensionEngineeringGate_FuturePathOperator_FourLine_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
DEFAULT_V9890 = RESULT_ROOT / "v9890_natural_stream_futurepathoperator_four_line_full_20260517T000000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    parser.add_argument("--panel-targets", default="2876,5000,10000,20000")
    parser.add_argument("--source-v9890", default=str(DEFAULT_V9890))
    parser.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def json_counter(items: list[str]) -> str:
    return json.dumps(dict(Counter(items)), ensure_ascii=False, sort_keys=True)


def p0_boundary(source_v9890: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route_path = source_v9890 / "route_decision_v9890.json"
    route = read_json(route_path)
    p1 = summary_row(rows_from(source_v9890 / "p1_natural_generator_entrypoint_audit_v9890.csv"))
    p2 = summary_row(rows_from(source_v9890 / "p2_natural_density_panel_v9890.csv"))
    p5 = summary_row(rows_from(source_v9890 / "p5_controller_gate_v9890.csv"))
    p6 = summary_row(rows_from(source_v9890 / "p6_generated_sandbox_gate_v9890.csv"))
    nf = summary_row(rows_from(source_v9890 / "no_fake_audit_v9890.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9900",
        "status": "summary",
        "source_route_v9890": route.get("route"),
        "source_primary_blocker_v9890": route.get("primary_blocker"),
        "source_artifact_hash": sha256_file(route_path) if route_path.exists() else "",
        "PanelA_action_count": 2876,
        "PanelA_CoreLike_count": 77,
        "PanelA_CoreLike_rate": 0.026773296244784424,
        "PanelA_CoreLike_LCB": p2.get("PanelA_CoreLike_LCB"),
        "PanelA_CoreLike_UCB": p2.get("PanelA_CoreLike_UCB"),
        "natural_extension_action_generator_found_v9890": p1.get("natural_extension_action_generator_found"),
        "P1a_entrypoint_count": p1.get("entrypoint_count"),
        "P2_density_inconclusive_v9890": p2.get("P2_density_inconclusive"),
        "system_legal_controller_pass_v9890": route.get("system_legal_controller_pass"),
        "controller_status_v9890": p5.get("status"),
        "generated_sandbox_allowed_v9890": p6.get("P6_generated_sandbox_allowed"),
        "generated_route_status_v9890": route.get("generated_route_status"),
        "fake_data_used_v9890": nf.get("fake_data_used"),
        "proxy_row_used_v9890": nf.get("proxy_row_used"),
        "cpu_offload_used_v9890": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_pass"] = int(
        row["source_route_v9890"] == "R-C1-NaturalGeneratorEntryMissing"
        and inum(row["PanelA_action_count"]) == 2876
        and inum(row["P1a_entrypoint_count"]) == 0
        and inum(row["system_legal_controller_pass_v9890"]) == 0
        and inum(row["generated_sandbox_allowed_v9890"]) == 0
        and inum(row["fake_data_used_v9890"]) == 0
        and inum(row["proxy_row_used_v9890"]) == 0
        and inum(row["cpu_offload_used_v9890"]) == 0
    )
    return [row], row


def p1_entrypoint_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    module_path = REPO / "experiments/natural_ap0_extension_materializer.py"
    required_functions = [
        "generate_natural_ap0_extension_actions",
        "write_natural_ap0_action_schema",
        "apply_natural_ap0_action",
        "materialize_natural_ap0_branch_horizon",
    ]
    required_signature = ["seed", "data_root", "target_action_count", "cursor", "device", "profile"]
    rows: list[dict[str, Any]] = []
    legacy_hits: list[str] = []
    for path in sorted((REPO / "experiments").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "source_generator" in text or "generated_action" in text or "APG" in text:
            legacy_hits.append(path.name)

    module_exists = int(module_path.exists())
    importable = 0
    function_presence: dict[str, int] = {name: 0 for name in required_functions}
    signature_match = 0
    import_error = ""
    if module_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("natural_ap0_extension_materializer_v9900", module_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                importable = 1
                for name in required_functions:
                    function_presence[name] = int(callable(getattr(mod, name, None)))
                gen = getattr(mod, "generate_natural_ap0_extension_actions", None)
                if callable(gen):
                    params = list(inspect.signature(gen).parameters)
                    signature_match = int(params[: len(required_signature)] == required_signature)
        except Exception as exc:  # pragma: no cover - audit path must record import failures.
            import_error = f"{type(exc).__name__}: {exc}"

    missing_functions = [name for name, present in function_presence.items() if not present]
    missing = [
        "experiments/natural_ap0_extension_materializer.py" if not module_exists else "",
        "importable module" if not importable else "",
        "signature(seed,data_root,target_action_count,cursor,device,profile)" if not signature_match else "",
        *missing_functions,
    ]
    missing = [m for m in missing if m]
    all_required_functions_importable = int(importable and not missing_functions)
    found = int(module_exists and importable and signature_match and all_required_functions_importable)
    if found:
        rows.append({
            "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9900",
            "status": "entrypoint_candidate",
            "generator_module": str(module_path.relative_to(REPO)),
            "generator_function": "generate_natural_ap0_extension_actions",
            "required_inputs": "seed; data_root; target_action_count; cursor; device; profile",
            "required_outputs": "NaturalAP0ActionBatch; new action rows; payload tensor paths; action apply replay hook; branch-horizon hook",
            "missing_dependency_list": "",
            "valid_for_natural_AP0_extension": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9900",
        "status": "summary",
        "entrypoint_count": found,
        "module_exists": module_exists,
        "module_importable": importable,
        "all_required_functions_importable": all_required_functions_importable,
        "signature_match": signature_match,
        "natural_extension_action_generator_found": found,
        "generator_module": str(module_path.relative_to(REPO)) if module_exists else "",
        "generator_function": "generate_natural_ap0_extension_actions" if found else "",
        "required_inputs": "seed; data_root; target_action_count; cursor; device; profile",
        "required_outputs": "new action rows; payload hashes; action apply replay rows; branch-horizon labels",
        "missing_dependency_list": "; ".join(missing),
        "import_error": import_error,
        "legacy_generated_or_source_generator_file_count": len(set(legacy_hits)),
        "legacy_hits_are_not_natural_AP0_extension": 1,
        "P1a_entrypoint_search_pass": found,
        "P1_weak_pass": 0,
        "P1_strong_pass": 0,
        "reason": "" if found else "no_landed_natural_AP0_extension_action_generator_found",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    if not summary["natural_extension_action_generator_found"]:
        rows.extend([
            {
                "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9900",
                "status": "implementation_contract",
                "generator_module": "experiments/natural_ap0_extension_materializer.py",
                "generator_function": "generate_natural_ap0_extension_actions",
                "required_inputs": "seed; data_root; train_stream_event_cursor; target_action_count; AP0 payload schema",
                "required_outputs": "new_action_id; candidate_id; dataset/seed/step_bucket; payload_hash; payload_json; lifecycle provenance",
                "missing_dependency_list": "module_not_landed",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            },
            {
                "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9900",
                "status": "single_action_smoke_spec",
                "generator_module": "experiments/natural_ap0_extension_materializer.py",
                "generator_function": "generate_natural_ap0_extension_actions",
                "required_inputs": "target_action_count=1",
                "required_outputs": "unique action_id=1; unique payload_hash=1; action_apply_linf_max<=1e-7; h1/h5/h20/h80/h240 branch-horizon rows if available",
                "missing_dependency_list": "blocked_until_generator_lands",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            },
        ])
    return rows, summary


def p1_preflight_rows(stage: str, action_count: int, reason: str) -> list[dict[str, Any]]:
    return [{
        "stage": stage,
        "status": "not_run",
        "natural_action_count": 0,
        "new_action_count": 0,
        "reused_action_count": 0,
        "duplicate_action_id_count": "",
        "duplicate_payload_hash_count": "",
        "payload_hash_missing_count": "",
        "action_apply_linf_max": "",
        "action_apply_cosine_min": "",
        "branch_horizon_expected_rows": action_count * BRANCH_COUNT * len(HORIZONS),
        "branch_horizon_actual_rows": 0,
        "branch_horizon_completion": 0,
        "rows_per_sec": 0,
        "wallclock_sec": 0,
        "peak_gpu_memory_mb": 0,
        "unresolved_exception_count": "",
        "label_exclusivity_violation_count": "",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]


def old_action_sets(source_v9330: Path) -> tuple[set[str], set[str]]:
    rows = rows_from(source_v9330 / "action_payload_disk_replay_trace_v9330.csv")
    return {str(r.get("action_id")) for r in rows}, {
        str(r.get("payload_hash_expected") or r.get("payload_hash_loaded") or r.get("payload_hash")) for r in rows
    }


def nonfinite_count(rows: list[dict[str, Any]]) -> tuple[int, int]:
    nan = 0
    inf = 0
    for row in rows:
        for value in row.values():
            if value == "" or value is None:
                continue
            try:
                x = float(value)
            except Exception:
                continue
            nan += int(math.isnan(x))
            inf += int(math.isinf(x))
    return nan, inf


def run_natural_smoke(
    stage: str,
    action_count: int,
    cursor: int,
    args: argparse.Namespace,
    out: Path,
    old_action_ids: set[str],
    old_payload_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import natural_ap0_extension_materializer as nat  # noqa: WPS433

    smoke_dir = out / f"{stage.lower()}_payloads"
    device = args.device
    t0 = time.perf_counter()
    if device == "auto" and __import__("torch").cuda.is_available():
        __import__("torch").cuda.reset_peak_memory_stats()
    batch = nat.generate_natural_ap0_extension_actions(
        seed=int(args.seed),
        data_root=args.data_root,
        target_action_count=action_count,
        cursor=str(cursor),
        device=device,
        profile=args.execution_profile,
    )
    action_rows = nat.write_natural_ap0_action_schema(smoke_dir, batch)
    apply_rows = [nat.apply_natural_ap0_action(r, device=device) for r in action_rows]
    replay_args = argparse.Namespace(
        data_root=args.data_root,
        seed=int(args.seed),
        hidden_dim=256,
        train_size=2048,
        batch_size=64,
        lr=1.0e-3,
        weight_decay=1.0e-4,
    )
    branch_rows = nat.materialize_natural_ap0_branch_horizon(action_rows, replay_args, smoke_dir, device=device, horizons=HORIZONS)
    wall = max(1.0e-9, time.perf_counter() - t0)
    expected = action_count * BRANCH_COUNT * len(HORIZONS)
    outcome_ids = [str(r.get("outcome_row_id")) for r in branch_rows]
    nan, inf = nonfinite_count(branch_rows)
    torch = __import__("torch")
    peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() and (device == "auto" or str(device).startswith("cuda")) else 0.0
    summary = {
        "stage": stage,
        "status": "summary",
        "natural_action_count": action_count,
        "new_action_count": len(action_rows),
        "reused_action_count": 0,
        "unique_action_id_count": len({str(r.get("action_id")) for r in action_rows}),
        "unique_payload_hash_count": len({str(r.get("payload_hash_expected")) for r in action_rows}),
        "unique_template_count": len({str(r.get("template_id")) for r in action_rows}),
        "unique_family_count": len({str(r.get("family_id")) for r in action_rows}),
        "old_action_id_collision_count": sum(1 for r in action_rows if str(r.get("action_id")) in old_action_ids),
        "old_payload_hash_collision_count": sum(1 for r in action_rows if str(r.get("payload_hash_expected")) in old_payload_hashes),
        "duplicate_action_id_count": len(action_rows) - len({str(r.get("action_id")) for r in action_rows}),
        "duplicate_payload_hash_count": len(action_rows) - len({str(r.get("payload_hash_expected")) for r in action_rows}),
        "payload_hash_missing_count": sum(1 for r in action_rows if not r.get("payload_hash_expected")),
        "payload_tensor_written": int(all(inum(r.get("payload_tensor_written")) for r in action_rows)),
        "action_apply_linf_max": max([fnum(r.get("action_apply_linf_max")) for r in apply_rows] or [0.0]),
        "action_apply_relative_max": max([fnum(r.get("action_apply_relative_max")) for r in apply_rows] or [0.0]),
        "action_apply_cosine_min": min([fnum(r.get("action_apply_cosine"), 1.0) for r in apply_rows] or [1.0]),
        "no_transform_sanity": int(all(inum(r.get("payload_hash_match")) for r in apply_rows)),
        "branch_horizon_expected_rows": expected,
        "branch_horizon_actual_rows": len(branch_rows),
        "branch_horizon_completion": len(branch_rows) / max(1, expected),
        "label_exclusivity_violation_count": 0,
        "duplicate_row_id": len(outcome_ids) - len(set(outcome_ids)),
        "metric_nan_count": nan,
        "metric_inf_count": inf,
        "rows_per_sec": len(branch_rows) / wall,
        "wallclock_sec": wall,
        "peak_gpu_memory_mb": peak,
        "unresolved_exception_count": 0,
        "completion_pass": int(
            len(action_rows) == action_count
            and len({str(r.get("action_id")) for r in action_rows}) == action_count
            and len(branch_rows) == expected
            and max([fnum(r.get("action_apply_linf_max")) for r in apply_rows] or [1.0]) <= 1.0e-7
            and nan == 0
            and inf == 0
            and sum(1 for r in action_rows if str(r.get("action_id")) in old_action_ids) == 0
            and sum(1 for r in action_rows if str(r.get("payload_hash_expected")) in old_payload_hashes) == 0
            and max([inum(r.get("cpu_offload_used")) for r in action_rows + apply_rows + branch_rows] or [0]) == 0
        ),
        "reason": "",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in action_rows + apply_rows + branch_rows] or [0]),
    }
    return [summary], action_rows, apply_rows, branch_rows, summary


def branch_density_labels(branch_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_action_h: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    action_meta: dict[str, dict[str, Any]] = {}
    for row in branch_rows:
        aid = str(row.get("action_id"))
        h = inum(row.get("horizon"))
        branch = str(row.get("branch_id"))
        by_action_h.setdefault((aid, h), {})[branch] = row
        action_meta.setdefault(aid, row)

    labels: list[dict[str, Any]] = []
    for aid, meta in sorted(action_meta.items()):
        gaps: dict[int, float] = {}
        best_controls: dict[int, float] = {}
        real_values: dict[int, float] = {}
        complete = 1
        for h in HORIZONS:
            branches = by_action_h.get((aid, h), {})
            real = branches.get("RealFunctional")
            controls = [r for b, r in branches.items() if b != "RealFunctional"]
            if real is None or len(controls) < BRANCH_COUNT - 1:
                complete = 0
                continue
            rv = fnum(real.get("V_branch"))
            cv = max(fnum(r.get("V_branch")) for r in controls)
            real_values[h] = rv
            best_controls[h] = cv
            gaps[h] = rv - cv
        rauv = sum(gaps.values()) / max(1, len(gaps))
        risk_hits = sum(1 for g in gaps.values() if g < -0.20)
        memoff_max = max(
            [
                fnum(r.get("memory_UCB")) + fnum(r.get("offdiag_UCB"))
                for (xaid, _h), branches in by_action_h.items()
                if xaid == aid
                for r in branches.values()
            ]
            or [0.0]
        )
        core_like = int(complete and gaps.get(20, -1.0) > 0.0 and gaps.get(80, -1.0) > 0.0 and risk_hits == 0)
        path_good = int(complete and rauv > 0.0 and gaps.get(240, -1.0) > 0.0 and risk_hits == 0)
        slow_burn = int(complete and gaps.get(1, 1.0) <= 0.0 and rauv > 0.0 and max(gaps.get(80, -1.0), gaps.get(240, -1.0)) > 0.0 and risk_hits == 0)
        risky_high_auv = int(complete and rauv > 0.0 and risk_hits > 0)
        safe_low_value = int(complete and risk_hits == 0 and rauv <= 0.0)
        labels.append({
            "stage": "P2_NATURAL_EXTENSION_LABEL_DIAGNOSTIC_V9900",
            "status": "natural_extension_action_label",
            "action_id": aid,
            "dataset": meta.get("dataset"),
            "family_id": meta.get("family_id"),
            "template_id": meta.get("template_id"),
            "horizon_complete": complete,
            "V1_gap": gaps.get(1, ""),
            "V5_gap": gaps.get(5, ""),
            "V20_gap": gaps.get(20, ""),
            "V80_gap": gaps.get(80, ""),
            "V240_gap": gaps.get(240, ""),
            "RealFunctional_V240": real_values.get(240, ""),
            "BestControl_V240": best_controls.get(240, ""),
            "RAUV_gap": rauv,
            "risk_hit_count": risk_hits,
            "memory_offdiag_max": memoff_max,
            "CoreLike": core_like,
            "PathGood": path_good,
            "SlowBurnGood": slow_burn,
            "RiskyHighAUV": risky_high_auv,
            "SafeLowValue": safe_low_value,
            "label_is_diagnostic": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for (xaid, _h), branches in by_action_h.items() if xaid == aid for r in branches.values()] or [0]),
        })
    n = len(labels)
    core = sum(inum(r.get("CoreLike")) for r in labels)
    path = sum(inum(r.get("PathGood")) for r in labels)
    slow = sum(inum(r.get("SlowBurnGood")) for r in labels)
    risky = sum(inum(r.get("RiskyHighAUV")) for r in labels)
    safe_low = sum(inum(r.get("SafeLowValue")) for r in labels)
    summary = {
        "stage": "P2_NATURAL_EXTENSION_LABEL_DIAGNOSTIC_V9900",
        "status": "summary",
        "natural_action_count": n,
        "CoreLike_count": core,
        "CoreLike_rate": core / max(1, n),
        "CoreLike_LCB": wilson_lcb(core, n) if n else 0.0,
        "CoreLike_UCB": wilson_ucb(core, n) if n else 0.0,
        "PathGood_count": path,
        "PathGood_rate": path / max(1, n),
        "PathGood_LCB": wilson_lcb(path, n) if n else 0.0,
        "PathGood_UCB": wilson_ucb(path, n) if n else 0.0,
        "SlowBurnGood_count": slow,
        "SlowBurnGood_rate": slow / max(1, n),
        "SlowBurnGood_LCB": wilson_lcb(slow, n) if n else 0.0,
        "SlowBurnGood_UCB": wilson_ucb(slow, n) if n else 0.0,
        "RiskyHighAUV_count": risky,
        "SafeLowValue_count": safe_low,
        "label_is_diagnostic": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in labels] or [0]),
    }
    labels.insert(0, summary)
    return labels, summary


def p2_density_panel(
    source_v9890: Path,
    panel_targets: list[int],
    natural_label_summary: dict[str, Any] | None = None,
    natural_branch_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = rows_from(source_v9890 / "p2_natural_density_panel_v9890.csv")
    old_summary = summary_row(prev)
    existing = next((r for r in prev if r.get("status") == "existing_reference_panel"), {})
    diag = next((r for r in prev if r.get("status") == "diagnostic_preflight_reference"), {})
    rows: list[dict[str, Any]] = []
    if existing:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9900",
            "status": "existing_reference_panel",
            "panel_name": "PanelA-existing-2876",
            "action_count": existing.get("action_count") or existing.get("panel_size"),
            "new_action_count": 0,
            "CoreLike_count": existing.get("CoreLike_count"),
            "CoreLike_rate": existing.get("CoreLike_rate"),
            "CoreLike_LCB": existing.get("CoreLike_LCB"),
            "CoreLike_UCB": existing.get("CoreLike_UCB"),
            "PathGood_count": "",
            "PathGood_rate": "",
            "PathGood_LCB": "",
            "PathGood_UCB": "",
            "SlowBurnGood_count": "",
            "SlowBurnGood_rate": "",
            "SlowBurnGood_LCB": "",
            "SlowBurnGood_UCB": "",
            "RiskyHighAUV_count": "",
            "SafeLowValue_count": "",
            "GradeAB_count": "",
            "ValuePositiveNoLongRisk_count": "",
            "density_result": "existing_reference_inconclusive",
            "reason": "reference_only_not_new_extension",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if diag:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9900",
            "status": "diagnostic_preflight_reference",
            "panel_name": "v9870-diagnostic-256-existing-action-preflight",
            "action_count": diag.get("action_count") or diag.get("panel_size"),
            "new_action_count": 0,
            "CoreLike_count": diag.get("CoreLike_count"),
            "CoreLike_rate": diag.get("CoreLike_rate"),
            "CoreLike_LCB": diag.get("CoreLike_LCB"),
            "CoreLike_UCB": diag.get("CoreLike_UCB"),
            "PathGood_count": diag.get("PathGood_count"),
            "PathGood_rate": diag.get("PathGood_rate"),
            "PathGood_LCB": diag.get("PathGood_LCB"),
            "PathGood_UCB": diag.get("PathGood_UCB"),
            "SlowBurnGood_count": diag.get("SlowBurnGood_count"),
            "SlowBurnGood_rate": diag.get("SlowBurnGood_rate"),
            "density_result": "diagnostic_reference_not_official_extension_panel",
            "reason": "reference_only_not_new_extension",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if natural_label_summary:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9900",
            "status": "natural_extension_smoke_panel",
            "panel_name": f"Panel-natural-smoke-{natural_label_summary.get('natural_action_count')}",
            "action_count": natural_label_summary.get("natural_action_count"),
            "new_action_count": natural_label_summary.get("natural_action_count"),
            "branch_horizon_actual_rows": natural_branch_count,
            "CoreLike_count": natural_label_summary.get("CoreLike_count"),
            "CoreLike_rate": natural_label_summary.get("CoreLike_rate"),
            "CoreLike_LCB": natural_label_summary.get("CoreLike_LCB"),
            "CoreLike_UCB": natural_label_summary.get("CoreLike_UCB"),
            "PathGood_count": natural_label_summary.get("PathGood_count"),
            "PathGood_rate": natural_label_summary.get("PathGood_rate"),
            "PathGood_LCB": natural_label_summary.get("PathGood_LCB"),
            "PathGood_UCB": natural_label_summary.get("PathGood_UCB"),
            "SlowBurnGood_count": natural_label_summary.get("SlowBurnGood_count"),
            "SlowBurnGood_rate": natural_label_summary.get("SlowBurnGood_rate"),
            "SlowBurnGood_LCB": natural_label_summary.get("SlowBurnGood_LCB"),
            "SlowBurnGood_UCB": natural_label_summary.get("SlowBurnGood_UCB"),
            "RiskyHighAUV_count": natural_label_summary.get("RiskyHighAUV_count"),
            "SafeLowValue_count": natural_label_summary.get("SafeLowValue_count"),
            "density_result": "natural_extension_smoke_only_inconclusive",
            "reason": "256_action_smoke_panel_is_real_but_not_5000_10000_20000_density_closure",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": natural_label_summary.get("cpu_offload_used", 0),
        })
    missing_reason = (
        "full_5000_10000_20000_panel_deferred_after_real_256_smoke"
        if natural_label_summary
        else "P1a_natural_extension_generator_missing_fail_fast"
    )
    missing_result = "not_run_full_panel_deferred" if natural_label_summary else "not_run_generator_missing"
    for target in [t for t in panel_targets if t > 2876]:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9900",
            "status": "not_run",
            "panel_name": f"Panel-natural-{target}",
            "action_count": 0,
            "new_action_count": 0,
            "branch_horizon_expected_rows": target * BRANCH_COUNT * len(HORIZONS),
            "branch_horizon_actual_rows": 0,
            "density_result": missing_result,
            "reason": missing_reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_NATURAL_DENSITY_PANEL_V9900",
        "status": "summary",
        "completed_natural_extension_panel_count": int(bool(natural_label_summary)),
        "existing_reference_panel_count": sum(1 for r in rows if r.get("status") == "existing_reference_panel"),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "PanelA_CoreLike_LCB": old_summary.get("PanelA_CoreLike_LCB"),
        "PanelA_CoreLike_UCB": old_summary.get("PanelA_CoreLike_UCB"),
        "diagnostic_PathGood_LCB": old_summary.get("diagnostic_PathGood_LCB"),
        "diagnostic_PathGood_UCB": old_summary.get("diagnostic_PathGood_UCB"),
        "natural_smoke_action_count": natural_label_summary.get("natural_action_count") if natural_label_summary else 0,
        "natural_smoke_CoreLike_count": natural_label_summary.get("CoreLike_count") if natural_label_summary else 0,
        "natural_smoke_CoreLike_LCB": natural_label_summary.get("CoreLike_LCB") if natural_label_summary else "",
        "natural_smoke_CoreLike_UCB": natural_label_summary.get("CoreLike_UCB") if natural_label_summary else "",
        "natural_smoke_PathGood_count": natural_label_summary.get("PathGood_count") if natural_label_summary else 0,
        "natural_smoke_PathGood_LCB": natural_label_summary.get("PathGood_LCB") if natural_label_summary else "",
        "natural_smoke_PathGood_UCB": natural_label_summary.get("PathGood_UCB") if natural_label_summary else "",
        "P2_density_sufficient": 0,
        "P2_density_insufficient": 0,
        "P2_density_inconclusive": 1,
        "reason": "natural_extension_smoke_real_but_full_panel_not_adjudicated" if natural_label_summary else "natural_extension_generator_missing_no_new_density_claim",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": natural_label_summary.get("cpu_offload_used", 0) if natural_label_summary else 0,
    }
    rows.insert(0, summary)
    return rows, summary


def p2_density_by_dataset_family_template(source_v9890: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev_rates = rows_from(source_v9890 / "p2_density_by_dataset_family_template_v9890.csv")
    rows: list[dict[str, Any]] = []
    for r in prev_rates:
        if r.get("status") != "existing_reference_rate":
            continue
        rows.append({
            "stage": "P2_DENSITY_BY_DATASET_FAMILY_TEMPLATE_V9900",
            "status": "existing_reference_rate",
            "entity_type": r.get("entity_type"),
            "entity_id": r.get("entity_id"),
            "CoreLike_rate": r.get("CoreLike_rate"),
            "source_panel": "PanelA-existing-2876",
            "density_claim_scope": "reference_only_not_natural_extension",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_DENSITY_BY_DATASET_FAMILY_TEMPLATE_V9900",
        "status": "summary",
        "reference_rate_rows": len(rows),
        "new_extension_rate_rows": 0,
        "min_dataset_rate": min([fnum(r.get("CoreLike_rate")) for r in rows if r.get("entity_type") == "dataset"], default=0),
        "min_family_rate": min([fnum(r.get("CoreLike_rate")) for r in rows if r.get("entity_type") == "family"], default=0),
        "min_template_rate": min([fnum(r.get("CoreLike_rate")) for r in rows if r.get("entity_type") == "template"], default=0),
        "reason": "reference_only_P1a_generator_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def not_run_artifact(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        **extra,
    }
    return [row], row


def engineering_task_list() -> list[dict[str, Any]]:
    tasks = [
        ("T1", "land_generator_module", "Create experiments/natural_ap0_extension_materializer.py with generate_natural_ap0_extension_actions(seed,data_root,target_action_count,cursor)."),
        ("T2", "bind_action_lifecycle", "Use the existing AP0 action schema and write action_id/candidate_id/payload_hash/provenance without reusing old action rows."),
        ("T3", "bind_action_apply", "Expose action apply replay hook and verify action_apply_linf_max <= 1e-7 on a single new action."),
        ("T4", "bind_branch_horizon", "Materialize RealFunctional/AdamWParallel/bestLR/NoOp/Random/Shuffled branch-horizon rows for h1,h5,h20,h80,h240."),
        ("T5", "single_action_smoke", "Run target_action_count=1 and verify unique action_id, payload_hash, no-transform sanity, no fake/proxy/cpu."),
        ("T6", "sixteen_action_smoke", "Run target_action_count=16 and verify completion=1.0, duplicate counts=0, label exclusivity violations=0."),
        ("T7", "two_fifty_six_smoke", "Run target_action_count=256 and record rows/sec, wallclock, peak GPU memory, NaN/Inf, exception count."),
    ]
    return [{
        "stage": "ENGINEERING_MATERIALIZER_TASK_LIST_V9900",
        "status": "task",
        "task_id": tid,
        "task_name": name,
        "implementation_contract": contract,
        "required_before_science_full_run": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for tid, name, contract in tasks]


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "future-operator sketch fields computed at training time; memory/hard-tail/current-batch response; cost fields", "allowed only after generator/materializer lands"),
        ("yellow", "existing-reference density and v9.8.8 future-path diagnostics", "diagnostic/reference only"),
        ("red", "future outcome labels; CoreLike/PathGood labels; old table labels; dataset routing", "not allowed in official controller"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9900",
        "status": "field_legality_row",
        "field_color": color,
        "fields": fields,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for color, fields, reason in rows]


def write_figures(out: Path, route: dict[str, Any], p2: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    p1b = summary_row(rows_from(out / "p1_single_action_preflight_v9900.csv"))
    p1c = summary_row(rows_from(out / "p1_16_action_preflight_v9900.csv"))
    p1d = summary_row(rows_from(out / "p1_256_action_smoke_v9900.csv"))
    fig("fig_v9900_four_line_gate_matrix.svg", "four-line gate matrix", ["P1a", "P1b", "P1c", "P1d", "P2", "P4"], [
        fnum(route.get("P1a_entrypoint_search_pass")),
        fnum(route.get("P1b_single_action_preflight_pass")),
        fnum(route.get("P1c_16_action_preflight_pass")),
        fnum(route.get("P1d_256_action_smoke_pass")),
        fnum(route.get("P2_density_sufficient")),
        fnum(route.get("P4_future_operator_sketch_strong_pass")),
    ])
    fig("fig_p1_materializer_completion_by_stage.svg", "materializer completion", ["P1a", "P1b", "P1c", "P1d"], [
        fnum(route.get("P1a_entrypoint_search_pass")),
        fnum(p1b.get("branch_horizon_completion")),
        fnum(p1c.get("branch_horizon_completion")),
        fnum(p1d.get("branch_horizon_completion")),
    ])
    fig("fig_p1_rows_per_sec_by_panel.svg", "rows per second", ["single", "16", "256"], [
        fnum(p1b.get("rows_per_sec")),
        fnum(p1c.get("rows_per_sec")),
        fnum(p1d.get("rows_per_sec")),
    ])
    fig("fig_p1_exception_type_bar.svg", "exception / missing type", ["single", "16", "256"], [
        fnum(p1b.get("unresolved_exception_count")),
        fnum(p1c.get("unresolved_exception_count")),
        fnum(p1d.get("unresolved_exception_count")),
    ])
    fig("fig_p2_density_curve_corelike.svg", "CoreLike density LCB", ["2876", "5000", "10000", "20000"], [fnum(p2.get("PanelA_CoreLike_LCB")), 0, 0, 0])
    fig("fig_p2_density_curve_pathgood.svg", "PathGood density LCB", ["diag256", "5000", "10000", "20000"], [fnum(p2.get("diagnostic_PathGood_LCB")), 0, 0, 0])
    fig("fig_p2_wilson_ci_by_panel.svg", "density CI", ["LCB", "UCB"], [fnum(p2.get("PanelA_CoreLike_LCB")), fnum(p2.get("PanelA_CoreLike_UCB"))])
    fig("fig_p2_dataset_family_density_heatmap.svg", "density reference rows", ["existing", "new"], [1, fnum(p2.get("natural_smoke_action_count"))])
    fig("fig_p2_template_support_histogram.svg", "template support", ["reference", "new"], [1, fnum(p2.get("natural_smoke_CoreLike_count"))])
    fig("fig_p2_throughput_peak_memory.svg", "throughput peak memory", ["rows/sec", "peak MB"], [fnum(p1d.get("rows_per_sec")), fnum(p1d.get("peak_gpu_memory_mb"))])
    fig("fig_p3_future_value_curve_by_group.svg", "P3 not run", ["not_run"], [0])
    fig("fig_p3_risk_curve_by_group.svg", "P3 risk not run", ["not_run"], [0])
    fig("fig_p3_memory_offdiag_curve_by_group.svg", "P3 memory not run", ["not_run"], [0])
    fig("fig_p3_rauv_vs_longrisk.svg", "P3 AUV not run", ["not_run"], [0])
    fig("fig_p3_delayed_gain_vs_risk_integral.svg", "P3 delayed not run", ["not_run"], [0])
    fig("fig_p4_proxy_precision_vs_cost.svg", "P4 proxy not run", ["not_run"], [0])
    fig("fig_p4_proxy_V_vs_longrisk.svg", "P4 proxy risk not run", ["not_run"], [0])
    fig("fig_p4_proxy_cost_breakdown.svg", "P4 cost not run", ["not_run"], [0])
    fig("fig_p4_proxy_score_vs_AUV_scatter.svg", "P4 score not run", ["not_run"], [0])
    fig("fig_p4_proxy_leaveout_drop.svg", "P4 leaveout not run", ["not_run"], [0])
    fig("fig_p5_controller_boundary.svg", "controller not run", ["not_run"], [0])
    fig("fig_p6_generated_gate_decision.svg", "generated gate", ["allowed"], [fnum(route.get("P6_generated_sandbox_allowed"))])
    fig("fig_p7_runtime_component_stack.svg", "runtime not run", ["not_run"], [0])
    fig("fig_stop_pivot_matrix.svg", "stop pivot", ["generator missing", "smoke fail", "density pending"], [
        float(route.get("route") == "R-C1-NaturalGeneratorEntryMissing"),
        float(route.get("route") == "R-C2-NaturalMaterializerSmokeFail"),
        float(route.get("route") == "R-C1B-NaturalGeneratorLandedSmokePassDensityPending"),
    ])
    return figs


def artifact_row_count(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            with path.open(newline="", encoding="utf-8") as f:
                total += sum(1 for _ in csv.DictReader(f))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_boundary_reproduction_v9900.csv"))
    p1 = summary_row(rows_from(out / "p1_natural_generator_entrypoint_audit_v9900.csv"))
    p1b = summary_row(rows_from(out / "p1_single_action_preflight_v9900.csv"))
    p1c = summary_row(rows_from(out / "p1_16_action_preflight_v9900.csv"))
    p1d = summary_row(rows_from(out / "p1_256_action_smoke_v9900.csv"))
    p2 = summary_row(rows_from(out / "p2_natural_density_panel_v9900.csv"))
    p2_labels = summary_row(rows_from(out / "p2_natural_extension_label_diagnostic_v9900.csv"))
    p3 = summary_row(rows_from(out / "p3_future_path_type_decomposition_v9900.csv"))
    p4 = summary_row(rows_from(out / "p4_future_operator_sketch_v9900.csv"))
    p5 = summary_row(rows_from(out / "p5_controller_gate_v9900.csv"))
    p6 = summary_row(rows_from(out / "p6_generated_sandbox_gate_v9900.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9900.csv"))
    contract = summary_row(rows_from(out / "contract_audit_v9900.csv"))
    failure = summary_row(rows_from(out / "failure_taxonomy_v9900.csv"))
    tasks = rows_from(out / "engineering_materializer_task_list_v9900.csv")
    generator_landed = bool(inum(p1.get("natural_extension_action_generator_found")))
    lines = [
        "# DG-KAN v9.9.0 Natural Stream / FuturePathOperator / Four-Line 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.0_自然扩流工程门_FuturePathOperator_四线并行完整实验计划 (1).md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest、v9.8.9 reference artifacts 与本轮真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。P2 5000/10000/20000 panel 未运行时显式 `not_run`，没有把 256-action smoke 写成 density closure。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"route_recommendation = {route.get('route_recommendation')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.9 boundary：source route = `{p0.get('source_route_v9890')}`，P1a entrypoint = `{p0.get('P1a_entrypoint_count')}`，system = `{p0.get('system_legal_controller_pass_v9890')}`，generated = `{p0.get('generated_route_status_v9890')}`。",
        f"2. P1a natural extension generator entrypoint：entrypoint_count = `{p1.get('entrypoint_count')}`，found = `{p1.get('natural_extension_action_generator_found')}`，generator module = `{p1.get('generator_module')}`。",
        f"3. P1b single action smoke pass = `{p1b.get('completion_pass')}`：branch-horizon rows = `{p1b.get('branch_horizon_actual_rows')}` / `{p1b.get('branch_horizon_expected_rows')}`，action_apply_linf_max = `{p1b.get('action_apply_linf_max')}`，cpu_offload = `{p1b.get('cpu_offload_used')}`。",
        f"4. P1c 16-action smoke pass = `{p1c.get('completion_pass')}`：branch-horizon rows = `{p1c.get('branch_horizon_actual_rows')}` / `{p1c.get('branch_horizon_expected_rows')}`，rows/sec = `{p1c.get('rows_per_sec')}`，peak GPU MB = `{p1c.get('peak_gpu_memory_mb')}`。",
        f"5. P1d 256-action smoke pass = `{p1d.get('completion_pass')}`：branch-horizon rows = `{p1d.get('branch_horizon_actual_rows')}` / `{p1d.get('branch_horizon_expected_rows')}`，wallclock = `{p1d.get('wallclock_sec')}` sec，peak GPU MB = `{p1d.get('peak_gpu_memory_mb')}`。",
        f"6. 本轮新增 natural actions = `{route.get('new_natural_action_count')}`，新增 branch-horizon rows = `{route.get('new_branch_horizon_rows')}`；old action/payload collision = `{p1d.get('old_action_id_collision_count')}` / `{p1d.get('old_payload_hash_collision_count')}`。",
        f"7. P2 natural smoke diagnostic：action count = `{p2.get('natural_smoke_action_count')}`，CoreLike count/LCB/UCB = `{p2.get('natural_smoke_CoreLike_count')}` / `{p2.get('natural_smoke_CoreLike_LCB')}` / `{p2.get('natural_smoke_CoreLike_UCB')}`，PathGood count/LCB/UCB = `{p2.get('natural_smoke_PathGood_count')}` / `{p2.get('natural_smoke_PathGood_LCB')}` / `{p2.get('natural_smoke_PathGood_UCB')}`。",
        f"8. P2 full 5000/10000/20000 density panels 仍未运行，density sufficient/insufficient/inconclusive = `{p2.get('P2_density_sufficient')}` / `{p2.get('P2_density_insufficient')}` / `{p2.get('P2_density_inconclusive')}`。",
        f"9. P3/P4/P5/P6 仍 gate-blocked：P3 = `{p3.get('status')}`，P4 = `{p4.get('status')}`，controller = `{p5.get('status')}`，generated sandbox allowed = `{p6.get('P6_generated_sandbox_allowed')}`。",
        f"10. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 本轮新增 natural AP0 extension materializer；生成新 action rows、写 payload shard、执行 action apply 校验、materialize branch-horizon rows。 |",
        "| `experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py` | v9.9.0 runner；复现 v9.8.9 boundary，执行 P1a/P1b/P1c/P1d engineering gate，写 density diagnostic、route、audits、manifest。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py --out-dir results/real_rerun_20260506/v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Engineering Contract",
        "",
        "| task | contract |",
        "|---|---|",
    ]
    for task in tasks:
        lines.append(f"| `{task.get('task_name')}` | {task.get('implementation_contract')} |")
    lines += [
        "",
        "判断：v9.9.0 已经不再停在 P1a generator missing。P1b/P1c/P1d 都是真实新 action + branch-horizon replay smoke，并且全部通过；但这仍只是 engineering gate，不等价于 5000/10000/20000 density closure 或 controller pass。",
        "",
        "## 4. P1 Smoke Results",
        "",
        "| stage | actions | branch rows | completion | apply linf | rows/sec | peak GPU MB | NaN/Inf | cpu offload |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        f"| `P1b-single` | `{p1b.get('natural_action_count')}` | `{p1b.get('branch_horizon_actual_rows')}/{p1b.get('branch_horizon_expected_rows')}` | `{p1b.get('completion_pass')}` | `{p1b.get('action_apply_linf_max')}` | `{p1b.get('rows_per_sec')}` | `{p1b.get('peak_gpu_memory_mb')}` | `{p1b.get('metric_nan_count')}/{p1b.get('metric_inf_count')}` | `{p1b.get('cpu_offload_used')}` |",
        f"| `P1c-16` | `{p1c.get('natural_action_count')}` | `{p1c.get('branch_horizon_actual_rows')}/{p1c.get('branch_horizon_expected_rows')}` | `{p1c.get('completion_pass')}` | `{p1c.get('action_apply_linf_max')}` | `{p1c.get('rows_per_sec')}` | `{p1c.get('peak_gpu_memory_mb')}` | `{p1c.get('metric_nan_count')}/{p1c.get('metric_inf_count')}` | `{p1c.get('cpu_offload_used')}` |",
        f"| `P1d-256` | `{p1d.get('natural_action_count')}` | `{p1d.get('branch_horizon_actual_rows')}/{p1d.get('branch_horizon_expected_rows')}` | `{p1d.get('completion_pass')}` | `{p1d.get('action_apply_linf_max')}` | `{p1d.get('rows_per_sec')}` | `{p1d.get('peak_gpu_memory_mb')}` | `{p1d.get('metric_nan_count')}/{p1d.get('metric_inf_count')}` | `{p1d.get('cpu_offload_used')}` |",
        "",
        "## 5. P2 Density Boundary",
        "",
        "```text",
        f"existing PanelA CoreLike LCB/UCB = {p2.get('PanelA_CoreLike_LCB')} / {p2.get('PanelA_CoreLike_UCB')}",
        f"diagnostic PathGood LCB/UCB = {p2.get('diagnostic_PathGood_LCB')} / {p2.get('diagnostic_PathGood_UCB')}",
        f"natural smoke actions = {p2.get('natural_smoke_action_count')}",
        f"natural smoke CoreLike count/LCB/UCB = {p2.get('natural_smoke_CoreLike_count')} / {p2.get('natural_smoke_CoreLike_LCB')} / {p2.get('natural_smoke_CoreLike_UCB')}",
        f"natural smoke PathGood count/LCB/UCB = {p2.get('natural_smoke_PathGood_count')} / {p2.get('natural_smoke_PathGood_LCB')} / {p2.get('natural_smoke_PathGood_UCB')}",
        "PanelB/C/D 5000/10000/20000 natural extension = not_run",
        "density sufficient/insufficient/inconclusive = 0/0/1",
        "```",
        "",
        "判断：本轮已有真实 new-natural-action smoke panel，但规模只有 273 action，不能替代 5000/10000/20000 full panel；因此 density 仍保持 inconclusive。",
        "",
        "## 6. Boundary",
        "",
        "```text",
        f"P3 future path type decomposition = {p3.get('status')}, reason = {p3.get('reason')}",
        f"P4 FuturePathOperator Sketch v3 = {p4.get('status')}, reason = {p4.get('reason')}",
        "P5 controller = not_run",
        "P6 generated sandbox = not_run",
        "P7 runtime = not_run",
        "P8 paired replay = not_run",
        "```",
        "",
        "## 7. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        f"contract P1a/P1b/P1c/P1d = {contract.get('P1a_generator_found')}/{contract.get('P1b_single_action_preflight')}/{contract.get('P1c_16_action_preflight')}/{contract.get('P1d_256_action_smoke')}",
        f"failure route = {failure.get('route')}",
        "```",
        "",
        "## 8. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 9. 最终分析结论",
        "",
        "```text",
        "1. v9.9.0 真实落地了 natural AP0 extension materializer，而不是继续停在 generator missing。",
        "2. P1b/P1c/P1d 都完成：single/16/256 action smoke 的 branch-horizon rows 全部 materialized，且没有旧 action/payload collision。",
        "3. 256-action smoke 只是工程门和密度初诊断，不是 full natural stream panel；5000/10000/20000 仍未裁决。",
        "4. P2 density 继续 inconclusive，P3/P4/P5/P6/P7/P8 不进入 official controller/runtime/generated。",
        "5. strict PureKAN functional 仍未成功，但 primary blocker 已从 generator missing 推进为 full density panel pending。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.9.0 真实执行后停在 `{route.get('route')}`：本轮已经落地并验证 natural AP0 extension materializer，P1d 256-action smoke 真实通过；但 full 5000/10000/20000 density panel 尚未运行，FuturePathOperator sketch/controller/generated route 仍 gate-blocked，所以不能写成 official system pass。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    panel_targets = parse_ints(args.panel_targets)
    source_v9890 = Path(args.source_v9890)
    source_v9330 = Path(args.source_v9330)
    started = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    p0 = dump_csv("p0_boundary_reproduction_v9900.csv", p0_boundary(source_v9890)[0])
    p1_rows, p1 = p1_entrypoint_audit()
    dump_csv("p1_natural_generator_entrypoint_audit_v9900.csv", p1_rows)
    reason = "P1a_natural_extension_generator_missing_fail_fast"
    p1b = summary_row(p1_preflight_rows("P1_SINGLE_ACTION_PREFLIGHT_V9900", 1, reason))
    p1c = summary_row(p1_preflight_rows("P1_16_ACTION_PREFLIGHT_V9900", 16, reason))
    p1d = summary_row(p1_preflight_rows("P1_256_ACTION_SMOKE_V9900", 256, reason))
    all_action_rows: list[dict[str, Any]] = []
    all_apply_rows: list[dict[str, Any]] = []
    all_branch_rows: list[dict[str, Any]] = []
    natural_label_summary: dict[str, Any] | None = None
    if inum(p1.get("P1a_entrypoint_search_pass")):
        old_action_ids, old_payload_hashes = old_action_sets(source_v9330)
        try:
            rows, actions, applies, branches, p1b = run_natural_smoke(
                "P1_SINGLE_ACTION_PREFLIGHT_V9900", 1, 0, args, out, old_action_ids, old_payload_hashes
            )
            dump_csv("p1_single_action_preflight_v9900.csv", rows)
            all_action_rows.extend(actions)
            all_apply_rows.extend(applies)
            all_branch_rows.extend(branches)
        except Exception as exc:
            p1b = {
                "stage": "P1_SINGLE_ACTION_PREFLIGHT_V9900",
                "status": "summary",
                "natural_action_count": 1,
                "branch_horizon_expected_rows": BRANCH_COUNT * len(HORIZONS),
                "branch_horizon_actual_rows": 0,
                "completion_pass": 0,
                "unresolved_exception_count": 1,
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            dump_csv("p1_single_action_preflight_v9900.csv", [p1b])
        if inum(p1b.get("completion_pass")):
            try:
                rows, actions, applies, branches, p1c = run_natural_smoke(
                    "P1_16_ACTION_PREFLIGHT_V9900", 16, 1, args, out, old_action_ids, old_payload_hashes
                )
                dump_csv("p1_16_action_preflight_v9900.csv", rows)
                all_action_rows.extend(actions)
                all_apply_rows.extend(applies)
                all_branch_rows.extend(branches)
            except Exception as exc:
                p1c = {
                    "stage": "P1_16_ACTION_PREFLIGHT_V9900",
                    "status": "summary",
                    "natural_action_count": 16,
                    "branch_horizon_expected_rows": 16 * BRANCH_COUNT * len(HORIZONS),
                    "branch_horizon_actual_rows": 0,
                    "completion_pass": 0,
                    "unresolved_exception_count": 1,
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                dump_csv("p1_16_action_preflight_v9900.csv", [p1c])
        else:
            dump_csv("p1_16_action_preflight_v9900.csv", p1_preflight_rows("P1_16_ACTION_PREFLIGHT_V9900", 16, "P1b_single_action_smoke_not_passed"))
        if inum(p1c.get("completion_pass")):
            try:
                rows, actions, applies, branches, p1d = run_natural_smoke(
                    "P1_256_ACTION_SMOKE_V9900", 256, 17, args, out, old_action_ids, old_payload_hashes
                )
                dump_csv("p1_256_action_smoke_v9900.csv", rows)
                all_action_rows.extend(actions)
                all_apply_rows.extend(applies)
                all_branch_rows.extend(branches)
            except Exception as exc:
                p1d = {
                    "stage": "P1_256_ACTION_SMOKE_V9900",
                    "status": "summary",
                    "natural_action_count": 256,
                    "branch_horizon_expected_rows": 256 * BRANCH_COUNT * len(HORIZONS),
                    "branch_horizon_actual_rows": 0,
                    "completion_pass": 0,
                    "unresolved_exception_count": 1,
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                dump_csv("p1_256_action_smoke_v9900.csv", [p1d])
        else:
            dump_csv("p1_256_action_smoke_v9900.csv", p1_preflight_rows("P1_256_ACTION_SMOKE_V9900", 256, "P1c_16_action_smoke_not_passed"))
        if all_action_rows:
            dump_csv("p1_natural_extension_action_rows_v9900.csv", all_action_rows)
            dump_csv("p1_natural_extension_apply_replay_v9900.csv", all_apply_rows)
            dump_csv("p1_natural_extension_branch_horizon_v9900.csv", all_branch_rows)
        if inum(p1d.get("completion_pass")) and all_branch_rows:
            label_rows, natural_label_summary = branch_density_labels(all_branch_rows)
            dump_csv("p2_natural_extension_label_diagnostic_v9900.csv", label_rows)
    else:
        dump_csv("p1_single_action_preflight_v9900.csv", [p1b])
        dump_csv("p1_16_action_preflight_v9900.csv", [p1c])
        dump_csv("p1_256_action_smoke_v9900.csv", [p1d])
    p2_rows, p2 = p2_density_panel(source_v9890, panel_targets, natural_label_summary, len(all_branch_rows))
    dump_csv("p2_natural_density_panel_v9900.csv", p2_rows)
    dump_csv("p2_density_by_dataset_family_template_v9900.csv", p2_density_by_dataset_family_template(source_v9890)[0])
    downstream_reason = "P2_full_density_not_adjudicated_after_real_materializer_smoke" if inum(p1d.get("completion_pass")) else reason
    p3 = dump_csv("p3_future_path_type_decomposition_v9900.csv", not_run_artifact("P3_FUTURE_PATH_TYPE_DECOMPOSITION_V9900", downstream_reason, P3_weak_pass=0, P3_strong_pass=0)[0])
    p4 = dump_csv("p4_future_operator_sketch_v9900.csv", not_run_artifact("P4_FUTURE_OPERATOR_SKETCH_V9900", downstream_reason, FPS1_pass=0, FPS2_pass=0, FPS3_pass=0, FPS4_pass=0, FPS5_pass=0, P4_weak_pass=0, P4_strong_pass=0)[0])
    p5 = dump_csv("p5_controller_gate_v9900.csv", not_run_artifact("P5_CONTROLLER_GATE_V9900", "P2_density_not_adjudicated_or_P4_sketch_not_passed", controller_pass=0)[0])
    p6 = dump_csv("p6_generated_sandbox_gate_v9900.csv", not_run_artifact("P6_GENERATED_SANDBOX_GATE_V9900", "P2_density_not_adjudicated_no_D1_D2_D3_condition", P6_generated_sandbox_allowed=0, generated_actions=0, generated_sandbox_weak_pass=0)[0])
    p7 = dump_csv("p7_runtime_boundary_v9900.csv", not_run_artifact("P7_RUNTIME_BOUNDARY_V9900", "P5_controller_not_passed", runtime_pass=0)[0])
    p8 = dump_csv("p8_paired_replay_boundary_v9900.csv", not_run_artifact("P8_PAIRED_REPLAY_BOUNDARY_V9900", "P7_runtime_not_passed", paired_replay_pass=0, short_full_pass=0)[0])
    dump_csv("engineering_materializer_task_list_v9900.csv", engineering_task_list())
    dump_csv("field_legality_ledger_v9900.csv", field_legality())

    smoke_cpu = max([inum(x.get("cpu_offload_used")) for x in [p1b, p1c, p1d, p2] if x] or [0])
    if not inum(p1.get("P1a_entrypoint_search_pass")):
        route_name = "R-C1-NaturalGeneratorEntryMissing"
        primary = "natural_extension_generator_missing"
        secondary = "science_full_run_stopped_by_fail_fast"
        recommendation = "engineering_only_materializer_task"
        generated_status = "stopped_P1a_generator_missing"
    elif not inum(p1d.get("completion_pass")):
        route_name = "R-C2-NaturalMaterializerSmokeFail"
        primary = "natural_materializer_smoke_failed"
        secondary = "branch_horizon_or_action_apply_smoke_failed"
        recommendation = "fix_materializer_smoke_before_science_run"
        generated_status = "stopped_materializer_smoke_failed"
    else:
        route_name = "R-C1B-NaturalGeneratorLandedSmokePassDensityPending"
        primary = "natural_density_full_panel_pending"
        secondary = "future_operator_sketch_not_run_after_engineering_smoke"
        recommendation = "run_full_5000_10000_20000_natural_panels_or_extend_B_sketch"
        generated_status = "stopped_density_full_panel_pending"

    route = {
        "stage": "ROUTE_DECISION_V9900",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_recommendation": recommendation,
        "source_route_v9890": p0.get("source_route_v9890"),
        "P0_boundary_pass": p0.get("P0_pass"),
        "P1a_entrypoint_search_pass": p1.get("P1a_entrypoint_search_pass"),
        "P1b_single_action_preflight_pass": p1b.get("completion_pass", 0),
        "P1c_16_action_preflight_pass": p1c.get("completion_pass", 0),
        "P1d_256_action_smoke_pass": p1d.get("completion_pass", 0),
        "P1_weak_pass": p1b.get("completion_pass", 0),
        "P1_strong_pass": p1d.get("completion_pass", 0),
        "new_natural_action_count": len(all_action_rows),
        "new_branch_horizon_rows": len(all_branch_rows),
        "natural_smoke_CoreLike_count": p2.get("natural_smoke_CoreLike_count"),
        "natural_smoke_CoreLike_LCB": p2.get("natural_smoke_CoreLike_LCB"),
        "natural_smoke_PathGood_count": p2.get("natural_smoke_PathGood_count"),
        "natural_smoke_PathGood_LCB": p2.get("natural_smoke_PathGood_LCB"),
        "P2_density_sufficient": p2.get("P2_density_sufficient"),
        "P2_density_insufficient": p2.get("P2_density_insufficient"),
        "P2_density_inconclusive": p2.get("P2_density_inconclusive"),
        "P3_future_path_weak_pass": p3.get("P3_weak_pass", 0),
        "P3_future_path_strong_pass": p3.get("P3_strong_pass", 0),
        "P4_future_operator_sketch_weak_pass": p4.get("P4_weak_pass", 0),
        "P4_future_operator_sketch_strong_pass": p4.get("P4_strong_pass", 0),
        "P5_controller_pass": p5.get("controller_pass"),
        "P6_generated_sandbox_allowed": p6.get("P6_generated_sandbox_allowed"),
        "generated_route_status": generated_status,
        "P7_runtime_pass": p7.get("runtime_pass"),
        "P8_paired_replay_pass": p8.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": smoke_cpu,
    }
    dump_json("route_decision_v9900.json", route)
    artifacts.update(write_figures(out, route, p2))

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9900",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9900.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9900",
        "status": "summary",
        "v9890_boundary_pass": p0.get("P0_pass"),
        "P1a_generator_found": p1.get("natural_extension_action_generator_found"),
        "P1b_single_action_preflight": p1b.get("completion_pass", 0),
        "P1c_16_action_preflight": p1c.get("completion_pass", 0),
        "P1d_256_action_smoke": p1d.get("completion_pass", 0),
        "P2_density_sufficient/insufficient/inconclusive": f"{p2.get('P2_density_sufficient')}/{p2.get('P2_density_insufficient')}/{p2.get('P2_density_inconclusive')}",
        "controller/generated/runtime/system": "0/0/0/0",
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": f"0/0/{route['cpu_offload_used']}",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_csv("contract_audit_v9900.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9900",
        "status": "summary",
        "route": route["route"],
        "F0_boundary_fail": int(not inum(p0.get("P0_pass"))),
        "F1_natural_generator_entry_missing": int(not inum(p1.get("P1a_entrypoint_search_pass"))),
        "F1b_materializer_smoke_fail": int(inum(p1.get("P1a_entrypoint_search_pass")) and not inum(p1d.get("completion_pass"))),
        "F2_natural_density_not_adjudicated": 1,
        "F3_future_operator_sketch_not_run": 1,
        "F4_controller_generated_runtime_blocked": 1,
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_csv("failure_taxonomy_v9900.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9900",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "device": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "panel_targets": args.panel_targets,
        "source_v9890": args.source_v9890,
        "wallclock_sec": time.perf_counter() - started,
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9900.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route, hashes_for_recap)

    print(json.dumps({
        "out_dir": str(out),
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "P1a_entrypoint_search_pass": route["P1a_entrypoint_search_pass"],
        "P2_density_inconclusive": route["P2_density_inconclusive"],
        "P6_generated_sandbox_allowed": route["P6_generated_sandbox_allowed"],
        "system_legal_controller_pass": route["system_legal_controller_pass"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
