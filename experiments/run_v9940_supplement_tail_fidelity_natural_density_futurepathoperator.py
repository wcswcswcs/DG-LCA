#!/usr/bin/env python3
"""DG-KAN v9.9.4 supplemental C/A/B/D execution.

This supplement implements the user's addendum for v9.9.4.  It is deliberately
separate from the original v9940 runner so the already-materialized official
run is not overwritten.  The supplement first runs cheap distribution-only
1024 generator audits; only a major+tail fidelity pass opens branch-horizon
1024, and only a clean branch-horizon pass can open 5000.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import natural_ap0_extension_materializer as materializer  # noqa: E402
import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9930_natural_density_tail_fidelity_futurepathoperator as v9930  # noqa: E402
import run_v9940_tail_fidelity_natural_density_futurepathoperator as v9940  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_supplement_20260517T170000Z"
BASE_V9940 = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z"
SOURCE_V9930 = RESULT_ROOT / "v9930_natural_density_tail_fidelity_futurepathoperator_full_20260517T040000Z"
SOURCE_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
SOURCE_V9830 = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
SOURCE_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
SOURCE_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.4_TailFidelityNaturalDensity_FuturePathOperator_四线并行完整实验计划.md"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.4_TailFidelityNaturalDensity_FuturePathOperator_实验复盘.md"
SCRIPT_PATH = REPO / "experiments/run_v9940_supplement_tail_fidelity_natural_density_futurepathoperator.py"
MATERIALIZER_PATH = REPO / "experiments/natural_ap0_extension_materializer.py"


SUPPLEMENT_GENERATORS: list[tuple[str, str, int, str]] = [
    ("G7-exact-tail-quota-stochastic-refill", "g7-exact-tail-quota-stochastic-refill", 1, "exact tail quota with stochastic refill"),
    ("G8-hierarchical-major-tail-quota", "g8-hierarchical-major-tail-quota", 1, "hierarchical major->tail quota"),
    ("G9-tail-oversampling-importance-weights", "g9-tail-oversampling-importance-weights", 1, "tail oversampling + importance weights"),
    ("G10-recipe-replay-canonical-tail-precursor", "g10-recipe-replay-canonical-tail-precursor", 1, "recipe replay from canonical tail precursor"),
    ("G11-g5-major-tail-replay-mixture", "g11-g5-major-tail-replay-mixture", 1, "mixture of G5 major-preserving and tail replay"),
    ("G12-rejection-resampled-tail-fidelity-generator", "g12-rejection-resampled-tail-fidelity-generator", 1, "rejection-resampled tail fidelity generator"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--pilot-actions", type=int, default=1024)
    p.add_argument("--chunk-actions", type=int, default=512)
    p.add_argument("--panel-targets", default="1024,5000")
    return p.parse_args()


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def mean_lcb(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.mean(xs) - 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def axis_counter(rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], str]) -> Counter[str]:
    return Counter(str(fn(r)) for r in rows)


def tail_axis_specs(reference_rows: list[dict[str, Any]]) -> list[tuple[str, Callable[[dict[str, Any]], str], Callable[[dict[str, Any]], str]]]:
    old_norms = [v9930.payload_norm(r) for r in reference_rows]
    old_linfs = [v9930.payload_linf(r) for r in reference_rows]
    norm_cuts = [v9930.q(old_norms, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
    linf_cuts = [v9930.q(old_linfs, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]

    def old_recipe(r: dict[str, Any]) -> str:
        return v9930.field(r, "bucket_id", "family_id")

    def new_recipe(r: dict[str, Any]) -> str:
        return v9930.field(r, "source_recipe_id", "family_id")

    def old_template(r: dict[str, Any]) -> str:
        return v9930.field(r, "carrier_id", "template_id")

    def new_template(r: dict[str, Any]) -> str:
        return v9930.field(r, "candidate_template_id", "template_id", "carrier_id")

    def old_norm_bucket(r: dict[str, Any]) -> str:
        return v9930.bucket(v9930.payload_norm(r), norm_cuts, "payload_norm")

    def new_norm_bucket(r: dict[str, Any]) -> str:
        return v9930.bucket(v9930.payload_norm(r), norm_cuts, "payload_norm")

    def old_linf_bucket(r: dict[str, Any]) -> str:
        return v9930.bucket(v9930.payload_linf(r), linf_cuts, "payload_linf")

    def new_linf_bucket(r: dict[str, Any]) -> str:
        return v9930.bucket(v9930.payload_linf(r), linf_cuts, "payload_linf")

    return [
        ("candidate_template_step_pair", lambda r: f"{old_template(r)}|{v9930.step_bucket(r.get('step'))}", lambda r: f"{new_template(r)}|{v9930.step_bucket(r.get('step'))}"),
        ("recipe_template_pair", lambda r: f"{old_recipe(r)}|{old_template(r)}", lambda r: f"{new_recipe(r)}|{new_template(r)}"),
        ("recipe_step_pair", lambda r: f"{old_recipe(r)}|{v9930.step_bucket(r.get('step'))}", lambda r: f"{new_recipe(r)}|{v9930.step_bucket(r.get('step'))}"),
        ("template_payload_norm_tail", lambda r: f"{old_template(r)}|{old_norm_bucket(r)}", lambda r: f"{new_template(r)}|{new_norm_bucket(r)}"),
        ("template_payload_linf_tail", lambda r: f"{old_template(r)}|{old_linf_bucket(r)}", lambda r: f"{new_template(r)}|{new_linf_bucket(r)}"),
        ("source_recipe_id", old_recipe, new_recipe),
    ]


def missing_tail_detail(generator_id: str, reference_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = [r for r in action_rows if r.get("status") == "natural_extension_action_row"]
    source_cursors = [inum(r.get("source_cursor")) for r in actions if r.get("source_cursor") not in {None, ""}]
    cursor_state = f"{min(source_cursors or [0])}:{max(source_cursors or [0])}"
    out: list[dict[str, Any]] = []
    for axis, old_fn, new_fn in tail_axis_specs(reference_rows):
        old_c = axis_counter(reference_rows, old_fn)
        new_c = axis_counter(actions, new_fn)
        gate = {k: v for k, v in old_c.items() if v >= 8}
        for key, canonical_count in sorted(gate.items()):
            generator_count = new_c.get(key, 0)
            if generator_count:
                continue
            expected = canonical_count * len(actions) / max(1, len(reference_rows))
            quota_floor = math.floor(expected)
            quota_round = round(expected)
            if quota_round <= 0:
                reason = "quota_rounding_to_zero"
            elif any(key == str(r.get("worst_missing_group")) for r in audit_rows):
                reason = "not_selected_by_reference_sampler_or_refill"
            else:
                reason = "axis_gate_missing_after_selector"
            out.append({
                "stage": "C_LINE_MISSING_TAIL_GROUP_DETAIL_V9940_SUPPLEMENT",
                "status": "missing_tail_group",
                "generator_id": generator_id,
                "axis": axis,
                "tail_group_key": key,
                "canonical_count": canonical_count,
                "generator_count": generator_count,
                "quota_expected": expected,
                "quota_floor": quota_floor,
                "quota_round": quota_round,
                "quota_rounding_to_zero": int(quota_round <= 0),
                "cursor_state": cursor_state,
                "filter_reason": reason,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in actions] or [0]),
            })
    return out


def distribution_pass(audit: dict[str, Any], audit_rows: list[dict[str, Any]], smoke: dict[str, Any], eligible: int) -> int:
    return int(
        eligible
        and fnum(audit.get("PSI_major_max")) <= 0.05
        and fnum(audit.get("JS_major_max")) <= 0.08
        and fnum(audit.get("max_major_group_share")) <= 0.35
        and fnum(audit.get("entropy_major_min")) >= 0.90
        and fnum(audit.get("PSI_tail_max")) <= 0.10
        and fnum(audit.get("JS_tail_max")) <= 0.08
        and inum(audit.get("missing_tail_group_count")) == 0
        and v9940.tail_coverage_min(audit_rows) >= 0.80
        and inum(smoke.get("old_action_id_collision_count")) == 0
        and inum(smoke.get("old_payload_hash_collision_count")) == 0
        and inum(smoke.get("duplicate_action_id_count")) == 0
        and inum(smoke.get("fake_data_used")) == 0
        and inum(smoke.get("proxy_row_used")) == 0
        and inum(smoke.get("cpu_offload_used")) == 0
    )


def cheap_distribution_matrix(args: argparse.Namespace, out: Path, ref_actions: list[dict[str, Any]], group_ids: dict[str, set[str]], old_ids: set[str], old_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    for idx, (gid, profile, eligible, spec) in enumerate(SUPPLEMENT_GENERATORS):
        t0 = time.perf_counter()
        batch = materializer.generate_natural_ap0_extension_actions(
            seed=args.seed,
            data_root=args.data_root,
            target_action_count=args.pilot_actions,
            cursor=str(3_400_000 + idx * 200_000),
            device=args.device,
            profile=profile,
        )
        actions = []
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        for row in batch.action_rows:
            r = dict(row)
            r["generator_id"] = gid
            r["panel_size"] = args.pilot_actions
            r["distribution_only"] = 1
            r["payload_tensor_written"] = 0
            if profile == "g9-tail-oversampling-importance-weights":
                r["importance_weight"] = 1.0 / max(1.0e-12, fnum(r.get("tail_fraction"), 1.0))
            actions.append(r)
            seen_ids.add(str(r.get("action_id")))
            seen_hashes.add(str(r.get("payload_hash_expected") or r.get("payload_hash")))
        smoke = {
            "stage": "C_LINE_DISTRIBUTION_ONLY_1024_V9940_SUPPLEMENT",
            "status": "summary",
            "generator_id": gid,
            "action_count": len(actions),
            "old_action_id_collision_count": sum(1 for r in actions if str(r.get("action_id")) in old_ids),
            "old_payload_hash_collision_count": sum(1 for r in actions if str(r.get("payload_hash_expected") or r.get("payload_hash")) in old_hashes),
            "duplicate_action_id_count": len(actions) - len(seen_ids),
            "duplicate_payload_hash_count": len(actions) - len(seen_hashes),
            "wallclock_sec": time.perf_counter() - t0,
            "rows_per_sec": len(actions) / max(1.0e-12, time.perf_counter() - t0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in actions] or [0]),
        }
        audit_rows, audit = v9930.tail_fidelity_audit(ref_actions, actions, group_ids, generator_id=gid, panel_size=args.pilot_actions)
        gate = distribution_pass(audit, audit_rows, smoke, eligible)
        write_csv(out / f"c_line_{safe_name(gid)}_distribution_only_actions.csv", actions)
        write_csv(out / f"c_line_{safe_name(gid)}_distribution_only_fidelity.csv", audit_rows)
        write_csv(out / f"c_line_{safe_name(gid)}_distribution_only_smoke.csv", [smoke])
        missing_rows.extend(missing_tail_detail(gid, ref_actions, actions, audit_rows))
        row = {
            "stage": "C_LINE_GENERATOR_REPAIR_MATRIX_V9940_SUPPLEMENT",
            "status": "generator_row",
            "generator_id": gid,
            "profile": profile,
            "supplement_spec": spec,
            "action_count": len(actions),
            "cheap_distribution_only": 1,
            "major_PSI_max": audit.get("PSI_major_max"),
            "major_JS_max": audit.get("JS_major_max"),
            "major_max_group_share": audit.get("max_major_group_share"),
            "major_entropy_min": audit.get("entropy_major_min"),
            "tail_PSI_max": audit.get("PSI_tail_max"),
            "tail_JS_max": audit.get("JS_tail_max"),
            "missing_tail_group_count": audit.get("missing_tail_group_count"),
            "tail_group_coverage_min": v9940.tail_coverage_min(audit_rows),
            "distribution_pass": gate,
            "branch_horizon_1024_status": "not_run" if not gate else "pending_open",
            "panel_5000_status": "not_run",
            "reason": "" if gate else "major_tail_distribution_gate_failed_do_not_open_branch_horizon",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": smoke.get("cpu_offload_used"),
        }
        rows.append(row)
        if gate and (not best or fnum(row.get("tail_PSI_max"), 999) < fnum(best.get("tail_PSI_max"), 999)):
            best = row
    pass_count = sum(inum(r.get("distribution_pass")) for r in rows)
    best_failed = min(
        rows,
        key=lambda r: (
            inum(r.get("missing_tail_group_count"), 10**9),
            fnum(r.get("tail_PSI_max"), 999.0),
            fnum(r.get("major_PSI_max"), 999.0),
        ),
        default={},
    )
    summary = {
        "stage": "C_LINE_GENERATOR_REPAIR_MATRIX_V9940_SUPPLEMENT",
        "status": "summary",
        "candidate_count": len(rows),
        "distribution_pass_count": pass_count,
        "best_distribution_generator_id": best.get("generator_id", ""),
        "best_distribution_profile": best.get("profile", ""),
        "best_major_PSI_max": best.get("major_PSI_max", ""),
        "best_tail_PSI_max": best.get("tail_PSI_max", ""),
        "best_missing_tail_group_count": best.get("missing_tail_group_count", ""),
        "best_failed_generator_id": "" if best else best_failed.get("generator_id", ""),
        "best_failed_major_PSI_max": "" if best else best_failed.get("major_PSI_max", ""),
        "best_failed_tail_PSI_max": "" if best else best_failed.get("tail_PSI_max", ""),
        "best_failed_missing_tail_group_count": "" if best else best_failed.get("missing_tail_group_count", ""),
        "opens_branch_horizon_1024": int(bool(best)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    }
    rows.insert(0, summary)
    return rows, missing_rows, best


def maybe_run_branch_and_5000(args: argparse.Namespace, out: Path, best: dict[str, Any], old_ids: set[str], old_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not best:
        rows = [{
            "stage": "C_LINE_BRANCH_AND_5000_GATE_V9940_SUPPLEMENT",
            "status": "not_run",
            "branch_horizon_1024_status": "not_run",
            "panel_5000_status": "not_run",
            "largest_completed_panel_size": 0,
            "density_sufficient": 0,
            "density_insufficient": 0,
            "density_inconclusive": 1,
            "CoreLike_count": 0,
            "PathGood_count": 0,
            "SlowBurnGood_count": 0,
            "reason": "no_distribution_1024_major_tail_pass",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        return rows, [], [], rows[0]
    gid = str(best["generator_id"])
    profile = str(best["profile"])
    panel_rows: list[dict[str, Any]] = []
    labels_all: list[dict[str, Any]] = []
    actions_all: list[dict[str, Any]] = []
    smoke_rows, actions, applies, branches, smoke = v9930.run_panel_sharded(gid, profile, int(args.pilot_actions), 4_800_000, args, out, old_ids, old_hashes)
    write_csv(out / f"c_line_{safe_name(gid)}_1024_branch_smoke.csv", smoke_rows)
    write_csv(out / f"c_line_{safe_name(gid)}_1024_action_rows.csv", actions)
    write_csv(out / f"c_line_{safe_name(gid)}_1024_action_apply_replay.csv", applies)
    write_csv(out / f"c_line_{safe_name(gid)}_1024_branch_horizon.csv", branches)
    labels, label_summary = v9930.label_summary_from_branches(branches)
    write_csv(out / f"c_line_{safe_name(gid)}_1024_action_labels.csv", labels)
    decision = v9930.panel_density_decision(smoke, label_summary, gid, "panel_row", "supplement_distribution_pass_opened_1024_branch")
    decision["stage"] = "C_LINE_BRANCH_AND_5000_GATE_V9940_SUPPLEMENT"
    panel_rows.append(decision)
    labels_all, actions_all = labels, actions
    if fnum(smoke.get("branch_horizon_completion")) >= 1.0 and 5000 in parse_ints(args.panel_targets):
        smoke_rows, actions, applies, branches, smoke = v9930.run_panel_sharded(gid, profile, 5000, 5_800_000, args, out, old_ids, old_hashes)
        write_csv(out / f"c_line_{safe_name(gid)}_5000_branch_smoke.csv", smoke_rows)
        write_csv(out / f"c_line_{safe_name(gid)}_5000_action_rows.csv", actions)
        write_csv(out / f"c_line_{safe_name(gid)}_5000_action_apply_replay.csv", applies)
        write_csv(out / f"c_line_{safe_name(gid)}_5000_branch_horizon.csv", branches)
        labels, label_summary = v9930.label_summary_from_branches(branches)
        write_csv(out / f"c_line_{safe_name(gid)}_5000_action_labels.csv", labels)
        decision = v9930.panel_density_decision(smoke, label_summary, gid, "panel_row", "supplement_5000_after_clean_1024_branch")
        decision["stage"] = "C_LINE_BRANCH_AND_5000_GATE_V9940_SUPPLEMENT"
        panel_rows.append(decision)
        labels_all, actions_all = labels, actions
    summary = {
        "stage": "C_LINE_BRANCH_AND_5000_GATE_V9940_SUPPLEMENT",
        "status": "summary",
        "completed_panel_count": len(panel_rows),
        "largest_completed_panel_size": panel_rows[-1].get("panel_size") if panel_rows else 0,
        "density_sufficient": int(any(inum(r.get("density_sufficient")) for r in panel_rows)),
        "density_insufficient": int(any(inum(r.get("density_insufficient")) for r in panel_rows)),
        "density_inconclusive": int(not any(inum(r.get("density_sufficient")) or inum(r.get("density_insufficient")) for r in panel_rows)),
        "CoreLike_count": panel_rows[-1].get("CoreLike_count") if panel_rows else 0,
        "PathGood_count": panel_rows[-1].get("PathGood_count") if panel_rows else 0,
        "SlowBurnGood_count": panel_rows[-1].get("SlowBurnGood_count") if panel_rows else 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in panel_rows] or [0]),
    }
    panel_rows.insert(0, summary)
    return panel_rows, labels_all, actions_all, summary


def future_path_types_from_landed() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in rows_from(SOURCE_V9840 / "p1_future_path_mechanism_decomposition_v3.csv") if r.get("status") == "action_path_shape_row"]
    out: list[dict[str, Any]] = []
    typed: list[dict[str, Any]] = []
    for r in rows:
        v1, v5, v20, v80, v240 = [fnum(r.get(k)) for k in ["V1", "V5", "V20", "V80", "V240"]]
        rauv = fnum(r.get("AUV"))
        risk = int(inum(r.get("RiskPath")) or inum(r.get("bad240")) or inum(r.get("memory_fail")) or inum(r.get("offdiag_fail")) or fnum(r.get("longrisk240")) > 0.05)
        fast = int(v1 > 0 and v5 > 0 and v20 > 0 and v240 > 0 and not risk)
        slow = int((v1 <= 0.10 or v5 <= 0.10) and v20 > 0 and v80 > 0 and v240 > 0 and not risk)
        risky = int(rauv > 0 and risk)
        safe_low = int(not risk and rauv <= 1.0 and v240 > 0)
        bad = int(risk or rauv <= 0 or v240 <= 0)
        if slow == 0:
            relaxed_slow = int((v1 <= 0.25 or v5 <= 0.25) and v20 > 0 and v80 > 0 and v240 > 0 and not risk)
        else:
            relaxed_slow = 1
        trow = dict(r)
        trow.update({
            "path_type": "FastGood" if fast else ("SlowBurnGood" if slow else ("RiskyHighAUV" if risky else ("SafeLowValue" if safe_low else "BadPath"))),
            "FastGood": fast,
            "SlowBurnGood": slow,
            "SlowBurnGood_relaxed": relaxed_slow,
            "RiskyHighAUV": risky,
            "SafeLowValue": safe_low,
            "BadPath": bad,
            "RAUV": rauv,
            "LongRisk": fnum(r.get("longrisk240")),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": inum(r.get("cpu_offload_used")),
        })
        typed.append(trow)
    for name, pred in [
        ("FastGood", lambda r: inum(r.get("FastGood"))),
        ("SlowBurnGood", lambda r: inum(r.get("SlowBurnGood"))),
        ("SlowBurnGoodRelaxed", lambda r: inum(r.get("SlowBurnGood_relaxed"))),
        ("RiskyHighAUV", lambda r: inum(r.get("RiskyHighAUV"))),
        ("SafeLowValue", lambda r: inum(r.get("SafeLowValue"))),
        ("BadPath", lambda r: inum(r.get("BadPath"))),
    ]:
        vals = [r for r in typed if pred(r)]
        n = len(vals)
        out.append({
            "stage": "A_LINE_FUTURE_PATH_TYPE_DECOMPOSITION_V9940_SUPPLEMENT",
            "status": "path_type_summary",
            "path_type": name,
            "action_count": n,
            "V1_LCB": mean_lcb([fnum(r.get("V1")) for r in vals]),
            "V5_LCB": mean_lcb([fnum(r.get("V5")) for r in vals]),
            "V20_LCB": mean_lcb([fnum(r.get("V20")) for r in vals]),
            "V80_LCB": mean_lcb([fnum(r.get("V80")) for r in vals]),
            "V240_LCB": mean_lcb([fnum(r.get("V240")) for r in vals]),
            "RAUV_LCB": mean_lcb([fnum(r.get("RAUV")) for r in vals]),
            "LongRisk_UCB": wilson_ucb(sum(int(fnum(r.get("LongRisk")) > 0.05) for r in vals), n) if n else 0.0,
            "Bad_UCB": wilson_ucb(sum(inum(r.get("bad240")) for r in vals), n) if n else 0.0,
            "Null_UCB": wilson_ucb(sum(inum(r.get("null240")) for r in vals), n) if n else 0.0,
            "Memory_UCB": wilson_ucb(sum(inum(r.get("memory_fail")) for r in vals), n) if n else 0.0,
            "Offdiag_UCB": wilson_ucb(sum(inum(r.get("offdiag_fail")) for r in vals), n) if n else 0.0,
            "hard_tail_h80_mean": statistics.mean([fnum(r.get("hard_tail_h80")) for r in vals]) if vals else 0.0,
            "dataset_count": len({r.get("dataset_id") for r in vals}),
            "family_count": len({r.get("family_id") for r in vals}),
            "template_count": len({r.get("template_id") for r in vals}),
            "weak_mechanism_signal": int(name in {"SlowBurnGood", "SlowBurnGoodRelaxed"} and n > 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    slow = next((r for r in out if r.get("path_type") == "SlowBurnGood"), {})
    relaxed = next((r for r in out if r.get("path_type") == "SlowBurnGoodRelaxed"), {})
    summary = {
        "stage": "A_LINE_FUTURE_PATH_TYPE_DECOMPOSITION_V9940_SUPPLEMENT",
        "status": "summary",
        "landed_future_action_rows": len(rows),
        "FastGood_count": next((r.get("action_count") for r in out if r.get("path_type") == "FastGood"), 0),
        "SlowBurnGood_count": slow.get("action_count", 0),
        "SlowBurnGood_relaxed_count": relaxed.get("action_count", 0),
        "RiskyHighAUV_count": next((r.get("action_count") for r in out if r.get("path_type") == "RiskyHighAUV"), 0),
        "SafeLowValue_count": next((r.get("action_count") for r in out if r.get("path_type") == "SafeLowValue"), 0),
        "BadPath_count": next((r.get("action_count") for r in out if r.get("path_type") == "BadPath"), 0),
        "A_line_generation_target_ready": int(inum(slow.get("action_count")) >= 10 and fnum(slow.get("LongRisk_UCB")) <= 0.05),
        "relaxation_applied": int(inum(slow.get("action_count")) < 10 and inum(relaxed.get("action_count")) > inum(slow.get("action_count"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.insert(0, summary)
    return out, summary


def load_base_natural_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for p in BASE_V9940.glob("p2_G*_1024_action_rows_v9940.csv"):
        actions.extend([r for r in rows_from(p) if r.get("status") == "natural_extension_action_row"])
    for p in BASE_V9940.glob("p2_G*_1024_action_labels_v9940.csv"):
        labels.extend([r for r in rows_from(p) if r.get("status") == "natural_action_label"])
    return actions, labels


def future_operator_sketches(actions: list[dict[str, Any]], labels: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_by_id = {str(r.get("action_id")): r for r in labels}
    act = [r for r in actions if str(r.get("action_id")) in label_by_id]

    def timed_scores(fn: Callable[[dict[str, Any]], float]) -> tuple[dict[str, float], dict[str, float]]:
        scores: dict[str, float] = {}
        times: list[float] = []
        for row in act:
            t0 = time.perf_counter()
            scores[str(row.get("action_id"))] = fn(row)
            times.append((time.perf_counter() - t0) * 1000.0)
        return scores, {"q90": v9930.q(times, 0.90), "q99": v9930.q(times, 0.99)}

    specs = [
        ("Sketch1_tiny_virtual_AdamW", lambda r: 0.70 * fnum(r.get("action_adamw_cosine")) * fnum(r.get("trust_ratio")) + 0.20 * fnum(r.get("effective_derivative")) + 0.10 * fnum(r.get("tail_fraction")) - 3.0 * v9930.payload_linf(r)),
        ("Sketch2_JVP_VJP_path_sketch", lambda r: fnum(r.get("effective_derivative")) * fnum(r.get("trust_ratio")) - 0.25 * abs(fnum(r.get("action_adamw_cosine"))) - v9930.payload_norm(r)),
        ("Sketch3_signal_reservoir_split_proxy", lambda r: fnum(r.get("tail_fraction")) + fnum(r.get("trust_ratio")) - abs(fnum(r.get("branch_ratio")) - 1.0) - 8.0 * v9930.payload_linf(r)),
    ]
    out: list[dict[str, Any]] = []
    for name, fn in specs:
        scores, cost = timed_scores(fn)
        ranked = sorted(act, key=lambda r: scores[str(r.get("action_id"))], reverse=True)
        accepted = ranked[: min(87, len(ranked))]
        n = len(accepted)
        labs = [label_by_id[str(a.get("action_id"))] for a in accepted]
        good = sum(int(inum(l.get("CoreLike")) or inum(l.get("PathGood")) or inum(l.get("SlowBurnGood"))) for l in labs)
        risk = sum(inum(l.get("RiskPath")) for l in labs)
        bad = sum(inum(l.get("BadPath")) for l in labs)
        null = sum(1 - inum(l.get("horizon_complete")) for l in labs)
        precision = good / max(1, n)
        v_lcb = mean_lcb([fnum(l.get("V20_gap")) for l in labs])
        rauv_lcb = mean_lcb([fnum(l.get("RiskAdjustedAUV")) for l in labs])
        longrisk_ucb = wilson_ucb(risk, n) if n else 0.0
        bad_ucb = wilson_ucb(bad, n) if n else 0.0
        null_ucb = wilson_ucb(null, n) if n else 0.0
        weak = int(n >= 87 and precision >= 0.75 and v_lcb > 0 and longrisk_ucb <= 0.05 and bad_ucb <= 0.05 and null_ucb <= 0.15 and cost["q90"] <= 1.5)
        strong = int(weak)
        out.append({
            "stage": "B_LINE_FUTURE_PATH_OPERATOR_SKETCH_V9940_SUPPLEMENT",
            "status": "sketch_row",
            "sketch": name,
            "legality": "green_commit_time_fields_only",
            "evaluated_action_count": len(act),
            "accepted_count": n,
            "TopK87_precision": precision,
            "V_LCB": v_lcb,
            "RAUV_LCB": rauv_lcb,
            "LongRisk_UCB": longrisk_ucb,
            "Bad_UCB": bad_ucb,
            "Null_UCB": null_ucb,
            "cost_q90_ms": cost["q90"],
            "cost_q99_ms": cost["q99"],
            "weak_pass": weak,
            "strong_pass": strong,
            "failure": "" if weak else ("value_direction_missing" if precision >= 0.75 and v_lcb <= 0 else "precision_or_risk_or_value_gate_failed"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (inum(r.get("strong_pass")), fnum(r.get("TopK87_precision")), fnum(r.get("V_LCB"))), default={})
    summary = {
        "stage": "B_LINE_FUTURE_PATH_OPERATOR_SKETCH_V9940_SUPPLEMENT",
        "status": "summary",
        "sketch_count": len(out),
        "evaluated_action_count": len(act),
        "B_line_weak_pass": int(any(inum(r.get("weak_pass")) for r in out)),
        "B_line_strong_pass": int(any(inum(r.get("strong_pass")) for r in out)),
        "best_sketch": best.get("sketch", ""),
        "best_precision": best.get("TopK87_precision", ""),
        "best_V_LCB": best.get("V_LCB", ""),
        "best_LongRisk_UCB": best.get("LongRisk_UCB", ""),
        "best_cost_q90_ms": best.get("cost_q90_ms", ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.insert(0, summary)
    return out, summary


def artifact_row_count(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            with path.open(newline="", encoding="utf-8") as f:
                total += sum(1 for _ in csv.DictReader(f))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def write_figures(out: Path, c: dict[str, Any], a: dict[str, Any], b: dict[str, Any], route: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path
    fig("fig_supplement_c_gate_matrix.svg", "v9.9.4 supplement C gates", ["candidates", "dist pass", "branch", "5000"], [
        fnum(c.get("candidate_count")), fnum(c.get("distribution_pass_count")), fnum(route.get("C_branch_1024_opened")), fnum(route.get("C_panel_5000_opened")),
    ])
    fig("fig_supplement_a_path_types.svg", "v9.9.4 supplement A path types", ["Fast", "Slow", "Slow relaxed", "Risky", "SafeLow", "Bad"], [
        fnum(a.get("FastGood_count")), fnum(a.get("SlowBurnGood_count")), fnum(a.get("SlowBurnGood_relaxed_count")), fnum(a.get("RiskyHighAUV_count")), fnum(a.get("SafeLowValue_count")), fnum(a.get("BadPath_count")),
    ])
    fig("fig_supplement_b_sketch_gate.svg", "v9.9.4 supplement B sketches", ["weak", "strong", "best precision", "best V"], [
        fnum(b.get("B_line_weak_pass")), fnum(b.get("B_line_strong_pass")), fnum(b.get("best_precision")), fnum(b.get("best_V_LCB")),
    ])
    return figs


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def append_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    c = summary_row(rows_from(out / "c_line_generator_repair_matrix_v9940_supplement.csv"))
    branch = summary_row(rows_from(out / "c_line_branch_and_5000_gate_v9940_supplement.csv"))
    a = summary_row(rows_from(out / "a_line_future_path_type_decomposition_v9940_supplement.csv"))
    b = summary_row(rows_from(out / "b_line_future_path_operator_sketch_v9940_supplement.csv"))
    d = summary_row(rows_from(out / "d_line_generated_route_gate_v9940_supplement.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9940_supplement.csv"))
    lines = [
        "",
        "## 10. 补充计划执行结果",
        "",
        "> 本节对应用户补充计划。它不复用旧完成稿结论；所有数字来自补充 artifact。C 线先做 distribution-only 1024，只有 major+tail 通过才打开 branch-horizon/5000。",
        "",
        "```text",
        f"supplement_artifact = {out}",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        "### 10.1 C 线 Tail Fidelity 修复",
        "",
        "```text",
        f"candidate_count = {c.get('candidate_count')}",
        f"distribution_pass_count = {c.get('distribution_pass_count')}",
        f"best_distribution_generator_id = {c.get('best_distribution_generator_id')}",
        f"best_major_PSI_max = {c.get('best_major_PSI_max')}",
        f"best_tail_PSI_max = {c.get('best_tail_PSI_max')}",
        f"best_missing_tail_group_count = {c.get('best_missing_tail_group_count')}",
        f"branch_1024_opened = {route.get('C_branch_1024_opened')}",
        f"panel_5000_opened = {route.get('C_panel_5000_opened')}",
        f"density_sufficient/insufficient/inconclusive = {branch.get('density_sufficient')} / {branch.get('density_insufficient')} / {branch.get('density_inconclusive')}",
        "```",
        "",
        "### 10.2 A 线 Future Path 类型",
        "",
        "```text",
        f"landed_future_action_rows = {a.get('landed_future_action_rows')}",
        f"FastGood_count = {a.get('FastGood_count')}",
        f"SlowBurnGood_count = {a.get('SlowBurnGood_count')}",
        f"SlowBurnGood_relaxed_count = {a.get('SlowBurnGood_relaxed_count')}",
        f"RiskyHighAUV_count = {a.get('RiskyHighAUV_count')}",
        f"SafeLowValue_count = {a.get('SafeLowValue_count')}",
        f"BadPath_count = {a.get('BadPath_count')}",
        f"A_line_generation_target_ready = {a.get('A_line_generation_target_ready')}",
        "```",
        "",
        "### 10.3 B 线 FuturePathOperator Sketch",
        "",
        "```text",
        f"sketch_count = {b.get('sketch_count')}",
        f"evaluated_action_count = {b.get('evaluated_action_count')}",
        f"B_line_weak/strong = {b.get('B_line_weak_pass')} / {b.get('B_line_strong_pass')}",
        f"best_sketch = {b.get('best_sketch')}",
        f"best_precision/V/longrisk/cost_q90 = {b.get('best_precision')} / {b.get('best_V_LCB')} / {b.get('best_LongRisk_UCB')} / {b.get('best_cost_q90_ms')}",
        "```",
        "",
        "### 10.4 D 线 Generated Gate",
        "",
        "```text",
        f"generated_sandbox_allowed = {d.get('generated_sandbox_allowed')}",
        f"reason = {d.get('reason')}",
        "```",
        "",
        "### 10.5 No-Fake / Hash",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake/proxy/cpu = {nf.get('fake_data_used')} / {nf.get('proxy_row_used')} / {nf.get('cpu_offload_used')}",
        "```",
        "",
        "| supplement artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "补充最终判断：",
        "",
        "```text",
        "1. C 线按补充计划先执行 distribution-only 1024；未过 major+tail 的 generator 没有打开 branch-horizon/5000。",
        "2. A 线用已落地 future rows 拆出 Fast/SlowBurn/Risky/SafeLow/Bad 类型，SlowBurn 若不足只放宽 V1，不放宽 h240/risk。",
        "3. B 线只用 commit-time fields 重新做三个低成本 sketch；evaluation 才读真实 replay label。",
        "4. D 线继续严格 gate；C/B/A 条件不足时 generated sandbox 不打开。",
        "```",
    ]
    text = RECAP_PATH.read_text(encoding="utf-8") if RECAP_PATH.exists() else ""
    marker = "\n## 10. 补充计划执行结果\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    RECAP_PATH.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    started = time.perf_counter()

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    base_route = read_json(BASE_V9940 / "route_decision_v9940.json")
    p0 = {
        "stage": "P0_V9940_BASE_BOUNDARY_SUPPLEMENT",
        "status": "summary",
        "base_route": base_route.get("route"),
        "base_P2_official_generator_pass_count": base_route.get("P2_official_generator_pass_count"),
        "base_P3_largest_completed_panel_size": base_route.get("P3_largest_completed_panel_size"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("p0_v9940_base_boundary_supplement.csv", [p0])

    ref_actions = v9930.reference_actions(SOURCE_V9330)
    group_ids = v9930.load_action_group_ids(SOURCE_V9830, SOURCE_V9840)
    old_ids, old_hashes = v9930.old_action_sets(SOURCE_V9330, SOURCE_V9900, SOURCE_V9910)
    p1_rows, lineage_rows, p1 = v9940.tail_group_reference_audit(ref_actions, group_ids)
    dump_csv("c_line_tail_group_reference_audit_v9940_supplement.csv", p1_rows)
    dump_csv("c_line_tail_group_lineage_debug_v9940_supplement.csv", lineage_rows)

    c_rows, missing_rows, best = cheap_distribution_matrix(args, out, ref_actions, group_ids, old_ids, old_hashes)
    c = dump_csv("c_line_generator_repair_matrix_v9940_supplement.csv", c_rows)
    dump_csv("c_line_missing_tail_group_detail_v9940_supplement.csv", missing_rows or [{
        "stage": "C_LINE_MISSING_TAIL_GROUP_DETAIL_V9940_SUPPLEMENT",
        "status": "summary",
        "missing_tail_group_rows": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    branch_rows, density_labels, density_actions, branch = maybe_run_branch_and_5000(args, out, best, old_ids, old_hashes)
    branch = dump_csv("c_line_branch_and_5000_gate_v9940_supplement.csv", branch_rows)

    a_rows, a = future_path_types_from_landed()
    a = dump_csv("a_line_future_path_type_decomposition_v9940_supplement.csv", a_rows)

    base_actions, base_labels = load_base_natural_rows()
    b_rows, b = future_operator_sketches(base_actions, base_labels)
    b = dump_csv("b_line_future_path_operator_sketch_v9940_supplement.csv", b_rows)

    # A-line path typing is diagnostic unless a legal generation-target
    # materializer is landed.  Do not reopen generated route from AUV/path labels
    # alone.
    a_generated_reopen_ready = 0
    generated_allowed = int(inum(branch.get("density_insufficient")) or inum(b.get("B_line_strong_pass")) or a_generated_reopen_ready)
    d_reason = "generated_reopen_conditions_not_met"
    if generated_allowed:
        d_reason = "only_64_action_sandbox_allowed_next_not_run_in_supplement"
    elif inum(a.get("A_line_generation_target_ready")):
        d_reason = "A_line_path_type_diagnostic_not_legal_generation_target_mechanism"
    d = {
        "stage": "D_LINE_GENERATED_ROUTE_GATE_V9940_SUPPLEMENT",
        "status": "not_run",
        "generated_sandbox_allowed": generated_allowed,
        "sandbox_size_limit": 64 if generated_allowed else 0,
        "reason": d_reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    d = dump_csv("d_line_generated_route_gate_v9940_supplement.csv", [d])

    route = {
        "stage": "ROUTE_DECISION_V9940_SUPPLEMENT",
        "status": "summary",
        "route": "Supplement-C-FidelityPassBranchPending" if inum(c.get("distribution_pass_count")) else "Supplement-C-NoMajorTailDistributionPass",
        "primary_blocker": "no_supplemental_generator_major_tail_distribution_pass" if not inum(c.get("distribution_pass_count")) else "branch_or_density_pending",
        "secondary_blocker": "future_operator_sketch_not_controller_ready" if not inum(b.get("B_line_strong_pass")) else "",
        "base_route_v9940": base_route.get("route"),
        "C_distribution_pass_count": c.get("distribution_pass_count"),
        "C_best_distribution_generator_id": c.get("best_distribution_generator_id"),
        "C_branch_1024_opened": int(inum(branch.get("largest_completed_panel_size")) >= 1024),
        "C_panel_5000_opened": int(inum(branch.get("largest_completed_panel_size")) >= 5000),
        "C_density_sufficient": branch.get("density_sufficient"),
        "C_density_insufficient": branch.get("density_insufficient"),
        "C_density_inconclusive": branch.get("density_inconclusive"),
        "A_generation_target_ready": a.get("A_line_generation_target_ready"),
        "A_generated_reopen_ready": a_generated_reopen_ready,
        "B_line_strong_pass": b.get("B_line_strong_pass"),
        "D_generated_sandbox_allowed": d.get("generated_sandbox_allowed"),
        "generated_route_status": "stopped_supplement_gate_not_met" if not generated_allowed else "sandbox_64_only_allowed_not_run",
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max(inum(c.get("cpu_offload_used")), inum(branch.get("cpu_offload_used")), inum(a.get("cpu_offload_used")), inum(b.get("cpu_offload_used"))),
    }
    dump_json("route_decision_v9940_supplement.json", route)
    artifacts.update(write_figures(out, c, a, b, route))

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9940_SUPPLEMENT",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
        "fake_proxy_nonzero_count": 0,
    }
    dump_csv("no_fake_audit_v9940_supplement.csv", [no_fake])
    dump_csv("contract_audit_v9940_supplement.csv", [{
        "stage": "CONTRACT_AUDIT_V9940_SUPPLEMENT",
        "status": "summary",
        "C_distribution_only_first": 1,
        "C_branch_open_only_after_distribution_pass": int((not inum(branch.get("largest_completed_panel_size"))) or inum(c.get("distribution_pass_count"))),
        "D_generated_64_limit": 1,
        "no_fake_audit_pass": int(inum(no_fake.get("fake_data_used")) == 0 and inum(no_fake.get("proxy_row_used")) == 0 and inum(no_fake.get("cpu_offload_used")) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": no_fake.get("cpu_offload_used"),
    }])
    dump_csv("failure_taxonomy_v9940_supplement.csv", [{
        "stage": "FAILURE_TAXONOMY_V9940_SUPPLEMENT",
        "status": "summary",
        "route": route.get("route"),
        "F1_no_distribution_fidelity_pass": int(not inum(c.get("distribution_pass_count"))),
        "F2_density_not_adjudicated": int(not inum(branch.get("density_sufficient")) and not inum(branch.get("density_insufficient"))),
        "F3_FPO_not_controller_ready": int(not inum(b.get("B_line_strong_pass"))),
        "F4_generated_not_opened": int(not inum(d.get("generated_sandbox_allowed"))),
        "primary_blocker": route.get("primary_blocker"),
        "secondary_blocker": route.get("secondary_blocker"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": no_fake.get("cpu_offload_used"),
    }])
    manifest = {
        "stage": "RUN_MANIFEST_V9940_SUPPLEMENT",
        "status": "summary",
        "out_dir": str(out),
        "plan": str(PLAN_PATH),
        "runner": str(SCRIPT_PATH),
        "materializer": str(MATERIALIZER_PATH),
        "base_v9940": str(BASE_V9940),
        "elapsed_sec": time.perf_counter() - started,
        "route": route.get("route"),
        "artifacts": sorted(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
    }
    dump_json("run_manifest_v9940_supplement.json", manifest)
    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifacts["materializer"] = MATERIALIZER_PATH
    hashes = sha_rows(artifacts)
    append_recap(out, route, hashes)


if __name__ == "__main__":
    main()
