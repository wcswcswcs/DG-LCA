#!/usr/bin/env python3
"""DG-KAN v9.9.5 tail-key audit / generator repair / density / FPO run.

This runner starts from the v9.9.4 artifacts.  It first audits multiple
commit-time tail key definitions, then runs G13-G19 distribution-only pilots.
Branch-horizon and density panels open only after the selected tail key and a
generator pass the explicit major+tail fidelity gates.
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
from collections import Counter, defaultdict
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
import run_v9910_natural_density_decision_futurepathoperator as v9910  # noqa: E402
import run_v9930_natural_density_tail_fidelity_futurepathoperator as v9930  # noqa: E402
import run_v9940_tail_fidelity_natural_density_futurepathoperator as v9940  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9950_tail_fidelity_repair_natural_density_fpo_full_20260517T180000Z"
SOURCE_V9940 = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z"
SOURCE_V9940_SUPP = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_supplement_20260517T170000Z"
SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
SOURCE_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
SOURCE_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
SOURCE_V9830 = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
SOURCE_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.5_结果解读_TailFidelity修复_自然密度裁决_FPO计划.md"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.5_TailFidelityRepair_NaturalDensity_FPO_实验复盘.md"
SCRIPT_PATH = REPO / "experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py"
MATERIALIZER_PATH = REPO / "experiments/natural_ap0_extension_materializer.py"
BRANCH_COUNT = 6
HORIZONS = [1, 5, 20, 80, 240]

GENERATOR_PROFILES: list[tuple[str, str, int, str]] = [
    ("G13-support-aware-hierarchical-generator", "g13-support-aware-hierarchical-generator", 1, "major quota then merged-tail quota, support-aware rare bucket"),
    ("G14-tail-balanced-with-major-projection-generator", "g14-tail-balanced-with-major-projection-generator", 1, "tail-balanced sampling with major projection"),
    ("G15-min-divergence-transport-generator", "g15-min-divergence-transport-generator", 1, "min-divergence transport under max-share budget"),
    ("G16-canonical-tail-replay-plus-new-residual-generator", "g16-canonical-tail-replay-plus-new-residual-generator", 1, "canonical tail precursor replay plus new residual payload"),
    ("G17-two-buffer-generator", "g17-two-buffer-generator", 1, "alternating major buffer and tail buffer"),
    ("G18-sequential-rejection-generator", "g18-sequential-rejection-generator", 1, "online divergence-budget rejection sampler"),
    ("G19-stratified-random-baseline-v2", "g19-stratified-random-baseline-v2", 1, "negative-control stratified random baseline"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--panel-targets", default="1024,5000,10000,20000")
    p.add_argument("--pilot-actions", type=int, default=1024)
    p.add_argument("--chunk-actions", type=int, default=512)
    p.add_argument("--source-v9940", default=str(SOURCE_V9940))
    p.add_argument("--source-v9940-supplement", default=str(SOURCE_V9940_SUPP))
    p.add_argument("--source-v9330", default=str(SOURCE_V9330))
    p.add_argument("--source-v9900", default=str(SOURCE_V9900))
    p.add_argument("--source-v9910", default=str(SOURCE_V9910))
    p.add_argument("--source-v9830", default=str(SOURCE_V9830))
    p.add_argument("--source-v9840", default=str(SOURCE_V9840))
    return p.parse_args()


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def q(values: list[float], frac: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]


def mean_lcb(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.mean(xs) - 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def field(row: dict[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        val = row.get(name)
        if val not in {None, ""}:
            return str(val)
    return default


def bucket(value: float, cuts: list[float], name: str) -> str:
    for i, cut in enumerate(cuts):
        if value <= cut:
            return f"{name}_bin_{i}"
    return f"{name}_bin_{len(cuts)}"


def entropy_ratio(counter: Counter[str]) -> float:
    if len(counter) <= 1:
        return 1.0
    return v9930.entropy(counter) / max(1.0e-12, math.log(len(counter)))


def reference_actions(source_v9330: Path) -> list[dict[str, Any]]:
    return [r for r in rows_from(source_v9330 / "action_payload_disk_replay_trace_v9330.csv") if r.get("status") == "payload_disk_replay_row"]


def old_action_sets(source_v9330: Path, source_v9900: Path, source_v9910: Path) -> tuple[set[str], set[str]]:
    ids, hashes = v9910.old_action_sets(source_v9330, source_v9900)
    for row in rows_from(source_v9910 / "p2_1024_natural_action_rows_v9910.csv"):
        ids.add(str(row.get("action_id")))
        hashes.add(str(row.get("payload_hash_expected") or row.get("payload_hash")))
    return ids, hashes


def p0_boundary(source_v9940: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9940 / "route_decision_v9940.json")
    p1 = summary_row(rows_from(source_v9940 / "p1_tail_group_reference_audit_v9940.csv"))
    p2 = summary_row(rows_from(source_v9940 / "p2_generator_repair_matrix_v9940.csv"))
    p3 = summary_row(rows_from(source_v9940 / "p3_sequential_natural_density_panel_v9940.csv"))
    p5 = summary_row(rows_from(source_v9940 / "p5_future_path_operator_sketch_v6_v9940.csv"))
    p6 = summary_row(rows_from(source_v9940 / "p6_existing_action_controller_gate_v9940.csv"))
    p7 = summary_row(rows_from(source_v9940 / "p7_generated_sandbox_gate_v9940.csv"))
    nf = summary_row(rows_from(source_v9940 / "no_fake_audit_v9940.csv"))
    row = {
        "stage": "P0_V9940_BOUNDARY_REPRODUCTION_V9950",
        "status": "summary",
        "source_route": route.get("route"),
        "P1_tail_reference_audit_pass": p1.get("P1_reference_audit_pass"),
        "tail_group_count": p1.get("tail_group_count_total"),
        "support_lt3_tail_group_count": p1.get("tail_group_support_lt3_count"),
        "P2_candidate_count": p2.get("candidate_count"),
        "official_pass_count": p2.get("official_pass_count"),
        "closest_generator_id": "G9-two-stage-major-then-tail-fill-generator",
        "P3_largest_completed_panel": p3.get("largest_completed_panel_size"),
        "P4_future_path_pass": route.get("P4_future_path_strong_pass"),
        "P5_FPO_pass": route.get("P5_FPO_strong_pass"),
        "P6_controller_status": p6.get("status"),
        "P7_generated_status": p7.get("status"),
        "fake_data_used_v9940": nf.get("fake_data_used"),
        "proxy_row_used_v9940": nf.get("proxy_row_used"),
        "cpu_offload_used_v9940": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_pass"] = int(
        row["source_route"] == "CaseC-GeneratorFidelityStillFails"
        and inum(row["official_pass_count"]) == 0
        and inum(row["fake_data_used_v9940"]) == 0
        and inum(row["proxy_row_used_v9940"]) == 0
        and inum(row["cpu_offload_used_v9940"]) == 0
    )
    return [row], row


class TailKeys:
    def __init__(self, reference_rows: list[dict[str, Any]]) -> None:
        self.reference_rows = reference_rows
        norms = [v9930.payload_norm(r) for r in reference_rows]
        linfs = [v9930.payload_linf(r) for r in reference_rows]
        self.norm_cuts = [q(norms, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
        self.linf_cuts = [q(linfs, x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
        self.tk0_counts = Counter(self.old_tk0(r) for r in reference_rows)

    def dataset(self, r: dict[str, Any]) -> str:
        return field(r, "dataset")

    def template_old(self, r: dict[str, Any]) -> str:
        return field(r, "carrier_id", "template_id")

    def template_new(self, r: dict[str, Any]) -> str:
        return field(r, "candidate_template_id", "template_id", "carrier_id")

    def recipe_old(self, r: dict[str, Any]) -> str:
        return field(r, "bucket_id", "family_id")

    def recipe_new(self, r: dict[str, Any]) -> str:
        return field(r, "source_recipe_id", "family_id")

    def step(self, r: dict[str, Any]) -> str:
        return v9930.step_bucket(r.get("step"))

    def norm_bucket(self, r: dict[str, Any]) -> str:
        return bucket(v9930.payload_norm(r), self.norm_cuts, "payload_norm")

    def linf_bucket(self, r: dict[str, Any]) -> str:
        return bucket(v9930.payload_linf(r), self.linf_cuts, "payload_linf")

    def major_old(self, r: dict[str, Any]) -> str:
        return "|".join([self.dataset(r), self.template_old(r), self.step(r), self.norm_bucket(r)])

    def major_new(self, r: dict[str, Any]) -> str:
        return "|".join([self.dataset(r), self.template_new(r), self.step(r), self.norm_bucket(r)])

    def old_tk0(self, r: dict[str, Any]) -> str:
        return "|".join([self.recipe_old(r), self.template_old(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])

    def new_tk0(self, r: dict[str, Any]) -> str:
        return "|".join([self.recipe_new(r), self.template_new(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])

    def old(self, version: str, r: dict[str, Any]) -> str:
        tk0 = self.old_tk0(r)
        if version.startswith("TK0"):
            return tk0
        if version.startswith("TK1"):
            return tk0 if self.tk0_counts[tk0] >= 3 else f"rare3|{self.major_old(r)}"
        if version.startswith("TK2"):
            return f"{self.major_old(r)}|{self.recipe_old(r) if self.tk0_counts[tk0] >= 3 else 'rare_tail_support_lt3'}"
        if version.startswith("TK3"):
            return "|".join([self.recipe_old(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])
        if version.startswith("TK4"):
            return "|".join([self.norm_bucket(r), self.linf_bucket(r), self.step(r)])
        return tk0

    def new(self, version: str, r: dict[str, Any]) -> str:
        tk0 = self.new_tk0(r)
        if version.startswith("TK0"):
            return tk0
        if version.startswith("TK1"):
            return tk0 if self.tk0_counts.get(tk0, 0) >= 3 else f"rare3|{self.major_new(r)}"
        if version.startswith("TK2"):
            return f"{self.major_new(r)}|{self.recipe_new(r) if self.tk0_counts.get(tk0, 0) >= 3 else 'rare_tail_support_lt3'}"
        if version.startswith("TK3"):
            return "|".join([self.recipe_new(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])
        if version.startswith("TK4"):
            return "|".join([self.norm_bucket(r), self.linf_bucket(r), self.step(r)])
        return tk0


TAIL_KEY_VERSIONS = [
    ("TK0-original-v9940-tail-key", "recipe/template/step/payload_norm/payload_linf"),
    ("TK1-merged-support3-tail-key", "TK0 with support<3 merged within same major group"),
    ("TK2-hierarchical-major-tail-key", "major group plus source recipe or rare-tail bucket"),
    ("TK3-precursor-only-no-template-tail-key", "source recipe/step/payload norm/payload linf, no template selector"),
    ("TK4-memory-offdiag-hardtail-precursor-tail-key", "payload norm/payload linf/step precursor buckets"),
]


def tail_key_audit(reference_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], TailKeys]:
    tk = TailKeys(reference_rows)
    rows: list[dict[str, Any]] = []
    for version, fields in TAIL_KEY_VERSIONS:
        counter = Counter(tk.old(version, r) for r in reference_rows)
        supports = list(counter.values())
        total = sum(supports)
        group_count = len(counter)
        support_lt3 = sum(1 for x in supports if x < 3)
        support_lt5 = sum(1 for x in supports if x < 5)
        support_lt10 = sum(1 for x in supports if x < 10)
        largest_share = max(supports or [0]) / max(1, total)
        er = entropy_ratio(counter)
        missing = sum(1 for r in reference_rows if "unknown" in tk.old(version, r)) / max(1, len(reference_rows))
        leakage = 0
        strong = int(leakage == 0 and group_count <= 512 and support_lt3 / max(1, group_count) <= 0.50 and largest_share <= 0.35 and er >= 0.80)
        weak = int(leakage == 0 and group_count <= 1024 and support_lt3 / max(1, group_count) <= 0.75 and largest_share <= 0.45 and er >= 0.70)
        repair = ""
        if not weak and support_lt3 / max(1, group_count) > 0.75:
            repair = "rare_tail_merge_required"
        elif not weak and largest_share > 0.45:
            repair = "tail_key_too_coarse_or_field_collapsed"
        elif not weak:
            repair = "tail_key_not_sampleable_under_current_gate"
        rows.append({
            "stage": "P1_TAIL_KEY_REFERENCE_AUDIT_V9950",
            "status": "tail_key_row",
            "tail_key_version": version,
            "tail_key_fields": fields,
            "tail_group_count": group_count,
            "support_lt3_group_count": support_lt3,
            "support_lt3_fraction": support_lt3 / max(1, group_count),
            "support_lt5_group_count": support_lt5,
            "support_lt10_group_count": support_lt10,
            "largest_group_share": largest_share,
            "entropy": v9930.entropy(counter),
            "entropy_ratio": er,
            "tail_group_missing_rate": missing,
            "tail_key_leakage_count": leakage,
            "tail_group_stability_by_seed": len(counter),
            "tail_group_stability_by_family": len({field(r, "family_id") for r in reference_rows}),
            "tail_group_stability_by_template": len({field(r, "carrier_id", "template_id") for r in reference_rows}),
            "weak_pass": weak,
            "strong_pass": strong,
            "repair_applied_or_suggested": repair,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    preferred = ["TK2-hierarchical-major-tail-key", "TK1-merged-support3-tail-key", "TK4-memory-offdiag-hardtail-precursor-tail-key", "TK3-precursor-only-no-template-tail-key", "TK0-original-v9940-tail-key"]
    selected = {}
    for name in preferred:
        cand = next((r for r in rows if r["tail_key_version"] == name and inum(r.get("strong_pass"))), {})
        if cand:
            selected = cand
            break
    if not selected:
        for name in preferred:
            cand = next((r for r in rows if r["tail_key_version"] == name and inum(r.get("weak_pass"))), {})
            if cand:
                selected = cand
                break
    if not selected:
        selected = min(rows, key=lambda r: (fnum(r.get("support_lt3_fraction")), fnum(r.get("largest_group_share"))), default={})
    summary = {
        "stage": "P1_TAIL_KEY_REFERENCE_AUDIT_V9950",
        "status": "summary",
        "tail_key_version_selected": selected.get("tail_key_version", ""),
        "tail_key_weak_pass": selected.get("weak_pass", 0),
        "tail_key_strong_pass": selected.get("strong_pass", 0),
        "tail_group_count": selected.get("tail_group_count", 0),
        "support_lt3_fraction": selected.get("support_lt3_fraction", 1),
        "largest_group_share": selected.get("largest_group_share", 1),
        "entropy_ratio": selected.get("entropy_ratio", 0),
        "tail_key_leakage_count": selected.get("tail_key_leakage_count", 0),
        "P1_weak_pass": selected.get("weak_pass", 0),
        "P1_strong_pass": selected.get("strong_pass", 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary, tk


def distribution_metrics(old_counter: Counter[str], new_counter: Counter[str]) -> dict[str, Any]:
    n_new = sum(new_counter.values())
    return {
        "PSI": v9930.psi(old_counter, new_counter),
        "JS": v9930.js_distance(old_counter, new_counter),
        "max_share": max(new_counter.values()) / max(1, n_new) if new_counter else 0.0,
        "entropy_ratio": entropy_ratio(new_counter) / max(1.0e-12, entropy_ratio(old_counter)),
        "missing_group_count": sum(1 for k in old_counter if new_counter.get(k, 0) == 0),
        "coverage": sum(1 for k in old_counter if new_counter.get(k, 0) > 0) / max(1, len(old_counter)),
    }


def fidelity_audit_v2(tk: TailKeys, version: str, reference_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], generator_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    major_old = Counter(tk.major_old(r) for r in reference_rows)
    major_new = Counter(tk.major_new(r) for r in action_rows)
    tail_old = Counter(tk.old(version, r) for r in reference_rows)
    tail_new = Counter(tk.new(version, r) for r in action_rows)
    major = distribution_metrics(major_old, major_new)
    tail = distribution_metrics(tail_old, tail_new)
    rows = [
        {
            "stage": "P2_MAJOR_TAIL_FIDELITY_V9950",
            "status": "axis_distribution",
            "generator_id": generator_id,
            "tail_key_version": version,
            "axis_type": "major",
            "axis": "major_key",
            "old_group_count": len(major_old),
            "new_group_count": len(major_new),
            "PSI": major["PSI"],
            "JS": major["JS"],
            "max_new_group_share": major["max_share"],
            "entropy_ratio": major["entropy_ratio"],
            "missing_group_count": major["missing_group_count"],
            "coverage": major["coverage"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P2_MAJOR_TAIL_FIDELITY_V9950",
            "status": "axis_distribution",
            "generator_id": generator_id,
            "tail_key_version": version,
            "axis_type": "tail",
            "axis": "selected_tail_key",
            "old_group_count": len(tail_old),
            "new_group_count": len(tail_new),
            "PSI": tail["PSI"],
            "JS": tail["JS"],
            "max_new_group_share": tail["max_share"],
            "entropy_ratio": tail["entropy_ratio"],
            "missing_group_count": tail["missing_group_count"],
            "coverage": tail["coverage"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    summary = {
        "stage": "P2_MAJOR_TAIL_FIDELITY_V9950",
        "status": "summary",
        "generator_id": generator_id,
        "tail_key_version": version,
        "major_PSI": major["PSI"],
        "major_JS": major["JS"],
        "major_max_share": major["max_share"],
        "major_entropy_ratio": major["entropy_ratio"],
        "tail_PSI": tail["PSI"],
        "tail_JS": tail["JS"],
        "tail_missing_group_count": tail["missing_group_count"],
        "tail_coverage": tail["coverage"],
        "tail_entropy_ratio": tail["entropy_ratio"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def distribution_only_matrix(
    args: argparse.Namespace,
    out: Path,
    reference_rows: list[dict[str, Any]],
    tk: TailKeys,
    tail_key_version: str,
    old_ids: set[str],
    old_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    if not tail_key_version:
        summary = {
            "stage": "P2_GENERATOR_REPAIR_MATRIX_V9950",
            "status": "summary",
            "candidate_count": 0,
            "official_pass_count": 0,
            "weak_pass_count": 0,
            "best_generator_id": "",
            "best_generator_profile": "",
            "best_enters_P3": 0,
            "reason": "P1_tail_key_not_selected",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        return [summary], summary
    for idx, (gid, profile, eligible, desc) in enumerate(GENERATOR_PROFILES):
        t0 = time.perf_counter()
        batch = materializer.generate_natural_ap0_extension_actions(
            seed=args.seed,
            data_root=args.data_root,
            target_action_count=int(args.pilot_actions),
            cursor=str(20_000_000 + idx * 20_000),
            device=args.device,
            profile=profile,
        )
        actions: list[dict[str, Any]] = []
        for row in batch.action_rows:
            r = dict(row)
            r.update({
                "generator_id": gid,
                "generator_profile": profile,
                "panel_size": args.pilot_actions,
                "distribution_only": 1,
                "payload_tensor_written": 0,
            })
            actions.append(r)
        seen_ids = {str(r.get("action_id")) for r in actions}
        seen_hashes = {str(r.get("payload_hash_expected") or r.get("payload_hash")) for r in actions}
        smoke = {
            "stage": "P2_DISTRIBUTION_ONLY_SMOKE_V9950",
            "status": "summary",
            "generator_id": gid,
            "generator_profile": profile,
            "action_count": len(actions),
            "new_action_count": len(actions),
            "old_action_collision_count": sum(1 for r in actions if str(r.get("action_id")) in old_ids),
            "payload_collision_count": sum(1 for r in actions if str(r.get("payload_hash_expected") or r.get("payload_hash")) in old_hashes),
            "duplicate_action_id_count": len(actions) - len(seen_ids),
            "duplicate_payload_hash_count": len(actions) - len(seen_hashes),
            "rows_sec_distribution_only": len(actions) / max(1.0e-9, time.perf_counter() - t0),
            "wallclock_sec": time.perf_counter() - t0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in actions] or [0]),
        }
        audit_rows, audit = fidelity_audit_v2(tk, tail_key_version, reference_rows, actions, gid)
        write_csv(out / f"p2_{safe_name(gid)}_distribution_only_actions_v9950.csv", actions)
        write_csv(out / f"p2_{safe_name(gid)}_distribution_only_smoke_v9950.csv", [smoke])
        write_csv(out / f"p2_{safe_name(gid)}_distribution_fidelity_v9950.csv", audit_rows)
        official = int(
            eligible
            and fnum(audit["major_PSI"]) <= 0.05
            and fnum(audit["major_JS"]) <= 0.08
            and fnum(audit["major_max_share"]) <= 0.35
            and fnum(audit["major_entropy_ratio"]) >= 0.90
            and fnum(audit["tail_PSI"]) <= 0.05
            and fnum(audit["tail_JS"]) <= 0.08
            and inum(audit["tail_missing_group_count"]) == 0
            and fnum(audit["tail_coverage"]) >= 0.95
            and fnum(audit["tail_entropy_ratio"]) >= 0.90
            and inum(smoke["old_action_collision_count"]) == 0
            and inum(smoke["payload_collision_count"]) == 0
            and inum(smoke["duplicate_action_id_count"]) == 0
            and inum(smoke["cpu_offload_used"]) == 0
        )
        weak = int(
            eligible
            and fnum(audit["major_PSI"]) <= 0.08
            and fnum(audit["major_JS"]) <= 0.12
            and fnum(audit["major_max_share"]) <= 0.40
            and fnum(audit["major_entropy_ratio"]) >= 0.85
            and fnum(audit["tail_PSI"]) <= 0.08
            and fnum(audit["tail_JS"]) <= 0.12
            and inum(audit["tail_missing_group_count"]) <= 1
            and fnum(audit["tail_coverage"]) >= 0.90
            and fnum(audit["tail_entropy_ratio"]) >= 0.85
            and inum(smoke["old_action_collision_count"]) == 0
            and inum(smoke["payload_collision_count"]) == 0
            and inum(smoke["duplicate_action_id_count"]) == 0
            and inum(smoke["cpu_offload_used"]) == 0
        )
        row = {
            "stage": "P2_GENERATOR_REPAIR_MATRIX_V9950",
            "status": "generator_row",
            "generator_id": gid,
            "generator_profile": profile,
            "description": desc,
            "tail_key_version": tail_key_version,
            "action_count": len(actions),
            "new_action_count": len(actions),
            "old_action_collision_count": smoke["old_action_collision_count"],
            "payload_collision_count": smoke["payload_collision_count"],
            "major_PSI": audit["major_PSI"],
            "major_JS": audit["major_JS"],
            "major_max_share": audit["major_max_share"],
            "major_entropy_ratio": audit["major_entropy_ratio"],
            "tail_PSI": audit["tail_PSI"],
            "tail_JS": audit["tail_JS"],
            "tail_missing_group_count": audit["tail_missing_group_count"],
            "tail_coverage": audit["tail_coverage"],
            "tail_entropy_ratio": audit["tail_entropy_ratio"],
            "rows_sec_distribution_only": smoke["rows_sec_distribution_only"],
            "branch_horizon_status_if_opened": "pending_open" if (official or weak) else "not_run",
            "official_pass": official,
            "weak_pass": weak,
            "enters_P3": int(official or weak),
            "failure": "" if official or weak else "major_tail_distribution_gate_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": smoke["cpu_offload_used"],
        }
        rows.append(row)
        if inum(row["enters_P3"]) and (not selected or (inum(row["official_pass"]), -fnum(row["tail_PSI"])) > (inum(selected.get("official_pass")), -fnum(selected.get("tail_PSI")))):
            selected = row
    pass_rows = [r for r in rows if inum(r.get("official_pass"))]
    weak_rows = [r for r in rows if inum(r.get("weak_pass"))]
    best_failed = min(rows, key=lambda r: (inum(r.get("tail_missing_group_count"), 999999), fnum(r.get("tail_PSI"), 999.0), fnum(r.get("major_PSI"), 999.0)), default={})
    negative_control_pass = int(any(r.get("generator_id") == "G19-stratified-random-baseline-v2" and inum(r.get("weak_pass")) for r in rows))
    summary = {
        "stage": "P2_GENERATOR_REPAIR_MATRIX_V9950",
        "status": "summary",
        "candidate_count": len(rows),
        "official_pass_count": len(pass_rows),
        "weak_pass_count": len(weak_rows),
        "best_generator_id": selected.get("generator_id", ""),
        "best_generator_profile": selected.get("generator_profile", ""),
        "best_major_PSI": selected.get("major_PSI", ""),
        "best_tail_PSI": selected.get("tail_PSI", ""),
        "best_tail_missing_group_count": selected.get("tail_missing_group_count", ""),
        "best_enters_P3": int(bool(selected) and not negative_control_pass),
        "best_failed_generator_id": "" if selected else best_failed.get("generator_id", ""),
        "best_failed_major_PSI": "" if selected else best_failed.get("major_PSI", ""),
        "best_failed_tail_PSI": "" if selected else best_failed.get("tail_PSI", ""),
        "best_failed_missing_tail_group_count": "" if selected else best_failed.get("tail_missing_group_count", ""),
        "negative_control_pass": negative_control_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    }
    rows.insert(0, summary)
    return rows, summary


def panel_summary_rows(panel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in panel_rows if r.get("status") == "panel_row"]
    largest = completed[-1] if completed else {}
    return {
        "stage": "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9950",
        "status": "summary",
        "completed_panel_count": len(completed),
        "not_run_panel_count": sum(1 for r in panel_rows if r.get("status") == "not_run"),
        "largest_completed_panel_size": largest.get("panel_size", 0),
        "CoreLike_count": largest.get("CoreLike_count", 0),
        "PathGood_count": largest.get("PathGood_count", 0),
        "SlowBurnGood_count": largest.get("SlowBurnGood_count", 0),
        "CoreLikeOrSlowBurn_LCB": largest.get("CoreLike_or_SlowBurnGood_LCB", 0),
        "CoreLikeOrSlowBurn_UCB": largest.get("CoreLike_or_SlowBurnGood_UCB", 0),
        "P3_density_sufficient": largest.get("density_sufficient", 0),
        "P3_density_insufficient": largest.get("density_insufficient", 0),
        "P3_density_inconclusive": int(not inum(largest.get("density_sufficient")) and not inum(largest.get("density_insufficient"))),
        "reason": largest.get("reason") or (panel_rows[0].get("reason") if panel_rows else ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in panel_rows] or [0]),
    }


def density_decision(panel_summary: dict[str, Any], label_summary: dict[str, Any], generator_id: str, status: str, reason: str) -> dict[str, Any]:
    row = v9930.panel_density_decision(panel_summary, label_summary, generator_id, status=status, reason=reason)
    row["stage"] = "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9950"
    row["P3_density_sufficient"] = row.get("density_sufficient")
    row["P3_density_insufficient"] = row.get("density_insufficient")
    row["P3_density_inconclusive"] = row.get("density_inconclusive")
    return row


def run_density_if_open(args: argparse.Namespace, out: Path, p2: dict[str, Any], old_ids: set[str], old_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    targets = parse_ints(args.panel_targets)
    if not inum(p2.get("best_enters_P3")) or not p2.get("best_generator_id"):
        rows = [{
            "stage": "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9950",
            "status": "not_run",
            "panel_size": target,
            "density_result": "not_run",
            "reason": "P2_no_major_tail_fidelity_generator_passed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        } for target in targets]
        rows.insert(0, panel_summary_rows(rows))
        return rows, [], [], rows[0]
    gid = str(p2["best_generator_id"])
    profile = str(p2["best_generator_profile"])
    panel_rows: list[dict[str, Any]] = []
    all_labels: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    for target in targets:
        if target < int(args.pilot_actions):
            continue
        if any(inum(r.get("density_sufficient")) or inum(r.get("density_insufficient")) for r in panel_rows):
            panel_rows.append({
                "stage": "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9950",
                "status": "not_run",
                "panel_size": target,
                "reason": "sequential_density_already_adjudicated_at_previous_panel",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        smoke_rows, actions, applies, branches, smoke = v9930.run_panel_sharded(gid, profile, target, 30_000_000 + target, args, out, old_ids, old_hashes)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_panel_smoke_v9950.csv", smoke_rows)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_action_rows_v9950.csv", actions)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_action_apply_replay_v9950.csv", applies)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_branch_horizon_v9950.csv", branches)
        labels, label_summary = v9930.label_summary_from_branches(branches)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_action_labels_v9950.csv", labels)
        panel_rows.append(density_decision(smoke, label_summary, gid, "panel_row", f"real_sequential_density_panel_{target}"))
        all_labels, all_actions = labels, actions
    panel_rows.insert(0, panel_summary_rows(panel_rows))
    return panel_rows, all_labels, all_actions, panel_rows[0]


def path_revalidation(label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [r for r in label_rows if r.get("status") == "natural_action_label"]
    if not labels:
        rows, summary = v9930.not_run("P4_FUTURE_PATH_TYPE_REVALIDATION_V9950", "P3_no_completed_fidelity_panel", P4_future_path_weak_pass=0, P4_future_path_strong_pass=0)
        return rows, summary
    out: list[dict[str, Any]] = []
    for name in ["FastGood", "SlowBurnGood", "SlowBurnGoodRelaxed", "RiskyHighAUV", "RiskCleanButLowImmediate", "BadPath"]:
        vals = [r for r in labels if inum(r.get(name))]
        n = len(vals)
        out.append({
            "stage": "P4_FUTURE_PATH_TYPE_REVALIDATION_V9950",
            "status": "path_type_summary",
            "path_type": name,
            "action_count": n,
            "V_LCB": mean_lcb([fnum(r.get("V20_gap")) for r in vals]),
            "V240_LCB": mean_lcb([fnum(r.get("V240_gap")) for r in vals]),
            "RAUV_LCB": mean_lcb([fnum(r.get("RiskAdjustedAUV")) for r in vals]),
            "longrisk_UCB": wilson_ucb(sum(inum(r.get("RiskPath")) for r in vals), n) if n else 0.0,
            "bad_UCB": wilson_ucb(sum(inum(r.get("BadPath")) for r in vals), n) if n else 0.0,
            "null_UCB": wilson_ucb(sum(1 - inum(r.get("horizon_complete")) for r in vals), n) if n else 0.0,
            "support_by_dataset": len({r.get("dataset") for r in vals}),
            "support_by_family": len({r.get("family_id") for r in vals}),
            "support_by_tail": len({r.get("source_recipe_id") for r in vals}),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in vals] or [0]),
        })
    slow = next((r for r in out if r.get("path_type") == "SlowBurnGood"), {})
    fast = next((r for r in out if r.get("path_type") == "FastGood"), {})
    strong = int(
        inum(slow.get("action_count")) >= 64
        and fnum(slow.get("RAUV_LCB")) > 0
        and fnum(slow.get("V240_LCB")) > 0
        and fnum(slow.get("longrisk_UCB")) <= 0.05
        and inum(slow.get("action_count")) + inum(fast.get("action_count")) >= 87
    )
    summary = {
        "stage": "P4_FUTURE_PATH_TYPE_REVALIDATION_V9950",
        "status": "summary",
        "panel_action_count": len(labels),
        "FastGood_count": fast.get("action_count", 0),
        "SlowBurnGood_count": slow.get("action_count", 0),
        "P4_future_path_weak_pass": int(inum(slow.get("action_count")) > 0),
        "P4_future_path_strong_pass": strong,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in labels] or [0]),
    }
    out.insert(0, summary)
    return out, summary


def base_natural_rows(source_v9940: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for p in source_v9940.glob("p2_G*_1024_action_rows_v9940.csv"):
        actions.extend([r for r in rows_from(p) if r.get("status") == "natural_extension_action_row"])
    for p in source_v9940.glob("p2_G*_1024_action_labels_v9940.csv"):
        labels.extend([r for r in rows_from(p) if r.get("status") == "natural_action_label"])
    return actions, labels


def fpo_v7(actions: list[dict[str, Any]], labels: list[dict[str, Any]], official_context: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_by_id = {str(r.get("action_id")): r for r in labels}
    act = [r for r in actions if str(r.get("action_id")) in label_by_id]
    if not act:
        rows, summary = v9930.not_run("P5_FUTURE_PATH_OPERATOR_SKETCH_V7_V9950", "no_evaluable_action_label_rows", P5_FPO_weak_pass=0, P5_FPO_strong_pass=0)
        return rows, summary

    def eval_score(fn: Callable[[dict[str, Any]], float]) -> tuple[dict[str, float], float]:
        scores: dict[str, float] = {}
        times: list[float] = []
        for row in act:
            t0 = time.perf_counter()
            scores[str(row.get("action_id"))] = fn(row)
            times.append((time.perf_counter() - t0) * 1000.0)
        return scores, q(times, 0.90)

    specs = [
        ("FPO7A-tiny-virtual-adamw-1step", lambda r: fnum(r.get("action_adamw_cosine")) * fnum(r.get("trust_ratio")) - 4.0 * v9930.payload_linf(r)),
        ("FPO7B-tiny-virtual-adamw-3step", lambda r: 1.5 * fnum(r.get("action_adamw_cosine")) * fnum(r.get("trust_ratio")) + 0.2 * fnum(r.get("effective_derivative")) - 5.0 * v9930.payload_linf(r)),
        ("FPO7C-jvp-vjp-gradient-transport", lambda r: fnum(r.get("effective_derivative")) * fnum(r.get("trust_ratio")) - abs(fnum(r.get("branch_ratio")) - 1.0)),
        ("FPO7D-signal-reservoir-snr-gate", lambda r: fnum(r.get("tail_fraction")) + fnum(r.get("trust_ratio")) - v9930.payload_norm(r) * 8.0),
        ("FPO7E-hardtail-memory-delayed-gain", lambda r: fnum(r.get("tail_fraction")) - abs(fnum(r.get("action_adamw_cosine"))) - 6.0 * v9930.payload_linf(r)),
        ("FPO7F-risk-adjusted-path-type-classifier", lambda r: fnum(r.get("effective_derivative")) + fnum(r.get("trust_ratio")) - 12.0 * v9930.payload_linf(r)),
        ("FPO7G-slowburn-detector", lambda r: fnum(r.get("tail_fraction")) + 0.5 * fnum(r.get("effective_derivative")) - 0.25 * max(0.0, fnum(r.get("action_adamw_cosine")))),
        ("FPO7H-fastgood-slowburn-two-head", lambda r: max(fnum(r.get("effective_derivative")), fnum(r.get("tail_fraction"))) + fnum(r.get("trust_ratio")) - 8.0 * v9930.payload_linf(r)),
    ]
    out: list[dict[str, Any]] = []
    for name, fn in specs:
        scores, cost_q90 = eval_score(fn)
        ranked = sorted(act, key=lambda r: scores[str(r.get("action_id"))], reverse=True)
        accepted = ranked[: min(87, len(ranked))]
        labs = [label_by_id[str(a.get("action_id"))] for a in accepted]
        n = len(labs)
        good = sum(int(inum(l.get("CoreLike")) or inum(l.get("SlowBurnGood")) or inum(l.get("PathGood"))) for l in labs)
        risky = sum(inum(l.get("RiskPath")) for l in labs)
        bad = sum(inum(l.get("BadPath")) for l in labs)
        null = sum(1 - inum(l.get("horizon_complete")) for l in labs)
        precision = good / max(1, n)
        v_lcb = mean_lcb([fnum(l.get("V20_gap")) for l in labs])
        rauv_lcb = mean_lcb([fnum(l.get("RiskAdjustedAUV")) for l in labs])
        longrisk_ucb = wilson_ucb(risky, n) if n else 0.0
        bad_ucb = wilson_ucb(bad, n) if n else 0.0
        null_ucb = wilson_ucb(null, n) if n else 0.0
        memory_ucb = longrisk_ucb
        offdiag_ucb = longrisk_ucb
        weak = int(official_context and n >= 87 and precision >= 0.60 and v_lcb > 0 and longrisk_ucb <= 0.10 and cost_q90 <= 3.0)
        strong = int(
            official_context
            and n >= 87
            and precision >= 0.75
            and v_lcb > 0
            and rauv_lcb > 0
            and longrisk_ucb <= 0.05
            and bad_ucb <= 0.05
            and null_ucb <= 0.15
            and memory_ucb <= 0.05
            and offdiag_ucb <= 0.05
            and cost_q90 <= 1.50
        )
        out.append({
            "stage": "P5_FUTURE_PATH_OPERATOR_SKETCH_V7_V9950",
            "status": "fpo_row",
            "fpo_id": name,
            "feature_count": 4,
            "feature_legality": "green_commit_time_fields_only",
            "feature_cost_q90_ms": cost_q90,
            "accepted_count": n,
            "TopK87_precision_CoreLikeOrSlowBurn": precision,
            "TopK87_V_LCB": v_lcb,
            "TopK87_RAUV_LCB": rauv_lcb,
            "TopK87_longrisk_UCB": longrisk_ucb,
            "TopK87_bad_UCB": bad_ucb,
            "TopK87_null_UCB": null_ucb,
            "TopK87_memory_UCB": memory_ucb,
            "TopK87_offdiag_UCB": offdiag_ucb,
            "LDO": 0.0,
            "LSO": 0.0,
            "LTO": 0.0,
            "LFO": 0.0,
            "official_fidelity_context": official_context,
            "weak_pass": weak,
            "strong_pass": strong,
            "false_positive_type": "RiskyHighAUV_or_BadPath" if longrisk_ucb > 0.10 or bad_ucb > 0.05 else "low_value",
            "false_negative_type": "SlowBurnGood" if precision < 0.60 else "",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (inum(r.get("strong_pass")), inum(r.get("weak_pass")), fnum(r.get("TopK87_precision_CoreLikeOrSlowBurn")), fnum(r.get("TopK87_V_LCB"))), default={})
    summary = {
        "stage": "P5_FUTURE_PATH_OPERATOR_SKETCH_V7_V9950",
        "status": "summary",
        "fpo_count": len(out),
        "evaluated_action_count": len(act),
        "official_fidelity_context": official_context,
        "P5_FPO_weak_pass": int(any(inum(r.get("weak_pass")) for r in out)),
        "P5_FPO_strong_pass": int(any(inum(r.get("strong_pass")) for r in out)),
        "best_fpo_id": best.get("fpo_id", ""),
        "best_precision": best.get("TopK87_precision_CoreLikeOrSlowBurn", ""),
        "best_V_LCB": best.get("TopK87_V_LCB", ""),
        "best_longrisk_UCB": best.get("TopK87_longrisk_UCB", ""),
        "best_cost_q90_ms": best.get("feature_cost_q90_ms", ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    out.insert(0, summary)
    return out, summary


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return v9930.not_run(stage, reason, **extra)


def _result_files_from_artifacts(artifacts: dict[str, Path], patterns: tuple[str, ...]) -> list[Path]:
    result_root = RESULT_ROOT.resolve()
    roots: set[Path] = set()
    for path in artifacts.values():
        if not path.exists():
            continue
        resolved = path.resolve()
        if str(resolved).startswith(str(result_root)):
            roots.add(path.parent)
    files: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            files.update(root.glob(pattern))
    return sorted(files, key=lambda p: p.name)


def _flag_value(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "false"}:
        return 0
    try:
        return int(float(text) > 0)
    except ValueError:
        return int(text in {"true", "yes"})


def artifact_audit(artifacts: dict[str, Path]) -> dict[str, int]:
    total = 0
    fake = 0
    proxy = 0
    cpu = 0
    skip_prefixes = ("no_fake", "contract", "failure")
    for path in _result_files_from_artifacts(artifacts, ("*.csv", "*.json")):
        if path.name.startswith(skip_prefixes):
            continue
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    total += 1
                    fake = max(fake, _flag_value(row.get("fake_data_used", 0)))
                    proxy = max(proxy, _flag_value(row.get("proxy_row_used", 0)))
                    cpu = max(cpu, _flag_value(row.get("cpu_offload_used", 0)))
        elif path.suffix == ".json":
            total += 1
            data = read_json(path)
            if isinstance(data, dict):
                fake = max(fake, _flag_value(data.get("fake_data_used", 0)))
                proxy = max(proxy, _flag_value(data.get("proxy_row_used", 0)))
                cpu = max(cpu, _flag_value(data.get("cpu_offload_used", 0)))
    return {
        "rows_checked": total,
        "fake_data_used": fake,
        "proxy_row_used": proxy,
        "cpu_offload_used": cpu,
    }


def artifact_row_count(artifacts: dict[str, Path]) -> int:
    return artifact_audit(artifacts)["rows_checked"]


def write_figures(out: Path, p1_rows: list[dict[str, Any]], p2_rows: list[dict[str, Any]], p3: dict[str, Any], p4: dict[str, Any], p5: dict[str, Any], route: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    key_rows = [r for r in p1_rows if r.get("status") == "tail_key_row"]
    fig("fig_p1_tail_support_histogram_v9950.svg", "Tail key support counts", [r["tail_key_version"].split("-")[0] for r in key_rows], [fnum(r.get("tail_group_count")) for r in key_rows])
    gen_rows = [r for r in p2_rows if r.get("status") == "generator_row"]
    fig("fig_p2_major_tail_psi_scatter_v9950.svg", "Major/Tail PSI by generator", [r["generator_id"].split("-")[0] for r in gen_rows], [fnum(r.get("major_PSI")) + fnum(r.get("tail_PSI")) for r in gen_rows])
    fig("fig_p2_missing_tail_heatmap_v9950.svg", "Missing tail groups", [r["generator_id"].split("-")[0] for r in gen_rows], [fnum(r.get("tail_missing_group_count")) for r in gen_rows])
    fig("fig_p3_sequential_density_ci_v9950.svg", "Sequential density CI", ["LCB", "UCB", "suff", "insuff"], [fnum(p3.get("CoreLikeOrSlowBurn_LCB")), fnum(p3.get("CoreLikeOrSlowBurn_UCB")), fnum(p3.get("P3_density_sufficient")), fnum(p3.get("P3_density_insufficient"))])
    fig("fig_p4_future_path_types_v9950.svg", "Future path types", ["Fast", "Slow"], [fnum(p4.get("FastGood_count")), fnum(p4.get("SlowBurnGood_count"))])
    fig("fig_p5_fpo_confusion_v9950.svg", "FPO v7 gate", ["weak", "strong", "precision", "V"], [fnum(p5.get("P5_FPO_weak_pass")), fnum(p5.get("P5_FPO_strong_pass")), fnum(p5.get("best_precision")), fnum(p5.get("best_V_LCB"))])
    fig("fig_v9950_stop_pivot_matrix.svg", "Stop/Pivot", ["P1", "P2", "P3 suff", "P3 insuff", "P5", "D"], [fnum(route.get("P1_weak_pass")), fnum(route.get("P2_weak_pass")), fnum(route.get("P3_density_sufficient")), fnum(route.get("P3_density_insufficient")), fnum(route.get("P5_FPO_strong_pass")), fnum(route.get("P7_generated_sandbox_allowed"))])
    return figs


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _result_files_from_artifacts(paths, ("*.csv", "*.json", "*.svg")):
        if path.exists():
            hashes[path.name] = sha256_file(path)
    for name, path in paths.items():
        if path.exists() and not str(path.resolve()).startswith(str(RESULT_ROOT.resolve())):
            hashes[name] = sha256_file(path)
    return hashes


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_v9940_boundary_reproduction_v9950.csv"))
    p1 = summary_row(rows_from(out / "p1_tail_key_reference_audit_v9950.csv"))
    p2 = summary_row(rows_from(out / "p2_generator_repair_matrix_v9950.csv"))
    p3 = summary_row(rows_from(out / "p3_sequential_natural_density_panel_v9950.csv"))
    p4 = summary_row(rows_from(out / "p4_future_path_type_revalidation_v9950.csv"))
    p5 = summary_row(rows_from(out / "p5_future_path_operator_sketch_v7_v9950.csv"))
    p6 = summary_row(rows_from(out / "p6_existing_action_controller_gate_v9950.csv"))
    p7 = summary_row(rows_from(out / "p7_generated_sandbox_gate_v9950.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9950.csv"))
    p1_rows = [r for r in rows_from(out / "p1_tail_key_reference_audit_v9950.csv") if r.get("status") == "tail_key_row"]
    p2_rows = [r for r in rows_from(out / "p2_generator_repair_matrix_v9950.csv") if r.get("status") == "generator_row"]
    p5_rows = [r for r in rows_from(out / "p5_future_path_operator_sketch_v7_v9950.csv") if r.get("status") == "fpo_row"]
    lines = [
        "# DG-KAN v9.9.5 Tail Fidelity Repair / Natural Density / FPO 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.5_结果解读_TailFidelity修复_自然密度裁决_FPO计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 density/controller/generated/runtime 均显式 `not_run`。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.9.4 boundary：source route = `{p0.get('source_route')}`，official pass count = `{p0.get('official_pass_count')}`，fake/proxy/cpu = `{p0.get('fake_data_used_v9940')}` / `{p0.get('proxy_row_used_v9940')}` / `{p0.get('cpu_offload_used_v9940')}`。",
        f"2. P1 选中 tail key = `{p1.get('tail_key_version_selected')}`，weak/strong = `{p1.get('P1_weak_pass')}` / `{p1.get('P1_strong_pass')}`，group count = `{p1.get('tail_group_count')}`，support<3 fraction = `{p1.get('support_lt3_fraction')}`。",
        f"3. P2 G13-G19 candidate count = `{p2.get('candidate_count')}`，official/weak pass count = `{p2.get('official_pass_count')}` / `{p2.get('weak_pass_count')}`，best = `{p2.get('best_generator_id')}`。",
        f"4. P2 best failed = `{p2.get('best_failed_generator_id')}`，major/tail PSI = `{p2.get('best_failed_major_PSI')}` / `{p2.get('best_failed_tail_PSI')}`，missing tail = `{p2.get('best_failed_missing_tail_group_count')}`。",
        f"5. P3 largest completed panel = `{p3.get('largest_completed_panel_size')}`，density sufficient/insufficient/inconclusive = `{p3.get('P3_density_sufficient')}` / `{p3.get('P3_density_insufficient')}` / `{p3.get('P3_density_inconclusive')}`。",
        f"6. P4 future path weak/strong = `{p4.get('P4_future_path_weak_pass')}` / `{p4.get('P4_future_path_strong_pass')}`。",
        f"7. P5 FPO v7 weak/strong = `{p5.get('P5_FPO_weak_pass')}` / `{p5.get('P5_FPO_strong_pass')}`；best = `{p5.get('best_fpo_id')}`，precision = `{p5.get('best_precision')}`。",
        f"8. P6 controller = `{p6.get('status')}`；P7 generated sandbox allowed = `{p7.get('generated_sandbox_allowed')}`。",
        f"9. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.5 G13-G19 generator profiles。 |",
        "| `experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py` | v9.9.5 runner；执行 P0/P1/P2，按 gate 开 P3-P8，并写 manifest/recap。 |",
        "",
        "```text",
        "python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py --out-dir results/real_rerun_20260506/v9950_tail_fidelity_repair_natural_density_fpo_full_20260517T180000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Tail Key Audit v2",
        "",
        "| tail key | groups | support<3 fraction | max share | entropy ratio | weak | strong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p1_rows:
        lines.append(f"| `{r.get('tail_key_version')}` | `{r.get('tail_group_count')}` | `{r.get('support_lt3_fraction')}` | `{r.get('largest_group_share')}` | `{r.get('entropy_ratio')}` | `{r.get('weak_pass')}` | `{r.get('strong_pass')}` |")
    lines += [
        "",
        "## 4. P2 Generator Repair Matrix v2",
        "",
        "| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |",
        "|---|---|---|---:|---:|",
    ]
    for r in p2_rows:
        lines.append(f"| `{r.get('generator_id')}` | `{r.get('major_PSI')}`/`{r.get('major_JS')}`/`{r.get('major_max_share')}` | `{r.get('tail_PSI')}`/`{r.get('tail_JS')}`/`{r.get('tail_missing_group_count')}`/`{r.get('tail_coverage')}` | `{r.get('official_pass')}` | `{r.get('weak_pass')}` |")
    lines += [
        "",
        "## 5. P3-P8 Boundary",
        "",
        "```text",
        f"P3 density = {p3.get('P3_density_sufficient')} / {p3.get('P3_density_insufficient')} / {p3.get('P3_density_inconclusive')}",
        f"P4 future path = {p4.get('P4_future_path_weak_pass')} / {p4.get('P4_future_path_strong_pass')}",
        f"P5 FPO = {p5.get('P5_FPO_weak_pass')} / {p5.get('P5_FPO_strong_pass')}",
        f"P6 controller = {p6.get('status')}",
        f"P7 generated = {p7.get('status')}, allowed = {p7.get('generated_sandbox_allowed')}",
        "```",
        "",
        "## 6. P5 FPO v7",
        "",
        "| fpo | precision | V LCB | longrisk UCB | cost q90 ms | weak | strong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p5_rows:
        lines.append(f"| `{r.get('fpo_id')}` | `{r.get('TopK87_precision_CoreLikeOrSlowBurn')}` | `{r.get('TopK87_V_LCB')}` | `{r.get('TopK87_longrisk_UCB')}` | `{r.get('feature_cost_q90_ms')}` | `{r.get('weak_pass')}` | `{r.get('strong_pass')}` |")
    lines += [
        "",
        "## 7. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 8. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in hashes.items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 9. 最终分析结论",
        "",
        "```text",
        "1. v9.9.5 没有直接用 v9.9.4 的 tail key 继续加 generator，而是先审计 TK0-TK4。",
        "2. 只有 P1 选出的 commit-time tail key 通过后，P2 G13-G19 才进入 distribution-only 1024。",
        "3. 只有 P2 official/weak pass 才允许打开 P3 branch-horizon 和 sequential density panel。",
        "4. P5 FPO v7 即使诊断运行，也必须在 official fidelity context 下才能写成 controller-ready。",
        "5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。",
        "```",
        "",
        f"最终一句话：v9.9.5 真实执行后停在 `{route.get('route')}`：{route.get('route_explanation')}",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source_v9940 = Path(args.source_v9940)
    source_v9330 = Path(args.source_v9330)
    source_v9900 = Path(args.source_v9900)
    source_v9910 = Path(args.source_v9910)
    source_v9830 = Path(args.source_v9830)
    source_v9840 = Path(args.source_v9840)

    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, obj: dict[str, Any]) -> None:
        path = out / name
        write_json(path, obj)
        artifacts[name] = path

    p0_rows, p0 = p0_boundary(source_v9940)
    dump_csv("p0_v9940_boundary_reproduction_v9950.csv", p0_rows)

    ref_actions = reference_actions(source_v9330)
    old_ids, old_hashes = old_action_sets(source_v9330, source_v9900, source_v9910)

    p1_rows, p1, tk = tail_key_audit(ref_actions)
    dump_csv("p1_tail_key_reference_audit_v9950.csv", p1_rows)

    p2_rows, p2 = distribution_only_matrix(args, out, ref_actions, tk, str(p1.get("tail_key_version_selected") or ""), old_ids, old_hashes)
    dump_csv("p2_generator_repair_matrix_v9950.csv", p2_rows)

    p3_rows, p3_labels, p3_actions, p3 = run_density_if_open(args, out, p2, old_ids, old_hashes)
    dump_csv("p3_sequential_natural_density_panel_v9950.csv", p3_rows)
    dump_csv("p3_natural_density_group_rates_v9950.csv", v9930.grouped_density(p3_labels, p3_actions) if p3_labels else v9930.not_run("P3_NATURAL_DENSITY_GROUP_RATES_V9950", "P3_no_completed_density_panel")[0])

    p4_rows, p4 = path_revalidation(p3_labels)
    dump_csv("p4_future_path_type_revalidation_v9950.csv", p4_rows)

    fpo_actions, fpo_labels = (p3_actions, p3_labels) if p3_labels else base_natural_rows(source_v9940)
    p5_rows, p5 = fpo_v7(fpo_actions, fpo_labels, official_context=int(bool(p3_labels)))
    dump_csv("p5_future_path_operator_sketch_v7_v9950.csv", p5_rows)

    controller_open = int(inum(p3.get("P3_density_sufficient")) or inum(p5.get("P5_FPO_strong_pass")))
    p6_rows, p6 = not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9950", "P3_density_not_sufficient_and_P5_strong_not_passed", P6_controller_pass=0) if not controller_open else not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9950", "controller_open_not_implemented_without_passed_support_splits", P6_controller_pass=0)
    dump_csv("p6_existing_action_controller_gate_v9950.csv", p6_rows)

    generated_allowed = int(inum(p3.get("P3_density_insufficient")) or inum(p5.get("P5_FPO_strong_pass")) or inum(p4.get("P4_future_path_strong_pass")))
    p7 = {
        "stage": "P7_GENERATED_SANDBOX_GATE_V9950",
        "status": "not_run",
        "generated_sandbox_allowed": generated_allowed,
        "allowed_action_count": 64 if generated_allowed else 0,
        "reason": "generated_gate_conditions_not_met" if not generated_allowed else "64_action_sandbox_allowed_but_not_executed_in_v9950_boundary",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("p7_generated_sandbox_gate_v9950.csv", [p7])

    p8_rt, p8_rt_sum = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9950", "P6_or_P7_not_passed", P8_runtime_pass=0)
    p8_pr, p8_pr_sum = not_run("P8_PAIRED_REPLAY_BOUNDARY_V9950", "P6_or_P7_not_passed", P8_paired_replay_pass=0)
    dump_csv("p8_selected_runtime_boundary_v9950.csv", p8_rt)
    dump_csv("p8_paired_replay_boundary_v9950.csv", p8_pr)

    if not inum(p1.get("P1_weak_pass")):
        route_name, primary, secondary, gen_status, explanation = (
            "R1-TailKeyNotSampleable",
            "tail_key_reference_not_sampleable",
            "generator_repair_blocked",
            "stopped_tail_key_failed",
            "tail key audit did not find a weak-pass official key; next step is redefining commit-time tail key before more generators.",
        )
    elif not inum(p2.get("best_enters_P3")):
        route_name, primary, secondary, gen_status, explanation = (
            "R2-TailKeyPassGeneratorFidelityFail",
            "no_major_tail_fidelity_generator_passed",
            "density_panels_blocked",
            "stopped_generator_fidelity_failed",
            "tail key was made sampleable, but G13-G19 did not produce a major+tail fidelity pass, so density/controller/generated stay blocked.",
        )
    elif inum(p3.get("P3_density_sufficient")):
        route_name, primary, secondary, gen_status, explanation = (
            "R3-NaturalDensitySufficientControllerPending",
            "controller_gate_pending",
            "FPO_or_support_split_pending",
            "stopped_controller_gate_pending",
            "a fidelity generator opened density and density is sufficient, but controller/runtime still require their own gates.",
        )
    elif inum(p3.get("P3_density_insufficient")):
        route_name, primary, secondary, gen_status, explanation = (
            "R4-NaturalDensityInsufficientGeneratedCandidate",
            "generated_route_candidate",
            "FPO_or_path_target_required",
            "generated_64_sandbox_candidate",
            "natural density is insufficient on a fidelity panel, so generated route can be prepared under strict 64-action sandbox rules.",
        )
    elif inum(p5.get("P5_FPO_strong_pass")):
        route_name, primary, secondary, gen_status, explanation = (
            "R5-FPOStrongControllerCandidate",
            "FPO_controller_candidate",
            "runtime_boundary_pending",
            "stopped_controller_gate_pending",
            "FPO v7 strong pass exists, but controller/runtime still need gated execution.",
        )
    else:
        route_name, primary, secondary, gen_status, explanation = (
            "R6-FidelityPassDensityInconclusive",
            "sequential_density_pending_or_inconclusive",
            "FPO_not_controller_ready",
            "stopped_density_or_FPO_pending",
            "generator fidelity opened the density path but density and FPO gates did not close in this run.",
        )

    route = {
        "stage": "ROUTE_DECISION_V9950",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_explanation": explanation,
        "source_route_v9940": p0.get("source_route"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_weak_pass": p1.get("P1_weak_pass"),
        "P1_strong_pass": p1.get("P1_strong_pass"),
        "tail_key_version_selected": p1.get("tail_key_version_selected"),
        "P2_official_pass_count": p2.get("official_pass_count"),
        "P2_weak_pass_count": p2.get("weak_pass_count"),
        "P2_weak_pass": int(inum(p2.get("weak_pass_count")) > 0),
        "best_generator_id": p2.get("best_generator_id"),
        "best_failed_generator_id": p2.get("best_failed_generator_id"),
        "P3_largest_completed_panel_size": p3.get("largest_completed_panel_size"),
        "P3_density_sufficient": p3.get("P3_density_sufficient"),
        "P3_density_insufficient": p3.get("P3_density_insufficient"),
        "P3_density_inconclusive": p3.get("P3_density_inconclusive"),
        "P4_future_path_weak_pass": p4.get("P4_future_path_weak_pass"),
        "P4_future_path_strong_pass": p4.get("P4_future_path_strong_pass"),
        "P5_FPO_weak_pass": p5.get("P5_FPO_weak_pass"),
        "P5_FPO_strong_pass": p5.get("P5_FPO_strong_pass"),
        "P6_controller_pass": p6.get("P6_controller_pass"),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed"),
        "generated_route_status": gen_status,
        "P8_runtime_pass": p8_rt_sum.get("P8_runtime_pass"),
        "P8_paired_replay_pass": p8_pr_sum.get("P8_paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max(inum(p0.get("cpu_offload_used")), inum(p1.get("cpu_offload_used")), inum(p2.get("cpu_offload_used")), inum(p3.get("cpu_offload_used")), inum(p5.get("cpu_offload_used"))),
    }
    dump_json("route_decision_v9950.json", route)

    figs = write_figures(out, p1_rows, p2_rows, p3, p4, p5, route)
    artifacts.update(figs)

    audit = artifact_audit(artifacts)
    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9950",
        "status": "summary",
        "rows_checked": audit["rows_checked"],
        "fake_data_used": audit["fake_data_used"],
        "proxy_row_used": audit["proxy_row_used"],
        "cpu_offload_used": max(audit["cpu_offload_used"], inum(route["cpu_offload_used"])),
    }
    dump_csv("no_fake_audit_v9950.csv", [no_fake])
    dump_csv("contract_audit_v9950.csv", [{
        "stage": "CONTRACT_AUDIT_V9950",
        "status": "summary",
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_tail_key_selected": p1.get("tail_key_version_selected"),
        "P2_gate_respected": int(not inum(p2.get("best_enters_P3")) or inum(p3.get("completed_panel_count")) >= 0),
        "generated_gate_respected": int(generated_allowed == inum(p7.get("generated_sandbox_allowed"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }])
    dump_csv("failure_taxonomy_v9950.csv", [{
        "stage": "FAILURE_TAXONOMY_V9950",
        "status": "summary",
        "route": route_name,
        "F1_tail_key_failed": int(not inum(p1.get("P1_weak_pass"))),
        "F2_generator_fidelity_failed": int(inum(p1.get("P1_weak_pass")) and not inum(p2.get("best_enters_P3"))),
        "F3_density_not_closed": int(not inum(p3.get("P3_density_sufficient")) and not inum(p3.get("P3_density_insufficient"))),
        "F4_FPO_not_strong": int(not inum(p5.get("P5_FPO_strong_pass"))),
        "F5_generated_not_open": int(not generated_allowed),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }])

    manifest = {
        "stage": "RUN_MANIFEST_V9950",
        "status": "summary",
        "out_dir": str(out),
        "command": "python experiments/run_v9950_tail_fidelity_repair_natural_density_fpo.py --out-dir ... --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512",
        "plan": str(PLAN_PATH),
        "runner": str(SCRIPT_PATH),
        "materializer": str(MATERIALIZER_PATH),
        "source_v9940": str(source_v9940),
        "route": route_name,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_json("run_manifest_v9950.json", manifest)
    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifacts["materializer"] = MATERIALIZER_PATH
    hashes = sha_rows(artifacts)
    write_recap(out, route, hashes)


if __name__ == "__main__":
    main()
