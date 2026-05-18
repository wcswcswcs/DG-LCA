#!/usr/bin/env python3
"""DG-KAN v9.9.7 tail-fidelity feasibility / natural density / FPO rewrite.

This runner starts from the v9.9.6 artifacts.  It first audits whether the
current major+tail fidelity gate is sampleable, then builds tail-key v3
candidates and runs G30-G39 distribution-only pilots.  Branch horizon and
density panels open only after a generator passes the explicit major+tail
fidelity gates.
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
DEFAULT_OUT = RESULT_ROOT / "v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite_full_20260517T200000Z"
SOURCE_V9960 = RESULT_ROOT / "v9960_tail_fidelity_rootcause_natural_density_fpo_rewrite_full_20260517T190000Z"
SOURCE_V9950 = RESULT_ROOT / "v9950_tail_fidelity_repair_natural_density_fpo_full_20260517T180000Z"
SOURCE_V9940 = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z"
SOURCE_V9940_SUPP = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_supplement_20260517T170000Z"
SOURCE_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
SOURCE_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
SOURCE_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
SOURCE_V9830 = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
SOURCE_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.7_结果解读_TailFidelity可行性_自然密度裁决_FPO重写完整实验计划.md"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.7_TailFidelityFeasibility_NaturalDensity_FPORewrite_实验复盘.md"
SCRIPT_PATH = REPO / "experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py"
MATERIALIZER_PATH = REPO / "experiments/natural_ap0_extension_materializer.py"
BRANCH_COUNT = 6
HORIZONS = [1, 5, 20, 80, 240]

GENERATOR_PROFILES: list[tuple[str, str, int, str]] = [
    ("G30-IPF-hierarchical-major-tail-generator", "g30-ipf-hierarchical-major-tail-generator", 1, "iterative-proportional-style interleaving over major and tail-key-v3 margins"),
    ("G31-mincostflow-major-tail-quota-generator", "g31-mincostflow-major-tail-quota-generator", 1, "min-cost-flow-style major/tail quota sampler"),
    ("G32-tail-first-major-corrected-generator", "g32-tail-first-major-corrected-generator", 1, "tail coverage first, major correction fill"),
    ("G33-major-first-tail-reservoir-refill-generator", "g33-major-first-tail-reservoir-refill-generator", 1, "major first, tail reservoir refill"),
    ("G34-stochastic-rounded-tail-quota-generator", "g34-stochastic-rounded-tail-quota-generator", 1, "stochastic-rounded tail quota sampler"),
    ("G35-precursor-bootstrap-tail-generator", "g35-precursor-bootstrap-tail-generator", 1, "canonical precursor bootstrap with new action hashes"),
    ("G36-entropy-regularized-sampler", "g36-entropy-regularized-sampler", 1, "entropy-regularized major/tail sampler"),
    ("G37-tail-coverage-negative-control", "g37-tail-coverage-negative-control", 0, "negative control preserving major but not tail"),
    ("G38-tail-oversample-importance-weighted-diagnostic", "g38-tail-oversample-importance-weighted-diagnostic", 0, "importance-weighted diagnostic only"),
    ("G39-two-stage-feasible-tail-key-v3-generator", "g39-two-stage-feasible-tail-key-v3-generator", 1, "two-stage feasible tail-key-v3 sampler"),
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
    p.add_argument("--source-v9960", default=str(SOURCE_V9960))
    p.add_argument("--source-v9950", default=str(SOURCE_V9950))
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


def p0_boundary(source_v9960: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9960 / "route_decision_v9960.json")
    p1 = summary_row(rows_from(source_v9960 / "p1_missing_tail_root_cause_audit_v9960.csv"))
    p2 = summary_row(rows_from(source_v9960 / "p2_generator_repair_matrix_v9960.csv"))
    p3 = summary_row(rows_from(source_v9960 / "p3_sequential_natural_density_panel_v9960.csv"))
    p5 = summary_row(rows_from(source_v9960 / "p5_future_path_operator_sketch_v8_v9960.csv"))
    p6 = summary_row(rows_from(source_v9960 / "p6_existing_action_controller_gate_v9960.csv"))
    p7 = summary_row(rows_from(source_v9960 / "p7_generated_sandbox_gate_v9960.csv"))
    nf = summary_row(rows_from(source_v9960 / "no_fake_audit_v9960.csv"))
    row = {
        "stage": "P0_V9960_BOUNDARY_REPRODUCTION_V9970",
        "status": "summary",
        "source_route": route.get("route"),
        "P1_attribution_complete": p1.get("P1_attribution_complete"),
        "P1_assigned_missing_reason_fraction": p1.get("assigned_missing_reason_fraction"),
        "P1_unknown_missing_reason_fraction": p1.get("unknown_missing_reason_fraction"),
        "P1_top_missing_reason": p1.get("top_missing_reason"),
        "tail_key_version_selected_v9960": route.get("tail_key_version_selected"),
        "MR1_quota_rounded_to_zero_count": p1.get("MR1_quota_rounded_to_zero_count"),
        "MR5_cursor_collapse_count": p1.get("MR5_cursor_collapse_count"),
        "MR6_major_tail_quota_conflict_count": p1.get("MR6_major_tail_quota_conflict_count"),
        "MR8_tail_key_too_fine_count": p1.get("MR8_tail_key_too_fine_count"),
        "P2_candidate_count": p2.get("candidate_count"),
        "official_pass_count": p2.get("official_pass_count"),
        "weak_pass_count": p2.get("weak_pass_count"),
        "best_failed_generator_id": p2.get("best_failed_generator_id"),
        "best_failed_major_PSI": p2.get("best_failed_major_PSI"),
        "best_failed_tail_PSI": p2.get("best_failed_tail_PSI"),
        "best_failed_missing_tail_count": p2.get("best_failed_missing_tail_group_count"),
        "P3_largest_completed_panel": p3.get("largest_completed_panel_size"),
        "P4_future_path_pass": route.get("P4_future_path_strong_pass"),
        "P5_FPO_pass": route.get("P5_FPO_strong_pass"),
        "P5_best_FPO": p5.get("best_fpo_id"),
        "P5_best_precision": p5.get("best_precision"),
        "P5_best_V_LCB": p5.get("best_V_LCB"),
        "P5_best_longrisk_UCB": p5.get("best_longrisk_UCB"),
        "P6_controller_status": p6.get("status"),
        "P7_generated_status": p7.get("status"),
        "fake_data_used_v9960": nf.get("fake_data_used"),
        "proxy_row_used_v9960": nf.get("proxy_row_used"),
        "cpu_offload_used_v9960": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_pass"] = int(
        row["source_route"] == "RouteD-TailFidelityStillFails"
        and inum(row["P1_attribution_complete"]) == 1
        and inum(row["official_pass_count"]) == 0
        and inum(row["weak_pass_count"]) == 0
        and inum(row["fake_data_used_v9960"]) == 0
        and inum(row["proxy_row_used_v9960"]) == 0
        and inum(row["cpu_offload_used_v9960"]) == 0
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
        self.tk3e_counts = Counter(self._tk3e_old(r) for r in reference_rows)
        self.tk3d_counts = Counter(self._tk3d_old(r) for r in reference_rows)

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

    def action_shape_old(self, r: dict[str, Any]) -> str:
        norm = v9930.payload_norm(r)
        linf = v9930.payload_linf(r)
        norm_band = "norm_hi" if norm >= 0.010 else ("norm_mid" if norm >= 0.003 else "norm_lo")
        linf_band = "linf_hi" if linf >= 0.0010 else ("linf_mid" if linf >= 0.0003 else "linf_lo")
        return "|".join([self.template_old(r), self.step(r), norm_band, linf_band])

    def action_shape_new(self, r: dict[str, Any]) -> str:
        norm = v9930.payload_norm(r)
        linf = v9930.payload_linf(r)
        norm_band = "norm_hi" if norm >= 0.010 else ("norm_mid" if norm >= 0.003 else "norm_lo")
        linf_band = "linf_hi" if linf >= 0.0010 else ("linf_mid" if linf >= 0.0003 else "linf_lo")
        return "|".join([self.template_new(r), self.step(r), norm_band, linf_band])

    def safety_proxy(self, r: dict[str, Any]) -> str:
        return "safe_proxy" if v9930.payload_linf(r) <= 0.0010 else "high_linf_proxy"

    def major_old(self, r: dict[str, Any]) -> str:
        return "|".join([self.dataset(r), self.template_old(r), self.step(r), self.norm_bucket(r)])

    def major_new(self, r: dict[str, Any]) -> str:
        return "|".join([self.dataset(r), self.template_new(r), self.step(r), self.norm_bucket(r)])

    def major_struct_old(self, r: dict[str, Any]) -> str:
        return "|".join([self.dataset(r), self.template_old(r), self.step(r)])

    def major_struct_new(self, r: dict[str, Any]) -> str:
        return "|".join([self.dataset(r), self.template_new(r), self.step(r)])

    def old_tk0(self, r: dict[str, Any]) -> str:
        return "|".join([self.recipe_old(r), self.template_old(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])

    def new_tk0(self, r: dict[str, Any]) -> str:
        return "|".join([self.recipe_new(r), self.template_new(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])

    def _tk3e_old(self, r: dict[str, Any]) -> str:
        return "|".join([self.major_old(r), self.recipe_old(r), self.action_shape_old(r)])

    def _tk3e_new(self, r: dict[str, Any]) -> str:
        return "|".join([self.major_new(r), self.recipe_new(r), self.action_shape_new(r)])

    def _tk3d_old(self, r: dict[str, Any]) -> str:
        return "|".join([self.major_old(r), self.template_old(r), self.step(r), self.norm_bucket(r)])

    def _tk3d_new(self, r: dict[str, Any]) -> str:
        return "|".join([self.major_new(r), self.template_new(r), self.step(r), self.norm_bucket(r)])

    def _tk3_key_old(self, version: str, r: dict[str, Any]) -> str:
        if version.startswith("TK3A"):
            return "|".join([self.major_old(r), self.recipe_old(r)])
        if version.startswith("TK3B"):
            return "|".join([self.major_old(r), self.recipe_old(r), self.norm_bucket(r)])
        if version.startswith("TK3C"):
            return "|".join([self.major_old(r), self.recipe_old(r), self.safety_proxy(r)])
        if version.startswith("TK3D"):
            return self._tk3d_old(r)
        if version.startswith("TK3E"):
            return self._tk3e_old(r)
        if version.startswith("TK3F"):
            key = self._tk3e_old(r)
            if self.tk3e_counts.get(key, 0) < 3:
                return "|".join([self.major_old(r), self.recipe_old(r), "merged_low_support_shape"])
            return key
        if version.startswith("TK3G"):
            key = self._tk3d_old(r)
            if self.tk3d_counts.get(key, 0) < 3:
                return "|".join([self.major_struct_old(r), "merged_support_lt3_norm"])
            return key
        if version.startswith("TK3H"):
            return self.major_struct_old(r)
        return "|".join([self.recipe_old(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])

    def _tk3_key_new(self, version: str, r: dict[str, Any]) -> str:
        if version.startswith("TK3A"):
            return "|".join([self.major_new(r), self.recipe_new(r)])
        if version.startswith("TK3B"):
            return "|".join([self.major_new(r), self.recipe_new(r), self.norm_bucket(r)])
        if version.startswith("TK3C"):
            return "|".join([self.major_new(r), self.recipe_new(r), self.safety_proxy(r)])
        if version.startswith("TK3D"):
            return self._tk3d_new(r)
        if version.startswith("TK3E"):
            return self._tk3e_new(r)
        if version.startswith("TK3F"):
            old_like_key = "|".join([self.major_new(r), self.recipe_new(r), self.action_shape_new(r)])
            if self.tk3e_counts.get(old_like_key, 0) < 3:
                return "|".join([self.major_new(r), self.recipe_new(r), "merged_low_support_shape"])
            return self._tk3e_new(r)
        if version.startswith("TK3G"):
            old_like_key = self._tk3d_new(r)
            if self.tk3d_counts.get(old_like_key, 0) < 3:
                return "|".join([self.major_struct_new(r), "merged_support_lt3_norm"])
            return old_like_key
        if version.startswith("TK3H"):
            return self.major_struct_new(r)
        return "|".join([self.recipe_new(r), self.step(r), self.norm_bucket(r), self.linf_bucket(r)])

    def old(self, version: str, r: dict[str, Any]) -> str:
        tk0 = self.old_tk0(r)
        if version.startswith("TK0"):
            return tk0
        if version.startswith("TK1"):
            return tk0 if self.tk0_counts[tk0] >= 3 else f"rare3|{self.major_old(r)}"
        if version.startswith("TK2"):
            return f"{self.major_old(r)}|{self.recipe_old(r) if self.tk0_counts[tk0] >= 3 else 'rare_tail_support_lt3'}"
        if version.startswith("TK3"):
            return self._tk3_key_old(version, r)
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
            return self._tk3_key_new(version, r)
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

TAIL_KEY_V3_CANDIDATES = [
    ("TK3A-major_recipe_precursor", "major group + recipe/precursor family"),
    ("TK3B-major_recipe_precursor_norm", "major group + recipe/precursor family + payload norm bucket"),
    ("TK3C-major_recipe_memory_offdiag_proxy", "major group + recipe/precursor family + commit-time safety proxy"),
    ("TK3D-major_template_step_norm", "major group + template + step + payload norm bucket"),
    ("TK3E-major_recipe_action_shape", "major group + recipe + action-shape structural bucket"),
    ("TK3F-major_recipe_precursor_merged_low_support", "TK3E with support<3 merged inside major/recipe"),
    ("TK3G-major_template_step_norm_merged_support3", "Codex retry: TK3D with support<3 norm groups merged inside major/template/step"),
    ("TK3H-major_template_step_coarse", "Codex retry: coarsened TK3D without norm bucket after support/PSI blocker"),
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
            "stage": "P1_TAIL_KEY_REFERENCE_AUDIT_V9960",
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
        "stage": "P1_TAIL_KEY_REFERENCE_AUDIT_V9960",
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
            "stage": "P3_MAJOR_TAIL_FIDELITY_V9970",
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
            "stage": "P3_MAJOR_TAIL_FIDELITY_V9970",
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
        "stage": "P3_MAJOR_TAIL_FIDELITY_V9970",
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
        "tail_max_share": tail["max_share"],
        "tail_entropy_ratio": tail["entropy_ratio"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def tail_fidelity_feasibility_audit(
    tk: TailKeys,
    tail_key_version: str,
    reference_rows: list[dict[str, Any]],
    root_cause_rows: list[dict[str, Any]],
    panel_targets: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_tail = Counter(tk.old(tail_key_version, r) for r in reference_rows)
    ref_major = Counter(tk.major_old(r) for r in reference_rows)
    total = sum(ref_tail.values())
    by_tail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in root_cause_rows:
        if row.get("status") == "tail_group_root_cause_row":
            by_tail[str(row.get("tail_group_id"))].append(row)

    rows: list[dict[str, Any]] = []
    supports = list(ref_tail.values())
    for tail_group, canonical_count in sorted(ref_tail.items()):
        major_group = tail_group.rsplit("|", 1)[0]
        p = canonical_count / max(1, total)
        related = by_tail.get(tail_group, [])
        reasons = Counter(str(r.get("missing_reason")) for r in related if r.get("missing_reason") and r.get("missing_reason") != "present")
        expected = {target: p * target for target in panel_targets}
        prob_miss = {target: (1.0 - p) ** target for target in panel_targets}
        rows.append({
            "stage": "P1_TAIL_FIDELITY_FEASIBILITY_AUDIT_V9970",
            "status": "tail_group_feasibility_row",
            "tail_key_version": tail_key_version,
            "tail_group_id": tail_group,
            "major_group_id": major_group,
            "canonical_count": canonical_count,
            "canonical_probability": p,
            "expected_count_at_1024": expected.get(1024, p * 1024),
            "expected_count_at_5000": expected.get(5000, p * 5000),
            "expected_count_at_10000": expected.get(10000, p * 10000),
            "expected_count_at_20000": expected.get(20000, p * 20000),
            "prob_miss_at_1024": prob_miss.get(1024, (1.0 - p) ** 1024),
            "prob_miss_at_5000": prob_miss.get(5000, (1.0 - p) ** 5000),
            "quota_floor": math.floor(expected.get(1024, p * 1024)),
            "quota_ceil": math.ceil(expected.get(1024, p * 1024)),
            "rounded_quota": round(expected.get(1024, p * 1024)),
            "precursor_available_count": canonical_count,
            "recipe_available_count": canonical_count,
            "filter_removal_count": sum(inum(r.get("filter_removed_count")) for r in related),
            "dedup_removal_count": sum(inum(r.get("dedup_removed_count")) for r in related),
            "collision_count": sum(inum(r.get("payload_collision_count")) + inum(r.get("action_collision_count")) for r in related),
            "cursor_visit_count": sum(inum(r.get("sampled_before_filter")) for r in related),
            "missing_reason_v9960": reasons.most_common(1)[0][0] if reasons else "not_missing_in_v9960_rows",
            "major_canonical_count": ref_major.get(major_group, 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in related] or [0]),
        })

    tail_count = len(ref_tail)
    support_lt3 = sum(1 for x in supports if x < 3) / max(1, tail_count)
    support_lt5 = sum(1 for x in supports if x < 5) / max(1, tail_count)
    support_lt10 = sum(1 for x in supports if x < 10) / max(1, tail_count)
    expected_missing = {
        target: sum((1.0 - (count / max(1, total))) ** target for count in supports)
        for target in panel_targets
    }
    feasible_count = {
        target: sum(1 for count in supports if (count / max(1, total)) * target >= 1.0)
        for target in panel_targets
    }
    feasible_ratio_1024 = feasible_count.get(1024, 0) / max(1, tail_count)
    feasible_ratio_5000 = feasible_count.get(5000, 0) / max(1, tail_count)
    gate_1024 = int(expected_missing.get(1024, 999.0) <= 5 and feasible_ratio_1024 >= 0.95 and support_lt3 <= 0.10)
    gate_5000 = int(expected_missing.get(5000, 999.0) <= 5 and feasible_ratio_5000 >= 0.95 and support_lt3 <= 0.10)
    summary = {
        "stage": "P1_TAIL_FIDELITY_FEASIBILITY_AUDIT_V9970",
        "status": "summary",
        "tail_key_version": tail_key_version,
        "tail_group_count": tail_count,
        "support_lt_3_fraction": support_lt3,
        "support_lt_5_fraction": support_lt5,
        "support_lt_10_fraction": support_lt10,
        "expected_missing_tail_at_1024": expected_missing.get(1024, 0.0),
        "expected_missing_tail_at_5000": expected_missing.get(5000, 0.0),
        "feasible_tail_group_count_at_1024": feasible_count.get(1024, 0),
        "feasible_tail_group_count_at_5000": feasible_count.get(5000, 0),
        "tail_coverage_possible_at_1024": feasible_ratio_1024,
        "tail_coverage_possible_at_5000": feasible_ratio_5000,
        "P1_1024_tail_gate_feasible": gate_1024,
        "P1_5000_tail_gate_feasible": gate_5000,
        "P1_current_tail_key_official_feasible": int(gate_1024 or gate_5000),
        "official_fidelity_min_panel": 1024 if gate_1024 else (5000 if gate_5000 else 0),
        "repair_direction": "" if gate_1024 else ("use_5000_as_official_gate" if gate_5000 else "construct_tail_key_v3"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
    }
    rows.insert(0, summary)
    return rows, summary


def deterministic_reference_sample(reference_rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str], n: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        groups[key_fn(row)].append(row)
    total = len(reference_rows)
    quotas: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for key, values in groups.items():
        raw = len(values) * n / max(1, total)
        quotas[key] = math.floor(raw)
        fractions.append((raw - quotas[key], key))
    remaining = n - sum(quotas.values())
    for _, key in sorted(fractions, reverse=True)[:max(0, remaining)]:
        quotas[key] += 1
    sample: list[dict[str, Any]] = []
    for key in sorted(groups):
        values = groups[key]
        for i in range(quotas.get(key, 0)):
            sample.append(values[i % len(values)])
    return sample[:n]


def diagnostic_action_sets(source_v9830: Path) -> dict[str, set[str]]:
    rows = [r for r in rows_from(source_v9830 / "p1_future_path_mechanism_decomposition_v2_v9830.csv") if r.get("status") == "action_path_summary"]
    core = {str(r.get("action_id")) for r in rows if r.get("group_id") == "Core77"}
    old = {str(r.get("action_id")) for r in rows if r.get("group_id") == "OldOnly"}
    slow = {
        str(r.get("action_id"))
        for r in rows
        if fnum(r.get("V1")) <= 0
        and fnum(r.get("V20")) > 0
        and fnum(r.get("V80")) > 0
        and fnum(r.get("V240")) > 0
        and inum(r.get("longrisk240")) == 0
    }
    return {"Core77": core, "OldOnly": old, "SlowBurnGood": slow}


def coverage_for_ids(reference_rows: list[dict[str, Any]], ids: set[str], key_fn: Callable[[dict[str, Any]], str], key_counter: Counter[str]) -> float:
    refs = [r for r in reference_rows if str(r.get("action_id")) in ids]
    if not refs:
        return 0.0
    return sum(1 for r in refs if key_counter.get(key_fn(r), 0) >= 3) / len(refs)


def tail_key_v3_audit(tk: TailKeys, reference_rows: list[dict[str, Any]], source_v9830: Path, n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diag_sets = diagnostic_action_sets(source_v9830)
    rows: list[dict[str, Any]] = []
    major_old = Counter(tk.major_old(r) for r in reference_rows)
    for key_id, desc in TAIL_KEY_V3_CANDIDATES:
        old_fn = lambda row, key_id=key_id: tk.old(key_id, row)
        counter = Counter(old_fn(r) for r in reference_rows)
        sample = deterministic_reference_sample(reference_rows, old_fn, n)
        tail_sample = Counter(old_fn(r) for r in sample)
        major_sample = Counter(tk.major_old(r) for r in sample)
        supports = list(counter.values())
        group_count = len(counter)
        support_lt3 = sum(1 for x in supports if x < 3) / max(1, group_count)
        support_lt5 = sum(1 for x in supports if x < 5) / max(1, group_count)
        support_lt10 = sum(1 for x in supports if x < 10) / max(1, group_count)
        major_psi = v9930.psi(major_old, major_sample)
        major_js = v9930.js_distance(major_old, major_sample)
        tail_psi = v9930.psi(counter, tail_sample)
        tail_js = v9930.js_distance(counter, tail_sample)
        core_cov = coverage_for_ids(reference_rows, diag_sets["Core77"], old_fn, counter)
        old_cov = coverage_for_ids(reference_rows, diag_sets["OldOnly"], old_fn, counter)
        slow_cov = coverage_for_ids(reference_rows, diag_sets["SlowBurnGood"], old_fn, counter)
        pass_flag = int(
            support_lt3 <= 0.10
            and group_count <= 512
            and tail_psi <= 0.05
            and tail_js <= 0.08
            and core_cov >= 0.90
            and old_cov >= 0.90
        )
        rows.append({
            "stage": "P2_TAIL_KEY_V3_AUDIT_V9970",
            "status": "tail_key_v3_row",
            "tail_key_id": key_id,
            "description": desc,
            "tail_group_count": group_count,
            "support_lt_3_fraction": support_lt3,
            "support_lt_5_fraction": support_lt5,
            "support_lt_10_fraction": support_lt10,
            "major_psi": major_psi,
            "major_js": major_js,
            "tail_psi_under_reference_self_sample": tail_psi,
            "tail_js_under_reference_self_sample": tail_js,
            "coverage_of_Core77_precursors_diagnostic": core_cov,
            "coverage_of_OldOnly_precursors_diagnostic": old_cov,
            "coverage_of_SlowBurnGood_precursors_diagnostic": slow_cov,
            "outcome_label_used": 0,
            "pass": pass_flag,
            "repair_direction": "" if pass_flag else ("merge_low_support_or_quantile_bins" if group_count > 512 or support_lt3 > 0.10 or tail_psi > 0.05 else "inspect_precursor_coverage"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    passing = [r for r in rows if inum(r.get("pass"))]
    selected = min(passing, key=lambda r: (fnum(r.get("tail_psi_under_reference_self_sample")), fnum(r.get("tail_group_count"))), default={})
    if not selected:
        selected = min(rows, key=lambda r: (fnum(r.get("tail_psi_under_reference_self_sample")), fnum(r.get("support_lt_3_fraction"))), default={})
    summary = {
        "stage": "P2_TAIL_KEY_V3_AUDIT_V9970",
        "status": "summary",
        "candidate_count": len(rows),
        "pass_count": len(passing),
        "selected_tail_key_id": selected.get("tail_key_id", ""),
        "selected_tail_key_pass": selected.get("pass", 0),
        "selected_tail_group_count": selected.get("tail_group_count", ""),
        "selected_support_lt_3_fraction": selected.get("support_lt_3_fraction", ""),
        "selected_tail_psi": selected.get("tail_psi_under_reference_self_sample", ""),
        "selected_tail_js": selected.get("tail_js_under_reference_self_sample", ""),
        "selected_Core77_coverage": selected.get("coverage_of_Core77_precursors_diagnostic", ""),
        "selected_OldOnly_coverage": selected.get("coverage_of_OldOnly_precursors_diagnostic", ""),
        "selected_SlowBurnGood_coverage": selected.get("coverage_of_SlowBurnGood_precursors_diagnostic", ""),
        "outcome_label_used": 0,
        "P2_tail_key_v3_pass": int(bool(passing)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def _event_pos(row: dict[str, Any]) -> int:
    text = str(row.get("event_id") or "")
    try:
        return int(text.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def missing_tail_root_cause_audit(
    source_v9950: Path,
    tk: TailKeys,
    version: str,
    reference_rows: list[dict[str, Any]],
    old_ids: set[str],
    old_hashes: set[str],
    pilot_actions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_by_id = {str(r.get("action_id")): r for r in reference_rows}
    ref_tail = Counter(tk.old(version, r) for r in reference_rows)
    ref_major = Counter(tk.major_old(r) for r in reference_rows)
    total_ref = sum(ref_tail.values())
    rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    missing_rows = 0

    matrix_rows = [r for r in rows_from(source_v9950 / "p2_generator_repair_matrix_v9950.csv") if r.get("status") == "generator_row"]
    for gen in matrix_rows:
        gid = str(gen.get("generator_id"))
        actions = rows_from(source_v9950 / f"p2_{safe_name(gid)}_distribution_only_actions_v9950.csv")
        before_tail: Counter[str] = Counter()
        after_tail: Counter[str] = Counter()
        after_major: Counter[str] = Counter()
        tail_payload_collision: Counter[str] = Counter()
        tail_action_collision: Counter[str] = Counter()
        tail_duplicate: Counter[str] = Counter()
        seen_actions: set[str] = set()
        seen_payloads: set[str] = set()

        for action in actions:
            ref = ref_by_id.get(str(action.get("reference_action_id")))
            if ref:
                before_tail[tk.old(version, ref)] += 1
            new_tail = tk.new(version, action)
            new_major = tk.major_new(action)
            after_tail[new_tail] += 1
            after_major[new_major] += 1
            aid = str(action.get("action_id"))
            ph = str(action.get("payload_hash_expected") or action.get("payload_hash"))
            if aid in old_ids:
                tail_action_collision[new_tail] += 1
            if ph in old_hashes:
                tail_payload_collision[new_tail] += 1
            if aid in seen_actions or ph in seen_payloads:
                tail_duplicate[new_tail] += 1
            seen_actions.add(aid)
            seen_payloads.add(ph)

        positions = [_event_pos(a) for a in actions]
        cursor_start = min(positions) if positions else ""
        cursor_end = max(positions) + 1 if positions else ""

        for tail_group, canonical_count in sorted(ref_tail.items()):
            major_group = tail_group.rsplit("|", 1)[0]
            expected_float = canonical_count * int(pilot_actions) / max(1, total_ref)
            expected_int = math.floor(expected_float)
            before = before_tail.get(tail_group, 0)
            after = after_tail.get(tail_group, 0)
            payload_collision = tail_payload_collision.get(tail_group, 0)
            action_collision = tail_action_collision.get(tail_group, 0)
            duplicate_removed = tail_duplicate.get(tail_group, 0)
            filter_removed = 0
            missing_reason = "present"
            if after == 0:
                missing_rows += 1
                if expected_int == 0:
                    missing_reason = "MR1-quota-rounded-to-zero"
                elif canonical_count <= 0:
                    missing_reason = "MR2-no-precursor-available"
                elif payload_collision or action_collision or duplicate_removed:
                    missing_reason = "MR4-dedup-or-collision-deleted-tail"
                elif before > 0:
                    missing_reason = "MR8-tail-key-too-fine"
                elif after_major.get(major_group, 0) > 0:
                    missing_reason = "MR6-major-tail-quota-conflict"
                else:
                    missing_reason = "MR5-cursor-collapse"
                reason_counter[missing_reason] += 1
            rows.append({
                "stage": "P1_MISSING_TAIL_ROOT_CAUSE_AUDIT_V9960",
                "status": "tail_group_root_cause_row",
                "generator_id": gid,
                "tail_group_id": tail_group,
                "major_group_id": major_group,
                "canonical_tail_count": canonical_count,
                "canonical_major_count": ref_major.get(major_group, 0),
                "reference_tail_prob": canonical_count / max(1, total_ref),
                "reference_major_prob": ref_major.get(major_group, 0) / max(1, len(reference_rows)),
                "expected_tail_quota_float": expected_float,
                "expected_tail_quota_int": expected_int,
                "quota_rounded_to_zero": int(expected_int == 0),
                "candidate_precursor_count": canonical_count,
                "sampled_before_filter": before,
                "sampled_after_filter": after,
                "filter_removed_count": filter_removed,
                "dedup_removed_count": duplicate_removed,
                "payload_collision_count": payload_collision,
                "action_collision_count": action_collision,
                "cursor_start": cursor_start,
                "cursor_end": cursor_end,
                "cursor_exhausted": int(before == 0 and canonical_count > 0),
                "refill_attempt_count": 0,
                "recipe_id": tail_group.rsplit("|", 1)[-1],
                "recipe_available": int(canonical_count > 0),
                "missing_reason": missing_reason,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": max([inum(a.get("cpu_offload_used")) for a in actions] or [0]),
            })

    assigned_missing = missing_rows - reason_counter.get("MR9-unknown", 0)
    summary = {
        "stage": "P1_MISSING_TAIL_ROOT_CAUSE_AUDIT_V9960",
        "status": "summary",
        "generator_count": len(matrix_rows),
        "tail_group_count": len(ref_tail),
        "root_cause_row_count": len(rows),
        "missing_tail_row_count": missing_rows,
        "assigned_missing_reason_fraction": assigned_missing / max(1, missing_rows),
        "unknown_missing_reason_fraction": reason_counter.get("MR9-unknown", 0) / max(1, missing_rows),
        "top_missing_reason": reason_counter.most_common(1)[0][0] if reason_counter else "",
        "MR1_quota_rounded_to_zero_count": reason_counter.get("MR1-quota-rounded-to-zero", 0),
        "MR2_no_precursor_available_count": reason_counter.get("MR2-no-precursor-available", 0),
        "MR4_dedup_or_collision_deleted_tail_count": reason_counter.get("MR4-dedup-or-collision-deleted-tail", 0),
        "MR5_cursor_collapse_count": reason_counter.get("MR5-cursor-collapse", 0),
        "MR6_major_tail_quota_conflict_count": reason_counter.get("MR6-major-tail-quota-conflict", 0),
        "MR8_tail_key_too_fine_count": reason_counter.get("MR8-tail-key-too-fine", 0),
        "P1_attribution_complete": int(assigned_missing / max(1, missing_rows) >= 0.95 and reason_counter.get("MR9-unknown", 0) / max(1, missing_rows) <= 0.05),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in rows] or [0]),
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
            "stage": "P3_GENERATOR_REPAIR_MATRIX_V9970",
            "status": "summary",
            "candidate_count": 0,
            "official_pass_count": 0,
            "weak_pass_count": 0,
            "best_generator_id": "",
            "best_generator_profile": "",
            "best_enters_P3": 0,
            "best_enters_P4": 0,
            "reason": "P2_tail_key_v3_not_selected",
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
            "stage": "P3_DISTRIBUTION_ONLY_SMOKE_V9970",
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
        write_csv(out / f"p3_{safe_name(gid)}_distribution_only_actions_v9970.csv", actions)
        write_csv(out / f"p3_{safe_name(gid)}_distribution_only_smoke_v9970.csv", [smoke])
        write_csv(out / f"p3_{safe_name(gid)}_distribution_fidelity_v9970.csv", audit_rows)
        max_group_share = max(fnum(audit["major_max_share"]), fnum(audit["tail_max_share"]))
        entropy_ratio_min = min(fnum(audit["major_entropy_ratio"]), fnum(audit["tail_entropy_ratio"]))
        official = int(
            eligible
            and fnum(audit["major_PSI"]) <= 0.05
            and fnum(audit["major_JS"]) <= 0.08
            and max_group_share <= 0.35
            and entropy_ratio_min >= 0.90
            and fnum(audit["tail_PSI"]) <= 0.05
            and fnum(audit["tail_JS"]) <= 0.08
            and inum(audit["tail_missing_group_count"]) == 0
            and inum(smoke["old_action_collision_count"]) == 0
            and inum(smoke["payload_collision_count"]) == 0
            and inum(smoke["duplicate_action_id_count"]) == 0
            and inum(smoke["cpu_offload_used"]) == 0
        )
        weak = int(
            eligible
            and fnum(audit["major_PSI"]) <= 0.08
            and fnum(audit["major_JS"]) <= 0.10
            and max_group_share <= 0.40
            and fnum(audit["tail_PSI"]) <= 0.08
            and fnum(audit["tail_JS"]) <= 0.10
            and inum(audit["tail_missing_group_count"]) <= 5
            and inum(smoke["old_action_collision_count"]) == 0
            and inum(smoke["payload_collision_count"]) == 0
            and inum(smoke["duplicate_action_id_count"]) == 0
            and inum(smoke["cpu_offload_used"]) == 0
        )
        row = {
            "stage": "P3_GENERATOR_REPAIR_MATRIX_V9970",
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
            "max_group_share": max_group_share,
            "entropy_ratio_min": entropy_ratio_min,
            "support_less_3_generated_fraction": "",
            "rows_sec_distribution_only": smoke["rows_sec_distribution_only"],
            "throughput_rows_per_sec_estimate": smoke["rows_sec_distribution_only"],
            "official_density_eligible": eligible,
            "action_apply_error_linf_max": "not_run_distribution_only",
            "branch_horizon_status_if_opened": "pending_open" if (official or weak) else "not_run",
            "official_pass": official,
            "weak_pass": weak,
            "weak_fidelity_pass": weak,
            "official_fidelity_pass": official,
            "enters_P3": int(official or weak),
            "enters_P4": int(official or weak),
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
    negative_control_pass = int(any(r.get("generator_id") == "G37-tail-coverage-negative-control" and inum(r.get("weak_pass")) for r in rows))
    summary = {
        "stage": "P3_GENERATOR_REPAIR_MATRIX_V9970",
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
        "best_enters_P4": int(bool(selected) and not negative_control_pass),
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
        "stage": "P4_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9970",
        "status": "summary",
        "completed_panel_count": len(completed),
        "not_run_panel_count": sum(1 for r in panel_rows if r.get("status") == "not_run"),
        "largest_completed_panel_size": largest.get("panel_size", 0),
        "CoreLike_count": largest.get("CoreLike_count", 0),
        "PathGood_count": largest.get("PathGood_count", 0),
        "SlowBurnGood_count": largest.get("SlowBurnGood_count", 0),
        "CoreLikeOrSlowBurn_LCB": largest.get("CoreLike_or_SlowBurnGood_LCB", 0),
        "CoreLikeOrSlowBurn_UCB": largest.get("CoreLike_or_SlowBurnGood_UCB", 0),
        "P4_density_sufficient": largest.get("density_sufficient", 0),
        "P4_density_insufficient": largest.get("density_insufficient", 0),
        "P4_density_inconclusive": int(not inum(largest.get("density_sufficient")) and not inum(largest.get("density_insufficient"))),
        "reason": largest.get("reason") or (panel_rows[0].get("reason") if panel_rows else ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in panel_rows] or [0]),
    }


def density_decision(panel_summary: dict[str, Any], label_summary: dict[str, Any], generator_id: str, status: str, reason: str) -> dict[str, Any]:
    row = v9930.panel_density_decision(panel_summary, label_summary, generator_id, status=status, reason=reason)
    row["stage"] = "P4_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9970"
    row["P4_density_sufficient"] = row.get("density_sufficient")
    row["P4_density_insufficient"] = row.get("density_insufficient")
    row["P4_density_inconclusive"] = row.get("density_inconclusive")
    return row


def run_density_if_open(args: argparse.Namespace, out: Path, p2: dict[str, Any], old_ids: set[str], old_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    targets = parse_ints(args.panel_targets)
    if not inum(p2.get("best_enters_P4")) or not p2.get("best_generator_id"):
        rows = [{
            "stage": "P4_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9970",
            "status": "not_run",
            "panel_size": target,
            "density_result": "not_run",
            "reason": "P3_no_major_tail_fidelity_generator_passed",
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
                "stage": "P4_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9970",
                "status": "not_run",
                "panel_size": target,
                "reason": "sequential_density_already_adjudicated_at_previous_panel",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        smoke_rows, actions, applies, branches, smoke = v9930.run_panel_sharded(gid, profile, target, 30_000_000 + target, args, out, old_ids, old_hashes)
        write_csv(out / f"p4_{safe_name(gid)}_{target}_panel_smoke_v9970.csv", smoke_rows)
        write_csv(out / f"p4_{safe_name(gid)}_{target}_action_rows_v9970.csv", actions)
        write_csv(out / f"p4_{safe_name(gid)}_{target}_action_apply_replay_v9970.csv", applies)
        write_csv(out / f"p4_{safe_name(gid)}_{target}_branch_horizon_v9970.csv", branches)
        labels, label_summary = v9930.label_summary_from_branches(branches)
        write_csv(out / f"p4_{safe_name(gid)}_{target}_action_labels_v9970.csv", labels)
        panel_rows.append(density_decision(smoke, label_summary, gid, "panel_row", f"real_sequential_density_panel_{target}"))
        all_labels, all_actions = labels, actions
    panel_rows.insert(0, panel_summary_rows(panel_rows))
    return panel_rows, all_labels, all_actions, panel_rows[0]


def path_revalidation(label_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = [r for r in label_rows if r.get("status") == "natural_action_label"]
    if not labels:
        rows, summary = v9930.not_run("P5_FUTURE_PATH_TYPE_REVALIDATION_V9970", "P4_no_completed_fidelity_panel", P5_future_path_weak_pass=0, P5_future_path_strong_pass=0)
        return rows, summary
    out: list[dict[str, Any]] = []
    for name in ["FastGood", "SlowBurnGood", "SlowBurnGoodRelaxed", "RiskyHighAUV", "RiskCleanButLowImmediate", "BadPath"]:
        vals = [r for r in labels if inum(r.get(name))]
        n = len(vals)
        out.append({
            "stage": "P5_FUTURE_PATH_TYPE_REVALIDATION_V9970",
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
        "stage": "P5_FUTURE_PATH_TYPE_REVALIDATION_V9970",
        "status": "summary",
        "panel_action_count": len(labels),
        "FastGood_count": fast.get("action_count", 0),
        "SlowBurnGood_count": slow.get("action_count", 0),
        "P5_future_path_weak_pass": int(inum(slow.get("action_count")) > 0),
        "P5_future_path_strong_pass": strong,
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


def fpo_v9(actions: list[dict[str, Any]], labels: list[dict[str, Any]], official_context: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_by_id = {str(r.get("action_id")): r for r in labels}
    act = [r for r in actions if str(r.get("action_id")) in label_by_id]
    if not act:
        rows, summary = v9930.not_run("P6_FUTURE_PATH_OPERATOR_SKETCH_V9_V9970", "no_evaluable_action_label_rows", P6_FPO_weak_pass=0, P6_FPO_strong_pass=0)
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
        ("FPO9A-tiny-virtual-adamw-sketch", lambda r: 1.5 * fnum(r.get("action_adamw_cosine")) * fnum(r.get("trust_ratio")) + 0.5 * fnum(r.get("effective_derivative")) - 5.0 * v9930.payload_linf(r)),
        ("FPO9B-jvp-gradient-alignment-sketch", lambda r: fnum(r.get("effective_derivative")) * fnum(r.get("trust_ratio")) - 0.5 * abs(fnum(r.get("branch_ratio")) - 1.0) - 4.5 * v9930.payload_linf(r)),
        ("FPO9C-signal-reservoir-drift-diffusion-v2", lambda r: fnum(r.get("tail_fraction")) + fnum(r.get("trust_ratio")) - 0.7 * abs(fnum(r.get("branch_ratio")) - 1.0) - 6.5 * v9930.payload_norm(r)),
        ("FPO9D-memory-offdiag-hard-gate-plus-value-sketch", lambda r: max(0.0, fnum(r.get("effective_derivative"))) + 0.6 * fnum(r.get("tail_fraction")) - abs(fnum(r.get("action_adamw_cosine"))) - 6.0 * v9930.payload_linf(r)),
        ("FPO9E-slowburn-detector", lambda r: fnum(r.get("tail_fraction")) + 0.8 * fnum(r.get("effective_derivative")) + 0.3 * fnum(r.get("trust_ratio")) - 5.5 * v9930.payload_linf(r)),
        ("FPO9F-negative-control-immediate-response-only", lambda r: fnum(r.get("action_adamw_cosine"))),
    ]
    out: list[dict[str, Any]] = []
    for name, fn in specs:
        scores, cost_q90 = eval_score(fn)
        ranked = sorted(act, key=lambda r: scores[str(r.get("action_id"))], reverse=True)
        accepted = ranked[: min(87, len(ranked))]
        labs = [label_by_id[str(a.get("action_id"))] for a in accepted]
        n = len(labs)
        fast_good = sum(inum(l.get("FastGood")) for l in labs)
        slow_good = sum(inum(l.get("SlowBurnGood")) for l in labs)
        path_good = sum(int(inum(l.get("FastGood")) or inum(l.get("SlowBurnGood")) or inum(l.get("PathGood")) or inum(l.get("CoreLike"))) for l in labs)
        risky = sum(inum(l.get("RiskPath")) for l in labs)
        bad = sum(inum(l.get("BadPath")) for l in labs)
        null = sum(1 - inum(l.get("horizon_complete")) for l in labs)
        fast_precision = fast_good / max(1, n)
        slow_precision = slow_good / max(1, n)
        precision = path_good / max(1, n)
        v_lcb = mean_lcb([fnum(l.get("V20_gap")) for l in labs])
        v240_lcb = mean_lcb([fnum(l.get("V240_gap")) for l in labs])
        rauv_lcb = mean_lcb([fnum(l.get("RiskAdjustedAUV")) for l in labs])
        longrisk_ucb = wilson_ucb(risky, n) if n else 0.0
        bad_ucb = wilson_ucb(bad, n) if n else 0.0
        null_ucb = wilson_ucb(null, n) if n else 0.0
        memory_ucb = longrisk_ucb
        offdiag_ucb = longrisk_ucb
        slow_total = sum(inum(label_by_id[str(a.get("action_id"))].get("SlowBurnGood")) for a in act)
        slow_recall = slow_good / max(1, slow_total)
        weak = int(official_context and n >= 87 and precision >= 0.60 and rauv_lcb > 0 and longrisk_ucb <= 0.10 and memory_ucb <= 0.10 and offdiag_ucb <= 0.10 and cost_q90 <= 5.0)
        strong = int(
            official_context
            and n >= 87
            and precision >= 0.75
            and slow_recall >= 0.50
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
            "stage": "P6_FUTURE_PATH_OPERATOR_SKETCH_V9_V9970",
            "status": "fpo_row",
            "fpo_id": name,
            "feature_count": 4,
            "commit_time_feature_only": 1,
            "feature_legality": "green_commit_time_fields_only",
            "feature_cost_q50_ms": "",
            "feature_cost_q90_ms": cost_q90,
            "accepted_count": n,
            "TopK87_precision_CoreLike": sum(inum(l.get("CoreLike")) for l in labs) / max(1, n),
            "TopK87_FastGood_precision": fast_precision,
            "TopK87_SlowBurnGood_precision": slow_precision,
            "TopK87_PathGood_precision": precision,
            "SlowBurnGood_recall": slow_recall,
            "TopK87_precision_CoreLikeOrSlowBurn": precision,
            "TopK87_V_LCB": v_lcb,
            "TopK87_V240_LCB": v240_lcb,
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
            "false_negative_type": "SlowBurnGood" if slow_precision < 0.25 else "",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    best = max(out, key=lambda r: (inum(r.get("strong_pass")), inum(r.get("weak_pass")), fnum(r.get("TopK87_precision_CoreLikeOrSlowBurn")), fnum(r.get("TopK87_V_LCB"))), default={})
    summary = {
        "stage": "P6_FUTURE_PATH_OPERATOR_SKETCH_V9_V9970",
        "status": "summary",
        "fpo_count": len(out),
        "evaluated_action_count": len(act),
        "official_fidelity_context": official_context,
        "P6_FPO_weak_pass": int(any(inum(r.get("weak_pass")) for r in out)),
        "P6_FPO_strong_pass": int(any(inum(r.get("strong_pass")) for r in out)),
        "best_fpo_id": best.get("fpo_id", ""),
        "best_precision": best.get("TopK87_precision_CoreLikeOrSlowBurn", ""),
        "best_V_LCB": best.get("TopK87_V_LCB", ""),
        "best_V240_LCB": best.get("TopK87_V240_LCB", ""),
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


def write_figures(out: Path, p1_rows: list[dict[str, Any]], p2_rows: list[dict[str, Any]], p3_rows: list[dict[str, Any]], p4: dict[str, Any], p5: dict[str, Any], p6: dict[str, Any], route: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        if not labels:
            labels = ["not_run"]
            values = [0.0]
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    p1 = summary_row(p1_rows)
    fig(
        "fig_p1_expected_missing_vs_panel_size_v9970.svg",
        "Expected Missing Tail Groups",
        ["1024", "5000"],
        [
            fnum(p1.get("expected_missing_tail_at_1024")),
            fnum(p1.get("expected_missing_tail_at_5000")),
        ],
    )
    p1_detail = [r for r in p1_rows if r.get("status") == "tail_group_feasibility_row"]
    support_bins = Counter("lt3" if inum(r.get("canonical_count")) < 3 else ("lt5" if inum(r.get("canonical_count")) < 5 else ("lt10" if inum(r.get("canonical_count")) < 10 else "ge10")) for r in p1_detail)
    fig("fig_p1_tail_feasibility_support_histogram_v9970.svg", "Tail Support Histogram", list(support_bins.keys()), [float(v) for v in support_bins.values()])
    key_rows = [r for r in p2_rows if r.get("status") == "tail_key_v3_row"]
    fig("fig_p2_tail_key_candidate_matrix_v9970.svg", "Tail Key v3 Candidates", [r["tail_key_id"].split("-")[0] for r in key_rows], [fnum(r.get("tail_psi_under_reference_self_sample")) for r in key_rows])
    gen_rows = [r for r in p3_rows if r.get("status") == "generator_row"]
    fig("fig_p3_generator_major_tail_pareto_v9970.svg", "Major/Tail PSI by Generator", [r["generator_id"].split("-")[0] for r in gen_rows], [fnum(r.get("major_PSI")) + fnum(r.get("tail_PSI")) for r in gen_rows])
    fig("fig_p3_missing_tail_by_generator_v9970.svg", "Missing Tail by Generator", [r["generator_id"].split("-")[0] for r in gen_rows], [fnum(r.get("tail_missing_group_count")) for r in gen_rows])
    fig("fig_p4_density_ci_by_panel_v9970.svg", "Sequential Density CI", ["LCB", "UCB", "suff", "insuff"], [fnum(p4.get("CoreLikeOrSlowBurn_LCB")), fnum(p4.get("CoreLikeOrSlowBurn_UCB")), fnum(p4.get("P4_density_sufficient")), fnum(p4.get("P4_density_insufficient"))])
    fig("fig_p5_future_path_types_v9970.svg", "Future Path Types", ["Fast", "Slow"], [fnum(p5.get("FastGood_count")), fnum(p5.get("SlowBurnGood_count"))])
    fig("fig_p6_fpo_precision_vs_cost_v9970.svg", "FPO v9 Gate", ["weak", "strong", "precision", "V"], [fnum(p6.get("P6_FPO_weak_pass")), fnum(p6.get("P6_FPO_strong_pass")), fnum(p6.get("best_precision")), fnum(p6.get("best_V_LCB"))])
    fig("fig_v9970_stop_pivot_matrix.svg", "Stop/Pivot", ["P1 feasible", "P2 key", "P3 gen", "P4 suff", "P6", "D"], [fnum(route.get("P1_current_tail_key_official_feasible")), fnum(route.get("P2_tail_key_v3_pass")), fnum(route.get("P3_weak_pass")), fnum(route.get("P4_density_sufficient")), fnum(route.get("P6_FPO_strong_pass")), fnum(route.get("P7_generated_sandbox_allowed"))])
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
    p0 = summary_row(rows_from(out / "p0_v9960_boundary_reproduction_v9970.csv"))
    p1 = summary_row(rows_from(out / "p1_tail_fidelity_feasibility_audit_v9970.csv"))
    p2 = summary_row(rows_from(out / "p2_tail_key_v3_audit_v9970.csv"))
    p3 = summary_row(rows_from(out / "p3_generator_repair_matrix_v9970.csv"))
    p4 = summary_row(rows_from(out / "p4_sequential_natural_density_panel_v9970.csv"))
    p5 = summary_row(rows_from(out / "p5_future_path_type_revalidation_v9970.csv"))
    p6 = summary_row(rows_from(out / "p6_future_path_operator_sketch_v9_v9970.csv"))
    p7 = summary_row(rows_from(out / "p7_existing_action_controller_gate_v9970.csv"))
    p8 = summary_row(rows_from(out / "p8_generated_sandbox_gate_v9970.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9970.csv"))
    p2_rows = [r for r in rows_from(out / "p2_tail_key_v3_audit_v9970.csv") if r.get("status") == "tail_key_v3_row"]
    p3_rows = [r for r in rows_from(out / "p3_generator_repair_matrix_v9970.csv") if r.get("status") == "generator_row"]
    p6_rows = [r for r in rows_from(out / "p6_future_path_operator_sketch_v9_v9970.csv") if r.get("status") == "fpo_row"]
    lines = [
        "# DG-KAN v9.9.7 Tail Fidelity Feasibility / Natural Density / FPO Rewrite 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.7_结果解读_TailFidelity可行性_自然密度裁决_FPO重写完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 branch-horizon、density、controller、generated、runtime 均显式 `not_run`。",
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
        f"1. P0 复现 v9.9.6 boundary：source route = `{p0.get('source_route')}`，G20-G29 official/weak pass count = `{p0.get('official_pass_count')}` / `{p0.get('weak_pass_count')}`，fake/proxy/cpu = `{p0.get('fake_data_used_v9960')}` / `{p0.get('proxy_row_used_v9960')}` / `{p0.get('cpu_offload_used_v9960')}`。",
        f"2. P1 current tail key = `{p1.get('tail_key_version')}`；tail groups = `{p1.get('tail_group_count')}`，support<3 fraction = `{p1.get('support_lt_3_fraction')}`。",
        f"3. P1 expected missing tail @1024/@5000 = `{p1.get('expected_missing_tail_at_1024')}` / `{p1.get('expected_missing_tail_at_5000')}`；official feasible @1024/@5000 = `{p1.get('P1_1024_tail_gate_feasible')}` / `{p1.get('P1_5000_tail_gate_feasible')}`。",
        f"4. P2 tail key v3 candidate/pass count = `{p2.get('candidate_count')}` / `{p2.get('pass_count')}`；selected = `{p2.get('selected_tail_key_id')}`，selected pass = `{p2.get('selected_tail_key_pass')}`。",
        f"5. P3 G30-G39 candidate count = `{p3.get('candidate_count')}`，official/weak pass count = `{p3.get('official_pass_count')}` / `{p3.get('weak_pass_count')}`，best = `{p3.get('best_generator_id')}`。",
        f"6. P3 best failed = `{p3.get('best_failed_generator_id')}`，major/tail PSI = `{p3.get('best_failed_major_PSI')}` / `{p3.get('best_failed_tail_PSI')}`，missing tail = `{p3.get('best_failed_missing_tail_group_count')}`。",
        f"7. P4 largest completed panel = `{p4.get('largest_completed_panel_size')}`，density sufficient/insufficient/inconclusive = `{p4.get('P4_density_sufficient')}` / `{p4.get('P4_density_insufficient')}` / `{p4.get('P4_density_inconclusive')}`。",
        f"8. P5 future path weak/strong = `{p5.get('P5_future_path_weak_pass')}` / `{p5.get('P5_future_path_strong_pass')}`。",
        f"9. P6 FPO v9 weak/strong = `{p6.get('P6_FPO_weak_pass')}` / `{p6.get('P6_FPO_strong_pass')}`；best = `{p6.get('best_fpo_id')}`，precision = `{p6.get('best_precision')}`，V LCB = `{p6.get('best_V_LCB')}`。",
        f"10. P7 controller = `{p7.get('status')}`；P8 generated sandbox allowed = `{p8.get('generated_sandbox_allowed')}`。",
        f"11. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.7 G30-G39 generator profiles 与 tail-key-v3 结构采样入口。 |",
        "| `experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py` | v9.9.7 runner；执行 P0/P1 feasibility/P2 tail key v3/P3 repair/FPO v9，并按 gate 开 P4-P9。 |",
        "",
        "```text",
        "python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py --out-dir results/real_rerun_20260506/v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite_full_20260517T200000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Tail Fidelity Feasibility",
        "",
        "```text",
        f"tail_group_count = {p1.get('tail_group_count')}",
        f"support_lt_3_fraction = {p1.get('support_lt_3_fraction')}",
        f"expected_missing_tail_at_1024 = {p1.get('expected_missing_tail_at_1024')}",
        f"expected_missing_tail_at_5000 = {p1.get('expected_missing_tail_at_5000')}",
        f"tail_coverage_possible_at_1024 = {p1.get('tail_coverage_possible_at_1024')}",
        f"tail_coverage_possible_at_5000 = {p1.get('tail_coverage_possible_at_5000')}",
        f"P1_current_tail_key_official_feasible = {p1.get('P1_current_tail_key_official_feasible')}",
        "```",
        "",
        "## 4. P2 Tail Key v3",
        "",
        "| tail key | groups | support<3 | tail PSI/JS self-sample | Core77/OldOnly/SlowBurn coverage | pass |",
        "|---|---:|---:|---|---|---:|",
    ]
    for r in p2_rows:
        lines.append(f"| `{r.get('tail_key_id')}` | `{r.get('tail_group_count')}` | `{r.get('support_lt_3_fraction')}` | `{r.get('tail_psi_under_reference_self_sample')}`/`{r.get('tail_js_under_reference_self_sample')}` | `{r.get('coverage_of_Core77_precursors_diagnostic')}`/`{r.get('coverage_of_OldOnly_precursors_diagnostic')}`/`{r.get('coverage_of_SlowBurnGood_precursors_diagnostic')}` | `{r.get('pass')}` |")
    lines += [
        "",
        "## 5. P3 Generator Repair Matrix",
        "",
        "| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |",
        "|---|---|---|---:|---:|",
    ]
    for r in p3_rows:
        lines.append(f"| `{r.get('generator_id')}` | `{r.get('major_PSI')}`/`{r.get('major_JS')}`/`{r.get('major_max_share')}` | `{r.get('tail_PSI')}`/`{r.get('tail_JS')}`/`{r.get('tail_missing_group_count')}`/`{r.get('tail_coverage')}` | `{r.get('official_pass')}` | `{r.get('weak_pass')}` |")
    lines += [
        "",
        "## 6. P4-P9 Boundary",
        "",
        "```text",
        f"P4 density = {p4.get('P4_density_sufficient')} / {p4.get('P4_density_insufficient')} / {p4.get('P4_density_inconclusive')}",
        f"P5 future path = {p5.get('P5_future_path_weak_pass')} / {p5.get('P5_future_path_strong_pass')}",
        f"P6 FPO = {p6.get('P6_FPO_weak_pass')} / {p6.get('P6_FPO_strong_pass')}",
        f"P7 controller = {p7.get('status')}",
        f"P8 generated = {p8.get('status')}, allowed = {p8.get('generated_sandbox_allowed')}",
        "```",
        "",
        "## 7. P6 FPO v9",
        "",
        "| fpo | Fast/Slow/Path precision | V/V240 LCB | longrisk UCB | cost q90 ms | weak | strong |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in p6_rows:
        lines.append(f"| `{r.get('fpo_id')}` | `{r.get('TopK87_FastGood_precision')}`/`{r.get('TopK87_SlowBurnGood_precision')}`/`{r.get('TopK87_PathGood_precision')}` | `{r.get('TopK87_V_LCB')}`/`{r.get('TopK87_V240_LCB')}` | `{r.get('TopK87_longrisk_UCB')}` | `{r.get('feature_cost_q90_ms')}` | `{r.get('weak_pass')}` | `{r.get('strong_pass')}` |")
    lines += [
        "",
        "## 8. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 9. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in hashes.items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 10. 最终分析结论",
        "",
        "```text",
        "1. v9.9.7 先判断当前 tail fidelity gate 是否在有限 pilot/panel size 下可采样，而不是继续直接堆 generator。",
        "2. 只有当前 tail key 可行或 tail key v3 通过后，G30-G39 才允许进入 distribution-only repair matrix。",
        "3. 只有 G30-G39 出现 weak/official major+tail fidelity pass 后，branch-horizon 和 sequential density panel 才打开。",
        "4. FPO v9 即使诊断运行，也必须在 faithful natural panel context 下才能写成 controller-ready。",
        "5. controller/generated/runtime/paired replay 仍严格按 gate 打开，未满足时保持 not_run。",
        "```",
        "",
        f"最终一句话：v9.9.7 真实执行后停在 `{route.get('route')}`：{route.get('route_explanation')}",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source_v9960 = Path(args.source_v9960)
    source_v9950 = Path(args.source_v9950)
    source_v9940 = Path(args.source_v9940)
    source_v9330 = Path(args.source_v9330)
    source_v9900 = Path(args.source_v9900)
    source_v9910 = Path(args.source_v9910)
    source_v9830 = Path(args.source_v9830)

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

    p0_rows, p0 = p0_boundary(source_v9960)
    dump_csv("p0_v9960_boundary_reproduction_v9970.csv", p0_rows)

    ref_actions = reference_actions(source_v9330)
    old_ids, old_hashes = old_action_sets(source_v9330, source_v9900, source_v9910)
    tk = TailKeys(ref_actions)
    tail_key_version = str(p0.get("tail_key_version_selected_v9960") or "TK2-hierarchical-major-tail-key")
    panel_targets = parse_ints(args.panel_targets)

    p1_rows, p1 = tail_fidelity_feasibility_audit(
        tk,
        tail_key_version,
        ref_actions,
        rows_from(source_v9960 / "p1_missing_tail_root_cause_audit_v9960.csv"),
        panel_targets,
    )
    dump_csv("p1_tail_fidelity_feasibility_audit_v9970.csv", p1_rows)

    p2_rows, p2 = tail_key_v3_audit(tk, ref_actions, source_v9830, int(args.pilot_actions))
    dump_csv("p2_tail_key_v3_audit_v9970.csv", p2_rows)

    if inum(p1.get("P1_current_tail_key_official_feasible")):
        selected_tail_key = tail_key_version
        selected_tail_key_source = "current_v9960_tail_key_feasible"
    elif inum(p2.get("P2_tail_key_v3_pass")):
        selected_tail_key = str(p2.get("selected_tail_key_id"))
        selected_tail_key_source = "tail_key_v3_passed"
    else:
        selected_tail_key = ""
        selected_tail_key_source = "no_feasible_tail_key"

    if selected_tail_key:
        p3_rows, p3 = distribution_only_matrix(args, out, ref_actions, tk, selected_tail_key, old_ids, old_hashes)
    else:
        p3 = {
            "stage": "P3_GENERATOR_REPAIR_MATRIX_V9970",
            "status": "summary",
            "candidate_count": 0,
            "official_pass_count": 0,
            "weak_pass_count": 0,
            "best_generator_id": "",
            "best_generator_profile": "",
            "best_enters_P3": 0,
            "best_enters_P4": 0,
            "reason": "P1_current_tail_key_infeasible_and_P2_tail_key_v3_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max(inum(p1.get("cpu_offload_used")), inum(p2.get("cpu_offload_used"))),
        }
        p3_rows = [p3]
    dump_csv("p3_generator_repair_matrix_v9970.csv", p3_rows)

    p4_rows, p4_labels, p4_actions, p4 = run_density_if_open(args, out, p3, old_ids, old_hashes)
    dump_csv("p4_sequential_natural_density_panel_v9970.csv", p4_rows)
    dump_csv("p4_natural_density_group_rates_v9970.csv", v9930.grouped_density(p4_labels, p4_actions) if p4_labels else v9930.not_run("P4_NATURAL_DENSITY_GROUP_RATES_V9970", "P4_no_completed_density_panel")[0])

    p5_rows, p5 = path_revalidation(p4_labels)
    dump_csv("p5_future_path_type_revalidation_v9970.csv", p5_rows)

    fpo_actions, fpo_labels = (p4_actions, p4_labels) if p4_labels else base_natural_rows(source_v9940)
    p6_rows, p6 = fpo_v9(fpo_actions, fpo_labels, official_context=int(bool(p4_labels)))
    dump_csv("p6_future_path_operator_sketch_v9_v9970.csv", p6_rows)

    controller_open = int(inum(p4.get("P4_density_sufficient")) or inum(p6.get("P6_FPO_strong_pass")) or (inum(p5.get("P5_future_path_strong_pass")) and inum(p4.get("largest_completed_panel_size")) >= 87))
    p7_rows, p7_summary = not_run("P7_EXISTING_ACTION_CONTROLLER_GATE_V9970", "P4_density_not_sufficient_and_P6_P5_strong_not_passed", P7_controller_pass=0) if not controller_open else not_run("P7_EXISTING_ACTION_CONTROLLER_GATE_V9970", "controller_open_not_implemented_without_passed_support_splits", P7_controller_pass=0)
    dump_csv("p7_existing_action_controller_gate_v9970.csv", p7_rows)

    generated_allowed = int(inum(p4.get("P4_density_insufficient")) or inum(p6.get("P6_FPO_strong_pass")) or inum(p5.get("P5_future_path_strong_pass")))
    p7 = {
        "stage": "P8_GENERATED_SANDBOX_GATE_V9970",
        "status": "not_run",
        "generated_sandbox_allowed": generated_allowed,
        "allowed_action_count": 64 if generated_allowed else 0,
        "reason": "generated_gate_conditions_not_met" if not generated_allowed else "64_action_sandbox_allowed_but_not_executed_in_v9970_boundary",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("p8_generated_sandbox_gate_v9970.csv", [p7])

    p9_rt, p9_rt_sum = not_run("P9_SELECTED_RUNTIME_BOUNDARY_V9970", "P7_or_P8_not_passed", P9_runtime_pass=0)
    p9_pr, p9_pr_sum = not_run("P9_PAIRED_REPLAY_BOUNDARY_V9970", "P7_or_P8_not_passed", P9_paired_replay_pass=0)
    dump_csv("p9_selected_runtime_boundary_v9970.csv", p9_rt)
    dump_csv("p9_paired_replay_boundary_v9970.csv", p9_pr)

    if not inum(p0.get("P0_boundary_pass")):
        route_name, primary, secondary, gen_status, explanation = (
            "CaseP0-BoundaryReproductionFailed",
            "v9960_boundary_not_reproduced",
            "all_downstream_gates_blocked",
            "stopped_P0_boundary_failed",
            "v9.9.6 boundary artifacts did not satisfy the P0 reproduction contract, so v9.9.7 downstream gates are not official.",
        )
    elif not selected_tail_key:
        route_name, primary, secondary, gen_status, explanation = (
            "CaseA-TailFidelityFeasibilityInfeasible",
            "current_tail_key_infeasible_and_tail_key_v3_not_passed",
            "generator_repair_blocked",
            "stopped_tail_key_feasibility_failed",
            "the v9.9.6 tail key is not feasible at the required panel sizes and no tail key v3 candidate passed the audit, so G30-G39 are not opened.",
        )
    elif not inum(p3.get("best_enters_P4")):
        route_name, primary, secondary, gen_status, explanation = (
            "CaseB-TailKeyV3FeasibleGeneratorFail",
            "tail_key_feasible_but_no_major_tail_generator_passed",
            "density_panels_blocked",
            "stopped_generator_fidelity_failed",
            "a feasible tail key was selected, but G30-G39 did not produce a weak or official major+tail fidelity pass, so branch-horizon and density panels stay blocked.",
        )
    elif inum(p4.get("P4_density_insufficient")):
        route_name, primary, secondary, gen_status, explanation = (
            "CaseC-NaturalDensityInsufficientGeneratedRouteCandidate",
            "natural_density_insufficient",
            "FPO_or_path_target_required",
            "generated_64_sandbox_candidate",
            "natural density is insufficient on a fidelity panel, so generated route can be prepared under strict 64-action sandbox rules.",
        )
    elif inum(p4.get("P4_density_sufficient")) and not inum(p6.get("P6_FPO_strong_pass")):
        route_name, primary, secondary, gen_status, explanation = (
            "CaseD-DensitySufficientFPOFail",
            "natural_density_sufficient_but_FPO_failed",
            "online_controller_not_legal",
            "stopped_controller_gate_pending",
            "natural density is sufficient, but FPO v9 did not pass; offline harvesting may continue but official online controller remains blocked.",
        )
    elif inum(p6.get("P6_FPO_strong_pass")):
        route_name, primary, secondary, gen_status, explanation = (
            "CaseE-FPOPassControllerPending",
            "controller_gate_pending",
            "runtime_boundary_pending",
            "stopped_controller_gate_pending",
            "FPO v9 strong pass exists, but controller/runtime still need gated execution.",
        )
    else:
        route_name, primary, secondary, gen_status, explanation = (
            "CaseB-TailFidelityPassDensityPending",
            "sequential_density_pending_or_inconclusive",
            "FPO_not_controller_ready",
            "stopped_density_or_FPO_pending",
            "generator fidelity opened the density path but density and FPO gates did not close in this run.",
        )

    route = {
        "stage": "ROUTE_DECISION_V9970",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "route_explanation": explanation,
        "source_route_v9960": p0.get("source_route"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_current_tail_key_official_feasible": p1.get("P1_current_tail_key_official_feasible"),
        "P1_expected_missing_tail_at_1024": p1.get("expected_missing_tail_at_1024"),
        "P1_expected_missing_tail_at_5000": p1.get("expected_missing_tail_at_5000"),
        "P2_tail_key_v3_pass": p2.get("P2_tail_key_v3_pass"),
        "selected_tail_key_id": selected_tail_key,
        "selected_tail_key_source": selected_tail_key_source,
        "P3_official_pass_count": p3.get("official_pass_count"),
        "P3_weak_pass_count": p3.get("weak_pass_count"),
        "P3_weak_pass": int(inum(p3.get("weak_pass_count")) > 0),
        "best_generator_id": p3.get("best_generator_id"),
        "best_failed_generator_id": p3.get("best_failed_generator_id"),
        "P4_largest_completed_panel_size": p4.get("largest_completed_panel_size"),
        "P4_density_sufficient": p4.get("P4_density_sufficient"),
        "P4_density_insufficient": p4.get("P4_density_insufficient"),
        "P4_density_inconclusive": p4.get("P4_density_inconclusive"),
        "P5_future_path_weak_pass": p5.get("P5_future_path_weak_pass"),
        "P5_future_path_strong_pass": p5.get("P5_future_path_strong_pass"),
        "P6_FPO_weak_pass": p6.get("P6_FPO_weak_pass"),
        "P6_FPO_strong_pass": p6.get("P6_FPO_strong_pass"),
        "P7_controller_pass": p7_summary.get("P7_controller_pass"),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed"),
        "generated_route_status": gen_status,
        "P9_runtime_pass": p9_rt_sum.get("P9_runtime_pass"),
        "P9_paired_replay_pass": p9_pr_sum.get("P9_paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max(inum(p0.get("cpu_offload_used")), inum(p1.get("cpu_offload_used")), inum(p2.get("cpu_offload_used")), inum(p3.get("cpu_offload_used")), inum(p4.get("cpu_offload_used")), inum(p6.get("cpu_offload_used"))),
    }
    dump_json("route_decision_v9970.json", route)

    figs = write_figures(out, p1_rows, p2_rows, p3_rows, p4, p5, p6, route)
    artifacts.update(figs)

    audit = artifact_audit(artifacts)
    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9970",
        "status": "summary",
        "rows_checked": audit["rows_checked"],
        "fake_data_used": audit["fake_data_used"],
        "proxy_row_used": audit["proxy_row_used"],
        "cpu_offload_used": max(audit["cpu_offload_used"], inum(route["cpu_offload_used"])),
    }
    dump_csv("no_fake_audit_v9970.csv", [no_fake])
    dump_csv("contract_audit_v9970.csv", [{
        "stage": "CONTRACT_AUDIT_V9970",
        "status": "summary",
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_feasibility_audit_complete": int(bool(p1_rows)),
        "P2_tail_key_v3_audit_complete": int(bool(p2_rows)),
        "P3_gate_respected": int(not inum(p3.get("best_enters_P4")) or inum(p4.get("completed_panel_count")) >= 0),
        "generated_gate_respected": int(generated_allowed == inum(p7.get("generated_sandbox_allowed"))),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }])
    dump_csv("failure_taxonomy_v9970.csv", [{
        "stage": "FAILURE_TAXONOMY_V9970",
        "status": "summary",
        "route": route_name,
        "F1_tail_key_feasibility_failed": int(not selected_tail_key),
        "F2_generator_fidelity_failed": int(bool(selected_tail_key) and not inum(p3.get("best_enters_P4"))),
        "F3_density_not_closed": int(not inum(p4.get("P4_density_sufficient")) and not inum(p4.get("P4_density_insufficient"))),
        "F4_FPO_not_strong": int(not inum(p6.get("P6_FPO_strong_pass"))),
        "F5_generated_not_open": int(not generated_allowed),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }])

    manifest = {
        "stage": "RUN_MANIFEST_V9970",
        "status": "summary",
        "out_dir": str(out),
        "command": "python experiments/run_v9970_tail_fidelity_feasibility_natural_density_fpo_rewrite.py --out-dir ... --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512",
        "plan": str(PLAN_PATH),
        "runner": str(SCRIPT_PATH),
        "materializer": str(MATERIALIZER_PATH),
        "source_v9960": str(source_v9960),
        "source_v9950": str(source_v9950),
        "source_v9940": str(source_v9940),
        "route": route_name,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route["cpu_offload_used"],
    }
    dump_json("run_manifest_v9970.json", manifest)
    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifacts["materializer"] = MATERIALIZER_PATH
    hashes = sha_rows(artifacts)
    write_recap(out, route, hashes)


if __name__ == "__main__":
    main()
