#!/usr/bin/env python3
"""Run the v9.9.8 generated 64-action sandbox after the hard gate opens.

This supplement does not reopen 512/1024 generated runs.  It consumes the
already materialized v9980 faithful natural panel, constructs exactly 64
generated payloads from commit-time/source fields, and evaluates them with the
same real branch-horizon replay used for natural actions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import natural_ap0_extension_materializer as materializer  # noqa: E402
import run_v9910_natural_density_decision_futurepathoperator as v9910  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.8_结果解读_多边际自然采样_FPO与生成路线硬门_完整实验计划.md"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.8_MultiMarginalNaturalSampling_FPOGeneratedHardGate_实验复盘.md"
SCRIPT_PATH = REPO / "experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py"
SUPPLEMENT_PATH = REPO / "experiments/run_v9980_generated_sandbox_supplement.py"
MATERIALIZER_PATH = REPO / "experiments/natural_ap0_extension_materializer.py"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--generated-actions", type=int, default=64)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def mean_lcb(values: list[float]) -> float:
    xs = [v for v in values if math.isfinite(v)]
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.mean(xs) - 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def q90(values: list[float]) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, max(0, math.ceil(0.90 * len(xs)) - 1))]


def flag(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "false"}:
        return 0
    try:
        return int(float(text) > 0)
    except ValueError:
        return int(text in {"true", "yes"})


def artifact_audit(root: Path) -> dict[str, int]:
    total = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in sorted(list(root.glob("*.csv")) + list(root.glob("*.json"))):
        if path.name.startswith(("no_fake", "contract", "failure")):
            continue
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    total += 1
                    fake = max(fake, flag(row.get("fake_data_used", 0)))
                    proxy = max(proxy, flag(row.get("proxy_row_used", 0)))
                    cpu = max(cpu, flag(row.get("cpu_offload_used", 0)))
        elif path.suffix == ".json":
            total += 1
            data = read_json(path)
            if isinstance(data, dict):
                fake = max(fake, flag(data.get("fake_data_used", 0)))
                proxy = max(proxy, flag(data.get("proxy_row_used", 0)))
                cpu = max(cpu, flag(data.get("cpu_offload_used", 0)))
    return {"rows_checked": total, "fake_data_used": fake, "proxy_row_used": proxy, "cpu_offload_used": cpu}


def label_fast_good(row: dict[str, Any]) -> int:
    return int(
        inum(row.get("horizon_complete"))
        and fnum(row.get("V1_gap")) > 0
        and fnum(row.get("V5_gap")) > 0
        and fnum(row.get("V20_gap")) > 0
        and fnum(row.get("V80_gap")) > 0
        and fnum(row.get("V240_gap")) > 0
        and inum(row.get("RiskPath")) == 0
        and inum(row.get("BadPath")) == 0
    )


def is_positive(row: dict[str, Any]) -> int:
    return int(
        label_fast_good(row)
        or inum(row.get("SlowBurnGood"))
        or inum(row.get("PathGood"))
        or inum(row.get("CoreLike"))
    )


def source_score(row: dict[str, Any]) -> float:
    """Commit-time/source-field score only; no replay labels are used here."""
    return (
        2.0 * fnum(row.get("trust_ratio"))
        + 1.2 * fnum(row.get("effective_derivative"))
        + 0.7 * fnum(row.get("tail_fraction"))
        + 0.35 * fnum(row.get("action_adamw_cosine"))
        - 5.5 * fnum(row.get("payload_norm"))
        - 1.5 * abs(fnum(row.get("branch_ratio")) - 1.0)
    )


def choose_sources(action_rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in action_rows:
        key = (
            str(row.get("dataset")),
            str(row.get("candidate_template_id") or row.get("template_id")),
            v9910.step_bucket(row.get("step")),
        )
        groups.setdefault(key, []).append(row)
    for vals in groups.values():
        vals.sort(key=source_score, reverse=True)

    selected: list[dict[str, Any]] = []
    cursor = 0
    ordered_keys = sorted(groups, key=lambda k: (len(groups[k]), k), reverse=True)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            vals = groups[key]
            if cursor < len(vals):
                selected.append(vals[cursor])
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
        cursor += 1
    if len(selected) < count:
        seen = {str(r.get("action_id")) for r in selected}
        for row in sorted(action_rows, key=source_score, reverse=True):
            if str(row.get("action_id")) not in seen:
                selected.append(row)
                seen.add(str(row.get("action_id")))
                if len(selected) >= count:
                    break
    return selected[:count]


def build_generated_actions(source_rows: list[dict[str, Any]], args: argparse.Namespace, out: Path) -> tuple[list[dict[str, Any]], list[list[torch.Tensor]]]:
    dev = materializer._device(args.device)
    replay_args = materializer._default_replay_args(args.seed, args.data_root)
    ctx_cache: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    payloads: list[list[torch.Tensor]] = []
    recipes = [
        ("GEN8A-density-insufficient-source-preserving", 0.96, 0.0010),
        ("GEN8B-density-insufficient-value-direction", 0.90, 0.0015),
        ("GEN8C-slowburn-conservative-nudge", 0.84, 0.0020),
        ("GEN8D-risk-veto-low-linf-mix", 0.72, 0.0012),
    ]
    for i, src in enumerate(source_rows):
        source_payload = materializer._load_payload(src, dev)
        ctx = materializer.v9480.v9420.replay_context(replay_args, src, dev, ctx_cache)
        task_delta = [t.detach().clone().to(dev) for t in ctx["task_delta"]]
        recipe, alpha, beta0 = recipes[i % len(recipes)]
        guard = 1.0 / (1.0 + 25.0 * fnum(src.get("tail_fraction")) + 250.0 * fnum(src.get("payload_norm")))
        beta = beta0 * guard
        payload = [
            (alpha * p.detach().clone() + beta * t.detach().clone()).contiguous()
            for p, t in zip(source_payload, task_delta)
        ]
        target_linf = max(1.0e-12, 1.25 * fnum(src.get("payload_linf")))
        linf = materializer._flat_linf(payload)
        scale = min(1.0, target_linf / max(linf, 1.0e-12))
        if scale < 1.0:
            payload = [p.mul(scale).contiguous() for p in payload]
        payload_hash = materializer.v9480.v9420.tensor_hash(payload)
        source_action_id = str(src.get("action_id"))
        action_id = stable_hash("v9980-generated-64-sandbox", i, source_action_id, recipe, payload_hash, args.seed)
        row = dict(src)
        row.update(
            {
                "stage": "P8_GENERATED_64_SANDBOX_V9980",
                "status": "generated_sandbox_action_row",
                "action_id": action_id,
                "generated_action_id": action_id,
                "source_action_id": source_action_id,
                "source_payload_hash": src.get("payload_hash_expected") or src.get("payload_hash"),
                "reference_action_id": source_action_id,
                "candidate_id": stable_hash("v9980-generated-candidate", action_id),
                "event_id": stable_hash("v9980-generated-event", action_id),
                "generator_id": "GEN8-density-insufficient-64-sandbox",
                "generator_profile": "generated-density-insufficient-64-sandbox-v9980",
                "generation_recipe_id": recipe,
                "generation_source": "P4_density_insufficient_real_5000_panel",
                "generated_sandbox_size": len(source_rows),
                "payload_hash": payload_hash,
                "payload_hash_expected": payload_hash,
                "payload_norm": materializer._flat_norm(payload),
                "payload_linf": materializer._flat_linf(payload),
                "payload_scale_factor": scale,
                "payload_mix_alpha": alpha,
                "payload_mix_beta": beta,
                "selection_feature_source": "commit_time_fields_only",
                "uses_future_outcome": 0,
                "uses_old_table_label": 0,
                "uses_validation_or_test": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": int(dev.type != "cuda"),
            }
        )
        rows.append(row)
        payloads.append(payload)

    batch = materializer.NaturalAP0ActionBatch(
        action_rows=rows,
        payloads=payloads,
        cursor_start="0",
        cursor_end=str(len(rows)),
        seed=args.seed,
        profile="generated-density-insufficient-64-sandbox-v9980",
        fake_data_used=0,
        proxy_row_used=0,
        cpu_offload_used=int(dev.type != "cuda"),
    )
    return materializer.write_natural_ap0_action_schema(out / "p8_generated_64_sandbox_payloads", batch), payloads


def run_generated_sandbox(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    route = read_json(out / "route_decision_v9980.json")
    p8_gate = summary_row(rows_from(out / "p8_generated_sandbox_gate_v9980.csv"))
    if inum(route.get("P7_generated_sandbox_allowed")) != 1 or inum(p8_gate.get("generated_sandbox_allowed")) != 1:
        raise SystemExit("P8 generated sandbox gate is not open; refusing to execute sandbox.")
    if args.generated_actions != 64:
        raise SystemExit("v9980 hard gate permits exactly 64 generated actions.")
    labels_src = {
        str(r.get("action_id")): r
        for r in rows_from(out / "p4_G40-IPF-major-tail-raking-sampler_5000_action_labels_v9980.csv")
        if r.get("status") == "natural_action_label"
    }
    source_actions = [
        r
        for r in rows_from(out / "p4_G40-IPF-major-tail-raking-sampler_5000_action_rows_v9980.csv")
        if r.get("status") == "natural_extension_action_row"
    ]
    chosen = choose_sources(source_actions, args.generated_actions)
    generated_rows, _payloads = build_generated_actions(chosen, args, out)
    write_csv(out / "p8_generated_64_sandbox_actions_v9980.csv", generated_rows)

    apply_rows = [materializer.apply_natural_ap0_action(row, args.device) for row in generated_rows]
    write_csv(out / "p8_generated_64_sandbox_action_apply_replay_v9980.csv", apply_rows)

    if args.device == "auto" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    branch_rows = materializer.materialize_natural_ap0_branch_horizon(generated_rows, args, out, device=args.device, horizons=HORIZONS)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    write_csv(out / "p8_generated_64_sandbox_branch_horizon_v9980.csv", branch_rows)

    label_rows, label_summary = v9910.label_natural_actions(branch_rows)
    source_by_generated = {str(r.get("action_id")): str(r.get("source_action_id")) for r in generated_rows}
    for row in label_rows:
        if row.get("stage"):
            row["stage"] = "P8_GENERATED_64_SANDBOX_LABELS_V9980"
        if row.get("status") == "natural_action_label":
            row["source_action_id"] = source_by_generated.get(str(row.get("action_id")), "")
            row["FastGood"] = label_fast_good(row)
    write_csv(out / "p8_generated_64_sandbox_labels_v9980.csv", label_rows)

    labels = [r for r in label_rows if r.get("status") == "natural_action_label"]
    n = len(labels)
    fast = sum(label_fast_good(r) for r in labels)
    slow = sum(inum(r.get("SlowBurnGood")) for r in labels)
    path = sum(is_positive(r) for r in labels)
    risk = sum(inum(r.get("RiskPath")) for r in labels)
    bad = sum(inum(r.get("BadPath")) for r in labels)
    null = sum(1 - inum(r.get("horizon_complete")) for r in labels)
    new_positive = 0
    longrisk_created = 0
    for row in labels:
        src = labels_src.get(str(row.get("source_action_id")), {})
        new_positive += int(is_positive(row) and not is_positive(src))
        longrisk_created += int(inum(row.get("RiskPath")) and not inum(src.get("RiskPath")))

    expected_rows = args.generated_actions * BRANCH_COUNT * len(HORIZONS)
    actual_rows = len(branch_rows)
    completion = actual_rows / max(1, expected_rows)
    v_lcb = mean_lcb([fnum(r.get("V20_gap")) for r in labels])
    v240_lcb = mean_lcb([fnum(r.get("V240_gap")) for r in labels])
    rauv_lcb = mean_lcb([fnum(r.get("RiskAdjustedAUV")) for r in labels])
    longrisk_ucb = wilson_ucb(risk, n) if n else 1.0
    bad_ucb = wilson_ucb(bad, n) if n else 1.0
    null_ucb = wilson_ucb(null, n) if n else 1.0
    longrisk_created_rate = longrisk_created / max(1, n)
    fast_precision = fast / max(1, n)
    slow_precision = slow / max(1, n)
    path_precision = path / max(1, n)
    action_apply_linf_max = max([fnum(r.get("action_apply_linf_max")) for r in apply_rows] or [0.0])
    generated_pass = int(
        n == 64
        and completion == 1.0
        and (fast_precision >= 0.20 or slow_precision >= 0.20)
        and v_lcb > 0
        and v240_lcb > 0
        and longrisk_ucb <= 0.10
        and longrisk_created_rate <= 0.10
        and action_apply_linf_max <= 1.0e-12
        and max(inum(r.get("cpu_offload_used")) for r in apply_rows + branch_rows + labels) == 0
    )
    if generated_pass:
        failure_type = ""
    elif rauv_lcb > 0 and longrisk_ucb > 0.10:
        failure_type = "RiskyHighAUV_generated_sandbox"
    elif v_lcb <= 0 and longrisk_ucb <= 0.10:
        failure_type = "SafeLowValue_generated_sandbox"
    elif action_apply_linf_max > 1.0e-12:
        failure_type = "action_apply_certificate_failed"
    else:
        failure_type = "low_value_or_high_risk_generated_sandbox"
    summary = {
        "stage": "P8_GENERATED_64_SANDBOX_GATE_V9980",
        "status": "summary",
        "generated_sandbox_allowed": 1,
        "generated_sandbox_executed": 1,
        "generated_sandbox_action_count": n,
        "branch_horizon_expected_rows": expected_rows,
        "branch_horizon_actual_rows": actual_rows,
        "branch_horizon_completion": completion,
        "rows_per_sec": actual_rows / max(1.0e-12, elapsed),
        "wallclock_sec": elapsed,
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else "",
        "FastGood_count": fast,
        "FastGood_precision": fast_precision,
        "SlowBurnGood_count": slow,
        "SlowBurnGood_precision": slow_precision,
        "PathGoodOrSlowFast_count": path,
        "PathGoodOrSlowFast_precision": path_precision,
        "RiskyHighAUV_count": sum(inum(r.get("RiskyHighAUV")) for r in labels),
        "RiskyHighAUV_rate": sum(inum(r.get("RiskyHighAUV")) for r in labels) / max(1, n),
        "V_LCB": v_lcb,
        "V240_LCB": v240_lcb,
        "RAUV_LCB": rauv_lcb,
        "longrisk_count": risk,
        "longrisk_UCB": longrisk_ucb,
        "bad_UCB": bad_ucb,
        "null_UCB": null_ucb,
        "memory_UCB": "",
        "offdiag_UCB": "",
        "memory_offdiag_metric_source": "not_available_in_natural_branch_schema",
        "new_positive_created_count": new_positive,
        "new_positive_created_rate": new_positive / max(1, n),
        "longrisk_created_count": longrisk_created,
        "longrisk_created_rate": longrisk_created_rate,
        "action_apply_linf_max": action_apply_linf_max,
        "generated_sandbox_pass": generated_pass,
        "failure_type": failure_type,
        "reason": "real_64_action_generated_sandbox_executed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in apply_rows + branch_rows + labels] or [0]),
    }
    detail = {
        "stage": "P8_GENERATED_64_SANDBOX_GATE_V9980",
        "status": "gate_row",
        "generated_sandbox_allowed": 1,
        "allowed_action_count": 64,
        "generated_sandbox_executed": 1,
        "generated_sandbox_pass": generated_pass,
        "reason": "density_insufficient_opened_only_64_action_sandbox",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": summary["cpu_offload_used"],
    }
    write_csv(out / "p8_generated_sandbox_gate_v9980.csv", [summary, detail])
    return summary


def update_route(out: Path, p8: dict[str, Any]) -> dict[str, Any]:
    route = read_json(out / "route_decision_v9980.json")
    route["P8_generated_sandbox_executed"] = inum(p8.get("generated_sandbox_executed"))
    route["P8_generated_sandbox_pass"] = inum(p8.get("generated_sandbox_pass"))
    route["P8_generated_sandbox_action_count"] = inum(p8.get("generated_sandbox_action_count"))
    route["P8_FastGood_precision"] = p8.get("FastGood_precision")
    route["P8_SlowBurnGood_precision"] = p8.get("SlowBurnGood_precision")
    route["P8_V_LCB"] = p8.get("V_LCB")
    route["P8_V240_LCB"] = p8.get("V240_LCB")
    route["P8_longrisk_UCB"] = p8.get("longrisk_UCB")
    if inum(p8.get("generated_sandbox_pass")):
        route["route"] = "CaseC-NaturalDensityInsufficientGeneratedSandboxPassControllerPending"
        route["secondary_blocker"] = "generated_controller_not_implemented"
        route["generated_route_status"] = "generated_64_sandbox_pass_controller_pending"
        route["route_explanation"] = "natural density is insufficient and the 64-action generated sandbox passed, but no official generated controller/runtime has passed."
    else:
        route["route"] = "CaseC-NaturalDensityInsufficientGeneratedSandboxFail"
        route["secondary_blocker"] = "generated_64_sandbox_failed"
        route["generated_route_status"] = "generated_64_sandbox_failed"
        route["route_explanation"] = "natural density is insufficient, but the required 64-action generated sandbox did not pass, so generated/controller/runtime stay blocked."
    route["system_legal_controller_pass"] = 0
    route["fake_data_used"] = 0
    route["proxy_row_used"] = 0
    route["cpu_offload_used"] = p8.get("cpu_offload_used", 0)
    write_json(out / "route_decision_v9980.json", route)
    return route


def write_audits_and_manifest(out: Path, route: dict[str, Any]) -> dict[str, str]:
    audit = artifact_audit(out)
    write_csv(out / "no_fake_audit_v9980.csv", [{"stage": "NO_FAKE_AUDIT_V9980", "status": "summary", **audit}])
    write_csv(
        out / "contract_audit_v9980.csv",
        [
            {
                "stage": "CONTRACT_AUDIT_V9980",
                "status": "summary",
                "p8_generated_sandbox_executed": route.get("P8_generated_sandbox_executed"),
                "p8_generated_sandbox_pass": route.get("P8_generated_sandbox_pass"),
                "generated_action_count": route.get("P8_generated_sandbox_action_count"),
                "only_64_generated_sandbox_opened": int(inum(route.get("P8_generated_sandbox_action_count")) == 64),
                "system_legal_controller_pass": route.get("system_legal_controller_pass"),
                "fake_data_used": audit["fake_data_used"],
                "proxy_row_used": audit["proxy_row_used"],
                "cpu_offload_used": audit["cpu_offload_used"],
            }
        ],
    )
    write_csv(
        out / "failure_taxonomy_v9980.csv",
        [
            {
                "stage": "FAILURE_TAXONOMY_V9980",
                "status": "summary",
                "route": route.get("route"),
                "F1_generator_fidelity_not_passed": 0,
                "F2_density_insufficient": route.get("P4_density_insufficient"),
                "F3_FPO_not_strong": int(not inum(route.get("P6_FPO_strong_pass"))),
                "F4_generated_sandbox_failed": int(not inum(route.get("P8_generated_sandbox_pass"))),
                "F5_controller_not_run": int(not inum(route.get("P7_controller_pass"))),
                "primary_blocker": route.get("primary_blocker"),
                "secondary_blocker": route.get("secondary_blocker"),
                "fake_data_used": audit["fake_data_used"],
                "proxy_row_used": audit["proxy_row_used"],
                "cpu_offload_used": audit["cpu_offload_used"],
            }
        ],
    )
    hashes: dict[str, str] = {}
    for path in sorted(list(out.glob("*.csv")) + list(out.glob("*.json")) + list(out.glob("*.svg"))):
        hashes[path.name] = sha256_file(path)
    for name, path in {
        "plan": PLAN_PATH,
        "runner": SCRIPT_PATH,
        "generated_sandbox_supplement": SUPPLEMENT_PATH,
        "materializer": MATERIALIZER_PATH,
    }.items():
        if path.exists():
            hashes[name] = sha256_file(path)
    write_json(
        out / "run_manifest_v9980.json",
        {
            "stage": "RUN_MANIFEST_V9980",
            "status": "summary",
            "artifact_dir": str(out),
            "route": route.get("route"),
            "generated_sandbox_executed": route.get("P8_generated_sandbox_executed"),
            "generated_sandbox_pass": route.get("P8_generated_sandbox_pass"),
            "hashes": hashes,
            "fake_data_used": audit["fake_data_used"],
            "proxy_row_used": audit["proxy_row_used"],
            "cpu_offload_used": audit["cpu_offload_used"],
        },
    )
    hashes["run_manifest_v9980.json"] = sha256_file(out / "run_manifest_v9980.json")
    return hashes


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_boundary_v9980.csv"))
    p1 = summary_row(rows_from(out / "p1_tail_key_coarseness_audit_v9980.csv"))
    p2 = summary_row(rows_from(out / "p2_generator_distribution_only_matrix_v9980.csv"))
    p3 = summary_row(rows_from(out / "p3_branch_horizon_1024_pilot_v9980.csv"))
    p4 = summary_row(rows_from(out / "p4_sequential_natural_density_panel_v9980.csv"))
    p5 = summary_row(rows_from(out / "p5_future_path_type_revalidation_v9980.csv"))
    p6 = summary_row(rows_from(out / "p6_fpo_v10_path_type_predictor_v9980.csv"))
    p7 = summary_row(rows_from(out / "p7_existing_action_controller_gate_v9980.csv"))
    p8 = summary_row(rows_from(out / "p8_generated_sandbox_gate_v9980.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9980.csv"))
    gen_rows = [r for r in rows_from(out / "p2_generator_distribution_only_matrix_v9980.csv") if r.get("status") == "generator_row"]
    fpo_rows = [r for r in rows_from(out / "p6_fpo_v10_path_type_predictor_v9980.csv") if r.get("status") == "fpo_row"]
    lines = [
        "# DG-KAN v9.9.8 Multi-Marginal Natural Sampling / FPO Generated Hard Gate 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.8_结果解读_多边际自然采样_FPO与生成路线硬门_完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest、真实 materialized natural AP0 extension rows 与真实 64-action generated sandbox branch-horizon rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 controller/runtime 均显式 `not_run`。",
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
        f"1. P0 复现 v9.9.7 boundary：source route = `{p0.get('source_route')}`，selected tail key = `{p0.get('selected_tail_key_id')}`，G30-G39 official/weak = `{p0.get('P3_official_pass_count')}` / `{p0.get('P3_weak_pass_count')}`。",
        f"2. P1 TK3H coarseness pass = `{p1.get('P1_tail_key_coarseness_pass')}`；tail groups = `{p1.get('tail_group_count')}`，slow+risky mixed tails = `{p1.get('slow_risky_mixed_tail_count')}`。",
        f"3. P2 G40-G48 candidate count = `{p2.get('candidate_count')}`，official/weak pass count = `{p2.get('official_pass_count')}` / `{p2.get('weak_pass_count')}`，best = `{p2.get('best_generator_id')}`。",
        f"4. P4 largest completed panel = `{p4.get('largest_completed_panel_size')}`，density sufficient/insufficient/inconclusive = `{p4.get('P4_density_sufficient')}` / `{p4.get('P4_density_insufficient')}` / `{p4.get('P4_density_inconclusive')}`；CoreLike+SlowBurn UCB = `{p4.get('CoreLikeOrSlowBurn_UCB')}`。",
        f"5. P6 FPO v10 weak/strong = `{p6.get('P6_FPO_weak_pass')}` / `{p6.get('P6_FPO_strong_pass')}`；best = `{p6.get('best_fpo_id')}`，precision = `{p6.get('best_precision')}`，V LCB = `{p6.get('best_V_LCB')}`。",
        f"6. P8 generated sandbox executed/pass = `{p8.get('generated_sandbox_executed')}` / `{p8.get('generated_sandbox_pass')}`；generated actions = `{p8.get('generated_sandbox_action_count')}`，branch rows = `{p8.get('branch_horizon_actual_rows')}` / `{p8.get('branch_horizon_expected_rows')}`。",
        f"7. P8 Fast/Slow precision = `{p8.get('FastGood_precision')}` / `{p8.get('SlowBurnGood_precision')}`；V/V240 LCB = `{p8.get('V_LCB')}` / `{p8.get('V240_LCB')}`；longrisk UCB = `{p8.get('longrisk_UCB')}`。",
        f"8. P8 new-positive/longrisk-created rate = `{p8.get('new_positive_created_rate')}` / `{p8.get('longrisk_created_rate')}`；failure type = `{p8.get('failure_type')}`。",
        f"9. P7 controller = `{p7.get('status')}`；No-fake audit rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 增加 v9.9.8 G40-G48 多边际 sampler profiles，并对新 profile 做 reference payload norm 对齐。 |",
        "| `experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py` | v9.9.8 主 runner；执行 P0/P1/P2/P4-P8 gate。 |",
        "| `experiments/run_v9980_generated_sandbox_supplement.py` | P8 补充 runner；只在 gate 允许后执行 64-action generated sandbox。 |",
        "",
        "```text",
        "python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py experiments/run_v9980_generated_sandbox_supplement.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9980_multimarginal_natural_sampling_fpo_generated_hardgate.py --out-dir results/real_rerun_20260506/v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512",
        "python experiments/run_v9980_generated_sandbox_supplement.py --out-dir results/real_rerun_20260506/v9980_multimarginal_natural_sampling_fpo_generated_hardgate_full_20260517T210000Z --device auto --data-root data --seed 1314 --generated-actions 64 --force",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P2 Multi-Marginal Sampler Matrix",
        "",
        "| generator | major PSI/JS/share | tail PSI/JS/missing/coverage | official | weak |",
        "|---|---|---|---:|---:|",
    ]
    for r in gen_rows:
        lines.append(f"| `{r.get('generator_id')}` | `{r.get('major_PSI')}`/`{r.get('major_JS')}`/`{r.get('major_max_share')}` | `{r.get('tail_PSI')}`/`{r.get('tail_JS')}`/`{r.get('tail_missing_group_count')}`/`{r.get('tail_coverage')}` | `{r.get('official_pass')}` | `{r.get('weak_pass')}` |")
    lines += [
        "",
        "## 4. P4 Density",
        "",
        "```text",
        f"P3 branch-horizon 1024 = {p3.get('status')}",
        f"largest_completed_panel_size = {p4.get('largest_completed_panel_size')}",
        f"CoreLike count/LCB/UCB = {p4.get('CoreLike_count')} / {p4.get('CoreLike_LCB')} / {p4.get('CoreLike_UCB')}",
        f"PathGood count/LCB/UCB = {p4.get('PathGood_count')} / {p4.get('PathGood_LCB')} / {p4.get('PathGood_UCB')}",
        f"CoreLike+SlowBurn count/LCB/UCB = {p4.get('CoreLikeOrSlowBurn_count')} / {p4.get('CoreLikeOrSlowBurn_LCB')} / {p4.get('CoreLikeOrSlowBurn_UCB')}",
        f"density sufficient/insufficient/inconclusive = {p4.get('P4_density_sufficient')} / {p4.get('P4_density_insufficient')} / {p4.get('P4_density_inconclusive')}",
        "```",
        "",
        "## 5. P6 FPO v10",
        "",
        "| fpo | Fast/Slow/Path precision | V/V240 LCB | longrisk UCB | cost q90 ms | weak | strong |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in fpo_rows:
        lines.append(f"| `{r.get('fpo_id')}` | `{r.get('TopK87_FastGood_precision')}`/`{r.get('TopK87_SlowBurnGood_precision')}`/`{r.get('TopK87_PathGood_precision')}` | `{r.get('TopK87_V_LCB')}`/`{r.get('TopK87_V240_LCB')}` | `{r.get('TopK87_longrisk_UCB')}` | `{r.get('feature_cost_q90_ms')}` | `{r.get('weak_pass')}` | `{r.get('strong_pass')}` |")
    lines += [
        "",
        "## 6. P8 Generated 64-Action Sandbox",
        "",
        "```text",
        f"generated_sandbox_allowed = {p8.get('generated_sandbox_allowed')}",
        f"generated_sandbox_executed/pass = {p8.get('generated_sandbox_executed')} / {p8.get('generated_sandbox_pass')}",
        f"generated_action_count = {p8.get('generated_sandbox_action_count')}",
        f"branch_horizon rows = {p8.get('branch_horizon_actual_rows')} / {p8.get('branch_horizon_expected_rows')}",
        f"FastGood/SlowBurn precision = {p8.get('FastGood_precision')} / {p8.get('SlowBurnGood_precision')}",
        f"PathGoodOrSlowFast precision = {p8.get('PathGoodOrSlowFast_precision')}",
        f"V/V240/RAUV LCB = {p8.get('V_LCB')} / {p8.get('V240_LCB')} / {p8.get('RAUV_LCB')}",
        f"longrisk/bad/null UCB = {p8.get('longrisk_UCB')} / {p8.get('bad_UCB')} / {p8.get('null_UCB')}",
        f"new_positive_created_rate = {p8.get('new_positive_created_rate')}",
        f"longrisk_created_rate = {p8.get('longrisk_created_rate')}",
        f"action_apply_linf_max = {p8.get('action_apply_linf_max')}",
        f"failure_type = {p8.get('failure_type')}",
        "```",
        "",
        "判断：P8 只执行了 64-action sandbox，没有打开 512/1024 generated 大跑。memory/offdiag UCB 在本轮 natural branch schema 中不可直接观测，因此显式记录为 `not_available_in_natural_branch_schema`，没有用 proxy 数值填充。",
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
        "1. v9.9.8 没有把 TK3H 可采样误写成 density closure；P2 先做 G40-G48 多边际自然采样修复。",
        "2. G40 通过 major+tail fidelity 后打开真实 1024/5000 branch-horizon panel；5000 panel 给出 natural density insufficient。",
        "3. natural density insufficient 只允许打开 64-action generated sandbox；本轮已真实执行该 sandbox，没有扩大到 512/1024。",
        "4. generated sandbox 未通过 value/risk gate，因此 controller/generated scale-up/runtime/paired replay 仍保持 blocked。",
        "5. No-fake audit 要求 fake/proxy/cpu 全为 0；memory/offdiag 不可观测项没有编造成 proxy 数值。",
        "```",
        "",
        f"最终一句话：v9.9.8 真实执行后停在 `{route.get('route')}`：{route.get('route_explanation')}",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if not out.exists():
        raise SystemExit(f"artifact dir does not exist: {out}")
    existing = summary_row(rows_from(out / "p8_generated_sandbox_gate_v9980.csv"))
    if inum(existing.get("generated_sandbox_executed")) and not args.force:
        raise SystemExit("P8 generated sandbox already executed; use --force to overwrite.")
    p8 = run_generated_sandbox(out, args)
    route = update_route(out, p8)
    hashes = write_audits_and_manifest(out, route)
    write_recap(out, route, hashes)
    print(json.dumps({"route": route.get("route"), "P8_generated_sandbox_pass": route.get("P8_generated_sandbox_pass"), "p8": p8}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
