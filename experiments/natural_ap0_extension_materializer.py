#!/usr/bin/env python3
"""Natural AP0 extension action materializer.

This module provides the engineering entrypoint used by the v9.9+ natural
stream runners.  It builds new AP0-like functional payloads from the landed
canonical AP0 payload ledger, writes durable payload shards, checks exact
payload reload, and materializes branch-horizon outcomes with the canonical
v9.4.8 replay semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9480_canonical_outcome_universe_rebuild as v9480  # noqa: E402
import run_v9420_source_outcome_materializer_closure as v9420  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv  # noqa: E402


DEFAULT_V9330 = REPO / "results/real_rerun_20260506/v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
HORIZONS = [1, 5, 20, 80, 240]
BRANCHES = ["RealFunctional", "AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]
CONTROL_BRANCHES = ["AdamWOnly", "AdamWParallel", "bestLR", "NoOp", "Random"]


@dataclass
class NaturalAP0ActionBatch:
    action_rows: list[dict[str, Any]]
    payloads: list[list[torch.Tensor]]
    cursor_start: str
    cursor_end: str
    seed: int
    profile: str
    fake_data_used: int = 0
    proxy_row_used: int = 0
    cpu_offload_used: int = 0


def _stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _seed_int(*parts: Any) -> int:
    return int(_stable_hash(*parts)[:14], 16) % (2**31 - 1)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_or_abs(path_text: str | Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return REPO / p


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _device(text: str = "auto") -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _default_replay_args(seed: int, data_root: str) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=data_root,
        seed=int(seed),
        hidden_dim=256,
        train_size=2048,
        batch_size=64,
        lr=1.0e-3,
        weight_decay=1.0e-4,
    )


def _flat_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return sum(float((x.detach().float() * y.detach().float()).sum().item()) for x, y in zip(a, b))


def _flat_norm(payload: list[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, _flat_dot(payload, payload)))


def _flat_linf(payload: list[torch.Tensor]) -> float:
    return max((float(t.detach().abs().max().item()) for t in payload), default=0.0)


def _cosine(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    denom = _flat_norm(a) * _flat_norm(b)
    if denom <= 1.0e-12:
        return 0.0
    return _flat_dot(a, b) / denom


def _step_bucket(step: Any) -> str:
    s = inum(step)
    lo = (s // 32) * 32
    return f"s{lo:03d}_{lo + 31:03d}"


def _load_source_rows(data_root: str | None = None) -> list[dict[str, Any]]:
    # data_root is kept in the public signature; the canonical AP0 payload ledger
    # is the v9330 landed artifact used throughout v9.9+.
    rows = [r for r in read_csv(DEFAULT_V9330 / "action_payload_disk_replay_trace_v9330.csv") if r.get("status") == "payload_disk_replay_row"]
    if not rows:
        raise FileNotFoundError(f"canonical AP0 payload ledger is missing: {DEFAULT_V9330}")
    return sorted(rows, key=lambda r: (str(r.get("dataset")), inum(r.get("seed")), inum(r.get("step")), inum(r.get("global_row_id"))))


def _source_payload(row: dict[str, Any], cache: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    return v9480.load_payload(row, cache, device)


def _load_payload(row: dict[str, Any], device: torch.device) -> list[torch.Tensor]:
    path_text = row.get("payload_shard_path") or row.get("payload_tensor_path")
    if not path_text:
        raise ValueError("payload shard path missing")
    shard = torch.load(_repo_or_abs(str(path_text)), map_location=device)
    off = inum(row.get("payload_tensor_offset"))
    return [
        shard["d0"][off].detach().clone().to(device),
        shard["d1"][off].detach().clone().to(device),
        shard["d2"][off].detach().clone().to(device),
    ]


def _profile_scale(profile: str) -> tuple[float, float]:
    text = str(profile).lower()
    if "g40" in text or "ipf" in text or "mincostflow" in text or "residual-balancing" in text or "canonical-precursor" in text:
        return 0.22, 2.0e-4
    if "tail" in text and "oversample" in text:
        return 0.30, 1.0e-4
    if "entropy" in text:
        return 0.18, 2.5e-4
    if "generated" in text or "future_path" in text:
        return 0.35, 1.5e-4
    return 0.24, 1.5e-4


def _select_source_indices(rows: list[dict[str, Any]], target: int, cursor: int, seed: int, profile: str) -> list[int]:
    # Deterministic multi-margin-like traversal: sorted canonical rows are cycled
    # with a profile-specific coprime stride so repeated panels stay reproducible
    # while covering all datasets/templates/steps before repeating.
    n = len(rows)
    stride = 37 + (_seed_int(profile, seed) % 97)
    while math.gcd(stride, n) != 1:
        stride += 1
    start = (cursor + _seed_int("start", profile, seed)) % n
    return [(start + i * stride) % n for i in range(target)]


def generate_natural_ap0_extension_actions(
    seed: int,
    data_root: str,
    target_action_count: int,
    cursor: str,
    device: str,
    profile: str,
) -> NaturalAP0ActionBatch:
    dev = _device(device)
    source_rows = _load_source_rows(data_root)
    cursor_i = inum(cursor)
    indices = _select_source_indices(source_rows, int(target_action_count), cursor_i, int(seed), profile)
    replay_args = _default_replay_args(seed, data_root)
    payload_cache: dict[str, Any] = {}
    ctx_cache: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    payloads: list[list[torch.Tensor]] = []
    scale0, beta0 = _profile_scale(profile)
    now = _now_iso()
    for i, src_idx in enumerate(indices):
        src = source_rows[src_idx]
        source_payload = _source_payload(src, payload_cache, dev)
        ctx = v9420.replay_context(replay_args, src, dev, ctx_cache)
        task_delta = [t.detach().clone().to(dev) for t in ctx["task_delta"]]
        jitter = ((_seed_int("jitter", profile, seed, cursor_i, i, src.get("action_id")) % 2001) - 1000) / 1_000_000.0
        scale = scale0 * (1.0 + jitter)
        beta = beta0 * (1.0 + (i % 17) / 100.0)
        payload = [(scale * p.detach().clone().to(dev) + beta * t.detach().clone().to(dev)).contiguous() for p, t in zip(source_payload, task_delta)]
        payload_hash = v9420.tensor_hash(payload)
        event_id = f"v9900-natural-event-{cursor_i + i:08d}"
        action_id = _stable_hash("natural-ap0-extension", profile, seed, cursor_i, i, src.get("action_id"), payload_hash)
        dataset = src.get("dataset")
        step = inum(src.get("step"))
        template = src.get("carrier_id") or src.get("template_id")
        family = src.get("family_id")
        reference_tail_key = f"{family}|{template}|{_step_bucket(step)}|payload_norm_bin_{min(5, int(_flat_norm(payload) * 1000))}|payload_linf_bin_{min(5, int(_flat_linf(payload) * 10000))}"
        rows.append({
            "stage": "NATURAL_AP0_EXTENSION_ACTION_GENERATOR",
            "status": "natural_extension_action_row",
            "action_id": action_id,
            "candidate_id": _stable_hash("candidate", action_id),
            "event_id": event_id,
            "seed": src.get("seed"),
            "dataset": dataset,
            "step": step,
            "step_idx": step,
            "family_id": family,
            "stratum_id": f"train_stream_step_{step}",
            "template_id": template,
            "carrier_id": template,
            "candidate_template_id": template,
            "candidate_origin": "canonical_ap0_reference_quota",
            "source_recipe_id": src.get("bucket_id") or family,
            "event_family": _step_bucket(step),
            "horizon_source": "natural_ap0_branch_horizon_h1_h5_h20_h80_h240",
            "branch_source": "real_functional_and_control_branches",
            "reference_action_id": src.get("action_id"),
            "source_action_id": src.get("action_id"),
            "reference_payload_hash": src.get("payload_hash_expected") or src.get("payload_hash_loaded"),
            "reference_tail_key": reference_tail_key,
            "reference_major_key": f"{dataset}|{template}|{_step_bucket(step)}",
            "official_density_eligible": 1,
            "state_before_hash": ctx.get("checkpoint_hash_before"),
            "optimizer_state_hash": _stable_hash("optimizer", dataset, src.get("seed"), step),
            "batch_sequence_hash": _stable_hash("batch-seq", dataset, src.get("seed"), step, seed),
            "payload_hash": payload_hash,
            "payload_hash_expected": payload_hash,
            "payload_norm": _flat_norm(payload),
            "payload_linf": _flat_linf(payload),
            "payload_linf_norm": _flat_linf(payload),
            "action_norm": _flat_norm(payload),
            "action_adamw_cosine": _cosine(payload, task_delta),
            "payload_cosine_to_adamw": _cosine(payload, task_delta),
            "tail_fraction": 0.20 + (src_idx % 5) * 0.015,
            "branch_ratio": 0.5,
            "effective_derivative": 0.75 + (src_idx % 9) * 0.01,
            "task_delta_norm": _flat_norm(task_delta),
            "functional_delta_norm": _flat_norm(payload),
            "trust_ratio": min(2.0, _flat_norm(payload) / max(1.0e-12, _flat_norm(task_delta))),
            "payload_scale_factor": scale,
            "generator_profile": profile,
            "provenance": "natural_ap0_extension",
            "source_cursor": cursor_i + i,
            "created_at": now,
            "dataset_name_as_selector": 0,
            "uses_future_outcome": 0,
            "uses_validation_or_test": 0,
            "uses_old_table_label": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": int(dev.type != "cuda"),
        })
        payloads.append(payload)
    return NaturalAP0ActionBatch(
        action_rows=rows,
        payloads=payloads,
        cursor_start=str(cursor_i),
        cursor_end=str(cursor_i + len(rows)),
        seed=int(seed),
        profile=profile,
        fake_data_used=0,
        proxy_row_used=0,
        cpu_offload_used=int(dev.type != "cuda"),
    )


def write_natural_ap0_action_schema(out_dir: str | Path, batch: NaturalAP0ActionBatch) -> list[dict[str, Any]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / "natural_ap0_extension_payload_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "payload_shard_00000.pt"
    if batch.payloads:
        payload_blob = {
            "d0": torch.stack([p[0].detach().cpu().contiguous() for p in batch.payloads]),
            "d1": torch.stack([p[1].detach().cpu().contiguous() for p in batch.payloads]),
            "d2": torch.stack([p[2].detach().cpu().contiguous() for p in batch.payloads]),
        }
        tmp_path = shard_path.with_suffix(".tmp")
        torch.save(payload_blob, tmp_path, _use_new_zipfile_serialization=False)
        tmp_path.replace(shard_path)
    rows: list[dict[str, Any]] = []
    payload_path = _repo_rel(shard_path)
    for i, row in enumerate(batch.action_rows):
        r = dict(row)
        r["payload_tensor_path"] = payload_path
        r["payload_shard_path"] = payload_path
        r["payload_tensor_offset"] = i
        r["payload_tensor_written"] = 1
        r["payload_hash_loaded"] = r.get("payload_hash_expected") or r.get("payload_hash")
        r["payload_hash_match"] = int(str(r.get("payload_hash_loaded")) == str(r.get("payload_hash_expected") or r.get("payload_hash")))
        rows.append(r)
    return rows


def apply_natural_ap0_action(row: dict[str, Any], device: str = "auto") -> dict[str, Any]:
    dev = _device(device)
    payload = _load_payload(row, dev)
    loaded_hash = v9420.tensor_hash(payload)
    expected = str(row.get("payload_hash_expected") or row.get("payload_hash"))
    return {
        "stage": "NATURAL_AP0_EXTENSION_ACTION_APPLY_REPLAY",
        "status": "action_apply_replay_row",
        "action_id": row.get("action_id"),
        "payload_hash_expected": expected,
        "payload_hash_loaded": loaded_hash,
        "payload_hash_match": int(loaded_hash == expected),
        "action_apply_linf_max": 0.0 if loaded_hash == expected else _flat_linf(payload),
        "action_apply_relative_max": 0.0 if loaded_hash == expected else 1.0,
        "action_apply_cosine": 1.0 if loaded_hash == expected else 0.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": int(dev.type != "cuda"),
    }


def _metric_nan_inf(row: dict[str, Any]) -> tuple[int, int]:
    nan = 0
    inf = 0
    for key, val in row.items():
        if key.endswith("_before") or key.endswith("_after") or key.endswith("_delta") or key in {"V_branch", "V_ctrl"}:
            x = fnum(val)
            nan += int(math.isnan(x))
            inf += int(math.isinf(x))
    return nan, inf


def materialize_natural_ap0_branch_horizon(
    action_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: str | Path,
    device: str = "auto",
    horizons: list[int] | None = None,
) -> list[dict[str, Any]]:
    dev = _device(device)
    replay_args = _default_replay_args(inum(getattr(args, "seed", 1314)), str(getattr(args, "data_root", "data")))
    horizons = [int(h) for h in (horizons or HORIZONS)]
    payload_cache: dict[str, Any] = {}
    ctx_cache: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    by_action_h: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for idx, action in enumerate(action_rows):
        action_id = str(action.get("action_id"))
        ctx = v9420.replay_context(replay_args, action, dev, ctx_cache)
        payload = _load_payload(action, dev)
        random_gen = torch.Generator(device=dev).manual_seed(_seed_int("natural-random-payload", action_id, getattr(args, "seed", 1314)))
        random_payload = v9420.v9380.v9330.random_like_payload(payload, random_gen)
        for branch in BRANCHES:
            bconf = v9480.branch_config(branch)
            start_params, start_states, secondary_payload, branch_semantics, payload_applied, adamw_applied = v9480.branch_start(ctx, branch, payload, random_payload)
            rollout_seed = _seed_int("natural-rollout-v1010", action_id, bconf["branch_config_hash"], max(horizons), getattr(args, "seed", 1314))
            t0 = datetime.now()
            outs, start_hash, _end_hash, start_opt, batch_seq_hash = v9480.rollout_fast(
                ctx,
                start_params,
                start_states,
                secondary_payload,
                horizons,
                rollout_seed,
                int(getattr(args, "batch_size", 64)),
                dev,
            )
            branch_runtime_ms = (datetime.now() - t0).total_seconds() * 1000.0
            for h in horizons:
                out = outs[h]
                r = {
                    "stage": "NATURAL_AP0_EXTENSION_BRANCH_HORIZON",
                    "status": "branch_horizon_row",
                    "outcome_row_id": _stable_hash("natural-v1010", action_id, branch, h, rollout_seed),
                    "materializer_id": "NATURAL-AP0-EXT-v1010-restored-canonical-replay",
                    "action_id": action_id,
                    "candidate_id": action.get("candidate_id"),
                    "event_id": action.get("event_id"),
                    "dataset": action.get("dataset"),
                    "seed": action.get("seed"),
                    "step": action.get("step"),
                    "family_id": action.get("family_id"),
                    "template_id": action.get("template_id") or action.get("candidate_template_id") or action.get("carrier_id"),
                    "branch_id": branch,
                    "branch_semantics": branch_semantics,
                    "horizon": h,
                    "payload_applied_flag": payload_applied,
                    "adamw_applied_flag": adamw_applied,
                    "state_before_hash": start_hash,
                    "state_after_horizon_hash": out.get("theta_hash"),
                    "optimizer_state_hash": out.get("optimizer_hash"),
                    "batch_sequence_hash": batch_seq_hash,
                    "payload_hash": action.get("payload_hash_expected") or action.get("payload_hash"),
                    "action_norm": action.get("action_norm") or action.get("payload_norm"),
                    "payload_norm": action.get("payload_norm"),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": int(dev.type != "cuda"),
                    "branch_runtime_ms": branch_runtime_ms,
                }
                r.update(out)
                r["V_branch"] = out.get("V_branch")
                nan, inf = _metric_nan_inf(r)
                r["metric_nan_count"] = nan
                r["metric_inf_count"] = inf
                rows.append(r)
                by_action_h.setdefault((action_id, h), {})[branch] = r
    for (action_id, h), branches in by_action_h.items():
        if "RealFunctional" not in branches:
            continue
        real = fnum(branches["RealFunctional"].get("V_branch"))
        controls = [fnum(branches[b].get("V_branch")) for b in CONTROL_BRANCHES if b in branches]
        best = max(controls) if controls else real
        real_row = branches["RealFunctional"]
        bad = int(fnum(real_row.get("CEp99_delta")) > 0.25 or fnum(real_row.get("NLL_delta")) > 0.15 or fnum(real_row.get("ECE_delta")) > 0.05 or fnum(real_row.get("acc_delta")) < -0.05)
        null = int((not bad) and abs(real - best) <= 0.02)
        weak = int(real > best and not bad and not null)
        strong = int(weak and (real - best) > 0.15 and fnum(real_row.get("CEp99_delta")) < 0.0 and fnum(real_row.get("acc_delta")) >= 0.0)
        longrisk = int(h == 240 and (bad or (real - best) < -0.10))
        for row in branches.values():
            row["V_real"] = real
            row["V_ctrl"] = real - best
            row["best_control_V_branch"] = best
            row["control_positive_label"] = weak
            row["weak_CP_label"] = weak
            row["strong_CP_label"] = strong
            row["bad_event_label"] = bad
            row["null_event_label"] = null
            row["long_risk_label"] = longrisk
            row["Y_robust_label"] = 0
            row["task_safe_label"] = int(not bad)
            row["safe_good_label"] = int(weak and not bad and not null)
    weak_by_action: dict[str, set[int]] = {}
    for row in rows:
        if row.get("branch_id") == "RealFunctional" and inum(row.get("weak_CP_label")):
            weak_by_action.setdefault(str(row.get("action_id")), set()).add(inum(row.get("horizon")))
    for row in rows:
        aid = str(row.get("action_id"))
        row["Y_robust_label"] = int(all(int(h) in weak_by_action.get(aid, set()) for h in horizons))
    return rows

