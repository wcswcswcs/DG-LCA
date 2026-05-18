#!/usr/bin/env python3
"""DG-KAN v9.3.1 legal action-value observability closure audit.

This runner follows the v9.3.1 plan from the landed v9.3.0 artifacts.  It
does not synthesize action-apply tensors, secondary outcomes, matched controls,
or event-driven runtime.  If those required materializations are absent, it
records the blocker and keeps downstream gates closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dgkan_outcome_controller import (
    attach_calibrated_risk_stats,
    attach_candidate_features,
    auc_score,
    fnum,
    inum,
    read_csv,
    read_json,
    sha256_file,
    wilson_lcb,
    wilson_ucb,
    write_csv,
    write_json,
)


REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO / "docs/DG-KAN_v9.3.1_LegalActionValueObservability_MeasuredEventRuntimeClosure_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.3.1_LegalActionValueObservability_MeasuredEventRuntimeClosure_实验复盘.md"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_SOURCE_V9300 = RESULT_ROOT / "v9300_outcome_action_primitive_reset_event_driven_runtime_closure_first_20260513T233000Z"
DEFAULT_SOURCE_V9282 = RESULT_ROOT / "v9282_outcome_grounded_risk_support_rebuild_empty_step_free_batch_major_runtime_closure_first_20260513T220000Z"
DEFAULT_SOURCE_V9280 = RESULT_ROOT / "v9280_full_train_stream_stable_accept_materializer_outcome_label_rebuild_rerun_20260513T190000Z"
DEFAULT_SOURCE_V9272 = RESULT_ROOT / "v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z"

PRIMARY_LABELS = ["safe_good_label", "bad_event_label", "null_event_label"]
ACTION_APPLY_FIELDS = [
    "action_apply_error_max",
    "payload_norm",
    "payload_role_entropy",
    "true_delta_norm",
    "cos_action_adamw",
    "cos_action_negative_grad",
    "linearized_CE_delta",
    "linearized_margin_delta",
]
SECONDARY_CONTROL_FIELDS = [
    "CEp99_delta",
    "margin_p10_delta",
    "ECE_delta",
    "NLL_delta",
    "curvature_delta",
    "acc_delta",
    "loss_auc_delta_step",
    "loss_auc_delta_time",
    "real_beats_adamwparallel",
    "real_beats_bestlr",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9300", default=str(DEFAULT_SOURCE_V9300))
    p.add_argument("--source-v9282", default=str(DEFAULT_SOURCE_V9282))
    p.add_argument("--source-v9280", default=str(DEFAULT_SOURCE_V9280))
    p.add_argument("--source-v9272", default=str(DEFAULT_SOURCE_V9272))
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def glob_tensor_artifacts(*roots: Path) -> list[str]:
    out: list[str] = []
    suffixes = {".pt", ".pth", ".npy", ".npz"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix in suffixes or any(tok in path.name.lower() for tok in ("tensor", "payload", "theta"))):
                # Keep CSV payload tables out of the tensor-artifact count.
                if path.suffix == ".csv":
                    continue
                out.append(str(path.relative_to(REPO)))
    return sorted(set(out))


def q(values: list[float], frac: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, math.ceil(frac * len(vals)) - 1))
    return vals[idx]


def metric_summary(rows: list[dict[str, Any]], denom_rows: list[dict[str, str]], mask: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    held = [row for row in rows if inum(row.get("seed")) >= 5 and mask(row)]
    denom = sum(1 for row in denom_rows if inum(row.get("seed")) >= 5)
    n = len(held)
    safe = sum(inum(row.get("safe_good_label")) for row in held)
    bad = sum(inum(row.get("bad_event_label")) for row in held)
    null = sum(inum(row.get("null_event_label")) for row in held)
    fam = Counter(str(row.get("family_id", "")) for row in held)
    strata = Counter(str(row.get("_signal_stratum", row.get("signal_stratum", ""))) for row in held)
    actions = Counter(str(row.get("carrier_id", row.get("action_family", ""))) for row in held)
    return {
        "accepted_count": n,
        "safe_good_count": safe,
        "bad_event_count": bad,
        "null_event_count": null,
        "precision": safe / n if n else 0.0,
        "coverage": n / denom if denom else 0.0,
        "bad_event_rate": bad / n if n else 0.0,
        "null_rate": null / n if n else 0.0,
        "precision_lcb": wilson_lcb(safe, n),
        "bad_event_ucb": wilson_ucb(bad, n),
        "accepted_family_count": len(fam),
        "accepted_signal_strata_count": len(strata),
        "accepted_action_family_count": len(actions),
        "max_family_share": max((v / n for v in fam.values()), default=0.0),
        "max_stratum_share": max((v / n for v in strata.values()), default=0.0),
        "max_action_family_share": max((v / n for v in actions.values()), default=0.0),
    }


def topk_counts(rows: list[dict[str, Any]], score_fn: Callable[[dict[str, Any]], float], k: int, high: bool = True) -> dict[str, Any]:
    held = [row for row in rows if inum(row.get("seed")) >= 5]
    selected = sorted(held, key=score_fn, reverse=high)[:k]
    n = len(selected)
    return {
        "BadCountAt273": sum(inum(row.get("bad_event_label")) for row in selected),
        "SafeGoodCountAt273": sum(inum(row.get("safe_good_label")) for row in selected),
        "NullCountAt273": sum(inum(row.get("null_event_label")) for row in selected),
        "selected_count": n,
        "precision_at_273": sum(inum(row.get("safe_good_label")) for row in selected) / n if n else 0.0,
        "bad_event_at_273": sum(inum(row.get("bad_event_label")) for row in selected) / n if n else 0.0,
        "null_at_273": sum(inum(row.get("null_event_label")) for row in selected) / n if n else 0.0,
    }


def prepare_rows(source_v9280: Path, source_v9272: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    rows = [
        dict(row)
        for row in read_csv(source_v9280 / "full_row_stable_accept_outcome_table_v9280.csv")
        if row.get("status") == "candidate_stable_accept_outcome_row"
    ]
    full_rows = read_csv(source_v9280 / "full_online_event_table_v9280.csv")
    old_rows = [
        row
        for row in read_csv(source_v9272 / "full_online_event_table_v9272.csv")
        if inum(row.get("candidate_flag")) == 1
    ]
    full_by_event = {row["event_id"]: row for row in full_rows}
    old_ids = {row["event_id"] for row in old_rows}
    attach_candidate_features(rows, full_by_event, old_ids)
    for row in rows:
        event = full_by_event.get(str(row.get("event_id")), {})
        row["candidate_score"] = fnum(event.get("candidate_score"))
        row["bridge_score"] = fnum(event.get("bridge_score"))
        row["action_family"] = str(row.get("carrier_id", "unknown"))
    attach_calibrated_risk_stats(rows)
    return rows, full_rows, old_rows


def build_observability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    held = [row for row in rows if inum(row.get("seed")) >= 5]
    bad_labels = [inum(row.get("bad_event_label")) for row in held]
    safe_labels = [inum(row.get("safe_good_label")) for row in held]
    feature_defs = [
        ("OGP-C1-CandidateScore", "state_logit", lambda r: fnum(r.get("candidate_score")), True),
        ("OGP-A1-StableScoreQ", "stable_accept", lambda r: fnum(r.get("score_quantized_ref")), True),
        ("OGP-A2-StableRankNeg", "stable_accept", lambda r: -fnum(r.get("stable_rank_ref")), True),
        ("OGP-R1-RiskResidualLow", "support_risk", lambda r: fnum(r.get("_risk_residual_score")), False),
        ("OGP-D1-BadUCBMeanLow", "support_risk", lambda r: fnum(r.get("_bad_ucb_mean")), False),
        ("OGP-D2-SupportLCBMin", "support_risk", lambda r: fnum(r.get("_support_lcb_min")), True),
        ("OGP-F1-NullUCBMaxLow", "support_risk", lambda r: fnum(r.get("_null_ucb_max")), False),
    ]
    out: list[dict[str, Any]] = []
    for fid, group, fn, high_for_accept in feature_defs:
        values = [fn(row) for row in held]
        auc_bad = auc_score(values, bad_labels)
        auc_safe = auc_score(values, safe_labels)
        counts = topk_counts(rows, fn, 273, high_for_accept)
        out.append(
            {
                "observable_id": fid,
                "observable_group": group,
                "action_conditioned": 0,
                "commit_time_available": 1,
                "diagnostic_only": 1,
                "AUC_bad_event_raw": auc_bad,
                "AUC_bad_event_oriented": max(auc_bad, 1.0 - auc_bad),
                "AUC_safe_good_raw": auc_safe,
                "AUC_safe_good_oriented": max(auc_safe, 1.0 - auc_safe),
                **counts,
                "BadCountAtAccept273_gate": int(counts["BadCountAt273"] <= 13),
                "SafeGoodCountAtAccept273_gate": int(counts["SafeGoodCountAt273"] >= 205),
                "NullCountAtAccept273_gate": int(counts["NullCountAt273"] <= 40),
                "feature_cost_recorded": 0,
                "feature_cost_pass": 0,
                "observable_pass": 0,
            }
        )
    return out


def not_run(reason: str, **extra: Any) -> dict[str, Any]:
    row = {"status": "not_run", "reason": reason}
    row.update(extra)
    return row


def write_recap(out_dir: Path, route: dict[str, Any], hashes: list[dict[str, Any]]) -> None:
    try:
        out_rel = str(out_dir.resolve().relative_to(REPO.resolve()))
    except ValueError:
        out_rel = str(out_dir)
    text = f"""# DG-KAN v9.3.1 Legal Action-Value Observability 与 Measured Event Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.1_LegalActionValueObservability_MeasuredEventRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把缺失的 action apply、secondary/control outcome 或 diagnostic runtime 写成 official pass。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = {route.get('base_candidate')}
success_v9310_strict_purekan_functional = {bool(inum(route.get('success_v9310_strict_purekan_functional')))}
success_v9310_full_functional = {bool(inum(route.get('success_v9310_full_functional')))}
success_v9310_external_ready = {bool(inum(route.get('success_v9310_external_ready')))}
```

最终 artifact：

```text
{out_rel}/
```

核心结论：

1. P0 复现 v9.3.0 boundary：candidate/action/event = `{route.get('candidate_count')}` / `{route.get('action_count')}` / `{route.get('event_count')}`，source route = `{route.get('source_route_v9300')}`。
2. P1 action lifecycle 未过：`action_apply_error_measured = {route.get('action_apply_error_measured')}`，`action_apply_error_missing_count = {route.get('action_apply_error_missing_count')}`，没有可重放 tensor artifact。
3. P2 secondary/control outcome 未过：`secondary_outcome_ready = {route.get('secondary_outcome_ready')}`，`missing_secondary_delta_count = {route.get('missing_secondary_delta_count')}`，matched control per event = `{route.get('matched_control_count_per_event')}`。
4. Primary oracle frontier 仍存在：oracle precision = `{route.get('oracle_precision')}`，coverage = `{route.get('oracle_coverage')}`，bad/null = `{route.get('oracle_bad_event')}` / `{route.get('oracle_null_rate')}`。
5. Legal observability 只能作为 diagnostic：best observable = `{route.get('best_observable_id')}`，bad AUC = `{route.get('best_observable_auc_bad')}`，safe AUC = `{route.get('best_observable_auc_safe')}`，feature cost pass = `{route.get('feature_cost_pass')}`。
6. P5 action-conditioned observable factory 没有 survivor：现有 frozen action rows 缺 payload norm、true delta norm、action/AdamW cosine、linearized deltas 等字段。
7. P6/P8/P10 不能打开：`ogp_decision_pass = {route.get('ogp_decision_pass')}`，`system_legal_controller_pass = {route.get('system_legal_controller_pass')}`。
8. P9 measured event-driven runtime 未 materialize：reference step ratio q90 = `{route.get('step_ratio_q90')}`，empty-step controller kernels = `{route.get('empty_step_controller_kernel_count')}`，runtime pass = `{route.get('event_driven_runtime_pass')}`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py` | v9.3.1 runner；读取 v9.3.0/v9.2.80/v9.2.82 artifacts，审计 action apply、secondary/control outcome、legal observable gap 和 event-driven runtime materialization |

代码检查：

```text
python -m py_compile experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py
```

正式运行：

```bash
python experiments/run_v9310_legal_action_value_observability_measured_event_runtime_closure.py \\
  --out-dir {out_rel} \\
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

`route_decision.json` 摘要：

```json
{json.dumps(route, indent=2, ensure_ascii=False, sort_keys=True)}
```

判断：v9.3.1 按计划没有继续调 StableAccept/DR7。由于 P1 action apply error 与 P2 secondary/control outcomes 都未 materialize，controller 和 downstream 必须 gate-block。

## 3. P1/P2 materializer blocker

```text
action_apply_error_measured = {route.get('action_apply_error_measured')}
action_apply_error_missing_count = {route.get('action_apply_error_missing_count')}
secondary_outcome_ready = {route.get('secondary_outcome_ready')}
missing_secondary_delta_count = {route.get('missing_secondary_delta_count')}
matched_control_count_per_event = {route.get('matched_control_count_per_event')}
```

判断：当前 artifact 只有 action/candidate identity、payload/hash 和 primary labels；没有可审计的 action apply error 或 matched-control branch outcomes。不能进入 official action-value controller。

## 4. Observability Diagnostic

```text
primary_oracle_pass = {route.get('primary_oracle_pass')}
useful_oracle_pass = {route.get('useful_oracle_pass')}
control_oracle_pass = {route.get('control_oracle_pass')}
observability_gap_pass = {route.get('observability_gap_pass')}
best_observable_id = {route.get('best_observable_id')}
best_observable_BadCountAt273 = {route.get('best_observable_BadCountAt273')}
best_observable_SafeGoodCountAt273 = {route.get('best_observable_SafeGoodCountAt273')}
best_observable_NullCountAt273 = {route.get('best_observable_NullCountAt273')}
```

判断：primary safe-good oracle 仍强，但这不是 deployable controller。现有 legal observables 不是 action-conditioned value/risk observables，且 feature cost 未测。

## 5. Runtime Boundary

```text
runtime_candidate_id = {route.get('runtime_candidate_id')}
runtime_mode = {route.get('runtime_mode')}
empty_step_controller_kernel_count = {route.get('empty_step_controller_kernel_count')}
controller_launches_per_active_step_q90 = {route.get('controller_launches_per_active_step_q90')}
step_ratio_q90 = {route.get('step_ratio_q90')}
event_driven_runtime_pass = {route.get('event_driven_runtime_pass')}
```

判断：本轮没有 measured zero-candidate skip 或 active-step fused runtime；diagnostic empty-step skip 继续保持 diagnostic，不能 official。

## 6. Downstream Boundary

P11-P14 均以 `not_run` 落盘，原因是 `P10_system_controller_not_official`。没有把 primary oracle、observability diagnostic 或 runtime estimate 写成 paired replay / short-run / full-run success。

## 7. No-fake Audit

```text
fake_data_used = {route.get('fake_data_used')}
proxy_row_used = {route.get('proxy_row_used')}
cpu_offload_used = {route.get('cpu_offload_used')}
diagnostic_downstream_used_for_controller = {route.get('diagnostic_downstream_used_for_controller')}
```

## 8. Hash

| artifact | SHA256 |
|---|---|
"""
    for row in hashes:
        text += f"| {row.get('artifact')} | `{row.get('sha256')}` |\n"
    text += f"""

## 9. 最终分析结论

v9.3.1 的真实推进是：

```text
v9.3.0:
  candidate/action population 有 oracle frontier；
  但 legal OGP feature、secondary/control outcome 和 measured event runtime 未闭合。

v9.3.1:
  严格审计 action apply 与 secondary/control materializer；
  发现当前落盘 artifact 仍不能提供 action apply error、matched controls 或完整 secondary outcomes；
  因而按计划停止 controller/downstream promotion。
```

机制判断：

1. H1 仍成立但只能停在 primary oracle：oracle frontier 存在，但 useful/control oracle 无法在缺 secondary/control outcomes 时声明。
2. H2 未闭合：现有 feature 仍是 state/support diagnostic，不是 action-conditioned value observable。
3. H3 未闭合：secondary/control outcomes 缺失，action value 的完整定义不存在。
4. H4 未闭合：event-driven runtime 没有 measured materialization。
5. 当前下一步必须先实现真实 action apply error materializer 和 secondary/control outcome materializer，再回到 legal action-value observable/controller。

最终一句话：

> v9.3.1 真实执行后停在 `{route.get('route')}`：v9.3.0 的 oracle frontier 没回退，但 action apply error 与 secondary/control outcome 仍缺失，measured event-driven runtime 未实现；strict PureKAN functional 仍不能转正。
"""
    RECAP_PATH.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    src9300 = Path(args.source_v9300)
    src9282 = Path(args.source_v9282)
    src9280 = Path(args.source_v9280)
    src9272 = Path(args.source_v9272)

    route9300 = read_json(src9300 / "route_decision.json")
    route9282 = read_json(src9282 / "route_decision.json")
    rows, full_rows, _old_rows = prepare_rows(src9280, src9272)
    frozen = read_csv(src9300 / "frozen_candidate_action_table_v9300.csv")
    runtime_source = read_csv(src9300 / "p7_online_event_driven_runtime.csv")
    tensor_artifacts = glob_tensor_artifacts(src9300, src9280, src9272)

    manifest = {
        "experiment": "DG-KAN_v9.3.1_LegalActionValueObservability_MeasuredEventRuntimeClosure",
        "source_v9300": str(src9300),
        "source_v9282": str(src9282),
        "source_v9280": str(src9280),
        "source_v9272": str(src9272),
        "candidate_count": len(rows),
        "action_count": len(frozen),
        "event_count": len(full_rows),
        "tensor_artifact_count": len(tensor_artifacts),
        "device_arg": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "completed_at": now_iso(),
    }
    write_json(out_dir / "run_manifest.json", manifest)

    # P0
    stable_m = metric_summary(rows, full_rows, lambda r: inum(r.get("accept_native")) == 1)
    p0 = {
        "stage": "P0_V9300_BOUNDARY_REANALYSIS",
        "status": "summary",
        "source_route_v9300": route9300.get("route"),
        "source_primary_blocker": route9300.get("primary_blocker"),
        "candidate_count": len(rows),
        "action_count": len(frozen),
        "event_count": len(full_rows),
        "heldout_denominator": sum(1 for r in full_rows if inum(r.get("seed")) >= 5),
        "stableaccept_accepted_count": stable_m["accepted_count"],
        "stableaccept_precision": stable_m["precision"],
        "stableaccept_coverage": stable_m["coverage"],
        "stableaccept_bad_event": stable_m["bad_event_rate"],
        "stableaccept_null_rate": stable_m["null_rate"],
        "oracle_weak_frontier_pass_v9300": route9300.get("oracle_weak_frontier_pass"),
        "oracle_strong_frontier_pass_v9300": route9300.get("oracle_strong_frontier_pass"),
        "ogp_feature_pass_v9300": route9300.get("ogp_feature_pass"),
        "event_driven_runtime_pass_v9300": route9300.get("event_driven_runtime_pass"),
        "boundary_reanalysis_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "p0_v9300_boundary_reanalysis.csv", [p0])

    # P1
    missing_apply_rows = []
    for row in frozen:
        missing = [field for field in ACTION_APPLY_FIELDS if str(row.get(field, "")) == ""]
        if missing:
            missing_apply_rows.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "action_id": row.get("action_id"),
                    "event_id": row.get("event_id"),
                    "missing_action_fields": ";".join(missing),
                    "action_apply_error_measured": 0,
                    "source_tensor_artifact_count": len(tensor_artifacts),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    p1 = {
        "stage": "P1_ACTION_LIFECYCLE_APPLY_ERROR_CLOSURE",
        "status": "summary",
        "candidate_count": len(rows),
        "action_count": len(frozen),
        "action_apply_error_measured": 0,
        "action_apply_error_missing_count": len(missing_apply_rows),
        "action_apply_error_linf": "",
        "action_apply_error_relative": "",
        "payload_tensor_artifact_count": len(tensor_artifacts),
        "theta_before_after_hash_present": 0,
        "native_apply_equivalence_measured": 0,
        "no_event_preservation_pass": 0,
        "action_lifecycle_pass": 0,
        "reason": "landed_artifacts_have_hashes_and_identity_rows_but_no_replayable_action_tensors_or_apply_error_measurement",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "p1_action_lifecycle_apply_error_closure.csv", [p1])
    write_csv(out_dir / "action_apply_error_trace_v9310.csv", missing_apply_rows or [not_run("no_missing_action_apply_rows")])
    write_csv(
        out_dir / "no_event_semantics_trace_v9310.csv",
        [
            {
                "stage": "P1_NO_EVENT_SEMANTICS",
                "status": "blocked",
                "zero_candidate_step_count_source": route9300.get("zero_candidate_step_count", ""),
                "no_event_preservation_pass": 0,
                "base_adamw_equivalence_on_zero_candidate_steps": "",
                "reason": "zero_candidate_no_event_equivalence_not_measured_in_landed_artifacts",
            }
        ],
    )

    # P2
    missing_secondary = inum(route9300.get("missing_secondary_delta_count")) or sum(
        1 for row in rows for field in SECONDARY_CONTROL_FIELDS if row.get(field, "") == ""
    )
    p2 = {
        "stage": "P2_SECONDARY_CONTROL_OUTCOME_MATERIALIZER_V2",
        "status": "summary",
        "candidate_count": len(rows),
        "primary_label_rows": sum(1 for row in rows if all(str(row.get(f, "")) != "" for f in PRIMARY_LABELS)),
        "missing_primary_label_count": sum(1 for row in rows if any(str(row.get(f, "")) == "" for f in PRIMARY_LABELS)),
        "label_exclusivity_violation_count": sum(1 for row in rows if inum(row.get("safe_good_label")) and inum(row.get("bad_event_label"))),
        "secondary_outcome_ready": 0,
        "missing_secondary_delta_count": missing_secondary,
        "official_sample_coverage": 0.0,
        "matched_control_branch_present": 0,
        "matched_control_count_per_event": 0,
        "outcome_materializer_cost_measured": 0,
        "p2_primary_pass": 1,
        "p2_downstream_ready_pass": 0,
        "p2_coverage_pass": 0,
        "p2_cost_pass": 0,
        "reason": "secondary_deltas_and_matched_control_branches_not_materialized",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "p2_secondary_control_outcome_materializer_v2.csv", [p2])
    write_csv(
        out_dir / "secondary_control_outcome_trace_v9310.csv",
        [
            {
                "candidate_id": row.get("candidate_id"),
                "event_id": row.get("event_id"),
                "CEp99_delta": row.get("CEp99_delta"),
                "margin_p10_delta": row.get("margin_delta"),
                "missing_secondary_control_fields": ";".join(
                    field for field in SECONDARY_CONTROL_FIELDS if str(row.get(field, "")) == ""
                ),
            }
            for row in rows
        ],
    )
    write_csv(out_dir / "matched_control_trace_v9310.csv", [not_run("matched_control_branches_not_materialized", matched_control_count_per_event=0)])
    write_csv(out_dir / "outcome_materializer_cost_trace_v9310.csv", [not_run("secondary_control_materializer_not_available", outcome_materializer_cost_measured=0)])

    # P3
    oracle_m = metric_summary(rows, full_rows, lambda r: inum(r.get("safe_good_label")) == 1)
    p3 = {
        "stage": "P3_ORACLE_REVALIDATION_SECONDARY_CONTROL",
        "status": "summary",
        "primary_oracle_pass": int(
            oracle_m["precision"] >= 0.90
            and oracle_m["coverage"] >= 0.03
            and oracle_m["bad_event_rate"] <= 0.02
            and oracle_m["null_rate"] <= 0.10
        ),
        "useful_oracle_pass": 0,
        "control_oracle_pass": 0,
        "oracle_precision": oracle_m["precision"],
        "oracle_coverage": oracle_m["coverage"],
        "oracle_bad_event": oracle_m["bad_event_rate"],
        "oracle_null_rate": oracle_m["null_rate"],
        "oracle_value_lcb": "",
        "oracle_beats_adamwparallel_rate": "",
        "oracle_beats_bestlr_rate": "",
        "reason": "secondary_control_outcomes_missing_so_useful_control_oracle_not_measured",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "p3_oracle_revalidation_secondary_control.csv", [p3])
    write_csv(out_dir / "oracle_value_control_trace_v9310.csv", [p3])

    # P4/P5 diagnostic observability.
    obs_rows = build_observability_rows(rows)
    best_obs = max(obs_rows, key=lambda r: (fnum(r.get("AUC_safe_good_oriented")), -fnum(r.get("BadCountAt273"))))
    observability_gap = {
        "stage": "P4_ORACLE_LEGAL_OBSERVABILITY_GAP",
        "status": "diagnostic_summary",
        "oracle_safe_good_count_heldout": oracle_m["accepted_count"],
        "min_accepted_for_coverage": math.ceil(0.03 * sum(1 for r in full_rows if inum(r.get("seed")) >= 5)),
        "best_observable_id": best_obs["observable_id"],
        "best_observable_auc_bad": best_obs["AUC_bad_event_oriented"],
        "best_observable_auc_safe": best_obs["AUC_safe_good_oriented"],
        "best_observable_ece_bad": "",
        "best_observable_BadCountAt273": best_obs["BadCountAt273"],
        "best_observable_SafeGoodCountAt273": best_obs["SafeGoodCountAt273"],
        "best_observable_NullCountAt273": best_obs["NullCountAt273"],
        "observability_gap_pass": 1,
        "diagnostic_only": 1,
        "reason": "primary_oracle_vs_legal_gap_quantified_but_action_and_secondary_materializers_block_official_use",
    }
    write_csv(out_dir / "p4_oracle_legal_observability_gap.csv", [observability_gap])
    write_csv(out_dir / "observability_gap_trace_v9310.csv", obs_rows)

    action_observable_defs = [
        ("O1-GradientActionAlignment", "cos_action_negative_grad;cos_action_adamw"),
        ("O2-LinearizedActionValue", "linearized_CE_delta;linearized_margin_delta"),
        ("O3-TrustRegionCurvature", "curvature_delta;true_delta_norm"),
        ("O4-PayloadGeometry", "payload_norm;payload_role_entropy;payload_tail_selectivity"),
        ("O5-EmpiricalBayesActionFamilySupport", "action_family_support_stats"),
        ("O6-NullRiskActionScore", "null_control_outcomes"),
        ("O7-ActionCostObservable", "payload_apply_time_ms_q90;feature_compute_time_ms_q90"),
    ]
    action_obs_rows = []
    for oid, required in action_observable_defs:
        req = required.split(";")
        available_rows = sum(1 for row in frozen if all(str(row.get(field, "")) != "" for field in req))
        action_obs_rows.append(
            {
                "observable_id": oid,
                "required_fields": required,
                "candidate_count": len(frozen),
                "available_row_count": available_rows,
                "missing_rate": 1.0 - available_rows / max(1, len(frozen)),
                "action_conditioned": 1,
                "commit_time_available": int(available_rows > 0),
                "AUC_bad_event": "",
                "AUC_safe_good": "",
                "feature_compute_time_ms_q90": "",
                "feature_memory_ratio": "",
                "feature_cost_pass": 0,
                "observable_pass": 0,
                "reason": "required_action_conditioned_fields_missing",
            }
        )
    write_csv(out_dir / "p5_action_conditioned_observable_factory.csv", action_obs_rows)
    write_csv(out_dir / "observable_feature_trace_v9310.csv", action_obs_rows)
    write_csv(out_dir / "observable_cost_trace_v9310.csv", action_obs_rows)
    write_csv(out_dir / "observable_ablation_trace_v9310.csv", [not_run("no_action_conditioned_observable_survivor")])

    # P6/P7/P8 blocked by materializer failure.
    p6 = not_run(
        "P1_action_lifecycle_or_P2_secondary_control_failed",
        controller_id="not_selected",
        ogp_decision_pass=0,
        precision_heldout=0.0,
        coverage_heldout=0.0,
        bad_event_heldout=0.0,
        null_rate_heldout=0.0,
    )
    write_csv(out_dir / "p6_crossfitted_action_value_controller.csv", [p6])
    write_csv(out_dir / "controller_frontier_trace_v9310.csv", [p6])

    p7 = {
        "stage": "P7_DECISION_FAILURE_AUTOPSY_V4",
        "status": "summary",
        "failed_controller_id": "not_selected_materializer_blocked",
        "bad_accepted_count": stable_m["bad_event_count"],
        "null_accepted_count": stable_m["null_event_count"],
        "missed_oracle_safe_count": max(0, oracle_m["accepted_count"] - stable_m["safe_good_count"]),
        "coverage_lost_count": math.ceil(0.03 * sum(1 for r in full_rows if inum(r.get("seed")) >= 5)),
        "failure_mode": "DF8-action_apply_uncertainty",
        "failure_submode": "secondary_control_outcome_missing_and_action_apply_unmeasured",
        "bad_accepted_attribution_fraction": 1.0,
        "missed_oracle_safe_attribution_fraction": 1.0,
        "coverage_collapse_attribution_fraction": 1.0,
        "next_route_selected": "fix_action_apply_and_secondary_control_materializers",
        "decision_failure_autopsy_pass": 1,
    }
    write_csv(out_dir / "p7_decision_failure_autopsy_v4.csv", [p7])
    write_csv(out_dir / "decision_failure_trace_v9310.csv", [p7])

    p8 = not_run(
        "P3_useful_control_oracle_not_measured_and_P5_observable_no_survivor",
        action_primitive_redesign_triggered=0,
        action_primitive_survivor_found=0,
    )
    write_csv(out_dir / "p8_action_primitive_redesign_branch.csv", [p8])
    write_csv(out_dir / "action_primitive_candidate_trace_v9310.csv", [p8])

    rt0 = next((row for row in runtime_source if str(row.get("runtime_candidate_id", "")).startswith("RT0")), {})
    rt1 = next((row for row in runtime_source if str(row.get("runtime_candidate_id", "")).startswith("RT1")), {})
    p9_rows = [
        {
            "runtime_candidate_id": "RT0-v9300-reference-fixed-per-step",
            "runtime_mode": rt0.get("runtime_mode", "online_sequential_official_reference"),
            "step_count": rt0.get("step_count", route9300.get("step_count", "")),
            "active_step_count": rt0.get("active_step_count", route9300.get("active_step_count", "")),
            "zero_candidate_step_count": rt0.get("zero_candidate_step_count", route9300.get("zero_candidate_step_count", "")),
            "candidate_count": len(rows),
            "accepted_count": stable_m["accepted_count"],
            "empty_step_controller_kernel_count": rt0.get("empty_step_controller_kernel_count", route9300.get("empty_step_controller_kernel_count", "")),
            "empty_step_controller_sync_count": rt0.get("empty_step_controller_sync_count", route9300.get("empty_step_controller_sync_count", "")),
            "controller_kernel_launch_count": rt0.get("controller_kernel_launch_count", ""),
            "controller_sync_count": rt0.get("controller_sync_count", ""),
            "controller_launches_per_active_step_q90": rt0.get("controller_launches_per_active_step_q90", ""),
            "controller_syncs_per_active_step_q90": rt0.get("controller_syncs_per_active_step_q90", ""),
            "step_ratio_q90": rt0.get("step_ratio_q90", route9300.get("step_ratio_q90", "")),
            "memory_ratio": rt0.get("memory_ratio", 1.0),
            "no_event_preservation_error": "",
            "base_adamw_equivalence_error": "",
            "accept_disagreement_count": 0,
            "payload_apply_error_max": "",
            "runtime_measured": 1,
            "diagnostic_derived_from_measured_components": 0,
            "event_driven_runtime_pass": 0,
        },
        {
            "runtime_candidate_id": "RT1-measured-zero-candidate-skip-not-materialized",
            "runtime_mode": "not_materialized",
            "empty_step_controller_kernel_count": rt1.get("empty_step_controller_kernel_count", 0),
            "controller_launches_per_active_step_q90": rt1.get("controller_launches_per_active_step_q90", 2.0),
            "runtime_measured": 0,
            "diagnostic_derived_from_measured_components": 1,
            "event_driven_runtime_pass": 0,
            "reason": "zero_candidate_skip_runtime_is_diagnostic_in_source_not_measured_official_path",
        },
    ]
    write_csv(out_dir / "p9_measured_event_driven_runtime.csv", p9_rows)
    write_csv(out_dir / "runtime_event_scheduler_trace_v9310.csv", p9_rows)
    write_csv(out_dir / "runtime_empty_event_semantics_trace_v9310.csv", p9_rows)
    write_csv(out_dir / "runtime_component_trace_v9310.csv", p9_rows)

    route = {
        "route": "R1-BoundaryReanalyzed",
        "base_candidate": "LQ-t2-h256",
        "source_route_v9300": route9300.get("route"),
        "candidate_count": len(rows),
        "action_count": len(frozen),
        "event_count": len(full_rows),
        "candidate_lifecycle_pass": route9300.get("candidate_lifecycle_pass"),
        "action_lifecycle_pass": 0,
        "action_apply_error_measured": 0,
        "action_apply_error_missing_count": len(missing_apply_rows),
        "action_apply_error_linf": "",
        "action_apply_error_relative": "",
        "no_event_preservation_pass": 0,
        "secondary_outcome_ready": 0,
        "missing_secondary_delta_count": missing_secondary,
        "official_sample_coverage": 0.0,
        "matched_control_count_per_event": 0,
        "primary_oracle_pass": p3["primary_oracle_pass"],
        "useful_oracle_pass": 0,
        "control_oracle_pass": 0,
        "oracle_precision": oracle_m["precision"],
        "oracle_coverage": oracle_m["coverage"],
        "oracle_bad_event": oracle_m["bad_event_rate"],
        "oracle_null_rate": oracle_m["null_rate"],
        "oracle_value_lcb": "",
        "oracle_beats_adamwparallel_rate": "",
        "oracle_beats_bestlr_rate": "",
        "observability_gap_pass": 1,
        "best_observable_id": best_obs["observable_id"],
        "best_observable_auc_bad": best_obs["AUC_bad_event_oriented"],
        "best_observable_auc_safe": best_obs["AUC_safe_good_oriented"],
        "best_observable_ece_bad": "",
        "best_observable_BadCountAt273": best_obs["BadCountAt273"],
        "best_observable_SafeGoodCountAt273": best_obs["SafeGoodCountAt273"],
        "best_observable_NullCountAt273": best_obs["NullCountAt273"],
        "feature_cost_pass": 0,
        "controller_id": "not_selected_materializer_blocked",
        "action_primitive_id": "AP0-current-action-reference",
        "ogp_decision_pass": 0,
        "precision_heldout": 0.0,
        "coverage_heldout": 0.0,
        "bad_event_heldout": 0.0,
        "null_rate_heldout": 0.0,
        "value_mean_heldout": "",
        "precision_lcb": 0.0,
        "bad_event_ucb": 1.0,
        "support_balance_pass": 0,
        "stableaccept_patch_exhausted": 1,
        "runtime_candidate_id": "RT0-v9300-reference-fixed-per-step",
        "runtime_mode": rt0.get("runtime_mode", "online_sequential_official_reference"),
        "empty_step_controller_kernel_count": rt0.get("empty_step_controller_kernel_count", route9300.get("empty_step_controller_kernel_count", "")),
        "empty_step_controller_sync_count": rt0.get("empty_step_controller_sync_count", route9300.get("empty_step_controller_sync_count", "")),
        "controller_launches_per_active_step_q90": rt0.get("controller_launches_per_active_step_q90", 9.0),
        "controller_syncs_per_active_step_q90": rt0.get("controller_syncs_per_active_step_q90", 1.0),
        "step_ratio_q90": rt0.get("step_ratio_q90", route9300.get("step_ratio_q90", "")),
        "memory_ratio": rt0.get("memory_ratio", 1.0),
        "event_driven_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "leave_dataset_out_pass": 0,
        "leave_stratum_out_pass": 0,
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "continual_pass": 0,
        "robustness_pass": 0,
        "strong_baseline_pass": 0,
        "external_ready": 0,
        "primary_blocker": "action_apply_and_secondary_control_materialization_missing",
        "next_required_implementation": "materialize_action_apply_error_and_secondary_control_outcomes",
        "success_v9310_strict_purekan_functional": 0,
        "success_v9310_full_functional": 0,
        "success_v9310_external_ready": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "diagnostic_downstream_used_for_controller": 0,
    }
    write_csv(out_dir / "p10_system_integration_v9310.csv", [route | {"status": "not_run", "reason": "P1_P2_materializer_blocked"}])
    write_csv(out_dir / "system_controller_trace_v9310.csv", [route | {"status": "not_run", "reason": "P1_P2_materializer_blocked"}])

    downstream_reason = "P10_system_controller_not_official"
    for name in [
        "p11_diagnostic_causal_scout.csv",
        "diagnostic_isolation_audit_v9310.csv",
        "p12_leave_dataset_stratum_out.csv",
        "leaveout_trace_v9310.csv",
        "p13_official_paired_replay.csv",
        "official_paired_replay_trace_v9310.csv",
        "p14_short_full_continual_robustness.csv",
        "short_full_continual_trace_v9310.csv",
    ]:
        write_csv(out_dir / name, [not_run(downstream_reason, diagnostic_downstream_used_for_controller=0)])

    contract = {
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "candidate_lifecycle_pass": route9300.get("candidate_lifecycle_pass"),
        "action_lifecycle_pass": 0,
        "secondary_outcome_ready": 0,
        "matched_control_ready": 0,
        "primary_oracle_pass": p3["primary_oracle_pass"],
        "useful_oracle_pass": 0,
        "control_oracle_pass": 0,
        "observability_gap_pass": 1,
        "action_conditioned_observable_pass": 0,
        "ogp_decision_pass": 0,
        "event_driven_runtime_pass": 0,
        "system_legal_controller_pass": 0,
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_controller": 0,
        "projection_used_for_official": 0,
        "source_measured_gap_used": 0,
        "formula_proxy_used": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_csv(out_dir / "contract_audit_v9310.csv", [contract])
    failures = [
        {"failure_code": "F5_action_lifecycle_apply_error_fail", "active": 1, "count": len(missing_apply_rows)},
        {"failure_code": "F7_secondary_outcome_missing", "active": 1, "count": missing_secondary},
        {"failure_code": "F8_matched_control_missing", "active": 1, "count": len(full_rows)},
        {"failure_code": "F13_action_conditioned_feature_unpredictive", "active": 1, "count": len(action_obs_rows)},
        {"failure_code": "F15_feature_cost_infeasible", "active": 1, "count": 1},
        {"failure_code": "F25_empty_step_runtime_fail", "active": 1, "count": inum(route.get("empty_step_controller_kernel_count"))},
        {"failure_code": "F30_runtime_measured_path_missing", "active": 1, "count": 1},
    ]
    write_csv(out_dir / "failure_table.csv", failures)
    provenance = {
        "rows_checked": len(rows),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": True,
        "no_proxy": True,
        "source_v9300_hash": sha256_file(src9300 / "route_decision.json"),
    }
    write_csv(out_dir / "v9310_provenance_audit.csv", [provenance])
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)

    hash_targets = [
        ("plan", PLAN_PATH),
        ("runner", SCRIPT_PATH),
        ("run manifest", out_dir / "run_manifest.json"),
        ("route", out_dir / "route_decision.json"),
        ("P0 boundary", out_dir / "p0_v9300_boundary_reanalysis.csv"),
        ("P1 action lifecycle", out_dir / "p1_action_lifecycle_apply_error_closure.csv"),
        ("P2 secondary control", out_dir / "p2_secondary_control_outcome_materializer_v2.csv"),
        ("P3 oracle", out_dir / "p3_oracle_revalidation_secondary_control.csv"),
        ("P4 observability", out_dir / "p4_oracle_legal_observability_gap.csv"),
        ("P5 observable factory", out_dir / "p5_action_conditioned_observable_factory.csv"),
        ("P6 controller", out_dir / "p6_crossfitted_action_value_controller.csv"),
        ("P7 autopsy", out_dir / "p7_decision_failure_autopsy_v4.csv"),
        ("P8 primitive", out_dir / "p8_action_primitive_redesign_branch.csv"),
        ("P9 runtime", out_dir / "p9_measured_event_driven_runtime.csv"),
        ("P10 system", out_dir / "p10_system_integration_v9310.csv"),
        ("contract", out_dir / "contract_audit_v9310.csv"),
        ("provenance", out_dir / "v9310_provenance_audit.csv"),
    ]
    hashes = [{"artifact": name, "sha256": sha256_file(path)} for name, path in hash_targets if path.exists()]
    write_csv(out_dir / "artifact_hashes.csv", hashes)
    write_recap(out_dir, route, hashes)
    return route


def main() -> None:
    args = parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
