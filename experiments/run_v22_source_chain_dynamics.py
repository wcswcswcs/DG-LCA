#!/usr/bin/env python3
"""v22 source-chain dynamics runner.

The training loop is reused from the audited v21.01 source-retention runner;
this script adds v22 source-chain classification and v22 artifact names.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.source_chain import RETENTION_EPS, classify_source_chain, finite_float, source_chain_row  # noqa: E402
from experiments import run_v21_01_common as v2101_common  # noqa: E402
from experiments import run_v21_01_source_retention as v2101_source  # noqa: E402
from experiments.run_v22_common import (  # noqa: E402
    PYTHON,
    V22_EXEC_DOC,
    append_exec,
    ensure_out,
    int_flag,
    read_rows,
    write_json,
    write_rows,
)


DEFAULT_TARGET_SPECS = ",".join(
    [
        "F3-T1-loss-cotangent-target",
        "F3-T5-random-matched-target",
        "F9-TCTRL-stable-random-target",
        "F10-T7-b1-cross-split-consensus-transfer",
        "F25-loss-warm-to-b1-consensus-migration",
        "F30-gain-gated-loss-warm-b1-consensus",
        "F33-loss-warm-to-b1-consensus-b3-null",
        "F35-loss-warm-to-view-consistent-loss",
        "F37-loss-warm-to-lowbank-loss-b3-null",
        "F39-loss-warm-to-gated-lowbank-loss-b3-null",
        "F68-adamw-boundary-to-gated-lowbank-b3-null",
        "F69-adamw-boundary-lowbank-anchor-antiwashout",
        "CTRL-SGD",
        "CTRL-AdamW",
        "CTRL-RandomMatchedNorm",
        "CTRL-NoOpMatchedOverhead",
    ]
)
DEFAULT_KAN_SPECS = ",".join(
    [
        "KSW1-basis-estimate-readout-commit",
        "KSW2-lowdegree-lowfreq-source-bank",
        "F6-KSW2-density-smallstep-alt50",
        "F8-KSW2-earlyboost-alt25",
        "F3-T1-loss-cotangent-target",
        "F3-T5-random-matched-target",
        "F9-TCTRL-stable-random-target",
        "CTRL-SGD",
        "CTRL-AdamW",
        "CTRL-RandomMatchedNorm",
        "CTRL-NoOpMatchedOverhead",
    ]
)
DEFAULT_MLP_SPECS = ",".join(
    [
        "MLP-F1-M2-strong-source",
        "MLP-F2-M15-weak-stable",
        "MLP-F11-dual-timescale-retention-warm1200",
        "MLP-F23-source-vs-sgd-lookahead-gate",
        "MLP-F40-adamw-boundary-to-momentum-source",
        "MLP-F41-adamw-boundary-to-dual-timescale-source",
        "MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source",
        "MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source",
        "MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source",
        "MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source",
        "MLP-F47-trainloss-gated-dual-timescale-tiny-late-source",
        "MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source",
        "MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source",
        "MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source",
        "MLP-F51-trainloss-late-hold-recovery-source",
        "MLP-F52-trainloss-late-lookahead-floor-source",
        "MLP-F53-trainloss-terminal-lookahead-floor-source",
        "MLP-F54-trainloss-early-terminal-lookahead-floor-source",
        "MLP-F57-adamw-boundary-dual-timescale-antiwashout-source",
        "MLP-F58-adamw-boundary-dual-timescale-source-anchor",
        "MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry",
        "MLP-F60-adamw-boundary-dual-timescale-readout-channel",
        "MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel",
        "MLP-F62-trainloss-terminal-projected-lookahead-floor-source",
        "MLP-F63-trainloss-terminal-consensus-lookahead-floor-source",
        "MLP-F64-trainloss-terminal-selector-lookahead-floor-source",
        "MLP-F70-trainloss-terminal-positive-lookahead-floor-source",
        "MLP-F71-trainloss-terminal-h3200-checkpoint-reentry",
        "MLP-F72-trainloss-terminal-hard-split-source",
        "MLP-F73-trainloss-terminal-adamw-lookahead",
        "MLP-F74-trainloss-terminal-optimizer-selector",
        "MLP-F65-adamw-split-fisher-residual-source",
        "CTRL-SGD",
        "CTRL-AdamW",
        "CTRL-RandomMatchedNorm",
        "CTRL-NoOpMatchedOverhead",
    ]
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--v21-scope", choices=["target", "kan", "mlp", "f0"], default="target")
    p.add_argument("--carriers", default="D-CHE,D-FOU")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--fu-lr", type=float, default=0.0001)
    p.add_argument("--alt-period", type=int, default=50)
    p.add_argument("--init-seed-offset", type=int, default=0)
    p.add_argument("--run-label", default="v2200_source_chain_smoke")
    p.add_argument("--spec-ids", default="")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge-only", action="store_true")
    return p


def default_specs(scope: str) -> str:
    if scope == "mlp":
        return DEFAULT_MLP_SPECS
    if scope == "kan":
        return DEFAULT_KAN_SPECS
    if scope == "target":
        return DEFAULT_TARGET_SPECS
    return ""


def call_v21_source(args: argparse.Namespace) -> None:
    out_dir = ensure_out(args.out_dir)
    v2101_common.V2101_EXEC_DOC = V22_EXEC_DOC
    scope = {"mlp": "mlp", "kan": "kan", "target": "target", "f0": "f0"}[args.v21_scope]
    argv = [
        "run_v21_01_source_retention.py",
        "--out-dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--data-root",
        str(args.data_root),
        "--scope",
        scope,
        "--carriers",
        str(args.carriers),
        "--datasets",
        str(args.datasets),
        "--seeds",
        str(args.seeds),
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--batch-size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--fu-lr",
        str(args.fu_lr),
        "--alt-period",
        str(args.alt_period),
        "--init-seed-offset",
        str(args.init_seed_offset),
        "--run-label",
        str(args.run_label),
        "--shard-count",
        str(args.shard_count),
        "--shard-index",
        str(args.shard_index),
    ]
    specs = args.spec_ids or default_specs(args.v21_scope)
    if specs:
        argv.extend(["--spec-ids", specs])
    if args.merge_only:
        argv.append("--merge-only")
    command = f"{PYTHON} experiments/run_v21_01_source_retention.py " + " ".join(argv[1:])
    append_exec(out_dir, command, status="started", note=f"delegates to v21.01 source runner; scope={scope}; specs={specs or 'default'}")
    old = sys.argv[:]
    try:
        sys.argv = argv
        v2101_source.main()
    finally:
        sys.argv = old
    append_exec(out_dir, command, status="completed", note="v21.01 raw/source-retention artifacts generated in v22 official dir")


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key)) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize_v22(out_dir: Path) -> None:
    matrix = read_rows(out_dir / "v21_01_source_retention_matrix.csv")
    enriched = []
    for row in matrix:
        item = dict(row)
        item["v22_id"] = item.get("v21_id", "")
        item.update(source_chain_row(item, prefix="source_h"))
        enriched.append(item)
    write_rows(out_dir / "v22_source_chain_matrix.csv", enriched)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[(str(row.get("carrier", "")), str(row.get("basis_repair_variant", "")), str(row.get("v22_id", "")))].append(row)
    summary = []
    for (carrier, variant, v22_id), rows in sorted(groups.items()):
        h = {f"h{step}": mean(rows, f"source_h{step}") for step in (100, 400, 800, 1600, 2400, 3200, 4800, 6400)}
        decision = classify_source_chain(h["h100"], h["h400"], h["h800"], h["h1600"], h["h3200"], h["h4800"])
        h4800_ok = int(decision.continuous_retention_chain and math.isfinite(h["h4800"]) and h["h4800"] >= RETENTION_EPS)
        summary.append(
            {
                "carrier": carrier,
                "variant": variant,
                "v22_id": v22_id,
                "rows": len(rows),
                "blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
                **h,
                "early_source_chain_group": decision.early_source_chain,
                "continuous_retention_group": decision.continuous_retention_chain,
                "h1600_retention_ratio": decision.h1600_retention_ratio,
                "h3200_retention_ratio": decision.h3200_retention_ratio,
                "h4800_retention_ratio": decision.h4800_retention_ratio,
                "productive_h3200_group": decision.continuous_retention_chain,
                "productive_h4800_group": h4800_ok,
                "late_rebound_group": decision.late_rebound,
                "row_early_chain_count": sum(int_flag(r.get("early_source_chain")) for r in rows),
                "row_continuous_count": sum(int_flag(r.get("continuous_retention_chain")) for r in rows),
                "row_late_rebound_count": sum(int_flag(r.get("late_rebound_v22")) for r in rows),
                "source_chain_blocker": decision.blocker,
            }
        )
    write_rows(out_dir / "v22_source_chain_summary.csv", summary)

    candidate = [r for r in summary if str(r.get("v22_id", "")).startswith(("F", "KSW", "MLP"))]
    decision = {
        "source_chain_rows": len(enriched),
        "source_chain_groups": len(summary),
        "candidate_groups": len(candidate),
        "candidate_early_chain_groups": sum(int_flag(r.get("early_source_chain_group")) for r in candidate),
        "candidate_continuous_groups": sum(int_flag(r.get("continuous_retention_group")) for r in candidate),
        "candidate_productive_h3200_groups": sum(int_flag(r.get("productive_h3200_group")) for r in candidate),
        "candidate_productive_h4800_groups": sum(int_flag(r.get("productive_h4800_group")) for r in candidate),
        "late_rebound_groups": sum(int_flag(r.get("late_rebound_group")) for r in summary),
        "full_escalation_recommended": int(any(int_flag(r.get("early_source_chain_group")) for r in candidate)),
        "promotion_candidate": int(any(int_flag(r.get("productive_h4800_group")) for r in candidate)),
    }
    if not decision["candidate_early_chain_groups"]:
        decision["decision"] = "NoEarlySourceChain"
    elif not decision["candidate_continuous_groups"]:
        decision["decision"] = "EarlyChainButNoContinuousRetention"
    elif not decision["candidate_productive_h4800_groups"]:
        decision["decision"] = "ContinuousH3200ButNoH4800"
    else:
        decision["decision"] = "CandidateReadyForIndependentConfirmation"
    write_json(out_dir / "v22_source_chain_decision.json", decision)
    write_rows(out_dir / "v22_source_chain_decision.csv", [decision])
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_source_chain_dynamics.py --merge-only", status="completed", note=f"v22 source groups={len(summary)} decision={decision['decision']}")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    call_v21_source(args)
    if args.merge_only:
        summarize_v22(out_dir)


if __name__ == "__main__":
    main()
