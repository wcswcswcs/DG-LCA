#!/usr/bin/env python3
"""Summarize v22 AdamW-boundary phase reset evidence."""

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

from dgkan.fu.source_chain import RETENTION_EPS, classify_source_chain, source_chain_row  # noqa: E402
from experiments.run_v22_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_rows, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2200_f40_")
    p.add_argument("--candidate-prefix", default="MLP-F40")
    p.add_argument("--artifact-prefix", default="v22_f40_phase_reset")
    p.add_argument("--phase-label", default="F40")
    return p


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key)) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    cmd = (
        f"{PYTHON} experiments/run_v22_f40_phase_reset_summary.py "
        f"--run-prefix {args.run_prefix} --candidate-prefix {args.candidate_prefix} "
        f"--artifact-prefix {args.artifact_prefix} --phase-label {args.phase_label}"
    )
    append_exec(out_dir, cmd, status="started")
    rows = [
        dict(r)
        for r in read_rows(out_dir / "v22_source_chain_matrix.csv")
        if str(r.get("run_label", "")).startswith(str(args.run_prefix))
    ]
    for row in rows:
        row.update(source_chain_row(row, prefix="source_h"))
    write_rows(out_dir / f"{args.artifact_prefix}_matrix.csv", rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("carrier", "")), str(row.get("v22_id", "")))].append(row)
    summary: list[dict[str, Any]] = []
    for (carrier, v22_id), group in sorted(groups.items()):
        h = {f"h{step}": mean(group, f"source_h{step}") for step in (100, 400, 800, 1600, 2400, 3200, 4800, 6400)}
        decision = classify_source_chain(h["h100"], h["h400"], h["h800"], h["h1600"], h["h3200"], h["h4800"])
        summary.append(
            {
                "carrier": carrier,
                "v22_id": v22_id,
                "rows": len(group),
                "blocked_rows": sum(1 for r in group if str(r.get("execution_status", "")).startswith("blocked")),
                **h,
                "early_source_chain_group": decision.early_source_chain,
                "continuous_retention_group": decision.continuous_retention_chain,
                "h1600_retention_ratio": decision.h1600_retention_ratio,
                "h3200_retention_ratio": decision.h3200_retention_ratio,
                "h4800_retention_ratio": decision.h4800_retention_ratio,
                "productive_h3200_group": decision.continuous_retention_chain,
                "productive_h4800_group": int(decision.continuous_retention_chain and math.isfinite(h["h4800"]) and h["h4800"] >= RETENTION_EPS),
                "row_early_chain_count": sum(int_flag(r.get("early_source_chain")) for r in group),
                "row_continuous_count": sum(int_flag(r.get("continuous_retention_chain")) for r in group),
                "source_chain_blocker": decision.blocker,
            }
        )
    write_rows(out_dir / f"{args.artifact_prefix}_summary.csv", summary)

    candidates = [r for r in summary if str(r.get("v22_id", "")).startswith(str(args.candidate_prefix))]
    candidate = candidates[0] if candidates else {}
    h3200_finite = math.isfinite(finite_float(candidate.get("h3200")))
    h4800_finite = math.isfinite(finite_float(candidate.get("h4800")))
    decision = {
        "run_prefix": str(args.run_prefix),
        "candidate_prefix": str(args.candidate_prefix),
        "artifact_prefix": str(args.artifact_prefix),
        "phase_label": str(args.phase_label),
        "rows": len(rows),
        "groups": len(summary),
        "candidate_rows": int(candidate.get("rows", 0) or 0),
        "candidate_blocked_rows": int(candidate.get("blocked_rows", 0) or 0),
        "candidate_h100": candidate.get("h100", ""),
        "candidate_h400": candidate.get("h400", ""),
        "candidate_h800": candidate.get("h800", ""),
        "candidate_h1600": candidate.get("h1600", ""),
        "candidate_h3200": candidate.get("h3200", ""),
        "candidate_h4800": candidate.get("h4800", ""),
        "candidate_h1600_retention_ratio": candidate.get("h1600_retention_ratio", ""),
        "candidate_h3200_retention_ratio": candidate.get("h3200_retention_ratio", ""),
        "candidate_h4800_retention_ratio": candidate.get("h4800_retention_ratio", ""),
        "candidate_early_chain": int_flag(candidate.get("early_source_chain_group")),
        "candidate_continuous": int_flag(candidate.get("continuous_retention_group")),
        "candidate_productive_h3200": int_flag(candidate.get("productive_h3200_group")),
        "candidate_productive_h4800": int_flag(candidate.get("productive_h4800_group")),
        "h3200_observed": int(h3200_finite),
        "h4800_observed": int(h4800_finite),
        "full_escalation_recommended": int(int_flag(candidate.get("early_source_chain_group")) and not h3200_finite),
        "promotion_allowed": 0,
    }
    phase = str(args.phase_label)
    if not rows:
        decision["decision"] = f"{phase}RowsMissing"
    elif not int_flag(decision["candidate_early_chain"]):
        decision["decision"] = f"{phase}PhaseResetNoEarlyChain"
    elif not int_flag(decision["candidate_productive_h3200"]):
        h3200_value = finite_float(candidate.get("h3200"))
        decision["decision"] = (
            f"{phase}EarlyChainNoContinuousRetentionRatio"
            if h3200_finite and h3200_value >= RETENTION_EPS
            else f"{phase}EarlyChainNoContinuousRetentionH3200"
            if h3200_finite
            else f"{phase}EarlyChainNeedsFullH3200Validation"
        )
    elif not int_flag(decision["candidate_productive_h4800"]):
        decision["decision"] = f"{phase}ContinuousH3200ButH4800Failed" if h4800_finite else f"{phase}ContinuousH3200NeedsH4800Validation"
    else:
        decision["decision"] = f"{phase}ReadyForIndependentConfirmation"
    write_json(out_dir / f"{args.artifact_prefix}_decision.json", decision)
    write_rows(out_dir / f"{args.artifact_prefix}_decision.csv", [decision])
    append_exec(
        out_dir,
        cmd,
        status="completed",
        note=f"decision={decision['decision']} rows={len(rows)} early={decision['candidate_early_chain']}",
    )


if __name__ == "__main__":
    main()
