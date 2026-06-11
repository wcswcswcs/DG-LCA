#!/usr/bin/env python3
"""Summarize the local source-observer no-go boundary for v22.07."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import (  # noqa: E402
    PYTHON,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


HORIZON_KEYS = [
    "source_vs_best_control_h100",
    "source_vs_best_control_h400",
    "source_vs_best_control_h800",
    "source_vs_best_control_h1600",
    "source_vs_best_control_h3200",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _best(rows: list[dict[str, Any]], key: str) -> float:
    vals = [finite_float(r.get(key), float("-inf")) for r in rows]
    vals = [v for v in vals if v != float("-inf")]
    return max(vals) if vals else float("nan")


def _count_positive(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for r in rows if finite_float(r.get(key), -999.0) >= 0.005)


def _family_summary(
    out_dir: Path,
    *,
    family: str,
    summary_file: str,
    route_file: str,
    c1_file: str = "",
    c1_col: str = "",
    early_col: str,
    full_col: str,
    official_col: str = "official_C2_solver_claimed",
) -> dict[str, Any]:
    summary = read_rows(out_dir / summary_file)
    c1_rows = read_rows(out_dir / c1_file) if c1_file else []
    route = read_json(out_dir / route_file)
    best_c1_b2 = _best(c1_rows, "B2_transfer_gain") if c1_rows else float("nan")
    c1_pass_rows = sum(int_flag(r.get(c1_col)) for r in c1_rows) if c1_col else ""
    early_rows = sum(int_flag(r.get(early_col)) for r in summary)
    full_rows = sum(int_flag(r.get(full_col)) for r in summary)
    official_rows = sum(int_flag(r.get(official_col)) for r in summary) if summary and official_col else int_flag(route.get("official_C2_solver_claimed_rows"))
    row = {
        "family": family,
        "route": route.get("route", route.get("decision", "")),
        "rows": len(summary),
        "c1_rows": len(c1_rows) if c1_rows else "",
        "c1_pass_rows": c1_pass_rows,
        "best_C1_B2_transfer_gain": best_c1_b2 if c1_rows else "",
        "early_pass_rows": early_rows,
        "full_pass_rows": full_rows,
        "official_C2_solver_claimed_rows": official_rows,
        "h1600_positive_rows": _count_positive(summary, "source_vs_best_control_h1600"),
        "h3200_positive_rows": _count_positive(summary, "source_vs_best_control_h3200"),
        "best_h100": _best(summary, "source_vs_best_control_h100"),
        "best_h400": _best(summary, "source_vs_best_control_h400"),
        "best_h800": _best(summary, "source_vs_best_control_h800"),
        "best_h1600": _best(summary, "source_vs_best_control_h1600"),
        "best_h3200": _best(summary, "source_vs_best_control_h3200"),
        "blocker": route.get("blocker", ""),
    }
    row["family_boundary_failed"] = int(
        int_flag(row.get("full_pass_rows")) == 0
        and int_flag(row.get("h3200_positive_rows")) == 0
        and (row.get("c1_pass_rows") == "" or int_flag(row.get("c1_pass_rows")) == 0)
    )
    return row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    families = [
        _family_summary(
            out_dir,
            family="semantic_target_repair",
            c1_file="v22_07_c3_semantic_target_repair_c1.csv",
            c1_col="C1_semantic_pass",
            summary_file="v22_07_c3_semantic_target_repair_summary.csv",
            route_file="v22_07_c3_semantic_target_repair_route.json",
            early_col="C3_semantic_early_pass",
            full_col="C3_semantic_full_pass",
        ),
        _family_summary(
            out_dir,
            family="preservation_repair",
            summary_file="v22_07_c3_preservation_repair_summary.csv",
            route_file="v22_07_c3_preservation_repair_route.json",
            early_col="C3_preservation_early_pass",
            full_col="C3_preservation_full_pass",
        ),
        _family_summary(
            out_dir,
            family="retained_metric_solver",
            summary_file="v22_07_retained_metric_solver_repair_summary.csv",
            route_file="v22_07_retained_metric_solver_repair_route.json",
            early_col="C3_retained_solver_early_pass",
            full_col="C3_retained_solver_full_pass",
        ),
        _family_summary(
            out_dir,
            family="stage_fcp_hybrid",
            summary_file="v22_07_stagefcp_hybrid_repair_summary.csv",
            route_file="v22_07_stagefcp_hybrid_repair_route.json",
            early_col="C3_stagefcp_early_pass",
            full_col="C3_stagefcp_full_pass",
        ),
    ]
    for idx, name in enumerate(
        [
            "observer2_source_theory",
            "observer3_flow_theory",
            "observer4_transfer_meta",
            "observer5_activation_manifold",
            "observer6_curvature_guard",
            "observer7_control_nullspace",
        ],
        start=2,
    ):
        file_stem = {
            2: "observer2_source_theory",
            3: "observer3_flow_theory",
            4: "observer4_transfer_meta",
            5: "observer5_activation_manifold",
            6: "observer6_curvature_guard",
            7: "observer7_control_nullspace",
        }[idx]
        families.append(
            _family_summary(
                out_dir,
                family=name,
                c1_file=f"v22_07_{file_stem}_c1.csv",
                c1_col=f"C1_observer{idx}_pass",
                summary_file=f"v22_07_{file_stem}_summary.csv",
                route_file=f"v22_07_{file_stem}_route.json",
                early_col=f"C3_observer{idx}_early_pass",
                full_col=f"C3_observer{idx}_full_pass",
            )
        )

    observer_families = [r for r in families if str(r.get("family", "")).startswith("observer")]
    all_observers_present = all(int_flag(r.get("rows")) > 0 for r in observer_families)
    observer_full_rows = sum(int_flag(r.get("full_pass_rows")) for r in observer_families)
    observer_c1_pass_rows = sum(int_flag(r.get("c1_pass_rows")) for r in observer_families)
    observer_h1600_pos = sum(int_flag(r.get("h1600_positive_rows")) for r in observer_families)
    observer_h3200_pos = sum(int_flag(r.get("h3200_positive_rows")) for r in observer_families)
    observer_early_rows = sum(int_flag(r.get("early_pass_rows")) for r in observer_families)
    best_observer_c1 = max(
        [finite_float(r.get("best_C1_B2_transfer_gain"), float("-inf")) for r in observer_families],
        default=float("nan"),
    )
    best_observer_h3200 = max(
        [finite_float(r.get("best_h3200"), float("-inf")) for r in observer_families],
        default=float("nan"),
    )
    local_no_go = int(
        all_observers_present
        and len(observer_families) >= 6
        and observer_c1_pass_rows == 0
        and observer_full_rows == 0
        and observer_h1600_pos == 0
        and observer_h3200_pos == 0
    )
    route = {
        "route": "C3-SourceObserverLocalNoGoBoundary" if local_no_go else "C3-SourceObserverBoundaryIncomplete",
        "promotion_allowed": 0,
        "local_no_go_boundary_claimed": local_no_go,
        "scope": "attempted_train_only_source_observer_families_only",
        "universal_scientific_no_go_claimed": 0,
        "future_source_used_for_direction": 0,
        "observer_family_count": len(observer_families),
        "observer_rows": sum(int_flag(r.get("rows")) for r in observer_families),
        "observer_early_pass_rows": observer_early_rows,
        "observer_full_pass_rows": observer_full_rows,
        "observer_c1_pass_rows": observer_c1_pass_rows,
        "observer_h1600_positive_rows": observer_h1600_pos,
        "observer_h3200_positive_rows": observer_h3200_pos,
        "best_observer_C1_B2_transfer_gain": best_observer_c1,
        "best_observer_h3200": best_observer_h3200,
        "blocker": "observer_families_failed_C1_and_h1600_h3200_retention" if local_no_go else "boundary_incomplete",
    }
    write_rows(out_dir / "v22_07_source_observer_nogo_family_summary.csv", families)
    write_json(out_dir / "v22_07_source_observer_nogo_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_07_source_observer_nogo.py --out-dir {out_dir}",
        status="completed",
        note=(
            f"route={route['route']} local_no_go={local_no_go} "
            f"observer_c1={observer_c1_pass_rows} observer_full={observer_full_rows} "
            f"h1600_pos={observer_h1600_pos} h3200_pos={observer_h3200_pos}"
        ),
    )


if __name__ == "__main__":
    main()
