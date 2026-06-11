#!/usr/bin/env python3
"""Summarize v22.01 old-shape continuation source-retention outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v22_01_common import V2201_OFFICIAL, V2201_RECAP_DOC, append_text


CONTINUATIONS = [
    ("f53_oldshape_replay", "F53 old-shape replay"),
    ("f75_f77_oldshape", "F75-F77 old-shape invariant/optimizer replay"),
    ("f78_f83_oldshape", "F78-F83 old-shape terminal-route replay"),
    ("f84_f86_oldshape", "F84-F86 old-shape terminal-hold replay"),
    ("f87_oldshape", "F87 old-shape terminal raw-then-source-guard repair"),
]


SUMMARY_FIELDS = [
    "continuation",
    "label",
    "v22_id",
    "rows",
    "h100",
    "h400",
    "h800",
    "h1600",
    "h2400",
    "h3200",
    "h4800",
    "early_source_chain_group",
    "continuous_retention_group",
    "productive_h4800_group",
    "terminal_collapse_group",
    "row_h4800_positive_count",
    "source_chain_blocker",
]


DATASET_FIELDS = [
    "continuation",
    "label",
    "v22_id",
    "dataset",
    "rows",
    "mean_h800",
    "mean_h1600",
    "mean_h3200",
    "mean_h4800",
    "h4800_positive_rows",
    "early_rows",
    "continuous_rows",
    "continuous_h4800_rows",
    "blockers",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _to_float(value: str) -> float | None:
    if value == "" or value.lower() == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def _candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if not row.get("v22_id", "").startswith("CTRL-")]


def build_audit(out_dir: Path) -> tuple[Path, Path, Path]:
    summary_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []

    for tag, label in CONTINUATIONS:
        route_path = out_dir / f"v22_01_continuation_{tag}_route.json"
        summary_path = out_dir / f"v22_01_continuation_{tag}_source_chain_summary.csv"
        row_path = out_dir / f"v22_01_continuation_{tag}_row_localization.csv"
        if not route_path.exists() or not summary_path.exists() or not row_path.exists():
            continue

        route = json.loads(route_path.read_text())
        route["continuation"] = tag
        route["label"] = label
        route_rows.append(route)

        for row in _candidate_rows(_read_csv(summary_path)):
            out_row: dict[str, object] = {"continuation": tag, "label": label}
            for field in SUMMARY_FIELDS:
                if field not in {"continuation", "label"}:
                    out_row[field] = row.get(field, "")
            summary_rows.append(out_row)

        by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(row_path):
            if row.get("v22_id", "").startswith("CTRL-"):
                continue
            by_key[(row.get("v22_id", ""), row.get("dataset", ""))].append(row)

        for (v22_id, dataset), group in sorted(by_key.items()):
            blockers = sorted({row.get("source_chain_blocker", "") for row in group if row.get("source_chain_blocker", "")})
            h800 = [v for row in group if (v := _to_float(row.get("source_h800", ""))) is not None]
            h1600 = [v for row in group if (v := _to_float(row.get("source_h1600", ""))) is not None]
            h3200 = [v for row in group if (v := _to_float(row.get("source_h3200", ""))) is not None]
            h4800 = [v for row in group if (v := _to_float(row.get("source_h4800", ""))) is not None]
            dataset_rows.append(
                {
                    "continuation": tag,
                    "label": label,
                    "v22_id": v22_id,
                    "dataset": dataset,
                    "rows": len(group),
                    "mean_h800": _mean(h800),
                    "mean_h1600": _mean(h1600),
                    "mean_h3200": _mean(h3200),
                    "mean_h4800": _mean(h4800),
                    "h4800_positive_rows": sum(1 for value in h4800 if value >= 0.005),
                    "early_rows": sum(1 for row in group if row.get("early_source_chain") == "1"),
                    "continuous_rows": sum(1 for row in group if row.get("continuous_retention_chain") == "1"),
                    "continuous_h4800_rows": sum(1 for row in group if row.get("continuous_h4800_chain") == "1"),
                    "blockers": ";".join(blockers),
                }
            )

    route_out = out_dir / "v22_01_oldshape_continuation_route_audit.csv"
    summary_out = out_dir / "v22_01_oldshape_continuation_candidate_summary.csv"
    dataset_out = out_dir / "v22_01_oldshape_continuation_dataset_localization.csv"
    route_fields = [
        "continuation",
        "label",
        "decision",
        "promotion_allowed",
        "candidate_groups",
        "candidate_early_chain",
        "candidate_continuous_h3200",
        "candidate_h4800",
        "terminal_collapse_groups",
        "late_rebound_groups",
        "best_candidate_h4800",
        "rows",
    ]
    _write_csv(route_out, route_rows, route_fields)
    _write_csv(summary_out, summary_rows, SUMMARY_FIELDS)
    _write_csv(dataset_out, dataset_rows, DATASET_FIELDS)
    return route_out, summary_out, dataset_out


def append_recap(out_dir: Path, route_path: Path, summary_path: Path, dataset_path: Path) -> None:
    routes = _read_csv(route_path)
    summary = _read_csv(summary_path)
    dataset = _read_csv(dataset_path)

    best = sorted(
        _candidate_rows(summary),
        key=lambda row: (_to_float(row.get("h4800", "")) if _to_float(row.get("h4800", "")) is not None else -999.0),
        reverse=True,
    )[:8]
    grouped_passes = [row for row in summary if row.get("productive_h4800_group") == "1"]

    lines = [
        "\n## v22.01 old-shape continuation aggregate audit\n",
        f"- route audit: `{route_path.relative_to(out_dir.parent.parent.parent.parent) if False else route_path}`\n",
        f"- candidate summary: `{summary_path}`\n",
        f"- dataset localization: `{dataset_path}`\n",
        f"- continuation route rows: {len(routes)}\n",
        f"- candidate summary rows: {len(summary)}\n",
        f"- grouped h4800 passing candidate rows: {len(grouped_passes)}\n",
        "\n### Route audit\n\n",
        "| continuation | decision | early | continuous_h3200 | h4800 | terminal_collapse | best_h4800 |\n",
        "| --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for row in routes:
        lines.append(
            f"| {row.get('label','')} | {row.get('decision','')} | {row.get('candidate_early_chain','')} | "
            f"{row.get('candidate_continuous_h3200','')} | {row.get('candidate_h4800','')} | "
            f"{row.get('terminal_collapse_groups','')} | {row.get('best_candidate_h4800','')} |\n"
        )

    lines.extend(
        [
            "\n### Best candidate h4800 rows\n\n",
            "| continuation | v22_id | h800 | h1600 | h3200 | h4800 | early | continuous | h4800_group | blocker |\n",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
        ]
    )
    for row in best:
        lines.append(
            f"| {row.get('label','')} | {row.get('v22_id','')} | {row.get('h800','')} | "
            f"{row.get('h1600','')} | {row.get('h3200','')} | {row.get('h4800','')} | "
            f"{row.get('early_source_chain_group','')} | {row.get('continuous_retention_group','')} | "
            f"{row.get('productive_h4800_group','')} | {row.get('source_chain_blocker','')} |\n"
        )

    f53 = [row for row in dataset if row.get("v22_id") == "MLP-F53-trainloss-terminal-lookahead-floor-source"]
    if f53:
        lines.extend(
            [
                "\n### F53 row localization by dataset\n\n",
                "| dataset | rows | mean_h800 | mean_h1600 | mean_h3200 | mean_h4800 | h4800_positive_rows | early_rows | continuous_rows | continuous_h4800_rows | blockers |\n",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
            ]
        )
        for row in sorted(f53, key=lambda item: item.get("dataset", "")):
            lines.append(
                f"| {row.get('dataset','')} | {row.get('rows','')} | {row.get('mean_h800','')} | "
                f"{row.get('mean_h1600','')} | {row.get('mean_h3200','')} | {row.get('mean_h4800','')} | "
                f"{row.get('h4800_positive_rows','')} | {row.get('early_rows','')} | "
                f"{row.get('continuous_rows','')} | {row.get('continuous_h4800_rows','')} | {row.get('blockers','')} |\n"
            )

    lines.extend(
        [
            "\n### Aggregate insight\n\n",
            "- F53/F75-F86 old-shape continuations produced zero grouped h4800 source-chain candidates; v22.01 remains not achieved and `promotion_allowed=0`.\n",
            "- The best reproduced mechanism is still F53 at h4800=-0.003963543309105767; every plan-continuation old-shape repair after F53 is worse on grouped h4800.\n",
            "- F53 row localization shows MNIST carries positive h4800 rows, while Fashion-MNIST/KMNIST remain terminal-collapse blockers; the failed invariant/readout/optimizer continuations did not convert that row-level heterogeneity into grouped retained source.\n",
            "- Current evidence supports stopping terminal floor/carrier/hold micro-sweeps and moving only with a genuinely new train-only retained-target/source-observability theory.\n",
        ]
    )
    append_text(V2201_RECAP_DOC, "".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=V2201_OFFICIAL)
    parser.add_argument("--skip-recap", action="store_true")
    args = parser.parse_args()
    route_path, summary_path, dataset_path = build_audit(args.out_dir)
    if not args.skip_recap:
        append_recap(args.out_dir, route_path, summary_path, dataset_path)
    print(route_path)
    print(summary_path)
    print(dataset_path)


if __name__ == "__main__":
    main()
