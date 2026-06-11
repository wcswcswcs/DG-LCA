#!/usr/bin/env python3
"""v19 H13 offline anatomy for H10/H11 M31 continuation artifacts.

This script does not run training. It reads existing H10/H11 matrix/trace
artifacts, writes H13 anatomy CSV/MD artifacts, updates route metadata, and
refreshes the official bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


HORIZONS = ("h800", "h1600", "h3200", "h4800")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return sum(clean) / len(clean) if clean else None


def _min(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return min(clean) if clean else None


def _fmt(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_matrix_rows(official_dir: Path) -> list[dict[str, str]]:
    inputs = [
        ("H10-offset0", official_dir / "v19_h10_m31_full_matrix.csv"),
        ("H11-independent", official_dir / "v19_h11_m31_independent_matrix.csv"),
    ]
    rows: list[dict[str, str]] = []
    for phase, path in inputs:
        for row in _read_csv(path):
            row["_phase"] = phase
            rows.append(row)
    return rows


def _summarize_specs(
    spec_groups: dict[tuple[str, str], list[dict[str, str]]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for (phase, spec), rows in sorted(spec_groups.items()):
        out = {"phase": phase, "continuation_id": spec, "rows": str(len(rows))}
        source_means: dict[str, float | None] = {}
        for horizon in HORIZONS:
            sources = [_float(row, f"source_{horizon}") for row in rows]
            losses = [_float(row, f"val_loss_{horizon}") for row in rows]
            source_means[horizon] = _mean(sources)
            out[f"{horizon}_mean_source"] = _fmt(source_means[horizon])
            out[f"{horizon}_positive_count"] = str(
                sum(1 for value in sources if value is not None and value > 0)
            )
            out[f"{horizon}_mean_m31_val_loss"] = _fmt(_mean(losses))
        for previous, current in (("h800", "h1600"), ("h1600", "h3200"), ("h3200", "h4800")):
            before = source_means[previous]
            after = source_means[current]
            out[f"retention_{current}_over_{previous}"] = (
                _fmt(max(0.0, after / before))
                if before is not None and before > 0 and after is not None
                else ""
            )
        out["retained_to_h3200"] = (
            "1"
            if all(source_means[h] is not None and source_means[h] > 0 for h in ("h800", "h1600", "h3200"))
            else "0"
        )
        out["h4800_positive"] = (
            "1"
            if source_means["h4800"] is not None and source_means["h4800"] > 0
            else "0"
        )
        output.append(out)
    return output


def _controls_vs_m31(matrix_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_context: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in matrix_rows:
        key = (
            row["_phase"],
            row.get("run_label", ""),
            row.get("init_seed_offset", ""),
            row.get("dataset", ""),
            row.get("seed", ""),
        )
        by_context[key].append(row)

    output: list[dict[str, str]] = []
    for key, rows in sorted(by_context.items()):
        phase, run_label, init_seed_offset, dataset, seed = key
        controls = [row for row in rows if row.get("continuation_id", "").startswith("CTRL-")]
        m31_rows = [row for row in rows if row.get("continuation_id", "").startswith("M31-")]
        if not controls or not m31_rows:
            continue
        for m31 in m31_rows:
            out = {
                "phase": phase,
                "run_label": run_label,
                "init_seed_offset": init_seed_offset,
                "dataset": dataset,
                "seed": seed,
                "continuation_id": m31.get("continuation_id", ""),
            }
            for horizon in HORIZONS:
                best_control = _min(_float(control, f"val_loss_{horizon}") for control in controls)
                m31_loss = _float(m31, f"val_loss_{horizon}")
                computed = best_control - m31_loss if best_control is not None and m31_loss is not None else None
                out[f"{horizon}_best_control_loss"] = _fmt(best_control)
                out[f"{horizon}_m31_loss"] = _fmt(m31_loss)
                out[f"{horizon}_computed_source"] = _fmt(computed)
                out[f"{horizon}_recorded_source"] = _fmt(_float(m31, f"source_{horizon}"))
            output.append(out)
    return output


def _gate_diagnostics(official_dir: Path) -> list[dict[str, str]]:
    inputs = [
        ("H10-offset0", official_dir / "v19_h10_m31_full_traces.csv"),
        ("H11-independent", official_dir / "v19_h11_m31_independent_traces.csv"),
    ]
    output: list[dict[str, str]] = []
    for phase, path in inputs:
        groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(path):
            continuation_id = row.get("continuation_id", "")
            if continuation_id.startswith("M31-") and row.get("source_state_gate_accept", "") != "":
                key = (
                    phase,
                    row.get("run_label", ""),
                    row.get("init_seed_offset", ""),
                    continuation_id,
                )
                groups[key].append(row)
        for (phase_name, run_label, init_seed_offset, continuation_id), rows in sorted(groups.items()):
            accepts = [_float(row, "source_state_gate_accept") for row in rows]
            output.append(
                {
                    "phase": phase_name,
                    "run_label": run_label,
                    "init_seed_offset": init_seed_offset,
                    "continuation_id": continuation_id,
                    "gate_trials": str(len(rows)),
                    "gate_accepts": str(sum(1 for value in accepts if value is not None and value > 0.5)),
                    "gate_accept_rate": _fmt(_mean(accepts)),
                    "density_mean": _fmt(_mean(_float(row, "source_state_consensus_density") for row in rows)),
                    "corrupt_cos_mean": _fmt(_mean(_float(row, "source_state_corrupt_cos") for row in rows)),
                    "signal_gain_mean": _fmt(_mean(_float(row, "source_state_signal_gain") for row in rows)),
                    "corrupt_gain_mean": _fmt(_mean(_float(row, "source_state_corrupt_gain") for row in rows)),
                }
            )
    return output


def _phase_rollup(
    spec_groups: dict[tuple[str, str], list[dict[str, str]]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for spec in sorted({spec for _, spec in spec_groups}):
        row = {"continuation_id": spec}
        for horizon in HORIZONS:
            h10 = _mean(_float(item, f"source_{horizon}") for item in spec_groups.get(("H10-offset0", spec), []))
            h11 = _mean(_float(item, f"source_{horizon}") for item in spec_groups.get(("H11-independent", spec), []))
            row[f"h10_{horizon}_mean_source"] = _fmt(h10)
            row[f"h11_{horizon}_mean_source"] = _fmt(h11)
            row[f"h11_minus_h10_{horizon}"] = _fmt(h11 - h10 if h10 is not None and h11 is not None else None)
        output.append(row)
    return output


def _write_decision(
    path: Path,
    summary_rows: list[dict[str, str]],
    controls_rows: list[dict[str, str]],
) -> tuple[int, int, int, int]:
    h10_h3200 = sum(
        1
        for row in summary_rows
        if row["phase"] == "H10-offset0" and row["h3200_mean_source"] and float(row["h3200_mean_source"]) > 0
    )
    h11_h3200 = sum(
        1
        for row in summary_rows
        if row["phase"] == "H11-independent" and row["h3200_mean_source"] and float(row["h3200_mean_source"]) > 0
    )
    h10_h4800 = sum(
        1
        for row in summary_rows
        if row["phase"] == "H10-offset0" and row["h4800_mean_source"] and float(row["h4800_mean_source"]) > 0
    )
    h11_h4800 = sum(
        1
        for row in summary_rows
        if row["phase"] == "H11-independent" and row["h4800_mean_source"] and float(row["h4800_mean_source"]) > 0
    )

    lines = [
        "# v19 H13 H10/H11 离线解剖决策",
        "",
        "本步骤不重跑训练，只读取 H10/H11 已生成 matrix/trace，核对 H10 offset0 正例是否能被独立初始化支持。",
        "",
        f"- H10 h3200 positive spec count: {h10_h3200}",
        f"- H11 h3200 positive spec count: {h11_h3200}",
        f"- H10 h4800 positive spec count: {h10_h4800}",
        f"- H11 h4800 positive spec count: {h11_h4800}",
        "- 结论：H10 的 M31 h3200 正例没有在 H11 独立 init offsets 中复现；H10 自身 h4800 也 washout，因此不能作为 retained source。",
        "- 行动：不继续同族 M31/M32 参数扫描；需要先设计新的 source-observability/toy-correctness 理论，再做下一轮 4GPU。",
        "",
        "## Mean Control/M31 Loss Check",
        "",
        "| phase | continuation_id | h3200 best control loss | h3200 M31 loss | h4800 best control loss | h4800 M31 loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for phase in ("H10-offset0", "H11-independent"):
        for spec in sorted({row["continuation_id"] for row in controls_rows}):
            rows = [row for row in controls_rows if row["phase"] == phase and row["continuation_id"] == spec]
            if not rows:
                continue
            c32 = _mean(float(row["h3200_best_control_loss"]) for row in rows if row["h3200_best_control_loss"])
            m32 = _mean(float(row["h3200_m31_loss"]) for row in rows if row["h3200_m31_loss"])
            c48 = _mean(float(row["h4800_best_control_loss"]) for row in rows if row["h4800_best_control_loss"])
            m48 = _mean(float(row["h4800_m31_loss"]) for row in rows if row["h4800_m31_loss"])
            lines.append(f"| {phase} | {spec} | {_fmt(c32)} | {_fmt(m32)} | {_fmt(c48)} | {_fmt(m48)} |")
    path.write_text("\n".join(lines) + "\n")
    return h10_h3200, h11_h3200, h10_h4800, h11_h4800


def _update_manifest(official_dir: Path, artifact_paths: list[Path]) -> None:
    manifest_path = official_dir / "v19_required_artifact_manifest.csv"
    existing: list[dict[str, str]] = []
    fields: list[str] = []
    if manifest_path.exists():
        with manifest_path.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            existing = list(reader)
    for required in ("artifact", "path", "exists", "size_bytes", "note"):
        if required not in fields:
            fields.append(required)

    seen = {row.get("path") for row in existing}
    for path in artifact_paths:
        rel = str(path)
        if rel in seen:
            for row in existing:
                if row.get("path") == rel:
                    row["exists"] = "1"
                    row["size_bytes"] = str(path.stat().st_size)
                    row.setdefault("note", "")
                    if not row["note"]:
                        row["note"] = "H13 offline anatomy artifact"
        else:
            row = {field: "" for field in fields}
            row.update(
                {
                    "artifact": path.name,
                    "path": rel,
                    "exists": "1",
                    "size_bytes": str(path.stat().st_size),
                    "note": "H13 offline anatomy artifact",
                }
            )
            existing.append(row)

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in existing:
            for field in fields:
                row.setdefault(field, "")
            writer.writerow(row)


def _refresh_bundle(official_dir: Path) -> int:
    bundle_path = official_dir / "v19_results_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(official_dir.rglob("*")):
            if path.is_file() and path.name != bundle_path.name:
                zf.write(path, path.relative_to(official_dir))
    return bundle_path.stat().st_size


def run(official_dir: Path) -> None:
    matrix_rows = _load_matrix_rows(official_dir)
    spec_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in matrix_rows:
        continuation_id = row.get("continuation_id", "")
        if continuation_id.startswith("M31-"):
            spec_groups[(row["_phase"], continuation_id)].append(row)

    summary_rows = _summarize_specs(spec_groups)
    controls_rows = _controls_vs_m31(matrix_rows)
    diagnostics_rows = _gate_diagnostics(official_dir)
    rollup_rows = _phase_rollup(spec_groups)

    summary_path = official_dir / "v19_h13_h10_h11_anatomy_summary.csv"
    controls_path = official_dir / "v19_h13_h10_h11_controls_vs_m31.csv"
    diagnostics_path = official_dir / "v19_h13_h10_h11_gate_diagnostics.csv"
    rollup_path = official_dir / "v19_h13_h10_h11_phase_rollup.csv"
    decision_path = official_dir / "v19_h13_h10_h11_anatomy_decision.md"
    route_path = official_dir / "v19_route_decision.json"

    _write_csv(summary_path, summary_rows)
    _write_csv(controls_path, controls_rows)
    _write_csv(diagnostics_path, diagnostics_rows)
    _write_csv(rollup_path, rollup_rows)
    h10_h3200, h11_h3200, h10_h4800, h11_h4800 = _write_decision(decision_path, summary_rows, controls_rows)

    route = json.loads(route_path.read_text())
    route.update(
        {
            "route_detail": (
                "H10 M31 partial positive was not independently confirmed by H11; "
                "H12 post-AdamW recompute M32 smoke was negative; H13 offline anatomy "
                "confirmed H10 h3200 positives wash out by h4800 and do not reproduce "
                "under independent init offsets. No same-family M31/M32 repair remains "
                "actionable without a new source-observability theory."
            ),
            "FunctionalRoute": (
                "S3-MLPGenericRetainedSource;H10M31PartialPositive;"
                "H11IndependentConfirmationFailed;H12M32PostAdamWSmokeNegative;"
                "H13H10H11AnatomyNoRepro;NoActionableSourceStateFamilyRepair"
            ),
            "h13_anatomy_completed": 1,
            "h13_h10_h3200_positive_spec_count": h10_h3200,
            "h13_h11_h3200_positive_spec_count": h11_h3200,
            "h13_h10_h4800_positive_spec_count": h10_h4800,
            "h13_h11_h4800_positive_spec_count": h11_h4800,
            "h13_actionable_same_family_repair": 0,
            "h13_decision": (
                "H10 offset0 M31 partial positives are not reproduced by H11 independent "
                "init offsets and wash out by h4800; stop same-family M31/M32 sweeps until "
                "new source-observability theory is specified."
            ),
            "route": "R-MLPGenericRetained-KANCarrierSourceStateFamilyNotRobust",
            "promotion_allowed": 0,
            "official_success_reached": 0,
            "continue_same_family_recommended": 0,
            "new_source_theory_required": 1,
            "conceptual_uncertainty": 1,
        }
    )
    route_path.write_text(json.dumps(route, indent=2, ensure_ascii=False) + "\n")

    artifact_paths = [summary_path, controls_path, diagnostics_path, rollup_path, decision_path, route_path]
    _update_manifest(official_dir, artifact_paths)
    bundle_size = _refresh_bundle(official_dir)

    for path in artifact_paths[:-1]:
        print(f"wrote {path}")
    print(f"updated {route_path}")
    print(f"bundle_size {bundle_size}")
    print(
        "h10_pos_h3200",
        h10_h3200,
        "h11_pos_h3200",
        h11_h3200,
        "h10_h4800",
        h10_h4800,
        "h11_h4800",
        h11_h4800,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path("results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19"),
    )
    args = parser.parse_args()
    run(args.official_dir)


if __name__ == "__main__":
    main()
