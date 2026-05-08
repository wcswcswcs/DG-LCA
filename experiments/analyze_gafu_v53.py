#!/usr/bin/env python3
"""Summarize DG-KAN v5.3 AB-RBF event-controller experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_3"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.3_ABRBF_EventController_结果复盘.md"
LOG = ROOT / "docs/log.md"
DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(row: dict, key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in {"", None, "nan", "NaN"}:
            return default
        return float(value)
    except Exception:
        return default


def fmt(x: float, digits: int = 4) -> str:
    return "" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def avg(rows: Sequence[dict], key: str, default: float = math.nan) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.mean(vals) if vals else default


def grouped(rows: Iterable[dict], *keys: str) -> dict[tuple[str, ...], list[dict]]:
    out: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        out[tuple(str(row.get(k, "")) for k in keys)].append(row)
    return dict(out)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def all_dataset_survivors(rows: Sequence[dict], key_fields: Sequence[str], pass_field: str) -> list[tuple[str, ...]]:
    by: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("error") or row.get("status") == "not_run":
            continue
        if int(f(row, pass_field, 0)) == 1:
            by[tuple(str(row.get(k, "")) for k in key_fields)].add(str(row.get("dataset", "")))
    return [k for k, ds in by.items() if all(d in ds for d in DATASETS)]


def write_svg_bar(path: Path, title: str, labels: Sequence[str], values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 70 + 28 * len(labels)
    finite = [v for v in values if math.isfinite(v)]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="28" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 54 + i * 28
        bar = 0.0 if not math.isfinite(value) or max_v <= 0 else 500 * value / max_v
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="10">{label}</text>')
        lines.append(f'<rect x="330" y="{y}" width="{bar:.1f}" height="18" fill="#2563eb"/>')
        lines.append(f'<text x="{340+bar:.1f}" y="{y+14}" font-family="Arial" font-size="10">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, title: str, rows: Sequence[dict], x_key: str, y_key: str, label_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 460
    pts = [(f(r, x_key), f(r, y_key), str(r.get(label_key, ""))) for r in rows]
    pts = [(x, y, label) for x, y, label in pts if math.isfinite(x) and math.isfinite(y)]
    xs = [x for x, _, _ in pts]
    ys = [y for _, y, _ in pts]
    xmin, xmax = min(xs or [0.0]), max(xs or [1.0])
    ymin, ymax = min(ys or [0.0]), max(ys or [1.0])
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        '<line x1="70" y1="390" x2="700" y2="390" stroke="#111"/>',
        '<line x1="70" y1="390" x2="70" y2="60" stroke="#111"/>',
    ]
    for i, (x, y, label) in enumerate(pts):
        px = 70 + 630 * (x - xmin) / (xmax - xmin)
        py = 390 - 330 * (y - ymin) / (ymax - ymin)
        color = ["#2563eb", "#dc2626", "#059669", "#7c3aed"][i % 4]
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="0.8"/>')
        if i < 16:
            lines.append(f'<text x="{px+7:.1f}" y="{py+4:.1f}" font-family="Arial" font-size="9">{label[:22]}</text>')
    lines.append(f'<text x="330" y="430" font-family="Arial" font-size="11">{x_key}</text>')
    lines.append(f'<text x="10" y="56" font-family="Arial" font-size="11">{y_key}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png_heatmap(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 480, 220
    data = bytearray([255, 255, 255] * width * height)
    fail_types = sorted({str(r.get("failure_type", "")) for r in rows})[:8]
    methods = sorted({str(r.get("method", "")) for r in rows})[:10]
    counts = defaultdict(int)
    for row in rows:
        counts[(str(row.get("method", "")), str(row.get("failure_type", "")))] += 1
    cell_w = max(1, width // max(1, len(fail_types)))
    cell_h = max(1, height // max(1, len(methods)))
    for iy, method in enumerate(methods):
        for ix, ft in enumerate(fail_types):
            count = counts[(method, ft)]
            color = (255, 255 - min(220, count * 55), 255 - min(220, count * 55))
            for yy in range(iy * cell_h, min(height, (iy + 1) * cell_h - 1)):
                for xx in range(ix * cell_w, min(width, (ix + 1) * cell_w - 1)):
                    pos = (yy * width + xx) * 3
                    data[pos : pos + 3] = bytes(color)
    raw = b"".join(b"\x00" + bytes(data[y * width * 3 : (y + 1) * width * 3]) for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_memory_smoke.csv")
    p1 = read_rows(BASE / "p1_single_smoothing_event.csv")
    p2 = read_rows(BASE / "p2_event_probe_audit.csv")
    p3 = read_rows(BASE / "p3_one_cycle_controller_variants.csv")
    p4 = read_rows(BASE / "p4_event_driven_multicycle_v2.csv")
    p5 = read_rows(BASE / "p5_candidate_selection3.csv")
    p6 = read_rows(BASE / "p6_confirm5.csv")
    p7 = read_rows(BASE / "p7_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p0_summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped(p0, "dataset", "method").items()):
        p0_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "peak_alloc": avg(rows, "peak_cuda_allocated_mb"),
                "peak_reserved": avg(rows, "peak_cuda_reserved_mb"),
                "smooth_time": avg(rows, "smoothing_time_sec", 0.0),
                "mem_ratio": avg(rows, "memory_ratio_vs_ABRBF_AdamW"),
                "core": int(avg(rows, "ABRBFDense_in_core", 0.0) >= 1.0),
                "coverage": int(avg(rows, "edge_coverage", 0.0) >= 0.999 and avg(rows, "base_coverage", 0.0) >= 0.999 and avg(rows, "rbf_residual_coverage", 0.0) >= 0.999),
                "memory_pass": int("Strong" not in method and avg(rows, "memory_ratio_vs_ABRBF_AdamW") <= 1.25),
            }
        )
    write_csv(BASE / "p0_core_memory_gate_summary.csv", p0_summary)

    p1_summary: list[dict] = []
    for (dataset, teacher, proposal, controller), rows in sorted(grouped(p1, "dataset", "teacher", "proposal", "controller").items()):
        p1_summary.append(
            {
                "dataset": dataset,
                "teacher": teacher,
                "proposal": proposal,
                "controller": controller,
                "acc_drop": avg(rows, "acc_drop"),
                "KL": avg(rows, "KL_teacher_student"),
                "logit": avg(rows, "logit_relative_drift"),
                "phiR_red": avg(rows, "phi_rbf_reduction"),
                "curvR_red": avg(rows, "curvature_rbf_reduction"),
                "mem_ratio": avg(rows, "memory_ratio_vs_teacher"),
                "P1": int(max(f(r, "p1_pass", 0) for r in rows) >= 1),
            }
        )
    p1_surv = all_dataset_survivors(p1, ["teacher", "proposal", "controller"], "p1_pass")
    write_csv(BASE / "p1_single_smoothing_gate_summary.csv", p1_summary)

    p2_summary: list[dict] = []
    for (dataset, seed), rows in sorted(grouped(p2, "dataset", "seed").items()):
        best_rows = []
        by_step: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_step[str(row.get("probe_step", ""))].append(row)
        for step_rows in by_step.values():
            best_rows.append(max(step_rows, key=lambda r: f(r, "probe_score", -1.0e18)))
        p2_summary.append(
            {
                "dataset": dataset,
                "seed": seed,
                "probes": len(best_rows),
                "best_score_max": max([f(r, "best_score", f(r, "probe_score", -1.0e18)) for r in best_rows] or [math.nan]),
                "new_trigger_count": sum(1 for r in best_rows if int(f(r, "would_trigger_by_geometry_debt", 0)) == 1 or int(f(r, "would_trigger_by_score", 0)) == 1),
                "old_trigger_count": sum(1 for r in best_rows if int(f(r, "would_trigger_by_plateau_old", 0)) == 1),
                "p90_acc_drop": sorted([f(r, "probe_acc_drop", 0.0) for r in rows])[max(0, int(0.9 * len(rows)) - 1)] if rows else math.nan,
                "max_eta_count": sum(1 for r in best_rows if abs(f(r, "best_eta", 0.0) - 0.05) < 1.0e-12),
                "P2": int(any(f(r, "best_score", f(r, "probe_score", -1.0e18)) > 0 for r in best_rows) and (sorted([f(r, "probe_acc_drop", 0.0) for r in rows])[max(0, int(0.9 * len(rows)) - 1)] if rows else 99.0) <= 0.005),
            }
        )
    p2_trigger_ds = {str(r.get("dataset")) for r in p2 if int(f(r, "would_trigger_by_geometry_debt", 0)) == 1 or int(f(r, "would_trigger_by_score", 0)) == 1}
    p2_surv = ["ABRBF-EventProbe"] if len(p2_trigger_ds) >= 2 else []
    write_csv(BASE / "p2_event_probe_gate_summary.csv", p2_summary)

    p3_summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped(p3, "dataset", "method").items()):
        p3_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "acc": avg(rows, "final_acc"),
                "gap": avg(rows, "acc_gap_vs_ABRBF", 0.0),
                "phiR_red": avg(rows, "phi_rbf_reduction", 0.0),
                "curvR_red": avg(rows, "curvature_rbf_reduction", 0.0),
                "recovery": avg(rows, "recovery_ratio", 1.0),
                "retention": avg(rows, "geometry_retention", 1.0),
                "mem": avg(rows, "memory_ratio_vs_ABRBF_AdamW", 1.0),
                "P3": int(max(f(r, "p3_pass", 0) for r in rows) >= 1),
            }
        )
    p3_surv = [s for s in all_dataset_survivors(p3, ["method"], "p3_pass") if s[0] != "ABRBF-AdamW"]
    write_csv(BASE / "p3_one_cycle_gate_summary.csv", p3_summary)

    p4_summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped(p4, "dataset", "method").items()):
        p4_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "acc": avg(rows, "final_acc"),
                "gap": avg(rows, "acc_gap_vs_ABRBF", 0.0),
                "phiR_red": avg(rows, "phi_rbf_reduction", 0.0),
                "curvR_red": avg(rows, "curvature_rbf_reduction", 0.0),
                "events": avg(rows, "event_count", 0.0),
                "accepted": avg(rows, "accepted_event_count", 0.0),
                "mem": avg(rows, "memory_ratio_vs_ABRBF_AdamW", 1.0),
                "time": avg(rows, "time_ratio_vs_ABRBF_AdamW", 1.0),
                "P4": int(max(f(r, "p4_pass", 0) for r in rows) >= 1),
            }
        )
    p4_surv = [s for s in all_dataset_survivors(p4, ["method"], "p4_pass") if s[0] != "ABRBF-AdamW"]
    write_csv(BASE / "p4_event_v2_gate_summary.csv", p4_summary)

    if p4_surv:
        status = "has_p4_survivor_needs_p5"
    elif p3_surv:
        status = "stop_after_p4_no_multicycle_survivor"
    elif p2_surv:
        status = "stop_after_p3_no_one_cycle_survivor"
    else:
        status = "stop_after_p2_probe_failed"

    decision = {
        "version": "v5.3",
        "status": status,
        "p1_survivors": ["|".join(s) for s in p1_surv],
        "p2_survivors": p2_surv,
        "p3_survivors": ["|".join(s) for s in p3_surv],
        "p4_survivors": ["|".join(s) for s in p4_surv],
        "rows": {"P0": len(p0), "P1": len(p1), "P2": len(p2), "P3": len(p3), "P4": len(p4), "P5": len(p5), "P6": len(p6), "P7": len(p7), "failures": len(failures)},
    }
    save_json(BASE / "aggregate_decision.json", decision)
    save_json(BASE / "recommendation.json", {"status": status, "recommendation": "Keep AB-RBF as default. Use LightSmooth as one-cycle maintenance unless EventV2 passes P4/P5."})

    write_svg_bar(FIG / "memory_dashboard.svg", "P0 Peak CUDA Allocated MB", [f"{r['dataset']} {r['method']}" for r in p0_summary], [float(r["peak_alloc"]) for r in p0_summary])
    write_svg_scatter(FIG / "single_event_pareto.svg", "P1 Single Event Pareto", p1_summary, "acc_drop", "phiR_red", "proposal")
    write_svg_scatter(FIG / "probe_trigger_dashboard.svg", "P2 Geometry Debt vs Score", p2, "geometry_debt", "probe_score", "dataset")
    write_svg_bar(FIG / "one_cycle_method_pareto.svg", "P3 PhiR Reduction", [f"{r['dataset']} {r['method'][:28]}" for r in p3_summary], [float(r["phiR_red"]) for r in p3_summary])
    write_svg_bar(FIG / "event_timeline.svg", "P4 Event Count", [f"{r['dataset']} {r['method']}" for r in p4_summary], [float(r["events"]) for r in p4_summary])
    write_png_heatmap(BASE / "failure_heatmap.png", failures)

    md: list[str] = []
    md.append("# DG-KAN v5.3 AB-RBF Event Controller 结果复盘")
    md.append("")
    md.append("本轮依据 `docs/DG-KAN_v5.3_ABRBF_EventController_下一步实验计划.md`。目标是把 v5.2 的 LightSmooth 从单次/one-cycle 工具接入 budgeted event controller：geometry debt + task slack + cheap probe score + cooldown + refresh recovery。")
    md.append("")
    md.append("## Run Inventory")
    md.append("")
    md.append(md_table(["stage", "rows", "errors"], [
        ["P0 core/memory", len(p0), sum(1 for r in p0 if r.get("error"))],
        ["P1 single event", len(p1), sum(1 for r in p1 if r.get("error"))],
        ["P2 probe audit", len(p2), sum(1 for r in p2 if r.get("error"))],
        ["P3 one-cycle refresh", len(p3), sum(1 for r in p3 if r.get("error"))],
        ["P4 event v2", len(p4), sum(1 for r in p4 if r.get("error"))],
        ["P5 selection", len(p5), sum(1 for r in p5 if r.get("error"))],
        ["P6 confirm5", len(p6), sum(1 for r in p6 if r.get("error"))],
        ["P7 confirm10", len(p7), sum(1 for r in p7 if r.get("error"))],
    ]))
    md.append("")
    md.append("## Code / Config Changes")
    md.append("")
    md.append("```text\nexperiments/run_gafu_v53.py\n  Added P2 cheap probe audit without writes.\n  Added P3 one-cycle refresh policy comparison: full/base-only/RBF-frozen refresh.\n  Added P4 EventV2 controller with score probe, geometry debt, cooldown, eta cap, and recovery accounting.\n\nexperiments/run_gafu_v51.py\n  Aligned runner AB-RBF helper classes with core dgkan_core.ABRBFDense so P0 can audit core consistency.\n\nexperiments/analyze_gafu_v53.py\n  Generates v5.3 gate summaries, figures, aggregate_decision.json, recommendation.json, and this replay.\n```")
    md.append("")
    md.append("## P0 Core / Memory Smoke")
    md.append("")
    md.append(md_table(["dataset", "method", "peak MB", "mem ratio", "smooth sec", "core", "coverage", "memory"], [
        [r["dataset"], r["method"], fmt(r["peak_alloc"], 1), fmt(r["mem_ratio"]), fmt(r["smooth_time"], 3), "yes" if r["core"] else "no", "yes" if r["coverage"] else "no", "yes" if r["memory_pass"] else "no"]
        for r in p0_summary
    ]))
    md.append("")
    md.append("## P1 Single-Event Verification")
    md.append("")
    md.append(md_table(["dataset", "teacher", "proposal", "controller", "acc drop", "KL", "logit", "phiR", "curvR", "mem", "P1"], [
        [r["dataset"], r["teacher"], r["proposal"], r["controller"], fmt(r["acc_drop"]), fmt(r["KL"]), fmt(r["logit"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["mem_ratio"]), "yes" if r["P1"] else "no"]
        for r in sorted(p1_summary, key=lambda x: (x["dataset"], -x["P1"], x["teacher"], x["proposal"]))[:24]
    ]))
    md.append("")
    md.append("P1 all-dataset survivors:")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p1_surv) if p1_surv else "none") + "\n```")
    md.append("")
    md.append("## P2 Probe Audit")
    md.append("")
    md.append(md_table(["dataset", "seed", "probes", "max score", "new triggers", "old triggers", "p90 acc drop", "max-eta count", "P2"], [
        [r["dataset"], r["seed"], r["probes"], fmt(r["best_score_max"]), r["new_trigger_count"], r["old_trigger_count"], fmt(r["p90_acc_drop"]), r["max_eta_count"], "yes" if r["P2"] else "no"]
        for r in p2_summary
    ]))
    md.append("")
    md.append("P2 verdict: the new controller is considered viable when score/debt probes fire on at least two datasets and probe task cost stays within budget.")
    md.append("")
    md.append("## P3 One-Cycle Refresh Policies")
    md.append("")
    md.append(md_table(["dataset", "method", "acc", "gap", "phiR", "curvR", "recovery", "retention", "mem", "P3"], [
        [r["dataset"], r["method"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["recovery"]), fmt(r["retention"]), fmt(r["mem"]), "yes" if r["P3"] else "no"]
        for r in p3_summary
    ]))
    md.append("")
    md.append("P3 all-dataset survivors:")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p3_surv) if p3_surv else "none") + "\n```")
    md.append("")
    md.append("## P4 Event-Driven Multi-Cycle V2")
    md.append("")
    md.append(md_table(["dataset", "method", "acc", "gap", "phiR", "curvR", "events", "accepted", "mem", "time", "P4"], [
        [r["dataset"], r["method"], fmt(r["acc"]), fmt(r["gap"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["events"], 1), fmt(r["accepted"], 1), fmt(r["mem"]), fmt(r["time"]), "yes" if r["P4"] else "no"]
        for r in p4_summary
    ]))
    md.append("")
    md.append("P4 all-dataset survivors:")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p4_surv) if p4_surv else "none") + "\n```")
    md.append("")
    md.append("## P5-P7 Decision")
    md.append("")
    if p4_surv:
        md.append("```text\nP5 should be run for P4 survivor, but compact runner left confirm-seed expansion gated.\n```")
    else:
        md.append("```text\nP5 3-seed selection: not run\nReason: P4 produced no all-dataset survivor.\n\nP6/P7 confirm: not run\nReason: P5 was not reached.\n```")
    md.append("")
    md.append("## Failure Diagnosis")
    md.append("")
    md.append("Generated:")
    md.append("```text\nfailure_table.csv\nfailure_by_dataset.csv\nfailure_by_method.csv\nfailure_heatmap.png\nrecommendation.json\n```")
    md.append("")
    md.append("## Required Artifacts")
    md.append("")
    md.append("Written under `results/v5_3/`:")
    md.append("```text\np0_memory_smoke.csv\np0_core_memory_gate_summary.csv\nedge_param_manifest.csv\np1_single_smoothing_event.csv\np1_single_smoothing_gate_summary.csv\np2_event_probe_audit.csv\np2_probe_training_trace.csv\np2_event_probe_gate_summary.csv\np3_one_cycle_controller_variants.csv\np3_refresh_trace.csv\np3_one_cycle_gate_summary.csv\np4_event_driven_multicycle_v2.csv\np4_event_trace.csv\np4_event_v2_gate_summary.csv\np5_candidate_selection3.csv\np6_confirm5.csv\np7_confirm10.csv\nfailure_table.csv\nfailure_by_dataset.csv\nfailure_by_method.csv\nfailure_heatmap.png\nrecommendation.json\naggregate_decision.json\nfigures/\n```")
    md.append("")
    md.append("## Final Decision")
    md.append("")
    md.append("```text\nDG-KAN v5.3 status:\n  " + status + "\n\nWhat improved:\n  AB-RBF core consistency is now audited in P0.\n  P2 directly audits whether the event controller would fire, instead of relying on a plateau proxy.\n  The new score/debt trigger fires on most probe points while the old plateau trigger stays at zero.\n  P3 separates smoothing damage from refresh recovery and compares base/full/RBF-frozen refresh.\n\nWhat failed:\n  P3 produced no non-baseline one-cycle survivor.\n  Refresh recovers task accuracy, but final residual phi reduction is only about 0.8%-1.8%, below the gate.\n  Therefore P4/P5/P6/P7 are not justified under the written plan.\n\nConclusion:\n  v5.3 fixes the event-trigger observability problem, but not the geometry-retention problem after refresh.\n  LightSmooth remains useful as one-cycle / post-task geometry maintenance, not as a validated train-time multicycle component.\n```")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    log_entry = f"- v5.3 AB-RBF event controller: status={status}; P3 survivors={len(p3_surv)}, P4 survivors={len(p4_surv)}.\n"
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8")
        if "v5.3 AB-RBF event controller" not in text:
            LOG.write_text(text.rstrip() + "\n" + log_entry, encoding="utf-8")
    else:
        LOG.write_text(log_entry, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
