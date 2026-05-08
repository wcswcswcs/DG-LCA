#!/usr/bin/env python3
"""Summarize DG-KAN v5.2 memory-budgeted residual smoothing experiments."""

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
BASE = ROOT / "results/v5_2"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_结果复盘.md"
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


def avg(rows: Sequence[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return statistics.mean(vals) if vals else math.nan


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
    width = 840
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
        bar = 0.0 if not math.isfinite(value) or max_v <= 0 else 480 * value / max_v
        lines.append(f'<text x="18" y="{y+14}" font-family="Arial" font-size="11">{label}</text>')
        lines.append(f'<rect x="290" y="{y}" width="{bar:.1f}" height="18" fill="#2563eb"/>')
        lines.append(f'<text x="{300+bar:.1f}" y="{y+14}" font-family="Arial" font-size="11">{fmt(value,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_scatter(path: Path, title: str, rows: Sequence[dict], x_key: str, y_key: str, label_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 460
    xs = [f(r, x_key) for r in rows if math.isfinite(f(r, x_key)) and math.isfinite(f(r, y_key))]
    ys = [f(r, y_key) for r in rows if math.isfinite(f(r, x_key)) and math.isfinite(f(r, y_key))]
    xmin, xmax = (min(xs or [0]), max(xs or [1]))
    ymin, ymax = (min(ys or [0]), max(ys or [1]))
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
        '<line x1="70" y1="390" x2="680" y2="390" stroke="#111"/>',
        '<line x1="70" y1="390" x2="70" y2="60" stroke="#111"/>',
    ]
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    for i, row in enumerate(rows):
        x = f(row, x_key)
        y = f(row, y_key)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        px = 70 + 610 * (x - xmin) / (xmax - xmin)
        py = 390 - 330 * (y - ymin) / (ymax - ymin)
        color = colors[i % len(colors)]
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity="0.8"/>')
        if i < 16:
            label = str(row.get(label_key, ""))[:22]
            lines.append(f'<text x="{px+7:.1f}" y="{py+4:.1f}" font-family="Arial" font-size="9">{label}</text>')
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
    p2 = read_rows(BASE / "p2_one_cycle_train_smooth_refresh.csv")
    p3 = read_rows(BASE / "p3_event_driven_multicycle.csv")
    p4 = read_rows(BASE / "p4_candidate_selection3.csv")
    p5 = read_rows(BASE / "p5_confirm5.csv")
    p6 = read_rows(BASE / "p6_confirm10.csv")
    p7 = read_rows(BASE / "p7_cifar_small_memory_precheck.csv")
    failures = read_rows(BASE / "failure_table.csv")

    p0_summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped(p0, "dataset", "method").items()):
        p0_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "peak_alloc": avg(rows, "peak_cuda_allocated_mb"),
                "peak_reserved": avg(rows, "peak_cuda_reserved_mb"),
                "teacher_cache": avg(rows, "teacher_cache_mb"),
                "proposal_temp": avg(rows, "proposal_temp_mb"),
                "smooth_time": avg(rows, "smoothing_time_sec"),
                "mem_ratio": avg(rows, "memory_ratio_vs_ABRBF_AdamW"),
                "time_ratio": avg(rows, "time_ratio_vs_ABRBF_AdamW"),
                "core_pass": int(avg(rows, "learnable_nonKAN_params") == 0 and avg(rows, "edge_coverage") >= 0.999),
                "memory_pass": int("Strong" not in method and avg(rows, "memory_ratio_vs_ABRBF_AdamW") <= 1.25),
            }
        )
    write_csv(BASE / "p0_memory_gate_summary.csv", p0_summary)

    p1_summary: list[dict] = []
    for (teacher, proposal, controller, dataset), rows in sorted(grouped(p1, "teacher", "proposal", "controller", "dataset").items()):
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
                "time_sec": avg(rows, "smoothing_time_sec"),
                "P1": int(max(f(r, "p1_pass", 0) for r in rows) >= 1),
            }
        )
    p1_surv = all_dataset_survivors(p1, ["teacher", "proposal", "controller"], "p1_pass")
    write_csv(BASE / "p1_single_smoothing_gate_summary.csv", p1_summary)

    p2_summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped(p2, "dataset", "method").items()):
        base = [r for r in p2 if r.get("dataset") == dataset and r.get("method") == "ABRBF-AdamW" and not r.get("error")]
        p2_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "final_acc": avg(rows, "final_acc"),
                "gap_vs_ABRBF": avg(base, "final_acc") - avg(rows, "final_acc") if base else math.nan,
                "phiR_red": avg(rows, "phi_rbf_reduction_final"),
                "curvR_red": avg(rows, "curvature_rbf_reduction_final"),
                "refresh_recovery": avg(rows, "refresh_recovery_ratio"),
                "geometry_retention": avg(rows, "geometry_retention_ratio"),
                "mem_ratio": avg(rows, "memory_ratio_vs_ABRBF_AdamW"),
                "smooth_time": avg(rows, "smoothing_time_sec"),
                "P2": int(max(f(r, "p2_pass", 0) for r in rows) >= 1),
            }
        )
    p2_surv = all_dataset_survivors(p2, ["method"], "p2_pass")
    write_csv(BASE / "p2_one_cycle_gate_summary.csv", p2_summary)

    p3_summary: list[dict] = []
    for (dataset, method), rows in sorted(grouped(p3, "dataset", "method").items()):
        p3_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "final_acc": avg(rows, "final_acc"),
                "gap_vs_ABRBF": avg(rows, "acc_gap_vs_ABRBF"),
                "phiR_red": avg(rows, "phi_rbf_reduction"),
                "curvR_red": avg(rows, "curvature_rbf_reduction"),
                "events": avg(rows, "smoothing_event_count"),
                "accepted": avg(rows, "accepted_event_count"),
                "smooth_time_frac": avg(rows, "smoothing_time_fraction"),
                "P3": int(max(f(r, "p3_pass", 0) for r in rows) >= 1),
            }
        )
    p3_surv = all_dataset_survivors(p3, ["method"], "p3_pass")
    write_csv(BASE / "p3_event_gate_summary.csv", p3_summary)

    status = "stop_after_p3_no_multicycle_survivor" if not p3_surv else "has_p3_survivor"
    decision = {
        "version": "v5.2",
        "status": status,
        "p1_survivors": ["|".join(s) for s in p1_surv],
        "p2_survivors": ["|".join(s) for s in p2_surv],
        "p3_survivors": ["|".join(s) for s in p3_surv],
        "rows": {"P0": len(p0), "P1": len(p1), "P2": len(p2), "P3": len(p3), "P4": len(p4), "P5": len(p5), "P6": len(p6), "P7": len(p7), "P8_failures": len(failures)},
    }
    save_json(BASE / "aggregate_decision.json", decision)

    recommendation = {
        "status": status,
        "recommendation": "Keep AB-RBF as the PureKAN default. Light residual smoothing is useful for single/one-cycle maintenance, but current event controller should be redesigned before confirm seeds.",
    }
    save_json(BASE / "recommendation.json", recommendation)

    write_svg_bar(
        FIG / "memory_dashboard.svg",
        "P0 Peak CUDA Allocated MB",
        [f"{r['dataset']} {r['method']}" for r in p0_summary],
        [float(r["peak_alloc"]) for r in p0_summary],
    )
    light_p1 = [r for r in p1_summary if "Strong" not in r["proposal"]]
    write_svg_scatter(FIG / "p1_geometry_cost_pareto.svg", "P1 Geometry-Cost Pareto", light_p1, "mem_ratio", "curvR_red", "proposal")
    write_svg_bar(
        FIG / "p2_refresh_geometry.svg",
        "P2 Geometry Retention",
        [f"{r['dataset']} {r['method'].replace('ABRBF-','')}" for r in p2_summary if r["method"] != "ABRBF-AdamW"],
        [float(r["geometry_retention"]) for r in p2_summary if r["method"] != "ABRBF-AdamW"],
    )
    write_svg_bar(
        FIG / "p3_phi_reduction.svg",
        "P3 Final PhiR Reduction",
        [f"{r['dataset']} {r['method'].replace('ABRBF-','')}" for r in p3_summary],
        [float(r["phiR_red"]) for r in p3_summary],
    )
    write_png_heatmap(BASE / "failure_heatmap.png", failures)

    p0_light = [r for r in p0_summary if "LightSmooth" in r["method"]]
    p0_strong = [r for r in p0_summary if "StrongSmooth" in r["method"]]
    p1_top = sorted(p1_summary, key=lambda r: (-r["P1"], r["dataset"], r["teacher"], r["proposal"]))[:18]
    p2_top = sorted(p2_summary, key=lambda r: (r["dataset"], r["method"]))
    p3_top = sorted(p3_summary, key=lambda r: (r["dataset"], r["method"]))

    md: list[str] = []
    md.append("# DG-KAN v5.2 Memory-Budgeted Residual Smoothing 结果复盘")
    md.append("")
    md.append("本轮依据 `docs/DG-KAN_v5.2_MemoryBudgetedResidualSmoothing_实验计划.md`。目标是保留 AB-RBF 作为 PureKAN 默认 edge primitive，同时把 v5.1 的 strong residual smoothing acceptor 降级为低频、轻量、显存预算内的 geometry maintenance。")
    md.append("")
    md.append("## Run Inventory")
    md.append("")
    md.append(md_table(["stage", "rows", "errors"], [
        ["P0 memory smoke", len(p0), sum(1 for r in p0 if r.get("error"))],
        ["P1 single smoothing", len(p1), sum(1 for r in p1 if r.get("error"))],
        ["P2 one-cycle", len(p2), sum(1 for r in p2 if r.get("error"))],
        ["P3 event-driven", len(p3), sum(1 for r in p3 if r.get("error"))],
        ["P4 selection", len(p4), sum(1 for r in p4 if r.get("error"))],
        ["P5 confirm5", len(p5), sum(1 for r in p5 if r.get("error"))],
        ["P6 confirm10", len(p6), sum(1 for r in p6 if r.get("error"))],
        ["P7 CIFAR precheck", len(p7), sum(1 for r in p7 if r.get("error"))],
        ["P8 failures", len(failures), 0],
    ]))
    md.append("")
    md.append("## Code / Config Changes")
    md.append("")
    md.append("```text\nexperiments/run_gafu_v52.py\n  Added memory/time instrumentation for AB-RBF and residual smoothing events.\n  Added light coefficient-only residual smoothing eta search.\n  Added P1 single-event, P2 one-cycle, and P3 event-driven multi-cycle probes.\n  StrongSmooth is retained only as offline reference.\n\nexperiments/analyze_gafu_v52.py\n  Generates memory/gate summaries, failure taxonomy, figures, aggregate decision, and this replay.\n```")
    md.append("")
    md.append("## P0 Memory Smoke")
    md.append("")
    md.append(md_table(["dataset", "method", "peak MB", "mem ratio", "smooth sec", "memory pass"], [
        [r["dataset"], r["method"], fmt(r["peak_alloc"], 1), fmt(r["mem_ratio"]), fmt(r["smooth_time"], 3), "yes" if r["memory_pass"] else "no"]
        for r in p0_summary
    ]))
    md.append("")
    md.append("P0 verdict: core invariants passed. LightSmooth reduced runtime by roughly two orders of magnitude versus StrongSmooth and used far less memory; StrongSmooth remains offline-only. A first CUDA MNIST smoke row is a borderline memory-ratio outlier, but P1/P2 teacher-relative light smoothing stays within budget.")
    md.append("")
    md.append("## P1 Single Smoothing Event")
    md.append("")
    md.append(md_table(["dataset", "teacher", "proposal", "controller", "acc drop", "KL", "logit", "phiR red", "curvR red", "mem", "P1"], [
        [r["dataset"], r["teacher"], r["proposal"], r["controller"], fmt(r["acc_drop"]), fmt(r["KL"]), fmt(r["logit"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["mem_ratio"]), "yes" if r["P1"] else "no"]
        for r in p1_top
    ]))
    md.append("")
    md.append("P1 all-dataset survivors:")
    md.append("")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p1_surv) if p1_surv else "none") + "\n```")
    md.append("")
    md.append("P1 verdict: light smoothing works as a single event. `S0-laplacian` and `S1-sobolev-grad` with logit/score gates preserve task outputs while reducing residual geometry. StrongSmooth does not pass the memory/task gate.")
    md.append("")
    md.append("## P2 One-Cycle Train/Smooth/Refresh")
    md.append("")
    md.append(md_table(["dataset", "method", "acc", "gap", "phiR red", "curvR red", "recovery", "retention", "mem", "P2"], [
        [r["dataset"], r["method"], fmt(r["final_acc"]), fmt(r["gap_vs_ABRBF"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["refresh_recovery"]), fmt(r["geometry_retention"]), fmt(r["mem_ratio"]), "yes" if r["P2"] else "no"]
        for r in p2_top
    ]))
    md.append("")
    md.append("P2 all-dataset survivors:")
    md.append("")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p2_surv) if p2_surv else "none") + "\n```")
    md.append("")
    md.append("P2 verdict: one-cycle light smoothing passes across MNIST, Fashion-MNIST, and KMNIST. Refresh can recover or preserve task performance while keeping around 5% residual phi reduction and large curvature reduction. StrongSmooth remains rejected due memory/time.")
    md.append("")
    md.append("## P3 Event-Driven Multi-Cycle")
    md.append("")
    md.append(md_table(["dataset", "method", "acc", "gap", "phiR red", "curvR red", "events", "accepted", "time frac", "P3"], [
        [r["dataset"], r["method"], fmt(r["final_acc"]), fmt(r["gap_vs_ABRBF"]), fmt(r["phiR_red"]), fmt(r["curvR_red"]), fmt(r["events"], 1), fmt(r["accepted"], 1), fmt(r["smooth_time_frac"]), "yes" if r["P3"] else "no"]
        for r in p3_top
    ]))
    md.append("")
    md.append("P3 all-dataset survivors:")
    md.append("")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p3_surv) if p3_surv else "none") + "\n```")
    md.append("")
    md.append("P3 verdict: no multicycle survivor. Fixed-interval smoothing keeps geometry gains but costs too much accuracy on MNIST/KMNIST. Event-driven smoothing with the written plateau trigger does not fire, so it preserves accuracy but gains no geometry.")
    md.append("")
    md.append("## P4-P7 Decision")
    md.append("")
    md.append("```text\nP4 3-seed selection: not run\nReason: P3 produced no all-dataset survivor.\n\nP5/P6 confirm: not run\nReason: P4 was not reached.\n\nP7 CIFAR-small precheck: not run\nReason: no unified light smoothing candidate to scale-check.\n```")
    md.append("")
    md.append("## Failure Diagnosis")
    md.append("")
    md.append("Generated:")
    md.append("")
    md.append("```text\nfailure_table.csv\nfailure_by_dataset.csv\nfailure_by_method.csv\nfailure_heatmap.png\nrecommendation.json\n```")
    md.append("")
    md.append("Diagnosis:")
    md.append("")
    md.append("```text\nF1 memory too high:\n  StrongSmooth clearly fails. LightSmooth is mostly inside budget, with a borderline P0 smoke outlier.\n\nF3/F4/F5 train-time integration:\n  P2 one-cycle succeeds, but P3 multicycle integration fails.\n\nF7 KMNIST-specific failure:\n  Fixed-interval P3 loses too much accuracy on KMNIST; MNIST also fails.\n\nController diagnosis:\n  The smoothing operator is now cheap and effective locally. The remaining blocker is event scheduling / recovery, not AB-RBF or single-event smoothing.\n```")
    md.append("")
    md.append("## Required Artifacts")
    md.append("")
    md.append("Written under `results/v5_2/`:")
    md.append("")
    md.append("```text\np0_memory_smoke.csv\np0_memory_gate_summary.csv\nedge_param_manifest.csv\np1_single_smoothing_event.csv\np1_single_smoothing_gate_summary.csv\np2_one_cycle_train_smooth_refresh.csv\np2_training_trace.csv\np2_one_cycle_gate_summary.csv\np3_event_driven_multicycle.csv\np3_event_trace.csv\np3_event_gate_summary.csv\np4_candidate_selection3.csv\np5_confirm5.csv\np6_confirm10.csv\np7_cifar_small_memory_precheck.csv\nfailure_table.csv\nfailure_by_dataset.csv\nfailure_by_method.csv\nfailure_heatmap.png\nrecommendation.json\naggregate_decision.json\nfigures/\n```")
    md.append("")
    md.append("## Final Decision")
    md.append("")
    md.append("```text\nDG-KAN v5.2 status:\n  " + status + "\n\nWhat improved:\n  AB-RBF remains the correct PureKAN edge primitive.\n  Light residual smoothing replaces v5.1 strong acceptor for single-event and one-cycle use.\n  Memory/time cost is dramatically lower than StrongSmooth.\n\nWhat failed:\n  Multicycle integration is not solved.\n  Fixed-interval smoothing accumulates task cost.\n  Event-driven smoothing did not trigger under the written plateau rule.\n\nConclusion:\n  v5.2 upgrades residual smoothing from too-heavy strong acceptor to practical one-cycle maintenance.\n  It should not yet be used as a multicycle training component.\n  Next work should redesign event scheduling and refresh policy, not the AB-RBF primitive.\n```")

    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    log_entry = "- v5.2 memory-budgeted residual smoothing: P1/P2 light smoothing passed, P3 multicycle failed; stopped before confirm seeds.\n"
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8")
        if "v5.2 memory-budgeted residual smoothing" not in text:
            LOG.write_text(text.rstrip() + "\n" + log_entry, encoding="utf-8")
    else:
        LOG.write_text(log_entry, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
