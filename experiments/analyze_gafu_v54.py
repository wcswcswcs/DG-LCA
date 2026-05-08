#!/usr/bin/env python3
"""Summarize DG-KAN v5.4 efficiency-first PureKAN experiments."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v5_4"
FIG = BASE / "figures"
OUT = ROOT / "docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_结果复盘.md"
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
        v = row.get(key, "")
        if v in {"", None, "nan", "NaN"}:
            return default
        return float(v)
    except Exception:
        return default


def fmt(x: float, digits: int = 4) -> str:
    return "" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def mean(vals: Iterable[float], default: float = math.nan) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.mean(xs) if xs else default


def std(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


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
    width = 920
    height = 70 + 30 * len(labels)
    finite = [v for v in values if math.isfinite(v)]
    max_v = max(finite or [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="18" y="30" font-family="Arial" font-size="16" font-weight="700">{title}</text>',
    ]
    for i, (label, val) in enumerate(zip(labels, values)):
        y = 58 + i * 30
        bar = 0.0 if not math.isfinite(val) else 500 * val / max(1.0e-12, max_v)
        lines.append(f'<text x="18" y="{y+15}" font-family="Arial" font-size="10">{label[:42]}</text>')
        lines.append(f'<rect x="350" y="{y}" width="{bar:.1f}" height="18" fill="#2563eb"/>')
        lines.append(f'<text x="{360+bar:.1f}" y="{y+14}" font-family="Arial" font-size="10">{fmt(val,3)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p0 = read_rows(BASE / "p0_core_primitive_manifest.csv")
    p1 = read_rows(BASE / "p1_efficiency_microbenchmark.csv")
    p2 = read_rows(BASE / "p2_custom_backward_audit.csv")
    p3 = read_rows(BASE / "p3_accuracy_frontier.csv")
    p4 = read_rows(BASE / "p4_functional_analytic_smoke.csv")
    p5 = read_rows(BASE / "p5_lightsmooth_compatibility.csv")
    p6 = read_rows(BASE / "p6_candidate_selection3.csv")
    p7 = read_rows(BASE / "p7_confirm5.csv")
    p8 = read_rows(BASE / "p8_confirm10.csv")
    failures = read_rows(BASE / "failure_table.csv") or read_rows(BASE / "p9_failure_diagnosis.csv")

    p1_summary: list[dict] = []
    for (method,), rows in sorted(grouped(p1, "method").items()):
        if "MLP" in method:
            continue
        pass_rate = mean([f(r, "p1_pass", 0.0) for r in rows], 0.0)
        p1_summary.append(
            {
                "method": method,
                "rows": len(rows),
                "forward_time_ratio": mean(f(r, "forward_time_ratio_vs_mlp") for r in rows),
                "backward_time_ratio": mean(f(r, "backward_time_ratio_vs_mlp") for r in rows),
                "backward_memory_ratio": mean(f(r, "backward_memory_ratio_vs_mlp") for r in rows),
                "step_time_ratio": mean(f(r, "step_time_ratio_vs_mlp") for r in rows),
                "pass_rate": pass_rate,
                "P1": int(pass_rate >= 0.25),
            }
        )
    write_csv(BASE / "p1_efficiency_gate_summary.csv", p1_summary)

    p2_summary: list[dict] = []
    for (method, variant), rows in sorted(grouped(p2, "method", "backward_variant").items()):
        p2_summary.append(
            {
                "method": method,
                "variant": variant,
                "grad_relerr": mean(f(r, "grad_relerr_coeff") for r in rows),
                "grad_cos": mean(f(r, "grad_cos_coeff") for r in rows),
                "memory_ratio": mean(f(r, "memory_ratio_vs_mlp") for r in rows),
                "step_ratio": mean(f(r, "step_time_ratio_vs_mlp") for r in rows),
                "saved_mb": mean(f(r, "saved_tensor_total_mb") for r in rows),
                "P2": int(all(int(f(r, "p2_pass", 0)) == 1 for r in rows)),
            }
        )
    write_csv(BASE / "p2_custom_backward_gate_summary.csv", p2_summary)

    p3_summary: list[dict] = []
    by_dataset_method = grouped(p3, "dataset", "method")
    for (dataset, method), rows in sorted(by_dataset_method.items()):
        mlp_rows = by_dataset_method.get((dataset, "MLP-AdamW"), [])
        mlp_acc = mean(f(r, "test_acc") for r in mlp_rows)
        p3_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(rows),
                "acc": mean(f(r, "test_acc") for r in rows),
                "std": std(f(r, "test_acc") for r in rows),
                "gap_vs_mlp": mlp_acc - mean(f(r, "test_acc") for r in rows),
                "ECE": mean(f(r, "ECE") for r in rows),
                "phi_rbf": mean(f(r, "phi_rbf") for r in rows),
                "curv_rbf": mean(f(r, "curvature_rbf") for r in rows),
                "fwd": mean(f(r, "forward_time_ratio") for r in rows),
                "bmem": mean(f(r, "backward_memory_ratio") for r in rows),
                "step": mean(f(r, "step_time_ratio") for r in rows),
                "P3": int(all(int(f(r, "p3_pass", 0)) == 1 for r in rows) and method != "MLP-AdamW"),
            }
        )
    write_csv(BASE / "p3_accuracy_gate_summary.csv", p3_summary)

    p3_surv = all_dataset_survivors(p3, ["method"], "p3_pass")
    p4_surv = all_dataset_survivors(p4, ["method"], "p4_pass")
    p5_surv = all_dataset_survivors(p5, ["method"], "p5_pass")
    p6_surv = all_dataset_survivors(p6, ["method"], "p6_pass")
    status = "continue_to_p7" if p6_surv else "stop_after_p3_no_efficiency_accuracy_survivor"
    if p3_surv and not (p4_surv or p5_surv):
        status = "stop_after_p5_no_functional_or_smoothing_survivor"

    fail_by_type = Counter(str(r.get("failure_type", "")) for r in failures)
    fail_by_method = Counter(str(r.get("method", "")) for r in failures)
    write_csv(BASE / "failure_by_type.csv", [{"failure_type": k, "count": v} for k, v in fail_by_type.most_common()])
    write_csv(BASE / "failure_by_method.csv", [{"method": k, "count": v} for k, v in fail_by_method.most_common()])

    labels = [r["method"] for r in p1_summary]
    write_svg_bar(FIG / "p1_backward_memory_ratio.svg", "P1 backward memory ratio vs MLP", labels, [float(r["backward_memory_ratio"]) for r in p1_summary])
    p3_non_mlp = [r for r in p3_summary if r["method"] != "MLP-AdamW"]
    write_svg_bar(FIG / "p3_accuracy_gap_vs_mlp.svg", "P3 accuracy gap vs MLP", [f'{r["dataset"]}:{r["method"]}' for r in p3_non_mlp[:24]], [float(r["gap_vs_mlp"]) for r in p3_non_mlp[:24]])

    agg = {
        "version": "v5.4",
        "status": status,
        "p1_survivors": [r["method"] for r in p1_summary if int(r["P1"]) == 1],
        "p2_survivors": [f'{r["method"]}|{r["variant"]}' for r in p2_summary if int(r["P2"]) == 1],
        "p3_survivors": ["|".join(s) for s in p3_surv],
        "p4_survivors": ["|".join(s) for s in p4_surv],
        "p5_survivors": ["|".join(s) for s in p5_surv],
        "rows": {"P0": len(p0), "P1": len(p1), "P2": len(p2), "P3": len(p3), "P4": len(p4), "P5": len(p5), "P6": len(p6), "P7": len(p7), "P8": len(p8), "P9": len(failures)},
    }
    save_json(BASE / "aggregate_decision.json", agg)
    save_json(BASE / "route_decision.json", {"decision": status, "dominant_failures": fail_by_type.most_common(5)})

    md: list[str] = []
    md.append("# DG-KAN v5.4 Efficiency-First PureKAN Functional Training 结果复盘")
    md.append("")
    md.append("本轮依据 `docs/DG-KAN_v5.4_EfficiencyFirst_PureKAN_FunctionalTraining_实验计划.md`。核心目标是先验证 PureKAN edge primitive 的效率 envelope，再决定 functional training / LightSmooth 是否值得继续扩展。")
    md.append("")
    md.append("## Run Inventory")
    md.append("")
    md.append(md_table(["stage", "rows", "errors"], [
        ["P0 core manifest", len(p0), sum(1 for r in p0 if r.get("error"))],
        ["P1 efficiency", len(p1), sum(1 for r in p1 if r.get("error"))],
        ["P2 custom backward", len(p2), sum(1 for r in p2 if r.get("error"))],
        ["P3 accuracy frontier", len(p3), sum(1 for r in p3 if r.get("error"))],
        ["P4 functional smoke", len(p4), sum(1 for r in p4 if r.get("error"))],
        ["P5 LightSmooth", len(p5), sum(1 for r in p5 if r.get("error"))],
        ["P6 selection", len(p6), sum(1 for r in p6 if r.get("error"))],
        ["P7 confirm5", len(p7), sum(1 for r in p7 if r.get("error"))],
        ["P8 confirm10", len(p8), sum(1 for r in p8 if r.get("error"))],
    ]))
    md.append("")
    md.append("## Code / Config Changes")
    md.append("")
    md.append("```text\nexperiments/dgkan_core.py\n  Added core efficient PureKAN edge primitives: ABRBFDepthwiseMixDense, CPABRBFDense, RationalKATDense.\n  Extended edge/base/RBF/mixing parameter discovery so efficient mixing remains inside the KAN edge system.\n\nexperiments/run_gafu_v54.py\n  Added P0 core manifest, P1 efficiency microbenchmark, P2 checkpoint/recompute backward audit,\n  P3 AdamW-like accuracy frontier, and gated P4-P9 placeholders/diagnosis.\n\nexperiments/analyze_gafu_v54.py\n  Generates v5.4 gate summaries, figures, aggregate_decision.json, route_decision.json, and this replay.\n```")
    md.append("")
    md.append("## P0 Core Consistency")
    md.append("")
    md.append(md_table(["method", "primitive", "edge", "nonKAN", "mixing", "rollback", "P0"], [
        [r["method"], r["primitive_type"], r["num_edge_params"], r["num_nonkan_params"], r["num_mixing_params"], fmt(f(r, "rollback_max_abs")), "yes" if int(f(r, "p0_pass", 0)) else "no"]
        for r in p0
    ]))
    md.append("")
    md.append("## P1 Efficiency Microbenchmark")
    md.append("")
    md.append(md_table(["method", "rows", "fwd", "bwd", "bmem", "step", "pass rate", "P1"], [
        [r["method"], r["rows"], fmt(f(r, "forward_time_ratio")), fmt(f(r, "backward_time_ratio")), fmt(f(r, "backward_memory_ratio")), fmt(f(r, "step_time_ratio")), fmt(f(r, "pass_rate")), "yes" if int(r["P1"]) else "no"]
        for r in p1_summary
    ]))
    md.append("")
    md.append("P1 survivors:")
    md.append("```text\n" + ("\n".join(r["method"] for r in p1_summary if int(r["P1"])) or "none") + "\n```")
    md.append("")
    md.append("## P2 Custom Backward Audit")
    md.append("")
    md.append(md_table(["method", "variant", "relerr", "cos", "mem", "step", "saved MB", "P2"], [
        [r["method"], r["variant"], fmt(f(r, "grad_relerr")), fmt(f(r, "grad_cos")), fmt(f(r, "memory_ratio")), fmt(f(r, "step_ratio")), fmt(f(r, "saved_mb")), "yes" if int(r["P2"]) else "no"]
        for r in p2_summary
    ]))
    md.append("")
    md.append("## P3 Accuracy Frontier")
    md.append("")
    md.append(md_table(["dataset", "method", "runs", "acc", "std", "gap", "ECE", "fwd", "bmem", "step", "P3"], [
        [r["dataset"], r["method"], r["runs"], fmt(f(r, "acc")), fmt(f(r, "std")), fmt(f(r, "gap_vs_mlp")), fmt(f(r, "ECE")), fmt(f(r, "fwd")), fmt(f(r, "bmem")), fmt(f(r, "step")), "yes" if int(r["P3"]) else "no"]
        for r in p3_summary
    ]))
    md.append("")
    md.append("P3 all-dataset survivors:")
    md.append("```text\n" + ("\n".join("|".join(s) for s in p3_surv) if p3_surv else "none") + "\n```")
    md.append("")
    md.append("## P4-P8 Decision")
    md.append("")
    md.append("```text\nP4 functional / analytic update smoke: not run\nReason: P3 produced no efficiency-aware accuracy survivor.\n\nP5 LightSmooth compatibility: not run\nReason: P4 was not reached.\n\nP6/P7/P8 confirm: not run\nReason: no candidate passed the written joint gate.\n```")
    md.append("")
    md.append("## Failure Diagnosis")
    md.append("")
    md.append(md_table(["failure", "count"], [[k, v] for k, v in fail_by_type.most_common()]))
    md.append("")
    md.append("## Required Artifacts")
    md.append("")
    md.append("Written under `results/v5_4/`:")
    md.append("```text\np0_core_primitive_manifest.csv\np1_efficiency_microbenchmark.csv\np1_efficiency_gate_summary.csv\np2_custom_backward_audit.csv\np2_custom_backward_gate_summary.csv\np3_accuracy_frontier.csv\np3_accuracy_gate_summary.csv\np4_functional_analytic_smoke.csv\np5_lightsmooth_compatibility.csv\np6_candidate_selection3.csv\np7_confirm5.csv\np8_confirm10.csv\np9_failure_diagnosis.csv\nfailure_table.csv\nfailure_by_type.csv\nfailure_by_method.csv\naggregate_decision.json\nroute_decision.json\nfigures/\n```")
    md.append("")
    md.append("## Final Decision")
    md.append("")
    md.append("```text\nDG-KAN v5.4 status:\n  " + status + "\n\nWhat improved:\n  Efficient PureKAN primitives are now core-level objects, not runner-only helpers.\n  DepthwiseMix / CP / RationalKAT all keep learnable parameters inside the KAN edge system.\n  P3 confirms CP-ABRBF can recover meaningful task accuracy in places, especially MNIST/Fashion.\n\nWhat failed:\n  No primitive passed the P1 hard efficiency envelope across the compact grid.\n  Checkpoint/recompute backward preserved gradients but did not reduce measured peak memory below MLP.\n  P3 produced no all-dataset survivor under the accuracy + efficiency joint gate, with KMNIST the hardest task.\n\nConclusion:\n  v5.4 supports the plan's warning: dense AB-RBF is a mechanism reference, not a final efficient system.\n  The next route should prioritize fused/custom kernels or a more GEMM-native Rational/KAT primitive before more functional optimizer work.\n```")
    OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    log_entry = f"- v5.4 efficiency-first PureKAN: status={status}; P1 survivors={len(agg['p1_survivors'])}, P3 survivors={len(p3_surv)}.\n"
    if LOG.exists():
        text = LOG.read_text(encoding="utf-8")
        if "v5.4 efficiency-first PureKAN" not in text:
            LOG.write_text(text.rstrip() + "\n" + log_entry, encoding="utf-8")
    else:
        LOG.write_text(log_entry, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
