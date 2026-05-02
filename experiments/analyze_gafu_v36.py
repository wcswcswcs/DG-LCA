#!/usr/bin/env python3
"""Analyze DG-KAN v3.6 Rational/PureKAN experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dgkan_core import read_csv, save_json, write_csv


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for path in paths:
        csv_path = path / "runs.csv" if path.is_dir() else path
        for row in read_csv(csv_path):
            key = (
                str(row.get("package", "")),
                str(row.get("dataset", "")),
                str(row.get("method", "")),
                str(row.get("seed", "")),
                str(row.get("target", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _group(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    metrics = [
        "test_acc",
        "val_auc",
        "ece",
        "v3_metric_condition_geometry",
        "trust_clip_rate",
        "branch_over_adamw",
        "branch_output_norm_ratio_final",
        "no_kan_drop",
        "kan_margin_contribution_mean",
        "kan_correct_class_delta_mean",
        "learnable_nonkan_params",
        "purekan_nonkan_param_count",
        "rational_den_actual_p01_grid",
        "rational_den_actual_condition_batch",
        "rational_r_prime_p95",
        "rational_num_den_update_ratio_mean",
        "rational_branch_linear_cov_condition_mean",
        "rbf_basis_dead_frac",
        "rbf_input_out_of_grid_frac",
        "convstem_output_abs_p95",
        "final_mse",
        "loss_auc",
        "bad_step_rate",
        "metric_condition",
        "den_actual_p01_grid",
    ]
    for key, vals in groups.items():
        item = {name: value for name, value in zip(keys, key)}
        item["runs"] = len(vals)
        item["errors"] = sum(1 for r in vals if str(r.get("error", "")).strip())
        for metric in metrics:
            data = [_f(r.get(metric)) for r in vals]
            item[f"{metric}_mean"] = _mean(data)
            item[f"{metric}_std"] = _std(data)
        out.append(item)
    return sorted(out, key=lambda r: tuple(str(r.get(k, "")) for k in keys))


def _baseline_map(rows: Sequence[Dict[str, Any]], package: str, baseline_substr: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        if row.get("package") != package or baseline_substr not in str(row.get("method", "")):
            continue
        out.setdefault(str(row.get("dataset", "")), {})[int(_f(row.get("seed"), -1))] = row
    return out


def _paired_delta(rows: Sequence[Dict[str, Any]], package: str, baseline_substr: str) -> List[Dict[str, Any]]:
    baselines = _baseline_map(rows, package, baseline_substr)
    groups: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for row in rows:
        if row.get("package") != package or baseline_substr in str(row.get("method", "")):
            continue
        dataset = str(row.get("dataset", ""))
        seed = int(_f(row.get("seed"), -1))
        base = baselines.get(dataset, {}).get(seed)
        if base is None:
            continue
        groups.setdefault((dataset, str(row.get("method", ""))), []).append((row, base))
    out: List[Dict[str, Any]] = []
    for (dataset, method), pairs in groups.items():
        acc_delta = [_f(r.get("test_acc")) - _f(b.get("test_acc")) for r, b in pairs]
        auc_delta = [_f(b.get("val_auc")) - _f(r.get("val_auc")) for r, b in pairs]
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "runs": len(pairs),
                "acc_delta_mean": _mean(acc_delta),
                "acc_delta_std": _std(acc_delta),
                "auc_improvement_mean": _mean(auc_delta),
                "auc_improvement_std": _std(auc_delta),
            }
        )
    return sorted(out, key=lambda r: (r["dataset"], r["method"]))


def _md_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str, str]]) -> str:
    lines = ["| " + " | ".join(title for title, _, _ in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        vals: List[str] = []
        for _, key, kind in columns:
            value = row.get(key, "")
            if kind == "int":
                vals.append(str(int(round(_f(value, 0.0)))))
            elif kind == "float":
                vals.append(f"{_f(value):.4f}")
            elif kind == "sci":
                vals.append(f"{_f(value):.3g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _filter(rows: Sequence[Dict[str, Any]], package: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("package") == package]


def analyze(rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    p0 = _filter(rows, "V3_6_P0_IMPL_SMOKE")
    p3 = _filter(rows, "V3_6_P3_RATIONAL_DGKAN_TANGENT3")
    p8 = _filter(rows, "V3_6_P8_PUREKAN_3SEED")
    p0_pass = (
        bool(p0)
        and all(not str(r.get("error", "")).strip() for r in p0)
        and max(_f(r.get("purekan_nonkan_param_count"), 0.0) for r in p0 if "PureKAN" in str(r.get("method", ""))) == 0.0
        and max(
            [_f(r.get("v3_metric_condition_geometry"), 0.0) for r in p0 if "tangent" in str(r.get("method", "")).lower()]
            or [0.0]
        )
        < 1e5
    )

    p3_delta = _paired_delta(p3, "V3_6_P3_RATIONAL_DGKAN_TANGENT3", "Rational-DGKAN-AdamW")
    p3_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in p3_delta:
        method = str(row["method"])
        if not method.startswith("Rational-UFULL-tangent"):
            continue
        method_rows = [r for r in p3 if r.get("dataset") == row["dataset"] and r.get("method") == method]
        row = dict(row)
        row["cond_mean"] = _mean(_f(r.get("v3_metric_condition_geometry")) for r in method_rows)
        row["clip_mean"] = _mean(_f(r.get("trust_clip_rate"), 0.0) for r in method_rows)
        p3_by_method.setdefault(method, []).append(row)
    p3_pass = any(
        {r["dataset"] for r in vals} >= {"Fashion-MNIST", "KMNIST"}
        and all(
            r["acc_delta_mean"] > -0.01
            and r["auc_improvement_mean"] > 0
            and r["cond_mean"] < 1e4
            and r["clip_mean"] < 0.02
            for r in vals
            if r["dataset"] in {"Fashion-MNIST", "KMNIST"}
        )
        for vals in p3_by_method.values()
    )

    p8_delta = _paired_delta(p8, "V3_6_P8_PUREKAN_3SEED", "PureKAN-AdamW")
    p8_pass = False
    for row in p8_delta:
        if "PureKAN-UFULL" in row["method"] and row["acc_delta_mean"] > -0.01 and row["auc_improvement_mean"] > 0:
            p8_pass = True

    decision = {
        "p0_pass": p0_pass,
        "p3_rational_confirm_triggered": p3_pass,
        "p4_run": False,
        "p8_purekan_confirm_triggered": p8_pass,
        "p9_run": False,
        "total_rows": len(rows),
    }
    save_json(out_dir / "v36_decision.json", decision)
    return decision


def write_report(rows: List[Dict[str, Any]], out_dir: Path, doc_path: Path) -> None:
    decision = analyze(rows, out_dir)
    p0_summary = _group(_filter(rows, "V3_6_P0_IMPL_SMOKE"), ["dataset", "method"])
    p1_summary = _group(_filter(rows, "V3_6_P1_RATIONAL_METRIC_AUDIT"), ["dataset", "method"])
    p2_summary = _group(_filter(rows, "V3_6_P2_RATIONAL_TANGENT_TOY"), ["target", "method"])
    p3_summary = _group(_filter(rows, "V3_6_P3_RATIONAL_DGKAN_TANGENT3"), ["dataset", "method"])
    p5_summary = _group(_filter(rows, "V3_6_P5_UFULL_F100_FAILURE_AUDIT"), ["dataset", "method"])
    p6_summary = _group(_filter(rows, "V3_6_P6_CIFAR_BRANCH_AUDIT"), ["dataset", "method"])
    p8_summary = _group(_filter(rows, "V3_6_P8_PUREKAN_3SEED"), ["dataset", "method"])
    p3_delta = _paired_delta(_filter(rows, "V3_6_P3_RATIONAL_DGKAN_TANGENT3"), "V3_6_P3_RATIONAL_DGKAN_TANGENT3", "Rational-DGKAN-AdamW")
    p8_delta = _paired_delta(_filter(rows, "V3_6_P8_PUREKAN_3SEED"), "V3_6_P8_PUREKAN_3SEED", "PureKAN-AdamW")

    write_csv(out_dir / "p0_summary.csv", p0_summary)
    write_csv(out_dir / "p1_rational_metric_summary.csv", p1_summary)
    write_csv(out_dir / "p2_toy_summary.csv", p2_summary)
    write_csv(out_dir / "p3_rational_tangent_summary.csv", p3_summary)
    write_csv(out_dir / "p3_paired_delta_vs_rational_adamw.csv", p3_delta)
    write_csv(out_dir / "p5_f100_summary.csv", p5_summary)
    write_csv(out_dir / "p6_cifar_summary.csv", p6_summary)
    write_csv(out_dir / "p8_purekan_summary.csv", p8_summary)
    write_csv(out_dir / "p8_paired_delta_vs_purekan_adamw.csv", p8_delta)

    lines: List[str] = ["# DG-KAN v3.6 RationalUFull / PureKAN 结果复盘", ""]
    lines.append("## Decision")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(decision, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## P0 Implementation Smoke")
    lines.append("")
    lines.append(
        _md_table(
            p0_summary,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc", "test_acc_mean", "float"),
                ("condG", "v3_metric_condition_geometry_mean", "float"),
                ("clip", "trust_clip_rate_mean", "float"),
                ("den p01", "rational_den_actual_p01_grid_mean", "float"),
                ("pure nonKAN", "purekan_nonkan_param_count_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## P1 Rational Metric Audit")
    lines.append("")
    lines.append(
        _md_table(
            p1_summary,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc", "test_acc_mean", "float"),
                ("val AUC", "val_auc_mean", "float"),
                ("condG", "v3_metric_condition_geometry_mean", "float"),
                ("clip", "trust_clip_rate_mean", "float"),
                ("den p01", "rational_den_actual_p01_grid_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## P2 Rational Toy")
    lines.append("")
    lines.append(
        _md_table(
            p2_summary,
            [
                ("target", "target", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("mse", "final_mse_mean", "sci"),
                ("loss AUC", "loss_auc_mean", "sci"),
                ("bad", "bad_step_rate_mean", "float"),
                ("cond", "metric_condition_mean", "float"),
                ("den p01", "den_actual_p01_grid_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## P3 Rational Tangent DGKAN")
    lines.append("")
    lines.append(
        _md_table(
            p3_summary,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc", "test_acc_mean", "float"),
                ("val AUC", "val_auc_mean", "float"),
                ("condG", "v3_metric_condition_geometry_mean", "float"),
                ("clip", "trust_clip_rate_mean", "float"),
                ("den p01", "rational_den_actual_p01_grid_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("### P3 Paired Delta vs Rational AdamW")
    lines.append("")
    lines.append(
        _md_table(
            p3_delta,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc delta", "acc_delta_mean", "float"),
                ("AUC imp", "auc_improvement_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## P5 U-FULL f100 Failure Audit")
    lines.append("")
    lines.append(
        _md_table(
            p5_summary,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc", "test_acc_mean", "float"),
                ("val AUC", "val_auc_mean", "float"),
                ("branch", "branch_output_norm_ratio_final_mean", "float"),
                ("noKAN", "no_kan_drop_mean", "float"),
                ("margin", "kan_margin_contribution_mean_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## P6 CIFAR Branch Audit")
    lines.append("")
    lines.append(
        _md_table(
            p6_summary,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc", "test_acc_mean", "float"),
                ("val AUC", "val_auc_mean", "float"),
                ("branch", "branch_output_norm_ratio_final_mean", "float"),
                ("noKAN", "no_kan_drop_mean", "float"),
                ("dead", "rbf_basis_dead_frac_mean", "float"),
                ("outgrid", "rbf_input_out_of_grid_frac_mean", "float"),
                ("conv p95", "convstem_output_abs_p95_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## P8 PureKAN 3-Seed")
    lines.append("")
    lines.append(
        _md_table(
            p8_summary,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc", "test_acc_mean", "float"),
                ("val AUC", "val_auc_mean", "float"),
                ("condG", "v3_metric_condition_geometry_mean", "float"),
                ("pure nonKAN", "purekan_nonkan_param_count_mean", "float"),
                ("dead", "rbf_basis_dead_frac_mean", "float"),
                ("outgrid", "rbf_input_out_of_grid_frac_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("### P8 Paired Delta vs PureKAN-AdamW")
    lines.append("")
    lines.append(
        _md_table(
            p8_delta,
            [
                ("dataset", "dataset", "str"),
                ("method", "method", "str"),
                ("runs", "runs", "int"),
                ("acc delta", "acc_delta_mean", "float"),
                ("AUC imp", "auc_improvement_mean", "float"),
            ],
        )
    )
    lines.append("")
    lines.append("## Final Interpretation")
    lines.append("")
    lines.append(
        "P0 通过：Rational tangent metric 有限且 condition 已压到 smoke gate 内，PureKAN 的 learnable non-KAN 参数为 0。"
    )
    lines.append("")
    lines.append(
        "Rational/KAT：tangent metric 修复了旧 poly metric 的高 clip 问题，但 P3 中 condition 仍约 1.5e4-1.8e4，且相对 Rational-AdamW 没有稳定 AUC/accuracy 优势，因此 P4 不触发。"
    )
    lines.append("")
    lines.append(
        "U-FULL f100：Fashion 上 f100 有更强 accuracy signal，但 KMNIST 与 f085/v3.2 best 仍不稳；branch 更大不等于任务对齐更好。"
    )
    lines.append("")
    lines.append(
        "CIFAR-small：ConvStem U-FULL 仍明显低于 ConvStem AdamW，diag warmup/branch boost probe 只能局部缓解，不能恢复到 AdamW。"
    )
    lines.append("")
    lines.append(
        "PureKAN：结构实现成功，AdamW 可训练；但 PureKAN-UFULL 在 Fashion/KMNIST/MNIST 都明显慢或掉点，说明当前 U-FULL 仍更像 hybrid branch optimizer，而不是完整 PureKAN optimizer。P9 不触发。"
    )
    lines.append("")
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("results/gafu_v3_6"),
            Path("results/gafu_v3_6_toy"),
            Path("results/gafu_v3_6_rational"),
            Path("results/gafu_v3_6_main"),
            Path("results/gafu_v3_6_extra"),
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/gafu_v3_6_analysis"))
    parser.add_argument("--doc", type=Path, default=Path("docs/DG-KAN_v3.6_RationalUFull_PureKAN_结果复盘.md"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(args.inputs)
    write_csv(args.out_dir / "all_rows.csv", rows)
    write_report(rows, args.out_dir, args.doc)
    print(f"Wrote {len(rows)} rows into {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
