#!/usr/bin/env python3
"""F22 rotated/cautious matrix source-state summary and route update."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_01_common import (  # noqa: E402
    PYTHON,
    V2101_RECAP_DOC,
    append_exec,
    append_text,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    read_json,
    read_rows,
    retention_ratio,
    write_json,
    write_rows,
)


F22_IDS = {
    "MLP-F22-rotated-cautious-matrix-source": "F22 rotated/cautious matrix slow state",
    "MLP-F1-M2-strong-source": "M2 momentum early-source reference",
    "MLP-F7-M2-source-with-matrix-block-retention": "M46 prior matrix-block retention reference",
}
CONTROL_IDS = {
    "CTRL-SGD": "SGD control",
    "CTRL-RandomMatchedNorm": "Random matched norm",
    "CTRL-NoOpMatchedOverhead": "NoOp matched overhead",
}
HORIZONS = (800, 1600, 2400, 3200, 4800, 6400)
F22_ARTIFACTS = [
    "v21_01_f22_rotated_cautious_summary.csv",
    "v21_01_f22_rotated_cautious_decision.csv",
    "v21_01_f22_rotated_cautious_examples.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-prefix", default="v2101_f22_")
    p.add_argument("--no-log", action="store_true")
    return p


def source(row: dict[str, str], h: int) -> float:
    return finite_float(row.get(f"source_h{h}"))


def mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else ""


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    return out


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (run_label, vid), group in sorted(grouped(rows, ("run_label", "v21_id")).items()):
        item: dict[str, Any] = {
            "run_label": run_label,
            "v21_id": vid,
            "condition": F22_IDS.get(vid, CONTROL_IDS.get(vid, "")),
            "rows": len(group),
            "blocked_rows": sum(1 for r in group if str(r.get("execution_status", "")).startswith("blocked")),
            "candidate": int(vid == "MLP-F22-rotated-cautious-matrix-source"),
            "reference": int(vid in F22_IDS and vid != "MLP-F22-rotated-cautious-matrix-source"),
            "control": int(vid in CONTROL_IDS or vid.startswith("CTRL")),
            "washout_rows": sum(int_flag(r.get("washout_flag")) for r in group),
            "late_rebound_rows": sum(int_flag(r.get("late_rebound_flag")) for r in group),
            "retained_rows": sum(int_flag(r.get("retained_flag")) for r in group),
            "gate_accept_h1600_rows": sum(int_flag(r.get("source_state_gate_accept_h1600")) for r in group),
            "current_cos_h1600_mean": mean([finite_float(r.get("source_state_current_cos_h1600")) for r in group]),
            "linec_accept_rate_mean": mean([finite_float(r.get("LineC_filter_accept_rate")) for r in group]),
        }
        for h in HORIZONS:
            vals = [source(r, h) for r in group]
            item[f"source_h{h}_mean"] = mean(vals)
            item[f"source_h{h}_positive_rows"] = sum(1 for v in vals if math.isfinite(v) and v >= 0.005)
        item["retention_h1600_over_h800"] = retention_ratio(item.get("source_h1600_mean"), item.get("source_h800_mean"))
        item["retention_h3200_over_h1600"] = retention_ratio(item.get("source_h3200_mean"), item.get("source_h1600_mean"))
        item["retention_h4800_over_h3200"] = retention_ratio(item.get("source_h4800_mean"), item.get("source_h3200_mean"))
        item["early_chain_group"] = int(finite_float(item.get("source_h800_mean")) >= 0.005 and finite_float(item.get("source_h1600_mean")) >= 0.005)
        item["productive_h3200_group"] = int(
            int_flag(item.get("early_chain_group"))
            and finite_float(item.get("source_h3200_mean")) >= 0.005
            and finite_float(item.get("retention_h3200_over_h1600"), 0.0) >= 0.50
        )
        item["productive_h4800_group"] = int(
            int_flag(item.get("productive_h3200_group"))
            and finite_float(item.get("source_h4800_mean")) >= 0.005
            and finite_float(item.get("retention_h4800_over_h3200"), 0.0) >= 0.50
        )
        out.append(item)
    return out


def append_unique_manifest(out_dir: Path, artifacts: list[str]) -> None:
    rows = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    seen = {r.get("artifact") for r in rows}
    for artifact in artifacts:
        if artifact not in seen:
            rows.append({"artifact": artifact, "exists": int((out_dir / artifact).exists()), "required": 1})
        else:
            for row in rows:
                if row.get("artifact") == artifact:
                    row["exists"] = int((out_dir / artifact).exists())
                    row["required"] = 1
    write_rows(out_dir / "v21_01_required_artifact_manifest.csv", rows)
    write_rows(out_dir / "required_manifest.csv", rows)


def render_section(decision: dict[str, Any], summary: list[dict[str, Any]], examples: list[dict[str, str]]) -> str:
    rows = "\n".join(
        "| {condition} | {run_label} | {v21_id} | {rows} | {blocked_rows} | {source_h800_mean} | {source_h1600_mean} | {source_h3200_mean} | {source_h4800_mean} | {source_h6400_mean} | {early_chain_group} | {productive_h3200_group} | {gate_accept_h1600_rows} | {current_cos_h1600_mean} |".format(**r)
        for r in summary
    )
    example_rows = "\n".join(
        "| {run_label} | {v21_id} | {dataset} | {seed} | {source_h800} | {source_h1600} | {source_h3200} | {source_h4800} | {source_h6400} | {washout_flag} | {late_rebound_flag} | {retained_flag} |".format(**r)
        for r in examples[:36]
    )
    return f"""
## 2026-06-04 F22 Rotated/Cautious Matrix Source-State

### 是否达成 v21.01 目标

没有达成。

- route: `{decision.get('route')}`
- promotion_allowed: {decision.get('promotion_allowed')}
- F22 rows / blocked rows: {decision.get('f22_rows')} / {decision.get('f22_blocked_rows')}
- F22 smoke rows / full rows: {decision.get('f22_smoke_rows')} / {decision.get('f22_full_rows')}
- F22 candidate early-chain groups: {decision.get('f22_candidate_early_chain_groups')}
- F22 candidate productive h3200 groups: {decision.get('f22_candidate_productive_h3200_groups')}
- full escalation recommended: {decision.get('f22_full_escalation_recommended')}

F22 是 F21 optimizer-dynamics decoupling 失败后的 F1-H/F1-I 实现：用每个矩阵参数的 low-rank SVD 分量维护 slow source state，并对与当前 low-rank 分量冲突的坐标做 cautious downweight，而不是 hard mask。方向仍只来自 train gradients；validation/source readback 只用于实验评估。

### F22 Group Evidence

| condition | run label | v21_id | rows | blocked | h800 | h1600 | h3200 | h4800 | h6400 | early chain | productive h3200 | gate accepts h1600 | current cos h1600 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

### F22 Row Examples

| run label | v21_id | dataset | seed | h800 | h1600 | h3200 | h4800 | h6400 | washout | late | retained |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{example_rows}

### 修改记录

- 修改 `dgkan/fu/mechanisms.py`：新增 `M69-RotatedCautiousMatrixSlowFU`，first-step/audit update 使用 low-rank matrix SVD slow state，并声明 matrix-block semantic contract。
- 修改 `experiments/run_v17_common.py`：新增 M69 训练分支；先保留 momentum primary early source，再维护 low-rank matrix slow state，对冲突分量做 cautious downweight，并记录 gate/cos/density/gain 诊断。
- 修改 `experiments/run_v21_common.py` 与 `experiments/run_v21_01_source_retention.py`：新增 `MLP-F22-rotated-cautious-matrix-source` spec 和 v21.01 白名单。
- 新增 `experiments/run_v21_01_f22_rotated_cautious_summary.py`，只从 fresh `v2101_f22_` rows 汇总 F22 evidence、更新 route/manifest/复盘。

### F22 结论 / Insight

- F22 不是 breakthrough：没有形成 productive h3200 group。
- 如果 `full escalation recommended=0`，说明 rotated/cautious matrix state 连 h800/h1600 grouped early-chain 都没有打开，不能升级 full。
- 如果 full rows > 0 但 productive h3200 仍为 0，说明 matrix-rotated cautious state 仍不能把 early source 转成 retained source。
- 当前仍不能写 promotion；`promotion_allowed` 保持 0。
"""


def replace_section(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "\n## 2026-06-04 F22 Rotated/Cautious Matrix Source-State"
    idx = text.find(marker)
    if idx >= 0:
        path.write_text(text[:idx].rstrip() + "\n" + section, encoding="utf-8")
    else:
        append_text(path, section)


def main() -> int:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows = [r for r in read_rows(out_dir / "v21_01_source_retention_matrix.csv") if str(r.get("run_label", "")).startswith(str(args.run_prefix))]
    summary = summarize(rows)
    examples = [r for r in rows if r.get("v21_id") in F22_IDS or r.get("v21_id") in CONTROL_IDS]
    candidates = [r for r in summary if int_flag(r.get("candidate"))]
    smoke = [r for r in rows if "smoke" in str(r.get("run_label", ""))]
    full = [r for r in rows if "full" in str(r.get("run_label", ""))]
    candidate_early = sum(int_flag(r.get("early_chain_group")) for r in candidates)
    candidate_h3200 = sum(int_flag(r.get("productive_h3200_group")) for r in candidates)
    full_recommended = int(bool(smoke) and not full and candidate_early > 0)
    if candidate_h3200 > 0:
        route_label = "F22RotatedCautiousCandidateNeedsConfirmation"
    elif full:
        route_label = "F22RotatedCautiousNoRetained"
    elif full_recommended:
        route_label = "F22RotatedCautiousSmokeNeedsFull"
    else:
        route_label = "F22RotatedCautiousSmokeNoEarlyChain"
    decision = {
        "decision": route_label,
        "f22_rows": len(rows),
        "f22_blocked_rows": sum(1 for r in rows if str(r.get("execution_status", "")).startswith("blocked")),
        "f22_smoke_rows": len(smoke),
        "f22_full_rows": len(full),
        "f22_candidate_groups": len(candidates),
        "f22_candidate_early_chain_groups": candidate_early,
        "f22_candidate_productive_h3200_groups": candidate_h3200,
        "f22_full_escalation_recommended": full_recommended,
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v21_01_f22_rotated_cautious_summary.csv", summary)
    write_rows(out_dir / "v21_01_f22_rotated_cautious_examples.csv", examples)
    write_rows(out_dir / "v21_01_f22_rotated_cautious_decision.csv", [decision])

    route = read_json(out_dir / "v21_01_route_decision.json")
    route.update(
        {
            "route": f"R2-LateReboundNoContinuousRetention-{route_label}",
            "promotion_allowed": 0,
            "f22_rows": decision["f22_rows"],
            "f22_blocked_rows": decision["f22_blocked_rows"],
            "f22_candidate_early_chain_groups": candidate_early,
            "f22_candidate_productive_h3200_groups": candidate_h3200,
            "f22_full_escalation_recommended": full_recommended,
            "rotated_cautious_matrix_state_completed": int(not full_recommended),
        }
    )
    write_json(out_dir / "v21_01_route_decision.json", route)
    write_json(out_dir / "route_decision.json", route)
    decision = {**decision, "route": route.get("route", ""), "promotion_allowed": route.get("promotion_allowed", 0)}
    append_unique_manifest(out_dir, F22_ARTIFACTS)
    replace_section(V2101_RECAP_DOC, render_section(decision, summary, examples))
    if not args.no_log:
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v21_01_f22_rotated_cautious_summary.py --out-dir {out_dir}",
            status="completed",
            note=(
                f"rows={decision['f22_rows']} blocked={decision['f22_blocked_rows']} "
                f"early={candidate_early} productive_h3200={candidate_h3200} "
                f"full_recommended={full_recommended} promotion=0"
            ),
        )
    manifest = read_rows(out_dir / "v21_01_required_artifact_manifest.csv")
    required = [str(r.get("artifact")) for r in manifest if int_flag(r.get("required", 1))]
    build_packet(out_dir, required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
