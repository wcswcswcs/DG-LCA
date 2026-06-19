#!/usr/bin/env python3
"""DG-KAN v22.46b debt-controlled signal identification runner.

This runner does not fabricate new experiment rows.  Its first responsibility is
to re-slice v22.46 artifacts under the v22.46b hypotheses:

* debt-orthogonal residual signal,
* KAN margin stability against same-basis controls,
* structural overhead evidence rather than cadence-only evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import py_compile
import shlex
import statistics
import sys
import threading
import time
import traceback
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_43_metric_preserving_continuous_functional_flow_fu as v2243

V2246_ROOT = ROOT / "results/v22_46"
OUT_ROOT = ROOT / "results/v22_46b"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.46b_DebtControlledSignalIdentification_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.46b_DebtControlledSignalIdentification_实验结果复盘.md"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.46b_DebtControlledSignalIdentification_补充计划.md"
LOG_LOCK = threading.RLock()

REPAIR_DIRECTION = V2246_ROOT / "v22_46_repair_support_direction_decomposition.csv"
REPAIR_MATRIX = V2246_ROOT / "v22_46_repair_all_training_matrix.csv"
REPAIR_ROUTE = V2246_ROOT / "v22_46_repair_route.json"
KAN_MATRIX = V2246_ROOT / "v22_46_kan_variant_repair_KAN_vs_MLP_matched_support_matrix.csv"
KAN_RAW_MATRIX = V2246_ROOT / "v22_46_kan_variant_repair_all_training_matrix.csv"
KAN_ROUTE = V2246_ROOT / "v22_46_kan_variant_repair_route.json"
OVERHEAD_MATRIX = V2246_ROOT / "v22_46_repair_overhead_no_debt_matrix.csv"
EMIT50_READOUT_MATRIX = (
    V2246_ROOT
    / "repair_snapshots"
    / "v22_46_repair_all_training_matrix_emit50_supp120_metric120_readoutcap0_02_overconfidence_only_before_emit20.csv"
)
EMIT50_READOUT_ROUTE = (
    V2246_ROOT
    / "repair_snapshots"
    / "v22_46_repair_route_emit50_supp120_metric120_readoutcap0_02_overconfidence_only_before_emit20.json"
)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.46b DebtControlledSignalIdentification 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行命令、输入 artifact、输出文件、状态与 blocker；"
            "posthoc 审计必须标明来源，不冒充新训练实验。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.46b DebtControlledSignalIdentification 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实 artifact；理论代理量必须说明局限；"
            "不编造缺失数据，不把 posthoc proxy 写成 causal proof。\n",
            encoding="utf-8",
        )


def safe_fragment(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)[:180]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(row) for row in rows]
    if fieldnames is None:
        seen: set[str] = set()
        fieldnames = []
        for row in data:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        if not fieldnames:
            fieldnames = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fval(value: Any, default: float | None = None) -> float | None:
    if value in {"", None}:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def iflag(value: Any) -> int:
    if value in {1, "1", True, "true", "True", "yes", "pass"}:
        return 1
    return 0


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(x for x in values if math.isfinite(x))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def mean(values: Iterable[float | None]) -> float | None:
    clean = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.fmean(clean) if clean else None


def median(values: Iterable[float | None]) -> float | None:
    clean = [x for x in values if x is not None and math.isfinite(x)]
    return statistics.median(clean) if clean else None


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: Any = "n/a",
) -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    with LOG_LOCK:
        journal = read_rows(OUT_ROOT / "v22_46b_command_journal.csv")
        journal.append({key: str(value) for key, value in row.items()})
        write_rows(
            OUT_ROOT / "v22_46b_command_journal.csv",
            journal,
            ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"],
        )
        with EXEC_DOC.open("a", encoding="utf-8") as f:
            f.write(f"\n## {row['timestamp']} {task_id}\n\n")
            f.write("```bash\n" + command + "\n```\n\n")
            f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
            if files:
                f.write(f"- files: {files}\n")
            if note:
                f.write(f"- note: {note}\n")


def bind_upstream() -> None:
    v2243.OUT_ROOT = OUT_ROOT
    v2243.CHUNK_ROOT = CHUNK_ROOT
    v2243.LOG_ROOT = LOG_ROOT
    if hasattr(v2243, "FIG_ROOT"):
        v2243.FIG_ROOT = FIG_ROOT
    v2243.EXEC_DOC = EXEC_DOC
    v2243.RECAP_DOC = RECAP_DOC
    v2243.ensure_out = ensure_out
    v2243.append_exec = append_exec


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "_无行。_\n"
    shown = rows[:limit]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown:
        cells = []
        for col in columns:
            text = str(row.get(col, ""))
            if len(text) > 96:
                text = text[:93] + "..."
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(f"\n_只显示前 {limit} 行，共 {len(rows)} 行。_")
    return "\n".join(out) + "\n"


def selected(row: dict[str, str], datasets: set[str], seeds: set[str]) -> bool:
    if datasets and row.get("dataset") not in datasets:
        return False
    if seeds and row.get("seed") not in seeds:
        return False
    return True


def stage_code_audit() -> dict[str, Any]:
    inputs = [
        PLAN_DOC,
        REPAIR_DIRECTION,
        REPAIR_MATRIX,
        REPAIR_ROUTE,
        KAN_MATRIX,
        KAN_RAW_MATRIX,
        KAN_ROUTE,
        OVERHEAD_MATRIX,
        EMIT50_READOUT_MATRIX,
        EMIT50_READOUT_ROUTE,
    ]
    compile_status = "pass"
    compile_error = ""
    try:
        py_compile.compile(__file__, doraise=True)
    except Exception as exc:  # pragma: no cover - written to artifact.
        compile_status = "fail"
        compile_error = repr(exc)
    manifest = []
    for path in inputs:
        rows = read_rows(path) if path.suffix == ".csv" else []
        manifest.append(
            {
                "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path),
                "csv_rows": len(rows) if path.suffix == ".csv" else "",
            }
        )
    write_rows(OUT_ROOT / "v22_46b_input_manifest.csv", manifest)
    result = {
        "timestamp": now_sg(),
        "runner_compile_status": compile_status,
        "runner_compile_error": compile_error,
        "required_inputs": len(inputs),
        "missing_inputs": sum(1 for item in manifest if not item["exists"]),
        "empty_csv_inputs": sum(1 for item in manifest if item["path"].endswith(".csv") and item["csv_rows"] == 0),
        "status": "pass" if compile_status == "pass" and all(item["exists"] for item in manifest) else "fail",
        "claim_limit": "code/input audit only; no experimental success inferred",
    }
    write_json(OUT_ROOT / "v22_46b_code_audit.json", result)
    return result


def stage_debt_orthogonal(datasets: set[str], seeds: set[str]) -> dict[str, Any]:
    rows = [row for row in read_rows(REPAIR_DIRECTION) if selected(row, datasets, seeds)]
    out: list[dict[str, Any]] = []
    for row in rows:
        tau_direction = fval(row.get("tau_direction"))
        tau_support = fval(row.get("tau_support"))
        ece_delta = fval(row.get("ECE_delta_vs_optimizer_baseline"))
        brier_delta = fval(row.get("Brier_delta_vs_optimizer_baseline"))
        tail_q99_delta = fval(row.get("tail_q99_delta_vs_optimizer_baseline"))
        debt_values = [ece_delta, brier_delta, tail_q99_delta]
        debt_available = all(value is not None for value in debt_values)
        debt_penalty = None
        if debt_available:
            debt_penalty = sum(max(float(value), 0.0) for value in debt_values if value is not None)
        proxy = None
        if tau_direction is not None and debt_penalty is not None:
            proxy = tau_direction - debt_penalty
        no_debt = iflag(row.get("no_debt")) or iflag(row.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        overhead = fval(row.get("controller_overhead"))
        beats_control = iflag(row.get("beats_matched_control"))
        direction_positive = tau_direction is not None and tau_direction > 0.0
        debt_orthogonal_positive = proxy is not None and proxy > 0.0
        overhead_le_0p25 = overhead is not None and overhead <= 0.25
        overhead_le_0p35 = overhead is not None and overhead <= 0.35
        route_gate = direction_positive and debt_orthogonal_positive and no_debt and overhead_le_0p25 and beats_control
        exploration_gate = direction_positive and debt_orthogonal_positive and no_debt and overhead_le_0p35 and beats_control
        out.append(
            {
                "part": row.get("part"),
                "mechanism": row.get("mechanism"),
                "recipe": row.get("recipe"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "architecture_key": row.get("architecture_key"),
                "variant": row.get("variant"),
                "control_used": row.get("control_used"),
                "tau_support": tau_support,
                "tau_direction": tau_direction,
                "ECE_delta_vs_optimizer_baseline": ece_delta,
                "Brier_delta_vs_optimizer_baseline": brier_delta,
                "tail_q99_delta_vs_optimizer_baseline": tail_q99_delta,
                "positive_ECE_debt": max(ece_delta or 0.0, 0.0) if ece_delta is not None else "",
                "positive_Brier_debt": max(brier_delta or 0.0, 0.0) if brier_delta is not None else "",
                "positive_tail_q99_debt": max(tail_q99_delta or 0.0, 0.0) if tail_q99_delta is not None else "",
                "debt_penalty_raw_metric_units": debt_penalty,
                "direction_gain_debt_orthogonal_proxy": proxy,
                "proxy_definition": "tau_direction - sum(max(ECE_delta,0), max(Brier_delta,0), max(tail_q99_delta,0))",
                "proxy_claim_limit": "posthoc proxy from held metrics; not gradient-overlap causal proof",
                "official_no_debt": int(no_debt),
                "beats_matched_control": beats_control,
                "controller_overhead": overhead,
                "direction_positive": int(direction_positive),
                "debt_orthogonal_positive": int(debt_orthogonal_positive),
                "overhead_le_0p25": int(overhead_le_0p25),
                "overhead_le_0p35": int(overhead_le_0p35),
                "B1_route_gate": int(route_gate),
                "B1_exploration_gate_0p35": int(exploration_gate),
                "source_artifact": str(REPAIR_DIRECTION.relative_to(ROOT)),
            }
        )
    write_rows(OUT_ROOT / "v22_46b_debt_orthogonal_matrix.csv", out)
    route_gate_rows = sum(iflag(row.get("B1_route_gate")) for row in out)
    exploration_gate_rows = sum(iflag(row.get("B1_exploration_gate_0p35")) for row in out)
    route = {
        "timestamp": now_sg(),
        "route": "B1_debt_orthogonal_opened" if route_gate_rows >= 8 else "not_opened",
        "reason": "B1 requires >=8 official no-debt, overhead<=0.25, beats-control debt-orthogonal rows",
        "rows": len(out),
        "direction_positive_rows": sum(iflag(row.get("direction_positive")) for row in out),
        "debt_orthogonal_positive_rows": sum(iflag(row.get("debt_orthogonal_positive")) for row in out),
        "official_no_debt_rows": sum(iflag(row.get("official_no_debt")) for row in out),
        "beats_matched_control_rows": sum(iflag(row.get("beats_matched_control")) for row in out),
        "overhead_le_0p25_rows": sum(iflag(row.get("overhead_le_0p25")) for row in out),
        "route_gate_rows": route_gate_rows,
        "exploration_gate_rows_0p35": exploration_gate_rows,
        "promotion_allowed": route_gate_rows >= 8,
        "source_artifact": str(REPAIR_DIRECTION.relative_to(ROOT)),
        "claim_limit": "posthoc metric-debt proxy; does not replace future gradient-overlap experiment",
    }
    write_json(OUT_ROOT / "v22_46b_debt_orthogonal_route.json", route)
    return route


def stage_kan_margin(datasets: set[str], seeds: set[str]) -> dict[str, Any]:
    summary_rows = [row for row in read_rows(KAN_MATRIX) if selected(row, datasets, seeds)]
    raw_rows = [row for row in read_rows(KAN_RAW_MATRIX) if selected(row, datasets, seeds)]
    raw_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    controls_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (row.get("dataset", ""), row.get("seed", ""), row.get("architecture_key", ""), row.get("variant", ""))
        nll = fval(row.get("final_NLL"))
        if nll is None:
            continue
        if row.get("row_role") == "signal_candidate" and row.get("control_mode") == "none":
            raw_by_key[key] = row
        elif row.get("row_role") == "control" and row.get("control_mode") not in {"", "none"}:
            controls_by_key.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        key = (row.get("dataset", ""), row.get("seed", ""), row.get("KAN_architecture", ""), row.get("KAN_variant", ""))
        candidate_nll = fval(row.get("KAN_final_NLL"))
        if candidate_nll is None and key in raw_by_key:
            candidate_nll = fval(raw_by_key[key].get("final_NLL"))
        controls = controls_by_key.get(key, [])
        margins: list[float] = []
        for control in controls:
            control_nll = fval(control.get("final_NLL"))
            if candidate_nll is None or control_nll is None:
                continue
            margin = control_nll - candidate_nll
            margins.append(margin)
            control_rows.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "KAN_architecture": key[2],
                    "KAN_variant": key[3],
                    "control_mode": control.get("control_mode"),
                    "KAN_final_NLL": candidate_nll,
                    "control_final_NLL": control_nll,
                    "control_margin_control_minus_KAN": margin,
                    "bootstrap_status": "not_resampled_posthoc_observed_controls_only",
                    "source_artifact": str(KAN_RAW_MATRIX.relative_to(ROOT)),
                }
            )
        margin_min = min(margins) if margins else None
        margin_p10 = percentile(margins, 0.10)
        margin_median = median(margins)
        stable_margin_positive = margin_p10 is not None and margin_p10 > 0.0 and len(margins) >= 3
        no_debt = iflag(row.get("no_ECE_Brier_tail_debt")) or iflag(row.get("derived_no_ECE_Brier_tail_debt_from_raw_metrics"))
        overhead = fval(row.get("controller_overhead_ratio"))
        overhead_le_0p25 = overhead is not None and overhead <= 0.25
        kan_beats_mlp = iflag(row.get("KAN_beats_MLP_matched_support"))
        control_explained = iflag(row.get("ControlExplained"))
        gate = kan_beats_mlp and stable_margin_positive and no_debt and overhead_le_0p25 and not control_explained
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "KAN_architecture": row.get("KAN_architecture"),
                "KAN_variant": row.get("KAN_variant"),
                "matched_MLP_variant": row.get("matched_MLP_variant"),
                "KAN_final_NLL": candidate_nll,
                "MLP_matched_final_NLL": fval(row.get("MLP_matched_final_NLL")),
                "control_count": len(margins),
                "control_margin_min_control_minus_KAN": margin_min,
                "control_margin_p10_control_minus_KAN": margin_p10,
                "control_margin_median_control_minus_KAN": margin_median,
                "KAN_beats_MLP_matched_support": kan_beats_mlp,
                "KAN_beats_best_same_basis_control": iflag(row.get("KAN_beats_best_same_basis_control")),
                "ControlExplained": control_explained,
                "TrueKANGain": iflag(row.get("TrueKANGain")),
                "BothGain": iflag(row.get("BothGain")),
                "official_no_debt": int(no_debt),
                "controller_overhead_ratio": overhead,
                "overhead_le_0p25": int(overhead_le_0p25),
                "KAN_margin_stable_p10_positive": int(stable_margin_positive),
                "B2_route_gate": int(gate),
                "source_artifact": str(KAN_MATRIX.relative_to(ROOT)),
                "claim_limit": "posthoc observed-control margin; bootstrap resampling not yet run",
            }
        )
    write_rows(OUT_ROOT / "v22_46b_kan_margin_stability_matrix.csv", out)
    write_rows(OUT_ROOT / "v22_46b_kan_control_bootstrap.csv", control_rows)
    route_gate_rows = sum(iflag(row.get("B2_route_gate")) for row in out)
    control_explained_rows = sum(iflag(row.get("ControlExplained")) for row in out)
    control_explained_rate = control_explained_rows / len(out) if out else None
    route = {
        "timestamp": now_sg(),
        "route": "B2_kan_margin_opened" if route_gate_rows >= 8 and (control_explained_rate or 1.0) <= 0.20 else "not_opened",
        "reason": "B2 requires >=8 KAN-vs-MLP rows with stable same-basis control p10 margin, no-debt, overhead<=0.25, and ControlExplained<=20%",
        "rows": len(out),
        "observed_control_margin_rows": len(control_rows),
        "KAN_beats_MLP_matched_support_rows": sum(iflag(row.get("KAN_beats_MLP_matched_support")) for row in out),
        "KAN_margin_stable_p10_positive_rows": sum(iflag(row.get("KAN_margin_stable_p10_positive")) for row in out),
        "official_no_debt_rows": sum(iflag(row.get("official_no_debt")) for row in out),
        "overhead_le_0p25_rows": sum(iflag(row.get("overhead_le_0p25")) for row in out),
        "ControlExplained_rows": control_explained_rows,
        "ControlExplained_rate": control_explained_rate,
        "route_gate_rows": route_gate_rows,
        "promotion_allowed": route_gate_rows >= 8 and (control_explained_rate or 1.0) <= 0.20,
        "source_artifact": str(KAN_MATRIX.relative_to(ROOT)),
        "claim_limit": "posthoc observed-control stability only; future true bootstrap/control-bank resampling still needed",
    }
    write_json(OUT_ROOT / "v22_46b_kan_margin_route.json", route)
    return route


def overhead_component_row(row: dict[str, str], source_artifact: str) -> dict[str, Any]:
    full = fval(row.get("full_step_ms"))
    base = fval(row.get("base_optimizer_ms"))
    controller = fval(row.get("controller_ms"))
    state = fval(row.get("state_update_ms"))
    calibration = fval(row.get("calibration_nuisance_grad_ms"))
    mirror = fval(row.get("mirror_grad_ms"))
    basis_jvp = fval(row.get("basis_JVP_time_ms"))
    basis_metric = fval(row.get("basis_metric_update_time_ms"))
    reported = fval(row.get("controller_overhead_ratio"))
    known = [base, controller, state, calibration, mirror, basis_jvp, basis_metric]
    known_sum = sum(value for value in known if value is not None)
    return {
        "source_artifact": source_artifact,
        "run_label": row.get("run_label"),
        "dataset": row.get("dataset"),
        "seed": row.get("seed"),
        "architecture_key": row.get("architecture_key"),
        "variant": row.get("variant"),
        "control_mode": row.get("control_mode"),
        "row_role": row.get("row_role"),
        "recipe": row.get("recipe"),
        "cached_controller_emit_cadence": row.get("cached_controller_emit_cadence"),
        "support_refresh_cadence": row.get("support_refresh_cadence"),
        "metric_refresh_cadence": row.get("metric_refresh_cadence"),
        "calibration_readout_radial_cap": row.get("calibration_readout_radial_cap"),
        "calibration_readout_policy": row.get("calibration_readout_policy"),
        "full_step_ms": full,
        "base_optimizer_ms": base,
        "controller_ms": controller,
        "state_update_ms": state,
        "calibration_nuisance_grad_ms": calibration,
        "mirror_grad_ms": mirror,
        "basis_JVP_time_ms": basis_jvp,
        "basis_metric_update_time_ms": basis_metric,
        "known_component_sum_ms": known_sum,
        "controller_share_of_full_step": controller / full if full and controller is not None else "",
        "known_component_share_of_full_step": known_sum / full if full else "",
        "reported_controller_overhead_ratio": reported,
        "reported_overhead_le_0p25": int(reported is not None and reported <= 0.25),
        "reported_overhead_le_0p35": int(reported is not None and reported <= 0.35),
    }


def stage_overhead(datasets: set[str], seeds: set[str]) -> dict[str, Any]:
    sources = [
        ("current_emit20_repair", REPAIR_MATRIX),
        ("emit50_readout_snapshot", EMIT50_READOUT_MATRIX),
        ("kan_variant_repair", KAN_RAW_MATRIX),
    ]
    component_rows: list[dict[str, Any]] = []
    for label, path in sources:
        for row in read_rows(path):
            if selected(row, datasets, seeds):
                component_rows.append(overhead_component_row(row, label))
    write_rows(OUT_ROOT / "v22_46b_overhead_breakdown_matrix.csv", component_rows)

    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in component_rows:
        key = (
            str(row.get("source_artifact", "")),
            str(row.get("recipe", "")),
            str(row.get("cached_controller_emit_cadence", "")),
            str(row.get("support_refresh_cadence", "")),
            str(row.get("metric_refresh_cadence", "")),
            str(row.get("calibration_readout_policy", "")),
        )
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        ratios = [fval(row.get("reported_controller_overhead_ratio")) for row in rows]
        controller_shares = [fval(row.get("controller_share_of_full_step")) for row in rows]
        summary_rows.append(
            {
                "source_artifact": key[0],
                "recipe": key[1],
                "cached_controller_emit_cadence": key[2],
                "support_refresh_cadence": key[3],
                "metric_refresh_cadence": key[4],
                "calibration_readout_policy": key[5],
                "rows": len(rows),
                "reported_overhead_le_0p25_rows": sum(iflag(row.get("reported_overhead_le_0p25")) for row in rows),
                "reported_overhead_le_0p35_rows": sum(iflag(row.get("reported_overhead_le_0p35")) for row in rows),
                "reported_controller_overhead_ratio_mean": mean(ratios),
                "reported_controller_overhead_ratio_median": median(ratios),
                "reported_controller_overhead_ratio_p90": percentile([x for x in ratios if x is not None], 0.90),
                "controller_share_of_full_step_mean": mean(controller_shares),
                "controller_share_of_full_step_median": median(controller_shares),
            }
        )
    write_rows(OUT_ROOT / "v22_46b_overhead_group_summary.csv", summary_rows)

    repair_route = read_json(REPAIR_ROUTE)
    emit50_route = read_json(EMIT50_READOUT_ROUTE)
    route = {
        "timestamp": now_sg(),
        "route": "not_opened",
        "reason": "posthoc overhead audit found timing components, but no fused/structural controller implementation has been run; cadence-only emit20 already reduced route gates",
        "rows": len(component_rows),
        "groups": len(summary_rows),
        "current_emit20_official_direction_route_gate_rows": repair_route.get("official_direction_route_gate_rows"),
        "current_emit20_exploration_gate_rows_0p35": repair_route.get("official_direction_exploration_gate_rows_0p35"),
        "emit50_readout_snapshot_official_direction_route_gate_rows": emit50_route.get("official_direction_route_gate_rows"),
        "emit50_readout_snapshot_exploration_gate_rows_0p35": emit50_route.get("official_direction_exploration_gate_rows_0p35"),
        "all_rows_reported_overhead_le_0p25": sum(iflag(row.get("reported_overhead_le_0p25")) for row in component_rows),
        "all_rows_reported_overhead_le_0p35": sum(iflag(row.get("reported_overhead_le_0p35")) for row in component_rows),
        "reported_controller_overhead_ratio_mean": mean(fval(row.get("reported_controller_overhead_ratio")) for row in component_rows),
        "controller_share_of_full_step_mean": mean(fval(row.get("controller_share_of_full_step")) for row in component_rows),
        "fused_controller_probe_available": False,
        "promotion_allowed": False,
        "source_artifacts": [label for label, _ in sources],
        "claim_limit": "timing decomposition only; does not claim structural overhead fix",
    }
    write_json(OUT_ROOT / "v22_46b_overhead_route.json", route)
    write_rows(
        OUT_ROOT / "v22_46b_fused_controller_probe.csv",
        [
            {
                "status": "not_run",
                "reason": "v22.46b initial pass is posthoc audit; fused/debt-shared controller implementation not yet added",
                "required_before_promotion": "run fused controller and show overhead<=0.25 without reducing direction/no-debt counts",
            }
        ],
    )
    return route


def gpu_list(text: str) -> list[str]:
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    return items or ["0"]


def build_fused_probe_specs(args: argparse.Namespace, datasets: set[str], seeds: set[str]) -> list[dict[str, Any]]:
    devices = gpu_list(args.gpus)
    dataset_list = sorted(datasets) if datasets else ["FashionMNIST", "KMNIST", "MNIST", "Wine"]
    seed_list = sorted(seeds, key=lambda x: int(x)) if seeds else ["0", "1", "2"]
    specs: list[dict[str, Any]] = []
    idx = 0
    common = {
        "architecture": "MLP",
        "optimizer_family": "AdamW",
        "steps": int(args.fused_probe_steps),
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "batch_size": int(args.batch_size),
        "hidden": int(args.hidden),
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "support_rank": 4,
        "nuisance_rank": 2,
        "support_refresh_cadence": 120,
        "metric_refresh_cadence": 120,
        "cached_controller_emit_cadence": 50,
        "calibration_nuisance_cadence": 50,
        "velocity_scale": 0.18,
        "metric_shrinkage": 0.10,
        "metric_eps": 1.0e-6,
        "beta_signal": 0.01,
        "beta_metric": 0.05,
        "beta_q": 0.05,
        "eta_rho": 0.20,
        "eta_debt": 0.10,
        "tau_safe": 0.02,
        "rho_min": 0.0,
        "rho_max": 0.25,
        "debt_velocity_barrier": 0.0,
        "calibration_velocity_barrier": 0.0,
        "safety_budget_velocity_barrier": 8.0,
        "calibration_readout_radial_cap": 0.02,
        "calibration_readout_policy": "overconfidence_only",
        "calibration_nuisance_weight": 0.50,
        "calibration_correction_weight": 0.50,
        "calibration_nuisance_mode": "tail_q99_brier_qp_margin",
        "pure_fu_mode": False,
        "warmup_steps": 0,
        "kan_init_variant": "default",
        "mirror_grad_cadence": 1,
    }
    variants = [
        ("optimizer_alone", "none"),
        ("S4-SignalMetric-OET", "none"),
        ("S4-SignalMetric-OET", "same-metric-support-random"),
    ]
    for dataset in dataset_list:
        for seed in seed_list:
            for fused in [False, True]:
                mode = "fused" if fused else "unfused"
                for variant, control_mode in variants:
                    spec = dict(common)
                    spec.update(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "variant": variant,
                            "control_mode": control_mode,
                            "device_name": f"cuda:{devices[idx % len(devices)]}",
                            "label": (
                                f"v22_46b_B3_FusedDebt_{mode}_{dataset}_s{seed}_"
                                f"MLP_{variant}_{safe_fragment(control_mode)}"
                            ),
                            "fused_debt_controller": fused,
                            "fused_probe_mode": mode,
                            "part": "B3",
                            "mechanism": "FusedDebtControllerProbe",
                            "recipe": "emit50_supp120_metric120_readoutcap0_02_overconfidence_only_debt_shared_controller",
                        }
                    )
                    specs.append(spec)
                    idx += 1
    return specs


def train_kwargs(spec: dict[str, Any], tier2_download: bool = False) -> dict[str, Any]:
    return {
        "dataset": str(spec["dataset"]),
        "seed": int(spec["seed"]),
        "architecture": str(spec["architecture"]),
        "optimizer_family": str(spec["optimizer_family"]),
        "variant": str(spec["variant"]),
        "control_mode": str(spec["control_mode"]),
        "device_name": str(spec["device_name"]),
        "steps": int(spec["steps"]),
        "train_size": int(spec["train_size"]),
        "held_size": int(spec["held_size"]),
        "batch_size": int(spec["batch_size"]),
        "hidden": int(spec["hidden"]),
        "lr": float(spec["lr"]),
        "weight_decay": float(spec["weight_decay"]),
        "support_rank": int(spec["support_rank"]),
        "nuisance_rank": int(spec["nuisance_rank"]),
        "support_refresh_cadence": int(spec["support_refresh_cadence"]),
        "beta_signal": float(spec["beta_signal"]),
        "beta_metric": float(spec["beta_metric"]),
        "beta_q": float(spec["beta_q"]),
        "eta_rho": float(spec["eta_rho"]),
        "eta_debt": float(spec["eta_debt"]),
        "tau_safe": float(spec["tau_safe"]),
        "rho_min": float(spec["rho_min"]),
        "rho_max": float(spec["rho_max"]),
        "velocity_scale": float(spec["velocity_scale"]),
        "metric_shrinkage": float(spec["metric_shrinkage"]),
        "metric_eps": float(spec["metric_eps"]),
        "metric_refresh_cadence": int(spec["metric_refresh_cadence"]),
        "debt_velocity_barrier": float(spec["debt_velocity_barrier"]),
        "calibration_velocity_barrier": float(spec["calibration_velocity_barrier"]),
        "safety_budget_velocity_barrier": float(spec["safety_budget_velocity_barrier"]),
        "calibration_readout_radial_cap": float(spec["calibration_readout_radial_cap"]),
        "calibration_nuisance_weight": float(spec["calibration_nuisance_weight"]),
        "calibration_correction_weight": float(spec["calibration_correction_weight"]),
        "calibration_nuisance_mode": str(spec["calibration_nuisance_mode"]),
        "calibration_nuisance_cadence": int(spec["calibration_nuisance_cadence"]),
        "pure_fu_mode": bool(spec["pure_fu_mode"]),
        "warmup_steps": int(spec["warmup_steps"]),
        "kan_init_variant": str(spec["kan_init_variant"]),
        "tier2_download": bool(tier2_download),
        "label": str(spec["label"]),
        "calibration_readout_policy": str(spec["calibration_readout_policy"]),
        "mirror_grad_cadence": int(spec["mirror_grad_cadence"]),
        "cached_controller_emit_cadence": int(spec["cached_controller_emit_cadence"]),
        "fused_debt_controller": bool(spec["fused_debt_controller"]),
        "debt_orthogonal_controller": bool(spec.get("debt_orthogonal_controller", False)),
    }


def command_for_spec(spec: dict[str, Any]) -> str:
    kw = train_kwargs(spec)
    argv = [
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python",
        "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py",
        "--stage",
        "collect",
        "--dataset",
        kw["dataset"],
        "--seed",
        str(kw["seed"]),
        "--architecture",
        kw["architecture"],
        "--optimizer",
        kw["optimizer_family"],
        "--variant",
        kw["variant"],
        "--control-mode",
        kw["control_mode"],
        "--device",
        kw["device_name"],
        "--steps",
        str(kw["steps"]),
        "--train-size",
        str(kw["train_size"]),
        "--held-size",
        str(kw["held_size"]),
        "--batch-size",
        str(kw["batch_size"]),
        "--hidden",
        str(kw["hidden"]),
        "--support-refresh-cadence",
        str(kw["support_refresh_cadence"]),
        "--metric-refresh-cadence",
        str(kw["metric_refresh_cadence"]),
        "--cached-controller-emit-cadence",
        str(kw["cached_controller_emit_cadence"]),
        "--calibration-nuisance-cadence",
        str(kw["calibration_nuisance_cadence"]),
        "--calibration-nuisance-weight",
        str(kw["calibration_nuisance_weight"]),
        "--calibration-correction-weight",
        str(kw["calibration_correction_weight"]),
        "--calibration-nuisance-mode",
        kw["calibration_nuisance_mode"],
        "--calibration-readout-radial-cap",
        str(kw["calibration_readout_radial_cap"]),
        "--calibration-readout-policy",
        kw["calibration_readout_policy"],
        "--safety-budget-velocity-barrier",
        str(kw["safety_budget_velocity_barrier"]),
        "--velocity-scale",
        str(kw["velocity_scale"]),
        "--beta-signal",
        str(kw["beta_signal"]),
        "--label",
        kw["label"],
    ]
    if kw["fused_debt_controller"]:
        argv.append("--fused-debt-controller")
    if kw["debt_orthogonal_controller"]:
        argv.append("--debt-orthogonal-controller")
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def run_fused_probe_spec(spec: dict[str, Any]) -> dict[str, Any]:
    bind_upstream()
    try:
        summary = v2243.train_variant(**train_kwargs(spec))
        summary.update(
            {
                "part": spec.get("part", ""),
                "mechanism": spec.get("mechanism", ""),
                "recipe": spec.get("recipe", ""),
                "fused_probe_mode": spec.get("fused_probe_mode", ""),
                "status_v22_46b": "pass",
                "error": "",
            }
        )
        return summary
    except Exception as exc:
        err_path = LOG_ROOT / f"{safe_fragment(spec.get('label', 'fused_probe_row'))}_error.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            command_for_spec(spec),
            task_id=f"fused_probe_row_failed_{safe_fragment(spec.get('label', 'row'))}",
            status="failed",
            gpu=str(spec.get("device_name", "")),
            files=str(err_path.relative_to(ROOT)),
            note=repr(exc),
            exit_code=1,
        )
        return {
            "run_label": spec.get("label", ""),
            "dataset": spec.get("dataset", ""),
            "seed": spec.get("seed", ""),
            "architecture_key": spec.get("architecture", ""),
            "variant": spec.get("variant", ""),
            "control_mode": spec.get("control_mode", ""),
            "fused_debt_controller": int(bool(spec.get("fused_debt_controller"))),
            "fused_probe_mode": spec.get("fused_probe_mode", ""),
            "part": spec.get("part", ""),
            "mechanism": spec.get("mechanism", ""),
            "recipe": spec.get("recipe", ""),
            "status_v22_46b": "fail",
            "error": repr(exc),
        }


def fused_base_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("architecture_key", row.get("architecture", ""))),
        str(row.get("fused_probe_mode", "fused" if iflag(row.get("fused_debt_controller")) else "unfused")),
    )


def fused_no_debt(row: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    out = {
        "ECE_delta_vs_optimizer_baseline": "",
        "Brier_delta_vs_optimizer_baseline": "",
        "tail_q99_delta_vs_optimizer_baseline": "",
        "official_no_debt": 0,
    }
    if not base:
        return out
    ece = fval(row.get("ECE"))
    brier = fval(row.get("Brier"))
    tail = fval(row.get("tail_loss_q99"))
    base_ece = fval(base.get("ECE"))
    base_brier = fval(base.get("Brier"))
    base_tail = fval(base.get("tail_loss_q99"))
    if None in {ece, brier, tail, base_ece, base_brier, base_tail}:
        return out
    ece_delta = float(ece) - float(base_ece)
    brier_delta = float(brier) - float(base_brier)
    tail_delta = float(tail) - float(base_tail)
    out.update(
        {
            "ECE_delta_vs_optimizer_baseline": ece_delta,
            "Brier_delta_vs_optimizer_baseline": brier_delta,
            "tail_q99_delta_vs_optimizer_baseline": tail_delta,
            "official_no_debt": int(ece_delta <= 0.0 and brier_delta <= 0.0 and tail_delta <= 0.0),
        }
    )
    return out


def build_fused_probe_direction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bases: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    controls: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status_v22_46b") != "pass":
            continue
        key = fused_base_key(row)
        if row.get("variant") == "optimizer_alone" and row.get("control_mode") == "none":
            bases[key] = row
        elif row.get("variant") == "S4-SignalMetric-OET" and row.get("control_mode") == "same-metric-support-random":
            controls[(*key, "S4-SignalMetric-OET")] = row
        elif row.get("variant") == "S4-SignalMetric-OET" and row.get("control_mode") == "none":
            candidates.append(row)
    out: list[dict[str, Any]] = []
    for row in candidates:
        key = fused_base_key(row)
        base = bases.get(key)
        control = controls.get((*key, "S4-SignalMetric-OET"))
        base_nll = fval(base.get("final_NLL")) if base else None
        control_nll = fval(control.get("final_NLL")) if control else None
        real_nll = fval(row.get("final_NLL"))
        tau_support = base_nll - control_nll if base_nll is not None and control_nll is not None else None
        tau_direction = control_nll - real_nll if control_nll is not None and real_nll is not None else None
        audit = fused_no_debt(row, base)
        overhead = fval(row.get("controller_overhead_ratio"))
        direction_positive = tau_direction is not None and tau_direction > 0.0
        overhead_le_0p25 = overhead is not None and overhead <= 0.25
        overhead_le_0p35 = overhead is not None and overhead <= 0.35
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "architecture_key": row.get("architecture_key"),
                "variant": row.get("variant"),
                "fused_probe_mode": row.get("fused_probe_mode"),
                "fused_debt_controller": row.get("fused_debt_controller"),
                "base_final_NLL": base_nll,
                "control_final_NLL": control_nll,
                "real_final_NLL": real_nll,
                "tau_support": tau_support,
                "tau_direction": tau_direction,
                "direction_positive": int(direction_positive),
                "beats_matched_control": int(direction_positive),
                **audit,
                "controller_overhead_ratio": overhead,
                "controller_ms": row.get("controller_ms"),
                "state_update_ms": row.get("state_update_ms"),
                "calibration_nuisance_grad_ms": row.get("calibration_nuisance_grad_ms"),
                "full_step_ms": row.get("full_step_ms"),
                "cached_controller_emit_cadence": row.get("cached_controller_emit_cadence"),
                "cached_controller_emit_fraction": row.get("cached_controller_emit_fraction"),
                "controller_emit_refresh_fraction": row.get("controller_emit_refresh_fraction"),
                "overhead_le_0p25": int(overhead_le_0p25),
                "overhead_le_0p35": int(overhead_le_0p35),
                "B3_route_gate": int(direction_positive and audit["official_no_debt"] and overhead_le_0p25),
                "B3_exploration_gate_0p35": int(direction_positive and audit["official_no_debt"] and overhead_le_0p35),
                "source_artifact": "results/v22_46b/v22_46b_fused_controller_training_matrix.csv",
            }
        )
    return out


def stage_fused_probe(args: argparse.Namespace, datasets: set[str], seeds: set[str]) -> dict[str, Any]:
    bind_upstream()
    specs = build_fused_probe_specs(args, datasets, seeds)
    write_rows(OUT_ROOT / "v22_46b_fused_controller_specs.csv", specs)
    append_exec(
        "dispatch_v22_46b_fused_controller_probe",
        task_id="v22.46b-fused-probe-dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46b/v22_46b_fused_controller_specs.csv, results/v22_46b/chunks",
        note=f"rows={len(specs)}; workers={args.workers}; steps={args.fused_probe_steps}; paired unfused/fused debt-controller probe",
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_fused_probe_spec, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46b") != "pass")
    rows.sort(key=lambda row: str(row.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46b_fused_controller_training_matrix.csv", rows)
    direction = build_fused_probe_direction(rows)
    write_rows(OUT_ROOT / "v22_46b_fused_controller_direction_matrix.csv", direction)
    write_rows(OUT_ROOT / "v22_46b_fused_controller_probe.csv", direction or [{"status": "no_direction_rows"}])

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in direction:
        by_mode.setdefault(str(row.get("fused_probe_mode", "")), []).append(row)
    mode_summary = []
    for mode, group in sorted(by_mode.items()):
        mode_summary.append(
            {
                "mode": mode,
                "direction_rows": len(group),
                "direction_positive_rows": sum(iflag(row.get("direction_positive")) for row in group),
                "official_no_debt_rows": sum(iflag(row.get("official_no_debt")) for row in group),
                "overhead_le_0p25_rows": sum(iflag(row.get("overhead_le_0p25")) for row in group),
                "overhead_le_0p35_rows": sum(iflag(row.get("overhead_le_0p35")) for row in group),
                "route_gate_rows": sum(iflag(row.get("B3_route_gate")) for row in group),
                "exploration_gate_rows_0p35": sum(iflag(row.get("B3_exploration_gate_0p35")) for row in group),
                "mean_controller_overhead_ratio": mean(fval(row.get("controller_overhead_ratio")) for row in group),
                "median_controller_overhead_ratio": median(fval(row.get("controller_overhead_ratio")) for row in group),
            }
        )
    write_rows(OUT_ROOT / "v22_46b_fused_controller_summary.csv", mode_summary)
    fused = next((row for row in mode_summary if row.get("mode") == "fused"), {})
    unfused = next((row for row in mode_summary if row.get("mode") == "unfused"), {})
    fused_route_gate = int(fused.get("route_gate_rows", 0) or 0)
    fused_direction = int(fused.get("direction_positive_rows", 0) or 0)
    unfused_direction = int(unfused.get("direction_positive_rows", 0) or 0)
    fused_no_debt = int(fused.get("official_no_debt_rows", 0) or 0)
    unfused_no_debt = int(unfused.get("official_no_debt_rows", 0) or 0)
    route_open = fused_route_gate >= 8 and fused_direction >= unfused_direction and fused_no_debt >= unfused_no_debt
    route = {
        "timestamp": now_sg(),
        "route": "B3_fused_controller_opened" if route_open else "not_opened",
        "reason": "B3 requires fused route_gate>=8 without reducing direction-positive or official no-debt counts versus paired unfused probe",
        "rows": len(rows),
        "failures": failures,
        "direction_rows": len(direction),
        "fused_route_gate_rows": fused_route_gate,
        "unfused_route_gate_rows": int(unfused.get("route_gate_rows", 0) or 0),
        "fused_direction_positive_rows": fused_direction,
        "unfused_direction_positive_rows": unfused_direction,
        "fused_official_no_debt_rows": fused_no_debt,
        "unfused_official_no_debt_rows": unfused_no_debt,
        "fused_overhead_le_0p25_rows": int(fused.get("overhead_le_0p25_rows", 0) or 0),
        "unfused_overhead_le_0p25_rows": int(unfused.get("overhead_le_0p25_rows", 0) or 0),
        "fused_mean_controller_overhead_ratio": fused.get("mean_controller_overhead_ratio", ""),
        "unfused_mean_controller_overhead_ratio": unfused.get("mean_controller_overhead_ratio", ""),
        "promotion_allowed": route_open,
        "source_artifacts": [
            "results/v22_46b/v22_46b_fused_controller_training_matrix.csv",
            "results/v22_46b/v22_46b_fused_controller_direction_matrix.csv",
        ],
        "claim_limit": "paired v22.46b training probe only; does not alter official v22.46 results",
    }
    write_json(OUT_ROOT / "v22_46b_fused_controller_route.json", route)
    append_exec(
        "dispatch_v22_46b_fused_controller_probe",
        task_id="v22.46b-fused-probe-complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46b/v22_46b_fused_controller_training_matrix.csv, results/v22_46b/v22_46b_fused_controller_route.json",
        note=f"rows={len(rows)}; failures={failures}; route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return route


def build_b1_gradient_probe_specs(args: argparse.Namespace, datasets: set[str], seeds: set[str]) -> list[dict[str, Any]]:
    devices = gpu_list(args.gpus)
    dataset_list = sorted(datasets) if datasets else ["FashionMNIST", "KMNIST", "MNIST", "Wine"]
    seed_list = sorted(seeds, key=lambda x: int(x)) if seeds else ["0", "1", "2"]
    specs: list[dict[str, Any]] = []
    idx = 0
    common = {
        "architecture": "MLP",
        "optimizer_family": "AdamW",
        "steps": int(args.b1_probe_steps),
        "train_size": int(args.train_size),
        "held_size": int(args.held_size),
        "batch_size": int(args.batch_size),
        "hidden": int(args.hidden),
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "support_rank": 4,
        "nuisance_rank": 2,
        "support_refresh_cadence": 120,
        "metric_refresh_cadence": 120,
        "cached_controller_emit_cadence": 50,
        "calibration_nuisance_cadence": 50,
        "velocity_scale": 0.18,
        "metric_shrinkage": 0.10,
        "metric_eps": 1.0e-6,
        "beta_signal": 0.01,
        "beta_metric": 0.05,
        "beta_q": 0.05,
        "eta_rho": 0.20,
        "eta_debt": 0.10,
        "tau_safe": 0.02,
        "rho_min": 0.0,
        "rho_max": 0.25,
        "debt_velocity_barrier": 0.0,
        "calibration_velocity_barrier": 0.0,
        "safety_budget_velocity_barrier": 8.0,
        "calibration_readout_radial_cap": 0.02,
        "calibration_readout_policy": "overconfidence_only",
        "calibration_nuisance_weight": 0.50,
        "calibration_correction_weight": 0.50,
        "calibration_nuisance_mode": "tail_q99_brier_qp_margin",
        "pure_fu_mode": False,
        "warmup_steps": 0,
        "kan_init_variant": "default",
        "mirror_grad_cadence": 1,
        "fused_debt_controller": True,
    }
    variants = [
        ("optimizer_alone", "none"),
        ("S4-SignalMetric-OET", "none"),
        ("S4-SignalMetric-OET", "same-metric-support-random"),
    ]
    for dataset in dataset_list:
        for seed in seed_list:
            for debt_orthogonal in [False, True]:
                mode = "debt_orthogonal" if debt_orthogonal else "standard"
                for variant, control_mode in variants:
                    spec = dict(common)
                    spec.update(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "variant": variant,
                            "control_mode": control_mode,
                            "device_name": f"cuda:{devices[idx % len(devices)]}",
                            "label": (
                                f"v22_46b_B1_DebtGradient_{mode}_{dataset}_s{seed}_"
                                f"MLP_{variant}_{safe_fragment(control_mode)}"
                            ),
                            "debt_orthogonal_controller": debt_orthogonal,
                            "b1_probe_mode": mode,
                            "part": "B1",
                            "mechanism": "DebtGradientOrthogonalProbe",
                            "recipe": "emit50_supp120_metric120_tailq99_brier_gradient_orthogonal_probe",
                        }
                    )
                    specs.append(spec)
                    idx += 1
    return specs


def b1_mode(row: dict[str, Any]) -> str:
    explicit = str(row.get("b1_probe_mode", "") or "").strip()
    if explicit:
        return explicit
    label = str(row.get("run_label", "") or "")
    if "_B1_DebtGradient_debt_orthogonal_" in label:
        return "debt_orthogonal"
    if "_B1_DebtGradient_standard_" in label:
        return "standard"
    return "debt_orthogonal" if iflag(row.get("debt_orthogonal_controller")) else "standard"


def b1_base_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    mode = b1_mode(row)
    return (
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("architecture_key", row.get("architecture", ""))),
        mode,
    )


def build_b1_gradient_direction(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bases: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    controls: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status_v22_46b") != "pass":
            continue
        key = b1_base_key(row)
        if row.get("variant") == "optimizer_alone" and row.get("control_mode") == "none":
            bases[key] = row
        elif row.get("variant") == "S4-SignalMetric-OET" and row.get("control_mode") == "same-metric-support-random":
            controls[(*key, "S4-SignalMetric-OET")] = row
        elif row.get("variant") == "S4-SignalMetric-OET" and row.get("control_mode") == "none":
            candidates.append(row)
        overlap_rows.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "architecture_key": row.get("architecture_key"),
                "variant": row.get("variant"),
                "control_mode": row.get("control_mode"),
                "b1_probe_mode": b1_mode(row),
                "debt_orthogonal_controller": row.get("debt_orthogonal_controller"),
                "debt_tail_alignment_before_orthogonal": row.get("debt_tail_alignment_before_orthogonal"),
                "debt_brier_alignment_before_orthogonal": row.get("debt_brier_alignment_before_orthogonal"),
                "debt_tail_alignment_after_orthogonal": row.get("debt_tail_alignment_after_orthogonal"),
                "debt_brier_alignment_after_orthogonal": row.get("debt_brier_alignment_after_orthogonal"),
                "debt_orthogonal_projection_rank_mean": row.get("debt_orthogonal_projection_rank_mean"),
                "debt_orthogonal_projection_overlap": row.get("debt_orthogonal_projection_overlap"),
                "pareto_guard_active_fraction": row.get("pareto_guard_active_fraction"),
                "pareto_tail_alignment_before": row.get("pareto_tail_alignment_before"),
                "pareto_brier_alignment_before": row.get("pareto_brier_alignment_before"),
                "calibration_nuisance_metric_overlap": row.get("calibration_nuisance_metric_overlap"),
                "controller_overhead_ratio": row.get("controller_overhead_ratio"),
                "source_artifact": "results/v22_46b/v22_46b_b1_debt_gradient_training_matrix.csv",
            }
        )
    direction_rows: list[dict[str, Any]] = []
    for row in candidates:
        key = b1_base_key(row)
        base = bases.get(key)
        control = controls.get((*key, "S4-SignalMetric-OET"))
        base_nll = fval(base.get("final_NLL")) if base else None
        control_nll = fval(control.get("final_NLL")) if control else None
        real_nll = fval(row.get("final_NLL"))
        tau_support = base_nll - control_nll if base_nll is not None and control_nll is not None else None
        tau_direction = control_nll - real_nll if control_nll is not None and real_nll is not None else None
        audit = fused_no_debt(row, base)
        overhead = fval(row.get("controller_overhead_ratio"))
        tail_before = fval(row.get("debt_tail_alignment_before_orthogonal"), 0.0) or 0.0
        brier_before = fval(row.get("debt_brier_alignment_before_orthogonal"), 0.0) or 0.0
        tail_after = fval(row.get("debt_tail_alignment_after_orthogonal"), 0.0) or 0.0
        brier_after = fval(row.get("debt_brier_alignment_after_orthogonal"), 0.0) or 0.0
        harmful_before = max(-tail_before, 0.0) + max(-brier_before, 0.0)
        harmful_after = max(-tail_after, 0.0) + max(-brier_after, 0.0)
        direction_positive = tau_direction is not None and tau_direction > 0.0
        overhead_le_0p25 = overhead is not None and overhead <= 0.25
        overhead_le_0p35 = overhead is not None and overhead <= 0.35
        debt_clean = harmful_after <= 1.0e-6 if iflag(row.get("debt_orthogonal_controller")) else harmful_before <= 1.0e-6
        route_gate = direction_positive and audit["official_no_debt"] and overhead_le_0p25 and debt_clean
        exploration_gate = direction_positive and audit["official_no_debt"] and overhead_le_0p35 and debt_clean
        direction_rows.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "architecture_key": row.get("architecture_key"),
                "variant": row.get("variant"),
                "b1_probe_mode": b1_mode(row),
                "debt_orthogonal_controller": row.get("debt_orthogonal_controller"),
                "base_final_NLL": base_nll,
                "control_final_NLL": control_nll,
                "real_final_NLL": real_nll,
                "tau_support": tau_support,
                "tau_direction": tau_direction,
                "direction_positive": int(direction_positive),
                "beats_matched_control": int(direction_positive),
                **audit,
                "debt_tail_alignment_before_orthogonal": tail_before,
                "debt_brier_alignment_before_orthogonal": brier_before,
                "debt_tail_alignment_after_orthogonal": tail_after,
                "debt_brier_alignment_after_orthogonal": brier_after,
                "harmful_debt_gradient_overlap_before": harmful_before,
                "harmful_debt_gradient_overlap_after": harmful_after,
                "debt_gradient_overlap_clean": int(debt_clean),
                "debt_orthogonal_projection_overlap": row.get("debt_orthogonal_projection_overlap"),
                "direction_gain_debt_orthogonal_training_gate": int(direction_positive and debt_clean),
                "controller_overhead_ratio": overhead,
                "overhead_le_0p25": int(overhead_le_0p25),
                "overhead_le_0p35": int(overhead_le_0p35),
                "B1_training_route_gate": int(route_gate),
                "B1_training_exploration_gate_0p35": int(exploration_gate),
                "source_artifact": "results/v22_46b/v22_46b_b1_debt_gradient_training_matrix.csv",
            }
        )
    return direction_rows, overlap_rows


def stage_b1_gradient_probe(args: argparse.Namespace, datasets: set[str], seeds: set[str]) -> dict[str, Any]:
    bind_upstream()
    specs = build_b1_gradient_probe_specs(args, datasets, seeds)
    write_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_specs.csv", specs)
    append_exec(
        "dispatch_v22_46b_b1_debt_gradient_probe",
        task_id="v22.46b-b1-gradient-probe-dispatch",
        status="started",
        gpu=args.gpus,
        files="results/v22_46b/v22_46b_b1_debt_gradient_specs.csv, results/v22_46b/chunks",
        note=f"rows={len(specs)}; workers={args.workers}; steps={args.b1_probe_steps}; paired standard/debt_orthogonal probe",
    )
    rows: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = [ex.submit(run_fused_probe_spec, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            rows.append(row)
            failures += int(row.get("status_v22_46b") != "pass")
    rows.sort(key=lambda row: str(row.get("run_label", "")))
    write_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_training_matrix.csv", rows)
    direction, overlaps = build_b1_gradient_direction(rows)
    write_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_direction_matrix.csv", direction)
    write_rows(OUT_ROOT / "v22_46b_debt_gradient_overlap.csv", overlaps)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in direction:
        by_mode.setdefault(str(row.get("b1_probe_mode", "")), []).append(row)
    mode_summary: list[dict[str, Any]] = []
    for mode, group in sorted(by_mode.items()):
        mode_summary.append(
            {
                "mode": mode,
                "direction_rows": len(group),
                "direction_positive_rows": sum(iflag(row.get("direction_positive")) for row in group),
                "official_no_debt_rows": sum(iflag(row.get("official_no_debt")) for row in group),
                "debt_gradient_overlap_clean_rows": sum(iflag(row.get("debt_gradient_overlap_clean")) for row in group),
                "overhead_le_0p25_rows": sum(iflag(row.get("overhead_le_0p25")) for row in group),
                "route_gate_rows": sum(iflag(row.get("B1_training_route_gate")) for row in group),
                "exploration_gate_rows_0p35": sum(iflag(row.get("B1_training_exploration_gate_0p35")) for row in group),
                "mean_harmful_debt_gradient_overlap_after": mean(fval(row.get("harmful_debt_gradient_overlap_after")) for row in group),
                "mean_controller_overhead_ratio": mean(fval(row.get("controller_overhead_ratio")) for row in group),
            }
        )
    write_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_summary.csv", mode_summary)
    ortho = next((row for row in mode_summary if row.get("mode") == "debt_orthogonal"), {})
    standard = next((row for row in mode_summary if row.get("mode") == "standard"), {})
    ortho_route_gate = int(ortho.get("route_gate_rows", 0) or 0)
    route_open = ortho_route_gate >= 8
    route = {
        "timestamp": now_sg(),
        "route": "B1_debt_gradient_orthogonal_opened" if route_open else "not_opened",
        "reason": "B1 requires >=8 debt-orthogonal rows with positive direction, clean harmful debt-gradient overlap, official no-debt, overhead<=0.25, and matched-control win",
        "rows": len(rows),
        "failures": failures,
        "direction_rows": len(direction),
        "debt_orthogonal_route_gate_rows": ortho_route_gate,
        "standard_route_gate_rows": int(standard.get("route_gate_rows", 0) or 0),
        "debt_orthogonal_direction_positive_rows": int(ortho.get("direction_positive_rows", 0) or 0),
        "standard_direction_positive_rows": int(standard.get("direction_positive_rows", 0) or 0),
        "debt_orthogonal_official_no_debt_rows": int(ortho.get("official_no_debt_rows", 0) or 0),
        "standard_official_no_debt_rows": int(standard.get("official_no_debt_rows", 0) or 0),
        "debt_orthogonal_overlap_clean_rows": int(ortho.get("debt_gradient_overlap_clean_rows", 0) or 0),
        "standard_overlap_clean_rows": int(standard.get("debt_gradient_overlap_clean_rows", 0) or 0),
        "debt_orthogonal_overhead_le_0p25_rows": int(ortho.get("overhead_le_0p25_rows", 0) or 0),
        "standard_overhead_le_0p25_rows": int(standard.get("overhead_le_0p25_rows", 0) or 0),
        "promotion_allowed": route_open,
        "source_artifacts": [
            "results/v22_46b/v22_46b_b1_debt_gradient_training_matrix.csv",
            "results/v22_46b/v22_46b_b1_debt_gradient_direction_matrix.csv",
            "results/v22_46b/v22_46b_debt_gradient_overlap.csv",
        ],
        "claim_limit": "paired v22.46b B1 training probe; official no-debt remains exact zero-tolerance vs same-mode optimizer baseline",
    }
    write_json(OUT_ROOT / "v22_46b_b1_debt_gradient_route.json", route)
    append_exec(
        "dispatch_v22_46b_b1_debt_gradient_probe",
        task_id="v22.46b-b1-gradient-probe-complete",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46b/v22_46b_b1_debt_gradient_training_matrix.csv, results/v22_46b/v22_46b_b1_debt_gradient_route.json",
        note=f"rows={len(rows)}; failures={failures}; route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return route


def stage_b1_gradient_recap(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_training_matrix.csv")
    failures = sum(row.get("status_v22_46b") != "pass" for row in rows)
    direction, overlaps = build_b1_gradient_direction(rows)
    write_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_direction_matrix.csv", direction)
    write_rows(OUT_ROOT / "v22_46b_debt_gradient_overlap.csv", overlaps)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in direction:
        by_mode.setdefault(str(row.get("b1_probe_mode", "")), []).append(row)
    mode_summary: list[dict[str, Any]] = []
    for mode, group in sorted(by_mode.items()):
        mode_summary.append(
            {
                "mode": mode,
                "direction_rows": len(group),
                "direction_positive_rows": sum(iflag(row.get("direction_positive")) for row in group),
                "official_no_debt_rows": sum(iflag(row.get("official_no_debt")) for row in group),
                "debt_gradient_overlap_clean_rows": sum(iflag(row.get("debt_gradient_overlap_clean")) for row in group),
                "overhead_le_0p25_rows": sum(iflag(row.get("overhead_le_0p25")) for row in group),
                "route_gate_rows": sum(iflag(row.get("B1_training_route_gate")) for row in group),
                "exploration_gate_rows_0p35": sum(iflag(row.get("B1_training_exploration_gate_0p35")) for row in group),
                "mean_harmful_debt_gradient_overlap_after": mean(fval(row.get("harmful_debt_gradient_overlap_after")) for row in group),
                "mean_controller_overhead_ratio": mean(fval(row.get("controller_overhead_ratio")) for row in group),
            }
        )
    write_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_summary.csv", mode_summary)
    ortho = next((row for row in mode_summary if row.get("mode") == "debt_orthogonal"), {})
    standard = next((row for row in mode_summary if row.get("mode") == "standard"), {})
    ortho_route_gate = int(ortho.get("route_gate_rows", 0) or 0)
    route_open = ortho_route_gate >= 8
    route = {
        "timestamp": now_sg(),
        "route": "B1_debt_gradient_orthogonal_opened" if route_open else "not_opened",
        "reason": "B1 requires >=8 debt-orthogonal rows with positive direction, clean harmful debt-gradient overlap, official no-debt, overhead<=0.25, and matched-control win",
        "rows": len(rows),
        "failures": failures,
        "direction_rows": len(direction),
        "debt_orthogonal_route_gate_rows": ortho_route_gate,
        "standard_route_gate_rows": int(standard.get("route_gate_rows", 0) or 0),
        "debt_orthogonal_direction_positive_rows": int(ortho.get("direction_positive_rows", 0) or 0),
        "standard_direction_positive_rows": int(standard.get("direction_positive_rows", 0) or 0),
        "debt_orthogonal_official_no_debt_rows": int(ortho.get("official_no_debt_rows", 0) or 0),
        "standard_official_no_debt_rows": int(standard.get("official_no_debt_rows", 0) or 0),
        "debt_orthogonal_overlap_clean_rows": int(ortho.get("debt_gradient_overlap_clean_rows", 0) or 0),
        "standard_overlap_clean_rows": int(standard.get("debt_gradient_overlap_clean_rows", 0) or 0),
        "debt_orthogonal_overhead_le_0p25_rows": int(ortho.get("overhead_le_0p25_rows", 0) or 0),
        "standard_overhead_le_0p25_rows": int(standard.get("overhead_le_0p25_rows", 0) or 0),
        "promotion_allowed": route_open,
        "source_artifacts": [
            "results/v22_46b/v22_46b_b1_debt_gradient_training_matrix.csv",
            "results/v22_46b/v22_46b_b1_debt_gradient_direction_matrix.csv",
            "results/v22_46b/v22_46b_debt_gradient_overlap.csv",
        ],
        "claim_limit": "paired v22.46b B1 training probe; official no-debt remains exact zero-tolerance vs same-mode optimizer baseline",
        "recap_only": True,
        "recap_note": "mode inferred from b1_probe_mode/run_label/debt_orthogonal_controller; no training rerun",
    }
    write_json(OUT_ROOT / "v22_46b_b1_debt_gradient_route.json", route)
    append_exec(
        "recompute_v22_46b_b1_debt_gradient_route_from_existing_training_matrix",
        task_id="v22.46b-b1-gradient-recap",
        status="pass" if failures == 0 else "fail",
        gpu=args.gpus,
        files="results/v22_46b/v22_46b_b1_debt_gradient_direction_matrix.csv, results/v22_46b/v22_46b_b1_debt_gradient_route.json",
        note=f"rows={len(rows)}; failures={failures}; route={route}",
        exit_code=0 if failures == 0 else 1,
    )
    return route


def stage_recap() -> None:
    code = read_json(OUT_ROOT / "v22_46b_code_audit.json")
    b1 = read_json(OUT_ROOT / "v22_46b_debt_orthogonal_route.json")
    b2 = read_json(OUT_ROOT / "v22_46b_kan_margin_route.json")
    b3 = read_json(OUT_ROOT / "v22_46b_overhead_route.json")
    fused = read_json(OUT_ROOT / "v22_46b_fused_controller_route.json")
    b1_gradient = read_json(OUT_ROOT / "v22_46b_b1_debt_gradient_route.json")
    if fused and b3:
        b3["fused_controller_probe_available"] = True
        b3["fused_controller_route"] = fused.get("route", "")
        b3["fused_controller_route_gate_rows"] = fused.get("fused_route_gate_rows", "")
        b3["fused_controller_promotion_allowed"] = fused.get("promotion_allowed", False)
        write_json(OUT_ROOT / "v22_46b_overhead_route.json", b3)
    debt_rows = read_rows(OUT_ROOT / "v22_46b_debt_orthogonal_matrix.csv")
    kan_rows = read_rows(OUT_ROOT / "v22_46b_kan_margin_stability_matrix.csv")
    overhead_summary = read_rows(OUT_ROOT / "v22_46b_overhead_group_summary.csv")
    fused_summary = read_rows(OUT_ROOT / "v22_46b_fused_controller_summary.csv")
    b1_gradient_summary = read_rows(OUT_ROOT / "v22_46b_b1_debt_gradient_summary.csv")

    routes = [
        {
            "part": "code/input audit",
            "route": code.get("status"),
            "key_count": f"missing_inputs={code.get('missing_inputs')}, empty_csv={code.get('empty_csv_inputs')}",
            "claim_limit": code.get("claim_limit"),
        },
        {
            "part": "B1 debt-orthogonal",
            "route": b1.get("route"),
            "key_count": f"route_gate={b1.get('route_gate_rows')}, exploration={b1.get('exploration_gate_rows_0p35')}",
            "claim_limit": b1.get("claim_limit"),
        },
        {
            "part": "B1 gradient probe",
            "route": b1_gradient.get("route", "not_run"),
            "key_count": f"ortho_gate={b1_gradient.get('debt_orthogonal_route_gate_rows', '')}, standard_gate={b1_gradient.get('standard_route_gate_rows', '')}",
            "claim_limit": b1_gradient.get("claim_limit", "not run yet"),
        },
        {
            "part": "B2 KAN margin",
            "route": b2.get("route"),
            "key_count": f"route_gate={b2.get('route_gate_rows')}, control_explained_rate={b2.get('ControlExplained_rate')}",
            "claim_limit": b2.get("claim_limit"),
        },
        {
            "part": "B3 overhead",
            "route": b3.get("route"),
            "key_count": f"emit20_gate={b3.get('current_emit20_official_direction_route_gate_rows')}, emit50_snapshot_gate={b3.get('emit50_readout_snapshot_official_direction_route_gate_rows')}",
            "claim_limit": b3.get("claim_limit"),
        },
        {
            "part": "B3 fused probe",
            "route": fused.get("route", "not_run"),
            "key_count": f"fused_gate={fused.get('fused_route_gate_rows', '')}, unfused_gate={fused.get('unfused_route_gate_rows', '')}",
            "claim_limit": fused.get("claim_limit", "not run yet"),
        },
    ]

    top_debt = sorted(
        debt_rows,
        key=lambda row: (
            -iflag(row.get("B1_exploration_gate_0p35")),
            -(fval(row.get("direction_gain_debt_orthogonal_proxy"), -1e9) or -1e9),
        ),
    )
    top_kan = sorted(
        kan_rows,
        key=lambda row: (
            -iflag(row.get("B2_route_gate")),
            -(fval(row.get("control_margin_p10_control_minus_KAN"), -1e9) or -1e9),
        ),
    )
    overhead_sorted = sorted(
        overhead_summary,
        key=lambda row: fval(row.get("reported_controller_overhead_ratio_median"), 1e9) or 1e9,
    )

    text = [
        "# DG-KAN v22.46b DebtControlledSignalIdentification 实验结果复盘\n",
        f"更新时间：{now_sg()}\n",
        "## 结论\n",
        "- 需要新实验理论和计划：是。v22.46 的继续扫参已经被 no-debt、overhead、ControlExplained 三个 gate 同时限制。\n",
        f"- B1 debt-orthogonal route: `{b1.get('route')}` / route_gate `{b1.get('route_gate_rows')}` / exploration_gate0p35 `{b1.get('exploration_gate_rows_0p35')}`。\n",
        f"- B1 debt-gradient training probe: `{b1_gradient.get('route', 'not_run')}` / debt_orthogonal_gate `{b1_gradient.get('debt_orthogonal_route_gate_rows', '')}` / standard_gate `{b1_gradient.get('standard_route_gate_rows', '')}`。\n",
        f"- B2 KAN margin route: `{b2.get('route')}` / route_gate `{b2.get('route_gate_rows')}` / ControlExplained_rate `{b2.get('ControlExplained_rate')}`。\n",
        f"- B3 overhead route: `{b3.get('route')}` / fused_controller_probe_available `{b3.get('fused_controller_probe_available')}`。\n",
        f"- B3 fused controller probe: `{fused.get('route', 'not_run')}` / fused_route_gate `{fused.get('fused_route_gate_rows', '')}` / unfused_route_gate `{fused.get('unfused_route_gate_rows', '')}`。\n",
        "- B1 posthoc/B2/overhead breakdown 是 posthoc 审计；B1 debt-gradient 与 B3 fused controller 是本轮新增 paired training probe，均使用 4 GPU workers 跑完且不再用旧失败 route 代替新证据。\n",
        "- 按 v22.46b 停止规则：B1/B2/B3 均未打开，v22.46 family 收尾为稳定 `SupportOnlyNoSignalIncrement`；该结论不是 DG-KAN 总体否定，只限定当前 v22.46/v22.46b 证据链。\n",
        "\n## Route Evidence\n",
        md_table(routes, ["part", "route", "key_count", "claim_limit"], limit=8),
        "\n## 执行与修复记录\n",
        "- 新增 `experiments/run_v22_46b_debt_controlled_signal_identification.py`：独立读取 v22.46 artifact，输出到 `results/v22_46b/`，不覆盖 v22.46 主结果。\n",
        "- 新增 B1 posthoc debt-orthogonal matrix：用 `tau_direction - sum(positive ECE/Brier/tail_q99 delta)` 作为保守 proxy，并明确标注不是 causal gradient-overlap proof。\n",
        "- 新增 B2 KAN observed-control margin audit：从 KAN variant repair raw matrix 重建 same-basis controls 的 margin min/p10/median；`v22_46b_kan_control_bootstrap.csv` 标注为 observed controls only，未伪装成真实 bootstrap。\n",
        "- 新增 B3 overhead breakdown：拆分 current emit20、emit50 readout snapshot、KAN variant repair 的 reported overhead 与 timing component，用于判断 cadence-only 是否足够。\n",
        "- 代码修复：在 `experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py` 增加默认关闭的 `fused_debt_controller`；开启后 calibration debt gradient refresh 与 controller emit refresh 对齐并复用缓存，新增 summary 字段 `fused_debt_controller`、`controller_emit_refresh_fraction`、`debt_grad_shared_with_controller_emit`。\n",
        "- 代码修复：在 `experiments/run_v22_46_support_quotient_causal_flow.py` adapter 增加 `fused_debt_controller` 透传与复现命令标记；默认关闭，不改变既有 v22.46 artifact 的解释。\n",
        "- 新增 B3 paired fused/unfused training probe：72 rows / failures 0，4 GPU workers；输出 `v22_46b_fused_controller_training_matrix.csv`、`v22_46b_fused_controller_direction_matrix.csv`、`v22_46b_fused_controller_route.json`。\n",
        "- 复盘一致性修复：如果 fused route JSON 已存在，`v22_46b_overhead_route.json` 自动回填 `fused_controller_probe_available=True` 与 fused route gate，避免 posthoc overhead 早于 probe 生成导致日志前后矛盾。\n",
        "- 新增 B1 debt-gradient training probe：paired standard/debt_orthogonal controller，输出 `v22_46b_b1_debt_gradient_training_matrix.csv`、`v22_46b_b1_debt_gradient_direction_matrix.csv`、`v22_46b_debt_gradient_overlap.csv`、`v22_46b_b1_debt_gradient_route.json`。\n",
        "- B1 修复记录：首次 B1 训练因 `MetricFlowController.update_and_emit` 未持有 `debt_orthogonal_controller` 参数而 72/72 failed；已在 controller `__init__` 中保存该参数并由 adapter/CLI 传入，`py_compile` 通过后重跑为 72 rows / failures 0。\n",
        "- B1 汇总修复：子训练 summary 未保留 `b1_probe_mode`，导致 route JSON 的 standard/debt_orthogonal 分组为 0；已从 `b1_probe_mode/run_label/debt_orthogonal_controller` 推断 mode，并用 `--stage b1-gradient-recap` 从既有训练矩阵重算 route，没有重复训练。\n",
        "\n## B1 Debt-Orthogonal Evidence\n",
        md_table(
            top_debt,
            [
                "part",
                "dataset",
                "seed",
                "variant",
                "tau_direction",
                "debt_penalty_raw_metric_units",
                "direction_gain_debt_orthogonal_proxy",
                "official_no_debt",
                "controller_overhead",
                "B1_route_gate",
                "B1_exploration_gate_0p35",
            ],
            limit=16,
        ),
        "\n## B1 Debt-Gradient Training Probe\n",
        md_table(
            b1_gradient_summary,
            [
                "mode",
                "direction_rows",
                "direction_positive_rows",
                "official_no_debt_rows",
                "debt_gradient_overlap_clean_rows",
                "overhead_le_0p25_rows",
                "route_gate_rows",
                "exploration_gate_rows_0p35",
                "mean_harmful_debt_gradient_overlap_after",
            ],
            limit=8,
        ),
        "\n## B2 KAN Margin Evidence\n",
        md_table(
            top_kan,
            [
                "dataset",
                "seed",
                "KAN_architecture",
                "KAN_variant",
                "control_count",
                "control_margin_p10_control_minus_KAN",
                "KAN_beats_MLP_matched_support",
                "KAN_margin_stable_p10_positive",
                "ControlExplained",
                "official_no_debt",
                "controller_overhead_ratio",
                "B2_route_gate",
            ],
            limit=16,
        ),
        "\n## B3 Overhead Evidence\n",
        md_table(
            overhead_sorted,
            [
                "source_artifact",
                "recipe",
                "cached_controller_emit_cadence",
                "rows",
                "reported_overhead_le_0p25_rows",
                "reported_controller_overhead_ratio_median",
                "controller_share_of_full_step_median",
            ],
            limit=20,
        ),
        "\n## B3 Fused Controller Probe\n",
        md_table(
            fused_summary,
            [
                "mode",
                "direction_rows",
                "direction_positive_rows",
                "official_no_debt_rows",
                "overhead_le_0p25_rows",
                "route_gate_rows",
                "exploration_gate_rows_0p35",
                "mean_controller_overhead_ratio",
            ],
            limit=8,
        ),
        "\n## 分析与 Insight\n",
        "- B1 真实训练 probe 未支持 debt-orthogonal 晋升：debt_orthogonal 方向正向 7/12、official no-debt 4/12、overhead<=0.25 9/12，但四个 gate 同时成立只有 1/12；standard 同条件为 3/12，说明当前投影没有扩大可行域。\n",
        "- B2 KAN margin 只有单行 route gate，且 ControlExplained_rate=0.8645833333333334；这更像 observed-control 下的局部偶然优势，不能证明 KAN carrier 有稳定独立增益。\n",
        "- B3 fused controller 平均 overhead 略低于 unfused，但 fused/unfused route_gate 都是 3，未达到打开路线标准；结构共享能省一点开销，但没有解决 no-debt/direction 联立瓶颈。\n",
        "- 综合证据链：v22.46/v22.46b 的有效信号主要停留在 support-positive 或局部 matched-control positive，未形成官方 no-debt、低 overhead、方向增益三者同时稳定成立的路线。\n",
        "\n## 下一步\n",
        "- 本轮实验收尾并进入审计：核心代码、执行日志、复盘、route JSON/CSV artifact、B1/B3 chunk 摘要与失败日志打包到 `code_audit_pack/`。\n",
    ]
    RECAP_DOC.write_text("\n".join(text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "code-audit", "debt-orthogonal", "kan-margin", "overhead", "fused-probe", "b1-gradient-probe", "b1-gradient-recap", "recap"], default="all")
    parser.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fused-probe-steps", type=int, default=200)
    parser.add_argument("--b1-probe-steps", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--held-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_out()
    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    datasets = {item for item in args.datasets.split(",") if item}
    seeds = {item for item in args.seeds.split(",") if item}
    outputs = [
        "results/v22_46b/v22_46b_code_audit.json",
        "results/v22_46b/v22_46b_debt_orthogonal_route.json",
        "results/v22_46b/v22_46b_kan_margin_route.json",
        "results/v22_46b/v22_46b_overhead_route.json",
        "results/v22_46b/v22_46b_fused_controller_route.json",
        "results/v22_46b/v22_46b_b1_debt_gradient_route.json",
        "docs/DG-KAN_v22.46b_DebtControlledSignalIdentification_实验结果复盘.md",
    ]
    append_exec(
        command,
        task_id=f"v22.46b-{args.stage}",
        status="started",
        files=", ".join(outputs),
        note=(
            f"datasets={args.datasets}; seeds={args.seeds}; gpus={args.gpus}; workers={args.workers}; "
            f"stage_type={'paired training probe' if args.stage in {'fused-probe', 'b1-gradient-probe'} else 'posthoc/recap audit'}"
        ),
    )
    try:
        if args.stage in {"all", "code-audit"}:
            stage_code_audit()
        if args.stage in {"all", "debt-orthogonal"}:
            stage_debt_orthogonal(datasets, seeds)
        if args.stage in {"all", "kan-margin"}:
            stage_kan_margin(datasets, seeds)
        if args.stage in {"all", "overhead"}:
            stage_overhead(datasets, seeds)
        if args.stage in {"fused-probe"}:
            stage_fused_probe(args, datasets, seeds)
        if args.stage in {"b1-gradient-probe"}:
            stage_b1_gradient_probe(args, datasets, seeds)
        if args.stage in {"b1-gradient-recap"}:
            stage_b1_gradient_recap(args)
        if args.stage in {"all", "recap"}:
            if args.stage == "recap":
                stage_code_audit()
            stage_recap()
        if args.stage in {"fused-probe", "b1-gradient-probe", "b1-gradient-recap"}:
            stage_recap()
        append_exec(command, task_id=f"v22.46b-{args.stage}", status="completed", files=", ".join(outputs), exit_code=0)
        return 0
    except Exception as exc:  # pragma: no cover - written to log for auditability.
        err_path = LOG_ROOT / f"{safe_fragment('v22.46b_' + args.stage)}_error.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        append_exec(
            command,
            task_id=f"v22.46b-{args.stage}",
            status="failed",
            files=str(err_path.relative_to(ROOT)),
            note=repr(exc),
            exit_code=1,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
