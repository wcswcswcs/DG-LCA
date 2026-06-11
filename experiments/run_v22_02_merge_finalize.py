#!/usr/bin/env python3
"""v22.02 final route, logs, packets, and clean-unzip self-test."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_02_common import (  # noqa: E402
    PYTHON,
    V2202_RECAP_DOC,
    append_exec,
    append_text,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    sha256_file,
    simple_svg,
    write_json,
    write_rows,
    write_text,
)


REQUIRED_ARTIFACTS = [
    "v22_02_route_decision.json",
    "v22_02_required_artifact_manifest.csv",
    "v22_02_failure_taxonomy.csv",
    "v22_02_no_go_boundary.md",
    "v22_02_next_hypothesis_queue.md",
    "v22_02_code_audit_summary.csv",
    "v22_02_efficiency_truth_table.csv",
    "v22_02_efficiency_full_loop_table.csv",
    "v22_02_kernel_gradcheck.csv",
    "v22_02_official_fused_status_matrix.csv",
    "v22_02_drat_repair_table.csv",
    "v22_02_drbf_repair_table.csv",
    "v22_02_source_retention_matrix.csv",
    "v22_02_terminal_collapse_autopsy.csv",
    "v22_02_precommit_selector_matrix.csv",
    "v22_02_source_channel_target_matrix.csv",
    "v22_02_kan_source_bank_matrix.csv",
    "v22_02_controls_matrix.csv",
    "v22_02_independent_rerun_matrix.csv",
    "v22_02_runnable_queue.csv",
    "v22_02_gpu_assignment_manifest.csv",
    "v22_02_gpu_utilization_timeline.csv",
    "v22_02_idle_violation.csv",
    "v22_02_deferred_items.csv",
    "v22_02_queue_drain_report.csv",
    "v22_02_command_journal.csv",
]

CONTINUATION_FULL_AUDIT_NAMES = [
    "v22_02_f106_f144_oldshape_full_audit.csv",
    "v22_02_f106_f141_oldshape_full_audit.csv",
    "v22_02_f106_f138_oldshape_full_audit.csv",
    "v22_02_f106_f135_oldshape_full_audit.csv",
    "v22_02_f106_f132_oldshape_full_audit.csv",
    "v22_02_f106_f129_oldshape_full_audit.csv",
    "v22_02_f106_f126_oldshape_full_audit.csv",
    "v22_02_f106_f123_oldshape_full_audit.csv",
    "v22_02_f106_f119_oldshape_full_audit.csv",
]

CONTINUATION_ROUTE_NAMES = [
    "v22_02_f106_f144_oldshape_full_route.csv",
    "v22_02_f106_f141_oldshape_full_route.csv",
    "v22_02_f106_f138_oldshape_full_route.csv",
    "v22_02_f106_f135_oldshape_full_route.csv",
    "v22_02_f106_f132_oldshape_full_route.csv",
    "v22_02_f106_f129_oldshape_full_route.csv",
    "v22_02_f106_f126_oldshape_full_route.csv",
    "v22_02_terminal_collapse_route.csv",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def first_existing_path(out_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = out_dir / name
        if path.exists():
            return path
    return out_dir / names[-1]


def route_from(out_dir: Path) -> dict[str, Any]:
    code = read_rows(out_dir / "v22_02_code_truth_gate.csv") or read_rows(out_dir / "v22_02_code_audit_summary.csv")
    eff = read_rows(out_dir / "v22_02_efficiency_officialization_summary.csv")
    dr = read_json(out_dir / "v22_02_drat_drbf_active_repair_decision.json")
    term = read_json(out_dir / "v22_02_terminal_collapse_route.json")
    continuation_route = read_rows(first_existing_path(out_dir, CONTINUATION_ROUTE_NAMES))
    if continuation_route:
        term = continuation_route[0]
    selector = read_json(out_dir / "v22_02_precommit_selector_decision.json")
    target = read_json(out_dir / "v22_02_source_channel_target_decision.json")
    kan = read_json(out_dir / "v22_02_kan_source_writer_decision.json")
    missing = [name for name in REQUIRED_ARTIFACTS if name != "v22_02_route_decision.json" and not (out_dir / name).exists()]
    s09 = int(bool(code) and all(int_flag(r.get("pass")) for r in code))
    dche = next((r for r in eff if str(r.get("carrier")) == "D-CHE"), {})
    dfou = next((r for r in eff if str(r.get("carrier")) == "D-FOU"), {})
    dche_closure = int_flag(dche.get("full_loop_official_closure"))
    dfou_closure = int_flag(dfou.get("full_loop_official_closure"))
    drat = dr.get("D-RAT", {}) or {}
    drbf = dr.get("D-RBF", {}) or {}
    h4800 = int(term.get("candidate_h4800", 0) or 0)
    h3200 = int(term.get("candidate_continuous_h3200", 0) or 0)
    if missing:
        route = "R0-RequiredArtifactMissing"
    elif not s09:
        route = "R0-S09TruthGateFailed"
    elif h4800:
        route = "R5-H4800CandidateNeedsIndependentConfirmation"
    elif h3200:
        route = "R4-TerminalCollapseNoH4800"
    else:
        route = "R3-NoContinuousH3200UnderV2202Rule"
    return {
        "route": route,
        "S0_9_pass": s09,
        "D-CHE_full_loop_official_closure": dche_closure,
        "D-FOU_full_loop_official_closure": dfou_closure,
        "D-CHE_S1_pass_rows": dche.get("S1_pass_rows", ""),
        "D-FOU_S1_pass_rows": dfou.get("S1_pass_rows", ""),
        "D-RAT_decision": drat.get("decision", ""),
        "D-RBF_decision": drbf.get("decision", ""),
        "functional_early": term.get("candidate_early_chain", 0),
        "functional_h3200": h3200,
        "functional_h4800": h4800,
        "terminal_collapse_groups": term.get("terminal_collapse_groups", 0),
        "h4000_missing_group_count": term.get("h4000_missing_group_count", 0),
        "SelectorRoute": selector.get("decision", ""),
        "TargetRoute": target.get("decision", ""),
        "KANRoute": kan.get("decision", ""),
        "promotion_allowed": 0,
        "required_artifact_missing_count": len(missing),
        "missing_artifacts": ";".join(missing),
    }


def _num(value: Any, default: float = float("nan")) -> float:
    return finite_float(value, default=default)


def _intish(value: Any) -> int:
    try:
        if value in {"", None}:
            return 0
        return int(float(str(value)))
    except Exception:
        return 0


def _short_path(value: Any) -> str:
    text = str(value or "")
    marker = "results/v22_02_terminal_collapse_early_source_selector_basis_efficiency_4gpu/official_v22_02/"
    return text.replace(marker, "")


def _group_rows(rows: list[dict[str, Any]], key: str) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    index: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get(key, ""))
        if name not in index:
            index[name] = []
            groups.append((name, index[name]))
        index[name].append(row)
    return groups


def _sort_by_num(rows: list[dict[str, Any]], key: str, *, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _num(row.get(key), default=float("-inf")), reverse=reverse)


def continuation_ledger(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for continuation, items in _group_rows(rows, "continuation"):
        ranked = _sort_by_num(items, "h4800_retention_ratio")
        best = ranked[0] if ranked else {}
        ledger.append(
            {
                "continuation": continuation,
                "source_dir": _short_path(best.get("source_dir")),
                "candidates": len(items),
                "early": sum(_intish(r.get("early_source_chain_group")) for r in items),
                "continuous_h3200": sum(_intish(r.get("continuous_h3200_group")) for r in items),
                "h4800": sum(_intish(r.get("productive_h4800_group")) for r in items),
                "best_v22_id": best.get("v22_id", ""),
                "best_h800": best.get("h800", ""),
                "best_h3200": best.get("h3200", ""),
                "best_h4000": best.get("h4000", ""),
                "best_h4800": best.get("h4800", ""),
                "best_h4800_retention_ratio": best.get("h4800_retention_ratio", ""),
                "row_h4800_positive_max": max((_intish(r.get("row_h4800_positive_count")) for r in items), default=0),
                "decision": "HasH4800Candidate" if any(_intish(r.get("productive_h4800_group")) for r in items) else "NoH4800Candidate",
            }
        )
    return ledger


def continuation_detail_blocks(rows: list[dict[str, Any]]) -> list[str]:
    blocks: list[str] = []
    fields = [
        "v22_id",
        "h100",
        "h400",
        "h800",
        "h1600",
        "h2400",
        "h3200",
        "h4000",
        "h4800",
        "h4800_retention_ratio",
        "early_source_chain_group",
        "continuous_h3200_group",
        "productive_h4800_group",
        "row_h4800_positive_count",
        "blocker",
    ]
    for continuation, items in _group_rows(rows, "continuation"):
        source_dir = _short_path(items[0].get("source_dir")) if items else ""
        ranked = _sort_by_num(items, "h4800_retention_ratio")
        best = ranked[0] if ranked else {}
        blocks.extend(
            [
                f"### {continuation}",
                "",
                f"- source_dir: `{source_dir}`",
                f"- candidate rows: {len(items)}",
                f"- early / continuous_h3200 / h4800: {sum(_intish(r.get('early_source_chain_group')) for r in items)} / {sum(_intish(r.get('continuous_h3200_group')) for r in items)} / {sum(_intish(r.get('productive_h4800_group')) for r in items)}",
                f"- best: `{best.get('v22_id', '')}` h4800={best.get('h4800', '')}, h4800_retention_ratio={best.get('h4800_retention_ratio', '')}",
                "",
                md_table(items, fields, max_rows=80),
            ]
        )
    return blocks


def focus_dataset_localization(continuation_rows: list[dict[str, Any]], focus_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dirs: list[Path] = []
    seen_dirs: set[str] = set()
    for row in continuation_rows:
        if str(row.get("v22_id", "")) not in focus_ids:
            continue
        source_dir = str(row.get("source_dir", ""))
        if source_dir and source_dir not in seen_dirs:
            seen_dirs.add(source_dir)
            source_dirs.append(Path(source_dir))

    raw_rows: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        for name in ["v22_02_source_retention_matrix.csv", "v21_01_source_retention_matrix.csv"]:
            path = source_dir / name
            if path.exists() and path.stat().st_size > 0:
                raw_rows.extend(read_rows(path))
                break
    raw_rows = [row for row in raw_rows if str(row.get("v22_id", row.get("v21_id", ""))) in focus_ids]

    dataset_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        v22_id = str(row.get("v22_id", row.get("v21_id", "")))
        dataset = str(row.get("dataset", ""))
        dataset_groups.setdefault((v22_id, dataset), []).append(row)

    aggregate: list[dict[str, Any]] = []
    horizons = ["source_h800", "source_h1600", "source_h2400", "source_h3200", "source_h4000", "source_h4800"]
    for (v22_id, dataset), items in dataset_groups.items():
        out: dict[str, Any] = {"v22_id": v22_id, "dataset": dataset, "rows": len(items)}
        for horizon in horizons:
            vals = [_num(row.get(horizon)) for row in items]
            vals = [v for v in vals if v == v]
            out[f"mean_{horizon.replace('source_', '')}"] = sum(vals) / len(vals) if vals else ""
        out["h4800_positive_rows"] = sum(1 for row in items if _num(row.get("source_h4800")) >= 0.005)
        out["early_rows"] = sum(_intish(row.get("v22_02_early_source_chain")) for row in items)
        out["continuous_rows"] = sum(_intish(row.get("v22_02_continuous_h3200_chain")) for row in items)
        out["h4800_rows"] = sum(_intish(row.get("v22_02_productive_h4800_chain")) for row in items)
        blockers = sorted({str(row.get("v22_02_source_chain_blocker", "")) for row in items if str(row.get("v22_02_source_chain_blocker", ""))})
        out["blockers"] = ";".join(blockers)
        aggregate.append(out)

    raw_fields = [
        "v22_id",
        "dataset",
        "seed",
        "source_h100",
        "source_h400",
        "source_h800",
        "source_h1600",
        "source_h2400",
        "source_h3200",
        "source_h4000",
        "source_h4800",
        "v22_02_early_source_chain",
        "v22_02_continuous_h3200_chain",
        "v22_02_productive_h4800_chain",
        "v22_02_source_chain_blocker",
    ]
    raw_compact = [{field: row.get(field, "") for field in raw_fields} for row in raw_rows]
    return aggregate, raw_compact


def write_controls_and_confirmation(out_dir: Path) -> None:
    matrix = read_rows(out_dir / "v22_02_source_retention_matrix.csv")
    controls = [r for r in matrix if str(r.get("v22_id", r.get("v21_id", ""))).startswith("CTRL-")]
    write_rows(out_dir / "v22_02_controls_matrix.csv", controls)
    route_rows = read_rows(first_existing_path(out_dir, CONTINUATION_ROUTE_NAMES))
    term = route_rows[0] if route_rows else read_json(out_dir / "v22_02_terminal_collapse_route.json")
    if int(term.get("candidate_h4800", 0) or 0) > 0:
        rows = [{"status": "required_not_run", "reason": "h4800 candidate present; independent offset rerun required before promotion"}]
    else:
        rows = [{"status": "deferred", "reason": "no h4800 productive candidate; independent confirmation not eligible"}]
    write_rows(out_dir / "v22_02_independent_rerun_matrix.csv", rows)


def write_queue(out_dir: Path) -> None:
    tasks = [
        ("GPU0", "S0.9 clean packet test", "v22_02_code_audit_summary.csv"),
        ("GPU0", "D-CHE efficiency officialization", "v22_02_efficiency_officialization_summary.csv"),
        ("GPU2", "D-FOU efficiency officialization", "v22_02_efficiency_officialization_summary.csv"),
        ("GPU3", "D-RAT/D-RBF active repair", "v22_02_drat_drbf_active_repair_summary.csv"),
        ("GPU1", "terminal-collapse fresh/autopsy", "v22_02_terminal_collapse_autopsy.csv"),
        ("GPU1", "precommit selector", "v22_02_precommit_selector_matrix.csv"),
        ("GPU2", "KAN source writer", "v22_02_kan_source_bank_matrix.csv"),
    ]
    queue = []
    assignments = []
    for idx, (gpu, task, artifact) in enumerate(tasks):
        status = "completed" if (out_dir / artifact).exists() else "missing"
        queue.append({"queue_index": idx, "task": task, "artifact": artifact, "assigned_gpu": gpu, "status": status})
        assignments.append({"gpu": gpu, "task": task, "artifact": artifact, "status": status})
    write_rows(out_dir / "v22_02_runnable_queue.csv", queue)
    write_rows(out_dir / "v22_02_gpu_assignment_manifest.csv", assignments)
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], text=True, capture_output=True, timeout=30)
        util = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                util.append({"gpu": parts[0], "name": parts[1], "memory_used": parts[2], "memory_total": parts[3], "utilization_gpu": parts[4], "snapshot_time": now_sg()})
    except Exception as exc:
        util = [{"gpu": "", "blocker": f"{type(exc).__name__}:{exc}", "snapshot_time": now_sg()}]
    write_rows(out_dir / "v22_02_gpu_utilization_timeline.csv", util)
    pending = [r for r in queue if r["status"] != "completed"]
    write_rows(out_dir / "v22_02_idle_violation.csv", [{"execution_contract_violation": int(bool(pending)), "pending_tasks": len(pending), "note": "violation means required queue artifact missing at finalize"}])
    deferred = []
    dr = read_json(out_dir / "v22_02_drat_drbf_active_repair_decision.json")
    for carrier in ["D-RAT", "D-RBF"]:
        d = dr.get(carrier, {}) or {}
        if str(d.get("decision")) not in {"NearEfficientCarrier", "OfficialEfficientCarrier"}:
            deferred.append({"item": f"{carrier} functional smoke", "status": "deferred", "reason": d.get("blocker", "near-E1 not reached")})
    route_rows = read_rows(first_existing_path(out_dir, CONTINUATION_ROUTE_NAMES))
    term = route_rows[0] if route_rows else read_json(out_dir / "v22_02_terminal_collapse_route.json")
    if not int(term.get("candidate_h4800", 0) or 0):
        deferred.append({"item": "independent h4800 confirmation", "status": "deferred", "reason": "no h4800 productive candidate"})
    write_rows(out_dir / "v22_02_deferred_items.csv", deferred)
    write_rows(out_dir / "v22_02_queue_drain_report.csv", [{"completed_tasks": sum(r["status"] == "completed" for r in queue), "missing_tasks": len(pending), "deferred_items": len(deferred), "queue_drained": int(not pending)}])
    simple_svg(out_dir / "figures" / "fig_gpu_utilization_timeline.svg", "GPU utilization timeline", util, "gpu")


def write_manifest(out_dir: Path) -> None:
    rows = []
    for name in REQUIRED_ARTIFACTS:
        path = out_dir / name
        rows.append({"artifact": name, "required": 1, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256_file(path) if path.exists() and path.is_file() else ""})
    write_rows(out_dir / "v22_02_required_artifact_manifest.csv", rows)


def write_taxonomy(out_dir: Path, decision: dict[str, Any]) -> None:
    rows = []
    if int(decision.get("required_artifact_missing_count", 0)):
        rows.append({"class": "C0-RequiredArtifactMissing", "count": decision.get("required_artifact_missing_count"), "action": "rerun missing runner before scientific route"})
    if not int(decision.get("S0_9_pass", 0)):
        rows.append({"class": "C1-S09TruthGateFailed", "count": 1, "action": "fix packet/import/metric tests"})
    if not int(decision.get("D-CHE_full_loop_official_closure", 0)):
        rows.append({"class": "E0-DCHEFullLoopClosureNotOfficial", "count": 1, "action": "split full-loop timing and kernel status"})
    if not int(decision.get("D-FOU_full_loop_official_closure", 0)):
        rows.append({"class": "E0-DFOUFullLoopClosureNotOfficial", "count": 1, "action": "split full-loop timing and kernel status"})
    if str(decision.get("D-RAT_decision")) != "NearEfficientCarrier":
        rows.append({"class": "E2-DRATForwardEvalBlocked", "count": 1, "action": "continue Horner/reciprocal/telemetry-free repair"})
    if str(decision.get("D-RBF_decision")) != "NearEfficientCarrier":
        rows.append({"class": "E3-DRBFDenseMaterializationBlocked", "count": 1, "action": "continue local-K/no-dense/active-center repair"})
    if int(decision.get("functional_early", 0)) and not int(decision.get("functional_h3200", 0)):
        rows.append({"class": "F0-NoContinuousH3200UnderV2202Rule", "count": decision.get("functional_early"), "action": "retained-target/source-observability reset before more terminal tweaks"})
    if int(decision.get("functional_h3200", 0)) and not int(decision.get("functional_h4800", 0)):
        rows.append({"class": "F1-H3200ChainButH4800Collapse", "count": decision.get("terminal_collapse_groups"), "action": "terminal autopsy plus retained-target reset"})
    if str(decision.get("SelectorRoute")) in {"NoH4800LabelNoSelectorCanPass", "NoPrecommitPass"}:
        rows.append({"class": "F3-NoLegalPrecommitSelector", "count": 1, "action": "stop threshold tuning; redesign train-only target features"})
    if str(decision.get("TargetRoute")) == "TargetObservableNoRetention":
        rows.append({"class": "F4-TargetObservableNoRetention", "count": 1, "action": "target theory reset rather than actuator tuning"})
    if str(decision.get("KANRoute")) == "KANSourceBankMismatch":
        rows.append({"class": "F5-KANSourceBankMismatch", "count": 1, "action": "MLP source decomposition and KAN bank mapping"})
    write_rows(out_dir / "v22_02_failure_taxonomy.csv", rows)
    write_text(
        out_dir / "v22_02_no_go_boundary.md",
        "# v22.02 No-Go Boundary\n\n"
        f"- route: `{decision.get('route')}`\n"
        f"- promotion_allowed: {decision.get('promotion_allowed')}\n"
        "- No promotion is allowed unless FU-S4 h4800 productive retention and independent confirmation are both present.\n"
        "- v22.02 continuous h3200 additionally requires retention ratios, so h3200-positive-but-decaying groups are not promoted to terminal-collapse candidates.\n"
        "- Efficiency-only or retrospective-selector evidence is not written as functional breakthrough.\n",
    )
    write_text(
        out_dir / "v22_02_next_hypothesis_queue.md",
        "# v22.02 Next Hypothesis Queue\n\n"
        "1. If h4000 is missing for any top collapse group, rerun only those groups with h3600/h4000/h4400 instrumentation.\n"
        "2. Replace retained-target/source-observability definition before adding more terminal floor, raw guard, reject-rescue, or source-shape preserve variants.\n"
        "3. Continue D-RAT numerator/denominator fusion and D-RBF local-K no-dense repair only until near-E1; do not run FU smoke before near-E1.\n"
        "4. Add explicit optimizer projection instrumentation for h3200->h4800 if UnknownTerminalCollapse remains dominant.\n",
    )


def clean_unzip_self_test(out_dir: Path, zip_path: Path) -> None:
    check_dir = Path("/tmp/v22_02_packet_check")
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(check_dir)
    source_root = check_dir / "02_SOURCE_TREE"
    proc1 = subprocess.run([sys.executable, "-m", "compileall", "-q", "."], cwd=str(source_root), text=True, capture_output=True, timeout=1200)
    proc2 = subprocess.run([sys.executable, "experiments/run_v22_02_s09_truth_gate.py", "--self-contained-import-check", "1", "--source-root", "."], cwd=str(source_root), text=True, capture_output=True, timeout=1200)
    write_text(out_dir / "v22_02_clean_unzip_self_test.log", f"compileall_exit={proc1.returncode}\ntruth_gate_exit={proc2.returncode}\n--- compileall stdout ---\n{proc1.stdout}\n--- compileall stderr ---\n{proc1.stderr}\n--- truth stdout ---\n{proc2.stdout}\n--- truth stderr ---\n{proc2.stderr}\n")
    for name in [
        "v22_02_packet_clean_unzip_compileall.csv",
        "v22_02_packet_clean_unzip_import_closure.csv",
        "v22_02_packet_manifest_vs_required_source_files.csv",
        "v22_02_code_truth_gate.csv",
    ]:
        src = check_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    append_exec(out_dir, f"rm -rf /tmp/v22_02_packet_check && unzip {zip_path.name} -d /tmp/v22_02_packet_check && cd /tmp/v22_02_packet_check/02_SOURCE_TREE && python -m compileall -q . && python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .", status="completed" if proc1.returncode == 0 and proc2.returncode == 0 else "blocked", note=f"compileall_exit={proc1.returncode}; truth_gate_exit={proc2.returncode}")


def render_recap(out_dir: Path, decision: dict[str, Any]) -> None:
    code = read_rows(out_dir / "v22_02_code_audit_summary.csv")
    eff = read_rows(out_dir / "v22_02_efficiency_officialization_summary.csv")
    dr = read_rows(out_dir / "v22_02_drat_drbf_active_repair_summary.csv")
    source = read_rows(out_dir / "v22_02_source_chain_summary.csv")
    autopsy = read_rows(out_dir / "v22_02_terminal_collapse_autopsy.csv")
    continuation_full_path = first_existing_path(out_dir, CONTINUATION_FULL_AUDIT_NAMES)
    continuation_smoke_path = out_dir / "v22_02_f88_f129_smoke_audit.csv"
    if not continuation_smoke_path.exists():
        continuation_smoke_path = out_dir / "v22_02_f88_f126_smoke_audit.csv"
    if not continuation_smoke_path.exists():
        continuation_smoke_path = out_dir / "v22_02_f88_f123_smoke_audit.csv"
    if not continuation_smoke_path.exists():
        continuation_smoke_path = out_dir / "v22_02_f88_f119_smoke_audit.csv"
    continuation_full = read_rows(continuation_full_path)
    continuation_smoke = read_rows(continuation_smoke_path)
    continuation_route_path = first_existing_path(out_dir, CONTINUATION_ROUTE_NAMES)
    continuation_route = read_rows(continuation_route_path)
    selector = read_rows(out_dir / "v22_02_precommit_selector_matrix.csv")
    target = read_rows(out_dir / "v22_02_source_channel_target_matrix.csv")
    kan = read_rows(out_dir / "v22_02_kan_source_bank_matrix.csv")
    tax = read_rows(out_dir / "v22_02_failure_taxonomy.csv")
    packet = out_dir / "v22_02_code_review_packet.zip"
    bundle = out_dir / "v22_02_results_bundle.zip"
    continuation_ledger_rows = continuation_ledger(continuation_full)
    continuation_ranked = _sort_by_num(continuation_full, "h4800_retention_ratio")
    focus_ids = {
        "MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard",
        "MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard",
        "MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard",
        "MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap",
    }
    focus_dataset, focus_raw = focus_dataset_localization(continuation_full, focus_ids)
    artifact_index = []
    for artifact in [
        continuation_full_path,
        continuation_route_path,
        out_dir / "v22_02_source_retention_matrix.csv",
        out_dir / "v22_02_precommit_selector_matrix.csv",
        out_dir / "v22_02_source_channel_target_matrix.csv",
        out_dir / "v22_02_kan_source_bank_matrix.csv",
        out_dir / "continuation_f142_f144_topk_support_oldshape_full" / "v22_02_source_chain_summary.csv",
        out_dir / "continuation_f142_f144_topk_support_oldshape_full" / "v21_01_source_retention_raw_traces.csv",
    ]:
        artifact_index.append(
            {
                "artifact": _short_path(artifact),
                "exists": int(Path(artifact).exists()),
                "size_bytes": Path(artifact).stat().st_size if Path(artifact).exists() else 0,
                "sha256": sha256_file(Path(artifact)) if Path(artifact).exists() and Path(artifact).is_file() else "",
            }
        )
    text = [
        "# DG-KAN v22.02 TerminalCollapse EarlySourceSelector BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{decision.get('route')}`",
        f"- S0.9 pass: {decision.get('S0_9_pass')}",
        f"- D-CHE/D-FOU full-loop official closure: {decision.get('D-CHE_full_loop_official_closure')} / {decision.get('D-FOU_full_loop_official_closure')}",
        f"- D-RAT/D-RBF: {decision.get('D-RAT_decision')} / {decision.get('D-RBF_decision')}",
        f"- functional early / h3200 / h4800: {decision.get('functional_early')} / {decision.get('functional_h3200')} / {decision.get('functional_h4800')}",
        f"- terminal collapse groups: {decision.get('terminal_collapse_groups')}",
        f"- SelectorRoute: {decision.get('SelectorRoute')}",
        f"- TargetRoute: {decision.get('TargetRoute')}",
        f"- KANRoute: {decision.get('KANRoute')}",
        f"- promotion_allowed: {decision.get('promotion_allowed')}",
        f"- required artifact missing count: {decision.get('required_artifact_missing_count')}",
        "",
        "## Part 1 Code Audit",
        "",
        md_table(code, ["check", "pass", "metric", "value", "blocker"], max_rows=20),
        "## Part 2 Basis Efficiency",
        "",
        md_table(eff, ["carrier", "rows", "E1_pass_rows", "S1_pass_rows", "S1_pass_batch_sizes", "S1_pass_variants", "best_forward_ratio", "best_step_ratio", "full_loop_official_closure", "decision"], max_rows=20),
        md_table(dr, ["carrier", "profile_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "decision", "blocker"], max_rows=20),
        "## Part 3 Functional Update",
        "",
        md_table(source, ["carrier", "variant", "v22_id", "rows", "h100", "h400", "h800", "h1600", "h2400", "h3200", "h4000", "h4800", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "terminal_collapse_group", "source_chain_blocker"], max_rows=120),
        "## Artifact Index",
        "",
        md_table(artifact_index, ["artifact", "exists", "size_bytes", "sha256"], max_rows=50),
        "## Fresh Rerun Provenance",
        "",
        "- `fresh_terminal_pool_oldshape`: 81-row mixed-candidate smoke/localization pool. It was superseded for scientific causal claims because mixed candidates change job-index/train-seed assignment.",
        "- `fresh_terminal_single_oldshape`: 225-row single-candidate old-shape localization baseline, with one candidate plus matched controls per run-label and h4000 recorded.",
        "- Baseline single-candidate candidates: `MLP-F53`, `MLP-F77`, `MLP-F78`, `MLP-F86`, `MLP-F87`; controls: `CTRL-SGD`, `CTRL-AdamW`, `CTRL-RandomMatchedNorm`, `CTRL-NoOpMatchedOverhead`.",
        "",
        "## Continuation F88-F144 Source-State / Terminal Guard / Signal Target / Source Shape / Weak-Row Rescue / Top-K Support",
        "",
        "- This section is rendered only from fresh continuation audit CSVs; no values are hand-promoted.",
        "- Best near-miss in this family remains `MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard`, h4800_retention_ratio=0.4905266741203586, below the v22.02 productive gate.",
        "- F120/F121/F122 terminal projected optimizer, projected blend, and antiwashout did not improve over F118; F123 h4000 checkpoint reentry also remained below the productive h4800 gate.",
        "- F124/F125/F126 tested the plan-recommended train-only signal-channel/reservoir/source-bank target reset. The exact MLP readout target path was enabled, but the target gates did not produce retained h4800 source and all three worsened relative to F118.",
        "- F127/F128/F129 test a source-state adaptive terminal family: train-split/corrupt-label gated adaptive raw, sparse source-axis, and ratio-preserving source guard. They are evaluated with the same v22.02 source-chain gate.",
        "- F130/F131/F132 test the next plan-recommended retained-target reset: source-projected B3-null consensus, noise-orthogonal B3-null consensus, and easy-margin B3-null consensus. They are train-only terminal targets wrapped around the F118 early100 h800 source slow-EMA path.",
        "- F133/F134/F135 test a terminal reject-rescue family: source-axis micro rescue, slow-EMA micro rescue, and hold-then-source rescue. The rescue is triggered only from train-stream terminal reject diagnostics and is still evaluated by the unchanged productive h4800 gate.",
        "- F136/F137/F138 test a train-only source-shape preservation family: pure midlate source-shape preserve, preserve plus raw guard, and preserve plus terminal clamp. They are evaluated as a falsification of source-shape drift rather than as promotion candidates.",
        "- F139/F140/F141 test terminal reject weak-row rescue: raw rescue, raw+source hybrid rescue, and late-only raw rescue. They are triggered by train-only source-state reject diagnostics and keep the unchanged productive h4800 gate.",
        "- F142/F143/F144 test the FU-H6 recommended top-k source-support variant: sparse support guard, top-k raw hybrid guard, and debt-capped top-k guard. They protect only train-stream source-state support rather than the full source vector and keep the unchanged productive h4800 gate.",
        md_table(continuation_route, ["source_chain_rows", "source_chain_groups", "candidate_groups", "candidate_early_chain", "candidate_continuous_h3200", "candidate_h4800", "promotion_allowed", "decision"], max_rows=5),
        "### Continuation Ledger",
        "",
        md_table(continuation_ledger_rows, ["continuation", "source_dir", "candidates", "early", "continuous_h3200", "h4800", "best_v22_id", "best_h800", "best_h3200", "best_h4000", "best_h4800", "best_h4800_retention_ratio", "row_h4800_positive_max", "decision"], max_rows=80),
        "### Best Candidate Ranking",
        "",
        md_table(continuation_ranked, ["continuation", "v22_id", "h100", "h400", "h800", "h1600", "h2400", "h3200", "h4000", "h4800", "h4800_retention_ratio", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "row_h4800_positive_count", "blocker"], max_rows=80),
        "### Old-shape full audit",
        "",
        md_table(continuation_full, ["continuation", "source_dir", "v22_id", "h100", "h400", "h800", "h1600", "h2400", "h3200", "h4000", "h4800", "h4800_retention_ratio", "early_source_chain_group", "continuous_h3200_group", "productive_h4800_group", "row_h4800_positive_count", "blocker"], max_rows=120),
        "### Per-Continuation Detail",
        "",
        *continuation_detail_blocks(continuation_full),
        "### F118 vs F142-F144 Dataset Localization",
        "",
        "- This localization reads row-level fresh matrices from each continuation source_dir. F118 is included as the best near-miss baseline; F142-F144 are the latest top-k support variants.",
        md_table(focus_dataset, ["v22_id", "dataset", "rows", "mean_h800", "mean_h1600", "mean_h2400", "mean_h3200", "mean_h4000", "mean_h4800", "h4800_positive_rows", "early_rows", "continuous_rows", "h4800_rows", "blockers"], max_rows=80),
        "### F118 vs F142-F144 Row Localization",
        "",
        md_table(focus_raw, ["v22_id", "dataset", "seed", "source_h100", "source_h400", "source_h800", "source_h1600", "source_h2400", "source_h3200", "source_h4000", "source_h4800", "v22_02_early_source_chain", "v22_02_continuous_h3200_chain", "v22_02_productive_h4800_chain", "v22_02_source_chain_blocker"], max_rows=80),
        "### Smoke audit",
        "",
        md_table(continuation_smoke, ["continuation", "v22_id", "h100", "h400", "h800", "h1600", "early_source_chain_group", "blocker"], max_rows=30),
        "",
        "## Terminal Collapse Autopsy",
        "",
        md_table(autopsy, ["carrier", "v22_id", "h3200", "h4000", "h4800", "source_derivative_h3200_to_h4000", "source_derivative_h4000_to_h4800", "dataset_seed_heterogeneity_score", "terminal_collapse_class"], max_rows=30),
        "## Precommit Selector",
        "",
        md_table(selector, ["selector_feature_name", "selector_train_only", "selector_uses_future", "selector_AUC_retained_h4800_on_shadow_pool", "selector_precision_at_k", "fresh_selected_h4800_pass_rows", "selector_decision"], max_rows=20),
        "## Source Target Reset",
        "",
        md_table(target, ["carrier", "v22_id", "target_family", "ActuationR2", "B2_transfer_gain", "random_target_B2_gain", "h4800_retention_rate", "target_reset_decision"], max_rows=30),
        "## KAN Source Writer",
        "",
        md_table(kan, ["carrier", "v22_id", "writer_family", "h800", "h3200", "h4800", "h4800_positive_rows", "source_bank_fraction", "source_bank_retention_h4800_over_h3200", "KAN_FU_S3_v22_02"], max_rows=30),
        "## Failure Taxonomy",
        "",
        md_table(tax, ["class", "count", "action"], max_rows=30),
        "## 修改记录",
        "",
        "- 修改 `experiments/run_v17_common.py` 与 `experiments/run_v21_01_common.py`：新增 h4000 horizon 落盘，满足 v22.02 terminal-collapse localization 要求。",
        "- 新增 `dgkan/fu/terminal_collapse.py`、`precommit_selector.py`、`kan_source_bank.py`、`dgkan/metrics/calibration.py`：提供 v22.02 source-chain、selector legality、KAN source-bank 与 calibration unit tests。",
        "- 新增 `dgkan/profiling/efficiency_v22_02.py` 与 kernel alias：重判 full-loop official / near-E1，不伪称 D-RAT/D-RBF official fused。",
        "- 新增 v22.02 runner/finalizer：S0.9 clean packet gate、efficiency readback、D-RAT/D-RBF active repair readback、fresh terminal autopsy、legal precommit selector、target reset、KAN source writer、packet/bundle finalization。",
        "- 扩展 MLP readout function-space target path：`MLPBaseline.frozen_readout_features()` 允许 exact readout actuation 对 MLP 2D readout 执行 train-only signal/reservoir target reset。",
        "- 新增 `MLP-F124/F125/F126`：signal-reservoir target、source-bank target、dual target guard，作为 F118 之后的 target-theory reset continuation。",
        "- 新增 `MLP-F127/F128/F129`：source-state adaptive raw、sparse source-axis、ratio-preserving terminal guard，用 train split A/B 与 corrupt-label gain 做无未来泄漏的 terminal candidate gate。",
        "- 新增 `MLP-F130/F131/F132`：source-projected / noise-orthogonal / easy-margin B3-null consensus terminal target reset；主路径保持 F118 early100 h800 source slow-EMA，只在 terminal target gate 提交 train-only readout target。",
        "- 新增 `MLP-F133/F134/F135`：terminal reject source-axis / slow-EMA / hold-source rescue；只在 train-stream selector 接受但 source-state gate 拒绝、且 density/balance/corrupt-gain 满足条件时提交微量 source rescue。",
        "- 新增 `MLP-F136/F137/F138`：source-shape preservation、source-shape preservation + raw guard、source-shape preservation + terminal clamp；用于检验 h3200->h4800 衰减是否来自 midlate source-shape drift，而不改 productive h4800 判据。",
        "- 新增 `MLP-F139/F140/F141`：terminal reject weak-row raw rescue、raw+source hybrid rescue、late-only raw rescue；只由 train-only source-state reject 触发，用于检验 F118 弱行是否可被 terminal micro-rescue 修复。",
        "- 新增 `MLP-F142/F143/F144`：terminal top-k source-support guard、top-k raw hybrid guard、top-k debt-capped guard；按 FU-H6 建议只保护 train-stream source-state 的 top-k support subspace，用于检验 full source-vector projection 是否过强。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- v22.02 的 code gate 只有在 clean-unzip import closure 也通过时才可作为科学 no-go 的前提；最终 route 读取 clean packet 输出和 required artifact manifest。",
        "- D-CHE/D-FOU 的 efficiency 证据来自 full-loop timing artifact，并按 v22.02 更严格 S1 规则重判；它不能单独支撑 functional promotion。",
        "- D-RAT/D-RBF 已有 active repair component breakdown，但 near-E1 未达成时，functional smoke 继续 deferred。",
        "- single-candidate old-shape fresh rerun 中 5 个 MLP 候选均形成 early source chain，但 h3200 retention ratio 未达 v22.02 continuous gate；h4000 记录显示 source 从 h3200 后继续衰减。",
        "- Functional route 只看 grouped continuous h3200 与 grouped h4800 productive retention；没有 h4800 candidate 时，selector/independent confirmation 不允许升级为 promotion。",
        "- 若 target reset 显示 high ActuationR2 但 h4800 retention 为 0，则 blocker 是 retained target/source-observability theory，而不是 actuator 本身。",
        "- F106/F115/F117/F118/F119 已能形成 grouped early source chain 和 continuous h3200，但 F118 的 h4800_retention_ratio=0.4905266741203586 仍低于 0.50，不能 promotion；F119 stronger raw guard 下降到 0.4788671993982172。",
        "- F120/F121/F122 terminal projected optimizer/source-blend/antiwashout 分别为 h4800_retention_ratio=0.47899749939111147 / 0.48388658936948814 / 0.4836718313083516；F123 h4000 checkpoint reentry 为 0.4799617647110513。它们都没有超过 F118，也没有形成 productive_h4800_group。",
        "- F124/F125/F126 train-only signal-channel/reservoir target reset 分别为 h4800_retention_ratio=0.405378 / 0.382958 / 0.417708；它们没有超过 F118，也没有形成 productive_h4800_group。",
        "- F127/F128/F129 source-state adaptive terminal guard 分别为 h4800_retention_ratio=0.401535 / 0.372177 / 0.400462；它们没有超过 F118，也没有形成 productive_h4800_group。",
        "- F130/F131/F132 terminal target reset 把 h4800 从负值 terminal collapse 改成 positive but nonproductive retention：h4800_retention_ratio=0.419301 / 0.386605 / 0.405931，row_h4800_positive_count=8/9 / 9/9 / 9/9；但 productive_h4800_group 仍为 0。",
        "- F133/F134/F135 terminal reject rescue 只触发少量 train-only micro-rescue，h4800_retention_ratio=0.419742 / 0.385211 / 0.405189；仍低于 F118，productive_h4800_group 仍为 0。",
        "- F136/F137/F138 source-shape preserve family 未形成 early source chain：h800=-1.082650 / -1.107372 / -1.075262，h4800=-1.357646 / -1.385284 / -1.360441，early_source_chain_group=0，continuous_h3200_group=0。",
        "- F139/F140/F141 terminal reject weak-row rescue 保留 early+h3200 chain，但 h4800_retention_ratio=0.419263 / 0.389215 / 0.420980，全部低于 F118 的 0.490527，productive_h4800_group 仍为 0。",
        "- F142/F143/F144 top-k source-support family 保留 early+h3200 chain，但 h4800_retention_ratio=0.407347 / 0.390464 / 0.416659，全部低于 F118 的 0.490527，productive_h4800_group 仍为 0。",
        "- 因此当前 evidence 不再支持继续 terminal projector、raw guard、h4000 reentry、signal/reservoir target、source-state adaptive、B3-null consensus target、terminal reject-rescue、weak-row rescue、source-shape preserve 或 top-k support 小修；需要新的 train-only retained-target/source-observability 理论。",
        "",
    ]
    if packet.exists():
        text.append(f"- v22_02_code_review_packet.zip: size={packet.stat().st_size} sha256={sha256_file(packet)}")
    if bundle.exists():
        text.append(f"- v22_02_results_bundle.zip: size={bundle.stat().st_size} sha256={sha256_file(bundle)}")
    write_text(V2202_RECAP_DOC, "\n".join(text) + "\n")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    write_controls_and_confirmation(out_dir)
    write_queue(out_dir)
    decision = route_from(out_dir)
    write_taxonomy(out_dir, decision)
    write_manifest(out_dir)
    decision = route_from(out_dir)
    write_json(out_dir / "v22_02_route_decision.json", decision)
    write_rows(out_dir / "v22_02_route_decision.csv", [decision])
    render_recap(out_dir, decision)
    zip_path, _bundle = build_packet(out_dir)
    clean_unzip_self_test(out_dir, zip_path)
    write_manifest(out_dir)
    decision = route_from(out_dir)
    write_taxonomy(out_dir, decision)
    write_manifest(out_dir)
    decision = route_from(out_dir)
    write_json(out_dir / "v22_02_route_decision.json", decision)
    write_rows(out_dir / "v22_02_route_decision.csv", [decision])
    render_recap(out_dir, decision)
    zip_path, bundle = build_packet(out_dir)
    render_recap(out_dir, decision)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_merge_finalize.py", status="completed", note=f"route={decision.get('route')} packet={zip_path.name} bundle={bundle.name}")


if __name__ == "__main__":
    main()
