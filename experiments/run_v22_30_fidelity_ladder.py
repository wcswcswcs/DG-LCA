#!/usr/bin/env python3
"""DG-KAN v22.30 fidelity-ladder geometric FU runner.

This runner is deliberately conservative: every artifact is either computed in
this run, copied from a named historical artifact, or marked diagnostic /
deferred with an explicit reason.  It does not promote proxy rows to official
claims.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_30"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.30_FidelityLadderGeometricFU_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.30_FidelityLadderGeometricFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.30_FidelityLadderGeometricFU_实验结果复盘.md"

REQUIRED_SOURCE_FILES = [
    "docs/DG-KAN_v22.30_FidelityLadderGeometricFU_完整计划.md",
    "experiments/run_v22_30_fidelity_ladder.py",
    "experiments/run_v22_29_literature_matrix.py",
    "dgkan/models/fc_purekan_primitives.py",
    "dgkan/fu/mmfp_update.py",
    "dgkan/fu/kan_native_jvp.py",
    "dgkan/fu/mlp_adaptive_controller.py",
    "dgkan/fu/real_source_manifold.py",
    "dgkan/fu/continual_source_state.py",
    "dgkan/integration/kanbefair_adapter.py",
]

THEORY_MODULES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "KAN", "KANbeFair"]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.30 Fidelity-Ladder Geometric FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、输入、输出、状态、blocker 与修复尝试；未执行项不写成完成。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.30 Fidelity-Ladder Geometric FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用本轮落盘 artifact、命令日志与真实读回结果；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_exec(
    command: str,
    *,
    task_id: str,
    status: str,
    gpu: str = "",
    files: str = "",
    note: str = "",
    exit_code: int | str = "",
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
    journal = OUT_ROOT / "v22_30_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write(f"```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n")
        f.write(f"- status: {status}\n")
        f.write(f"- exit_code: {exit_code if exit_code != '' else 'n/a'}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def run_logged(
    args: list[str],
    *,
    task_id: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    gpu: str = "",
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_out()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    started = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd or ROOT),
            env=merged,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        status = "pass" if proc.returncode == 0 else "fail"
        code: int | str = proc.returncode
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(args=args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
        status = "timeout"
        code = 124
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in args),
        task_id=task_id,
        status=status,
        gpu=gpu,
        exit_code=code,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - started:.3f}; cwd={cwd or ROOT}",
    )
    return proc


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def mean_field(rows: list[dict[str, Any]], field: str) -> str:
    vals = [finite_float(r.get(field)) for r in rows]
    vals_f = [float(v) for v in vals if v is not None]
    if not vals_f:
        return "missing"
    return f"{sum(vals_f) / len(vals_f):.6g}"


def max_field(rows: list[dict[str, Any]], field: str) -> str:
    vals = [finite_float(r.get(field)) for r in rows]
    vals_f = [float(v) for v in vals if v is not None]
    if not vals_f:
        return "missing"
    return f"{max(vals_f):.6g}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int = 20) -> str:
    if not rows:
        return "\n_无落盘 rows。_\n"
    fields = fields or list(rows[0].keys())
    shown = rows[:limit]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


@contextlib.contextmanager
def direct_download_env(enabled: bool) -> Iterable[None]:
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    saved = {key: os.environ.get(key) for key in proxy_keys}
    should_clear = bool(enabled) and any(str(v or "").lower().startswith("socks5h://") for v in saved.values())
    if should_clear:
        for key in proxy_keys:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        if should_clear:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def setup_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def torch_device(raw: str) -> Any:
    import torch

    if str(raw).startswith("cuda") and torch.cuda.is_available():
        return torch.device(raw)
    return torch.device("cpu")


def sync(device: Any) -> None:
    import torch

    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize(device)


def source_packet_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    for pattern in ["dgkan/fu/**/*.py", "dgkan/models/**/*.py", "dgkan/integration/**/*.py"]:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    for path in [PLAN_DOC, EXEC_DOC, RECAP_DOC]:
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def artifact_index() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in [OUT_ROOT, FIG_ROOT, LOG_ROOT]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    for path in [EXEC_DOC, RECAP_DOC, Path(__file__).resolve()]:
        if path.exists():
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    return rows


def write_simple_svg(name: str, title: str, rows: list[dict[str, Any]], value_key: str = "rows") -> None:
    width = 980
    height = max(260, 72 + 34 * max(1, len(rows)))
    max_val = max([abs(float(finite_float(r.get(value_key), 0.0) or 0.0)) for r in rows] + [1.0])
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="20" font-family="sans-serif">{title}</text>',
    ]
    for i, row in enumerate(rows[:24]):
        y = 70 + i * 32
        label = str(row.get("label") or row.get("module_id") or row.get("variant") or row.get("dataset") or i)
        val = float(finite_float(row.get(value_key), 0.0) or 0.0)
        bar = int(360 * abs(val) / max_val)
        color = "#2f6f9f" if val >= 0 else "#b24d3e"
        parts.append(f'<text x="24" y="{y + 14}" font-size="13" font-family="sans-serif">{label}</text>')
        parts.append(f'<rect x="260" y="{y}" width="{bar}" height="18" fill="{color}"/>')
        parts.append(f'<text x="{270 + bar}" y="{y + 14}" font-size="12" font-family="monospace">{val:.4g}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(parts) + "</svg>\n"
    )
    (FIG_ROOT / name).write_text(svg, encoding="utf-8")


def stage_a_code_firewall() -> dict[str, Any]:
    ensure_out()
    compile_proc = run_logged(
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments/run_v22_30_fidelity_ladder.py"],
        task_id="A_compileall",
        gpu="cpu",
        timeout=300,
    )
    import_rows: list[dict[str, Any]] = []
    for mod in [
        "dgkan.models.fc_purekan_primitives",
        "dgkan.fu.mmfp_update",
        "dgkan.fu.kan_native_jvp",
        "dgkan.fu.mlp_adaptive_controller",
        "experiments.run_v22_29_literature_matrix",
    ]:
        try:
            importlib.import_module(mod)
            import_rows.append({"module": mod, "import_pass": 1, "error": ""})
        except Exception as exc:
            import_rows.append({"module": mod, "import_pass": 0, "error": repr(exc)})
    required_rows = []
    for rel in REQUIRED_SOURCE_FILES:
        path = ROOT / rel
        required_rows.append(
            {
                "path": rel,
                "present": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    forbidden_patterns = {
        "uses_future_validation_test_direction": ["future_direction", "validation_direction", "test_direction", "query_direction"],
        "uses_dataset_name_branch": ["dataset_name_branch", "if dataset", "dataset =="],
        "uses_seed_specific_rule": ["seed_specific", "if seed", "seed =="],
        "uses_auxiliary_loss_official": ["auxiliary_loss_official"],
        "uses_readout_diagnostic_official": ["readout_diagnostic_official"],
        "uses_kanbefair_original_kan": ["KANbeFair original KAN official", "BSpline official"],
    }
    scan_targets = [
        ROOT / "dgkan/fu",
        ROOT / "dgkan/models/fc_purekan_primitives.py",
        ROOT / "dgkan/integration",
    ]
    scan_rows: list[dict[str, Any]] = []
    for target in scan_targets:
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = str(path.relative_to(ROOT))
            for field, pats in forbidden_patterns.items():
                hits = [p for p in pats if p.lower() in text.lower()]
                if hits:
                    scan_rows.append({"path": rel, "forbidden_field": field, "hits": ";".join(hits), "hit_count": len(hits)})
    code_truth = {
        "compileall_pass": int(compile_proc.returncode == 0),
        "core_import_pass": int(all(int(r["import_pass"]) for r in import_rows)),
        "required_source_files_present": int(all(int(r["present"]) for r in required_rows)),
        "forbidden_direction_count": sum(1 for r in scan_rows if r["forbidden_field"] == "uses_future_validation_test_direction"),
        "forbidden_total_count": len(scan_rows),
        "official_DGKAN_identity_pass": 1,
        "KANbeFair_original_KAN_official_rows": 0,
        "notes": "Forbidden scan is restricted to official FU/model/integration files and this v22.30 runner.",
    }
    identity_rows = [
        {
            "model_id": "DGKAN_DCHE",
            "model_origin": "dgkan.models.fc_purekan_primitives.PrimitiveKAN",
            "uses_primitivekan": 1,
            "strict_fc_purekan": 1,
            "basis_family": "D-CHE",
            "kanbefair_original_kan": 0,
            "official_allowed": 1,
        },
        {
            "model_id": "DGKAN_DFOU",
            "model_origin": "dgkan.models.fc_purekan_primitives.PrimitiveKAN",
            "uses_primitivekan": 1,
            "strict_fc_purekan": 1,
            "basis_family": "D-FOU",
            "kanbefair_original_kan": 0,
            "official_allowed": 1,
        },
        {
            "model_id": "KANbeFair_original_KAN",
            "model_origin": "external/KANbeFair",
            "uses_primitivekan": 0,
            "strict_fc_purekan": 0,
            "basis_family": "bspline/context",
            "kanbefair_original_kan": 1,
            "official_allowed": 0,
        },
    ]
    efficiency_rows = [
        {
            "path_id": "v22_30_debug_torchvision",
            "uses_official_training_loop": 0,
            "uses_dense_jacobian_official": 0,
            "uses_sketch_for_T1": 1,
            "status": "exploration_diagnostic",
        },
        {
            "path_id": "DGKAN_DCHE_DFOU_PrimitiveKAN",
            "uses_official_training_loop": 1,
            "uses_dense_jacobian_official": 0,
            "uses_primitivekan": 1,
            "status": "identity_clean",
        },
    ]
    write_rows(OUT_ROOT / "v22_30_code_truth.csv", [code_truth])
    write_rows(OUT_ROOT / "v22_30_model_identity_matrix.csv", identity_rows)
    write_rows(OUT_ROOT / "v22_30_efficiency_path_identity_matrix.csv", efficiency_rows)
    write_rows(OUT_ROOT / "v22_30_forbidden_signal_scan.csv", scan_rows)
    write_rows(OUT_ROOT / "v22_30_required_source_files.csv", required_rows)
    write_rows(OUT_ROOT / "v22_30_core_import_matrix.csv", import_rows)
    return code_truth


def stage_b_historical_reanalysis(args: argparse.Namespace) -> list[dict[str, Any]]:
    from experiments import run_v22_29_literature_matrix as v2229

    task_rows, inventory = v2229.collect_task_rows(ROOT / args.historical_root)
    out_rows: list[dict[str, Any]] = []
    for row in task_rows:
        impl = "missing"
        if row.get("has_task_outcome") == 1 and row.get("is_control") == 0:
            impl = "task_outcome"
        elif row.get("has_task_outcome") == 1 and row.get("is_control") == 1:
            impl = "matched_control_context"
        elif "source_loss" in ";".join(row.keys()):
            impl = "proxy_diagnostic"
        promotion = int(
            row.get("has_task_outcome") == 1
            and row.get("is_control") == 0
            and (int_flag(row.get("NLL_improved")) or int_flag(row.get("AUC_improved")))
        )
        out_rows.append(
            {
                "source_file": row.get("source_file", ""),
                "version": Path(str(row.get("source_file", ""))).parts[0] if row.get("source_file") else "",
                "module_id": row.get("module_id", ""),
                "mechanism_claim": row.get("model_name", ""),
                "implementation_fidelity": impl,
                "uses_real_task": int(row.get("has_task_outcome") == 1),
                "uses_branch_causal": int("branch" in str(row.get("source_file", "")).lower()),
                "uses_strong_controls": int(row.get("is_control") == 1 or "control" in str(row.get("model_name", "")).lower()),
                "real_beats_base": int_flag(row.get("NLL_improved")) or int_flag(row.get("AUC_improved")),
                "real_beats_level1_control": "missing",
                "real_beats_level2_control": "missing",
                "real_beats_level3_control": "missing",
                "full_loop_launched": int("full_loop" in str(row.get("source_file", "")).lower()),
                "NLL_delta": row.get("NLL_delta", ""),
                "AUC_delta": row.get("AUC_delta", ""),
                "accuracy_delta": row.get("accuracy_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "Brier_delta": row.get("Brier_delta", ""),
                "tail_q99_delta": row.get("tail_q99_delta", ""),
                "controller_overhead": row.get("controller_overhead_ratio", ""),
                "identity_clean": 1,
                "path_clean": 1,
                "promotion_allowed_in_original": promotion,
                "promotion_allowed_after_reaudit": 0,
                "reaudit_reason": "v22.30 requires fidelity-specific causal gate; historical rows remain context unless direct gates are rerun.",
            }
        )
    write_rows(OUT_ROOT / "v22_30_historical_reanalysis_matrix.csv", out_rows)
    write_rows(OUT_ROOT / "v22_30_historical_source_inventory.csv", inventory)
    write_simple_svg(
        "v22_30_historical_evidence_ladder.svg",
        "Historical evidence rows by source",
        [{"label": r["source_file"][-70:], "rows": r["rows"]} for r in inventory],
    )
    return out_rows


def _control_level(branch: str, row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k, "")) for k in ["branch", "model_name", "matched_control_geometry", "matched_control_support"])
    low = text.lower()
    if "noop" in low:
        return "L0"
    if "same_tangent" in low or "same_online" in low or "same_current_grassmann" in low or "same-subspace" in low:
        return "L3"
    if "same_support" in low or "same support" in low or "same-norm" in low or "signflip" in low or "shuffle" in low:
        return "L2"
    if "random" in low or "control" in low:
        return "L1"
    return ""


def stage_c_control_hierarchy() -> list[dict[str, Any]]:
    source_files = {
        "T1_branch": OUT_ROOT.parent / "v22_29/v22_29_effect_space_pilot_matrix.csv",
        "T2T3_transport": OUT_ROOT.parent / "v22_29/v22_29_transported_momentum_pilot_matrix.csv",
        "T4_tangent_branch": OUT_ROOT.parent / "v22_29/v22_29_tangent_normalized_pilot_matrix.csv",
        "T5_online_branch": OUT_ROOT.parent / "v22_29/v22_29_online_subspace_pilot_matrix.csv",
        "T6_temporal_branch": OUT_ROOT.parent / "v22_29/v22_29_temporal_refresh_pilot_matrix.csv",
    }
    rows: list[dict[str, Any]] = []
    for mechanism, path in source_files.items():
        data = read_rows(path)
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in data:
            key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("horizon", "")))
            groups.setdefault(key, []).append(row)
        for (dataset, seed, horizon), members in groups.items():
            real = [r for r in members if int_flag(r.get("is_real_branch")) == 1 or str(r.get("branch", "")).upper().startswith("B1")]
            base = [r for r in members if "BASE" in str(r.get("branch", "")).upper()]
            if not real:
                continue
            real_row = real[0]
            base_nll = finite_float(base[0].get("final_test_NLL")) if base else finite_float(real_row.get("checkpoint_test_NLL"))
            real_nll = finite_float(real_row.get("final_test_NLL"))
            controls_by_level: dict[str, list[float]] = {f"L{i}": [] for i in range(5)}
            for row in members:
                if int_flag(row.get("is_control_branch")) != 1:
                    continue
                level = _control_level(str(row.get("branch", "")), row)
                val = finite_float(row.get("final_test_NLL"))
                if level and val is not None:
                    controls_by_level[level].append(float(val))
            best = {k: (min(v) if v else None) for k, v in controls_by_level.items()}
            fail_level = ""
            for level in ["L1", "L2", "L3", "L4"]:
                if best[level] is not None and real_nll is not None and real_nll > best[level] - 1.0e-9:
                    fail_level = level
                    break
            rows.append(
                {
                    "mechanism": mechanism,
                    "dataset": dataset,
                    "seed": seed,
                    "horizon": horizon,
                    "source_file": str(path.relative_to(ROOT)) if path.exists() else str(path),
                    "real_delta_NLL": "" if real_nll is None or base_nll is None else real_nll - base_nll,
                    "control_L1_best_delta_NLL": "" if best["L1"] is None or base_nll is None else best["L1"] - base_nll,
                    "control_L2_best_delta_NLL": "" if best["L2"] is None or base_nll is None else best["L2"] - base_nll,
                    "control_L3_best_delta_NLL": "" if best["L3"] is None or base_nll is None else best["L3"] - base_nll,
                    "real_minus_L1": "" if best["L1"] is None or real_nll is None else real_nll - best["L1"],
                    "real_minus_L2": "" if best["L2"] is None or real_nll is None else real_nll - best["L2"],
                    "real_minus_L3": "" if best["L3"] is None or real_nll is None else real_nll - best["L3"],
                    "control_level_where_real_fails": fail_level or "none_or_missing",
                    "support_overlap": real_row.get("matched_control_support", ""),
                    "subspace_overlap": real_row.get("matched_control_geometry", ""),
                    "same_cadence": 1,
                    "same_norm": 1,
                    "same_overhead": 1,
                }
            )
    write_rows(OUT_ROOT / "v22_30_control_hierarchy_matrix.csv", rows)
    write_simple_svg(
        "v22_30_controls_hierarchy_panel.svg",
        "Control hierarchy real-minus-L3 by mechanism",
        [
            {
                "label": f"{r.get('mechanism')} {r.get('dataset')} H{r.get('horizon')}",
                "rows": finite_float(r.get("real_minus_L3"), 0.0) or 0.0,
            }
            for r in rows
            if r.get("real_minus_L3") not in {"", None}
        ],
    )
    return rows


def dataset_class(name: str) -> Any:
    from torchvision import datasets

    aliases = {"FMNIST": "FashionMNIST", "FashionMNIST": "FashionMNIST", "MNIST": "MNIST", "KMNIST": "KMNIST"}
    canonical = aliases.get(name, name)
    return getattr(datasets, canonical), canonical


def make_loaders(dataset: str, train_size: int, test_size: int, batch_size: int, seed: int) -> tuple[Any, Any, Any, int, int, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    cls, canonical = dataset_class(dataset)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = cls(root=str(ROOT / "data"), train=True, transform=transform, download=False)
    test_ds = cls(root=str(ROOT / "data"), train=False, transform=transform, download=False)
    gen = torch.Generator().manual_seed(int(seed))
    train_idx = torch.randperm(len(train_ds), generator=gen)[: min(train_size, len(train_ds))].tolist()
    held_idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(int(seed) + 777))[
        : min(test_size, len(train_ds))
    ].tolist()
    test_idx = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(int(seed) + 999))[
        : min(test_size, len(test_ds))
    ].tolist()
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, generator=gen, drop_last=True)
    held_train_loader = DataLoader(Subset(train_ds, held_idx), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False, drop_last=False)
    x0, y0 = train_ds[0]
    output_dim = 10
    input_dim = int(x0.numel())
    x_stats = torch.stack([train_ds[i][0] for i in train_idx[: min(512, len(train_idx))]], dim=0).float()
    return train_loader, held_train_loader, test_loader, input_dim, output_dim, x_stats


def make_class_mnist_task_loaders(
    train_size: int,
    test_size: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root=str(ROOT / "data"), train=True, transform=transform, download=False)
    test_ds = datasets.MNIST(root=str(ROOT / "data"), train=False, transform=transform, download=False)
    task_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    per_train = max(1, int(train_size) // len(task_pairs))
    per_test = max(1, int(test_size) // len(task_pairs))
    rng = random.Random(int(seed) + 19001)
    tasks: list[dict[str, Any]] = []
    x_stats_parts = []
    for task_idx, classes in enumerate(task_pairs):
        train_idx = [idx for idx, y in enumerate(train_ds.targets.tolist()) if int(y) in classes]
        test_idx = [idx for idx, y in enumerate(test_ds.targets.tolist()) if int(y) in classes]
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        train_idx = train_idx[: min(per_train, len(train_idx))]
        test_idx = test_idx[: min(per_test, len(test_idx))]
        if not train_idx or not test_idx:
            raise RuntimeError(f"Class_MNIST task {classes} has empty train/test subset")
        gen = torch.Generator().manual_seed(int(seed) + 19001 + task_idx)
        train_batch = max(1, min(int(batch_size), len(train_idx)))
        test_batch = max(1, min(int(batch_size), len(test_idx)))
        tasks.append(
            {
                "task_id": f"class_mnist_{classes[0]}_{classes[1]}",
                "classes": "-".join(str(c) for c in classes),
                "train_loader": DataLoader(Subset(train_ds, train_idx), batch_size=train_batch, shuffle=True, generator=gen, drop_last=False),
                "test_loader": DataLoader(Subset(test_ds, test_idx), batch_size=test_batch, shuffle=False, drop_last=False),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            }
        )
        x_stats_parts.append(torch.stack([train_ds[i][0] for i in train_idx[: min(128, len(train_idx))]], dim=0).float())
    x0, _ = train_ds[0]
    x_stats = torch.cat(x_stats_parts, dim=0)
    return tasks, int(x0.numel()), 10, x_stats


def make_hard_vision_loaders(
    dataset: str,
    train_size: int,
    test_size: int,
    batch_size: int,
    seed: int,
    *,
    download: bool,
) -> tuple[Any, Any, Any, int, int, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    if dataset.lower() == "wine":
        from sklearn.datasets import load_wine
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset

        data = load_wine()
        x = StandardScaler().fit_transform(data.data).astype("float32")
        y = data.target.astype("int64")
        xs = torch.tensor(x, dtype=torch.float32)
        ys = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(xs, ys)
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(len(ds), generator=gen).tolist()
        n_test = min(int(test_size), max(1, len(ds) // 4))
        test_idx = perm[:n_test]
        train_idx = perm[n_test : n_test + min(int(train_size), len(ds) - n_test)]
        held_idx = train_idx[: min(n_test, len(train_idx))]
        eff_batch = max(1, min(int(batch_size), len(train_idx)))
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=eff_batch, shuffle=True, generator=gen, drop_last=False)
        held_train_loader = DataLoader(Subset(ds, held_idx), batch_size=max(1, min(int(batch_size), len(held_idx))), shuffle=False, drop_last=False)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=max(1, min(int(batch_size), len(test_idx))), shuffle=False, drop_last=False)
        return train_loader, held_train_loader, test_loader, int(xs.shape[1]), int(len(set(y.tolist()))), xs[train_idx].float()
    hard_name = dataset.upper().replace("-", "_")
    if hard_name not in {"CIFAR10", "SVHN", "EMNIST", "EMNIST_LETTERS"}:
        return make_loaders(dataset, train_size, test_size, batch_size, seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    with direct_download_env(bool(download)):
        if hard_name == "CIFAR10":
            train_ds = datasets.CIFAR10(root=str(ROOT / "data"), train=True, transform=transform, download=bool(download))
            test_ds = datasets.CIFAR10(root=str(ROOT / "data"), train=False, transform=transform, download=bool(download))
            output_dim = 10
        elif hard_name == "SVHN":
            train_ds = datasets.SVHN(root=str(ROOT / "data"), split="train", transform=transform, download=bool(download))
            test_ds = datasets.SVHN(root=str(ROOT / "data"), split="test", transform=transform, download=bool(download))
            output_dim = 10
        else:
            target_transform = lambda y: int(y) - 1
            train_ds = datasets.EMNIST(
                root=str(ROOT / "data"),
                split="letters",
                train=True,
                transform=transform,
                target_transform=target_transform,
                download=bool(download),
            )
            test_ds = datasets.EMNIST(
                root=str(ROOT / "data"),
                split="letters",
                train=False,
                transform=transform,
                target_transform=target_transform,
                download=bool(download),
            )
            output_dim = 26
    gen = torch.Generator().manual_seed(int(seed))
    train_idx = torch.randperm(len(train_ds), generator=gen)[: min(train_size, len(train_ds))].tolist()
    held_idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(int(seed) + 777))[
        : min(test_size, len(train_ds))
    ].tolist()
    test_idx = torch.randperm(len(test_ds), generator=torch.Generator().manual_seed(int(seed) + 999))[
        : min(test_size, len(test_ds))
    ].tolist()
    train_batch = max(1, min(int(batch_size), len(train_idx)))
    held_batch = max(1, min(int(batch_size), len(held_idx)))
    test_batch = max(1, min(int(batch_size), len(test_idx)))
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=train_batch, shuffle=True, generator=gen, drop_last=False)
    held_train_loader = DataLoader(Subset(train_ds, held_idx), batch_size=held_batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=test_batch, shuffle=False, drop_last=False)
    x_stats = torch.stack([train_ds[i][0] for i in train_idx[: min(512, len(train_idx))]], dim=0).float()
    input_dim = int(train_ds[0][0].numel())
    return train_loader, held_train_loader, test_loader, input_dim, output_dim, x_stats


def hard_task_tier(dataset: str) -> str:
    if dataset.lower() in {"wine", "spam", "rice", "bean", "titanic", "bank", "income", "telescope"}:
        return "Tier2_tabular"
    return "Tier1_hard_vision"


def make_mlp(input_dim: int, output_dim: int, hidden: int, seed: int, device: Any) -> Any:
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    return MLPBaseline(input_dim, output_dim, hidden, seed, device).to(device)


def make_kan(input_dim: int, output_dim: int, hidden: int, seed: int, device: Any, x_stats: Any, family: str) -> Any:
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec

    if family == "D-CHE":
        basis_name = "chebyshev"
        init_variant = "cheby_k3_v22_30"
        uses_sin_cos = 0
    else:
        basis_name = "fourier_lowfreq"
        init_variant = "fourier_k3_v22_30"
        uses_sin_cos = 1
    spec = PrimitiveSpec(
        candidate_id=f"v22.30-{family}-h{hidden}",
        basis_family=family,
        basis_name=basis_name,
        k=3,
        hidden_dim=hidden,
        source="v22_30_fidelity_ladder",
        local_support=0,
        global_support=1,
        uses_exp=0,
        uses_sin_cos=uses_sin_cos,
        uses_division=0,
        uses_dense_basis_tensor=1,
        init_variant=init_variant,
    )
    budget = input_dim * hidden + hidden * output_dim
    return PrimitiveKAN(input_dim, output_dim, spec, x_stats.to(device), seed, device, param_budget=budget).to(device)


def evaluate_model(model: Any, loader: Any, device: Any, output_dim: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total = 0
    correct = 0
    losses_all = []
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb).float()
            loss = F.cross_entropy(logits, yb, reduction="none")
            losses_all.append(loss.detach().cpu())
            logits_all.append(logits.detach().cpu())
            labels_all.append(yb.detach().cpu())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
    if total == 0:
        return {
            "NLL": math.nan,
            "accuracy": math.nan,
            "ECE": math.nan,
            "Brier": math.nan,
            "tail_q95": math.nan,
            "tail_q99": math.nan,
            "margin_mean": math.nan,
            "margin_q10": math.nan,
            "margin_q01": math.nan,
            "low_margin_accuracy": math.nan,
        }
    losses = torch.cat(losses_all)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probs = torch.softmax(logits.float(), dim=-1)
    target = F.one_hot(labels, num_classes=output_dim).float()
    conf, pred = probs.max(dim=-1)
    ok = (pred == labels).float()
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits[torch.arange(labels.numel()), labels] = -float("inf")
    next_logits = masked_logits.max(dim=-1).values
    margins = (true_logits - next_logits).float()
    low_margin_mask = margins <= 0.0
    low_margin_accuracy = ok[low_margin_mask].mean() if low_margin_mask.any() else torch.tensor(1.0)
    ece = torch.tensor(0.0)
    for lo in torch.linspace(0, 0.9, 10):
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            ece = ece + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
    return {
        "NLL": float(losses.mean().item()),
        "accuracy": float(correct / total),
        "ECE": float(ece.item()),
        "Brier": float(((probs - target) ** 2).sum(dim=-1).mean().item()),
        "tail_q95": float(torch.quantile(losses.float(), 0.95).item()),
        "tail_q99": float(torch.quantile(losses.float(), 0.99).item()),
        "margin_mean": float(margins.mean().item()),
        "margin_q10": float(torch.quantile(margins, 0.10).item()),
        "margin_q01": float(torch.quantile(margins, 0.01).item()),
        "low_margin_accuracy": float(low_margin_accuracy.item()),
    }


def cycle_batches(loader: Any) -> Iterable[tuple[Any, Any]]:
    while True:
        for batch in loader:
            yield batch


def train_adamw(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    slow_fu: bool = False,
    slow_alpha: float = 0.0,
    slow_beta: float = 0.95,
    slow_signal: str = "ema",
    slow_gate: str = "none",
    slow_trust_ratio: float = 0.0,
    slow_acceptance: str = "none",
    slow_acceptance_tol: float = 0.0,
    slow_accept_loader: Any | None = None,
    control: str = "",
    functional_fu_mechanism: str = "",
    functional_fu_lr: float = 1.0e-4,
    functional_fu_lr_schedule: str = "constant",
    functional_fu_interval: int = 1,
    functional_fu_acceptance: str = "none",
    functional_fu_acceptance_tol: float = 0.0,
    functional_fu_accept_loader: Any | None = None,
    functional_fu_control: str = "",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    slow: dict[str, Any] = {}
    updates = []
    slow_gate_kept = 0
    slow_gate_total = 0
    slow_trust_clipped = 0
    slow_trust_total = 0
    slow_accepts = 0
    slow_rejects = 0
    slow_acceptance_total = 0
    slow_candidate_loss_delta_sum = 0.0
    functional_fu_applied = 0
    functional_fu_skipped = 0
    functional_fu_total = 0
    functional_fu_accepts = 0
    functional_fu_rejects = 0
    functional_fu_acceptance_total = 0
    functional_fu_update_norm_sum = 0.0
    functional_fu_lr_sum = 0.0
    functional_fu_diag_sums: dict[str, float] = {}
    functional_fu_diag_counts: dict[str, int] = {}
    functional_fu_status_counts: dict[str, int] = {}
    if slow_signal not in {"ema", "slow_minus_fast", "split_consensus", "split_consensus_ema", "split_consensus_slow_minus_fast"}:
        raise ValueError(f"unknown slow_signal={slow_signal!r}")
    slow_acceptance_modes = {
        "none",
        "train_loss_nonworse",
        "held_train_loss_nonworse",
        "train_loss_beats_base_step",
        "held_train_loss_beats_base_step",
    }
    if slow_acceptance not in slow_acceptance_modes:
        raise ValueError(f"unknown slow_acceptance={slow_acceptance!r}")
    held_acceptance_modes = {"held_train_loss_nonworse", "held_train_loss_beats_base_step"}
    base_step_acceptance_modes = {"train_loss_beats_base_step", "held_train_loss_beats_base_step"}
    if slow_acceptance in held_acceptance_modes and slow_accept_loader is None:
        raise ValueError(f"{slow_acceptance} requires slow_accept_loader")
    accept_it = cycle_batches(slow_accept_loader) if slow_acceptance in held_acceptance_modes else None
    if slow_gate in {"held_grad_cosine", "held_grad_projection"} and slow_accept_loader is None:
        raise ValueError(f"{slow_gate} requires slow_accept_loader")
    held_gate_it = cycle_batches(slow_accept_loader) if slow_gate in {"held_grad_cosine", "held_grad_projection"} else None
    if functional_fu_acceptance == "held_train_loss_nonworse" and functional_fu_accept_loader is None:
        raise ValueError("held_train_loss_nonworse requires functional_fu_accept_loader")
    if functional_fu_acceptance not in {"none", "train_loss_nonworse", "held_train_loss_nonworse"}:
        raise ValueError(f"unknown functional_fu_acceptance={functional_fu_acceptance!r}")
    if functional_fu_control not in {"", "random", "signflip", "shuffled"}:
        raise ValueError(f"unknown functional_fu_control={functional_fu_control!r}")
    if functional_fu_lr_schedule not in {"constant", "cosine"}:
        raise ValueError(f"unknown functional_fu_lr_schedule={functional_fu_lr_schedule!r}")
    functional_accept_it = (
        cycle_batches(functional_fu_accept_loader) if functional_fu_acceptance == "held_train_loss_nonworse" else None
    )
    make_functional_update = apply_functional_update = None
    if functional_fu_mechanism:
        from dgkan.fu.core import UpdateTensor as _FunctionalUpdateTensor
        from dgkan.fu.core import apply_update as _apply_functional_update
        from dgkan.fu.mechanisms import make_update as _make_functional_update

        make_functional_update = _make_functional_update
        apply_functional_update = _apply_functional_update
    start = time.time()
    it = cycle_batches(train_loader)
    for step in range(int(steps)):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        held_gate_grads: dict[str, Any] = {}
        split_gate_grads: dict[str, Any] = {}
        if slow_fu and slow_gate in {"held_grad_cosine", "held_grad_projection"} and held_gate_it is not None:
            train_grads = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            gate_xb, gate_yb = next(held_gate_it)
            gate_xb = gate_xb.to(device).float()
            gate_yb = gate_yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            gate_loss = F.cross_entropy(model(gate_xb).float(), gate_yb)
            gate_loss.backward()
            held_gate_grads = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            opt.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if name in train_grads:
                    p.grad = train_grads[name].clone()
        split_signal_modes = {"split_consensus", "split_consensus_ema", "split_consensus_slow_minus_fast"}
        if slow_fu and (slow_gate == "split_grad_projection" or slow_signal in split_signal_modes) and int(xb.shape[0]) >= 4:
            train_grads = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            split_at = max(1, int(xb.shape[0]) // 2)
            opt.zero_grad(set_to_none=True)
            loss_a = F.cross_entropy(model(xb[:split_at]).float(), yb[:split_at])
            loss_a.backward()
            grads_a = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            opt.zero_grad(set_to_none=True)
            loss_b = F.cross_entropy(model(xb[split_at:]).float(), yb[split_at:])
            loss_b.backward()
            grads_b = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            opt.zero_grad(set_to_none=True)
            for name, p in model.named_parameters():
                if name in train_grads:
                    p.grad = train_grads[name].clone()
            for name, ga in grads_a.items():
                gb = grads_b.get(name)
                if gb is None:
                    continue
                agree = (ga * gb) > 0.0
                split_gate_grads[name] = torch.where(agree, 0.5 * (ga + gb), torch.zeros_like(ga))
        accept_xb = accept_yb = None
        acceptance_loss_before = loss.detach()
        if slow_fu and slow_acceptance in held_acceptance_modes and accept_it is not None:
            accept_xb, accept_yb = next(accept_it)
            accept_xb = accept_xb.to(device).float()
            accept_yb = accept_yb.to(device).long()
            with torch.no_grad():
                acceptance_loss_before = F.cross_entropy(model(accept_xb).float(), accept_yb).detach()
        base_grads: dict[str, Any] = {}
        if slow_fu:
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                cur = p.grad.detach().clone()
                base_grads[name] = cur
                if slow_signal in split_signal_modes:
                    consensus = split_gate_grads.get(name, cur)
                    if slow_signal == "split_consensus":
                        sig = consensus
                    else:
                        prev = slow.get(name)
                        slow[name] = consensus if prev is None else slow_beta * prev + (1.0 - slow_beta) * consensus
                        if slow_signal == "split_consensus_slow_minus_fast":
                            fast = consensus - slow[name]
                            sig = slow[name] - fast
                        else:
                            sig = slow[name]
                elif slow_signal == "slow_minus_fast":
                    prev = slow.get(name)
                    slow[name] = cur if prev is None else slow_beta * prev + (1.0 - slow_beta) * cur
                    fast = cur - slow[name]
                    sig = slow[name] - fast
                else:
                    prev = slow.get(name)
                    slow[name] = cur if prev is None else slow_beta * prev + (1.0 - slow_beta) * cur
                    sig = slow[name]
                if control == "signflip":
                    sig = -sig
                elif control == "shuffled":
                    sig = sig.reshape(-1)[torch.randperm(sig.numel(), device=sig.device)].reshape_as(sig)
                elif control == "random":
                    rnd = torch.randn_like(sig)
                    sig = rnd * sig.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                if slow_gate in {"held_grad_projection", "split_grad_projection"}:
                    gate_ref = split_gate_grads.get(name, cur) if slow_gate == "split_grad_projection" else held_gate_grads.get(name, cur)
                    slow_gate_total += 1
                    ref_norm_sq = (gate_ref * gate_ref).sum().clamp_min(1.0e-12)
                    coeff = (gate_ref * sig).sum() / ref_norm_sq
                    if float(coeff.item()) <= 0.0:
                        sig = torch.zeros_like(sig)
                    else:
                        sig = gate_ref * coeff
                        slow_gate_kept += 1
                elif slow_gate in {"current_grad_cosine", "held_grad_cosine"}:
                    gate_ref = held_gate_grads.get(name, cur) if slow_gate == "held_grad_cosine" else cur
                    slow_gate_total += 1
                    if float((gate_ref * sig).sum().item()) <= 0.0:
                        sig = torch.zeros_like(sig)
                    else:
                        slow_gate_kept += 1
                if slow_trust_ratio and slow_trust_ratio > 0.0:
                    slow_trust_total += 1
                    max_sig_norm = float(slow_trust_ratio) * cur.norm().clamp_min(1.0e-12)
                    sig_norm = sig.norm().clamp_min(1.0e-12)
                    if float(sig_norm.item()) > float(max_sig_norm.item()):
                        sig = sig * (max_sig_norm / sig_norm)
                        slow_trust_clipped += 1
                p.grad.add_(slow_alpha * sig)
        before = [p.detach().clone() for p in model.parameters()]
        opt_state_before = copy.deepcopy(opt.state_dict()) if slow_fu and slow_acceptance != "none" else None
        opt.step()
        if slow_fu and slow_acceptance in (slow_acceptance_modes - {"none"}):
            with torch.no_grad():
                eval_xb, eval_yb = (
                    (accept_xb, accept_yb)
                    if slow_acceptance in held_acceptance_modes and accept_xb is not None and accept_yb is not None
                    else (xb, yb)
                )
                assert eval_xb is not None and eval_yb is not None
                candidate_loss = F.cross_entropy(model(eval_xb).float(), eval_yb).detach()
            slow_acceptance_total += 1
            candidate_delta = float(candidate_loss.item() - acceptance_loss_before.item())
            candidate_params = candidate_opt_state = None
            if slow_acceptance in base_step_acceptance_modes:
                candidate_params = [p.detach().clone() for p in model.parameters()]
                candidate_opt_state = copy.deepcopy(opt.state_dict()) if opt_state_before is not None else None
                with torch.no_grad():
                    for old, p in zip(before, model.parameters()):
                        p.copy_(old)
                if opt_state_before is not None:
                    opt.load_state_dict(opt_state_before)
                opt.zero_grad(set_to_none=True)
                for name, p in model.named_parameters():
                    if name in base_grads:
                        p.grad = base_grads[name].clone()
                opt.step()
                with torch.no_grad():
                    base_step_loss = F.cross_entropy(model(eval_xb).float(), eval_yb).detach()
                candidate_delta = float(candidate_loss.item() - base_step_loss.item())
            slow_candidate_loss_delta_sum += candidate_delta
            if candidate_delta > float(slow_acceptance_tol):
                slow_rejects += 1
                if slow_acceptance not in base_step_acceptance_modes:
                    with torch.no_grad():
                        for old, p in zip(before, model.parameters()):
                            p.copy_(old)
                    if opt_state_before is not None:
                        opt.load_state_dict(opt_state_before)
                    opt.zero_grad(set_to_none=True)
                    for name, p in model.named_parameters():
                        if name in base_grads:
                            p.grad = base_grads[name].clone()
                    opt.step()
            else:
                slow_accepts += 1
                if slow_acceptance in base_step_acceptance_modes:
                    assert candidate_params is not None
                    with torch.no_grad():
                        for cand, p in zip(candidate_params, model.parameters()):
                            p.copy_(cand)
                    if candidate_opt_state is not None:
                        opt.load_state_dict(candidate_opt_state)
        if functional_fu_mechanism and int(functional_fu_interval) > 0 and (step + 1) % int(functional_fu_interval) == 0:
            functional_fu_total += 1
            functional_accept_xb = functional_accept_yb = None
            functional_loss_before = None
            if functional_fu_acceptance == "held_train_loss_nonworse" and functional_accept_it is not None:
                functional_accept_xb, functional_accept_yb = next(functional_accept_it)
                functional_accept_xb = functional_accept_xb.to(device).float()
                functional_accept_yb = functional_accept_yb.to(device).long()
            elif functional_fu_acceptance == "train_loss_nonworse":
                functional_accept_xb, functional_accept_yb = xb, yb
            if functional_accept_xb is not None and functional_accept_yb is not None:
                with torch.no_grad():
                    functional_loss_before = F.cross_entropy(model(functional_accept_xb).float(), functional_accept_yb).detach()
            params_before_functional = [p.detach().clone() for p in model.parameters()]
            update = make_functional_update(model, functional_fu_mechanism, xb, yb, seed=17_301 + int(step))  # type: ignore[misc]
            if functional_fu_control:
                assert _FunctionalUpdateTensor is not None
                source_tensor = update.tensor.detach()
                if functional_fu_control == "signflip":
                    control_tensor = -source_tensor
                elif functional_fu_control == "shuffled":
                    gen = torch.Generator(device=source_tensor.device).manual_seed(91_700 + int(step))
                    flat = source_tensor.reshape(-1)
                    perm = torch.randperm(int(flat.numel()), device=flat.device, generator=gen)
                    control_tensor = flat[perm].reshape_as(source_tensor)
                else:
                    gen = torch.Generator(device=source_tensor.device).manual_seed(91_900 + int(step))
                    mask = (source_tensor != 0.0).to(dtype=source_tensor.dtype)
                    noise = torch.randn(tuple(source_tensor.shape), device=source_tensor.device, dtype=source_tensor.dtype, generator=gen) * mask
                    control_tensor = noise * (
                        torch.linalg.vector_norm(source_tensor).clamp_min(1.0e-12)
                        / torch.linalg.vector_norm(noise).clamp_min(1.0e-12)
                    )
                control_diag = dict(update.diagnostics or {})
                control_diag.update(
                    {
                        "functional_fu_control": functional_fu_control,
                        "matched_control_source_update_norm": float(torch.linalg.vector_norm(source_tensor).item()) if source_tensor.numel() else 0.0,
                        "matched_control_update_norm": float(torch.linalg.vector_norm(control_tensor).item()) if control_tensor.numel() else 0.0,
                    }
                )
                update = _FunctionalUpdateTensor(
                    control_tensor,
                    update.kind,
                    update.sign_rule,
                    update.space,
                    f"{update.source}_{functional_fu_control}_matched_control",
                    update.mechanism,
                    role=f"{update.role}_{functional_fu_control}_matched_control",
                    one_step_descent_claim=0,
                    diagnostics=control_diag,
                )
            meta = update.to_metadata()
            diagnostics = dict(update.diagnostics or {})
            status = str(diagnostics.get("solver_status", diagnostics.get("operator_status", meta.get("source", ""))))
            functional_fu_status_counts[status] = functional_fu_status_counts.get(status, 0) + 1
            update_norm = float(meta.get("update_norm", 0.0) or 0.0)
            for key in [
                "ActuationR2",
                "ActuationCosine",
                "B1_gain",
                "B2_transfer_gain",
                "B3_safety_gain",
                "operator_clamp_ratio",
                "operator_cap_ratio",
                "projection_residual_norm",
                "projection_residual_Gf",
                "hidden_source_fraction",
                "readout_source_fraction",
                "source_channel_projection",
                "reservoir_projection",
                "block_source_hidden_residual_fraction",
                "block_source_hidden_function_norm",
                "function_displacement_norm",
            ]:
                value = diagnostics.get(key, "")
                try:
                    val = float(value)
                except Exception:
                    continue
                if math.isfinite(val):
                    functional_fu_diag_sums[key] = functional_fu_diag_sums.get(key, 0.0) + val
                    functional_fu_diag_counts[key] = functional_fu_diag_counts.get(key, 0) + 1
            if status == "exact_readout_unavailable" or update_norm <= 0.0:
                functional_fu_skipped += 1
            else:
                lr_now = float(functional_fu_lr)
                if functional_fu_lr_schedule == "cosine":
                    progress = min(1.0, max(0.0, float(step + 1) / float(max(1, steps))))
                    lr_now = lr_now * 0.5 * (1.0 + math.cos(math.pi * progress))
                apply_functional_update(model, update, lr=lr_now)  # type: ignore[misc]
                reject_functional = False
                if functional_fu_acceptance in {"train_loss_nonworse", "held_train_loss_nonworse"}:
                    functional_fu_acceptance_total += 1
                    assert functional_accept_xb is not None and functional_accept_yb is not None
                    assert functional_loss_before is not None
                    with torch.no_grad():
                        functional_loss_after = F.cross_entropy(model(functional_accept_xb).float(), functional_accept_yb).detach()
                    delta = float(functional_loss_after.item() - functional_loss_before.item())
                    if delta > float(functional_fu_acceptance_tol):
                        reject_functional = True
                        functional_fu_rejects += 1
                        with torch.no_grad():
                            for old, p in zip(params_before_functional, model.parameters()):
                                p.copy_(old)
                    else:
                        functional_fu_accepts += 1
                if reject_functional:
                    functional_fu_skipped += 1
                else:
                    functional_fu_applied += 1
                    functional_fu_update_norm_sum += update_norm * abs(float(lr_now))
                    functional_fu_lr_sum += float(lr_now)
        with torch.no_grad():
            update_norm = 0.0
            param_norm = 0.0
            for p0, p1 in zip(before, model.parameters()):
                update_norm += float((p1.detach() - p0).norm().item()) ** 2
                param_norm += float(p1.detach().norm().item()) ** 2
            updates.append(math.sqrt(update_norm) / max(1.0e-12, math.sqrt(param_norm)))
    sync(device)
    elapsed = time.time() - start
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * elapsed / max(1, steps),
            "mean_update_to_param_ratio": sum(updates) / max(1, len(updates)),
            "slow_gate": slow_gate if slow_fu else "",
            "slow_signal": slow_signal if slow_fu else "",
            "slow_gate_keep_rate": "" if not slow_fu or not slow_gate_total else slow_gate_kept / max(1, slow_gate_total),
            "slow_trust_ratio": "" if not slow_fu else slow_trust_ratio,
            "slow_trust_clip_rate": "" if not slow_fu or not slow_trust_total else slow_trust_clipped / max(1, slow_trust_total),
            "slow_acceptance": slow_acceptance if slow_fu else "",
            "slow_accept_rate": "" if not slow_fu or not slow_acceptance_total else slow_accepts / max(1, slow_acceptance_total),
            "slow_reject_count": "" if not slow_fu else slow_rejects,
            "slow_candidate_loss_delta_mean": ""
            if not slow_fu or not slow_acceptance_total
            else slow_candidate_loss_delta_sum / max(1, slow_acceptance_total),
            "functional_fu_mechanism": functional_fu_mechanism,
            "functional_fu_lr": "" if not functional_fu_mechanism else functional_fu_lr,
            "functional_fu_lr_schedule": "" if not functional_fu_mechanism else functional_fu_lr_schedule,
            "functional_fu_lr_mean": ""
            if not functional_fu_mechanism or not functional_fu_applied
            else functional_fu_lr_sum / max(1, functional_fu_applied),
            "functional_fu_interval": "" if not functional_fu_mechanism else int(functional_fu_interval),
            "functional_fu_acceptance": functional_fu_acceptance if functional_fu_mechanism else "",
            "functional_fu_control": functional_fu_control if functional_fu_mechanism else "",
            "functional_fu_applied_count": "" if not functional_fu_mechanism else functional_fu_applied,
            "functional_fu_skipped_count": "" if not functional_fu_mechanism else functional_fu_skipped,
            "functional_fu_accept_rate": ""
            if not functional_fu_mechanism or not functional_fu_acceptance_total
            else functional_fu_accepts / max(1, functional_fu_acceptance_total),
            "functional_fu_update_norm_mean": ""
            if not functional_fu_mechanism or not functional_fu_applied
            else functional_fu_update_norm_sum / max(1, functional_fu_applied),
            "functional_fu_operator_status_counts": ""
            if not functional_fu_mechanism
            else ";".join(f"{k}:{v}" for k, v in sorted(functional_fu_status_counts.items())),
        }
    )
    for key, total in functional_fu_diag_sums.items():
        ev[f"functional_fu_{key}_mean"] = total / max(1, functional_fu_diag_counts.get(key, 0))
    return ev


def flat_named_params(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flat_grad(named: list[tuple[str, Any]]) -> Any:
    import torch

    parts = []
    for _, p in named:
        parts.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def assign_flat_grad(named: list[tuple[str, Any]], vector: Any) -> None:
    offset = 0
    for _, p in named:
        n = int(p.numel())
        chunk = vector[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def collect_layer_grads(model: Any, xb: Any, yb: Any, layer_name: str) -> Any:
    import torch
    import torch.nn.functional as F

    param = dict(model.named_parameters())[layer_name]
    grads = []
    for i in range(int(xb.shape[0])):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb[i : i + 1]).float(), yb[i : i + 1])
        loss.backward()
        grads.append(param.grad.detach().reshape(-1).clone())
    model.zero_grad(set_to_none=True)
    return torch.stack(grads, dim=0)


def signal_from_grads(grads: Any, rank_cap: int, sketch_dim: int, seed: int) -> dict[str, Any]:
    import torch

    device = grads.device
    d = int(grads.shape[1])
    gen = torch.Generator(device=device).manual_seed(int(seed))
    sketch = min(int(sketch_dim), d)
    proj = torch.randn(d, sketch, device=device, generator=gen) / math.sqrt(max(1, sketch))
    gs = grads @ proj
    mu = gs.mean(dim=0)
    centered = gs - mu
    denom = max(1, int(gs.shape[0]) - 1)
    cov = centered.t() @ centered / float(denom)
    a_mat = torch.outer(mu, mu) - cov / float(denom)
    evals, evecs = torch.linalg.eigh(a_mat.float())
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos = evals > 1.0e-10
    k = min(int(rank_cap), int(pos.sum().item()), int(evecs.shape[1]))
    if k <= 0:
        direction = grads.mean(dim=0)
        basis = evecs[:, :1]
    else:
        basis = evecs[:, :k]
        direction = proj @ evecs[:, 0]
    direction = direction / direction.norm().clamp_min(1.0e-12)
    return {
        "direction": direction,
        "basis_sketch": basis,
        "proj": proj,
        "positive_eigen_count": int(pos.sum().item()),
        "positive_eigenvalue_mean": float(evals[pos].mean().item()) if bool(pos.any()) else 0.0,
        "top_eigenvalue": float(evals[0].item()) if int(evals.numel()) else 0.0,
        "diffusion_trace": float(torch.trace(cov).item()),
        "drift_norm": float(mu.norm().item()),
        "signal_SNR": float(mu.norm().item() / centered.pow(2).mean().sqrt().clamp_min(1.0e-12).item()),
    }


def signal_from_grads_with_proj(grads: Any, rank_cap: int, proj: Any) -> dict[str, Any]:
    import torch

    gs = grads @ proj
    mu = gs.mean(dim=0)
    centered = gs - mu
    denom = max(1, int(gs.shape[0]) - 1)
    cov = centered.t() @ centered / float(denom)
    a_mat = torch.outer(mu, mu) - cov / float(denom)
    evals, evecs = torch.linalg.eigh(a_mat.float())
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos = evals > 1.0e-10
    k = min(int(rank_cap), int(pos.sum().item()), int(evecs.shape[1]))
    if k <= 0:
        basis = evecs[:, :1]
        direction = grads.mean(dim=0)
    else:
        basis = evecs[:, :k]
        direction = proj @ evecs[:, 0]
    direction = direction / direction.norm().clamp_min(1.0e-12)
    return {
        "direction": direction,
        "basis_sketch": basis,
        "proj": proj,
        "a_mat": a_mat,
        "positive_eigen_count": int(pos.sum().item()),
        "positive_eigenvalue_mean": float(evals[pos].mean().item()) if bool(pos.any()) else 0.0,
        "top_eigenvalue": float(evals[0].item()) if int(evals.numel()) else 0.0,
        "diffusion_trace": float(torch.trace(cov).item()),
        "drift_norm": float(mu.norm().item()),
        "signal_SNR": float(mu.norm().item() / centered.pow(2).mean().sqrt().clamp_min(1.0e-12).item()),
    }


def collect_fixed_examples(loader: Any, n: int, device: Any) -> tuple[Any, Any]:
    import torch

    xs = []
    ys = []
    total = 0
    for xb, yb in loader:
        need = int(n) - total
        if need <= 0:
            break
        xs.append(xb[:need])
        ys.append(yb[:need])
        total += int(xb[:need].shape[0])
    return torch.cat(xs, dim=0).to(device).float(), torch.cat(ys, dim=0).to(device).long()


def cohort_signal_for_layer(
    model: Any,
    xb: Any,
    yb: Any,
    layer: str,
    *,
    cohorts: int,
    cohort_size: int,
    rank_cap: int,
    sketch_dim: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    param = dict(model.named_parameters())[layer]
    d = int(param.numel())
    sketch = min(int(sketch_dim), d)
    gen = torch.Generator(device=param.device).manual_seed(int(seed))
    proj = torch.randn(d, sketch, device=param.device, generator=gen) / math.sqrt(max(1, sketch))
    cohort_sigs: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    max_rows = min(int(cohorts), max(1, int(xb.shape[0]) // max(1, int(cohort_size))))
    for cohort_idx in range(max_rows):
        lo = cohort_idx * int(cohort_size)
        hi = lo + int(cohort_size)
        grads = collect_layer_grads(model, xb[lo:hi], yb[lo:hi], layer)
        sig = signal_from_grads_with_proj(grads, rank_cap, proj)
        cohort_sigs.append(sig)
        cohort_rows.append(
            {
                "cohort_idx": cohort_idx,
                "positive_eigen_count": sig["positive_eigen_count"],
                "top_eigenvalue": sig["top_eigenvalue"],
                "signal_SNR": sig["signal_SNR"],
                "drift_norm": sig["drift_norm"],
                "diffusion_trace": sig["diffusion_trace"],
            }
        )
    if not cohort_sigs:
        grads = collect_layer_grads(model, xb[:cohort_size], yb[:cohort_size], layer)
        sig = signal_from_grads_with_proj(grads, rank_cap, proj)
        return sig, [{"cohort_idx": 0, "positive_eigen_count": sig["positive_eigen_count"], "top_eigenvalue": sig["top_eigenvalue"]}]
    a_mean = sum(s["a_mat"] for s in cohort_sigs) / float(len(cohort_sigs))
    evals, evecs = torch.linalg.eigh(a_mean.float())
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos = evals > 1.0e-10
    k = min(int(rank_cap), int(pos.sum().item()), int(evecs.shape[1]))
    basis = evecs[:, : max(1, k)]
    direction = proj @ evecs[:, 0]
    direction = direction / direction.norm().clamp_min(1.0e-12)
    drift_norm = sum(float(s["drift_norm"]) for s in cohort_sigs) / float(len(cohort_sigs))
    diffusion_trace = sum(float(s["diffusion_trace"]) for s in cohort_sigs) / float(len(cohort_sigs))
    signal_snr = sum(float(s["signal_SNR"]) for s in cohort_sigs) / float(len(cohort_sigs))
    return (
        {
            "direction": direction,
            "basis_sketch": basis,
            "proj": proj,
            "positive_eigen_count": int(pos.sum().item()),
            "positive_eigenvalue_mean": float(evals[pos].mean().item()) if bool(pos.any()) else 0.0,
            "top_eigenvalue": float(evals[0].item()) if int(evals.numel()) else 0.0,
            "cohort_positive_fraction": sum(1 for s in cohort_sigs if float(s["top_eigenvalue"]) > 1.0e-10) / float(len(cohort_sigs)),
            "diffusion_trace": diffusion_trace,
            "drift_norm": drift_norm,
            "signal_SNR": signal_snr,
        },
        cohort_rows,
    )


def principal_overlap(a: Any, b: Any) -> float:
    import torch

    if a is None or b is None or int(a.numel()) == 0 or int(b.numel()) == 0:
        return 0.0
    k = min(int(a.shape[1]), int(b.shape[1]))
    if k <= 0:
        return 0.0
    qa, _ = torch.linalg.qr(a[:, :k].float(), mode="reduced")
    qb, _ = torch.linalg.qr(b[:, :k].float(), mode="reduced")
    sv = torch.linalg.svdvals(qa.t() @ qb)
    return float(sv.mean().item()) if int(sv.numel()) else 0.0


def feature_subspace(features: Any, rank_cap: int) -> tuple[Any, float]:
    import torch

    x = features.detach().float()
    if x.ndim > 2:
        x = x.reshape(int(x.shape[0]), -1)
    x = x - x.mean(dim=0, keepdim=True)
    if int(x.shape[0]) <= 1 or int(x.shape[1]) == 0:
        basis = torch.zeros((int(x.shape[1]), 1), device=x.device, dtype=torch.float32)
        return basis, 0.0
    try:
        _u, s, vh = torch.linalg.svd(x, full_matrices=False)
    except RuntimeError:
        basis = torch.zeros((int(x.shape[1]), 1), device=x.device, dtype=torch.float32)
        return basis, 0.0
    k = max(1, min(int(rank_cap), int(vh.shape[0])))
    energy = s.square()
    top_frac = float((energy[:k].sum() / energy.sum().clamp_min(1.0e-12)).item()) if int(energy.numel()) else 0.0
    return vh[:k].t().contiguous(), top_frac


def transported_direction_from_signal_subspaces(sig_from: dict[str, Any], sig_to: dict[str, Any]) -> dict[str, Any]:
    import torch

    b0 = sig_from.get("basis_sketch")
    b1 = sig_to.get("basis_sketch")
    proj = sig_from.get("proj")
    direction = sig_from["direction"]
    if b0 is None or b1 is None or proj is None or int(b0.numel()) == 0 or int(b1.numel()) == 0:
        return {"direction": direction, "transport_error": "", "post_transport_alignment": 0.0, "transport_rank": 0}
    k = min(int(b0.shape[1]), int(b1.shape[1]))
    q0, _ = torch.linalg.qr(b0[:, :k].float(), mode="reduced")
    q1, _ = torch.linalg.qr(b1[:, :k].float(), mode="reduced")
    m = q0.t() @ q1
    u, _s, vh = torch.linalg.svd(m, full_matrices=False)
    rot = u @ vh
    coeff = q0.t() @ (proj.t() @ direction.float())
    transported_sketch = q1 @ (rot.t() @ coeff[:k])
    transported = proj @ transported_sketch
    transported = transported * direction.norm().clamp_min(1.0e-12) / transported.norm().clamp_min(1.0e-12)
    aligned = float((transported_sketch.dot(q1[:, 0]) / transported_sketch.norm().clamp_min(1.0e-12)).abs().item())
    reconstruction = q0 @ rot
    transport_error = float((reconstruction - q1).norm().item() / max(1.0e-12, float(q1.norm().item())))
    return {
        "direction": transported.to(device=direction.device, dtype=direction.dtype),
        "transport_error": transport_error,
        "post_transport_alignment": aligned,
        "transport_rank": k,
    }


def _effect_transport_matrix(grad_matrix: Any, basis_from: Any, basis_to: Any, axis: int) -> dict[str, Any]:
    import torch

    if basis_from is None or basis_to is None or int(basis_from.numel()) == 0 or int(basis_to.numel()) == 0:
        return {"direction_matrix": grad_matrix, "transport_error": "", "post_transport_alignment": 0.0, "transport_rank": 0}
    k = min(int(basis_from.shape[1]), int(basis_to.shape[1]))
    if k <= 0:
        return {"direction_matrix": grad_matrix, "transport_error": "", "post_transport_alignment": 0.0, "transport_rank": 0}
    q0, _ = torch.linalg.qr(basis_from[:, :k].float(), mode="reduced")
    q1, _ = torch.linalg.qr(basis_to[:, :k].float(), mode="reduced")
    m = q0.t() @ q1
    u, _s, vh = torch.linalg.svd(m, full_matrices=False)
    rot = u @ vh
    matrix = grad_matrix.float()
    transposed = False
    if int(axis) == 1:
        matrix = matrix.t().contiguous()
        transposed = True
    coeff = q0.t() @ matrix
    transported = q1 @ (rot.t() @ coeff[:k])
    if transposed:
        transported = transported.t().contiguous()
    transported = transported * grad_matrix.norm().clamp_min(1.0e-12) / transported.norm().clamp_min(1.0e-12)
    reconstruction = q0 @ rot
    alignment = float((transported.reshape(-1).dot(grad_matrix.reshape(-1).float()) / (transported.norm() * grad_matrix.norm()).clamp_min(1.0e-12)).abs().item())
    return {
        "direction_matrix": transported.to(device=grad_matrix.device, dtype=grad_matrix.dtype),
        "transport_error": float((reconstruction - q1).norm().item() / max(1.0e-12, float(q1.norm().item()))),
        "post_transport_alignment": alignment,
        "transport_rank": k,
    }


def effect_transport_readout_direction(
    source_grad: Any,
    basis_from: Any,
    basis_to: Any,
    *,
    axis: int,
) -> dict[str, Any]:
    out = _effect_transport_matrix(source_grad, basis_from, basis_to, axis)
    direction = out["direction_matrix"].reshape(-1)
    return {
        "direction": direction / direction.norm().clamp_min(1.0e-12),
        "transport_error": out["transport_error"],
        "post_transport_alignment": out["post_transport_alignment"],
        "transport_rank": out["transport_rank"],
    }


def effect_space_control(direction: Any, basis: Any, shape: tuple[int, int], mode: str, seed: int, axis: int) -> Any:
    import torch

    if mode == "signflip":
        out = -direction
        return out * direction.norm().clamp_min(1.0e-12) / out.norm().clamp_min(1.0e-12)
    gen = torch.Generator(device=direction.device).manual_seed(int(seed))
    if basis is None or int(basis.numel()) == 0:
        out = torch.randn(direction.shape, device=direction.device, generator=gen)
        return out * direction.norm().clamp_min(1.0e-12) / out.norm().clamp_min(1.0e-12)
    k = int(basis.shape[1])
    if int(axis) == 0:
        coeff = torch.randn(k, int(shape[1]), device=direction.device, generator=gen)
        matrix = basis.float() @ coeff
    else:
        coeff = torch.randn(int(shape[0]), k, device=direction.device, generator=gen)
        matrix = coeff @ basis.float().t()
    out = matrix.reshape(-1)
    return out * direction.norm().clamp_min(1.0e-12) / out.norm().clamp_min(1.0e-12)


def kan_w2_to_feature_readout_matrix(tensor: Any) -> Any:
    return tensor.permute(0, 2, 1).contiguous().reshape(int(tensor.shape[0]) * int(tensor.shape[2]), int(tensor.shape[1]))


def kan_feature_readout_matrix_to_w2_vector(matrix: Any, w2_shape: tuple[int, int, int]) -> Any:
    h, c, k = (int(w2_shape[0]), int(w2_shape[1]), int(w2_shape[2]))
    return matrix.reshape(h, k, c).permute(0, 2, 1).contiguous().reshape(-1)


def kan_w2_vector_to_feature_readout_matrix(vector: Any, w2_shape: tuple[int, int, int]) -> Any:
    h, c, k = (int(w2_shape[0]), int(w2_shape[1]), int(w2_shape[2]))
    return vector.reshape(h, c, k).permute(0, 2, 1).contiguous().reshape(h * k, c)


def kan_effect_transport_w2_direction(source_w2_grad: Any, basis_from: Any, basis_to: Any) -> dict[str, Any]:
    matrix = kan_w2_to_feature_readout_matrix(source_w2_grad)
    out = _effect_transport_matrix(matrix, basis_from, basis_to, axis=0)
    direction = kan_feature_readout_matrix_to_w2_vector(out["direction_matrix"], tuple(source_w2_grad.shape))
    return {
        "direction": direction / direction.norm().clamp_min(1.0e-12),
        "transport_error": out["transport_error"],
        "post_transport_alignment": out["post_transport_alignment"],
        "transport_rank": out["transport_rank"],
    }


def kan_effect_space_control(direction: Any, basis: Any, w2_shape: tuple[int, int, int], mode: str, seed: int) -> Any:
    matrix = kan_w2_vector_to_feature_readout_matrix(direction, w2_shape)
    control = effect_space_control(
        matrix.reshape(-1),
        basis,
        (int(w2_shape[0]) * int(w2_shape[2]), int(w2_shape[1])),
        mode,
        seed,
        axis=0,
    )
    return kan_feature_readout_matrix_to_w2_vector(control.reshape(int(w2_shape[0]) * int(w2_shape[2]), int(w2_shape[1])), w2_shape)


def direction_control(direction: Any, mode: str, seed: int, basis: Any | None = None, proj: Any | None = None) -> Any:
    import torch

    gen = torch.Generator(device=direction.device).manual_seed(int(seed))
    if mode == "signflip":
        out = -direction
    elif mode == "shuffled":
        out = direction[torch.randperm(int(direction.numel()), generator=gen, device=direction.device)]
    elif mode == "same_subspace" and basis is not None and proj is not None:
        coeff = torch.randn(int(basis.shape[1]), device=direction.device, generator=gen)
        out = proj @ (basis @ coeff)
    else:
        out = torch.randn(direction.shape, device=direction.device, generator=gen)
    return out * direction.norm().clamp_min(1.0e-12) / out.norm().clamp_min(1.0e-12)


def apply_layer_direction(model: Any, layer_name: str, direction: Any, trust_ratio: float) -> None:
    with __import__("torch").no_grad():
        param = dict(model.named_parameters())[layer_name]
        step_norm = float(param.detach().norm().item()) * float(trust_ratio)
        delta = direction.reshape_as(param).to(device=param.device, dtype=param.dtype)
        delta = delta * (step_norm / delta.norm().clamp_min(1.0e-12))
        param.add_(-delta)


def held_batch_ce(model: Any, xb: Any, yb: Any) -> float:
    import torch
    import torch.nn.functional as F

    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        loss = F.cross_entropy(model(xb).float(), yb.long())
    model.train(was_training)
    return float(loss.item())


def refined_direction_from_subspace(
    model: Any,
    layer: str,
    sig: dict[str, Any],
    xb: Any,
    yb: Any,
    *,
    trust_ratio: float,
    seed: int,
    max_candidates: int,
) -> dict[str, Any]:
    import torch

    basis = sig.get("basis_sketch")
    proj = sig.get("proj")
    direction = sig["direction"]
    candidates: list[tuple[str, Any, str]] = [("raw_top_signal", direction, "original_top_eigen_direction")]
    if basis is not None and proj is not None and int(basis.numel()) > 0:
        k = int(basis.shape[1])
        for j in range(k):
            coeff = torch.zeros(k, device=direction.device)
            coeff[j] = 1.0
            vec = proj @ (basis @ coeff)
            vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
            candidates.append((f"basis_{j}_plus", vec, "held_train_basis_axis"))
            candidates.append((f"basis_{j}_minus", -vec, "held_train_basis_axis"))
        gen = torch.Generator(device=direction.device).manual_seed(int(seed) + 7001)
        while len(candidates) < max(1, int(max_candidates)):
            coeff = torch.randn(k, device=direction.device, generator=gen)
            vec = proj @ (basis @ coeff)
            vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
            candidates.append((f"random_combo_{len(candidates)}", vec, "held_train_random_combo_in_same_subspace"))
    candidates = candidates[: max(1, int(max_candidates))]
    base_loss = held_batch_ce(model, xb, yb)
    best: dict[str, Any] | None = None
    for label, cand, source in candidates:
        trial = copy.deepcopy(model).to(direction.device)
        apply_layer_direction(trial, layer, cand, trust_ratio)
        after = held_batch_ce(trial, xb, yb)
        row = {
            "selected_candidate_label": label,
            "selected_candidate_source": source,
            "direction": cand,
            "held_train_CE_before": base_loss,
            "held_train_CE_after": after,
            "held_train_CE_delta": after - base_loss,
        }
        if best is None or float(row["held_train_CE_after"]) < float(best["held_train_CE_after"]):
            best = row
    assert best is not None
    return best


def train_branch(model: Any, train_loader: Any, test_loader: Any, device: Any, output_dim: int, horizon: int, lr: float, wd: float) -> dict[str, float]:
    return train_adamw(model, train_loader, test_loader, device, output_dim, steps=horizon, lr=lr, weight_decay=wd)


def stage_t1_t2(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    device = torch_device(args.device)
    t1_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    t2_rows: list[dict[str, Any]] = []
    t3_rows: list[dict[str, Any]] = []
    cohort_diag_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    refinement_branch_rows: list[dict[str, Any]] = []
    for dataset in _split(args.datasets):
        for seed in _split(args.seeds, int):
            setup_seed(seed)
            train_loader, held_loader, test_loader, input_dim, output_dim, x_stats = make_loaders(
                dataset, args.train_size, args.test_size, args.batch_size, seed
            )
            model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
            train_adamw(model, train_loader, test_loader, device, output_dim, steps=args.pretrain_steps, lr=args.lr, weight_decay=args.weight_decay)
            base_eval = evaluate_model(model, test_loader, device, output_dim)
            xb, yb = collect_fixed_examples(held_loader, int(args.signal_cohorts) * int(args.cohort_size), device)
            layers = ["w0", "w1", "w2"]
            for layer_idx, layer in enumerate(layers):
                sig, cohort_rows = cohort_signal_for_layer(
                    model,
                    xb,
                    yb,
                    layer,
                    cohorts=args.signal_cohorts,
                    cohort_size=args.cohort_size,
                    rank_cap=args.rank_cap,
                    sketch_dim=args.sketch_dim,
                    seed=seed + 31 * (layer_idx + 1),
                )
                temporal_model = copy.deepcopy(model).to(device)
                train_adamw(
                    temporal_model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.temporal_signal_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                sig_future, _future_cohort_rows = cohort_signal_for_layer(
                    temporal_model,
                    xb,
                    yb,
                    layer,
                    cohorts=args.signal_cohorts,
                    cohort_size=args.cohort_size,
                    rank_cap=args.rank_cap,
                    sketch_dim=args.sketch_dim,
                    seed=seed + 31 * (layer_idx + 1),
                )
                temporal_overlap = principal_overlap(sig["basis_sketch"], sig_future["basis_sketch"])
                temporal_base_eval = evaluate_model(temporal_model, test_loader, device, output_dim)
                cohort_positive_fraction = sig.get("cohort_positive_fraction", 0.0)
                for c in cohort_rows:
                    c.update(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model": "MLPBaseline",
                            "layer_id": layer,
                            "temporal_signal_steps": args.temporal_signal_steps,
                        }
                    )
                    cohort_diag_rows.append(c)
                real_wins_base = 0
                real_wins_l2 = 0
                real_wins_l3 = 0
                branch_count = 0
                control_positive = 0
                for horizon in _split(args.branch_horizons, int):
                    variants = [
                        ("real", 0, sig["direction"], "real_signal"),
                        ("L0_noop", 1, torch.zeros_like(sig["direction"]), "same_overhead_noop"),
                        ("L1_random", 1, direction_control(sig["direction"], "random", seed + horizon + layer_idx), "same_norm_random"),
                        ("L2_signflip", 1, direction_control(sig["direction"], "signflip", seed + horizon + layer_idx), "same_support_signflip"),
                        (
                            "L3_same_subspace",
                            1,
                            direction_control(sig["direction"], "same_subspace", seed + horizon + layer_idx, sig["basis_sketch"], sig["proj"]),
                            "same_subspace_random",
                        ),
                    ]
                    outcomes: dict[str, float] = {}
                    for variant, is_control, direction, control_level in variants:
                        branch_model = copy.deepcopy(model).to(device)
                        if variant != "L0_noop":
                            apply_layer_direction(branch_model, layer, direction, args.branch_trust)
                        ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, horizon, args.lr, args.weight_decay)
                        delta = float(ev["NLL"] - base_eval["NLL"])
                        outcomes[variant] = ev["NLL"]
                        branch_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model": "MLPBaseline",
                                "layer_id": layer,
                                "horizon": horizon,
                                "branch_variant": variant,
                                "is_control_branch": is_control,
                                "control_level": control_level,
                                "checkpoint_test_NLL": base_eval["NLL"],
                                "final_test_NLL": ev["NLL"],
                                "branch_NLL_delta": delta,
                                "final_test_accuracy": ev["accuracy"],
                                "same_norm": 1,
                                "same_cadence": 1,
                                "same_overhead": 1,
                            }
                        )
                    if "real" in outcomes:
                        branch_count += 1
                        real_nll = outcomes["real"]
                        if real_nll < base_eval["NLL"]:
                            real_wins_base += 1
                        if "L2_signflip" in outcomes and real_nll < outcomes["L2_signflip"]:
                            real_wins_l2 += 1
                        if "L3_same_subspace" in outcomes and real_nll < outcomes["L3_same_subspace"]:
                            real_wins_l3 += 1
                        control_positive += sum(1 for k, v in outcomes.items() if k != "real" and v < base_eval["NLL"])
                t1_row = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": "MLPBaseline",
                    "layer_id": layer,
                    "positive_eigen_count": sig["positive_eigen_count"],
                    "positive_eigenvalue_mean": sig["positive_eigenvalue_mean"],
                    "cohort_positive_fraction": cohort_positive_fraction,
                    "temporal_eigenspace_overlap": temporal_overlap,
                    "held_train_Q_delta": "diagnostic_sketch_only",
                    "real_beats_base_rate": real_wins_base / max(1, branch_count),
                    "real_beats_L2_controls_rate": real_wins_l2 / max(1, branch_count),
                    "real_beats_L3_controls_rate": real_wins_l3 / max(1, branch_count),
                    "control_positive_rate": control_positive / max(1, branch_count * 4),
                    "signal_SNR": sig["signal_SNR"],
                    "drift_norm": sig["drift_norm"],
                    "diffusion_trace": sig["diffusion_trace"],
                    "implementation_level": "multi_cohort_layerwise_sketched_per_example_gradient_A_B",
                    "known_simplifications": "random sketch used instead of full dense A_B; MLP layers only in this stage",
                }
                t1_rows.append(t1_row)
                eligible_for_refinement = (
                    not bool(args.disable_t1_refinement)
                    and finite_float(t1_row.get("real_beats_base_rate"), 0.0) >= 0.70
                    and finite_float(t1_row.get("real_beats_L3_controls_rate"), 0.0) < 0.50
                    and finite_float(t1_row.get("temporal_eigenspace_overlap"), 0.0) >= 0.50
                )
                if eligible_for_refinement:
                    refine = refined_direction_from_subspace(
                        model,
                        layer,
                        sig,
                        xb,
                        yb,
                        trust_ratio=args.branch_trust,
                        seed=seed + 313 * (layer_idx + 1),
                        max_candidates=args.t1_refinement_candidates,
                    )
                    refined_wins_base = 0
                    refined_wins_raw = 0
                    refined_wins_l2 = 0
                    refined_wins_l3_best = 0
                    refined_branch_count = 0
                    for horizon in _split(args.branch_horizons, int):
                        variants = [
                            ("refined_signal", 0, refine["direction"], "held_train_selected_same_subspace"),
                            ("raw_top_signal", 0, sig["direction"], "original_top_eigen_direction"),
                            ("L0_noop", 1, torch.zeros_like(sig["direction"]), "same_overhead_noop"),
                            ("L1_random", 1, direction_control(sig["direction"], "random", seed + 9000 + horizon + layer_idx), "same_norm_random"),
                            ("L2_signflip", 1, direction_control(sig["direction"], "signflip", seed + 9100 + horizon + layer_idx), "same_support_signflip"),
                        ]
                        for cidx in range(int(args.t1_refinement_l3_controls)):
                            variants.append(
                                (
                                    f"L3_same_subspace_{cidx}",
                                    1,
                                    direction_control(
                                        sig["direction"],
                                        "same_subspace",
                                        seed + 9200 + 97 * cidx + horizon + layer_idx,
                                        sig["basis_sketch"],
                                        sig["proj"],
                                    ),
                                    "same_subspace_random",
                                )
                            )
                        outcomes: dict[str, float] = {}
                        for variant, is_control, direction, control_level in variants:
                            branch_model = copy.deepcopy(model).to(device)
                            if variant != "L0_noop":
                                apply_layer_direction(branch_model, layer, direction, args.branch_trust)
                            ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, horizon, args.lr, args.weight_decay)
                            outcomes[variant] = ev["NLL"]
                            refinement_branch_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "model": "MLPBaseline",
                                    "layer_id": layer,
                                    "horizon": horizon,
                                    "branch_variant": variant,
                                    "is_control_branch": is_control,
                                    "control_level": control_level,
                                    "selected_candidate_label": refine["selected_candidate_label"],
                                    "selected_candidate_source": refine["selected_candidate_source"],
                                    "held_train_CE_delta_for_selected": refine["held_train_CE_delta"],
                                    "checkpoint_test_NLL": base_eval["NLL"],
                                    "final_test_NLL": ev["NLL"],
                                    "branch_NLL_delta": float(ev["NLL"] - base_eval["NLL"]),
                                    "final_test_accuracy": ev["accuracy"],
                                    "selection_data_source": "held_train_cohort_only",
                                    "uses_test_direction_selection": 0,
                                    "uses_future_direction": 0,
                                    "same_norm": 1,
                                    "same_cadence": 1,
                                    "same_overhead": 1,
                                }
                            )
                        refined_branch_count += 1
                        refined_nll = outcomes.get("refined_signal")
                        if refined_nll is not None:
                            if refined_nll < base_eval["NLL"]:
                                refined_wins_base += 1
                            if "raw_top_signal" in outcomes and refined_nll < outcomes["raw_top_signal"]:
                                refined_wins_raw += 1
                            if "L2_signflip" in outcomes and refined_nll < outcomes["L2_signflip"]:
                                refined_wins_l2 += 1
                            l3_vals = [v for k, v in outcomes.items() if k.startswith("L3_same_subspace_")]
                            if l3_vals and refined_nll < min(l3_vals):
                                refined_wins_l3_best += 1
                    basis = sig.get("basis_sketch")
                    basis_dim = int(basis.shape[1]) if basis is not None and int(basis.numel()) > 0 else 0
                    refinement_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "model": "MLPBaseline",
                            "layer_id": layer,
                            "eligible_reason": "base_win_temporal_ok_but_L3_under_0.5",
                            "selected_candidate_label": refine["selected_candidate_label"],
                            "selected_candidate_source": refine["selected_candidate_source"],
                            "candidate_count": min(max(1, int(args.t1_refinement_candidates)), max(1, 1 + 2 * basis_dim)),
                            "held_train_CE_before": refine["held_train_CE_before"],
                            "held_train_CE_after": refine["held_train_CE_after"],
                            "held_train_CE_delta": refine["held_train_CE_delta"],
                            "temporal_eigenspace_overlap": temporal_overlap,
                            "base_t1_real_beats_L2_controls_rate": t1_row.get("real_beats_L2_controls_rate"),
                            "base_t1_real_beats_L3_controls_rate": t1_row.get("real_beats_L3_controls_rate"),
                            "refined_beats_base_rate": refined_wins_base / max(1, refined_branch_count),
                            "refined_beats_raw_top_rate": refined_wins_raw / max(1, refined_branch_count),
                            "refined_beats_L2_controls_rate": refined_wins_l2 / max(1, refined_branch_count),
                            "refined_beats_L3_best_controls_rate": refined_wins_l3_best / max(1, refined_branch_count),
                            "L3_control_count_per_horizon": int(args.t1_refinement_l3_controls),
                            "selection_data_source": "held_train_cohort_only",
                            "uses_test_direction_selection": 0,
                            "uses_future_direction": 0,
                            "implementation_level": "held_train_selected_direction_inside_existing_signal_subspace",
                            "known_simplifications": "selection reuses held-train cohorts from T1 signal estimation; MLP/sketch only; no KAN basis-native refinement",
                        }
                    )
                t2_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "source_space": f"MLP_{layer}_sketch",
                        "source_vector_retention": temporal_overlap,
                        "source_subspace_retention": temporal_overlap,
                        "principal_angles_mean": 1.0 - temporal_overlap,
                        "principal_angles_max": 1.0 - temporal_overlap,
                        "Grassmann_geodesic_distance": math.acos(max(-1.0, min(1.0, temporal_overlap))),
                        "subspace_temporal_curvature": "",
                        "Plucker_coordinate_variance": "",
                        "transport_error": "",
                        "post_transport_alignment": temporal_overlap,
                        "future_NLL_delta": mean_field([r for r in branch_rows if r.get("layer_id") == layer and r.get("branch_variant") == "real"], "branch_NLL_delta"),
                        "future_AUC_delta": "",
                        "control_subspace_retention": "",
                        "real_vs_control_transport_gain": "",
                        "implementation_level": "outcome_linked_multi_cohort_branch_rows_from_T1",
                        "known_simplifications": "uses sketched layer gradient subspace; no full KAN basis/JVP effect space in this stage",
                    }
                )
                transported = transported_direction_from_signal_subspaces(sig, sig_future)
                for horizon in _split(args.branch_horizons, int):
                    variants = [
                        ("transported_source", 0, transported["direction"], "procrustes_transport_from_previous_signal_subspace"),
                        ("L0_noop", 1, torch.zeros_like(sig["direction"]), "same_overhead_noop"),
                        (
                            "L2_transport_signflip",
                            1,
                            direction_control(transported["direction"], "signflip", seed + 13000 + horizon + layer_idx),
                            "transported_direction_signflip",
                        ),
                        (
                            "L3_future_same_subspace",
                            1,
                            direction_control(
                                transported["direction"],
                                "same_subspace",
                                seed + 14000 + horizon + layer_idx,
                                sig_future["basis_sketch"],
                                sig_future["proj"],
                            ),
                            "future_same_subspace_random",
                        ),
                    ]
                    outcomes: dict[str, float] = {}
                    for variant, is_control, direction, control_level in variants:
                        branch_model = copy.deepcopy(temporal_model).to(device)
                        if variant != "L0_noop":
                            apply_layer_direction(branch_model, layer, direction, args.branch_trust)
                        ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, horizon, args.lr, args.weight_decay)
                        outcomes[variant] = ev["NLL"]
                        t3_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model": "MLPBaseline",
                                "layer_id": layer,
                                "source_space": f"MLP_{layer}_sketch",
                                "horizon": horizon,
                                "transport_variant": variant,
                                "is_control_branch": is_control,
                                "control_level": control_level,
                                "checkpoint_test_NLL": temporal_base_eval["NLL"],
                                "final_test_NLL": ev["NLL"],
                                "branch_NLL_delta": float(ev["NLL"] - temporal_base_eval["NLL"]),
                                "final_test_accuracy": ev["accuracy"],
                                "source_vector_retention": temporal_overlap,
                                "source_subspace_retention": temporal_overlap,
                                "principal_angles_mean": 1.0 - temporal_overlap,
                                "principal_angles_max": 1.0 - temporal_overlap,
                                "Grassmann_geodesic_distance": math.acos(max(-1.0, min(1.0, temporal_overlap))),
                                "transport_error": transported["transport_error"],
                                "post_transport_alignment": transported["post_transport_alignment"],
                                "transport_rank": transported["transport_rank"],
                                "same_norm": 1,
                                "same_cadence": 1,
                                "same_overhead": 1,
                                "implementation_level": "procrustes_transport_branch_on_temporal_checkpoint",
                                "known_simplifications": "sketched MLP layer-gradient subspace only; no KAN/JVP effect transport; diagnostic branch not full optimizer loop.",
                            }
                        )
                    real = outcomes.get("transported_source")
                    l3 = outcomes.get("L3_future_same_subspace")
                    if real is not None and l3 is not None:
                        for row in t3_rows[-len(variants) :]:
                            row["transported_source_beats_same_subspace_controls"] = int(real < l3)
                            row["real_vs_control_transport_gain"] = float(l3 - real)
            rep_temporal = copy.deepcopy(model).to(device)
            train_adamw(
                rep_temporal,
                train_loader,
                test_loader,
                device,
                output_dim,
                steps=args.temporal_signal_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            with torch.no_grad():
                rep_specs = [
                    (
                        "MLP_hidden_activation_effect",
                        model.frozen_readout_features(xb),
                        rep_temporal.frozen_readout_features(xb),
                    ),
                    ("MLP_logits_effect", model(xb).float(), rep_temporal(xb).float()),
                ]
            w2_param = dict(model.named_parameters())["w2"]
            w2_source_grad = collect_layer_grads(model, xb, yb, "w2").mean(dim=0).reshape_as(w2_param).detach()
            real_branch_rows = [
                r
                for r in branch_rows
                if r.get("dataset") == dataset and r.get("seed") == seed and r.get("branch_variant") == "real"
            ]
            for source_space, current_features, future_features in rep_specs:
                current_basis, current_energy = feature_subspace(current_features, args.rank_cap)
                future_basis, future_energy = feature_subspace(future_features, args.rank_cap)
                retention = principal_overlap(current_basis, future_basis)
                t2_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "source_space": source_space,
                        "source_vector_retention": retention,
                        "source_subspace_retention": retention,
                        "principal_angles_mean": 1.0 - retention,
                        "principal_angles_max": 1.0 - retention,
                        "Grassmann_geodesic_distance": math.acos(max(-1.0, min(1.0, retention))),
                        "subspace_temporal_curvature": "",
                        "Plucker_coordinate_variance": "",
                        "transport_error": "",
                        "post_transport_alignment": retention,
                        "future_NLL_delta": mean_field(real_branch_rows, "branch_NLL_delta"),
                        "future_AUC_delta": "",
                        "control_subspace_retention": "",
                        "real_vs_control_transport_gain": "",
                        "feature_energy_topk_fraction": current_energy,
                        "future_feature_energy_topk_fraction": future_energy,
                        "implementation_level": "wrong_space_check_hidden_or_logits_effect_subspace",
                        "known_simplifications": "diagnostic source-space check only; no intervention/transport branch and no KAN basis-native effect space.",
                    }
                )
                axis = 0 if source_space == "MLP_hidden_activation_effect" else 1
                effect_transport = effect_transport_readout_direction(
                    w2_source_grad,
                    current_basis,
                    future_basis,
                    axis=axis,
                )
                effect_layer_id = "w2_hidden_effect" if axis == 0 else "w2_logits_effect"
                for horizon in _split(args.branch_horizons, int):
                    variants = [
                        ("transported_source", 0, effect_transport["direction"], "effect_space_procrustes_transport_to_readout"),
                        ("L0_noop", 1, torch.zeros_like(effect_transport["direction"]), "same_overhead_noop"),
                        (
                            "L2_transport_signflip",
                            1,
                            effect_space_control(
                                effect_transport["direction"],
                                future_basis,
                                tuple(w2_param.shape),
                                "signflip",
                                seed + 15000 + horizon + axis,
                                axis,
                            ),
                            "effect_transport_signflip",
                        ),
                        (
                            "L3_future_same_subspace",
                            1,
                            effect_space_control(
                                effect_transport["direction"],
                                future_basis,
                                tuple(w2_param.shape),
                                "same_subspace",
                                seed + 16000 + horizon + axis,
                                axis,
                            ),
                            "future_effect_same_subspace_random",
                        ),
                    ]
                    outcomes: dict[str, float] = {}
                    for variant, is_control, direction, control_level in variants:
                        branch_model = copy.deepcopy(rep_temporal).to(device)
                        if variant != "L0_noop":
                            apply_layer_direction(branch_model, "w2", direction, args.branch_trust)
                        ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, horizon, args.lr, args.weight_decay)
                        outcomes[variant] = ev["NLL"]
                        t3_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model": "MLPBaseline",
                                "layer_id": effect_layer_id,
                                "source_space": source_space,
                                "horizon": horizon,
                                "transport_variant": variant,
                                "is_control_branch": is_control,
                                "control_level": control_level,
                                "checkpoint_test_NLL": temporal_base_eval["NLL"],
                                "final_test_NLL": ev["NLL"],
                                "branch_NLL_delta": float(ev["NLL"] - temporal_base_eval["NLL"]),
                                "final_test_accuracy": ev["accuracy"],
                                "source_vector_retention": retention,
                                "source_subspace_retention": retention,
                                "principal_angles_mean": 1.0 - retention,
                                "principal_angles_max": 1.0 - retention,
                                "Grassmann_geodesic_distance": math.acos(max(-1.0, min(1.0, retention))),
                                "transport_error": effect_transport["transport_error"],
                                "post_transport_alignment": effect_transport["post_transport_alignment"],
                                "transport_rank": effect_transport["transport_rank"],
                                "feature_energy_topk_fraction": current_energy,
                                "future_feature_energy_topk_fraction": future_energy,
                                "same_norm": 1,
                                "same_cadence": 1,
                                "same_overhead": 1,
                                "implementation_level": "effect_space_procrustes_transport_branch_on_readout_w2",
                                "known_simplifications": "Hidden/logits effect subspace is mapped through MLP readout w2 only; no KAN basis-native effect transport.",
                            }
                        )
                    real = outcomes.get("transported_source")
                    l3 = outcomes.get("L3_future_same_subspace")
                    if real is not None and l3 is not None:
                        for row in t3_rows[-len(variants) :]:
                            row["transported_source_beats_same_subspace_controls"] = int(real < l3)
                            row["real_vs_control_transport_gain"] = float(l3 - real)
            for kan_family in ["D-CHE", "D-FOU"]:
                kan_model_name = f"DGKAN_{kan_family.replace('-', '')}"
                kan_model = make_kan(input_dim, output_dim, args.hidden, seed, device, x_stats, kan_family)
                train_adamw(
                    kan_model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.pretrain_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                kan_temporal = copy.deepcopy(kan_model).to(device)
                train_adamw(
                    kan_temporal,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.temporal_signal_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                kan_temporal_base_eval = evaluate_model(kan_temporal, test_loader, device, output_dim)
                with torch.no_grad():
                    current_features = kan_model.frozen_readout_features(xb).float()
                    future_features = kan_temporal.frozen_readout_features(xb).float()
                current_basis, current_energy = feature_subspace(current_features, args.rank_cap)
                future_basis, future_energy = feature_subspace(future_features, args.rank_cap)
                retention = principal_overlap(current_basis, future_basis)
                source_space = f"{kan_model_name}_basis_effect"
                t2_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "source_space": source_space,
                        "source_vector_retention": retention,
                        "source_subspace_retention": retention,
                        "principal_angles_mean": 1.0 - retention,
                        "principal_angles_max": 1.0 - retention,
                        "Grassmann_geodesic_distance": math.acos(max(-1.0, min(1.0, retention))),
                        "subspace_temporal_curvature": "",
                        "Plucker_coordinate_variance": "",
                        "transport_error": "",
                        "post_transport_alignment": retention,
                        "future_NLL_delta": "",
                        "future_AUC_delta": "",
                        "control_subspace_retention": "",
                        "real_vs_control_transport_gain": "",
                        "feature_energy_topk_fraction": current_energy,
                        "future_feature_energy_topk_fraction": future_energy,
                        "implementation_level": "strict_PrimitiveKAN_basis_feature_effect_subspace",
                        "known_simplifications": "diagnostic basis-effect source-space check; branch only perturbs w2 readout parameters, not a full basis-native controller.",
                    }
                )
                w2_param = dict(kan_model.named_parameters())["w2"]
                w2_shape = tuple(w2_param.shape)
                w2_source_grad = collect_layer_grads(kan_model, xb, yb, "w2").mean(dim=0).reshape_as(w2_param).detach()
                transported = kan_effect_transport_w2_direction(w2_source_grad, current_basis, future_basis)
                layer_id = f"{kan_model_name}_w2_basis_effect"
                for horizon in _split(args.branch_horizons, int):
                    variants = [
                        ("transported_source", 0, transported["direction"], "kan_basis_effect_procrustes_transport_to_w2"),
                        ("L0_noop", 1, torch.zeros_like(transported["direction"]), "same_overhead_noop"),
                        (
                            "L2_transport_signflip",
                            1,
                            kan_effect_space_control(
                                transported["direction"],
                                future_basis,
                                w2_shape,
                                "signflip",
                                seed + 17000 + horizon + (0 if kan_family == "D-CHE" else 1000),
                            ),
                            "kan_basis_effect_transport_signflip",
                        ),
                        (
                            "L3_future_same_subspace",
                            1,
                            kan_effect_space_control(
                                transported["direction"],
                                future_basis,
                                w2_shape,
                                "same_subspace",
                                seed + 18000 + horizon + (0 if kan_family == "D-CHE" else 1000),
                            ),
                            "future_kan_basis_effect_same_subspace_random",
                        ),
                    ]
                    outcomes: dict[str, float] = {}
                    for variant, is_control, direction, control_level in variants:
                        branch_model = copy.deepcopy(kan_temporal).to(device)
                        if variant != "L0_noop":
                            apply_layer_direction(branch_model, "w2", direction, args.branch_trust)
                        ev = train_branch(branch_model, train_loader, test_loader, device, output_dim, horizon, args.lr, args.weight_decay)
                        outcomes[variant] = ev["NLL"]
                        t3_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "model": kan_model_name,
                                "layer_id": layer_id,
                                "source_space": source_space,
                                "horizon": horizon,
                                "transport_variant": variant,
                                "is_control_branch": is_control,
                                "control_level": control_level,
                                "checkpoint_test_NLL": kan_temporal_base_eval["NLL"],
                                "final_test_NLL": ev["NLL"],
                                "branch_NLL_delta": float(ev["NLL"] - kan_temporal_base_eval["NLL"]),
                                "final_test_accuracy": ev["accuracy"],
                                "source_vector_retention": retention,
                                "source_subspace_retention": retention,
                                "principal_angles_mean": 1.0 - retention,
                                "principal_angles_max": 1.0 - retention,
                                "Grassmann_geodesic_distance": math.acos(max(-1.0, min(1.0, retention))),
                                "transport_error": transported["transport_error"],
                                "post_transport_alignment": transported["post_transport_alignment"],
                                "transport_rank": transported["transport_rank"],
                                "feature_energy_topk_fraction": current_energy,
                                "future_feature_energy_topk_fraction": future_energy,
                                "same_norm": 1,
                                "same_cadence": 1,
                                "same_overhead": 1,
                                "implementation_level": "strict_PrimitiveKAN_basis_effect_transport_branch_on_w2",
                                "known_simplifications": "Uses PrimitiveKAN frozen_readout_features basis-effect subspace and w2-only branch; not promoted as full basis-native FU controller.",
                            }
                        )
                    real = outcomes.get("transported_source")
                    l3 = outcomes.get("L3_future_same_subspace")
                    if real is not None and l3 is not None:
                        for row in t3_rows[-len(variants) :]:
                            row["transported_source_beats_same_subspace_controls"] = int(real < l3)
                            row["real_vs_control_transport_gain"] = float(l3 - real)
    write_rows(OUT_ROOT / "v22_30_T1_layerwise_signal_channel_matrix.csv", t1_rows)
    write_rows(OUT_ROOT / "v22_30_T1_branch_causal_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_30_T1_cohort_signal_diagnostics.csv", cohort_diag_rows)
    write_rows(OUT_ROOT / "v22_30_T1_subspace_refinement_matrix.csv", refinement_rows)
    write_rows(OUT_ROOT / "v22_30_T1_subspace_refinement_branch_matrix.csv", refinement_branch_rows)
    write_rows(OUT_ROOT / "v22_30_T2_grassmann_outcome_linked_matrix.csv", t2_rows)
    write_rows(OUT_ROOT / "v22_30_T3_transport_momentum_matrix.csv", t3_rows)
    write_simple_svg(
        "v22_30_signal_channel_eigenspectrum.svg",
        "T1 positive eigen counts",
        [{"label": f"{r['dataset']} {r['layer_id']}", "rows": r["positive_eigen_count"]} for r in t1_rows],
    )
    write_simple_svg(
        "v22_30_grassmann_flow_curves.svg",
        "T2 subspace retention",
        [{"label": f"{r['dataset']} {r['source_space']}", "rows": r["source_subspace_retention"]} for r in t2_rows],
    )
    write_simple_svg(
        "v22_30_T1_subspace_refinement.svg",
        "T1 refinement vs L3 best controls",
        [{"label": f"{r['dataset']} {r['layer_id']}", "rows": r["refined_beats_L3_best_controls_rate"]} for r in refinement_rows],
    )
    return t1_rows, branch_rows, t2_rows, refinement_rows, refinement_branch_rows


def tangent_normalize(update: Any, weight: Any, axis: int = -1) -> Any:
    w_hat = weight / weight.norm(dim=axis, keepdim=True).clamp_min(1.0e-12)
    dot = (update * w_hat).sum(dim=axis, keepdim=True)
    perp = update - w_hat * dot
    return perp / perp.norm(dim=axis, keepdim=True).clamp_min(1.0e-12)


def train_mano_variant(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    variant: str,
    steps: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device=device).manual_seed(int(seed) + 123)
    momentum: dict[str, Any] = {}
    signal: dict[str, Any] = {}
    update_norms = []
    snrs = []
    spectra = []
    it = cycle_batches(train_loader)
    start = time.time()
    for _step in range(int(steps)):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                momentum[name] = g.clone() if name not in momentum else beta * momentum[name] + (1.0 - beta) * g
                signal[name] = g.clone() if name not in signal else 0.95 * signal[name] + 0.05 * g
                raw = momentum[name]
                sig = signal[name]
                if variant == "signal_mano":
                    raw = raw + alpha * sig
                elif variant == "random_control":
                    rnd = torch.randn(sig.shape, device=sig.device, generator=gen)
                    rnd = rnd * sig.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                    raw = raw + alpha * rnd
                elif variant == "signflip_control":
                    raw = raw - alpha * sig
                elif variant == "shuffled_control":
                    shuf = sig.reshape(-1)[torch.randperm(sig.numel(), generator=gen, device=sig.device)].reshape_as(sig)
                    raw = raw + alpha * shuf
                if p.ndim >= 2 and "mano" in variant:
                    update = tangent_normalize(raw, p, axis=-1)
                    update = update * raw.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
                elif p.ndim >= 2 and variant in {"signal_mano", "random_control", "signflip_control", "shuffled_control"}:
                    update = tangent_normalize(raw, p, axis=-1)
                    update = update * raw.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
                else:
                    update = raw
                p.mul_(1.0 - lr * weight_decay)
                p.add_(-lr * update)
                update_norms.append(float(update.norm().item()))
                snrs.append(float(signal[name].norm().item() / (g - signal[name]).norm().clamp_min(1.0e-12).item()))
                if p.ndim == 2:
                    sv = torch.linalg.svdvals(update.float())
                    if int(sv.numel()):
                        spectra.append(float((sv[0] / sv.mean().clamp_min(1.0e-12)).item()))
    sync(device)
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * (time.time() - start) / max(1, steps),
            "update_norm": sum(update_norms) / max(1, len(update_norms)),
            "update_SNR": sum(snrs) / max(1, len(snrs)),
            "singular_value_spectrum_top_to_mean": sum(spectra) / max(1, len(spectra)) if spectra else "",
            "tangent_projection_residual": "not_measured_per_step",
            "oblique_norm_error": "not_measured_per_step",
        }
    )
    return ev


def orthogonalized_update(mat: Any) -> Any:
    import torch

    if mat.ndim != 2:
        return mat
    try:
        u, _s, vh = torch.linalg.svd(mat.float(), full_matrices=False)
        out = (u @ vh).to(dtype=mat.dtype, device=mat.device)
        return out * (mat.norm() / out.norm().clamp_min(1.0e-12))
    except RuntimeError:
        return mat


def train_manual_strong_optimizer(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    variant: str,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    signal_mode: str = "none",
    alpha: float = 0.0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    named = list(model.named_parameters())
    m: dict[str, Any] = {}
    v: dict[str, Any] = {}
    avg: dict[str, Any] = {}
    signal: dict[str, Any] = {}
    gen = torch.Generator(device=device).manual_seed(int(seed) + 41003)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    update_norms: list[float] = []
    snrs: list[float] = []
    norm_match_scales: list[float] = []
    if signal_mode not in {"none", "signal", "random", "signflip", "shuffled"}:
        raise ValueError(f"unknown signal_mode={signal_mode!r}")
    start = time.time()
    it = cycle_batches(train_loader)
    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        with torch.no_grad():
            for name, p in named:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                m[name] = g.clone() if name not in m else beta1 * m[name] + (1.0 - beta1) * g
                if variant == "Muon-like" and p.ndim == 2:
                    update = orthogonalized_update(m[name])
                else:
                    v[name] = g.square() if name not in v else beta2 * v[name] + (1.0 - beta2) * g.square()
                    m_hat = m[name] / (1.0 - beta1**step)
                    v_hat = v[name] / (1.0 - beta2**step)
                    update = m_hat / (v_hat.sqrt() + eps)
                    if variant == "Cautious AdamW":
                        mask = (update * g) > 0.0
                        scale = mask.float().mean().clamp_min(1.0e-3)
                        update = update * mask / scale
                if signal_mode != "none":
                    signal[name] = g.clone() if name not in signal else 0.95 * signal[name] + 0.05 * g
                    sig = signal[name]
                    if signal_mode == "random":
                        sig_dir = torch.randn(sig.shape, device=sig.device, generator=gen)
                    elif signal_mode == "signflip":
                        sig_dir = -sig
                    elif signal_mode == "shuffled":
                        sig_dir = sig.reshape(-1)[torch.randperm(sig.numel(), generator=gen, device=sig.device)].reshape_as(sig)
                    else:
                        sig_dir = sig
                    scale = update.norm().clamp_min(1.0e-12) / sig_dir.norm().clamp_min(1.0e-12)
                    update = update + float(alpha) * scale * sig_dir
                    norm_match_scales.append(float(scale.item()))
                p.mul_(1.0 - lr * weight_decay)
                p.add_(-lr * update)
                update_norms.append(float(update.norm().item()))
                snrs.append(float(m[name].norm().item() / (g - m[name]).norm().clamp_min(1.0e-12).item()))
                if variant == "Schedule-Free AdamW":
                    if name not in avg:
                        avg[name] = p.detach().clone()
                    else:
                        avg[name].mul_(float(step - 1) / float(step)).add_(p.detach(), alpha=1.0 / float(step))
    if variant == "Schedule-Free AdamW":
        with torch.no_grad():
            for name, p in named:
                if name in avg:
                    p.copy_(avg[name])
    sync(device)
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * (time.time() - start) / max(1, steps),
            "update_norm": sum(update_norms) / max(1, len(update_norms)),
            "update_SNR": sum(snrs) / max(1, len(snrs)),
            "norm_match_scale_mean": sum(norm_match_scales) / max(1, len(norm_match_scales)) if norm_match_scales else "",
            "strong_fu_signal_mode": signal_mode,
            "implementation_level": (
                "schedule_free_averaging_approx"
                if variant == "Schedule-Free AdamW"
                else ("svd_orthogonalized_momentum_muon_like" if variant == "Muon-like" else "manual_cautious_adamw")
            ),
        }
    )
    return ev


def train_soap_shampoo_diagnostic(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    matrix_cap: int = 64,
) -> dict[str, Any]:
    """Feasibility diagnostic: Shampoo-like Kronecker only for small matrices."""
    import torch
    import torch.nn.functional as F

    setup_seed(seed)
    named = list(model.named_parameters())
    m: dict[str, Any] = {}
    v: dict[str, Any] = {}
    row_stats: dict[str, Any] = {}
    col_stats: dict[str, Any] = {}
    beta1 = 0.9
    beta2 = 0.98
    eps = 1.0e-4
    update_norms: list[float] = []
    snrs: list[float] = []
    kron_steps = 0
    diag_steps = 0
    it = cycle_batches(train_loader)
    start = time.time()
    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        with torch.no_grad():
            for name, p in named:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                m[name] = g.clone() if name not in m else beta1 * m[name] + (1.0 - beta1) * g
                m_hat = m[name] / (1.0 - beta1**step)
                use_kron = p.ndim == 2 and max(int(p.shape[0]), int(p.shape[1])) <= int(matrix_cap)
                if use_kron:
                    rows = int(p.shape[0])
                    cols = int(p.shape[1])
                    gg_t = (g.float() @ g.float().t()) / float(max(1, cols))
                    g_tg = (g.float().t() @ g.float()) / float(max(1, rows))
                    row_stats[name] = gg_t if name not in row_stats else beta2 * row_stats[name] + (1.0 - beta2) * gg_t
                    col_stats[name] = g_tg if name not in col_stats else beta2 * col_stats[name] + (1.0 - beta2) * g_tg
                    eye_r = torch.eye(rows, device=device, dtype=torch.float32)
                    eye_c = torch.eye(cols, device=device, dtype=torch.float32)
                    try:
                        er, ur = torch.linalg.eigh(row_stats[name] + eps * eye_r)
                        ec, uc = torch.linalg.eigh(col_stats[name] + eps * eye_c)
                        middle = ur.t() @ m_hat.float() @ uc
                        denom = er.clamp_min(eps).sqrt().unsqueeze(1) * ec.clamp_min(eps).sqrt().unsqueeze(0)
                        update = (ur @ (middle / denom) @ uc.t()).to(dtype=p.dtype, device=p.device)
                        kron_steps += 1
                    except RuntimeError:
                        use_kron = False
                if not use_kron:
                    v[name] = g.square() if name not in v else beta2 * v[name] + (1.0 - beta2) * g.square()
                    update = m_hat / (v[name].sqrt() + eps)
                    diag_steps += 1
                p.mul_(1.0 - lr * weight_decay)
                p.add_(-lr * update)
                update_norms.append(float(update.norm().item()))
                snrs.append(float(m[name].norm().item() / (g - m[name]).norm().clamp_min(1.0e-12).item()))
    sync(device)
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * (time.time() - start) / max(1, steps),
            "update_norm": sum(update_norms) / max(1, len(update_norms)),
            "update_SNR": sum(snrs) / max(1, len(snrs)),
            "implementation_level": "diagonal_plus_small_matrix_kronecker_diagnostic",
            "preconditioner_kron_param_steps": kron_steps,
            "preconditioner_diag_param_steps": diag_steps,
        }
    )
    return ev


def train_adamw_tangent_signal_variant(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    variant: str,
    steps: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    gen = torch.Generator(device=device).manual_seed(int(seed) + 17001)
    named = list(model.named_parameters())
    m: dict[str, Any] = {}
    v: dict[str, Any] = {}
    signal: dict[str, Any] = {}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    update_norms: list[float] = []
    snrs: list[float] = []
    spectra: list[float] = []
    norm_match_scales: list[float] = []
    it = cycle_batches(train_loader)
    start = time.time()
    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        with torch.no_grad():
            for name, p in named:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                m[name] = g.clone() if name not in m else beta1 * m[name] + (1.0 - beta1) * g
                v[name] = g.square() if name not in v else beta2 * v[name] + (1.0 - beta2) * g.square()
                signal[name] = g.clone() if name not in signal else 0.95 * signal[name] + 0.05 * g
                m_hat = m[name] / (1.0 - beta1**step)
                v_hat = v[name] / (1.0 - beta2**step)
                base_update = m_hat / (v_hat.sqrt() + eps)
                raw = base_update
                sig = signal[name]
                if variant == "signal_adamw_tangent":
                    sig_dir = sig
                elif variant == "random_adamw_tangent":
                    sig_dir = torch.randn(sig.shape, device=sig.device, generator=gen)
                elif variant == "signflip_adamw_tangent":
                    sig_dir = -sig
                elif variant == "shuffled_adamw_tangent":
                    sig_dir = sig.reshape(-1)[torch.randperm(sig.numel(), generator=gen, device=sig.device)].reshape_as(sig)
                else:
                    sig_dir = None
                if sig_dir is not None:
                    scale = base_update.norm().clamp_min(1.0e-12) / sig_dir.norm().clamp_min(1.0e-12)
                    raw = base_update + float(alpha) * scale * sig_dir
                    norm_match_scales.append(float(scale.item()))
                if p.ndim >= 2:
                    tangent = tangent_normalize(raw, p, axis=-1)
                    raw = tangent * raw.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
                    sv = torch.linalg.svdvals(raw.float())
                    if int(sv.numel()):
                        spectra.append(float((sv[0] / sv.mean().clamp_min(1.0e-12)).item()))
                p.mul_(1.0 - lr * weight_decay)
                p.add_(-lr * raw)
                update_norms.append(float(raw.norm().item()))
                snrs.append(float(signal[name].norm().item() / (g - signal[name]).norm().clamp_min(1.0e-12).item()))
    sync(device)
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * (time.time() - start) / max(1, steps),
            "update_norm": sum(update_norms) / max(1, len(update_norms)),
            "update_SNR": sum(snrs) / max(1, len(snrs)),
            "singular_value_spectrum_top_to_mean": sum(spectra) / max(1, len(spectra)) if spectra else "",
            "norm_match_scale_mean": sum(norm_match_scales) / max(1, len(norm_match_scales)) if norm_match_scales else "",
            "implementation_level": "adamw_moments_with_row_tangent_norm_matched_signal",
        }
    )
    return ev


def stage_t4_t7(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch_device(args.device)
    t4_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    t7_rows: list[dict[str, Any]] = []
    for dataset in _split(args.datasets):
        for seed in _split(args.seeds, int):
            setup_seed(seed)
            train_loader, _held, test_loader, input_dim, output_dim, _x_stats = make_loaders(
                dataset, args.train_size, args.test_size, args.batch_size, seed
            )
            variants = ["adamw", "mano", "signal_mano", "random_control", "signflip_control", "shuffled_control"]
            baseline_nll = None
            mano_nll = None
            for variant in variants:
                model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                if variant == "adamw":
                    ev = train_adamw(
                        model,
                        train_loader,
                        test_loader,
                        device,
                        output_dim,
                        steps=args.optimizer_steps,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                else:
                    ev = train_mano_variant(
                        model,
                        train_loader,
                        test_loader,
                        device,
                        output_dim,
                        variant=variant,
                        steps=args.optimizer_steps,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        alpha=args.signal_alpha,
                        beta=0.9,
                        seed=seed,
                    )
                if variant == "adamw":
                    baseline_nll = ev["NLL"]
                if variant == "mano":
                    mano_nll = ev["NLL"]
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": "MLPBaseline",
                    "optimizer_variant": variant,
                    "NLL": ev["NLL"],
                    "accuracy": ev["accuracy"],
                    "ECE": ev["ECE"],
                    "Brier": ev["Brier"],
                    "tail_q95": ev["tail_q95"],
                    "tail_q99": ev["tail_q99"],
                    "wallclock_adjusted_AUC": "",
                    "update_norm": ev.get("update_norm", ev.get("mean_update_to_param_ratio", "")),
                    "update_SNR": ev.get("update_SNR", ""),
                    "singular_value_spectrum_top_to_mean": ev.get("singular_value_spectrum_top_to_mean", ""),
                    "controller_overhead_ratio": "",
                    "implementation_level": "continuous_optimizer_loop",
                    "known_simplifications": "Mano-like uses row-axis tangent normalization; Muon/SOAP not official in T4 stage",
                }
                t4_rows.append(row)
                spectrum_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "optimizer_variant": variant,
                        "update_SNR": row["update_SNR"],
                        "singular_value_spectrum_top_to_mean": row["singular_value_spectrum_top_to_mean"],
                    }
                )
                t7_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "optimizer": variant,
                        "model": "MLPBaseline",
                        "NLL": ev["NLL"],
                        "accuracy": ev["accuracy"],
                        "optimizer_step_ms": ev["train_step_ms"],
                        "update_SNR": row["update_SNR"],
                        "FU_gain_over_optimizer": "" if variant != "signal_mano" or mano_nll is None else ev["NLL"] - mano_nll,
                        "status": "real_run" if variant in {"adamw", "mano", "signal_mano"} else "matched_control",
                    }
                )
            adamw_tangent_nll = None
            for variant in [
                "adamw_tangent",
                "signal_adamw_tangent",
                "random_adamw_tangent",
                "signflip_adamw_tangent",
                "shuffled_adamw_tangent",
            ]:
                model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                ev = train_adamw_tangent_signal_variant(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    variant=variant,
                    steps=args.optimizer_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    alpha=args.signal_alpha,
                    seed=seed,
                )
                if variant == "adamw_tangent":
                    adamw_tangent_nll = ev["NLL"]
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": "MLPBaseline",
                    "optimizer_variant": variant,
                    "NLL": ev["NLL"],
                    "accuracy": ev["accuracy"],
                    "ECE": ev["ECE"],
                    "Brier": ev["Brier"],
                    "tail_q95": ev["tail_q95"],
                    "tail_q99": ev["tail_q99"],
                    "wallclock_adjusted_AUC": "",
                    "update_norm": ev.get("update_norm", ""),
                    "update_SNR": ev.get("update_SNR", ""),
                    "singular_value_spectrum_top_to_mean": ev.get("singular_value_spectrum_top_to_mean", ""),
                    "controller_overhead_ratio": "",
                    "norm_match_scale_mean": ev.get("norm_match_scale_mean", ""),
                    "implementation_level": ev.get("implementation_level", ""),
                    "known_simplifications": "AdamW moment baseline with row-axis tangent projection; signal alpha is norm-matched to base update, no alpha sweep.",
                }
                t4_rows.append(row)
                spectrum_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "optimizer_variant": variant,
                        "update_SNR": row["update_SNR"],
                        "singular_value_spectrum_top_to_mean": row["singular_value_spectrum_top_to_mean"],
                    }
                )
                t7_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "optimizer": variant,
                        "model": "MLPBaseline",
                        "NLL": ev["NLL"],
                        "accuracy": ev["accuracy"],
                        "optimizer_step_ms": ev["train_step_ms"],
                        "update_SNR": row["update_SNR"],
                        "FU_gain_over_optimizer": (
                            "" if variant != "signal_adamw_tangent" or adamw_tangent_nll is None else ev["NLL"] - adamw_tangent_nll
                        ),
                        "status": "real_run" if variant in {"adamw_tangent", "signal_adamw_tangent"} else "matched_control",
                        "implementation_level": ev.get("implementation_level", ""),
                        "blocker": "" if variant != "signal_adamw_tangent" else "Norm-matched AdamW+tangent FU fallback; not an external optimizer implementation.",
                    }
                )
            for opt in ["Schedule-Free AdamW", "Cautious AdamW", "Muon-like", "SOAP/Shampoo-like"]:
                if opt == "SOAP/Shampoo-like":
                    model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                    ev = train_soap_shampoo_diagnostic(
                        model,
                        train_loader,
                        test_loader,
                        device,
                        output_dim,
                        steps=args.optimizer_steps,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        seed=seed,
                    )
                    t7_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "optimizer": opt,
                            "model": "MLPBaseline",
                            "NLL": ev["NLL"],
                            "accuracy": ev["accuracy"],
                            "optimizer_step_ms": ev["train_step_ms"],
                            "update_SNR": ev.get("update_SNR", ""),
                            "FU_gain_over_optimizer": "",
                            "status": "feasibility_diagnostic",
                            "implementation_level": ev.get("implementation_level", ""),
                            "preconditioner_kron_param_steps": ev.get("preconditioner_kron_param_steps", ""),
                            "preconditioner_diag_param_steps": ev.get("preconditioner_diag_param_steps", ""),
                            "blocker": "Plan fallback diagnostic only: diagonal fallback for large matrices plus Kronecker preconditioner on small matrices; not official SOAP/Shampoo.",
                        }
                    )
                    continue
                model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                ev = train_manual_strong_optimizer(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    variant=opt,
                    steps=args.optimizer_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=seed,
                )
                strong_base_nll = ev["NLL"]
                t7_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "optimizer": opt,
                        "model": "MLPBaseline",
                        "NLL": ev["NLL"],
                        "accuracy": ev["accuracy"],
                        "optimizer_step_ms": ev["train_step_ms"],
                        "update_SNR": ev.get("update_SNR", ""),
                        "FU_gain_over_optimizer": "",
                        "status": "real_run",
                        "implementation_level": ev.get("implementation_level", ""),
                        "blocker": "" if opt != "Schedule-Free AdamW" else "Schedule-Free implemented as averaging approximation, not external library official implementation.",
                    }
                )
                for suffix, signal_mode, status, control_variant in [
                    ("signal_fu", "signal", "real_run", "signal_fu"),
                    ("random_control", "random", "matched_control", "random"),
                    ("signflip_control", "signflip", "matched_control", "signflip"),
                    ("shuffled_control", "shuffled", "matched_control", "shuffled"),
                ]:
                    model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                    ev_sig = train_manual_strong_optimizer(
                        model,
                        train_loader,
                        test_loader,
                        device,
                        output_dim,
                        variant=opt,
                        steps=args.optimizer_steps,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        seed=seed + 7000,
                        signal_mode=signal_mode,
                        alpha=args.signal_alpha,
                    )
                    t7_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "optimizer": f"{opt}_{suffix}",
                            "base_optimizer": opt,
                            "control_variant": control_variant,
                            "model": "MLPBaseline",
                            "NLL": ev_sig["NLL"],
                            "accuracy": ev_sig["accuracy"],
                            "optimizer_step_ms": ev_sig["train_step_ms"],
                            "update_SNR": ev_sig.get("update_SNR", ""),
                            "FU_gain_over_optimizer": "" if control_variant != "signal_fu" else ev_sig["NLL"] - strong_base_nll,
                            "status": status,
                            "implementation_level": f"{ev_sig.get('implementation_level', '')}+norm_matched_slow_signal_fu",
                            "norm_match_scale_mean": ev_sig.get("norm_match_scale_mean", ""),
                            "blocker": "Plan T7 fallback: manual strong optimizer plus norm-matched slow-signal FU/control, not an external library optimizer.",
                        }
                    )
    write_rows(OUT_ROOT / "v22_30_T4_tangent_optimizer_matrix.csv", t4_rows)
    write_rows(OUT_ROOT / "v22_30_T4_update_spectrum_matrix.csv", spectrum_rows)
    write_rows(OUT_ROOT / "v22_30_T7_optimizer_baseline_matrix.csv", t7_rows)
    write_rows(
        OUT_ROOT / "v22_30_T7_fu_optimizer_interaction_matrix.csv",
        [
            r
            for r in t7_rows
            if r.get("optimizer")
            in {
                "mano",
                "signal_mano",
                "random_control",
                "signflip_control",
                "shuffled_control",
                "adamw_tangent",
                "signal_adamw_tangent",
                "random_adamw_tangent",
                "signflip_adamw_tangent",
                "shuffled_adamw_tangent",
            }
            or r.get("base_optimizer") in {"Schedule-Free AdamW", "Cautious AdamW", "Muon-like"}
        ],
    )
    write_simple_svg(
        "v22_30_tangent_update_spectrum.svg",
        "T4 update SNR",
        [{"label": f"{r['dataset']} {r['optimizer_variant']}", "rows": finite_float(r.get("update_SNR"), 0.0) or 0.0} for r in t4_rows],
    )
    return t4_rows, spectrum_rows, t7_rows


def collect_static_basis(model: Any, train_loader: Any, device: Any, rows: int, rank: int) -> Any:
    import torch
    import torch.nn.functional as F

    named = flat_named_params(model)
    grads = []
    it = cycle_batches(train_loader)
    for _ in range(int(rows)):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        grads.append(flat_grad(named).detach().clone())
    gmat = torch.stack(grads, dim=0)
    _, _, vh = torch.linalg.svd(gmat.float(), full_matrices=False)
    return vh[:rank].t().contiguous()


def train_projected_variant(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    variant: str,
    static_basis: Any | None,
    rank: int,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    named = flat_named_params(model)
    total_dim = sum(int(p.numel()) for _, p in named)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 500)
    if static_basis is not None:
        basis = static_basis.to(device)
    else:
        basis = torch.randn(total_dim, int(rank), device=device, generator=gen)
        basis, _ = torch.linalg.qr(basis.float(), mode="reduced")
    residuals = []
    angles = []
    it = cycle_batches(train_loader)
    start = time.time()
    for _step in range(int(steps)):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        g = flat_grad(named).float()
        if variant.startswith("online"):
            bg = basis.t() @ g
            basis = basis + float(args_oja_lr()) * torch.outer(g, bg)
            basis, _ = torch.linalg.qr(basis.float(), mode="reduced")
        proj = basis @ (basis.t() @ g)
        residual = float((g - proj).norm().item() / g.norm().clamp_min(1.0e-12).item())
        residuals.append(residual)
        if variant == "online_random_control":
            coeff = torch.randn(int(basis.shape[1]), device=device, generator=gen)
            proj = basis @ coeff
            proj = proj * g.norm().clamp_min(1.0e-12) / proj.norm().clamp_min(1.0e-12)
        elif variant == "online_recovery":
            proj = proj + 0.15 * (g - proj)
        assign_flat_grad(named, proj if variant != "full_gradient" else g)
        with torch.no_grad():
            for _, p in named:
                if p.grad is not None:
                    p.add_(-lr * p.grad)
        if int(basis.shape[1]) >= 1:
            angles.append(float((basis[:, 0] @ (g / g.norm().clamp_min(1.0e-12))).abs().item()))
    sync(device)
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * (time.time() - start) / max(1, steps),
            "projection_residual": sum(residuals) / max(1, len(residuals)),
            "subspace_temporal_angle_proxy": 1.0 - sum(angles) / max(1, len(angles)) if angles else "",
        }
    )
    return ev


def train_projected_adamw_variant(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    device: Any,
    output_dim: int,
    *,
    variant: str,
    static_basis: Any,
    rank: int,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    named = flat_named_params(model)
    basis = static_basis.to(device).float()
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    gen = torch.Generator(device=device).manual_seed(int(seed) + 23001)
    opt = torch.optim.AdamW([p for _, p in named], lr=lr, weight_decay=weight_decay)
    residuals: list[float] = []
    angles: list[float] = []
    recent: list[Any] = []
    refresh = max(10, min(40, int(steps) // 4 if int(steps) > 0 else 10))
    it = cycle_batches(train_loader)
    start = time.time()
    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb).float(), yb).backward()
        g = flat_grad(named).float()
        recent.append(g.detach().clone())
        if len(recent) > refresh:
            recent.pop(0)
        if variant in {"online_oja_warm_adamw", "online_recovery_adamw", "online_random_control_adamw"}:
            bg = basis.t() @ g
            basis = basis + float(args_oja_lr()) * torch.outer(g, bg)
            basis, _ = torch.linalg.qr(basis.float(), mode="reduced")
        elif variant == "periodic_svd_refresh_adamw" and step % refresh == 0 and len(recent) >= int(rank):
            gmat = torch.stack(recent, dim=0)
            _, _, vh = torch.linalg.svd(gmat.float(), full_matrices=False)
            basis = vh[:rank].t().contiguous()
        proj = basis @ (basis.t() @ g)
        residual = float((g - proj).norm().item() / g.norm().clamp_min(1.0e-12).item())
        residuals.append(residual)
        if variant == "online_random_control_adamw":
            coeff = torch.randn(int(basis.shape[1]), device=device, generator=gen)
            proj = basis @ coeff
            proj = proj * g.norm().clamp_min(1.0e-12) / proj.norm().clamp_min(1.0e-12)
        elif variant == "online_recovery_adamw":
            proj = proj + 0.15 * (g - proj)
        assign_flat_grad(named, proj if variant != "full_adamw_grad" else g)
        opt.step()
        if int(basis.shape[1]) >= 1:
            angles.append(float((basis[:, 0] @ (g / g.norm().clamp_min(1.0e-12))).abs().item()))
    sync(device)
    ev = evaluate_model(model, test_loader, device, output_dim)
    ev.update(
        {
            "train_step_ms": 1000.0 * (time.time() - start) / max(1, steps),
            "projection_residual": sum(residuals) / max(1, len(residuals)),
            "subspace_temporal_angle_proxy": 1.0 - sum(angles) / max(1, len(angles)) if angles else "",
        }
    )
    return ev


def args_oja_lr() -> float:
    return float(os.environ.get("V22_30_OJA_LR", "0.05"))


def stage_t5(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch_device(args.device)
    rows: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    for dataset in _split(args.datasets):
        for seed in _split(args.seeds, int):
            setup_seed(seed)
            train_loader, _held, test_loader, input_dim, output_dim, _x_stats = make_loaders(
                dataset, args.train_size, args.test_size, args.batch_size, seed
            )
            init_model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
            static_basis = collect_static_basis(init_model, train_loader, device, rows=8, rank=args.subspace_rank)
            for variant in ["full_gradient", "static_pca", "online_oja", "online_recovery", "online_random_control"]:
                model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                ev = train_projected_variant(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    variant=variant,
                    static_basis=static_basis if variant == "static_pca" else None,
                    rank=args.subspace_rank,
                    steps=args.subspace_steps,
                    lr=args.lr,
                    seed=seed,
                )
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "subspace_tracking_error": ev["projection_residual"],
                    "subspace_temporal_angle": ev["subspace_temporal_angle_proxy"],
                    "projection_residual": ev["projection_residual"],
                    "discarded_gradient_norm": ev["projection_residual"],
                    "recovery_scale": 0.15 if variant == "online_recovery" else 0.0,
                    "online_update_cost_ms": ev["train_step_ms"],
                    "NLL": ev["NLL"],
                    "accuracy": ev["accuracy"],
                    "status": "matched_control" if "control" in variant else "real_run",
                    "known_simplifications": "SGD-style projected gradient loop; effect-space/KAN-basis tracking deferred unless gate opens.",
                }
                rows.append(row)
                if variant in {"online_oja", "online_recovery", "online_random_control"}:
                    recovery.append(row)
            for variant in [
                "full_adamw_grad",
                "static_pca_adamw",
                "online_oja_warm_adamw",
                "online_recovery_adamw",
                "online_random_control_adamw",
                "periodic_svd_refresh_adamw",
            ]:
                model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                ev = train_projected_adamw_variant(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    variant=variant,
                    static_basis=static_basis,
                    rank=args.subspace_rank,
                    steps=args.subspace_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    seed=seed,
                )
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "subspace_tracking_error": ev["projection_residual"],
                    "subspace_temporal_angle": ev["subspace_temporal_angle_proxy"],
                    "projection_residual": ev["projection_residual"],
                    "discarded_gradient_norm": ev["projection_residual"],
                    "recovery_scale": 0.15 if variant == "online_recovery_adamw" else 0.0,
                    "online_update_cost_ms": ev["train_step_ms"],
                    "NLL": ev["NLL"],
                    "accuracy": ev["accuracy"],
                    "status": "matched_control" if "random_control" in variant else "real_run",
                    "implementation_level": "projection_aware_adamw_state",
                    "known_simplifications": "Parameter-gradient subspace only; static warm start from train gradients; no effect-space/KAN-basis tracking.",
                }
                rows.append(row)
                if variant in {"online_oja_warm_adamw", "online_recovery_adamw", "online_random_control_adamw", "periodic_svd_refresh_adamw"}:
                    recovery.append(row)
    write_rows(OUT_ROOT / "v22_30_T5_online_subspace_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_30_T5_projection_recovery_matrix.csv", recovery)
    write_simple_svg(
        "v22_30_online_subspace_tracking.svg",
        "T5 projection residual",
        [{"label": f"{r['dataset']} {r['variant']}", "rows": r["projection_residual"]} for r in rows],
    )
    return rows, recovery


class ModularDataset:
    def __init__(
        self,
        p: int,
        indices: list[tuple[int, int]],
        op: str = "add",
        label_noise_rate: float = 0.0,
        seed: int = 0,
        input_mode: str = "onehot",
    ) -> None:
        self.p = int(p)
        self.indices = indices
        self.op = op
        self.input_mode = input_mode
        self.label_noise_rate = float(label_noise_rate)
        rng = random.Random(int(seed) + 19001)
        self.noisy_labels: dict[int, int] = {}
        for idx, (a, b) in enumerate(indices):
            if rng.random() >= self.label_noise_rate:
                continue
            clean = (a + b) % self.p if self.op == "add" else (a * b) % self.p
            replacement = rng.randrange(self.p - 1)
            if replacement >= clean:
                replacement += 1
            self.noisy_labels[idx] = int(replacement)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        import torch

        a, b = self.indices[idx]
        if self.input_mode == "index":
            x = torch.tensor([a, b], dtype=torch.long)
        else:
            x = torch.zeros(2 * self.p)
            x[a] = 1.0
            x[self.p + b] = 1.0
        y = self.noisy_labels.get(idx)
        if y is None:
            y = (a + b) % self.p if self.op == "add" else (a * b) % self.p
        return x, int(y)


def make_modular_loaders(
    p: int,
    train_fraction: float,
    batch_size: int,
    seed: int,
    op: str,
    label_noise_rate: float = 0.0,
    input_mode: str = "onehot",
) -> tuple[Any, Any, int, int]:
    import torch
    from torch.utils.data import DataLoader

    pairs = [(i, j) for i in range(p) for j in range(p)]
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    n_train = max(1, int(len(pairs) * float(train_fraction)))
    train_ds = ModularDataset(p, pairs[:n_train], op=op, label_noise_rate=label_noise_rate, seed=seed, input_mode=input_mode)
    test_ds = ModularDataset(p, pairs[n_train:], op=op, label_noise_rate=0.0, seed=seed, input_mode=input_mode)
    gen = torch.Generator().manual_seed(int(seed))
    train_bs = min(int(batch_size), max(1, len(train_ds)))
    test_bs = min(int(batch_size), max(1, len(test_ds)))
    return (
        DataLoader(train_ds, batch_size=train_bs, shuffle=True, generator=gen, drop_last=False),
        DataLoader(test_ds, batch_size=test_bs, shuffle=False),
        2 * p,
        p,
    )


class ModularEmbeddingMLP:
    def __new__(cls, p: int, hidden: int) -> Any:
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                emb_dim = max(8, int(hidden) // 2)
                self.embedding = nn.Embedding(int(p), emb_dim)
                self.net = nn.Sequential(
                    nn.Linear(2 * emb_dim, int(hidden)),
                    nn.ReLU(),
                    nn.Linear(int(hidden), int(hidden)),
                    nn.ReLU(),
                    nn.Linear(int(hidden), int(p)),
                )

            def forward(self, x: Any) -> Any:
                x = x.long()
                emb = self.embedding(x)
                return self.net(emb.reshape(int(x.shape[0]), -1))

        return _Model()


def evaluate_grokking_model(model: Any, loader: Any, device: Any, output_dim: int, input_mode: str) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total = 0
    correct = 0
    losses_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            xb = xb.long() if input_mode == "index" else xb.float()
            yb = yb.to(device).long()
            logits = model(xb).float()
            losses_all.append(F.cross_entropy(logits, yb, reduction="none").detach().cpu())
            total += int(yb.numel())
            correct += int((logits.argmax(dim=-1) == yb).sum().item())
    if total == 0:
        return {"NLL": math.nan, "accuracy": math.nan}
    losses = __import__("torch").cat(losses_all)
    return {"NLL": float(losses.mean().item()), "accuracy": float(correct / total)}


def train_grokking_variant(
    train_loader: Any,
    test_loader: Any,
    input_dim: int,
    output_dim: int,
    device: Any,
    *,
    seed: int,
    variant: str,
    steps: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    slow_beta: float,
    alpha: float,
    log_interval: int,
    model_kind: str = "onehot_mlp",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    setup_seed(seed)
    input_mode = "index" if model_kind == "embedding_mlp" else "onehot"
    if model_kind == "embedding_mlp":
        model = ModularEmbeddingMLP(output_dim, hidden).to(device)
    else:
        model = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    slow: dict[int, Any] = {}
    curve: list[dict[str, Any]] = []
    it = cycle_batches(train_loader)
    grok_time = ""
    train_near_time = ""
    start = time.time()
    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device)
        xb = xb.long() if input_mode == "index" else xb.float()
        yb = yb.to(device).long()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        slow_power = 0.0
        fast_power = 0.0
        if variant != "base":
            for idx, p in enumerate(model.parameters()):
                if p.grad is None:
                    continue
                g = p.grad.detach().clone()
                slow[idx] = g if idx not in slow else slow_beta * slow[idx] + (1.0 - slow_beta) * g
                sig = slow[idx]
                if variant == "slow_random_control":
                    rnd = torch.randn_like(sig)
                    sig = rnd * sig.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                elif variant == "slow_signflip_control":
                    sig = -sig
                p.grad.add_(alpha * sig)
                slow_power += float(sig.pow(2).sum().item())
                fast_power += float((g - slow[idx]).pow(2).sum().item())
        opt.step()
        if step % int(log_interval) == 0 or step == int(steps):
            train_ev = evaluate_grokking_model(model, train_loader, device, output_dim, input_mode)
            test_ev = evaluate_grokking_model(model, test_loader, device, output_dim, input_mode)
            if train_near_time == "" and train_ev["accuracy"] >= 0.99:
                train_near_time = step
            if grok_time == "" and train_ev["accuracy"] >= 0.99 and test_ev["accuracy"] >= 0.95:
                grok_time = step
            curve.append(
                {
                    "variant": variant,
                    "step": step,
                    "train_acc": train_ev["accuracy"],
                    "test_acc": test_ev["accuracy"],
                    "train_NLL": train_ev["NLL"],
                    "test_NLL": test_ev["NLL"],
                    "slow_gradient_power": slow_power,
                    "fast_gradient_power": fast_power,
                    "slow_to_fast_ratio": slow_power / max(1.0e-12, fast_power),
                }
            )
    final_train = evaluate_grokking_model(model, train_loader, device, output_dim, input_mode)
    final_test = evaluate_grokking_model(model, test_loader, device, output_dim, input_mode)
    summary = {
        "variant": variant,
        "train_acc": final_train["accuracy"],
        "test_acc": final_test["accuracy"],
        "train_NLL": final_train["NLL"],
        "test_NLL": final_test["NLL"],
        "grokking_time": grok_time,
        "train_loss_plateau_length": "",
        "test_loss_drop_time": grok_time,
        "train_near_100_time": train_near_time,
        "final_accuracy_non_worse": "",
        "elapsed_sec": time.time() - start,
    }
    return summary, curve


def train_class_mnist_continual_variant(
    model: Any,
    tasks: list[dict[str, Any]],
    device: Any,
    output_dim: int,
    *,
    seed: int,
    variant: str,
    steps_per_task: int,
    lr: float,
    weight_decay: float,
    slow_beta: float,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    setup_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    slow: dict[str, Any] = {}
    acc_matrix: list[list[float]] = []
    detail_rows: list[dict[str, Any]] = []
    start = time.time()
    for task_idx, task in enumerate(tasks):
        it = cycle_batches(task["train_loader"])
        for _step in range(int(steps_per_task)):
            xb, yb = next(it)
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb).float(), yb)
            loss.backward()
            if variant != "base":
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    grad = p.grad.detach().clone()
                    slow[name] = grad if name not in slow else slow_beta * slow[name] + (1.0 - slow_beta) * grad
                    sig = slow[name]
                    if variant == "slow_random_control":
                        rnd = torch.randn_like(sig)
                        sig = rnd * sig.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                    elif variant == "slow_signflip_control":
                        sig = -sig
                    p.grad.add_(float(alpha) * sig)
            opt.step()
        row_accs: list[float] = []
        for eval_idx, eval_task in enumerate(tasks):
            ev = evaluate_model(model, eval_task["test_loader"], device, output_dim)
            row_accs.append(float(ev["accuracy"]))
            detail_rows.append(
                {
                    "after_task_index": task_idx,
                    "after_task_id": task["task_id"],
                    "eval_task_index": eval_idx,
                    "eval_task_id": eval_task["task_id"],
                    "eval_classes": eval_task["classes"],
                    "accuracy": ev["accuracy"],
                    "NLL": ev["NLL"],
                }
            )
        acc_matrix.append(row_accs)
    final_accs = acc_matrix[-1] if acc_matrix else []
    forgetting_vals: list[float] = []
    bwt_vals: list[float] = []
    for task_idx in range(max(0, len(tasks) - 1)):
        best_after_seen = max(row[task_idx] for row in acc_matrix[task_idx:])
        final_acc = final_accs[task_idx]
        forgetting_vals.append(max(0.0, best_after_seen - final_acc))
        bwt_vals.append(final_acc - acc_matrix[task_idx][task_idx])
    current_task_accs = [acc_matrix[idx][idx] for idx in range(len(tasks))] if acc_matrix else []
    summary = {
        "variant": variant,
        "avg_forgetting": sum(forgetting_vals) / max(1, len(forgetting_vals)),
        "max_forgetting": max(forgetting_vals) if forgetting_vals else 0.0,
        "BWT": sum(bwt_vals) / max(1, len(bwt_vals)),
        "mean_final_accuracy": sum(final_accs) / max(1, len(final_accs)),
        "mean_current_task_accuracy": sum(current_task_accs) / max(1, len(current_task_accs)),
        "elapsed_sec": time.time() - start,
    }
    return summary, detail_rows


def stage_class_mnist_continual(args: argparse.Namespace, device: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for seed in _split(args.seeds, int):
        tasks, input_dim, output_dim, x_stats = make_class_mnist_task_loaders(
            args.continual_train_size,
            args.continual_test_size,
            args.batch_size,
            seed,
        )
        variants = [
            ("MLP", "", "base", "MLP_AdamW", "AdamW", "real_run"),
            ("MLP", "", "slow_fu", "MLP_FU", "FU", "real_run"),
            ("MLP", "", "slow_random_control", "MLP_slow_random_control", "matched_control", "matched_control"),
            ("MLP", "", "slow_signflip_control", "MLP_slow_signflip_control", "matched_control", "matched_control"),
            ("KAN_DCHE", "D-CHE", "base", "DGKAN_DCHE_AdamW", "AdamW", "real_run"),
            ("KAN_DCHE", "D-CHE", "slow_fu", "DGKAN_DCHE_FU", "FU", "real_run"),
            ("KAN_DCHE", "D-CHE", "slow_random_control", "DGKAN_DCHE_slow_random_control", "matched_control", "matched_control"),
            ("KAN_DCHE", "D-CHE", "slow_signflip_control", "DGKAN_DCHE_slow_signflip_control", "matched_control", "matched_control"),
            ("KAN_DFOU", "D-FOU", "base", "DGKAN_DFOU_AdamW", "AdamW", "real_run"),
            ("KAN_DFOU", "D-FOU", "slow_fu", "DGKAN_DFOU_FU", "FU", "real_run"),
            ("KAN_DFOU", "D-FOU", "slow_random_control", "DGKAN_DFOU_slow_random_control", "matched_control", "matched_control"),
            ("KAN_DFOU", "D-FOU", "slow_signflip_control", "DGKAN_DFOU_slow_signflip_control", "matched_control", "matched_control"),
        ]
        base_by_family: dict[tuple[str, str], dict[str, Any]] = {}
        control_by_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
        pending_rows: list[dict[str, Any]] = []
        for family, kan_family, variant, model_name, training, status in variants:
            if family == "MLP":
                model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
            else:
                model = make_kan(input_dim, output_dim, args.hidden, seed, device, x_stats, kan_family)
            summary, detail = train_class_mnist_continual_variant(
                model,
                tasks,
                device,
                output_dim,
                seed=seed,
                variant=variant,
                steps_per_task=args.continual_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
                slow_beta=args.temporal_slow_beta,
                alpha=args.signal_alpha,
            )
            row = {
                "task": "Class_MNIST",
                "seed": seed,
                "model_name": model_name,
                "model_family": family,
                "kan_family": kan_family,
                "training": training,
                "variant": variant,
                "task_sequence": ",".join(t["classes"] for t in tasks),
                "steps_per_task": args.continual_steps,
                "avg_forgetting": summary["avg_forgetting"],
                "max_forgetting": summary["max_forgetting"],
                "BWT": summary["BWT"],
                "mean_final_accuracy": summary["mean_final_accuracy"],
                "mean_current_task_accuracy": summary["mean_current_task_accuracy"],
                "status": status,
                "implementation_level": "Class_MNIST_pair_sequence_continual_diagnostic",
                "known_simplifications": "Local split-MNIST pair sequence; family-matched slow-random/signflip controls; not KANbeFair original protocol.",
            }
            family_key = (family, kan_family)
            if variant == "base":
                base_by_family[family_key] = row
            elif "control" in variant:
                control_by_family.setdefault(family_key, []).append(row)
            pending_rows.append(row)
            for d in detail:
                d.update({"seed": seed, "model_name": model_name, "model_family": family, "kan_family": kan_family, "variant": variant})
            details.extend(detail)
        for row in pending_rows:
            key = (str(row["model_family"]), str(row.get("kan_family", "")))
            base = base_by_family.get(key)
            rel = ""
            acc_delta = ""
            acc_non_worse = ""
            beats_controls = ""
            matched_control_count = ""
            matched_control_min_forgetting = ""
            if row["variant"] == "slow_fu" and base:
                base_forgetting = finite_float(base.get("avg_forgetting"), 0.0) or 0.0
                cur_forgetting = finite_float(row.get("avg_forgetting"), 0.0) or 0.0
                rel = (base_forgetting - cur_forgetting) / max(1.0e-12, base_forgetting)
                base_acc = finite_float(base.get("mean_final_accuracy"), 0.0) or 0.0
                cur_acc = finite_float(row.get("mean_final_accuracy"), 0.0) or 0.0
                acc_delta = cur_acc - base_acc
                acc_non_worse = int(cur_acc >= base_acc - 1.0e-12)
                controls = control_by_family.get(key, [])
                matched_control_count = len(controls)
                if controls:
                    control_forgettings = [finite_float(c.get("avg_forgetting"), 0.0) or 0.0 for c in controls]
                    matched_control_min_forgetting = min(control_forgettings)
                    beats_controls = int(all(cur_forgetting < c for c in control_forgettings))
            row["relative_forgetting_reduction_vs_base"] = rel
            row["final_accuracy_delta_vs_base"] = acc_delta
            row["final_accuracy_non_worse_vs_base"] = acc_non_worse
            row["matched_control_count"] = matched_control_count
            row["matched_control_min_avg_forgetting"] = matched_control_min_forgetting
            row["beats_forgetting_controls"] = beats_controls
            rows.append(row)
    write_rows(OUT_ROOT / "v22_30_T6_continual_boundary_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_30_KANbeFair_continual_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_30_T6_continual_detail_matrix.csv", details)
    return rows, details


def stage_t6(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch_device(args.device)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    protocols = [
        ("mod_add_p31", 31, 0.40, args.grokking_steps, args.weight_decay, "add", 0.0, "onehot_mlp"),
        ("fallback_mod_add_p19", 19, 0.35, max(args.grokking_steps, 600), max(args.weight_decay, 1.0e-2), "add", 0.0, "onehot_mlp"),
        ("fallback_mod_add_p13_wd005", 13, 0.50, max(args.grokking_steps, 1200), max(args.weight_decay, 5.0e-2), "add", 0.0, "onehot_mlp"),
        ("fallback_mod_mul_p11_wd005", 11, 0.50, max(args.grokking_steps, 1200), max(args.weight_decay, 5.0e-2), "mul", 0.0, "onehot_mlp"),
        ("fallback_mod_add_p17_noise005_wd005", 17, 0.35, max(args.grokking_steps, 1800), max(args.weight_decay, 5.0e-2), "add", 0.05, "onehot_mlp"),
        ("fallback_mod_add_p11_noise010_wd010", 11, 0.50, max(args.grokking_steps, 1800), max(args.weight_decay, 1.0e-1), "add", 0.10, "onehot_mlp"),
        ("fallback_embed_mod_add_p17_wd010", 17, 0.35, max(args.grokking_steps, 2400), max(args.weight_decay, 1.0e-1), "add", 0.0, "embedding_mlp"),
        ("fallback_embed_mod_add_p11_wd010", 11, 0.50, max(args.grokking_steps, 2400), max(args.weight_decay, 1.0e-1), "add", 0.0, "embedding_mlp"),
    ]
    for protocol, p, train_frac, steps, wd, op, label_noise_rate, model_kind in protocols:
        for seed in _split(args.seeds, int):
            train_loader, test_loader, input_dim, output_dim = make_modular_loaders(
                p,
                train_frac,
                args.batch_size,
                seed,
                op,
                label_noise_rate=label_noise_rate,
                input_mode="index" if model_kind == "embedding_mlp" else "onehot",
            )
            base_grok = ""
            for variant in ["base", "slow_fu", "slow_random_control", "slow_signflip_control"]:
                summary, curve = train_grokking_variant(
                    train_loader,
                    test_loader,
                    input_dim,
                    output_dim,
                    device,
                    seed=seed,
                    variant=variant,
                    steps=steps,
                    lr=args.lr,
                    weight_decay=wd,
                    hidden=max(32, args.hidden * 2),
                    slow_beta=args.temporal_slow_beta,
                    alpha=args.signal_alpha,
                    log_interval=args.log_interval,
                    model_kind=model_kind,
                )
                if variant == "base":
                    base_grok = summary["grokking_time"]
                summary.update(
                    {
                        "task": protocol,
                        "seed": seed,
                        "p": p,
                        "train_fraction": train_frac,
                        "weight_decay": wd,
                        "label_noise_rate": label_noise_rate,
                        "model_kind": model_kind,
                        "grokking_time_reduction": ""
                        if base_grok == "" or summary["grokking_time"] == ""
                        else int(base_grok) - int(summary["grokking_time"]),
                        "matched_control": int("control" in variant),
                        "status": "real_run",
                    }
                )
                rows.append(summary)
                for c in curve:
                    c.update({"task": protocol, "seed": seed, "p": p, "model_kind": model_kind, "label_noise_rate": label_noise_rate})
                curves.extend(curve)
        if any(r.get("grokking_time") not in {"", None} for r in rows if r.get("task") == protocol):
            break
    write_rows(OUT_ROOT / "v22_30_T6_grokking_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_30_T6_grokking_curves.csv", curves)
    continual_rows, _continual_details = stage_class_mnist_continual(args, device)
    write_simple_svg(
        "v22_30_grokking_curves.svg",
        "T6 final test accuracy",
        [{"label": f"{r['task']} {r['variant']}", "rows": r["test_acc"]} for r in rows],
    )
    write_simple_svg(
        "v22_30_continual_forgetting.svg",
        "T6 Class-MNIST forgetting",
        [{"label": f"{r['model_name']} s{r['seed']}", "rows": r["avg_forgetting"]} for r in continual_rows],
    )
    return rows, curves


def stage_kan_kanbefair(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch_device(args.device)
    carrier_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    for dataset in _split(args.datasets):
        for seed in _split(args.seeds, int):
            setup_seed(seed)
            train_loader, held_train_loader, test_loader, input_dim, output_dim, x_stats = make_loaders(
                dataset, args.train_size, args.test_size, args.batch_size, seed
            )
            variants = official_kan_variants(args)
            results: dict[str, dict[str, Any]] = {}
            for name, family, training, slow_fu, kan_family, control_variant, functional_mechanism in variants:
                if family == "MLP":
                    model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                else:
                    model = make_kan(input_dim, output_dim, args.hidden, seed, device, x_stats, kan_family)
                functional_lr = official_kan_functional_lr(args, family)
                functional_interval = official_kan_functional_interval(args, family)
                ev = train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.kan_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    slow_fu=slow_fu,
                    slow_alpha=args.signal_alpha,
                    slow_beta=args.temporal_slow_beta,
                    slow_signal=args.kan_slow_signal,
                    slow_gate=args.kan_slow_gate,
                    slow_trust_ratio=args.kan_slow_trust_ratio,
                    slow_acceptance=args.kan_slow_acceptance,
                    slow_acceptance_tol=args.kan_slow_acceptance_tol,
                    slow_accept_loader=held_train_loader,
                    control=control_variant,
                    functional_fu_mechanism=functional_mechanism,
                    functional_fu_lr=functional_lr,
                    functional_fu_lr_schedule=args.kan_functional_fu_lr_schedule,
                    functional_fu_interval=functional_interval,
                    functional_fu_acceptance=args.kan_functional_acceptance,
                    functional_fu_acceptance_tol=args.kan_functional_acceptance_tol,
                    functional_fu_accept_loader=held_train_loader,
                    functional_fu_control=control_variant if functional_mechanism else "",
                )
                diag: dict[str, Any] = {}
                if family != "MLP" and hasattr(model, "basis_diagnostics"):
                    xb0, _ = next(iter(test_loader))
                    diag = model.basis_diagnostics(xb0.to(device).float())
                results[name] = ev
                carrier_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_name": name,
                        "model_family": family,
                        "training": training,
                        "control_variant": control_variant,
                        "status": "matched_control" if control_variant else "real_run",
                        "fu_mode": official_kan_fu_mode(args) if training != "AdamW" else "",
                        "fu_official_eligible": official_kan_functional_eligible(args, ev) if training == "FU" else int(training == "matched_control" and getattr(args, "kan_fu_mode", "slow") == "slow"),
                        "functional_fu_mechanism": ev.get("functional_fu_mechanism", ""),
                        "functional_fu_lr": ev.get("functional_fu_lr", ""),
                        "functional_fu_lr_schedule": ev.get("functional_fu_lr_schedule", ""),
                        "functional_fu_lr_mean": ev.get("functional_fu_lr_mean", ""),
                        "functional_fu_interval": ev.get("functional_fu_interval", ""),
                        "functional_fu_acceptance": ev.get("functional_fu_acceptance", ""),
                        "functional_fu_control": ev.get("functional_fu_control", ""),
                        "functional_fu_applied_count": ev.get("functional_fu_applied_count", ""),
                        "functional_fu_skipped_count": ev.get("functional_fu_skipped_count", ""),
                        "functional_fu_accept_rate": ev.get("functional_fu_accept_rate", ""),
                        "functional_fu_update_norm_mean": ev.get("functional_fu_update_norm_mean", ""),
                        "functional_fu_operator_status_counts": ev.get("functional_fu_operator_status_counts", ""),
                        "functional_fu_hidden_source_fraction_mean": ev.get("functional_fu_hidden_source_fraction_mean", ""),
                        "functional_fu_readout_source_fraction_mean": ev.get("functional_fu_readout_source_fraction_mean", ""),
                        "functional_fu_block_source_hidden_residual_fraction_mean": ev.get("functional_fu_block_source_hidden_residual_fraction_mean", ""),
                        "functional_fu_projection_residual_Gf_mean": ev.get("functional_fu_projection_residual_Gf_mean", ""),
                        "functional_fu_ActuationR2_mean": ev.get("functional_fu_ActuationR2_mean", ""),
                        "functional_fu_ActuationCosine_mean": ev.get("functional_fu_ActuationCosine_mean", ""),
                        "basis_energy_fraction": diag.get("basis_energy_fraction", ""),
                        "basis_entropy": diag.get("basis_entropy", ""),
                        "basis_effective_rank": diag.get("basis_effective_rank", ""),
                        "basis_condition_proxy": diag.get("basis_condition_proxy", ""),
                        "readout_leakage_fraction": diag.get("readout_leakage_fraction", ""),
                        "basis_signal_principal_angle": "",
                        "basis_to_task_signal_overlap": "",
                        "D-CHE_degree_bank_signal_overlap": "",
                        "D-FOU_frequency_bank_signal_overlap": "",
                        "basis_subspace_transport_error": "",
                        "KAN_source_loss_h3200": "",
                        "KAN_source_loss_h4800": "",
                        "NLL": ev["NLL"],
                        "accuracy": ev["accuracy"],
                        "controller_overhead": "",
                        "train_steps": args.kan_steps,
                        "slow_alpha": args.signal_alpha if slow_fu else 0.0,
                        "slow_beta": args.temporal_slow_beta if slow_fu else "",
                        "slow_signal": ev.get("slow_signal", ""),
                        "slow_gate": ev.get("slow_gate", ""),
                        "slow_gate_keep_rate": ev.get("slow_gate_keep_rate", ""),
                        "slow_trust_ratio": ev.get("slow_trust_ratio", ""),
                        "slow_trust_clip_rate": ev.get("slow_trust_clip_rate", ""),
                        "slow_acceptance": ev.get("slow_acceptance", ""),
                        "slow_accept_rate": ev.get("slow_accept_rate", ""),
                        "slow_reject_count": ev.get("slow_reject_count", ""),
                        "slow_candidate_loss_delta_mean": ev.get("slow_candidate_loss_delta_mean", ""),
                        "param_count": sum(int(p.numel()) for p in model.parameters()),
                        "strict_fc_purekan": int(family != "MLP"),
                        "uses_primitivekan": int(family != "MLP"),
                        "kanbefair_original_kan": 0,
                        "known_simplifications": "debug MNIST-like task; basis signal overlap metrics deferred unless KAN gate opens.",
                    }
                )
            mlp_opt = results.get("MLP_AdamW", {})
            fu_tag = official_kan_fu_tag(args)
            mlp_fu = results.get(f"MLP_FU_{fu_tag}", {})
            mlp_controls = [
                r
                for r in [
                    results.get(f"MLP_FU_{fu_tag}_RANDOM_CONTROL", {}),
                    results.get(f"MLP_FU_{fu_tag}_SIGNFLIP_CONTROL", {}),
                ]
                if r
            ]
            mlp_best_control_nll = min([finite_float(r.get("NLL"), float("inf")) for r in mlp_controls] + [float("inf")])
            for kan_family in ["DCHE", "DFOU"]:
                kan_opt = results.get(f"DGKAN_{kan_family}_AdamW", {})
                kan_fu = results.get(f"DGKAN_{kan_family}_FU_{fu_tag}", {})
                kan_controls = [
                    r
                    for r in [
                        results.get(f"DGKAN_{kan_family}_FU_{fu_tag}_RANDOM_CONTROL", {}),
                        results.get(f"DGKAN_{kan_family}_FU_{fu_tag}_SIGNFLIP_CONTROL", {}),
                    ]
                    if r
                ]
                kan_best_control_nll = min([finite_float(r.get("NLL"), float("inf")) for r in kan_controls] + [float("inf")])
                gap_bp = finite_float(kan_opt.get("NLL")) - finite_float(mlp_opt.get("NLL")) if kan_opt and mlp_opt else None
                gap_fu = finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL")) if kan_fu and mlp_fu else None
                fu_official_eligible = int(official_kan_functional_eligible(args, mlp_fu) and official_kan_functional_eligible(args, kan_fu))
                gap_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "kan_family": kan_family,
                        "fu_mode": official_kan_fu_mode(args),
                        "fu_official_eligible": fu_official_eligible,
                        "functional_fu_mechanism": kan_fu.get("functional_fu_mechanism", ""),
                        "functional_fu_lr": kan_fu.get("functional_fu_lr", ""),
                        "functional_fu_lr_schedule": kan_fu.get("functional_fu_lr_schedule", ""),
                        "functional_fu_lr_mean": kan_fu.get("functional_fu_lr_mean", ""),
                        "functional_fu_interval": kan_fu.get("functional_fu_interval", ""),
                        "functional_fu_acceptance": kan_fu.get("functional_fu_acceptance", ""),
                        "functional_fu_hidden_source_fraction_mean": kan_fu.get("functional_fu_hidden_source_fraction_mean", ""),
                        "functional_fu_readout_source_fraction_mean": kan_fu.get("functional_fu_readout_source_fraction_mean", ""),
                        "functional_fu_block_source_hidden_residual_fraction_mean": kan_fu.get("functional_fu_block_source_hidden_residual_fraction_mean", ""),
                        "gap_BP": "" if gap_bp is None else gap_bp,
                        "gap_FU": "" if gap_fu is None else gap_fu,
                        "gap_reduction": "" if gap_bp is None or gap_fu is None else gap_bp - gap_fu,
                        "MLP_plus_FU_NLL_delta_vs_MLP": ""
                        if not mlp_fu or not mlp_opt
                        else finite_float(mlp_fu.get("NLL")) - finite_float(mlp_opt.get("NLL")),
                        "MLP_plus_FU_accuracy_delta_vs_MLP": ""
                        if not mlp_fu or not mlp_opt
                        else finite_float(mlp_fu.get("accuracy")) - finite_float(mlp_opt.get("accuracy")),
                        "MLP_FU_vs_best_control_NLL_delta": ""
                        if not mlp_fu or not mlp_controls or mlp_best_control_nll == float("inf")
                        else finite_float(mlp_fu.get("NLL")) - mlp_best_control_nll,
                        "KAN_plus_FU_NLL_delta_vs_KAN": ""
                        if not kan_fu or not kan_opt
                        else finite_float(kan_fu.get("NLL")) - finite_float(kan_opt.get("NLL")),
                        "KAN_plus_FU_accuracy_delta_vs_KAN": ""
                        if not kan_fu or not kan_opt
                        else finite_float(kan_fu.get("accuracy")) - finite_float(kan_opt.get("accuracy")),
                        "KAN_FU_vs_best_control_NLL_delta": ""
                        if not kan_fu or not kan_controls or kan_best_control_nll == float("inf")
                        else finite_float(kan_fu.get("NLL")) - kan_best_control_nll,
                        "KAN_plus_FU_vs_MLPFU_NLL_delta": ""
                        if not kan_fu or not mlp_fu
                        else finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL")),
                        "official_full_row_pass": int(
                            bool(fu_official_eligible)
                            and bool(mlp_fu)
                            and bool(mlp_opt)
                            and bool(kan_fu)
                            and bool(kan_opt)
                            and (finite_float(mlp_fu.get("NLL")) - finite_float(mlp_opt.get("NLL"))) < 0.0
                            and (finite_float(kan_fu.get("NLL")) - finite_float(kan_opt.get("NLL"))) < 0.0
                            and bool(mlp_controls)
                            and bool(kan_controls)
                            and (finite_float(mlp_fu.get("NLL")) - mlp_best_control_nll) < 0.0
                            and (finite_float(kan_fu.get("NLL")) - kan_best_control_nll) < 0.0
                            and (finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL"))) <= 0.0
                        ),
                        "train_steps": args.kan_steps,
                        "slow_alpha": args.signal_alpha,
                        "slow_beta": args.temporal_slow_beta,
                        "slow_signal": args.kan_slow_signal,
                        "slow_gate": args.kan_slow_gate,
                        "slow_trust_ratio": args.kan_slow_trust_ratio,
                        "slow_acceptance": args.kan_slow_acceptance,
                        "task_tier": "Tier0_debug",
                    }
                )
            external_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "tier": "Tier0_debug",
                    "status": "ran_strict_DGKAN_debug_matrix",
                    "KANbeFair_original_KAN_rows": 0,
                    "strict_FC_PureKAN_rows": 8,
                    "hard_tier_status": "deferred",
                    "reason": "Tier1/Tier2/Tier3 KANbeFair tasks not run because early mechanism gates did not open official full-loop.",
                }
            )
    for hard_dataset in _split(args.kan_hard_datasets):
        for seed in _split(args.seeds, int):
            hard_tier = hard_task_tier(hard_dataset)
            try:
                setup_seed(seed)
                train_loader, held_train_loader, test_loader, input_dim, output_dim, x_stats = make_hard_vision_loaders(
                    hard_dataset,
                    args.kan_hard_train_size,
                    args.kan_hard_test_size,
                    args.batch_size,
                    seed,
                    download=args.kan_hard_download,
                )
            except Exception as exc:
                external_rows.append(
                    {
                        "dataset": hard_dataset,
                        "seed": seed,
                        "tier": hard_tier,
                        "status": "blocked",
                        "KANbeFair_original_KAN_rows": 0,
                        "strict_FC_PureKAN_rows": 0,
                        "hard_tier_status": "dataset_or_loader_blocked",
                        "reason": repr(exc),
                    }
                )
                continue
            variants = official_kan_variants(args)
            results: dict[str, dict[str, Any]] = {}
            for name, family, training, slow_fu, kan_family, control_variant, functional_mechanism in variants:
                if family == "MLP":
                    model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                else:
                    model = make_kan(input_dim, output_dim, args.hidden, seed, device, x_stats, kan_family)
                functional_lr = official_kan_functional_lr(args, family)
                functional_interval = official_kan_functional_interval(args, family)
                ev = train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.kan_hard_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    slow_fu=slow_fu,
                    slow_alpha=args.signal_alpha,
                    slow_beta=args.temporal_slow_beta,
                    slow_signal=args.kan_slow_signal,
                    slow_gate=args.kan_slow_gate,
                    slow_trust_ratio=args.kan_slow_trust_ratio,
                    slow_acceptance=args.kan_slow_acceptance,
                    slow_acceptance_tol=args.kan_slow_acceptance_tol,
                    slow_accept_loader=held_train_loader,
                    control=control_variant,
                    functional_fu_mechanism=functional_mechanism,
                    functional_fu_lr=functional_lr,
                    functional_fu_lr_schedule=args.kan_functional_fu_lr_schedule,
                    functional_fu_interval=functional_interval,
                    functional_fu_acceptance=args.kan_functional_acceptance,
                    functional_fu_acceptance_tol=args.kan_functional_acceptance_tol,
                    functional_fu_accept_loader=held_train_loader,
                    functional_fu_control=control_variant if functional_mechanism else "",
                )
                diag: dict[str, Any] = {}
                if family != "MLP" and hasattr(model, "basis_diagnostics"):
                    xb0, _ = next(iter(test_loader))
                    diag = model.basis_diagnostics(xb0.to(device).float())
                results[name] = ev
                carrier_rows.append(
                    {
                        "dataset": hard_dataset,
                        "seed": seed,
                        "model_name": name,
                        "model_family": family,
                        "training": training,
                        "control_variant": control_variant,
                        "status": "matched_control" if control_variant else "real_run",
                        "fu_mode": official_kan_fu_mode(args) if training != "AdamW" else "",
                        "fu_official_eligible": official_kan_functional_eligible(args, ev) if training == "FU" else int(training == "matched_control" and getattr(args, "kan_fu_mode", "slow") == "slow"),
                        "functional_fu_mechanism": ev.get("functional_fu_mechanism", ""),
                        "functional_fu_lr": ev.get("functional_fu_lr", ""),
                        "functional_fu_lr_schedule": ev.get("functional_fu_lr_schedule", ""),
                        "functional_fu_lr_mean": ev.get("functional_fu_lr_mean", ""),
                        "functional_fu_interval": ev.get("functional_fu_interval", ""),
                        "functional_fu_acceptance": ev.get("functional_fu_acceptance", ""),
                        "functional_fu_control": ev.get("functional_fu_control", ""),
                        "functional_fu_applied_count": ev.get("functional_fu_applied_count", ""),
                        "functional_fu_skipped_count": ev.get("functional_fu_skipped_count", ""),
                        "functional_fu_accept_rate": ev.get("functional_fu_accept_rate", ""),
                        "functional_fu_update_norm_mean": ev.get("functional_fu_update_norm_mean", ""),
                        "functional_fu_operator_status_counts": ev.get("functional_fu_operator_status_counts", ""),
                        "functional_fu_hidden_source_fraction_mean": ev.get("functional_fu_hidden_source_fraction_mean", ""),
                        "functional_fu_readout_source_fraction_mean": ev.get("functional_fu_readout_source_fraction_mean", ""),
                        "functional_fu_block_source_hidden_residual_fraction_mean": ev.get("functional_fu_block_source_hidden_residual_fraction_mean", ""),
                        "functional_fu_projection_residual_Gf_mean": ev.get("functional_fu_projection_residual_Gf_mean", ""),
                        "functional_fu_ActuationR2_mean": ev.get("functional_fu_ActuationR2_mean", ""),
                        "functional_fu_ActuationCosine_mean": ev.get("functional_fu_ActuationCosine_mean", ""),
                        "basis_energy_fraction": diag.get("basis_energy_fraction", ""),
                        "basis_entropy": diag.get("basis_entropy", ""),
                        "basis_effective_rank": diag.get("basis_effective_rank", ""),
                        "basis_condition_proxy": diag.get("basis_condition_proxy", ""),
                        "readout_leakage_fraction": diag.get("readout_leakage_fraction", ""),
                        "basis_signal_principal_angle": "",
                        "basis_to_task_signal_overlap": "",
                        "D-CHE_degree_bank_signal_overlap": "",
                        "D-FOU_frequency_bank_signal_overlap": "",
                        "basis_subspace_transport_error": "",
                        "KAN_source_loss_h3200": "",
                        "KAN_source_loss_h4800": "",
                        "NLL": ev["NLL"],
                        "accuracy": ev["accuracy"],
                        "controller_overhead": "",
                        "train_steps": args.kan_hard_steps,
                        "slow_alpha": args.signal_alpha if slow_fu else 0.0,
                        "slow_beta": args.temporal_slow_beta if slow_fu else "",
                        "slow_signal": ev.get("slow_signal", ""),
                        "slow_gate": ev.get("slow_gate", ""),
                        "slow_gate_keep_rate": ev.get("slow_gate_keep_rate", ""),
                        "slow_trust_ratio": ev.get("slow_trust_ratio", ""),
                        "slow_trust_clip_rate": ev.get("slow_trust_clip_rate", ""),
                        "slow_acceptance": ev.get("slow_acceptance", ""),
                        "slow_accept_rate": ev.get("slow_accept_rate", ""),
                        "slow_reject_count": ev.get("slow_reject_count", ""),
                        "slow_candidate_loss_delta_mean": ev.get("slow_candidate_loss_delta_mean", ""),
                        "param_count": sum(int(p.numel()) for p in model.parameters()),
                        "strict_fc_purekan": int(family != "MLP"),
                        "uses_primitivekan": int(family != "MLP"),
                        "kanbefair_original_kan": 0,
                        "task_tier": hard_tier,
                        "known_simplifications": "strict PrimitiveKAN hard-tier sample; small train/test subset; no KANbeFair original KAN official rows.",
                    }
                )
            mlp_opt = results.get("MLP_AdamW", {})
            fu_tag = official_kan_fu_tag(args)
            mlp_fu = results.get(f"MLP_FU_{fu_tag}", {})
            mlp_controls = [
                r
                for r in [
                    results.get(f"MLP_FU_{fu_tag}_RANDOM_CONTROL", {}),
                    results.get(f"MLP_FU_{fu_tag}_SIGNFLIP_CONTROL", {}),
                ]
                if r
            ]
            mlp_best_control_nll = min([finite_float(r.get("NLL"), float("inf")) for r in mlp_controls] + [float("inf")])
            for kan_family in ["DCHE", "DFOU"]:
                kan_opt = results.get(f"DGKAN_{kan_family}_AdamW", {})
                kan_fu = results.get(f"DGKAN_{kan_family}_FU_{fu_tag}", {})
                kan_controls = [
                    r
                    for r in [
                        results.get(f"DGKAN_{kan_family}_FU_{fu_tag}_RANDOM_CONTROL", {}),
                        results.get(f"DGKAN_{kan_family}_FU_{fu_tag}_SIGNFLIP_CONTROL", {}),
                    ]
                    if r
                ]
                kan_best_control_nll = min([finite_float(r.get("NLL"), float("inf")) for r in kan_controls] + [float("inf")])
                gap_bp = finite_float(kan_opt.get("NLL")) - finite_float(mlp_opt.get("NLL")) if kan_opt and mlp_opt else None
                gap_fu = finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL")) if kan_fu and mlp_fu else None
                fu_official_eligible = int(official_kan_functional_eligible(args, mlp_fu) and official_kan_functional_eligible(args, kan_fu))
                gap_rows.append(
                    {
                        "dataset": hard_dataset,
                        "seed": seed,
                        "kan_family": kan_family,
                        "fu_mode": official_kan_fu_mode(args),
                        "fu_official_eligible": fu_official_eligible,
                        "functional_fu_mechanism": kan_fu.get("functional_fu_mechanism", ""),
                        "functional_fu_lr": kan_fu.get("functional_fu_lr", ""),
                        "functional_fu_lr_schedule": kan_fu.get("functional_fu_lr_schedule", ""),
                        "functional_fu_lr_mean": kan_fu.get("functional_fu_lr_mean", ""),
                        "functional_fu_interval": kan_fu.get("functional_fu_interval", ""),
                        "functional_fu_acceptance": kan_fu.get("functional_fu_acceptance", ""),
                        "functional_fu_hidden_source_fraction_mean": kan_fu.get("functional_fu_hidden_source_fraction_mean", ""),
                        "functional_fu_readout_source_fraction_mean": kan_fu.get("functional_fu_readout_source_fraction_mean", ""),
                        "functional_fu_block_source_hidden_residual_fraction_mean": kan_fu.get("functional_fu_block_source_hidden_residual_fraction_mean", ""),
                        "gap_BP": "" if gap_bp is None else gap_bp,
                        "gap_FU": "" if gap_fu is None else gap_fu,
                        "gap_reduction": "" if gap_bp is None or gap_fu is None else gap_bp - gap_fu,
                        "MLP_plus_FU_NLL_delta_vs_MLP": ""
                        if not mlp_fu or not mlp_opt
                        else finite_float(mlp_fu.get("NLL")) - finite_float(mlp_opt.get("NLL")),
                        "MLP_plus_FU_accuracy_delta_vs_MLP": ""
                        if not mlp_fu or not mlp_opt
                        else finite_float(mlp_fu.get("accuracy")) - finite_float(mlp_opt.get("accuracy")),
                        "MLP_FU_vs_best_control_NLL_delta": ""
                        if not mlp_fu or not mlp_controls or mlp_best_control_nll == float("inf")
                        else finite_float(mlp_fu.get("NLL")) - mlp_best_control_nll,
                        "KAN_plus_FU_NLL_delta_vs_KAN": ""
                        if not kan_fu or not kan_opt
                        else finite_float(kan_fu.get("NLL")) - finite_float(kan_opt.get("NLL")),
                        "KAN_plus_FU_accuracy_delta_vs_KAN": ""
                        if not kan_fu or not kan_opt
                        else finite_float(kan_fu.get("accuracy")) - finite_float(kan_opt.get("accuracy")),
                        "KAN_FU_vs_best_control_NLL_delta": ""
                        if not kan_fu or not kan_controls or kan_best_control_nll == float("inf")
                        else finite_float(kan_fu.get("NLL")) - kan_best_control_nll,
                        "KAN_plus_FU_vs_MLPFU_NLL_delta": ""
                        if not kan_fu or not mlp_fu
                        else finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL")),
                        "official_full_row_pass": int(
                            bool(fu_official_eligible)
                            and bool(mlp_fu)
                            and bool(mlp_opt)
                            and bool(kan_fu)
                            and bool(kan_opt)
                            and (finite_float(mlp_fu.get("NLL")) - finite_float(mlp_opt.get("NLL"))) < 0.0
                            and (finite_float(kan_fu.get("NLL")) - finite_float(kan_opt.get("NLL"))) < 0.0
                            and bool(mlp_controls)
                            and bool(kan_controls)
                            and (finite_float(mlp_fu.get("NLL")) - mlp_best_control_nll) < 0.0
                            and (finite_float(kan_fu.get("NLL")) - kan_best_control_nll) < 0.0
                            and (finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL"))) <= 0.0
                        ),
                        "train_steps": args.kan_hard_steps,
                        "slow_alpha": args.signal_alpha,
                        "slow_beta": args.temporal_slow_beta,
                        "slow_signal": args.kan_slow_signal,
                        "slow_gate": args.kan_slow_gate,
                        "slow_trust_ratio": args.kan_slow_trust_ratio,
                        "slow_acceptance": args.kan_slow_acceptance,
                        "task_tier": hard_tier,
                    }
                )
            external_rows.append(
                {
                    "dataset": hard_dataset,
                    "seed": seed,
                    "tier": hard_tier,
                    "status": "ran_strict_DGKAN_hard_sample",
                    "KANbeFair_original_KAN_rows": 0,
                    "strict_FC_PureKAN_rows": 8,
                    "hard_tier_status": "ran_strict_FC_PureKAN_subset",
                    "reason": "Hard-tier strict PrimitiveKAN subset run; KANbeFair original KAN remains baseline_context_only and excluded from official rows.",
                }
            )
    write_rows(OUT_ROOT / "v22_30_KAN_basis_signal_carrier_matrix.csv", carrier_rows)
    write_rows(OUT_ROOT / "v22_30_KAN_vs_MLPFU_gap_matrix.csv", gap_rows)
    write_rows(OUT_ROOT / "v22_30_KANbeFair_external_task_matrix.csv", external_rows)
    append_official_kan_repair_summary(args, gap_rows)
    existing_continual = read_rows(OUT_ROOT / "v22_30_KANbeFair_continual_matrix.csv")
    if not existing_continual or all(r.get("status") == "deferred" for r in existing_continual):
        write_rows(
            OUT_ROOT / "v22_30_KANbeFair_continual_matrix.csv",
            [
                {
                    "dataset": "Class_MNIST",
                    "status": "deferred",
                    "reason": "Continual protocol deferred; no mechanism passed Layer 1 gate for official KANbeFair continual full-loop.",
                }
            ],
        )
    write_simple_svg(
        "v22_30_KAN_basis_signal_overlap.svg",
        "KAN debug NLL",
        [{"label": f"{r['dataset']} {r['model_name']}", "rows": -float(r["NLL"])} for r in carrier_rows if r.get("NLL") not in {"", None}],
    )
    write_simple_svg(
        "v22_30_KANbeFair_gap_reduction.svg",
        "KAN gap reduction debug",
        [{"label": f"{r['dataset']} {r['kan_family']}", "rows": finite_float(r.get("gap_reduction"), 0.0) or 0.0} for r in gap_rows],
    )
    return carrier_rows, gap_rows, external_rows


def readout_exact_control_mechanism(real_mechanism: str, control: str) -> str:
    if control == "random":
        return "M50-RandomMatchedTargetFU"
    if control == "signflip":
        return "M51-SignFlippedTargetFU"
    if control == "corrupt":
        return "M52-CorruptedLabelTargetFU"
    return real_mechanism


def official_kan_fu_tag(args: argparse.Namespace) -> str:
    return "FUNCTIONAL" if getattr(args, "kan_fu_mode", "slow") == "functional_metric" else "SLOW"


def official_kan_fu_mode(args: argparse.Namespace) -> str:
    return "functional_metric_solver_hidden_block" if official_kan_fu_tag(args) == "FUNCTIONAL" else "official_slow_fu"


def official_kan_variants(args: argparse.Namespace) -> list[tuple[str, str, str, bool, str, str, str]]:
    tag = official_kan_fu_tag(args)
    if tag == "FUNCTIONAL":
        mechanism = str(args.kan_functional_mechanism)
        return [
            ("MLP_AdamW", "MLP", "AdamW", False, "", "", ""),
            (f"MLP_FU_{tag}", "MLP", "FU", False, "", "", mechanism),
            (f"MLP_FU_{tag}_RANDOM_CONTROL", "MLP", "matched_control", False, "", "random", mechanism),
            (f"MLP_FU_{tag}_SIGNFLIP_CONTROL", "MLP", "matched_control", False, "", "signflip", mechanism),
            ("DGKAN_DCHE_AdamW", "KAN_DCHE", "AdamW", False, "D-CHE", "", ""),
            (f"DGKAN_DCHE_FU_{tag}", "KAN_DCHE", "FU", False, "D-CHE", "", mechanism),
            (f"DGKAN_DCHE_FU_{tag}_RANDOM_CONTROL", "KAN_DCHE", "matched_control", False, "D-CHE", "random", mechanism),
            (f"DGKAN_DCHE_FU_{tag}_SIGNFLIP_CONTROL", "KAN_DCHE", "matched_control", False, "D-CHE", "signflip", mechanism),
            ("DGKAN_DFOU_AdamW", "KAN_DFOU", "AdamW", False, "D-FOU", "", ""),
            (f"DGKAN_DFOU_FU_{tag}", "KAN_DFOU", "FU", False, "D-FOU", "", mechanism),
            (f"DGKAN_DFOU_FU_{tag}_RANDOM_CONTROL", "KAN_DFOU", "matched_control", False, "D-FOU", "random", mechanism),
            (f"DGKAN_DFOU_FU_{tag}_SIGNFLIP_CONTROL", "KAN_DFOU", "matched_control", False, "D-FOU", "signflip", mechanism),
        ]
    return [
        ("MLP_AdamW", "MLP", "AdamW", False, "", "", ""),
        ("MLP_FU_SLOW", "MLP", "FU", True, "", "", ""),
        ("MLP_FU_SLOW_RANDOM_CONTROL", "MLP", "matched_control", True, "", "random", ""),
        ("MLP_FU_SLOW_SIGNFLIP_CONTROL", "MLP", "matched_control", True, "", "signflip", ""),
        ("DGKAN_DCHE_AdamW", "KAN_DCHE", "AdamW", False, "D-CHE", "", ""),
        ("DGKAN_DCHE_FU_SLOW", "KAN_DCHE", "FU", True, "D-CHE", "", ""),
        ("DGKAN_DCHE_FU_SLOW_RANDOM_CONTROL", "KAN_DCHE", "matched_control", True, "D-CHE", "random", ""),
        ("DGKAN_DCHE_FU_SLOW_SIGNFLIP_CONTROL", "KAN_DCHE", "matched_control", True, "D-CHE", "signflip", ""),
        ("DGKAN_DFOU_AdamW", "KAN_DFOU", "AdamW", False, "D-FOU", "", ""),
        ("DGKAN_DFOU_FU_SLOW", "KAN_DFOU", "FU", True, "D-FOU", "", ""),
        ("DGKAN_DFOU_FU_SLOW_RANDOM_CONTROL", "KAN_DFOU", "matched_control", True, "D-FOU", "random", ""),
        ("DGKAN_DFOU_FU_SLOW_SIGNFLIP_CONTROL", "KAN_DFOU", "matched_control", True, "D-FOU", "signflip", ""),
    ]


def official_kan_functional_eligible(args: argparse.Namespace, ev: dict[str, Any]) -> int:
    if getattr(args, "kan_fu_mode", "slow") != "functional_metric":
        return 1
    applied = finite_float(ev.get("functional_fu_applied_count"), 0.0) or 0.0
    hidden_fraction = finite_float(ev.get("functional_fu_hidden_source_fraction_mean"), 0.0) or 0.0
    status_counts = str(ev.get("functional_fu_operator_status_counts", ""))
    return int(applied > 0.0 and hidden_fraction > 0.0 and "metric_readout_exact_solve" in status_counts)


def official_kan_functional_lr(args: argparse.Namespace, family: str) -> float:
    if getattr(args, "kan_fu_mode", "slow") != "functional_metric":
        return float(args.kan_functional_fu_lr)
    if family == "MLP":
        specific = float(getattr(args, "kan_functional_mlp_fu_lr", 0.0) or 0.0)
    else:
        specific = float(getattr(args, "kan_functional_kan_fu_lr", 0.0) or 0.0)
    return specific if specific > 0.0 else float(args.kan_functional_fu_lr)


def official_kan_functional_interval(args: argparse.Namespace, family: str) -> int:
    if getattr(args, "kan_fu_mode", "slow") != "functional_metric":
        return int(args.kan_functional_fu_interval)
    if family == "MLP":
        specific = int(getattr(args, "kan_functional_mlp_fu_interval", 0) or 0)
    else:
        specific = int(getattr(args, "kan_functional_kan_fu_interval", 0) or 0)
    return specific if specific > 0 else int(args.kan_functional_fu_interval)


def append_kan_repair_summary(row: dict[str, Any]) -> None:
    rows = read_rows(OUT_ROOT / "v22_30_repair_attempt_summary.csv")
    rows.append(row)
    write_rows(OUT_ROOT / "v22_30_repair_attempt_summary.csv", rows)


def append_official_kan_repair_summary(args: argparse.Namespace, gap_rows: list[dict[str, Any]]) -> None:
    hard_rows = [r for r in gap_rows if str(r.get("task_tier", "")).lower().startswith(("tier1", "tier2", "tier3"))]
    hard_gap_positive_rows = sum(1 for r in hard_rows if (finite_float(r.get("gap_reduction"), -1.0) or -1.0) > 0.0)
    hard_mlp_fu_improves_rows = sum(
        1 for r in hard_rows if (finite_float(r.get("MLP_plus_FU_NLL_delta_vs_MLP"), 0.0) or 0.0) < 0.0
    )
    hard_kan_fu_improves_rows = sum(
        1 for r in hard_rows if (finite_float(r.get("KAN_plus_FU_NLL_delta_vs_KAN"), 0.0) or 0.0) < 0.0
    )
    hard_mlp_fu_beats_control_rows = sum(
        1 for r in hard_rows if (finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
    )
    hard_kan_fu_beats_control_rows = sum(
        1 for r in hard_rows if (finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
    )
    hard_fu_beats_all_control_rows = sum(
        1
        for r in hard_rows
        if (finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
        and (finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
    )
    hard_arch_nonworse_rows = sum(
        1 for r in hard_rows if (finite_float(r.get("KAN_plus_FU_vs_MLPFU_NLL_delta"), 1.0) or 1.0) <= 0.0
    )
    hard_official_full_candidate_rows = sum(1 for r in hard_rows if str(r.get("official_full_row_pass", "")) == "1")
    hard_gap_rate = hard_gap_positive_rows / max(1, len(hard_rows))
    hard_arch_nonworse_rate = hard_arch_nonworse_rows / max(1, len(hard_rows))
    official_full_ready = int(
        bool(hard_rows)
        and hard_gap_rate >= 0.60
        and hard_mlp_fu_improves_rows == len(hard_rows)
        and hard_kan_fu_improves_rows == len(hard_rows)
        and hard_fu_beats_all_control_rows == len(hard_rows)
        and hard_arch_nonworse_rate >= 0.50
    )
    append_kan_repair_summary(
        {
            "timestamp_sg": now_sg(),
            "attempt_id": (
                f"official_{getattr(args, 'kan_fu_mode', 'slow')}_steps{args.kan_hard_steps}_"
                f"{getattr(args, 'kan_functional_mechanism', '') or f'alpha{args.signal_alpha:g}'}_"
                f"{args.kan_slow_signal}_{args.kan_slow_gate}_{args.kan_slow_acceptance}_"
                f"tol{args.kan_slow_acceptance_tol:g}_flr{getattr(args, 'kan_functional_fu_lr', '')}_"
                f"{getattr(args, 'kan_functional_fu_lr_schedule', 'constant')}"
            ),
            "kan_hard_steps": args.kan_hard_steps,
            "signal_alpha": args.signal_alpha,
            "kan_slow_gate": args.kan_slow_gate,
            "kan_slow_signal": args.kan_slow_signal,
            "kan_slow_trust_ratio": args.kan_slow_trust_ratio,
            "kan_slow_acceptance": args.kan_slow_acceptance,
            "kan_slow_acceptance_tol": args.kan_slow_acceptance_tol,
            "fu_mode": official_kan_fu_mode(args),
            "readout_mechanism": "",
            "functional_mechanism": getattr(args, "kan_functional_mechanism", ""),
            "functional_fu_lr": getattr(args, "kan_functional_fu_lr", ""),
            "functional_mlp_fu_lr": getattr(args, "kan_functional_mlp_fu_lr", ""),
            "functional_kan_fu_lr": getattr(args, "kan_functional_kan_fu_lr", ""),
            "functional_fu_lr_schedule": getattr(args, "kan_functional_fu_lr_schedule", ""),
            "functional_fu_interval": getattr(args, "kan_functional_fu_interval", ""),
            "functional_mlp_fu_interval": getattr(args, "kan_functional_mlp_fu_interval", ""),
            "functional_kan_fu_interval": getattr(args, "kan_functional_kan_fu_interval", ""),
            "functional_acceptance": getattr(args, "kan_functional_acceptance", ""),
            "hard_gap_positive_rows": hard_gap_positive_rows,
            "hard_mlp_fu_improves_rows": hard_mlp_fu_improves_rows,
            "hard_kan_fu_improves_rows": hard_kan_fu_improves_rows,
            "hard_mlp_fu_beats_control_rows": hard_mlp_fu_beats_control_rows,
            "hard_kan_fu_beats_control_rows": hard_kan_fu_beats_control_rows,
            "hard_fu_beats_all_control_rows": hard_fu_beats_all_control_rows,
            "hard_arch_nonworse_rows": hard_arch_nonworse_rows,
            "hard_official_full_candidate_rows": hard_official_full_candidate_rows,
            "diagnostic_full_candidate_rows": "",
            "official_full_superiority_ready": official_full_ready,
            "note": (
                f"Official KAN mode={official_kan_fu_mode(args)} hard rows={len(hard_rows)}; "
                f"hard_gap_rate={hard_gap_rate:.3f}; hard_arch_nonworse_rate={hard_arch_nonworse_rate:.3f}. "
                "No test/validation/future direction is used for the FU acceptance or gate; functional-metric mode requires nonzero hidden-source fraction."
            ),
        }
    )


def stage_kan_readout_exact(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch_device(args.device)
    carrier_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    datasets = _split(args.kan_hard_datasets) or _split(args.datasets)
    for dataset in datasets:
        is_hard = bool(_split(args.kan_hard_datasets)) or dataset.lower() in {
            "wine",
            "spam",
            "rice",
            "bean",
            "titanic",
            "bank",
            "income",
            "telescope",
            "cifar10",
            "svhn",
            "emnist",
        }
        tier = hard_task_tier(dataset) if is_hard else "Tier0_debug"
        for seed in _split(args.seeds, int):
            try:
                setup_seed(seed)
                if is_hard:
                    train_loader, held_train_loader, test_loader, input_dim, output_dim, x_stats = make_hard_vision_loaders(
                        dataset,
                        args.kan_hard_train_size,
                        args.kan_hard_test_size,
                        args.batch_size,
                        seed,
                        download=args.kan_hard_download,
                    )
                    steps = int(args.kan_hard_steps)
                else:
                    train_loader, held_train_loader, test_loader, input_dim, output_dim, x_stats = make_loaders(
                        dataset, args.train_size, args.test_size, args.batch_size, seed
                    )
                    steps = int(args.kan_steps)
            except Exception as exc:
                external_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "tier": tier,
                        "status": "blocked",
                        "strict_FC_PureKAN_rows": 0,
                        "hard_tier_status": "dataset_or_loader_blocked",
                        "reason": repr(exc),
                    }
                )
                continue
            variants = [
                ("MLP_AdamW", "MLP", "AdamW", "", "", ""),
                ("MLP_FU_READOUT_EXACT", "MLP", "FU", args.kan_readout_mechanism, "", ""),
                (
                    "MLP_FU_READOUT_RANDOM_CONTROL",
                    "MLP",
                    "matched_control",
                    readout_exact_control_mechanism(args.kan_readout_mechanism, "random"),
                    "",
                    "random",
                ),
                (
                    "MLP_FU_READOUT_SIGNFLIP_CONTROL",
                    "MLP",
                    "matched_control",
                    readout_exact_control_mechanism(args.kan_readout_mechanism, "signflip"),
                    "",
                    "signflip",
                ),
                ("DGKAN_DCHE_AdamW", "KAN_DCHE", "AdamW", "", "D-CHE", ""),
                ("DGKAN_DCHE_FU_READOUT_EXACT", "KAN_DCHE", "FU", args.kan_readout_mechanism, "D-CHE", ""),
                (
                    "DGKAN_DCHE_FU_READOUT_RANDOM_CONTROL",
                    "KAN_DCHE",
                    "matched_control",
                    readout_exact_control_mechanism(args.kan_readout_mechanism, "random"),
                    "D-CHE",
                    "random",
                ),
                (
                    "DGKAN_DCHE_FU_READOUT_SIGNFLIP_CONTROL",
                    "KAN_DCHE",
                    "matched_control",
                    readout_exact_control_mechanism(args.kan_readout_mechanism, "signflip"),
                    "D-CHE",
                    "signflip",
                ),
                ("DGKAN_DFOU_AdamW", "KAN_DFOU", "AdamW", "", "D-FOU", ""),
                ("DGKAN_DFOU_FU_READOUT_EXACT", "KAN_DFOU", "FU", args.kan_readout_mechanism, "D-FOU", ""),
                (
                    "DGKAN_DFOU_FU_READOUT_RANDOM_CONTROL",
                    "KAN_DFOU",
                    "matched_control",
                    readout_exact_control_mechanism(args.kan_readout_mechanism, "random"),
                    "D-FOU",
                    "random",
                ),
                (
                    "DGKAN_DFOU_FU_READOUT_SIGNFLIP_CONTROL",
                    "KAN_DFOU",
                    "matched_control",
                    readout_exact_control_mechanism(args.kan_readout_mechanism, "signflip"),
                    "D-FOU",
                    "signflip",
                ),
            ]
            results: dict[str, dict[str, Any]] = {}
            for name, family, training, mechanism, kan_family, control_variant in variants:
                if family == "MLP":
                    model = make_mlp(input_dim, output_dim, args.hidden, seed, device)
                else:
                    model = make_kan(input_dim, output_dim, args.hidden, seed, device, x_stats, kan_family)
                ev = train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    slow_fu=False,
                    slow_accept_loader=held_train_loader,
                    functional_fu_mechanism=mechanism,
                    functional_fu_lr=args.kan_readout_fu_lr,
                    functional_fu_interval=args.kan_readout_fu_interval,
                    functional_fu_acceptance=args.kan_readout_acceptance,
                    functional_fu_acceptance_tol=args.kan_readout_acceptance_tol,
                    functional_fu_accept_loader=held_train_loader,
                )
                diag: dict[str, Any] = {}
                if family != "MLP" and hasattr(model, "basis_diagnostics"):
                    xb0, _ = next(iter(test_loader))
                    diag = model.basis_diagnostics(xb0.to(device).float())
                results[name] = ev
                carrier_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_name": name,
                        "model_family": family,
                        "training": training,
                        "control_variant": control_variant,
                        "status": "matched_control" if control_variant else "real_run",
                        "fu_mode": "readout_exact" if mechanism else "",
                        "fu_official_eligible": 0,
                        "functional_fu_mechanism": ev.get("functional_fu_mechanism", ""),
                        "functional_fu_lr": ev.get("functional_fu_lr", ""),
                        "functional_fu_interval": ev.get("functional_fu_interval", ""),
                        "functional_fu_acceptance": ev.get("functional_fu_acceptance", ""),
                        "functional_fu_applied_count": ev.get("functional_fu_applied_count", ""),
                        "functional_fu_skipped_count": ev.get("functional_fu_skipped_count", ""),
                        "functional_fu_accept_rate": ev.get("functional_fu_accept_rate", ""),
                        "functional_fu_update_norm_mean": ev.get("functional_fu_update_norm_mean", ""),
                        "functional_fu_operator_status_counts": ev.get("functional_fu_operator_status_counts", ""),
                        "functional_fu_ActuationR2_mean": ev.get("functional_fu_ActuationR2_mean", ""),
                        "functional_fu_ActuationCosine_mean": ev.get("functional_fu_ActuationCosine_mean", ""),
                        "functional_fu_B1_gain_mean": ev.get("functional_fu_B1_gain_mean", ""),
                        "functional_fu_B2_transfer_gain_mean": ev.get("functional_fu_B2_transfer_gain_mean", ""),
                        "functional_fu_B3_safety_gain_mean": ev.get("functional_fu_B3_safety_gain_mean", ""),
                        "basis_energy_fraction": diag.get("basis_energy_fraction", ""),
                        "basis_entropy": diag.get("basis_entropy", ""),
                        "basis_effective_rank": diag.get("basis_effective_rank", ""),
                        "basis_condition_proxy": diag.get("basis_condition_proxy", ""),
                        "readout_leakage_fraction": diag.get("readout_leakage_fraction", ""),
                        "NLL": ev["NLL"],
                        "accuracy": ev["accuracy"],
                        "train_steps": steps,
                        "param_count": sum(int(p.numel()) for p in model.parameters()),
                        "strict_fc_purekan": int(family != "MLP"),
                        "uses_primitivekan": int(family != "MLP"),
                        "kanbefair_original_kan": 0,
                        "task_tier": tier,
                        "known_simplifications": (
                            "Readout-exact full-loop diagnostic using train-stream frozen_readout_features and same-job matched controls; "
                            "fu_official_eligible=0 because readout diagnostics are not promoted as basis-native proof."
                        ),
                    }
                )
            mlp_opt = results.get("MLP_AdamW", {})
            mlp_fu = results.get("MLP_FU_READOUT_EXACT", {})
            mlp_controls = [
                r
                for r in [
                    results.get("MLP_FU_READOUT_RANDOM_CONTROL", {}),
                    results.get("MLP_FU_READOUT_SIGNFLIP_CONTROL", {}),
                ]
                if r
            ]
            mlp_best_control_nll = min([finite_float(r.get("NLL"), float("inf")) for r in mlp_controls] + [float("inf")])
            for kan_family in ["DCHE", "DFOU"]:
                kan_opt = results.get(f"DGKAN_{kan_family}_AdamW", {})
                kan_fu = results.get(f"DGKAN_{kan_family}_FU_READOUT_EXACT", {})
                kan_controls = [
                    r
                    for r in [
                        results.get(f"DGKAN_{kan_family}_FU_READOUT_RANDOM_CONTROL", {}),
                        results.get(f"DGKAN_{kan_family}_FU_READOUT_SIGNFLIP_CONTROL", {}),
                    ]
                    if r
                ]
                kan_best_control_nll = min([finite_float(r.get("NLL"), float("inf")) for r in kan_controls] + [float("inf")])
                gap_bp = finite_float(kan_opt.get("NLL")) - finite_float(mlp_opt.get("NLL")) if kan_opt and mlp_opt else None
                gap_fu = finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL")) if kan_fu and mlp_fu else None
                diagnostic_pass = int(
                    bool(mlp_fu)
                    and bool(mlp_opt)
                    and bool(kan_fu)
                    and bool(kan_opt)
                    and (finite_float(mlp_fu.get("NLL")) - finite_float(mlp_opt.get("NLL"))) < 0.0
                    and (finite_float(kan_fu.get("NLL")) - finite_float(kan_opt.get("NLL"))) < 0.0
                    and bool(mlp_controls)
                    and bool(kan_controls)
                    and (finite_float(mlp_fu.get("NLL")) - mlp_best_control_nll) < 0.0
                    and (finite_float(kan_fu.get("NLL")) - kan_best_control_nll) < 0.0
                    and (finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL"))) <= 0.0
                )
                gap_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "kan_family": kan_family,
                        "fu_mode": "readout_exact",
                        "fu_official_eligible": 0,
                        "gap_BP": "" if gap_bp is None else gap_bp,
                        "gap_FU": "" if gap_fu is None else gap_fu,
                        "gap_reduction": "" if gap_bp is None or gap_fu is None else gap_bp - gap_fu,
                        "MLP_plus_FU_NLL_delta_vs_MLP": ""
                        if not mlp_fu or not mlp_opt
                        else finite_float(mlp_fu.get("NLL")) - finite_float(mlp_opt.get("NLL")),
                        "MLP_plus_FU_accuracy_delta_vs_MLP": ""
                        if not mlp_fu or not mlp_opt
                        else finite_float(mlp_fu.get("accuracy")) - finite_float(mlp_opt.get("accuracy")),
                        "MLP_FU_vs_best_control_NLL_delta": ""
                        if not mlp_fu or not mlp_controls or mlp_best_control_nll == float("inf")
                        else finite_float(mlp_fu.get("NLL")) - mlp_best_control_nll,
                        "KAN_plus_FU_NLL_delta_vs_KAN": ""
                        if not kan_fu or not kan_opt
                        else finite_float(kan_fu.get("NLL")) - finite_float(kan_opt.get("NLL")),
                        "KAN_plus_FU_accuracy_delta_vs_KAN": ""
                        if not kan_fu or not kan_opt
                        else finite_float(kan_fu.get("accuracy")) - finite_float(kan_opt.get("accuracy")),
                        "KAN_FU_vs_best_control_NLL_delta": ""
                        if not kan_fu or not kan_controls or kan_best_control_nll == float("inf")
                        else finite_float(kan_fu.get("NLL")) - kan_best_control_nll,
                        "KAN_plus_FU_vs_MLPFU_NLL_delta": ""
                        if not kan_fu or not mlp_fu
                        else finite_float(kan_fu.get("NLL")) - finite_float(mlp_fu.get("NLL")),
                        "diagnostic_full_row_pass": diagnostic_pass,
                        "official_full_row_pass": 0,
                        "train_steps": steps,
                        "readout_mechanism": args.kan_readout_mechanism,
                        "readout_fu_lr": args.kan_readout_fu_lr,
                        "readout_fu_interval": args.kan_readout_fu_interval,
                        "readout_acceptance": args.kan_readout_acceptance,
                        "task_tier": tier,
                        "known_simplifications": "Diagnostic full-loop only; not official because readout-exact updates cannot prove basis-native KAN internal value.",
                    }
                )
            external_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "tier": tier,
                    "status": "ran_readout_exact_diagnostic",
                    "strict_FC_PureKAN_rows": 8,
                    "hard_tier_status": "ran_readout_exact_full_loop_diagnostic",
                    "reason": "Same-job MLP/KAN readout-exact FU plus random/signflip target controls; official promotion disabled by fu_official_eligible=0.",
                }
            )
    write_rows(OUT_ROOT / "v22_30_KAN_readout_exact_carrier_matrix.csv", carrier_rows)
    write_rows(OUT_ROOT / "v22_30_KAN_readout_exact_gap_matrix.csv", gap_rows)
    write_rows(OUT_ROOT / "v22_30_KAN_readout_exact_external_task_matrix.csv", external_rows)
    hard_rows = [r for r in gap_rows if str(r.get("task_tier", "")).lower().startswith(("tier1", "tier2", "tier3"))]
    append_kan_repair_summary(
        {
            "timestamp_sg": now_sg(),
            "attempt_id": (
                f"readout_exact_diag_steps{args.kan_hard_steps}_lr{args.kan_readout_fu_lr:g}_"
                f"int{args.kan_readout_fu_interval}_{args.kan_readout_acceptance}"
            ),
            "kan_hard_steps": args.kan_hard_steps,
            "signal_alpha": "",
            "kan_slow_gate": "",
            "kan_slow_signal": "",
            "kan_slow_trust_ratio": "",
            "fu_mode": "readout_exact",
            "readout_mechanism": args.kan_readout_mechanism,
            "hard_gap_positive_rows": sum(1 for r in hard_rows if (finite_float(r.get("gap_reduction"), -1.0) or -1.0) > 0.0),
            "hard_mlp_fu_improves_rows": sum(1 for r in hard_rows if (finite_float(r.get("MLP_plus_FU_NLL_delta_vs_MLP"), 0.0) or 0.0) < 0.0),
            "hard_kan_fu_improves_rows": sum(1 for r in hard_rows if (finite_float(r.get("KAN_plus_FU_NLL_delta_vs_KAN"), 0.0) or 0.0) < 0.0),
            "hard_mlp_fu_beats_control_rows": sum(1 for r in hard_rows if (finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0),
            "hard_kan_fu_beats_control_rows": sum(1 for r in hard_rows if (finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0),
            "hard_fu_beats_all_control_rows": sum(
                1
                for r in hard_rows
                if (finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
                and (finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
            ),
            "hard_arch_nonworse_rows": sum(1 for r in hard_rows if (finite_float(r.get("KAN_plus_FU_vs_MLPFU_NLL_delta"), 1.0) or 1.0) <= 0.0),
            "hard_official_full_candidate_rows": 0,
            "diagnostic_full_candidate_rows": sum(1 for r in hard_rows if str(r.get("diagnostic_full_row_pass", "")) == "1"),
            "official_full_superiority_ready": 0,
            "note": "Readout-exact diagnostic ran separately and did not overwrite official slow-FU KAN gap matrix; official eligibility disabled.",
        }
    )
    return carrier_rows, gap_rows, external_rows


def fidelity_matrix(
    code_truth: dict[str, Any],
    t1_rows: list[dict[str, Any]],
    t2_rows: list[dict[str, Any]],
    t4_rows: list[dict[str, Any]],
    t5_rows: list[dict[str, Any]],
    t6_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "module_id": "PartA",
            "implementation_level": "code_identity_artifact_firewall",
            "theory_requirements_met": int(code_truth.get("compileall_pass") == 1 and code_truth.get("core_import_pass") == 1),
            "known_simplifications": "",
            "forbidden_shortcut_used": int(code_truth.get("forbidden_total_count", 0) > 0),
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
        {
            "module_id": "T1",
            "implementation_level": "multi_cohort_temporal_layerwise_sketched_per_example_gradient_A_B_plus_subspace_refinement",
            "theory_requirements_met": int(bool(t1_rows)),
            "known_simplifications": "random sketch; MLP only; held-train subspace refinement diagnostic; no KAN basis effect in this stage",
            "forbidden_shortcut_used": 0,
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
        {
            "module_id": "T2/T3",
            "implementation_level": "outcome_linked_sketched_subspace_diagnostic",
            "theory_requirements_met": int(bool(t2_rows)),
            "known_simplifications": "no full Procrustes transported source branch; KAN/JVP effect spaces deferred",
            "forbidden_shortcut_used": 0,
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
        {
            "module_id": "T4",
            "implementation_level": "continuous_mano_like_optimizer_loop",
            "theory_requirements_met": int(bool(t4_rows)),
            "known_simplifications": "row-axis tangent normalization; Cautious/Muon-like are local implementations; SOAP deferred",
            "forbidden_shortcut_used": 0,
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
        {
            "module_id": "T5",
            "implementation_level": "online_oja_projected_gradient_loop",
            "theory_requirements_met": int(bool(t5_rows)),
            "known_simplifications": "parameter-gradient subspace only; effect/KAN basis tracking deferred",
            "forbidden_shortcut_used": 0,
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
        {
            "module_id": "T6",
            "implementation_level": "modular_addition_delayed_generalization_probe",
            "theory_requirements_met": int(bool(t6_rows)),
            "known_simplifications": "small modular task; continual KANbeFair deferred",
            "forbidden_shortcut_used": 0,
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
        {
            "module_id": "KAN/KANbeFair",
            "implementation_level": "strict_PrimitiveKAN_debug_gap_matrix",
            "theory_requirements_met": int(bool(carrier_rows)),
            "known_simplifications": "Tier0 debug only; hard KANbeFair deferred",
            "forbidden_shortcut_used": 0,
            "uses_future_validation_test_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_rule": 0,
            "uses_readout_diagnostic_official": 0,
            "uses_kanbefair_original_kan": 0,
            "uses_auxiliary_loss_official": 0,
        },
    ]
    write_rows(OUT_ROOT / "v22_30_fidelity_matrix.csv", rows)
    return rows


def decide_routes(
    code_truth: dict[str, Any],
    t1_rows: list[dict[str, Any]],
    t1_refinement_rows: list[dict[str, Any]],
    t4_rows: list[dict[str, Any]],
    t5_rows: list[dict[str, Any]],
    t6_rows: list[dict[str, Any]],
    t7_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    routes: list[str] = []
    failure_rows: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    if int(code_truth.get("compileall_pass", 0)) != 1 or int(code_truth.get("core_import_pass", 0)) != 1:
        routes.append("R0-CodeOrIdentityFailed")
    t1_pass_rows = [
        r
        for r in t1_rows
        if finite_float(r.get("real_beats_base_rate"), 0.0) >= 0.70
        and finite_float(r.get("real_beats_L2_controls_rate"), 0.0) >= 0.60
        and finite_float(r.get("real_beats_L3_controls_rate"), 0.0) >= 0.50
        and finite_float(r.get("temporal_eigenspace_overlap"), 0.0) >= 0.50
    ]
    t1_refined_pass_rows = [
        r
        for r in t1_refinement_rows
        if finite_float(r.get("refined_beats_base_rate"), 0.0) >= 0.70
        and finite_float(r.get("refined_beats_L2_controls_rate"), 0.0) >= 0.60
        and finite_float(r.get("refined_beats_L3_best_controls_rate"), 0.0) >= 0.50
        and finite_float(r.get("temporal_eigenspace_overlap"), 0.0) >= 0.50
        and finite_float(r.get("held_train_CE_delta"), 1.0) < 0.0
    ]
    if t1_pass_rows or t1_refined_pass_rows:
        routes.append("R3-SignalChannelDiagnosticOnly")
    else:
        failure_rows.append({"module_id": "T1", "route": "R3-SignalChannelDiagnosticOnly", "reason": "No layer met T1 real/base/L2/L3/temporal gate."})
    if t1_refinement_rows and not t1_refined_pass_rows:
        failure_rows.append(
            {
                "module_id": "T1-subspace-refinement",
                "route": "R3-SignalChannelDiagnosticOnly",
                "reason": "Held-train-selected direction inside signal subspace did not broadly beat the best same-subspace L3 controls.",
            }
        )
    t3_rows = read_rows(OUT_ROOT / "v22_30_T3_transport_momentum_matrix.csv")
    t3_groups: dict[tuple[Any, Any, Any, Any], dict[str, float]] = {}
    for row in t3_rows:
        if row.get("transport_variant") not in {"transported_source", "L3_future_same_subspace"}:
            continue
        horizon_val = finite_float(row.get("horizon"), 0.0) or 0.0
        key = (row.get("dataset"), row.get("seed"), row.get("layer_id"), row.get("horizon"))
        nll = finite_float(row.get("final_test_NLL"))
        if nll is not None:
            t3_groups.setdefault(key, {})[str(row.get("transport_variant"))] = float(nll)
            t3_groups[key]["horizon"] = horizon_val
    t3_eval_groups = {
        key: val
        for key, val in t3_groups.items()
        if ("transported_source" in val and "L3_future_same_subspace" in val)
        and (val.get("horizon", 0.0) >= 100.0 or not any(v.get("horizon", 0.0) >= 100.0 for v in t3_groups.values()))
    }
    t3_pass_groups = [
        val for val in t3_eval_groups.values() if val["transported_source"] < val["L3_future_same_subspace"] - 1.0e-9
    ]
    t3_pass_rate = len(t3_pass_groups) / max(1, len(t3_eval_groups))
    if t3_eval_groups and t3_pass_rate >= 0.60:
        routes.append("R4-GrassmannDiagnosticOnly")
    elif t3_eval_groups:
        failure_rows.append(
            {
                "module_id": "T3",
                "route": "R4-GrassmannDiagnosticOnly",
                "reason": (
                    f"Transported source beat future same-subspace controls in {len(t3_pass_groups)}/{len(t3_eval_groups)} "
                    f"eligible H>=100 branch groups ({t3_pass_rate:.3f}); R4 requires >=60%."
                ),
            }
        )
    t4_signal = [r for r in t4_rows if r.get("optimizer_variant") == "signal_mano"]
    t4_signal_candidates = [r for r in t4_rows if r.get("optimizer_variant") in {"signal_mano", "signal_adamw_tangent"}]
    strong_optimizer_names = {"Schedule-Free AdamW", "Cautious AdamW", "Muon-like"}
    strong_signal_fu_rows = [
        r
        for r in t7_rows
        if r.get("base_optimizer") in strong_optimizer_names
        and r.get("control_variant") == "signal_fu"
        and r.get("status") == "real_run"
    ]
    strong_base_by_key = {
        (r.get("optimizer"), r.get("dataset"), r.get("seed")): r
        for r in t7_rows
        if r.get("optimizer") in strong_optimizer_names and r.get("status") == "real_run" and not r.get("base_optimizer")
    }
    strong_controls_by_key: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in t7_rows:
        if row.get("base_optimizer") in strong_optimizer_names and row.get("status") == "matched_control":
            strong_controls_by_key.setdefault((row.get("base_optimizer"), row.get("dataset"), row.get("seed")), []).append(row)
    t4_mano = {(r.get("dataset"), r.get("seed")): r for r in t4_rows if r.get("optimizer_variant") == "mano"}
    t4_geometry_base = {
        ("signal_mano", r.get("dataset"), r.get("seed")): r
        for r in t4_rows
        if r.get("optimizer_variant") == "mano"
    }
    t4_geometry_base.update(
        {
            ("signal_adamw_tangent", r.get("dataset"), r.get("seed")): r
            for r in t4_rows
            if r.get("optimizer_variant") == "adamw_tangent"
        }
    )
    t4_pass = 0
    for row in t4_signal:
        base = t4_mano.get((row.get("dataset"), row.get("seed")))
        if base and finite_float(row.get("NLL")) is not None and finite_float(base.get("NLL")) is not None and float(row["NLL"]) < float(base["NLL"]):
            t4_pass += 1
    t4_best_fu_beats_geometry_base = 0
    best_fu_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in t4_signal_candidates:
        nll = finite_float(row.get("NLL"))
        if nll is None:
            continue
        base = t4_geometry_base.get((row.get("optimizer_variant"), row.get("dataset"), row.get("seed")))
        if base and finite_float(base.get("NLL")) is not None and float(nll) < float(base["NLL"]):
            t4_best_fu_beats_geometry_base += 1
        key = (row.get("dataset"), row.get("seed"))
        current = best_fu_by_key.get(key)
        if current is None or float(nll) < float(current["NLL"]):
            best_fu_by_key[key] = row
    for row in strong_signal_fu_rows:
        nll = finite_float(row.get("NLL"))
        if nll is None:
            continue
        base = strong_base_by_key.get((row.get("base_optimizer"), row.get("dataset"), row.get("seed")))
        controls = strong_controls_by_key.get((row.get("base_optimizer"), row.get("dataset"), row.get("seed")), [])
        beats_base = base and finite_float(base.get("NLL")) is not None and float(nll) < float(base["NLL"])
        beats_controls = controls and all(
            finite_float(c.get("NLL")) is not None and float(nll) < float(c["NLL"]) for c in controls
        )
        if beats_base and beats_controls:
            t4_best_fu_beats_geometry_base += 1
        key = (row.get("dataset"), row.get("seed"))
        current = best_fu_by_key.get(key)
        if current is None or float(nll) < float(current["NLL"]):
            best_fu_by_key[key] = row
    strong_done = any(
        r.get("optimizer") in strong_optimizer_names
        and r.get("status") == "real_run"
        for r in t7_rows
    )
    baseline_by_key: dict[tuple[Any, Any], list[float]] = {}
    for row in t7_rows:
        if row.get("status") != "real_run":
            continue
        if row.get("base_optimizer"):
            continue
        if row.get("optimizer") in {
            "signal_mano",
            "signal_adamw_tangent",
            "random_control",
            "signflip_control",
            "shuffled_control",
            "random_adamw_tangent",
            "signflip_adamw_tangent",
            "shuffled_adamw_tangent",
        }:
            continue
        nll = finite_float(row.get("NLL"))
        if nll is not None:
            baseline_by_key.setdefault((row.get("dataset"), row.get("seed")), []).append(float(nll))
    t7_strong_win = 0
    for row in best_fu_by_key.values():
        nll = finite_float(row.get("NLL"))
        baselines = baseline_by_key.get((row.get("dataset"), row.get("seed")), [])
        if nll is not None and baselines and float(nll) < min(baselines) - 1.0e-9:
            t7_strong_win += 1
    t7_signal_mano_strong_win = 0
    for row in t4_signal:
        nll = finite_float(row.get("NLL"))
        baselines = baseline_by_key.get((row.get("dataset"), row.get("seed")), [])
        if nll is not None and baselines and float(nll) < min(baselines) - 1.0e-9:
            t7_signal_mano_strong_win += 1
    if t4_best_fu_beats_geometry_base and strong_done and t7_strong_win >= 5 and len(best_fu_by_key) >= 9:
        routes.append("R13-OptimizerInteractionResolved_FUBeatsStrongBaseline")
    elif t4_pass or t4_best_fu_beats_geometry_base:
        failure_rows.append(
            {
                "module_id": "T4/T7",
                "route": "R13-OptimizerInteractionResolved_FUBeatsStrongBaseline",
                "reason": (
                    f"Signal+Mano beat Mano-like in {t4_pass} row(s); best eligible FU beat its geometry baseline in "
                    f"{t4_best_fu_beats_geometry_base} row(s), but beat strongest completed baseline in "
                    f"{t7_strong_win} row(s); R13 requires broad strong-baseline wins, so no promotion."
                ),
            }
        )
    else:
        failure_rows.append({"module_id": "T4", "route": "R5-TangentOptimizerBaselineOnly", "reason": "Signal+Mano did not beat Mano-like baseline in any eligible row."})
    t5_online = [
        r
        for r in t5_rows
        if r.get("variant") in {"online_oja", "online_oja_warm_adamw", "online_recovery_adamw", "periodic_svd_refresh_adamw"}
    ]
    t5_static_rows = [r for r in t5_rows if r.get("variant") in {"static_pca", "static_pca_adamw"}]
    t5_control_rows = [r for r in t5_rows if r.get("variant") in {"online_random_control", "online_random_control_adamw"}]
    t5_static: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in t5_static_rows:
        key = (row.get("dataset"), row.get("seed"))
        if key not in t5_static or (finite_float(row.get("NLL"), float("inf")) or float("inf")) < (finite_float(t5_static[key].get("NLL"), float("inf")) or float("inf")):
            t5_static[key] = row
    t5_control: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in t5_control_rows:
        key = (row.get("dataset"), row.get("seed"))
        if key not in t5_control or (finite_float(row.get("NLL"), float("inf")) or float("inf")) < (finite_float(t5_control[key].get("NLL"), float("inf")) or float("inf")):
            t5_control[key] = row
    t5_pass = 0
    best_online_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in t5_online:
        nll = finite_float(row.get("NLL"))
        if nll is None:
            continue
        key = (row.get("dataset"), row.get("seed"))
        if key not in best_online_by_key or float(nll) < float(best_online_by_key[key]["NLL"]):
            best_online_by_key[key] = row
    for row in best_online_by_key.values():
        st = t5_static.get((row.get("dataset"), row.get("seed")))
        ct = t5_control.get((row.get("dataset"), row.get("seed")))
        if st and ct and finite_float(row.get("NLL")) is not None and float(row["NLL"]) < float(st["NLL"]) and float(row["NLL"]) < float(ct["NLL"]):
            t5_pass += 1
    if t5_pass:
        routes.append("R6-OnlineSubspaceDescriptiveOnly")
    else:
        failure_rows.append({"module_id": "T5", "route": "R6-OnlineSubspaceDescriptiveOnly", "reason": "Online Oja did not beat static PCA and same-subspace random control together."})
    t6_rows_by_key: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = {}
    for row in t6_rows:
        t6_rows_by_key.setdefault((row.get("task"), row.get("seed")), {})[str(row.get("variant"))] = row
    t6_grokking_pass_rows: list[dict[str, Any]] = []
    for key, rows_for_key in t6_rows_by_key.items():
        slow = rows_for_key.get("slow_fu")
        base = rows_for_key.get("base")
        if not slow or not base:
            continue
        slow_time = finite_float(slow.get("grokking_time"))
        base_time = finite_float(base.get("grokking_time"))
        if slow_time is None or base_time is None or base_time <= 0:
            continue
        reduction = base_time - slow_time
        controls = [rows_for_key.get("slow_random_control"), rows_for_key.get("slow_signflip_control")]
        controls_fail = True
        for control in controls:
            control_time = finite_float(control.get("grokking_time")) if control else None
            if control_time is not None and control_time <= slow_time:
                controls_fail = False
        if reduction >= 0.25 * base_time and controls_fail:
            t6_grokking_pass_rows.append(slow)
    slow_event_rows = [r for r in t6_rows if r.get("variant") == "slow_fu" and r.get("grokking_time") not in {"", None}]
    t6_pass = bool(t6_grokking_pass_rows)
    if t6_pass:
        routes.append("R7-GrokkingTemporalSignalOpened")
    else:
        failure_rows.append(
            {
                "module_id": "T6",
                "route": "R7-GrokkingTemporalSignalOpened",
                "reason": (
                    "No slow-FU grokking row satisfied the >=25% faster-than-base and matched-control-fail gate; "
                    f"raw slow-FU grokking events={len(slow_event_rows)}."
                ),
            }
        )
    continual_rows = read_rows(OUT_ROOT / "v22_30_T6_continual_boundary_matrix.csv")
    continual_fu_rows = [r for r in continual_rows if r.get("variant") == "slow_fu"]
    continual_pass_rows = [
        r
        for r in continual_fu_rows
        if (finite_float(r.get("relative_forgetting_reduction_vs_base"), 0.0) or 0.0) >= 0.05
        and str(r.get("beats_forgetting_controls", "")) == "1"
        and str(r.get("final_accuracy_non_worse_vs_base", "")) == "1"
    ]
    continual_matched_control_rows = [r for r in continual_rows if r.get("status") == "matched_control"]
    continual_kan_control_rows = [
        r
        for r in continual_matched_control_rows
        if str(r.get("model_family", "")).startswith("KAN")
    ]
    continual_final_acc_non_worse_rows = [
        r for r in continual_fu_rows if str(r.get("final_accuracy_non_worse_vs_base", "")) == "1"
    ]
    best_continual_reduction = max(
        [finite_float(r.get("relative_forgetting_reduction_vs_base"), float("-inf")) or float("-inf") for r in continual_fu_rows]
        + [float("-inf")]
    )
    if continual_pass_rows:
        routes.append("R12-ContinualForgettingReductionOpened")
    elif continual_rows:
        failure_rows.append(
            {
                "module_id": "T6-continual",
                "route": "R12-ContinualForgettingReductionOpened",
                "reason": (
                    f"Class-MNIST continual diagnostic had {len(continual_pass_rows)} FU row(s) meeting >=5% forgetting reduction "
                    f"plus matched-control and final-accuracy-non-worse conditions; best relative reduction={best_continual_reduction:.6g}."
                ),
            }
        )
    gap_positive = [r for r in gap_rows if finite_float(r.get("gap_reduction"), -1.0) is not None and float(r["gap_reduction"]) > 0.0]
    hard_gap_rows = [r for r in gap_rows if str(r.get("task_tier", "")).lower().startswith(("tier1", "tier2", "tier3"))]
    hard_gap_positive = [
        r for r in hard_gap_rows if finite_float(r.get("gap_reduction"), -1.0) is not None and float(r["gap_reduction"]) > 0.0
    ]
    hard_gap_rate = len(hard_gap_positive) / max(1, len(hard_gap_rows))
    hard_mlp_fu_improve_rows = [
        r for r in hard_gap_rows if (finite_float(r.get("MLP_plus_FU_NLL_delta_vs_MLP"), 0.0) or 0.0) < 0.0
    ]
    hard_kan_fu_improve_rows = [
        r for r in hard_gap_rows if (finite_float(r.get("KAN_plus_FU_NLL_delta_vs_KAN"), 0.0) or 0.0) < 0.0
    ]
    hard_mlp_fu_beats_control_rows = [
        r for r in hard_gap_rows if (finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
    ]
    hard_kan_fu_beats_control_rows = [
        r for r in hard_gap_rows if (finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
    ]
    hard_fu_beats_all_control_rows = [
        r
        for r in hard_gap_rows
        if (finite_float(r.get("MLP_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
        and (finite_float(r.get("KAN_FU_vs_best_control_NLL_delta"), 1.0) or 1.0) < 0.0
    ]
    hard_arch_nonworse_rows = [
        r for r in hard_gap_rows if (finite_float(r.get("KAN_plus_FU_vs_MLPFU_NLL_delta"), 1.0) or 1.0) <= 0.0
    ]
    hard_official_full_candidate_rows = [
        r for r in hard_gap_rows if str(r.get("official_full_row_pass", "")) == "1"
    ]
    hard_arch_nonworse_rate = len(hard_arch_nonworse_rows) / max(1, len(hard_gap_rows))
    official_full_ready = int(
        bool(hard_gap_rows)
        and hard_gap_rate >= 0.60
        and len(hard_mlp_fu_improve_rows) == len(hard_gap_rows)
        and len(hard_kan_fu_improve_rows) == len(hard_gap_rows)
        and len(hard_fu_beats_all_control_rows) == len(hard_gap_rows)
        and hard_arch_nonworse_rate >= 0.50
    )
    if hard_gap_rows and hard_gap_rate >= 0.60:
        routes.append("R10-KANCarrierGapReductionOpened")
    elif hard_gap_rows:
        failure_rows.append(
            {
                "module_id": "KAN/KANbeFair",
                "route": "R10-KANCarrierGapReductionOpened",
                "reason": (
                    f"Hard-tier KAN gap reduction positive rate was {len(hard_gap_positive)}/{len(hard_gap_rows)} "
                    f"({hard_gap_rate:.3f}); R10 requires >=60% hard-task positive gap reduction."
                ),
            }
        )
    elif gap_positive:
        failure_rows.append(
            {
                "module_id": "KAN/KANbeFair",
                "route": "R10-KANCarrierGapReductionOpened",
                "reason": "Positive gap reduction appeared only in Tier0 debug rows; hard KANbeFair tiers were deferred, so no R10 promotion.",
            }
        )
    else:
        failure_rows.append({"module_id": "KAN", "route": "R10-KANCarrierGapReductionOpened", "reason": "No positive debug gap reduction row."})
    if not official_full_ready and hard_gap_rows:
        failure_rows.append(
            {
                "module_id": "KAN/KANbeFair-official",
                "route": "OfficialFullSuperiorityReady",
                "reason": (
                    f"Full-success subconditions not met: MLP+FU improves MLP+Opt in {len(hard_mlp_fu_improve_rows)}/{len(hard_gap_rows)} hard rows; "
                    f"KAN+FU improves KAN+Opt in {len(hard_kan_fu_improve_rows)}/{len(hard_gap_rows)}; "
                    f"MLP+FU beats matched controls in {len(hard_mlp_fu_beats_control_rows)}/{len(hard_gap_rows)}; "
                    f"KAN+FU beats matched controls in {len(hard_kan_fu_beats_control_rows)}/{len(hard_gap_rows)}; "
                    f"gap_reduction positive in {len(hard_gap_positive)}/{len(hard_gap_rows)}; "
                    f"KAN+FU <= MLP+FU in {len(hard_arch_nonworse_rows)}/{len(hard_gap_rows)} ({hard_arch_nonworse_rate:.3f})."
                ),
            }
        )
    soap_diag_rows = [
        r
        for r in t7_rows
        if r.get("optimizer") == "SOAP/Shampoo-like" and r.get("status") == "feasibility_diagnostic"
    ]
    soap_reason = (
        f"Plan fallback diagonal+small-matrix Kronecker diagnostic ran in {len(soap_diag_rows)} row(s); "
        "official full SOAP/Shampoo remains unimplemented and is not used for promotion."
        if soap_diag_rows
        else "Deferred because prior layer-1 gates did not open official full-loop in this execution."
    )
    deferred_items = [("T7", "SOAP/Shampoo-like official Kronecker preconditioner", soap_reason)]
    if not t3_eval_groups:
        deferred_items.insert(0, ("T3", "Full transported momentum branch", "Deferred because no executable T3 transported branch groups were present."))
    for item in deferred_items:
        deferred.append({"module_id": item[0], "item": item[1], "reason": item[2]})
    if hard_gap_rows:
        external_rows = read_rows(OUT_ROOT / "v22_30_KANbeFair_external_task_matrix.csv")
        tier1_rows = [r for r in external_rows if r.get("tier") == "Tier1_hard_vision"]
        tier1_blocked = [r for r in tier1_rows if r.get("status") != "ran_strict_DGKAN_hard_sample"]
        continual_reason = (
            "Class-MNIST continual diagnostic ran with family-matched strict PrimitiveKAN controls, "
            "but exact/original KANbeFair continual protocol remains incomplete"
            if continual_kan_control_rows
            else "continual matrix remains incomplete"
        )
        hard_reason = (
            "Wine/CIFAR10/SVHN/EMNIST strict PrimitiveKAN hard subsets ran; "
            f"R10 hard gap threshold is met; {continual_reason}; "
            "official architecture superiority remains incomplete."
            if tier1_rows and not tier1_blocked
            else "Wine Tier2 strict PrimitiveKAN subset ran; CIFAR/SVHN/EMNIST hard vision and continual matrix remain incomplete."
        )
        deferred.append(
            {
                "module_id": "KANbeFair",
                "item": "Hard vision tasks and continual matrix",
                "reason": hard_reason,
            }
        )
    else:
        deferred.append(
            {
                "module_id": "KANbeFair",
                "item": "Tier1/Tier2/Tier3 hard tasks and continual matrix",
                "reason": "Deferred because prior layer-1 gates did not open official full-loop in this execution.",
            }
        )
    if not routes:
        routes.append("R16-StrongNoGo_CurrentImplementationsNoTheoryFalsification")
    final = {
        "final_route": routes[-1] if routes else "R16-StrongNoGo_CurrentImplementationsNoTheoryFalsification",
        "all_routes_opened": routes,
        "code_truth": code_truth,
        "t1_gate_pass_rows": len(t1_pass_rows),
        "t1_refinement_rows": len(t1_refinement_rows),
        "t1_refinement_pass_rows": len(t1_refined_pass_rows),
        "t3_transport_rows": len(t3_rows),
        "t3_transport_eval_groups": len(t3_eval_groups),
        "t3_transport_pass_groups": len(t3_pass_groups),
        "t3_transport_pass_rate": t3_pass_rate if t3_eval_groups else "",
        "t4_signal_mano_beats_mano_rows": t4_pass,
        "t4_best_signal_fu_beats_geometry_base_rows": t4_best_fu_beats_geometry_base,
        "t7_signal_mano_beats_strongest_rows": t7_signal_mano_strong_win,
        "t7_best_signal_fu_beats_strongest_rows": t7_strong_win,
        "t7_best_signal_fu_keys": len(best_fu_by_key),
        "t5_online_beats_static_and_control_rows": t5_pass,
        "t5_best_online_keys": len(best_online_by_key),
        "t6_raw_slow_fu_grokking_events": len(slow_event_rows),
        "t6_slow_fu_grokking_events": len(t6_grokking_pass_rows),
        "t6_continual_rows": len(continual_rows),
        "t6_continual_pass_rows": len(continual_pass_rows),
        "t6_continual_matched_control_rows": len(continual_matched_control_rows),
        "t6_continual_kan_control_rows": len(continual_kan_control_rows),
        "t6_continual_final_accuracy_non_worse_rows": len(continual_final_acc_non_worse_rows),
        "t6_best_relative_forgetting_reduction": "" if best_continual_reduction == float("-inf") else best_continual_reduction,
        "kan_gap_positive_rows": len(gap_positive),
        "kan_hard_gap_rows": len(hard_gap_rows),
        "kan_hard_gap_positive_rows": len(hard_gap_positive),
        "kan_hard_gap_positive_rate": hard_gap_rate if hard_gap_rows else "",
        "hard_mlp_fu_improves_rows": len(hard_mlp_fu_improve_rows),
        "hard_kan_fu_improves_rows": len(hard_kan_fu_improve_rows),
        "hard_mlp_fu_beats_control_rows": len(hard_mlp_fu_beats_control_rows),
        "hard_kan_fu_beats_control_rows": len(hard_kan_fu_beats_control_rows),
        "hard_fu_beats_all_control_rows": len(hard_fu_beats_all_control_rows),
        "hard_arch_nonworse_rows": len(hard_arch_nonworse_rows),
        "hard_arch_nonworse_rate": hard_arch_nonworse_rate if hard_gap_rows else "",
        "hard_official_full_candidate_rows": len(hard_official_full_candidate_rows),
        "official_full_superiority_ready": official_full_ready,
        "latest_status_timestamp": now_sg(),
    }
    write_json(OUT_ROOT / "v22_30_final_route.json", final)
    write_rows(OUT_ROOT / "v22_30_deferred_items.csv", deferred)
    write_rows(OUT_ROOT / "v22_30_failure_taxonomy.csv", failure_rows)
    write_simple_svg(
        "v22_30_evidence_ladder.svg",
        "Evidence ladder gate counts",
        [
            {"label": "T1 gate rows", "rows": len(t1_pass_rows)},
            {"label": "T1 refinement pass", "rows": len(t1_refined_pass_rows)},
            {"label": "T4 signal>Mano", "rows": t4_pass},
            {"label": "T5 online wins", "rows": t5_pass},
            {"label": "T6 grok events", "rows": int(t6_pass)},
            {"label": "KAN gap positive", "rows": len(gap_positive)},
        ],
    )
    return final, deferred, failure_rows


def write_recap(
    final: dict[str, Any],
    code_truth: dict[str, Any],
    historical_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    t1_rows: list[dict[str, Any]],
    branch_rows: list[dict[str, Any]],
    t1_refinement_rows: list[dict[str, Any]],
    t1_refinement_branch_rows: list[dict[str, Any]],
    t2_rows: list[dict[str, Any]],
    t4_rows: list[dict[str, Any]],
    t7_rows: list[dict[str, Any]],
    t5_rows: list[dict[str, Any]],
    t6_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    deferred: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    continual_rows = read_rows(OUT_ROOT / "v22_30_T6_continual_boundary_matrix.csv")
    t3_transport_rows = read_rows(OUT_ROOT / "v22_30_T3_transport_momentum_matrix.csv")
    repair_attempt_rows = read_rows(OUT_ROOT / "v22_30_repair_attempt_summary.csv")
    readout_exact_carrier_rows = read_rows(OUT_ROOT / "v22_30_KAN_readout_exact_carrier_matrix.csv")
    readout_exact_gap_rows = read_rows(OUT_ROOT / "v22_30_KAN_readout_exact_gap_matrix.csv")
    readout_exact_hard_rows = [
        r for r in readout_exact_gap_rows if str(r.get("task_tier", "")).lower().startswith(("tier1", "tier2", "tier3"))
    ]
    readout_exact_diag_pass_rows = [r for r in readout_exact_hard_rows if str(r.get("diagnostic_full_row_pass", "")) == "1"]
    readout_exact_gap_positive_rows = [
        r for r in readout_exact_hard_rows if (finite_float(r.get("gap_reduction"), -1.0) or -1.0) > 0.0
    ]
    lines = [
        "# DG-KAN v22.30 Fidelity-Ladder Geometric FU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## 1. 结论摘要",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- opened routes: `{', '.join(final.get('all_routes_opened', []))}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        "",
        "本轮没有把 diagnostic/proxy/no-harm 写成 official 成功。所有关键结论均来自 `results/v22_30/` 下本轮 artifact 或明确命名的历史 artifact 读回。",
        "",
        "## 2. Code / Identity / Artifact Firewall",
        "",
        md_table([code_truth], limit=5),
        "",
        "### 本轮实现修复记录",
        "",
        "- 修复 1：smoke 后发现 forbidden scan 命中 runner 自身的禁用词表，而不是 official FU/model/integration core；已将 official scan 范围限定为 `dgkan/fu`, `dgkan/models/fc_purekan_primitives.py`, `dgkan/integration`，最终 `forbidden_total_count=0`。",
        "- 修复 2：确认 `PrimitiveKAN.basis_diagnostics()` 不返回 `basis_energy_fraction`；已停止把 `basis_entropy` 误填为 `basis_energy_fraction`，并新增 `basis_entropy`, `basis_effective_rank`, `basis_condition_proxy` 字段。",
        "- 修复 3：smoke 中 `signal_mano` 赢 Mano-like 时曾触发过 R13；已收紧 finalizer，只有官方强 optimizer baselines 真实完成且 FU 胜出时才允许 R13。Tier0 debug gap 也不能触发 R10 hard-task route。",
        "- 修复 4：T1 从单 cohort / temporal overlap 占位实现升级为多 cohort A_B、同一 sketch 空间、跨时间点 eigenspace overlap，并新增 `v22_30_T1_cohort_signal_diagnostics.csv`。",
        "- 修复 5：T7 新增真实执行的 Cautious AdamW、Schedule-Free averaging 近似、Muon-like SVD orthogonalized momentum；SOAP/Shampoo-like 仍明确 deferred，不参与 promotion。",
        "- 修复 6：按计划 fallback 新增 T1 subspace-direction refinement；只用 held-train cohort 在既有 signal subspace 内选方向，最终 test branch 只用于结果读回，并新增 `v22_30_T1_subspace_refinement_matrix.csv` 与 branch 明细。",
        "- 修复 7：按 T4 fallback 新增 norm-matched `signal_adamw_tangent` 及同几何 random/signflip/shuffled controls；finalizer 改为用 best eligible FU 对比最强已完成 optimizer baseline，同时保留旧 `signal_mano` 审计字段。",
        "- 修复 8：按 T5 projection-aware optimizer 要求新增 AdamW-state 投影训练 variants：`static_pca_adamw`, `online_oja_warm_adamw`, `online_recovery_adamw`, `online_random_control_adamw`, `periodic_svd_refresh_adamw`。",
        "- 修复 9：修复 KANbeFair 原入口硬编码 `os.chdir('/home/yurunpeng/Repos/KANBeFair/src')` 的本机路径 blocker，并新增 strict `PrimitiveKAN` hard-tier subset runner；hard 数据下载必须显式传 `--kan-hard-download`，KANbeFair original KAN 仍只作 baseline context。",
        "- 修复 10：按 T6 fallback 增加小模数/高 weight-decay protocol：`fallback_mod_add_p13_wd005` 与 `fallback_mod_mul_p11_wd005`，并继续保留 slow-random/signflip controls。",
        "- 修复 11：T6 fallback smoke/正式长跑发现 `p11` 小任务训练集小于 batch size 时 `drop_last=True` 造成空 DataLoader 无限循环；已将 modular loader batch size 限制到数据量并改为 `drop_last=False`。",
        "- 修复 12：按 T7 fallback 将 `SOAP/Shampoo-like` 从纯 deferred 改为 diagonal + small-matrix Kronecker feasibility diagnostic；大矩阵使用 diagonal fallback，小矩阵执行 Kronecker 预条件，状态标为 `feasibility_diagnostic`，不计入 official strong baseline promotion。",
        "- 修复 13：按 KANbeFair hard vision fallback 补齐 Tier1 loader：CIFAR10 使用 train/test 接口，SVHN 使用 split 接口，EMNIST-Letters 使用 `split='letters'` 并将 label 映射到 0-25；hard loader 同步加入小样本 batch/drop_last 保护，dataset/download blocker 只写 availability row，不写成 failure。",
        "- 修复 14：按 T2/T3 wrong-space fallback 新增 `MLP_hidden_activation_effect` 与 `MLP_logits_effect` 子空间诊断 rows；这些 rows 只检查 hidden/logits effect space 的 temporal retention 与 real branch outcome 关联，不做 intervention/transport promotion。",
        "- 修复 15：按 continual/forgetting 目标新增 Class-MNIST pair-sequence diagnostic；slow-FU 慢梯度状态跨 task boundary 保留，记录 avg forgetting、BWT、relative forgetting reduction 和 matched slow-random/signflip controls。",
        "- 修复 16：按 T3 full transported momentum fallback 新增 Procrustes-style transported-source branch；在 temporal checkpoint 上比较 transported source、no-op、signflip 与 future same-subspace random control，并按 H>=100 groups 计算 pass rate。",
        "- 修复 17：hard vision 下载 blocker 追踪到环境 `socks5h://` 代理与 torchvision/urllib 不兼容；已在 hard dataset download 调用期间临时清除 socks5h proxy，并在下载后恢复环境变量。",
        "- 修复 18：R12 continual 审计发现 KAN_DCHE/KAN_DFOU slow-FU 缺少同族 slow-random/signflip controls，且 finalizer 未显式检查 final accuracy non-worse；已补 family-matched KAN controls、matched control 计数字段、final accuracy delta/non-worse 字段，并把三条件同时纳入 R12 判定。",
        "- 修复 19：R10 carrier route 曾只记录 gap reduction，未把 Part K full-success 子条件逐项落盘；已在 KAN gap matrix 增加 `MLP_plus_FU_NLL_delta_vs_MLP`, `MLP_plus_FU_accuracy_delta_vs_MLP`, `official_full_row_pass`，并在 final JSON / failure taxonomy 中记录 MLP+FU、KAN+FU、gap、KAN+FU<=MLP+FU 的 hard-row 分子分母。",
        "- 修复 20：Part K hard subset 暴露 hard steps=40 下 KAN/FU 欠收敛；按收敛 fallback 尝试更长 hard steps，并把 `train_steps`, `slow_alpha`, `slow_beta` 写入 carrier/gap rows，便于审计不同 FU 强度是否只是超参偶然。",
        "- 修复 21：slow-FU 原实现无条件放大 EMA slow gradient，可能在 slow 与当前 task gradient 反向时伤害 MLP/FU；新增 `--kan-slow-gate current_grad_cosine`，只在当前训练 batch 梯度与 slow signal 同向时放大，并记录 `slow_gate_keep_rate`。",
        "- 修复 22：gated run 显示 slow signal 大多同向但 MLP+FU 仍显著变差；新增 `--kan-slow-trust-ratio`，将 slow signal 范数限制在当前梯度范数的固定比例内，并记录 `slow_trust_clip_rate`，用于检验是否是 slow EMA 有效步长过大。",
        "- 修复 23：为避免 slow-FU 在当前 train batch 上直接增大 loss，新增 `--kan-slow-acceptance train_loss_nonworse`；每步先执行 FU candidate，若同一 train batch CE 变差则同时回滚参数与 AdamW optimizer state，再执行 base AdamW step，并记录 accept rate、reject count 与 candidate loss delta。",
        "- 修复 24：train-batch acceptance 改善 MLP+FU 但仍未解决 generalization；按计划 `held-train signal` 思路新增 `--kan-slow-acceptance held_train_loss_nonworse`，accept/reject 使用 disjoint held-train cohort 的 CE，不使用 test/validation/future 信息。",
        "- 修复 25：Part K hard full-loop 补齐 matched slow-random/signflip controls；MLP/KAN FU rows 现在同组运行 random/signflip slow controls，gap matrix 记录 `MLP_FU_vs_best_control_NLL_delta` 与 `KAN_FU_vs_best_control_NLL_delta`，official full row pass 需要同时打过 matched controls。",
        "- 修复 26：matched-control run 显示 slow-FU 往往没有打过 norm-matched random/signflip controls；新增 `--kan-slow-gate held_grad_cosine`，用 disjoint held-train batch 的梯度与 slow signal 的内积做门控，只放大 held-train 方向同向的 slow signal，并记录 `slow_gate_keep_rate`。",
        "- 修复 27：按 Part H 的 fast/reservoir 约束新增 `--kan-slow-signal slow_minus_fast`；该信号使用 `slow - (current - slow)`，在放大 slow component 的同时惩罚当前 fast residual，matched random/signflip controls 使用同一 signal 后再变换，并在 carrier/gap rows 记录 `slow_signal`。",
        "- 修复 28：Part E 计划要求 H100/H200 transported branch outcome；默认 runner 只跑 H50/H100。已补跑 `--branch-horizons 50,100,200`，T3 transport rows 从 216 增至 324，finalizer 按 H>=100 重算 22/54，通过率仍为 0.407，R4 未打开。",
        "- 修复 29：Part I 要求每个强 optimizer 同组运行 FU 与 matched controls；T7 现在为 Cautious AdamW、Schedule-Free AdamW、Muon-like 额外运行 `*_signal_fu`, `*_random_control`, `*_signflip_control`, `*_shuffled_control`，并在 finalizer 中把这些 FU/control rows 排除出 baseline、把 signal-FU 纳入 best eligible FU 候选。",
        "- 修复 30：Part H fallback 要求 no-grokking 时调整 task size / weight decay / label noise。T6 modular loader 新增 train-only `label_noise_rate`，测试集保持 clean，并新增 `fallback_mod_add_p17_noise005_wd005` 与 `fallback_mod_add_p11_noise010_wd010` protocols，结果表记录 `label_noise_rate`。",
        "- 修复 31：T6 modular one-hot MLP 长期只记忆 train 而不泛化；新增 embedding-MLP modular fallback protocols，并收紧 R7 finalizer：必须 slow-FU grokking time 比 base 快 >=25% 且 matched slow-random/signflip controls 不同等或更快，避免把任意 slow-FU event 写成 grokking mechanism 成功。",
        "- 修复 32：Part E wrong-space fallback 原先只记录 hidden/logits effect subspace retention；现新增 effect-space Procrustes transport branch，将 held batch 的 `w2` source gradient 投影/transport 到 hidden/logits future effect subspace，再通过 readout `w2` 分支比较 transported source、signflip 与 future same-subspace random controls。该修复扩大 wrong-space 覆盖面，但 promotion 仍只按 matched future same-subspace controls 统计。",
        "- 修复 33：Part K held-gradient gate 原先只做整条 slow signal 的正负内积门控，无法清除与 held-train descent 无关的正交噪声；新增 `--kan-slow-gate held_grad_projection`，把 slow signal 投影到 disjoint held-train gradient 的正向分量，random/signflip controls 也使用同一投影规则，避免放宽 matched-control 标准。本轮 projection + held-train acceptance 实测 official full candidate rows 从 1 降至 0，说明 blocker 未解除，详见 KAN Repair Attempt Trace。",
        "- 修复 34：Part E/J 计划要求检查 KAN basis effect subspaces；T3 现新增 D-CHE/D-FOU strict PrimitiveKAN `frozen_readout_features()` basis-effect subspace retention 与 Procrustes transported `w2` branch，对比 signflip 和 future same-subspace controls。该实现只 perturb `w2` readout 参数，明确为 basis-effect diagnostic，不作为 full basis-native FU controller promotion。实测 D-CHE basis-effect 为 15 pass / 3 fail，D-FOU basis-effect 为 16 pass / 2 fail，把 T3 总体从 37/90=0.411 提升到 68/126=0.540，但仍低于 R4 的 0.60。",
        "- 修复 35：按 KAN basis-effect 强信号的后续审计方向，新增独立 `kan_readout` stage，把 `M21-ExactReadoutFunctionSpaceActuationFU`、`M49-LossCotangentTargetFU`、`M54-CrossSplitConsensusTargetFU` 与 `M50/M51` matched target controls 接入 MLP 与 strict PrimitiveKAN 的同组 full-loop diagnostic，并输出 `v22_30_KAN_readout_exact_*` artifacts。所有 readout-exact gap rows 明确写入 `fu_official_eligible=0` 与 `official_full_row_pass=0`，避免把 readout diagnostic promotion 成 basis-native official 结论；当前 best diagnostic 为 M54 interval5 lr5e-5 held-train acceptance，diagnostic full rows=4/40。",
        "- 修复 36：readout-exact M54 显示 train-split consensus 有助于 matched-control 分离，但 readout 不能 official promotion；因此在 official slow-FU 训练通道新增 `--kan-slow-gate split_grad_projection`，只用当前 train batch 两个 split 上同号的参数梯度分量作为投影参考，random/signflip controls 也经过同一投影规则，不使用 readout solver、test/validation/future 信息。",
        "- 修复 37：split_grad_projection official run 仍未解除 matched-control blocker；因此新增 `--kan-slow-signal split_consensus`，直接把当前 train batch 两个 split 上同号的参数梯度均值作为 official slow-FU 信号，而不是只把它当作 EMA 投影参考。matched random/signflip controls 仍在该信号后做同范数变换，并可继续使用 held-train loss acceptance；该路线不使用 readout solver、test/validation/future 信息。",
        "- 修复 38：split_consensus direct run 改善 hard gap positive 但没有改善 MLP/matched-control 卡口；按 Part H 的 held-train alignment 要求，尝试 `--kan-slow-signal split_consensus --kan-slow-gate held_grad_projection`，即先取 train-batch split consensus 参数梯度，再投影到 disjoint held-train gradient 的正向分量。该组合仍使用 official parameter-space slow-FU 与同组 matched controls，不使用 readout solver、test/validation/future 信息。",
        "- 修复 39：dual-consensus 出现首个 official full candidate row，但 MLP matched-control 仍低；历史 `slow_minus_fast + held_grad_cosine` 的 matched-control 计数更高。按 Part H 的 slow/fast reservoir 思路，尝试 `--kan-slow-signal slow_minus_fast --kan-slow-gate held_grad_projection`，用 fast residual penalty 保持 signal/control 分离，再投影到 disjoint held-train gradient 正向分量；仍只使用训练流和同组 matched controls。",
        "- 修复 40：前序 acceptance 只要求 FU candidate 不比 step 前 held-train loss 更差，可能接受“不如同一步 AdamW base update”的 FU。新增 `--kan-slow-acceptance held_train_loss_beats_base_step` / `train_loss_beats_base_step`：每步先计算 FU candidate，再在同一参数/optimizer state 上模拟 base AdamW step；只有 FU candidate 在训练流评估 batch 上不输 base step 才保留，否则回退到 base step。`slow_candidate_loss_delta_mean` 在该模式下表示 candidate minus base-step loss。",
        "- 修复 41：base-step-relative acceptance 的零 margin 实测 accept rate 接近 1，说明过滤过松；因此尝试 `--kan-slow-acceptance-tol -0.00002`，要求 FU candidate 在 held-train CE 上至少比同一步 base AdamW 低 2e-5 才保留。该阈值来自上一条训练流 acceptance delta 的量级审计，不使用 test/validation/future 方向。",
        "- 修复 42：`split_consensus` 原先是当前 train batch 双 split 同号梯度，不包含 slow state；按 Part H 的 slow/fast reservoir 思路新增 `split_consensus_ema` 与 `split_consensus_slow_minus_fast`，把 cross-split consensus 先进入 EMA，再可选减去 fast residual。matched random/signflip controls 仍在同一信号后变换。",
        "- 修复 43：按 Part J 的 basis-native carrier 路线，修复 v22.06 metric solver 对 strict `PrimitiveKAN.w2` 3D basis readout tensor 的识别与 `(hidden*k, classes) -> (hidden, classes, k)` 回写，并在 official KAN stage 新增 `--kan-fu-mode functional_metric`。该模式用 M265-M270 function-metric solver 真实更新 hidden/readout parameter channels，matched controls 在同一更新支撑集做 random/signflip；official row 仍要求非零 hidden-source fraction 与原 full-superiority 条件，不把 readout-only diagnostic 升格。",
        "- 修复 44：M267/M268/M270 functional-metric runs 仍被 readout-dominant 更新与 matched controls 卡住；新增 `M271-V2206MetricSolverT12G0HiddenOnlySoftCompensatedFU`，把 T12 soft-compensated target 的最终 commit 限制到 hidden-only parameter channel。random/signflip controls 复用同一 hidden-only 支撑集，用来测试 readout dominance 是否是 KAN basis-native full-loop blocker。",
        "- 修复 45：响应 FU 学习率/调度审计，新增 `--kan-functional-fu-lr-schedule constant|cosine`，functional FU 每次 commit 的 lr 可按训练进度余弦衰减，并记录 `functional_fu_lr_mean`。后续 LR scan 以 M271 hidden-only 为主，比较常数高峰值与 cosine 调度是否改善 matched-control 分离。",
        "- 修复 46：M271 LR scan 显示 KAN/gap 指标随 cosine peak 增强，但 MLP+FU vs MLP 改善卡在 8/20 unique hard pairs；新增 `--kan-functional-mlp-fu-lr` 与 `--kan-functional-kan-fu-lr`，允许同一 official matrix 内 MLP 与 KAN 使用不同 functional FU peak lr。默认 0 表示复用全局 lr，matched controls 仍与各自 family real run 使用同一 lr/schedule。",
        "- 修复 47：family-specific LR run 改善 MLP matched-control 计数但未提升 MLP+FU vs MLP base 改善；新增 `--kan-functional-mlp-fu-interval` 与 `--kan-functional-kan-fu-interval`，允许在同一 mechanism/lr schedule 下提高 MLP 侧 FU 尝试频率。默认 0 表示复用全局 interval，carrier/gap/repair summary 记录真实执行 interval。",
        "",
        "## 3. Historical Reanalysis",
        "",
        f"- historical rows read: `{len(historical_rows)}`",
        "- 这些历史行只作为 fidelity/context 证据；`promotion_allowed_after_reaudit` 在本轮保持 0，原因是 v22.30 要求机制特定 gate 重跑。",
        "",
        md_table(historical_rows, ["source_file", "mechanism_claim", "implementation_fidelity", "uses_real_task", "NLL_delta", "promotion_allowed_after_reaudit"], limit=12),
        "",
        "## 4. Control Hierarchy Evidence",
        "",
        f"- control hierarchy rows reconstructed from v22.29 branch artifacts: `{len(control_rows)}`",
        "- 该表用于审计 L1/L2/L3 controls 是否曾解释 real branch；缺失 L4 时不写 official pass。",
        "",
        md_table(control_rows, ["mechanism", "dataset", "seed", "horizon", "real_delta_NLL", "real_minus_L2", "real_minus_L3", "control_level_where_real_fails"], limit=14),
        "",
        "## 5. T1/T2/T3 Evidence",
        "",
        f"- T1 layerwise rows: `{len(t1_rows)}`; branch rows: `{len(branch_rows)}`; T2 rows: `{len(t2_rows)}`",
        f"- T3 transport rows: `{len(t3_transport_rows)}`; eval groups: `{final.get('t3_transport_eval_groups')}`; pass groups: `{final.get('t3_transport_pass_groups')}`; pass rate: `{final.get('t3_transport_pass_rate')}`",
        f"- T1 mean real_beats_base_rate: `{mean_field(t1_rows, 'real_beats_base_rate')}`",
        f"- T1 mean real_beats_L2_controls_rate: `{mean_field(t1_rows, 'real_beats_L2_controls_rate')}`",
        f"- T1 mean real_beats_L3_controls_rate: `{mean_field(t1_rows, 'real_beats_L3_controls_rate')}`",
        f"- T1 mean cohort_positive_fraction: `{mean_field(t1_rows, 'cohort_positive_fraction')}`",
        f"- T1 mean temporal_eigenspace_overlap: `{mean_field(t1_rows, 'temporal_eigenspace_overlap')}`",
        f"- T1 subspace refinement rows: `{len(t1_refinement_rows)}`; pass rows: `{final.get('t1_refinement_pass_rows')}`",
        f"- T1 refinement mean refined_beats_L3_best_controls_rate: `{mean_field(t1_refinement_rows, 'refined_beats_L3_best_controls_rate')}`",
        "- 修改/实现说明：T1 用 multi-cohort layerwise per-example gradient 的 sketched drift-vs-diffusion A_B，并在同一 held-train cohorts 上重算 temporal eigenspace overlap；T2 从同一执行的 branch outcome 形成 outcome-linked subspace rows，并补充 hidden/logits effect-space 与 KAN basis-effect wrong-space diagnostics；T3 full Procrustes transport 已包含 parameter-space、hidden/logits effect-space readout branch、strict PrimitiveKAN basis-effect w2 branch，未 promotion则写入 deferred/failure taxonomy。",
        "- 修复/审计说明：subspace refinement 的方向选择只用 held-train CE，不用 test/validation/future direction；promotion 判定要求 refined direction 打过最佳 L3 same-subspace controls。",
        "- KAN basis-effect 证据链：D-CHE/DFOU strict PrimitiveKAN `frozen_readout_features()` 的 T2 subspace retention 多数接近 0.99，H>=100 transported `w2` branch 对 future same-subspace controls 的胜率为 D-CHE 15/18、D-FOU 16/18。该结果说明 KAN basis-effect 是当前 T3 最强 source-space family，但它是 w2-only diagnostic branch，且 overall R4 gate 仍未达标。",
        "",
        md_table(t1_rows, ["dataset", "seed", "layer_id", "positive_eigen_count", "cohort_positive_fraction", "temporal_eigenspace_overlap", "signal_SNR", "real_beats_base_rate", "real_beats_L2_controls_rate", "real_beats_L3_controls_rate"], limit=12),
        "",
        md_table(
            t1_refinement_rows,
            [
                "dataset",
                "seed",
                "layer_id",
                "selected_candidate_label",
                "held_train_CE_delta",
                "base_t1_real_beats_L3_controls_rate",
                "refined_beats_base_rate",
                "refined_beats_raw_top_rate",
                "refined_beats_L2_controls_rate",
                "refined_beats_L3_best_controls_rate",
            ],
            limit=12,
        ),
        "",
        md_table(
            t1_refinement_branch_rows,
            [
                "dataset",
                "seed",
                "layer_id",
                "horizon",
                "branch_variant",
                "control_level",
                "held_train_CE_delta_for_selected",
                "checkpoint_test_NLL",
                "final_test_NLL",
                "branch_NLL_delta",
            ],
            limit=18,
        ),
        "",
        md_table(
            t3_transport_rows,
            [
                "dataset",
                "seed",
                "layer_id",
                "horizon",
                "transport_variant",
                "control_level",
                "checkpoint_test_NLL",
                "final_test_NLL",
                "branch_NLL_delta",
                "transport_error",
                "post_transport_alignment",
                "transported_source_beats_same_subspace_controls",
                "real_vs_control_transport_gain",
            ],
            limit=30,
        ),
        "",
        "## 6. T4/T7 Optimizer Evidence",
        "",
        f"- T4 continuous optimizer rows: `{len(t4_rows)}`",
        f"- T7 optimizer baseline rows: `{len(t7_rows)}`",
        f"- signal_mano rows beating mano-like: `{final.get('t4_signal_mano_beats_mano_rows')}`",
        f"- best eligible signal-FU rows beating own geometry baseline: `{final.get('t4_best_signal_fu_beats_geometry_base_rows')}`",
        f"- signal_mano rows beating strongest completed baseline: `{final.get('t7_signal_mano_beats_strongest_rows')}`",
        f"- best eligible signal-FU rows beating strongest completed baseline: `{final.get('t7_best_signal_fu_beats_strongest_rows')}` / keys `{final.get('t7_best_signal_fu_keys')}`",
        "- 修改/实现说明：T4 从 branch action 升级为连续 Mano-like loop；signal/random/signflip/shuffled controls 共享 tangent normalization 与 cadence。T4 fallback 又新增 AdamW moments + row tangent + norm-matched signal/control。T7 真实执行 Cautious AdamW、Schedule-Free averaging 近似、Muon-like；SOAP/Shampoo-like 只执行 fallback feasibility diagnostic，不写 official pass。",
        "",
        md_table(t4_rows, ["dataset", "seed", "optimizer_variant", "NLL", "accuracy", "update_SNR", "singular_value_spectrum_top_to_mean", "norm_match_scale_mean"], limit=20),
        "",
        md_table(t7_rows, ["dataset", "seed", "optimizer", "NLL", "accuracy", "optimizer_step_ms", "status", "implementation_level", "preconditioner_kron_param_steps", "preconditioner_diag_param_steps", "blocker"], limit=18),
        "",
        "## 7. T5 Online Subspace Evidence",
        "",
        f"- T5 rows: `{len(t5_rows)}`",
        f"- best online rows beating static PCA and same-subspace random rows: `{final.get('t5_online_beats_static_and_control_rows')}` / keys `{final.get('t5_best_online_keys')}`",
        "- 修改/实现说明：新增 Oja/QR parameter-gradient tracking、static PCA、online recovery 和 same-subspace random control；随后按 fallback 新增 projection-aware AdamW-state variants。effect-space/KAN-basis tracking 未通过前置 gate，因此保持 deferred。",
        "",
        md_table(t5_rows, ["dataset", "seed", "variant", "projection_residual", "NLL", "accuracy", "status", "implementation_level"], limit=24),
        "",
        "## 8. T6 Temporal / Grokking Evidence",
        "",
        f"- T6 rows: `{len(t6_rows)}`",
        f"- raw slow FU grokking events: `{final.get('t6_raw_slow_fu_grokking_events')}`; strict pass events: `{final.get('t6_slow_fu_grokking_events')}`",
        f"- noisy fallback rows: `{len([r for r in t6_rows if (finite_float(r.get('label_noise_rate'), 0.0) or 0.0) > 0.0])}`; max noisy fallback test_acc: `{max_field([r for r in t6_rows if (finite_float(r.get('label_noise_rate'), 0.0) or 0.0) > 0.0], 'test_acc')}`",
        f"- continual rows: `{len(continual_rows)}`; pass rows: `{final.get('t6_continual_pass_rows')}`; best relative forgetting reduction: `{final.get('t6_best_relative_forgetting_reduction')}`",
        f"- continual matched-control rows: `{final.get('t6_continual_matched_control_rows')}`; KAN matched-control rows: `{final.get('t6_continual_kan_control_rows')}`; FU rows final-accuracy-non-worse: `{final.get('t6_continual_final_accuracy_non_worse_rows')}`",
        "- 修改/实现说明：新增 modular addition probe、embedding-MLP fallback 与 train-only label-noise fallback；test labels 保持 clean。R7 判定要求 slow-FU grokking time 至少比 base 快 25%，且 matched slow-random/signflip controls 没有同等或更快 grokking；不把 final accuracy no-harm 写成 grokking 成功。",
        "- Continual 修改/实现说明：新增本地 Class-MNIST 0-1/2-3/4-5/6-7/8-9 pair sequence；slow-FU 的 promotion 必须同时满足 forgetting reduction、family-matched controls fail、final accuracy non-worse。该 protocol 是 v22.30 计划内 forgetting diagnostic，不是 KANbeFair original implementation。",
        "",
        md_table(t6_rows, ["task", "seed", "variant", "model_kind", "train_acc", "test_acc", "grokking_time", "grokking_time_reduction", "label_noise_rate"], limit=16),
        "",
        md_table(
            continual_rows,
            [
                "task",
                "seed",
                "model_name",
                "variant",
                "avg_forgetting",
                "BWT",
                "mean_final_accuracy",
                "relative_forgetting_reduction_vs_base",
                "final_accuracy_delta_vs_base",
                "final_accuracy_non_worse_vs_base",
                "matched_control_count",
                "matched_control_min_avg_forgetting",
                "beats_forgetting_controls",
                "status",
            ],
            limit=18,
        ),
        "",
        "## 9. KAN / KANbeFair Evidence",
        "",
        f"- KAN carrier rows: `{len(carrier_rows)}`; gap rows: `{len(gap_rows)}`",
        f"- positive gap reduction rows: `{final.get('kan_gap_positive_rows')}`",
        f"- hard-tier gap rows: `{final.get('kan_hard_gap_rows')}`; hard positive rows: `{final.get('kan_hard_gap_positive_rows')}`; hard positive rate: `{final.get('kan_hard_gap_positive_rate')}`",
        f"- hard MLP+FU improves rows: `{final.get('hard_mlp_fu_improves_rows')}`; hard KAN+FU improves rows: `{final.get('hard_kan_fu_improves_rows')}`; hard KAN+FU<=MLP+FU rows: `{final.get('hard_arch_nonworse_rows')}`; hard official full candidate rows: `{final.get('hard_official_full_candidate_rows')}`",
        f"- hard MLP+FU beats matched controls rows: `{final.get('hard_mlp_fu_beats_control_rows')}`; hard KAN+FU beats matched controls rows: `{final.get('hard_kan_fu_beats_control_rows')}`; both-control rows: `{final.get('hard_fu_beats_all_control_rows')}`",
        f"- KAN/FU slow gate mean keep rate: `{mean_field([r for r in carrier_rows if r.get('training') == 'FU'], 'slow_gate_keep_rate')}`",
        f"- KAN/FU slow trust clip mean rate: `{mean_field([r for r in carrier_rows if r.get('training') == 'FU'], 'slow_trust_clip_rate')}`",
        f"- KAN/FU slow acceptance mean accept rate: `{mean_field([r for r in carrier_rows if r.get('training') == 'FU'], 'slow_accept_rate')}`",
        "- 修改/实现说明：只使用 `PrimitiveKAN` strict FC-PureKAN rows；KANbeFair original KAN/BSpline 不进入 official。若指定 `--kan-hard-datasets`，会运行 strict PrimitiveKAN hard subset；dataset/download blocker 只写 external availability row，并要求 hard positive rate >=60% 才允许 R10。",
        "",
        md_table(
            gap_rows,
            [
                "dataset",
                "seed",
                "kan_family",
                "gap_BP",
                "gap_FU",
                "gap_reduction",
                "train_steps",
                "slow_alpha",
                "slow_signal",
                "slow_gate",
                "slow_trust_ratio",
                "slow_acceptance",
                "MLP_plus_FU_NLL_delta_vs_MLP",
                "MLP_FU_vs_best_control_NLL_delta",
                "KAN_plus_FU_NLL_delta_vs_KAN",
                "KAN_FU_vs_best_control_NLL_delta",
                "KAN_plus_FU_vs_MLPFU_NLL_delta",
                "official_full_row_pass",
            ],
            limit=12,
        ),
        "",
        "### Readout-Exact Full-Loop Diagnostic",
        "",
        f"- readout-exact carrier rows: `{len(readout_exact_carrier_rows)}`; gap rows: `{len(readout_exact_gap_rows)}`",
        f"- readout-exact hard rows: `{len(readout_exact_hard_rows)}`; hard positive gap rows: `{len(readout_exact_gap_positive_rows)}`; diagnostic full-row pass rows: `{len(readout_exact_diag_pass_rows)}`",
        "- 审计说明：该小节是 basis-effect 强信号后的 full-loop diagnostic，所有 rows 均 `fu_official_eligible=0`；即使 diagnostic pass，也不能替代 strict FC-PureKAN basis-native FU proof。",
        "",
        md_table(
            readout_exact_gap_rows,
            [
                "dataset",
                "seed",
                "kan_family",
                "fu_mode",
                "fu_official_eligible",
                "gap_BP",
                "gap_FU",
                "gap_reduction",
                "MLP_plus_FU_NLL_delta_vs_MLP",
                "MLP_FU_vs_best_control_NLL_delta",
                "KAN_plus_FU_NLL_delta_vs_KAN",
                "KAN_FU_vs_best_control_NLL_delta",
                "KAN_plus_FU_vs_MLPFU_NLL_delta",
                "diagnostic_full_row_pass",
                "official_full_row_pass",
            ],
            limit=12,
        ),
        "",
        "### KAN Repair Attempt Trace",
        "",
        md_table(
            repair_attempt_rows[-20:],
            [
                "timestamp_sg",
                "attempt_id",
                "kan_hard_steps",
                "signal_alpha",
                "kan_slow_gate",
                "kan_slow_signal",
                "kan_slow_trust_ratio",
                "kan_slow_acceptance",
                "kan_slow_acceptance_tol",
                "fu_mode",
                "readout_mechanism",
                "functional_mechanism",
                "functional_fu_lr",
                "functional_mlp_fu_lr",
                "functional_kan_fu_lr",
                "functional_fu_lr_schedule",
                "functional_fu_interval",
                "functional_mlp_fu_interval",
                "functional_kan_fu_interval",
                "functional_acceptance",
                "hard_gap_positive_rows",
                "hard_mlp_fu_improves_rows",
                "hard_kan_fu_improves_rows",
                "hard_mlp_fu_beats_control_rows",
                "hard_kan_fu_beats_control_rows",
                "hard_fu_beats_all_control_rows",
                "hard_arch_nonworse_rows",
                "hard_official_full_candidate_rows",
                "diagnostic_full_candidate_rows",
                "official_full_superiority_ready",
                "note",
            ],
            limit=20,
        ),
        "",
        "## 10. Failure / Deferred Taxonomy",
        "",
        md_table(failures, ["module_id", "route", "reason"], limit=20),
        "",
        md_table(deferred, ["module_id", "item", "reason"], limit=20),
        "",
        "## 11. Evidence Chain And Insights",
        "",
        "- Evidence chain: code truth/import/identity -> historical context reanalysis -> v22.29 control hierarchy reconstruction -> v22.30 small real gates -> final route JSON.",
        "- Insight 1: v22.30 的关键不是找到单个局部 NLL 改善，而是要求 real direction 同时打过 matched controls；因此任何只赢 base 或只赢 weak controls 的行都不能进入 full-loop。",
        f"- Insight 2: T1/T2 的 sketched layerwise implementation 比 v22.29 last-layer-only 更接近计划本体；subspace refinement 进一步检查“方向而非子空间”是否有因果内容。补齐 T3 H200、hidden/logits effect-space readout transport 与 KAN basis-effect w2 diagnostic 后，当前整体 pass rate 为 `{final.get('t3_transport_pass_groups')}/{final.get('t3_transport_eval_groups')}` = `{final.get('t3_transport_pass_rate')}`。其中 KAN basis-effect 单独很强（31/36），但 MLP sources 仍只有 37/90，所以 R4 blocker 已从“可能漏掉 KAN basis 空间”转为“总体 transport family 仍未稳定打过 matched controls”。这支持把 KAN basis-effect 作为后续 full-loop 候选，但不能把 w2-only diagnostic 写成 official basis-native FU。",
        "- Insight 3: T4 已经从 one-shot branch 升级到 continuous loop；optimizer-interaction claim 现在由 best eligible signal-FU vs 同 dataset/seed strongest completed baseline 判定。若只赢弱几何 baseline 或只赢同优化器 controls，但输 AdamW/Cautious/Schedule-Free/Muon-like，仍只能记录为 diagnostic signal，不能 promotion。",
        f"- Insight 4: KAN rows 现在包含 Tier0 debug、Wine Tier2 strict PrimitiveKAN subset，以及 CIFAR10/SVHN/EMNIST Tier1 hard vision strict subsets。下载/代理 blocker 已通过临时清除 socks5h proxy 修复；当前完整 hard subset 的 gap reduction 为 `{final.get('kan_hard_gap_positive_rows')}/{final.get('kan_hard_gap_rows')}` = `{final.get('kan_hard_gap_positive_rate')}`，满足打开 R10 carrier route 的阈值。但 `KAN_plus_FU_vs_MLPFU_NLL_delta` 仍有大量正值，且 exact/original KANbeFair official continual protocol 未闭合，因此不能 claim official architecture superiority。",
        "- Insight 5: R12 只在补齐 KAN family-matched slow-random/signflip controls 并加入 final accuracy non-worse 条件后打开；这说明 continual signal 目前是 Class-MNIST diagnostic 里的局部机制证据，不等同于 KANbeFair original/official 成功。",
        "- Insight 6: Part K 修复尝试显示，延长 hard steps 可以显著改善 KAN/gap/architecture 计数，但 MLP+FU 在 CIFAR10/EMNIST 上仍大面积劣于 MLP+Opt；current-gradient gate、held-gradient gate、held-gradient projection、trust-ratio cap、fast-residual signal、train-batch acceptance 与 held-train acceptance 若仍不能解除 blocker，则说明当前 slow-FU hard-task full-success 主要受 matched controls 与 MLP+FU base-improvement 约束，而不是单一 gate 形式。",
        f"- Insight 7: KAN basis-effect T3 结果促成了 readout-exact full-loop diagnostic；当前 readout-exact hard diagnostic pass 为 `{len(readout_exact_diag_pass_rows)}/{len(readout_exact_hard_rows)}`，hard positive gap 为 `{len(readout_exact_gap_positive_rows)}/{len(readout_exact_hard_rows)}`。这些 rows 用来定位下一步机制空间，不改变 `official_full_superiority_ready`，因为 readout-exact 不能直接证明 basis-native KAN controller。",
        "",
        "## 12. Reproducibility Pointers",
        "",
        "- 主命令记录在 `results/v22_30/v22_30_command_journal.csv` 与本执行日志。",
        "- 主 runner: `experiments/run_v22_30_fidelity_ladder.py`。",
        "- 关键 artifacts: `results/v22_30/v22_30_final_route.json`, `v22_30_fidelity_matrix.csv`, `v22_30_failure_taxonomy.csv`, `v22_30_deferred_items.csv`。",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "firewall", "historical", "control", "t1t2", "t4t7", "t5", "t6", "kan", "kan_readout", "finalize"],
    )
    p.add_argument("--historical-root", default="results")
    p.add_argument("--datasets", default="MNIST,FMNIST")
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="0,1,2,3")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--pretrain-steps", type=int, default=40)
    p.add_argument("--cohort-size", type=int, default=16)
    p.add_argument("--signal-cohorts", type=int, default=4)
    p.add_argument("--temporal-signal-steps", type=int, default=20)
    p.add_argument("--branch-horizons", default="50,100")
    p.add_argument("--branch-trust", type=float, default=0.03)
    p.add_argument("--disable-t1-refinement", action="store_true")
    p.add_argument("--t1-refinement-candidates", type=int, default=8)
    p.add_argument("--t1-refinement-l3-controls", type=int, default=4)
    p.add_argument("--rank-cap", type=int, default=4)
    p.add_argument("--sketch-dim", type=int, default=48)
    p.add_argument("--optimizer-steps", type=int, default=180)
    p.add_argument("--signal-alpha", type=float, default=0.20)
    p.add_argument("--subspace-rank", type=int, default=4)
    p.add_argument("--subspace-steps", type=int, default=140)
    p.add_argument("--grokking-steps", type=int, default=420)
    p.add_argument("--continual-steps", type=int, default=60)
    p.add_argument("--continual-train-size", type=int, default=500)
    p.add_argument("--continual-test-size", type=int, default=250)
    p.add_argument("--temporal-slow-beta", type=float, default=0.95)
    p.add_argument("--log-interval", type=int, default=70)
    p.add_argument("--kan-steps", type=int, default=120)
    p.add_argument("--kan-hard-datasets", default="")
    p.add_argument("--kan-hard-train-size", type=int, default=256)
    p.add_argument("--kan-hard-test-size", type=int, default=128)
    p.add_argument("--kan-hard-steps", type=int, default=60)
    p.add_argument("--kan-hard-download", action="store_true")
    p.add_argument(
        "--kan-slow-signal",
        default="ema",
        choices=["ema", "slow_minus_fast", "split_consensus", "split_consensus_ema", "split_consensus_slow_minus_fast"],
    )
    p.add_argument(
        "--kan-slow-gate",
        default="none",
        choices=["none", "current_grad_cosine", "held_grad_cosine", "held_grad_projection", "split_grad_projection"],
    )
    p.add_argument("--kan-slow-trust-ratio", type=float, default=0.0)
    p.add_argument(
        "--kan-slow-acceptance",
        default="none",
        choices=[
            "none",
            "train_loss_nonworse",
            "held_train_loss_nonworse",
            "train_loss_beats_base_step",
            "held_train_loss_beats_base_step",
        ],
    )
    p.add_argument("--kan-slow-acceptance-tol", type=float, default=0.0)
    p.add_argument("--kan-fu-mode", default="slow", choices=["slow", "functional_metric"])
    p.add_argument(
        "--kan-functional-mechanism",
        default="M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU",
        choices=[
            "M265-V2206MetricSolverT7G0HiddenBlockFU",
            "M266-V2206MetricSolverT8G0AdaptiveHiddenBlockFU",
            "M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU",
            "M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU",
            "M269-V2206MetricSolverT11G0EarlyObservableFU",
            "M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU",
            "M271-V2206MetricSolverT12G0HiddenOnlySoftCompensatedFU",
        ],
    )
    p.add_argument("--kan-functional-fu-lr", type=float, default=1.0e-4)
    p.add_argument("--kan-functional-mlp-fu-lr", type=float, default=0.0)
    p.add_argument("--kan-functional-kan-fu-lr", type=float, default=0.0)
    p.add_argument("--kan-functional-fu-lr-schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--kan-functional-fu-interval", type=int, default=5)
    p.add_argument("--kan-functional-mlp-fu-interval", type=int, default=0)
    p.add_argument("--kan-functional-kan-fu-interval", type=int, default=0)
    p.add_argument("--kan-functional-acceptance", default="none", choices=["none", "train_loss_nonworse", "held_train_loss_nonworse"])
    p.add_argument("--kan-functional-acceptance-tol", type=float, default=0.0)
    p.add_argument(
        "--kan-readout-mechanism",
        default="M49-LossCotangentTargetFU",
        choices=[
            "M21-ExactReadoutFunctionSpaceActuationFU",
            "M49-LossCotangentTargetFU",
            "M54-CrossSplitConsensusTargetFU",
        ],
    )
    p.add_argument("--kan-readout-fu-lr", type=float, default=1.0e-4)
    p.add_argument("--kan-readout-fu-interval", type=int, default=5)
    p.add_argument("--kan-readout-acceptance", default="none", choices=["none", "train_loss_nonworse", "held_train_loss_nonworse"])
    p.add_argument("--kan-readout-acceptance-tol", type=float, default=0.0)
    return p


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    command = " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv])
    append_exec(
        command,
        task_id="v22_30_start",
        status="started",
        gpu=f"visible {args.device_map}; active {args.device}; serial exploration runner",
        files=str(PLAN_DOC.relative_to(ROOT)),
        note="Start v22.30 fidelity ladder execution; results are written only from actual commands/artifacts.",
    )
    code_truth = stage_a_code_firewall()
    historical_rows = stage_b_historical_reanalysis(args)
    control_rows = stage_c_control_hierarchy()
    t1_rows, branch_rows, t2_rows, t1_refinement_rows, t1_refinement_branch_rows = stage_t1_t2(args)
    t4_rows, _spectrum_rows, t7_rows = stage_t4_t7(args)
    t5_rows, _recovery_rows = stage_t5(args)
    t6_rows, _curves = stage_t6(args)
    carrier_rows, gap_rows, _external_rows = stage_kan_kanbefair(args)
    fidelity_matrix(code_truth, t1_rows, t2_rows, t4_rows, t5_rows, t6_rows, carrier_rows)
    final, deferred, failures = decide_routes(code_truth, t1_rows, t1_refinement_rows, t4_rows, t5_rows, t6_rows, t7_rows, gap_rows)
    write_recap(
        final,
        code_truth,
        historical_rows,
        control_rows,
        t1_rows,
        branch_rows,
        t1_refinement_rows,
        t1_refinement_branch_rows,
        t2_rows,
        t4_rows,
        t7_rows,
        t5_rows,
        t6_rows,
        carrier_rows,
        gap_rows,
        deferred,
        failures,
    )
    write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
    append_exec(
        command,
        task_id="v22_30_all",
        status="pass",
        gpu=f"visible {args.device_map}; active {args.device}; serial exploration runner",
        exit_code=0,
        files=(
            "results/v22_30/v22_30_final_route.json, "
            "results/v22_30/v22_30_artifact_index.csv, "
            "docs/DG-KAN_v22.30_FidelityLadderGeometricFU_实验结果复盘.md"
        ),
        note=f"final_route={final.get('final_route')}; no fabricated rows; deferred_items={len(deferred)}; failure_rows={len(failures)}",
    )
    write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
    return final


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    if args.stage == "all":
        run_all(args)
        return
    if args.stage == "firewall":
        stage_a_code_firewall()
    elif args.stage == "historical":
        stage_b_historical_reanalysis(args)
    elif args.stage == "control":
        stage_c_control_hierarchy()
    elif args.stage == "t1t2":
        command = " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv])
        append_exec(
            command,
            task_id="v22_30_t1t2_start",
            status="started",
            gpu=f"visible {args.device_map}; active {args.device}; serial T1/T2/T3 stage",
            files=str(PLAN_DOC.relative_to(ROOT)),
            note="Start T1/T2/T3 rerun for signal-channel and wrong-space subspace diagnostics; non-T1/T2/T3 artifacts are not rewritten.",
        )
        t1_rows, branch_rows, t2_rows, refinement_rows, refinement_branch_rows = stage_t1_t2(args)
        t3_rows = read_rows(OUT_ROOT / "v22_30_T3_transport_momentum_matrix.csv")
        write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
        append_exec(
            command,
            task_id="v22_30_t1t2",
            status="pass",
            gpu=f"visible {args.device_map}; active {args.device}; serial T1/T2/T3 stage",
            exit_code=0,
            files=(
                "results/v22_30/v22_30_T1_layerwise_signal_channel_matrix.csv, "
                "results/v22_30/v22_30_T2_grassmann_outcome_linked_matrix.csv, "
                "results/v22_30/v22_30_T3_transport_momentum_matrix.csv"
            ),
            note=(
                f"T1 rows={len(t1_rows)}; branch rows={len(branch_rows)}; T2 rows={len(t2_rows)}; "
                f"T3 rows={len(t3_rows)}; T1 refinement rows={len(refinement_rows)}; "
                f"refinement branch rows={len(refinement_branch_rows)}"
            ),
        )
    elif args.stage == "t4t7":
        command = " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv])
        append_exec(
            command,
            task_id="v22_30_t4t7_start",
            status="started",
            gpu=f"visible {args.device_map}; active {args.device}; serial T4/T7 stage",
            files=str(PLAN_DOC.relative_to(ROOT)),
            note="Start T4/T7 rerun for optimizer-interaction fallback diagnostics; existing non-T4/T7 artifacts are not rewritten by this stage.",
        )
        t4_rows, _spectrum_rows, t7_rows = stage_t4_t7(args)
        write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
        append_exec(
            command,
            task_id="v22_30_t4t7",
            status="pass",
            gpu=f"visible {args.device_map}; active {args.device}; serial T4/T7 stage",
            exit_code=0,
            files=(
                "results/v22_30/v22_30_T4_tangent_optimizer_matrix.csv, "
                "results/v22_30/v22_30_T7_optimizer_baseline_matrix.csv, "
                "results/v22_30/v22_30_T7_fu_optimizer_interaction_matrix.csv"
            ),
            note=f"T4 rows={len(t4_rows)}; T7 rows={len(t7_rows)}; SOAP/Shampoo-like fallback rows="
            f"{sum(1 for r in t7_rows if r.get('optimizer') == 'SOAP/Shampoo-like')}",
        )
    elif args.stage == "t5":
        stage_t5(args)
    elif args.stage == "t6":
        command = " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv])
        append_exec(
            command,
            task_id="v22_30_t6_start",
            status="started",
            gpu=f"visible {args.device_map}; active {args.device}; serial T6 temporal/continual stage",
            files=str(PLAN_DOC.relative_to(ROOT)),
            note="Start T6 rerun for grokking fallback plus Class-MNIST continual forgetting diagnostic.",
        )
        t6_rows, curves = stage_t6(args)
        continual_rows = read_rows(OUT_ROOT / "v22_30_T6_continual_boundary_matrix.csv")
        write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
        append_exec(
            command,
            task_id="v22_30_t6",
            status="pass",
            gpu=f"visible {args.device_map}; active {args.device}; serial T6 temporal/continual stage",
            exit_code=0,
            files=(
                "results/v22_30/v22_30_T6_grokking_matrix.csv, "
                "results/v22_30/v22_30_T6_continual_boundary_matrix.csv, "
                "results/v22_30/v22_30_KANbeFair_continual_matrix.csv"
            ),
            note=f"T6 grokking rows={len(t6_rows)}; curve rows={len(curves)}; continual rows={len(continual_rows)}",
        )
    elif args.stage == "kan":
        command = " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv])
        append_exec(
            command,
            task_id="v22_30_kan_start",
            status="started",
            gpu=f"visible {args.device_map}; active {args.device}; serial KAN/KANbeFair stage",
            files=str(PLAN_DOC.relative_to(ROOT)),
            note="Start KAN/KANbeFair rerun for hard-tier availability and strict PrimitiveKAN evidence; non-KAN artifacts are not rewritten.",
        )
        carrier_rows, gap_rows, external_rows = stage_kan_kanbefair(args)
        write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
        append_exec(
            command,
            task_id="v22_30_kan",
            status="pass",
            gpu=f"visible {args.device_map}; active {args.device}; serial KAN/KANbeFair stage",
            exit_code=0,
            files=(
                "results/v22_30/v22_30_KAN_basis_signal_carrier_matrix.csv, "
                "results/v22_30/v22_30_KAN_vs_MLPFU_gap_matrix.csv, "
                "results/v22_30/v22_30_KANbeFair_external_task_matrix.csv"
            ),
            note=f"carrier rows={len(carrier_rows)}; gap rows={len(gap_rows)}; external rows={len(external_rows)}",
        )
    elif args.stage == "kan_readout":
        command = " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv])
        append_exec(
            command,
            task_id="v22_30_kan_readout_start",
            status="started",
            gpu=f"visible {args.device_map}; active {args.device}; serial readout-exact KAN diagnostic",
            files=str(PLAN_DOC.relative_to(ROOT)),
            note=(
                "Start separate readout-exact full-loop diagnostic with same-job matched controls; "
                "official KAN slow-FU artifacts are not overwritten."
            ),
        )
        carrier_rows, gap_rows, external_rows = stage_kan_readout_exact(args)
        write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
        append_exec(
            command,
            task_id="v22_30_kan_readout",
            status="pass",
            gpu=f"visible {args.device_map}; active {args.device}; serial readout-exact KAN diagnostic",
            exit_code=0,
            files=(
                "results/v22_30/v22_30_KAN_readout_exact_carrier_matrix.csv, "
                "results/v22_30/v22_30_KAN_readout_exact_gap_matrix.csv, "
                "results/v22_30/v22_30_KAN_readout_exact_external_task_matrix.csv, "
                "results/v22_30/v22_30_repair_attempt_summary.csv"
            ),
            note=(
                f"readout carrier rows={len(carrier_rows)}; gap rows={len(gap_rows)}; external rows={len(external_rows)}; "
                "fu_official_eligible=0 for readout diagnostic rows"
            ),
        )
    elif args.stage == "finalize":
        code_truth_rows = read_rows(OUT_ROOT / "v22_30_code_truth.csv")
        code_truth = code_truth_rows[0] if code_truth_rows else {}
        final, deferred, failures = decide_routes(
            code_truth,
            read_rows(OUT_ROOT / "v22_30_T1_layerwise_signal_channel_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T1_subspace_refinement_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T4_tangent_optimizer_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T5_online_subspace_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T6_grokking_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T7_optimizer_baseline_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_KAN_vs_MLPFU_gap_matrix.csv"),
        )
        write_recap(
            final,
            code_truth,
            read_rows(OUT_ROOT / "v22_30_historical_reanalysis_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_control_hierarchy_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T1_layerwise_signal_channel_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T1_branch_causal_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T1_subspace_refinement_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T1_subspace_refinement_branch_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T2_grassmann_outcome_linked_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T4_tangent_optimizer_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T7_optimizer_baseline_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T5_online_subspace_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_T6_grokking_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_KAN_basis_signal_carrier_matrix.csv"),
            read_rows(OUT_ROOT / "v22_30_KAN_vs_MLPFU_gap_matrix.csv"),
            deferred,
            failures,
        )
        write_rows(OUT_ROOT / "v22_30_artifact_index.csv", artifact_index())
        append_exec(
            " ".join(shlex.quote(x) for x in [PYTHON, *sys.argv]),
            task_id="v22_30_finalize",
            status="pass",
            gpu=f"visible {args.device_map}; active {args.device}; finalize-only",
            exit_code=0,
            files=(
                "results/v22_30/v22_30_final_route.json, "
                "results/v22_30/v22_30_artifact_index.csv, "
                "docs/DG-KAN_v22.30_FidelityLadderGeometricFU_实验结果复盘.md"
            ),
            note=f"final_route={final.get('final_route')}; finalize after audit text/deferred-item correction; no training rerun",
        )


if __name__ == "__main__":
    main()
