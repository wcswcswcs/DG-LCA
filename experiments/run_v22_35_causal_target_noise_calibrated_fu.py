#!/usr/bin/env python3
"""DG-KAN v22.35 noise-calibrated causal-target FU runner.

This runner is intentionally conservative. It produces v22.35-owned artifacts,
keeps command/log provenance, runs small direct row-local no-op repeat probes
and Level-0 readout-exact target probes, and re-audits named v22.34/v22.33/v22.30
artifacts under the v22.35 gates. Missing or gate-blocked evidence is written as
such; no missing rows are inferred.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_35"
FIG_ROOT = OUT_ROOT / "figures"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_完整计划.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md"
V22_34 = ROOT / "results/v22_34"
V22_33 = ROOT / "results/v22_33"
V22_30 = ROOT / "results/v22_30"

TIER2_TABULAR_SPECS: dict[str, dict[str, Any]] = {
    "spam": {
        "canonical": "Spam",
        "uci_id": 94,
        "official_url": "https://archive.ics.uci.edu/dataset/94/spambase",
        "direct_urls": ["https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"],
        "direct_format": "csv_target_last",
        "openml_names": ["spambase"],
    },
    "rice": {
        "canonical": "Rice",
        "uci_id": 545,
        "official_url": "https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik",
        "direct_urls": ["https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip"],
        "direct_format": "zip_arff_target_last",
        "openml_names": ["Rice_Cammeo_Osmancik"],
    },
    "bean": {
        "canonical": "Bean",
        "uci_id": 602,
        "official_url": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        "direct_urls": ["https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"],
        "direct_format": "zip_arff_target_last",
        "openml_names": ["Dry_Bean_Dataset"],
    },
    "telescope": {
        "canonical": "Telescope",
        "uci_id": 159,
        "official_url": "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
        "direct_urls": ["https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"],
        "direct_format": "csv_target_last",
        "openml_names": ["MagicTelescope", "magic"],
    },
}
TIER2_TABULAR_ALIASES = {
    "spambase": "spam",
    "spam": "spam",
    "rice": "rice",
    "rice_cammeo_osmancik": "rice",
    "rice_cammeo_and_osmancik": "rice",
    "bean": "bean",
    "dry_bean": "bean",
    "dry_bean_dataset": "bean",
    "telescope": "telescope",
    "magic": "telescope",
    "magictelescope": "telescope",
    "magic_gamma_telescope": "telescope",
}


class TaskUnavailableError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.35 Causal Target Noise-Calibrated FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、真实输入/输出文件、gate 状态、blocker 与修复尝试；"
            "未执行或被 gate 阻断的项目必须明确写为 not_run/gate_blocked。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.35 Causal Target Noise-Calibrated FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用 v22.35 本轮 artifact、真实命令日志与明确命名的上游 artifact；禁止编造数据。\n",
            encoding="utf-8",
        )


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


def rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def pstdev(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0
    return float(statistics.pstdev(vals))


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    journal = OUT_ROOT / "v22_35_command_journal.csv"
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
    gpu: str = "",
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_out()
    merged = os.environ.copy()
    if env:
        merged.update(env)
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
    except subprocess.TimeoutExpired as exc:
        proc = subprocess.CompletedProcess(args=args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
    stdout_path = LOG_ROOT / f"{task_id}_stdout.log"
    stderr_path = LOG_ROOT / f"{task_id}_stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    status = "pass" if proc.returncode == 0 else ("timeout" if proc.returncode == 124 else "fail")
    append_exec(
        " ".join(shlex.quote(x) for x in args),
        task_id=task_id,
        status=status,
        gpu=gpu,
        exit_code=proc.returncode,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
    )
    return proc


def create_review_bundle(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    include_roots = [
        ROOT / "dgkan",
        ROOT / "experiments",
        PLAN_DOC,
        ROOT / "requirement.txt",
    ]
    with tarfile.open(path, "w:gz") as tar:
        for source in include_roots:
            if not source.exists():
                continue
            if source.is_file():
                tar.add(source, arcname=str(source.relative_to(ROOT)))
                continue
            for item in sorted(source.rglob("*")):
                if item.is_dir():
                    continue
                rel = item.relative_to(ROOT)
                parts = set(rel.parts)
                if "__pycache__" in parts or item.suffix in {".pyc", ".pyo"}:
                    continue
                tar.add(item, arcname=str(rel))
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def md_table(rows: list[dict[str, Any]], fields: list[str] | None = None, limit: int = 12) -> str:
    materialized = [dict(r) for r in rows[:limit]]
    if not materialized:
        return "_无可用行。_"
    cols = fields or list(materialized[0].keys())
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in materialized:
        lines.append("|" + "|".join(str(row.get(c, "")).replace("\n", " ") for c in cols) + "|")
    if len(rows) > limit:
        lines.append(f"\n_仅显示前 {limit} 行，共 {len(rows)} 行；完整 CSV 见 artifact。_")
    return "\n".join(lines)


def write_simple_svg(path: Path, title: str, rows: list[dict[str, Any]], value_key: str = "value") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals: list[tuple[str, float]] = []
    for i, row in enumerate(rows[:14]):
        label = str(row.get("label") or row.get("selector_name") or row.get("target_family") or row.get("gap_reduction_class") or i)
        vals.append((label[:34], finite_float(row.get(value_key), 0.0) or 0.0))
    if not vals:
        vals = [("no_data", 0.0)]
    max_abs = max(1.0e-12, max(abs(v) for _, v in vals))
    width, height = 940, 260 + 24 * len(vals)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="24" y="36" font-family="monospace" font-size="18">{title}</text>',
    ]
    x0, y0, bar_w = 300, 70, 500
    parts.append(f'<line x1="{x0}" y1="56" x2="{x0}" y2="{height-24}" stroke="#777" stroke-width="1"/>')
    for i, (label, value) in enumerate(vals):
        y = y0 + i * 24
        scaled = int(abs(value) / max_abs * bar_w)
        color = "#2f6f9f" if value >= 0 else "#b44b3e"
        x = x0 if value >= 0 else x0 - scaled
        parts.append(f'<text x="24" y="{y+14}" font-family="monospace" font-size="12">{label}</text>')
        parts.append(f'<rect x="{x}" y="{y}" width="{max(1, scaled)}" height="16" fill="{color}"/>')
        parts.append(f'<text x="{x0 + bar_w + 12}" y="{y+13}" font-family="monospace" font-size="11">{value:.6g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def artifact_index() -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file() and path.name != "v22_35_artifact_index.csv":
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(path.stat().st_mtime)),
                }
            )
    write_rows(OUT_ROOT / "v22_35_artifact_index.csv", rows)
    digest = hashlib.sha256()
    for row in rows:
        digest.update((row["artifact"] + row["sha256"]).encode("utf-8"))
    return rows, digest.hexdigest()


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def torch_device(name: str) -> Any:
    import torch

    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def stage_a() -> dict[str, Any]:
    bundle = OUT_ROOT / "v22_35_review_bundle_current.tar.gz"
    bundle_sha = create_review_bundle(bundle)
    clean_root = OUT_ROOT / "clean_unzip_check"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(clean_root)
    append_exec(
        "create and extract v22.35 review bundle for clean compile/import closure",
        task_id="A_clean_unzip_bundle",
        status="pass",
        gpu="0",
        files=f"{bundle.relative_to(ROOT)}, {bundle.with_suffix(bundle.suffix + '.sha256').relative_to(ROOT)}, {clean_root.relative_to(ROOT)}",
        note=f"sha256={bundle_sha}",
    )
    compile_proc = run_logged(
        [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"],
        task_id="A_clean_unzip_compileall",
        gpu="0",
        timeout=600,
        cwd=clean_root,
    )
    import_code = (
        "mods=['dgkan','dgkan.fu.real_jacobian_commit','dgkan.fu.basis_native_controller',"
        "'dgkan.fu.metric_solver','dgkan.models.fc_purekan_primitives',"
        "'experiments.run_v22_30_fidelity_ladder']; missing=[]\n"
        "for m in mods:\n"
        "    try: __import__(m)\n"
        "    except Exception as exc: missing.append((m,type(exc).__name__,str(exc)))\n"
        "print({'clean_import_ok': not missing, 'missing': missing})\n"
        "raise SystemExit(1 if missing else 0)\n"
    )
    import_proc = run_logged(
        [PYTHON, "-c", import_code],
        task_id="A_clean_unzip_import_closure",
        gpu="0",
        timeout=180,
        cwd=clean_root,
        env={"PYTHONPATH": str(clean_root)},
    )
    upstream = (read_rows(V22_34 / "v22_34_code_truth_gate.csv") or [{}])[0]
    row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "missing_module_names": "" if import_proc.returncode == 0 else "see logs/A_import_closure_stderr.log",
        "official_DGKAN_identity_pass": upstream.get("official_DGKAN_identity_pass", 1 if import_proc.returncode == 0 else 0),
        "KANbeFair_original_KAN_official_rows": upstream.get("KANbeFair_original_KAN_official_rows", 0),
        "uses_pykan_official_rows": upstream.get("uses_pykan_official_rows", 0),
        "uses_bspline_official_rows": upstream.get("uses_bspline_official_rows", 0),
        "uses_readout_diagnostic_official_rows": upstream.get("uses_readout_diagnostic_official_rows", 0),
        "artifact_manifest_hash": "",
        "review_bundle_path": str(bundle.relative_to(ROOT)),
        "review_bundle_sha256": bundle_sha,
        "runner_command_journal_complete": 1,
        "source_artifact": "direct_v22_35_clean_unzip_compile_import_plus_v22_34_code_truth_readback",
    }
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [row])
    append_exec(
        "write v22.35 Part A code truth gate",
        task_id="A_code_truth_matrix",
        status="pass" if int_flag(row["clean_unzip_compileall_pass"]) and int_flag(row["clean_unzip_import_pass"]) else "fail",
        gpu="0",
        files="results/v22_35/v22_35_code_truth_gate.csv",
        note=f"compileall_exit_code={compile_proc.returncode}; import_exit_code={import_proc.returncode}; clean_unzip_compileall_pass={row['clean_unzip_compileall_pass']}; clean_unzip_import_pass={row['clean_unzip_import_pass']}; bundle_sha256={bundle_sha}",
    )
    return row


def run_row_local_noise_repeats(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from experiments.run_v22_30_fidelity_ladder import (
        evaluate_model,
        make_kan,
        make_mlp,
        train_adamw,
        train_branch,
    )

    device = torch_device(args.noise_device)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    horizons = [int(x) for x in split_csv(getattr(args, "noise_branch_horizons", ""), int) if int(x) > 0]
    if not horizons:
        horizons = [int(args.noise_branch_horizon)]
    for dataset in split_csv(args.noise_datasets):
        for seed in split_csv(args.noise_seeds, int):
            train_loader, _held_loader, test_loader, input_dim, output_dim, x_stats, task_meta = make_v22_35_c11_loaders(
                dataset,
                args.noise_train_size,
                args.noise_test_size,
                args.batch_size,
                seed,
                tier2_download=bool(int(args.tier2_download)),
            )
            for model_name in split_csv(args.noise_models):
                family = ""
                if model_name.upper() == "MLP":
                    model = make_mlp(input_dim, output_dim, args.hidden, seed + 3500, device).to(device)
                    arch = "MLP"
                else:
                    family = model_name
                    model = make_kan(input_dim, output_dim, args.hidden, seed + 3510, device, x_stats, family).to(device)
                    arch = "KAN"
                train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.noise_pretrain_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                base_eval = evaluate_model(model, test_loader, device, output_dim)
                for horizon in horizons:
                    deltas: list[float] = []
                    for repeat_idx in range(int(args.noise_repeats)):
                        # Fixed eval set; repeated iterator construction advances the
                        # train DataLoader generator and captures no-op branch variance.
                        branch = copy.deepcopy(model).to(device)
                        ev = train_branch(
                            branch,
                            train_loader,
                            test_loader,
                            device,
                            output_dim,
                            horizon,
                            args.lr,
                            args.weight_decay,
                        )
                        delta = float(ev["NLL"] - base_eval["NLL"])
                        deltas.append(delta)
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "task_tier": task_meta.get("task_tier", ""),
                                "source_kind": task_meta.get("source_kind", ""),
                                "model_hidden": int(args.hidden),
                                "architecture": arch,
                                "carrier": family or "MLP",
                                "model_name": model_name,
                                "checkpoint_step": args.noise_pretrain_steps,
                                "branch_H": horizon,
                                "repeat_idx": repeat_idx,
                                "base_NLL": base_eval["NLL"],
                                "repeat_NLL": ev["NLL"],
                                "delta_NLL_noop_branch": delta,
                                "base_accuracy": base_eval["accuracy"],
                                "repeat_accuracy": ev["accuracy"],
                                "delta_accuracy_noop_branch": ev["accuracy"] - base_eval["accuracy"],
                                "epsilon_source": "paired_same_checkpoint_noop_repeat_direct_v22_35_fixed_eval",
                                "source_artifact": "direct_v22_35_row_local_noop_repeat",
                            }
                        )
                    epsilon = 2.0 * pstdev(deltas)
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "task_tier": task_meta.get("task_tier", ""),
                            "source_kind": task_meta.get("source_kind", ""),
                            "model_hidden": int(args.hidden),
                            "architecture": arch,
                            "carrier": family or "MLP",
                            "model_name": model_name,
                            "checkpoint_step": args.noise_pretrain_steps,
                            "branch_H": horizon,
                            "repeat_count": len(deltas),
                            "delta_NLL_mean": sum(deltas) / max(1, len(deltas)),
                            "delta_NLL_std": pstdev(deltas),
                            "epsilon_row": epsilon,
                            "epsilon_source": "paired_same_checkpoint_noop_repeat_direct_v22_35_fixed_eval",
                            "source_artifact": "direct_v22_35_row_local_noop_repeat",
                        }
                    )
    write_rows(OUT_ROOT / "v22_35_row_local_noise_repeats.csv", rows)
    write_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv", summary_rows)
    append_exec(
        "run direct row-local same-checkpoint no-op branch repeats",
        task_id="B_row_local_noop_repeats",
        status="pass" if summary_rows else "warn",
        gpu=args.noise_device,
        files="results/v22_35/v22_35_row_local_noise_repeats.csv, results/v22_35/v22_35_row_local_noise_matrix.csv",
        note=f"summary_rows={len(summary_rows)}; repeats={args.noise_repeats}",
    )
    return {"repeat_rows": len(rows), "summary_rows": len(summary_rows)}


def run_repair_row_local_noise_repeats(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from experiments.run_v22_30_fidelity_ladder import (
        evaluate_model,
        make_kan,
        train_adamw,
        train_branch,
    )

    device = torch_device(args.noise_device)
    rows: list[dict[str, Any]] = read_rows(OUT_ROOT / "v22_35_row_local_noise_repeats.csv")
    summary_rows: list[dict[str, Any]] = read_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv")
    new_repeat_rows: list[dict[str, Any]] = []
    new_summary_rows: list[dict[str, Any]] = []
    horizons = sorted(
        {
            int(x)
            for x in (
                split_csv(args.repair_branch_horizons, int)
                + split_csv(getattr(args, "basis_repair_branch_horizons", ""), int)
            )
            if int(x) > 0
        }
    )
    for dataset in split_csv(args.repair_datasets):
        for seed in split_csv(args.repair_seeds, int):
            try:
                train_loader, _held_loader, test_loader, input_dim, output_dim, x_stats, task_meta = make_v22_35_c11_loaders(
                    dataset,
                    args.repair_train_size,
                    args.repair_test_size,
                    args.batch_size,
                    seed,
                    tier2_download=bool(int(args.tier2_download)),
                )
            except Exception as exc:
                new_summary_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "task_tier": "Tier2_tabular" if (str(dataset).strip().lower() == "wine" or tier2_spec(dataset) is not None) else "",
                        "status": "task_unavailable",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "epsilon_source": "repair_row_local_noise_unavailable",
                        "source_artifact": "direct_v22_35_repair_row_local_noop_repeat",
                    }
                )
                continue
            for family in split_csv(args.repair_families):
                torch.manual_seed(235_000 + seed + (0 if family == "D-CHE" else 10_000))
                model = make_kan(input_dim, output_dim, args.hidden, seed + 3535, device, x_stats, family).to(device)
                train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.repair_pretrain_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                base_eval = evaluate_model(model, test_loader, device, output_dim)
                for horizon in horizons:
                    deltas: list[float] = []
                    for repeat_idx in range(int(args.noise_repeats)):
                        branch = copy.deepcopy(model).to(device)
                        ev = train_branch(
                            branch,
                            train_loader,
                            test_loader,
                            device,
                            output_dim,
                            horizon,
                            args.lr,
                            args.weight_decay,
                        )
                        delta = float(ev["NLL"] - base_eval["NLL"])
                        deltas.append(delta)
                        new_repeat_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "task_tier": task_meta.get("task_tier", ""),
                                "source_kind": task_meta.get("source_kind", ""),
                                "model_hidden": int(args.hidden),
                                "architecture": "KAN",
                                "carrier": family,
                                "model_name": family,
                                "checkpoint_step": args.repair_pretrain_steps,
                                "branch_H": horizon,
                                "repeat_idx": repeat_idx,
                                "base_NLL": base_eval["NLL"],
                                "repeat_NLL": ev["NLL"],
                                "delta_NLL_noop_branch": delta,
                                "base_accuracy": base_eval["accuracy"],
                                "repeat_accuracy": ev["accuracy"],
                                "delta_accuracy_noop_branch": ev["accuracy"] - base_eval["accuracy"],
                                "epsilon_source": "paired_same_checkpoint_noop_repeat_direct_v22_35_repair_fixed_eval",
                                "source_artifact": "direct_v22_35_repair_row_local_noop_repeat",
                            }
                        )
                    new_summary_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "task_tier": task_meta.get("task_tier", ""),
                            "source_kind": task_meta.get("source_kind", ""),
                            "model_hidden": int(args.hidden),
                            "architecture": "KAN",
                            "carrier": family,
                            "model_name": family,
                            "checkpoint_step": args.repair_pretrain_steps,
                            "branch_H": horizon,
                            "repeat_count": len(deltas),
                            "delta_NLL_mean": sum(deltas) / max(1, len(deltas)),
                            "delta_NLL_std": pstdev(deltas),
                            "epsilon_row": 2.0 * pstdev(deltas),
                            "epsilon_source": "paired_same_checkpoint_noop_repeat_direct_v22_35_repair_fixed_eval",
                            "source_artifact": "direct_v22_35_repair_row_local_noop_repeat",
                        }
                    )
    rows.extend(new_repeat_rows)
    summary_rows.extend(new_summary_rows)
    write_rows(OUT_ROOT / "v22_35_row_local_noise_repeats.csv", rows)
    write_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv", summary_rows)
    append_exec(
        "run repair-config row-local same-checkpoint no-op repeats for H20/H60/H200",
        task_id="B_repair_row_local_noop_repeats",
        status="pass" if new_summary_rows else "warn",
        gpu=args.noise_device,
        files="results/v22_35/v22_35_row_local_noise_repeats.csv, results/v22_35/v22_35_row_local_noise_matrix.csv",
        note=f"new_summary_rows={len(new_summary_rows)}; new_repeat_rows={len(new_repeat_rows)}; horizons={','.join(str(h) for h in horizons)}",
    )
    return {"repair_repeat_rows": len(new_repeat_rows), "repair_summary_rows": len(new_summary_rows)}


def epsilon_lookup(hidden: Any | None = None) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    hidden_filter = "" if hidden is None else str(hidden)
    for row in read_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv"):
        row_hidden = str(row.get("model_hidden", row.get("hidden", "16")) or "16")
        if hidden_filter and row_hidden != hidden_filter:
            continue
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        carrier = str(row.get("carrier", ""))
        branch_h = str(row.get("branch_H", ""))
        lookup[(dataset, seed, carrier, branch_h)] = row
        lookup.setdefault((dataset, seed, carrier), row)
        if carrier in {"D-CHE", "DCHE"}:
            lookup[(dataset, seed, "DCHE", branch_h)] = row
            lookup[(dataset, seed, "D-CHE", branch_h)] = row
            lookup.setdefault((dataset, seed, "DCHE"), row)
            lookup.setdefault((dataset, seed, "D-CHE"), row)
        if carrier in {"D-FOU", "DFOU"}:
            lookup[(dataset, seed, "DFOU", branch_h)] = row
            lookup[(dataset, seed, "D-FOU", branch_h)] = row
            lookup.setdefault((dataset, seed, "DFOU"), row)
            lookup.setdefault((dataset, seed, "D-FOU"), row)
        if carrier == "MLP":
            lookup[(dataset, seed, "MLP", branch_h)] = row
            lookup.setdefault((dataset, seed, "MLP"), row)
    return lookup


def control_bootstrap_epsilon(dataset: str, seed: str, horizon: str = "") -> tuple[float, str]:
    vals: list[float] = []
    for path in [
        V22_34 / "v22_34_T1_C9C10_repair_branch_matrix.csv",
        V22_34 / "v22_34_basis_target_repair_control_win_matrix.csv",
    ]:
        for row in read_rows(path):
            if str(row.get("dataset")) != str(dataset) or str(row.get("seed")) != str(seed):
                continue
            if horizon and str(row.get("branch_H", "")) not in {"", str(horizon)}:
                continue
            for key in [
                "NLL_delta_vs_base",
                "same_basis_random_NLL_delta",
                "signflip_basis_control_NLL_delta",
                "real_NLL_delta",
            ]:
                value = finite_float(row.get(key))
                if value is not None:
                    vals.append(value)
    eps = 2.0 * pstdev(vals)
    if eps <= 0.0:
        return 1.0e-6, "epsilon_float_no_repeat_or_bootstrap_available"
    return eps, "paired_branch_control_bootstrap_from_named_v22_34_rows"


def row_epsilon_for(dataset: str, seed: str, carrier: str, horizon: str = "", hidden: Any | None = None) -> tuple[float, str]:
    horizon_s = str(horizon)
    if horizon_s == "0":
        return 1.0e-6, "epsilon_float_immediate_H0_no_branch_diagnostic"
    lookup = epsilon_lookup(hidden)
    aliases = [carrier]
    if carrier == "D-CHE":
        aliases.append("DCHE")
    if carrier == "D-FOU":
        aliases.append("DFOU")
    direct_fallback: tuple[float, str] | None = None
    for alias in aliases:
        row = lookup.get((str(dataset), str(seed), alias, horizon_s)) if horizon_s else None
        if row:
            eps = finite_float(row.get("epsilon_row"), 0.0) or 0.0
            return max(float(eps), 1.0e-6), str(row.get("epsilon_source"))
    for alias in aliases:
        row = lookup.get((str(dataset), str(seed), alias))
        if row:
            eps = finite_float(row.get("epsilon_row"), 0.0) or 0.0
            row_h = str(row.get("branch_H", ""))
            source = str(row.get("epsilon_source"))
            if not horizon_s or row_h in {"", horizon_s}:
                return max(float(eps), 1.0e-6), source
            direct_fallback = (
                max(float(eps), 1.0e-6),
                f"{source}_H{row_h}_any_horizon_fallback_for_H{horizon_s}",
            )
    eps, source = control_bootstrap_epsilon(dataset, seed, horizon)
    if source == "epsilon_float_no_repeat_or_bootstrap_available" and str(horizon) not in {"", "0"}:
        eps, source_any = control_bootstrap_epsilon(dataset, seed, "")
        if source_any != "epsilon_float_no_repeat_or_bootstrap_available":
            return eps, f"{source_any}_any_horizon_fallback_for_H{horizon}"
    if source == "epsilon_float_no_repeat_or_bootstrap_available" and direct_fallback is not None:
        return direct_fallback
    return eps, source


def stage_b_gap_recalibration() -> dict[str, Any]:
    gap_rows = read_rows(V22_33 / "v22_33_gap_truth_matrix.csv")
    lookup = epsilon_lookup()
    out_rows: list[dict[str, Any]] = []
    classes = {name: 0 for name in ["TrueKANGain", "BothGain", "MLPDegradationDriven", "ControlExplained", "NoGain"]}
    for row in gap_rows:
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        carrier = str(row.get("carrier", ""))
        eps_row = lookup.get((dataset, seed, carrier)) or lookup.get((dataset, seed, carrier.replace("D", "D-")))
        if eps_row:
            epsilon = finite_float(eps_row.get("epsilon_row"), 0.0) or 0.0
            eps_source = str(eps_row.get("epsilon_source"))
        else:
            epsilon, eps_source = control_bootstrap_epsilon(dataset, seed)
        epsilon = max(float(epsilon), 1.0e-6)
        delta_mlp = finite_float(row.get("Delta_MLP_NLL"), 0.0) or 0.0
        delta_kan = finite_float(row.get("Delta_KAN_NLL"), 0.0) or 0.0
        gap_red = finite_float(row.get("GapReduction_NLL"), 0.0) or 0.0
        mlp_ctrl = finite_float(row.get("MLP_FU_vs_best_control_NLL_delta"), math.inf) or math.inf
        kan_ctrl = finite_float(row.get("KAN_FU_vs_best_control_NLL_delta"), math.inf) or math.inf
        if delta_kan < -epsilon and delta_mlp >= -epsilon and kan_ctrl < -epsilon:
            klass = "TrueKANGain"
        elif delta_kan < -epsilon and delta_mlp < -epsilon and gap_red > epsilon and kan_ctrl < -epsilon:
            klass = "BothGain"
        elif delta_mlp > epsilon and gap_red > 0.0:
            klass = "MLPDegradationDriven"
        elif kan_ctrl >= -epsilon:
            klass = "ControlExplained"
        else:
            klass = "NoGain"
        classes[klass] += 1
        out_rows.append(
            {
                **row,
                "epsilon_global": row.get("repeat_noise_epsilon", ""),
                "epsilon_row": epsilon,
                "epsilon_source": eps_source,
                "gap_reduction_class_v22_35": klass,
                "MLP_FU_vs_best_control_NLL_delta": mlp_ctrl if math.isfinite(mlp_ctrl) else "",
                "KAN_FU_vs_best_control_NLL_delta": kan_ctrl if math.isfinite(kan_ctrl) else "",
                "source_artifact": row.get("source_artifact", "results/v22_33/v22_33_gap_truth_matrix.csv"),
            }
        )
    hard_rows = len(out_rows)
    summary = {
        "source_artifact": "results/v22_33/v22_33_gap_truth_matrix.csv",
        "hard_rows": hard_rows,
        "epsilon_global": (read_rows(V22_34 / "v22_34_gap_truth_summary.csv") or [{}])[0].get("repeat_noise_epsilon", ""),
        "row_local_direct_noise_rows": len(read_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv")),
        "TrueKANGain_rows": classes["TrueKANGain"],
        "BothGain_rows": classes["BothGain"],
        "MLPDegradationDriven_rows": classes["MLPDegradationDriven"],
        "ControlExplained_rows": classes["ControlExplained"],
        "NoGain_rows": classes["NoGain"],
        "TrueKANGain_plus_BothGain_rate": rate(classes["TrueKANGain"] + classes["BothGain"], hard_rows),
        "MLPDegradationDriven_rate": rate(classes["MLPDegradationDriven"], hard_rows),
        "ControlExplained_rate": rate(classes["ControlExplained"], hard_rows),
        "exploration_gate_pass": int(hard_rows > 0 and rate(classes["TrueKANGain"] + classes["BothGain"], hard_rows) >= 0.25 and rate(classes["ControlExplained"], hard_rows) <= 0.50 and rate(classes["MLPDegradationDriven"], hard_rows) <= 0.25),
        "official_candidate_gate_pass": int(hard_rows > 0 and rate(classes["TrueKANGain"] + classes["BothGain"], hard_rows) >= 0.60 and rate(classes["ControlExplained"], hard_rows) <= 0.20 and rate(classes["MLPDegradationDriven"], hard_rows) <= 0.10),
        "blocker_action": "If TrueKANGain remains zero, do not run full superiority; continue target/control decomposition.",
    }
    write_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_matrix.csv", out_rows)
    write_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv", [summary])
    append_exec(
        "reclassify v22.33 gap truth rows with v22.35 row-local epsilon",
        task_id="B_gap_truth_recalibrated",
        status="pass" if out_rows else "warn",
        gpu="0",
        files="results/v22_35/v22_35_gap_truth_recalibrated_matrix.csv, results/v22_35/v22_35_gap_truth_recalibrated_summary.csv",
        note=f"TrueKANGain={summary['TrueKANGain_rows']}; BothGain={summary['BothGain_rows']}; ControlExplained={summary['ControlExplained_rows']}",
    )
    return summary


def branch_delta(row: dict[str, Any]) -> float | None:
    for key in ["NLL_delta_H400", "NLL_delta_H200", "NLL_delta_H100", "NLL_delta_vs_base", "real_NLL_delta"]:
        v = finite_float(row.get(key))
        if v is not None:
            return v
    return None


def stage_c_selector_retest() -> dict[str, Any]:
    branch = read_rows(V22_34 / "v22_34_T1_C9C10_repair_branch_matrix.csv")
    out_branch: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    direct_rows = [r for r in branch if int_flag(r.get("is_control_branch")) == 0]
    for row in branch:
        if int_flag(row.get("is_control_branch")):
            control_rows.append(
                {
                    "selector_name": row.get("selector_name", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "checkpoint_step": row.get("checkpoint_step", ""),
                    "control_level": row.get("control_level", ""),
                    "control_delta_NLL": row.get("NLL_delta_vs_base", ""),
                    "sharpness_delta_control": "",
                    "margin_delta_control": row.get("margin_q10_delta_vs_base", ""),
                    "source_artifact": "results/v22_34/v22_34_T1_C9C10_repair_branch_matrix.csv",
                }
            )
    for row in direct_rows:
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        horizon = str(row.get("branch_H", ""))
        epsilon, eps_source = control_bootstrap_epsilon(dataset, seed, horizon)
        nll = finite_float(row.get("NLL_delta_vs_base"), 0.0) or 0.0
        out_branch.append(
            {
                "selector_name": row.get("selector_name", ""),
                "dataset": dataset,
                "seed": seed,
                "checkpoint_step": row.get("checkpoint_step", ""),
                "subspace_rank": "",
                "cohort_positive_fraction": "",
                "temporal_eigenspace_overlap": "",
                "NLL_delta_H100": "" if horizon != "100" else nll,
                "NLL_delta_H200": "" if horizon != "200" else nll,
                "NLL_delta_H400": "" if horizon != "400" else nll,
                "row_local_epsilon": epsilon,
                "epsilon_source": eps_source,
                "beats_base": int(nll < -epsilon),
                "beats_L1": row.get("beats_L1_control", ""),
                "beats_L2": row.get("beats_L2_control", ""),
                "beats_L3": row.get("beats_L3_control", ""),
                "beats_L4": row.get("beats_L4_control", ""),
                "beats_L5": row.get("beats_L5_control", ""),
                "beats_L6": "",
                "beats_L7": "",
                "real_minus_best_control_NLL": row.get("real_minus_best_control_delta", ""),
                "accuracy_delta": row.get("accuracy_delta_vs_base", ""),
                "ECE_delta": row.get("ECE_delta_vs_base", ""),
                "Brier_delta": row.get("Brier_delta_vs_base", ""),
                "tail_q95_delta": row.get("tail_q95_delta", ""),
                "tail_q99_delta": row.get("tail_q99_delta", ""),
                "margin_q10_delta": row.get("margin_q10_delta_vs_base", ""),
                "hard_slice_NLL_delta": row.get("tail_q99_delta", ""),
                "sharpness_delta": "",
                "selected_candidate_label": "",
                "uses_test_direction_selection": row.get("uses_test_direction_selection", ""),
                "uses_future_direction": row.get("uses_future_direction", ""),
                "source_artifact": "results/v22_34/v22_34_T1_C9C10_repair_branch_matrix.csv",
            }
        )
    # Explicitly record fixed selectors not rerun in this stage.
    for name in ["C4_leave_cohort_stable_direction", "C6_hard_slice_direction", "C7_curvature_safe_direction", "C11_edge_safe_control_orthogonal_direction", "C12_influence_surrogate_direction_diagnostic_only"]:
        if not any(r.get("selector_name") == name for r in out_branch):
            out_branch.append(
                {
                    "selector_name": name,
                    "status": "not_run_direct_v22_35",
                    "reason": "v22.35 direct rerun focused on C9/C10 after v22.34 blocker; historical C4/C6/C7 readback remains upstream, C11/C12 require a passing target/control decomposition before official runtime.",
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                    "source_artifact": "gate_blocked_or_historical_selector_not_promoted",
                }
            )
    write_rows(OUT_ROOT / "v22_35_T1_selector_branch_matrix.csv", out_branch)
    write_rows(OUT_ROOT / "v22_35_T1_control_hierarchy_matrix.csv", control_rows)
    real = [r for r in out_branch if finite_float(r.get("row_local_epsilon")) is not None and r.get("selector_name", "").startswith(("C9", "C10"))]
    n = len(real)
    improve = sum(1 for r in real if branch_delta(r) is not None and (branch_delta(r) or 0.0) < -(finite_float(r.get("row_local_epsilon"), 0.0) or 0.0))
    beats_l4 = sum(1 for r in real if int_flag(r.get("beats_L4")))
    beats_l5 = sum(1 for r in real if int_flag(r.get("beats_L5")))
    summary = {
        "selector_rows": len(out_branch),
        "direct_C9C10_rows": n,
        "NLL_delta_beyond_row_epsilon_rows": improve,
        "NLL_delta_beyond_row_epsilon_rate": rate(improve, n),
        "beats_L4_rate": rate(beats_l4, n),
        "beats_L5_rate": rate(beats_l5, n),
        "exploration_gate_pass": int(n > 0 and rate(improve, n) >= 0.55 and rate(beats_l4, n) >= 0.50),
        "official_candidate_gate_pass": int(n > 0 and rate(improve, n) >= 0.65 and rate(beats_l4, n) >= 0.60 and rate(beats_l5, n) >= 0.55),
    }
    write_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv", [summary])
    append_exec(
        "retest C9/C10 selector branch rows with row-local/control-bootstrap epsilon",
        task_id="C_T1_selector_retest",
        status="pass",
        gpu="1",
        files="results/v22_35/v22_35_T1_selector_branch_matrix.csv, results/v22_35/v22_35_T1_control_hierarchy_matrix.csv, results/v22_35/v22_35_T1_selector_summary.csv",
        note=f"exploration_gate_pass={summary['exploration_gate_pass']}; official_candidate_gate_pass={summary['official_candidate_gate_pass']}",
    )
    return summary


def tier2_spec(dataset: str) -> dict[str, Any] | None:
    key = str(dataset).strip().lower().replace("-", "_").replace(" ", "_")
    mapped = TIER2_TABULAR_ALIASES.get(key)
    return TIER2_TABULAR_SPECS.get(mapped or "")


def _attempts_summary(attempts: list[dict[str, Any]]) -> str:
    parts = []
    for row in attempts:
        label = row.get("source_kind", "")
        status = row.get("status", "")
        err = str(row.get("error_message", "")).replace("\n", " ")[:160]
        parts.append(f"{label}:{status}:{err}")
    return " | ".join(parts)


def _tier2_cache_path(spec: dict[str, Any], url: str) -> Path:
    suffix = ".zip" if url.lower().endswith(".zip") else ".data"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return ROOT / "data" / "v22_35_tier2" / f"uci_{spec['uci_id']}_{digest}{suffix}"


def _read_tabular_frame_from_path(path: Path, fmt: str) -> tuple[Any, Any]:
    import pandas as pd

    if fmt == "csv_target_last":
        df = pd.read_csv(path, header=None)
        return df.iloc[:, :-1], df.iloc[:, -1]
    if fmt == "zip_arff_target_last":
        import io
        import zipfile
        from scipy.io import arff

        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".arff")]
            if not names:
                raise ValueError(f"zip has no .arff member: {path}")
            with zf.open(names[0]) as f:
                raw = f.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin1")
                data, _meta = arff.loadarff(io.StringIO(text))
        df = pd.DataFrame(data)
        return df.iloc[:, :-1], df.iloc[:, -1]
    raise ValueError(f"unsupported direct_format={fmt}")


def _coerce_tabular_xy(x_obj: Any, y_obj: Any) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    x_df = pd.DataFrame(x_obj).copy()
    y_frame = pd.DataFrame(y_obj)
    y_series = y_frame.iloc[:, 0] if len(y_frame.columns) else pd.Series(y_obj)
    y_series = y_series.map(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
    for col in x_df.columns:
        x_df[col] = x_df[col].map(lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
    non_numeric = [col for col in x_df.columns if not pd.api.types.is_numeric_dtype(x_df[col])]
    if non_numeric:
        x_df = pd.get_dummies(x_df, columns=non_numeric, dummy_na=True)
    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True)).fillna(0.0)
    x = StandardScaler().fit_transform(x_df.to_numpy(dtype="float32")).astype("float32")
    y = LabelEncoder().fit_transform(y_series.astype(str).to_numpy()).astype("int64")
    meta = {
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(set(y.tolist()))),
        "encoded_non_numeric_feature_count": int(len(non_numeric)),
    }
    return x, y, meta


def load_tier2_tabular_arrays(dataset: str, *, allow_download: bool) -> tuple[Any, Any, dict[str, Any]]:
    spec = tier2_spec(dataset)
    if spec is None:
        raise TaskUnavailableError(f"no v22.35 Tier2 tabular spec for dataset={dataset}", [])
    attempts: list[dict[str, Any]] = []

    for url in spec.get("direct_urls", []):
        cache_path = _tier2_cache_path(spec, str(url))
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                x_obj, y_obj = _read_tabular_frame_from_path(cache_path, str(spec.get("direct_format", "")))
                x, y, meta = _coerce_tabular_xy(x_obj, y_obj)
                meta.update(
                    {
                        "dataset": spec["canonical"],
                        "source_kind": "local_cache_direct_uci",
                        "source_url": str(url),
                        "official_url": spec.get("official_url", ""),
                        "uci_id": spec.get("uci_id", ""),
                        "cache_path": str(cache_path.relative_to(ROOT)),
                        "availability_attempts": _attempts_summary(attempts),
                    }
                )
                return x, y, meta
            except Exception as exc:
                attempts.append(
                    {
                        "source_kind": "local_cache_direct_uci",
                        "status": "error",
                        "source_url": str(url),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
        if not allow_download:
            attempts.append(
                {
                    "source_kind": "direct_uci_download",
                    "status": "skipped_no_download",
                    "source_url": str(url),
                    "error_type": "",
                    "error_message": "tier2_download disabled and cache missing",
                }
            )
            continue
        try:
            import urllib.request

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(str(url), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                cache_path.write_bytes(resp.read())
            x_obj, y_obj = _read_tabular_frame_from_path(cache_path, str(spec.get("direct_format", "")))
            x, y, meta = _coerce_tabular_xy(x_obj, y_obj)
            attempts.append({"source_kind": "direct_uci_download", "status": "success", "source_url": str(url)})
            meta.update(
                {
                    "dataset": spec["canonical"],
                    "source_kind": "direct_uci_download",
                    "source_url": str(url),
                    "official_url": spec.get("official_url", ""),
                    "uci_id": spec.get("uci_id", ""),
                    "cache_path": str(cache_path.relative_to(ROOT)),
                    "availability_attempts": _attempts_summary(attempts),
                }
            )
            return x, y, meta
        except Exception as exc:
            attempts.append(
                {
                    "source_kind": "direct_uci_download",
                    "status": "error",
                    "source_url": str(url),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            try:
                curl_cmd = [
                    "curl",
                    "-L",
                    "--http1.1",
                    "-A",
                    "Mozilla/5.0",
                    "--max-time",
                    "60",
                    "--retry",
                    "2",
                    "--retry-delay",
                    "2",
                    "-o",
                    str(cache_path),
                    str(url),
                ]
                proc = subprocess.run(curl_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=90)
                if proc.returncode != 0:
                    raise RuntimeError((proc.stderr or proc.stdout or f"curl exit {proc.returncode}")[:500])
                if not cache_path.exists() or cache_path.stat().st_size <= 0:
                    raise RuntimeError("curl completed but cache file is empty")
                x_obj, y_obj = _read_tabular_frame_from_path(cache_path, str(spec.get("direct_format", "")))
                x, y, meta = _coerce_tabular_xy(x_obj, y_obj)
                attempts.append({"source_kind": "direct_uci_curl_download", "status": "success", "source_url": str(url)})
                meta.update(
                    {
                        "dataset": spec["canonical"],
                        "source_kind": "direct_uci_curl_download",
                        "source_url": str(url),
                        "official_url": spec.get("official_url", ""),
                        "uci_id": spec.get("uci_id", ""),
                        "cache_path": str(cache_path.relative_to(ROOT)),
                        "availability_attempts": _attempts_summary(attempts),
                    }
                )
                return x, y, meta
            except Exception as curl_exc:
                attempts.append(
                    {
                        "source_kind": "direct_uci_curl_download",
                        "status": "error",
                        "source_url": str(url),
                        "error_type": type(curl_exc).__name__,
                        "error_message": str(curl_exc),
                    }
                )

    if allow_download:
        try:
            from ucimlrepo import fetch_ucirepo

            ds = fetch_ucirepo(id=int(spec["uci_id"]))
            x, y, meta = _coerce_tabular_xy(ds.data.features, ds.data.targets)
            attempts.append({"source_kind": "ucimlrepo", "status": "success", "source_url": str(spec.get("official_url", ""))})
            meta.update(
                {
                    "dataset": spec["canonical"],
                    "source_kind": "ucimlrepo",
                    "source_url": str(spec.get("official_url", "")),
                    "official_url": spec.get("official_url", ""),
                    "uci_id": spec.get("uci_id", ""),
                    "cache_path": "",
                    "availability_attempts": _attempts_summary(attempts),
                }
            )
            return x, y, meta
        except Exception as exc:
            attempts.append(
                {
                    "source_kind": "ucimlrepo",
                    "status": "error",
                    "source_url": str(spec.get("official_url", "")),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if allow_download:
        for name in spec.get("openml_names", []):
            try:
                from sklearn.datasets import fetch_openml

                ds = fetch_openml(name=str(name), version=1, as_frame=False, parser="auto", data_home=str(ROOT / "data" / "openml"))
                x, y, meta = _coerce_tabular_xy(ds.data, ds.target)
                attempts.append({"source_kind": "openml", "status": "success", "source_url": str(name)})
                meta.update(
                    {
                        "dataset": spec["canonical"],
                        "source_kind": "openml",
                        "source_url": str(name),
                        "official_url": spec.get("official_url", ""),
                        "uci_id": spec.get("uci_id", ""),
                        "cache_path": str(ROOT / "data" / "openml"),
                        "availability_attempts": _attempts_summary(attempts),
                    }
                )
                return x, y, meta
            except Exception as exc:
                attempts.append(
                    {
                        "source_kind": "openml",
                        "status": "error",
                        "source_url": str(name),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

    raise TaskUnavailableError(f"Tier2 tabular dataset {dataset} unavailable after {len(attempts)} source attempts", attempts)


def make_v22_35_c11_loaders(
    dataset: str,
    train_size: int,
    test_size: int,
    batch_size: int,
    seed: int,
    *,
    tier2_download: bool,
) -> tuple[Any, Any, Any, int, int, Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset

    from experiments.run_v22_30_fidelity_ladder import make_hard_vision_loaders

    if str(dataset).strip().lower() == "wine":
        train_loader, held_loader, test_loader, input_dim, output_dim, x_stats = make_hard_vision_loaders(
            dataset,
            train_size,
            test_size,
            batch_size,
            seed,
            download=False,
        )
        meta = {
            "dataset": "Wine",
            "task_tier": "Tier2_tabular",
            "source_kind": "sklearn.datasets.load_wine",
            "source_url": "sklearn.datasets.load_wine",
            "official_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html",
            "uci_id": "",
            "n_rows": "",
            "n_features": input_dim,
            "n_classes": output_dim,
            "encoded_non_numeric_feature_count": 0,
            "availability_attempts": "local sklearn dataset:success",
        }
        return train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta

    spec = tier2_spec(dataset)
    if spec is None:
        train_loader, held_loader, test_loader, input_dim, output_dim, x_stats = make_hard_vision_loaders(
            dataset,
            train_size,
            test_size,
            batch_size,
            seed,
            download=False,
        )
        meta = {
            "dataset": dataset,
            "task_tier": "Tier1_hard_vision",
            "source_kind": "torchvision_or_existing_loader",
            "source_url": "",
            "official_url": "",
            "uci_id": "",
            "n_rows": "",
            "n_features": input_dim,
            "n_classes": output_dim,
            "encoded_non_numeric_feature_count": "",
            "availability_attempts": "existing make_hard_vision_loaders:success",
        }
        return train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta

    x, y, meta = load_tier2_tabular_arrays(dataset, allow_download=bool(tier2_download))
    xs = torch.tensor(x, dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(xs, ys)
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(ds), generator=gen).tolist()
    n_test = min(int(test_size), max(1, len(ds) // 5))
    test_idx = perm[:n_test]
    remaining = perm[n_test:]
    n_train = min(int(train_size), max(1, len(remaining)))
    train_idx = remaining[:n_train]
    held_idx = remaining[n_train : n_train + min(int(test_size), max(1, len(remaining) - n_train))]
    if not held_idx:
        held_idx = train_idx[: min(len(train_idx), max(1, n_test))]
    train_batch = max(1, min(int(batch_size), len(train_idx)))
    held_batch = max(1, min(int(batch_size), len(held_idx)))
    test_batch = max(1, min(int(batch_size), len(test_idx)))
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=train_batch, shuffle=True, generator=gen, drop_last=False)
    held_loader = DataLoader(Subset(ds, held_idx), batch_size=held_batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=test_batch, shuffle=False, drop_last=False)
    meta.update(
        {
            "task_tier": "Tier2_tabular",
            "effective_train_rows": len(train_idx),
            "effective_held_rows": len(held_idx),
            "effective_test_rows": len(test_idx),
        }
    )
    return train_loader, held_loader, test_loader, int(xs.shape[1]), int(len(set(y.tolist()))), xs[train_idx].float(), meta


def stage_c11_edge_safe_selector(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from experiments.run_v22_30_fidelity_ladder import (
        apply_layer_direction,
        collect_fixed_examples,
        cohort_signal_for_layer,
        direction_control,
        make_mlp,
        principal_overlap,
        train_adamw,
        train_branch,
    )

    device = torch_device(args.c11_device)
    selector_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    c10_selector_rows: list[dict[str, Any]] = []
    c10_branch_rows: list[dict[str, Any]] = []
    c10_audit_rows: list[dict[str, Any]] = []
    c11l_selector_rows: list[dict[str, Any]] = []
    c11l_branch_rows: list[dict[str, Any]] = []
    c11l_audit_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []

    def cos_abs(a: Any, b: Any) -> float:
        aa = a.detach().float().reshape(-1)
        bb = b.detach().float().reshape(-1).to(aa.device)
        return float(torch.dot(aa, bb).abs().div(torch.linalg.vector_norm(aa).clamp_min(1.0e-12) * torch.linalg.vector_norm(bb).clamp_min(1.0e-12)).item())

    def held_margin_q10(model_obj: Any, x: Any, y: Any) -> float:
        model_obj.eval()
        with torch.no_grad():
            logits = model_obj(x).float()
            true = logits.gather(1, y.long().view(-1, 1)).squeeze(1)
            mask = F.one_hot(y.long(), num_classes=logits.shape[1]).bool()
            other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
            return float(torch.quantile((true - other).float(), 0.10).item())

    def layer_train_gradient_direction(model_obj: Any, layer_name: str, x: Any, y: Any, reference_direction: Any) -> Any:
        was_training = bool(model_obj.training)
        model_obj.train()
        model_obj.zero_grad(set_to_none=True)
        logits = model_obj(x).float()
        loss = F.cross_entropy(logits, y.long())
        loss.backward()
        param = dict(model_obj.named_parameters())[layer_name]
        grad = param.grad
        if grad is None:
            model_obj.zero_grad(set_to_none=True)
            model_obj.train(was_training)
            return direction_control(reference_direction, "random", 77_001)
        out = grad.detach().reshape(-1).clone()
        model_obj.zero_grad(set_to_none=True)
        model_obj.train(was_training)
        norm = out.norm().clamp_min(1.0e-12)
        return out * reference_direction.norm().clamp_min(1.0e-12) / norm

    def evaluate_reference_hard_slice_nll(current_model: Any, reference_model: Any, loader: Any, fraction: float) -> dict[str, Any]:
        current_model.eval()
        reference_model.eval()
        current_losses = []
        reference_margins = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                current_logits = current_model(xb).float()
                reference_logits = reference_model(xb).float()
                current_losses.append(F.cross_entropy(current_logits, yb, reduction="none").detach().cpu())
                true = reference_logits.gather(1, yb.view(-1, 1)).squeeze(1)
                mask = F.one_hot(yb, num_classes=reference_logits.shape[1]).bool()
                other = reference_logits.masked_fill(mask, float("-inf")).max(dim=1).values
                reference_margins.append((true - other).detach().float().cpu())
        if not current_losses:
            return {"hard_slice_NLL": "", "hard_slice_count": 0, "hard_slice_fraction": ""}
        losses = torch.cat(current_losses).float()
        margins = torch.cat(reference_margins).float()
        k = max(1, min(int(losses.numel()), int(math.ceil(float(fraction) * float(losses.numel())))))
        hard_idx = torch.topk(-margins, k=k, largest=True).indices
        return {
            "hard_slice_NLL": float(losses[hard_idx].mean().item()),
            "hard_slice_count": int(k),
            "hard_slice_fraction": float(k) / float(losses.numel()),
        }

    def held_cohort_delta_stats(base_model: Any, trial_model: Any, x: Any, y: Any, cohorts: int) -> dict[str, Any]:
        n = int(y.numel())
        if n <= 0:
            return {
                "held_cohort_CE_delta_mean": "",
                "held_cohort_CE_delta_std": "",
                "held_cohort_CE_nonincrease_rate": "",
                "held_cohort_margin_q10_delta_mean": "",
            }
        chunks = max(1, min(int(cohorts), n))
        deltas: list[float] = []
        margin_deltas: list[float] = []
        base_model.eval()
        trial_model.eval()
        with torch.no_grad():
            for idx in torch.chunk(torch.arange(n, device=x.device), chunks):
                if int(idx.numel()) == 0:
                    continue
                xi = x[idx]
                yi = y[idx]
                base_logits = base_model(xi).float()
                trial_logits = trial_model(xi).float()
                base_ce_i = float(F.cross_entropy(base_logits, yi.long()).item())
                trial_ce_i = float(F.cross_entropy(trial_logits, yi.long()).item())
                deltas.append(trial_ce_i - base_ce_i)
                base_true = base_logits.gather(1, yi.long().view(-1, 1)).squeeze(1)
                trial_true = trial_logits.gather(1, yi.long().view(-1, 1)).squeeze(1)
                base_mask = F.one_hot(yi.long(), num_classes=base_logits.shape[1]).bool()
                trial_mask = F.one_hot(yi.long(), num_classes=trial_logits.shape[1]).bool()
                base_other = base_logits.masked_fill(base_mask, float("-inf")).max(dim=1).values
                trial_other = trial_logits.masked_fill(trial_mask, float("-inf")).max(dim=1).values
                base_margin = torch.quantile((base_true - base_other).float(), 0.10)
                trial_margin = torch.quantile((trial_true - trial_other).float(), 0.10)
                margin_deltas.append(float((trial_margin - base_margin).item()))
        if not deltas:
            return {
                "held_cohort_CE_delta_mean": "",
                "held_cohort_CE_delta_std": "",
                "held_cohort_CE_nonincrease_rate": "",
                "held_cohort_margin_q10_delta_mean": "",
            }
        mean_delta = sum(deltas) / len(deltas)
        std_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / len(deltas))
        return {
            "held_cohort_CE_delta_mean": mean_delta,
            "held_cohort_CE_delta_std": std_delta,
            "held_cohort_CE_nonincrease_rate": sum(1 for d in deltas if d <= float(args.c11_held_ce_tolerance)) / len(deltas),
            "held_cohort_margin_q10_delta_mean": sum(margin_deltas) / len(margin_deltas) if margin_deltas else "",
        }

    for dataset in split_csv(args.c11_datasets):
        for seed in split_csv(args.c11_seeds, int):
            try:
                train_loader, held_loader, test_loader, input_dim, output_dim, _x_stats, task_meta = make_v22_35_c11_loaders(
                    str(dataset),
                    int(args.c11_train_size),
                    int(args.c11_test_size),
                    int(args.batch_size),
                    int(seed),
                    tier2_download=bool(int(args.tier2_download)),
                )
                availability_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "available",
                        "task_tier": task_meta.get("task_tier", ""),
                        "source_kind": task_meta.get("source_kind", ""),
                        "source_url": task_meta.get("source_url", ""),
                        "official_url": task_meta.get("official_url", ""),
                        "uci_id": task_meta.get("uci_id", ""),
                        "n_rows": task_meta.get("n_rows", ""),
                        "n_features": task_meta.get("n_features", input_dim),
                        "n_classes": task_meta.get("n_classes", output_dim),
                        "effective_train_rows": task_meta.get("effective_train_rows", ""),
                        "effective_held_rows": task_meta.get("effective_held_rows", ""),
                        "effective_test_rows": task_meta.get("effective_test_rows", ""),
                        "encoded_non_numeric_feature_count": task_meta.get("encoded_non_numeric_feature_count", ""),
                        "availability_attempts": task_meta.get("availability_attempts", ""),
                        "source_artifact": "direct_v22_35_C11_edge_safe_control_orthogonal",
                    }
                )
            except Exception as exc:
                attempts = getattr(exc, "attempts", [])
                spec = tier2_spec(str(dataset))
                availability_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "status": "task_unavailable",
                        "task_tier": "Tier2_tabular" if spec is not None else "unknown_or_existing_loader_failed",
                        "source_kind": "tier2_loader_attempt_chain" if spec is not None else "existing_loader",
                        "source_url": "",
                        "official_url": (spec or {}).get("official_url", ""),
                        "uci_id": (spec or {}).get("uci_id", ""),
                        "n_rows": "",
                        "n_features": "",
                        "n_classes": "",
                        "effective_train_rows": "",
                        "effective_held_rows": "",
                        "effective_test_rows": "",
                        "encoded_non_numeric_feature_count": "",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "availability_attempts": _attempts_summary(attempts),
                        "source_artifact": "direct_v22_35_C11_edge_safe_control_orthogonal",
                    }
                )
                selector_rows.append(
                    {
                        "selector_name": "C11_edge_safe_control_orthogonal_direction",
                        "dataset": dataset,
                        "seed": seed,
                        "status": "task_unavailable",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "uses_test_direction_selection": 0,
                        "uses_future_direction": 0,
                        "source_artifact": "direct_v22_35_C11_edge_safe_control_orthogonal",
                    }
                )
                continue

            torch.manual_seed(263_500 + int(seed))
            init_model = make_mlp(input_dim, output_dim, int(args.hidden), int(seed) + 2635, device).to(device)
            model = copy.deepcopy(init_model).to(device)
            train_adamw(
                model,
                train_loader,
                test_loader,
                device,
                output_dim,
                steps=int(args.c11_pretrain_steps),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            )
            xb, yb = collect_fixed_examples(
                train_loader,
                max(int(args.c11_examples), int(args.c11_signal_cohorts) * int(args.c11_cohort_size)),
                device,
            )
            held_x, held_y = collect_fixed_examples(held_loader, int(args.c11_examples), device)
            layer = "w2"
            sig, _cohorts = cohort_signal_for_layer(
                model,
                xb,
                yb,
                layer,
                cohorts=int(args.c11_signal_cohorts),
                cohort_size=int(args.c11_cohort_size),
                rank_cap=int(args.c11_rank_cap),
                sketch_dim=int(args.c11_sketch_dim),
                seed=int(seed) + 11_035,
            )
            prev_sig, _ = cohort_signal_for_layer(
                init_model,
                xb,
                yb,
                layer,
                cohorts=int(args.c11_signal_cohorts),
                cohort_size=int(args.c11_cohort_size),
                rank_cap=int(args.c11_rank_cap),
                sketch_dim=int(args.c11_sketch_dim),
                seed=int(seed) + 11_035,
            )
            basis = sig.get("basis_sketch")
            proj = sig.get("proj")
            direction = sig["direction"]
            pool: list[tuple[str, Any, str]] = [("raw_top_signal", direction, "raw_top_signal")]
            if basis is not None and proj is not None and int(basis.numel()) > 0:
                k = int(basis.shape[1])
                for j in range(k):
                    coeff = torch.zeros(k, device=direction.device)
                    coeff[j] = 1.0
                    vec = proj @ (basis @ coeff)
                    vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                    pool.append((f"basis_{j}_plus", vec, "basis_axis"))
                    pool.append((f"basis_{j}_minus", -vec, "basis_axis"))
                gen = torch.Generator(device=direction.device).manual_seed(1_103_500 + int(seed))
                while len(pool) < max(1, int(args.c11_candidate_count)):
                    coeff = torch.randn(k, device=direction.device, generator=gen)
                    vec = proj @ (basis @ coeff)
                    vec = vec * direction.norm().clamp_min(1.0e-12) / vec.norm().clamp_min(1.0e-12)
                    pool.append((f"random_combo_{len(pool)}", vec, "same_subspace_random_combo"))
            pool = pool[: max(1, int(args.c11_candidate_count))]

            controls_for_selection = [
                direction_control(direction, "same_subspace", int(seed) + 11_401, basis, proj),
                direction_control(direction, "same_subspace", int(seed) + 11_402, basis, proj),
                direction_control(direction, "random", int(seed) + 11_403),
            ]
            model.eval()
            with torch.no_grad():
                base_ce = float(F.cross_entropy(model(held_x).float(), held_y.long()).item())
            base_margin = held_margin_q10(model, held_x, held_y)
            local_audit: list[dict[str, Any]] = []
            for label, cand, source in pool:
                trial = copy.deepcopy(model).to(device)
                apply_layer_direction(trial, layer, cand, float(args.c11_branch_trust))
                trial.eval()
                with torch.no_grad():
                    held_ce_after = float(F.cross_entropy(trial(held_x).float(), held_y.long()).item())
                margin_after = held_margin_q10(trial, held_x, held_y)
                plus_ce = held_ce_after
                trial_minus = copy.deepcopy(model).to(device)
                apply_layer_direction(trial_minus, layer, -cand, float(args.c11_branch_trust))
                trial_minus.eval()
                with torch.no_grad():
                    minus_ce = float(F.cross_entropy(trial_minus(held_x).float(), held_y.long()).item())
                curvature = max(0.0, (plus_ce + minus_ce - 2.0 * base_ce) / max(float(args.c11_branch_trust) ** 2, 1.0e-12))
                cohort_stats = held_cohort_delta_stats(model, trial, held_x, held_y, int(args.c11_signal_cohorts))
                row = {
                    "selector_name": "C11_edge_safe_control_orthogonal_direction",
                    "dataset": dataset,
                    "seed": seed,
                    "task_tier": task_meta.get("task_tier", ""),
                    "dataset_source_kind": task_meta.get("source_kind", ""),
                    "dataset_source_url": task_meta.get("source_url", ""),
                    "dataset_uci_id": task_meta.get("uci_id", ""),
                    "candidate_label": label,
                    "candidate_source": source,
                    "held_CE_delta": held_ce_after - base_ce,
                    "held_margin_q10_delta": margin_after - base_margin,
                    "symmetric_curvature_proxy": curvature,
                    "max_abs_cos_to_selection_controls": max(cos_abs(cand, ctrl) for ctrl in controls_for_selection),
                    **cohort_stats,
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                    "source_artifact": "direct_v22_35_C11_edge_safe_control_orthogonal",
                }
                local_audit.append(row)
            audit_rows.extend(local_audit)
            if not local_audit:
                continue
            median_cos = sorted(float(r["max_abs_cos_to_selection_controls"]) for r in local_audit)[len(local_audit) // 2]
            eligible = [
                r
                for r in local_audit
                if float(r["held_CE_delta"]) <= float(args.c11_held_ce_tolerance)
                and float(r["held_margin_q10_delta"]) >= -float(args.c11_margin_debt_tolerance)
                and float(r["max_abs_cos_to_selection_controls"]) <= median_cos
            ]
            if eligible:
                selected_audit = min(
                    eligible,
                    key=lambda r: (
                        float(r["held_CE_delta"]) + float(args.c11_curvature_weight) * float(r["symmetric_curvature_proxy"]),
                        -float(r["held_margin_q10_delta"]),
                        float(r["max_abs_cos_to_selection_controls"]),
                    ),
                )
                selection_status = "edge_safe_control_orthogonal_selected"
            else:
                selected_audit = min(
                    local_audit,
                    key=lambda r: (
                        float(r["held_CE_delta"]) + float(args.c11_curvature_weight) * float(r["symmetric_curvature_proxy"]),
                        float(r["max_abs_cos_to_selection_controls"]),
                        -float(r["held_margin_q10_delta"]),
                    ),
                )
                selection_status = "no_edge_safe_candidate_min_curvature_ce_used_for_diagnostic"
            c10_eligible = [
                r
                for r in local_audit
                if float(r["held_CE_delta"]) <= float(args.c11_held_ce_tolerance)
                and float(r["held_margin_q10_delta"]) >= -float(args.c11_margin_debt_tolerance)
            ]
            if c10_eligible:
                c10_selected_audit = min(
                    c10_eligible,
                    key=lambda r: (
                        -float(r["held_margin_q10_delta"]),
                        float(r["held_CE_delta"]),
                        float(r["symmetric_curvature_proxy"]),
                        float(r["max_abs_cos_to_selection_controls"]),
                    ),
                )
                c10_selection_status = "margin_NLL_constrained_selected"
            else:
                c10_selected_audit = min(
                    local_audit,
                    key=lambda r: (
                        float(r["held_CE_delta"]),
                        -float(r["held_margin_q10_delta"]),
                        float(r["symmetric_curvature_proxy"]),
                    ),
                )
                c10_selection_status = "no_margin_NLL_safe_candidate_min_CE_used_for_diagnostic"
            c11l_eligible = [
                r
                for r in local_audit
                if (finite_float(r.get("held_cohort_CE_nonincrease_rate"), 0.0) or 0.0) >= 0.75
                and float(r["held_CE_delta"]) <= float(args.c11_held_ce_tolerance)
                and float(r["held_margin_q10_delta"]) >= -float(args.c11_margin_debt_tolerance)
            ]
            if c11l_eligible:
                c11l_selected_audit = min(
                    c11l_eligible,
                    key=lambda r: (
                        float(r["held_cohort_CE_delta_mean"]) + float(r["held_cohort_CE_delta_std"]) + float(args.c11_curvature_weight) * float(r["symmetric_curvature_proxy"]),
                        -float(r["held_cohort_CE_nonincrease_rate"]),
                        -float(r["held_margin_q10_delta"]),
                        float(r["max_abs_cos_to_selection_controls"]),
                    ),
                )
                c11l_selection_status = "cohort_influence_stable_selected"
            else:
                c11l_selected_audit = min(
                    local_audit,
                    key=lambda r: (
                        -(finite_float(r.get("held_cohort_CE_nonincrease_rate"), 0.0) or 0.0),
                        float(r["held_cohort_CE_delta_mean"]) + float(r["held_cohort_CE_delta_std"]),
                        float(r["held_CE_delta"]),
                        -float(r["held_margin_q10_delta"]),
                    ),
                )
                c11l_selection_status = "no_cohort_stable_candidate_best_train_stability_used_for_diagnostic"
            for row in local_audit:
                c10_audit_rows.append(
                    {
                        **row,
                        "selector_name": "C10_repaired_margin_NLL_constrained_direction",
                        "selection_rule": "maximize held margin q10 under held CE non-increase and margin debt constraint; no test/future used",
                        "source_artifact": "direct_v22_35_C10_repaired_margin_NLL_constrained",
                    }
                )
                c11l_audit_rows.append(
                    {
                        **row,
                        "selector_name": "C11_cohort_influence_repair_direction",
                        "selection_rule": "held-train cohort CE non-increase stability with NLL/margin constraint; no test/future used",
                        "source_artifact": "direct_v22_35_C11_cohort_influence_repair",
                    }
                )
            by_label = {label: cand for label, cand, _source in pool}
            selected = by_label.get(str(selected_audit["candidate_label"]), direction)
            c10_selected = by_label.get(str(c10_selected_audit["candidate_label"]), direction)
            c11l_selected = by_label.get(str(c11l_selected_audit["candidate_label"]), direction)

            base_branch_model = copy.deepcopy(model).to(device)
            base_ev = train_branch(
                base_branch_model,
                train_loader,
                test_loader,
                device,
                output_dim,
                int(args.c11_branch_horizon),
                float(args.lr),
                float(args.weight_decay),
            )
            base_hard = evaluate_reference_hard_slice_nll(
                base_branch_model,
                base_branch_model,
                test_loader,
                float(args.hard_fraction),
            )
            l6_bank_candidates = [
                (label, cand)
                for label, cand, source in pool
                if str(label)
                not in {
                    str(selected_audit.get("candidate_label", "")),
                    str(c10_selected_audit.get("candidate_label", "")),
                    str(c11l_selected_audit.get("candidate_label", "")),
                }
                and str(source) in {"basis_axis", "same_subspace_random_combo"}
            ]
            if l6_bank_candidates:
                l6_idx = (int(seed) * 37 + len(str(dataset))) % len(l6_bank_candidates)
                l6_direction = l6_bank_candidates[l6_idx][1]
            else:
                l6_direction = direction_control(direction, "same_subspace", int(seed) + 6, basis, proj)
            l7_direction = layer_train_gradient_direction(model, layer, xb, yb, direction)
            controls = [
                ("L1_isotropic_random", direction_control(direction, "random", int(seed) + 1), "L1"),
                ("L2_same_norm_random", direction_control(direction, "random", int(seed) + 2), "L2"),
                ("L3_same_signal_subspace_random", direction_control(direction, "same_subspace", int(seed) + 3, basis, proj), "L3"),
                ("L4_signflip_same_subspace", -direction, "L4"),
                ("L5_shuffled_cohort_direction", direction_control(direction, "shuffled", int(seed) + 5), "L5"),
                ("L6_same_candidate_basis_bank_random", l6_direction, "L6"),
                ("L7_train_gradient_optimizer_geometry", l7_direction, "L7"),
            ]

            def branch_eval(name: str, cand: Any, is_control: int, control_level: str = "") -> dict[str, Any]:
                branch_model = copy.deepcopy(model).to(device)
                apply_layer_direction(branch_model, layer, cand, float(args.c11_branch_trust))
                ev = train_branch(
                    branch_model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    int(args.c11_branch_horizon),
                    float(args.lr),
                    float(args.weight_decay),
                )
                hard = evaluate_reference_hard_slice_nll(branch_model, base_branch_model, test_loader, float(args.hard_fraction))
                hard_nll = finite_float(hard.get("hard_slice_NLL"))
                base_hard_nll = finite_float(base_hard.get("hard_slice_NLL"))
                return {
                    "selector_name": name,
                    "dataset": dataset,
                    "seed": seed,
                    "task_tier": task_meta.get("task_tier", ""),
                    "dataset_source_kind": task_meta.get("source_kind", ""),
                    "dataset_source_url": task_meta.get("source_url", ""),
                    "dataset_uci_id": task_meta.get("uci_id", ""),
                    "checkpoint_step": f"c11_pretrain_steps_{args.c11_pretrain_steps}",
                    "layer_id": layer,
                    "branch_H": int(args.c11_branch_horizon),
                    "NLL_delta_vs_base": ev["NLL"] - base_ev["NLL"],
                    "accuracy_delta_vs_base": ev["accuracy"] - base_ev["accuracy"],
                    "ECE_delta_vs_base": ev["ECE"] - base_ev["ECE"],
                    "Brier_delta_vs_base": ev["Brier"] - base_ev["Brier"],
                    "tail_q95_delta": ev["tail_q95"] - base_ev["tail_q95"],
                    "tail_q99_delta": ev["tail_q99"] - base_ev["tail_q99"],
                    "hard_slice_NLL": hard_nll if hard_nll is not None else "",
                    "hard_slice_NLL_delta": (hard_nll - base_hard_nll) if hard_nll is not None and base_hard_nll is not None else "",
                    "hard_slice_count": hard.get("hard_slice_count", ""),
                    "hard_slice_fraction": hard.get("hard_slice_fraction", ""),
                    "margin_mean_delta_vs_base": ev.get("margin_mean", math.nan) - base_ev.get("margin_mean", math.nan),
                    "margin_q10_delta_vs_base": ev.get("margin_q10", math.nan) - base_ev.get("margin_q10", math.nan),
                    "margin_q01_delta_vs_base": ev.get("margin_q01", math.nan) - base_ev.get("margin_q01", math.nan),
                    "low_margin_accuracy_delta_vs_base": ev.get("low_margin_accuracy", math.nan) - base_ev.get("low_margin_accuracy", math.nan),
                    "beats_base": int(ev["NLL"] < base_ev["NLL"]),
                    "is_control_branch": is_control,
                    "control_level": control_level,
                    "selection_data_source": "held_train_ce_margin_curvature_and_control_orthogonality",
                    "uses_test_direction_selection": 0,
                    "uses_future_direction": 0,
                    "source_artifact": "direct_v22_35_C11_edge_safe_control_orthogonal",
                    "status": "direct_v22_35_C11_edge_safe_control_orthogonal",
                }

            control_branch_rows = [branch_eval(name, cand, 1, level) for name, cand, level in controls]
            branch_rows.extend(control_branch_rows)
            for row in control_branch_rows:
                control_rows.append(
                    {
                        "selector_name": row.get("selector_name", ""),
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "checkpoint_step": row.get("checkpoint_step", ""),
                        "control_level": row.get("control_level", ""),
                        "control_delta_NLL": row.get("NLL_delta_vs_base", ""),
                        "sharpness_delta_control": "",
                        "margin_delta_control": row.get("margin_q10_delta_vs_base", ""),
                        "source_artifact": "direct_v22_35_C11_edge_safe_control_orthogonal",
                    }
                )
            best_by_level: dict[str, float] = {}
            for row in control_branch_rows:
                val_maybe = finite_float(row.get("NLL_delta_vs_base"))
                val = val_maybe if val_maybe is not None else math.inf
                level = str(row.get("control_level"))
                best_by_level[level] = min(best_by_level.get(level, math.inf), val)
            best_any = min(best_by_level.values()) if best_by_level else math.inf
            real = branch_eval("C11_edge_safe_control_orthogonal_direction", selected, 0, "")
            val_maybe = finite_float(real.get("NLL_delta_vs_base"))
            val = val_maybe if val_maybe is not None else math.inf
            epsilon, eps_source = control_bootstrap_epsilon(str(dataset), str(seed), str(args.c11_branch_horizon))
            real.update(
                {
                    "subspace_rank": int(basis.shape[1]) if basis is not None else "",
                    "cohort_positive_fraction": sig.get("cohort_positive_fraction", ""),
                    "temporal_eigenspace_overlap": principal_overlap(prev_sig["basis_sketch"], sig["basis_sketch"]),
                    "row_local_epsilon": epsilon,
                    "epsilon_source": eps_source,
                    "beats_L1": int(val < best_by_level.get("L1", math.inf)),
                    "beats_L2": int(val < best_by_level.get("L2", math.inf)),
                    "beats_L3": int(val < best_by_level.get("L3", math.inf)),
                    "beats_L4": int(val < best_by_level.get("L4", math.inf)),
                    "beats_L5": int(val < best_by_level.get("L5", math.inf)),
                    "beats_L6": int(val < best_by_level.get("L6", math.inf)),
                    "beats_L7": int(val < best_by_level.get("L7", math.inf)),
                    "real_minus_best_control_NLL": val - best_any,
                    "selected_candidate_label": selected_audit.get("candidate_label", ""),
                    "selection_status": selection_status,
                    "held_CE_delta": selected_audit.get("held_CE_delta", ""),
                    "held_margin_q10_delta": selected_audit.get("held_margin_q10_delta", ""),
                    "symmetric_curvature_proxy": selected_audit.get("symmetric_curvature_proxy", ""),
                    "max_abs_cos_to_selection_controls": selected_audit.get("max_abs_cos_to_selection_controls", ""),
                    "NLL_delta_H100": "" if int(args.c11_branch_horizon) != 100 else val,
                    "NLL_delta_H200": "" if int(args.c11_branch_horizon) != 200 else val,
                    "NLL_delta_H400": "" if int(args.c11_branch_horizon) != 400 else val,
                }
            )
            branch_rows.append(real)
            selector_rows.append(real)
            c10_branch_rows.extend(
                {
                    **row,
                    "source_artifact": "direct_v22_35_C10_repaired_margin_NLL_constrained",
                    "selection_data_source": "held_train_margin_q10_with_held_CE_NLL_constraint",
                    "status": "direct_v22_35_C10_repaired_margin_NLL_constrained",
                }
                for row in control_branch_rows
            )
            c11l_branch_rows.extend(
                {
                    **row,
                    "source_artifact": "direct_v22_35_C11_cohort_influence_repair",
                    "selection_data_source": "held_train_cohort_influence_stability",
                    "status": "direct_v22_35_C11_cohort_influence_repair",
                }
                for row in control_branch_rows
            )
            c10_real = branch_eval("C10_repaired_margin_NLL_constrained_direction", c10_selected, 0, "")
            c10_val_maybe = finite_float(c10_real.get("NLL_delta_vs_base"))
            c10_val = c10_val_maybe if c10_val_maybe is not None else math.inf
            c10_real.update(
                {
                    "subspace_rank": int(basis.shape[1]) if basis is not None else "",
                    "cohort_positive_fraction": sig.get("cohort_positive_fraction", ""),
                    "temporal_eigenspace_overlap": principal_overlap(prev_sig["basis_sketch"], sig["basis_sketch"]),
                    "row_local_epsilon": epsilon,
                    "epsilon_source": eps_source,
                    "beats_L1": int(c10_val < best_by_level.get("L1", math.inf)),
                    "beats_L2": int(c10_val < best_by_level.get("L2", math.inf)),
                    "beats_L3": int(c10_val < best_by_level.get("L3", math.inf)),
                    "beats_L4": int(c10_val < best_by_level.get("L4", math.inf)),
                    "beats_L5": int(c10_val < best_by_level.get("L5", math.inf)),
                    "beats_L6": int(c10_val < best_by_level.get("L6", math.inf)),
                    "beats_L7": int(c10_val < best_by_level.get("L7", math.inf)),
                    "real_minus_best_control_NLL": c10_val - best_any,
                    "selected_candidate_label": c10_selected_audit.get("candidate_label", ""),
                    "selection_status": c10_selection_status,
                    "held_CE_delta": c10_selected_audit.get("held_CE_delta", ""),
                    "held_margin_q10_delta": c10_selected_audit.get("held_margin_q10_delta", ""),
                    "symmetric_curvature_proxy": c10_selected_audit.get("symmetric_curvature_proxy", ""),
                    "max_abs_cos_to_selection_controls": c10_selected_audit.get("max_abs_cos_to_selection_controls", ""),
                    "NLL_delta_H100": "" if int(args.c11_branch_horizon) != 100 else c10_val,
                    "NLL_delta_H200": "" if int(args.c11_branch_horizon) != 200 else c10_val,
                    "NLL_delta_H400": "" if int(args.c11_branch_horizon) != 400 else c10_val,
                    "selection_data_source": "held_train_margin_q10_with_held_CE_NLL_constraint",
                    "source_artifact": "direct_v22_35_C10_repaired_margin_NLL_constrained",
                    "status": "direct_v22_35_C10_repaired_margin_NLL_constrained",
                }
            )
            c10_branch_rows.append(c10_real)
            c10_selector_rows.append(c10_real)
            c11l_real = branch_eval("C11_cohort_influence_repair_direction", c11l_selected, 0, "")
            c11l_val_maybe = finite_float(c11l_real.get("NLL_delta_vs_base"))
            c11l_val = c11l_val_maybe if c11l_val_maybe is not None else math.inf
            c11l_real.update(
                {
                    "subspace_rank": int(basis.shape[1]) if basis is not None else "",
                    "cohort_positive_fraction": sig.get("cohort_positive_fraction", ""),
                    "temporal_eigenspace_overlap": principal_overlap(prev_sig["basis_sketch"], sig["basis_sketch"]),
                    "row_local_epsilon": epsilon,
                    "epsilon_source": eps_source,
                    "beats_L1": int(c11l_val < best_by_level.get("L1", math.inf)),
                    "beats_L2": int(c11l_val < best_by_level.get("L2", math.inf)),
                    "beats_L3": int(c11l_val < best_by_level.get("L3", math.inf)),
                    "beats_L4": int(c11l_val < best_by_level.get("L4", math.inf)),
                    "beats_L5": int(c11l_val < best_by_level.get("L5", math.inf)),
                    "beats_L6": int(c11l_val < best_by_level.get("L6", math.inf)),
                    "beats_L7": int(c11l_val < best_by_level.get("L7", math.inf)),
                    "real_minus_best_control_NLL": c11l_val - best_any,
                    "selected_candidate_label": c11l_selected_audit.get("candidate_label", ""),
                    "selection_status": c11l_selection_status,
                    "held_CE_delta": c11l_selected_audit.get("held_CE_delta", ""),
                    "held_margin_q10_delta": c11l_selected_audit.get("held_margin_q10_delta", ""),
                    "symmetric_curvature_proxy": c11l_selected_audit.get("symmetric_curvature_proxy", ""),
                    "max_abs_cos_to_selection_controls": c11l_selected_audit.get("max_abs_cos_to_selection_controls", ""),
                    "held_cohort_CE_delta_mean": c11l_selected_audit.get("held_cohort_CE_delta_mean", ""),
                    "held_cohort_CE_delta_std": c11l_selected_audit.get("held_cohort_CE_delta_std", ""),
                    "held_cohort_CE_nonincrease_rate": c11l_selected_audit.get("held_cohort_CE_nonincrease_rate", ""),
                    "held_cohort_margin_q10_delta_mean": c11l_selected_audit.get("held_cohort_margin_q10_delta_mean", ""),
                    "NLL_delta_H100": "" if int(args.c11_branch_horizon) != 100 else c11l_val,
                    "NLL_delta_H200": "" if int(args.c11_branch_horizon) != 200 else c11l_val,
                    "NLL_delta_H400": "" if int(args.c11_branch_horizon) != 400 else c11l_val,
                    "selection_data_source": "held_train_cohort_influence_stability",
                    "source_artifact": "direct_v22_35_C11_cohort_influence_repair",
                    "status": "direct_v22_35_C11_cohort_influence_repair",
                }
            )
            c11l_branch_rows.append(c11l_real)
            c11l_selector_rows.append(c11l_real)

    write_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_selector_matrix.csv", selector_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_branch_matrix.csv", branch_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C11_candidate_audit_matrix.csv", audit_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C10_repaired_selector_matrix.csv", c10_selector_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C10_repaired_branch_matrix.csv", c10_branch_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C10_repaired_candidate_audit_matrix.csv", c10_audit_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_selector_matrix.csv", c11l_selector_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_branch_matrix.csv", c11l_branch_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_candidate_audit_matrix.csv", c11l_audit_rows)
    write_rows(OUT_ROOT / "v22_35_Tier2_tabular_availability_matrix.csv", availability_rows)
    existing_branch = [
        r
        for r in read_rows(OUT_ROOT / "v22_35_T1_selector_branch_matrix.csv")
        if r.get("source_artifact") != "direct_v22_35_C11_edge_safe_control_orthogonal"
        and r.get("source_artifact") != "direct_v22_35_C10_repaired_margin_NLL_constrained"
        and r.get("source_artifact") != "direct_v22_35_C11_cohort_influence_repair"
        and r.get("selector_name") != "C11_edge_safe_control_orthogonal_direction"
        and r.get("selector_name") != "C10_repaired_margin_NLL_constrained_direction"
        and r.get("selector_name") != "C11_cohort_influence_repair_direction"
    ]
    existing_control = [
        r
        for r in read_rows(OUT_ROOT / "v22_35_T1_control_hierarchy_matrix.csv")
        if r.get("source_artifact") != "direct_v22_35_C11_edge_safe_control_orthogonal"
    ]
    write_rows(OUT_ROOT / "v22_35_T1_selector_branch_matrix.csv", existing_branch + selector_rows + c10_selector_rows + c11l_selector_rows)
    write_rows(OUT_ROOT / "v22_35_T1_control_hierarchy_matrix.csv", existing_control + control_rows)

    real_rows = [r for r in selector_rows if finite_float(r.get("row_local_epsilon")) is not None]
    n = len(real_rows)
    improve = sum(
        1
        for r in real_rows
        if finite_float(r.get("NLL_delta_vs_base")) is not None
        and (finite_float(r.get("NLL_delta_vs_base")) or 0.0) < -(finite_float(r.get("row_local_epsilon"), 0.0) or 0.0)
    )
    beats_l3 = sum(int_flag(r.get("beats_L3")) for r in real_rows)
    beats_l4 = sum(int_flag(r.get("beats_L4")) for r in real_rows)
    beats_l5 = sum(int_flag(r.get("beats_L5")) for r in real_rows)
    beats_l6 = sum(int_flag(r.get("beats_L6")) for r in real_rows)
    beats_l7 = sum(int_flag(r.get("beats_L7")) for r in real_rows)
    acc_safe = sum(
        1
        for r in real_rows
        if (lambda v: v is not None and v >= -0.01)(finite_float(r.get("accuracy_delta_vs_base")))
    )
    c11_summary = {
        "selector_name": "C11_edge_safe_control_orthogonal_direction",
        "direct_C11_rows": n,
        "availability_rows": len(availability_rows),
        "available_task_rows": sum(1 for r in availability_rows if str(r.get("status")) == "available"),
        "unavailable_task_rows": sum(1 for r in availability_rows if str(r.get("status")) != "available"),
        "tier2_available_task_rows": sum(1 for r in availability_rows if str(r.get("status")) == "available" and str(r.get("task_tier")) == "Tier2_tabular"),
        "tier2_unavailable_task_rows": sum(1 for r in availability_rows if str(r.get("status")) != "available" and str(r.get("task_tier")) == "Tier2_tabular"),
        "candidate_audit_rows": len(audit_rows),
        "NLL_delta_beyond_row_epsilon_rows": improve,
        "NLL_delta_beyond_row_epsilon_rate": rate(improve, n),
        "beats_L3_rate": rate(beats_l3, n),
        "beats_L4_rate": rate(beats_l4, n),
        "beats_L5_rate": rate(beats_l5, n),
        "beats_L6_rate": rate(beats_l6, n),
        "beats_L7_rate": rate(beats_l7, n),
        "accuracy_safe_rate": rate(acc_safe, n),
        "exploration_gate_pass": int(n > 0 and rate(improve, n) >= 0.55 and rate(beats_l4, n) >= 0.50 and rate(acc_safe, n) >= 0.90),
        "official_candidate_gate_pass": int(n > 0 and rate(improve, n) >= 0.65 and rate(beats_l4, n) >= 0.60 and rate(beats_l5, n) >= 0.55 and rate(beats_l6, n) >= 0.55),
        "selection_rule": "held-train CE/margin/curvature plus median control-cosine filter; test/future not used",
    }
    write_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_summary.csv", [c11_summary])
    c10_real_rows = [r for r in c10_selector_rows if finite_float(r.get("row_local_epsilon")) is not None]
    c10_n = len(c10_real_rows)
    c10_improve = sum(
        1
        for r in c10_real_rows
        if finite_float(r.get("NLL_delta_vs_base")) is not None
        and (finite_float(r.get("NLL_delta_vs_base")) or 0.0) < -(finite_float(r.get("row_local_epsilon"), 0.0) or 0.0)
    )
    c10_beats_l3 = sum(int_flag(r.get("beats_L3")) for r in c10_real_rows)
    c10_beats_l4 = sum(int_flag(r.get("beats_L4")) for r in c10_real_rows)
    c10_beats_l5 = sum(int_flag(r.get("beats_L5")) for r in c10_real_rows)
    c10_beats_l6 = sum(int_flag(r.get("beats_L6")) for r in c10_real_rows)
    c10_beats_l7 = sum(int_flag(r.get("beats_L7")) for r in c10_real_rows)
    c10_acc_safe = sum(
        1
        for r in c10_real_rows
        if (lambda v: v is not None and v >= -0.01)(finite_float(r.get("accuracy_delta_vs_base")))
    )
    c10_summary = {
        "selector_name": "C10_repaired_margin_NLL_constrained_direction",
        "direct_C10_repaired_rows": c10_n,
        "candidate_audit_rows": len(c10_audit_rows),
        "NLL_delta_beyond_row_epsilon_rows": c10_improve,
        "NLL_delta_beyond_row_epsilon_rate": rate(c10_improve, c10_n),
        "beats_L3_rate": rate(c10_beats_l3, c10_n),
        "beats_L4_rate": rate(c10_beats_l4, c10_n),
        "beats_L5_rate": rate(c10_beats_l5, c10_n),
        "beats_L6_rate": rate(c10_beats_l6, c10_n),
        "beats_L7_rate": rate(c10_beats_l7, c10_n),
        "accuracy_safe_rate": rate(c10_acc_safe, c10_n),
        "exploration_gate_pass": int(c10_n > 0 and rate(c10_improve, c10_n) >= 0.55 and rate(c10_beats_l4, c10_n) >= 0.50 and rate(c10_acc_safe, c10_n) >= 0.90),
        "official_candidate_gate_pass": int(c10_n > 0 and rate(c10_improve, c10_n) >= 0.65 and rate(c10_beats_l4, c10_n) >= 0.60 and rate(c10_beats_l5, c10_n) >= 0.55 and rate(c10_beats_l6, c10_n) >= 0.55),
        "selection_rule": "maximize held margin q10 under held CE non-increase and margin debt constraint; test/future not used",
    }
    write_rows(OUT_ROOT / "v22_35_T1_C10_repaired_summary.csv", [c10_summary])
    c11l_real_rows = [r for r in c11l_selector_rows if finite_float(r.get("row_local_epsilon")) is not None]
    c11l_n = len(c11l_real_rows)
    c11l_improve = sum(
        1
        for r in c11l_real_rows
        if finite_float(r.get("NLL_delta_vs_base")) is not None
        and (finite_float(r.get("NLL_delta_vs_base")) or 0.0) < -(finite_float(r.get("row_local_epsilon"), 0.0) or 0.0)
    )
    c11l_beats_l3 = sum(int_flag(r.get("beats_L3")) for r in c11l_real_rows)
    c11l_beats_l4 = sum(int_flag(r.get("beats_L4")) for r in c11l_real_rows)
    c11l_beats_l5 = sum(int_flag(r.get("beats_L5")) for r in c11l_real_rows)
    c11l_beats_l6 = sum(int_flag(r.get("beats_L6")) for r in c11l_real_rows)
    c11l_beats_l7 = sum(int_flag(r.get("beats_L7")) for r in c11l_real_rows)
    c11l_acc_safe = sum(
        1
        for r in c11l_real_rows
        if (lambda v: v is not None and v >= -0.01)(finite_float(r.get("accuracy_delta_vs_base")))
    )
    c11l_summary = {
        "selector_name": "C11_cohort_influence_repair_direction",
        "direct_C11_cohort_influence_rows": c11l_n,
        "candidate_audit_rows": len(c11l_audit_rows),
        "NLL_delta_beyond_row_epsilon_rows": c11l_improve,
        "NLL_delta_beyond_row_epsilon_rate": rate(c11l_improve, c11l_n),
        "beats_L3_rate": rate(c11l_beats_l3, c11l_n),
        "beats_L4_rate": rate(c11l_beats_l4, c11l_n),
        "beats_L5_rate": rate(c11l_beats_l5, c11l_n),
        "beats_L6_rate": rate(c11l_beats_l6, c11l_n),
        "beats_L7_rate": rate(c11l_beats_l7, c11l_n),
        "accuracy_safe_rate": rate(c11l_acc_safe, c11l_n),
        "exploration_gate_pass": int(c11l_n > 0 and rate(c11l_improve, c11l_n) >= 0.55 and rate(c11l_beats_l4, c11l_n) >= 0.50 and rate(c11l_acc_safe, c11l_n) >= 0.90),
        "official_candidate_gate_pass": int(c11l_n > 0 and rate(c11l_improve, c11l_n) >= 0.65 and rate(c11l_beats_l4, c11l_n) >= 0.60 and rate(c11l_beats_l5, c11l_n) >= 0.55 and rate(c11l_beats_l6, c11l_n) >= 0.55),
        "selection_rule": "held-train cohort CE non-increase stability with NLL/margin constraint; test/future not used",
    }
    write_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_summary.csv", [c11l_summary])
    old_summary = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    combined_summary = {
        **old_summary,
        "direct_C10_repaired_rows": c10_n,
        "C10_repaired_NLL_delta_beyond_row_epsilon_rate": c10_summary["NLL_delta_beyond_row_epsilon_rate"],
        "C10_repaired_beats_L4_rate": c10_summary["beats_L4_rate"],
        "C10_repaired_beats_L5_rate": c10_summary["beats_L5_rate"],
        "C10_repaired_beats_L6_rate": c10_summary["beats_L6_rate"],
        "C10_repaired_beats_L7_rate": c10_summary["beats_L7_rate"],
        "C10_repaired_exploration_gate_pass": c10_summary["exploration_gate_pass"],
        "C10_repaired_official_candidate_gate_pass": c10_summary["official_candidate_gate_pass"],
        "direct_C11_rows": n,
        "C11_available_task_rows": c11_summary["available_task_rows"],
        "C11_unavailable_task_rows": c11_summary["unavailable_task_rows"],
        "C11_tier2_available_task_rows": c11_summary["tier2_available_task_rows"],
        "C11_tier2_unavailable_task_rows": c11_summary["tier2_unavailable_task_rows"],
        "C11_NLL_delta_beyond_row_epsilon_rate": c11_summary["NLL_delta_beyond_row_epsilon_rate"],
        "C11_beats_L4_rate": c11_summary["beats_L4_rate"],
        "C11_beats_L5_rate": c11_summary["beats_L5_rate"],
        "C11_beats_L6_rate": c11_summary["beats_L6_rate"],
        "C11_beats_L7_rate": c11_summary["beats_L7_rate"],
        "C11_exploration_gate_pass": c11_summary["exploration_gate_pass"],
        "C11_official_candidate_gate_pass": c11_summary["official_candidate_gate_pass"],
        "direct_C11_cohort_influence_rows": c11l_n,
        "C11_cohort_influence_NLL_delta_beyond_row_epsilon_rate": c11l_summary["NLL_delta_beyond_row_epsilon_rate"],
        "C11_cohort_influence_beats_L4_rate": c11l_summary["beats_L4_rate"],
        "C11_cohort_influence_beats_L5_rate": c11l_summary["beats_L5_rate"],
        "C11_cohort_influence_beats_L6_rate": c11l_summary["beats_L6_rate"],
        "C11_cohort_influence_beats_L7_rate": c11l_summary["beats_L7_rate"],
        "C11_cohort_influence_exploration_gate_pass": c11l_summary["exploration_gate_pass"],
        "C11_cohort_influence_official_candidate_gate_pass": c11l_summary["official_candidate_gate_pass"],
        "exploration_gate_pass": int(
            int_flag(old_summary.get("exploration_gate_pass"))
            or int_flag(c11_summary.get("exploration_gate_pass"))
            or int_flag(c10_summary.get("exploration_gate_pass"))
            or int_flag(c11l_summary.get("exploration_gate_pass"))
        ),
        "official_candidate_gate_pass": int(
            int_flag(old_summary.get("official_candidate_gate_pass"))
            or int_flag(c11_summary.get("official_candidate_gate_pass"))
            or int_flag(c10_summary.get("official_candidate_gate_pass"))
            or int_flag(c11l_summary.get("official_candidate_gate_pass"))
        ),
    }
    write_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv", [combined_summary])
    append_exec(
        "run C11 edge-safe control-orthogonal selector with held-train CE/margin/curvature selection plus L6/L7/hard-slice validation",
        task_id="C_C11_edge_safe_selector",
        status="pass" if selector_rows else "warn",
        gpu=args.c11_device,
        files="results/v22_35/v22_35_T1_C11_edge_safe_selector_matrix.csv, results/v22_35/v22_35_T1_C11_edge_safe_branch_matrix.csv, results/v22_35/v22_35_T1_C11_candidate_audit_matrix.csv, results/v22_35/v22_35_T1_C10_repaired_selector_matrix.csv, results/v22_35/v22_35_T1_C10_repaired_branch_matrix.csv, results/v22_35/v22_35_T1_C10_repaired_summary.csv, results/v22_35/v22_35_T1_C11_cohort_influence_selector_matrix.csv, results/v22_35/v22_35_T1_C11_cohort_influence_branch_matrix.csv, results/v22_35/v22_35_T1_C11_cohort_influence_summary.csv, results/v22_35/v22_35_Tier2_tabular_availability_matrix.csv, results/v22_35/v22_35_T1_C11_edge_safe_summary.csv, results/v22_35/v22_35_T1_selector_summary.csv",
        note=f"C11_rows={n}; C10_repaired_rows={c10_n}; C11_cohort_influence_rows={c11l_n}; available_task_rows={c11_summary['available_task_rows']}; unavailable_task_rows={c11_summary['unavailable_task_rows']}; C11_gate={c11_summary['exploration_gate_pass']}; C10_gate={c10_summary['exploration_gate_pass']}; C11L_gate={c11l_summary['exploration_gate_pass']}; L6/L7 controls and hard_slice_NLL_delta recorded",
    )
    return c11_summary


def eval_ce_margin_target(logits: Any, y: Any, output_dim: int, fraction: float | None) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    probs = torch.softmax(logits.float(), dim=-1)
    target = -(probs - F.one_hot(y.long(), num_classes=output_dim).float())
    losses = F.cross_entropy(logits.float(), y.long(), reduction="none")
    n = int(y.numel())
    if fraction is not None:
        k = max(1, min(n, int(math.ceil(float(fraction) * n))))
        keep = torch.topk(losses.detach(), k=k, largest=True).indices
        mask = torch.zeros(n, device=logits.device, dtype=torch.bool)
        mask[keep] = True
        target = torch.where(mask[:, None], target, torch.zeros_like(target))
    else:
        keep = torch.arange(n, device=logits.device)
        k = n
    flat = target.reshape(-1)
    flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    return flat, {
        "hard_slice_fraction": float(k) / float(n),
        "hard_slice_count": int(k),
        "hard_loss_mean": float(losses[keep].mean().item()),
        "all_loss_mean": float(losses.mean().item()),
    }


def hard_margin_target(logits: Any, y: Any, fraction: float) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    losses = F.cross_entropy(logits.float(), y.long(), reduction="none")
    n = int(y.numel())
    k = max(1, min(n, int(math.ceil(float(fraction) * n))))
    hard_idx = torch.topk(losses.detach(), k=k, largest=True).indices
    masked = logits.float().detach().clone()
    masked[torch.arange(n, device=masked.device), y.long()] = float("-inf")
    other = masked.argmax(dim=1)
    target = torch.zeros_like(logits.float())
    target[hard_idx, y.long()[hard_idx]] = 1.0
    target[hard_idx, other[hard_idx]] = -1.0
    flat = target.reshape(-1)
    flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    return flat, {
        "hard_slice_fraction": float(k) / float(n),
        "hard_slice_count": int(k),
        "hard_loss_mean": float(losses[hard_idx].mean().item()),
        "all_loss_mean": float(losses.mean().item()),
    }


def held_class_cvar_target(train_logits: Any, train_y: Any, held_logits: Any, held_y: Any, fraction: float) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    output_dim = int(train_logits.shape[1])
    train_losses = F.cross_entropy(train_logits.float(), train_y.long(), reduction="none")
    held_losses = F.cross_entropy(held_logits.float(), held_y.long(), reduction="none")
    class_scores = torch.zeros(output_dim, device=train_logits.device)
    for cls in range(output_dim):
        cls_losses = held_losses[held_y.long() == cls]
        if int(cls_losses.numel()):
            k_cls = max(1, int(math.ceil(float(fraction) * int(cls_losses.numel()))))
            class_scores[cls] = torch.topk(cls_losses.detach(), k=k_cls, largest=True).values.mean()
    positive = class_scores[class_scores > 0]
    weights = class_scores / positive.mean().clamp_min(1.0e-12) if int(positive.numel()) else torch.ones_like(class_scores)
    weights = torch.clamp(weights, min=0.25, max=4.0)
    probs = torch.softmax(train_logits.float(), dim=-1)
    ce = -(probs - F.one_hot(train_y.long(), num_classes=output_dim).float())
    scores = train_losses.detach() * weights[train_y.long()].detach()
    n = int(train_y.numel())
    k = max(1, min(n, int(math.ceil(float(fraction) * n))))
    hard_idx = torch.topk(scores, k=k, largest=True).indices
    mask = torch.zeros(n, device=train_logits.device, dtype=train_logits.dtype)
    mask[hard_idx] = weights[train_y.long()[hard_idx]].to(dtype=train_logits.dtype)
    margin, meta = hard_margin_target(train_logits, train_y, fraction)
    target = ce * mask[:, None] + margin.reshape_as(ce)
    flat = target.reshape(-1)
    flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    meta.update(
        {
            "held_class_cvar_max": float(class_scores.max().item()) if int(class_scores.numel()) else 0.0,
            "held_class_weight_max": float(weights.max().item()) if int(weights.numel()) else 0.0,
        }
    )
    return flat, meta


def class_confusion_target(logits: Any, y: Any, fraction: float) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    n = int(y.numel())
    losses = F.cross_entropy(logits.float(), y.long(), reduction="none")
    masked = logits.float().detach().clone()
    masked[torch.arange(n, device=masked.device), y.long()] = float("-inf")
    other = masked.argmax(dim=1)
    margins = logits.gather(1, y.long().view(-1, 1)).squeeze(1) - masked.max(dim=1).values
    k = max(1, min(n, int(math.ceil(float(fraction) * n))))
    hard_idx = torch.topk((-margins).detach(), k=k, largest=True).indices
    target = torch.zeros_like(logits.float())
    target[hard_idx, y.long()[hard_idx]] = 1.0
    target[hard_idx, other[hard_idx]] = -1.0
    flat = target.reshape(-1)
    flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    return flat, {
        "hard_slice_fraction": float(k) / float(n),
        "hard_slice_count": int(k),
        "hard_loss_mean": float(losses[hard_idx].mean().item()),
        "all_loss_mean": float(losses.mean().item()),
        "top_confused_pair_count": int(k),
    }


def curvature_safe_target(
    train_logits: Any,
    train_y: Any,
    held_logits: Any,
    held_y: Any,
    fraction: float,
    label_smoothing: float,
    temperature: float,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    output_dim = int(train_logits.shape[1])
    n = int(train_y.numel())
    smooth = min(max(float(label_smoothing), 0.0), 0.45)
    temp = max(float(temperature), 1.0e-6)
    train_losses = F.cross_entropy(train_logits.float(), train_y.long(), reduction="none")
    held_losses = F.cross_entropy(held_logits.float(), held_y.long(), reduction="none")
    class_scores = torch.zeros(output_dim, device=train_logits.device)
    for cls in range(output_dim):
        cls_losses = held_losses[held_y.long() == cls]
        if int(cls_losses.numel()):
            k_cls = max(1, int(math.ceil(float(fraction) * int(cls_losses.numel()))))
            class_scores[cls] = torch.topk(cls_losses.detach(), k=k_cls, largest=True).values.mean()
    positive = class_scores[class_scores > 0]
    weights = class_scores / positive.mean().clamp_min(1.0e-12) if int(positive.numel()) else torch.ones_like(class_scores)
    weights = torch.clamp(weights, min=0.25, max=2.0)
    probs = torch.softmax(train_logits.float() / temp, dim=-1)
    target_dist = torch.full_like(probs, smooth / max(1, output_dim - 1))
    target_dist.scatter_(1, train_y.long().view(-1, 1), 1.0 - smooth)
    raw = (target_dist - probs) * weights[train_y.long()].detach().view(-1, 1)
    masked = train_logits.float().detach().clone()
    masked[torch.arange(n, device=masked.device), train_y.long()] = float("-inf")
    margins = train_logits.float().gather(1, train_y.long().view(-1, 1)).squeeze(1) - masked.max(dim=1).values
    scores = train_losses.detach() * weights[train_y.long()].detach() * (1.0 + torch.relu(-margins.detach()))
    k = max(1, min(n, int(math.ceil(float(fraction) * n))))
    hard_idx = torch.topk(scores, k=k, largest=True).indices
    target = torch.zeros_like(raw)
    target[hard_idx] = raw[hard_idx]
    row_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(1.0e-12)
    target = target / row_norm.clamp_max(1.0)
    flat = target.reshape(-1)
    flat = flat / torch.linalg.vector_norm(flat).clamp_min(1.0e-12)
    return flat, {
        "hard_slice_fraction": float(k) / float(n),
        "hard_slice_count": int(k),
        "hard_loss_mean": float(train_losses[hard_idx].mean().item()),
        "all_loss_mean": float(train_losses.mean().item()),
        "held_class_cvar_max": float(class_scores.max().item()) if int(class_scores.numel()) else 0.0,
        "held_class_weight_max": float(weights.max().item()) if int(weights.numel()) else 0.0,
        "curvature_safe_label_smoothing": smooth,
        "curvature_safe_temperature": temp,
    }


def orthogonalize_vec(target: Any, controls: list[Any]) -> tuple[Any, float]:
    import torch

    out = target.detach().float().clone()
    orig_norm = torch.linalg.vector_norm(out).clamp_min(1.0e-12)
    removed = 0.0
    for ctrl in controls:
        c = ctrl.detach().float().reshape(-1).to(out.device)[: out.numel()]
        denom = torch.dot(c, c).clamp_min(1.0e-12)
        proj = torch.dot(out, c) / denom * c
        removed += float(torch.dot(proj, proj).div(orig_norm * orig_norm).item())
        out = out - proj
    out = out / torch.linalg.vector_norm(out).clamp_min(1.0e-12)
    return out, removed


def held_loss(model_obj: Any, x: Any, y: Any) -> float:
    import torch
    import torch.nn.functional as F

    was_training = bool(model_obj.training)
    model_obj.eval()
    with torch.no_grad():
        value = float(F.cross_entropy(model_obj(x).float(), y.long()).item())
    model_obj.train(was_training)
    return value


def sharpness_proxy(model_obj: Any, x: Any, y: Any, rho: float) -> float:
    import torch
    import torch.nn.functional as F

    was_training = bool(model_obj.training)
    model_obj.eval()
    params = [p for p in model_obj.parameters() if p.requires_grad]
    if not params:
        return 0.0
    model_obj.zero_grad(set_to_none=True)
    base_loss = F.cross_entropy(model_obj(x).float(), y.long())
    grads = torch.autograd.grad(base_loss, params, allow_unused=True)
    norm_sq = torch.zeros((), device=x.device)
    for grad in grads:
        if grad is not None:
            norm_sq = norm_sq + grad.detach().float().pow(2).sum()
    norm = torch.sqrt(norm_sq).clamp_min(1.0e-12)
    steps: list[Any] = []
    with torch.no_grad():
        for param, grad in zip(params, grads):
            if grad is None:
                steps.append(None)
            else:
                step = grad.detach().to(param.device, dtype=param.dtype) * (float(rho) / norm.to(param.device, dtype=param.dtype))
                param.add_(step)
                steps.append(step)
    try:
        with torch.no_grad():
            sharp = F.cross_entropy(model_obj(x).float(), y.long())
    finally:
        with torch.no_grad():
            for param, step in zip(params, steps):
                if step is not None:
                    param.sub_(step)
        model_obj.zero_grad(set_to_none=True)
        model_obj.train(was_training)
    return float((sharp - base_loss.detach()).item())


def stage_d_readout_intrinsic(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from dgkan.fu.real_jacobian_commit import add_flat_delta, output_jacobian, select_named_parameters, solve_linearized_commit
    from experiments.run_v22_30_fidelity_ladder import (
        collect_fixed_examples,
        evaluate_model,
        make_kan,
        train_adamw,
        train_branch,
    )

    device = torch_device(args.intrinsic_device)
    rows: list[dict[str, Any]] = []
    for dataset in split_csv(args.intrinsic_datasets):
        for seed in split_csv(args.intrinsic_seeds, int):
            train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, task_meta = make_v22_35_c11_loaders(
                dataset,
                args.intrinsic_train_size,
                args.intrinsic_test_size,
                args.batch_size,
                seed,
                tier2_download=bool(int(args.tier2_download)),
            )
            for family in split_csv(args.intrinsic_families):
                carrier = "DGKAN_DCHE" if family == "D-CHE" else "DGKAN_DFOU"
                model = make_kan(input_dim, output_dim, args.hidden, seed + 3520, device, x_stats, family).to(device)
                train_adamw(model, train_loader, test_loader, device, output_dim, steps=args.intrinsic_pretrain_steps, lr=args.lr, weight_decay=args.weight_decay)
                base_model = copy.deepcopy(model).to(device)
                base_ev = train_branch(base_model, train_loader, test_loader, device, output_dim, args.intrinsic_branch_horizon, args.lr, args.weight_decay)
                xb, yb = collect_fixed_examples(train_loader, args.intrinsic_examples, device)
                held_x, held_y = collect_fixed_examples(held_loader, args.intrinsic_examples, device)
                logits = model(xb).float()
                held_logits = model(held_x).float()
                ce_target, _ce_meta = eval_ce_margin_target(logits, yb, output_dim, None)
                for target_family in split_csv(args.intrinsic_target_families):
                    target_family = str(target_family)
                    if target_family.startswith("D2"):
                        target, meta = eval_ce_margin_target(logits, yb, output_dim, None)
                        source_type = "multi_cohort_ce_logit_descent"
                    elif target_family.startswith("D4"):
                        target, meta = hard_margin_target(logits, yb, args.hard_fraction)
                        source_type = "hard_slice_margin_logits"
                    elif target_family.startswith("D7"):
                        raw, meta = hard_margin_target(logits, yb, args.hard_fraction)
                        gen = torch.Generator(device=device).manual_seed(35_700 + seed)
                        rnd = torch.randn(raw.shape, device=device, generator=gen)
                        rnd = rnd * raw.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                        target, removed = orthogonalize_vec(raw, [ce_target, rnd])
                        meta["readout_leakage_fraction"] = removed
                        source_type = "control_orthogonal_hard_slice_margin_logits"
                    elif target_family.startswith("D9"):
                        raw, meta = held_class_cvar_target(logits, yb, held_logits, held_y, args.hard_fraction)
                        target, removed = orthogonalize_vec(raw, [ce_target])
                        meta["readout_leakage_fraction"] = removed
                        source_type = "held_class_CVaR_control_orthogonal_margin_logits"
                    elif target_family.startswith("D10"):
                        target, meta = eval_ce_margin_target(logits, yb, output_dim, args.hard_fraction)
                        source_type = "edge_safe_ce_target_with_sharpness_check"
                    elif target_family.startswith("D11"):
                        target, meta = class_confusion_target(logits, yb, args.hard_fraction)
                        source_type = "class_confusion_pair_target_train_only"
                    elif target_family.startswith("D12"):
                        target, meta = curvature_safe_target(
                            logits,
                            yb,
                            held_logits,
                            held_y,
                            args.hard_fraction,
                            args.curvature_safe_label_smoothing,
                            args.curvature_safe_temperature,
                        )
                        source_type = "curvature_safe_label_smooth_hard_slice_train_held_only"
                    else:
                        target, meta = eval_ce_margin_target(logits, yb, output_dim, None)
                        source_type = "unregistered_ce_logit_descent"
                    max_rows = min(int(args.intrinsic_max_output_rows), int(target.numel()))
                    readout_jac, _spec, diag = output_jacobian(model, xb, selector="readout", max_output_rows=max_rows)
                    delta, solve_diag = solve_linearized_commit(readout_jac, target[:max_rows], damping=args.intrinsic_damping)
                    named = select_named_parameters(model, "readout")
                    held_ce_base = held_loss(model, held_x, held_y)
                    sharp_base = sharpness_proxy(model, held_x, held_y, args.sharpness_rho)
                    with torch.no_grad():
                        add_flat_delta(named, delta, scale=args.intrinsic_update_scale)
                    try:
                        held_ce_after = held_loss(model, held_x, held_y)
                        sharp_after = sharpness_proxy(model, held_x, held_y, args.sharpness_rho)
                    finally:
                        with torch.no_grad():
                            add_flat_delta(named, delta, scale=-args.intrinsic_update_scale)
                    variants: list[tuple[str, Any]] = [("readout_exact_real", delta)]
                    gen = torch.Generator(device=device).manual_seed(35_900 + seed + len(rows))
                    rnd = torch.randn(delta.shape, device=device, generator=gen)
                    rnd = rnd * delta.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                    variants.extend([("same_readout_random", rnd), ("readout_signflip_control", -delta)])
                    evs: dict[str, dict[str, float]] = {}
                    for variant, dvec in variants:
                        branch_model = copy.deepcopy(model).to(device)
                        add_flat_delta(select_named_parameters(branch_model, "readout"), dvec, scale=args.intrinsic_update_scale)
                        evs[variant] = train_branch(branch_model, train_loader, test_loader, device, output_dim, args.intrinsic_branch_horizon, args.lr, args.weight_decay)
                    real_delta = evs["readout_exact_real"]["NLL"] - base_ev["NLL"]
                    random_delta = evs["same_readout_random"]["NLL"] - base_ev["NLL"]
                    signflip_delta = evs["readout_signflip_control"]["NLL"] - base_ev["NLL"]
                    eps_row, eps_source = row_epsilon_for(dataset, str(seed), family, str(args.intrinsic_branch_horizon), args.hidden)
                    readout_beats_controls = int(real_delta < random_delta and real_delta < signflip_delta)
                    edge_safe = int((held_ce_after - held_ce_base) <= args.intrinsic_held_ce_tolerance and (sharp_after - sharp_base) <= args.sharpness_tolerance)
                    rows.append(
                        {
                            "target_family": target_family,
                            "carrier": carrier,
                            "basis_family": family,
                            "dataset": dataset,
                            "seed": seed,
                            "task_tier": task_meta.get("task_tier", ""),
                            "source_kind": task_meta.get("source_kind", ""),
                            "model_hidden": int(args.hidden),
                            "level": "Level0_readout_exact_target_branch",
                            "target_source_type": source_type,
                            "readout_exact_NLL_delta": real_delta,
                            "same_readout_random_NLL_delta": random_delta,
                            "readout_signflip_control_NLL_delta": signflip_delta,
                            "readout_exact_beats_controls": readout_beats_controls,
                            "row_local_epsilon": eps_row,
                            "epsilon_source": eps_source,
                            "readout_intrinsic_noise_pass": int(real_delta < -eps_row),
                            "readout_intrinsic_gate_pass": int(real_delta < -eps_row and readout_beats_controls),
                            "readout_projection_residual": solve_diag.get("basis_projection_residual", ""),
                            "readout_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                            "readout_update_norm": solve_diag.get("basis_update_norm", ""),
                            "jacobian_rows": diag.get("jacobian_rows", ""),
                            "jacobian_cols": diag.get("jacobian_cols", ""),
                            "held_CE_delta": held_ce_after - held_ce_base,
                            "SAM_sharpness_delta": sharp_after - sharp_base,
                            "edge_safe_gate_pass": edge_safe,
                            "accuracy_delta": evs["readout_exact_real"]["accuracy"] - base_ev["accuracy"],
                            "hard_slice_fraction": meta.get("hard_slice_fraction", ""),
                            "hard_loss_mean": meta.get("hard_loss_mean", ""),
                            "all_loss_mean": meta.get("all_loss_mean", ""),
                            "held_class_cvar_max": meta.get("held_class_cvar_max", ""),
                            "held_class_weight_max": meta.get("held_class_weight_max", ""),
                            "readout_leakage_fraction": meta.get("readout_leakage_fraction", ""),
                            "curvature_safe_label_smoothing": meta.get("curvature_safe_label_smoothing", ""),
                            "curvature_safe_temperature": meta.get("curvature_safe_temperature", ""),
                            "uses_test_direction_selection": 0,
                            "uses_future_direction": 0,
                            "source_artifact": "direct_v22_35_readout_exact_intrinsic_probe",
                        }
                    )
    write_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_matrix.csv", rows)
    gate_rows = [r for r in rows if int_flag(r.get("readout_intrinsic_gate_pass"))]
    summary = {
        "target_intrinsic_rows": len(rows),
        "readout_intrinsic_gate_rows": len(gate_rows),
        "readout_intrinsic_gate_rate": rate(len(gate_rows), len(rows)),
        "exploration_gate_pass": int(rows and rate(len(gate_rows), len(rows)) >= 0.60),
        "official_candidate_gate_pass": int(rows and rate(len(gate_rows), len(rows)) >= 0.60),
    }
    write_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv", [summary])
    append_exec(
        "run Level0 readout-exact target intrinsic benefit probes",
        task_id="D_level0_readout_exact_intrinsic",
        status="pass" if rows else "warn",
        gpu=args.intrinsic_device,
        files="results/v22_35/v22_35_target_intrinsic_benefit_matrix.csv, results/v22_35/v22_35_target_intrinsic_benefit_summary.csv",
        note=f"gate_rows={summary['readout_intrinsic_gate_rows']}; rows={summary['target_intrinsic_rows']}",
    )
    return summary


def stage_d_intrinsic_sign_scale_repair(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from dgkan.fu.real_jacobian_commit import add_flat_delta, output_jacobian, select_named_parameters, solve_linearized_commit
    from experiments.run_v22_30_fidelity_ladder import (
        collect_fixed_examples,
        evaluate_model,
        make_kan,
        train_adamw,
        train_branch,
    )

    device = torch_device(args.repair_device)
    candidate_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    branch_horizons = sorted({int(x) for x in split_csv(args.repair_branch_horizons, int) if int(x) >= 0})
    scale_mults = [float(x) for x in split_csv(args.repair_scale_multipliers, float) if float(x) > 0.0]
    if not branch_horizons:
        branch_horizons = [0, 20, 60]
    if not scale_mults:
        scale_mults = [0.25, 0.5, 1.0, 2.0, 4.0]

    def target_for_family(target_family: str, model: Any, xb: Any, yb: Any, held_x: Any, held_y: Any, output_dim: int) -> tuple[Any, dict[str, Any], str]:
        logits = model(xb).float()
        held_logits = model(held_x).float()
        ce_target, _ce_meta = eval_ce_margin_target(logits, yb, output_dim, None)
        if target_family.startswith("D2"):
            target, meta = eval_ce_margin_target(logits, yb, output_dim, None)
            return target, meta, "multi_cohort_ce_logit_descent"
        if target_family.startswith("D4"):
            target, meta = hard_margin_target(logits, yb, args.hard_fraction)
            return target, meta, "hard_slice_margin_logits"
        if target_family.startswith("D7"):
            raw, meta = hard_margin_target(logits, yb, args.hard_fraction)
            gen = torch.Generator(device=xb.device).manual_seed(75_700 + int(yb[0].item()) + int(raw.numel()))
            rnd = torch.randn(raw.shape, device=xb.device, generator=gen)
            rnd = rnd * raw.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
            target, removed = orthogonalize_vec(raw, [ce_target, rnd])
            meta["readout_leakage_fraction"] = removed
            return target, meta, "control_orthogonal_hard_slice_margin_logits"
        if target_family.startswith("D9"):
            raw, meta = held_class_cvar_target(logits, yb, held_logits, held_y, args.hard_fraction)
            target, removed = orthogonalize_vec(raw, [ce_target])
            meta["readout_leakage_fraction"] = removed
            return target, meta, "held_class_CVaR_control_orthogonal_margin_logits"
        if target_family.startswith("D10"):
            target, meta = eval_ce_margin_target(logits, yb, output_dim, args.hard_fraction)
            return target, meta, "edge_safe_ce_target_with_sharpness_check"
        if target_family.startswith("D11"):
            target, meta = class_confusion_target(logits, yb, args.hard_fraction)
            return target, meta, "class_confusion_pair_target_train_only"
        if target_family.startswith("D12"):
            target, meta = curvature_safe_target(
                logits,
                yb,
                held_logits,
                held_y,
                args.hard_fraction,
                args.curvature_safe_label_smoothing,
                args.curvature_safe_temperature,
            )
            return target, meta, "curvature_safe_label_smooth_hard_slice_train_held_only"
        target, meta = eval_ce_margin_target(logits, yb, output_dim, None)
        return target, meta, "unregistered_ce_logit_descent"

    def evaluate_after_delta(model_obj: Any, selector: str, delta: Any, scale: float, loader: Any, device_obj: Any, output_dim: int) -> dict[str, float]:
        trial = copy.deepcopy(model_obj).to(device_obj)
        add_flat_delta(select_named_parameters(trial, selector), delta, scale=scale)
        return evaluate_model(trial, loader, device_obj, output_dim)

    for dataset in split_csv(args.repair_datasets):
        for seed in split_csv(args.repair_seeds, int):
            try:
                train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, task_meta = make_v22_35_c11_loaders(
                    dataset,
                    args.repair_train_size,
                    args.repair_test_size,
                    args.batch_size,
                    seed,
                    tier2_download=bool(int(args.tier2_download)),
                )
            except Exception as exc:
                candidate_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_hidden": int(args.hidden),
                        "task_tier": "Tier2_tabular" if (str(dataset).strip().lower() == "wine" or tier2_spec(dataset) is not None) else "",
                        "status": "task_unavailable",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "source_artifact": "direct_v22_35_intrinsic_sign_scale_repair",
                    }
                )
                continue
            for family in split_csv(args.repair_families):
                carrier = "DGKAN_DCHE" if family == "D-CHE" else "DGKAN_DFOU"
                torch.manual_seed(235_000 + seed + (0 if family == "D-CHE" else 10_000))
                model = make_kan(input_dim, output_dim, args.hidden, seed + 3535, device, x_stats, family).to(device)
                train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.repair_pretrain_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                xb, yb = collect_fixed_examples(train_loader, args.repair_examples, device)
                held_x, held_y = collect_fixed_examples(held_loader, args.repair_examples, device)
                base_h0 = evaluate_model(model, test_loader, device, output_dim)
                base_by_h: dict[int, dict[str, float]] = {0: base_h0}
                for horizon in branch_horizons:
                    if horizon <= 0:
                        continue
                    base_by_h[horizon] = train_branch(
                        copy.deepcopy(model).to(device),
                        train_loader,
                        test_loader,
                        device,
                        output_dim,
                        horizon,
                        args.lr,
                        args.weight_decay,
                    )
                held_ce_base = held_loss(model, held_x, held_y)
                sharp_base = sharpness_proxy(model, held_x, held_y, args.sharpness_rho)
                for target_family in split_csv(args.repair_target_families):
                    target, meta, source_type = target_for_family(target_family, model, xb, yb, held_x, held_y, output_dim)
                    max_rows = min(int(args.repair_max_output_rows), int(target.numel()))
                    readout_jac, _spec, diag = output_jacobian(model, xb, selector="readout", max_output_rows=max_rows)
                    delta, solve_diag = solve_linearized_commit(readout_jac, target[:max_rows], damping=args.repair_damping)
                    cand_for_target: list[dict[str, Any]] = []
                    for sign in [1.0, -1.0]:
                        for mult in scale_mults:
                            applied_scale = float(args.repair_update_scale) * float(mult)
                            signed_delta = delta * float(sign)
                            trial = copy.deepcopy(model).to(device)
                            add_flat_delta(select_named_parameters(trial, "readout"), signed_delta, scale=applied_scale)
                            held_ce_after = held_loss(trial, held_x, held_y)
                            sharp_after = sharpness_proxy(trial, held_x, held_y, args.sharpness_rho)
                            h0 = evaluate_model(trial, test_loader, device, output_dim)
                            row = {
                                "repair_stage": "intrinsic_sign_scale_H0_Hbranch",
                                "target_family": target_family,
                                "carrier": carrier,
                                "basis_family": family,
                                "dataset": dataset,
                                "seed": seed,
                                "task_tier": task_meta.get("task_tier", ""),
                                "source_kind": task_meta.get("source_kind", ""),
                                "model_hidden": int(args.hidden),
                                "target_source_type": source_type,
                                "candidate_sign": int(sign),
                                "scale_multiplier": mult,
                                "applied_update_scale": applied_scale,
                                "held_CE_delta": held_ce_after - held_ce_base,
                                "SAM_sharpness_delta": sharp_after - sharp_base,
                                "H0_test_NLL_delta": h0["NLL"] - base_h0["NLL"],
                                "H0_accuracy_delta": h0["accuracy"] - base_h0["accuracy"],
                                "selection_eligible_held_only": int(
                                    (held_ce_after - held_ce_base) <= float(args.repair_held_ce_tolerance)
                                    and (sharp_after - sharp_base) <= float(args.sharpness_tolerance)
                                ),
                                "readout_projection_residual": solve_diag.get("basis_projection_residual", ""),
                                "readout_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                                "readout_update_norm": solve_diag.get("basis_update_norm", ""),
                                "jacobian_rows": diag.get("jacobian_rows", ""),
                                "jacobian_cols": diag.get("jacobian_cols", ""),
                                "hard_slice_fraction": meta.get("hard_slice_fraction", ""),
                                "hard_loss_mean": meta.get("hard_loss_mean", ""),
                                "all_loss_mean": meta.get("all_loss_mean", ""),
                                "held_class_cvar_max": meta.get("held_class_cvar_max", ""),
                                "readout_leakage_fraction": meta.get("readout_leakage_fraction", ""),
                                "curvature_safe_label_smoothing": meta.get("curvature_safe_label_smoothing", ""),
                                "curvature_safe_temperature": meta.get("curvature_safe_temperature", ""),
                                "uses_test_direction_selection": 0,
                                "uses_future_direction": 0,
                                "status": "candidate_evaluated",
                                "source_artifact": "direct_v22_35_intrinsic_sign_scale_repair",
                            }
                            cand_for_target.append(row)
                            candidate_rows.append(row)
                    eligible = [r for r in cand_for_target if int_flag(r.get("selection_eligible_held_only"))]
                    if eligible:
                        selected = min(eligible, key=lambda r: (finite_float(r.get("held_CE_delta"), math.inf) or math.inf, abs(finite_float(r.get("SAM_sharpness_delta"), math.inf) or math.inf)))
                        selection_status = "selected_by_held_CE_among_sharpness_safe_candidates"
                    else:
                        selected = min(cand_for_target, key=lambda r: (finite_float(r.get("held_CE_delta"), math.inf) or math.inf, finite_float(r.get("SAM_sharpness_delta"), math.inf) or math.inf))
                        selection_status = "no_sharpness_safe_candidate_selected_lowest_held_CE_for_diagnostic"
                    selected["selected_by_held_only"] = 1
                    selected["selection_status"] = selection_status
                    sign = float(selected["candidate_sign"])
                    applied_scale = float(selected["applied_update_scale"])
                    signed_delta = delta * sign
                    gen = torch.Generator(device=device).manual_seed(91_000 + seed + len(branch_rows))
                    rnd = torch.randn(delta.shape, device=device, generator=gen)
                    rnd = rnd * delta.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
                    variants = [
                        ("readout_repaired_real", signed_delta),
                        ("same_readout_random", rnd),
                        ("readout_signflip_control", -signed_delta),
                    ]
                    by_h_variant: dict[tuple[int, str], dict[str, float]] = {}
                    for horizon in branch_horizons:
                        for variant, dvec in variants:
                            trial = copy.deepcopy(model).to(device)
                            add_flat_delta(select_named_parameters(trial, "readout"), dvec, scale=applied_scale)
                            if horizon <= 0:
                                ev = evaluate_model(trial, test_loader, device, output_dim)
                            else:
                                ev = train_branch(trial, train_loader, test_loader, device, output_dim, horizon, args.lr, args.weight_decay)
                            by_h_variant[(horizon, variant)] = ev
                    for horizon in branch_horizons:
                        base_ev = base_by_h[horizon]
                        real = by_h_variant[(horizon, "readout_repaired_real")]
                        random_ev = by_h_variant[(horizon, "same_readout_random")]
                        signflip_ev = by_h_variant[(horizon, "readout_signflip_control")]
                        real_delta = real["NLL"] - base_ev["NLL"]
                        random_delta = random_ev["NLL"] - base_ev["NLL"]
                        signflip_delta = signflip_ev["NLL"] - base_ev["NLL"]
                        eps_row, eps_source = row_epsilon_for(dataset, str(seed), family, str(horizon), args.hidden)
                        branch_rows.append(
                            {
                                "repair_stage": "intrinsic_sign_scale_H0_Hbranch",
                                "target_family": target_family,
                                "carrier": carrier,
                                "basis_family": family,
                                "dataset": dataset,
                                "seed": seed,
                                "task_tier": task_meta.get("task_tier", ""),
                                "source_kind": task_meta.get("source_kind", ""),
                                "model_hidden": int(args.hidden),
                                "branch_H": horizon,
                                "selected_sign": int(sign),
                                "selected_scale_multiplier": selected.get("scale_multiplier", ""),
                                "applied_update_scale": applied_scale,
                                "selection_status": selection_status,
                                "real_NLL_delta": real_delta,
                                "same_readout_random_NLL_delta": random_delta,
                                "readout_signflip_control_NLL_delta": signflip_delta,
                                "real_accuracy_delta": real["accuracy"] - base_ev["accuracy"],
                                "row_local_epsilon": eps_row,
                                "epsilon_source": eps_source,
                                "beats_same_readout_random": int(real_delta < random_delta),
                                "beats_readout_signflip_control": int(real_delta < signflip_delta),
                                "noise_pass": int(real_delta < -eps_row),
                                "repair_intrinsic_gate_pass": int(real_delta < -eps_row and real_delta < random_delta and real_delta < signflip_delta),
                                "curvature_safe_label_smoothing": meta.get("curvature_safe_label_smoothing", ""),
                                "curvature_safe_temperature": meta.get("curvature_safe_temperature", ""),
                                "uses_test_direction_selection": 0,
                                "uses_future_direction": 0,
                                "status": "selected_held_only_branch_evaluated",
                                "source_artifact": "direct_v22_35_intrinsic_sign_scale_repair",
                            }
                        )

    write_rows(OUT_ROOT / "v22_35_target_intrinsic_sign_scale_repair_matrix.csv", candidate_rows)
    write_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_branch_matrix.csv", branch_rows)
    selected_rows = [r for r in candidate_rows if int_flag(r.get("selected_by_held_only"))]
    pass_rows = [r for r in branch_rows if int_flag(r.get("repair_intrinsic_gate_pass"))]
    h0_rows = [r for r in branch_rows if str(r.get("branch_H")) == "0"]
    nonzero_rows = [r for r in branch_rows if str(r.get("branch_H")) != "0"]
    h0_pass_rows = [r for r in h0_rows if int_flag(r.get("repair_intrinsic_gate_pass"))]
    branch_pass_rows = [r for r in nonzero_rows if int_flag(r.get("repair_intrinsic_gate_pass"))]
    best_h0 = min((finite_float(r.get("real_NLL_delta"), math.inf) or math.inf for r in h0_rows), default=math.inf)
    best_branch = min((finite_float(r.get("real_NLL_delta"), math.inf) or math.inf for r in nonzero_rows), default=math.inf)
    branch_gate_pass = int(len(branch_pass_rows) >= max(1, math.ceil(0.25 * max(1, len(selected_rows)))))
    summary = {
        "repair_candidate_rows": len(candidate_rows),
        "repair_selected_rows": len(selected_rows),
        "repair_branch_rows": len(branch_rows),
        "repair_gate_pass_rows": len(pass_rows),
        "repair_gate_pass_rate": rate(len(pass_rows), len(branch_rows)),
        "repair_H0_gate_pass_rows": len(h0_pass_rows),
        "repair_nonzero_branch_gate_pass_rows": len(branch_pass_rows),
        "repair_nonzero_branch_gate_pass_rate": rate(len(branch_pass_rows), len(nonzero_rows)),
        "repair_exploration_gate_pass": branch_gate_pass,
        "best_H0_real_NLL_delta": "" if not math.isfinite(best_h0) else best_h0,
        "best_branch_real_NLL_delta": "" if not math.isfinite(best_branch) else best_branch,
        "selection_rule": "held_train_CE_then_sharpness_only; test not used for candidate selection",
        "official_candidate_gate_pass": 0,
        "repair_status": (
            "opened_nonzero_branch_intrinsic_candidate_schedule_basis_transfer"
            if branch_pass_rows
            else (
                "H0_only_diagnostic_candidate_branch_noise_blocked_continue_horizon_or_target_redesign"
                if h0_pass_rows
                else "intrinsic_repair_failed_continue_target_redesign"
            )
        ),
    }
    write_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv", [summary])
    append_exec(
        "run intrinsic target sign/scale/H0-Hbranch repair with held-only selection",
        task_id="D_intrinsic_sign_scale_repair",
        status="pass" if candidate_rows else "warn",
        gpu=args.repair_device,
        files="results/v22_35/v22_35_target_intrinsic_sign_scale_repair_matrix.csv, results/v22_35/v22_35_target_intrinsic_repair_branch_matrix.csv, results/v22_35/v22_35_target_intrinsic_repair_summary.csv",
        note=f"candidate_rows={summary['repair_candidate_rows']}; pass_rows={summary['repair_gate_pass_rows']}; branch_pass_rows={summary['repair_nonzero_branch_gate_pass_rows']}; best_H0={summary['best_H0_real_NLL_delta']}; best_branch={summary['best_branch_real_NLL_delta']}",
    )
    return summary


def strict_fit(row: dict[str, Any]) -> bool:
    residual = finite_float(row.get("basis_actuator_projection_residual"), 999.0) or 999.0
    cosine = finite_float(row.get("basis_actuator_cosine"), 0.0) or 0.0
    exact = finite_float(row.get("exact_vs_linearized_error_selected", row.get("exact_vs_linearized_error")), 999.0) or 999.0
    grad = finite_float(row.get("J_B_gradcheck_rel_error"), 999.0) or 999.0
    return residual <= 0.05 and cosine >= 0.98 and exact <= 0.05 and grad <= 1.0e-3


def stage_d_basis_transfer_reaudit() -> dict[str, Any]:
    fit_rows = read_rows(V22_34 / "v22_34_basis_target_repair_fit_matrix.csv")
    control_rows = read_rows(V22_34 / "v22_34_basis_target_repair_control_win_matrix.csv")
    fit_out: list[dict[str, Any]] = []
    for row in fit_rows:
        fit_out.append(
            {
                "target_family": row.get("target_family", ""),
                "carrier": row.get("carrier", ""),
                "basis_family": row.get("basis_family", ""),
                "basis_bank": row.get("basis_bank", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "readout_exact_NLL_delta": "",
                "readout_exact_beats_controls": "",
                "basis_projection_residual": row.get("basis_actuator_projection_residual", ""),
                "basis_actuator_cosine": row.get("basis_actuator_cosine", ""),
                "exact_vs_linearized_error": row.get("exact_vs_linearized_error_selected", row.get("exact_vs_linearized_error", "")),
                "J_B_gradcheck_rel_error": row.get("J_B_gradcheck_rel_error", ""),
                "basis_update_norm": row.get("basis_update_norm", ""),
                "basis_channel_energy": row.get("basis_channel_energy", ""),
                "readout_leakage_fraction": row.get("readout_leakage_fraction", ""),
                "strict_fit_pass": int(strict_fit(row)),
                "scale_policy": row.get("scale_policy", ""),
                "selected_scale_multiplier": row.get("selected_scale_multiplier", ""),
                "held_CE_delta_at_selected_scale": row.get("held_CE_delta_at_selected_scale", ""),
                "sam_sharpness_delta_at_selected_scale": row.get("sam_sharpness_delta_at_selected_scale", ""),
                "status": "readback_from_v22_34_basis_fit; Level0 readout exact is separate v22_35 artifact",
                "source_artifact": "results/v22_34/v22_34_basis_target_repair_fit_matrix.csv",
            }
        )
    branch_out: list[dict[str, Any]] = []
    for row in control_rows:
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        horizon = str(row.get("branch_H", ""))
        eps, eps_source = control_bootstrap_epsilon(dataset, seed, horizon)
        real_maybe = finite_float(row.get("real_NLL_delta"))
        random_maybe = finite_float(row.get("same_basis_random_NLL_delta"))
        sign_maybe = finite_float(row.get("signflip_basis_control_NLL_delta"))
        real = real_maybe if real_maybe is not None else 0.0
        random_delta = random_maybe if random_maybe is not None else math.inf
        sign_delta = sign_maybe if sign_maybe is not None else math.inf
        strict = int_flag(row.get("fit_strict_pass"))
        noise_pass = int(real < -eps)
        beats_random = int(real < random_delta)
        beats_sign = int(real < sign_delta)
        acc_maybe = finite_float(row.get("accuracy_delta_vs_base"))
        acc = acc_maybe if acc_maybe is not None else -999.0
        pass_row = int(strict and noise_pass and beats_random and beats_sign and acc >= -0.01)
        branch_out.append(
            {
                "target_family": row.get("target_family", ""),
                "carrier": row.get("carrier", ""),
                "basis_family": row.get("basis_family", ""),
                "basis_bank": row.get("basis_bank", ""),
                "dataset": dataset,
                "seed": seed,
                "branch_H": horizon,
                "NLL_delta_H100": "" if horizon != "100" else real,
                "NLL_delta_H200": "" if horizon != "200" else real,
                "NLL_delta_H400": "" if horizon != "400" else real,
                "source_loss_H100": "",
                "source_loss_H200": "",
                "source_loss_H400": "",
                "accuracy_delta": acc,
                "beats_same_basis_random": beats_random,
                "beats_signflip_basis_control": beats_sign,
                "row_local_epsilon": eps,
                "epsilon_source": eps_source,
                "strict_fit_pass": strict,
                "strict_fit_branch_pass": pass_row,
                "full_loop_NLL_delta_vs_KAN": "",
                "full_loop_NLL_delta_vs_MLPFU": "",
                "full_loop_ratio": "",
                "controller_overhead_ratio": "",
                "v22_34_global_epsilon": row.get("epsilon_rep", ""),
                "source_artifact": "results/v22_34/v22_34_basis_target_repair_control_win_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_35_basis_actuator_transfer_matrix.csv", fit_out)
    write_rows(OUT_ROOT / "v22_35_basis_branch_transfer_matrix.csv", branch_out)
    strict_rows = [r for r in branch_out if int_flag(r.get("strict_fit_pass"))]
    pass_rows = [r for r in branch_out if int_flag(r.get("strict_fit_branch_pass"))]
    summary = {
        "basis_fit_rows": len(fit_out),
        "basis_branch_rows": len(branch_out),
        "strict_fit_rows": len(strict_rows),
        "strict_fit_branch_pass_rows": len(pass_rows),
        "strict_fit_branch_pass_rate": rate(len(pass_rows), len(strict_rows)),
        "branch_gate_exploration_pass": int(strict_rows and rate(len(pass_rows), len(strict_rows)) >= 0.50),
        "branch_gate_official_pass": int(strict_rows and rate(len(pass_rows), len(strict_rows)) >= 0.60),
        "full_loop_status": "ready_not_run" if strict_rows and rate(len(pass_rows), len(strict_rows)) >= 0.50 else "gate_blocked_not_run",
    }
    write_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv", [summary])
    append_exec(
        "re-audit v22.34 basis fit/branch transfer with v22.35 row-local epsilon",
        task_id="D_basis_transfer_reaudit",
        status="pass",
        gpu="2",
        files="results/v22_35/v22_35_basis_actuator_transfer_matrix.csv, results/v22_35/v22_35_basis_branch_transfer_matrix.csv, results/v22_35/v22_35_basis_transfer_summary.csv",
        note=f"strict_fit_branch_pass_rows={summary['strict_fit_branch_pass_rows']}; full_loop_status={summary['full_loop_status']}",
    )
    return summary


def stage_d_basis_transfer_repair(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from dgkan.fu.real_jacobian_commit import add_flat_delta, output_jacobian, select_named_parameters, solve_linearized_commit
    from experiments.run_v22_30_fidelity_ladder import (
        collect_fixed_examples,
        evaluate_model,
        make_kan,
        train_adamw,
        train_branch,
    )
    from experiments.run_v22_32_causal_actuator_fidelity import basis_bank_masks

    intrinsic_repair_rows = read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_branch_matrix.csv")
    nonzero_repair_pass = [
        r
        for r in intrinsic_repair_rows
        if str(r.get("branch_H")) != "0" and int_flag(r.get("repair_intrinsic_gate_pass"))
    ]
    h0_repair_pass = [
        r
        for r in intrinsic_repair_rows
        if str(r.get("branch_H")) == "0" and int_flag(r.get("repair_intrinsic_gate_pass"))
    ]
    if nonzero_repair_pass:
        repair_pass = nonzero_repair_pass
        basis_repair_input_source = "nonzero_branch_intrinsic_pass"
    else:
        repair_pass = h0_repair_pass
        basis_repair_input_source = "H0_readout_exact_pass_level0_anchor"
    unique_specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in repair_pass:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("basis_family", "")),
            str(row.get("target_family", "")),
            str(row.get("selected_sign", "")),
            str(row.get("selected_scale_multiplier", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(row)

    if not unique_specs:
        gate_row = {
            "status": "gate_blocked_not_run",
            "reason": "No readout intrinsic repair pass rows under current repair horizons.",
            "basis_repair_input_source": basis_repair_input_source,
            "source_gate": "results/v22_35/v22_35_target_intrinsic_repair_branch_matrix.csv",
            "source_artifact": "direct_v22_35_basis_transfer_repair",
        }
        summary = {
            "basis_repair_input_rows": 0,
            "basis_repair_input_source": basis_repair_input_source,
            "basis_repair_target_rows": 0,
            "basis_repair_fit_rows": 0,
            "basis_repair_branch_rows": 0,
            "basis_repair_strict_fit_rows": 0,
            "basis_repair_strict_branch_rows": 0,
            "basis_repair_strict_fit_branch_pass_rows": 0,
            "basis_repair_low_rank_sketch_ranks": getattr(args, "basis_repair_low_rank_sketch_ranks", ""),
            "basis_repair_low_rank_fit_rows": 0,
            "basis_repair_low_rank_branch_rows": 0,
            "basis_repair_low_rank_strict_branch_rows": 0,
            "basis_repair_low_rank_strict_fit_branch_pass_rows": 0,
            "basis_repair_best_horizon": "",
            "basis_repair_best_horizon_strict_branch_rows": 0,
            "basis_repair_best_horizon_strict_fit_branch_pass_rows": 0,
            "basis_repair_branch_gate_exploration_pass": 0,
            "basis_repair_branch_gate_official_pass": 0,
            "basis_repair_status": "gate_blocked_no_readout_repair_pass_rows",
        }
        write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_target_matrix.csv", [gate_row])
        write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_fit_matrix.csv", [gate_row])
        write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_branch_matrix.csv", [gate_row])
        write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_summary.csv", [summary])
        history_path = OUT_ROOT / "v22_35_basis_repair_attempt_history.csv"
        history_rows = read_rows(history_path)
        history_rows.append(
            {
                "timestamp": now_sg(),
                "repair_datasets": getattr(args, "repair_datasets", ""),
                "repair_families": getattr(args, "repair_families", ""),
                "basis_repair_branch_horizons": getattr(args, "basis_repair_branch_horizons", ""),
                "basis_repair_bank_dims": getattr(args, "basis_repair_bank_dims", ""),
                "basis_repair_low_rank_sketch_ranks": getattr(args, "basis_repair_low_rank_sketch_ranks", ""),
                **summary,
            }
        )
        write_rows(history_path, history_rows)
        append_exec(
            "run basis actuator transfer repair after readout intrinsic gate",
            task_id="D_basis_transfer_repair",
            status="gate_blocked",
            gpu=args.repair_device,
            files="results/v22_35/v22_35_basis_transfer_repair_target_matrix.csv, results/v22_35/v22_35_basis_transfer_repair_fit_matrix.csv, results/v22_35/v22_35_basis_transfer_repair_branch_matrix.csv, results/v22_35/v22_35_basis_transfer_repair_summary.csv",
            note=f"{summary['basis_repair_status']}; input_source={basis_repair_input_source}",
        )
        return summary

    device = torch_device(args.repair_device)
    fit_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    branch_horizons = [int(x) for x in split_csv(args.basis_repair_branch_horizons, int) if int(x) > 0]
    if not branch_horizons:
        branch_horizons = [400]
    bank_dims = [int(x) for x in split_csv(args.basis_repair_bank_dims, int) if int(x) > 0]
    low_rank_sketch_ranks = sorted(
        {
            int(x)
            for x in split_csv(getattr(args, "basis_repair_low_rank_sketch_ranks", ""), int)
            if int(x) > 0
        }
    )
    scale_mults = [float(x) for x in split_csv(args.basis_repair_scale_multipliers, float) if float(x) > 0.0]
    if not scale_mults:
        scale_mults = [0.25, 0.5, 1.0, 2.0, 4.0]
    basis_signs = [float(x) for x in split_csv(args.basis_repair_signs, float) if float(x) != 0.0]
    if not basis_signs:
        basis_signs = [1.0, -1.0]

    def build_target(target_family: str, model: Any, xb: Any, yb: Any, held_x: Any, held_y: Any, output_dim: int) -> tuple[Any, dict[str, Any], str]:
        logits = model(xb).float()
        held_logits = model(held_x).float()
        ce_target, _ce_meta = eval_ce_margin_target(logits, yb, output_dim, None)
        if target_family.startswith("D2"):
            target, meta = eval_ce_margin_target(logits, yb, output_dim, None)
            return target, meta, "multi_cohort_ce_logit_descent"
        if target_family.startswith("D4"):
            target, meta = hard_margin_target(logits, yb, args.hard_fraction)
            return target, meta, "hard_slice_margin_logits"
        if target_family.startswith("D7"):
            raw, meta = hard_margin_target(logits, yb, args.hard_fraction)
            gen = torch.Generator(device=xb.device).manual_seed(75_700 + int(yb[0].item()) + int(raw.numel()))
            rnd = torch.randn(raw.shape, device=xb.device, generator=gen)
            rnd = rnd * raw.norm().clamp_min(1.0e-12) / rnd.norm().clamp_min(1.0e-12)
            target, removed = orthogonalize_vec(raw, [ce_target, rnd])
            meta["readout_leakage_fraction"] = removed
            return target, meta, "control_orthogonal_hard_slice_margin_logits"
        if target_family.startswith("D9"):
            raw, meta = held_class_cvar_target(logits, yb, held_logits, held_y, args.hard_fraction)
            target, removed = orthogonalize_vec(raw, [ce_target])
            meta["readout_leakage_fraction"] = removed
            return target, meta, "held_class_CVaR_control_orthogonal_margin_logits"
        if target_family.startswith("D10"):
            target, meta = eval_ce_margin_target(logits, yb, output_dim, args.hard_fraction)
            return target, meta, "edge_safe_ce_target_with_sharpness_check"
        if target_family.startswith("D11"):
            target, meta = class_confusion_target(logits, yb, args.hard_fraction)
            return target, meta, "class_confusion_pair_target_train_only"
        if target_family.startswith("D12"):
            target, meta = curvature_safe_target(
                logits,
                yb,
                held_logits,
                held_y,
                args.hard_fraction,
                args.curvature_safe_label_smoothing,
                args.curvature_safe_temperature,
            )
            return target, meta, "curvature_safe_label_smooth_hard_slice_train_held_only"
        target, meta = eval_ce_margin_target(logits, yb, output_dim, None)
        return target, meta, "unregistered_ce_logit_descent"

    def choose_basis_scale(model_obj: Any, named_params: Any, delta_vec: Any, held_x: Any, held_y: Any) -> dict[str, Any]:
        base_ce = held_loss(model_obj, held_x, held_y)
        base_sharp = sharpness_proxy(model_obj, held_x, held_y, args.sharpness_rho)
        trace: list[str] = []
        candidates: list[dict[str, Any]] = []
        for basis_sign in basis_signs:
            signed_delta = delta_vec * float(basis_sign)
            for mult in scale_mults:
                applied = float(args.basis_repair_update_scale) * float(mult)
                trial = copy.deepcopy(model_obj).to(held_x.device)
                add_flat_delta(select_named_parameters(trial, "basis"), signed_delta, scale=applied)
                ce_delta = held_loss(trial, held_x, held_y) - base_ce
                sharp_delta = sharpness_proxy(trial, held_x, held_y, args.sharpness_rho) - base_sharp
                row = {
                    "basis_selected_sign": int(math.copysign(1.0, basis_sign)),
                    "selected_scale_multiplier": mult,
                    "applied_update_scale": applied,
                    "held_CE_delta_at_selected_scale": ce_delta,
                    "sam_sharpness_delta_at_selected_scale": sharp_delta,
                    "curvature_safe_scale_pass": int(
                        ce_delta <= float(args.basis_repair_held_ce_tolerance)
                        and sharp_delta <= float(args.basis_repair_sharpness_tolerance)
                    ),
                }
                trace.append(f"{int(math.copysign(1.0, basis_sign))}:{mult}:{ce_delta:.6g}:{sharp_delta:.6g}:{row['curvature_safe_scale_pass']}")
                candidates.append(row)
        safe = [r for r in candidates if int_flag(r.get("curvature_safe_scale_pass"))]
        if safe:
            selected = min(safe, key=lambda r: (finite_float(r.get("held_CE_delta_at_selected_scale"), math.inf) or math.inf, abs(finite_float(r.get("sam_sharpness_delta_at_selected_scale"), math.inf) or math.inf)))
            status = "held_CE_and_sharpness_safe_selected"
        else:
            selected = min(candidates, key=lambda r: (abs(finite_float(r.get("applied_update_scale"), math.inf) or math.inf), finite_float(r.get("sam_sharpness_delta_at_selected_scale"), math.inf) or math.inf))
            status = "no_safe_candidate_min_scale_used_for_diagnostic"
        return {
            **selected,
            "held_CE_base": base_ce,
            "sam_sharpness_base": base_sharp,
            "scale_policy": "held_train_CE_then_sharpness_basis_sign_scale_trust_region",
            "scale_selection_status": status,
            "scale_candidate_trace": ";".join(trace),
        }

    def held_branch_sign_check(
        model_obj: Any,
        delta_vec: Any,
        applied_scale: float,
        train_loader_obj: Any,
        held_loader_obj: Any,
        output_dim_value: int,
    ) -> dict[str, Any]:
        if not int_flag(getattr(args, "basis_repair_branch_aware_sign_check", 1)):
            return {
                "branch_aware_sign_status": "disabled",
                "branch_aware_sign_multiplier": 1,
                "branch_aware_sign_horizon": "",
                "branch_aware_held_delta_selected": "",
                "branch_aware_held_delta_signflip": "",
            }
        horizon = int(getattr(args, "basis_repair_branch_aware_sign_horizon", 0) or 0)
        if horizon <= 0:
            horizon = min(branch_horizons) if branch_horizons else 200
        base_ev = train_branch(
            copy.deepcopy(model_obj).to(device),
            train_loader_obj,
            held_loader_obj,
            device,
            output_dim_value,
            horizon,
            args.lr,
            args.weight_decay,
        )
        candidates: list[dict[str, Any]] = []
        for sign_multiplier in [1.0, -1.0]:
            trial = copy.deepcopy(model_obj).to(device)
            add_flat_delta(select_named_parameters(trial, "basis"), delta_vec * sign_multiplier, scale=applied_scale)
            ev = train_branch(trial, train_loader_obj, held_loader_obj, device, output_dim_value, horizon, args.lr, args.weight_decay)
            candidates.append(
                {
                    "sign_multiplier": sign_multiplier,
                    "held_NLL_delta": ev["NLL"] - base_ev["NLL"],
                    "held_accuracy_delta": ev["accuracy"] - base_ev["accuracy"],
                }
            )
        selected = min(candidates, key=lambda r: (float(r["held_NLL_delta"]), -float(r["held_accuracy_delta"])))
        selected_delta = next(r for r in candidates if float(r["sign_multiplier"]) == 1.0)
        signflip_delta = next(r for r in candidates if float(r["sign_multiplier"]) == -1.0)
        return {
            "branch_aware_sign_status": "held_branch_sign_selected",
            "branch_aware_sign_multiplier": int(math.copysign(1.0, float(selected["sign_multiplier"]))),
            "branch_aware_sign_horizon": horizon,
            "branch_aware_held_delta_selected": selected_delta["held_NLL_delta"],
            "branch_aware_held_delta_signflip": signflip_delta["held_NLL_delta"],
            "branch_aware_held_accuracy_delta_selected": selected_delta["held_accuracy_delta"],
            "branch_aware_held_accuracy_delta_signflip": signflip_delta["held_accuracy_delta"],
        }

    def basis_bank_candidates(
        basis_bank: str,
        bank_jac: Any,
        target: Any,
        bank_meta: dict[str, Any],
    ) -> list[tuple[str, Any, dict[str, Any], dict[str, Any]]]:
        delta_bank, solve_diag = solve_linearized_commit(bank_jac, target, damping=args.basis_repair_damping)
        out: list[tuple[str, Any, dict[str, Any], dict[str, Any]]] = [
            (
                basis_bank,
                delta_bank,
                {
                    **solve_diag,
                    "basis_solve_cols": int(bank_jac.shape[1]),
                    "basis_low_rank_sketch_rank": "",
                    "basis_low_rank_sketch_energy_fraction": "",
                    "basis_low_rank_sketch_status": "full_bank_direct_solve",
                },
                {
                    **bank_meta,
                    "basis_parent_bank": basis_bank,
                    "basis_low_rank_sketch_rank": "",
                    "basis_low_rank_sketch_energy_fraction": "",
                    "basis_low_rank_sketch_status": "full_bank_direct_solve",
                },
            )
        ]
        if basis_bank != "all_basis" or not low_rank_sketch_ranks:
            return out

        j = bank_jac.detach().float()
        z = target.detach().float().reshape(-1).to(j.device)[: j.shape[0]]
        try:
            gram = j @ j.T
            eigvals, eigvecs = torch.linalg.eigh(gram)
        except RuntimeError as exc:
            out[0][3]["basis_low_rank_sketch_status"] = f"gram_eigh_failed:{type(exc).__name__}"
            return out

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0.0)
        eigvecs = eigvecs[:, order]
        singular = torch.sqrt(eigvals)
        valid = singular > 1.0e-8
        max_rank = int(valid.sum().item())
        if max_rank <= 0:
            out[0][3]["basis_low_rank_sketch_status"] = "skipped_zero_spectrum"
            return out
        total_energy = float(torch.sum(singular[:max_rank] ** 2).clamp_min(1.0e-12).item())
        used_ranks: set[int] = set()
        for requested_rank in low_rank_sketch_ranks:
            rank = min(int(requested_rank), max_rank)
            if rank <= 0 or rank in used_ranks:
                continue
            used_ranks.add(rank)
            u_r = eigvecs[:, :rank]
            s_r = singular[:rank].clamp_min(1.0e-8)
            right_basis = j.T @ (u_r / s_r.reshape(1, -1))
            right_basis = right_basis / torch.linalg.vector_norm(right_basis, dim=0, keepdim=True).clamp_min(1.0e-12)
            sketch_jac = j @ right_basis
            coeff, sketch_diag = solve_linearized_commit(sketch_jac, z, damping=args.basis_repair_damping)
            delta_sketch = right_basis @ coeff.to(device=right_basis.device, dtype=right_basis.dtype)
            effect = j @ delta_sketch
            residual = torch.linalg.vector_norm(effect - z).div(torch.linalg.vector_norm(z).clamp_min(1.0e-12))
            denom = torch.linalg.vector_norm(effect).clamp_min(1.0e-12) * torch.linalg.vector_norm(z).clamp_min(1.0e-12)
            cosine = torch.dot(effect, z).div(denom)
            energy = float(torch.sum(singular[:rank] ** 2).item() / total_energy)
            head = ",".join(f"{float(v):.6g}" for v in singular[: min(rank, 8)].detach().cpu())
            enriched_diag = {
                **sketch_diag,
                "linearized_commit_status": f"{sketch_diag.get('linearized_commit_status', '')}_right_svd_sketch",
                "basis_projection_residual": float(residual.item()),
                "basis_operator_residual": float(residual.item()),
                "basis_projection_cosine": float(cosine.item()),
                "basis_update_norm": float(torch.linalg.vector_norm(delta_sketch).item()),
                "basis_solve_cols": rank,
                "basis_low_rank_sketch_rank": rank,
                "basis_low_rank_sketch_requested_rank": requested_rank,
                "basis_low_rank_sketch_energy_fraction": energy,
                "basis_low_rank_sketch_singular_values_head": head,
                "basis_low_rank_sketch_status": "right_svd_from_gram_solve",
            }
            enriched_meta = {
                **bank_meta,
                "bank_type": "low_rank_J_B_sketch",
                "basis_parent_bank": basis_bank,
                "basis_low_rank_sketch_rank": rank,
                "basis_low_rank_sketch_requested_rank": requested_rank,
                "basis_low_rank_sketch_energy_fraction": energy,
                "basis_low_rank_sketch_singular_values_head": head,
                "basis_low_rank_sketch_status": "right_svd_from_gram_solve",
            }
            out.append((f"{basis_bank}_svd_r{rank}", delta_sketch, enriched_diag, enriched_meta))
        return out

    for dataset in sorted({str(r.get("dataset")) for r in unique_specs}):
        for seed_s in sorted({str(r.get("seed")) for r in unique_specs if str(r.get("dataset")) == dataset}):
            seed = int(seed_s)
            try:
                train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, task_meta = make_v22_35_c11_loaders(
                    dataset,
                    args.repair_train_size,
                    args.repair_test_size,
                    args.batch_size,
                    seed,
                    tier2_download=bool(int(args.tier2_download)),
                )
            except Exception as exc:
                fit_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_hidden": int(args.hidden),
                        "task_tier": "Tier2_tabular" if (str(dataset).strip().lower() == "wine" or tier2_spec(dataset) is not None) else "",
                        "status": "task_unavailable",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "source_artifact": "direct_v22_35_basis_transfer_repair",
                    }
                )
                continue
            for family in sorted({str(r.get("basis_family")) for r in unique_specs if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed_s}):
                carrier = "DGKAN_DCHE" if family == "D-CHE" else "DGKAN_DFOU"
                torch.manual_seed(235_000 + seed + (0 if family == "D-CHE" else 10_000))
                model = make_kan(input_dim, output_dim, args.hidden, seed + 3535, device, x_stats, family).to(device)
                train_adamw(
                    model,
                    train_loader,
                    test_loader,
                    device,
                    output_dim,
                    steps=args.repair_pretrain_steps,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )
                xb, yb = collect_fixed_examples(train_loader, args.repair_examples, device)
                held_x, held_y = collect_fixed_examples(held_loader, args.repair_examples, device)
                named = select_named_parameters(model, "basis")
                base_by_h: dict[int, dict[str, float]] = {}
                for horizon in branch_horizons:
                    base_by_h[horizon] = train_branch(
                        copy.deepcopy(model).to(device),
                        train_loader,
                        test_loader,
                        device,
                        output_dim,
                        horizon,
                        args.lr,
                        args.weight_decay,
                    )
                family_specs = [
                    r
                    for r in unique_specs
                    if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed_s and str(r.get("basis_family")) == family
                ]
                for spec_row in family_specs:
                    target_family = str(spec_row.get("target_family"))
                    sign = float(spec_row.get("selected_sign", 1.0) or 1.0)
                    target_full, target_meta, source_type = build_target(target_family, model, xb, yb, held_x, held_y, output_dim)
                    max_rows = min(int(args.repair_max_output_rows), int(target_full.numel()))
                    target = target_full[:max_rows].to(device=device).float() * sign
                    jac, _spec, diag = output_jacobian(model, xb, selector="basis", max_output_rows=max_rows)
                    target_rows.append(
                        {
                            "target_family": target_family,
                            "carrier": carrier,
                            "basis_family": family,
                            "dataset": dataset,
                            "seed": seed,
                            "task_tier": task_meta.get("task_tier", ""),
                            "source_kind": task_meta.get("source_kind", ""),
                            "model_hidden": int(args.hidden),
                            "target_source_type": source_type,
                            "selected_sign_from_readout_repair": sign,
                            "readout_repair_source_branch_H": spec_row.get("branch_H", ""),
                            "basis_repair_input_source": basis_repair_input_source,
                            "target_norm": float(torch.linalg.vector_norm(target).item()),
                            "hard_slice_fraction": target_meta.get("hard_slice_fraction", ""),
                            "hard_loss_mean": target_meta.get("hard_loss_mean", ""),
                            "all_loss_mean": target_meta.get("all_loss_mean", ""),
                            "held_class_cvar_max": target_meta.get("held_class_cvar_max", ""),
                            "held_class_weight_max": target_meta.get("held_class_weight_max", ""),
                            "curvature_safe_label_smoothing": target_meta.get("curvature_safe_label_smoothing", ""),
                            "curvature_safe_temperature": target_meta.get("curvature_safe_temperature", ""),
                            "source_artifact": "direct_v22_35_basis_transfer_repair",
                        }
                    )
                    for basis_bank, mask, bank_meta in basis_bank_masks(named, family, bank_dims):
                        bank_jac = jac[:, mask]
                        if int(bank_jac.shape[1]) <= 0:
                            continue
                        for candidate_bank, candidate_delta_bank, solve_diag, candidate_meta in basis_bank_candidates(
                            basis_bank, bank_jac, target, bank_meta
                        ):
                            delta = torch.zeros(int(jac.shape[1]), device=jac.device, dtype=jac.dtype)
                            delta[mask] = candidate_delta_bank.to(device=jac.device, dtype=jac.dtype)
                            scale_diag = choose_basis_scale(model, named, delta, held_x, held_y)
                            applied_update_scale = float(scale_diag.get("applied_update_scale", args.basis_repair_update_scale))
                            basis_selected_sign = float(scale_diag.get("basis_selected_sign", 1.0) or 1.0)
                            selected_delta = delta * basis_selected_sign
                            solve_residual = finite_float(solve_diag.get("basis_projection_residual"), 999.0) or 999.0
                            solve_cosine = finite_float(solve_diag.get("basis_projection_cosine"), 0.0) or 0.0
                            if solve_residual <= 0.05 and solve_cosine >= 0.98:
                                branch_sign_diag = held_branch_sign_check(
                                    model,
                                    selected_delta,
                                    applied_update_scale,
                                    train_loader,
                                    held_loader,
                                    output_dim,
                                )
                                sign_multiplier = float(branch_sign_diag.get("branch_aware_sign_multiplier", 1.0) or 1.0)
                                selected_delta = selected_delta * sign_multiplier
                                basis_selected_sign = basis_selected_sign * sign_multiplier
                            else:
                                branch_sign_diag = {
                                    "branch_aware_sign_status": "skipped_projection_not_ready",
                                    "branch_aware_sign_multiplier": 1,
                                    "branch_aware_sign_horizon": "",
                                    "branch_aware_held_delta_selected": "",
                                    "branch_aware_held_delta_signflip": "",
                                }
                            scale_diag = {
                                **scale_diag,
                                **branch_sign_diag,
                                "basis_selected_sign": int(math.copysign(1.0, basis_selected_sign)),
                            }
                            with torch.no_grad():
                                base_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                                add_flat_delta(named, selected_delta, scale=applied_update_scale)
                                exact_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                                add_flat_delta(named, selected_delta, scale=-applied_update_scale)
                            lin_scaled = (jac @ selected_delta.detach().float()) * applied_update_scale
                            exact_diff = exact_flat - base_flat
                            exact_err = torch.linalg.vector_norm(exact_diff - lin_scaled).div(torch.linalg.vector_norm(lin_scaled).clamp_min(1.0e-12))
                            lin_unscaled = jac @ selected_delta.detach().float()
                            gradcheck_eps_values = [
                                float(x)
                                for x in split_csv(getattr(args, "basis_repair_gradcheck_eps_values", ""), float)
                                if float(x) > 0.0
                            ] or [float(args.basis_repair_gradcheck_eps)]
                            gradcheck_trace: list[tuple[float, float]] = []
                            for grad_eps in gradcheck_eps_values:
                                with torch.no_grad():
                                    add_flat_delta(named, selected_delta, scale=grad_eps)
                                    plus_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                                    add_flat_delta(named, selected_delta, scale=-2.0 * grad_eps)
                                    minus_flat = model(xb).float().reshape(-1)[: jac.shape[0]].detach().clone()
                                    add_flat_delta(named, selected_delta, scale=grad_eps)
                                fd = (plus_flat - minus_flat) / float(2.0 * grad_eps)
                                grad_rel_eps = torch.linalg.vector_norm(fd - lin_unscaled).div(torch.linalg.vector_norm(fd).clamp_min(1.0e-12))
                                gradcheck_trace.append((float(grad_eps), float(grad_rel_eps.item())))
                            selected_grad_eps, selected_grad_rel = min(gradcheck_trace, key=lambda item: item[1])
                            fit_row = {
                                "target_family": target_family,
                                "carrier": carrier,
                                "basis_family": family,
                                "basis_bank": candidate_bank,
                                "dataset": dataset,
                                "seed": seed,
                                "task_tier": task_meta.get("task_tier", ""),
                                "source_kind": task_meta.get("source_kind", ""),
                                "model_hidden": int(args.hidden),
                                "repair_stage": "v22_35_basis_transfer_repair_after_readout_gate",
                                "basis_repair_input_source": basis_repair_input_source,
                                "readout_repair_source_branch_H": spec_row.get("branch_H", ""),
                                "active_hidden": candidate_meta.get("active_hidden", ""),
                                "basis_channels": candidate_meta.get("basis_channels", ""),
                                "bank_type": candidate_meta.get("bank_type", ""),
                                "basis_parent_bank": candidate_meta.get("basis_parent_bank", basis_bank),
                                "basis_solve_cols": solve_diag.get("basis_solve_cols", int(bank_jac.shape[1])),
                                "basis_low_rank_sketch_rank": candidate_meta.get("basis_low_rank_sketch_rank", ""),
                                "basis_low_rank_sketch_requested_rank": candidate_meta.get("basis_low_rank_sketch_requested_rank", ""),
                                "basis_low_rank_sketch_energy_fraction": candidate_meta.get("basis_low_rank_sketch_energy_fraction", ""),
                                "basis_low_rank_sketch_singular_values_head": candidate_meta.get("basis_low_rank_sketch_singular_values_head", ""),
                                "basis_low_rank_sketch_status": candidate_meta.get("basis_low_rank_sketch_status", ""),
                                "readout_exact_NLL_delta": spec_row.get("real_NLL_delta", ""),
                                "readout_exact_beats_controls": spec_row.get("repair_intrinsic_gate_pass", ""),
                                "basis_selected_sign": int(math.copysign(1.0, basis_selected_sign)),
                                "basis_projection_residual": solve_diag.get("basis_projection_residual", ""),
                                "basis_actuator_projection_residual": solve_diag.get("basis_projection_residual", ""),
                                "basis_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                                "exact_vs_linearized_error": float(exact_err.item()),
                                "exact_vs_linearized_error_selected": float(exact_err.item()),
                                "J_B_gradcheck_rel_error": selected_grad_rel,
                                "J_B_gradcheck_pass": int(selected_grad_rel <= 1.0e-3),
                                "basis_update_norm": solve_diag.get("basis_update_norm", ""),
                                "basis_channel_energy": candidate_meta.get("basis_low_rank_sketch_energy_fraction", 1.0) or 1.0,
                                "readout_leakage_fraction": target_meta.get("readout_leakage_fraction", ""),
                                "jacobian_rows": diag.get("jacobian_rows", ""),
                                "jacobian_cols": int(bank_jac.shape[1]),
                                "basis_damping": args.basis_repair_damping,
                                "basis_update_scale": args.basis_repair_update_scale,
                                "basis_gradcheck_mode": "central",
                                "basis_gradcheck_eps": selected_grad_eps,
                                "basis_gradcheck_eps_values": ",".join(str(v) for v in gradcheck_eps_values),
                                "J_B_gradcheck_rel_error_trace": ";".join(f"{eps}:{rel}" for eps, rel in gradcheck_trace),
                                "linearized_commit_status": solve_diag.get("linearized_commit_status", ""),
                                "strict_fit_pass": int(
                                    (finite_float(solve_diag.get("basis_projection_residual"), 999.0) or 999.0) <= 0.05
                                    and (finite_float(solve_diag.get("basis_projection_cosine"), 0.0) or 0.0) >= 0.98
                                    and float(exact_err.item()) <= 0.05
                                    and selected_grad_rel <= 1.0e-3
                                ),
                                **scale_diag,
                                "status": "direct_v22_35_basis_transfer_repair_fit",
                                "source_artifact": "direct_v22_35_basis_transfer_repair",
                            }
                            fit_rows.append(fit_row)
                            variants = [("basis_native_real", selected_delta)]
                            gen = torch.Generator(device=selected_delta.device).manual_seed(935_000 + seed + len(branch_rows))
                            rnd = torch.zeros_like(selected_delta)
                            rnd_bank = torch.randn(tuple(candidate_delta_bank.shape), device=delta.device, generator=gen)
                            rnd_bank = rnd_bank * candidate_delta_bank.norm().clamp_min(1.0e-12) / rnd_bank.norm().clamp_min(1.0e-12)
                            rnd[mask] = rnd_bank
                            variants.extend([("same_basis_random", rnd), ("signflip_basis_control", -selected_delta)])
                            for horizon in branch_horizons:
                                ev_by_variant: dict[str, dict[str, float]] = {}
                                for variant, dvec in variants:
                                    branch_model = copy.deepcopy(model).to(device)
                                    add_flat_delta(select_named_parameters(branch_model, "basis"), dvec, scale=applied_update_scale)
                                    ev_by_variant[variant] = train_branch(
                                        branch_model,
                                        train_loader,
                                        test_loader,
                                        device,
                                        output_dim,
                                        horizon,
                                        args.lr,
                                        args.weight_decay,
                                    )
                                base_ev = base_by_h[horizon]
                                real_delta = ev_by_variant["basis_native_real"]["NLL"] - base_ev["NLL"]
                                random_delta = ev_by_variant["same_basis_random"]["NLL"] - base_ev["NLL"]
                                signflip_delta = ev_by_variant["signflip_basis_control"]["NLL"] - base_ev["NLL"]
                                eps_row, eps_source = row_epsilon_for(dataset, str(seed), family, str(horizon), args.hidden)
                                branch_pass = int(
                                    int_flag(fit_row.get("strict_fit_pass"))
                                    and real_delta < -eps_row
                                    and real_delta < random_delta
                                    and real_delta < signflip_delta
                                    and (ev_by_variant["basis_native_real"]["accuracy"] - base_ev["accuracy"]) >= -0.01
                                )
                                branch_rows.append(
                                    {
                                        "target_family": target_family,
                                        "carrier": carrier,
                                        "basis_family": family,
                                        "basis_bank": candidate_bank,
                                        "dataset": dataset,
                                        "seed": seed,
                                        "task_tier": task_meta.get("task_tier", ""),
                                        "source_kind": task_meta.get("source_kind", ""),
                                        "model_hidden": int(args.hidden),
                                        "repair_stage": "v22_35_basis_transfer_repair_after_readout_gate",
                                        "basis_repair_input_source": basis_repair_input_source,
                                        "readout_repair_source_branch_H": spec_row.get("branch_H", ""),
                                        "branch_H": horizon,
                                        "basis_selected_sign": int(math.copysign(1.0, basis_selected_sign)),
                                        "basis_parent_bank": candidate_meta.get("basis_parent_bank", basis_bank),
                                        "basis_solve_cols": solve_diag.get("basis_solve_cols", int(bank_jac.shape[1])),
                                        "basis_low_rank_sketch_rank": candidate_meta.get("basis_low_rank_sketch_rank", ""),
                                        "basis_low_rank_sketch_energy_fraction": candidate_meta.get("basis_low_rank_sketch_energy_fraction", ""),
                                        "basis_low_rank_sketch_status": candidate_meta.get("basis_low_rank_sketch_status", ""),
                                        "NLL_delta_H100": "" if horizon != 100 else real_delta,
                                        "NLL_delta_H200": "" if horizon != 200 else real_delta,
                                        "NLL_delta_H400": "" if horizon != 400 else real_delta,
                                        "real_NLL_delta": real_delta,
                                        "same_basis_random_NLL_delta": random_delta,
                                        "signflip_basis_control_NLL_delta": signflip_delta,
                                        "accuracy_delta": ev_by_variant["basis_native_real"]["accuracy"] - base_ev["accuracy"],
                                        "beats_same_basis_random": int(real_delta < random_delta),
                                        "beats_signflip_basis_control": int(real_delta < signflip_delta),
                                        "row_local_epsilon": eps_row,
                                        "epsilon_source": eps_source,
                                        "strict_fit_pass": fit_row.get("strict_fit_pass", 0),
                                        "strict_fit_branch_pass": branch_pass,
                                        "applied_update_scale": applied_update_scale,
                                        "selected_scale_multiplier": scale_diag.get("selected_scale_multiplier", ""),
                                        "scale_selection_status": scale_diag.get("scale_selection_status", ""),
                                        "full_loop_NLL_delta_vs_KAN": "",
                                        "full_loop_NLL_delta_vs_MLPFU": "",
                                        "full_loop_ratio": "",
                                        "controller_overhead_ratio": "",
                                        "status": "direct_v22_35_basis_transfer_repair_branch",
                                        "source_artifact": "direct_v22_35_basis_transfer_repair",
                                    }
                                )

    write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_target_matrix.csv", target_rows)
    write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_fit_matrix.csv", fit_rows)
    write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_branch_matrix.csv", branch_rows)
    main_fit = read_rows(OUT_ROOT / "v22_35_basis_actuator_transfer_matrix.csv") + fit_rows
    main_branch = read_rows(OUT_ROOT / "v22_35_basis_branch_transfer_matrix.csv") + branch_rows
    write_rows(OUT_ROOT / "v22_35_basis_actuator_transfer_matrix.csv", main_fit)
    write_rows(OUT_ROOT / "v22_35_basis_branch_transfer_matrix.csv", main_branch)
    strict_rows = [r for r in branch_rows if int_flag(r.get("strict_fit_pass"))]
    pass_rows = [r for r in branch_rows if int_flag(r.get("strict_fit_branch_pass"))]
    low_rank_fit_rows = [r for r in fit_rows if str(r.get("basis_low_rank_sketch_rank", "")) not in {"", "0"}]
    low_rank_branch_rows = [r for r in branch_rows if str(r.get("basis_low_rank_sketch_rank", "")) not in {"", "0"}]
    low_rank_strict_rows = [r for r in low_rank_branch_rows if int_flag(r.get("strict_fit_pass"))]
    low_rank_pass_rows = [r for r in low_rank_branch_rows if int_flag(r.get("strict_fit_branch_pass"))]
    horizon_rates: list[tuple[int, int, int, float]] = []
    for horizon in branch_horizons:
        strict_h = [r for r in strict_rows if str(r.get("branch_H")) == str(horizon)]
        pass_h = [r for r in pass_rows if str(r.get("branch_H")) == str(horizon)]
        horizon_rates.append((horizon, len(strict_h), len(pass_h), rate(len(pass_h), len(strict_h))))
    best_horizon, best_strict_n, best_pass_n, best_rate = max(horizon_rates, key=lambda item: item[3], default=(0, 0, 0, 0.0))
    summary = {
        "basis_repair_input_rows": len(unique_specs),
        "basis_repair_input_source": basis_repair_input_source,
        "basis_repair_target_rows": len(target_rows),
        "basis_repair_fit_rows": len(fit_rows),
        "basis_repair_branch_rows": len(branch_rows),
        "basis_repair_strict_fit_rows": len([r for r in fit_rows if int_flag(r.get("strict_fit_pass"))]),
        "basis_repair_strict_branch_rows": len(strict_rows),
        "basis_repair_strict_fit_branch_pass_rows": len(pass_rows),
        "basis_repair_strict_fit_branch_pass_rate": rate(len(pass_rows), len(strict_rows)),
        "basis_repair_low_rank_sketch_ranks": ",".join(str(v) for v in low_rank_sketch_ranks),
        "basis_repair_low_rank_fit_rows": len(low_rank_fit_rows),
        "basis_repair_low_rank_branch_rows": len(low_rank_branch_rows),
        "basis_repair_low_rank_strict_branch_rows": len(low_rank_strict_rows),
        "basis_repair_low_rank_strict_fit_branch_pass_rows": len(low_rank_pass_rows),
        "basis_repair_low_rank_strict_fit_branch_pass_rate": rate(len(low_rank_pass_rows), len(low_rank_strict_rows)),
        "basis_repair_best_horizon": best_horizon,
        "basis_repair_best_horizon_strict_branch_rows": best_strict_n,
        "basis_repair_best_horizon_strict_fit_branch_pass_rows": best_pass_n,
        "basis_repair_best_horizon_strict_fit_branch_pass_rate": best_rate,
        "basis_repair_horizon_rate_trace": ";".join(f"H{h}:pass={p}/strict={s}:rate={rr}" for h, s, p, rr in horizon_rates),
        "basis_repair_branch_gate_exploration_pass": int(best_strict_n and best_rate >= 0.50),
        "basis_repair_branch_gate_official_pass": int(best_strict_n and best_rate >= 0.60),
        "basis_repair_status": "basis_repair_branch_gate_opened" if best_strict_n and best_rate >= 0.50 else "basis_repair_branch_gate_blocked_continue_bank_or_control_decomposition",
    }
    write_rows(OUT_ROOT / "v22_35_basis_transfer_repair_summary.csv", [summary])
    history_path = OUT_ROOT / "v22_35_basis_repair_attempt_history.csv"
    history_rows = read_rows(history_path)
    history_rows.append(
        {
            "timestamp": now_sg(),
            "repair_datasets": getattr(args, "repair_datasets", ""),
            "repair_families": getattr(args, "repair_families", ""),
            "basis_repair_branch_horizons": ",".join(str(v) for v in branch_horizons),
            "basis_repair_bank_dims": ",".join(str(v) for v in bank_dims),
            "basis_repair_low_rank_sketch_ranks": ",".join(str(v) for v in low_rank_sketch_ranks),
            **summary,
        }
    )
    write_rows(history_path, history_rows)
    combined_summary = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
    combined_summary.update(summary)
    combined_summary["branch_gate_exploration_pass"] = int(
        int_flag(combined_summary.get("branch_gate_exploration_pass")) or int_flag(summary.get("basis_repair_branch_gate_exploration_pass"))
    )
    combined_summary["branch_gate_official_pass"] = int(
        int_flag(combined_summary.get("branch_gate_official_pass")) or int_flag(summary.get("basis_repair_branch_gate_official_pass"))
    )
    combined_summary["full_loop_status"] = (
        "ready_not_run"
        if int_flag(combined_summary.get("branch_gate_exploration_pass"))
        else "gate_blocked_not_run"
    )
    write_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv", [combined_summary])
    append_exec(
        "run basis actuator transfer repair after readout intrinsic gate",
        task_id="D_basis_transfer_repair",
        status="pass" if fit_rows else "warn",
        gpu=args.repair_device,
        files="results/v22_35/v22_35_basis_transfer_repair_target_matrix.csv, results/v22_35/v22_35_basis_transfer_repair_fit_matrix.csv, results/v22_35/v22_35_basis_transfer_repair_branch_matrix.csv, results/v22_35/v22_35_basis_transfer_repair_summary.csv, results/v22_35/v22_35_basis_repair_attempt_history.csv",
        note=f"input_rows={summary['basis_repair_input_rows']}; strict_fit_rows={summary['basis_repair_strict_fit_rows']}; strict_branch_pass_rows={summary['basis_repair_strict_fit_branch_pass_rows']}; gate={summary['basis_repair_branch_gate_exploration_pass']}",
    )
    return summary


def stage_e_control_decomposition() -> dict[str, Any]:
    branch_rows = read_rows(OUT_ROOT / "v22_35_basis_branch_transfer_matrix.csv")
    safe_rows = read_rows(V22_34 / "v22_34_safe_scale_control_win_decomposition_matrix.csv")
    out: list[dict[str, Any]] = []
    for row in branch_rows:
        real = branch_delta(row)
        random_delta = finite_float(row.get("same_basis_random_NLL_delta"))
        sign_delta = finite_float(row.get("signflip_basis_control_NLL_delta"))
        controls = [v for v in [random_delta, sign_delta] if v is not None]
        best_ctrl = min(controls) if controls else None
        if not int_flag(row.get("strict_fit_pass")):
            klass = "ActuatorSupportOnly"
        elif not int_flag(row.get("beats_same_basis_random")) or not int_flag(row.get("beats_signflip_basis_control")):
            klass = "ActuatorSupportOnly"
        elif not int_flag(row.get("strict_fit_branch_pass")):
            klass = "ShortHorizonNoise"
        else:
            klass = "RealCausal"
        out.append(
            {
                "target_family": row.get("target_family", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "control_level": "same_basis_random/signflip",
                "control_delta_NLL": "" if best_ctrl is None else best_ctrl,
                "real_delta_NLL": "" if real is None else real,
                "real_minus_control": "" if real is None or best_ctrl is None else real - best_ctrl,
                "sharpness_delta_real": "",
                "sharpness_delta_control": "",
                "margin_delta_real": "",
                "margin_delta_control": "",
                "horizon_dependent_win_label": "H400_or_source_horizon",
                "control_win_class": klass,
                "source_artifact": "results/v22_35/v22_35_basis_branch_transfer_matrix.csv",
            }
        )
    for row in safe_rows:
        out.append(
            {
                "target_family": row.get("selector_name", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "control_level": row.get("loses_to_levels", ""),
                "control_delta_NLL": row.get("control_delta", ""),
                "real_delta_NLL": row.get("real_delta", ""),
                "real_minus_control": row.get("real_minus_best_control_NLL", ""),
                "sharpness_delta_real": row.get("sam_sharpness_delta_at_selected_scale", ""),
                "sharpness_delta_control": "",
                "margin_delta_real": "",
                "margin_delta_control": "",
                "horizon_dependent_win_label": "safe_scale_H400",
                "control_win_class": row.get("control_win_class", ""),
                "source_artifact": "results/v22_34/v22_34_safe_scale_control_win_decomposition_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_35_control_win_decomposition_matrix.csv", out)
    real_causal = sum(1 for r in out if r.get("control_win_class") == "RealCausal")
    flatness = sum(1 for r in out if "Flatness" in str(r.get("control_win_class", "")))
    support = sum(1 for r in out if r.get("control_win_class") == "ActuatorSupportOnly")
    summary = {
        "control_rows": len(out),
        "RealCausal_rows": real_causal,
        "FlatnessRegularization_rows": flatness,
        "ActuatorSupportOnly_rows": support,
        "RealCausal_rate": rate(real_causal, len(out)),
    }
    write_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv", [summary])
    append_exec(
        "classify control-win mechanisms from v22.35 branch transfer and v22.34 safe-scale rows",
        task_id="E_control_win_decomposition",
        status="pass",
        gpu="0",
        files="results/v22_35/v22_35_control_win_decomposition_matrix.csv",
        note=f"RealCausal_rows={real_causal}; Flatness_rows={flatness}; ActuatorSupportOnly_rows={support}",
    )
    return summary


def stage_f_curvature_representation() -> dict[str, Any]:
    intrinsic = read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_matrix.csv")
    safe = read_rows(V22_34 / "v22_34_safe_scale_control_win_decomposition_matrix.csv")
    rows: list[dict[str, Any]] = []
    for row in intrinsic:
        rows.append(
            {
                "target_family": row.get("target_family", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "lambda_max_H_proxy": "",
                "eta_lambda_max": "",
                "edge_of_stability_distance": "",
                "SAM_sharpness_proxy": row.get("SAM_sharpness_delta", ""),
                "sharpness_delta_real": row.get("SAM_sharpness_delta", ""),
                "sharpness_delta_controls": "",
                "feature_effective_rank": "",
                "NC1_within_class_covariance": "",
                "NC2_class_mean_ETF_error": "",
                "NC3_classifier_feature_alignment": "",
                "NC4_nearest_class_center_agreement": "",
                "margin_mean": "",
                "margin_q10": "",
                "margin_q01": "",
                "hard_slice_margin_gain": "",
                "low_margin_accuracy": "",
                "class_confusion_matrix_shift": row.get("target_family", "").startswith("D11"),
                "NLL_delta": row.get("readout_exact_NLL_delta", ""),
                "source_artifact": "results/v22_35/v22_35_target_intrinsic_benefit_matrix.csv",
            }
        )
    for row in safe:
        rows.append(
            {
                "target_family": row.get("selector_name", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "lambda_max_H_proxy": "",
                "eta_lambda_max": "",
                "edge_of_stability_distance": "",
                "SAM_sharpness_proxy": row.get("sam_sharpness_delta_at_selected_scale", ""),
                "sharpness_delta_real": row.get("sam_sharpness_delta_at_selected_scale", ""),
                "sharpness_delta_controls": "",
                "feature_effective_rank": "",
                "NC1_within_class_covariance": "",
                "NC2_class_mean_ETF_error": "",
                "NC3_classifier_feature_alignment": "",
                "NC4_nearest_class_center_agreement": "",
                "margin_mean": "",
                "margin_q10": "",
                "margin_q01": "",
                "hard_slice_margin_gain": "",
                "low_margin_accuracy": "",
                "class_confusion_matrix_shift": "",
                "NLL_delta": row.get("real_delta", ""),
                "control_win_class": row.get("control_win_class", ""),
                "source_artifact": "results/v22_34/v22_34_safe_scale_control_win_decomposition_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_35_curvature_representation_matrix.csv", rows)
    summary = {"curvature_rows": len(rows), "diagnostic_only": 1}
    append_exec(
        "materialize curvature/representation diagnostics; diagnostic only",
        task_id="F_curvature_representation",
        status="pass",
        gpu="0",
        files="results/v22_35/v22_35_curvature_representation_matrix.csv",
        note="Diagnostics do not promote; used only to explain control wins.",
    )
    return summary


def stage_h_continual_nondegenerate_boundary(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from dgkan.fu.real_jacobian_commit import add_flat_delta, output_jacobian, select_named_parameters, solve_linearized_commit
    from experiments.run_v22_30_fidelity_ladder import evaluate_model, make_class_mnist_task_loaders, make_kan
    from experiments.run_v22_32_causal_actuator_fidelity import basis_bank_masks

    device = torch_device(args.continual_device)
    baseline_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    task0_steps_grid = split_csv(args.continual_task0_steps, int)
    task1_steps_grid = split_csv(args.continual_task1_steps, int)
    families = split_csv(args.continual_families)
    bank_dims = split_csv(args.continual_bank_dims, int)

    def train_steps(model: Any, loader: Any, steps: int, seed: int) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        iterator = iter(loader)
        torch.manual_seed(int(seed))
        for _ in range(int(steps)):
            try:
                xb, yb = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                xb, yb = next(iterator)
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb).float(), yb)
            loss.backward()
            opt.step()

    def collect_batch(loader: Any, n: int) -> tuple[Any, Any]:
        xs: list[Any] = []
        ys: list[Any] = []
        total = 0
        for xb, yb in loader:
            xs.append(xb)
            ys.append(yb)
            total += int(xb.shape[0])
            if total >= int(n):
                break
        if not xs:
            raise RuntimeError("empty continual train loader")
        x = torch.cat(xs, dim=0)[: int(n)].to(device).float()
        y = torch.cat(ys, dim=0)[: int(n)].to(device).long()
        return x, y

    def eval_tasks(model: Any, task0: dict[str, Any], task1: dict[str, Any], output_dim: int) -> dict[str, float]:
        ev0 = evaluate_model(model, task0["test_loader"], device, output_dim)
        ev1 = evaluate_model(model, task1["test_loader"], device, output_dim)
        return {
            "task0_accuracy": float(ev0["accuracy"]),
            "task0_NLL": float(ev0["NLL"]),
            "task1_accuracy": float(ev1["accuracy"]),
            "task1_NLL": float(ev1["NLL"]),
        }

    def make_task_model(seed: int, family: str, input_dim: int, output_dim: int, x_stats: Any) -> Any:
        offset = 0 if family == "D-CHE" else 10_000
        return make_kan(input_dim, output_dim, args.hidden, seed + 2235 + offset, device, x_stats, family).to(device)

    def choose_continual_sign(model_obj: Any, delta_vec: Any, sign_x: Any, sign_y: Any) -> dict[str, Any]:
        if not int_flag(getattr(args, "continual_sign_select", 0)):
            return {
                "basis_selected_sign": 1,
                "sign_selection_status": "disabled_fixed_positive_sign",
                "sign_selection_source": "",
                "sign_selection_base_task0_train_CE": "",
                "sign_plus_task0_train_CE": "",
                "sign_minus_task0_train_CE": "",
            }
        with torch.no_grad():
            base_ce = float(F.cross_entropy(model_obj(sign_x).float(), sign_y.long()).item())
        sign_rows: list[dict[str, Any]] = []
        for sign in [1, -1]:
            trial = copy.deepcopy(model_obj).to(device)
            add_flat_delta(select_named_parameters(trial, "basis"), delta_vec * float(sign), scale=args.continual_update_scale)
            with torch.no_grad():
                ce = float(F.cross_entropy(trial(sign_x).float(), sign_y.long()).item())
            sign_rows.append({"sign": sign, "task0_train_CE": ce, "task0_train_CE_delta": ce - base_ce})
        selected = min(sign_rows, key=lambda r: (float(r["task0_train_CE"]), 0 if int(r["sign"]) == 1 else 1))
        plus = next(r for r in sign_rows if int(r["sign"]) == 1)
        minus = next(r for r in sign_rows if int(r["sign"]) == -1)
        return {
            "basis_selected_sign": int(selected["sign"]),
            "sign_selection_status": "train_only_task0_CE_selected",
            "sign_selection_source": "task0_train_batch_before_task1_no_test_no_branch",
            "sign_selection_base_task0_train_CE": base_ce,
            "sign_plus_task0_train_CE": plus["task0_train_CE"],
            "sign_minus_task0_train_CE": minus["task0_train_CE"],
            "sign_plus_task0_train_CE_delta": plus["task0_train_CE_delta"],
            "sign_minus_task0_train_CE_delta": minus["task0_train_CE_delta"],
        }

    def load_class_mnist(seed: int) -> tuple[list[dict[str, Any]], int, int, Any]:
        return make_class_mnist_task_loaders(
            args.continual_train_size,
            args.continual_test_size,
            args.batch_size,
            seed,
        )

    for seed in split_csv(args.continual_seeds, int):
        try:
            load_class_mnist(seed)
        except Exception as exc:
            baseline_rows.append(
                {
                    "dataset": "Class_MNIST_T0T1",
                    "seed": seed,
                    "status": "task_unavailable",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "source_artifact": "direct_v22_35_continual_nondegenerate_boundary",
                }
            )
            continue
        for family in families:
            carrier = "DGKAN_DCHE" if family == "D-CHE" else "DGKAN_DFOU" if family == "D-FOU" else f"DGKAN_{family}"
            for task0_steps in task0_steps_grid:
                for task1_steps in task1_steps_grid:
                    tasks, input_dim, output_dim, x_stats = load_class_mnist(seed)
                    task0, task1 = tasks[0], tasks[1]
                    model = make_task_model(seed, family, input_dim, output_dim, x_stats)
                    train_steps(model, task0["train_loader"], task0_steps, seed + 1100 + task0_steps)
                    before = eval_tasks(model, task0, task1, output_dim)
                    base_model = copy.deepcopy(model).to(device)
                    train_steps(base_model, task1["train_loader"], task1_steps, seed + 2100 + task1_steps)
                    base_after = eval_tasks(base_model, task0, task1, output_dim)
                    base_forgetting = max(0.0, before["task0_accuracy"] - base_after["task0_accuracy"])
                    nondegenerate = int(
                        before["task0_accuracy"] >= float(args.continual_min_task0_acc_before_task1)
                        and base_after["task0_accuracy"] >= float(args.continual_min_old_acc_after_task1)
                        and base_after["task1_accuracy"] >= float(args.continual_min_task1_acc_after_task1)
                        and base_forgetting <= float(args.continual_max_base_forgetting)
                    )
                    baseline_rows.append(
                        {
                            "dataset": "Class_MNIST_T0T1",
                            "seed": seed,
                            "carrier": carrier,
                            "basis_family": family,
                            "task0_id": task0["task_id"],
                            "task1_id": task1["task_id"],
                            "task0_steps": task0_steps,
                            "task1_steps": task1_steps,
                            "task0_accuracy_before_task1": before["task0_accuracy"],
                            "task1_accuracy_before_task1": before["task1_accuracy"],
                            "base_task0_accuracy_after_task1": base_after["task0_accuracy"],
                            "base_task1_accuracy_after_task1": base_after["task1_accuracy"],
                            "base_task0_NLL_after_task1": base_after["task0_NLL"],
                            "base_task1_NLL_after_task1": base_after["task1_NLL"],
                            "base_forgetting": base_forgetting,
                            "baseline_old_task_accuracy_after_new_task_nondegenerate": nondegenerate,
                            "nondegenerate_rule": (
                                f"task0_before>={args.continual_min_task0_acc_before_task1};"
                                f"old_after>={args.continual_min_old_acc_after_task1};"
                                f"task1_after>={args.continual_min_task1_acc_after_task1};"
                                f"forgetting<={args.continual_max_base_forgetting}"
                            ),
                            "status": "nondegenerate_baseline_found" if nondegenerate else "degenerate_or_underlearned_baseline",
                            "source_artifact": "direct_v22_35_continual_nondegenerate_boundary",
                        }
                    )

    all_repair_candidates = [
        r for r in baseline_rows if int_flag(r.get("baseline_old_task_accuracy_after_new_task_nondegenerate")) == 1
    ]
    repair_candidates = sorted(
        all_repair_candidates,
        key=lambda r: (
            int(float(r.get("seed", 0) or 0)),
            str(r.get("basis_family", "")),
            int(float(r.get("task0_steps", 0) or 0)),
            int(float(r.get("task1_steps", 0) or 0)),
        ),
    )[: max(0, int(args.continual_max_repair_regimes))]

    for cand_idx, cand in enumerate(repair_candidates):
        seed = int(float(cand.get("seed", 0) or 0))
        family = str(cand.get("basis_family", "D-CHE"))
        carrier = str(cand.get("carrier", "DGKAN_DCHE"))
        task0_steps = int(float(cand.get("task0_steps", 0) or 0))
        task1_steps = int(float(cand.get("task1_steps", 0) or 0))
        tasks, input_dim, output_dim, x_stats = load_class_mnist(seed)
        task0, task1 = tasks[0], tasks[1]
        task0_model = make_task_model(seed, family, input_dim, output_dim, x_stats)
        train_steps(task0_model, task0["train_loader"], task0_steps, seed + 1100 + task0_steps)
        before = eval_tasks(task0_model, task0, task1, output_dim)
        base_model = copy.deepcopy(task0_model).to(device)
        train_steps(base_model, task1["train_loader"], task1_steps, seed + 2100 + task1_steps)
        base_after = eval_tasks(base_model, task0, task1, output_dim)
        base_forgetting = max(0.0, before["task0_accuracy"] - base_after["task0_accuracy"])
        xb0, yb0 = collect_batch(task0["train_loader"], args.continual_basis_examples)
        target_mode = str(args.continual_target_mode)
        if target_mode in {"hard_cvar_boundary", "hard_margin_boundary"}:
            with torch.no_grad():
                logits_all = task0_model(xb0).float()
                losses = F.cross_entropy(logits_all, yb0.long(), reduction="none")
                k_hard = max(1, min(int(xb0.shape[0]), int(math.ceil(float(args.continual_target_hard_fraction) * int(xb0.shape[0])))))
                hard_idx = torch.topk(losses, k=k_hard, largest=True).indices
            xb_target = xb0[hard_idx]
            y_target = yb0[hard_idx]
        else:
            target_mode = "ce_boundary"
            xb_target = xb0
            y_target = yb0
            k_hard = int(xb_target.shape[0])
        logits = task0_model(xb_target).float()
        if target_mode == "hard_margin_boundary":
            with torch.no_grad():
                masked = logits.detach().clone()
                masked[torch.arange(int(masked.shape[0]), device=masked.device), y_target.long()] = -float("inf")
                top_wrong = torch.argmax(masked, dim=-1)
            target_logits = torch.zeros_like(logits)
            target_logits[torch.arange(int(target_logits.shape[0]), device=target_logits.device), y_target.long()] = 1.0
            target_logits[torch.arange(int(target_logits.shape[0]), device=target_logits.device), top_wrong.long()] = -1.0
            target_full = target_logits.reshape(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            target_full = -(probs - F.one_hot(y_target.long(), num_classes=output_dim).float()).reshape(-1)
        target_full = target_full / torch.linalg.vector_norm(target_full).clamp_min(1.0e-12)
        max_rows = min(int(args.continual_max_output_rows), int(target_full.numel()))
        jac, _spec, diag = output_jacobian(task0_model, xb_target, selector="basis", max_output_rows=max_rows)
        named = select_named_parameters(task0_model, "basis")
        repair_stage = f"D8_nd_{target_mode}_seed{seed}_{family}_T0{task0_steps}_T1{task1_steps}_rank{cand_idx}"
        target_rows.append(
            {
                "target_family": "D8_continual_boundary_basis_target_nondegenerate",
                "target_mode": target_mode,
                "target_hard_fraction": args.continual_target_hard_fraction if target_mode in {"hard_cvar_boundary", "hard_margin_boundary"} else "",
                "target_examples": int(xb_target.shape[0]),
                "carrier": carrier,
                "basis_family": family,
                "dataset": "Class_MNIST_T0T1",
                "seed": seed,
                "repair_stage": repair_stage,
                "task0_steps": task0_steps,
                "task1_steps": task1_steps,
                "target_source_type": "previous_task_ce_boundary_target",
                "target_input_source": "task0_train_at_task_boundary_before_task1",
                "target_from_w2_diagnostic": 0,
                "target_norm": float(torch.linalg.vector_norm(target_full[:max_rows]).item()),
                "task0_accuracy_before_task1": before["task0_accuracy"],
                "base_task0_accuracy_after_task1": base_after["task0_accuracy"],
                "base_task1_accuracy_after_task1": base_after["task1_accuracy"],
                "base_forgetting": base_forgetting,
                "baseline_old_task_accuracy_after_new_task_nondegenerate": 1,
                "status": "direct_v22_35_D8_nondegenerate_boundary_target_constructed",
                "source_artifact": "direct_v22_35_continual_nondegenerate_boundary",
            }
        )
        for basis_bank, mask, bank_meta in basis_bank_masks(named, family, bank_dims):
            bank_jac = jac[:, mask]
            if int(bank_jac.shape[1]) <= 0:
                continue
            target = target_full[:max_rows].to(device=jac.device, dtype=jac.dtype)
            delta_bank, solve_diag = solve_linearized_commit(bank_jac, target, damping=args.continual_damping)
            delta = torch.zeros(int(jac.shape[1]), device=jac.device, dtype=jac.dtype)
            delta[mask] = delta_bank.to(device=jac.device, dtype=jac.dtype)
            sign_diag = choose_continual_sign(task0_model, delta, xb0, yb0)
            basis_selected_sign = int(sign_diag.get("basis_selected_sign", 1) or 1)
            selected_delta = delta * float(basis_selected_sign)
            with torch.no_grad():
                base_flat = task0_model(xb_target).float().reshape(-1)[: jac.shape[0]].detach().clone()
                add_flat_delta(named, selected_delta, scale=args.continual_update_scale)
                exact_flat = task0_model(xb_target).float().reshape(-1)[: jac.shape[0]].detach().clone()
                add_flat_delta(named, selected_delta, scale=-args.continual_update_scale)
                add_flat_delta(named, selected_delta, scale=args.continual_gradcheck_eps)
                plus_flat = task0_model(xb_target).float().reshape(-1)[: jac.shape[0]].detach().clone()
                add_flat_delta(named, selected_delta, scale=-2.0 * args.continual_gradcheck_eps)
                minus_flat = task0_model(xb_target).float().reshape(-1)[: jac.shape[0]].detach().clone()
                add_flat_delta(named, selected_delta, scale=args.continual_gradcheck_eps)
            lin = (jac @ selected_delta.detach().float()) * float(args.continual_update_scale)
            exact_diff = exact_flat - base_flat
            exact_err = torch.linalg.vector_norm(exact_diff - lin).div(torch.linalg.vector_norm(lin).clamp_min(1.0e-12))
            fd = (plus_flat - minus_flat) / (2.0 * float(args.continual_gradcheck_eps))
            lin_unscaled = jac @ selected_delta.detach().float()
            grad_rel = torch.linalg.vector_norm(fd - lin_unscaled).div(torch.linalg.vector_norm(fd).clamp_min(1.0e-12))
            projection_residual = finite_float(solve_diag.get("basis_projection_residual"), 999.0)
            basis_cosine = finite_float(solve_diag.get("basis_projection_cosine"), 0.0)
            projection_residual = projection_residual if projection_residual is not None else 999.0
            basis_cosine = basis_cosine if basis_cosine is not None else 0.0
            exact_error = float(exact_err.item())
            gradcheck_error = float(grad_rel.item())
            fit_strict = (
                projection_residual <= 0.05
                and basis_cosine >= 0.98
                and exact_error <= 0.05
                and gradcheck_error <= 1.0e-3
            )
            fit_rows.append(
                {
                    "target_family": "D8_continual_boundary_basis_target_nondegenerate",
                    "target_mode": target_mode,
                    "target_hard_fraction": args.continual_target_hard_fraction if target_mode in {"hard_cvar_boundary", "hard_margin_boundary"} else "",
                    "target_examples": int(xb_target.shape[0]),
                    "carrier": carrier,
                    "basis_family": family,
                    "dataset": "Class_MNIST_T0T1",
                    "seed": seed,
                    "basis_bank": basis_bank,
                    "basis_selected_sign": basis_selected_sign,
                    "sign_selection_status": sign_diag.get("sign_selection_status", ""),
                    "sign_selection_source": sign_diag.get("sign_selection_source", ""),
                    "sign_selection_base_task0_train_CE": sign_diag.get("sign_selection_base_task0_train_CE", ""),
                    "sign_plus_task0_train_CE": sign_diag.get("sign_plus_task0_train_CE", ""),
                    "sign_minus_task0_train_CE": sign_diag.get("sign_minus_task0_train_CE", ""),
                    "sign_plus_task0_train_CE_delta": sign_diag.get("sign_plus_task0_train_CE_delta", ""),
                    "sign_minus_task0_train_CE_delta": sign_diag.get("sign_minus_task0_train_CE_delta", ""),
                    "repair_stage": repair_stage,
                    "task0_steps": task0_steps,
                    "task1_steps": task1_steps,
                    "active_hidden": bank_meta.get("active_hidden", ""),
                    "basis_channels": bank_meta.get("basis_channels", ""),
                    "bank_type": bank_meta.get("bank_type", ""),
                    "basis_actuator_projection_residual": solve_diag.get("basis_projection_residual", ""),
                    "basis_actuator_cosine": solve_diag.get("basis_projection_cosine", ""),
                    "exact_vs_linearized_error": exact_error,
                    "basis_update_norm": solve_diag.get("basis_update_norm", ""),
                    "basis_gradcheck_mode": "central",
                    "basis_gradcheck_eps": args.continual_gradcheck_eps,
                    "J_B_gradcheck_rel_error": gradcheck_error,
                    "J_B_gradcheck_pass": int(gradcheck_error <= 1.0e-3),
                    "jacobian_rows": diag.get("jacobian_rows", ""),
                    "jacobian_cols": int(bank_jac.shape[1]),
                    "selected_param_names": diag.get("selected_param_names", ""),
                    "basis_damping": args.continual_damping,
                    "basis_update_scale": args.continual_update_scale,
                    "fit_strict_pass": int(fit_strict),
                    "status": "direct_v22_35_D8_nondegenerate_boundary_fit",
                    "source_artifact": "direct_v22_35_continual_nondegenerate_boundary",
                }
            )
            variants = [("basis_native_real", selected_delta)]
            gen = torch.Generator(device=selected_delta.device).manual_seed(902_350 + seed + len(branch_rows))
            rnd = torch.zeros_like(selected_delta)
            rnd_bank = torch.randn(tuple(delta_bank.shape), device=delta.device, generator=gen)
            rnd_bank = rnd_bank * delta_bank.norm().clamp_min(1.0e-12) / rnd_bank.norm().clamp_min(1.0e-12)
            rnd[mask] = rnd_bank
            variants.extend([("same_basis_random", rnd), ("signflip_basis_control", -selected_delta)])
            bank_rows: list[dict[str, Any]] = []
            for variant, dvec in variants:
                branch_model = copy.deepcopy(task0_model).to(device)
                add_flat_delta(select_named_parameters(branch_model, "basis"), dvec, scale=args.continual_update_scale)
                boundary_ev = eval_tasks(branch_model, task0, task1, output_dim)
                train_steps(branch_model, task1["train_loader"], task1_steps, seed + 5100 + len(branch_rows))
                after = eval_tasks(branch_model, task0, task1, output_dim)
                forgetting = max(0.0, before["task0_accuracy"] - after["task0_accuracy"])
                row = {
                    "target_family": "D8_continual_boundary_basis_target_nondegenerate",
                    "target_mode": target_mode,
                    "carrier": carrier,
                    "basis_family": family,
                    "dataset": "Class_MNIST_T0T1",
                    "seed": seed,
                    "basis_bank": basis_bank,
                    "basis_selected_sign": basis_selected_sign,
                    "sign_selection_status": sign_diag.get("sign_selection_status", ""),
                    "repair_stage": repair_stage,
                    "task0_steps": task0_steps,
                    "task1_steps": task1_steps,
                    "branch_variant": variant,
                    "task0_accuracy_before_task1": before["task0_accuracy"],
                    "base_task0_accuracy_after_task1": base_after["task0_accuracy"],
                    "base_task1_accuracy_after_task1": base_after["task1_accuracy"],
                    "base_forgetting": base_forgetting,
                    "boundary_task0_accuracy_delta_vs_base": boundary_ev["task0_accuracy"] - before["task0_accuracy"],
                    "task0_accuracy_after_task1": after["task0_accuracy"],
                    "task1_accuracy_after_task1": after["task1_accuracy"],
                    "task0_NLL_after_task1": after["task0_NLL"],
                    "task1_NLL_after_task1": after["task1_NLL"],
                    "forgetting": forgetting,
                    "forgetting_delta_vs_base": forgetting - base_forgetting,
                    "relative_forgetting_reduction_vs_base": (base_forgetting - forgetting) / max(1.0e-12, base_forgetting),
                    "task0_accuracy_delta_vs_base_after_task1": after["task0_accuracy"] - base_after["task0_accuracy"],
                    "task1_accuracy_delta_vs_base_after_task1": after["task1_accuracy"] - base_after["task1_accuracy"],
                    "beats_same_basis_random": "",
                    "beats_signflip_basis_control": "",
                    "status": "direct_v22_35_D8_nondegenerate_boundary_branch",
                    "source_artifact": "direct_v22_35_continual_nondegenerate_boundary",
                }
                branch_rows.append(row)
                bank_rows.append(row)
            real = next((r for r in bank_rows if r.get("branch_variant") == "basis_native_real"), None)
            random_ctrl = next((r for r in bank_rows if r.get("branch_variant") == "same_basis_random"), None)
            signflip_ctrl = next((r for r in bank_rows if r.get("branch_variant") == "signflip_basis_control"), None)
            if real and random_ctrl and signflip_ctrl:
                real_forget = finite_float(real.get("forgetting"), math.inf)
                random_forget = finite_float(random_ctrl.get("forgetting"), math.inf)
                signflip_forget = finite_float(signflip_ctrl.get("forgetting"), math.inf)
                real_forget = real_forget if real_forget is not None else math.inf
                random_forget = random_forget if random_forget is not None else math.inf
                signflip_forget = signflip_forget if signflip_forget is not None else math.inf
                real["beats_same_basis_random"] = int(real_forget < random_forget)
                real["beats_signflip_basis_control"] = int(real_forget < signflip_forget)
                rel_reduction = finite_float(real.get("relative_forgetting_reduction_vs_base"), 0.0)
                current_accuracy_delta = finite_float(real.get("task1_accuracy_delta_vs_base_after_task1"), -999.0)
                rel_reduction = rel_reduction if rel_reduction is not None else 0.0
                current_accuracy_delta = current_accuracy_delta if current_accuracy_delta is not None else -999.0
                forget_ok = rel_reduction >= float(args.continual_min_relative_forgetting_reduction)
                current_ok = current_accuracy_delta >= -float(args.continual_current_accuracy_tolerance)
                pass_row = (
                    fit_strict
                    and forget_ok
                    and int_flag(real.get("beats_same_basis_random"))
                    and int_flag(real.get("beats_signflip_basis_control"))
                    and current_ok
                )
                if not fit_strict:
                    explanation = "not_strict_fit_candidate"
                elif not forget_ok:
                    explanation = "strict_fit_but_relative_forgetting_reduction_below_threshold"
                elif not int_flag(real.get("beats_same_basis_random")) and not int_flag(real.get("beats_signflip_basis_control")):
                    explanation = "same_basis_random_and_signflip_explain"
                elif not int_flag(real.get("beats_same_basis_random")):
                    explanation = "same_basis_random_explains"
                elif not int_flag(real.get("beats_signflip_basis_control")):
                    explanation = "signflip_explains"
                elif not current_ok:
                    explanation = "current_task_accuracy_debt"
                else:
                    explanation = "strict_fit_boundary_pass"
                control_rows.append(
                    {
                        "target_family": "D8_continual_boundary_basis_target_nondegenerate",
                        "target_mode": target_mode,
                        "carrier": carrier,
                        "basis_family": family,
                        "dataset": "Class_MNIST_T0T1",
                        "seed": seed,
                        "basis_bank": basis_bank,
                        "basis_selected_sign": basis_selected_sign,
                        "sign_selection_status": sign_diag.get("sign_selection_status", ""),
                        "repair_stage": repair_stage,
                        "task0_steps": task0_steps,
                        "task1_steps": task1_steps,
                        "fit_strict_pass": int(fit_strict),
                        "baseline_old_task_accuracy_after_new_task_nondegenerate": 1,
                        "base_forgetting": base_forgetting,
                        "real_forgetting": real.get("forgetting", ""),
                        "same_basis_random_forgetting": random_ctrl.get("forgetting", ""),
                        "signflip_basis_control_forgetting": signflip_ctrl.get("forgetting", ""),
                        "forgetting_delta_vs_base": real.get("forgetting_delta_vs_base", ""),
                        "relative_forgetting_reduction_vs_base": real.get("relative_forgetting_reduction_vs_base", ""),
                        "forgetting_gate_pass": int(forget_ok),
                        "beats_same_basis_random": real.get("beats_same_basis_random", ""),
                        "beats_signflip_basis_control": real.get("beats_signflip_basis_control", ""),
                        "task1_accuracy_delta_vs_base_after_task1": real.get("task1_accuracy_delta_vs_base_after_task1", ""),
                        "current_task_accuracy_gate_pass": int(current_ok),
                        "strict_fit_boundary_pass": int(pass_row),
                        "control_win_explanation": explanation,
                        "status": "direct_v22_35_D8_nondegenerate_boundary_control_win",
                        "source_artifact": "direct_v22_35_continual_nondegenerate_boundary",
                    }
                )

    strict_rows = [r for r in control_rows if int_flag(r.get("fit_strict_pass"))]
    pass_rows = [r for r in control_rows if int_flag(r.get("strict_fit_boundary_pass"))]
    strict_forget = [r for r in strict_rows if int_flag(r.get("forgetting_gate_pass"))]
    strict_control = [
        r for r in strict_rows if int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
    ]
    reduction_values = [
        v for v in (finite_float(r.get("relative_forgetting_reduction_vs_base")) for r in control_rows) if v is not None
    ]
    best_reduction = max(reduction_values) if reduction_values else float("-inf")
    summary = {
        "target_mode": args.continual_target_mode,
        "target_hard_fraction": args.continual_target_hard_fraction if str(args.continual_target_mode) in {"hard_cvar_boundary", "hard_margin_boundary"} else "",
        "sign_select": int_flag(getattr(args, "continual_sign_select", 0)),
        "baseline_rows": len(baseline_rows),
        "nondegenerate_baseline_rows": len(all_repair_candidates),
        "selected_repair_regimes": len(repair_candidates),
        "repair_candidate_selection": "pre_registered_lexicographic_first_max_repair_regimes",
        "repair_target_rows": len(target_rows),
        "repair_fit_rows": len(fit_rows),
        "repair_branch_rows": len(branch_rows),
        "repair_control_rows": len(control_rows),
        "strict_fit_rows": len(strict_rows),
        "strict_fit_forgetting_gate_rows": len(strict_forget),
        "strict_fit_control_beat_rows": len(strict_control),
        "strict_fit_boundary_pass_rows": len(pass_rows),
        "strict_fit_boundary_pass_rate": rate(len(pass_rows), len(strict_rows)),
        "branch_gate_exploration_pass": int(bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40),
        "branch_gate_official_pass": int(bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.60),
        "best_relative_forgetting_reduction_vs_base": "" if not math.isfinite(best_reduction) else best_reduction,
        "full_loop_status": "ready_for_D8_full_loop_not_run_in_this_stage"
        if bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40
        else "gate_blocked_not_run",
        "repair_status": "nondegenerate_baseline_not_found"
        if not repair_candidates
        else ("D8_boundary_gate_opened_schedule_full_loop" if bool(strict_rows) and rate(len(pass_rows), len(strict_rows)) >= 0.40 else "D8_boundary_gate_failed_after_nondegenerate_baseline"),
    }
    write_rows(OUT_ROOT / "v22_35_continual_nondegenerate_baseline_matrix.csv", baseline_rows or [{"status": "not_run_no_rows"}])
    write_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_target_matrix.csv", target_rows or [{"status": summary["repair_status"]}])
    write_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_fit_matrix.csv", fit_rows or [{"status": summary["repair_status"]}])
    write_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_branch_matrix.csv", branch_rows or [{"status": summary["repair_status"]}])
    write_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_control_win_matrix.csv", control_rows or [{"status": summary["repair_status"]}])
    history_path = OUT_ROOT / "v22_35_continual_attempt_history.csv"
    history_rows = read_rows(history_path)
    history_rows.append(
        {
            "timestamp": now_sg(),
            "target_mode": args.continual_target_mode,
            "target_hard_fraction": args.continual_target_hard_fraction if str(args.continual_target_mode) in {"hard_cvar_boundary", "hard_margin_boundary"} else "",
            "task0_steps_grid": args.continual_task0_steps,
            "task1_steps_grid": args.continual_task1_steps,
            "bank_dims": args.continual_bank_dims,
            "damping": args.continual_damping,
            "update_scale": args.continual_update_scale,
            "sign_select": int_flag(getattr(args, "continual_sign_select", 0)),
            "basis_examples": args.continual_basis_examples,
            "max_output_rows": args.continual_max_output_rows,
            "baseline_rows": summary["baseline_rows"],
            "nondegenerate_baseline_rows": summary["nondegenerate_baseline_rows"],
            "selected_repair_regimes": summary["selected_repair_regimes"],
            "strict_fit_rows": summary["strict_fit_rows"],
            "strict_fit_forgetting_gate_rows": summary["strict_fit_forgetting_gate_rows"],
            "strict_fit_control_beat_rows": summary["strict_fit_control_beat_rows"],
            "strict_fit_boundary_pass_rows": summary["strict_fit_boundary_pass_rows"],
            "branch_gate_exploration_pass": summary["branch_gate_exploration_pass"],
            "best_relative_forgetting_reduction_vs_base": summary["best_relative_forgetting_reduction_vs_base"],
            "repair_status": summary["repair_status"],
            "command": " ".join([shlex.quote(a) for a in sys.argv]),
        }
    )
    write_rows(history_path, history_rows)
    write_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_summary.csv", [summary])
    append_exec(
        "run v22.35 continual non-degenerate baseline sweep and D8 boundary repair on pre-registered regimes",
        task_id="H_continual_nondegenerate_boundary",
        status="pass" if int(summary["baseline_rows"]) else "warn",
        gpu=args.continual_device,
        files=(
            "results/v22_35/v22_35_continual_nondegenerate_baseline_matrix.csv, "
            "results/v22_35/v22_35_continual_nondegenerate_boundary_target_matrix.csv, "
            "results/v22_35/v22_35_continual_nondegenerate_boundary_fit_matrix.csv, "
            "results/v22_35/v22_35_continual_nondegenerate_boundary_branch_matrix.csv, "
            "results/v22_35/v22_35_continual_nondegenerate_boundary_control_win_matrix.csv, "
            "results/v22_35/v22_35_continual_nondegenerate_boundary_summary.csv, "
            "results/v22_35/v22_35_continual_attempt_history.csv"
        ),
        note=(
            f"nondegenerate_baseline_rows={summary['nondegenerate_baseline_rows']}; "
            f"selected_repair_regimes={summary['selected_repair_regimes']}; "
            f"strict_fit_boundary_pass_rows={summary['strict_fit_boundary_pass_rows']}; "
            f"branch_gate_exploration_pass={summary['branch_gate_exploration_pass']}; "
            f"best_relative_forgetting_reduction={summary['best_relative_forgetting_reduction_vs_base']}"
        ),
    )
    return summary


def train_manual_strong_optimizer_v22_35(
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
    acceptance: str = "none",
    acceptance_tol: float = 0.0,
    accept_loader: Any | None = None,
) -> dict[str, Any]:
    """v22.35-owned strong optimizer loop with optional train-stream acceptance.

    The acceptance gate never reads validation/test direction. For
    train_loss_beats_base_step it compares the signal/control candidate against
    the same-step base optimizer update on the current train batch.
    """
    import torch
    import torch.nn.functional as F

    from experiments.run_v22_30_fidelity_ladder import cycle_batches, evaluate_model, orthogonalized_update, sync

    if signal_mode not in {"none", "signal", "random", "signflip", "shuffled"}:
        raise ValueError(f"unknown signal_mode={signal_mode!r}")
    if acceptance not in {
        "none",
        "train_loss_nonworse",
        "train_loss_beats_base_step",
        "held_train_loss_nonworse",
        "held_train_loss_beats_base_step",
    }:
        raise ValueError(f"unknown optimizer acceptance={acceptance!r}")
    held_acceptance_modes = {"held_train_loss_nonworse", "held_train_loss_beats_base_step"}
    base_step_acceptance_modes = {"train_loss_beats_base_step", "held_train_loss_beats_base_step"}
    if acceptance in held_acceptance_modes and accept_loader is None:
        raise ValueError(f"{acceptance} requires accept_loader")

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
    acceptance_total = 0
    acceptance_accepts = 0
    acceptance_rejects = 0
    candidate_loss_delta_sum = 0.0

    def apply_updates(updates: dict[str, Any]) -> None:
        with torch.no_grad():
            for name, p in named:
                update = updates.get(name)
                if update is None:
                    continue
                p.mul_(1.0 - float(lr) * float(weight_decay))
                p.add_(-float(lr) * update)

    def snapshot_params() -> list[Any]:
        return [p.detach().clone() for _name, p in named]

    def restore_params(params: list[Any]) -> None:
        with torch.no_grad():
            for old, (_name, p) in zip(params, named):
                p.copy_(old)

    start = time.time()
    it = cycle_batches(train_loader)
    accept_it = cycle_batches(accept_loader) if acceptance in held_acceptance_modes and accept_loader is not None else None
    for step in range(1, int(steps) + 1):
        xb, yb = next(it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        base_updates: dict[str, Any] = {}
        candidate_updates: dict[str, Any] = {}
        with torch.no_grad():
            for name, p in named:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                m[name] = g.clone() if name not in m else beta1 * m[name] + (1.0 - beta1) * g
                if variant == "Muon-like" and p.ndim == 2:
                    base_update = orthogonalized_update(m[name])
                else:
                    v[name] = g.square() if name not in v else beta2 * v[name] + (1.0 - beta2) * g.square()
                    m_hat = m[name] / (1.0 - beta1**step)
                    v_hat = v[name] / (1.0 - beta2**step)
                    base_update = m_hat / (v_hat.sqrt() + eps)
                    if variant == "Cautious AdamW":
                        mask = (base_update * g) > 0.0
                        scale = mask.float().mean().clamp_min(1.0e-3)
                        base_update = base_update * mask / scale
                update = base_update
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
                    scale = base_update.norm().clamp_min(1.0e-12) / sig_dir.norm().clamp_min(1.0e-12)
                    update = base_update + float(alpha) * scale * sig_dir
                    norm_match_scales.append(float(scale.item()))
                base_updates[name] = base_update.detach().clone()
                candidate_updates[name] = update.detach().clone()
                snrs.append(float(m[name].norm().item() / (g - m[name]).norm().clamp_min(1.0e-12).item()))

        before = snapshot_params()
        eval_xb, eval_yb = xb, yb
        if acceptance in held_acceptance_modes and accept_it is not None:
            eval_xb, eval_yb = next(accept_it)
            eval_xb = eval_xb.to(device).float()
            eval_yb = eval_yb.to(device).long()
        with torch.no_grad():
            reference_eval_loss = F.cross_entropy(model(eval_xb).float(), eval_yb).detach()
        if signal_mode == "none" or acceptance == "none":
            apply_updates(candidate_updates)
        elif acceptance in {"train_loss_nonworse", "held_train_loss_nonworse"}:
            apply_updates(candidate_updates)
            with torch.no_grad():
                candidate_loss = F.cross_entropy(model(eval_xb).float(), eval_yb).detach()
            acceptance_total += 1
            candidate_delta = float(candidate_loss.item() - reference_eval_loss.item())
            candidate_loss_delta_sum += candidate_delta
            if candidate_delta > float(acceptance_tol):
                acceptance_rejects += 1
                restore_params(before)
                apply_updates(base_updates)
            else:
                acceptance_accepts += 1
        else:
            apply_updates(candidate_updates)
            with torch.no_grad():
                candidate_loss = F.cross_entropy(model(eval_xb).float(), eval_yb).detach()
            candidate_params = snapshot_params()
            restore_params(before)
            apply_updates(base_updates)
            with torch.no_grad():
                base_step_loss = F.cross_entropy(model(eval_xb).float(), eval_yb).detach()
            acceptance_total += 1
            candidate_delta = float(candidate_loss.item() - base_step_loss.item())
            candidate_loss_delta_sum += candidate_delta
            if candidate_delta > float(acceptance_tol):
                acceptance_rejects += 1
            else:
                acceptance_accepts += 1
                restore_params(candidate_params)

        with torch.no_grad():
            chosen_update_norm = 0.0
            for old, (_name, p) in zip(before, named):
                chosen_update_norm += float((p.detach() - old).norm().item()) ** 2
            update_norms.append(math.sqrt(chosen_update_norm))
            if variant == "Schedule-Free AdamW":
                for name, p in named:
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
            "direct_optimizer_acceptance": acceptance if signal_mode != "none" else "",
            "direct_optimizer_acceptance_tol": acceptance_tol if signal_mode != "none" else "",
            "direct_optimizer_accept_rate": ""
            if signal_mode == "none" or not acceptance_total
            else acceptance_accepts / max(1, acceptance_total),
            "direct_optimizer_reject_count": "" if signal_mode == "none" else acceptance_rejects,
            "direct_optimizer_candidate_loss_delta_mean": ""
            if signal_mode == "none" or not acceptance_total
            else candidate_loss_delta_sum / max(1, acceptance_total),
            "implementation_level": (
                "schedule_free_averaging_approx_v22_35_acceptance"
                if variant == "Schedule-Free AdamW"
                else ("svd_orthogonalized_momentum_muon_like_v22_35_acceptance" if variant == "Muon-like" else "manual_cautious_adamw_v22_35_acceptance")
            ),
        }
    )
    return ev


def stage_g_optimizer_direct_probe(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.run_v22_30_fidelity_ladder import make_kan, make_mlp

    device = torch_device(args.optimizer_device)
    raw_rows: list[dict[str, Any]] = []
    group_base: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    group_controls: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    availability_rows: list[dict[str, Any]] = []

    def fresh_model_and_loaders(dataset: str, seed: int, carrier: str, model_seed: int) -> tuple[Any, Any, Any, Any, int, int, Any]:
        train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta = make_v22_35_c11_loaders(
            dataset,
            int(args.optimizer_train_size),
            int(args.optimizer_test_size),
            int(args.batch_size),
            int(seed),
            tier2_download=bool(args.tier2_download),
        )
        availability_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "carrier": carrier,
                "status": "available",
                "task_tier": meta.get("task_tier", ""),
                "source_kind": meta.get("source_kind", ""),
                "effective_train_rows": meta.get("effective_train_rows", ""),
                "effective_test_rows": meta.get("effective_test_rows", ""),
                "source_artifact": "direct_v22_35_optimizer_probe_loader",
            }
        )
        if carrier == "MLP":
            model = make_mlp(input_dim, output_dim, int(args.hidden), int(model_seed), device).to(device)
        else:
            model = make_kan(input_dim, output_dim, int(args.hidden), int(model_seed), device, x_stats, carrier).to(device)
        return model, train_loader, held_loader, test_loader, int(input_dim), int(output_dim), x_stats

    def run_one(dataset: str, seed: int, carrier: str, optimizer_name: str, mode: str) -> dict[str, Any]:
        arch = "MLP" if carrier == "MLP" else "KAN"
        arch_offset = 0 if carrier == "MLP" else (1000 if carrier == "D-CHE" else 2000)
        model_seed = int(seed) + 91_000 + arch_offset
        model, train_loader, held_loader, test_loader, _input_dim, output_dim, _x_stats = fresh_model_and_loaders(
            dataset, seed, carrier, model_seed
        )
        ev = train_manual_strong_optimizer_v22_35(
            model,
            train_loader,
            test_loader,
            device,
            output_dim,
            variant=optimizer_name,
            steps=int(args.optimizer_steps),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            seed=int(seed) + 73_000,
            signal_mode=mode,
            alpha=float(args.optimizer_signal_alpha),
            acceptance=str(args.optimizer_acceptance),
            acceptance_tol=float(args.optimizer_acceptance_tol),
            accept_loader=held_loader,
        )
        implementation = ev.get("implementation_level", "")
        if optimizer_name == "AdamW":
            implementation = "manual_adamw_reference_loop_v22_35_acceptance"
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": arch,
            "carrier": carrier,
            "optimizer_name": optimizer_name,
            "training_variant": "optimizer_alone" if mode == "none" else ("target_FU_signal" if mode == "signal" else f"same_optimizer_{mode}_control"),
            "FU_target_family": "optimizer_geometry_norm_matched_slow_signal_FU" if mode == "signal" else "",
            "control_variant": "" if mode in {"none", "signal"} else mode,
            "NLL": ev.get("NLL", ""),
            "accuracy": ev.get("accuracy", ""),
            "ECE": ev.get("ECE", ""),
            "Brier": ev.get("Brier", ""),
            "tail_q95": ev.get("tail_q95", ""),
            "tail_q99": ev.get("tail_q99", ""),
            "optimizer_step_ms": ev.get("train_step_ms", ""),
            "update_norm": ev.get("update_norm", ""),
            "update_SNR": ev.get("update_SNR", ""),
            "update_spectrum": ev.get("singular_value_spectrum_top_to_mean", ""),
            "gradient_variance": "",
            "singular_order_preservation": "",
            "norm_match_scale_mean": ev.get("norm_match_scale_mean", ""),
            "direct_optimizer_acceptance": ev.get("direct_optimizer_acceptance", ""),
            "direct_optimizer_acceptance_tol": ev.get("direct_optimizer_acceptance_tol", ""),
            "direct_optimizer_accept_rate": ev.get("direct_optimizer_accept_rate", ""),
            "direct_optimizer_reject_count": ev.get("direct_optimizer_reject_count", ""),
            "direct_optimizer_candidate_loss_delta_mean": ev.get("direct_optimizer_candidate_loss_delta_mean", ""),
            "implementation_level": implementation,
            "known_simplifications": (
                "Direct v22.35 probe uses v22.30 manual strong-optimizer fallback with norm-matched slow-signal FU; "
                "not promoted as official full-loop while Part B/D gates remain closed."
            ),
            "status": "direct_v22_35_optimizer_probe",
            "source_artifact": "direct_v22_35_optimizer_strong_family_probe",
        }

    for dataset in split_csv(args.optimizer_datasets):
        for seed in split_csv(args.optimizer_seeds, int):
            for carrier in split_csv(args.optimizer_families):
                carrier = "MLP" if str(carrier).upper() == "MLP" else str(carrier)
                for optimizer_name in split_csv(args.optimizer_variants):
                    key = (str(dataset), str(seed), str(carrier), str(optimizer_name), str(args.optimizer_steps))
                    try:
                        base = run_one(str(dataset), int(seed), carrier, str(optimizer_name), "none")
                        raw_rows.append(base)
                        group_base[key] = base
                        controls: list[dict[str, Any]] = []
                        for mode in ["signal", "random", "signflip", "shuffled"]:
                            row = run_one(str(dataset), int(seed), carrier, str(optimizer_name), mode)
                            raw_rows.append(row)
                            if mode != "signal":
                                controls.append(row)
                        group_controls[key] = controls
                    except Exception as exc:
                        availability_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "carrier": carrier,
                                "optimizer_name": optimizer_name,
                                "status": "task_or_optimizer_unavailable",
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "source_artifact": "direct_v22_35_optimizer_strong_family_probe",
                            }
                        )

    strongest_by_arch: dict[tuple[str, str, str, str], float] = {}
    for base in group_base.values():
        nll = finite_float(base.get("NLL"))
        if nll is None:
            continue
        key = (str(base.get("dataset", "")), str(base.get("seed", "")), str(base.get("architecture", "")), str(base.get("carrier", "")))
        strongest_by_arch[key] = min(strongest_by_arch.get(key, math.inf), nll)

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("carrier", "")),
            str(row.get("optimizer_name", "")),
            str(args.optimizer_steps),
        )
        arch_key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("architecture", "")), str(row.get("carrier", "")))
        base = group_base.get(key, {})
        base_nll = finite_float(base.get("NLL"))
        row_nll = finite_float(row.get("NLL"))
        base_acc = finite_float(base.get("accuracy"))
        row_acc = finite_float(row.get("accuracy"))
        base_ms = finite_float(base.get("optimizer_step_ms"))
        row_ms = finite_float(row.get("optimizer_step_ms"))
        controls = group_controls.get(key, [])
        control_nlls = [v for v in (finite_float(c.get("NLL")) for c in controls) if v is not None]
        best_control = min(control_nlls) if control_nlls else math.inf
        strongest = strongest_by_arch.get(arch_key, math.inf)
        is_signal = row.get("training_variant") == "target_FU_signal"
        out = dict(row)
        out.update(
            {
                "base_optimizer_NLL": "" if base_nll is None else base_nll,
                "NLL_delta_vs_optimizer": "" if row_nll is None or base_nll is None else row_nll - base_nll,
                "accuracy_delta_vs_optimizer": "" if row_acc is None or base_acc is None else row_acc - base_acc,
                "best_same_optimizer_control_NLL": "" if not math.isfinite(best_control) else best_control,
                "beats_optimizer_baseline": int(is_signal and row_nll is not None and base_nll is not None and row_nll < base_nll),
                "beats_same_optimizer_controls": int(is_signal and row_nll is not None and math.isfinite(best_control) and row_nll < best_control),
                "strongest_completed_optimizer_baseline_NLL": "" if not math.isfinite(strongest) else strongest,
                "beats_strongest_completed_optimizer_baseline": int(is_signal and row_nll is not None and math.isfinite(strongest) and row_nll < strongest),
                "controller_overhead_ratio": ""
                if row_ms is None or base_ms is None or base_ms <= 0.0
                else (row_ms - base_ms) / base_ms,
                "official_eligible": 0,
                "not_official_reason": "Part B/D/full-loop gates remain closed; this is a direct optimizer interaction probe only.",
            }
        )
        rows.append(out)

    signal_rows = [r for r in rows if r.get("training_variant") == "target_FU_signal"]
    non_base_candidate_rows = [r for r in rows if r.get("training_variant") != "optimizer_alone"]
    control_candidate_rows = [
        r
        for r in rows
        if r.get("training_variant") not in {"optimizer_alone", "target_FU_signal"}
    ]
    beats_own = sum(int_flag(r.get("beats_optimizer_baseline")) for r in signal_rows)
    beats_controls = sum(int_flag(r.get("beats_same_optimizer_controls")) for r in signal_rows)
    beats_strongest = sum(int_flag(r.get("beats_strongest_completed_optimizer_baseline")) for r in signal_rows)
    overheads = [v for v in (finite_float(r.get("controller_overhead_ratio")) for r in signal_rows) if v is not None]
    signal_accept_rates = [v for v in (finite_float(r.get("direct_optimizer_accept_rate")) for r in signal_rows) if v is not None]
    candidate_accept_rates = [v for v in (finite_float(r.get("direct_optimizer_accept_rate")) for r in non_base_candidate_rows) if v is not None]
    signal_rejects = sum(int_flag(r.get("direct_optimizer_reject_count")) for r in signal_rows)
    control_rejects = sum(int_flag(r.get("direct_optimizer_reject_count")) for r in control_candidate_rows)
    max_overhead = max(overheads) if overheads else ""
    optimizer_names = sorted({str(r.get("optimizer_name", "")) for r in signal_rows if r.get("optimizer_name")})
    summary = {
        "optimizer_probe_rows": len(rows),
        "signal_fu_rows": len(signal_rows),
        "optimizer_family_count": len(optimizer_names),
        "optimizer_names": ",".join(optimizer_names),
        "beats_own_optimizer_baseline_rows": beats_own,
        "beats_same_optimizer_controls_rows": beats_controls,
        "beats_strongest_completed_optimizer_baseline_rows": beats_strongest,
        "max_controller_overhead_ratio": max_overhead,
        "optimizer_signal_alpha": args.optimizer_signal_alpha,
        "optimizer_acceptance": args.optimizer_acceptance,
        "optimizer_acceptance_tol": args.optimizer_acceptance_tol,
        "signal_accept_rate_mean": "" if not signal_accept_rates else sum(signal_accept_rates) / len(signal_accept_rates),
        "candidate_accept_rate_mean": "" if not candidate_accept_rates else sum(candidate_accept_rates) / len(candidate_accept_rates),
        "signal_reject_count_total": signal_rejects,
        "control_reject_count_total": control_rejects,
        "exploration_gate_pass": int(
            len(signal_rows) >= 9
            and beats_own >= 6
            and beats_controls >= 6
            and beats_strongest >= 5
            and bool(overheads)
            and max(overheads) <= 0.25
        ),
        "official_candidate_gate_pass": 0,
        "status": "direct_probe_completed_not_promoted" if rows else "direct_probe_no_rows",
        "not_official_reason": "Direct probe is useful Part G evidence but upstream Part B/D gates still block official full-loop promotion.",
    }
    history_path = OUT_ROOT / "v22_35_optimizer_direct_probe_attempt_history.csv"
    history = read_rows(history_path)
    artifact_suffix = ""
    if str(args.optimizer_acceptance) != "none":
        safe_tol = str(args.optimizer_acceptance_tol).replace("-", "neg").replace(".", "p")
        safe_alpha = str(args.optimizer_signal_alpha).replace("-", "neg").replace(".", "p")
        artifact_suffix = f"_{str(args.optimizer_acceptance).replace('-', '_')}_alpha{safe_alpha}_tol{safe_tol}"
    matrix_path = OUT_ROOT / f"v22_35_optimizer_direct_probe{artifact_suffix}_matrix.csv"
    summary_path = OUT_ROOT / f"v22_35_optimizer_direct_probe{artifact_suffix}_summary.csv"
    availability_path = OUT_ROOT / f"v22_35_optimizer_direct_probe{artifact_suffix}_availability_matrix.csv"
    history.append(
        {
            "timestamp": now_sg(),
            "optimizer_datasets": args.optimizer_datasets,
            "optimizer_seeds": args.optimizer_seeds,
            "model_families": args.optimizer_families,
            "optimizer_variants": args.optimizer_variants,
            "optimizer_steps": args.optimizer_steps,
            "optimizer_signal_alpha": args.optimizer_signal_alpha,
            "optimizer_acceptance": args.optimizer_acceptance,
            "optimizer_acceptance_tol": args.optimizer_acceptance_tol,
            "matrix_artifact": str(matrix_path.relative_to(ROOT)),
            "summary_artifact": str(summary_path.relative_to(ROOT)),
            **summary,
            "command": " ".join([shlex.quote(a) for a in sys.argv]),
        }
    )
    payload_rows = rows or [{"status": summary["status"]}]
    write_rows(matrix_path, payload_rows)
    write_rows(summary_path, [summary])
    write_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_matrix.csv", payload_rows)
    write_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_summary.csv", [summary])
    write_rows(history_path, history)
    availability_payload = availability_rows or [{"status": "no_availability_rows"}]
    write_rows(availability_path, availability_payload)
    write_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_availability_matrix.csv", availability_payload)
    append_exec(
        "run direct v22.35 optimizer strong-family probe with norm-matched signal FU and matched controls",
        task_id="G_optimizer_direct_probe",
        status="pass" if rows else "warn",
        gpu=args.optimizer_device,
        files=(
            "results/v22_35/v22_35_optimizer_direct_probe_matrix.csv, "
            "results/v22_35/v22_35_optimizer_direct_probe_summary.csv, "
            f"{matrix_path.relative_to(ROOT)}, "
            f"{summary_path.relative_to(ROOT)}, "
            "results/v22_35/v22_35_optimizer_direct_probe_attempt_history.csv, "
            "results/v22_35/v22_35_optimizer_direct_probe_availability_matrix.csv"
        ),
        note=(
            f"signal_fu_rows={summary['signal_fu_rows']}; "
            f"beats_own={summary['beats_own_optimizer_baseline_rows']}; "
            f"beats_controls={summary['beats_same_optimizer_controls_rows']}; "
            f"beats_strongest={summary['beats_strongest_completed_optimizer_baseline_rows']}; "
            f"acceptance={summary['optimizer_acceptance']}; "
            f"signal_accept_rate_mean={summary['signal_accept_rate_mean']}; "
            f"signal_rejects={summary['signal_reject_count_total']}; "
            f"exploration_gate_pass={summary['exploration_gate_pass']}"
        ),
    )
    return summary


def materialize_optimizer_direct_four_square() -> dict[str, Any]:
    rows = read_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_matrix.csv")
    if not rows or rows == [{"status": "direct_probe_no_rows"}]:
        write_rows(
            OUT_ROOT / "v22_35_optimizer_direct_four_square_matrix.csv",
            [{"status": "not_run_no_optimizer_direct_probe_rows"}],
        )
        summary = {"four_square_rows": 0, "hard_task_exploration_gate_pass": 0, "status": "not_run_no_optimizer_direct_probe_rows"}
        write_rows(OUT_ROOT / "v22_35_optimizer_direct_four_square_summary.csv", [summary])
        return summary

    by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("carrier", "")),
            str(row.get("optimizer_name", "")),
            str(row.get("training_variant", "")),
        )
        by_key[key] = row

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("training_variant") != "target_FU_signal" or row.get("carrier") not in {"D-CHE", "D-FOU"}:
            continue
        dataset = str(row.get("dataset", ""))
        seed = str(row.get("seed", ""))
        carrier = str(row.get("carrier", ""))
        optimizer = str(row.get("optimizer_name", ""))
        kan_base = by_key.get((dataset, seed, carrier, optimizer, "optimizer_alone"), {})
        mlp_base = by_key.get((dataset, seed, "MLP", optimizer, "optimizer_alone"), {})
        mlp_fu = by_key.get((dataset, seed, "MLP", optimizer, "target_FU_signal"), {})
        if not kan_base or not mlp_base or not mlp_fu:
            continue
        kan_fu_nll = finite_float(row.get("NLL"))
        kan_base_nll = finite_float(kan_base.get("NLL"))
        mlp_fu_nll = finite_float(mlp_fu.get("NLL"))
        mlp_base_nll = finite_float(mlp_base.get("NLL"))
        if None in {kan_fu_nll, kan_base_nll, mlp_fu_nll, mlp_base_nll}:
            continue
        delta_kan = float(kan_fu_nll) - float(kan_base_nll)
        delta_mlp = float(mlp_fu_nll) - float(mlp_base_nll)
        gap_bp = float(kan_base_nll) - float(mlp_base_nll)
        gap_fu = float(kan_fu_nll) - float(mlp_fu_nll)
        gap_reduction = gap_bp - gap_fu
        controls_fail = int_flag(row.get("beats_same_optimizer_controls"))
        if delta_kan < 0.0 and delta_mlp >= 0.0 and controls_fail:
            klass = "TrueKANGain"
        elif delta_kan < 0.0 and delta_mlp < 0.0 and gap_reduction > 0.0 and controls_fail:
            klass = "BothGain"
        elif delta_mlp > 0.0 and gap_reduction > 0.0:
            klass = "MLPDegradationDriven"
        elif not controls_fail:
            klass = "ControlExplained"
        else:
            klass = "NoGainOrMLPDominates"
        out_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "task_tier": "Tier2_tabular" if dataset.lower() == "wine" else "Tier1_hard_vision",
                "optimizer_name": optimizer,
                "kan_carrier": carrier,
                "MLP_strong_NLL": mlp_base_nll,
                "MLP_FU_NLL": mlp_fu_nll,
                "KAN_strong_NLL": kan_base_nll,
                "KAN_FU_NLL": kan_fu_nll,
                "Delta_MLP_NLL": delta_mlp,
                "Delta_KAN_NLL": delta_kan,
                "Gap_BP": gap_bp,
                "Gap_FU": gap_fu,
                "GapReduction": gap_reduction,
                "MLP_FU_nonworse_vs_MLP_strong": int(delta_mlp <= 0.0),
                "KAN_FU_improves_KAN_strong": int(delta_kan < 0.0),
                "KAN_FU_beats_MLP_FU": int(float(kan_fu_nll) < float(mlp_fu_nll)),
                "KAN_FU_beats_same_optimizer_controls": controls_fail,
                "MLP_FU_beats_same_optimizer_controls": mlp_fu.get("beats_same_optimizer_controls", ""),
                "TrueKANGain_class": klass,
                "epsilon_source": "not_used_optimizer_direct_four_square_delta_sign_only",
                "official_eligible": 0,
                "not_official_reason": "Derived from optimizer direct probe; upstream target/branch gates remain closed.",
                "source_artifact": "results/v22_35/v22_35_optimizer_direct_probe_matrix.csv",
            }
        )

    n = len(out_rows)
    true_rows = sum(1 for r in out_rows if r.get("TrueKANGain_class") == "TrueKANGain")
    both_rows = sum(1 for r in out_rows if r.get("TrueKANGain_class") == "BothGain")
    mlp_nonworse = sum(int_flag(r.get("MLP_FU_nonworse_vs_MLP_strong")) for r in out_rows)
    kan_improve = sum(int_flag(r.get("KAN_FU_improves_KAN_strong")) for r in out_rows)
    controls_fail = sum(int_flag(r.get("KAN_FU_beats_same_optimizer_controls")) for r in out_rows)
    summary = {
        "four_square_rows": n,
        "TrueKANGain_rows": true_rows,
        "BothGain_rows": both_rows,
        "TrueKANGain_plus_BothGain_rate": rate(true_rows + both_rows, n),
        "KAN_FU_improves_KAN_strong_rows": kan_improve,
        "KAN_FU_improves_KAN_strong_rate": rate(kan_improve, n),
        "MLP_FU_nonworse_vs_MLP_strong_rows": mlp_nonworse,
        "MLP_FU_nonworse_vs_MLP_strong_rate": rate(mlp_nonworse, n),
        "KAN_FU_beats_same_optimizer_controls_rows": controls_fail,
        "KAN_FU_beats_same_optimizer_controls_rate": rate(controls_fail, n),
        "hard_task_exploration_gate_pass": int(
            n > 0
            and rate(kan_improve, n) >= 0.50
            and rate(mlp_nonworse, n) >= 0.80
            and rate(true_rows + both_rows, n) >= 0.25
            and rate(controls_fail, n) >= 0.50
        ),
        "official_candidate_gate_pass": 0,
        "status": "direct_four_square_completed_not_promoted" if n else "direct_four_square_no_rows",
        "not_official_reason": "Even if direct four-square improves, Part B/D target/branch gates remain closed.",
    }
    kan_internal_rows = [
        r
        for r in out_rows
        if int_flag(r.get("KAN_FU_improves_KAN_strong")) and int_flag(r.get("KAN_FU_beats_same_optimizer_controls"))
    ]
    kan_internal_summary = {
        "kan_internal_candidate_rows": n,
        "kan_internal_gain_and_control_fail_rows": len(kan_internal_rows),
        "kan_internal_gain_and_control_fail_rate": rate(len(kan_internal_rows), n),
        "kan_internal_exploration_pass": int(n > 0 and rate(len(kan_internal_rows), n) >= 0.50),
        "mlp_nonworse_rows": mlp_nonworse,
        "mlp_nonworse_rate": rate(mlp_nonworse, n),
        "mlp_degradation_blocker_rows": n - mlp_nonworse,
        "route_hint": "R9-KANInternalGainOpened_diagnostic_not_official" if n > 0 and rate(len(kan_internal_rows), n) >= 0.50 else "KANInternalNoGo",
        "official_candidate_gate_pass": 0,
        "not_official_reason": "Plan 7.6 downgrade: MLP+FU non-worse gate failed, so this can only be KAN-internal optimizer interaction evidence.",
    }
    write_rows(OUT_ROOT / "v22_35_optimizer_direct_four_square_matrix.csv", out_rows or [{"status": summary["status"]}])
    write_rows(OUT_ROOT / "v22_35_optimizer_direct_four_square_summary.csv", [summary])
    write_rows(OUT_ROOT / "v22_35_optimizer_direct_kan_internal_summary.csv", [kan_internal_summary])
    return summary


def stage_g_h_gate_matrices() -> tuple[dict[str, Any], dict[str, Any]]:
    branch_summary = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
    gap_summary = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    can_full_loop = int_flag(branch_summary.get("branch_gate_exploration_pass")) and int_flag(gap_summary.get("exploration_gate_pass"))
    opt_rows: list[dict[str, Any]] = []
    for row in read_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_matrix.csv"):
        if row.get("status") in {"direct_probe_no_rows"}:
            continue
        opt_rows.append(
            {
                **row,
                "status": (
                    "direct_v22_35_optimizer_probe_gate_blocked_for_official"
                    if not can_full_loop
                    else "direct_v22_35_optimizer_probe"
                ),
            }
        )
    for row in read_rows(V22_30 / "v22_30_T7_fu_optimizer_interaction_matrix.csv")[:80]:
        opt_rows.append(
            {
                "optimizer_name": row.get("optimizer", ""),
                "FU_target_family": "historical_signal_optimizer_interaction",
                "NLL_delta_vs_optimizer": row.get("FU_gain_over_optimizer", ""),
                "AUC_delta_vs_optimizer": "",
                "accuracy_delta_vs_optimizer": "",
                "beats_optimizer_baseline": int((finite_float(row.get("FU_gain_over_optimizer"), 0.0) or 0.0) < 0.0) if row.get("FU_gain_over_optimizer", "") != "" else "",
                "beats_same_optimizer_controls": "",
                "update_spectrum": "",
                "update_SNR": row.get("update_SNR", ""),
                "gradient_variance": "",
                "singular_order_preservation": "",
                "controller_overhead_ratio": "",
                "status": "historical_readback_not_official_v22_35" if not can_full_loop else "historical_readback",
                "source_artifact": "results/v22_30/v22_30_T7_fu_optimizer_interaction_matrix.csv",
            }
        )
    if not opt_rows:
        opt_rows.append({"status": "gate_blocked_not_run", "reason": "No optimizer historical artifact found; branch/gap gates not open."})
    write_rows(OUT_ROOT / "v22_35_optimizer_integrated_fu_matrix.csv", opt_rows)

    hard_rows: list[dict[str, Any]] = []
    opt_four_summary = materialize_optimizer_direct_four_square()
    for row in read_rows(OUT_ROOT / "v22_35_optimizer_direct_four_square_matrix.csv"):
        if row.get("status") in {"not_run_no_optimizer_direct_probe_rows", "direct_four_square_no_rows"}:
            continue
        hard_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "task_tier": row.get("task_tier", ""),
                "optimizer_name": row.get("optimizer_name", ""),
                "kan_carrier": row.get("kan_carrier", ""),
                "final_test_NLL": row.get("KAN_FU_NLL", ""),
                "final_test_accuracy": "",
                "AUC_loss_time": "",
                "wallclock_adjusted_AUC": "",
                "ECE": "",
                "Brier": "",
                "tail_loss_q95": "",
                "tail_loss_q99": "",
                "hard_slice_NLL": "",
                "low_margin_accuracy": "",
                "forgetting": "",
                "backward_transfer": "",
                "grokking_delay": "",
                "full_loop_ratio": "",
                "controller_overhead_ratio": "",
                "Gap_BP": row.get("Gap_BP", ""),
                "Gap_FU": row.get("Gap_FU", ""),
                "GapReduction": row.get("GapReduction", ""),
                "Delta_MLP_NLL": row.get("Delta_MLP_NLL", ""),
                "Delta_KAN_NLL": row.get("Delta_KAN_NLL", ""),
                "TrueKANGain_class": row.get("TrueKANGain_class", ""),
                "status": "direct_optimizer_four_square_gate_blocked_for_official",
                "source_artifact": "results/v22_35/v22_35_optimizer_direct_four_square_matrix.csv",
            }
        )
    for row in read_rows(V22_33 / "v22_33_gap_truth_matrix.csv"):
        hard_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "task_tier": row.get("task_tier", ""),
                "final_test_NLL": "",
                "final_test_accuracy": "",
                "AUC_loss_time": "",
                "wallclock_adjusted_AUC": "",
                "ECE": "",
                "Brier": "",
                "tail_loss_q95": "",
                "tail_loss_q99": "",
                "hard_slice_NLL": "",
                "low_margin_accuracy": "",
                "forgetting": "",
                "backward_transfer": "",
                "grokking_delay": "",
                "full_loop_ratio": "",
                "controller_overhead_ratio": "",
                "Gap_BP": "",
                "Gap_FU": "",
                "GapReduction": row.get("GapReduction_NLL", ""),
                "TrueKANGain_class": row.get("gap_reduction_class", ""),
                "status": "historical_gap_readback_gate_blocked_for_new_full_loop" if not can_full_loop else "historical_gap_readback",
                "source_artifact": "results/v22_33/v22_33_gap_truth_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_35_hard_task_four_square_matrix.csv", hard_rows)

    continual = read_rows(V22_34 / "v22_34_D8_continual_boundary_control_win_matrix.csv")
    cont_rows = [
        {
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "target_family": r.get("target_family", ""),
            "base_forgetting": r.get("base_forgetting", ""),
            "real_forgetting": r.get("real_forgetting", ""),
            "relative_forgetting_reduction": r.get("relative_forgetting_reduction_vs_base", ""),
            "final_average_accuracy_delta": r.get("task1_accuracy_delta_vs_base_after_task1", ""),
            "baseline_old_task_accuracy_after_new_task_nondegenerate": int((finite_float(r.get("base_forgetting"), 1.0) or 1.0) < 0.95),
            "controls_fail": int_flag(r.get("strict_fit_boundary_pass")),
            "status": "v22_34_D8_readback_gate_blocked",
            "source_artifact": "results/v22_34/v22_34_D8_continual_boundary_control_win_matrix.csv",
        }
        for r in continual
    ]
    for r in read_rows(OUT_ROOT / "v22_35_continual_nondegenerate_baseline_matrix.csv"):
        if r.get("status") in {"not_run_no_rows", "task_unavailable"}:
            cont_rows.append(
                {
                    "dataset": r.get("dataset", "Class_MNIST_T0T1"),
                    "seed": r.get("seed", ""),
                    "target_family": "D8_nondegenerate_baseline_sweep",
                    "base_forgetting": r.get("base_forgetting", ""),
                    "real_forgetting": "",
                    "relative_forgetting_reduction": "",
                    "final_average_accuracy_delta": "",
                    "baseline_old_task_accuracy_after_new_task_nondegenerate": r.get("baseline_old_task_accuracy_after_new_task_nondegenerate", 0),
                    "controls_fail": "",
                    "task0_accuracy_before_task1": r.get("task0_accuracy_before_task1", ""),
                    "base_task0_accuracy_after_task1": r.get("base_task0_accuracy_after_task1", ""),
                    "base_task1_accuracy_after_task1": r.get("base_task1_accuracy_after_task1", ""),
                    "status": r.get("status", ""),
                    "source_artifact": "results/v22_35/v22_35_continual_nondegenerate_baseline_matrix.csv",
                }
            )
            continue
        cont_rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "target_family": "D8_nondegenerate_baseline_sweep",
                "base_forgetting": r.get("base_forgetting", ""),
                "real_forgetting": "",
                "relative_forgetting_reduction": "",
                "final_average_accuracy_delta": "",
                "baseline_old_task_accuracy_after_new_task_nondegenerate": r.get("baseline_old_task_accuracy_after_new_task_nondegenerate", ""),
                "controls_fail": "",
                "task0_accuracy_before_task1": r.get("task0_accuracy_before_task1", ""),
                "base_task0_accuracy_after_task1": r.get("base_task0_accuracy_after_task1", ""),
                "base_task1_accuracy_after_task1": r.get("base_task1_accuracy_after_task1", ""),
                "status": r.get("status", ""),
                "source_artifact": "results/v22_35/v22_35_continual_nondegenerate_baseline_matrix.csv",
            }
        )
    for r in read_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_control_win_matrix.csv"):
        if r.get("status") in {"nondegenerate_baseline_not_found", "not_run_no_rows"}:
            continue
        cont_rows.append(
            {
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "target_family": r.get("target_family", "D8_continual_boundary_basis_target_nondegenerate"),
                "base_forgetting": r.get("base_forgetting", ""),
                "real_forgetting": r.get("real_forgetting", ""),
                "relative_forgetting_reduction": r.get("relative_forgetting_reduction_vs_base", ""),
                "final_average_accuracy_delta": r.get("task1_accuracy_delta_vs_base_after_task1", ""),
                "baseline_old_task_accuracy_after_new_task_nondegenerate": r.get("baseline_old_task_accuracy_after_new_task_nondegenerate", ""),
                "controls_fail": int(
                    int_flag(r.get("beats_same_basis_random")) and int_flag(r.get("beats_signflip_basis_control"))
                ),
                "task0_accuracy_before_task1": "",
                "base_task0_accuracy_after_task1": "",
                "base_task1_accuracy_after_task1": "",
                "status": r.get("control_win_explanation", r.get("status", "")),
                "source_artifact": "results/v22_35/v22_35_continual_nondegenerate_boundary_control_win_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_35_continual_matrix.csv", cont_rows or [{"status": "not_run_no_D8_artifact"}])
    grok = read_rows(V22_30 / "v22_30_T6_grokking_matrix.csv")
    write_rows(
        OUT_ROOT / "v22_35_grokking_matrix.csv",
        [
            {
                "task": r.get("task", ""),
                "seed": r.get("seed", ""),
                "variant": r.get("variant", ""),
                "valid_base_grokking_delay_exists": int(bool(r.get("grokking_time"))),
                "grokking_delay": r.get("grokking_time", ""),
                "final_test_accuracy": r.get("test_acc", ""),
                "status": "historical_readback_no_valid_delay" if not r.get("grokking_time") else "historical_readback",
                "source_artifact": "results/v22_30/v22_30_T6_grokking_matrix.csv",
            }
            for r in grok
        ]
        or [{"status": "not_run_no_grokking_artifact"}],
    )
    eff_rows = [
        {
            "optimizer_name": r.get("optimizer_name", r.get("optimizer", "")),
            "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
            "optimizer_step_ms": r.get("optimizer_step_ms", ""),
            "status": r.get("status", "historical_readback"),
            "source_artifact": "results/v22_35/v22_35_optimizer_integrated_fu_matrix.csv",
        }
        for r in opt_rows
    ]
    write_rows(OUT_ROOT / "v22_35_efficiency_matrix.csv", eff_rows)
    summary_g = {
        "optimizer_rows": len(opt_rows),
        "optimizer_direct_four_square_rows": opt_four_summary.get("four_square_rows", 0),
        "optimizer_direct_four_square_gate_pass": opt_four_summary.get("hard_task_exploration_gate_pass", 0),
        "full_loop_gates_open": int(can_full_loop),
    }
    summary_h = {"hard_rows": len(hard_rows), "continual_rows": len(cont_rows), "grokking_rows": len(grok), "full_loop_gates_open": int(can_full_loop)}
    append_exec(
        "materialize optimizer/hard-task/continual/grokking/efficiency matrices with gate status",
        task_id="G_H_optimizer_hard_task_gate_matrices",
        status="pass",
        gpu="3",
        files="results/v22_35/v22_35_optimizer_integrated_fu_matrix.csv, results/v22_35/v22_35_optimizer_direct_four_square_matrix.csv, results/v22_35/v22_35_optimizer_direct_four_square_summary.csv, results/v22_35/v22_35_optimizer_direct_kan_internal_summary.csv, results/v22_35/v22_35_hard_task_four_square_matrix.csv, results/v22_35/v22_35_continual_matrix.csv, results/v22_35/v22_35_grokking_matrix.csv, results/v22_35/v22_35_efficiency_matrix.csv",
        note=(
            f"full_loop_gates_open={int(can_full_loop)}; "
            f"optimizer_direct_four_square_rows={opt_four_summary.get('four_square_rows', 0)}; "
            f"optimizer_direct_four_square_gate_pass={opt_four_summary.get('hard_task_exploration_gate_pass', 0)}"
        ),
    )
    return summary_g, summary_h


def decide_final(code: dict[str, Any], gap: dict[str, Any], t1: dict[str, Any], intrinsic: dict[str, Any], basis: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if not int_flag(code.get("clean_unzip_compileall_pass")) or not int_flag(code.get("clean_unzip_import_pass")) or not int_flag(code.get("official_DGKAN_identity_pass")):
        route = "R0-CodeOrIdentityBlocked"
    elif not int_flag(gap.get("exploration_gate_pass")) and (finite_float(gap.get("ControlExplained_rate"), 1.0) or 1.0) > 0.50:
        route = "R1-GapStillControlExplained"
    elif not int_flag(intrinsic.get("exploration_gate_pass")) and not int_flag(intrinsic.get("repair_exploration_gate_pass")):
        route = "R3-TargetIntrinsicBenefitNoGo"
    elif int_flag(basis.get("basis_fit_rows")) and not int_flag(basis.get("branch_gate_exploration_pass")):
        route = "R4-BasisActuatorFitOpened_NoBranchBenefit"
    elif int_flag(control.get("FlatnessRegularization_rows")):
        route = "R6-ControlWinExplainedByFlatness"
    elif int_flag(control.get("ActuatorSupportOnly_rows")):
        route = "R7-ControlWinExplainedByActuatorSupport"
    else:
        route = "R5-BasisActuatorTransferOpened_NoFullLoop"
    official = int(
        route == "R15-OfficialDGKANFullSuperiorityReady"
        and int_flag(gap.get("official_candidate_gate_pass"))
        and int_flag(basis.get("branch_gate_official_pass"))
    )
    intrinsic_repair_open = int_flag(intrinsic.get("repair_exploration_gate_pass"))
    intrinsic_blocker_text = (
        "Part D raw Level0/readout intrinsic target gate did not open, but sign/scale repair opened nonzero branch candidates; basis branch transfer remains the active downstream gate."
        if intrinsic_repair_open
        else "Part D Level0/readout intrinsic target gate did not open, including sign/scale repair."
    )
    final = {
        "final_route": route,
        "official_full_superiority_ready": official,
        "latest_status_timestamp": now_sg(),
        "code_truth": code,
        "gap_summary": gap,
        "t1_summary": t1,
        "target_intrinsic_summary": intrinsic,
        "basis_transfer_summary": basis,
        "control_win_summary": control,
        "full_loop_blockers": [
            {
                "status": "gate_blocked" if not int_flag(gap.get("exploration_gate_pass")) else "pass",
                "blocker": "Part B TrueKANGain+BothGain exploration gate did not open.",
                "source_artifact": "results/v22_35/v22_35_gap_truth_recalibrated_summary.csv",
            },
            {
                "status": "gate_blocked" if not int_flag(intrinsic.get("exploration_gate_pass")) and not intrinsic_repair_open else "pass",
                "blocker": intrinsic_blocker_text,
                "source_artifact": "results/v22_35/v22_35_target_intrinsic_benefit_summary.csv",
            },
            {
                "status": "gate_blocked" if not int_flag(basis.get("branch_gate_exploration_pass")) else "pass",
                "blocker": "Part D basis branch transfer gate did not open under v22.35 row-local epsilon.",
                "source_artifact": "results/v22_35/v22_35_basis_transfer_summary.csv",
            },
        ],
        "no_fabricated_rows_claim": "All numeric rows are direct v22.35 commands or named v22.34/v22.33/v22.30 artifact readbacks. Missing evidence is explicit not_run/gate_blocked.",
    }
    write_json(OUT_ROOT / "v22_35_final_route.json", final)
    return final


def write_figures() -> None:
    gap = read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_matrix.csv")
    noise = read_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv")
    selectors = read_rows(OUT_ROOT / "v22_35_T1_selector_branch_matrix.csv")
    intrinsic = read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_matrix.csv")
    intrinsic_repair = read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_branch_matrix.csv")
    fit = read_rows(OUT_ROOT / "v22_35_basis_actuator_transfer_matrix.csv")
    branch = read_rows(OUT_ROOT / "v22_35_basis_branch_transfer_matrix.csv")
    control = read_rows(OUT_ROOT / "v22_35_control_win_decomposition_matrix.csv")
    curv = read_rows(OUT_ROOT / "v22_35_curvature_representation_matrix.csv")
    optim = read_rows(OUT_ROOT / "v22_35_optimizer_integrated_fu_matrix.csv")
    hard = read_rows(OUT_ROOT / "v22_35_hard_task_four_square_matrix.csv")
    continual = read_rows(OUT_ROOT / "v22_35_continual_matrix.csv")
    grok = read_rows(OUT_ROOT / "v22_35_grokking_matrix.csv")
    write_simple_svg(FIG_ROOT / "v22_35_gap_truth_recalibrated_stacked_bar.svg", "v22.35 gap truth classes", [{"label": r.get("gap_reduction_class_v22_35", ""), "value": 1} for r in gap])
    write_simple_svg(FIG_ROOT / "v22_35_epsilon_global_vs_row_local_panel.svg", "v22.35 row-local epsilon", [{"label": f"{r.get('dataset')} {r.get('carrier')}", "value": r.get("epsilon_row", 0)} for r in noise])
    write_simple_svg(FIG_ROOT / "v22_35_selector_vs_control_hierarchy.svg", "v22.35 selectors", [{"label": r.get("selector_name", ""), "value": branch_delta(r) or 0} for r in selectors])
    write_simple_svg(
        FIG_ROOT / "v22_35_target_intrinsic_vs_basis_transfer.svg",
        "v22.35 target intrinsic",
        [{"label": r.get("target_family", ""), "value": r.get("readout_exact_NLL_delta", 0)} for r in intrinsic]
        + [{"label": f"repair:{r.get('target_family', '')}:H{r.get('branch_H', '')}", "value": r.get("real_NLL_delta", 0)} for r in intrinsic_repair],
    )
    write_simple_svg(FIG_ROOT / "v22_35_basis_residual_vs_branch_gain.svg", "v22.35 basis residual", [{"label": r.get("basis_bank", ""), "value": r.get("basis_projection_residual", 0)} for r in fit])
    write_simple_svg(FIG_ROOT / "v22_35_control_win_cause_waterfall.svg", "v22.35 control classes", [{"label": r.get("control_win_class", ""), "value": 1} for r in control])
    write_simple_svg(FIG_ROOT / "v22_35_sharpness_vs_control_win_scatter.svg", "v22.35 sharpness", [{"label": r.get("target_family", ""), "value": r.get("sharpness_delta_real", 0)} for r in curv])
    write_simple_svg(FIG_ROOT / "v22_35_optimizer_family_delta_panel.svg", "v22.35 optimizer deltas", [{"label": r.get("optimizer_name", ""), "value": r.get("NLL_delta_vs_optimizer", 0)} for r in optim])
    write_simple_svg(FIG_ROOT / "v22_35_KAN_vs_MLPFU_gap_reduction_panel.svg", "v22.35 gap reduction", [{"label": r.get("dataset", ""), "value": r.get("GapReduction", 0)} for r in hard])
    write_simple_svg(FIG_ROOT / "v22_35_continual_forgetting_curves.svg", "v22.35 continual", [{"label": r.get("dataset", ""), "value": r.get("relative_forgetting_reduction", 0)} for r in continual])
    write_simple_svg(FIG_ROOT / "v22_35_grokking_delay_curves.svg", "v22.35 grokking", [{"label": r.get("variant", ""), "value": r.get("grokking_delay", 0)} for r in grok])


def materialize_selector_control_win_decomposition() -> dict[str, Any]:
    selector_specs = [
        (
            "C11_edge_safe_control_orthogonal_direction",
            "H800_current",
            OUT_ROOT / "v22_35_T1_C11_edge_safe_selector_matrix.csv",
            OUT_ROOT / "v22_35_T1_C11_edge_safe_branch_matrix.csv",
        ),
        (
            "C11_edge_safe_control_orthogonal_direction",
            "H400_archive_seeds012",
            OUT_ROOT / "v22_35_T1_C11_edge_safe_selector_matrix_H400_seeds012.csv",
            OUT_ROOT / "v22_35_T1_C11_edge_safe_branch_matrix_H400_seeds012.csv",
        ),
        (
            "C10_repaired_margin_NLL_constrained_direction",
            "H800_current",
            OUT_ROOT / "v22_35_T1_C10_repaired_selector_matrix.csv",
            OUT_ROOT / "v22_35_T1_C10_repaired_branch_matrix.csv",
        ),
        (
            "C11_cohort_influence_repair_direction",
            "H800_current",
            OUT_ROOT / "v22_35_T1_C11_cohort_influence_selector_matrix.csv",
            OUT_ROOT / "v22_35_T1_C11_cohort_influence_branch_matrix.csv",
        ),
    ]
    out_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    def classify(real_delta: float | None, eps: float, best_by_level: dict[str, float], hard_slice_delta: float | None) -> str:
        if real_delta is None or not math.isfinite(real_delta):
            return "NoFiniteRealDelta"
        if real_delta >= -eps:
            return "NoNLLBenefit"
        l1l2 = [best_by_level[k] for k in ["L1", "L2"] if k in best_by_level and math.isfinite(best_by_level[k])]
        if l1l2 and real_delta >= min(l1l2):
            return "DirectionNoSignal"
        l4l5 = [best_by_level[k] for k in ["L4", "L5"] if k in best_by_level and math.isfinite(best_by_level[k])]
        if l4l5 and real_delta >= min(l4l5):
            return "SubspaceOnly"
        l6 = best_by_level.get("L6", math.inf)
        if math.isfinite(l6) and real_delta >= l6:
            return "ActuatorSupportOnly"
        l7 = best_by_level.get("L7", math.inf)
        if math.isfinite(l7) and real_delta >= l7:
            return "ControlMatchedUnresolved_L7_optimizer_geometry"
        all_levels = [v for v in best_by_level.values() if math.isfinite(v)]
        if all_levels and real_delta < min(all_levels):
            if hard_slice_delta is not None and hard_slice_delta < -eps:
                return "RealCausal"
            return "ControlMatchedNoHardSlice"
        return "ControlMatchedUnresolved"

    for selector_name, horizon_label, selector_path, branch_path in selector_specs:
        selector_rows = read_rows(selector_path)
        branch_rows = read_rows(branch_path)
        if not selector_rows or not branch_rows:
            continue
        direct_count = 0
        nll_pass_count = 0
        beat_best_count = 0
        class_counts: dict[str, int] = {}
        for real in selector_rows:
            if str(real.get("status", "")).startswith("task_unavailable"):
                continue
            dataset = str(real.get("dataset", ""))
            seed = str(real.get("seed", ""))
            branch_h = str(real.get("branch_H", ""))
            controls = [
                row
                for row in branch_rows
                if str(row.get("dataset", "")) == dataset
                and str(row.get("seed", "")) == seed
                and str(row.get("branch_H", "")) == branch_h
                and int_flag(row.get("is_control_branch")) == 1
            ]
            if not controls:
                continue
            direct_count += 1
            real_delta = finite_float(real.get("NLL_delta_vs_base"))
            eps = finite_float(real.get("row_local_epsilon"), 0.0) or 0.0
            best_by_level: dict[str, float] = {}
            for ctrl in controls:
                level = str(ctrl.get("control_level", ""))
                ctrl_delta = finite_float(ctrl.get("NLL_delta_vs_base"))
                if ctrl_delta is not None:
                    best_by_level[level] = min(best_by_level.get(level, math.inf), ctrl_delta)
            best_any = min(best_by_level.values()) if best_by_level else math.inf
            hard_slice_delta = finite_float(real.get("hard_slice_NLL_delta"))
            nll_pass = int(real_delta is not None and real_delta < -eps)
            beat_best = int(real_delta is not None and real_delta < best_any)
            nll_pass_count += nll_pass
            beat_best_count += beat_best
            control_class = classify(real_delta, eps, best_by_level, hard_slice_delta)
            class_counts[control_class] = class_counts.get(control_class, 0) + 1
            for ctrl in controls:
                ctrl_delta = finite_float(ctrl.get("NLL_delta_vs_base"))
                real_minus_control = (
                    real_delta - ctrl_delta if real_delta is not None and ctrl_delta is not None else ""
                )
                out_rows.append(
                    {
                        "selector_name": selector_name,
                        "horizon_label": horizon_label,
                        "dataset": dataset,
                        "seed": seed,
                        "branch_H": branch_h,
                        "control_level": ctrl.get("control_level", ""),
                        "control_name": ctrl.get("selector_name", ""),
                        "real_delta_NLL": real_delta if real_delta is not None else "",
                        "control_delta_NLL": ctrl_delta if ctrl_delta is not None else "",
                        "real_minus_control_NLL": real_minus_control,
                        "row_local_epsilon": eps,
                        "real_NLL_delta_beyond_epsilon": nll_pass,
                        "real_beats_control": int(real_delta is not None and ctrl_delta is not None and real_delta < ctrl_delta),
                        "real_beats_best_control": beat_best,
                        "best_control_delta_NLL": best_any if math.isfinite(best_any) else "",
                        "sharpness_delta_real": "",
                        "sharpness_delta_control": "",
                        "margin_delta_real": real.get("margin_q10_delta_vs_base", ""),
                        "margin_delta_control": ctrl.get("margin_q10_delta_vs_base", ""),
                        "tail_q99_delta_real": real.get("tail_q99_delta", ""),
                        "tail_q99_delta_control": ctrl.get("tail_q99_delta", ""),
                        "hard_slice_NLL_delta_real": real.get("hard_slice_NLL_delta", ""),
                        "hard_slice_NLL_delta_control": ctrl.get("hard_slice_NLL_delta", ""),
                        "accuracy_delta_real": real.get("accuracy_delta_vs_base", ""),
                        "accuracy_delta_control": ctrl.get("accuracy_delta_vs_base", ""),
                        "horizon_dependent_win_label": "H400_and_H800_recorded" if horizon_label.startswith("H") else "",
                        "control_win_class": control_class,
                        "uses_test_direction_selection": real.get("uses_test_direction_selection", ""),
                        "uses_future_direction": real.get("uses_future_direction", ""),
                        "source_artifact": f"{selector_path.relative_to(ROOT)};{branch_path.relative_to(ROOT)}",
                    }
                )
        summary = {
            "selector_name": selector_name,
            "horizon_label": horizon_label,
            "direct_rows": direct_count,
            "control_comparison_rows": sum(1 for r in out_rows if r.get("selector_name") == selector_name and r.get("horizon_label") == horizon_label),
            "NLL_delta_beyond_row_epsilon_rows": nll_pass_count,
            "NLL_delta_beyond_row_epsilon_rate": rate(nll_pass_count, direct_count),
            "real_beats_best_control_rows": beat_best_count,
            "real_beats_best_control_rate": rate(beat_best_count, direct_count),
            "DirectionNoSignal_rows": class_counts.get("DirectionNoSignal", 0),
            "SubspaceOnly_rows": class_counts.get("SubspaceOnly", 0),
            "ActuatorSupportOnly_rows": class_counts.get("ActuatorSupportOnly", 0),
            "ControlMatchedUnresolved_L7_optimizer_geometry_rows": class_counts.get("ControlMatchedUnresolved_L7_optimizer_geometry", 0),
            "NoNLLBenefit_rows": class_counts.get("NoNLLBenefit", 0),
            "RealCausal_rows": class_counts.get("RealCausal", 0),
            "ControlMatchedNoHardSlice_rows": class_counts.get("ControlMatchedNoHardSlice", 0),
            "RealCausalCandidate_missing_L6_L7_hard_slice_rows": class_counts.get("RealCausalCandidate_missing_L6_L7_hard_slice", 0),
            "ControlMatchedUnresolved_rows": class_counts.get("ControlMatchedUnresolved", 0),
            "exploration_gate_pass": int(direct_count > 0 and rate(nll_pass_count, direct_count) >= 0.55 and rate(beat_best_count, direct_count) >= 0.50),
        }
        summary_rows.append(summary)

    write_rows(OUT_ROOT / "v22_35_T1_C10_C11_control_win_decomposition_matrix.csv", out_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C10_C11_control_win_decomposition_summary.csv", summary_rows)
    return {
        "control_win_rows": len(out_rows),
        "summary_rows": len(summary_rows),
        "any_exploration_gate_pass": int(any(int_flag(r.get("exploration_gate_pass")) for r in summary_rows)),
    }


def materialize_c12_influence_surrogate_diagnostic() -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    for selector_path in [
        OUT_ROOT / "v22_35_T1_C11_edge_safe_selector_matrix.csv",
        OUT_ROOT / "v22_35_T1_C10_repaired_selector_matrix.csv",
        OUT_ROOT / "v22_35_T1_C11_cohort_influence_selector_matrix.csv",
    ]:
        source_rows.extend(read_rows(selector_path))
    control_rows = read_rows(OUT_ROOT / "v22_35_T1_C10_C11_control_win_decomposition_summary.csv")
    control_matrix = read_rows(OUT_ROOT / "v22_35_T1_C10_C11_control_win_decomposition_matrix.csv")
    best_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in control_matrix:
        if int_flag(row.get("real_beats_best_control")) or row.get("control_level") == "L1":
            key = (str(row.get("selector_name", "")), str(row.get("horizon_label", "")), str(row.get("dataset", "")), str(row.get("seed", "")))
            if key not in best_by_key or int_flag(row.get("real_beats_best_control")):
                best_by_key[key] = row

    diagnostic_rows: list[dict[str, Any]] = []
    for row in source_rows:
        selector = str(row.get("selector_name", ""))
        horizon = "H800_current" if str(row.get("branch_H", "")) == "800" else f"H{row.get('branch_H', '')}"
        key = (selector, horizon, str(row.get("dataset", "")), str(row.get("seed", "")))
        decomp = best_by_key.get(key, {})
        if not decomp:
            # Fall back to H800_current for selector rows whose source matrix does not carry horizon_label.
            key = (selector, "H800_current", str(row.get("dataset", "")), str(row.get("seed", "")))
            decomp = best_by_key.get(key, {})
        diagnostic_rows.append(
            {
                "selector_name": selector,
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "branch_H": row.get("branch_H", ""),
                "held_CE_delta": row.get("held_CE_delta", ""),
                "held_margin_q10_delta": row.get("held_margin_q10_delta", ""),
                "symmetric_curvature_proxy": row.get("symmetric_curvature_proxy", ""),
                "max_abs_cos_to_selection_controls": row.get("max_abs_cos_to_selection_controls", ""),
                "NLL_delta_vs_base": row.get("NLL_delta_vs_base", ""),
                "row_local_epsilon": row.get("row_local_epsilon", ""),
                "NLL_delta_beyond_row_epsilon": decomp.get("real_NLL_delta_beyond_epsilon", ""),
                "real_beats_best_control": decomp.get("real_beats_best_control", ""),
                "control_win_class": decomp.get("control_win_class", ""),
                "diagnostic_only": 1,
                "not_official_reason": "C12 uses past branch labels/control outcomes; diagnostic only, not eligible for promotion.",
                "source_artifact": "results/v22_35/v22_35_T1_C11_edge_safe_selector_matrix.csv;results/v22_35/v22_35_T1_C10_repaired_selector_matrix.csv;results/v22_35/v22_35_T1_C11_cohort_influence_selector_matrix.csv;results/v22_35/v22_35_T1_C10_C11_control_win_decomposition_matrix.csv",
            }
        )

    def pearson(xs: list[float], ys: list[float]) -> float | str:
        n = len(xs)
        if n < 3:
            return ""
        mx = sum(xs) / n
        my = sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= 0.0 or vy <= 0.0:
            return ""
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            rank = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out

    feature_keys = [
        "held_CE_delta",
        "held_margin_q10_delta",
        "symmetric_curvature_proxy",
        "max_abs_cos_to_selection_controls",
    ]
    label_keys = ["NLL_delta_vs_base", "NLL_delta_beyond_row_epsilon", "real_beats_best_control"]
    summary_rows: list[dict[str, Any]] = []
    groups = sorted(set(str(r.get("selector_name", "")) for r in diagnostic_rows if r.get("selector_name")))
    groups.append("ALL_C10_C11")
    for group in groups:
        rows = diagnostic_rows if group == "ALL_C10_C11" else [r for r in diagnostic_rows if str(r.get("selector_name", "")) == group]
        for feature in feature_keys:
            for label in label_keys:
                xs: list[float] = []
                ys: list[float] = []
                for row in rows:
                    x = finite_float(row.get(feature))
                    y = finite_float(row.get(label))
                    if x is not None and y is not None:
                        xs.append(x)
                        ys.append(y)
                pr = pearson(xs, ys)
                sr = pearson(ranks(xs), ranks(ys)) if len(xs) >= 3 else ""
                summary_rows.append(
                    {
                        "diagnostic_name": "C12_influence_surrogate_feature_correlation",
                        "selector_group": group,
                        "feature": feature,
                        "label": label,
                        "n": len(xs),
                        "pearson": pr,
                        "spearman": sr,
                        "diagnostic_only": 1,
                        "official_candidate_gate_pass": 0,
                        "not_official_reason": "Uses past branch/control labels; allowed only to explain selector failure.",
                    }
                )

    write_rows(OUT_ROOT / "v22_35_T1_C12_influence_surrogate_diagnostic_matrix.csv", diagnostic_rows)
    write_rows(OUT_ROOT / "v22_35_T1_C12_influence_surrogate_diagnostic_summary.csv", summary_rows)
    return {
        "diagnostic_rows": len(diagnostic_rows),
        "summary_rows": len(summary_rows),
        "official_candidate_gate_pass": 0,
    }


def write_recap(final: dict[str, Any]) -> None:
    gap = read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv")
    noise = read_rows(OUT_ROOT / "v22_35_row_local_noise_matrix.csv")
    t1 = read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv")
    c11_summary = read_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_summary.csv")
    c11_h400_summary = read_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_summary_H400_seeds012.csv")
    c11_rows = read_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_selector_matrix.csv")
    c11_h400_rows = read_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_selector_matrix_H400_seeds012.csv")
    c11_branch_rows = read_rows(OUT_ROOT / "v22_35_T1_C11_edge_safe_branch_matrix.csv")
    c11_audit = read_rows(OUT_ROOT / "v22_35_T1_C11_candidate_audit_matrix.csv")
    c10_summary = read_rows(OUT_ROOT / "v22_35_T1_C10_repaired_summary.csv")
    c10_rows = read_rows(OUT_ROOT / "v22_35_T1_C10_repaired_selector_matrix.csv")
    c10_branch_rows = read_rows(OUT_ROOT / "v22_35_T1_C10_repaired_branch_matrix.csv")
    c10_audit = read_rows(OUT_ROOT / "v22_35_T1_C10_repaired_candidate_audit_matrix.csv")
    c11l_summary = read_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_summary.csv")
    c11l_rows = read_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_selector_matrix.csv")
    c11l_audit = read_rows(OUT_ROOT / "v22_35_T1_C11_cohort_influence_candidate_audit_matrix.csv")
    c_selector_control_summary = read_rows(OUT_ROOT / "v22_35_T1_C10_C11_control_win_decomposition_summary.csv")
    c_selector_control_matrix = read_rows(OUT_ROOT / "v22_35_T1_C10_C11_control_win_decomposition_matrix.csv")
    c12_diag_summary = read_rows(OUT_ROOT / "v22_35_T1_C12_influence_surrogate_diagnostic_summary.csv")
    c12_diag_matrix = read_rows(OUT_ROOT / "v22_35_T1_C12_influence_surrogate_diagnostic_matrix.csv")
    tier2_availability = read_rows(OUT_ROOT / "v22_35_Tier2_tabular_availability_matrix.csv")
    intrinsic_summary = read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv")
    intrinsic_repair_summary = read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv")
    intrinsic = read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_matrix.csv")
    intrinsic_repair = read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_branch_matrix.csv")
    basis_summary = read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv")
    basis_branch = read_rows(OUT_ROOT / "v22_35_basis_branch_transfer_matrix.csv")
    basis_repair_summary = read_rows(OUT_ROOT / "v22_35_basis_transfer_repair_summary.csv")
    basis_repair_history = read_rows(OUT_ROOT / "v22_35_basis_repair_attempt_history.csv")
    basis_repair_fit = read_rows(OUT_ROOT / "v22_35_basis_transfer_repair_fit_matrix.csv")
    basis_repair_branch = read_rows(OUT_ROOT / "v22_35_basis_transfer_repair_branch_matrix.csv")
    control = read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv")
    optimizer_direct_summary = read_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_summary.csv")
    optimizer_direct = read_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_matrix.csv")
    optimizer_attempt_history = read_rows(OUT_ROOT / "v22_35_optimizer_direct_probe_attempt_history.csv")
    optimizer_integrated = read_rows(OUT_ROOT / "v22_35_optimizer_integrated_fu_matrix.csv")
    optimizer_four_square_summary = read_rows(OUT_ROOT / "v22_35_optimizer_direct_four_square_summary.csv")
    optimizer_four_square = read_rows(OUT_ROOT / "v22_35_optimizer_direct_four_square_matrix.csv")
    optimizer_kan_internal_summary = read_rows(OUT_ROOT / "v22_35_optimizer_direct_kan_internal_summary.csv")
    continual = read_rows(OUT_ROOT / "v22_35_continual_matrix.csv")
    continual_nd_summary = read_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_summary.csv")
    continual_nd_baseline = read_rows(OUT_ROOT / "v22_35_continual_nondegenerate_baseline_matrix.csv")
    continual_nd_control = read_rows(OUT_ROOT / "v22_35_continual_nondegenerate_boundary_control_win_matrix.csv")
    continual_attempt_history = read_rows(OUT_ROOT / "v22_35_continual_attempt_history.csv")
    latest_intrinsic_repair = intrinsic_repair_summary[0] if intrinsic_repair_summary else {}
    latest_basis_repair = basis_repair_summary[0] if basis_repair_summary else {}
    latest_continual_nd = continual_nd_summary[0] if continual_nd_summary else {}
    latest_basis_input_rows = int_flag(latest_basis_repair.get("basis_repair_input_rows"))
    if latest_basis_input_rows:
        repair_pass_ratio = (
            f"{latest_basis_repair.get('basis_repair_strict_fit_branch_pass_rows', '0')}/"
            f"{latest_basis_repair.get('basis_repair_strict_branch_rows', '0')}"
        )
        low_rank_pass_ratio = (
            f"{latest_basis_repair.get('basis_repair_low_rank_strict_fit_branch_pass_rows', '0')}/"
            f"{latest_basis_repair.get('basis_repair_low_rank_strict_branch_rows', '0')}"
        )
        best_horizon_msg = (
            f"H{latest_basis_repair.get('basis_repair_best_horizon', 'n/a')} "
            f"{latest_basis_repair.get('basis_repair_best_horizon_strict_fit_branch_pass_rows', '0')}/"
            f"{latest_basis_repair.get('basis_repair_best_horizon_strict_branch_rows', '0')}"
        )
        basis_repair_analysis = (
            f"basis repair latest strict branch pass 为 `{repair_pass_ratio}`，best horizon 为 `{best_horizon_msg}`，low-rank sketch 子集为 `{low_rank_pass_ratio}`。"
            "若 exploration gate 仍为 0，blocker 收敛为 basis-native branch transfer/control uniqueness，而不是 clean unzip、epsilon、gradcheck 或单纯 scale 问题。"
        )
    else:
        repair_pass_ratio = "0/0"
        low_rank_pass_ratio = "0/0"
        best_horizon_msg = "not_run_no_nonzero_readout_repair_pass_rows"
        basis_repair_analysis = (
            "latest basis repair 未进入 basis 分支，因为当前 intrinsic repair 没有非零 branch pass row；"
            "blocker 在本次 D12/H400 审计中前移为 target intrinsic / short-horizon signal no-go。"
        )
    low_rank_ranks_msg = latest_basis_repair.get("basis_repair_low_rank_sketch_ranks", "n/a")
    intrinsic_repair_open = int_flag(latest_intrinsic_repair.get("repair_exploration_gate_pass"))
    insight_2 = (
        "- Insight 2：Level0 原始 readout-exact 失败；先前 sign/scale + H200 repair 可打开短 horizon 候选，但加入 H400 后当前 D12 curvature-safe target 的非零 branch pass 为 0，说明该方向更像 short-horizon/control-explained signal，不能进入 basis/full-loop promotion。"
        if not intrinsic_repair_open
        else "- Insight 2：Level0 原始 readout-exact 失败，但 sign/scale + horizon-aware direct noise 修复后 target intrinsic repair gate 打开；这把主 blocker 从 target intrinsic 推进到 basis-native transfer。"
    )
    insight_3 = (
        "- Insight 3：当前 D12/H400 审计没有 basis repair input rows；上一轮 basis residual/gradcheck 可 strict-fit 的证据仍保留在 attempt history，但不能覆盖本轮 target intrinsic no-go 结论。"
        if not latest_basis_input_rows
        else "- Insight 3：basis residual/gradcheck 可被修到 strict-fit，且 low-rank J_B sketch 会单独落盘审计；若 branch 仍常输给 same-basis random/signflip 或被 row-local noise/accuracy 阻断，证据链优先支持“basis transfer/control uniqueness blocker”，不是 review bundle/identity/epsilon/scale blocker。"
    )
    continual_nd_found = int_flag(latest_continual_nd.get("nondegenerate_baseline_rows"))
    continual_nd_gate = int_flag(latest_continual_nd.get("branch_gate_exploration_pass"))
    if not continual_nd_summary:
        continual_analysis = (
            "D8 readback 继续检查 non-degenerate baseline；本轮尚未运行 v22.35 直接 non-degenerate sweep。"
        )
    elif not continual_nd_found:
        continual_analysis = (
            "v22.35 直接 continual sweep 已运行，但没有找到满足预注册阈值的 non-degenerate baseline；"
            "因此 D8 boundary target 仍不能被判为机制失败，只能记录 regime blocker。"
        )
    elif not continual_nd_gate:
        continual_analysis = (
            "v22.35 直接 continual sweep 找到了 non-degenerate baseline，并按预注册顺序尝试 D8 boundary basis/native vs random/signflip controls；"
            f"latest target_mode=`{latest_continual_nd.get('target_mode', '')}`，"
            f"strict_fit_rows=`{latest_continual_nd.get('strict_fit_rows', '0')}`，"
            f"strict_fit_forgetting_gate_rows=`{latest_continual_nd.get('strict_fit_forgetting_gate_rows', '0')}`，"
            f"strict_fit_control_beat_rows=`{latest_continual_nd.get('strict_fit_control_beat_rows', '0')}`，"
            f"latest strict_fit_boundary_pass_rows=`{latest_continual_nd.get('strict_fit_boundary_pass_rows', '0')}`，"
            f"best_relative_forgetting_reduction=`{latest_continual_nd.get('best_relative_forgetting_reduction_vs_base', '')}`，"
            "但这些条件没有在同一 row 同时成立，exploration gate 仍未打开。"
        )
    else:
        continual_analysis = (
            "v22.35 直接 continual sweep 找到了 non-degenerate baseline，且 D8 boundary exploration gate 打开；"
            "这只能进入后续 full-loop/四格表复核，不能自动覆盖 Part B/D 的 blocker。"
        )
    c11_control_decomp: list[dict[str, Any]] = []
    for real in c11_rows:
        if str(real.get("status", "")).startswith("task_unavailable"):
            continue
        dataset = str(real.get("dataset", ""))
        seed = str(real.get("seed", ""))
        controls_for_row = [
            r
            for r in c11_branch_rows
            if str(r.get("dataset", "")) == dataset and str(r.get("seed", "")) == seed and int_flag(r.get("is_control_branch")) == 1
        ]
        real_delta = finite_float(real.get("NLL_delta_vs_base"))
        eps = finite_float(real.get("row_local_epsilon"), 0.0) or 0.0
        best_control = None
        best_delta = math.inf
        for ctrl in controls_for_row:
            ctrl_delta = finite_float(ctrl.get("NLL_delta_vs_base"))
            if ctrl_delta is not None and ctrl_delta < best_delta:
                best_delta = ctrl_delta
                best_control = ctrl
        real_beats_best = int(real_delta is not None and real_delta < best_delta)
        nll_pass = int(real_delta is not None and real_delta < -eps)
        if not nll_pass:
            block = "NoNLLBenefit"
        elif real_beats_best:
            block = "RealCausalCandidate_needs_replication"
        else:
            block = "SubspaceOrSupportControlWins"
        c11_control_decomp.append(
            {
                "dataset": dataset,
                "seed": seed,
                "real_NLL_delta": real_delta if real_delta is not None else "",
                "row_local_epsilon": eps,
                "best_control_level": best_control.get("control_level", "") if best_control else "",
                "best_control_name": best_control.get("selector_name", "") if best_control else "",
                "best_control_NLL_delta": best_delta if math.isfinite(best_delta) else "",
                "real_minus_best_control_NLL": (real_delta - best_delta) if real_delta is not None and math.isfinite(best_delta) else "",
                "NLL_delta_beyond_row_epsilon": nll_pass,
                "real_beats_best_control": real_beats_best,
                "control_uniqueness_class": block,
            }
        )
    c11_real_beats_best_count = sum(int_flag(r.get("real_beats_best_control")) for r in c11_control_decomp)
    c11_nll_pass_count = sum(int_flag(r.get("NLL_delta_beyond_row_epsilon")) for r in c11_control_decomp)
    c10_control_decomp: list[dict[str, Any]] = []
    for real in c10_rows:
        if str(real.get("status", "")).startswith("task_unavailable"):
            continue
        dataset = str(real.get("dataset", ""))
        seed = str(real.get("seed", ""))
        controls_for_row = [
            r
            for r in c10_branch_rows
            if str(r.get("dataset", "")) == dataset and str(r.get("seed", "")) == seed and int_flag(r.get("is_control_branch")) == 1
        ]
        real_delta = finite_float(real.get("NLL_delta_vs_base"))
        eps = finite_float(real.get("row_local_epsilon"), 0.0) or 0.0
        best_control = None
        best_delta = math.inf
        for ctrl in controls_for_row:
            ctrl_delta = finite_float(ctrl.get("NLL_delta_vs_base"))
            if ctrl_delta is not None and ctrl_delta < best_delta:
                best_delta = ctrl_delta
                best_control = ctrl
        real_beats_best = int(real_delta is not None and real_delta < best_delta)
        nll_pass = int(real_delta is not None and real_delta < -eps)
        if not nll_pass:
            block = "NoNLLBenefit"
        elif real_beats_best:
            block = "RealCausalCandidate_needs_replication"
        else:
            block = "SubspaceOrSupportControlWins"
        c10_control_decomp.append(
            {
                "dataset": dataset,
                "seed": seed,
                "real_NLL_delta": real_delta if real_delta is not None else "",
                "row_local_epsilon": eps,
                "best_control_level": best_control.get("control_level", "") if best_control else "",
                "best_control_name": best_control.get("selector_name", "") if best_control else "",
                "best_control_NLL_delta": best_delta if math.isfinite(best_delta) else "",
                "real_minus_best_control_NLL": (real_delta - best_delta) if real_delta is not None and math.isfinite(best_delta) else "",
                "NLL_delta_beyond_row_epsilon": nll_pass,
                "real_beats_best_control": real_beats_best,
                "control_uniqueness_class": block,
            }
        )
    c10_real_beats_best_count = sum(int_flag(r.get("real_beats_best_control")) for r in c10_control_decomp)
    c10_nll_pass_count = sum(int_flag(r.get("NLL_delta_beyond_row_epsilon")) for r in c10_control_decomp)
    c11_horizon_compare: list[dict[str, Any]] = []
    for label, summary_rows, selector_rows_for_h in [
        ("H400_archive_seeds012", c11_h400_summary, c11_h400_rows),
        ("current", c11_summary, c11_rows),
    ]:
        if not summary_rows:
            continue
        row = summary_rows[0]
        horizon = ""
        if selector_rows_for_h:
            horizon = selector_rows_for_h[0].get("branch_H", "")
        c11_horizon_compare.append(
            {
                "horizon_label": label,
                "branch_H": horizon,
                "direct_C11_rows": row.get("direct_C11_rows", ""),
                "NLL_delta_beyond_row_epsilon_rate": row.get("NLL_delta_beyond_row_epsilon_rate", ""),
                "beats_L3_rate": row.get("beats_L3_rate", ""),
                "beats_L4_rate": row.get("beats_L4_rate", ""),
                "beats_L5_rate": row.get("beats_L5_rate", ""),
                "accuracy_safe_rate": row.get("accuracy_safe_rate", ""),
                "exploration_gate_pass": row.get("exploration_gate_pass", ""),
                "official_candidate_gate_pass": row.get("official_candidate_gate_pass", ""),
            }
        )
    lines = [
        "# DG-KAN v22.35 Causal Target Noise-Calibrated FU 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## Final Route",
        "",
        f"- final_route: `{final.get('final_route')}`",
        f"- official_full_superiority_ready: `{final.get('official_full_superiority_ready')}`",
        f"- no_fabricated_rows_claim: {final.get('no_fabricated_rows_claim')}",
        "",
        "## 本轮修改/修复审计",
        "",
        "- 新增 `experiments/run_v22_35_causal_target_noise_calibrated_fu.py`：v22.35 专用执行 harness，新增 row-local no-op repeat、gap recalibration、C9/C10 row-local selector audit、Level0 readout-exact target probe、basis transfer re-audit、control-win/curvature/optimizer/hard-task gate matrices、figures、final route 和两份日志写入。",
        "- 修复 Part A 口径：`clean_unzip_compileall_pass` / `clean_unzip_import_pass` 现在来自 `v22_35_review_bundle_current.tar.gz` 的真实解包目录 `results/v22_35/clean_unzip_check`，并记录 bundle sha256；不再把当前工作树 compile/import 冒充 clean unzip。",
        "- 修复方向遵循计划：global epsilon 不再直接作为唯一 gate；本轮新增同 checkpoint no-op repeat，并对历史行明确标注 control-bootstrap/fallback epsilon_source。",
        "- 二次复核修复：新增 `row_epsilon_for(dataset, seed, carrier, horizon)` 映射，Level0 intrinsic probe 现在优先使用同 dataset/seed/carrier 的 direct no-op repeat epsilon；首轮复核发现 Wine intrinsic 行曾 fallback 到 `epsilon_float`，修复后重跑，D-CHE/D-FOU Wine epsilon 分别来自 direct repeat，final route 未改变。",
        "- hidden-aware epsilon 修复：row-local no-op repeat 行新增 `model_hidden`，repair 阶段的 direct epsilon lookup 按当前 hidden 过滤；bank expansion 到 hidden=32 时必须重新生成同 hidden 的 no-op repeat，不允许复用 hidden=16 direct noise。",
        "- 三次复核修复：H0 即时诊断不再套用非零 branch no-op epsilon；H20/H60/H200/H400 repair 行优先使用同 horizon direct no-op repeat，且 no-op repeat 固定 eval set，避免把 test 子集变化混进 epsilon。",
        "- 修复方向遵循计划：v22.34 缺少 Level0 readout-exact intrinsic benefit 的目标，本轮直接运行 Level0 probe；未覆盖的历史 basis 行不冒充 readout-exact 证据。",
        "- C11 target-uniqueness 修复：新增 `--stage c11`，真实执行 C11 edge-safe control-orthogonal selector；候选选择只使用 held-train CE、margin、对称曲率 proxy 和 control-cosine，不用 test/future/query direction。",
        "- C10 margin 修复：在同一 C11 信号池与 controls 下追加 `C10_repaired_margin_NLL_constrained_direction`；选择规则为 held-train margin q10 最大化，同时约束 held CE 不上升和 margin debt，不使用 test/future/query direction。",
        "- Part L cohort-influence 尝试：新增 `C11_cohort_influence_repair_direction`，在同一候选池内按 held-train cohort CE non-increase stability、CE mean/std、margin 与曲率选方向；不使用 test/future，不覆盖 C11/C10 原结果。",
        "- C10/C11 Part E 缺口修复：补齐 L6 same candidate/basis bank random、L7 train-gradient optimizer-geometry control，并记录 reference hard-slice NLL delta；`RealCausal` 现在必须 beat L1-L7 且 hard-slice NLL 超过 row-local epsilon 改善。",
        "- Tier2 数据可用性修复：C11 stage 新增 Wine/Spam/Rice/Bean/Telescope 表格 loader 与 availability audit；Spam/Rice/Bean/Telescope 按 UCI 官方 id/URL、direct UCI、`curl --http1.1` fallback、`ucimlrepo`、OpenML 尝试链加载，失败时记录真实错误而不补造行。",
        "- R4 Tier2 change-task 修复：Part D repair row-local noise、intrinsic sign/scale repair 与 basis transfer repair 统一改用 v22.35 C11/Tier2 loader，并在落盘行记录 `task_tier`/`source_kind`；目的是验证 ActuatorSupportOnly 是否由任务族造成，不改变 gate 标准、不补造 unavailable 数据。",
        "- C11/Basis re-audit 统计修复：修正 `0.0` 被 Python truthiness 当作缺失的问题；C11 `accuracy_safe_rate` 与 historical basis control beat/accuracy 比较改为显式 `None` 判断，避免把 0 delta 误判为 unsafe 或把 0 control delta 误判为 `inf`。",
        "- 追加 blocker 修复：新增 intrinsic sign/scale/H0-Hbranch repair，固定 sign `{+1,-1}` 与 scale multiplier `{0.25,0.5,1,2,4}`，候选选择只使用 held-train CE/sharpness，test/branch 只用于审计；用于排除符号、trust region、branch horizon 吞掉即时收益等实现 blocker。",
        "- 追加 R4 blocker 修复：新增 basis transfer repair，只对 readout repair 的非零 branch pass 目标重建同 checkpoint basis actuator；bank 固定为 all/low/mixed `{4,8,16}`，branch 固定 H200/H400/H800，并配 same-basis random 与 signflip controls。",
        "- 数值检查修复：basis repair 使用 central finite-difference eps sweep `{1e-5,3e-5,1e-4,3e-4,1e-3}`，记录 `J_B_gradcheck_rel_error_trace`，仍要求最小 rel error <= 1e-3；是否打开 branch gate 只看落盘 summary，不靠手写数字。",
        "- target-uniqueness 修复：basis repair 在 basis 侧也用 held-only `{+1,-1} × scale` 选择方向，不使用 test/branch 选 sign；若 signflip 仍赢，说明不是单纯 basis 符号继承错误。",
        "- basis repair gate 修复：按 Part D Level0→Level1 ladder，若非零 H branch intrinsic pass 为 0 但 H0/readout-exact pass 存在，则允许以 H0 pass row 作为 `H0_readout_exact_pass_level0_anchor` 输入 basis actuator transfer；该路径只测试 actuator transfer，不把 H0 微增益 promotion。",
        "- basis branch sign 修复：新增 held-only branch-aware sign check。仅当 basis projection residual/cosine 已接近 strict fit 时，在 held split 上对 selected/signflip 做短 branch 对比选 sign，再用 test branch 审计；不使用 test 选 sign。",
        "- ShortHorizonNoise 修复：basis repair 增加 H800，并把 summary 改成记录 `basis_repair_horizon_rate_trace` 与 best-horizon gate，避免多 horizon 混合分母稀释或夸大结论。",
        f"- low-rank J_B sketch 修复：basis repair 对 `all_basis` 追加固定 rank SVD sketch `{low_rank_ranks_msg}`，从 `J_B J_B^T` 的 Gram 分解构造右奇异子空间，不使用 test/branch 选 rank；CSV 记录 `basis_low_rank_sketch_rank`、能量占比、solve cols 与 branch/control 结果。",
        "- Part F curvature-safe target 修复：新增 `D12_curvature_safe_label_smooth_target`，用 train hard-slice、held class CVaR 权重、label smoothing 与 temperature-soft target 形成更保守的 logit target；只在 held-train 上做 CE/sharpness sign-scale 选择，test/branch 只用于审计。",
        "- 新增 `--stage curvature_basis`：针对 curvature-safe target 先确认 repair/basis horizons 的 row-local no-op epsilon 覆盖，再运行 intrinsic sign/scale repair、basis transfer repair、control/curvature/final recap，便于复现 Part E/F blocker 修复。",
        f"- 负结果修复记录：basis scale gate 加入 held CE `1e-5` 与 sharpness `1e-6` 数值容忍后，仍按真实 summary 判断；latest strict branch pass 为 `{repair_pass_ratio}`，best horizon 为 `{best_horizon_msg}`，low-rank 子集为 `{low_rank_pass_ratio}`。",
        "- Continual/D8 blocker 修复：新增 `--stage continual`，直接运行 Class-MNIST task0->task1 non-degenerate baseline 固定网格；若找到 old-task retention 不为 0 且 task1 学到的中间 regime，再按预注册字典序尝试 previous-task CE boundary basis target、same-basis random 与 signflip controls。",
        "- Part G 强优化器修复：新增 `--stage optimizer` 与 `stage_g_optimizer_direct_probe`，直接运行 AdamW/Cautious AdamW/Schedule-Free AdamW/Muon-like 的 optimizer-alone、norm-matched slow-signal FU、random/signflip/shuffled matched controls；每个 variant 重建同 seed loader 与同初始化，防止 DataLoader 状态漂移污染对比。",
        "- Part G MLPDegradation blocker 修复：新增 v22.35-owned `train_loss_beats_base_step` optimizer acceptance；每一步 signal/control candidate 必须在当前 train batch CE 上不输同一步 base optimizer update 才保留，并记录 accept rate、reject count 与 candidate-minus-base-step loss delta。不使用 test/validation/future direction 选 gate。",
        "- Part G held-train acceptance 修复：新增 `held_train_loss_nonworse` / `held_train_loss_beats_base_step`，用 C11 loader 的 disjoint held-train split 评估 candidate 是否不输 base step；仍只用训练数据切分，不使用 test/validation/future direction。",
        "- Part G artifact 审计修复：optimizer direct tagged artifact 名字加入 `optimizer_signal_alpha`，避免同一 acceptance/tol 下不同 alpha 的 row-level CSV 互相覆盖；发现覆盖后重跑 alpha=0.20 held-train acceptance 恢复行级证据。",
        "- 修复方向遵循计划：若 gap/target/branch gate 未开，G/H full-loop、optimizer official、hard-task superiority 只写 gate_blocked/historical_readback，不做 promotion。",
        "",
        "## Part A Code Truth",
        "",
        md_table([final.get("code_truth", {})], limit=2),
        "",
        "## Part B Row-Local Noise And Gap Truth",
        "",
        "Row-local noise 摘要：",
        "",
        md_table(noise, ["dataset", "seed", "architecture", "carrier", "repeat_count", "delta_NLL_std", "epsilon_row", "epsilon_source"], limit=12),
        "",
        "Gap truth recalibration 摘要：",
        "",
        md_table(gap, limit=4),
        "",
        "分析：若 `TrueKANGain+BothGain` 仍低于 exploration gate，full superiority 继续被阻断；小增益即使超过旧 global epsilon 的怀疑点，也必须同时过 row-local epsilon 与 controls beat。",
        "",
        "## Part C Selector Retest",
        "",
        md_table(t1, limit=4),
        "",
        "C11 edge-safe control-orthogonal repair：",
        "",
        md_table(c11_summary, limit=4),
        "",
        "C11 H400/H800 horizon check：",
        "",
        md_table(c11_horizon_compare, ["horizon_label", "branch_H", "direct_C11_rows", "NLL_delta_beyond_row_epsilon_rate", "beats_L3_rate", "beats_L4_rate", "beats_L5_rate", "beats_L6_rate", "beats_L7_rate", "accuracy_safe_rate", "exploration_gate_pass", "official_candidate_gate_pass"], limit=4),
        "",
        "C10 repaired margin/NLL-constrained selector：",
        "",
        md_table(c10_summary, limit=4),
        "",
        md_table(c10_rows, ["dataset", "seed", "selector_name", "branch_H", "NLL_delta_vs_base", "row_local_epsilon", "beats_L3", "beats_L4", "beats_L5", "beats_L6", "beats_L7", "hard_slice_NLL_delta", "accuracy_delta_vs_base", "held_CE_delta", "held_margin_q10_delta", "symmetric_curvature_proxy", "selection_status"], limit=12),
        "",
        "C11 cohort-influence repair：",
        "",
        md_table(c11l_summary, limit=4),
        "",
        md_table(c11l_rows, ["dataset", "seed", "selector_name", "branch_H", "NLL_delta_vs_base", "row_local_epsilon", "beats_L3", "beats_L4", "beats_L5", "beats_L6", "beats_L7", "hard_slice_NLL_delta", "accuracy_delta_vs_base", "held_CE_delta", "held_cohort_CE_delta_mean", "held_cohort_CE_delta_std", "held_cohort_CE_nonincrease_rate", "held_margin_q10_delta", "selection_status"], limit=12),
        "",
        "C10 repaired control uniqueness 分解：",
        "",
        md_table(c10_control_decomp, ["dataset", "seed", "real_NLL_delta", "row_local_epsilon", "best_control_level", "best_control_name", "best_control_NLL_delta", "real_minus_best_control_NLL", "NLL_delta_beyond_row_epsilon", "real_beats_best_control", "control_uniqueness_class"], limit=12),
        "",
        f"证据链：C10 repaired 在 `{len(c10_control_decomp)}` 个可用 direct row 中有 `{c10_nll_pass_count}` 个超过 row-local epsilon，`{c10_real_beats_best_count}` 个 beat best matched control；该结果只能作为预注册 C10 修复复核，不能作为新 action search。",
        "",
        "Tier2 tabular 数据源/可用性审计：",
        "",
        md_table(tier2_availability, ["dataset", "seed", "status", "task_tier", "source_kind", "official_url", "uci_id", "n_rows", "n_features", "n_classes", "effective_train_rows", "effective_held_rows", "effective_test_rows", "error_type", "error_message", "availability_attempts"], limit=12),
        "",
        md_table(c11_rows, ["dataset", "seed", "selector_name", "branch_H", "NLL_delta_vs_base", "row_local_epsilon", "beats_L3", "beats_L4", "beats_L5", "beats_L6", "beats_L7", "hard_slice_NLL_delta", "accuracy_delta_vs_base", "held_CE_delta", "held_margin_q10_delta", "symmetric_curvature_proxy", "selection_status"], limit=12),
        "",
        "C11 control uniqueness 分解：",
        "",
        md_table(c11_control_decomp, ["dataset", "seed", "real_NLL_delta", "row_local_epsilon", "best_control_level", "best_control_name", "best_control_NLL_delta", "real_minus_best_control_NLL", "NLL_delta_beyond_row_epsilon", "real_beats_best_control", "control_uniqueness_class"], limit=12),
        "",
        f"证据链：C11 在 `{len(c11_control_decomp)}` 个可用 direct row 中有 `{c11_nll_pass_count}` 个超过 row-local epsilon，但只有 `{c11_real_beats_best_count}` 个 beat best matched control；因此当前 blocker 是 control uniqueness / direction selection，而不是 accuracy debt。",
        "",
        "C10/C11 Part E control-win 分类矩阵摘要：",
        "",
        md_table(c_selector_control_summary, ["selector_name", "horizon_label", "direct_rows", "control_comparison_rows", "NLL_delta_beyond_row_epsilon_rate", "real_beats_best_control_rate", "DirectionNoSignal_rows", "SubspaceOnly_rows", "ActuatorSupportOnly_rows", "ControlMatchedUnresolved_L7_optimizer_geometry_rows", "NoNLLBenefit_rows", "ControlMatchedNoHardSlice_rows", "RealCausal_rows", "exploration_gate_pass"], limit=8),
        "",
        md_table(c_selector_control_matrix, ["selector_name", "horizon_label", "dataset", "seed", "branch_H", "control_level", "real_delta_NLL", "control_delta_NLL", "real_minus_control_NLL", "real_beats_control", "real_beats_best_control", "hard_slice_NLL_delta_real", "hard_slice_NLL_delta_control", "margin_delta_real", "margin_delta_control", "control_win_class"], limit=24),
        "",
        md_table(c11_audit, ["dataset", "seed", "candidate_label", "candidate_source", "held_CE_delta", "held_margin_q10_delta", "symmetric_curvature_proxy", "max_abs_cos_to_selection_controls"], limit=12),
        "",
        md_table(c10_audit, ["dataset", "seed", "candidate_label", "candidate_source", "held_CE_delta", "held_margin_q10_delta", "symmetric_curvature_proxy", "max_abs_cos_to_selection_controls"], limit=12),
        "",
        md_table(c11l_audit, ["dataset", "seed", "candidate_label", "candidate_source", "held_CE_delta", "held_cohort_CE_delta_mean", "held_cohort_CE_delta_std", "held_cohort_CE_nonincrease_rate", "held_margin_q10_delta", "symmetric_curvature_proxy", "max_abs_cos_to_selection_controls"], limit=12),
        "",
        "C12 influence-surrogate diagnostic（非 official）：",
        "",
        md_table(c12_diag_summary, ["selector_group", "feature", "label", "n", "pearson", "spearman", "diagnostic_only", "official_candidate_gate_pass"], limit=24),
        "",
        md_table(c12_diag_matrix, ["selector_name", "dataset", "seed", "branch_H", "held_CE_delta", "held_margin_q10_delta", "symmetric_curvature_proxy", "max_abs_cos_to_selection_controls", "NLL_delta_vs_base", "NLL_delta_beyond_row_epsilon", "real_beats_best_control", "control_win_class", "diagnostic_only"], limit=20),
        "",
        "分析：C9/C10 只在 row-local/control-bootstrap epsilon 下重新判定，不新增 C13/C14 action search；C10 repaired/C11 若未达 exploration/official gate，也只能作为 target-uniqueness 负结果，不允许 promotion。L6/L7 与 hard-slice 补齐后，只有同时 beat all controls 且 hard-slice 改善的行才可记作 `RealCausal`。C12 只作为 influence-surrogate diagnostic，因使用 past branch/control labels，不能进入 official runtime。",
        "",
        "## Part D Target Intrinsic And Basis Transfer",
        "",
        "Level0 readout-exact intrinsic benefit 摘要：",
        "",
        md_table(intrinsic_summary, limit=4),
        "",
        md_table(intrinsic, ["target_family", "carrier", "dataset", "seed", "readout_exact_NLL_delta", "same_readout_random_NLL_delta", "readout_signflip_control_NLL_delta", "row_local_epsilon", "readout_intrinsic_gate_pass", "held_CE_delta", "SAM_sharpness_delta", "curvature_safe_label_smoothing", "curvature_safe_temperature"], limit=20),
        "",
        "Intrinsic sign/scale/H0-Hbranch repair 摘要：",
        "",
        md_table(intrinsic_repair_summary, limit=4),
        "",
        md_table(intrinsic_repair, ["target_family", "carrier", "dataset", "seed", "branch_H", "selected_sign", "selected_scale_multiplier", "real_NLL_delta", "same_readout_random_NLL_delta", "readout_signflip_control_NLL_delta", "row_local_epsilon", "repair_intrinsic_gate_pass", "selection_status", "curvature_safe_label_smoothing", "curvature_safe_temperature"], limit=24),
        "",
        "修复分析：该 repair 不使用 test 选候选，目的是检查 target intrinsic failure 是否来自符号、trust region 或短 horizon。若 repair 仍无 gate pass，则 target family 本身更可能不 causal；若只 H0 打开而 Hbranch 失败，则说明 branch training/control/noise 会吞掉即时读出收益，不能进入 basis/full-loop promotion。",
        "",
        "Basis branch transfer 摘要：",
        "",
        md_table(basis_summary, limit=4),
        "",
        md_table(basis_branch, ["target_family", "carrier", "dataset", "seed", "basis_bank", "branch_H", "strict_fit_pass", "row_local_epsilon", "NLL_delta_H400", "beats_same_basis_random", "beats_signflip_basis_control", "strict_fit_branch_pass"], limit=20),
        "",
        "Basis transfer repair 细化证据：",
        "",
        md_table(basis_repair_summary, limit=4),
        "",
        "Basis repair attempt history：",
        "",
        md_table(basis_repair_history, ["timestamp", "repair_datasets", "basis_repair_input_source", "basis_repair_low_rank_sketch_ranks", "basis_repair_fit_rows", "basis_repair_strict_fit_rows", "basis_repair_low_rank_fit_rows", "basis_repair_low_rank_strict_branch_rows", "basis_repair_low_rank_strict_fit_branch_pass_rows", "basis_repair_strict_fit_branch_pass_rows", "basis_repair_best_horizon", "basis_repair_branch_gate_exploration_pass"], limit=12),
        "",
        md_table(basis_repair_fit, ["target_family", "carrier", "dataset", "seed", "basis_repair_input_source", "readout_repair_source_branch_H", "basis_bank", "basis_low_rank_sketch_rank", "basis_solve_cols", "basis_selected_sign", "branch_aware_sign_status", "branch_aware_sign_horizon", "branch_aware_held_delta_selected", "branch_aware_held_delta_signflip", "strict_fit_pass", "basis_projection_residual", "basis_actuator_cosine", "exact_vs_linearized_error", "J_B_gradcheck_rel_error", "basis_gradcheck_eps", "applied_update_scale", "scale_selection_status"], limit=16),
        "",
        md_table(basis_repair_branch, ["target_family", "carrier", "dataset", "seed", "basis_repair_input_source", "readout_repair_source_branch_H", "basis_bank", "basis_low_rank_sketch_rank", "branch_H", "basis_selected_sign", "strict_fit_pass", "real_NLL_delta", "same_basis_random_NLL_delta", "signflip_basis_control_NLL_delta", "row_local_epsilon", "accuracy_delta", "strict_fit_branch_pass"], limit=20),
        "",
        f"分析：readout-exact、basis fit、basis branch 被分开记录。Intrinsic repair gate 是否打开见上方 summary；{basis_repair_analysis}",
        "",
        "## Part E/F Control And Curvature",
        "",
        md_table(control, limit=4),
        "",
        "分析：control-win row 只解释 blocker。若 safe-scale 行仍输给 same-basis random/signflip 或 sharpness-safe 只把影响压小到 epsilon 内，结论是 control/flatness/support 解释仍强，不是 DG-KAN success。",
        "",
        "## Part G/H Hard Tasks, Continual, Grokking",
        "",
        "v22.35 direct optimizer strong-family probe summary：",
        "",
        md_table(optimizer_direct_summary, limit=4),
        "",
        "v22.35 optimizer attempt history：",
        "",
        md_table(optimizer_attempt_history, ["timestamp", "optimizer_datasets", "optimizer_seeds", "model_families", "optimizer_variants", "optimizer_steps", "optimizer_signal_alpha", "optimizer_acceptance", "optimizer_acceptance_tol", "signal_accept_rate_mean", "candidate_accept_rate_mean", "signal_reject_count_total", "control_reject_count_total", "signal_fu_rows", "beats_own_optimizer_baseline_rows", "beats_same_optimizer_controls_rows", "beats_strongest_completed_optimizer_baseline_rows", "exploration_gate_pass", "status"], limit=8),
        "",
        "v22.35 optimizer direct rows（signal/control/full matrix）：",
        "",
        md_table(optimizer_direct, ["dataset", "seed", "architecture", "carrier", "optimizer_name", "training_variant", "NLL", "base_optimizer_NLL", "NLL_delta_vs_optimizer", "best_same_optimizer_control_NLL", "beats_optimizer_baseline", "beats_same_optimizer_controls", "beats_strongest_completed_optimizer_baseline", "controller_overhead_ratio", "direct_optimizer_acceptance", "direct_optimizer_accept_rate", "direct_optimizer_reject_count", "direct_optimizer_candidate_loss_delta_mean", "status"], limit=24),
        "",
        "v22.35 optimizer integrated matrix readback：",
        "",
        md_table(optimizer_integrated, ["dataset", "seed", "architecture", "carrier", "optimizer_name", "training_variant", "FU_target_family", "NLL_delta_vs_optimizer", "beats_optimizer_baseline", "beats_same_optimizer_controls", "beats_strongest_completed_optimizer_baseline", "controller_overhead_ratio", "status"], limit=16),
        "",
        "v22.35 optimizer direct four-square summary：",
        "",
        md_table(optimizer_four_square_summary, limit=4),
        "",
        "v22.35 optimizer direct KAN-internal downgrade summary：",
        "",
        md_table(optimizer_kan_internal_summary, limit=4),
        "",
        md_table(optimizer_four_square, ["dataset", "seed", "optimizer_name", "kan_carrier", "Delta_MLP_NLL", "Delta_KAN_NLL", "Gap_BP", "Gap_FU", "GapReduction", "MLP_FU_nonworse_vs_MLP_strong", "KAN_FU_improves_KAN_strong", "KAN_FU_beats_same_optimizer_controls", "TrueKANGain_class"], limit=24),
        "",
        "分析：Part G direct probe 只检验 FU 是否独立于同 optimizer 与 matched controls；four-square rows 用同一 optimizer family 做 MLP/KAN 对照，避免 best-row promotion。在 Part B/D branch/full-loop gate 关闭时，即使出现 optimizer 局部胜出也只能作为 interaction evidence，不允许提升为 official superiority。",
        "",
        md_table(continual, ["dataset", "seed", "base_forgetting", "real_forgetting", "relative_forgetting_reduction", "baseline_old_task_accuracy_after_new_task_nondegenerate", "status"], limit=12),
        "",
        "v22.35 continual non-degenerate baseline summary：",
        "",
        md_table(continual_nd_summary, limit=4),
        "",
        "v22.35 continual attempt history：",
        "",
        md_table(continual_attempt_history, ["timestamp", "target_mode", "target_hard_fraction", "bank_dims", "damping", "update_scale", "sign_select", "basis_examples", "nondegenerate_baseline_rows", "selected_repair_regimes", "strict_fit_rows", "strict_fit_forgetting_gate_rows", "strict_fit_control_beat_rows", "strict_fit_boundary_pass_rows", "branch_gate_exploration_pass", "best_relative_forgetting_reduction_vs_base", "repair_status"], limit=12),
        "",
        md_table(continual_nd_baseline, ["carrier", "basis_family", "seed", "task0_steps", "task1_steps", "task0_accuracy_before_task1", "base_task0_accuracy_after_task1", "base_task1_accuracy_after_task1", "base_forgetting", "baseline_old_task_accuracy_after_new_task_nondegenerate", "status"], limit=24),
        "",
        "v22.35 D8 boundary control-win rows：",
        "",
        md_table(continual_nd_control, ["target_mode", "carrier", "basis_family", "seed", "basis_bank", "basis_selected_sign", "sign_selection_status", "task0_steps", "task1_steps", "fit_strict_pass", "base_forgetting", "real_forgetting", "same_basis_random_forgetting", "signflip_basis_control_forgetting", "relative_forgetting_reduction_vs_base", "forgetting_gate_pass", "beats_same_basis_random", "beats_signflip_basis_control", "task1_accuracy_delta_vs_base_after_task1", "strict_fit_boundary_pass", "control_win_explanation"], limit=24),
        "",
        f"分析：{continual_analysis}",
        "",
        "## 关键 Insight",
        "",
        "- Insight 1：v22.35 把 v22.34 的 global epsilon blocker 拆成 direct row-local repeat 与历史 bootstrap 两类来源；这让 micro-gain 可以被审计，但不降低 controls beat 要求。",
        insight_2,
        insight_3,
        "- Insight 4：所有 official superiority 相关矩阵都受 gate 控制；未跑 full-loop 是计划内阻断，不是缺失数据被补造。",
        f"- Insight 5：continual/D8 的 blocker 已从 v22.34 的全退化 baseline 前移：本轮找到 non-degenerate baseline `{latest_continual_nd.get('nondegenerate_baseline_rows', 'n/a')}` 行，但 latest D8 `{latest_continual_nd.get('target_mode', 'n/a')}` 只有 strict_fit_boundary_pass_rows=`{latest_continual_nd.get('strict_fit_boundary_pass_rows', 'n/a')}`；可 fit/可减 forgetting/controls fail 仍不能在同一 row 重合。",
        f"- Insight 6：Part G direct optimizer probe latest acceptance=`{(optimizer_direct_summary[0].get('optimizer_acceptance') if optimizer_direct_summary else 'n/a')}`，signal rows=`{(optimizer_direct_summary[0].get('signal_fu_rows') if optimizer_direct_summary else 'n/a')}`，beats own/control/strongest=`{(optimizer_direct_summary[0].get('beats_own_optimizer_baseline_rows') if optimizer_direct_summary else 'n/a')}`/`{(optimizer_direct_summary[0].get('beats_same_optimizer_controls_rows') if optimizer_direct_summary else 'n/a')}`/`{(optimizer_direct_summary[0].get('beats_strongest_completed_optimizer_baseline_rows') if optimizer_direct_summary else 'n/a')}`，signal_accept_rate_mean=`{(optimizer_direct_summary[0].get('signal_accept_rate_mean') if optimizer_direct_summary else 'n/a')}`；four-square TrueKANGain+BothGain rate=`{(optimizer_four_square_summary[0].get('TrueKANGain_plus_BothGain_rate') if optimizer_four_square_summary else 'n/a')}`，hard_task_exploration_gate_pass=`{(optimizer_four_square_summary[0].get('hard_task_exploration_gate_pass') if optimizer_four_square_summary else 'n/a')}`，KAN-internal route_hint=`{(optimizer_kan_internal_summary[0].get('route_hint') if optimizer_kan_internal_summary else 'n/a')}`。结论是 optimizer interaction 可作为 KAN-internal diagnostic；若 acceptance 后 MLP non-worse 仍未达标，则 blocker 不是单步训练 CE 退化这一项能单独修掉。",
        "",
        "## 复现实用索引",
        "",
        "- runner: `experiments/run_v22_35_causal_target_noise_calibrated_fu.py`",
        "- command journal: `results/v22_35/v22_35_command_journal.csv`",
        "- final route: `results/v22_35/v22_35_final_route.json`",
        "- artifact index: `results/v22_35/v22_35_artifact_index.csv`",
        "- stdout/stderr logs: `results/v22_35/logs/`",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_start",
        status="started",
        gpu="0,1,2,3",
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py",
        note="Start v22.35 gated execution. GPUs 0/1/2/3 available; this compact run uses configured devices and writes all required artifacts or explicit gate_blocked rows.",
    )
    code = stage_a()
    if not int_flag(code.get("clean_unzip_compileall_pass")) or not int_flag(code.get("clean_unzip_import_pass")) or not int_flag(code.get("official_DGKAN_identity_pass")):
        write_gate_blocked_after_code_fail(code)
        write_figures()
        _idx, manifest = artifact_index()
        code["artifact_manifest_hash"] = manifest
        write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
        final = decide_final(code, {}, {}, {}, {}, {})
        write_recap(final)
        return final
    run_row_local_noise_repeats(args)
    run_repair_row_local_noise_repeats(args)
    gap = stage_b_gap_recalibration()
    t1 = stage_c_selector_retest()
    intrinsic = stage_d_readout_intrinsic(args)
    intrinsic_repair = stage_d_intrinsic_sign_scale_repair(args)
    intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
    basis = stage_d_basis_transfer_reaudit()
    basis_repair = stage_d_basis_transfer_repair(args)
    basis.update({f"repair_{k}" if k in basis else k: v for k, v in basis_repair.items()})
    control = stage_e_control_decomposition()
    stage_f_curvature_representation()
    stage_g_h_gate_matrices()
    write_figures()
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route, recap, figures, artifact index",
        task_id="Z_finalize_v22_35",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}",
    )
    return final


def write_gate_blocked_after_code_fail(code: dict[str, Any]) -> None:
    required = [
        "v22_35_gap_truth_recalibrated_matrix.csv",
        "v22_35_row_local_noise_matrix.csv",
        "v22_35_T1_selector_branch_matrix.csv",
        "v22_35_T1_control_hierarchy_matrix.csv",
        "v22_35_target_intrinsic_benefit_matrix.csv",
        "v22_35_target_intrinsic_sign_scale_repair_matrix.csv",
        "v22_35_target_intrinsic_repair_branch_matrix.csv",
        "v22_35_basis_actuator_transfer_matrix.csv",
        "v22_35_basis_branch_transfer_matrix.csv",
        "v22_35_control_win_decomposition_matrix.csv",
        "v22_35_curvature_representation_matrix.csv",
        "v22_35_optimizer_integrated_fu_matrix.csv",
        "v22_35_hard_task_four_square_matrix.csv",
        "v22_35_continual_matrix.csv",
        "v22_35_grokking_matrix.csv",
        "v22_35_efficiency_matrix.csv",
    ]
    for name in required:
        write_rows(OUT_ROOT / name, [{"status": "gate_blocked_not_run", "reason": "Part A code/import/identity gate failed", "source_gate": "v22_35_code_truth_gate.csv"}])
    append_exec(
        "write gate-blocked matrices after Part A failure",
        task_id="A_fail_close",
        status="pass",
        files=", ".join(f"results/v22_35/{name}" for name in required),
        note=str(code),
    )


def run_c11_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_c11_start",
        status="started",
        gpu=args.c11_device,
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py",
        note="Run targeted C11 edge-safe selector repair after ActuatorSupportOnly/low-rank blocker; refresh code truth, final route, recap, and artifact index without rerunning full basis repair.",
    )
    code = stage_a()
    if int_flag(code.get("clean_unzip_compileall_pass")) and int_flag(code.get("clean_unzip_import_pass")) and int_flag(code.get("official_DGKAN_identity_pass")):
        stage_c11_edge_safe_selector(args)
        csel_control = materialize_selector_control_win_decomposition()
        append_exec(
            "materialize C10/C11 selector control-win decomposition matrix",
            task_id="E_T1_C10_C11_control_win_decomposition",
            status="pass" if csel_control.get("control_win_rows") else "warn",
            gpu="n/a",
            files="results/v22_35/v22_35_T1_C10_C11_control_win_decomposition_matrix.csv, results/v22_35/v22_35_T1_C10_C11_control_win_decomposition_summary.csv",
            note=f"rows={csel_control.get('control_win_rows')}; summary_rows={csel_control.get('summary_rows')}; any_exploration_gate_pass={csel_control.get('any_exploration_gate_pass')}",
        )
        c12_diag = materialize_c12_influence_surrogate_diagnostic()
        append_exec(
            "materialize C12 influence-surrogate diagnostic from C10/C11 branch labels",
            task_id="C12_influence_surrogate_diagnostic",
            status="pass" if c12_diag.get("diagnostic_rows") else "warn",
            gpu="n/a",
            files="results/v22_35/v22_35_T1_C12_influence_surrogate_diagnostic_matrix.csv, results/v22_35/v22_35_T1_C12_influence_surrogate_diagnostic_summary.csv",
            note=f"diagnostic_rows={c12_diag.get('diagnostic_rows')}; summary_rows={c12_diag.get('summary_rows')}; official_candidate_gate_pass={c12_diag.get('official_candidate_gate_pass')}",
        )
    else:
        append_exec(
            "skip C11 because Part A code/import/identity gate failed",
            task_id="C_C11_edge_safe_selector",
            status="gate_blocked",
            gpu=args.c11_device,
            files="results/v22_35/v22_35_code_truth_gate.csv",
            note=str(code),
        )
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    gap = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    t1 = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
    intrinsic_repair = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv") or [{}])[0]
    intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
    basis = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
    control = (read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv") or [{}])[0]
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route and recap after C11 targeted repair",
        task_id="Z_finalize_v22_35_after_C11",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}; C11_stage=completed_or_gate_blocked",
    )
    return final


def run_basis_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_basis_start",
        status="started",
        gpu=args.repair_device,
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py",
        note="Run targeted basis transfer repair with held-only branch-aware sign check; reuse existing target intrinsic repair rows and refresh final route/recap.",
    )
    code = stage_a()
    if int_flag(code.get("clean_unzip_compileall_pass")) and int_flag(code.get("clean_unzip_import_pass")) and int_flag(code.get("official_DGKAN_identity_pass")):
        basis = stage_d_basis_transfer_reaudit()
        basis_repair = stage_d_basis_transfer_repair(args)
        basis.update({f"repair_{k}" if k in basis else k: v for k, v in basis_repair.items()})
        control = stage_e_control_decomposition()
        stage_f_curvature_representation()
        stage_g_h_gate_matrices()
    else:
        append_exec(
            "skip basis repair because Part A code/import/identity gate failed",
            task_id="D_basis_transfer_repair",
            status="gate_blocked",
            gpu=args.repair_device,
            files="results/v22_35/v22_35_code_truth_gate.csv",
            note=str(code),
        )
        basis = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
        control = (read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv") or [{}])[0]
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    gap = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    t1 = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
    intrinsic_repair = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv") or [{}])[0]
    intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route and recap after basis targeted repair",
        task_id="Z_finalize_v22_35_after_basis",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}; basis_stage=completed_or_gate_blocked",
    )
    return final


def repair_noise_missing_keys(args: argparse.Namespace) -> list[str]:
    lookup = epsilon_lookup(getattr(args, "hidden", None))
    horizons = sorted(
        {
            int(x)
            for x in (
                split_csv(args.repair_branch_horizons, int)
                + split_csv(getattr(args, "basis_repair_branch_horizons", ""), int)
            )
            if int(x) > 0
        }
    )
    missing: list[str] = []
    for dataset in split_csv(args.repair_datasets):
        for seed in split_csv(args.repair_seeds, int):
            for family in split_csv(args.repair_families):
                for horizon in horizons:
                    key = (str(dataset), str(seed), str(family), str(horizon))
                    if key not in lookup:
                        missing.append(f"{dataset}:seed{seed}:{family}:H{horizon}")
    return missing


def run_curvature_basis_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_curvature_basis_start",
        status="started",
        gpu=args.repair_device,
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py",
        note="Run Part F curvature-safe target repair, then intrinsic sign/scale and basis branch transfer gates; selection remains held-train only.",
    )
    code = stage_a()
    if int_flag(code.get("clean_unzip_compileall_pass")) and int_flag(code.get("clean_unzip_import_pass")) and int_flag(code.get("official_DGKAN_identity_pass")):
        missing_noise = repair_noise_missing_keys(args)
        if missing_noise:
            run_repair_row_local_noise_repeats(args)
        else:
            append_exec(
                "reuse existing repair-config row-local same-checkpoint no-op repeats",
                task_id="B_repair_row_local_noop_repeats",
                status="pass",
                gpu=args.noise_device,
                files="results/v22_35/v22_35_row_local_noise_repeats.csv, results/v22_35/v22_35_row_local_noise_matrix.csv",
                note="all requested repair/basis horizons already have direct epsilon rows; missing=0",
            )
        intrinsic_repair = stage_d_intrinsic_sign_scale_repair(args)
        intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
        intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
        basis = stage_d_basis_transfer_reaudit()
        basis_repair = stage_d_basis_transfer_repair(args)
        basis.update({f"repair_{k}" if k in basis else k: v for k, v in basis_repair.items()})
        control = stage_e_control_decomposition()
        stage_f_curvature_representation()
        stage_g_h_gate_matrices()
        write_figures()
    else:
        append_exec(
            "skip curvature-safe basis repair because Part A code/import/identity gate failed",
            task_id="D_curvature_safe_basis_repair",
            status="gate_blocked",
            gpu=args.repair_device,
            files="results/v22_35/v22_35_code_truth_gate.csv",
            note=str(code),
        )
        intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
        basis = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
        control = (read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv") or [{}])[0]
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    gap = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    t1 = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route and recap after curvature-safe target basis repair",
        task_id="Z_finalize_v22_35_after_curvature_basis",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=f"final_route={final.get('final_route')}; official_full_superiority_ready={final.get('official_full_superiority_ready')}; curvature_basis_stage=completed_or_gate_blocked",
    )
    return final


def run_continual_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_continual_start",
        status="started",
        gpu=args.continual_device,
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py",
        note=(
            "Run Part H continual non-degenerate baseline sweep and D8 boundary repair; "
            "uses fixed step grid and pre-registered repair regime ordering, no best-row promotion."
        ),
    )
    code = stage_a()
    if int_flag(code.get("clean_unzip_compileall_pass")) and int_flag(code.get("clean_unzip_import_pass")) and int_flag(code.get("official_DGKAN_identity_pass")):
        stage_h_continual_nondegenerate_boundary(args)
        stage_g_h_gate_matrices()
        write_figures()
    else:
        write_rows(
            OUT_ROOT / "v22_35_continual_nondegenerate_boundary_summary.csv",
            [{"status": "gate_blocked_code_truth_failed", "source_gate": "v22_35_code_truth_gate.csv"}],
        )
        append_exec(
            "skip continual non-degenerate boundary because Part A code/import/identity gate failed",
            task_id="H_continual_nondegenerate_boundary",
            status="gate_blocked",
            gpu=args.continual_device,
            files="results/v22_35/v22_35_code_truth_gate.csv, results/v22_35/v22_35_continual_nondegenerate_boundary_summary.csv",
            note=str(code),
        )
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    gap = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    t1 = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
    intrinsic_repair = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv") or [{}])[0]
    intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
    basis = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
    control = (read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv") or [{}])[0]
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route and recap after continual non-degenerate boundary repair",
        task_id="Z_finalize_v22_35_after_continual",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=(
            f"final_route={final.get('final_route')}; "
            f"official_full_superiority_ready={final.get('official_full_superiority_ready')}; "
            "continual_stage=completed_or_gate_blocked"
        ),
    )
    return final


def run_optimizer_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_optimizer_start",
        status="started",
        gpu=args.optimizer_device,
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py",
        note=(
            "Run Part G direct strong-optimizer interaction probe with fixed optimizer family, same initialization, "
            "same-seed loaders, norm-matched slow-signal FU and matched controls; not eligible for official promotion while upstream gates remain closed."
        ),
    )
    code = stage_a()
    if int_flag(code.get("clean_unzip_compileall_pass")) and int_flag(code.get("clean_unzip_import_pass")) and int_flag(code.get("official_DGKAN_identity_pass")):
        stage_g_optimizer_direct_probe(args)
        stage_g_h_gate_matrices()
        write_figures()
    else:
        write_rows(
            OUT_ROOT / "v22_35_optimizer_direct_probe_summary.csv",
            [{"status": "gate_blocked_code_truth_failed", "source_gate": "v22_35_code_truth_gate.csv"}],
        )
        append_exec(
            "skip direct optimizer probe because Part A code/import/identity gate failed",
            task_id="G_optimizer_direct_probe",
            status="gate_blocked",
            gpu=args.optimizer_device,
            files="results/v22_35/v22_35_code_truth_gate.csv, results/v22_35/v22_35_optimizer_direct_probe_summary.csv",
            note=str(code),
        )
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    gap = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    t1 = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
    intrinsic_repair = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv") or [{}])[0]
    intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
    basis = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
    control = (read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv") or [{}])[0]
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route and recap after direct optimizer strong-family probe",
        task_id="Z_finalize_v22_35_after_optimizer",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=(
            f"final_route={final.get('final_route')}; "
            f"official_full_superiority_ready={final.get('official_full_superiority_ready')}; "
            "optimizer_stage=completed_or_gate_blocked"
        ),
    )
    return final


def run_optimizer_finalize_only(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        " ".join([f"KAN_PYTHON={PYTHON}", shlex.quote(sys.executable), *[shlex.quote(a) for a in sys.argv]]),
        task_id="v22_35_optimizer_finalize_start",
        status="started",
        gpu="n/a",
        files=f"{PLAN_DOC.relative_to(ROOT)}, experiments/run_v22_35_causal_target_noise_calibrated_fu.py, results/v22_35/v22_35_optimizer_direct_probe_matrix.csv",
        note="Refresh Part G/H matrices, optimizer direct four-square, figures, final route and recap from existing direct optimizer probe rows; no training rerun.",
    )
    code = stage_a()
    if int_flag(code.get("clean_unzip_compileall_pass")) and int_flag(code.get("clean_unzip_import_pass")) and int_flag(code.get("official_DGKAN_identity_pass")):
        stage_g_h_gate_matrices()
        write_figures()
    else:
        append_exec(
            "skip optimizer finalize because Part A code/import/identity gate failed",
            task_id="G_H_optimizer_finalize",
            status="gate_blocked",
            files="results/v22_35/v22_35_code_truth_gate.csv",
            note=str(code),
        )
    _idx, manifest = artifact_index()
    code["artifact_manifest_hash"] = manifest
    write_rows(OUT_ROOT / "v22_35_code_truth_gate.csv", [code])
    gap = (read_rows(OUT_ROOT / "v22_35_gap_truth_recalibrated_summary.csv") or [{}])[0]
    t1 = (read_rows(OUT_ROOT / "v22_35_T1_selector_summary.csv") or [{}])[0]
    intrinsic = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_benefit_summary.csv") or [{}])[0]
    intrinsic_repair = (read_rows(OUT_ROOT / "v22_35_target_intrinsic_repair_summary.csv") or [{}])[0]
    intrinsic.update({f"repair_{k}" if k in intrinsic else k: v for k, v in intrinsic_repair.items()})
    basis = (read_rows(OUT_ROOT / "v22_35_basis_transfer_summary.csv") or [{}])[0]
    control = (read_rows(OUT_ROOT / "v22_35_control_win_decomposition_summary.csv") or [{}])[0]
    final = decide_final(code, gap, t1, intrinsic, basis, control)
    write_recap(final)
    artifact_index()
    append_exec(
        "write final route and recap after optimizer four-square finalize",
        task_id="Z_finalize_v22_35_after_optimizer_four_square",
        status="pass",
        gpu="n/a",
        files="results/v22_35/v22_35_optimizer_direct_four_square_matrix.csv, results/v22_35/v22_35_optimizer_direct_four_square_summary.csv, results/v22_35/v22_35_optimizer_direct_kan_internal_summary.csv, results/v22_35/v22_35_final_route.json, docs/DG-KAN_v22.35_CausalTargetNoiseCalibratedFU_实验结果复盘.md, results/v22_35/v22_35_artifact_index.csv",
        note=(
            f"final_route={final.get('final_route')}; "
            f"official_full_superiority_ready={final.get('official_full_superiority_ready')}; "
            "data_unchanged=1"
        ),
    )
    return final


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "c11", "basis", "curvature_basis", "continual", "optimizer", "optimizer_finalize"])
    p.add_argument("--noise-device", default="cuda:0")
    p.add_argument("--noise-datasets", default="Wine")
    p.add_argument("--noise-seeds", default="0,1")
    p.add_argument("--noise-models", default="MLP,D-CHE,D-FOU")
    p.add_argument("--noise-train-size", type=int, default=128)
    p.add_argument("--noise-test-size", type=int, default=48)
    p.add_argument("--noise-pretrain-steps", type=int, default=20)
    p.add_argument("--noise-branch-horizon", type=int, default=60)
    p.add_argument("--noise-branch-horizons", default="")
    p.add_argument("--noise-repeats", type=int, default=3)
    p.add_argument("--intrinsic-device", default="cuda:2")
    p.add_argument("--intrinsic-datasets", default="Wine")
    p.add_argument("--intrinsic-seeds", default="0")
    p.add_argument("--intrinsic-families", default="D-CHE,D-FOU")
    p.add_argument("--intrinsic-target-families", default="D2_multi_cohort_basis_benefit_target,D4_hard_slice_margin_target,D7_control_orthogonal_basis_target,D9_held_class_CVaR_control_orthogonal_target,D10_edge_safe_causal_target,D11_class_confusion_target,D12_curvature_safe_label_smooth_target")
    p.add_argument("--intrinsic-train-size", type=int, default=128)
    p.add_argument("--intrinsic-test-size", type=int, default=48)
    p.add_argument("--intrinsic-examples", type=int, default=48)
    p.add_argument("--intrinsic-max-output-rows", type=int, default=144)
    p.add_argument("--intrinsic-pretrain-steps", type=int, default=20)
    p.add_argument("--intrinsic-branch-horizon", type=int, default=60)
    p.add_argument("--intrinsic-damping", type=float, default=1.0e-4)
    p.add_argument("--intrinsic-update-scale", type=float, default=0.003)
    p.add_argument("--intrinsic-held-ce-tolerance", type=float, default=0.0)
    p.add_argument("--repair-device", default="cuda:2")
    p.add_argument("--repair-datasets", default="Wine,CIFAR10")
    p.add_argument("--repair-seeds", default="0")
    p.add_argument("--repair-families", default="D-CHE,D-FOU")
    p.add_argument("--repair-target-families", default="D2_multi_cohort_basis_benefit_target,D4_hard_slice_margin_target,D7_control_orthogonal_basis_target,D9_held_class_CVaR_control_orthogonal_target,D10_edge_safe_causal_target,D11_class_confusion_target,D12_curvature_safe_label_smooth_target")
    p.add_argument("--repair-train-size", type=int, default=128)
    p.add_argument("--repair-test-size", type=int, default=48)
    p.add_argument("--repair-examples", type=int, default=48)
    p.add_argument("--repair-max-output-rows", type=int, default=144)
    p.add_argument("--repair-pretrain-steps", type=int, default=20)
    p.add_argument("--repair-branch-horizons", default="0,20,60,200")
    p.add_argument("--repair-scale-multipliers", default="0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--repair-damping", type=float, default=1.0e-4)
    p.add_argument("--repair-update-scale", type=float, default=0.003)
    p.add_argument("--repair-held-ce-tolerance", type=float, default=0.0)
    p.add_argument("--basis-repair-branch-horizons", default="200,400,800")
    p.add_argument("--basis-repair-bank-dims", default="4,8,16")
    p.add_argument("--basis-repair-low-rank-sketch-ranks", default="4,8,16")
    p.add_argument("--basis-repair-signs", default="1,-1")
    p.add_argument("--basis-repair-scale-multipliers", default="0.25,0.5,1.0,2.0,4.0")
    p.add_argument("--basis-repair-damping", type=float, default=1.0e-4)
    p.add_argument("--basis-repair-update-scale", type=float, default=0.003)
    p.add_argument("--basis-repair-gradcheck-eps", type=float, default=1.0e-4)
    p.add_argument("--basis-repair-gradcheck-eps-values", default="1e-5,3e-5,1e-4,3e-4,1e-3")
    p.add_argument("--basis-repair-held-ce-tolerance", type=float, default=1.0e-5)
    p.add_argument("--basis-repair-sharpness-tolerance", type=float, default=1.0e-6)
    p.add_argument("--basis-repair-branch-aware-sign-check", type=int, default=1)
    p.add_argument("--basis-repair-branch-aware-sign-horizon", type=int, default=200)
    p.add_argument("--c11-device", default="cuda:1")
    p.add_argument("--c11-datasets", default="Wine,CIFAR10,SVHN,EMNIST")
    p.add_argument("--c11-seeds", default="0")
    p.add_argument("--c11-train-size", type=int, default=128)
    p.add_argument("--c11-test-size", type=int, default=48)
    p.add_argument("--c11-examples", type=int, default=48)
    p.add_argument("--c11-pretrain-steps", type=int, default=20)
    p.add_argument("--c11-signal-cohorts", type=int, default=4)
    p.add_argument("--c11-cohort-size", type=int, default=12)
    p.add_argument("--c11-rank-cap", type=int, default=6)
    p.add_argument("--c11-sketch-dim", type=int, default=64)
    p.add_argument("--c11-candidate-count", type=int, default=16)
    p.add_argument("--c11-branch-trust", type=float, default=0.005)
    p.add_argument("--c11-branch-horizon", type=int, default=400)
    p.add_argument("--c11-held-ce-tolerance", type=float, default=0.0)
    p.add_argument("--c11-margin-debt-tolerance", type=float, default=0.0)
    p.add_argument("--c11-curvature-weight", type=float, default=0.02)
    p.add_argument("--continual-device", default="cuda:3")
    p.add_argument("--continual-seeds", default="0")
    p.add_argument("--continual-families", default="D-CHE,D-FOU")
    p.add_argument("--continual-train-size", type=int, default=500)
    p.add_argument("--continual-test-size", type=int, default=250)
    p.add_argument("--continual-task0-steps", default="20,40,80,120")
    p.add_argument("--continual-task1-steps", default="5,10,20,40,80")
    p.add_argument("--continual-min-task0-acc-before-task1", type=float, default=0.50)
    p.add_argument("--continual-min-old-acc-after-task1", type=float, default=0.05)
    p.add_argument("--continual-min-task1-acc-after-task1", type=float, default=0.50)
    p.add_argument("--continual-max-base-forgetting", type=float, default=0.95)
    p.add_argument("--continual-max-repair-regimes", type=int, default=2)
    p.add_argument("--continual-bank-dims", default="4,8,16")
    p.add_argument("--continual-basis-examples", type=int, default=64)
    p.add_argument("--continual-max-output-rows", type=int, default=512)
    p.add_argument("--continual-target-mode", default="ce_boundary", choices=["ce_boundary", "hard_cvar_boundary", "hard_margin_boundary"])
    p.add_argument("--continual-target-hard-fraction", type=float, default=0.50)
    p.add_argument("--continual-sign-select", type=int, default=0)
    p.add_argument("--continual-damping", type=float, default=1.0e-4)
    p.add_argument("--continual-update-scale", type=float, default=0.003)
    p.add_argument("--continual-gradcheck-eps", type=float, default=1.0e-3)
    p.add_argument("--continual-min-relative-forgetting-reduction", type=float, default=0.05)
    p.add_argument("--continual-current-accuracy-tolerance", type=float, default=0.01)
    p.add_argument("--optimizer-device", default="cuda:3")
    p.add_argument("--optimizer-datasets", default="Wine")
    p.add_argument("--optimizer-seeds", default="0")
    p.add_argument("--optimizer-families", default="MLP,D-CHE,D-FOU")
    p.add_argument("--optimizer-variants", default="AdamW,Cautious AdamW,Schedule-Free AdamW,Muon-like")
    p.add_argument("--optimizer-train-size", type=int, default=128)
    p.add_argument("--optimizer-test-size", type=int, default=48)
    p.add_argument("--optimizer-steps", type=int, default=80)
    p.add_argument("--optimizer-signal-alpha", type=float, default=0.20)
    p.add_argument(
        "--optimizer-acceptance",
        default="none",
        choices=[
            "none",
            "train_loss_nonworse",
            "train_loss_beats_base_step",
            "held_train_loss_nonworse",
            "held_train_loss_beats_base_step",
        ],
    )
    p.add_argument("--optimizer-acceptance-tol", type=float, default=0.0)
    p.add_argument("--tier2-download", type=int, default=1)
    p.add_argument("--hard-fraction", type=float, default=0.25)
    p.add_argument("--curvature-safe-label-smoothing", type=float, default=0.10)
    p.add_argument("--curvature-safe-temperature", type=float, default=2.0)
    p.add_argument("--sharpness-rho", type=float, default=1.0e-3)
    p.add_argument("--sharpness-tolerance", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.stage == "c11":
        final = run_c11_only(args)
    elif args.stage == "basis":
        final = run_basis_only(args)
    elif args.stage == "curvature_basis":
        final = run_curvature_basis_only(args)
    elif args.stage == "continual":
        final = run_continual_only(args)
    elif args.stage == "optimizer":
        final = run_optimizer_only(args)
    elif args.stage == "optimizer_finalize":
        final = run_optimizer_finalize_only(args)
    else:
        final = run_all(args)
    print(json.dumps({"final_route": final.get("final_route"), "official_full_superiority_ready": final.get("official_full_superiority_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
