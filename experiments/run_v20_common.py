"""Shared helpers for v20 execution artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
V20_ROOT = ROOT / "results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu"
V20_OFFICIAL = V20_ROOT / "official_v20"
V19_OFFICIAL = ROOT / "results/v19_0_source_channel_fu_basis_kernel_breakthrough_4gpu/official_v19"
V20_EXEC_DOC = ROOT / "docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_执行日志.md"
V20_RECAP_DOC = ROOT / "docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_实验结果复盘.md"
V20_PLAN_DOC = ROOT / "docs/DG-KAN_v20.0_SourceChannelFU_BasisKernelOfficialization_4GPU_完整计划.md"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V20_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def append_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text)


def append_exec(out_dir: Path, command: str, *, status: str = "", note: str = "") -> None:
    row = {"timestamp": now_sg(), "command": command, "status": status, "note": note}
    journal = read_rows(out_dir / "v20_command_journal.csv")
    journal.append({k: str(v) for k, v in row.items()})
    write_rows(out_dir / "v20_command_journal.csv", journal)
    append_text(V20_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V20_EXEC_DOC, f"- note: {note}\n")


def run_cmd(command: list[str], *, cwd: Path = ROOT, timeout: int = 300) -> tuple[int, str]:
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return proc.returncode, "\n".join(
        [
            "$ " + " ".join(command),
            f"exit={proc.returncode}",
            "--- stdout ---",
            proc.stdout,
            "--- stderr ---",
            proc.stderr,
        ]
    )


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_csvs(out_dir: Path, pattern: str, target: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(out_dir.glob(pattern)):
        if path.name == target:
            continue
        rows.extend(read_rows(path))
    write_rows(out_dir / target, rows)
    return rows


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def placeholder_svg(path: Path, title: str, rows: list[dict[str, Any]], metric: str = "") -> None:
    vals = [finite_float(r.get(metric)) for r in rows] if metric else []
    vals = [v for v in vals if math.isfinite(v)]
    summary = f"rows={len(rows)}"
    if vals:
        summary += f" min={min(vals):.4g} mean={sum(vals)/len(vals):.4g} max={max(vals):.4g}"
    text = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='220'>"
        "<rect width='100%' height='100%' fill='#f7f7f5'/>"
        f"<text x='24' y='60' font-family='monospace' font-size='22'>{title}</text>"
        f"<text x='24' y='110' font-family='monospace' font-size='16'>{summary}</text>"
        "</svg>\n"
    )
    write_text(path, text)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


V20_EFFICIENCY_VARIANTS = {
    "D-CHE": [
        "CHE20-R2-low-degree-k3-triton-officialize",
        "CHE20-R3-k4-triton-no-materialize-officialize",
        "CHE20-R4-k3-gradbuf-triton-officialize",
    ],
    "D-FOU": [
        "FOU20-R2-officialize-lowfreq-k2-stream",
        "FOU20-R4-k4-triton-no-materialize-officialize",
    ],
    "LQ": ["LQ20-recurrence-fixed-frame-smoke"],
    "D-RAT": ["RAT20-branchless-horner-smoke"],
    "D-RBF": ["RBF20-compact-local-k4-smoke"],
    "D-WAV": ["WAV20-sparse-support-smoke"],
}


def to_v19_repair_variant(family: str, v20_variant: str) -> str:
    text = str(v20_variant)
    mapping = {
        "CHE20-R2-low-degree-k3-triton-officialize": "CHE-R2-low-degree-k3-triton",
        "CHE20-R3-k4-triton-no-materialize-officialize": "CHE-R3-k4-triton-no-materialize",
        "CHE20-R4-k3-gradbuf-triton-officialize": "CHE-R4-k3-gradbuf-triton",
        "FOU20-R2-officialize-lowfreq-k2-stream": "FOU-R2-low-frequency-k2-stream",
        "FOU20-R4-k4-triton-no-materialize-officialize": "FOU-R4-k4-triton-no-materialize",
    }
    return mapping.get(text, "R0-current")


def v20_functional_specs(scope: str) -> list[dict[str, Any]]:
    if scope == "mlp_anatomy":
        return [
            {"v20_id": "F1-MLP-M16-replay-exact", "continuation_id": "M16-warm800-alt50-fu0p0001", "mechanism": "M16-TwoPhaseMomentumThenLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "replay"},
            {"v20_id": "F1a-remove-two-phase-warmup", "continuation_id": "M15-linec-alt50-fu0p0005", "mechanism": "M15-LineCFilteredAlternatingFU", "fu_lr": 0.0005, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "remove_warmup"},
            {"v20_id": "F1b-remove-alternation", "continuation_id": "M2-fu0p0001", "mechanism": "M2-SGDMomentumPrimaryFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "momentum_only"},
            {"v20_id": "F1c-remove-LineC-filtered-source-estimator", "continuation_id": "M5-alt50-fu0p0005", "mechanism": "M5-AlternatingFUGradient", "fu_lr": 0.0005, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "alternation_no_linec_filter"},
            {"v20_id": "F1d-random-same-norm-source-state", "continuation_id": "CTRL-RandomMatchedNorm", "mechanism": "CTRL-RandomMatchedNorm", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_random_norm"},
            {"v20_id": "F1f-AdamW-primary-only", "continuation_id": "CTRL-AdamW", "mechanism": "CTRL-AdamW", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_adamw"},
            {"v20_id": "F1j-matrix-block-hidden-writer", "continuation_id": "M13-lowrank-r4-fu0p0001", "mechanism": "M13-LowRankMatrixBlockFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "matrix_block"},
            {"v20_id": "F1k-output-readout-only-writer", "continuation_id": "M17-readout-warm800-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "readout_only"},
            {"v20_id": "F3a-MLP-PopRiskSlowState-fu0p0005", "continuation_id": "M14-popriskslow-fu0p0005", "mechanism": "M14-SourceChannelPopRiskSlowFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_slow_state"},
            {"v20_id": "F3a2-MLP-PopRiskSlowState-fu0p0001", "continuation_id": "M14-popriskslow-fu0p0001", "mechanism": "M14-SourceChannelPopRiskSlowFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_slow_state"},
            {"v20_id": "F3b-MLP-PopRiskMatrixBlock-fu0p0005", "continuation_id": "M33-poprisk-blockslow-fu0p0005", "mechanism": "M33-PopRiskMatrixBlockSlowFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_matrix_block"},
            {"v20_id": "F3b2-MLP-PopRiskMatrixBlock-fu0p0001", "continuation_id": "M33-poprisk-blockslow-fu0p0001", "mechanism": "M33-PopRiskMatrixBlockSlowFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_matrix_block"},
            {"v20_id": "F3r1-MLP-ExactPopRiskK32-fu0p0005", "continuation_id": "M34-exact-poprisk-k32-fu0p0005", "mechanism": "M34-ExactPopRiskSlowFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_exact_k32"},
            {"v20_id": "F3r2-MLP-ExactPopRiskK32-fu0p0001", "continuation_id": "M34-exact-poprisk-k32-fu0p0001", "mechanism": "M34-ExactPopRiskSlowFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_exact_k32"},
            {"v20_id": "F3r3-MLP-ExactPopRiskK32Block-fu0p0005", "continuation_id": "M35-exact-poprisk-block-k32-fu0p0005", "mechanism": "M35-ExactPopRiskMatrixBlockSlowFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_exact_k32_block"},
            {"v20_id": "F3r4-MLP-ExactPopRiskK32Block-fu0p0001", "continuation_id": "M35-exact-poprisk-block-k32-fu0p0001", "mechanism": "M35-ExactPopRiskMatrixBlockSlowFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_poprisk_exact_k32_block"},
            {"v20_id": "F5a-MLP-LowRankMatrixBlock-fu0p0005", "continuation_id": "M13-lowrank-r4-fu0p0005", "mechanism": "M13-LowRankMatrixBlockFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_matrix_block"},
            {"v20_id": "F5a2-MLP-ExplicitMatrixBlock-fu0p0005", "continuation_id": "M7-matrixblock-fu0p0005", "mechanism": "M7-MatrixBlockFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "case_a_matrix_block"},
            {"v20_id": "F5b-MLP-OutputReadoutBlock-fu0p0001", "continuation_id": "M17-readout-warm1-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1, "hypothesis": "case_a_output_readout_block"},
            {"v20_id": "F5r1-MLP-MomentumWarmReadoutBlock-alt50-fu0p0001", "continuation_id": "M36-momwarm800-readout-alt50-fu0p0001", "mechanism": "M36-MomentumWarmReadoutBlockFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "case_a_momentum_warm_readout_block"},
            {"v20_id": "F5r2-MLP-MomentumWarmReadoutBlock-alt100-fu0p00005", "continuation_id": "M36-momwarm800-readout-alt100-fu0p00005", "mechanism": "M36-MomentumWarmReadoutBlockFU", "fu_lr": 0.00005, "alt_period": 100, "source_warmup_steps": 800, "hypothesis": "case_a_momentum_warm_readout_block"},
            {"v20_id": "F7a-MLP-SplitFisherAgreement-alt50-fu0p0001", "continuation_id": "M37-split-fisher-alt50-fu0p0001", "mechanism": "M37-SplitFisherAgreementSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "case_a_split_fisher_agreement_source_state"},
            {"v20_id": "F7b-MLP-SplitFisherAgreement-alt100-fu0p00005", "continuation_id": "M37-split-fisher-alt100-fu0p00005", "mechanism": "M37-SplitFisherAgreementSlowFU", "fu_lr": 0.00005, "alt_period": 100, "source_warmup_steps": 0, "hypothesis": "case_a_split_fisher_agreement_source_state"},
            {"v20_id": "F7c-MLP-AdamWSplitFisherResidual-alt100-fu0p00005", "continuation_id": "M38-adamw-split-fisher-alt100-fu0p00005", "mechanism": "M38-AdamWSplitFisherAgreementResidualFU", "fu_lr": 0.00005, "alt_period": 100, "source_warmup_steps": 0, "hypothesis": "case_a_adamw_split_fisher_residual"},
            {"v20_id": "F7r1-MLP-MomentumWarmSplitFisher-warm400-alt50-fu0p0001", "continuation_id": "M39-momwarm400-split-fisher-alt50-fu0p0001", "mechanism": "M39-MomentumWarmSplitFisherFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "case_a_momentum_warm_split_fisher_repair"},
            {"v20_id": "F7r2-MLP-MomentumWarmSplitFisher-warm800-alt50-fu0p0001", "continuation_id": "M39-momwarm800-split-fisher-alt50-fu0p0001", "mechanism": "M39-MomentumWarmSplitFisherFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "case_a_momentum_warm_split_fisher_repair"},
            {"v20_id": "F7r3-MLP-MomentumWarmSplitFisher-warm1200-alt50-fu0p0001", "continuation_id": "M39-momwarm1200-split-fisher-alt50-fu0p0001", "mechanism": "M39-MomentumWarmSplitFisherFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200, "hypothesis": "case_a_momentum_warm_split_fisher_repair"},
            {"v20_id": "F8a-MLP-MomentumWarmAntiWashout-warm800", "continuation_id": "M40-momwarm800-antiwashout", "mechanism": "M40-MomentumWarmAntiWashoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "case_a_momentum_source_antiwashout_projection"},
            {"v20_id": "F8b-MLP-MomentumWarmAntiWashout-warm1200", "continuation_id": "M40-momwarm1200-antiwashout", "mechanism": "M40-MomentumWarmAntiWashoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200, "hypothesis": "case_a_momentum_source_antiwashout_projection"},
            {"v20_id": "F8c-MLP-MomentumWarmSourceAnchor-warm800", "continuation_id": "M41-momwarm800-source-anchor", "mechanism": "M41-MomentumWarmSourceAnchorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "case_a_momentum_source_anchor_residual"},
            {"v20_id": "F8d-MLP-MomentumWarmSourceAnchor-warm1200", "continuation_id": "M41-momwarm1200-source-anchor", "mechanism": "M41-MomentumWarmSourceAnchorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200, "hypothesis": "case_a_momentum_source_anchor_residual"},
            {"v20_id": "F9a-MLP-MomentumWarmHold-warm800", "continuation_id": "M42-momwarm800-hold", "mechanism": "M42-MomentumWarmHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "case_a_postwarmup_hold_diagnostic"},
            {"v20_id": "F9b-MLP-MomentumWarmHold-warm1200", "continuation_id": "M42-momwarm1200-hold", "mechanism": "M42-MomentumWarmHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200, "hypothesis": "case_a_postwarmup_hold_diagnostic"},
            {"v20_id": "F10a-MLP-MomentumCycleHold-warm800", "continuation_id": "M43-momcycle800-hold", "mechanism": "M43-MomentumCycleHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "case_a_periodic_momentum_refresh_hold"},
            {"v20_id": "CTRL-NoOpMatchedOverhead", "continuation_id": "CTRL-NoOpMatchedOverhead", "mechanism": "CTRL-NoOpMatchedOverhead", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_noop"},
            {"v20_id": "CTRL-SGD", "continuation_id": "CTRL-SGD", "mechanism": "CTRL-SGD", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_sgd"},
        ]
    return [
        {"v20_id": "F3-PopRiskSlowState", "continuation_id": "M14-popriskslow-fu0p0005", "mechanism": "M14-SourceChannelPopRiskSlowFU", "fu_lr": 0.0005, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "poprisk_slow_state"},
        {"v20_id": "F4-ExactReadoutActuation", "continuation_id": "M23-exact-readout-ultracap-alt50-fu0p0001", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_actuation"},
        {"v20_id": "F5-MatrixBlockWriter", "continuation_id": "M13-lowrank-r4-fu0p0001", "mechanism": "M13-LowRankMatrixBlockFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "matrix_block"},
        {"v20_id": "F6-ReadoutCarrierWriter", "continuation_id": "M17-readout-warm800-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "readout_carrier"},
        {"v20_id": "F6-TwoPhaseSourceWriter", "continuation_id": "M16-warm800-alt50-fu0p0001", "mechanism": "M16-TwoPhaseMomentumThenLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "two_phase_port"},
        {"v20_id": "CTRL-AdamW", "continuation_id": "CTRL-AdamW", "mechanism": "CTRL-AdamW", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_adamw"},
        {"v20_id": "CTRL-SGD", "continuation_id": "CTRL-SGD", "mechanism": "CTRL-SGD", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_sgd"},
        {"v20_id": "CTRL-RandomMatchedNorm", "continuation_id": "CTRL-RandomMatchedNorm", "mechanism": "CTRL-RandomMatchedNorm", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_random_norm"},
        {"v20_id": "CTRL-NoOpMatchedOverhead", "continuation_id": "CTRL-NoOpMatchedOverhead", "mechanism": "CTRL-NoOpMatchedOverhead", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_noop"},
    ]


def build_packet(out_dir: Path, required_artifacts: list[str]) -> None:
    packet = out_dir / "v20_code_review_packet"
    if packet.exists():
        shutil.rmtree(packet)
    sections = [
        "00_README.md",
        "01_ENVIRONMENT",
        "02_SOURCE_TREE",
        "03_IMPORT_CLOSURE",
        "04_LINEC_FAST_CHANNEL",
        "05_RETENTION_DEBT_ROUTE",
        "06_MECHANISM_SEMANTICS",
        "07_FUNCTIONAL_SOURCE_CHANNEL",
        "08_OPTIMIZER_COUPLING",
        "09_EFFICIENCY_KERNELS",
        "10_EXPERIMENT_RUNNERS",
        "11_RAW_MATRICES",
        "12_FIGURES",
        "13_FAILURE_TAXONOMY",
        "14_REPRO_COMMANDS",
        "15_GPU_QUEUE",
    ]
    for section in sections:
        target = packet / section
        if Path(section).suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
    _, git_status = run_cmd(["git", "status", "--short"], timeout=60)
    _, git_head = run_cmd(["git", "rev-parse", "HEAD"], timeout=60)
    _, py_version = run_cmd([PYTHON, "--version"], timeout=60)
    _, nvidia = run_cmd(["nvidia-smi"], timeout=60)
    write_text(packet / "01_ENVIRONMENT/git_status.txt", git_status)
    write_text(packet / "01_ENVIRONMENT/git_head.txt", git_head)
    write_text(packet / "01_ENVIRONMENT/python_version.txt", py_version)
    write_text(packet / "01_ENVIRONMENT/cuda_info.txt", nvidia)
    copy_if_exists(V20_PLAN_DOC, packet / "02_SOURCE_TREE/docs" / V20_PLAN_DOC.name)
    copy_if_exists(V20_EXEC_DOC, packet / "02_SOURCE_TREE/docs" / V20_EXEC_DOC.name)
    copy_if_exists(V20_RECAP_DOC, packet / "02_SOURCE_TREE/docs" / V20_RECAP_DOC.name)
    for rel in [
        "dgkan/metrics/linec.py",
        "dgkan/fu/mechanisms.py",
        "dgkan/fu/source_channel.py",
        "dgkan/fu/slow_state.py",
        "dgkan/fu/function_space_actuation.py",
        "dgkan/fu/matrix_block.py",
        "dgkan/profiling/efficiency_v20.py",
        "dgkan/kernels/cheby_fused.py",
        "dgkan/kernels/fourier_fused.py",
        "dgkan/kernels/lq_fused.py",
        "dgkan/kernels/rational_fused.py",
        "dgkan/kernels/rbf_sparse.py",
        "dgkan/kernels/wavelet_sparse.py",
        "experiments/run_v20_s04_truth_gate.py",
        "experiments/run_v20_mlp_retained_source_anatomy.py",
        "experiments/run_v20_kan_source_channel_writer.py",
        "experiments/run_v20_basis_kernel_officialization.py",
        "experiments/run_v20_function_space_actuation.py",
        "experiments/run_v20_merge_finalize.py",
        "experiments/run_v20_common.py",
    ]:
        copy_if_exists(ROOT / rel, packet / "02_SOURCE_TREE" / rel)
    missing_v19 = []
    for rel in [
        "experiments/run_v19_h10_m31_full.py",
        "experiments/run_v19_h11_m31_independent.py",
        "experiments/run_v19_h12_m32_postadamw.py",
        "experiments/run_v19_h13_anatomy.py",
    ]:
        if (ROOT / rel).exists():
            copy_if_exists(ROOT / rel, packet / "02_SOURCE_TREE" / rel)
        else:
            missing_v19.append({"source_file": rel, "exists": 0, "missing_source_reason": "v19 continuation implemented in run_v19_functional_continuation.py plus run_v19_h13_h10_h11_anatomy.py; standalone named wrapper absent", "cannot_independently_audit_continuation": 1})
    write_rows(out_dir / "v20_missing_v19_continuation_source.csv", missing_v19)

    section_map = {
        "03_IMPORT_CLOSURE": ["v20_code_truth_gate.csv", "v20_import_closure.csv", "v20_compileall.log"],
        "04_LINEC_FAST_CHANNEL": ["v20_linec_fast_golden.csv", "v20_linec_channel_golden.csv"],
        "05_RETENTION_DEBT_ROUTE": ["v20_debt_route_unit_tests.csv", "v20_debt_accounting_matrix.csv", "v20_source_retention_matrix.csv"],
        "06_MECHANISM_SEMANTICS": ["v20_mechanism_semantic_contract.csv"],
        "07_FUNCTIONAL_SOURCE_CHANNEL": ["v20_mlp_m16_anatomy.csv", "v20_kan_source_writer_matrix.csv", "v20_poprisk_slow_state_matrix.csv", "v20_matrix_block_fu_matrix.csv", "v20_function_space_actuation_matrix.csv"],
        "08_OPTIMIZER_COUPLING": ["v20_optimizer_washout_matrix.csv", "v19_h13_h10_h11_gate_diagnostics.csv"],
        "09_EFFICIENCY_KERNELS": ["v20_efficiency_truth_table.csv", "v20_efficiency_waterfall.csv", "v20_kernel_gradcheck.csv"],
        "10_EXPERIMENT_RUNNERS": ["v20_command_journal.csv"],
        "11_RAW_MATRICES": ["v20_mlp_m16_anatomy_matrix.csv", "v20_mlp_m16_anatomy_traces.csv", "v20_kan_source_writer_raw_matrix.csv", "v20_kan_source_writer_raw_traces.csv"],
        "12_FIGURES": [f"figures/{p.name}" for p in sorted((out_dir / "figures").glob("*.svg"))],
        "13_FAILURE_TAXONOMY": ["v20_failure_taxonomy.csv", "v20_route_decision.json"],
        "14_REPRO_COMMANDS": ["v20_command_journal.csv"],
        "15_GPU_QUEUE": ["v20_runnable_queue.csv", "v20_gpu_assignment_manifest.csv", "v20_gpu_utilization_dashboard.csv", "v20_idle_violation.csv", "v20_deferred_items.csv", "v20_gpu_queue_drain_report.csv"],
    }
    for section, names in section_map.items():
        for name in names:
            copy_if_exists(out_dir / name, packet / section / Path(name).name)
    copy_if_exists(V20_EXEC_DOC, packet / "14_REPRO_COMMANDS" / V20_EXEC_DOC.name)
    write_text(packet / "00_README.md", "\n".join(["# v20 Code Review Packet", f"- generated_at: {now_sg()}", f"- result_dir: {out_dir}", "- evidence-first: missing data remains blocked, not promoted."]) + "\n")
    rows = []
    for path in sorted(p for p in packet.rglob("*") if p.is_file()):
        rel = path.relative_to(packet)
        if rel.name in {"packet_manifest.csv", "packet_sha256_manifest.csv"}:
            continue
        rows.append({"relative_path": str(rel), "sha256": sha256_file(path), "bytes": path.stat().st_size, "artifact_type": rel.parts[0], "required": 1})
    write_rows(packet / "packet_manifest.csv", rows)
    write_rows(packet / "packet_sha256_manifest.csv", [{"relative_path": r["relative_path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in rows])
    with zipfile.ZipFile(out_dir / "v20_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in packet.rglob("*") if p.is_file()):
            zf.write(path, Path("v20_code_review_packet") / path.relative_to(packet))
    with zipfile.ZipFile(out_dir / "v20_results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and "v20_code_review_packet/" not in str(p.relative_to(out_dir)) and p.name != "v20_results_bundle.zip"):
            zf.write(path, path.relative_to(out_dir))
