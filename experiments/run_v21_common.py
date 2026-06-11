"""Shared helpers for v21 execution artifacts."""

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
V21_ROOT = ROOT / "results/v21_0_source_retention_kernel_officialization_4gpu"
V21_OFFICIAL = V21_ROOT / "official_v21"
V20_OFFICIAL = ROOT / "results/v20_0_source_channel_fu_basis_kernel_officialization_4gpu/official_v20"
V21_EXEC_DOC = ROOT / "docs/DG-KAN_v21.0_SourceRetention_KernelOfficialization_4GPU_执行日志.md"
V21_RECAP_DOC = ROOT / "docs/DG-KAN_v21.0_SourceRetention_KernelOfficialization_4GPU_实验结果复盘.md"
V21_PLAN_DOC = ROOT / "docs/DG-KAN_v21.0_SourceRetention_KernelOfficialization_4GPU_完整计划.md"


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else V21_OFFICIAL
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
    journal_path = out_dir / "v21_command_journal.csv"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    exists = journal_path.exists() and journal_path.stat().st_size > 0
    with journal_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "command", "status", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow({k: str(v) for k, v in row.items()})
    append_text(V21_EXEC_DOC, f"\n## {row['timestamp']}\n\n```bash\n{command}\n```\n\n- status: {status or 'recorded'}\n")
    if note:
        append_text(V21_EXEC_DOC, f"- note: {note}\n")


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


V21_EFFICIENCY_VARIANTS = {
    "D-CHE": [
        "CHE21-R4-k3-gradbuf-triton-official",
        "CHE21-R2-low-degree-k3-official",
        "CHE21-R4-k5-gradbuf-triton-ablation",
    ],
    "D-FOU": [
        "FOU21-R4-k4-triton-no-materialize-official",
        "FOU21-R2-lowfreq-k2-stream-official",
        "FOU21-R3-tablelookup-bandreadout-official",
    ],
    "LQ": ["LQ21-recurrence-fixed-frame-smoke"],
    "D-RAT": ["RAT21-branchless-horner-smoke"],
    "D-RBF": ["RBF21-compact-local-k4-smoke"],
    "D-WAV": ["WAV21-sparse-support-smoke"],
}


def to_v19_repair_variant(family: str, v21_variant: str) -> str:
    mapping = {
        "CHE21-R4-k3-gradbuf-triton-official": "CHE-R4-k3-gradbuf-triton",
        "CHE21-R2-low-degree-k3-official": "CHE-R2-low-degree-k3-triton",
        "CHE21-R4-k5-gradbuf-triton-ablation": "CHE-R4-k3-gradbuf-triton",
        "FOU21-R4-k4-triton-no-materialize-official": "FOU-R4-k4-triton-no-materialize",
        "FOU21-R2-lowfreq-k2-stream-official": "FOU-R2-low-frequency-k2-stream",
        "FOU21-R3-tablelookup-bandreadout-official": "FOU-R2-low-frequency-k2-stream",
        "RBF21-compact-local-k4-smoke": "RBF22.03-R1-compact-local-k4-no-dense",
    }
    return mapping.get(str(v21_variant), "R0-current")


def v21_functional_specs(scope: str) -> list[dict[str, Any]]:
    if scope == "mlp_source":
        return [
            {"v21_id": "MLP-F21-adamw-primary-fu-residual", "continuation_id": "M1-adamw-primary-fu0p0001", "mechanism": "M1-AdamWPrimaryFUResidual", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "adamw_primary_fu_residual_optimizer_decoupling"},
            {"v21_id": "MLP-F1-M2-strong-source", "continuation_id": "M2-fu0p0001", "mechanism": "M2-SGDMomentumPrimaryFU", "fu_lr": 0.0001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "m2_strong_early_source"},
            {"v21_id": "MLP-F2-M15-weak-stable", "continuation_id": "M15-linec-alt50-fu0p0005", "mechanism": "M15-LineCFilteredAlternatingFU", "fu_lr": 0.0005, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "linec_filtered_weak_stable"},
            {"v21_id": "MLP-F3-F9-hold", "continuation_id": "M42-momwarm800-hold", "mechanism": "M42-MomentumWarmHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "post_warmup_hold"},
            {"v21_id": "MLP-F4-F10-cycle-hold", "continuation_id": "M43-momcycle800-hold", "mechanism": "M43-MomentumCycleHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "cycle_hold"},
            {"v21_id": "MLP-F5-M2-plus-M15-anchor", "continuation_id": "M44-m2-linec-anchor-slow", "mechanism": "M44-MomentumLineCAnchorSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "m2_plus_linec_anchor"},
            {"v21_id": "MLP-F6-M2-source-with-slow-anchor", "continuation_id": "M45-m2-slow-anchor", "mechanism": "M45-MomentumSlowAnchorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "m2_slow_anchor"},
            {"v21_id": "MLP-F7-M2-source-with-matrix-block-retention", "continuation_id": "M46-m2-matrix-block-retention", "mechanism": "M46-MomentumMatrixBlockRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "m2_matrix_block_retention"},
            {"v21_id": "MLP-F8-cycle-hold-to-M15-anchor", "continuation_id": "M47-cycle800-linec-anchor", "mechanism": "M47-MomentumCycleThenLineCAnchorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "cycle_hold_then_linec_anchor_bridge"},
            {"v21_id": "MLP-F9-cycle1200-to-M15-anchor", "continuation_id": "M47-cycle1200-linec-anchor", "mechanism": "M47-MomentumCycleThenLineCAnchorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200, "hypothesis": "long_cycle_hold_then_linec_anchor_bridge"},
            {"v21_id": "MLP-F10-dual-timescale-retention-warm800", "continuation_id": "M48-dual-timescale-warm800", "mechanism": "M48-DualTimescaleSourceRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "dual_timescale_short_long_source_retention"},
            {"v21_id": "MLP-F11-dual-timescale-retention-warm1200", "continuation_id": "M48-dual-timescale-warm1200", "mechanism": "M48-DualTimescaleSourceRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1200, "hypothesis": "dual_timescale_long_warm_source_retention"},
            {"v21_id": "MLP-F20-margin-split-consensus-source", "continuation_id": "M68-margin-split-consensus-alt50-fu0p0001", "mechanism": "M68-MarginSplitConsensusSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "top_wrong_margin_split_consensus_source_state"},
            {"v21_id": "MLP-F22-rotated-cautious-matrix-source", "continuation_id": "M69-rotated-cautious-matrix-alt50-fu0p0001", "mechanism": "M69-RotatedCautiousMatrixSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "soap_like_rotated_cautious_matrix_source_state"},
            {"v21_id": "MLP-F23-source-vs-sgd-lookahead-gate", "continuation_id": "M70-train-lookahead-cautious-alt50-fu0p0001", "mechanism": "M70-TrainLookaheadCautiousSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "train_only_source_vs_gradient_lookahead_selector"},
            {"v21_id": "MLP-F40-adamw-boundary-to-momentum-source", "continuation_id": "M87-adamw400-to-m2-alt50-fu0p0001", "mechanism": "M87-AdamWBoundaryThenMomentumSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "phase_reset_adamw_h100_h400_boundary_then_momentum_source"},
            {"v21_id": "MLP-F41-adamw-boundary-to-dual-timescale-source", "continuation_id": "M88-adamw400-to-dual-timescale-alt50-fu0p0001", "mechanism": "M88-AdamWBoundaryThenDualTimescaleSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "phase_reset_adamw_boundary_then_dual_timescale_source_retention"},
            {"v21_id": "MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source", "continuation_id": "M89-adamw400-dual-timescale-sgd-floor-alt50-fu0p0001", "mechanism": "M89-AdamWBoundaryDualTimescaleSGDFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "phase_reset_dual_timescale_source_with_sgd_floor_retention_repair"},
            {"v21_id": "MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source", "continuation_id": "M90-adamw400-dual-timescale-late-sgd-floor-alt50-fu0p0001", "mechanism": "M90-AdamWBoundaryDualTimescaleLateSGDFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "phase_reset_dual_timescale_source_with_h3200_late_sgd_floor_retention_repair"},
            {"v21_id": "MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source", "continuation_id": "M91-adamw400-dual-timescale-tiny-late-sgd-floor-alt50-fu0p0001", "mechanism": "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "phase_reset_dual_timescale_source_with_tiny_late_sgd_floor_retention_repair"},
            {"v21_id": "MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source", "continuation_id": "M92-adamw400-dual-timescale-reinforced-tiny-late-sgd-floor-alt50-fu0p0001", "mechanism": "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "phase_reset_reinforced_dual_timescale_source_with_tiny_late_sgd_floor_retention_repair"},
            {"v21_id": "MLP-F47-trainloss-gated-dual-timescale-tiny-late-source", "continuation_id": "M93-trainloss-gated-dual-timescale-tiny-late-alt50-fu0p0001", "mechanism": "M93-TrainLossGatedDualTimescaleTinyLateFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f46_train_loss_selector_online_gated_late_retention_repair"},
            {"v21_id": "MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source", "continuation_id": "M94-trainloss-gated-dual-timescale-hold-fallback-alt50-fu0p0001", "mechanism": "M94-TrainLossGatedDualTimescaleHoldFallbackFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f47_reject_rows_hold_fallback_repair"},
            {"v21_id": "MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source", "continuation_id": "M95-trainloss-gated-dual-timescale-tiny-late-fallback-alt50-fu0p0001", "mechanism": "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f48_reject_rows_hold_then_tiny_late_floor_repair"},
            {"v21_id": "MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source", "continuation_id": "M96-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-alt50-fu0p0001", "mechanism": "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f49_reject_rows_boosted_tiny_late_floor_h4800_repair"},
            {"v21_id": "MLP-F51-trainloss-late-hold-recovery-source", "continuation_id": "M97-trainloss-late-hold-recovery-alt50-fu0p0001", "mechanism": "M97-TrainLossLateHoldRecoveryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f49_h3200_source_preserving_late_hold_recovery"},
            {"v21_id": "MLP-F52-trainloss-late-lookahead-floor-source", "continuation_id": "M98-trainloss-late-lookahead-floor-alt50-fu0p0001", "mechanism": "M98-TrainLossLateLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f51_train_split_late_micro_floor_precommit_gate"},
            {"v21_id": "MLP-F53-trainloss-terminal-lookahead-floor-source", "continuation_id": "M99-trainloss-terminal-lookahead-floor-alt50-fu0p0001", "mechanism": "M99-TrainLossTerminalLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f52_terminal_train_split_catchup_after_source_hold"},
            {"v21_id": "MLP-F54-trainloss-early-terminal-lookahead-floor-source", "continuation_id": "M100-trainloss-early-terminal-lookahead-floor-alt50-fu0p0001", "mechanism": "M100-TrainLossEarlyTerminalLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f53_early_terminal_train_split_catchup_after_source_hold"},
            {"v21_id": "MLP-F55-split-fisher-source-observability-reset", "continuation_id": "M37-split-fisher-alt50-fu0p0001", "mechanism": "M37-SplitFisherAgreementSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "c4_split_fisher_source_observability_reset"},
            {"v21_id": "MLP-F56-momentum-warm-split-fisher-source", "continuation_id": "M39-momwarm400-split-fisher-alt50-fu0p0001", "mechanism": "M39-MomentumWarmSplitFisherFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "c4_momentum_warm_split_fisher_source_observability_reset"},
            {"v21_id": "MLP-F57-adamw-boundary-dual-timescale-antiwashout-source", "continuation_id": "M101-adamw400-dual-timescale-antiwashout-alt50-fu0p0001", "mechanism": "M101-AdamWBoundaryDualTimescaleAntiWashoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "source_theory_reset_antiwashout_projection_after_dual_timescale_source"},
            {"v21_id": "MLP-F58-adamw-boundary-dual-timescale-source-anchor", "continuation_id": "M102-adamw400-dual-timescale-anchor-alt50-fu0p0001", "mechanism": "M102-AdamWBoundaryDualTimescaleSourceAnchorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "source_theory_reset_anchor_residual_after_dual_timescale_source"},
            {"v21_id": "MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry", "continuation_id": "M103-adamw400-dual-timescale-param-ema-reentry-alt50-fu0p0001", "mechanism": "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "source_theory_reset_schedule_free_parameter_ema_reentry"},
            {"v21_id": "MLP-F60-adamw-boundary-dual-timescale-readout-channel", "continuation_id": "M104-adamw400-dual-timescale-readout-channel-alt50-fu0p0001", "mechanism": "M104-AdamWBoundaryDualTimescaleReadoutChannelFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "source_theory_reset_readout_channel_retained_target"},
            {"v21_id": "MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel", "continuation_id": "M105-adamw400-dual-timescale-hidden-matrix-channel-alt50-fu0p0001", "mechanism": "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "source_theory_reset_hidden_matrix_channel_retained_target"},
            {"v21_id": "MLP-F62-trainloss-terminal-projected-lookahead-floor-source", "continuation_id": "M106-terminal-projected-lookahead-floor-alt50-fu0p0001", "mechanism": "M106-TrainLossTerminalProjectedLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "source_preserving_terminal_recovery_projected_gradient_floor"},
            {"v21_id": "MLP-F63-trainloss-terminal-consensus-lookahead-floor-source", "continuation_id": "M107-terminal-consensus-lookahead-floor-alt50-fu0p0001", "mechanism": "M107-TrainLossTerminalConsensusLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "c4_cross_split_noise_reservoir_terminal_recovery"},
            {"v21_id": "MLP-F64-trainloss-terminal-selector-lookahead-floor-source", "continuation_id": "M108-terminal-selector-lookahead-floor-alt50-fu0p0001", "mechanism": "M108-TrainLossTerminalSelectorLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "c4_train_only_terminal_direction_selector_recovery"},
            {"v21_id": "MLP-F70-trainloss-terminal-positive-lookahead-floor-source", "continuation_id": "M111-terminal-positive-lookahead-floor-alt50-fu0p0001", "mechanism": "M111-TrainLossTerminalPositiveLookaheadFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "f53_terminal_positive_only_lookahead_gate_repair"},
            {"v21_id": "MLP-F71-trainloss-terminal-h3200-checkpoint-reentry", "continuation_id": "M112-terminal-h3200-checkpoint-reentry-alt50-fu0p0001", "mechanism": "M112-TrainLossTerminalCheckpointReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "h3200_source_checkpoint_as_terminal_retained_target"},
            {"v21_id": "MLP-F72-trainloss-terminal-hard-split-source", "continuation_id": "M113-terminal-hard-split-source-alt50-fu0p0001", "mechanism": "M113-TrainLossTerminalHardSplitSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "hard_example_split_consensus_terminal_source_theory_reset"},
            {"v21_id": "MLP-F73-trainloss-terminal-adamw-lookahead", "continuation_id": "M114-terminal-adamw-lookahead-alt50-fu0p0001", "mechanism": "M114-TrainLossTerminalAdamWLookaheadFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "terminal_adamw_carrier_lookahead_after_source_hold"},
            {"v21_id": "MLP-F74-trainloss-terminal-optimizer-selector", "continuation_id": "M115-terminal-optimizer-selector-alt50-fu0p0001", "mechanism": "M115-TrainLossTerminalOptimizerSelectorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "train_only_terminal_raw_vs_adamw_carrier_selector"},
            {"v21_id": "MLP-F65-adamw-split-fisher-residual-source", "continuation_id": "M38-adamw-split-fisher-residual-alt50-fu0p0001", "mechanism": "M38-AdamWSplitFisherAgreementResidualFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "c4_adamw_split_fisher_source_observability_reset"},
            {"v21_id": "MLP-F75-dataset-invariant-poprisk-slow-source", "continuation_id": "M116-dataset-invariant-poprisk-alt50-fu0p0001", "mechanism": "M116-DatasetInvariantPopRiskSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2201_dataset_heterogeneity_train_only_split_poprisk_invariant_selector"},
            {"v21_id": "MLP-F76-dataset-invariant-readout-consensus-source", "continuation_id": "M117-dataset-invariant-readout-consensus-alt50-fu0p0001", "mechanism": "M117-DatasetInvariantReadoutConsensusFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2201_dataset_heterogeneity_train_split_readout_consensus_corrupt_filtered_selector"},
            {"v21_id": "MLP-F77-source-conserving-optimizer-only", "continuation_id": "M118-source-conserving-optimizer-alt50-fu0p0001", "mechanism": "M118-SourceConservingOptimizerOnlyFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_optimizer_washout_source_conserving_no_sgd_fallback"},
            {"v21_id": "MLP-F78-terminal-source-conserving-route", "continuation_id": "M119-terminal-source-conserving-route-alt50-fu0p0001", "mechanism": "M119-TerminalSourceConservingRouteFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_retained_target_terminal_source_conserving_projected_consensus_route"},
            {"v21_id": "MLP-F79-trainloss-risk-profile-route", "continuation_id": "M120-trainloss-risk-profile-route-alt50-fu0p0001", "mechanism": "M120-TrainLossRiskProfileRouteFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_dataset_heterogeneity_train_loss_risk_profile_terminal_route"},
            {"v21_id": "MLP-F80-debt-aware-source-gate", "continuation_id": "M121-debt-aware-source-gate-alt50-fu0p0001", "mechanism": "M121-DebtAwareSourceGateFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_debt_collapse_train_only_debt_aware_terminal_source_gate"},
            {"v21_id": "MLP-F81-ungated-warm-terminal-source-route", "continuation_id": "M122-ungated-warm-terminal-source-route-alt50-fu0p0001", "mechanism": "M122-UngatedWarmTerminalSourceRouteFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_ungated_h3200_source_warm_then_terminal_source_route"},
            {"v21_id": "MLP-F82-ungated-warm-risk-profile-route", "continuation_id": "M123-ungated-warm-risk-profile-route-alt50-fu0p0001", "mechanism": "M123-UngatedWarmRiskProfileRouteFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_ungated_h3200_source_warm_then_train_loss_risk_profile_route"},
            {"v21_id": "MLP-F83-ungated-warm-debt-raw-bailout", "continuation_id": "M124-ungated-warm-debt-raw-bailout-alt50-fu0p0001", "mechanism": "M124-UngatedWarmDebtRawBailoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_ungated_h3200_source_warm_then_debt_aware_raw_terminal_bailout"},
            {"v21_id": "MLP-F84-h2400-terminal-hold-source", "continuation_id": "M125-h2400-terminal-hold-alt50-fu0p0001", "mechanism": "M125-TrainLossH2400CheckpointHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_terminal_collapse_h2400_hold_without_future_leakage"},
            {"v21_id": "MLP-F85-h2800-terminal-hold-source", "continuation_id": "M126-h2800-terminal-hold-alt50-fu0p0001", "mechanism": "M126-TrainLossH2800CheckpointHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_terminal_collapse_h2800_hold_without_future_leakage"},
            {"v21_id": "MLP-F86-h2400-debt-bailout-source", "continuation_id": "M127-h2400-debt-bailout-alt50-fu0p0001", "mechanism": "M127-TrainLossH2400DebtBailoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_terminal_collapse_h2400_hold_train_only_debt_bailout"},
            {"v21_id": "MLP-F87-terminal-raw-then-source-guard", "continuation_id": "M128-terminal-raw-then-source-guard-alt50-fu0p0001", "mechanism": "M128-TrainLossTerminalRawThenSourceGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2201_terminal_collapse_raw_catchup_then_final_source_guard"},
            {"v21_id": "MLP-F88-h800-source-slow-ema-retention", "continuation_id": "M129-h800-source-slow-ema-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_h800_to_h3200_source_slow_ema_retention"},
            {"v21_id": "MLP-F89-h800-readout-channel-retention", "continuation_id": "M130-h800-readout-channel-alt50-fu0p0001", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_h800_readout_channel_retained_target_source_observability"},
            {"v21_id": "MLP-F90-h800-dual-memory-source-retention", "continuation_id": "M131-h800-dual-memory-alt50-fu0p0001", "mechanism": "M131-H800DualMemorySourceRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_h800_short_long_dual_memory_source_retention"},
            {"v21_id": "MLP-F91-h800-readout-channel-alt25-retention", "continuation_id": "M130-h800-readout-channel-alt25-fu0p0001", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.0001, "alt_period": 25, "source_warmup_steps": 400, "hypothesis": "v2202_readout_channel_more_frequent_source_observability_retention"},
            {"v21_id": "MLP-F92-h800-readout-channel-strong-retention", "continuation_id": "M130-h800-readout-channel-alt50-fu0p00015", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.00015, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_readout_channel_stronger_source_observability_retention"},
            {"v21_id": "MLP-F93-h800-readout-channel-alt25-strong-retention", "continuation_id": "M130-h800-readout-channel-alt25-fu0p00015", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.00015, "alt_period": 25, "source_warmup_steps": 400, "hypothesis": "v2202_readout_channel_frequency_plus_strength_retention"},
            {"v21_id": "MLP-F94-h1600-source-checkpoint-reentry", "continuation_id": "M132-h1600-source-checkpoint-reentry-alt50-fu0p0001", "mechanism": "M132-H1600SourceCheckpointReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_h1600_train_time_source_checkpoint_reentry"},
            {"v21_id": "MLP-F95-h2400-source-checkpoint-reentry", "continuation_id": "M133-h2400-source-checkpoint-reentry-alt50-fu0p0001", "mechanism": "M133-H2400SourceCheckpointReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_h2400_train_time_source_checkpoint_reentry"},
            {"v21_id": "MLP-F106-early100-h800-source-slow-ema-retention", "continuation_id": "M129-early100-h800-source-slow-ema-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_early_h400_source_state_anchor_h800_slow_ema"},
            {"v21_id": "MLP-F107-early100-h800-readout-channel-retention", "continuation_id": "M130-early100-h800-readout-channel-alt50-fu0p0001", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_early_h400_readout_channel_source_state_anchor"},
            {"v21_id": "MLP-F108-early100-h800-readout-channel-alt25-strong-retention", "continuation_id": "M130-early100-h800-readout-channel-alt25-fu0p00015", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.00015, "alt_period": 25, "source_warmup_steps": 100, "hypothesis": "v2202_early_h400_readout_channel_frequency_strength_anchor"},
            {"v21_id": "MLP-F109-early100-h1600-source-checkpoint-reentry", "continuation_id": "M132-early100-h1600-source-checkpoint-reentry-alt50-fu0p0001", "mechanism": "M132-H1600SourceCheckpointReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_early_h400_h1600_source_checkpoint_reentry"},
            {"v21_id": "MLP-F110-early1-h800-source-slow-ema-retention", "continuation_id": "M129-early1-h800-source-slow-ema-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1, "hypothesis": "v2202_h100_source_state_anchor_from_step1_slow_ema"},
            {"v21_id": "MLP-F111-early1-h800-readout-channel-retention", "continuation_id": "M130-early1-h800-readout-channel-alt50-fu0p0001", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 1, "hypothesis": "v2202_h100_readout_channel_anchor_from_step1"},
            {"v21_id": "MLP-F112-early50-h800-source-slow-ema-retention", "continuation_id": "M129-early50-h800-source-slow-ema-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 50, "hypothesis": "v2202_h100_source_state_anchor_from_step50_slow_ema"},
            {"v21_id": "MLP-F113-early50-h800-readout-channel-retention", "continuation_id": "M130-early50-h800-readout-channel-alt50-fu0p0001", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 50, "hypothesis": "v2202_h100_readout_channel_anchor_from_step50"},
            {"v21_id": "MLP-F114-early100-h800-source-slow-ema-alt25-retention", "continuation_id": "M129-early100-h800-source-slow-ema-alt25-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 25, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_repair_early100_slow_ema_alt25"},
            {"v21_id": "MLP-F115-early100-h800-source-slow-ema-strong-retention", "continuation_id": "M129-early100-h800-source-slow-ema-alt50-fu0p00015", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.00015, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_repair_early100_slow_ema_strong"},
            {"v21_id": "MLP-F116-early100-h800-source-slow-ema-alt25-strong-retention", "continuation_id": "M129-early100-h800-source-slow-ema-alt25-fu0p00015", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.00015, "alt_period": 25, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_repair_early100_slow_ema_alt25_strong"},
            {"v21_id": "MLP-F117-early100-h800-source-slow-ema-terminal-guard", "continuation_id": "M139-early100-h800-source-slow-ema-terminal-guard-alt50-fu0p0001", "mechanism": "M139-EarlySourceSlowEMATerminalGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_repair_early100_slow_ema_terminal_source_guard"},
            {"v21_id": "MLP-F118-early100-h800-source-slow-ema-terminal-raw-guard", "continuation_id": "M140-early100-h800-source-slow-ema-terminal-raw-guard-alt50-fu0p0001", "mechanism": "M140-EarlySourceSlowEMATerminalRawGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_repair_early100_slow_ema_terminal_raw_then_source_guard"},
            {"v21_id": "MLP-F119-early100-h800-source-slow-ema-terminal-raw-guard-strong", "continuation_id": "M141-early100-h800-source-slow-ema-terminal-raw-guard-strong-alt50-fu0p0001", "mechanism": "M141-EarlySourceSlowEMATerminalRawGuardStrongFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_repair_early100_slow_ema_stronger_terminal_raw_then_source_guard"},
            {"v21_id": "MLP-F120-early100-h800-source-slow-ema-terminal-projected-optimizer", "continuation_id": "M142-early100-h800-source-slow-ema-terminal-projected-optimizer-alt50-fu0p0001", "mechanism": "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_optimizer_projection_terminal_current_gradient_antiwashout"},
            {"v21_id": "MLP-F121-early100-h800-source-slow-ema-terminal-projected-blend", "continuation_id": "M143-early100-h800-source-slow-ema-terminal-projected-blend-alt50-fu0p0001", "mechanism": "M143-EarlySourceSlowEMATerminalProjectedBlendFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_optimizer_projection_terminal_current_gradient_plus_slow_source_blend"},
            {"v21_id": "MLP-F122-early100-h800-source-slow-ema-terminal-antiwashout", "continuation_id": "M144-early100-h800-source-slow-ema-terminal-antiwashout-alt50-fu0p0001", "mechanism": "M144-EarlySourceSlowEMATerminalAntiWashoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_optimizer_source_conflict_terminal_antiwashout_projector"},
            {"v21_id": "MLP-F123-early100-h800-source-slow-ema-terminal-h4000-reentry", "continuation_id": "M145-early100-h800-source-slow-ema-terminal-h4000-reentry-alt50-fu0p0001", "mechanism": "M145-EarlySourceSlowEMATerminalH4000ReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_h4000_source_state_terminal_checkpoint_reentry_after_raw_guard"},
            {"v21_id": "MLP-F124-early100-h800-source-slow-ema-signal-reservoir-target", "continuation_id": "M146-early100-h800-source-slow-ema-signal-reservoir-target-alt50-fu0p0001", "mechanism": "M146-EarlySourceSlowEMASignalReservoirTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_signal_reservoir_target_gate_after_source_state"},
            {"v21_id": "MLP-F125-early100-h800-source-slow-ema-source-bank-target", "continuation_id": "M147-early100-h800-source-slow-ema-source-bank-target-alt50-fu0p0001", "mechanism": "M147-EarlySourceSlowEMASourceBankTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_source_bank_target_gate_after_source_state"},
            {"v21_id": "MLP-F126-early100-h800-source-slow-ema-dual-target-guard", "continuation_id": "M148-early100-h800-source-slow-ema-dual-target-guard-alt50-fu0p0001", "mechanism": "M148-EarlySourceSlowEMADualTargetGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_dual_signal_reservoir_source_bank_target_gate_after_source_state"},
            {"v21_id": "MLP-F127-early100-h800-source-slow-ema-terminal-adaptive-raw", "continuation_id": "M149-early100-h800-source-slow-ema-terminal-adaptive-raw-alt50-fu0p0001", "mechanism": "M149-EarlySourceSlowEMATerminalAdaptiveRawFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_source_state_adaptive_raw_terminal_guard"},
            {"v21_id": "MLP-F128-early100-h800-source-slow-ema-terminal-sparse-source", "continuation_id": "M150-early100-h800-source-slow-ema-terminal-sparse-source-alt50-fu0p0001", "mechanism": "M150-EarlySourceSlowEMATerminalSparseSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_sparse_source_axis_terminal_retention_guard"},
            {"v21_id": "MLP-F129-early100-h800-source-slow-ema-terminal-ratio-preserve", "continuation_id": "M151-early100-h800-source-slow-ema-terminal-ratio-preserve-alt50-fu0p0001", "mechanism": "M151-EarlySourceSlowEMATerminalRatioPreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_h4800_ratio_preserving_terminal_source_guard"},
            {"v21_id": "MLP-F130-early100-h800-source-slow-ema-terminal-source-projection-target", "continuation_id": "M152-early100-h800-source-slow-ema-terminal-source-projection-target-alt50-fu0p0001", "mechanism": "M152-SourceProjectionB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_source_projection_target_with_terminal_raw_guard"},
            {"v21_id": "MLP-F131-early100-h800-source-slow-ema-terminal-noise-orthogonal-target", "continuation_id": "M153-early100-h800-source-slow-ema-terminal-noise-orthogonal-target-alt50-fu0p0001", "mechanism": "M153-NoiseOrthogonalB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_noise_orthogonal_target_with_terminal_raw_guard"},
            {"v21_id": "MLP-F132-early100-h800-source-slow-ema-terminal-easy-margin-target", "continuation_id": "M154-early100-h800-source-slow-ema-terminal-easy-margin-target-alt50-fu0p0001", "mechanism": "M154-EasyMarginB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_easy_margin_target_with_terminal_raw_guard"},
            {"v21_id": "MLP-F133-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue", "continuation_id": "M158-early100-h800-source-slow-ema-terminal-reject-source-axis-rescue-alt50-fu0p0001", "mechanism": "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_terminal_reject_source_axis_micro_rescue"},
            {"v21_id": "MLP-F134-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue", "continuation_id": "M159-early100-h800-source-slow-ema-terminal-reject-slow-ema-rescue-alt50-fu0p0001", "mechanism": "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_terminal_reject_slow_ema_micro_rescue"},
            {"v21_id": "MLP-F135-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue", "continuation_id": "M160-early100-h800-source-slow-ema-terminal-reject-hold-source-rescue-alt50-fu0p0001", "mechanism": "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_terminal_reject_hold_then_source_rescue"},
            {"v21_id": "MLP-F136-early100-h800-source-slow-ema-shape-preserve", "continuation_id": "M161-early100-h800-source-slow-ema-shape-preserve-alt50-fu0p0001", "mechanism": "M161-EarlySourceSlowEMAShapePreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_midlate_train_only_source_shape_preservation_without_terminal_raw"},
            {"v21_id": "MLP-F137-early100-h800-source-slow-ema-shape-preserve-raw-guard", "continuation_id": "M162-early100-h800-source-slow-ema-shape-preserve-raw-guard-alt50-fu0p0001", "mechanism": "M162-EarlySourceSlowEMAShapePreserveRawGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_midlate_source_shape_preservation_wrapped_around_f118_raw_guard"},
            {"v21_id": "MLP-F138-early100-h800-source-slow-ema-shape-preserve-clamp", "continuation_id": "M163-early100-h800-source-slow-ema-shape-preserve-clamp-alt50-fu0p0001", "mechanism": "M163-EarlySourceSlowEMAShapePreserveClampFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_midlate_source_shape_preservation_with_h4400_source_axis_clamp"},
            {"v21_id": "MLP-F139-early100-h800-source-slow-ema-terminal-reject-raw-rescue", "continuation_id": "M164-early100-h800-source-slow-ema-terminal-reject-raw-rescue-alt50-fu0p0001", "mechanism": "M164-EarlySourceSlowEMATerminalRejectRawRescueFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_terminal_reject_raw_rescue_for_weak_source_rows"},
            {"v21_id": "MLP-F140-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue", "continuation_id": "M165-early100-h800-source-slow-ema-terminal-reject-hybrid-rescue-alt50-fu0p0001", "mechanism": "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_terminal_reject_raw_source_hybrid_rescue_for_weak_rows"},
            {"v21_id": "MLP-F141-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue", "continuation_id": "M166-early100-h800-source-slow-ema-terminal-reject-lateonly-raw-rescue-alt50-fu0p0001", "mechanism": "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_late_only_terminal_reject_raw_rescue_no_pre4400_raw"},
            {"v21_id": "MLP-F142-early100-h800-source-slow-ema-terminal-topk-support-guard", "continuation_id": "M167-early100-h800-source-slow-ema-terminal-topk-support-guard-alt50-fu0p0001", "mechanism": "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_topk_source_support_terminal_guard"},
            {"v21_id": "MLP-F143-early100-h800-source-slow-ema-terminal-topk-raw-guard", "continuation_id": "M168-early100-h800-source-slow-ema-terminal-topk-raw-guard-alt50-fu0p0001", "mechanism": "M168-EarlySourceSlowEMATerminalTopKRawGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_topk_source_support_with_raw_terminal_guard"},
            {"v21_id": "MLP-F144-early100-h800-source-slow-ema-terminal-topk-debt-cap", "continuation_id": "M169-early100-h800-source-slow-ema-terminal-topk-debt-cap-alt50-fu0p0001", "mechanism": "M169-EarlySourceSlowEMATerminalTopKDebtCapFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_train_only_topk_source_support_debt_capped_terminal_guard"},
            {"v21_id": "MLP-F145-early100-h800-source-slow-ema-terminal-source-preserve-strong", "continuation_id": "M170-early100-h800-source-slow-ema-terminal-source-preserve-strong-alt50-fu0p0001", "mechanism": "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_train_only_terminal_source_preserve_strong_no_raw"},
            {"v21_id": "MLP-F146-early100-h800-source-slow-ema-terminal-source-preserve-gentle", "continuation_id": "M171-early100-h800-source-slow-ema-terminal-source-preserve-gentle-alt50-fu0p0001", "mechanism": "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_train_only_terminal_source_preserve_gentle_no_raw"},
            {"v21_id": "MLP-F147-early100-h800-source-slow-ema-terminal-info-volume-guard", "continuation_id": "M172-early100-h800-source-slow-ema-terminal-info-volume-guard-alt50-fu0p0001", "mechanism": "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_train_only_terminal_source_channel_info_volume_guard"},
            {"v21_id": "MLP-F148-early100-h800-source-slow-ema-low-nds-diffeomorphic-target", "continuation_id": "M173-early100-h800-source-slow-ema-low-nds-diffeomorphic-target-alt50-fu0p0001", "mechanism": "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c2_low_nds_diffeomorphic_target_after_f145_f147_no_h4800"},
            {"v21_id": "MLP-F149-early100-h800-source-slow-ema-info-volume-diffeomorphic-target", "continuation_id": "M174-early100-h800-source-slow-ema-info-volume-diffeomorphic-target-alt50-fu0p0001", "mechanism": "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c2_info_volume_diffeomorphic_target_after_f145_f147_no_h4800"},
            {"v21_id": "MLP-F150-early100-h800-source-slow-ema-low-rank-readout-transport", "continuation_id": "M175-early100-h800-source-slow-ema-low-rank-readout-transport-alt50-fu0p0001", "mechanism": "M175-EarlySourceSlowEMALowRankReadoutTransportFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c2_low_rank_readout_transport_after_f145_f147_no_h4800"},
            {"v21_id": "MLP-F151-early100-h800-source-slow-ema-terminal-source-preserve-very-strong", "continuation_id": "M179-early100-h800-source-slow-ema-terminal-source-preserve-very-strong-alt50-fu0p0001", "mechanism": "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_fallback_increase_source_preservation_strength_no_h800_amplitude_boost"},
            {"v21_id": "MLP-F152-early100-h800-source-slow-ema-h3600-terminal-source-preserve", "continuation_id": "M180-early100-h800-source-slow-ema-h3600-terminal-source-preserve-alt50-fu0p0001", "mechanism": "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_fallback_attach_source_preservation_at_h3600_terminal_phase"},
            {"v21_id": "MLP-F153-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source", "continuation_id": "M181-early100-h800-source-slow-ema-terminal-nora-row-orthogonal-source-alt50-fu0p0001", "mechanism": "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_fallback_nora_style_row_orthogonal_source_update_for_terminal_jitter"},
            {"v21_id": "MLP-F154-early100-h800-source-slow-ema-terminal-debt-aware-preserve", "continuation_id": "M182-early100-h800-source-slow-ema-terminal-debt-aware-preserve-alt50-fu0p0001", "mechanism": "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_fallback_debt_aware_terminal_preservation_when_train_loss_debt_persists"},
            {"v21_id": "MLP-F155-early100-h800-source-slow-ema-terminal-low-nds-matrix-block", "continuation_id": "M183-early100-h800-source-slow-ema-terminal-low-nds-matrix-block-alt50-fu0p0001", "mechanism": "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_fallback_low_nds_hidden_readout_matrix_block_source_preserve"},
            {"v21_id": "MLP-F156-early100-h800-source-slow-ema-terminal-dual-memory-preserve", "continuation_id": "M184-early100-h800-source-slow-ema-terminal-dual-memory-preserve-alt50-fu0p0001", "mechanism": "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_c1_fallback_long_memory_terminal_source_preservation"},
            {"v21_id": "MLP-F157-early100-h800-source-slow-ema-terminal-snr-predictor", "continuation_id": "M185-early100-h800-source-slow-ema-terminal-snr-predictor-alt50-fu0p0001", "mechanism": "M185-EarlySourceSlowEMASNRTerminalPredictorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_signal_estimator_poprisk_snr_as_terminal_retention_predictor_not_parameter_mask"},
            {"v21_id": "MLP-F158-early100-h800-source-slow-ema-split-consensus-estimator", "continuation_id": "M186-early100-h800-source-slow-ema-split-consensus-estimator-alt50-fu0p0001", "mechanism": "M186-EarlySourceSlowEMASplitConsensusEstimatorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_signal_estimator_split_consensus_as_train_only_terminal_gate_not_h4800_selector"},
            {"v21_id": "MLP-F159-early100-h800-source-slow-ema-signal-reservoir-transport", "continuation_id": "M187-early100-h800-source-slow-ema-signal-reservoir-transport-alt50-fu0p0001", "mechanism": "M187-EarlySourceSlowEMASignalReservoirTransportFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_signal_reservoir_transport_from_train_stream_noise_estimator_without_direct_target_mask"},
            {"v21_id": "MLP-F160-early100-h800-source-slow-ema-terminal-source-floor", "continuation_id": "M188-early100-h800-source-slow-ema-terminal-source-floor-alt50-fu0p0001", "mechanism": "M188-EarlySourceSlowEMATerminalSourceFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_terminal_erosion_source_floor_train_stream_only"},
            {"v21_id": "MLP-F161-early100-h800-source-slow-ema-terminal-h4000-anchor-floor", "continuation_id": "M189-early100-h800-source-slow-ema-terminal-h4000-anchor-floor-alt50-fu0p0001", "mechanism": "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_terminal_erosion_source_anchor_floor_no_future_selector"},
            {"v21_id": "MLP-F162-early100-h800-source-slow-ema-terminal-decay-aware-floor", "continuation_id": "M190-early100-h800-source-slow-ema-terminal-decay-aware-floor-alt50-fu0p0001", "mechanism": "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_terminal_erosion_decay_aware_source_floor_train_stream_only"},
            {"v21_id": "MLP-F163-early100-h800-source-slow-ema-terminal-raw-guard-source-floor", "continuation_id": "M191-early100-h800-source-slow-ema-terminal-raw-guard-source-floor-alt50-fu0p0001", "mechanism": "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_f118_nearmiss_raw_guard_until_h4000_then_source_floor"},
            {"v21_id": "MLP-F164-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor", "continuation_id": "M192-early100-h800-source-slow-ema-terminal-raw-guard-h4000-anchor-floor-alt50-fu0p0001", "mechanism": "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_f118_nearmiss_raw_guard_until_h4000_then_anchor_floor"},
            {"v21_id": "MLP-F165-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor", "continuation_id": "M193-early100-h800-source-slow-ema-terminal-raw-guard-decay-aware-floor-alt50-fu0p0001", "mechanism": "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_f118_nearmiss_raw_guard_until_h4000_then_decay_aware_floor"},
            {"v21_id": "MLP-F166-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal", "continuation_id": "M194-early100-h800-source-slow-ema-terminal-anti-erosion-orthogonal-alt50-fu0p0001", "mechanism": "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_remove_anti_source_gradient_then_source_floor"},
            {"v21_id": "MLP-F167-early100-h800-source-slow-ema-terminal-source-reflection-guard", "continuation_id": "M195-early100-h800-source-slow-ema-terminal-source-reflection-guard-alt50-fu0p0001", "mechanism": "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_reflect_train_stream_anti_source_gradient"},
            {"v21_id": "MLP-F168-early100-h800-source-slow-ema-terminal-h4000-transport-corrector", "continuation_id": "M196-early100-h800-source-slow-ema-terminal-h4000-transport-corrector-alt50-fu0p0001", "mechanism": "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_source_anchor_transport_corrector_train_stream_only"},
            {"v21_id": "MLP-F169-early100-h800-source-slow-ema-terminal-h3200-anchor-transport", "continuation_id": "M197-early100-h800-source-slow-ema-terminal-h3200-anchor-transport-alt50-fu0p0001", "mechanism": "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_post_h4000_h3200_anchor_transport_train_stream_only"},
            {"v21_id": "MLP-F170-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport", "continuation_id": "M198-early100-h800-source-slow-ema-terminal-raw-guard-h3200-anchor-transport-alt50-fu0p0001", "mechanism": "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_raw_guard_post_h4000_h3200_anchor_transport"},
            {"v21_id": "MLP-F171-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry", "continuation_id": "M199-early100-h800-source-slow-ema-terminal-h3200-ratio-reentry-alt50-fu0p0001", "mechanism": "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_h3200_anchor_ratio_reentry_after_post_h4000_erosion"},
            {"v21_id": "MLP-F172-early100-h800-source-slow-ema-terminal-h3200-progress-carry", "continuation_id": "M200-early100-h800-source-slow-ema-terminal-h3200-progress-carry-alt50-fu0p0001", "mechanism": "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_h3200_terminal_source_preserve_plus_train_stream_progress_carry"},
            {"v21_id": "MLP-F173-early100-h800-source-slow-ema-terminal-h4000-progress-carry", "continuation_id": "M201-early100-h800-source-slow-ema-terminal-h4000-progress-carry-alt50-fu0p0001", "mechanism": "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_h4000_terminal_source_preserve_plus_late_progress_carry"},
            {"v21_id": "MLP-F174-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend", "continuation_id": "M202-early100-h800-source-slow-ema-terminal-h3200-source-progress-blend-alt50-fu0p0001", "mechanism": "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_h3200_source_projected_progress_blend_train_stream_only"},
            {"v21_id": "MLP-F175-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress", "continuation_id": "M203-early100-h800-source-slow-ema-terminal-raw-guard-h3600-gentle-progress-alt50-fu0p0001", "mechanism": "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_raw_guard_h3600_attach_gentle_train_split_progress"},
            {"v21_id": "MLP-F176-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress", "continuation_id": "M204-early100-h800-source-slow-ema-terminal-raw-guard-h4000-gentle-progress-alt50-fu0p0001", "mechanism": "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_raw_guard_h4000_attach_gentle_train_split_progress"},
            {"v21_id": "MLP-F177-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress", "continuation_id": "M205-early100-h800-source-slow-ema-terminal-raw-guard-h4400-projected-progress-alt50-fu0p0001", "mechanism": "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_raw_guard_h4400_projected_progress_after_h3200_drop"},
            {"v21_id": "MLP-F178-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup", "continuation_id": "M206-early100-h800-source-slow-ema-terminal-control-relative-sgd-catchup-alt50-fu0p0001", "mechanism": "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_terminal_control_relative_sgd_catchup_train_split_corrupt_gated"},
            {"v21_id": "MLP-F179-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup", "continuation_id": "M207-early100-h800-source-slow-ema-terminal-control-relative-adamw-catchup-alt50-fu0p0001", "mechanism": "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_terminal_control_relative_adamw_catchup_train_split_corrupt_gated"},
            {"v21_id": "MLP-F180-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup", "continuation_id": "M208-early100-h800-source-slow-ema-terminal-control-relative-source-balanced-catchup-alt50-fu0p0001", "mechanism": "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_terminal_control_relative_source_balanced_catchup_train_only"},
            {"v21_id": "MLP-F181-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve", "continuation_id": "M209-early100-h800-source-slow-ema-terminal-trajectory-adaptive-preserve-alt50-fu0p0001", "mechanism": "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_train_only_horizon_anchor_trajectory_adaptive_terminal_preserve"},
            {"v21_id": "MLP-F182-early100-h800-source-slow-ema-terminal-mid-erosion-bridge", "continuation_id": "M210-early100-h800-source-slow-ema-terminal-mid-erosion-bridge-alt50-fu0p0001", "mechanism": "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_h1600_h2400_anchor_mid_erosion_bridge_before_terminal"},
            {"v21_id": "MLP-F183-early100-h800-source-slow-ema-terminal-two-phase-ratio-repair", "continuation_id": "M211-early100-h800-source-slow-ema-terminal-two-phase-ratio-repair-alt50-fu0p0001", "mechanism": "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_two_phase_mid_bridge_then_post_h4000_ratio_repair_train_only"},
            {"v21_id": "MLP-F184-early100-h800-source-slow-ema-terminal-anti-source-clip", "continuation_id": "M212-early100-h800-source-slow-ema-terminal-anti-source-clip-alt50-fu0p0001", "mechanism": "M212-EarlySourceSlowEMATerminalAntiSourceClipFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_reduce_terminal_intervention_by_clipping_train_stream_anti_source_gradient"},
            {"v21_id": "MLP-F185-early100-h800-source-slow-ema-terminal-debt-aware-hold", "continuation_id": "M213-early100-h800-source-slow-ema-terminal-debt-aware-hold-alt50-fu0p0001", "mechanism": "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_debt_aware_preserve_until_h4000_then_terminal_hold_or_clip"},
            {"v21_id": "MLP-F186-early100-h800-source-slow-ema-terminal-anchor-flow-tiny", "continuation_id": "M214-early100-h800-source-slow-ema-terminal-anchor-flow-tiny-alt50-fu0p0001", "mechanism": "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_tiny_diffeomorphic_anchor_flow_after_h4000_without_h4800_selector"},
            {"v21_id": "MLP-F187-early100-h800-source-slow-ema-terminal-accept-memory", "continuation_id": "M215-early100-h800-source-slow-ema-terminal-accept-memory-alt50-fu0p0001", "mechanism": "M215-EarlySourceSlowEMATerminalAcceptMemoryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_replay_train_only_last_accepted_terminal_transport_when_current_signal_noisy"},
            {"v21_id": "MLP-F188-early100-h800-source-slow-ema-terminal-accept-memory-debt", "continuation_id": "M216-early100-h800-source-slow-ema-terminal-accept-memory-debt-alt50-fu0p0001", "mechanism": "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_debt_aware_accept_memory_preserves_terminal_transport_under_noisy_one_step_gate"},
            {"v21_id": "MLP-F189-early100-h800-source-slow-ema-terminal-source-progress-memory", "continuation_id": "M217-early100-h800-source-slow-ema-terminal-source-progress-memory-alt50-fu0p0001", "mechanism": "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2203_last_good_source_progress_transport_memory_train_only"},
            {"v21_id": "MLP-D1a-SPP-lambda025", "continuation_id": "D1a-spp-lambda025-alt50-fu0p000075", "mechanism": "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU", "fu_lr": 0.000075, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1a_source_preserving_late_projection_lambda025_train_only"},
            {"v21_id": "MLP-D1a-SPP-lambda050", "continuation_id": "D1a-spp-lambda050-alt50-fu0p0001", "mechanism": "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1a_source_preserving_late_projection_lambda050_train_only"},
            {"v21_id": "MLP-D1a-SPP-lambda100", "continuation_id": "D1a-spp-lambda100-alt50-fu0p0001", "mechanism": "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1a_source_preserving_late_projection_lambda100_train_only"},
            {"v21_id": "MLP-D1a-SPP-readout-only", "continuation_id": "D1a-spp-readout-only-alt50-fu0p0001", "mechanism": "M130-H800ReadoutChannelRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1a_readout_only_source_preserving_projection"},
            {"v21_id": "MLP-D1b-roworth-hidden-readout", "continuation_id": "D1b-roworth-hidden-readout-alt50-fu0p0001", "mechanism": "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1b_nora_style_row_orthogonal_source_update"},
            {"v21_id": "MLP-D1c-lowNDS-with-source-preservation", "continuation_id": "D1c-lownds-preserve-alt50-fu0p0001", "mechanism": "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1c_low_NDS_matrix_block_with_source_preservation"},
            {"v21_id": "MLP-D1d-dualmem-longonly-terminal", "continuation_id": "D1d-dualmem-longonly-terminal-alt50-fu0p0001", "mechanism": "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1d_short_long_dual_memory_terminal_preservation"},
            {"v21_id": "MLP-D1e-combined-debt-aware", "continuation_id": "D1e-combined-debt-aware-alt50-fu0p0001", "mechanism": "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1e_train_stream_debt_aware_terminal_FU"},
            {"v21_id": "MLP-D1f-info-volume-plus-roworth", "continuation_id": "D1f-info-volume-roworth-alt50-fu0p0001", "mechanism": "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2204_D1f_information_volume_preserving_source_FU"},
            {"v21_id": "CTRL-SGD", "continuation_id": "CTRL-SGD", "mechanism": "CTRL-SGD", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_sgd"},
            {"v21_id": "CTRL-AdamW", "continuation_id": "CTRL-AdamW", "mechanism": "CTRL-AdamW", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_adamw"},
            {"v21_id": "CTRL-RandomMatchedNorm", "continuation_id": "CTRL-RandomMatchedNorm", "mechanism": "CTRL-RandomMatchedNorm", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_random_norm"},
            {"v21_id": "CTRL-NoOpMatchedOverhead", "continuation_id": "CTRL-NoOpMatchedOverhead", "mechanism": "CTRL-NoOpMatchedOverhead", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_noop"},
        ]
    return [
        {"v21_id": "RAT-FU1-readout-only-rational-source", "continuation_id": "C4-rat-fu1-readout-only-rational-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "v2203_c4_rat_readout_only_rational_source"},
        {"v21_id": "RAT-FU2-denominator-safe-low-degree-source", "continuation_id": "C4-rat-fu2-denominator-safe-low-degree-alt50-fu0p00005", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.00005, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2203_c4_rat_denominator_safe_low_degree_source"},
        {"v21_id": "RAT-FU3-numerator-only-source-writer", "continuation_id": "C4-rat-fu3-numerator-only-source-writer-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2203_c4_rat_numerator_only_source_writer"},
        {"v21_id": "RBF-FU1-readout-only-local-support-source", "continuation_id": "C4-rbf-fu1-readout-only-local-support-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "v2203_c4_rbf_readout_only_local_support_source"},
        {"v21_id": "RBF-FU2-active-center-low-k-source", "continuation_id": "C4-rbf-fu2-active-center-low-k-alt50-fu0p00005", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.00005, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2203_c4_rbf_active_center_low_k_source"},
        {"v21_id": "RBF-FU3-compact-local-source-writer", "continuation_id": "C4-rbf-fu3-compact-local-source-writer-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2203_c4_rbf_compact_local_source_writer"},
        {"v21_id": "KSW1-basis-estimate-readout-commit", "continuation_id": "M17-readout-warm800-alt50-fu0p0001", "mechanism": "M17-ReadoutCarrierTwoPhaseLineCFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "basis_estimate_readout_commit"},
        {"v21_id": "KSW2-lowdegree-lowfreq-source-bank", "continuation_id": "M23-exact-readout-ultracap-alt50-fu0p0001", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "low_degree_low_frequency_source_bank"},
        {"v21_id": "F6-KSW2-density-smallstep-alt50", "continuation_id": "M23-exact-readout-ultracap-alt50-fu0p00005", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.00005, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "ksw2_early_source_density_smallstep_alt50"},
        {"v21_id": "F6-KSW2-density-alt100", "continuation_id": "M23-exact-readout-ultracap-alt100-fu0p0001", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0001, "alt_period": 100, "source_warmup_steps": 0, "hypothesis": "ksw2_early_source_density_alt100"},
        {"v21_id": "F6-KSW2-density-smallstep-alt100", "continuation_id": "M23-exact-readout-ultracap-alt100-fu0p00005", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.00005, "alt_period": 100, "source_warmup_steps": 0, "hypothesis": "ksw2_early_source_density_smallstep_alt100"},
        {"v21_id": "F7-KSW2-warm400-smallstep-alt50", "continuation_id": "M24-warm400-exact-readout-ultracap-alt50-fu0p00005", "mechanism": "M24-WarmupExactReadoutUltraCapFU", "fu_lr": 0.00005, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "ksw2_phase_repair_warm400_smallstep_alt50"},
        {"v21_id": "F7-KSW2-warm800-smallstep-alt50", "continuation_id": "M24-warm800-exact-readout-ultracap-alt50-fu0p00005", "mechanism": "M24-WarmupExactReadoutUltraCapFU", "fu_lr": 0.00005, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "ksw2_phase_repair_warm800_smallstep_alt50"},
        {"v21_id": "F8-KSW2-earlyboost-alt25", "continuation_id": "M23-exact-readout-ultracap-alt25-fu0p0001", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0001, "alt_period": 25, "source_warmup_steps": 0, "hypothesis": "ksw2_early_boost_more_frequent_commit_alt25"},
        {"v21_id": "F8-KSW2-earlyboost-highstep-alt50", "continuation_id": "M23-exact-readout-ultracap-alt50-fu0p0002", "mechanism": "M23-ExactReadoutUltraCapScheduledFU", "fu_lr": 0.0002, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "ksw2_early_boost_larger_step_alt50"},
        {"v21_id": "KSW3-dualbank-source-reservoir", "continuation_id": "M46-m2-matrix-block-retention", "mechanism": "M46-MomentumMatrixBlockRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "dual_bank_source_reservoir"},
        {"v21_id": "KSW5-spectral-hidden-basis-block-source", "continuation_id": "M69-rotated-cautious-matrix-alt50-fu0p0001", "mechanism": "M69-RotatedCautiousMatrixSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2202_hidden_matrix_source_mapping_spectral_basis_block"},
        {"v21_id": "KSW6-hidden-matrix-channel-source", "continuation_id": "M105-adamw400-hidden-matrix-channel-alt50-fu0p0001", "mechanism": "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_hidden_matrix_channel_source_mapping_to_kan_basis_block"},
        {"v21_id": "KSW7-early-hidden-matrix-channel-source", "continuation_id": "M105-adamw100-hidden-matrix-channel-alt50-fu0p0001", "mechanism": "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_hidden_matrix_channel_early_phase_midretention_repair"},
        {"v21_id": "KSW8-strong-hidden-matrix-channel-source", "continuation_id": "M105-adamw400-hidden-matrix-channel-alt25-fu0p00015", "mechanism": "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU", "fu_lr": 0.00015, "alt_period": 25, "source_warmup_steps": 400, "hypothesis": "v2202_hidden_matrix_channel_frequency_strength_midretention_repair"},
        {"v21_id": "KSW9-h800-source-slow-ema-bank", "continuation_id": "M129-h800-source-slow-ema-alt50-fu0p0001", "mechanism": "M129-H800SourceSlowEMARetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_kan_h800_source_anchor_slow_ema_bank_repair"},
        {"v21_id": "KSW10-h800-dual-memory-source-bank", "continuation_id": "M131-h800-dual-memory-alt50-fu0p0001", "mechanism": "M131-H800DualMemorySourceRetentionFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_kan_h800_short_long_dual_memory_source_bank_repair"},
        {"v21_id": "KSW4-margin-split-consensus-source", "continuation_id": "M68-margin-split-consensus-alt50-fu0p0001", "mechanism": "M68-MarginSplitConsensusSlowFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "kan_top_wrong_margin_split_consensus_source_state"},
        {"v21_id": "F3-T1-loss-cotangent-target", "continuation_id": "M49-loss-cotangent-target-alt50-fu0p0001", "mechanism": "M49-LossCotangentTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_source_loss_cotangent"},
        {"v21_id": "F3-T1c-lowrank-loss-target", "continuation_id": "M53-lowrank-loss-target-alt50-fu0p0001", "mechanism": "M53-LowRankLossCotangentTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_source_low_rank_loss"},
        {"v21_id": "F9-T2-cross-split-consensus-target", "continuation_id": "M54-cross-split-consensus-target-alt50-fu0p0001", "mechanism": "M54-CrossSplitConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_cross_split_consensus"},
        {"v21_id": "F9-T3-lowdegree-readout-target", "continuation_id": "M55-lowdegree-readout-target-alt50-fu0p0001", "mechanism": "M55-LowDegreeReadoutTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_low_degree_readout"},
        {"v21_id": "F9-T4-weak-stable-target", "continuation_id": "M56-weak-stable-target-alt50-fu0p0001", "mechanism": "M56-WeakStableTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_weak_stable"},
        {"v21_id": "F9-T5-b1-readout-transfer-target", "continuation_id": "M57-b1-readout-transfer-target-alt50-fu0p0001", "mechanism": "M57-B1ReadoutTransferTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_b1_fit_b2_transfer"},
        {"v21_id": "F9-T6-reservoir-excluding-consensus-target", "continuation_id": "M58-reservoir-excluding-consensus-target-alt50-fu0p0001", "mechanism": "M58-ReservoirExcludingConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_reservoir_excluding_consensus"},
        {"v21_id": "F96-signal-reservoir-b3-null-consensus-target", "continuation_id": "M134-signal-reservoir-b3-null-consensus-alt50-fu0p0001", "mechanism": "M134-SignalReservoirB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2202_train_only_signal_reservoir_b3_null_consensus_target_reset"},
        {"v21_id": "F97-source-bank-b3-null-consensus-target", "continuation_id": "M135-source-bank-b3-null-consensus-alt50-fu0p0001", "mechanism": "M135-SourceBankB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2202_train_only_source_bank_b3_null_consensus_target_reset"},
        {"v21_id": "F98-source-bank-b3-null-consensus-lowstep-target", "continuation_id": "M135-source-bank-b3-null-consensus-alt50-fu0p00002", "mechanism": "M135-SourceBankB3NullConsensusTargetFU", "fu_lr": 0.00002, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2202_source_bank_b3_null_consensus_target_lowstep_overwrite_repair"},
        {"v21_id": "F99-source-bank-b3-null-consensus-tinystep-target", "continuation_id": "M135-source-bank-b3-null-consensus-alt50-fu0p00001", "mechanism": "M135-SourceBankB3NullConsensusTargetFU", "fu_lr": 0.00001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2202_source_bank_b3_null_consensus_target_tinystep_overwrite_repair"},
        {"v21_id": "F100-source-bank-b3-null-earlypulse100-target", "continuation_id": "M137-source-bank-b3-null-earlypulse100-fu0p0001", "mechanism": "M137-EarlyPulseSourceBankB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_source_bank_b3_null_early_pulse_then_sgd_retention_test"},
        {"v21_id": "F101-source-bank-b3-null-earlypulse100-lowstep-target", "continuation_id": "M137-source-bank-b3-null-earlypulse100-fu0p00002", "mechanism": "M137-EarlyPulseSourceBankB3NullConsensusTargetFU", "fu_lr": 0.00002, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_source_bank_b3_null_lowstep_early_pulse_then_sgd_retention_test"},
        {"v21_id": "F102-source-bank-b3-null-earlypulse200-target", "continuation_id": "M137-source-bank-b3-null-earlypulse200-fu0p0001", "mechanism": "M137-EarlyPulseSourceBankB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 200, "hypothesis": "v2202_source_bank_b3_null_h200_pulse_then_sgd_retention_test"},
        {"v21_id": "F103-source-bank-b3-null-earlypulse400-target", "continuation_id": "M137-source-bank-b3-null-earlypulse400-fu0p0001", "mechanism": "M137-EarlyPulseSourceBankB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "v2202_source_bank_b3_null_h400_pulse_then_sgd_retention_test"},
        {"v21_id": "F104-source-bank-b3-null-earlypulse100-adamw-target", "continuation_id": "M138-source-bank-b3-null-earlypulse100-adamw-fu0p0001", "mechanism": "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 100, "hypothesis": "v2202_source_bank_b3_null_h100_pulse_then_adamw_washout_test"},
        {"v21_id": "F105-source-bank-b3-null-earlypulse200-adamw-target", "continuation_id": "M138-source-bank-b3-null-earlypulse200-adamw-fu0p0001", "mechanism": "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 200, "hypothesis": "v2202_source_bank_b3_null_h200_pulse_then_adamw_washout_test"},
        {"v21_id": "F9-TCTRL2-stable-random-b3-null-target", "continuation_id": "M136-stable-random-b3-null-target-alt50-fu0p0001", "mechanism": "M136-StableRandomB3NullTargetControlFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "v2202_same_shape_stable_random_b3_null_target_control"},
        {"v21_id": "F10-T7-b1-cross-split-consensus-transfer", "continuation_id": "M60-b1-cross-split-consensus-transfer-alt50-fu0p0001", "mechanism": "M60-B1CrossSplitConsensusTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "b1_transfer_plus_cross_split_consensus"},
        {"v21_id": "F10-T8-stable-b1-consensus-transfer", "continuation_id": "M61-stable-b1-consensus-transfer-alt50-fu0p0001", "mechanism": "M61-StableB1ConsensusTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "stable_subspace_b1_consensus_transfer"},
        {"v21_id": "F10-T9-b1-weak-stable-transfer", "continuation_id": "M62-b1-weak-stable-transfer-alt50-fu0p0001", "mechanism": "M62-B1WeakStableTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "b1_transfer_plus_weak_stable_target"},
        {"v21_id": "F10-T10-lowdegree-b1-consensus-transfer", "continuation_id": "M63-lowdegree-b1-consensus-transfer-alt50-fu0p0001", "mechanism": "M63-LowDegreeB1ConsensusTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "low_degree_b1_consensus_transfer"},
        {"v21_id": "F18-T11-contrastive-loss-random-orthogonal", "continuation_id": "M64-contrastive-loss-random-orthogonal-alt50-fu0p0001", "mechanism": "M64-ContrastiveLossRandomOrthogonalTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "loss_target_after_rowwise_random_control_projection"},
        {"v21_id": "F18-T12-b1-contrastive-loss-random-orthogonal", "continuation_id": "M65-b1-contrastive-loss-random-orthogonal-alt50-fu0p0001", "mechanism": "M65-B1ContrastiveLossRandomOrthogonalTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "b1_transfer_loss_target_after_rowwise_random_control_projection"},
        {"v21_id": "F19-T13-topwrong-margin-target", "continuation_id": "M66-topwrong-margin-target-alt50-fu0p0001", "mechanism": "M66-TopWrongMarginTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "correct_class_vs_top_wrong_margin_target"},
        {"v21_id": "F19-T14-b1-topwrong-margin-transfer", "continuation_id": "M67-b1-topwrong-margin-transfer-alt50-fu0p0001", "mechanism": "M67-B1TopWrongMarginTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "b1_transfer_correct_class_vs_top_wrong_margin_target"},
        {"v21_id": "F24-B1-consensus-lookahead-gated-transfer", "continuation_id": "M71-b1-consensus-lookahead-gate-alt50-fu0p0001", "mechanism": "M71-TrainLookaheadB1ConsensusTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "b1_consensus_target_vs_gradient_train_lookahead_selector"},
        {"v21_id": "F25-loss-warm-to-b1-consensus-migration", "continuation_id": "M72-losswarm800-to-b1-consensus-alt50-fu0p0001", "mechanism": "M72-LossWarmToB1ConsensusMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_b1_consensus_delayed_migration"},
        {"v21_id": "F26-easy-b1-consensus-transfer", "continuation_id": "M73-easy-b1-consensus-alt50-fu0p0001", "mechanism": "M73-EasyB1ConsensusTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "train_low_loss_easy_example_b1_consensus_source_target"},
        {"v21_id": "F27-loss-warm-to-easy-b1-consensus-migration", "continuation_id": "M74-losswarm800-to-easy-b1-consensus-alt50-fu0p0001", "mechanism": "M74-LossWarmToEasyB1ConsensusMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_easy_example_b1_consensus_migration"},
        {"v21_id": "F28-loss-easy-b1-consensus-blend", "continuation_id": "M75-loss-easy-b1-consensus-blend-alt50-fu0p0001", "mechanism": "M75-LossEasyB1ConsensusBlendFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "loss_cotangent_preserved_plus_easy_b1_consensus_blend"},
        {"v21_id": "F29-loss-warm-to-loss-easy-b1-consensus-blend", "continuation_id": "M76-losswarm800-to-loss-easy-b1-consensus-blend-alt50-fu0p0001", "mechanism": "M76-LossWarmToLossEasyB1ConsensusBlendFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_blended_easy_consensus_migration"},
        {"v21_id": "F30-gain-gated-loss-warm-b1-consensus", "continuation_id": "M77-gain-gated-losswarm800-b1-consensus-alt50-fu0p0001", "mechanism": "M77-GainGatedLossWarmB1ConsensusMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "train_split_b2_gain_precommit_gate_for_b1_consensus_migration"},
        {"v21_id": "F31-gain-gated-loss-warm-blend-consensus", "continuation_id": "M78-gain-gated-losswarm800-blend-consensus-alt50-fu0p0001", "mechanism": "M78-GainGatedLossWarmBlendMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "train_split_b2_gain_precommit_gate_for_blended_consensus_migration"},
        {"v21_id": "F32-b1-consensus-b3-null-transfer", "continuation_id": "M79-b1-consensus-b3-null-alt50-fu0p0001", "mechanism": "M79-B1ConsensusB3NullTransferFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "b1_consensus_target_with_b3_safety_null_channel_projection"},
        {"v21_id": "F33-loss-warm-to-b1-consensus-b3-null", "continuation_id": "M80-losswarm800-to-b1-consensus-b3-null-alt50-fu0p0001", "mechanism": "M80-LossWarmToB1ConsensusB3NullMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_b1_consensus_b3_null_channel_migration"},
        {"v21_id": "F34-view-consistent-loss-b3-null", "continuation_id": "M81-view-consistent-loss-b3-null-alt50-fu0p0001", "mechanism": "M81-ViewConsistentLossTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "train_input_perturbation_view_consistent_loss_target_with_b3_null"},
        {"v21_id": "F35-loss-warm-to-view-consistent-loss", "continuation_id": "M82-losswarm800-to-view-consistent-loss-alt50-fu0p0001", "mechanism": "M82-LossWarmToViewConsistentLossMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_view_consistent_loss_b3_null_migration"},
        {"v21_id": "F36-lowbank-loss-b3-null", "continuation_id": "M83-lowbank-loss-b3-null-alt50-fu0p0001", "mechanism": "M83-LowBankLossB3NullFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "low_degree_low_frequency_source_bank_loss_target_with_b3_reservoir_null"},
        {"v21_id": "F37-loss-warm-to-lowbank-loss-b3-null", "continuation_id": "M84-losswarm800-to-lowbank-loss-b3-null-alt50-fu0p0001", "mechanism": "M84-LossWarmToLowBankLossB3NullMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_low_bank_source_channel_migration"},
        {"v21_id": "F38-gain-gated-lowbank-loss-b3-null", "continuation_id": "M85-gain-gated-lowbank-loss-b3-null-alt50-fu0p0001", "mechanism": "M85-GainGatedLowBankLossB3NullFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "train_split_b2_gain_precommit_gate_for_low_bank_source_channel"},
        {"v21_id": "F39-loss-warm-to-gated-lowbank-loss-b3-null", "continuation_id": "M86-losswarm800-to-gated-lowbank-loss-b3-null-alt50-fu0p0001", "mechanism": "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 800, "hypothesis": "loss_cotangent_early_closure_then_gain_gated_low_bank_source_channel_migration"},
        {"v21_id": "F68-adamw-boundary-to-gated-lowbank-b3-null", "continuation_id": "M109-adamw400-to-gated-lowbank-b3-null-alt50-fu0p0001", "mechanism": "M109-AdamWBoundaryToGainGatedLowBankB3NullFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "c3_boundary_conditioned_lowbank_b3_null_source_channel"},
        {"v21_id": "F69-adamw-boundary-lowbank-anchor-antiwashout", "continuation_id": "M110-adamw400-lowbank-anchor-antiwashout-alt50-fu0p0001", "mechanism": "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 400, "hypothesis": "c3_lowbank_source_anchor_antiwashout_after_boundary"},
        {"v21_id": "F3-T5-random-matched-target", "continuation_id": "M50-random-matched-target-alt50-fu0p0001", "mechanism": "M50-RandomMatchedTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_random_control"},
        {"v21_id": "F9-TCTRL-stable-random-target", "continuation_id": "M59-stable-random-target-alt50-fu0p0001", "mechanism": "M59-StableRandomTargetControlFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_stable_random_control"},
        {"v21_id": "F3-T6-sign-flipped-target", "continuation_id": "M51-sign-flipped-target-alt50-fu0p0001", "mechanism": "M51-SignFlippedTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_sign_flip_control"},
        {"v21_id": "F3-T7-corrupted-label-target", "continuation_id": "M52-corrupted-target-alt50-fu0p0001", "mechanism": "M52-CorruptedLabelTargetFU", "fu_lr": 0.0001, "alt_period": 50, "source_warmup_steps": 0, "hypothesis": "function_space_target_corrupt_control"},
        {"v21_id": "CTRL-AdamW", "continuation_id": "CTRL-AdamW", "mechanism": "CTRL-AdamW", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_adamw"},
        {"v21_id": "CTRL-SGD", "continuation_id": "CTRL-SGD", "mechanism": "CTRL-SGD", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_sgd"},
        {"v21_id": "CTRL-RandomMatchedNorm", "continuation_id": "CTRL-RandomMatchedNorm", "mechanism": "CTRL-RandomMatchedNorm", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_random_norm"},
        {"v21_id": "CTRL-NoOpMatchedOverhead", "continuation_id": "CTRL-NoOpMatchedOverhead", "mechanism": "CTRL-NoOpMatchedOverhead", "fu_lr": 0.001, "alt_period": 10, "source_warmup_steps": 0, "hypothesis": "control_noop"},
    ]


def build_packet(out_dir: Path, required_artifacts: list[str]) -> None:
    packet = out_dir / "v21_code_review_packet"
    if packet.exists():
        shutil.rmtree(packet)
    packet.mkdir(parents=True, exist_ok=True)
    for name in [
        "00_README.md",
        "01_ENVIRONMENT",
        "02_SOURCE_TREE",
        "03_IMPORT_CLOSURE",
        "04_EFFICIENCY_KERNELS",
        "05_SOURCE_RETENTION",
        "06_RAW_MATRICES",
        "07_REPRO_COMMANDS",
        "08_FAILURE_TAXONOMY",
    ]:
        target = packet / name
        if Path(name).suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
    for cmd, name in [
        (["git", "status", "--short"], "git_status.txt"),
        (["git", "rev-parse", "HEAD"], "git_head.txt"),
        ([PYTHON, "--version"], "python_version.txt"),
        (["nvidia-smi"], "cuda_info.txt"),
    ]:
        _code, report = run_cmd(cmd, timeout=120)
        write_text(packet / "01_ENVIRONMENT" / name, report)
    for rel in [
        "dgkan/metrics/linec.py",
        "dgkan/fu/mechanisms.py",
        "dgkan/fu/source_channel.py",
        "dgkan/fu/source_state.py",
        "dgkan/fu/function_space_actuation.py",
        "dgkan/fu/matrix_block.py",
        "dgkan/fu/poprisk_source.py",
        "dgkan/profiling/efficiency_v21.py",
        "dgkan/kernels/che_official.py",
        "dgkan/kernels/fou_official.py",
        "experiments/run_v17_common.py",
        "experiments/run_v21_common.py",
        "experiments/run_v21_s05_truth_gate.py",
        "experiments/run_v21_efficiency_officialization.py",
        "experiments/run_v21_mlp_source_dynamics.py",
        "experiments/run_v21_kan_source_channel_writer.py",
        "experiments/run_v21_function_space_target_reset.py",
        "experiments/run_v21_function_space_target_contrast.py",
        "experiments/run_v21_function_space_target_source.py",
        "experiments/run_v21_source_retention_estimator_audit.py",
        "experiments/run_v21_optimizer_overwrite_diagnostic.py",
        "experiments/run_v21_finalize.py",
    ]:
        copy_if_exists(ROOT / rel, packet / "02_SOURCE_TREE" / rel)
    copy_if_exists(V21_PLAN_DOC, packet / "02_SOURCE_TREE/docs" / V21_PLAN_DOC.name)
    copy_if_exists(V21_EXEC_DOC, packet / "07_REPRO_COMMANDS" / V21_EXEC_DOC.name)
    copy_if_exists(V21_RECAP_DOC, packet / "07_REPRO_COMMANDS" / V21_RECAP_DOC.name)
    section_map = {
        "03_IMPORT_CLOSURE": ["v21_code_truth_gate.csv", "v21_import_closure.csv", "v21_compileall.log", "v21_mechanism_semantic_contract.csv"],
        "04_EFFICIENCY_KERNELS": ["v21_efficiency_truth_table.csv", "v21_efficiency_waterfall.csv", "v21_kernel_gradcheck.csv"],
        "05_SOURCE_RETENTION": ["v21_mlp_source_dynamics.csv", "v21_kan_source_writer_matrix.csv", "v21_source_retention_matrix.csv", "v21_debt_accounting_matrix.csv", "v21_function_space_target_reset_matrix.csv", "v21_function_space_target_contrast_summary.csv", "v21_function_space_target_contrast_decision.csv", "v21_function_space_target_source_matrix.csv", "v21_f5_source_retention_estimator_summary.csv", "v21_f5_source_retention_estimator_decision.csv", "v21_f5_early_source_stratification.csv", "v21_adamw_overwrite_diagnostic.csv"],
        "06_RAW_MATRICES": ["v21_mlp_source_dynamics_matrix.csv", "v21_mlp_source_dynamics_traces.csv", "v21_kan_source_writer_raw_matrix.csv", "v21_kan_source_writer_raw_traces.csv"],
        "07_REPRO_COMMANDS": ["v21_command_journal.csv"],
        "08_FAILURE_TAXONOMY": ["v21_failure_taxonomy.csv", "v21_route_decision.json", "v21_required_artifact_manifest.csv"],
    }
    for section, names in section_map.items():
        for name in names:
            copy_if_exists(out_dir / name, packet / section / Path(name).name)
    write_text(packet / "00_README.md", "\n".join(["# v21 Code Review Packet", f"- generated_at: {now_sg()}", f"- result_dir: {out_dir}", "- no fabricated data: absent evidence is blocked, not promoted."]) + "\n")
    rows = []
    for path in sorted(p for p in packet.rglob("*") if p.is_file()):
        rel = path.relative_to(packet)
        if rel.name in {"packet_manifest.csv", "packet_sha256_manifest.csv"}:
            continue
        rows.append({"relative_path": str(rel), "sha256": sha256_file(path), "bytes": path.stat().st_size, "artifact_type": rel.parts[0], "required": 1})
    write_rows(packet / "packet_manifest.csv", rows)
    write_rows(packet / "packet_sha256_manifest.csv", [{"relative_path": r["relative_path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in rows])
    with zipfile.ZipFile(out_dir / "v21_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in packet.rglob("*") if p.is_file()):
            zf.write(path, Path("v21_code_review_packet") / path.relative_to(packet))
    with zipfile.ZipFile(out_dir / "v21_results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and "v21_code_review_packet/" not in str(p.relative_to(out_dir)) and p.name != "v21_results_bundle.zip"):
            zf.write(path, path.relative_to(out_dir))
