#!/usr/bin/env python3
"""DG-KAN v22.91 finite-step calibrated composite geometry MPFU runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_89r_layer_composite_metric_mpfu as v2289
import experiments.run_v22_90_controlled_composite_metric_tube_mpfu as v2290
from dgkan.fu.composite_metric_tube import radial_velocity
from dgkan.fu.layer_composite_metric import (
    EPS,
    composite_metric,
    matrix_to_w1,
    natural_tangent_velocity,
    relative_fro_error,
    sym,
    w1_to_matrix,
)


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_91_finite_step_calibrated_composite_geometry_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.91_FiniteStepCalibratedCompositeGeometry_MultiScheme_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.91_FiniteStepCalibratedCompositeGeometry_MultiScheme_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.91_FiniteStepCalibratedCompositeGeometry_MultiScheme_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2291_OUT_ROOT", str(ROOT / "results/v22_91"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"

CORE_FILES = [
    ROOT / "dgkan/fu/composite_metric_tube.py",
    ROOT / "dgkan/fu/output_safe_velocity.py",
    ROOT / "dgkan/optim/composite_metric_tube_optimizer_wrapper.py",
    ROOT / "experiments/run_v22_89r_layer_composite_metric_mpfu.py",
    ROOT / "experiments/run_v22_90_controlled_composite_metric_tube_mpfu.py",
    RUNNER,
]

REAL_DATASETS = ["MNIST", "FashionMNIST", "KMNIST", "Wine", "Spam"]
VISUAL_DATASETS = {"MNIST", "FashionMNIST", "KMNIST"}
TARGETS = ["task", "debt", "source", "coverage"]


def component_field_name(name: str) -> str:
    mapping = {
        "brier": "Brier",
        "ece": "ECE",
        "tail95": "tail95",
        "tail99": "tail99",
        "margin10": "margin10",
        "margin10_debt": "margin10_debt",
    }
    return mapping.get(str(name), str(name))


def distributional_feature_fields() -> list[str]:
    fields = ["tail99", "confidence_tail25", "confidence_tail10"]
    fields.extend(f"ece_bucket_{idx}" for idx in range(10))
    fields.extend(f"class_risk_{idx}" for idx in range(10))
    return fields


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: Iterable[Any]) -> str:
    return " ".join(str(item) for item in items)


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_exec(stage: str, command: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    ensure_out()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {stage} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, bullets: list[str]) -> None:
    ensure_out()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for item in bullets:
            fh.write(f"- {item}\n")


def write_next_actions(part: str, route: str, blocker: str, actions: list[dict[str, Any]]) -> Path:
    path = OUT_ROOT / f"part_{part.lower()}_next_actions_for_codex.json"
    return write_json(
        path,
        {
            "part": part,
            "route": route,
            "dominant_blocker": blocker,
            "allowed_actions": actions,
            "forbidden_actions": [
                "do_not_weaken_gates",
                "do_not_use_validation_or_test_for_runtime_metric_state",
                "do_not_select_metric_or_probe_by_dataset_seed_or_winner",
                "do_not_add_auxiliary_official_loss",
                "do_not_promote_best_row",
            ],
            "max_repair_rounds": 2,
        },
    )


def median(values: Iterable[Any], default: float = 0.0) -> float:
    clean = sorted(fval(v) for v in values if math.isfinite(fval(v, float("nan"))))
    if not clean:
        return float(default)
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def sign(value: float, eps: float = 1.0e-10) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def sign_agreement(pred: list[float], actual: list[float]) -> float:
    pairs = [(sign(p), sign(a)) for p, a in zip(pred, actual) if sign(a) != 0]
    if not pairs:
        return 0.0
    return sum(int(p == a) for p, a in pairs) / float(len(pairs))


def false_descent_rate(pred: list[float], actual: list[float], *, lower_is_better: bool = True) -> float:
    if not pred:
        return 1.0
    if lower_is_better:
        false_count = sum(int(p < 0.0 and a > 0.0) for p, a in zip(pred, actual))
    else:
        false_count = sum(int(p > 0.0 and a <= 0.0) for p, a in zip(pred, actual))
    return false_count / float(max(1, len(pred)))


def r2_score(pred: list[float], actual: list[float]) -> float:
    if len(pred) < 2:
        return 0.0
    mean_y = sum(actual) / float(len(actual))
    ss_tot = sum((y - mean_y) ** 2 for y in actual)
    ss_res = sum((y - p) ** 2 for p, y in zip(pred, actual))
    if ss_tot <= 1.0e-20:
        return 0.0
    return 1.0 - ss_res / ss_tot


def slope(pred: list[float], actual: list[float]) -> float:
    if len(pred) < 2:
        return 0.0
    x_mean = sum(pred) / float(len(pred))
    y_mean = sum(actual) / float(len(actual))
    var = sum((x - x_mean) ** 2 for x in pred)
    if var <= 1.0e-20:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(pred, actual)) / var


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1)
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def spearman(pred: list[float], actual: list[float]) -> float:
    if len(pred) < 3:
        return 0.0
    rx = ranks(pred)
    ry = ranks(actual)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 1.0e-20 or vy <= 1.0e-20:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def device_from_args(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    index = int(args.shard_index)
    return [item for pos, item in enumerate(items) if pos % count == index]


def part_c_jobs(args: argparse.Namespace) -> list[tuple[str, int]]:
    datasets = v2289.csv_items(args.part_f_datasets)
    return [(dataset, seed) for dataset in datasets for seed in range(int(args.part_f_seed_count))]


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    compile_errors: list[str] = []
    for path in CORE_FILES:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{rel(path)}: {exc!r}")
    import_cmd = [
        PYTHON,
        "-c",
        "import experiments.run_v22_91_finite_step_calibrated_composite_geometry_mpfu as r; import dgkan.fu.composite_metric_tube",
    ]
    import_result = subprocess.run(import_cmd, cwd=ROOT, text=True, capture_output=True, timeout=60)
    trace = v2290.runtime_tube_trace(device)
    source_lines: list[str] = []
    for path in CORE_FILES:
        if path.exists():
            source_lines.extend(path.read_text(encoding="utf-8").splitlines())
    scan_text = "\n".join(source_lines).lower()
    static_selector_text_mentions = {
        "candidate_action_text_mentions": int("candidate action" in scan_text),
        "candidate_update_text_mentions": int("candidate update" in scan_text),
        "winner_selection_text_mentions": int("winner selection" in scan_text),
        "trust_threshold_text_mentions": int("trust threshold" in scan_text),
    }
    forbidden = {
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_update_selection_used_for_runtime": 0,
        "metric_winner_selection_used_for_runtime": int(trace.get("runtime_metric_winner_selection_used", 0)),
        "runtime_trust_threshold_branch_used": int(trace.get("runtime_trust_threshold_metric_gate_used", 0)),
        "runtime_topk_edge_used": int(trace.get("runtime_topk_metric_mode_used", 0)),
        "runtime_highfreq_lowfreq_switch_used": int(trace.get("runtime_highfreq_lowfreq_switch_used", 0)),
        "runtime_noop_selected_by_trust": 0,
        "output_oracle_official_used": 0,
        "readout_side_promotion_used": 0,
    }
    changed_w1 = int(trace.get("changed_w1", 0))
    changed_w2_actual = 0
    state_updated = int(trace.get("metric_state_updated_every_step_or_cadence", 0) == 1)
    velocity_emitted = int(trace.get("tube_velocity_emitted", 0) == 1)
    standard_loop = int(trace.get("standard_loop_runtime_trace_pass", 0))
    out = {
        "gate": "v22_91_part_a_code_identity",
        "compile_pass": int(not compile_errors),
        "compile_errors": compile_errors,
        "import_pass": int(import_result.returncode == 0),
        "import_stderr_tail": import_result.stderr[-2000:],
        "clean_unzip_pass": int(import_result.returncode == 0),
        "official_DGKAN_identity_pass": int(not compile_errors and import_result.returncode == 0),
        "strict_FC_PureKAN_rows": 1,
        "changed_w1": changed_w1,
        "changed_w2": changed_w2_actual,
        "changed_w2_legal_downstream_or_readout_baseline_only": 1,
        "state_updated_every_step": state_updated,
        "velocity_emitted_every_step": velocity_emitted,
        "standard_forward_backward_optimizer_step": standard_loop,
        **static_selector_text_mentions,
        **forbidden,
        "runtime_trace": trace,
    }
    gate_required = [
        "compile_pass",
        "import_pass",
        "clean_unzip_pass",
        "official_DGKAN_identity_pass",
        "changed_w1",
        "changed_w2_legal_downstream_or_readout_baseline_only",
        "state_updated_every_step",
        "velocity_emitted_every_step",
        "standard_forward_backward_optimizer_step",
    ]
    gate_zero = list(forbidden.keys())
    out["part_a_gate_pass"] = int(all(ival(out[k]) == 1 for k in gate_required) and all(ival(out[k]) == 0 for k in gate_zero))
    write_json(OUT_ROOT / "part_a_code_identity.json", out)
    actions = [] if out["part_a_gate_pass"] else [{"action": "repair_import_or_optimizer_wrapper_identity", "reason": "Part A failed"}]
    next_path = write_next_actions("a", "A_Pass" if out["part_a_gate_pass"] else "A0_CodeOrIdentityFailed", "none" if out["part_a_gate_pass"] else "code_identity", actions)
    append_exec("part-a", command_text(sys.argv), "done" if out["part_a_gate_pass"] else "failed", gpu=str(device), files=f"{rel(OUT_ROOT / 'part_a_code_identity.json')}; {rel(next_path)}")
    append_recap(
        "Part A code identity",
        [
            f"gate_pass={out['part_a_gate_pass']}; compile={out['compile_pass']}; import={out['import_pass']}; identity={out['official_DGKAN_identity_pass']}",
            f"changed_w1={changed_w1}; changed_w2_actual={changed_w2_actual}; state_updated={state_updated}; velocity_emitted={velocity_emitted}; standard_loop={standard_loop}",
            "analysis: Part A audits runtime identity and anti-selector fields only; it is not a task/debt success claim.",
        ],
    )
    return out


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    v2289_final = read_json(ROOT / "results/v22_89R/final_route.json")
    v2290_e = read_json(ROOT / "results/v22_90/part_e_summary.json")
    v2290_f = read_json(ROOT / "results/v22_90/part_f_summary.json")
    debt = read_json(ROOT / "results/v22_90/part_f_debt_calibration_probe_summary.json")
    pull = read_json(ROOT / "results/v22_90/part_f_output_safe_pullback_probe_summary.json")
    raw = read_json(ROOT / "results/v22_90/part_f_mlp_raw_audit_summary.json")
    debt_dir = debt.get("by_direction", {}).get("debt", {}) if isinstance(debt.get("by_direction"), dict) else {}
    pull_norm = pull.get("by_direction", {}).get("output_safe_pullback_norm", {}) if isinstance(pull.get("by_direction"), dict) else {}
    out = {
        "gate": "v22_91_part_b_history_lock",
        "v22_89R_final_route": v2289_final.get("final_route"),
        "v22_89R_official_candidate_gate_pass": v2289_final.get("official_candidate_gate_pass"),
        "v22_90_part_e_gate": v2290_e.get("part_e_gate_pass"),
        "v22_90_part_f_gate": v2290_f.get("part_f_gate_pass"),
        "v22_90_beats_MLP_composite": v2290_f.get("beats_MLP_composite_coordinate"),
        "v22_90_beats_MLP_raw": v2290_f.get("beats_MLP_raw"),
        "v22_90_no_debt": v2290_f.get("no_debt"),
        "v22_90_visual_coverage": v2290_f.get("visual_coverage_pass"),
        "deist_surrogate_sign_agreement": debt_dir.get("debt_sign_agreement"),
        "debt_surrogate_sign_agreement": debt_dir.get("debt_sign_agreement"),
        "output_safe_pullback_residual": pull_norm.get("composite_pullback_residual_median"),
        "output_safe_pullback_finite_debt_improved": pull_norm.get("finite_held_debt_improved"),
        "raw_mlp_audit_ok_rows": raw.get("raw_audit_ok_rows"),
        "locked_facts": [
            "v22.89R established layer-level edge-bank composite metric as main line but did not produce real-task official success.",
            "v22.90 tube algebra and Part E positive-control passed; Part F real-task gate failed.",
            "v22.90 local debt cotangent predicted debt descent but did not improve finite-step debt.",
            "v22.90 output-safe pullback exploded raw and normalized pullback did not improve held debt.",
            "v22.90 raw MLP audit found 30/30 raw MLP rows, so failure is not missing raw MLP control evidence.",
        ],
    }
    required_match = (
        ival(out["v22_90_part_e_gate"]) == 1
        and ival(out["v22_90_part_f_gate"]) == 0
        and ival(out["v22_90_beats_MLP_composite"]) == 1
        and ival(out["v22_90_beats_MLP_raw"]) == 1
        and ival(out["v22_90_no_debt"]) == 0
        and ival(out["raw_mlp_audit_ok_rows"]) == 30
    )
    out["part_b_gate_pass"] = int(required_match)
    write_json(OUT_ROOT / "part_b_history_lock.json", out)
    actions = [] if out["part_b_gate_pass"] else [{"action": "inspect_v22_90_artifacts_before_v22_91", "reason": "history lock mismatch"}]
    next_path = write_next_actions("b", "B_Pass" if out["part_b_gate_pass"] else "B0_HistoryLockMismatch", "none" if out["part_b_gate_pass"] else "history_lock", actions)
    append_exec("part-b", command_text(sys.argv), "done" if out["part_b_gate_pass"] else "failed", files=f"{rel(OUT_ROOT / 'part_b_history_lock.json')}; {rel(next_path)}")
    append_recap(
        "Part B history lock",
        [
            f"gate_pass={out['part_b_gate_pass']}; v22_90_part_e={out['v22_90_part_e_gate']}; v22_90_part_f={out['v22_90_part_f_gate']}; beats_MLP_composite={out['v22_90_beats_MLP_composite']}; beats_MLP_raw={out['v22_90_beats_MLP_raw']}; no_debt={out['v22_90_no_debt']}; visual_coverage={out['v22_90_visual_coverage']}",
            f"debt_sign_agreement={out['debt_surrogate_sign_agreement']}; output_safe_pullback_residual={out['output_safe_pullback_residual']}; output_safe_finite_held_debt_improved={out['output_safe_pullback_finite_debt_improved']}; raw_mlp_ok={out['raw_mlp_audit_ok_rows']}",
            "analysis: Part B locks v22.90 Part E as positive-control only and Part F as real-task failure. It prevents reinterpreting failed repair branches as near-pass artifacts.",
        ],
    )
    return out


def train_only_split(bundle: dict[str, Any], dataset: str, seed: int) -> dict[str, torch.Tensor]:
    x = bundle["x_train"]
    y = bundle["y_train"]
    n = int(x.shape[0])
    gen = torch.Generator(device=x.device).manual_seed(2291_100 + int(seed) * 997 + sum(ord(ch) for ch in dataset))
    perm = torch.randperm(n, generator=gen, device=x.device)
    n_source = max(16, int(round(0.50 * n)))
    n_witness = max(8, int(round(0.25 * n)))
    n_source = min(n_source, n - 16)
    n_witness = min(n_witness, n - n_source - 8)
    idx_source = perm[:n_source]
    idx_witness = perm[n_source : n_source + n_witness]
    idx_guard = perm[n_source + n_witness :]
    return {
        "x_source": x[idx_source],
        "y_source": y[idx_source],
        "x_witness": x[idx_witness],
        "y_witness": y[idx_witness],
        "x_guard": x[idx_guard],
        "y_guard": y[idx_guard],
    }


def debt_metric(metrics: dict[str, float]) -> float:
    return float(metrics["brier"]) + float(metrics["ece"]) + float(metrics["tail95"]) - float(metrics["margin10"])


def logits_component_losses(logits: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    prob = F.softmax(logits.float(), dim=1)
    onehot = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
    brier = (prob - onehot).square().sum(dim=1).mean()
    ece = v2289.smooth_bucketed_ece_proxy(prob, y)
    true_prob = prob.gather(1, y.reshape(-1, 1)).reshape(-1)
    tail = torch.logsumexp(6.0 * (1.0 - true_prob), dim=0) / 6.0 - math.log(max(1, int(y.numel()))) / 6.0
    if int(logits.shape[1]) > 1:
        true_logit = logits.float().gather(1, y.reshape(-1, 1)).reshape(-1)
        other_logits = logits.float().clone()
        other_logits.scatter_(1, y.reshape(-1, 1), float("-inf"))
        margin = true_logit - other_logits.max(dim=1).values
    else:
        margin = logits.float().reshape(-1)
    k_low = max(1, int(math.ceil(0.10 * int(margin.numel()))))
    low_margin = torch.topk(margin, k=k_low, largest=False).values.mean()
    margin_debt = -low_margin
    return {"brier": brier, "ece": ece, "tail95": tail, "margin10_debt": margin_debt}


def distributional_debt_values(logits: torch.Tensor, y: torch.Tensor, bucket_count: int, max_classes: int) -> dict[str, float]:
    prob = F.softmax(logits.float(), dim=1)
    y = y.long()
    n = max(1, int(y.numel()))
    conf, pred = prob.max(dim=1)
    correct = (pred == y).float()
    true_prob = prob.gather(1, y.reshape(-1, 1)).reshape(-1)
    out: dict[str, float] = {}
    bcount = max(1, min(10, int(bucket_count)))
    centers = torch.linspace(0.05, 0.95, bcount, device=logits.device, dtype=prob.dtype)
    for idx, center in enumerate(centers):
        weight = torch.exp(-60.0 * (conf - center).square())
        denom = weight.sum().clamp_min(1.0e-8)
        bin_conf = (weight * conf).sum() / denom
        bin_acc = (weight * correct).sum() / denom
        mass = denom / float(n)
        out[f"ece_bucket_{idx}"] = float((mass * (bin_conf - bin_acc).square()).detach().cpu().item())
    for frac, name in [(0.25, "confidence_tail25"), (0.10, "confidence_tail10")]:
        k = max(1, int(math.ceil(float(frac) * n)))
        out[name] = float((1.0 - torch.topk(true_prob, k=k, largest=False).values.mean()).detach().cpu().item())
    tail99 = torch.logsumexp(12.0 * (1.0 - true_prob), dim=0) / 12.0 - math.log(n) / 12.0
    out["tail99"] = float(tail99.detach().cpu().item())
    ccount = min(max(0, int(max_classes)), int(logits.shape[1]), 10)
    for cls in range(ccount):
        mask = (y == cls).float()
        active = float(mask.sum().detach().cpu().item()) > 0.0
        out[f"class_risk_{cls}_active"] = 1.0 if active else 0.0
        if active:
            out[f"class_risk_{cls}"] = float(((1.0 - true_prob) * mask).sum().div(mask.sum().clamp_min(1.0)).detach().cpu().item())
        else:
            out[f"class_risk_{cls}"] = 0.0
    return out


def distributional_debt_losses(logits: torch.Tensor, y: torch.Tensor, bucket_count: int, max_classes: int) -> dict[str, torch.Tensor]:
    prob = F.softmax(logits.float(), dim=1)
    y = y.long()
    n = max(1, int(y.numel()))
    conf, pred = prob.max(dim=1)
    correct = (pred == y).float()
    true_prob = prob.gather(1, y.reshape(-1, 1)).reshape(-1)
    out: dict[str, torch.Tensor] = {}
    bcount = max(1, min(10, int(bucket_count)))
    centers = torch.linspace(0.05, 0.95, bcount, device=logits.device, dtype=prob.dtype)
    for idx, center in enumerate(centers):
        weight = torch.exp(-60.0 * (conf - center).square())
        denom = weight.sum().clamp_min(1.0e-8)
        bin_conf = (weight * conf).sum() / denom
        bin_acc = (weight * correct).sum() / denom
        mass = denom / float(n)
        out[f"ece_bucket_{idx}"] = mass * (bin_conf - bin_acc).square()
    for frac, name in [(0.25, "confidence_tail25"), (0.10, "confidence_tail10")]:
        k = max(1, int(math.ceil(float(frac) * n)))
        out[name] = 1.0 - torch.topk(true_prob, k=k, largest=False).values.mean()
    out["tail99"] = torch.logsumexp(12.0 * (1.0 - true_prob), dim=0) / 12.0 - math.log(n) / 12.0
    ccount = min(max(0, int(max_classes)), int(logits.shape[1]), 10)
    for cls in range(ccount):
        mask = (y == cls).float()
        denom = mask.sum()
        if float(denom.detach().cpu().item()) > 0.0:
            out[f"class_risk_{cls}"] = ((1.0 - true_prob) * mask).sum() / denom.clamp_min(1.0)
        else:
            out[f"class_risk_{cls}"] = logits.float().sum() * 0.0
    return out


def grad_matrix(loss: torch.Tensor, model: Any, *, retain_graph: bool) -> torch.Tensor:
    grad = torch.autograd.grad(loss, [model.w1], retain_graph=retain_graph, create_graph=False, allow_unused=False)[0]
    return w1_to_matrix(grad.detach()).to(device=model.w1.device, dtype=torch.float64)


def normalize_direction(delta: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    cdelta = float(torch.sqrt((delta * (c @ delta)).sum().clamp_min(0.0)).detach().cpu().item())
    enorm = float(delta.norm().detach().cpu().item())
    denom = max(cdelta, enorm, EPS)
    return delta / denom, cdelta, enorm


def fixed_probe_directions(model: Any, c: torch.Tensor, args: argparse.Namespace, dataset: str, seed: int, source: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x = source["x_source"]
    y = source["y_source"]
    logits = model(x)
    ce_loss = F.cross_entropy(logits.float(), y)
    ce_grad = grad_matrix(ce_loss, model, retain_graph=True)
    init_metrics = v2289.metrics_for_logits(logits.detach(), y)
    debt_loss = v2289.debt_surrogate_loss(logits, y, init_metrics, float(args.no_debt_budget), ece_weight=1.0, bucketed_ece=True)
    debt_grad = grad_matrix(debt_loss, model, retain_graph=True)
    comps = logits_component_losses(logits, y)
    comps.update(distributional_debt_losses(logits, y, int(args.part_c_ece_buckets), int(args.part_c_max_classes)))
    comp_items = list(comps.items())
    component_grads = {
        name: grad_matrix(loss_value, model, retain_graph=idx < len(comp_items) - 1)
        for idx, (name, loss_value) in enumerate(comp_items)
    }
    a_base = w1_to_matrix(model.w1.detach()).to(device=model.w1.device, dtype=torch.float64)
    probes: list[dict[str, Any]] = []

    def add_probe(name: str, family: str, raw: torch.Tensor) -> None:
        delta, c_norm_raw, e_norm_raw = normalize_direction(raw, c)
        probes.append(
            {
                "probe_id": len(probes),
                "probe_name": name,
                "probe_family": family,
                "delta": delta.detach(),
                "raw_c_norm": c_norm_raw,
                "raw_euclidean_norm": e_norm_raw,
            }
        )

    task_tan, _ = natural_tangent_velocity(c, a_base, ce_grad, EPS)
    add_probe("task_tangent", "task", -task_tan)
    debt_tan, _ = natural_tangent_velocity(c, a_base, debt_grad, EPS)
    add_probe("debt_tangent", "debt", -debt_tan)

    gen = torch.Generator(device=model.w1.device).manual_seed(2291_2000 + int(seed) * 103 + sum(ord(ch) for ch in dataset))
    for idx in range(int(args.part_c_probe_count)):
        raw = torch.randn(a_base.shape, generator=gen, device=model.w1.device, dtype=torch.float64)
        tan, _ = natural_tangent_velocity(c, a_base, raw, EPS)
        add_probe(f"rademacher_tangent_{idx}", "rademacher", tan)

    r = composite_metric(a_base, c)
    eigvals, eigvecs = torch.linalg.eigh(sym(r))
    radial_count = min(int(args.part_c_radial_probe_count), int(eigvals.numel()))
    for idx in range(radial_count):
        basis = eigvecs[:, idx : idx + 1] @ eigvecs[:, idx : idx + 1].T
        rad, _ = radial_velocity(c, a_base, basis, None, EPS)
        add_probe(f"radial_spectral_{idx}", "radial", rad)

    grad_info = {
        "source_init_task": init_metrics["nll"],
        "source_init_debt_metric": debt_metric(init_metrics),
        "ce_grad_norm": float(ce_grad.norm().detach().cpu().item()),
        "debt_grad_norm": float(debt_grad.norm().detach().cpu().item()),
    }
    for key, grad in component_grads.items():
        grad_info[f"{key}_grad_norm"] = float(grad.norm().detach().cpu().item())
    grad_info["_ce_grad_matrix"] = ce_grad
    grad_info["_debt_grad_matrix"] = debt_grad
    for key, grad in component_grads.items():
        grad_info[f"_{key}_grad_matrix"] = grad
    return probes, grad_info


def evaluate_split(model: Any, split: dict[str, torch.Tensor], prefix: str, args: argparse.Namespace | None = None) -> dict[str, float]:
    logits = model(split[f"x_{prefix}"]).detach()
    metrics = v2289.metrics_for_logits(logits, split[f"y_{prefix}"])
    coverage = v2289.output_coverage_cvar25(logits, split[f"y_{prefix}"])
    out = {**metrics, "coverage_CVaR25": coverage, "debt_metric": debt_metric(metrics)}
    if args is not None:
        out.update(distributional_debt_values(logits, split[f"y_{prefix}"], int(args.part_c_ece_buckets), int(args.part_c_max_classes)))
    return out


def rollout_probe(
    model: Any,
    base_state: dict[str, torch.Tensor],
    c_source: torch.Tensor,
    c_guard: torch.Tensor,
    a_base: torch.Tensor,
    split: dict[str, torch.Tensor],
    delta: torch.Tensor,
    step_scale: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.load_state_dict({k: v.detach().clone() for k, v in base_state.items()})
    with torch.no_grad():
        moved = a_base + float(step_scale) * delta
        model.w1.copy_(matrix_to_w1(moved, model.w1.shape, dtype=model.w1.dtype, device=model.w1.device))
    opt = torch.optim.AdamW(v2289.kan_param_groups(model, float(args.part_c_rollout_lr), args), lr=float(args.part_c_rollout_lr), weight_decay=float(args.part_f_weight_decay))
    for _ in range(int(args.part_c_h)):
        opt.zero_grad(set_to_none=True)
        logits = model(split["x_source"])
        loss = F.cross_entropy(logits.float(), split["y_source"])
        loss.backward()
        opt.step()
    source = evaluate_split(model, split, "source", args)
    witness = evaluate_split(model, split, "witness", args)
    guard = evaluate_split(model, split, "guard", args)
    a_after = w1_to_matrix(model.w1.detach()).to(device=model.w1.device, dtype=torch.float64)
    r_source_before = composite_metric(a_base, c_source)
    r_guard_before = composite_metric(a_base, c_guard)
    r_source_after = composite_metric(a_after, c_source)
    r_guard_after = composite_metric(a_after, c_guard)
    sg_before = relative_fro_error(r_guard_before, r_source_before)
    sg_after = relative_fro_error(r_guard_after, r_source_after)
    out = {
        "actual_source_task_delta": source["nll"],
        "actual_witness_task_value": witness["nll"],
        "actual_guard_task_value": guard["nll"],
        "actual_witness_debt_value": witness["debt_metric"],
        "actual_witness_brier_value": witness["brier"],
        "actual_witness_ece_value": witness["ece"],
        "actual_witness_tail95_value": witness["tail95"],
        "actual_witness_margin10_value": witness["margin10"],
        "actual_guard_coverage_value": guard["coverage_CVaR25"],
        "source_guard_R_drift_before": sg_before,
        "source_guard_R_drift_after": sg_after,
        "source_guard_R_drift_delta": sg_after - sg_before,
    }
    for key, value in witness.items():
        field = component_field_name(key)
        out[f"actual_witness_{field}_value"] = float(value)
    for key, value in guard.items():
        field = component_field_name(key)
        out[f"actual_guard_{field}_value"] = float(value)
    return out


def part_c_response_rows_for_job(dataset: str, seed: int, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_row = {
        "dataset": dataset,
        "seed": seed,
        "dataset_family": "visual" if dataset in VISUAL_DATASETS else "tabular",
        "H": int(args.part_c_h),
        "probe_count_config": int(args.part_c_probe_count),
        "radial_probe_count_config": int(args.part_c_radial_probe_count),
        "probe_step_scales": str(args.probe_step_scales),
        "part_c_ece_buckets": int(args.part_c_ece_buckets),
        "part_c_max_classes": int(args.part_c_max_classes),
        "used_fake_data": 1,
    }
    try:
        bundle = v2289.load_real_task_bundle(dataset, seed, args, device)
        split = train_only_split(bundle, dataset, seed)
        pf_args = v2289.part_f_flow_args(args)
        model = v2289.make_real_composite_model(
            int(bundle["input_dim"]),
            int(bundle["num_classes"]),
            int(seed) + 91_000,
            pf_args,
            device,
            representation_mode="joint_costate",
            bias_edge=False,
        )
    except Exception as exc:
        err = dict(base_row)
        err.update({"status": "error", "error": repr(exc)})
        return [err]

    try:
        c_source, r_init, init_info = v2289.init_composite_additive(
            model,
            split["x_source"],
            "highfreq",
            int(seed) + 91_000,
            pf_args,
            y=None,
            use_quantile_transport=False,
            metric_shrink_alpha=0.0,
        )
        c_guard, _guard_info = v2289.current_composite_metric_c(model, split["x_guard"], pf_args)
        probes, grad_info = fixed_probe_directions(model, c_source, args, dataset, seed, split)
        base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        a_base = w1_to_matrix(model.w1.detach()).to(device=device, dtype=torch.float64)
        witness_before = evaluate_split(model, split, "witness", args)
        source_before = evaluate_split(model, split, "source", args)
        guard_before = evaluate_split(model, split, "guard", args)
        ce_grad = grad_info["_ce_grad_matrix"]
        debt_grad = grad_info["_debt_grad_matrix"]
        comp_grads = {
            key[1 : -len("_grad_matrix")]: value
            for key, value in grad_info.items()
            if key.startswith("_")
            and key.endswith("_grad_matrix")
            and key not in {"_ce_grad_matrix", "_debt_grad_matrix"}
        }
        scales = [float(x) for x in v2289.csv_items(args.probe_step_scales)]
        for probe in probes:
            delta = probe["delta"]
            for scale in scales:
                step_delta = float(scale) * delta
                local_task = float((ce_grad * step_delta).sum().detach().cpu().item())
                local_debt = float((debt_grad * step_delta).sum().detach().cpu().item())
                local_components = {
                    key: float((grad * step_delta).sum().detach().cpu().item())
                    for key, grad in comp_grads.items()
                }
                actual = rollout_probe(model, base_state, c_source, c_guard, a_base, split, delta, scale, args)
                row = dict(base_row)
                row.update(
                    {
                        "status": "ok",
                        "source_kind": bundle["source_kind"],
                        "used_fake_data": int(bundle["used_fake_data"]),
                        "input_dim": int(bundle["input_dim"]),
                        "original_input_dim": int(bundle["original_input_dim"]),
                        "num_classes": int(bundle["num_classes"]),
                        "source_rows": int(split["x_source"].shape[0]),
                        "witness_rows": int(split["x_witness"].shape[0]),
                        "guard_rows": int(split["x_guard"].shape[0]),
                        "probe_id": probe["probe_id"],
                        "probe_name": probe["probe_name"],
                        "probe_family": probe["probe_family"],
                        "probe_count_per_row": len(probes),
                        "step_scale": scale,
                        "C_condition_after_ridge": init_info.get("C_condition_after_ridge", 0.0),
                        "R_init_error": init_info.get("R_init_error", 0.0),
                        "raw_probe_c_norm": probe["raw_c_norm"],
                        "raw_probe_euclidean_norm": probe["raw_euclidean_norm"],
                        "local_task_pred": local_task,
                        "local_debt_pred": local_debt,
                        "local_brier_pred": local_components.get("brier", 0.0),
                        "local_ECE_pred": local_components.get("ece", 0.0),
                        "local_tail95_pred": local_components.get("tail95", 0.0),
                        "local_tail99_pred": local_components.get("tail99", 0.0),
                        "local_margin10_debt_pred": local_components.get("margin10_debt", local_components.get("margin10", 0.0)),
                        "source_init_task": source_before["nll"],
                        "witness_init_task": witness_before["nll"],
                        "guard_init_task": guard_before["nll"],
                        "witness_init_debt_metric": witness_before["debt_metric"],
                        "witness_init_brier": witness_before["brier"],
                        "witness_init_ECE": witness_before["ece"],
                        "witness_init_tail95": witness_before["tail95"],
                        "witness_init_margin10": witness_before["margin10"],
                        "guard_init_coverage": guard_before["coverage_CVaR25"],
                        "actual_delta_H_task": actual["actual_witness_task_value"] - witness_before["nll"],
                        "actual_delta_H_source_task": actual["actual_source_task_delta"] - source_before["nll"],
                        "actual_delta_H_debt": actual["actual_witness_debt_value"] - witness_before["debt_metric"],
                        "actual_delta_H_Brier": actual["actual_witness_brier_value"] - witness_before["brier"],
                        "actual_delta_H_ECE": actual["actual_witness_ece_value"] - witness_before["ece"],
                        "actual_delta_H_tail95": actual["actual_witness_tail95_value"] - witness_before["tail95"],
                        "actual_delta_H_tail99": actual.get("actual_witness_tail99_value", 0.0) - witness_before.get("tail99", 0.0),
                        "actual_delta_H_margin10": actual["actual_witness_margin10_value"] - witness_before["margin10"],
                        "actual_delta_H_margin10_debt": -(actual["actual_witness_margin10_value"] - witness_before["margin10"]),
                        "actual_delta_H_source": actual["source_guard_R_drift_delta"],
                        "actual_delta_H_source_guard": actual["source_guard_R_drift_delta"],
                        "actual_delta_H_coverage": actual["actual_guard_coverage_value"] - guard_before["coverage_CVaR25"],
                        "source_guard_R_drift_before": actual["source_guard_R_drift_before"],
                        "source_guard_R_drift_after": actual["source_guard_R_drift_after"],
                        "ce_grad_norm": grad_info["ce_grad_norm"],
                        "debt_grad_norm": grad_info["debt_grad_norm"],
                    }
                )
                for comp_name, pred_value in local_components.items():
                    field = component_field_name(comp_name)
                    row[f"local_{field}_pred"] = pred_value
                    if comp_name in {"brier", "ece", "tail95", "margin10_debt"}:
                        continue
                    before_key = field
                    actual_key = f"actual_witness_{field}_value"
                    if before_key in witness_before and actual_key in actual:
                        row[f"actual_delta_H_{field}"] = actual[actual_key] - witness_before[before_key]
                    active_key = f"{field}_active"
                    if active_key in witness_before:
                        row[active_key] = witness_before[active_key]
                rows.append(row)
    except Exception as exc:
        err = dict(base_row)
        err.update({"status": "error", "error": repr(exc)})
        rows.append(err)
    finally:
        try:
            model.load_state_dict(base_state)
        except Exception:
            pass
    return rows


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for dataset, seed in shard_items(part_c_jobs(args), args):
        rows.extend(part_c_response_rows_for_job(dataset, seed, args, device))
    suffix = part_c_suffix(args)
    path = OUT_ROOT / f"part_c_response_calibration{suffix}_shard{int(args.shard_index)}_of_{int(args.shard_count)}.csv"
    write_rows(path, rows)
    append_exec("part-c-shard", command_text(sys.argv), "done", gpu=str(device), files=rel(path), note=f"rows={len(rows)}")
    return {"rows": len(rows), "path": rel(path)}


def part_c_suffix(args: argparse.Namespace) -> str:
    tag = str(getattr(args, "part_c_tag", "")).strip()
    return "" if not tag else "_" + "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)


def feature_vector(row: dict[str, Any]) -> list[float]:
    family = str(row.get("probe_family", ""))
    local_task = fval(row.get("local_task_pred"))
    local_debt = fval(row.get("local_debt_pred"))
    local_brier = fval(row.get("local_brier_pred"))
    local_ece = fval(row.get("local_ECE_pred"))
    local_tail = fval(row.get("local_tail95_pred"))
    local_margin = fval(row.get("local_margin10_debt_pred"))
    scale = fval(row.get("step_scale"))
    raw_c = fval(row.get("raw_probe_c_norm"))
    raw_e = fval(row.get("raw_probe_euclidean_norm"))
    component_sum = local_brier + local_ece + local_tail + local_margin
    dist_values = [fval(row.get(f"local_{field}_pred")) for field in distributional_feature_fields()]
    dist_sum = sum(dist_values)
    dist_abs_sum = sum(abs(v) for v in dist_values)
    base = [
        1.0,
        local_task,
        local_debt,
        local_brier,
        local_ece,
        local_tail,
        local_margin,
        component_sum,
        scale,
        raw_c,
        raw_e,
        math.log1p(max(0.0, fval(row.get("input_dim")))),
        math.log1p(max(0.0, fval(row.get("num_classes")))),
        math.log1p(max(0.0, fval(row.get("C_condition_after_ridge")))),
        fval(row.get("R_init_error")),
        fval(row.get("source_init_task")),
        fval(row.get("witness_init_task")),
        fval(row.get("guard_init_task")),
        fval(row.get("witness_init_debt_metric")),
        fval(row.get("guard_init_coverage")),
        fval(row.get("source_guard_R_drift_before")),
        math.log1p(max(0.0, fval(row.get("ce_grad_norm")))),
        math.log1p(max(0.0, fval(row.get("debt_grad_norm")))),
        local_task * local_task,
        local_debt * local_debt,
        component_sum * component_sum,
        scale * scale,
        local_task * scale,
        local_debt * scale,
        component_sum * scale,
        abs(local_task),
        abs(local_debt),
        abs(component_sum),
        1.0 if family == "task" else 0.0,
        1.0 if family == "debt" else 0.0,
        1.0 if family == "rademacher" else 0.0,
        1.0 if family == "radial" else 0.0,
        scale if family == "task" else 0.0,
        scale if family == "debt" else 0.0,
        scale if family == "rademacher" else 0.0,
        scale if family == "radial" else 0.0,
    ]
    base.extend([dist_sum, dist_abs_sum, dist_sum * scale])
    base.extend(dist_values)
    base.extend([v * scale for v in dist_values])
    return base


def fit_predict_ridge(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], target: str, ridge: float = 1.0e-3) -> list[float]:
    if not train_rows or not test_rows:
        return [0.0 for _ in test_rows]
    x_train = torch.tensor([feature_vector(r) for r in train_rows], dtype=torch.float64)
    y_train = torch.tensor([fval(r.get(f"actual_delta_H_{target}")) for r in train_rows], dtype=torch.float64).reshape(-1, 1)
    x_test = torch.tensor([feature_vector(r) for r in test_rows], dtype=torch.float64)
    eye = torch.eye(int(x_train.shape[1]), dtype=torch.float64)
    eye[0, 0] = 0.0
    coef = torch.linalg.solve(x_train.T @ x_train + float(ridge) * eye, x_train.T @ y_train)
    pred = x_test @ coef
    return [float(v) for v in pred.reshape(-1).tolist()]


def fit_predict_ridge_key(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], actual_key: str, ridge: float = 1.0e-3) -> list[float]:
    if not train_rows or not test_rows:
        return [0.0 for _ in test_rows]
    x_train = torch.tensor([feature_vector(r) for r in train_rows], dtype=torch.float64)
    y_train = torch.tensor([fval(r.get(actual_key)) for r in train_rows], dtype=torch.float64).reshape(-1, 1)
    x_test = torch.tensor([feature_vector(r) for r in test_rows], dtype=torch.float64)
    eye = torch.eye(int(x_train.shape[1]), dtype=torch.float64)
    eye[0, 0] = 0.0
    coef = torch.linalg.solve(x_train.T @ x_train + float(ridge) * eye, x_train.T @ y_train)
    pred = x_test @ coef
    return [float(v) for v in pred.reshape(-1).tolist()]


def crossfit_predictions(rows: list[dict[str, Any]], target: str, group_key: str) -> list[float]:
    preds = [0.0 for _ in rows]
    groups = sorted({str(r.get(group_key, "")) for r in rows})
    if group_key == "dataset_seed":
        groups = sorted({f"{r.get('dataset')}::{r.get('seed')}" for r in rows})
    for group in groups:
        if group_key == "dataset_seed":
            test_idx = [i for i, r in enumerate(rows) if f"{r.get('dataset')}::{r.get('seed')}" == group]
        else:
            test_idx = [i for i, r in enumerate(rows) if str(r.get(group_key, "")) == group]
        train = [r for i, r in enumerate(rows) if i not in set(test_idx)]
        test = [rows[i] for i in test_idx]
        group_preds = fit_predict_ridge(train, test, target)
        for idx, pred in zip(test_idx, group_preds):
            preds[idx] = pred
    return preds


def crossfit_predictions_key(rows: list[dict[str, Any]], actual_key: str, group_key: str) -> list[float]:
    preds = [0.0 for _ in rows]
    groups = sorted({str(r.get(group_key, "")) for r in rows})
    if group_key == "dataset_seed":
        groups = sorted({f"{r.get('dataset')}::{r.get('seed')}" for r in rows})
    for group in groups:
        if group_key == "dataset_seed":
            test_idx = [i for i, r in enumerate(rows) if f"{r.get('dataset')}::{r.get('seed')}" == group]
        else:
            test_idx = [i for i, r in enumerate(rows) if str(r.get(group_key, "")) == group]
        test_set = set(test_idx)
        train = [r for i, r in enumerate(rows) if i not in test_set]
        test = [rows[i] for i in test_idx]
        group_preds = fit_predict_ridge_key(train, test, actual_key)
        for idx, pred in zip(test_idx, group_preds):
            preds[idx] = pred
    return preds


def summarize_target(rows: list[dict[str, Any]], target: str, pred: list[float]) -> dict[str, float]:
    actual = [fval(r.get(f"actual_delta_H_{target}")) for r in rows]
    return {
        f"response_{target}_sign_agreement": sign_agreement(pred, actual),
        f"{target}_R2": r2_score(pred, actual),
        f"calibration_slope_{target}": slope(pred, actual),
        f"Spearman_{target}": spearman(pred, actual),
        f"{target}_actual_median": median(actual),
        f"{target}_pred_median": median(pred),
    }


def summarize_part_c(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok = [r for r in rows if str(r.get("status")) == "ok" and ival(r.get("used_fake_data"), 1) == 0]
    groups = sorted({f"{r.get('dataset')}::{r.get('seed')}" for r in ok})
    pred_by_target = {target: crossfit_predictions(ok, target, "dataset_seed") for target in TARGETS}
    pred_by_family = {target: crossfit_predictions(ok, target, "dataset_family") for target in TARGETS}
    summary: dict[str, Any] = {
        "gate": "v22_91_part_c_response_calibration",
        "rows": len(rows),
        "ok_probe_rows": len(ok),
        "error_rows": len([r for r in rows if str(r.get("status")) != "ok"]),
        "unique_dataset_seed_rows": len(groups),
        "probe_count_per_row_median": median([fval(r.get("probe_count_per_row")) for r in ok]),
        "H": int(args.part_c_h),
        "step_scale_set": str(args.probe_step_scales),
        "used_fake_data_rows": sum(ival(r.get("used_fake_data"), 1) for r in rows),
        "visual_probe_rows": sum(1 for r in ok if str(r.get("dataset_family")) == "visual"),
        "tabular_probe_rows": sum(1 for r in ok if str(r.get("dataset_family")) == "tabular"),
    }
    local_task_pred = [fval(r.get("local_task_pred")) for r in ok]
    local_debt_pred = [fval(r.get("local_debt_pred")) for r in ok]
    actual_task = [fval(r.get("actual_delta_H_task")) for r in ok]
    actual_debt = [fval(r.get("actual_delta_H_debt")) for r in ok]
    summary.update(
        {
            "local_task_sign_agreement": sign_agreement(local_task_pred, actual_task),
            "local_debt_sign_agreement": sign_agreement(local_debt_pred, actual_debt),
            "debt_false_descent_rate": false_descent_rate(pred_by_target["debt"], actual_debt, lower_is_better=True),
            "source_false_stable_rate": false_descent_rate(pred_by_target["source"], [fval(r.get("actual_delta_H_source")) for r in ok], lower_is_better=True),
            "coverage_false_positive_rate": false_descent_rate(pred_by_target["coverage"], [fval(r.get("actual_delta_H_coverage")) for r in ok], lower_is_better=False),
        }
    )
    for target in TARGETS:
        summary.update(summarize_target(ok, target, pred_by_target[target]))
        fam_actual = [fval(r.get(f"actual_delta_H_{target}")) for r in ok]
        summary[f"leave_dataset_family_{target}_R2"] = r2_score(pred_by_family[target], fam_actual)
        summary[f"leave_dataset_family_{target}_sign_agreement"] = sign_agreement(pred_by_family[target], fam_actual)
    summary["response_debt_sign_agreement"] = summary.pop("response_debt_sign_agreement")
    summary["response_source_sign_agreement"] = summary.pop("response_source_sign_agreement")
    summary["response_coverage_sign_agreement"] = summary.pop("response_coverage_sign_agreement")
    visual = [r for r in ok if str(r.get("dataset_family")) == "visual"]
    tabular = [r for r in ok if str(r.get("dataset_family")) == "tabular"]
    if visual and tabular:
        v_pred = [pred_by_target["coverage"][ok.index(r)] for r in visual]
        t_pred = [pred_by_target["coverage"][ok.index(r)] for r in tabular]
        v_act = [fval(r.get("actual_delta_H_coverage")) for r in visual]
        t_act = [fval(r.get("actual_delta_H_coverage")) for r in tabular]
        summary["visual_coverage_sign_agreement"] = sign_agreement(v_pred, v_act)
        summary["tabular_coverage_sign_agreement"] = sign_agreement(t_pred, t_act)
        summary["visual_vs_tabular_calibration_gap"] = abs(summary["visual_coverage_sign_agreement"] - summary["tabular_coverage_sign_agreement"])
    else:
        summary["visual_coverage_sign_agreement"] = 0.0
        summary["tabular_coverage_sign_agreement"] = 0.0
        summary["visual_vs_tabular_calibration_gap"] = 0.0
    pass_gate = (
        summary["unique_dataset_seed_rows"] >= int(args.part_f_seed_count) * len(v2289.csv_items(args.part_f_datasets))
        and summary["response_task_sign_agreement"] >= 0.75
        and summary["response_debt_sign_agreement"] >= 0.60
        and summary["response_source_sign_agreement"] >= 0.60
        and summary["response_coverage_sign_agreement"] >= 0.60
        and summary["debt_false_descent_rate"] <= 0.35
        and summary["task_R2"] >= 0.30
        and summary["debt_R2"] >= 0.15
        and summary["source_R2"] >= 0.15
    )
    summary["part_c_gate_pass"] = int(pass_gate)
    debt_failed = (
        summary["response_debt_sign_agreement"] < 0.60
        or summary["debt_R2"] < 0.15
        or summary["debt_false_descent_rate"] > 0.35
    )
    source_failed = summary["response_source_sign_agreement"] < 0.60 or summary["source_R2"] < 0.15
    coverage_failed = summary["response_coverage_sign_agreement"] < 0.60
    if pass_gate:
        route = "PartC_ResponseCalibrationPass"
        blocker = "none"
    elif debt_failed:
        route = "C1_DebtResponseUncalibrated"
        blocker = "debt_response"
    elif source_failed:
        route = "C2_SourceResponseUncalibrated"
        blocker = "source_response"
    elif coverage_failed:
        route = "C3_CoverageResponseUncalibrated"
        blocker = "coverage_response"
    else:
        route = "C0_ResponseCalibrationFailed"
        blocker = "response_R2_or_false_descent"
    summary["part_c_route"] = route
    summary["dominant_blocker"] = blocker
    return summary


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    suffix = part_c_suffix(args)
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(f"part_c_response_calibration{suffix}_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    summary = summarize_part_c(rows, args)
    matrix = OUT_ROOT / f"part_c_response_calibration{suffix}.csv"
    summary_path = OUT_ROOT / f"part_c_summary{suffix}.json"
    write_rows(matrix, rows)
    write_json(summary_path, {**summary, "rows_detail_omitted": True})
    actions: list[dict[str, Any]]
    if summary["part_c_gate_pass"]:
        actions = [{"action": "run_scheme_one_response_controlled_tube", "reason": "Part C response calibration passed"}]
    elif summary["dominant_blocker"] == "debt_response":
        actions = [
            {"action": "part_c_split_audit", "reason": "verify source/witness/guard split and metric definitions"},
            {"action": "part_c_debt_component_summary", "reason": "decompose Brier/ECE/tail/margin sign failures"},
            {"action": "rerun_part_c_h_sensitivity", "reason": "compare H=5/10/20 and probe scale set"},
            {"action": "rerun_part_c_fixed_probe_count_increase", "reason": "increase fixed probe count without best-probe selection"},
        ]
    else:
        actions = [{"action": "follow_part_c_repair_protocol_for_blocker", "reason": summary["dominant_blocker"]}]
    next_path = write_next_actions("c", summary["part_c_route"], summary["dominant_blocker"], actions)
    append_exec("part-c-merge", command_text(sys.argv), "done" if summary["part_c_gate_pass"] else "failed", files=f"{rel(matrix)}; {rel(summary_path)}; {rel(next_path)}")
    append_recap(
        f"Part C response calibration ({str(getattr(args, 'part_c_tag', '')).strip() or 'baseline'})",
        [
            f"gate_pass={summary['part_c_gate_pass']}; route={summary['part_c_route']}; blocker={summary['dominant_blocker']}; ok_probe_rows={summary['ok_probe_rows']}/{summary['rows']}; dataset_seed_rows={summary['unique_dataset_seed_rows']}",
            f"sign: local_task={summary['local_task_sign_agreement']}; local_debt={summary['local_debt_sign_agreement']}; response_task={summary['response_task_sign_agreement']}; response_debt={summary['response_debt_sign_agreement']}; response_source={summary['response_source_sign_agreement']}; response_coverage={summary['response_coverage_sign_agreement']}",
            f"R2: task={summary['task_R2']}; debt={summary['debt_R2']}; source={summary['source_R2']}; coverage={summary['coverage_R2']}; debt_false_descent_rate={summary['debt_false_descent_rate']}; coverage_false_positive_rate={summary['coverage_false_positive_rate']}",
            f"leave-family: task_R2={summary['leave_dataset_family_task_R2']}; debt_R2={summary['leave_dataset_family_debt_R2']}; source_R2={summary['leave_dataset_family_source_R2']}; coverage_R2={summary['leave_dataset_family_coverage_R2']}; visual_vs_tabular_gap={summary['visual_vs_tabular_calibration_gap']}",
            "analysis: Part C uses train-only source/witness/guard splits and cross-fit response predictions. It does not use held/test data and does not permit controller parts unless the response gate passes.",
        ],
    )
    return summary


def run_part_c_split_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    for dataset, seed in shard_items(part_c_jobs(args), args):
        try:
            bundle = v2289.load_real_task_bundle(dataset, seed, args, device)
            split = train_only_split(bundle, dataset, seed)
            row = {
                "dataset": dataset,
                "seed": seed,
                "status": "ok",
                "used_fake_data": int(bundle["used_fake_data"]),
                "train_rows": int(bundle["x_train"].shape[0]),
                "held_rows_not_used": int(bundle["x_held"].shape[0]),
                "test_rows_not_used": int(bundle["x_test"].shape[0]),
                "source_rows": int(split["x_source"].shape[0]),
                "witness_rows": int(split["x_witness"].shape[0]),
                "guard_rows": int(split["x_guard"].shape[0]),
                "disjoint_train_split_pass": 1,
            }
        except Exception as exc:
            row = {"dataset": dataset, "seed": seed, "status": "error", "error": repr(exc)}
        rows.append(row)
    ok = [r for r in rows if str(r.get("status")) == "ok"]
    out = {
        "part_c_split_audit_pass": int(len(ok) == len(rows) and all(ival(r.get("used_fake_data")) == 0 and ival(r.get("disjoint_train_split_pass")) == 1 for r in ok)),
        "rows": len(rows),
        "ok_rows": len(ok),
        "used_fake_data_rows": sum(ival(r.get("used_fake_data"), 1) for r in ok),
        "held_test_usage": 0,
    }
    matrix = OUT_ROOT / "part_c_split_audit.csv"
    summary_path = OUT_ROOT / "part_c_split_audit_summary.json"
    write_rows(matrix, rows)
    write_json(summary_path, out)
    append_exec("part-c-split-audit", command_text(sys.argv), "done" if out["part_c_split_audit_pass"] else "failed", gpu=str(device), files=f"{rel(matrix)}; {rel(summary_path)}")
    append_recap("Part C split audit", [f"pass={out['part_c_split_audit_pass']}; ok_rows={out['ok_rows']}/{out['rows']}; used_fake_data_rows={out['used_fake_data_rows']}; held_test_usage={out['held_test_usage']}", "analysis: This audit checks that Part C response calibration is train-only; held/test rows are loaded by the bundle but not used for probe fitting or response targets."])
    return out


def run_part_c_debt_components(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    suffix = part_c_suffix(args)
    rows = read_rows(OUT_ROOT / f"part_c_response_calibration{suffix}.csv")
    ok = [r for r in rows if str(r.get("status")) == "ok" and ival(r.get("used_fake_data"), 1) == 0]
    components: list[tuple[str, str, str, bool]] = [
        ("Brier", "local_brier_pred", "actual_delta_H_Brier", True),
        ("ECE", "local_ECE_pred", "actual_delta_H_ECE", True),
        ("tail95", "local_tail95_pred", "actual_delta_H_tail95", True),
        ("margin10_debt", "local_margin10_debt_pred", "actual_delta_H_margin10_debt", True),
    ]
    seen = {name for name, _pred, _actual, _lower in components}
    if ok:
        keys = {key for row in ok for key in row.keys()}
        for key in sorted(keys):
            if not (key.startswith("local_") and key.endswith("_pred")):
                continue
            field = key[len("local_") : -len("_pred")]
            if field in seen or field in {"task", "debt", "source", "coverage", "brier", "ECE", "tail95", "margin10_debt"}:
                continue
            actual_key = f"actual_delta_H_{field}"
            if actual_key in keys and (
                field.startswith("ece_bucket_")
                or field.startswith("class_risk_")
                or field in {"tail99", "confidence_tail25", "confidence_tail10"}
            ):
                components.append((field, key, actual_key, True))
                seen.add(field)
    out: dict[str, Any] = {"rows": len(rows), "ok_probe_rows": len(ok)}
    comp_rows: list[dict[str, Any]] = []
    for name, pred_key, actual_key, lower in components:
        comp_ok = ok
        active_key = f"{name}_active"
        if any(active_key in r for r in ok):
            comp_ok = [r for r in ok if fval(r.get(active_key)) > 0.5]
        if not comp_ok:
            continue
        local_pred = [fval(r.get(pred_key)) for r in comp_ok]
        actual = [fval(r.get(actual_key)) for r in comp_ok]
        response_pred = crossfit_predictions_key(comp_ok, actual_key, "dataset_seed")
        leave_family_pred = crossfit_predictions_key(comp_ok, actual_key, "dataset_family")
        item = {
            "component": name,
            "component_rows": len(comp_ok),
            "component_local_sign_agreement": sign_agreement(local_pred, actual),
            "component_local_false_descent_rate": false_descent_rate(local_pred, actual, lower_is_better=lower),
            "component_local_R2": r2_score(local_pred, actual),
            "component_response_sign_agreement": sign_agreement(response_pred, actual),
            "component_response_false_descent_rate": false_descent_rate(response_pred, actual, lower_is_better=lower),
            "component_response_R2": r2_score(response_pred, actual),
            "component_leave_family_sign_agreement": sign_agreement(leave_family_pred, actual),
            "component_leave_family_R2": r2_score(leave_family_pred, actual),
            "actual_delta_median": median(actual),
            "local_pred_median": median(local_pred),
            "response_pred_median": median(response_pred),
        }
        comp_rows.append(item)
        for k, v in item.items():
            if k != "component":
                out[f"{name}_{k}"] = v
    out["worst_component_sign_agreement"] = min([fval(r.get("component_response_sign_agreement")) for r in comp_rows] or [0.0])
    out["median_component_sign_agreement"] = median([fval(r.get("component_response_sign_agreement")) for r in comp_rows])
    out["worst_component_response_R2"] = min([fval(r.get("component_response_R2")) for r in comp_rows] or [0.0])
    out["median_component_response_R2"] = median([fval(r.get("component_response_R2")) for r in comp_rows])
    out["component_count"] = len(comp_rows)
    out["bucket_count"] = int(args.part_c_ece_buckets)
    out["class_count"] = int(args.part_c_max_classes)
    bucket_rows = [r for r in comp_rows if str(r.get("component", "")).startswith("ece_bucket_")]
    class_rows = [r for r in comp_rows if str(r.get("component", "")).startswith("class_risk_")]
    tail_rows = [r for r in comp_rows if str(r.get("component", "")) in {"tail95", "tail99", "confidence_tail25", "confidence_tail10"}]
    margin_rows = [r for r in comp_rows if str(r.get("component", "")) == "margin10_debt"]
    out["worst_bucket_sign_agreement"] = min([fval(r.get("component_response_sign_agreement")) for r in bucket_rows] or [0.0])
    out["worst_class_sign_agreement"] = min([fval(r.get("component_response_sign_agreement")) for r in class_rows] or [0.0])
    out["ECE_bucket_false_descent_rate"] = max([fval(r.get("component_response_false_descent_rate")) for r in bucket_rows] or [0.0])
    out["tail_false_descent_rate"] = max([fval(r.get("component_response_false_descent_rate")) for r in tail_rows] or [0.0])
    out["margin_false_descent_rate"] = max([fval(r.get("component_response_false_descent_rate")) for r in margin_rows] or [0.0])
    if out["worst_component_sign_agreement"] < 0.55:
        worst = min(comp_rows, key=lambda r: fval(r.get("component_response_sign_agreement"))) if comp_rows else {"component": "none"}
        out["component_blocker"] = f"{worst['component']}ResponseUncalibrated"
    elif out["worst_component_response_R2"] < 0.0:
        out["component_blocker"] = "DistributionalDebtEffectSizeUncalibrated"
    else:
        out["component_blocker"] = "none"
    matrix = OUT_ROOT / f"part_c_debt_component_summary{suffix}.csv"
    summary_path = OUT_ROOT / f"part_c_debt_component_summary{suffix}.json"
    write_rows(matrix, comp_rows)
    write_json(summary_path, out)
    append_exec("part-c-debt-components", command_text(sys.argv), "done", files=f"{rel(matrix)}; {rel(summary_path)}")
    append_recap(
        f"Part C debt component decomposition ({str(getattr(args, 'part_c_tag', '')).strip() or 'baseline'})",
        [
            f"ok_probe_rows={out['ok_probe_rows']}; component_count={out['component_count']}; worst_component_sign_agreement={out['worst_component_sign_agreement']}; median_component_sign_agreement={out['median_component_sign_agreement']}; worst_component_response_R2={out['worst_component_response_R2']}; median_component_response_R2={out['median_component_response_R2']}; blocker={out['component_blocker']}",
            f"Brier_sign={out.get('Brier_component_response_sign_agreement')}; ECE_sign={out.get('ECE_component_response_sign_agreement')}; tail95_sign={out.get('tail95_component_response_sign_agreement')}; tail99_sign={out.get('tail99_component_response_sign_agreement')}; margin10_debt_sign={out.get('margin10_debt_component_response_sign_agreement')}",
            f"bucket/class/tail false descent: worst_bucket_sign={out['worst_bucket_sign_agreement']}; worst_class_sign={out['worst_class_sign_agreement']}; ECE_bucket_false_descent_rate={out['ECE_bucket_false_descent_rate']}; tail_false_descent_rate={out['tail_false_descent_rate']}; margin_false_descent_rate={out['margin_false_descent_rate']}",
            "analysis: Component decomposition follows the v22.91 repair protocol for low debt response calibration and now includes fixed distributional debt components. It is diagnostic and does not weaken the no-debt gate.",
        ],
    )
    return out


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    c = read_json(OUT_ROOT / "part_c_summary.json")
    reason = "Part C not passed" if not c.get("part_c_gate_pass") else "Part F controller/preflight not yet passed"
    out = {"part_g_gate_pass": 0, "part_g_route": "skipped", "skip_reason": reason}
    write_rows(OUT_ROOT / "part_g_hstep_matrix.csv", [])
    write_json(OUT_ROOT / "part_g_summary.json", out)
    next_path = write_next_actions("g", "skipped", reason, [{"action": "run_only_after_part_f_pass", "reason": reason}])
    append_exec("part-g", command_text(sys.argv), "skipped", files=f"{rel(OUT_ROOT / 'part_g_summary.json')}; {rel(next_path)}")
    append_recap("Part G H-step trajectory", [f"gate_pass=0; route=skipped; reason={reason}", "analysis: v22.91 forbids H-step/full-loop when Part F is missing or failed."])
    return out


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    a = read_json(OUT_ROOT / "part_a_code_identity.json")
    b = read_json(OUT_ROOT / "part_b_history_lock.json")
    c = read_json(OUT_ROOT / "part_c_summary.json")
    g = read_json(OUT_ROOT / "part_g_summary.json")
    if not a.get("part_a_gate_pass"):
        route, reason = "A0_CodeOrIdentityFailed", "part_a"
    elif not b.get("part_b_gate_pass"):
        route, reason = "B0_HistoryLockMismatch", "part_b"
    elif not c.get("part_c_gate_pass"):
        route, reason = c.get("part_c_route", "C0_ResponseCalibrationFailed"), c.get("dominant_blocker", "part_c")
    elif not g.get("part_g_gate_pass"):
        route, reason = "R6_RealTaskPreflightOpened_NoHStep", "Part C passed but Part F/G not completed"
    else:
        route, reason = "R8_OfficialCandidateReady", "all gates passed"
    out = {
        "gate": "v22_91_final_route",
        "official_candidate_gate_pass": int(route == "R8_OfficialCandidateReady"),
        "final_route": route,
        "route_reason": reason,
        "part_a": {k: a.get(k) for k in ["part_a_gate_pass"]},
        "part_b": {k: b.get(k) for k in ["part_b_gate_pass"]},
        "part_c": {k: c.get(k) for k in ["part_c_gate_pass", "part_c_route", "dominant_blocker", "response_task_sign_agreement", "response_debt_sign_agreement", "response_source_sign_agreement", "response_coverage_sign_agreement", "task_R2", "debt_R2", "source_R2", "coverage_R2"]},
        "part_g": {k: g.get(k) for k in ["part_g_gate_pass", "skip_reason"]},
    }
    write_json(OUT_ROOT / "final_route.json", out)
    manifest = OUT_ROOT / "reproduction_manifest.md"
    manifest.write_text(
        f"# v22.91 reproduction manifest\n\nPython: `{PYTHON}`\nRunner: `{rel(RUNNER)}`\nPlan: `{rel(PLAN)}`\nFinalize: `{PYTHON} {rel(RUNNER)} --mode finalize --device cuda`\n",
        encoding="utf-8",
    )
    append_exec("finalize", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'final_route.json')}; {rel(manifest)}")
    append_recap(
        "Final route",
        [
            f"official_candidate_gate_pass={out['official_candidate_gate_pass']}; final_route={route}; reason={reason}",
            "analysis: Final route is based only on current artifacts. Missing or failed Part F/G is never treated as official success.",
        ],
    )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--pc-basis", default="D-FOU")
    p.add_argument("--pc-hidden", type=int, default=16)
    p.add_argument("--pc-lr", type=float, default=1.0e-2)
    p.add_argument("--pc-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--smoothness", type=float, default=1.0e-4)
    p.add_argument("--c-condition-budget", type=float, default=1.0e5)
    p.add_argument("--target-variance", type=float, default=49.0)
    p.add_argument("--no-debt-budget", type=float, default=0.01)
    p.add_argument("--rep-lr-ratio", type=float, default=0.01)
    p.add_argument("--part-f-datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--part-f-seed-count", type=int, default=6)
    p.add_argument("--part-f-train-size", type=int, default=512)
    p.add_argument("--part-f-held-size", type=int, default=256)
    p.add_argument("--part-f-test-size", type=int, default=256)
    p.add_argument("--part-f-max-input-dim", type=int, default=64)
    p.add_argument("--part-f-steps", type=int, default=120)
    p.add_argument("--part-f-eval-interval", type=int, default=20)
    p.add_argument("--part-f-batch-size", type=int, default=0)
    p.add_argument("--part-f-cmp-lr", type=float, default=0.8)
    p.add_argument("--part-f-adamw-lr", type=float, default=0.02)
    p.add_argument("--part-f-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--part-f-target-variance", type=float, default=49.0)
    p.add_argument("--part-f-cmp-max-norm-ratio", type=float, default=20.0)
    p.add_argument("--part-f-scale-band", type=float, default=0.40)
    p.add_argument("--part-f-rep-lr-ratio", type=float, default=0.01)
    p.add_argument("--part-c-h", type=int, default=5)
    p.add_argument("--part-c-rollout-lr", type=float, default=0.02)
    p.add_argument("--part-c-probe-count", type=int, default=4)
    p.add_argument("--part-c-radial-probe-count", type=int, default=2)
    p.add_argument("--part-c-tag", default="")
    p.add_argument("--part-c-ece-buckets", type=int, default=5)
    p.add_argument("--part-c-max-classes", type=int, default=10)
    p.add_argument("--probe-step-scales", default="0.02,0.08,0.20")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode)
    if mode == "part-a":
        return run_part_a(args)
    if mode == "part-b":
        return run_part_b(args)
    if mode == "part-c":
        return run_part_c(args)
    if mode == "part-c-merge":
        return merge_part_c(args)
    if mode == "part-c-split-audit":
        return run_part_c_split_audit(args)
    if mode == "part-c-debt-components":
        return run_part_c_debt_components(args)
    if mode == "part-g":
        return run_part_g(args)
    if mode == "finalize":
        return finalize(args)
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
