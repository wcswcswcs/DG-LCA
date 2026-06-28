#!/usr/bin/env python3
"""Static and runtime audit helpers for v22.55 standard training loops."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


OFFICIAL_FILES = [
    ROOT / "dgkan/fu/witnessed_gradient_operator.py",
    ROOT / "dgkan/optim/witnessed_optimizer_wrapper.py",
    ROOT / "experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py",
]


def _forbidden_patterns() -> dict[str, str]:
    return {
        "apply_flat_update_called": r"apply_" + r"flat_update\s*\(",
        "p_data_write_detected": r"\.data\s*(?:\[|\.add_|\.copy_|=)",
        "copy_param_write_detected": r"(?:Parameter|param|p)\.copy_\s*\(",
        "manual_param_update_detected": r"(?:param|p)\.add_\s*\(",
        "no_grad_param_mutation_detected": r"with\s+torch\.no_grad\s*\(\)\s*:\s*(?:\n|.){0,240}(?:param|p)\.",
        "branch_replay_used_as_training": r"branch[_ -]?replay\s*\(",
        "proxy_direction_used_as_training": r"proxy[_ -]?direction\s*=",
        "candidate_action_selection_used_for_runtime": r"(?:argmax|\.topk\s*\(|winner).*candidate",
        "cohort_topk_selection_used": r"\.topk\s*\(",
        "score_selector_used": r"score[_ -]?selector\s*=",
        "class_weight_or_sampler_used_as_fu": r"(?:WeightedRandomSampler|class_weight\s*=)",
        "uses_validation_test_future_direction": r"(?:x_held|x_test|validation|future).*direction\s*=",
    }


def scan_files(paths: Iterable[Path] | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    aggregate = {name: 0 for name in _forbidden_patterns()}
    files = [p for p in (paths or OFFICIAL_FILES) if p.exists()]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in _forbidden_patterns().items():
            hits = [m.start() for m in re.finditer(pattern, text, flags=re.MULTILINE)]
            if hits:
                aggregate[name] += len(hits)
                rows.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "pattern": name,
                        "hits": len(hits),
                    }
                )
    official_loop_text = ""
    runner = ROOT / "experiments/run_v22_55_true_training_witnessed_gradient_operator_fu.py"
    if runner.exists():
        official_loop_text = runner.read_text(encoding="utf-8", errors="replace")
    standard_loop_static_scan_pass = int(
        "logits = model(xb)" in official_loop_text
        and "loss_task.backward()" in official_loop_text
        and "opt.step()" in official_loop_text
    )
    manual_update_forbidden_scan_pass = int(sum(aggregate.values()) == 0)
    return {
        "rows": rows,
        "summary": {
            **aggregate,
            "official_files_scanned": len(files),
            "standard_loop_static_scan_pass": standard_loop_static_scan_pass,
            "manual_update_forbidden_scan_pass": manual_update_forbidden_scan_pass,
        },
    }


class RuntimeTrainingLoopAudit:
    """Snapshot-based detector for parameter writes before optimizer.step()."""

    def __init__(self, params: Iterable[Any]) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.before_step: list[Any] = []
        self.manual_param_update_detected = 0
        self.no_grad_param_mutation_detected = 0
        self.apply_flat_update_called = 0
        self.p_data_write_detected = 0
        self.copy_param_write_detected = 0

    def snapshot_before_backward(self) -> None:
        self.before_step = [p.detach().clone() for p in self.params]

    def check_before_optimizer_step(self) -> None:
        if not self.before_step or len(self.before_step) != len(self.params):
            return
        for param, before in zip(self.params, self.before_step):
            delta = (param.detach() - before.to(device=param.device, dtype=param.dtype)).abs().max()
            val = float(delta.item()) if hasattr(delta, "item") else float(delta)
            if math.isfinite(val) and val > 1.0e-12:
                self.manual_param_update_detected = 1
                self.no_grad_param_mutation_detected = 1
                break

    def as_dict(self) -> dict[str, int]:
        return {
            "manual_param_update_detected": self.manual_param_update_detected,
            "no_grad_param_mutation_detected": self.no_grad_param_mutation_detected,
            "apply_flat_update_called": self.apply_flat_update_called,
            "p_data_write_detected": self.p_data_write_detected,
            "copy_param_write_detected": self.copy_param_write_detected,
        }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = scan_files()
    if args.out:
        write_rows(Path(args.out), result["rows"] or [{"status": "no_forbidden_hits"}])
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if result["summary"]["manual_update_forbidden_scan_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
