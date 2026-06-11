"""Small calibration metric helpers used by v22.02 audit scripts."""

from __future__ import annotations

from typing import Any, Sequence


def brier_from_probs(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    total = 0.0
    count = 0
    for row, label in zip(probs, labels):
        for idx, p in enumerate(row):
            target = 1.0 if idx == int(label) else 0.0
            total += (float(p) - target) ** 2
        count += 1
    return total / count if count else 0.0


def ece_from_confidence(confidences: Sequence[Any], correct: Sequence[Any], bins: int = 15) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    pairs = [(float(c), float(ok)) for c, ok in zip(confidences, correct)]
    if not pairs:
        return 0.0
    out = 0.0
    n = float(len(pairs))
    for b in range(int(bins)):
        lo = b / float(bins)
        hi = (b + 1) / float(bins)
        bucket = [(c, ok) for c, ok in pairs if c >= lo and (c < hi or b == bins - 1 and c <= hi)]
        if not bucket:
            continue
        conf = sum(c for c, _ok in bucket) / len(bucket)
        acc = sum(ok for _c, ok in bucket) / len(bucket)
        out += len(bucket) / n * abs(conf - acc)
    return out


def calibration_unit_tests() -> list[dict[str, Any]]:
    brier = brier_from_probs([[1.0, 0.0], [0.0, 1.0]], [0, 1])
    ece = ece_from_confidence([1.0, 1.0], [1.0, 1.0], bins=2)
    return [
        {"test": "perfect_brier", "expected": 0.0, "actual": brier, "pass": int(abs(brier) < 1.0e-12)},
        {"test": "perfect_ece", "expected": 0.0, "actual": ece, "pass": int(abs(ece) < 1.0e-12)},
    ]


__all__ = ["brier_from_probs", "calibration_unit_tests", "ece_from_confidence"]
