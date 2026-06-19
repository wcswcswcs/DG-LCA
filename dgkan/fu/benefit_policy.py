"""Small dependency-light benefit policy utilities for v22.18."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def auc_binary(y_true: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    pos = sum(int(y) for _s, y in pairs)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for idx, (_score, y) in enumerate(pairs, start=1):
        if int(y):
            rank_sum += idx
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def spearman(y: list[float], scores: list[float]) -> float:
    if len(y) < 3:
        return 0.0
    yr = _ranks(np.asarray(y, dtype=float))
    sr = _ranks(np.asarray(scores, dtype=float))
    yc = yr - yr.mean()
    sc = sr - sr.mean()
    denom = float(np.linalg.norm(yc) * np.linalg.norm(sc))
    return float(yc.dot(sc) / denom) if denom > 0 else 0.0


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


@dataclass
class LinearBenefitPolicy:
    feature_names: list[str]
    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    intercept: float

    def predict_one(self, row: dict[str, Any]) -> float:
        x = np.asarray([safe_float(row.get(name)) for name in self.feature_names], dtype=float)
        z = (x - self.mean) / self.scale
        return float(z.dot(self.coef) + self.intercept)

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.predict_one(row) for row in rows]


def fit_ridge(rows: list[dict[str, Any]], y: list[float], feature_names: list[str], ridge: float = 1.0e-3) -> LinearBenefitPolicy:
    if not rows:
        return LinearBenefitPolicy(feature_names, np.zeros(len(feature_names)), np.zeros(len(feature_names)), np.ones(len(feature_names)), 0.0)
    x = np.asarray([[safe_float(row.get(name)) for name in feature_names] for row in rows], dtype=float)
    target = np.asarray(y, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    z = (x - mean) / scale
    z_aug = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    reg = np.eye(z_aug.shape[1]) * float(ridge)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(z_aug.T @ z_aug + reg) @ z_aug.T @ target
    return LinearBenefitPolicy(feature_names, coef[1:], mean, scale, float(coef[0]))


def validation_metrics(rows: list[dict[str, Any]], y: list[float], positive: list[int], policy: LinearBenefitPolicy) -> dict[str, float]:
    scores = policy.predict(rows)
    return {
        "rows": float(len(rows)),
        "benefit_AUC_binary_positive": auc_binary([int(v) for v in positive], scores),
        "benefit_spearman": spearman([float(v) for v in y], scores),
        "predicted_benefit_mean": float(np.mean(scores)) if scores else 0.0,
        "realized_benefit_mean": float(np.mean(y)) if y else 0.0,
    }
