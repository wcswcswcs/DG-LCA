"""Legal precommit selector audit helpers for v22.02.

Selector scores are computed from caller-supplied train-stream/shadow features.
Outcome labels may be passed only for offline evaluation of a frozen selector.
"""

from __future__ import annotations

from typing import Any, Iterable

from dgkan.fu.source_chain import finite_float


LEGAL_PRECOMMIT_FEATURES = {
    "train_loss_h100",
    "train_loss_h400",
    "split_transfer_gain_h400",
    "B2_transfer_gain_h400",
    "B3_safety_gain_h400",
    "ActuationR2_h400",
    "source_state_short_long_agreement_h400",
    "source_state_current_cos_h400",
    "optimizer_source_conflict_cos_h400",
    "readout_source_energy_h400",
    "basis_source_energy_h400",
    "LineC_fast_loss_h400",
}

FORBIDDEN_SELECTOR_FEATURE_TOKENS = ("h800", "h1600", "h2400", "h3200", "h4000", "h4800", "val_", "test_", "future", "query")


def is_legal_precommit_feature(name: str) -> int:
    text = str(name)
    if any(tok in text for tok in FORBIDDEN_SELECTOR_FEATURE_TOKENS):
        return 0
    return int(text in LEGAL_PRECOMMIT_FEATURES or text.endswith("_h100") or text.endswith("_h400"))


def roc_auc(scores: Iterable[Any], labels: Iterable[Any]) -> float | str:
    pairs = [(finite_float(s), int(finite_float(y, 0.0) > 0.0)) for s, y in zip(scores, labels)]
    pairs = [(s, y) for s, y in pairs if s == s]
    positives = [s for s, y in pairs if y]
    negatives = [s for s, y in pairs if not y]
    if not positives or not negatives:
        return ""
    wins = 0.0
    total = 0.0
    for ps in positives:
        for ns in negatives:
            total += 1.0
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total if total else ""


def precision_at_k(scores: Iterable[Any], labels: Iterable[Any], k: int) -> float | str:
    pairs = [(finite_float(s), int(finite_float(y, 0.0) > 0.0)) for s, y in zip(scores, labels)]
    pairs = [(s, y) for s, y in pairs if s == s]
    if not pairs or k <= 0:
        return ""
    chosen = sorted(pairs, key=lambda item: item[0], reverse=True)[: min(k, len(pairs))]
    return sum(y for _s, y in chosen) / len(chosen)


def evaluate_selector(rows: list[dict[str, Any]], feature: str, label: str = "productive_h4800_group", k: int = 2) -> dict[str, Any]:
    legal = is_legal_precommit_feature(feature)
    scores = [r.get(feature) for r in rows]
    labels = [r.get(label) for r in rows]
    auc = roc_auc(scores, labels) if legal else ""
    precision = precision_at_k(scores, labels, k) if legal else ""
    return {
        "selector_feature_name": feature,
        "selector_train_only": legal,
        "selector_uses_future": int(not legal),
        "selector_AUC_retained_h4800_on_shadow_pool": auc,
        "selector_precision_at_k": precision,
        "selected_candidate_count": min(k, len(rows)) if legal else 0,
        "precommit_selector_pass": int(legal and precision != "" and float(precision) >= 0.50),
    }


def precommit_selector_unit_tests() -> list[dict[str, Any]]:
    rows = [
        {"feature": "train_loss_h400", "expected_legal": 1},
        {"feature": "train_loss_h800", "expected_legal": 0},
        {"feature": "source_h4800", "expected_legal": 0},
        {"feature": "B2_transfer_gain_h400", "expected_legal": 1},
    ]
    for row in rows:
        row["actual_legal"] = is_legal_precommit_feature(str(row["feature"]))
        row["pass"] = int(row["actual_legal"] == row["expected_legal"])
    auc = roc_auc([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
    rows.append({"feature": "auc_known_order", "expected_legal": 1, "actual_legal": auc, "pass": int(abs(float(auc) - 1.0) < 1.0e-12)})
    return rows


__all__ = [
    "FORBIDDEN_SELECTOR_FEATURE_TOKENS",
    "LEGAL_PRECOMMIT_FEATURES",
    "evaluate_selector",
    "is_legal_precommit_feature",
    "precision_at_k",
    "precommit_selector_unit_tests",
    "roc_auc",
]
