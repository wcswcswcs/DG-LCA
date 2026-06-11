"""Audit-only debt accounting helpers for long-horizon FU runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


EPS = 1.0e-12


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def recovery_rate(peak: Any, final: Any) -> float | str:
    p = finite_float(peak)
    f = finite_float(final)
    if not math.isfinite(p) or not math.isfinite(f) or p <= 0.0:
        return ""
    return 1.0 - f / max(EPS, p)


def debt_peak_final(values: Iterable[Any], baseline: Any = 0.0) -> dict[str, float | str]:
    base = finite_float(baseline, 0.0)
    debts = [max(0.0, finite_float(v, base) - base) for v in values]
    if not debts:
        return {"debt_peak": "", "debt_final": "", "debt_recovery": ""}
    peak = max(debts)
    final = debts[-1]
    return {"debt_peak": peak, "debt_final": final, "debt_recovery": recovery_rate(peak, final)}


def auc_time_ratio(candidate_auc: Any, best_control_auc: Any) -> float | str:
    cand = finite_float(candidate_auc)
    ctrl = finite_float(best_control_auc)
    if not math.isfinite(cand) or not math.isfinite(ctrl) or ctrl <= 0.0:
        return ""
    return cand / max(EPS, ctrl)


@dataclass(frozen=True)
class DebtTest:
    test: str
    expected: Any
    actual: Any
    passed: int

    def row(self) -> dict[str, Any]:
        return {"test": self.test, "expected": self.expected, "actual": self.actual, "pass": self.passed}


def debt_accounting_unit_tests() -> list[dict[str, Any]]:
    normal = recovery_rate(2.0, 0.5)
    curve = debt_peak_final([1.0, 3.0, 2.0], baseline=1.0)
    auc = auc_time_ratio(1.2, 1.0)
    tests = [
        DebtTest("recovery_rate_normal", 0.75, normal, int(abs(float(normal) - 0.75) < 1.0e-12)).row(),
        DebtTest("recovery_rate_zero_peak_undefined", "", recovery_rate(0.0, 0.0), int(recovery_rate(0.0, 0.0) == "")).row(),
        DebtTest("debt_peak_final_curve", "2.0/1.0/0.5", f"{curve['debt_peak']}/{curve['debt_final']}/{curve['debt_recovery']}", int(curve["debt_peak"] == 2.0 and curve["debt_final"] == 1.0 and abs(float(curve["debt_recovery"]) - 0.5) < 1.0e-12)).row(),
        DebtTest("auc_time_ratio", 1.2, auc, int(abs(float(auc) - 1.2) < 1.0e-12)).row(),
        DebtTest("auc_time_missing_control", "", auc_time_ratio(1.0, 0.0), int(auc_time_ratio(1.0, 0.0) == "")).row(),
    ]
    return tests


__all__ = [
    "EPS",
    "auc_time_ratio",
    "debt_accounting_unit_tests",
    "debt_peak_final",
    "finite_float",
    "recovery_rate",
]
