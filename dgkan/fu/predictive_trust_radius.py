"""Predictive trust-radius helpers for DG-KAN v23.02.

The classes in this module are deliberately small and auditable. They do not
read held/test data and they expose explicit state dictionaries so experiment
runners can snapshot and roll back trust state together with model/optimizer
state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable


DEFAULT_SCALE_GRID: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0)
DEBT_COMPONENTS: tuple[str, ...] = ("Brier", "ECE", "tail95", "tail99", "margin10")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    x = max(-40.0, min(40.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def safe_scale_from_quadratic(a: float, b: float, *, grid: Iterable[float] = DEFAULT_SCALE_GRID, budget: float = 0.0) -> float:
    """Return the largest grid scale satisfying ``a*s + b*s^2 <= budget``."""
    aa = _finite(a)
    bb = _finite(b)
    for scale in sorted((float(x) for x in grid), reverse=True):
        if aa * scale + bb * scale * scale <= float(budget):
            return float(scale)
    return 0.0


@dataclass
class AnalyticQuadraticTrust:
    """Componentwise quadratic trust predictor.

    The runner provides per-component directional linear and quadratic debt
    coefficients. The predictor chooses the largest scale that is safe for all
    recorded components on a fixed, auditable scale grid.
    """

    b_min: float = -1.0e3
    b_max: float = 1.0e3
    safety_margin: float = 0.0
    scale_grid: tuple[float, ...] = DEFAULT_SCALE_GRID

    def predict(self, features: dict[str, Any]) -> float:
        scale = 1.0
        for component in DEBT_COMPONENTS:
            a = _finite(features.get(f"predicted_debt_delta_linear_{component}", 0.0))
            b = _finite(features.get(f"predicted_debt_delta_quadratic_{component}", 0.0))
            b = max(float(self.b_min), min(float(self.b_max), b))
            component_scale = safe_scale_from_quadratic(a, b, grid=self.scale_grid, budget=-float(self.safety_margin))
            scale = min(scale, component_scale)
        return float(scale)

    def update(self, accepted: bool, scale: float, features: dict[str, Any] | None = None) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"b_min": self.b_min, "b_max": self.b_max, "safety_margin": self.safety_margin, "scale_grid": list(self.scale_grid)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.b_min = _finite(state.get("b_min", self.b_min), self.b_min)
        self.b_max = _finite(state.get("b_max", self.b_max), self.b_max)
        self.safety_margin = _finite(state.get("safety_margin", self.safety_margin), self.safety_margin)
        if isinstance(state.get("scale_grid"), list):
            self.scale_grid = tuple(float(x) for x in state["scale_grid"])


@dataclass
class OnlineLogisticTrust:
    """Tiny online logistic predictor over handcrafted scalar features."""

    feature_names: tuple[str, ...] = (
        "risk_score",
        "update_norm_Gedge",
        "qpop_gate_density",
        "recent_accept_rate_ema",
        "recent_scale_mean_ema",
    )
    lr: float = 0.05
    threshold: float = 0.90
    scale_grid: tuple[float, ...] = DEFAULT_SCALE_GRID
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0

    def score(self, features: dict[str, Any], scale: float = 1.0) -> float:
        z = self.bias + math.log(max(float(scale), 1.0e-12))
        for name in self.feature_names:
            z += self.weights.get(name, 0.0) * _finite(features.get(name, 0.0))
        return _sigmoid(z)

    def predict(self, features: dict[str, Any]) -> float:
        for scale in sorted(self.scale_grid, reverse=True):
            if self.score(features, scale) >= float(self.threshold):
                return float(scale)
        return 0.0

    def update(self, accepted: bool, scale: float, features: dict[str, Any] | None = None) -> None:
        feats = features or {}
        y = 1.0 if bool(accepted) else 0.0
        p = self.score(feats, scale)
        err = y - p
        self.bias += float(self.lr) * err
        for name in self.feature_names:
            self.weights[name] = self.weights.get(name, 0.0) + float(self.lr) * err * _finite(feats.get(name, 0.0))

    def state_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "bias": self.bias, "lr": self.lr, "threshold": self.threshold}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.weights = {str(k): _finite(v) for k, v in dict(state.get("weights", {})).items()}
        self.bias = _finite(state.get("bias", self.bias), self.bias)
        self.lr = _finite(state.get("lr", self.lr), self.lr)
        self.threshold = _finite(state.get("threshold", self.threshold), self.threshold)


@dataclass
class PIDTrustRadiusController:
    """Simple adaptive trust radius driven by recent accept/reject outcomes."""

    radius: float = 0.05
    target_accept_rate: float = 0.35
    eta_p: float = 0.10
    eta_i: float = 0.01
    integral_error: float = 0.0
    radius_min: float = 1.0e-4
    radius_max: float = 1.0

    def predict(self, features: dict[str, Any]) -> float:
        norm = max(_finite(features.get("update_norm_Gedge", features.get("update_norm_L2", 1.0))), 1.0e-12)
        return float(max(0.0, min(1.0, self.radius / norm)))

    def update(self, accepted: bool, scale: float, features: dict[str, Any] | None = None) -> None:
        accept = 1.0 if bool(accepted) else 0.0
        err = accept - float(self.target_accept_rate)
        self.integral_error += err
        log_radius = math.log(max(self.radius, self.radius_min))
        log_radius += float(self.eta_p) * err + float(self.eta_i) * self.integral_error
        self.radius = max(self.radius_min, min(self.radius_max, math.exp(log_radius)))

    def state_dict(self) -> dict[str, Any]:
        return {
            "radius": self.radius,
            "target_accept_rate": self.target_accept_rate,
            "eta_p": self.eta_p,
            "eta_i": self.eta_i,
            "integral_error": self.integral_error,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.radius = _finite(state.get("radius", self.radius), self.radius)
        self.target_accept_rate = _finite(state.get("target_accept_rate", self.target_accept_rate), self.target_accept_rate)
        self.eta_p = _finite(state.get("eta_p", self.eta_p), self.eta_p)
        self.eta_i = _finite(state.get("eta_i", self.eta_i), self.eta_i)
        self.integral_error = _finite(state.get("integral_error", self.integral_error), self.integral_error)


@dataclass
class ComponentRiskTrust:
    """Persistent component-risk trust baseline."""

    alpha0: float = 0.25
    rho: float = 0.90
    component_weight: float = 25.0
    violations: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in DEBT_COMPONENTS})

    def predict(self, features: dict[str, Any]) -> float:
        risk = sum(max(0.0, v) for v in self.violations.values())
        return float(max(0.0, min(1.0, self.alpha0 / (1.0 + float(self.component_weight) * risk))))

    def update(self, accepted: bool, scale: float, features: dict[str, Any] | None = None) -> None:
        feats = features or {}
        for component in DEBT_COMPONENTS:
            key = f"observed_debt_delta_{component}"
            violation = max(0.0, _finite(feats.get(key, 0.0)))
            old = self.violations.get(component, 0.0)
            self.violations[component] = float(self.rho) * old + (1.0 - float(self.rho)) * violation

    def state_dict(self) -> dict[str, Any]:
        return {"alpha0": self.alpha0, "rho": self.rho, "component_weight": self.component_weight, "violations": dict(self.violations)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.alpha0 = _finite(state.get("alpha0", self.alpha0), self.alpha0)
        self.rho = _finite(state.get("rho", self.rho), self.rho)
        self.component_weight = _finite(state.get("component_weight", self.component_weight), self.component_weight)
        self.violations = {str(k): _finite(v) for k, v in dict(state.get("violations", self.violations)).items()}


def predictor_smoke_test() -> dict[str, int | float]:
    features = {f"predicted_debt_delta_linear_{c}": -0.01 for c in DEBT_COMPONENTS}
    features.update({f"predicted_debt_delta_quadratic_{c}": 0.02 for c in DEBT_COMPONENTS})
    features.update({"update_norm_Gedge": 0.1, "qpop_gate_density": 0.5, "recent_accept_rate_ema": 0.5, "recent_scale_mean_ema": 0.1, "risk_score": 0.0})
    analytic = AnalyticQuadraticTrust()
    online = OnlineLogisticTrust()
    pid = PIDTrustRadiusController()
    component = ComponentRiskTrust()
    scales = [analytic.predict(features), online.predict(features), pid.predict(features), component.predict(features)]
    state = pid.state_dict()
    pid.update(False, scales[2], features)
    changed = int(pid.radius != state["radius"])
    pid.load_state_dict(state)
    restored = int(abs(pid.radius - state["radius"]) <= 1.0e-15)
    return {
        "predictive_trust_analytic_available": int(scales[0] >= 0.0),
        "predictive_trust_online_available": int(scales[1] >= 0.0),
        "predictive_trust_pid_available": int(scales[2] >= 0.0),
        "predictive_trust_component_risk_available": int(scales[3] >= 0.0),
        "predictor_state_changes": changed,
        "predictor_state_restores": restored,
    }


__all__ = [
    "DEFAULT_SCALE_GRID",
    "DEBT_COMPONENTS",
    "AnalyticQuadraticTrust",
    "OnlineLogisticTrust",
    "PIDTrustRadiusController",
    "ComponentRiskTrust",
    "safe_scale_from_quadratic",
    "predictor_smoke_test",
]
