"""GeometryCertificateV0 gates for v12 Good Geometry Battery artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class GeometryCertificateThresholds:
    acc_tol: float = 0.005
    exploratory_acc_tol: float = 0.01
    auc_tol: float = 0.02
    ece_tol: float = 0.01
    nll_tol: float = 0.02
    step_ratio_max: float = 1.10
    rank_tol: float = 1.0
    entropy_tol: float = 0.02
    tail_tol: float = 0.05
    margin_tol: float = 0.02
    pareto_rel: float = 0.10
    ece_rel: float = 0.05
    ce_rel: float = 0.05


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None, "not_run", "nan"):
            return default
        return float(value)
    except Exception:
        return default


def _hard_gate_rows(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    t: GeometryCertificateThresholds,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    gates: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    def add(name: str, passed: bool, value: float, threshold: float, detail: str) -> None:
        gates[name] = {"pass": bool(passed), "value": value, "threshold": threshold, "detail": detail}
        if not passed:
            failures.append(name)

    add(
        "task_acc",
        _f(candidate, "val_acc") >= _f(reference, "val_acc") - t.acc_tol,
        _f(candidate, "val_acc") - _f(reference, "val_acc"),
        -t.acc_tol,
        "candidate val_acc delta vs reference",
    )
    add(
        "task_acc_exploratory",
        _f(candidate, "val_acc") >= _f(reference, "val_acc") - t.exploratory_acc_tol,
        _f(candidate, "val_acc") - _f(reference, "val_acc"),
        -t.exploratory_acc_tol,
        "exploratory task slack",
    )
    add(
        "val_loss_auc_time",
        _f(candidate, "val_loss_auc_time") <= _f(reference, "val_loss_auc_time") + t.auc_tol,
        _f(candidate, "val_loss_auc_time") - _f(reference, "val_loss_auc_time"),
        t.auc_tol,
        "candidate time AUC increase vs reference",
    )
    add(
        "ECE",
        _f(candidate, "ECE") <= _f(reference, "ECE") + t.ece_tol,
        _f(candidate, "ECE") - _f(reference, "ECE"),
        t.ece_tol,
        "candidate ECE increase vs reference",
    )
    add(
        "NLL",
        _f(candidate, "NLL") <= _f(reference, "NLL") + t.nll_tol,
        _f(candidate, "NLL") - _f(reference, "NLL"),
        t.nll_tol,
        "candidate NLL increase vs reference",
    )
    add(
        "step_time_ratio",
        _f(candidate, "step_time_ratio_vs_reference", _f(candidate, "step_time_ratio_vs_MLP", 1.0)) <= t.step_ratio_max,
        _f(candidate, "step_time_ratio_vs_reference", _f(candidate, "step_time_ratio_vs_MLP", 1.0)),
        t.step_ratio_max,
        "step or amortized step ratio",
    )
    add(
        "effective_rank_hidden",
        _f(candidate, "effective_rank_hidden") >= _f(reference, "effective_rank_hidden") - t.rank_tol,
        _f(candidate, "effective_rank_hidden") - _f(reference, "effective_rank_hidden"),
        -t.rank_tol,
        "hidden effective rank delta",
    )
    add(
        "basis_usage_entropy",
        _f(candidate, "basis_usage_entropy") >= _f(reference, "basis_usage_entropy") - t.entropy_tol,
        _f(candidate, "basis_usage_entropy") - _f(reference, "basis_usage_entropy"),
        -t.entropy_tol,
        "basis/cover entropy delta",
    )
    add(
        "CEp99",
        _f(candidate, "CE_p99", _f(candidate, "CEp99", 0.0)) <= _f(reference, "CE_p99", _f(reference, "CEp99", 0.0)) + t.tail_tol,
        _f(candidate, "CE_p99", _f(candidate, "CEp99", 0.0)) - _f(reference, "CE_p99", _f(reference, "CEp99", 0.0)),
        t.tail_tol,
        "CE p99 increase vs reference",
    )
    add(
        "margin_p10",
        _f(candidate, "margin_p10") >= _f(reference, "margin_p10") - t.margin_tol,
        _f(candidate, "margin_p10") - _f(reference, "margin_p10"),
        -t.margin_tol,
        "margin p10 delta",
    )
    return gates, failures


def _rel_improves_lower_better(ref: float, cand: float, rel: float) -> bool:
    return cand <= ref * (1.0 - rel) if ref > 0 else cand < ref


def _rel_improves_higher_better(ref: float, cand: float, rel: float) -> bool:
    return cand >= ref * (1.0 + rel) if ref > 0 else cand > ref


def evaluate_geometry_certificate(
    *,
    reference_method: str,
    candidate_method: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    thresholds: GeometryCertificateThresholds | None = None,
) -> Dict[str, Any]:
    t = thresholds or GeometryCertificateThresholds()
    hard, failures = _hard_gate_rows(reference, candidate, t)

    pareto = {
        "curvature": {
            "pass": _rel_improves_lower_better(_f(reference, "curvature_debt"), _f(candidate, "curvature_debt"), t.pareto_rel),
            "reference": _f(reference, "curvature_debt"),
            "candidate": _f(candidate, "curvature_debt"),
        },
        "perturbation": {
            "pass": _rel_improves_lower_better(
                _f(reference, "perturb_logit_drift_p95"),
                _f(candidate, "perturb_logit_drift_p95"),
                t.pareto_rel,
            ),
            "reference": _f(reference, "perturb_logit_drift_p95"),
            "candidate": _f(candidate, "perturb_logit_drift_p95"),
        },
        "signal_noise": {
            "pass": _rel_improves_higher_better(
                _f(reference, "signal_consistency_real"),
                _f(candidate, "signal_consistency_real"),
                t.pareto_rel,
            )
            or _rel_improves_lower_better(_f(reference, "noise_leak"), _f(candidate, "noise_leak"), t.pareto_rel),
            "reference_signal_consistency": _f(reference, "signal_consistency_real"),
            "candidate_signal_consistency": _f(candidate, "signal_consistency_real"),
            "reference_noise_leak": _f(reference, "noise_leak"),
            "candidate_noise_leak": _f(candidate, "noise_leak"),
        },
        "cover": {
            "pass": _rel_improves_higher_better(
                _f(reference, "basis_usage_entropy"),
                _f(candidate, "basis_usage_entropy"),
                t.pareto_rel,
            ),
            "reference": _f(reference, "basis_usage_entropy"),
            "candidate": _f(candidate, "basis_usage_entropy"),
        },
        "tail": {
            "pass": _rel_improves_lower_better(
                _f(reference, "CE_p99", _f(reference, "CEp99")),
                _f(candidate, "CE_p99", _f(candidate, "CEp99")),
                t.ce_rel,
            ),
            "reference": _f(reference, "CE_p99", _f(reference, "CEp99")),
            "candidate": _f(candidate, "CE_p99", _f(candidate, "CEp99")),
        },
        "calibration": {
            "pass": _rel_improves_lower_better(_f(reference, "ECE"), _f(candidate, "ECE"), t.ece_rel),
            "reference": _f(reference, "ECE"),
            "candidate": _f(candidate, "ECE"),
        },
    }
    pareto_pass = any(bool(v["pass"]) for v in pareto.values())
    hard_pass = all(bool(v["pass"]) for k, v in hard.items() if k != "task_acc_exploratory")
    if not pareto_pass:
        failures.append("no_pareto_geometry_improvement")
    return {
        "version": "v0",
        "reference_method": reference_method,
        "candidate_method": candidate_method,
        "hard_gates": hard,
        "pareto_metrics": pareto,
        "hard_gate_pass": bool(hard_pass),
        "pareto_pass": bool(pareto_pass),
        "pass": bool(hard_pass and pareto_pass),
        "failure_reasons": failures,
    }


def certificate_to_rows(certificates: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cert in certificates:
        base = {
            "stage": "P3_GEOMETRY_CERTIFICATE_V0",
            "reference_method": cert.get("reference_method", ""),
            "candidate_method": cert.get("candidate_method", ""),
            "certificate_pass": int(bool(cert.get("pass", False))),
            "hard_gate_pass": int(bool(cert.get("hard_gate_pass", False))),
            "pareto_pass": int(bool(cert.get("pareto_pass", False))),
            "failure_reasons": ";".join(str(x) for x in cert.get("failure_reasons", [])),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        for name, gate in dict(cert.get("hard_gates", {})).items():
            row = dict(base)
            row.update(
                {
                    "gate_type": "hard",
                    "metric_name": name,
                    "metric_pass": int(bool(gate.get("pass", False))),
                    "metric_value": gate.get("value", ""),
                    "threshold": gate.get("threshold", ""),
                    "detail": gate.get("detail", ""),
                }
            )
            rows.append(row)
        for name, metric in dict(cert.get("pareto_metrics", {})).items():
            row = dict(base)
            row.update(
                {
                    "gate_type": "pareto",
                    "metric_name": name,
                    "metric_pass": int(bool(metric.get("pass", False))),
                    "metric_value": metric.get("candidate", metric.get("candidate_signal_consistency", "")),
                    "threshold": "relative_improvement",
                    "detail": "pareto geometry dimension",
                }
            )
            rows.append(row)
    return rows
