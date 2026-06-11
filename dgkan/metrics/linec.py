"""LineC metric implementation and golden tests for v17.

LineC is treated as a measurement. Exceptions and non-finite values are
reported as measurement-invalid rows, not as geometry failures.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Callable

import torch
import torch.nn.functional as F


EPS = 1.0e-12


class LineCMeasurementInvalid(RuntimeError):
    """Raised when a LineC measurement cannot be trusted."""


@dataclass
class LineCResult:
    linec_measurement_valid: int
    CouplingR2: float
    NoiseSignalLeak: float
    RealSignalReservoirRatio: float
    linec_exception_type: str = ""
    linec_exception_message: str = ""
    route: str = "R-LineCMeasured"

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise LineCMeasurementInvalid("expected torch.Tensor")
    out = value.detach().float().reshape(-1)
    if out.numel() == 0:
        raise LineCMeasurementInvalid("empty measurement vector")
    if not bool(torch.isfinite(out).all()):
        raise LineCMeasurementInvalid("non-finite measurement vector")
    return out


def _positive_corr_r2(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(int(a.numel()), int(b.numel()))
    if n < 2:
        raise LineCMeasurementInvalid("need at least two samples for coupling")
    x = a[:n] - a[:n].mean()
    y = b[:n] - b[:n].mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= EPS:
        return 0.0
    corr = float((x @ y / denom).clamp(-1.0, 1.0).item())
    return float(max(0.0, corr) ** 2)


def linec_from_improvements(split_a_improvement: torch.Tensor, split_b_improvement: torch.Tensor) -> LineCResult:
    """Measure transfer coupling from per-example loss improvements.

    Positive improvement means loss decreased after the proposed update.
    Coupling uses only the positive part of correlation so anti-transfer does
    not receive a high R2. NoiseSignalLeak increases when B1 improves while B2
    is damaged. ReservoirRatio is the residual variance after fitting B2 from
    B1; lower is better.
    """

    try:
        a = _as_float_tensor(split_a_improvement)
        b = _as_float_tensor(split_b_improvement)
        n = min(int(a.numel()), int(b.numel()))
        a = a[:n]
        b = b[:n]
        coupling = _positive_corr_r2(a, b)
        mean_a = float(a.mean().item())
        mean_b = float(b.mean().item())
        leak = max(0.0, mean_a) * max(0.0, -mean_b) / (abs(mean_a) + abs(mean_b) + EPS)
        ac = a - a.mean()
        bc = b - b.mean()
        denom = float((ac @ ac).item())
        if denom <= EPS:
            reservoir = 0.0 if float(torch.linalg.vector_norm(bc).item()) <= EPS else 1.0
        else:
            beta = (bc @ ac) / (ac @ ac).clamp_min(EPS)
            resid = bc - beta * ac
            reservoir = float((resid.square().mean() / bc.square().mean().clamp_min(EPS)).clamp(0.0, 9.0).item())
        if not all(math.isfinite(v) for v in [coupling, leak, reservoir]):
            raise LineCMeasurementInvalid("non-finite LineC scalar")
        return LineCResult(1, coupling, leak, reservoir)
    except Exception as exc:
        return LineCResult(
            linec_measurement_valid=0,
            CouplingR2=float("nan"),
            NoiseSignalLeak=float("nan"),
            RealSignalReservoirRatio=float("nan"),
            linec_exception_type=type(exc).__name__,
            linec_exception_message=str(exc),
            route="R0-LineCMeasurementInvalid",
        )


def linec_model_update(
    model: torch.nn.Module,
    split_a: tuple[torch.Tensor, torch.Tensor],
    split_b: tuple[torch.Tensor, torch.Tensor],
    apply_update: Callable[[], None],
    restore_state: Callable[[], None],
) -> LineCResult:
    """Measure LineC around a temporary model update."""

    xa, ya = split_a
    xb, yb = split_b
    try:
        with torch.no_grad():
            before_a = torch.nn.functional.cross_entropy(model(xa).float(), ya, reduction="none")
            before_b = torch.nn.functional.cross_entropy(model(xb).float(), yb, reduction="none")
        apply_update()
        with torch.no_grad():
            after_a = torch.nn.functional.cross_entropy(model(xa).float(), ya, reduction="none")
            after_b = torch.nn.functional.cross_entropy(model(xb).float(), yb, reduction="none")
        restore_state()
        return linec_from_improvements(before_a - after_a, before_b - after_b)
    except Exception as exc:
        try:
            restore_state()
        except Exception:
            pass
        return LineCResult(
            linec_measurement_valid=0,
            CouplingR2=float("nan"),
            NoiseSignalLeak=float("nan"),
            RealSignalReservoirRatio=float("nan"),
            linec_exception_type=type(exc).__name__,
            linec_exception_message=str(exc),
            route="R0-LineCMeasurementInvalid",
        )


def linec_channel_from_logits(
    initial_a_logits: torch.Tensor,
    current_a_logits: torch.Tensor,
    labels_a: torch.Tensor,
    initial_b_logits: torch.Tensor,
    current_b_logits: torch.Tensor,
    labels_b: torch.Tensor,
) -> LineCResult:
    """Long-horizon train-split channel readback.

    This is an audit metric, not a direction source. It measures whether the
    trajectory-level NLL improvement on one train split is coupled to the
    trajectory-level NLL improvement on a disjoint train split.
    """

    try:
        before_a = F.cross_entropy(initial_a_logits.detach().float(), labels_a.detach(), reduction="none")
        after_a = F.cross_entropy(current_a_logits.detach().float(), labels_a.detach(), reduction="none")
        before_b = F.cross_entropy(initial_b_logits.detach().float(), labels_b.detach(), reduction="none")
        after_b = F.cross_entropy(current_b_logits.detach().float(), labels_b.detach(), reduction="none")
        return linec_from_improvements(before_a - after_a, before_b - after_b)
    except Exception as exc:
        return LineCResult(
            linec_measurement_valid=0,
            CouplingR2=float("nan"),
            NoiseSignalLeak=float("nan"),
            RealSignalReservoirRatio=float("nan"),
            linec_exception_type=type(exc).__name__,
            linec_exception_message=str(exc),
            route="R0-LineCMeasurementInvalid",
        )


def run_linec_channel_golden_tests(seed: int = 1900) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    labels = torch.arange(96) % 3

    def logits_from_margin(margin: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(int(margin.numel()), 3)
        logits[torch.arange(int(margin.numel())), labels[: int(margin.numel())]] = margin
        return logits

    rows: list[dict[str, Any]] = []
    base_a = logits_from_margin(torch.zeros(96))
    base_b = logits_from_margin(torch.zeros(96))
    noop = linec_channel_from_logits(base_a, base_a.clone(), labels, base_b, base_b.clone(), labels)
    rows.append({"golden": "C1-NoOpNull", **noop.to_row(), "pass": int(noop.linec_measurement_valid and noop.CouplingR2 <= 1.0e-9)})

    transfer = torch.linspace(0.0, 1.5, 96)
    current_a = logits_from_margin(transfer)
    current_b = logits_from_margin(transfer + 0.03 * torch.randn(96, generator=gen))
    known = linec_channel_from_logits(base_a, current_a, labels, base_b, current_b, labels)
    rows.append({"golden": "C2-KnownTransferPositive", **known.to_row(), "pass": int(known.linec_measurement_valid and known.CouplingR2 >= 0.90 and known.NoiseSignalLeak <= 0.05)})

    bad_b = logits_from_margin(-transfer)
    leak = linec_channel_from_logits(base_a, current_a, labels, base_b, bad_b, labels)
    rows.append({"golden": "C3-NoiseLeakPositive", **leak.to_row(), "pass": int(leak.linec_measurement_valid and leak.NoiseSignalLeak >= 0.05)})

    reservoir_b = logits_from_margin(0.03 * torch.randn(96, generator=gen))
    reservoir = linec_channel_from_logits(base_a, current_a, labels, base_b, reservoir_b, labels)
    rows.append({"golden": "C4-ReservoirOnlyPositive", **reservoir.to_row(), "pass": int(reservoir.linec_measurement_valid and reservoir.CouplingR2 <= 0.15)})

    perm = torch.randperm(96, generator=gen)
    permuted = linec_channel_from_logits(base_a, current_a[perm], labels, base_b, current_b, labels)
    rows.append(
        {
            "golden": "C5-BatchPermutationMismatch",
            **permuted.to_row(),
            "known_transfer_coupling": known.CouplingR2,
            "coupling_drop": float(known.CouplingR2) - float(permuted.CouplingR2),
            "pass": int(permuted.linec_measurement_valid and float(known.CouplingR2) - float(permuted.CouplingR2) >= 0.50),
        }
    )

    invalid = linec_channel_from_logits(torch.tensor([float("nan")]), torch.zeros(1, 3), torch.zeros(1, dtype=torch.long), base_b, current_b, labels)
    rows.append({"golden": "C6-ExceptionMeasurementInvalid", **invalid.to_row(), "pass": int(invalid.route == "R0-LineCMeasurementInvalid" and not invalid.linec_measurement_valid)})

    before = F.cross_entropy(base_a, labels, reduction="none")
    after_a = F.cross_entropy(current_a, labels, reduction="none")
    after_b = F.cross_entropy(current_b, labels, reduction="none")
    improvement_a = before - after_a
    improvement_b = before - after_b
    base_scale = linec_from_improvements(improvement_a, improvement_b)
    scaled = linec_from_improvements(2.5 * improvement_a, 2.5 * improvement_b)
    rows.append({"golden": "C7-ScaleInvariance", **scaled.to_row(), "scale_coupling_delta": abs(float(base_scale.CouplingR2) - float(scaled.CouplingR2)), "pass": int(scaled.linec_measurement_valid and abs(float(base_scale.CouplingR2) - float(scaled.CouplingR2)) <= 1.0e-6)})

    long_initial = logits_from_margin(torch.zeros(96))
    long_mid = logits_from_margin(0.5 * transfer)
    long_final = logits_from_margin(1.25 * transfer)
    mid = linec_channel_from_logits(long_initial, long_mid, labels, long_initial, long_mid, labels)
    final = linec_channel_from_logits(long_initial, long_final, labels, long_initial, long_final, labels)
    rows.append(
        {
            "golden": "C8-LongHorizonRecoverySynthetic",
            **final.to_row(),
            "mid_CouplingR2": mid.CouplingR2,
            "final_CouplingR2": final.CouplingR2,
            "pass": int(mid.linec_measurement_valid and final.linec_measurement_valid and final.CouplingR2 >= mid.CouplingR2 and final.CouplingR2 >= 0.90),
        }
    )
    for row in rows:
        row["uses_validation_test_future_query"] = 0
        row["mode"] = "LineC-channel"
    return rows


def run_linec_golden_tests(seed: int = 1700, trials: int = 200) -> list[dict[str, Any]]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rows: list[dict[str, Any]] = []

    base = torch.linspace(-1.0, 1.0, 64)
    noop = linec_from_improvements(torch.zeros_like(base), torch.zeros_like(base))
    rows.append(
        {
            "golden": "G1-NoOpNull",
            **noop.to_row(),
            "pass": int(noop.linec_measurement_valid and abs(noop.CouplingR2) <= 1.0e-9 and abs(noop.NoiseSignalLeak) <= 1.0e-9 and abs(noop.RealSignalReservoirRatio) <= 1.0e-9),
        }
    )

    false_positive = 0
    valid = 0
    for _ in range(int(trials)):
        a = torch.randn(64, generator=gen)
        b = torch.randn(64, generator=gen)
        lm = linec_from_improvements(a, b)
        if lm.linec_measurement_valid:
            valid += 1
            false_positive += int(lm.CouplingR2 >= 0.50 and lm.NoiseSignalLeak <= 0.10 and lm.RealSignalReservoirRatio <= 0.50)
    fpr = float(false_positive) / max(1, valid)
    rows.append(
        {
            "golden": "G2-RandomMatchedNormNull",
            "linec_measurement_valid": int(valid == int(trials)),
            "CouplingR2": "",
            "NoiseSignalLeak": "",
            "RealSignalReservoirRatio": "",
            "linec_null_false_positive_rate": fpr,
            "pass": int(valid == int(trials) and fpr <= 0.05),
            "route": "R-LineCMeasured" if valid == int(trials) else "R0-LineCMeasurementInvalid",
        }
    )

    transfer_a = torch.linspace(-0.2, 0.8, 96)
    transfer_b = transfer_a + 0.03 * torch.randn(96, generator=gen)
    transfer = linec_from_improvements(transfer_a, transfer_b)
    rows.append(
        {
            "golden": "G3-SyntheticKnownTransfer",
            **transfer.to_row(),
            "linec_golden_transfer_pass": int(transfer.linec_measurement_valid and transfer.CouplingR2 >= 0.90 and transfer.NoiseSignalLeak <= 0.05 and transfer.RealSignalReservoirRatio <= 0.10),
            "pass": int(transfer.linec_measurement_valid and transfer.CouplingR2 >= 0.90 and transfer.NoiseSignalLeak <= 0.05 and transfer.RealSignalReservoirRatio <= 0.10),
        }
    )

    leak_a = 0.15 + torch.linspace(0.0, 0.8, 96)
    leak_b = -0.15 - torch.linspace(0.0, 0.8, 96) + 0.02 * torch.randn(96, generator=gen)
    leak = linec_from_improvements(leak_a, leak_b)
    rows.append(
        {
            "golden": "G4-SyntheticNoiseLeak",
            **leak.to_row(),
            "linec_golden_noise_fail_pass": int(leak.linec_measurement_valid and leak.CouplingR2 <= 0.05 and leak.NoiseSignalLeak >= 0.05),
            "pass": int(leak.linec_measurement_valid and leak.CouplingR2 <= 0.05 and leak.NoiseSignalLeak >= 0.05),
        }
    )

    reservoir_a = torch.sin(torch.linspace(0.0, 6.0, 96))
    reservoir_b = 0.03 * torch.randn(96, generator=gen)
    reservoir = linec_from_improvements(reservoir_a, reservoir_b)
    rows.append(
        {
            "golden": "G5-ReservoirOnlyPerturbation",
            **reservoir.to_row(),
            "reservoir_only_pass": int(reservoir.linec_measurement_valid and reservoir.CouplingR2 <= 0.10),
            "pass": int(reservoir.linec_measurement_valid and reservoir.CouplingR2 <= 0.10),
        }
    )

    perm = torch.randperm(int(transfer_a.numel()), generator=gen)
    permuted = linec_from_improvements(transfer_a[perm], transfer_b)
    coupling_drop = float(transfer.CouplingR2) - float(permuted.CouplingR2)
    rows.append(
        {
            "golden": "G6-BatchPermutationMismatchDrop",
            **permuted.to_row(),
            "batch_permutation_coupling_drop": coupling_drop,
            "pass": int(permuted.linec_measurement_valid and coupling_drop >= 0.50 and permuted.CouplingR2 <= 0.50),
        }
    )

    invalid = linec_from_improvements(torch.tensor([float("nan")]), torch.zeros(1))
    rows.append(
        {
            "golden": "G7-ExceptionPathMeasurementInvalid",
            **invalid.to_row(),
            "exception_as_fail_count": 0 if invalid.route == "R0-LineCMeasurementInvalid" and not invalid.linec_measurement_valid else 1,
            "pass": int(invalid.route == "R0-LineCMeasurementInvalid" and not invalid.linec_measurement_valid),
        }
    )

    scaled = linec_from_improvements(2.5 * transfer_a, 2.5 * transfer_b)
    scale_delta = abs(float(transfer.CouplingR2) - float(scaled.CouplingR2))
    rows.append(
        {
            "golden": "G8-ScaleInvariance",
            **scaled.to_row(),
            "scale_coupling_delta": scale_delta,
            "pass": int(scaled.linec_measurement_valid and scale_delta <= 1.0e-6),
        }
    )

    control_like = linec_from_improvements(transfer_a, transfer_b)
    rows.append(
        {
            "golden": "G9-ControlEquivalence",
            **control_like.to_row(),
            "control_equivalent": int(control_like.CouplingR2 >= 0.50),
            "pass": int(control_like.linec_measurement_valid and control_like.CouplingR2 >= 0.50),
        }
    )

    return rows
