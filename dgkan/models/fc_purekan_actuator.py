"""FC-PureKAN edge-owned actuator primitives for LQ functional tests.

The v9.2.13 runners use this module for strict FC-PureKAN actuator math.  The
actuator is not an external residual: it is an additional output-layer
edge-basis channel over the same lifted coordinates used by the LQ primitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from dgkan.models import fc_purekan_lq as lq
from dgkan.training.manual_full_edge import ce_loss_and_grad


@dataclass(frozen=True)
class ActuatorSpec:
    candidate_id: str
    actuator_type: str
    hidden_dim: int = 256
    output_scale: float = 1.0
    actuator_init_scale: float = 0.0
    beta: float = 1.0
    gamma: float = 2.0
    center: float = 0.0


ACTUATOR_SPECS: Dict[str, ActuatorSpec] = {
    "A0-LQ-current": ActuatorSpec("A0-LQ-current", "none"),
    "A1-LQ-T2ActuatorZeroInit": ActuatorSpec("A1-LQ-T2ActuatorZeroInit", "t2_duplicate", actuator_init_scale=0.0),
    "A2-LQ-NormalizedT2Actuator": ActuatorSpec("A2-LQ-NormalizedT2Actuator", "normalized_t2", actuator_init_scale=0.0),
    "A3-LQ-CenteredT2Actuator": ActuatorSpec("A3-LQ-CenteredT2Actuator", "centered_t2", actuator_init_scale=0.0),
    "A4-LQ-BoundedRationalActuator": ActuatorSpec("A4-LQ-BoundedRationalActuator", "bounded_rational", actuator_init_scale=0.0),
    "A5-LQ-PiecewiseLinear2Actuator": ActuatorSpec("A5-LQ-PiecewiseLinear2Actuator", "piecewise_linear_2", actuator_init_scale=0.0),
    "A6-LQ-LocalRBFSharedCenterActuator": ActuatorSpec("A6-LQ-LocalRBFSharedCenterActuator", "local_rbf", actuator_init_scale=0.0),
    "A7-LQ-BasisEntropyActuator": ActuatorSpec("A7-LQ-BasisEntropyActuator", "basis_entropy", actuator_init_scale=0.0),
    "A4b-BoundedRational-BranchlessDerivative": ActuatorSpec("A4b-BoundedRational-BranchlessDerivative", "bounded_rational", actuator_init_scale=0.0),
    "A4c-BoundedRational-FixedBeta": ActuatorSpec("A4c-BoundedRational-FixedBeta", "bounded_rational", actuator_init_scale=0.0, beta=1.0),
    "A4d-BoundedRational-ValueOnlyActuator": ActuatorSpec("A4d-BoundedRational-ValueOnlyActuator", "bounded_rational_value_only", actuator_init_scale=0.0),
    "A4e-BoundedRational-FusedCoeffGrad": ActuatorSpec("A4e-BoundedRational-FusedCoeffGrad", "bounded_rational", actuator_init_scale=0.0),
    "A5b-PiecewiseLinear2-Branchless": ActuatorSpec("A5b-PiecewiseLinear2-Branchless", "piecewise_linear_2", actuator_init_scale=0.0),
    "A5c-PiecewiseLinear2-FixedKnots": ActuatorSpec("A5c-PiecewiseLinear2-FixedKnots", "piecewise_linear_2", actuator_init_scale=0.0),
    "A7c-BasisEntropy-ValueOnly": ActuatorSpec("A7c-BasisEntropy-ValueOnly", "basis_entropy_value_only", actuator_init_scale=0.0),
    "OP1-ObservableTailLinearChannel": ActuatorSpec(
        "OP1-ObservableTailLinearChannel", "observable_tail_linear", actuator_init_scale=1.0e-3, gamma=6.0, center=0.55
    ),
    "OP2-ObservablePiecewiseTailChannel": ActuatorSpec(
        "OP2-ObservablePiecewiseTailChannel", "observable_piecewise_tail_4", actuator_init_scale=1.0e-3
    ),
    "OP3-ObservableSharedRBFLocalChannel": ActuatorSpec(
        "OP3-ObservableSharedRBFLocalChannel", "observable_shared_rbf4", actuator_init_scale=1.0e-3, gamma=4.0
    ),
    "OP4-ObservableOrthogonalTailChannel": ActuatorSpec(
        "OP4-ObservableOrthogonalTailChannel", "observable_orthogonal_tail", actuator_init_scale=1.0e-3, gamma=6.0, center=0.45
    ),
    "OP5-ObservableControlGapChannel": ActuatorSpec(
        "OP5-ObservableControlGapChannel", "observable_control_gap", actuator_init_scale=1.0e-3, gamma=6.0, center=0.50
    ),
    "OP6-LightHybridObservable": ActuatorSpec(
        "OP6-LightHybridObservable", "observable_light_hybrid", actuator_init_scale=1.0e-3, gamma=4.0
    ),
    "VAOP1-TailGradientAlignedLinearChannel": ActuatorSpec(
        "VAOP1-TailGradientAlignedLinearChannel", "observable_tail_linear", actuator_init_scale=1.0e-3, gamma=7.0, center=0.60
    ),
    "VAOP2-MarginJacobianPiecewiseChannel": ActuatorSpec(
        "VAOP2-MarginJacobianPiecewiseChannel", "observable_piecewise_tail_4", actuator_init_scale=1.0e-3
    ),
    "VAOP3-ControlGapBoundedSharedRBFChannel": ActuatorSpec(
        "VAOP3-ControlGapBoundedSharedRBFChannel", "observable_shared_rbf4", actuator_init_scale=1.0e-3, gamma=5.0
    ),
    "VAOP4-RoleWiseFT7EdgeChannel": ActuatorSpec(
        "VAOP4-RoleWiseFT7EdgeChannel", "observable_orthogonal_tail", actuator_init_scale=1.0e-3, gamma=6.0, center=0.45
    ),
    "VAOP5-ConservativeControlGapLowerBoundChannel": ActuatorSpec(
        "VAOP5-ConservativeControlGapLowerBoundChannel", "observable_control_gap", actuator_init_scale=1.0e-3, gamma=7.0, center=0.55
    ),
    "VAOP6-FamilyValueChannel": ActuatorSpec(
        "VAOP6-FamilyValueChannel", "observable_orthogonal_tail", actuator_init_scale=1.0e-3, gamma=4.0, center=0.55
    ),
    "BPFS1-ZeroInitDormantTailLinear": ActuatorSpec(
        "BPFS1-ZeroInitDormantTailLinear", "observable_tail_linear", actuator_init_scale=0.0, gamma=6.0, center=0.55
    ),
    "BPFS2-FrozenTaskFunctionalAttach": ActuatorSpec(
        "BPFS2-FrozenTaskFunctionalAttach", "bounded_rational", actuator_init_scale=0.0, beta=1.0
    ),
    "BPFS3-TaskOrthogonalFunctionalSubspace": ActuatorSpec(
        "BPFS3-TaskOrthogonalFunctionalSubspace", "observable_orthogonal_tail", actuator_init_scale=0.0, gamma=6.0, center=0.45
    ),
    "BPFS4-ControlGapFunctionalChannel": ActuatorSpec(
        "BPFS4-ControlGapFunctionalChannel", "observable_control_gap", actuator_init_scale=0.0, gamma=6.0, center=0.50
    ),
    "BPFS5-FamilyValueFunctionalChannel": ActuatorSpec(
        "BPFS5-FamilyValueFunctionalChannel", "observable_orthogonal_tail", actuator_init_scale=0.0, gamma=4.0, center=0.55
    ),
    "BPFS6-RoleWiseFT7EdgeOwnedChannel": ActuatorSpec(
        "BPFS6-RoleWiseFT7EdgeOwnedChannel", "observable_light_hybrid", actuator_init_scale=0.0, gamma=4.0
    ),
    "TPEA1-RegisteredZeroExcluded": ActuatorSpec(
        "TPEA1-RegisteredZeroExcluded", "bounded_rational", actuator_init_scale=0.0, beta=1.0
    ),
    "TPEA2-RegisteredZeroNoOpGroup": ActuatorSpec(
        "TPEA2-RegisteredZeroNoOpGroup", "observable_control_gap", actuator_init_scale=0.0, gamma=6.0, center=0.50
    ),
    "TPEA3-LateAttachOrthogonalTail": ActuatorSpec(
        "TPEA3-LateAttachOrthogonalTail", "observable_orthogonal_tail", actuator_init_scale=0.0, gamma=6.0, center=0.45
    ),
    "TPEA4-ShadowSpecUntilEvent": ActuatorSpec(
        "TPEA4-ShadowSpecUntilEvent", "observable_tail_linear", actuator_init_scale=0.0, gamma=6.0, center=0.55
    ),
    "TPEA5-RoleWiseFT7LateAttach": ActuatorSpec(
        "TPEA5-RoleWiseFT7LateAttach", "observable_light_hybrid", actuator_init_scale=0.0, gamma=4.0
    ),
    "TPEA6-ControlGapLateAttach": ActuatorSpec(
        "TPEA6-ControlGapLateAttach", "observable_control_gap", actuator_init_scale=0.0, gamma=6.0, center=0.50
    ),
}


def actuator_specs_from_ids(ids: Sequence[str]) -> List[ActuatorSpec]:
    out: List[ActuatorSpec] = []
    for cid in ids:
        if cid not in ACTUATOR_SPECS:
            raise ValueError(f"unknown actuator candidate {cid}")
        out.append(ACTUATOR_SPECS[cid])
    return out


def _normed_lift(
    h: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = (h - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdh = (((raw >= -clip) & (raw <= clip)).to(h.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    return z, dzdh


def actuator_basis_from_lift(
    h: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: ActuatorSpec,
    clip: float,
    out_div: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """Return base LQ channels plus any actuator channels.

    Channel 0 is identity and channel 1 is Chebyshev T2.  Extra channels are
    edge-owned basis functions of the lifted coordinate.
    """

    z, dzdh = _normed_lift(h, mu, std, clip, out_div)
    t2 = 2.0 * z.square() - 1.0
    dt2 = 4.0 * z * dzdh
    vals: List[torch.Tensor] = [h, t2]
    ders: List[torch.Tensor] = [torch.ones_like(h), dt2]
    names: List[str] = ["identity", "t2_task"]
    kind = str(spec.actuator_type)
    if kind == "none":
        return vals, ders, names
    if kind == "t2_duplicate":
        vals.append(t2)
        ders.append(dt2)
        names.append("actuator_t2_duplicate")
    elif kind == "normalized_t2":
        scale = t2.detach().float().std(dim=0, keepdim=True).clamp_min(1.0e-3).to(t2.dtype)
        vals.append(t2 / scale)
        ders.append(dt2 / scale)
        names.append("actuator_normalized_t2")
    elif kind == "centered_t2":
        center = t2.detach().mean(dim=0, keepdim=True)
        vals.append(t2 - center)
        ders.append(dt2)
        names.append("actuator_centered_t2")
    elif kind == "bounded_rational":
        beta = float(spec.beta)
        denom = 1.0 + beta * z.square()
        vals.append(z.square() / denom)
        ders.append((2.0 * z / denom.square()) * dzdh)
        names.append("actuator_bounded_rational")
    elif kind == "bounded_rational_value_only":
        beta = float(spec.beta)
        denom = 1.0 + beta * z.square()
        # Value-only actuator: the output-edge coefficient is trainable, while
        # the actuator basis is treated as a fixed event/value channel.
        vals.append((z.square() / denom).detach())
        ders.append(torch.zeros_like(h))
        names.append("actuator_bounded_rational_value_only")
    elif kind == "piecewise_linear_2":
        pos = torch.relu(z)
        neg = torch.relu(-z)
        vals.extend([pos, neg])
        ders.extend([(z > 0).to(h.dtype) * dzdh, -(z < 0).to(h.dtype) * dzdh])
        names.extend(["actuator_piecewise_pos", "actuator_piecewise_neg"])
    elif kind == "local_rbf":
        gamma = float(spec.gamma)
        diff = z - float(spec.center)
        val = torch.exp(-gamma * diff.square())
        vals.append(val)
        ders.append((-2.0 * gamma * diff) * val * dzdh)
        names.append("actuator_local_rbf")
    elif kind == "basis_entropy":
        gate = (1.0 - z.abs()).clamp_min(0.0)
        dgate = torch.where(z.abs() < 1.0, -z.sign() * dzdh, torch.zeros_like(z))
        vals.append(t2 * gate)
        ders.append(dt2 * gate + t2 * dgate)
        names.append("actuator_basis_entropy")
    elif kind == "basis_entropy_value_only":
        gate = (1.0 - z.abs()).clamp_min(0.0)
        vals.append((t2 * gate).detach())
        ders.append(torch.zeros_like(h))
        names.append("actuator_basis_entropy_value_only")
    elif kind == "observable_tail_linear":
        gamma = float(spec.gamma)
        center = float(spec.center)
        abs_z = z.abs()
        sign_z = z.sign()
        gate = torch.sigmoid(gamma * (abs_z - center))
        dgate = gamma * gate * (1.0 - gate) * sign_z * dzdh
        vals.append(z * gate)
        ders.append(dzdh * gate + z * dgate)
        names.append("observable_tail_linear")
    elif kind == "observable_piecewise_tail_4":
        abs_z = z.abs()
        sign_z = z.sign()
        for idx, threshold in enumerate((0.25, 0.50, 0.75, 1.00)):
            tail = torch.relu(abs_z - float(threshold))
            vals.append(tail)
            ders.append((abs_z > float(threshold)).to(h.dtype) * sign_z * dzdh)
            names.append(f"observable_piecewise_tail_{idx + 1}")
    elif kind == "observable_shared_rbf4":
        gamma = float(spec.gamma)
        for idx, center in enumerate((-0.75, -0.25, 0.25, 0.75)):
            diff = z - float(center)
            val = torch.exp(-gamma * diff.square())
            vals.append(val)
            ders.append((-2.0 * gamma * diff) * val * dzdh)
            names.append(f"observable_shared_rbf4_{idx + 1}")
    elif kind == "observable_orthogonal_tail":
        gamma = float(spec.gamma)
        center = float(spec.center)
        abs_z = z.abs()
        sign_z = z.sign()
        gate = torch.sigmoid(gamma * (abs_z - center))
        dgate = gamma * gate * (1.0 - gate) * sign_z * dzdh
        poly = z - z.pow(3)
        dpoly = (1.0 - 3.0 * z.square()) * dzdh
        vals.append(poly * gate)
        ders.append(dpoly * gate + poly * dgate)
        names.append("observable_orthogonal_tail")
    elif kind == "observable_control_gap":
        gamma = float(spec.gamma)
        center = float(spec.center)
        abs_z = z.abs()
        sign_z = z.sign()
        gate = torch.sigmoid(gamma * (abs_z - center))
        dgate = gamma * gate * (1.0 - gate) * sign_z * dzdh
        vals.append(t2 * gate * sign_z)
        ders.append((dt2 * gate + t2 * dgate) * sign_z)
        names.append("observable_control_gap")
    elif kind == "observable_light_hybrid":
        beta = float(spec.beta)
        gamma = float(spec.gamma)
        denom = 1.0 + beta * z.square()
        rational = z.square() / denom
        drational = (2.0 * z / denom.square()) * dzdh
        rbf = torch.exp(-gamma * z.square())
        drbf = (-2.0 * gamma * z) * rbf * dzdh
        vals.extend([rational, rbf])
        ders.extend([drational, drbf])
        names.extend(["observable_light_hybrid_rational", "observable_light_hybrid_rbf"])
    else:
        raise ValueError(f"unknown actuator_type {spec.actuator_type}")
    return vals, ders, names


def actuator_channel_count(spec: ActuatorSpec) -> int:
    if spec.actuator_type == "none":
        return 0
    if spec.actuator_type in {"piecewise_linear_2", "observable_light_hybrid"}:
        return 2
    if spec.actuator_type in {"observable_piecewise_tail_4", "observable_shared_rbf4"}:
        return 4
    return 1


def basis_formula(spec: ActuatorSpec) -> str:
    formulas = {
        "none": "B0(h)=h; B2(h)=2z^2-1",
        "t2_duplicate": "B_a(h)=2z^2-1 duplicate zero-init actuator",
        "normalized_t2": "B_a(h)=(2z^2-1)/std_batch(2z^2-1)",
        "centered_t2": "B_a(h)=(2z^2-1)-mean_batch(2z^2-1)",
        "bounded_rational": "B_a(h)=z^2/(1+beta z^2)",
        "bounded_rational_value_only": "B_a(h)=stopgrad(z^2/(1+beta z^2)); train output-edge coefficient only",
        "piecewise_linear_2": "B_a+(h)=max(z,0), B_a-(h)=max(-z,0)",
        "local_rbf": "B_a(h)=exp(-gamma (z-center)^2)",
        "basis_entropy": "B_a(h)=(2z^2-1) max(1-|z|,0)",
        "basis_entropy_value_only": "B_a(h)=stopgrad((2z^2-1) max(1-|z|,0)); train output-edge coefficient only",
        "observable_tail_linear": "B_a(h)=z sigmoid(gamma(|z|-center))",
        "observable_piecewise_tail_4": "B_a,k(h)=max(|z|-tau_k,0), tau={0.25,0.50,0.75,1.00}",
        "observable_shared_rbf4": "B_a,k(h)=exp(-gamma(z-c_k)^2), c={-0.75,-0.25,0.25,0.75}",
        "observable_orthogonal_tail": "B_a(h)=(z-z^3) sigmoid(gamma(|z|-center))",
        "observable_control_gap": "B_a(h)=(2z^2-1) sigmoid(gamma(|z|-center)) sign(z)",
        "observable_light_hybrid": "B_a(h)={z^2/(1+beta z^2), exp(-gamma z^2)}",
    }
    return formulas[str(spec.actuator_type)]


def actuator_forward(
    x: torch.Tensor,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: ActuatorSpec,
    clip: float = 2.0,
    out_div: float = 2.0,
) -> torch.Tensor:
    h = x @ params[0]
    vals, _ders, _names = actuator_basis_from_lift(h, mu, std, spec, clip, out_div)
    logits = vals[0] @ params[1]
    for val, weight in zip(vals[1:], params[2:]):
        logits = logits + val @ weight
    return logits


def actuator_fwd_bwd(
    x: torch.Tensor,
    labels: torch.Tensor,
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: ActuatorSpec,
    clip: float = 2.0,
    out_div: float = 2.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    h = x @ params[0]
    vals, ders, _names = actuator_basis_from_lift(h, mu, std, spec, clip, out_div)
    logits = vals[0] @ params[1]
    for val, weight in zip(vals[1:], params[2:]):
        logits = logits + val @ weight
    loss, dy = ce_loss_and_grad(logits, labels)
    dweights = [val.T @ dy for val in vals]
    dh = torch.zeros_like(h)
    for der, weight in zip(ders, params[1:]):
        dh = dh + (dy @ weight.T) * der
    d_a = x.T @ dh
    return loss, [d_a, *dweights]


def init_actuator_params(
    input_dim: int,
    output_dim: int,
    spec: ActuatorSpec,
    x_for_stats: torch.Tensor,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    base = lq.LQSpec(spec.candidate_id, "t2", spec.hidden_dim, "default", spec.output_scale)
    params, mu, std = lq.init_lq_params(input_dim, output_dim, base, x_for_stats, device, seed)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 917)
    for _ in range(actuator_channel_count(spec)):
        scale = float(spec.actuator_init_scale) / math.sqrt(max(1, spec.hidden_dim))
        if scale == 0.0:
            params.append(torch.zeros(spec.hidden_dim, output_dim, device=device))
        else:
            params.append(torch.randn(spec.hidden_dim, output_dim, device=device, generator=gen) * scale)
    return params, mu, std


def output_vjp_direction(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: ActuatorSpec,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
) -> List[torch.Tensor]:
    h = x @ params[0]
    vals, ders, _names = actuator_basis_from_lift(h, mu, std, spec, 2.0, 2.0)
    scale = 1.0 / max(1, int(x.shape[0]))
    dy = target_delta_logits.detach() * scale
    dweights = [val.T @ dy for val in vals]
    dh = torch.zeros_like(h)
    for der, weight in zip(ders, params[1:]):
        dh = dh + (dy @ weight.T) * der
    d_a = x.T @ dh
    return [d_a, *dweights]


def actuator_only_least_squares_delta(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: ActuatorSpec,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
    ridge: float = 1.0e-3,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Fit target logits using only the extra actuator output coefficients."""

    delta = [torch.zeros_like(p) for p in params]
    n_act = actuator_channel_count(spec)
    if n_act <= 0:
        return delta, {"ls_rank": 0, "ls_residual_norm": float(target_delta_logits.float().norm().detach().cpu())}
    h = x @ params[0]
    vals, _ders, names = actuator_basis_from_lift(h, mu, std, spec, 2.0, 2.0)
    phi = torch.cat(vals[-n_act:], dim=1).float()
    target = target_delta_logits.float()
    eye = torch.eye(phi.shape[1], device=phi.device, dtype=phi.dtype)
    gram = phi.T @ phi + float(ridge) * eye
    rhs = phi.T @ target
    coef = torch.linalg.solve(gram, rhs).to(params[0].dtype)
    start = 0
    for idx in range(n_act):
        block = coef[start : start + spec.hidden_dim]
        delta[len(params) - n_act + idx] = block
        start += spec.hidden_dim
    actual = phi @ coef.float()
    residual = actual - target
    return delta, {
        "actuator_channel_names": ",".join(names[-n_act:]),
        "ls_rank": int(torch.linalg.matrix_rank(phi).detach().cpu()),
        "ls_residual_norm": float(residual.norm().detach().cpu()),
    }


def output_fit_metrics(
    params: Sequence[torch.Tensor],
    delta: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    spec: ActuatorSpec,
    x: torch.Tensor,
    target_delta_logits: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        before = actuator_forward(x, params, mu, std, spec)
        after = actuator_forward(x, [p.detach() + d.detach() for p, d in zip(params, delta)], mu, std, spec)
        actual = after - before
        target = target_delta_logits.detach()
        sse = (actual.float() - target.float()).square().sum()
        centered = target.float() - target.float().mean()
        sst = centered.square().sum().clamp_min(1.0e-12)
        r2 = 1.0 - sse / sst
    return {
        "output_target_fit_r2": float(r2.detach().cpu()),
        "output_displacement_norm": float(actual.float().norm().detach().cpu()),
        "target_norm": float(target.float().norm().detach().cpu()),
        "output_displacement_to_target_ratio": float((actual.float().norm() / target.float().norm().clamp_min(1.0e-12)).detach().cpu()),
    }


def basis_condition_metrics(h: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, spec: ActuatorSpec) -> Dict[str, float]:
    vals, _ders, _names = actuator_basis_from_lift(h, mu, std, spec, 2.0, 2.0)
    feat = torch.stack([v.reshape(-1) for v in vals], dim=1)
    feat = feat - feat.mean(dim=0, keepdim=True)
    cov = feat.T @ feat / max(1, feat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.float()).clamp_min(1.0e-12)
    energy = eig / eig.sum().clamp_min(1.0e-12)
    entropy = float((-(energy * energy.log()).sum() / math.log(max(2, int(eig.numel())))).detach().cpu())
    return {
        "basis_condition_number": float((eig.max() / eig.min()).detach().cpu()),
        "basis_usage_entropy": entropy,
        "dominant_basis_fraction": float(energy.max().detach().cpu()),
    }
