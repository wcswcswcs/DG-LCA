"""Hard contracts for official v9 DG-KAN claims."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from .specs import CandidateSpec, ContractFlags


class ContractViolation(AssertionError):
    """Raised when a candidate is not eligible for an official clean claim."""


def _as_bool_int(value: int | bool) -> int:
    return int(bool(value))


def contract_violations(spec: CandidateSpec, flags: ContractFlags) -> List[Tuple[str, str]]:
    violations: List[Tuple[str, str]] = []
    if spec.official_eligible:
        if spec.loss_type != "CE":
            violations.append(("F1_contract_loss_violation", f"loss_type={spec.loss_type}"))
        if float(spec.label_smoothing) != 0.0:
            violations.append(("F2_label_smoothing_violation", f"label_smoothing={spec.label_smoothing}"))
    if _as_bool_int(flags.external_teacher_used) or _as_bool_int(flags.self_teacher_used):
        violations.append(("F3_teacher_or_distill_violation", "teacher flag is nonzero"))
    if _as_bool_int(flags.geometry_loss_used):
        violations.append(("F1_contract_loss_violation", "geometry_loss_used=1"))
    if _as_bool_int(flags.sampler_changed):
        violations.append(("F1_contract_loss_violation", "sampler_changed=1"))
    if _as_bool_int(flags.class_weight_used):
        violations.append(("F1_contract_loss_violation", "class_weight_used=1"))
    if _as_bool_int(flags.cpu_offload_used):
        violations.append(("F16_fake_or_proxy_violation", "cpu_offload_used=1"))
    if _as_bool_int(flags.uses_loss_backward):
        violations.append(("F1_contract_loss_violation", "uses_loss_backward=1"))
    if _as_bool_int(flags.fake_data_used) or _as_bool_int(flags.proxy_row_used):
        violations.append(("F16_fake_or_proxy_violation", "fake/proxy flag is nonzero"))
    if spec.model_level == "full_edge" and "linear" in spec.stack_type.lower():
        violations.append(("F4_full_edge_contract_fail", f"full_edge stack_type={spec.stack_type}"))
    return violations


def validate_official_contract(spec: CandidateSpec, flags: ContractFlags) -> None:
    violations = contract_violations(spec, flags)
    if violations:
        joined = "; ".join(f"{code}:{detail}" for code, detail in violations)
        raise ContractViolation(joined)


def clean_ce_contract(spec: CandidateSpec, flags: ContractFlags) -> None:
    validate_official_contract(spec, flags)


def full_edge_purekan_contract(spec: CandidateSpec, flags: ContractFlags) -> None:
    validate_official_contract(spec, flags)
    if spec.model_level != "full_edge":
        raise ContractViolation(f"F4_full_edge_contract_fail:model_level={spec.model_level}")
    if not spec.edge_basis:
        raise ContractViolation("F4_full_edge_contract_fail:edge_basis_missing")


def make_contract_audit_row(spec: CandidateSpec, flags: ContractFlags, *, stage: str) -> dict:
    violations = contract_violations(spec, flags)
    return {
        **spec.to_row(),
        **flags.to_row(),
        "stage": stage,
        "contract_pass": int(not violations),
        "violation_count": len(violations),
        "violation_codes": ",".join(code for code, _ in violations),
        "violation_details": " | ".join(detail for _, detail in violations),
    }
