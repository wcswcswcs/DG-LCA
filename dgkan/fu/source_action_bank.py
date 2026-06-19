"""Source/action bank utilities for v22.18 benefit-conditioned audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionFamily:
    action_id: str
    name: str
    is_control: int
    description: str


ACTION_FAMILIES: tuple[ActionFamily, ...] = (
    ActionFamily("A0", "no-op / AdamW base", 0, "ordinary optimizer update without FU"),
    ActionFamily("A1", "weak predictive cotangent prox", 0, "legacy low-ratio FU prox"),
    ActionFamily("A2", "source-usefulness gated prox", 0, "FU prox gated by source usefulness"),
    ActionFamily("A3", "source release", 0, "release stale or low-usefulness source"),
    ActionFamily("A4", "split-consensus param source", 0, "split-batch gradient consensus source"),
    ActionFamily("A5", "split-consensus + virtual safety", 0, "split source with current-step virtual train-loss veto"),
    ActionFamily("A6", "JVP-history refresh scale0.05", 0, "history refresh through JVP/effect-space estimate"),
    ActionFamily("A7", "JVP-history refresh scale0.10", 0, "higher-scale JVP/effect-space refresh"),
    ActionFamily("A8", "D-CHE h8 basis-native source", 0, "strict FC-PureKAN D-CHE h8 basis source"),
    ActionFamily("A9", "D-CHE h8 basis-native + leakage penalty", 0, "basis source with readout leakage penalty"),
    ActionFamily("A10", "pairwise geometry source", 0, "pairwise margin or geometry source"),
    ActionFamily("A11", "random useful-source control", 1, "random matched source control"),
    ActionFamily("A12", "stable-random useful-source control", 1, "stable random matched source control"),
    ActionFamily("A13", "signflip useful-source control", 1, "sign-flipped source control"),
    ActionFamily("A14", "shuffled-source control", 1, "shuffled or corrupt source control"),
)

ACTION_BY_ID = {family.action_id: family for family in ACTION_FAMILIES}


RUNTIME_FEATURES = [
    "U_task_usefulness",
    "source_loss_boundary_distance",
    "action_family_numeric",
    "source_age",
    "source_norm",
    "J_reachable_residual_estimate",
    "JBa_source_cosine_estimate",
    "risk_score",
    "destructive_projection",
    "NDS",
    "control_projection_fraction",
    "virtual_train_loss_delta_current_step",
    "controller_to_base_update_ratio",
    "basis_channel_energy_estimate",
    "readout_leakage_estimate",
]


def model_to_action_family(model_name: str, source_candidate_type: str = "", controller_policy: str = "") -> ActionFamily:
    upper = str(model_name).upper()
    source = str(source_candidate_type).lower()
    policy = str(controller_policy).lower()
    if "_FU" not in upper and not upper.endswith("_FU_NATIVE_JVP_REFRESH"):
        return ACTION_BY_ID["A0"]
    if "STABLE_RANDOM" in upper or "stable_random" in source:
        return ACTION_BY_ID["A12"]
    if "SIGNFLIP" in upper or "signflip" in source:
        return ACTION_BY_ID["A13"]
    if "CONTROL_CORRUPT" in upper or "SHUFFLED" in upper or "corrupt" in source or "shuffled" in source:
        return ACTION_BY_ID["A14"]
    if "CONTROL_RANDOM" in upper or "random" in source:
        return ACTION_BY_ID["A11"]
    if "DCHE" in upper and "FU_NATIVE" in upper:
        return ACTION_BY_ID["A8"]
    if ("SCALE010" in upper or "SCALE0_10" in upper or "scale0.10" in source or "scale0.10" in policy) and (
        "JVP_REFRESH" in upper or "jvp" in source or "jvp" in policy
    ):
        return ACTION_BY_ID["A7"]
    if "JVP_REFRESH" in upper or "jvp" in source or "jvp" in policy:
        return ACTION_BY_ID["A6"]
    if "SPLIT" in upper and "VIRTUAL" in upper:
        return ACTION_BY_ID["A5"]
    if "SPLIT" in upper:
        return ACTION_BY_ID["A4"]
    if "RELEASE" in upper or "release" in policy:
        return ACTION_BY_ID["A3"]
    if "GATED" in upper or "ACTIONABLE" in upper or "virtual" in policy or "gated" in policy:
        return ACTION_BY_ID["A2"]
    return ACTION_BY_ID["A1"]


def action_family_numeric(action_id: str) -> int:
    try:
        return int(str(action_id).lstrip("A"))
    except ValueError:
        return -1


def is_control_action(action_id: str) -> int:
    family = ACTION_BY_ID.get(str(action_id))
    return int(family.is_control) if family else 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in {float("inf"), float("-inf")}:
        return default
    return out


def runtime_feature_row(row: dict[str, Any], action: ActionFamily) -> dict[str, Any]:
    virtual_delta = safe_float(row.get("virtual_loss_guided")) - safe_float(row.get("virtual_loss_base"))
    return {
        "U_task_usefulness": safe_float(row.get("U_task_usefulness"), safe_float(row.get("U_task_usefulness_mean"))),
        "source_loss_boundary_distance": safe_float(row.get("source_loss_boundary_distance")),
        "action_family_numeric": action_family_numeric(action.action_id),
        "source_age": safe_float(row.get("source_age"), safe_float(row.get("Staleness"))),
        "source_norm": safe_float(row.get("source_norm")),
        "J_reachable_residual_estimate": safe_float(row.get("jacobian_reachable_projection_residual")),
        "JBa_source_cosine_estimate": safe_float(row.get("JBa_source_cosine")),
        "risk_score": safe_float(row.get("risk_score")),
        "destructive_projection": safe_float(row.get("destructive_projection")),
        "NDS": safe_float(row.get("NDS")),
        "control_projection_fraction": safe_float(row.get("ControlProjection"), safe_float(row.get("control_projection_fraction"))),
        "virtual_train_loss_delta_current_step": virtual_delta,
        "controller_to_base_update_ratio": safe_float(row.get("controller_to_base_update_ratio")),
        "basis_channel_energy_estimate": safe_float(row.get("basis_channel_energy_fraction")),
        "readout_leakage_estimate": safe_float(row.get("readout_channel_energy_fraction")),
    }
