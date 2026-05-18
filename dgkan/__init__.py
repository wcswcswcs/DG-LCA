"""DG-KAN v9 clean core package.

The package starts by making the experiment contract explicit.  Later phases
can move performance-critical code here without letting runner defaults decide
scientific claims.
"""

from .contracts import ContractViolation, validate_official_contract
from .specs import CandidateSpec, ContractFlags

__all__ = [
    "CandidateSpec",
    "ContractFlags",
    "ContractViolation",
    "validate_official_contract",
]
