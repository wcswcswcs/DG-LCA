"""Source-loss readback helpers for v22.13.

These helpers score already-produced trajectories.  They do not construct FU
directions and must stay out of operator core decisions.
"""

from __future__ import annotations

from typing import Any


def source_gate_row(source_func: dict[int, float], source_loss: dict[int, float]) -> dict[str, Any]:
    h3200_func = float(source_func.get(3200, -999.0))
    h4800_func = float(source_func.get(4800, -999.0))
    h3200_loss = float(source_loss.get(3200, -999.0))
    h4800_loss = float(source_loss.get(4800, -999.0))
    r4800_func: float | str = h4800_func / h3200_func if abs(h3200_func) > 1.0e-12 else ""
    r4800_loss: float | str = h4800_loss / h3200_loss if abs(h3200_loss) > 1.0e-12 else ""
    c3 = int(
        all(float(source_func.get(h, -999.0)) >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and h3200_loss >= -1.0e-6
    )
    c4 = int(c3 and h4800_func >= 0.005 and h4800_loss >= -1.0e-6 and isinstance(r4800_func, float) and r4800_func >= 0.50)
    return {
        "R4800_over_3200_func": r4800_func,
        "R4800_over_3200_loss": r4800_loss,
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "TargetRetentionOnly_NotTaskUseful": int(any(float(source_func.get(h, -999.0)) >= 0.005 for h in [3200, 4800]) and h3200_loss < -1.0e-6),
    }


__all__ = ["source_gate_row"]

