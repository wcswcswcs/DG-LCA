"""Top-level v22.13 loss-interface compatibility module.

The official FU operator consumes only the cotangent tensors produced by the
adapters re-exported here.  Adapter internals may define task losses; operator
core modules must remain name/formula blind.
"""

from __future__ import annotations

from dgkan.fu.loss_interface import *  # noqa: F401,F403

