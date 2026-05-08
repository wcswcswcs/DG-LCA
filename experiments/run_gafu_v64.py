#!/usr/bin/env python3
"""Blocked non-empirical DG-KAN artifact generator.

This file previously generated or reported proxy/synthetic experiment rows. It
has been intentionally replaced by a hard-fail stub so fabricated data cannot be
regenerated or mistaken for real training output.
"""

from __future__ import annotations

from provenance_guard import forbid_non_empirical_execution

forbid_non_empirical_execution(__file__, 'v6.4 was mixed: P2 had real training, but later stages generated proxy/derived rows. Use experiments/run_gafu_v64_real_p2.py for the real P2 path.')
