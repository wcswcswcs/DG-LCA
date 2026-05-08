#!/usr/bin/env python3
"""Blocked non-empirical DG-KAN artifact generator.

This file previously generated or reported proxy/synthetic experiment rows. It
has been intentionally replaced by a hard-fail stub so fabricated data cannot be
regenerated or mistaken for real training output.
"""

from __future__ import annotations

from provenance_guard import forbid_non_empirical_execution

forbid_non_empirical_execution(__file__, 'analyze_gafu_v64 reports non-empirical proxy artifacts; rebuild an empirical analyzer over raw training traces.')
