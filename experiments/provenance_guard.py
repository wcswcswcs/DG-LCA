#!/usr/bin/env python3
"""Guardrails that permanently block non-empirical artifact generators."""

from __future__ import annotations

from pathlib import Path


PROXY_NOTICE = """NOT_EMPIRICAL: this script is a proxy/dry-run artifact generator.

It does not perform real dataloader + optimizer + epoch training. It may use
fixed score tables, deterministic jitter, or metric-row formulas. Its outputs
must not be cited as empirical results.

This repository is now in no-proxy mode. Fabricated/proxy experiment rows are
not allowed. Implement a real trainable module and real training runner instead.
"""


def forbid_non_empirical_execution(script: str | Path, reason: str = "") -> None:
    extra = f"\nReason: {reason}" if reason else ""
    raise SystemExit(f"{PROXY_NOTICE}{extra}\nBlocked file: {Path(script).name}")
