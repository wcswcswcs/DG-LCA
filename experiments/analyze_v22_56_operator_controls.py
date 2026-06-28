#!/usr/bin/env python3
"""Standalone entry point for the v22.56 operator-control audit."""

from __future__ import annotations

from experiments.run_v22_56_task_compatible_witnessed_preconditioner_fu import build_parser, analyze_operator_controls


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    analyze_operator_controls(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
