#!/usr/bin/env python3
"""DG-KAN v7.4 real-only mechanism and kernelization runner.

v7.4 intentionally reuses the audited v7.3/v7.2 execution path for task,
gradient, efficiency, provenance, and no-fake/no-proxy checks.  The wrapper
only changes the plan/script identity and mirrors v7.3 postprocess artifacts
under v7.4 names so that v7.4 runs remain traceable to the v7.4 plan.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import run_gafu_v73_real as v73


PLAN_PATH = "docs/DG-KAN_v7.4_Expressivity_Optimizer_Geometry_Kernelization_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v74_real.py"


def _mirror_v74_artifacts(out_dir: Path) -> None:
    mapping = {
        "v73_route_decision.json": "v74_route_decision.json",
        "v73_provenance_audit.csv": "v74_provenance_audit.csv",
        "v73_manifest.json": "v74_manifest.json",
    }
    for src_name, dst_name in mapping.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)
    manifest_path = out_dir / "v74_manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["plan_path"] = PLAN_PATH
        data["script_path"] = SCRIPT_PATH
        data["runner_reuse"] = "run_gafu_v73_real.py"
        manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args) -> None:
    v73.PLAN_PATH = PLAN_PATH
    v73.SCRIPT_PATH = SCRIPT_PATH
    v73.run(args)
    _mirror_v74_artifacts(Path(args.out_dir))


if __name__ == "__main__":
    run(v73.parse_args())
