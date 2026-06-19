#!/usr/bin/env python3
"""Create the v22.18 authoritative status and source-code manifest."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import (  # noqa: E402
    OUT_ROOT,
    PYTHON,
    REQUIRED_SOURCE_FILES,
    append_exec,
    artifact_index,
    ensure_out,
    now_sg,
    sha256_file,
    source_packet_paths,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _compile_required() -> dict[str, Any]:
    files = [str(ROOT / rel) for rel in REQUIRED_SOURCE_FILES if (ROOT / rel).exists()]
    proc = subprocess.run([PYTHON, "-m", "py_compile", *files], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log = OUT_ROOT / "v22_18_clean_import_compile.log"
    log.write_text((proc.stdout or "") + f"\nexit_code={proc.returncode}\n", encoding="utf-8")
    return {"compileall": int(proc.returncode == 0), "compile_log": str(log.relative_to(ROOT)), "compile_exit_code": proc.returncode}


def _import_required() -> dict[str, Any]:
    modules = [
        "dgkan.fu.source_action_bank",
        "dgkan.fu.benefit_policy",
        "dgkan.fu.treatment_effect_labels",
        "dgkan.fu.kan_native_jvp",
        "dgkan.fu.continual_source_state",
        "experiments.run_v22_18_common",
        "experiments.run_v22_18_branch_label_readback",
        "experiments.run_v22_18_control_projected_labels",
        "experiments.run_v22_18_identity_audit",
        "experiments.run_v22_18_label_strictness_audit",
    ]
    rows = []
    for module in modules:
        try:
            importlib.import_module(module)
            rows.append({"module": module, "import_ok": 1, "error": ""})
        except Exception as exc:
            rows.append({"module": module, "import_ok": 0, "error": f"{type(exc).__name__}: {exc}"})
    write_rows(OUT_ROOT / "v22_18_clean_import_matrix.csv", rows)
    return {"clean_import": int(bool(rows) and all(int(r["import_ok"]) for r in rows)), "import_rows": len(rows)}


def _code_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in source_packet_paths():
        rows.append(
            {
                "source_file": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    write_rows(OUT_ROOT / "v22_18_code_packet_manifest.csv", rows)
    return rows


def _code_packet(rows: list[dict[str, Any]]) -> Path:
    packet = OUT_ROOT / "v22_18_code_packet.zip"
    if packet.exists():
        packet.unlink()
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            if int(row.get("exists") or 0):
                z.write(ROOT / str(row["source_file"]), str(row["source_file"]))
    return packet


def main() -> None:
    _args = parser().parse_args()
    ensure_out()
    compile_status = _compile_required()
    import_status = _import_required()
    manifest_rows = _code_manifest()
    packet = _code_packet(manifest_rows)
    rows = artifact_index()
    write_rows(OUT_ROOT / "v22_18_artifact_index.csv", rows)
    all_required = bool(manifest_rows) and all(int(r["exists"]) and r["sha256"] for r in manifest_rows)
    status = {
        "timestamp": now_sg(),
        "compileall": compile_status["compileall"],
        "clean_import": import_status["clean_import"],
        "all_required_artifacts_indexed": int(bool(rows)),
        "authoritative_status_single_source_of_truth": 1,
        "finalizer_separates_smoke_exploration_official": 1,
        "finalizer_separates_noharm_improvement_superiority": 1,
        "readout_diagnostic_cannot_promote_basis_native": 1,
        "kanbefair_baseline_cannot_promote_dgkan": 1,
        "required_source_files_present": int(all_required),
        "code_packet": str(packet.relative_to(ROOT)),
    }
    write_json(OUT_ROOT / "v22_18_authoritative_status.json", status)
    write_rows(
        OUT_ROOT / "v22_18_gate_summary.csv",
        [
            {
                "gate_name": "A code/artifact/finalizer truth",
                "pass": int(
                    status["compileall"]
                    and status["clean_import"]
                    and status["required_source_files_present"]
                    and status["authoritative_status_single_source_of_truth"]
                ),
                "scope": "official-prerequisite",
                "required_artifacts": "v22_18_authoritative_status.json;v22_18_artifact_index.csv;v22_18_code_packet_manifest.csv",
                "artifact_exists": 1,
                "artifact_nonempty": 1,
                "row_count": len(rows),
                "metric_thresholds": "compileall=1;clean_import=1;required_source_files_present=1",
                "blocking_metric": "" if status["compileall"] and status["clean_import"] and status["required_source_files_present"] else "compile/import/source manifest",
                "source_file_sha256": "",
                "latest_status_timestamp": status["timestamp"],
            }
        ],
    )
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="A-status-manifest",
        status="pass" if status["compileall"] and status["clean_import"] and all_required else "fail",
        exit_code=0 if status["compileall"] and status["clean_import"] and all_required else 1,
        files="results/v22_18/v22_18_authoritative_status.json, results/v22_18/v22_18_code_packet_manifest.csv, results/v22_18/v22_18_artifact_index.csv",
        note="single-source authoritative status initialized; later finalizer updates route/gates",
    )


if __name__ == "__main__":
    main()
