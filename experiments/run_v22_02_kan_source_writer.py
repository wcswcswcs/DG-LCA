#!/usr/bin/env python3
"""v22.02 KAN source-bank writer and MLP-to-KAN mapping readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.kan_source_bank import kan_source_bank_row  # noqa: E402
from experiments.run_v22_02_common import (  # noqa: E402
    PYTHON,
    V2201_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    mean,
    read_rows,
    sha256_file,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2201_OFFICIAL))
    p.add_argument("--fresh-source-dir", default="")
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    source_path = source_dir / "v22_01_kan_source_writer_summary.csv"
    rows = []
    for row in read_rows(source_path):
        item = dict(row)
        item["v22_02_source_artifact"] = str(source_path)
        item["v22_02_source_artifact_sha256"] = sha256_file(source_path) if source_path.exists() else ""
        item.update(kan_source_bank_row(item))
        item["KAN_FU_S3_v22_02"] = int(
            finite_float(item.get("h800")) >= 0.005
            and finite_float(item.get("h3200")) >= 0.005
            and finite_float(item.get("h4800")) >= 0.005
            and int(item.get("h4800_positive_rows", 0) or 0) >= 4
        )
        rows.append(item)
    write_rows(out_dir / "v22_02_kan_source_bank_matrix.csv", rows)

    fresh_dir = Path(args.fresh_source_dir) if args.fresh_source_dir else out_dir
    source_summary = read_rows(fresh_dir / "v22_02_source_chain_summary.csv")
    mlp_rows = [r for r in source_summary if str(r.get("carrier")) == "MLP"]
    mapping = []
    for row in mlp_rows:
        hidden = finite_float(row.get("hidden_source_energy"), 0.0)
        readout = finite_float(row.get("readout_source_energy"), 0.0)
        total = hidden + readout
        mapping.append(
            {
                "v22_id": row.get("v22_id", ""),
                "h800": row.get("h800", ""),
                "h3200": row.get("h3200", ""),
                "h4800": row.get("h4800", ""),
                "hidden_source_energy": hidden,
                "readout_source_energy": readout,
                "bias_source_energy": row.get("bias_source_energy", ""),
                "other_source_energy": row.get("other_source_energy", ""),
                "hidden_source_fraction": row.get("hidden_source_fraction", ""),
                "readout_source_fraction": row.get("readout_source_fraction", ""),
                "bias_source_fraction": row.get("bias_source_fraction", ""),
                "source_subspace_rank": row.get("source_subspace_rank", ""),
                "source_singular_values": row.get("source_singular_values", ""),
                "interpretable_block_fraction": max(hidden, readout) / total if total > 0.0 else "",
                "matrix_block_alignment": row.get("matrix_block_alignment", ""),
                "parameter_alignment": row.get("parameter_alignment", ""),
                "source_reconstruction_error": row.get("source_reconstruction_error", ""),
                "MLP_to_DCHE_projection_retention": row.get("MLP_to_DCHE_projection_retention", ""),
                "MLP_to_DFOU_projection_retention": row.get("MLP_to_DFOU_projection_retention", ""),
                "mapping_status": "not_measured_by_current_runner" if total <= 0.0 else "measured",
            }
        )
    write_rows(out_dir / "v22_02_mlp_source_subspace_matrix.csv", mapping)
    pass_rows = sum(int_flag(r.get("KAN_FU_S3_v22_02")) for r in rows)
    decision = {
        "decision": "KANSourceBankWriterPass" if pass_rows else "KANSourceBankMismatch",
        "rows": len(rows),
        "KAN_FU_S3_rows": pass_rows,
        "best_h4800": max([finite_float(r.get("h4800"), -999.0) for r in rows] or [-999.0]),
        "mlp_source_mapping_rows": len(mapping),
        "mlp_mapping_measured_rows": sum(str(r.get("mapping_status")) == "measured" for r in mapping),
        "promotion_allowed": 0,
    }
    write_json(out_dir / "v22_02_kan_source_writer_decision.json", decision)
    simple_svg(out_dir / "figures" / "fig_kan_source_bank_energy.svg", "KAN source bank energy", rows, "source_bank_fraction")
    simple_svg(out_dir / "figures" / "fig_mlp_source_subspace_spectrum.svg", "MLP source subspace spectrum", mapping, "interpretable_block_fraction")
    simple_svg(out_dir / "figures" / "fig_mlp_to_kan_projection_retention.svg", "MLP to KAN projection retention", mapping, "MLP_to_DCHE_projection_retention")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_kan_source_writer.py --source-dir {source_dir} --fresh-source-dir {fresh_dir}", status="completed", note=f"rows={len(rows)} decision={decision['decision']} source_artifact={source_path}")


if __name__ == "__main__":
    main()
