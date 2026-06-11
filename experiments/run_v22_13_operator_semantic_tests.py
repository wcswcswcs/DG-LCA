#!/usr/bin/env python3
"""Standalone v22.13 role-blind operator semantic tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.operator_atoms_v22_13 import adapter_renaming_test, operator_law_test  # noqa: E402
from dgkan.fu.operator_core import OFFICIAL_OPERATOR_IDS, apply_operator  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, int_flag, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=2213)
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    torch.manual_seed(int(args.seed))
    logits = torch.randn(32, 5)
    d1 = torch.randn_like(logits)
    d2 = torch.randn_like(logits)
    provenance = []
    renaming = []
    law_rows = []
    for op_id in OFFICIAL_OPERATOR_IDS:
        out, telem = apply_operator(operator_id=op_id, logits=logits, cotangent=d1, seed=int(args.seed))
        row = telem.to_row()
        row["runtime_provenance_hash"] = str(hash(tuple(round(float(x), 6) for x in out.reshape(-1)[:16])))
        provenance.append(row)
        renaming.append(adapter_renaming_test(logits, d1, op_id, seed=int(args.seed)))
        law = operator_law_test(lambda d, local=op_id: apply_operator(operator_id=local, logits=logits, cotangent=d, seed=int(args.seed))[0], d1, d2)
        linear_required = 0
        law["operator_id"] = op_id
        law["operator_class"] = "nonlinear_norm_capped_operator"
        law["linearity_gate_required"] = linear_required
        law["homogeneity_gate_required"] = linear_required
        law["operator_law_pass"] = int(
            float(law["operator_lipschitz_ratio"]) <= 5.0
            and float(law["cotangent_permutation_equivariance_error"]) <= 1.0e-5
            and (not linear_required or (float(law["operator_linearity_error"]) <= 0.15 and float(law["operator_homogeneity_error"]) <= 0.10))
        )
        law_rows.append(law)
    write_rows(out_dir / "v22_13_runtime_provenance.csv", provenance)
    write_rows(out_dir / "v22_13_adapter_renaming_tests.csv", renaming)
    write_rows(out_dir / "v22_13_operator_law_tests.csv", law_rows)
    route = {
        "route": "OperatorSemanticTestsPass" if all(int_flag(r.get("adapter_renaming_pass")) for r in renaming) and all(int_flag(r.get("operator_law_pass")) for r in law_rows) else "R1-OperatorSemanticGateFailed",
        "adapter_renaming_pass_rows": sum(int_flag(r.get("adapter_renaming_pass")) for r in renaming),
        "operator_law_pass_rows": sum(int_flag(r.get("operator_law_pass")) for r in law_rows),
        "runtime_provenance_pass": int(all(int_flag(r.get("uses_adapter_name_for_direction")) == 0 and int_flag(r.get("uses_loss_formula_for_direction")) == 0 and int_flag(r.get("uses_labels_for_fu_core")) == 0 for r in provenance)),
        "blocker": "",
    }
    route["semantic_pass"] = int(route["route"] == "OperatorSemanticTestsPass" and route["runtime_provenance_pass"])
    write_json(out_dir / "v22_13_operator_semantic_route.json", route)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_operator_semantic_tests.py --seed {int(args.seed)} --out-dir {out_dir}", status="completed" if route["semantic_pass"] else "blocked", note=f"route={route['route']} semantic_pass={route['semantic_pass']}")


if __name__ == "__main__":
    main()
