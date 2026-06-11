#!/usr/bin/env python3
"""v22.05 S0.12 code, metric, and mechanism semantic truth gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.core import cosine, flat_grad  # noqa: E402
from dgkan.fu.debt_accounting import debt_accounting_unit_tests  # noqa: E402
from dgkan.fu.diffeomorphic_target import diffeomorphic_target_unit_tests  # noqa: E402
from dgkan.fu.fisher_metric import fisher_metric_unit_tests  # noqa: E402
from dgkan.fu.function_space_metrics import function_space_metric_unit_tests  # noqa: E402
from dgkan.fu.mechanisms import MECHANISMS, make_update, mechanism_contract_rows  # noqa: E402
from dgkan.fu.metric_projection import metric_projection_unit_tests  # noqa: E402
from dgkan.fu.rkhs_metric import rkhs_metric_unit_tests  # noqa: E402
from dgkan.fu.sobolev_metric import sobolev_metric_unit_tests  # noqa: E402
from dgkan.fu.source_chain import source_chain_unit_tests  # noqa: E402
from dgkan.fu.source_preservation import source_preservation_unit_tests  # noqa: E402
from dgkan.fu.terminal_erosion import terminal_erosion_unit_tests  # noqa: E402
from dgkan.fu.terminal_retention import terminal_retention_unit_tests  # noqa: E402
from dgkan.metrics.linec import run_linec_channel_golden_tests, run_linec_golden_tests  # noqa: E402
from dgkan.profiling.efficiency_v22_05 import efficiency_v22_05_unit_tests  # noqa: E402
from dgkan.profiling.kernel_gradcheck import kernel_correctness_row  # noqa: E402
from experiments.run_v22_05_common import PYTHON, append_exec, ensure_out, run_cmd, write_json, write_rows  # noqa: E402


REQUIRED_SOURCE_FILES = [
    "dgkan/fu/core.py",
    "dgkan/fu/source_chain.py",
    "dgkan/fu/terminal_retention.py",
    "dgkan/fu/terminal_erosion.py",
    "dgkan/fu/function_space_metrics.py",
    "dgkan/fu/sobolev_metric.py",
    "dgkan/fu/rkhs_metric.py",
    "dgkan/fu/fisher_metric.py",
    "dgkan/fu/metric_projection.py",
    "dgkan/fu/mechanisms.py",
    "dgkan/fu/debt_accounting.py",
    "dgkan/metrics/linec.py",
    "dgkan/kernels/fused_chebyshev_k3.py",
    "dgkan/kernels/fused_fourier_k2.py",
    "dgkan/kernels/rational_fused.py",
    "dgkan/kernels/rbf_sparse.py",
    "dgkan/profiling/efficiency_v20.py",
    "dgkan/profiling/efficiency_v21.py",
    "dgkan/profiling/efficiency_v22.py",
    "dgkan/profiling/efficiency_v22_03.py",
    "dgkan/profiling/efficiency_v22_04.py",
    "dgkan/profiling/efficiency_v22_05.py",
    "experiments/run_v22_05_common.py",
    "experiments/run_v22_05_s012_truth_gate.py",
    "experiments/run_v22_05_efficiency_reconfirm.py",
    "experiments/run_v22_05_drat_drbf_repair.py",
    "experiments/run_v22_05_metric_first_fu.py",
    "experiments/run_v22_05_metric_autopsy.py",
    "experiments/run_v22_05_kan_source_mapping.py",
    "experiments/run_v22_05_queue.py",
    "experiments/run_v22_05_finalize.py",
]


EXPECTED_V2205_MECHANISMS = [
    "M220-MetricFirstG0L2FU",
    "M221-MetricFirstDiagFisherFU",
    "M222-MetricFirstPopRiskDiagFU",
    "M223-MetricFirstSobolevH1FU",
    "M224-MetricFirstRKHSKNNFU",
    "M225-MetricFirstFisherRKHSFU",
    "M226-MetricFirstLowNDSFU",
    "M227-MetricFirstEnsembleFU",
    "M228-MetricSourceMemoryG0L2FU",
    "M229-MetricSourceMemoryDiagFisherFU",
    "M230-MetricSourceMemoryPopRiskDiagFU",
    "M231-MetricSourceMemorySobolevH1FU",
    "M232-MetricSourceMemoryRKHSKNNFU",
    "M233-MetricSourceMemoryFisherRKHSFU",
    "M234-MetricSourceMemoryLowNDSFU",
    "M235-MetricSourceMemoryEnsembleFU",
    "M236-MetricTargetLossCotangentFU",
    "M237-MetricTargetLowDegreeReadoutFU",
    "M238-MetricTargetB1TransferFU",
    "M239-MetricTargetSourceProjectedB3NullFU",
    "M240-HC2H3200NoProjectionFU",
    "M241-HC2H3200L2ProjectionFU",
    "M242-HC2H3200HalfProjectionFU",
    "M243-HC2H4000L2RecomputeProjectionFU",
    "M244-HC2H4000HalfRecomputeProjectionFU",
    "M245-MetricTargetSplitConsensusFU",
    "M246-MetricTargetPopRiskSNRFU",
    "M247-MetricTargetDiffeomorphicNoFoldFU",
    "M248-MetricTargetRandomMatchedFU",
    "M249-MetricTargetSignFlippedFU",
    "M250-MetricTargetDebtCalibratedSourceFU",
    "M251-MetricTargetObservableMidDebtSourceFU",
    "M252-MetricTargetObservableMidDebtM181BridgeFU",
    "M253-MetricTargetGradientObservableMidDebtFU",
    "M254-HC8PreH3200SourceChannelAntiWashoutFU",
    "M255-HC9RiskWeightedSplitConsensusSourceFU",
    "M256-HC10ViewConsistentSourceCarryFU",
    "M257-HC11ContinuousSourceCarryAntiWashoutFU",
    "M258-HC12NoiseOrthogonalSourceCarryFU",
    "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
    "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
]


BOOL_COLUMNS = [
    "uses_optimizer_primary",
    "uses_slow_state",
    "uses_matrix_block",
    "uses_function_space_metric",
    "uses_sobolev_metric",
    "uses_rkhs_metric",
    "uses_fisher_metric",
    "uses_basis_channel_metric",
    "implementation_is_prototype",
    "semantic_contract_declared",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-root", default=str(ROOT))
    p.add_argument("--mode", default="all")
    p.add_argument("--self-contained-import-check", type=int, default=0)
    return p


def _mode_enabled(mode: str, name: str) -> bool:
    return mode == "all" or mode == name


def _schema_rows(contract: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool, str]:
    out = []
    ok = True
    blockers = []
    for row in contract:
        mech = str(row.get("mechanism", ""))
        source_pattern = str(row.get("source_pattern", ""))
        for col in BOOL_COLUMNS:
            value = row.get(col, "")
            good = value in {0, 1, "0", "1"}
            out.append({"mechanism": mech, "column": col, "value": value, "pass": int(good)})
            if not good:
                ok = False
                blockers.append(f"{mech}:{col}")
        if source_pattern in {"0", "1", ""} and mech in EXPECTED_V2205_MECHANISMS:
            ok = False
            blockers.append(f"{mech}:source_pattern_missing")
    return out, ok, ";".join(blockers)


def _semantic_noncollapse_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(2205)
    device = torch.device("cpu")

    class TinyReadoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(16, 12)
            self.w2 = torch.nn.Parameter(torch.empty(12, 4))
            self.b2 = torch.nn.Parameter(torch.zeros(4))
            torch.nn.init.xavier_uniform_(self.w2)

        def frozen_readout_features(self, xb: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.fc1(self.flatten(xb)))

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            return self.frozen_readout_features(xb) @ self.w2 + self.b2

    model = TinyReadoutModel().to(device)
    x = torch.randn(8, 1, 4, 4, device=device)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device)
    contract = {str(r.get("mechanism")): r for r in mechanism_contract_rows()}
    updates: dict[str, Any] = {}
    for mech in EXPECTED_V2205_MECHANISMS:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).float(), y).backward()
        updates[mech] = make_update(model, mech, x, y, seed=2205).tensor.detach().clone()
    pair_rows = []
    mechanisms = list(updates)
    for i, left in enumerate(mechanisms):
        for right in mechanisms[i + 1 :]:
            a = updates[left]
            b = updates[right]
            an = float(torch.linalg.vector_norm(a.float()).item())
            bn = float(torch.linalg.vector_norm(b.float()).item())
            rel = float(torch.linalg.vector_norm((a - b).float()).item() / max(1.0e-12, an))
            norm_ratio = an / max(1.0e-12, bn)
            cos = cosine(a, b)
            alias = int(cos > 0.999 and 0.99 <= norm_ratio <= 1.01)
            group_l = str(contract.get(left, {}).get("semantic_noncollapse_group", ""))
            group_r = str(contract.get(right, {}).get("semantic_noncollapse_group", ""))
            declared = int((not alias) or (bool(group_l) and group_l == group_r))
            pair_rows.append(
                {
                    "mechanism_i": left,
                    "mechanism_j": right,
                    "update_cosine": cos,
                    "relative_l2_distance": rel,
                    "norm_ratio": norm_ratio,
                    "semantic_alias": alias,
                    "declared_alias_group_i": group_l,
                    "declared_alias_group_j": group_r,
                    "declared_alias_if_needed": declared,
                }
            )
    summary = [
        {
            "pairs": len(pair_rows),
            "semantic_alias_pairs": sum(int(r["semantic_alias"]) for r in pair_rows),
            "undeclared_alias_pairs": sum(1 for r in pair_rows if int(r["semantic_alias"]) and not int(r["declared_alias_if_needed"])),
            "pass": int(all(int(r["declared_alias_if_needed"]) for r in pair_rows)),
            "gradient_norm": float(torch.linalg.vector_norm(flat_grad(model, device).float()).item()),
        }
    ]
    return pair_rows, summary


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_root = Path(args.source_root)
    mode = str(args.mode)
    rows: list[dict[str, Any]] = []

    if _mode_enabled(mode, "required_source_files"):
        file_rows = [
            {"path": rel, "exists": int((source_root / rel).exists()), "size_bytes": (source_root / rel).stat().st_size if (source_root / rel).exists() else ""}
            for rel in REQUIRED_SOURCE_FILES
        ]
        write_rows(out_dir / "v22_05_required_source_files.csv", file_rows)
        rows.append(
            {
                "check": "required_source_files",
                "pass": int(all(int(r["exists"]) for r in file_rows)),
                "metric": "exists",
                "value": f"{sum(int(r['exists']) for r in file_rows)}/{len(file_rows)}",
                "blocker": ";".join(r["path"] for r in file_rows if not int(r["exists"])),
            }
        )

    if _mode_enabled(mode, "import_closure") or mode == "all":
        compile_cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "tests"]
        code, log = run_cmd(compile_cmd, cwd=source_root, timeout=600)
        (out_dir / "v22_05_compileall.log").write_text(log, encoding="utf-8")
        write_rows(out_dir / "v22_05_compileall.csv", [{"command": " ".join(compile_cmd), "returncode": code, "pass": int(code == 0)}])
        import_code, import_log = run_cmd(
            [
                PYTHON,
                "-c",
                "import dgkan.fu.function_space_metrics, dgkan.fu.sobolev_metric, dgkan.fu.rkhs_metric, dgkan.fu.fisher_metric, dgkan.fu.metric_projection, dgkan.profiling.efficiency_v22_05, experiments.run_v22_05_common, experiments.run_v22_05_s012_truth_gate",
            ],
            cwd=source_root,
            timeout=180,
        )
        (out_dir / "v22_05_import_closure.log").write_text(import_log, encoding="utf-8")
        import_pass = int(import_code == 0 and int(args.self_contained_import_check) == 1)
        write_rows(
            out_dir / "v22_05_import_closure.csv",
            [{"returncode": import_code, "pass": import_pass, "self_contained_import_check": int(args.self_contained_import_check)}],
        )
        write_rows(
            out_dir / "v22_05_clean_unzip_self_test.csv",
            [{"source_root": str(source_root), "self_contained_import_check": int(args.self_contained_import_check), "pass": import_pass}],
        )
        rows.extend(
            [
                {"check": "compileall", "pass": int(code == 0), "metric": "py_compile", "value": code, "blocker": "" if code == 0 else "compile_failed"},
                {"check": "import_closure", "pass": import_pass, "metric": "self_contained_import", "value": import_code, "blocker": "" if import_pass else "import_failed_or_not_self_contained"},
            ]
        )

    if _mode_enabled(mode, "linec_golden"):
        fast = run_linec_golden_tests(trials=24)
        channel = run_linec_channel_golden_tests()
        write_rows(out_dir / "v22_05_linec_fast_golden.csv", fast)
        write_rows(out_dir / "v22_05_linec_channel_golden.csv", channel)
        rows.extend(
            [
                {"check": "linec_fast_golden", "pass": int(all(int(r.get("pass", r.get("linec_golden_transfer_pass", 1))) for r in fast)), "metric": "tests", "value": f"{len(fast)}", "blocker": ""},
                {"check": "linec_channel_golden", "pass": int(all(int(r.get("pass", 0)) for r in channel)), "metric": "tests", "value": f"{len(channel)}", "blocker": ""},
            ]
        )

    if _mode_enabled(mode, "source_chain_tests"):
        tests = source_chain_unit_tests()
        write_rows(out_dir / "v22_05_source_chain_unit_tests.csv", tests)
        rows.append({"check": "source_chain_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "terminal_retention_tests"):
        tests = terminal_retention_unit_tests() + terminal_erosion_unit_tests() + source_preservation_unit_tests() + diffeomorphic_target_unit_tests() + debt_accounting_unit_tests()
        write_rows(out_dir / "v22_05_terminal_retention_unit_tests.csv", tests)
        rows.append({"check": "terminal_retention_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "metric_tests"):
        tests = function_space_metric_unit_tests() + fisher_metric_unit_tests() + sobolev_metric_unit_tests() + rkhs_metric_unit_tests() + metric_projection_unit_tests()
        write_rows(out_dir / "v22_05_function_space_metric_unit_tests.csv", tests)
        rows.append({"check": "metric_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if _mode_enabled(mode, "mechanism_contracts"):
        contract = mechanism_contract_rows()
        selected = [r for r in contract if str(r.get("mechanism", "")) in EXPECTED_V2205_MECHANISMS]
        schema_rows, schema_ok, schema_blocker = _schema_rows(selected)
        alias_rows, alias_summary = _semantic_noncollapse_audit()
        write_rows(out_dir / "v22_05_mechanism_contracts.csv", selected)
        write_rows(out_dir / "v22_05_mechanism_contract_schema.csv", schema_rows)
        write_rows(out_dir / "v22_05_semantic_alias_matrix.csv", alias_rows)
        write_rows(out_dir / "v22_05_semantic_noncollapse_summary.csv", alias_summary)
        present_ok = len(selected) == len(EXPECTED_V2205_MECHANISMS) and all(m in MECHANISMS for m in EXPECTED_V2205_MECHANISMS)
        alias_ok = bool(alias_summary) and int(alias_summary[0].get("pass", 0)) == 1
        rows.append(
            {
                "check": "mechanism_contracts",
                "pass": int(present_ok and schema_ok and alias_ok),
                "metric": "v22.05 schema+semantic",
                "value": f"{len(selected)}/{len(EXPECTED_V2205_MECHANISMS)};alias_undeclared={alias_summary[0].get('undeclared_alias_pairs', '') if alias_summary else ''}",
                "blocker": ";".join(x for x in ["" if present_ok else "missing_mechanism", schema_blocker, "" if alias_ok else "undeclared_semantic_alias"] if x),
            }
        )

    if _mode_enabled(mode, "kernel_gradcheck"):
        kernel_rows = [kernel_correctness_row("D-CHE"), kernel_correctness_row("D-FOU"), kernel_correctness_row("D-RAT"), kernel_correctness_row("D-RBF")]
        write_rows(out_dir / "v22_05_kernel_gradcheck.csv", kernel_rows)
        rows.append({"check": "kernel_gradcheck", "pass": int(all(int(r.get("kernel_correctness_official_pass", r.get("pass", 0))) for r in kernel_rows)), "metric": "kernels", "value": f"{len(kernel_rows)}", "blocker": ""})

    if _mode_enabled(mode, "profiler_phase_tests"):
        tests = efficiency_v22_05_unit_tests()
        write_rows(out_dir / "v22_05_profiler_phase_tests.csv", tests)
        rows.append({"check": "profiler_phase_tests", "pass": int(all(int(r.get("pass", 0)) for r in tests)), "metric": "tests", "value": f"{sum(int(r.get('pass', 0)) for r in tests)}/{len(tests)}", "blocker": ""})

    if rows:
        existing = []
        if (out_dir / "v22_05_code_truth_gate.csv").exists() and mode != "all":
            import csv

            with (out_dir / "v22_05_code_truth_gate.csv").open(newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        all_rows = existing + rows
        write_rows(out_dir / "v22_05_code_truth_gate.csv", all_rows)
        route = {
            "mode": mode,
            "pass": int(all(int(r.get("pass", 0)) for r in rows)),
            "failed_checks": ";".join(r.get("check", "") for r in rows if not int(r.get("pass", 0))),
            "self_contained_import_check": int(args.self_contained_import_check),
        }
        write_json(out_dir / "v22_05_code_route_decision.json", route)
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_05_s012_truth_gate.py --mode {mode} --source-root {source_root} --self-contained-import-check {int(args.self_contained_import_check)}",
            status="completed",
            note=f"pass={route['pass']} failed={route['failed_checks']}",
        )


if __name__ == "__main__":
    main()
