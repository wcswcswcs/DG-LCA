#!/usr/bin/env python
"""v12.26.1 label-free-only functional bridge runner.

This runner deliberately avoids the B320-current label-initialized path.  It
constructs every base from an explicit A-LF label-free candidate, forces
``y_stats=None``, and reuses only train-stream/precommit functional primitives
from v12.25.  Labels are used only by the ordinary supervised training/eval
loop and by audit controls, never for base initialization or source direction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1223_failclosed_explore_open2_functional_rebuild as exp  # noqa: E402
import run_v1224_train_stream_functional_bridge as bridge  # noqa: E402
import run_v1225_composite_functional_bridge as comp  # noqa: E402


FORBIDDEN_TOKENS = (
    "trainprobe",
    "smalllabeloracle",
    "oraclelabel",
    "labelinit",
    "classmeanfromy",
    "yforstats",
)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def forbidden_token_present(*parts: Any) -> int:
    text = " ".join(str(p).lower() for p in parts if p is not None)
    return int(any(tok in text for tok in FORBIDDEN_TOKENS))


def clean_base_spec(input_dim: int, output_dim: int, base_candidate_id: str) -> tuple[Any, dict[str, Any]]:
    specs = exp.linea.ablation_specs(int(input_dim), int(output_dim))
    if base_candidate_id not in specs:
        raise KeyError(f"unknown v12.26 label-free base candidate: {base_candidate_id}")
    item = specs[base_candidate_id]
    spec = item["spec"]
    if int(item.get("uses_y_for_stats", 1)) != 0:
        raise RuntimeError(f"{base_candidate_id} is not label-free: uses_y_for_stats={item.get('uses_y_for_stats')}")
    if forbidden_token_present(base_candidate_id, getattr(spec, "candidate_id", ""), getattr(spec, "init_variant", "")):
        raise RuntimeError(
            f"{base_candidate_id} contains a forbidden official token: "
            f"spec_id={getattr(spec, 'candidate_id', '')} init={getattr(spec, 'init_variant', '')}"
        )
    return spec, item


def buffer_int(model: torch.nn.Module, name: str) -> int:
    value = getattr(model, name, None)
    if torch.is_tensor(value):
        try:
            return int(value.detach().cpu().item())
        except Exception:
            return 0
    return 0


def build_label_free_context(args: argparse.Namespace, source: dict[str, Any], device: torch.device):
    dataset = exp.v120._canonical_dataset(str(source["dataset"]))
    seed = int(float(source["seed"]))
    base_candidate_id = str(source["base_candidate_id"])
    load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=seed)
    data = exp.v120._load_vision_split(
        load_args,
        dataset,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.val_size),
    )
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    spec, _item = clean_base_spec(int(input_dim), int(output_dim), base_candidate_id)
    base = exp.linea.make_model(
        base_candidate_id,
        spec,
        int(input_dim),
        int(output_dim),
        x_train,
        y_train,
        device,
        int(seed) + int(args.base_seed_offset),
        0,
        "actual",
    )
    if buffer_int(base, "trainprobe_signal_init_applied") != 0:
        raise RuntimeError(f"{base_candidate_id} unexpectedly applied train-probe signal init")
    b = min(int(args.linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
    xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
    with torch.no_grad():
        base_query_logits = base(xq).detach()
    try:
        opt_updated = exp.v1252._take_adamw_window(base, xb, yb, 2.0e-3, 1.0e-3)
        opt_delta = exp.v1221._copy_state_delta(base, opt_updated)
    except Exception:
        opt_delta = {}
    return base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_query_logits, opt_delta, dataset, seed


def v1226_candidates() -> dict[str, dict[str, Any]]:
    def take(name: str, new_name: str, component: str | None = None) -> tuple[str, dict[str, Any]]:
        spec = copy.deepcopy(comp.CANDIDATES[name])
        spec["component_ids"] = component or str(spec.get("component_ids", name))
        return new_name, spec

    pairs = [
        take("F25-NG52-centeredDenoiseDirectBranch", "F26-D2-directGainTask"),
        take("F25-NG144-lowRankCouplingAlpha060PostCal125", "F26-D2-lowRankQuadGuard"),
        take("F25-NG156-crossRefCouplingAlpha060PostCal125", "F26-D2-crossRefQuadGuard"),
        take("F25-NG148-stableCenteredLowRankMixAlpha060PostCal125", "F26-D2-compositeTaskGuard"),
        take("F25-NG127-ng70SeedWeightAnchor060PostCal125", "F26-D2-postCalWeightAnchor"),
        take("F25-NG40-quarterResponseResidual-quadDirect-I27I26", "F26-D2-controlResidualComposite"),
        take("F25-NG114-tailClippedDenoiseLowTailMix-7030", "F26-D3-riskAwareTailMix"),
        take("F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect", "F26-D3-roleSwitchFreezeDirect"),
        take("F25-NG160-stableCenteredCrossRefMixAlpha060PostCal125", "F26-D3-sequentialGeometryTask"),
        take("F25-NG154-lowRankQuadDirectAlpha085PostCal125", "F26-D3-downscaleQuadPolicy"),
        take("F25-NG37-quarterResponseResidual-I27I26-8515", "F26-D3-partialResponseRiskVeto"),
        take("F25-NG138-stableCenteredDenoiseAlpha060PostCal125", "F26-D4-roleGainTransport"),
        take("F25-NG147-tailClippedLowRankCouplingAlpha060PostCal125", "F26-D4-quadReservoirGuard"),
        take("F25-NG158-entropyCrossRefCouplingAlpha060PostCal125", "F26-D4-lowRankLogitSubspace"),
        take("F25-NG44-featureCovDirectLift", "F26-D4-unlabeledCovarianceTransport"),
    ]
    out = dict(pairs)
    for name in out:
        if forbidden_token_present(name):
            raise RuntimeError(f"v12.26 functional candidate has forbidden token: {name}")
    return out


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v1226_functional_depth2")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--base-candidates", required=True)
    ap.add_argument("--datasets", default="KMNIST")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--norm-budget", type=float, default=0.009)
    ap.add_argument("--base-seed-offset", type=int, default=1226100)
    ap.add_argument("--compensation-batch", type=int, default=32)
    ap.add_argument("--ensemble-count", type=int, default=8)
    ap.add_argument("--train-seed-bases", default="12260400,12261400,12262400")
    ap.add_argument("--linec-seeds", default="12259500,12260600,12261600,12262600,12263600")
    ap.add_argument("--linec-batch", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--tail-proxy-tau", type=float, default=0.50)
    ap.add_argument("--ref-mode", default="bootstrap_mom", choices=["sequential", "bootstrap_mom"])
    ap.add_argument("--ref-seed-base", type=int, default=12262525)
    ap.add_argument("--train-seed-salt-override", type=int, default=39)
    ap.add_argument("--candidates", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    exp.ensure_dir(out_dir)
    device = torch.device("cuda:0" if str(args.device) == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    candidates = v1226_candidates()
    comp.CANDIDATES = candidates
    comp.role_scan.build_context = build_label_free_context  # type: ignore[assignment]

    selected_candidates = parse_csv(args.candidates) if str(args.candidates).strip() else list(candidates)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for base_candidate in parse_csv(args.base_candidates):
        if forbidden_token_present(base_candidate):
            raise RuntimeError(f"base candidate has forbidden token: {base_candidate}")
        for dataset in parse_csv(args.datasets):
            for seed in parse_ints(args.seeds):
                source = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "norm_budget": float(args.norm_budget),
                    "signed_direction": -1.0,
                    "actuator_id": "V1226LabelFreeOnlyFunctionalSource",
                    "base_candidate_id": base_candidate,
                }
                for candidate_id in selected_candidates:
                    if candidate_id not in candidates:
                        raise ValueError(f"unknown v12.26 candidate {candidate_id}")
                    for train_seed_base in parse_ints(args.train_seed_bases):
                        rr, ss = comp.run_candidate(args, source, candidate_id, train_seed_base, device)
                        for row in rr:
                            row.update(
                                {
                                    "stage": str(row.get("stage", "")).replace("V1225", "V1226"),
                                    "base_candidate_id": base_candidate,
                                    "base_is_label_free": 1,
                                    "uses_label_for_init": 0,
                                    "uses_y_for_stats": 0,
                                    "trainprobe_signal_init_applied": 0,
                                    "probe_dirs_from_labels": 0,
                                    "trainprobeP_enabled": 0,
                                    "trainprobeDirect_enabled": 0,
                                    "label_used_for_initialization": 0,
                                    "small_label_oracle_used": 0,
                                    "shuffled_label_init_used": 0,
                                    "validation_or_test_used_for_init": 0,
                                    "dataset_name_used_for_init": 0,
                                    "forbidden_token_present": forbidden_token_present(base_candidate, candidate_id),
                                    "uses_query_reference": 0,
                                    "uses_validation_or_test": 0,
                                    "uses_linec_hard_target_for_direction": 0,
                                }
                            )
                        ss.update(
                            {
                                "base_candidate_id": base_candidate,
                                "base_is_label_free": 1,
                                "uses_label_for_init": 0,
                                "uses_y_for_stats": 0,
                                "label_used_for_initialization": 0,
                                "forbidden_token_present": forbidden_token_present(base_candidate, candidate_id),
                            }
                        )
                        rows.extend(rr)
                        summaries.append(ss)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    aggregate_rows: list[dict[str, Any]] = []
    keys = sorted({(str(s["base_candidate_id"]), str(s["candidate_id"]), str(s["dataset"]), str(s["seed"])) for s in summaries})
    for base_candidate, candidate_id, dataset, seed in keys:
        group = [
            s for s in summaries
            if str(s["base_candidate_id"]) == base_candidate
            and str(s["candidate_id"]) == candidate_id
            and str(s["dataset"]) == dataset
            and str(s["seed"]) == seed
        ]
        maj = sum(exp.safe_int(g.get("strict_majority_pass"), 0) for g in group)
        allp = sum(exp.safe_int(g.get("strict_all_pass"), 0) for g in group)
        exp_count = sum(exp.safe_int(g.get("exploration_gate_pass"), 0) for g in group)
        aggregate_rows.append(
            {
                "stage": "V1226_LABEL_FREE_FUNCTIONAL_AGGREGATE",
                "base_candidate_id": base_candidate,
                "candidate_id": candidate_id,
                "dataset": dataset,
                "seed": seed,
                "train_shuffle_rows": len(group),
                "train_shuffle_strict_majority_pass_count": maj,
                "train_shuffle_strict_all_pass_count": allp,
                "train_shuffle_exploration_pass_count": exp_count,
                "train_shuffle_robust_majority_pass": int(maj >= 2),
                "train_shuffle_robust_all_pass": int(allp >= 2),
                "exploration_train_shuffle_robust_pass": int(exp_count >= 3),
                "best_source_vs_noop": max(comp.fnum(g.get("source_vs_noop"), -999.0) for g in group),
                "best_source_vs_control": max(comp.fnum(g.get("source_vs_best_control"), -999.0) for g in group),
                "best_linec_seed_pass_count": max(exp.safe_int(g.get("LineC_seed_pass_count"), 0) for g in group),
                "base_is_label_free": 1,
                "promotion_allowed": 0,
                "no_fake": 1,
            }
        )
    rows.extend(aggregate_rows)
    csv_path = out_dir / f"{args.artifact_prefix}.csv"
    exp.write_csv_rows(csv_path, rows)
    result = {
        "stage": "V1226_LABEL_FREE_ONLY_FUNCTIONAL_RESULT",
        "artifact_csv": exp.rel(csv_path),
        "base_candidates": parse_csv(args.base_candidates),
        "candidates": selected_candidates,
        "candidate_rows": len(summaries),
        "aggregate_rows": len(aggregate_rows),
        "any_exploration_pass": int(any(exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0) for r in aggregate_rows)),
        "any_strict_majority_pass": int(any(exp.safe_int(r.get("strict_majority_pass"), 0) for r in summaries)),
        "any_strict_all_pass": int(any(exp.safe_int(r.get("strict_all_pass"), 0) for r in summaries)),
        "combined_bridge_any_train_shuffle_robust_majority_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0) for r in aggregate_rows)),
        "combined_bridge_any_train_shuffle_robust_all_pass": int(any(exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregate_rows)),
        "best_candidate_aggregates": sorted(
            aggregate_rows,
            key=lambda r: (
                exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0),
                exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0),
                comp.fnum(r.get("best_source_vs_control"), -999.0),
                comp.fnum(r.get("best_source_vs_noop"), -999.0),
            ),
            reverse=True,
        )[:20],
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(out_dir / f"{args.artifact_prefix}_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
