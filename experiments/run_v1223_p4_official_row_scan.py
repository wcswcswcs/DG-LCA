#!/usr/bin/env python
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m


def official_sources(out_dir, top_k):
    rows = exp.read_csv_rows(out_dir / 'v1223_actuator_safety_roleaware.csv')
    rows = [r for r in rows if r.get('basis_type') != 'matched_control' and exp.safe_int(r.get('official_release_gate'), 0) == 1]
    rows.sort(key=lambda r: (exp.safe_float(r.get('CouplingR2_delta'), -999.0), exp.safe_float(r.get('control_gap'), -999.0), -exp.safe_float(r.get('CEp99_delta'), 999.0)), reverse=True)
    if int(top_k) > 0:
        rows = rows[:int(top_k)]
    return rows


def build_context(args, source, device):
    dataset = exp.v120._canonical_dataset(str(source['dataset']))
    seed = int(float(source['seed']))
    load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=seed)
    data = exp.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    spec = exp.linea.ablation_specs(int(input_dim), int(output_dim))['A1-noYForStats']['spec']
    base = exp.linea.make_model('A1-noYForStats', spec, int(input_dim), int(output_dim), x_train, y_train, device, seed + 1223000, 0)
    b = min(int(args.linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
    xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
    with torch.no_grad():
        base_query_logits = base(xq).detach()
        base_train_logits = base(xb).detach()
    try:
        opt_updated = exp.v1252._take_adamw_window(base, xb, yb, 2.0e-3, 1.0e-3)
        opt_delta = exp.v1221._copy_state_delta(base, opt_updated)
    except Exception:
        opt_delta = {}
    return base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_train_logits, base_query_logits, opt_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no-download', action='store_true')
    ap.add_argument('--train-size', type=int, default=256)
    ap.add_argument('--val-size', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2.0e-3)
    ap.add_argument('--weight-decay', type=float, default=1.0e-3)
    ap.add_argument('--linec-batch', type=int, default=32)
    ap.add_argument('--linec-sketch-dim', type=int, default=8)
    ap.add_argument('--modes', default='per_variant_query')
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--artifact-prefix', default='v1223_functional_p4_official_row_scan')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('P4 official-row scan requires CUDA')
    torch.cuda.set_device(device)
    sources = official_sources(out_dir, int(args.top_k))
    all_rows = []
    for rank, source in enumerate(sources, start=1):
        ctx = build_context(args, source, device)
        for mode in [m.strip() for m in str(args.modes).split(',') if m.strip()]:
            rows = p4m.run_mode(args, mode, *ctx[:1], source, *ctx[1:], device)
            for row in rows:
                row['source_rank'] = rank
                row['source_dataset'] = source.get('dataset', '')
                row['source_seed'] = source.get('seed', '')
                row['source_norm_budget'] = source.get('norm_budget', '')
                row['source_signed_direction'] = source.get('signed_direction', '')
                row['source_CouplingR2_delta'] = source.get('CouplingR2_delta', '')
                row['source_control_gap'] = source.get('control_gap', '')
            all_rows.extend(rows)
    csv_path = out_dir / f'{args.artifact_prefix}.csv'
    json_path = out_dir / f'{args.artifact_prefix}_summary.json'
    exp.write_csv_rows(csv_path, all_rows)
    summaries = [r for r in all_rows if r.get('stage') == 'V1223_FUNCTIONAL_P4_COMPENSATION_MODE_SUMMARY']
    summaries.sort(key=lambda r: (exp.safe_int(r.get('p4_pass'), 0), exp.safe_float(r.get('source_vs_control_acc_delta'), -999.0), exp.safe_float(r.get('source_vs_noop_acc_delta'), -999.0)), reverse=True)
    result = {'stage': 'V1223_FUNCTIONAL_P4_OFFICIAL_ROW_SCAN', 'source_rows': len(sources), 'summary_rows': len(summaries), 'any_p4_pass': int(any(exp.safe_int(r.get('p4_pass'), 0) for r in summaries)), 'promotion_allowed': 0, 'artifact_csv': exp.rel(csv_path), 'best_summaries': summaries[:10], 'no_fake': 1}
    exp.write_json(json_path, result)
    state = json.loads((out_dir / 'v1223_route_decision.json').read_text())
    state.update({'p4_official_row_scan_rows': len(all_rows), 'p4_official_row_scan_any_pass': result['any_p4_pass'], 'p4_official_row_scan_summary': exp.rel(json_path)})
    zip_path = exp.package_zip(out_dir)
    state['code_review_packet'] = exp.rel(zip_path)
    state['code_review_packet_sha256'] = exp.sha256_file(zip_path)
    state.update(exp.write_required_manifest(out_dir))
    exp.write_hash_manifest(out_dir)
    exp.write_json(out_dir / 'v1223_route_decision.json', state)
    print(json.dumps(result | {'code_review_packet_sha256': state.get('code_review_packet_sha256'), 'required_artifact_missing_count': state.get('required_artifact_missing_count')}, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == '__main__':
    main()
