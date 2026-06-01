#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, json, math, sys, time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp


def train_short(model, x_train, y_train, batch_size, epochs, lr, weight_decay, seed, device):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed))
    times = []
    for _ in range(int(epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(batch_size)):
            idx = perm[off: off + int(batch_size)]
            opt.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(torch.tensor(times).quantile(0.90).item()) if times else ''


def pick_source(out_dir: Path) -> dict[str, str]:
    p3 = [r for r in exp.read_csv_rows(out_dir / 'v1223_functional_p3.csv') if exp.safe_int(r.get('p3_cloned_gate_pass'), 0) == 1]
    if not p3:
        raise RuntimeError('No passing P3 row; P4 short-run is not allowed')
    row = p3[0]
    safety = exp.read_csv_rows(out_dir / 'v1223_actuator_safety_roleaware.csv')
    candidates = [r for r in safety if r.get('actuator_id') == row.get('source_actuator') and r.get('dataset') == row.get('source_dataset') and str(r.get('seed')) == str(row.get('source_seed')) and exp.safe_int(r.get('role_safe_movement_pass'), 0) == 1 and exp.safe_int(r.get('release_audit_pass'), 0) == 1 and exp.safe_int(r.get('exploratory_release_gate'), 0) == 1]
    if not candidates:
        raise RuntimeError('Passing P3 source row not found in actuator safety CSV')
    return max(candidates, key=lambda r: (exp.safe_float(r.get('CouplingR2_delta'), -999.0), exp.safe_float(r.get('control_gap'), -999.0)))


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
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('P4 short-run audit requires CUDA; CPU-offload is not allowed')
    torch.cuda.set_device(device)
    source = pick_source(out_dir)
    dataset = exp.v120._canonical_dataset(str(source['dataset']))
    seed = int(float(source['seed']))
    budget = float(source['norm_budget'])
    signed = float(source['signed_direction'])
    actuator = str(source['actuator_id'])
    load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=seed)
    data = exp.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    spec = exp.linea.ablation_specs(int(input_dim), int(output_dim))['A1-noYForStats']['spec']
    base = exp.linea.make_model('A1-noYForStats', spec, int(input_dim), int(output_dim), x_train, y_train, device, int(seed) + 1223000, 0)
    b = min(int(args.linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
    xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
    with torch.no_grad():
        base_logits = base(xq).detach()
    try:
        opt_updated = exp.v1252._take_adamw_window(base, xb, yb, 2.0e-3, 1.0e-3)
        opt_delta = exp.v1221._copy_state_delta(base, opt_updated)
    except Exception:
        opt_delta = {}
    variants = [
        ('NoOpMatchedOverhead', 'noop_control', 0.0, signed),
        (actuator, 'p3_source_actuator', budget, signed),
        ('DirectLogitCompensatedRandomControl', 'direct_logit_compensated_control', budget, signed),
    ]
    rows: list[dict[str, Any]] = []
    for idx, (aid, kind, bgt, sgn) in enumerate(variants):
        trial = copy.deepcopy(base).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(bgt) * 100000) + len(aid) * 17 + (1 if sgn > 0 else 2))
        role = exp.apply_v1223_actuator(trial, aid, float(bgt), float(sgn), gen, opt_delta)
        direct = {'direct_logit_compensation_applied': 0, 'direct_logit_compensation_pre_drift': '', 'direct_logit_compensation_post_drift': '', 'direct_logit_compensation_norm': ''}
        if aid in {'I24-DirectLogitCompensatedQuadRelease', 'I25-DirectLogitCompensatedShadowRelease', 'DirectLogitCompensatedRandomControl'}:
            direct = exp.apply_direct_readout_logit_compensation(trial, xq, base_logits)
        init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        q90 = train_short(trial, x_train, y_train, int(args.batch_size), int(args.epochs), float(args.lr), float(args.weight_decay), seed + 12234000 + idx, device)
        final_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        try:
            linec = exp.v1221._linec_metrics(trial, xb, yb, xq, yq, seed + 12235000 + idx, int(args.linec_sketch_dim))
        except Exception as exc:
            linec = {'error': f'{type(exc).__name__}: {exc}'}
        rows.append({'stage': 'V1223_FUNCTIONAL_P4_SHORT', 'variant_kind': kind, 'actuator_id': aid, 'source_actuator': actuator, 'dataset': dataset, 'seed': seed, 'norm_budget': bgt, 'signed_direction': sgn, 'role': role, 'epochs': int(args.epochs), 'train_size': int(args.train_size), 'val_size': int(args.val_size), 'init_acc': init_eval.get('acc', ''), 'init_NLL': init_eval.get('NLL', ''), 'init_ECE': init_eval.get('ECE', ''), 'init_CEp99': init_eval.get('CEp99', ''), 'final_acc': final_eval.get('acc', ''), 'final_NLL': final_eval.get('NLL', ''), 'final_ECE': final_eval.get('ECE', ''), 'final_CEp99': final_eval.get('CEp99', ''), **direct, 'step_time_q90_ms': q90, **{f'linec_{k}': v for k, v in linec.items()}})
    by = {r['variant_kind']: r for r in rows}
    src, noop, ctrl = by['p3_source_actuator'], by['noop_control'], by['direct_logit_compensated_control']
    src_acc = exp.safe_float(src.get('final_acc'), float('nan'))
    noop_acc = exp.safe_float(noop.get('final_acc'), float('nan'))
    ctrl_acc = exp.safe_float(ctrl.get('final_acc'), float('nan'))
    src_nll = exp.safe_float(src.get('final_NLL'), float('nan'))
    noop_nll = exp.safe_float(noop.get('final_NLL'), float('nan'))
    src_cep99 = exp.safe_float(src.get('final_CEp99'), float('nan'))
    noop_cep99 = exp.safe_float(noop.get('final_CEp99'), float('nan'))
    p4_pass = int(math.isfinite(src_acc) and math.isfinite(noop_acc) and math.isfinite(ctrl_acc) and src_acc >= max(noop_acc, ctrl_acc) + 0.005 and (not math.isfinite(src_nll) or not math.isfinite(noop_nll) or src_nll <= noop_nll) and (not math.isfinite(src_cep99) or not math.isfinite(noop_cep99) or src_cep99 <= noop_cep99 + 0.05))
    summary = {'stage': 'V1223_FUNCTIONAL_P4_SHORT_SUMMARY', 'p4_opened': 1, 'p4_executed': 1, 'p4_pass': p4_pass, 'promotion_allowed': 0, 'status': 'executed_audit_only_no_s5_route' if p4_pass else 'executed_no_pass', 'source_actuator': actuator, 'dataset': dataset, 'seed': seed, 'source_final_acc': src_acc, 'noop_final_acc': noop_acc, 'control_final_acc': ctrl_acc, 'source_vs_noop_acc_delta': src_acc - noop_acc if math.isfinite(src_acc) and math.isfinite(noop_acc) else '', 'source_vs_control_acc_delta': src_acc - ctrl_acc if math.isfinite(src_acc) and math.isfinite(ctrl_acc) else '', 'no_fake': 1}
    rows.append(summary)
    exp.write_csv_rows(out_dir / 'v1223_functional_p4_short.csv', rows)
    exp.write_json(out_dir / 'v1223_functional_p4_short_summary.json', summary)
    state = json.loads((out_dir / 'v1223_route_decision.json').read_text())
    state.update({'p4_open': 1, 'p4_pass': p4_pass, 'p4_executed': 1, 'functional_p4_short_summary': exp.rel(out_dir / 'v1223_functional_p4_short_summary.json')})
    route, minimum_success, fail_reason = exp.decide_route(state)
    state.update({'route': route, 'minimum_success': minimum_success, 'fail_reason': fail_reason, 'official_success_reached': int(route.startswith('S5'))})
    zip_path = exp.package_zip(out_dir)
    state['code_review_packet'] = exp.rel(zip_path)
    state['code_review_packet_sha256'] = exp.sha256_file(zip_path)
    state.update(exp.write_required_manifest(out_dir))
    exp.write_hash_manifest(out_dir)
    exp.write_json(out_dir / 'v1223_route_decision.json', state)
    print(json.dumps(summary | {'route': state.get('route'), 'code_review_packet_sha256': state.get('code_review_packet_sha256'), 'required_artifact_missing_count': state.get('required_artifact_missing_count')}, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == '__main__':
    main()
