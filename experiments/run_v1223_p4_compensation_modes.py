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


def pick_source(out_dir: Path):
    p3 = [r for r in exp.read_csv_rows(out_dir / 'v1223_functional_p3.csv') if exp.safe_int(r.get('p3_cloned_gate_pass'), 0) == 1]
    if not p3:
        raise RuntimeError('No passing P3 row')
    row = p3[0]
    safety = exp.read_csv_rows(out_dir / 'v1223_actuator_safety_roleaware.csv')
    candidates = [r for r in safety if r.get('actuator_id') == row.get('source_actuator') and r.get('dataset') == row.get('source_dataset') and str(r.get('seed')) == str(row.get('source_seed')) and exp.safe_int(r.get('role_safe_movement_pass'), 0) == 1 and exp.safe_int(r.get('release_audit_pass'), 0) == 1 and exp.safe_int(r.get('exploratory_release_gate'), 0) == 1]
    if not candidates:
        raise RuntimeError('P3 source row missing')
    return max(candidates, key=lambda r: (exp.safe_float(r.get('CouplingR2_delta'), -999.0), exp.safe_float(r.get('control_gap'), -999.0)))


def solve_direct_delta(model, x_ref, base_logits_ref, ridge=1.0e-3):
    direct = getattr(model, 'direct_readout', None)
    if not torch.is_tensor(direct) or not hasattr(model, '_direct_logits_and_features'):
        return None, {'direct_logit_compensation_applied': 0, 'direct_logit_compensation_pre_drift': '', 'direct_logit_compensation_post_drift': '', 'direct_logit_compensation_norm': ''}
    with torch.no_grad():
        logits_before = model(x_ref).detach()
        pre = float((logits_before - base_logits_ref).abs().max().item())
        _dl, feats = model._direct_logits_and_features(x_ref)
        f = feats.detach().float()
        target = (base_logits_ref - logits_before).detach().float()
        gram = f @ f.transpose(0, 1)
        eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
        coef = torch.linalg.solve(gram + float(ridge) * eye, target)
        scale = math.sqrt(max(1, int(getattr(model, 'input_dim', f.shape[1]))))
        delta_w = (f.transpose(0, 1) @ coef) * float(scale)
        if tuple(delta_w.shape) != tuple(direct.shape):
            return None, {'direct_logit_compensation_applied': 0, 'direct_logit_compensation_pre_drift': pre, 'direct_logit_compensation_post_drift': pre, 'direct_logit_compensation_norm': ''}
    return delta_w.detach(), {'direct_logit_compensation_applied': 1, 'direct_logit_compensation_pre_drift': pre, 'direct_logit_compensation_norm': float(delta_w.norm().item())}


def apply_delta(model, delta_w, x_ref, base_logits_ref, info):
    direct = getattr(model, 'direct_readout', None)
    if delta_w is None or not torch.is_tensor(direct):
        info = dict(info)
        info.setdefault('direct_logit_compensation_post_drift', '')
        return info
    with torch.no_grad():
        direct.add_(delta_w.to(device=direct.device, dtype=direct.dtype))
        post = float((model(x_ref).detach() - base_logits_ref).abs().max().item())
    info = dict(info)
    info['direct_logit_compensation_post_drift'] = post
    return info


def run_mode(args, mode, base, source, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_train_logits, base_query_logits, opt_delta, device):
    seed = int(float(source['seed']))
    budget = float(source['norm_budget'])
    signed = float(source['signed_direction'])
    actuator = str(source['actuator_id'])
    ref_x = xb if mode.endswith('train') else xq
    ref_base = base_train_logits if mode.endswith('train') else base_query_logits
    shared_delta = None
    shared_info = None
    if mode.startswith('shared_source'):
        tmp = copy.deepcopy(base).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 100000) + len(actuator) * 17 + (1 if signed > 0 else 2))
        exp.apply_v1223_actuator(tmp, actuator, budget, signed, gen, opt_delta)
        shared_delta, shared_info = solve_direct_delta(tmp, ref_x, ref_base)
    variants = [('NoOpMatchedOverhead', 'noop_control', 0.0, signed), (actuator, 'p3_source_actuator', budget, signed), ('DirectLogitCompensatedRandomControl', 'direct_logit_compensated_control', budget, signed)]
    rows = []
    for idx, item in enumerate(variants):
        aid, kind, bgt, sgn = item
        trial = copy.deepcopy(base).to(device)
        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(bgt) * 100000) + len(aid) * 17 + (1 if sgn > 0 else 2))
        role = exp.apply_v1223_actuator(trial, aid, float(bgt), float(sgn), gen, opt_delta)
        direct = {'direct_logit_compensation_applied': 0, 'direct_logit_compensation_pre_drift': '', 'direct_logit_compensation_post_drift': '', 'direct_logit_compensation_norm': ''}
        if mode == 'none':
            pass
        elif mode.startswith('shared_source'):
            direct = apply_delta(trial, shared_delta, ref_x, ref_base, shared_info or direct)
        else:
            delta, info = solve_direct_delta(trial, ref_x, ref_base)
            direct = apply_delta(trial, delta, ref_x, ref_base, info)
        init_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        q90 = train_short(trial, x_train, y_train, int(args.batch_size), int(args.epochs), float(args.lr), float(args.weight_decay), seed + 12236000 + idx, device)
        final_eval = exp.v1252._classification_basic(trial, x_val, y_val)
        try:
            linec = exp.v1221._linec_metrics(trial, xb, yb, xq, yq, seed + 12237000 + idx, int(args.linec_sketch_dim))
        except Exception as exc:
            linec = {'error': f'{type(exc).__name__}: {exc}'}
        rows.append({'stage': 'V1223_FUNCTIONAL_P4_COMPENSATION_MODE', 'compensation_mode': mode, 'variant_kind': kind, 'actuator_id': aid, 'source_actuator': actuator, 'dataset': source['dataset'], 'seed': seed, 'norm_budget': bgt, 'signed_direction': sgn, 'role': role, 'epochs': int(args.epochs), 'train_size': int(args.train_size), 'val_size': int(args.val_size), 'init_acc': init_eval.get('acc', ''), 'init_NLL': init_eval.get('NLL', ''), 'init_ECE': init_eval.get('ECE', ''), 'init_CEp99': init_eval.get('CEp99', ''), 'final_acc': final_eval.get('acc', ''), 'final_NLL': final_eval.get('NLL', ''), 'final_ECE': final_eval.get('ECE', ''), 'final_CEp99': final_eval.get('CEp99', ''), **direct, 'step_time_q90_ms': q90, **{f'linec_{k}': v for k, v in linec.items()}})
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
    rows.append({'stage': 'V1223_FUNCTIONAL_P4_COMPENSATION_MODE_SUMMARY', 'compensation_mode': mode, 'p4_pass': p4_pass, 'promotion_allowed': 0, 'source_final_acc': src_acc, 'noop_final_acc': noop_acc, 'control_final_acc': ctrl_acc, 'source_vs_noop_acc_delta': src_acc - noop_acc if math.isfinite(src_acc) and math.isfinite(noop_acc) else '', 'source_vs_control_acc_delta': src_acc - ctrl_acc if math.isfinite(src_acc) and math.isfinite(ctrl_acc) else '', 'status': 'pass_audit_only_no_s5_route' if p4_pass else 'executed_no_pass', 'no_fake': 1})
    return rows


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
    ap.add_argument('--modes', default='per_variant_query,shared_source_query,per_variant_train,shared_source_train,none')
    ap.add_argument('--artifact-prefix', default='v1223_functional_p4_compensation_modes')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('P4 compensation-mode audit requires CUDA')
    torch.cuda.set_device(device)
    source = pick_source(out_dir)
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
    rows = []
    for mode in [m.strip() for m in str(args.modes).split(',') if m.strip()]:
        rows.extend(run_mode(args, mode, base, source, x_train, y_train, x_val, y_val, xb, yb, xq, yq, base_train_logits, base_query_logits, opt_delta, device))
    csv_path = out_dir / f'{args.artifact_prefix}.csv'
    json_path = out_dir / f'{args.artifact_prefix}_summary.json'
    exp.write_csv_rows(csv_path, rows)
    summaries = [r for r in rows if r.get('stage') == 'V1223_FUNCTIONAL_P4_COMPENSATION_MODE_SUMMARY']
    any_pass = int(any(exp.safe_int(r.get('p4_pass'), 0) for r in summaries))
    result = {'stage': 'V1223_FUNCTIONAL_P4_COMPENSATION_MODE_AGGREGATE', 'artifact_csv': exp.rel(csv_path), 'modes': len(summaries), 'any_p4_pass': any_pass, 'promotion_allowed': 0, 'summaries': summaries, 'no_fake': 1}
    exp.write_json(json_path, result)
    state = json.loads((out_dir / 'v1223_route_decision.json').read_text())
    state.update({'p4_compensation_mode_rows': len(rows), 'p4_compensation_mode_any_pass': any_pass, 'p4_compensation_mode_summary': exp.rel(json_path)})
    zip_path = exp.package_zip(out_dir)
    state['code_review_packet'] = exp.rel(zip_path)
    state['code_review_packet_sha256'] = exp.sha256_file(zip_path)
    state.update(exp.write_required_manifest(out_dir))
    exp.write_hash_manifest(out_dir)
    exp.write_json(out_dir / 'v1223_route_decision.json', state)
    print(json.dumps(result | {'code_review_packet_sha256': state.get('code_review_packet_sha256'), 'required_artifact_missing_count': state.get('required_artifact_missing_count')}, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == '__main__':
    main()
