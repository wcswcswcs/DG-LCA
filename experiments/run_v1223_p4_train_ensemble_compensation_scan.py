#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, csv, json, math, sys, time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp
import experiments.run_v1223_p4_compensation_modes as p4m
import experiments.run_v1223_p4_trainable_role_scan as role_scan


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        if x == "" or x is None:
            return default
        return float(x)
    except Exception:
        return default


def auc_error_time(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    total = max(points[-1][0] - points[0][0], 1.0e-9)
    area = 0.0
    for (t0, a0), (t1, a1) in zip(points[:-1], points[1:]):
        area += max(t1 - t0, 0.0) * (((1.0 - a0) + (1.0 - a1)) / 2.0)
    return float(area / total)


def direct_delta(model, x_ref, base_logits, ridge=1.0e-3):
    direct = getattr(model, "direct_readout", None)
    if not torch.is_tensor(direct) or not hasattr(model, "_direct_logits_and_features"):
        return None, float("nan")
    logits_before = model(x_ref).detach()
    pre = float((logits_before - base_logits).abs().max().item())
    _dl, feats = model._direct_logits_and_features(x_ref)  # type: ignore[attr-defined]
    f = feats.detach().float()
    target = (base_logits - logits_before).detach().float()
    gram = f @ f.transpose(0, 1)
    eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    coef = torch.linalg.solve(gram + float(ridge) * eye, target)
    scale = math.sqrt(max(1, int(getattr(model, "input_dim", f.shape[1]))))
    delta = (f.transpose(0, 1) @ coef) * float(scale)
    return (delta, pre) if tuple(delta.shape) == tuple(direct.shape) else (None, pre)


def apply_ensemble(model, refs):
    direct = getattr(model, "direct_readout", None)
    if not torch.is_tensor(direct) or not refs:
        return {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": float("nan"), "direct_logit_compensation_post_drift": float("nan"), "direct_logit_compensation_norm": float("nan")}
    deltas, pres = [], []
    with torch.no_grad():
        for x_ref, base_logits in refs:
            d, pre = direct_delta(model, x_ref, base_logits)
            pres.append(pre)
            if d is not None:
                deltas.append(d)
        if not deltas:
            pre = max([x for x in pres if math.isfinite(x)] or [float("nan")])
            return {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": pre, "direct_logit_compensation_post_drift": pre, "direct_logit_compensation_norm": float("nan")}
        delta = torch.stack(deltas).mean(dim=0)
        direct.add_(delta.to(device=direct.device, dtype=direct.dtype))
        posts = [float((model(x_ref).detach() - base_logits).abs().max().item()) for x_ref, base_logits in refs]
    return {"direct_logit_compensation_applied": 1, "direct_logit_compensation_pre_drift": max([x for x in pres if math.isfinite(x)] or [float("nan")]), "direct_logit_compensation_post_drift": max(posts), "direct_logit_compensation_norm": float(delta.norm().item())}


def train_eval(model, x_train, y_train, x_val, y_val, batch_size, epochs, lr, wd, seed, device, role_policy):
    params = [p for n, p in model.named_parameters() if p.requires_grad and role_scan.include_param(n, role_policy)]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(wd)) if params else None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    points = [(0.0, fnum(exp.v1252._classification_basic(model, x_val, y_val).get("acc")))]
    elapsed, times = 0.0, []
    if opt is not None:
        for _epoch in range(int(epochs)):
            perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
            for off in range(0, int(x_train.shape[0]), int(batch_size)):
                idx = perm[off:off + int(batch_size)]
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
                loss.backward(); opt.step()
                torch.cuda.synchronize(device)
                dt = (time.perf_counter() - t0) * 1000.0
                elapsed += dt; times.append(dt)
            points.append((elapsed, fnum(exp.v1252._classification_basic(model, x_val, y_val).get("acc"))))
    ev = exp.v1252._classification_basic(model, x_val, y_val)
    q90 = float(torch.tensor(times).quantile(0.90).item()) if times else float("nan")
    return ev, q90, auc_error_time(points)


def run_one(args, ensemble_count: int, device):
    source = p4m.pick_source(Path(args.out_dir))
    base, x_train, y_train, x_val, y_val, xb, yb, xq, yq, _base_query_logits, opt_delta, dataset, seed = role_scan.build_context(args, source, device)
    micro = min(int(args.compensation_batch), int(x_train.shape[0]))
    refs = []
    with torch.no_grad():
        for i in range(int(ensemble_count)):
            start = i * micro
            if start >= int(x_train.shape[0]):
                break
            x_ref = x_train[start:min(start + micro, int(x_train.shape[0]))]
            refs.append((x_ref, base(x_ref).detach()))
    budget, signed = float(source["norm_budget"]), float(source["signed_direction"])
    specs = [("noop_control", "NoOpMatchedOverhead", 0.0), ("p3_source_actuator", "I26-TrainDirectLogitCompensatedQuadRelease", budget), ("train_ensemble_random_control", "TrainDirectLogitCompensatedRandomControl", budget)]
    rows=[]
    finals={}
    for idx,(kind,aid,bgt) in enumerate(specs):
        trial=copy.deepcopy(base).to(device)
        gen=torch.Generator(device=device).manual_seed(int(seed)+int(abs(bgt)*100000)+len(aid)*17+(1 if signed>0 else 2))
        role=exp.apply_v1223_actuator(trial,aid,bgt,signed,gen,opt_delta)
        direct={"direct_logit_compensation_applied":0,"direct_logit_compensation_pre_drift":"","direct_logit_compensation_post_drift":"","direct_logit_compensation_norm":""}
        if aid in {"I26-TrainDirectLogitCompensatedQuadRelease","TrainDirectLogitCompensatedRandomControl"}:
            direct=apply_ensemble(trial,refs)
        ev,q90,auc=train_eval(trial,x_train,y_train,x_val,y_val,args.batch_size,args.epochs,args.lr,args.weight_decay,int(seed)+int(args.train_seed_base)+idx+len(args.role_policy),device,args.role_policy)
        try:
            linec=exp.v1221._linec_metrics(trial,xb,yb,xq,yq,int(seed)+int(args.linec_seed_base)+idx,int(args.linec_sketch_dim))
        except Exception as exc:
            linec={"error":f"{type(exc).__name__}: {exc}"}
        row={"stage":"V1223_P4_TRAIN_ENSEMBLE_COMPENSATION_VARIANT","variant_kind":kind,"actuator_id":aid,"role":role,"ensemble_count":ensemble_count,"compensation_batch":micro,"dataset":dataset,"seed":seed,"norm_budget":bgt,"signed_direction":signed,"epochs":args.epochs,"lr":args.lr,"weight_decay":args.weight_decay,"final_acc":ev.get("acc",""),"final_NLL":ev.get("NLL",""),"final_ECE":ev.get("ECE",""),"final_CEp99":ev.get("CEp99",""),"step_time_q90_ms":q90,"auc_error_time":auc,**direct,**{f"linec_{k}":v for k,v in linec.items()}}
        rows.append(row); finals[kind]=row
    src,noop,ctrl=finals["p3_source_actuator"],finals["noop_control"],finals["train_ensemble_random_control"]
    src_acc,noop_acc,ctrl_acc=fnum(src.get("final_acc")),fnum(noop.get("final_acc")),fnum(ctrl.get("final_acc"))
    src_nll,noop_nll=fnum(src.get("final_NLL")),fnum(noop.get("final_NLL"))
    src_cep,noop_cep=fnum(src.get("final_CEp99")),fnum(noop.get("final_CEp99"))
    src_ece,noop_ece=fnum(src.get("final_ECE")),fnum(noop.get("final_ECE"))
    src_cpl,noop_cpl=fnum(src.get("linec_CouplingR2")),fnum(noop.get("linec_CouplingR2"))
    src_noise,noop_noise=fnum(src.get("linec_NoiseSignalLeak")),fnum(noop.get("linec_NoiseSignalLeak"))
    src_res,noop_res=fnum(src.get("linec_RealSignalReservoirRatio")),fnum(noop.get("linec_RealSignalReservoirRatio"))
    src_auc,noop_auc=fnum(src.get("auc_error_time")),fnum(noop.get("auc_error_time"))
    p=int(math.isfinite(src_acc) and math.isfinite(noop_acc) and math.isfinite(ctrl_acc) and src_acc>=max(noop_acc,ctrl_acc)+0.005 and src_nll<=noop_nll and src_cep<=noop_cep+0.05 and src_ece<=noop_ece+0.02 and src_auc<=noop_auc and src_cpl>=noop_cpl and src_noise<=noop_noise and src_res<=noop_res)
    summary={"stage":"V1223_P4_TRAIN_ENSEMBLE_COMPENSATION_SUMMARY","ensemble_count":ensemble_count,"compensation_batch":micro,"p4_train_ensemble_pass":p,"promotion_allowed":0,"source_final_acc":src_acc,"noop_final_acc":noop_acc,"control_final_acc":ctrl_acc,"source_vs_noop_acc_delta":src_acc-noop_acc,"source_vs_control_acc_delta":src_acc-ctrl_acc,"source_NLL":src_nll,"noop_NLL":noop_nll,"source_CEp99":src_cep,"noop_CEp99":noop_cep,"source_ECE":src_ece,"noop_ECE":noop_ece,"source_CouplingR2":src_cpl,"noop_CouplingR2":noop_cpl,"source_NoiseSignalLeak":src_noise,"noop_NoiseSignalLeak":noop_noise,"source_RealSignalReservoirRatio":src_res,"noop_RealSignalReservoirRatio":noop_res,"status":"pass_audit_only_train_ensemble_not_promoted" if p else "executed_no_pass","no_fake":1}
    rows.append(summary)
    return rows, summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True); ap.add_argument("--data-root",default="data"); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--no-download",action="store_true")
    ap.add_argument("--train-size",type=int,default=512); ap.add_argument("--val-size",type=int,default=256); ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--lr",type=float,default=0.002); ap.add_argument("--weight-decay",type=float,default=0.001); ap.add_argument("--role-policy",default="quad_only")
    ap.add_argument("--compensation-batch",type=int,default=32); ap.add_argument("--ensemble-counts",default="2,4,8"); ap.add_argument("--train-seed-base",type=int,default=12239400); ap.add_argument("--linec-seed-base",type=int,default=12239500); ap.add_argument("--linec-batch",type=int,default=32); ap.add_argument("--linec-sketch-dim",type=int,default=8); ap.add_argument("--artifact-prefix",default="v1223_p4_train_ensemble_compensation_scan")
    args=ap.parse_args(); out_dir=Path(args.out_dir); device=torch.device(args.device)
    if device.type != "cuda": raise RuntimeError("train ensemble compensation scan requires CUDA")
    torch.cuda.set_device(device)
    rows=[]; summaries=[]
    for count in [int(x.strip()) for x in str(args.ensemble_counts).split(",") if x.strip()]:
        rr,ss=run_one(args,count,device); rows.extend(rr); summaries.append(ss)
    csv_path=out_dir/f"{args.artifact_prefix}.csv"; json_path=out_dir/f"{args.artifact_prefix}_summary.json"
    exp.write_csv_rows(csv_path,rows)
    summaries.sort(key=lambda r:(exp.safe_int(r.get("p4_train_ensemble_pass"),0),exp.safe_float(r.get("source_vs_noop_acc_delta"),-999.0),exp.safe_float(r.get("source_vs_control_acc_delta"),-999.0)), reverse=True)
    result={"stage":"V1223_P4_TRAIN_ENSEMBLE_COMPENSATION_SCAN","artifact_csv":exp.rel(csv_path),"summary_rows":len(summaries),"any_pass":int(any(exp.safe_int(r.get("p4_train_ensemble_pass"),0) for r in summaries)),"best_summaries":summaries,"promotion_allowed":0,"no_fake":1}
    exp.write_json(json_path,result)
    state_path=out_dir/"v1223_route_decision.json"
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding="utf-8")); state.update({"p4_train_ensemble_compensation_summary":exp.rel(json_path),"p4_train_ensemble_compensation_any_pass":result["any_pass"],"p4_train_ensemble_compensation_promotion_allowed":0,"no_fake":1}); state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True))
if __name__=="__main__": main()
