# DG-KAN v13.8 GenericPopRisk and KAN SNRtoBasisLift 执行日志

生成时间：2026-05-28（Asia/Singapore）

本日志只记录实际执行过的命令、代码文件、artifact 路径与可复现信息；不把 smoke / fallback 写成 promotion。

## 1. 计划文件

```bash
sed -n '1,220p' docs/DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md
sed -n '220,520p' docs/DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md
sed -n '520,760p' docs/DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md
sed -n '760,1030p' docs/DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md
```

## 2. 代码修改文件

```text
experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
experiments/run_v137_boundary_conditioned_poprisk_training.py
```

修改摘要：

```text
1. 新增 v13.8 runner，分离 Line G MLP generic optimizer、Line K0 SNR transfer、Line K1 KAN SNR-to-basis lift、Line D Non-RAT substrate audit。
2. v13.8 runner 输出 required manifest、route、forbidden audit、loss-interface audit、implementation readback、code packet、figures。
3. 修复 K1 writeback trace 被误拼入 failure table 的审计边界，新增 v138_rat_snr_lift_writeback_trace.csv。
4. v13.8 显式加入 K1-K7 alias：
   K1-RAT-ParamSNR-Only
   K2-RAT-ParamSNR-MinCoverGuard
   K3-RAT-RoleWiseSNRLift
   K4-RAT-DynamicSNRClusterLift
   K5-RAT-LowRankSNRCorrector
   K6-RAT-ParamSNR-Then-BasisConsolidation
   K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase
5. v13.7 shared helper 新增 train-stream Blend、rank-1 LowRankCorrector、MinCoverGuard、ThenBasisConsolidation、ThenCoverPhase。
6. route 中写入 compute_budgeted_run、real/synthetic budget 参数，避免把预算版写成 full 200-step plan-scale。
```

## 3. 语法检查

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py
conda run -n kan python -m py_compile experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
```

结果：

```text
py_compile pass
```

## 4. smoke：初版 runner

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/smoke_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/smoke_v138 --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X1 --synthetic-seeds 0 --synthetic-train-size 32 --synthetic-val-size 16 --k-losses CE --k-methods RAT-AdamW,RAT-ParameterSNRSoft,RAT-BasisSNR --train-steps 8 --batch-size 16 --log-interval 4 --mlp-hidden 64 --no-download
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
kan_s3_task_pass_count = 0
```

说明：只验证 artifact surface，不能作为 v13.8 结论。

## 5. prepatch full official 尝试

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138 --datasets MNIST,Fashion-MNIST,KMNIST --mlp-seeds 0,1,2,3,4,5,6,7,8,9 --mlp-seed-threshold 6 --mlp-methods MLP-AdamW,MLP-AdamW-SNREMA-Blend25-TrainLossQuantileTrust,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust,MLP-AdamW-SNREMA-Blend50-PerExampleGradientClip,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust-PerExampleGradientClip --real-train-size 1024 --real-val-size 512 --real-test-size 512 --real-epochs 3 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,RAT-ParameterSNRSoft,RAT-ParameterSNREMA,RAT-ParameterSNRRoleNorm,RAT-GroupSNR,RAT-GroupSNREMA,RAT-BasisSNR,RAT-BasisSNR-CoverPhaseSchedule --train-steps 120 --batch-size 64 --log-interval 40 --mlp-hidden 160 --loss-interface CE --no-download
```

结果：

```text
运行约 35 分钟后人工中断。
原因：实现复核发现 K1 surface 仍是 v13.7 方法名，未显式覆盖 v13.8 K1-K7；同时 runner 末尾统一写 artifact，长跑期间不可审计。
该次没有 route artifact，不作为实验结论。
```

## 6. smoke：K1-K7 修复后

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/smoke_v138_k1fix && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/smoke_v138_k1fix --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X1 --synthetic-seeds 0 --synthetic-train-size 32 --synthetic-val-size 16 --k-losses CE --k-methods RAT-AdamW,K1-RAT-ParamSNR-Only,K5-RAT-LowRankSNRCorrector,K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase --train-steps 6 --batch-size 8 --log-interval 3 --mlp-hidden 64 --no-download
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
kan_s3_task_pass_count = 0
```

## 7. official_v138：compute-budgeted plan-covered run

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138 --datasets MNIST,Fashion-MNIST,KMNIST --mlp-seeds 0,1,2,3,4,5,6,7,8,9 --mlp-seed-threshold 6 --mlp-methods MLP-AdamW,MLP-AdamW-SNREMA-Blend25-TrainLossQuantileTrust,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust,MLP-AdamW-SNREMA-Blend50-ActiveFractionSchedule,MLP-AdamW-SNREMA-Blend50-PerExampleGradientClip --real-train-size 512 --real-val-size 256 --real-test-size 256 --real-epochs 2 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K1-RAT-ParamSNR-Only,K2-RAT-ParamSNR-MinCoverGuard,K3-RAT-RoleWiseSNRLift,K4-RAT-DynamicSNRClusterLift,K5-RAT-LowRankSNRCorrector,K6-RAT-ParamSNR-Then-BasisConsolidation,K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase --train-steps 24 --batch-size 16 --log-interval 12 --mlp-hidden 160 --loss-interface CE --no-download
```

结果 artifact：

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138
```

route 摘要：

```text
route = R1-GenericSNROptimizerNoGo
compute_budgeted_run = 1
mlp_generic_10seed_dataset_pass_count = 0 / 3
kan_s3_task_pass_count = 3 / 7
kan_s4_task_pass_count = 1 / 7
snr_transfer_gate_pass = 1
nonrat_substrate_health_pass_count = 0
required_artifact_missing_count = 0
```

## 8. K1 targeted repair

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/targeted_k1_repair_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/targeted_k1_repair_v138 --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 64 --real-val-size 32 --real-test-size 32 --real-epochs 1 --synthetic-tasks X2,X3,X5,X6 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K2-RAT-ParamSNR-MinCoverGuard,K4-RAT-DynamicSNRClusterLift,K5-RAT-LowRankSNRCorrector,K6-RAT-ParamSNR-Then-BasisConsolidation,K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase --train-steps 80 --batch-size 16 --log-interval 40 --mlp-hidden 64 --loss-interface CE --no-download
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
kan_s3_task_pass_count = 1 / 4
kan_s4_task_pass_count = 0 / 4
required_artifact_missing_count = 0
```

## 9. MLP full-size 10-seed repair

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_full10seed_repair_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_full10seed_repair_v138 --datasets MNIST,Fashion-MNIST,KMNIST --mlp-seeds 0,1,2,3,4,5,6,7,8,9 --mlp-seed-threshold 6 --mlp-methods MLP-AdamW,MLP-AdamW-SNREMA-Blend50-TrainLossQuantileTrust,MLP-AdamW-SNREMA-Blend50-ActiveFractionSchedule,MLP-AdamW-SNREMA-Blend50-PerExampleGradientClip --real-train-size 1024 --real-val-size 512 --real-test-size 512 --real-epochs 3 --synthetic-tasks X1 --synthetic-seeds 0 --synthetic-train-size 32 --synthetic-val-size 16 --k-losses CE --k-methods RAT-AdamW --train-steps 1 --batch-size 64 --log-interval 1 --mlp-hidden 160 --loss-interface CE --no-download
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
mlp_generic_10seed_dataset_pass_count = 0 / 3
Fashion-MNIST seed_pass_count = 0 / 10
KMNIST seed_pass_count = 2 / 10
MNIST seed_pass_count = 2 / 10
required_artifact_missing_count = 0
```

## 10. 关键 artifact

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_mlp_generic_optimizer_10seed.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_snr_transfer_audit.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_rat_snr_lift_summary.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_nonrat_substrate_health.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/targeted_k1_repair_v138/v138_route_decision.json
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_full10seed_repair_v138/v138_route_decision.json
```

## 11. 用户再次追问后的 Non-RAT exact vertical slice 补跑

复核 v13.8 第 8.3 节后发现：official_v138 已执行 two Non-RAT family substrate-health repair scout，但没有单独记录 one Non-RAT exact lifetime / task-health vertical slice。因此补跑 Fourier exact vertical slice。

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_fou20_v138 && conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --run-id v138_nonrat_fou20_exact --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_fou20_v138 --artifact-prefix v138_nonrat_fou20 --device cuda:0 --data-root data --no-download --datasets MNIST --seeds 0 --candidates D-FOU20-LowFreqIdentityResidualHealthSubstrate --candidate-registry v1235 --train-size 64 --val-size 32 --batch-size 32 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 2 --workspace-profile-steps 5 --linec-batch-size 32 --linec-sketch-dim 32 --linec-seeds 0
```

输出 summary：

```text
workspace_rows = 1
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_rows = 1
hardening_executed_rows = 0
family_near_pass_rows = 0
promotion_allowed = 0
```

关键 artifact：

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_fou20_v138/v138_nonrat_fou20_workspace_truth.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_fou20_v138/v138_nonrat_fou20_hardening.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_fou20_v138/v138_nonrat_fou20_linec.csv
```

关键结果：

```text
candidate = D-FOU20-LowFreqIdentityResidualHealthSubstrate
dataset = MNIST
seed = 0
raw_memory_ratio_vs_mlp = 1.1053239288134742
incremental_memory_ratio_vs_mlp = 2.9649122807017543
step_ratio_vs_mlp = 1.5105504843235176
workspace_gate_pass = 0
hardening_executed = 0
hardening_skip_reason = workspace_gate_fail
linec_executed = 0
linec_skip_reason = workspace_gate_fail
```

## 12. 用户再次追问后的 route metadata 审计修复

复核计划 stop/go 条件与 official route：

```bash
rg -n "R1|R2|R3|R4|R5|stop|Stop|NoGo|no-go|continue|Continue|repair|修复|推荐|Recommendation|fallback|14\\.|8\\.3|14\\.1|14\\.3|14\\.5|17\\." docs/DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md
```

读取 official route：

```bash
python - <<'PY'
import json, pathlib
p=pathlib.Path('results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json')
d=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','compute_budgeted_run','mlp_generic_10seed_confirmed','mlp_generic_10seed_dataset_pass_count','snr_transfer_gate_pass','kan_s3_task_pass_count','kan_s4_task_pass_count','nonrat_substrate_health_pass_count','required_artifact_missing_count','forbidden_information_violation_count','kan_real_short_run_open_allowed']:
    print(f'{k}={d.get(k)}')
PY
```

发现：

```text
official_v138/v138_route_decision.json 缺少 final_stop_allowed 字段。
```

代码修改：

```text
experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
  build_route() 新增 final_stop_allowed 派生字段。
```

当前 official route metadata 修复：

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json
  补入 final_stop_allowed = 1。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
```

结果：

```text
py_compile pass
```

JSON 校验：

```bash
python -m json.tool results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json >/tmp/v138_route_check.json && python - <<'PY'
import json
p='results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json'
d=json.load(open(p))
for k in ['route','official_success_reached','promotion_allowed','final_stop_allowed','required_artifact_missing_count','kan_real_short_run_open_allowed']:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
kan_real_short_run_open_allowed = 0
```

说明：

```text
本次没有新增训练实验，没有新增 CSV 指标，没有修改 gate。
该修改只补齐 route metadata，方便后续审计确认 v13.8 no-go 是否允许停止。
```

刷新 code review packet：

```bash
python - <<'PY'
from pathlib import Path
import hashlib, json, zipfile
root = Path.cwd()
out = root / 'results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138'
packet = out / 'v138_code_review_packet.zip'
files = [
    root / 'experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py',
    root / 'experiments/run_v137_boundary_conditioned_poprisk_training.py',
    root / 'experiments/run_v137_mlp_snr_real_triage.py',
    root / 'docs/DG-KAN_v13.8_GenericPopRisk_and_KAN_SNRtoBasisLift_完整计划.md',
]
with zipfile.ZipFile(packet, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        if path.exists():
            zf.write(path, path.relative_to(root).as_posix())
sha = hashlib.sha256(packet.read_bytes()).hexdigest()
route_path = out / 'v138_route_decision.json'
route = json.loads(route_path.read_text())
route['code_review_packet_sha256'] = sha
route_path.write_text(json.dumps(route, indent=2, sort_keys=True) + '\n')
print(sha)
PY
python -m json.tool results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json >/tmp/v138_route_check_after_packet.json
```

结果：

```text
code_review_packet_sha256 = a27ad37b9ef8e10dcb657cd3fe0ca3e5844d38979959843905cf719e31d2e5e1
route JSON parse pass
```

最终 route 复核：

```bash
python - <<'PY'
import json
p='results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/official_v138/v138_route_decision.json'
d=json.load(open(p))
for k in ['route','official_success_reached','promotion_allowed','final_stop_allowed','required_artifact_missing_count','code_review_packet_sha256']:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
code_review_packet_sha256 = a27ad37b9ef8e10dcb657cd3fe0ca3e5844d38979959843905cf719e31d2e5e1
```

## 13. 用户再次追问后的 KAN 200-step full-step repair

复核发现 official_v138 是 compute-budgeted run，KAN train_steps 只有 24；为了避免把预算版当成 KAN line 闭合结论，本次补跑 KAN 200-step repair。MLP 侧只用 MNIST seed0 AdamW placeholder，因为 v13.8 第 14.1 节已要求 MLP 10-seed 失败后停止扩展 MLP SNR 小修。

执行指令：

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138 --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K1-RAT-ParamSNR-Only,K2-RAT-ParamSNR-MinCoverGuard,K3-RAT-RoleWiseSNRLift,K4-RAT-DynamicSNRClusterLift,K5-RAT-LowRankSNRCorrector,K6-RAT-ParamSNR-Then-BasisConsolidation,K7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase --train-steps 200 --batch-size 64 --log-interval 100 --mlp-hidden 160 --loss-interface CE --no-download
```

备注：

```text
进程约 72 分钟完成；期间 ps 观察到 python 子进程持续约 84%-85% CPU，占用稳定。
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
compute_budgeted_run = 1
synthetic_train_steps = 200
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.9576625823974609
snr_transfer_median_cos_group_vs_param = 0.8467055559158325
kan_s3_task_pass_count = 4 / 7
kan_s4_task_pass_count = 2 / 7
nonrat_substrate_health_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
kan_real_short_run_open_allowed = 0
```

关键 artifact：

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138/v138_route_decision.json
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138/v138_rat_snr_lift_summary.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138/v138_snr_transfer_audit.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138/v138_required_manifest.csv
```

聚合脚本：

```bash
python - <<'PY'
import csv,json,pathlib
base=pathlib.Path('results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/kan_full200_repair_v138')
route=json.loads((base/'v138_route_decision.json').read_text())
for k in ['route','promotion_allowed','kan_real_short_run_open_allowed','snr_transfer_gate_pass','kan_s3_task_pass_count','kan_s4_task_pass_count','required_artifact_missing_count']:
    print(f'{k}={route.get(k)}')
rows=list(csv.DictReader((base/'v138_rat_snr_lift_summary.csv').open()))
for task in sorted({r['task_or_dataset'] for r in rows}):
    rs=[r for r in rows if r['task_or_dataset']==task]
    s3=[r for r in rs if int(r.get('pass_s3','0') or 0)==1]
    seeds=sorted({r['seed'] for r in s3})
    losses=sorted({r['loss_interface'] for r in s3})
    task_s3=int(len(seeds)>=2 or len(losses)>=2)
    print(task, task_s3, len(s3), ','.join(seeds) or '-', ','.join(losses) or '-')
PY
```

结果摘要：

```text
X1 task_s3 = 0
X2 task_s3 = 1
X3 task_s3 = 1
X4 task_s3 = 0
X5 task_s3 = 1
X6 task_s3 = 1
X7 task_s3 = 0
```

## 14. K1-C freeze-cluster fallback

计划 K1-C 要求 dynamic cluster unstable across seeds 时尝试 freeze clusters after warmup。本次新增 K4F fallback。

代码修改：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
  SNRState 增加 frozen_gate / frozen_group_scores。
  snr_gate() 对 FreezeCluster 方法在 plasticity-open 后冻结 train-stream gate。

experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
  新增 K4F-RAT-DynamicSNRClusterFreeze alias。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py
conda run -n kan python -m py_compile experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py
```

结果：

```text
py_compile pass
```

执行指令：

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/k4_freeze_targeted_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/k4_freeze_targeted_v138 --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X1,X4,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K4-RAT-DynamicSNRClusterLift,K4F-RAT-DynamicSNRClusterFreeze --train-steps 200 --batch-size 64 --log-interval 100 --mlp-hidden 160 --loss-interface CE --no-download
```

结果：

```text
route = R1-GenericSNROptimizerNoGo
snr_transfer_gate_pass = 1
kan_s3_task_pass_count = 0 / 3
kan_s4_task_pass_count = 0 / 3
required_artifact_missing_count = 0
promotion_allowed = 0
code_review_packet_sha256 = c676357849a645614b429dd82b337f9b4fe53b60d7a84f0935a001ffa1910c74
```

K4F 对比摘要：

```text
X1 K4  task_pass = 0, s3_rows = 1, best_source = 0.4532723129555286
X1 K4F task_pass = 0, s3_rows = 1, best_source = 0.45247159472554344
X4 K4  task_pass = 0, s3_rows = 1, best_source = 0.4091217412041853
X4 K4F task_pass = 0, s3_rows = 1, best_source = 0.4098742882429317
X7 K4  task_pass = 0, s3_rows = 0, best_source = 0.30647107199183976
X7 K4F task_pass = 0, s3_rows = 0, best_source = 0.3065164241870677
```

说明：

```text
K4F 使用 train-stream SNR gate freeze，不使用 CEp99/NLL/ECE/LineC 生成方向。
K4F 没有把 X1/X4/X7 任一 family 推过 task-family gate。
```

## 15. Non-RAT RBF17/WAV16 exact substrate repair

计划 v13.8 要求 Line D 在 Non-RAT substrate-health blocker 时优先尝试 exact / fused / non-materialized repair。本次根据 200-step run 的 Non-RAT scout，选择最接近 workspace gate 的 RBF/WAV task-health candidates：

```text
D-RBF17-CompactCapacityK4HealthSubstrate
D-WAV16-SupportStableHatHealthSubstrate
```

执行指令：

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_rbf17_wav16_v138 && conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --run-id v138_nonrat_rbf17_wav16_exact --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_rbf17_wav16_v138 --artifact-prefix v138_nonrat_rbf17_wav16 --device cuda:0 --data-root data --no-download --datasets MNIST --seeds 0 --candidates D-RBF17-CompactCapacityK4HealthSubstrate,D-WAV16-SupportStableHatHealthSubstrate --candidate-registry v1235 --train-size 64 --val-size 32 --batch-size 32 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 2 --workspace-profile-steps 5 --linec-batch-size 32 --linec-sketch-dim 32 --linec-seeds 0
```

主要 artifact：

```text
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_rbf17_wav16_v138/v138_nonrat_rbf17_wav16_workspace_aggregate.json
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_rbf17_wav16_v138/v138_nonrat_rbf17_wav16_workspace_truth.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_rbf17_wav16_v138/v138_nonrat_rbf17_wav16_hardening.csv
results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/nonrat_exact_rbf17_wav16_v138/v138_nonrat_rbf17_wav16_linec.csv
```

aggregate 结果：

```text
workspace_rows = 2
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_rows = 2
hardening_executed_rows = 0
family_near_pass_rows = 0
promotion_allowed = 0
```

逐 candidate 结果：

| candidate | raw_memory_ratio_vs_mlp | incremental_memory_ratio_vs_mlp | step_ratio_vs_mlp | workspace_gate_pass | exact_kernel_implemented | materializes_basis_tensor | materializes_derivative_tensor | materializes_readout_grad_tensor | hardening_executed | linec_executed | skip_reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-RBF17-CompactCapacityK4HealthSubstrate | 1.2209141795727805 | 2.5254803675856308 | 1.2397470757373803 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | workspace_gate_fail |
| D-WAV16-SupportStableHatHealthSubstrate | 1.1849946032781948 | 2.5254803675856308 | 1.2997384483553653 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | workspace_gate_fail |

说明：

```text
1. RBF17/WAV16 raw memory ratio 接近 S1 workspace gate，但 incremental ratio 仍为 2.52548 > 2.00。
2. 两个 candidate 的 exact_kernel_implemented = 0，仍 materializes basis / derivative / readout_grad tensors。
3. hardening 与 LineC 因 workspace_gate_fail 被正确跳过。
4. 本次不能写成 Non-RAT substrate pass 或 promotion。
```

## 16. MLP G7 LogitNormTrust full-size 10-seed repair

用户再次追问“未达成则继续”后，重新对照 v13.8 计划第 5.7 节，发现 full-size MLP repair 已覆盖 G6 TrainLossQuantileTrust、G8 ActiveFractionSchedule、G9 PerExampleGradientClip，但未单独覆盖 G7 LogitNormTrust。为补齐计划内 trust repair chain，本次运行 G7 full-size 10-seed。

执行指令：

```bash
rm -rf results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_g7_logitnorm_full10seed_v138 && conda run -n kan python experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py --out-dir results/v13_8_generic_poprisk_and_kan_snr_to_basis_lift/mlp_g7_logitnorm_full10seed_v138 --datasets MNIST,Fashion-MNIST,KMNIST --mlp-seeds 0,1,2,3,4,5,6,7,8,9 --mlp-seed-threshold 6 --mlp-methods MLP-AdamW,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust --real-train-size 1024 --real-val-size 512 --real-test-size 512 --real-epochs 3 --synthetic-tasks X1 --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --k-losses CE --k-methods RAT-AdamW --train-steps 2 --batch-size 64 --log-interval 2 --mlp-hidden 160 --loss-interface CE --no-download
```

说明：

```text
该 run 只用于补齐 MLP G7 full-size repair。
KAN/Non-RAT 部分使用最小占位，因此该 run 的 KAN S3/S4 不用于 v13.8 KAN 结论。
```

route 结果：

```text
route = R1-GenericSNROptimizerNoGo
compute_budgeted_run = 1
real_train_size = 1024
real_val_size = 512
real_test_size = 512
real_epochs = 3
mlp_generic_10seed_confirmed = 0
mlp_generic_10seed_dataset_pass_count = 0 / 3
required_artifact_missing_count = 0
promotion_allowed = 0
```

10-seed 结果：

| dataset | seed_pass_count | seed_threshold | dataset_pass | passing_seeds |
|---|---:|---:|---:|---|
| Fashion-MNIST | 0 | 6 | 0 | - |
| KMNIST | 0 | 6 | 0 | - |
| MNIST | 0 | 6 | 0 | - |

最接近 rows：

| dataset | seed | source_vs_adamw | AUC_time_ratio | CEp99_delta | NLL_delta | ECE_delta | real_triage_pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | 0.05841256490086766 | 0.9135080659086646 | -0.31749391555786133 | 0.05088728666305542 | 0.008245181292295456 | 0 |
| Fashion-MNIST | 7 | -0.03371183383521881 | 1.0481259675329708 | 0.3629751205444336 | 0.013575732707977295 | 0.014368921518325806 | 0 |
| KMNIST | 9 | -0.03697864633723191 | 1.046387831223096 | -2.0538525581359863 | -0.056354641914367676 | -0.008521147072315216 | 0 |

判断：

```text
1. G7 full-size LogitNormTrust 没有打开任何 dataset。
2. MNIST seed0 source/AUC/CEp99/ECE 满足或改善，但 NLL_delta = 0.050887 > 0.02，因此不 pass。
3. Fashion-MNIST/KMNIST 最接近 rows 仍为 source negative / AUC_time_ratio > 1。
4. 至此 full-size MLP repair 已覆盖 G6/G7/G8/G9，仍没有 10-seed generic optimizer confirmation。
5. 该结果不能写成 KAN promotion。
```
