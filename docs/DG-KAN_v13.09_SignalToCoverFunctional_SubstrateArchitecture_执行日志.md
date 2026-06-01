# DG-KAN v13.09 SignalToCoverFunctional SubstrateArchitecture 执行日志

生成时间：2026-05-28（Asia/Singapore）

本日志只记录实际执行过的命令、文件与结果；不把 smoke / focused repair / Non-RAT substrate audit 写成 promotion。

## 1. 计划与代码读取

读取计划文件：

```bash
sed -n '1,220p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
rg -n "R2|cover formation|repair|blocker|K8|K9|K10|K11|Non-RAT|substrate" docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
sed -n '600,720p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
sed -n '720,835p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
sed -n '835,925p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
sed -n '930,1045p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
```

理解到的执行边界：

```text
1. v13.09 目标是验证 signal retention 是否能形成 stable basis cover，并带来 KAN-specific S3/S4。
2. K0 gate 需要 signal_retention_group / cos_group_vs_param / cover_purity / cover_churn。
3. K1 official 需要 K8-K12 signal-to-cover candidates，不能继续 K1-K7 小修。
4. LineC / CEp99 / NLL / ECE 只能 audit / gate，不能生成方向。
5. Non-RAT 必须先过 exact no-materialize substrate-health gate，不能直接进入 functional proof。
6. R2 表示 signal retention pass 但 cover formation fail；计划建议优先 substrate/base architecture reset。
```

## 2. 代码修改

修改文件：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
```

核心修改：

```text
1. 在 v13.7/v13.8 的 SNR runner 上增加 train-stream per-example gradient cover telemetry：
   cover_purity_mean / cover_purity_p10 / cover_churn / cover_load_gini /
   cover_specialization_entropy / signal_to_cover_score / split/merge/freeze /
   false_drop_fraction / false_keep_fraction。
2. 新增 K8-K12 方法语义：
   K8 GradientClusterCover k4/k8；
   K9 CoverSplitMerge lite/noMerge；
   K10 ParamSNRThenCover 3phase/slowConsolidate；
   K11 ReadoutBasisDecoupledSNR / ReadoutFirstBasisConsolidate；
   K12 VarianceReservoirProxyCoverGrowth / LowVarianceSignalCoverGrowth。
3. 新增 v13.09 runner：
   experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
4. 新 runner 输出 v139 required artifacts：
   route / manifest / provenance / K0 audit / K1 training+summary+LineC /
   Non-RAT exact substrate / MLP control / figures / code packet。
5. Non-RAT exact substrate 表明确区分 exact_kernel_implemented 与 materializes_*；
   不把 scout / proxy alias 写成 exact success。
```

合法性说明：

```text
1. K8-K12 update 只使用 train-stream gradient、parameter/group/basis telemetry。
2. 不使用 validation/test/future/query batch 生成方向。
3. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
4. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
5. 不改变 promotion gate。
```

## 3. 语法检查

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
```

结果：

```text
py_compile pass
```

## 4. Smoke

执行：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/smoke_v139 && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/smoke_v139 --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X1 --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --k-losses CE --k-methods RAT-AdamW,K8-RAT-GradientClusterCover-k4,K10-RAT-ParamSNRThenCover-3phase --train-steps 4 --batch-size 16 --log-interval 2 --mlp-hidden 80 --loss-interface CE --no-download
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 1.089290976524353
snr_transfer_median_cos_group_vs_param = 0.9345244765281677
cover_purity_median = 0.055969491600990295
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
nonrat_substrate_health_pass_count = 0
```

说明：smoke 只证明 runner / artifacts / K8/K10 surface 可执行，不能 promotion。

## 5. Official v13.09

执行：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139 && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139 --datasets MNIST,Fashion-MNIST,KMNIST --mlp-seeds 0,1,2,3,4 --mlp-seed-threshold 3 --mlp-methods MLP-AdamW,MLP-AdamW-SNREMA-Blend50-ActiveFractionSchedule,MLP-AdamW-SNRRoleNorm-Blend50-LogitNormTrust --real-train-size 1024 --real-val-size 512 --real-test-size 512 --real-epochs 3 --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,OLDK4-RAT-DynamicSNRClusterLift,OLDK7-RAT-ParamSNR-BlendAdamW-Then-CoverPhase,K8-RAT-GradientClusterCover-k4,K8-RAT-GradientClusterCover-k8,K9-RAT-CoverSplitMerge-lite,K9-RAT-CoverSplitMerge-noMerge,K10-RAT-ParamSNRThenCover-3phase,K10-RAT-ParamSNRThenCover-slowConsolidate,K11-RAT-ReadoutBasisDecoupledSNR,K11-RAT-ReadoutFirstBasisConsolidate,K12-RAT-VarianceReservoirProxyCoverGrowth,K12-RAT-LowVarianceSignalCoverGrowth --train-steps 200 --batch-size 32 --log-interval 50 --mlp-hidden 160 --loss-interface CE --no-download
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
promotion_allowed = 0
official_success_reached = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
compute_budgeted_run = 0
k_method_count = 13
mlp_method_count = 3
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7663904428482056
snr_transfer_median_cos_group_vs_param = 0.6665693521499634
cover_purity_median = 0.030393941327929497
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 4
kan_s4_task_pass_count = 0
kan_real_short_run_open_allowed = 0
nonrat_substrate_health_pass_count = 0
mlp_generic_10seed_dataset_pass_count = 2
mlp_generic_10seed_dataset_count = 3
forbidden_information_violation_count = 0
failure_rows = 0
```

artifact 检查：

```bash
conda run -n kan python -c "import json, pandas as pd; from pathlib import Path; base=Path('results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139'); route=json.loads((base/'v139_route_decision.json').read_text()); s=pd.read_csv(base/'v139_rat_signal_to_cover_summary.csv'); k0=pd.read_csv(base/'v139_signal_to_cover_audit.csv'); print('route', route['route']); print('required_missing', route['required_artifact_missing_count']); print('k0_rows', len(k0)); print('s3_rows', int(s.pass_s3.sum()), 's4_rows', int(s.pass_s4.sum())); print('manifest_rows', len(pd.read_csv(base/'v139_required_manifest.csv')));"
```

结果：

```text
route R2-SignalRetentionPassCoverFormationFail
required_missing 0
k0_rows 42
s3_rows 55
s4_rows 0
manifest_rows 30
```

主要 artifact：

```text
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_route_decision.json
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_required_manifest.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_code_provenance_audit.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_signal_to_cover_audit.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_rat_signal_to_cover_training.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_rat_signal_to_cover_summary.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_rat_signal_to_cover_linec.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_nonrat_exact_substrate.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_mlp_control_monitor.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_no_go_boundary.md
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_next_hypothesis_queue.md
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_code_review_packet.zip
```

## 6. Non-RAT exact substrate repair

执行：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/nonrat_exact_v139 && conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --run-id v139_nonrat_exact --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/nonrat_exact_v139 --artifact-prefix v139_nonrat_exact --device cuda:0 --data-root data --no-download --datasets MNIST --seeds 0 --candidates D-FOU20-LowFreqIdentityResidualHealthSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate,D-WAV16-SupportStableHatHealthSubstrate,D-CHE20-DegreeNormalizedReadoutHealthSubstrate --candidate-registry v1235 --train-size 64 --val-size 32 --batch-size 32 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 2 --workspace-profile-steps 5 --linec-batch-size 32 --linec-sketch-dim 32 --linec-seeds 0
```

结果：

```text
workspace_rows = 4
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_rows = 4
hardening_executed_rows = 0
family_near_pass_rows = 0
promotion_allowed = 0
```

逐候选 workspace 结果：

| candidate | raw_memory_ratio_vs_mlp | incremental_memory_ratio_vs_mlp | step_ratio_vs_mlp | workspace_gate_pass |
|---|---:|---:|---:|---:|
| D-FOU20-LowFreqIdentityResidualHealthSubstrate | 1.105324 | 2.964912 | 1.426523 | 0 |
| D-RBF17-CompactCapacityK4HealthSubstrate | 1.220914 | 2.525480 | 1.226483 | 0 |
| D-WAV16-SupportStableHatHealthSubstrate | 1.184995 | 2.525480 | 1.358163 | 0 |
| D-CHE20-DegreeNormalizedReadoutHealthSubstrate | 1.228871 | 2.811195 | 1.151777 | 0 |

exact audit：

```text
D-FOU20 exact_kernel_implemented=1, no_materialize_hard_gate_pass=1, manual_gradcheck_pass=1；
但 incremental_memory_ratio_vs_mlp=2.964912 > 2.00，因此 workspace gate fail。
D-RBF17 / D-WAV16 / D-CHE20 仍不是 exact fused no-materialize kernel，不能写成 exact repair success。
```

## 7. Focused Case B/C repair

原因：

```text
official 中 X5 source 很好但 NLL/ECE tail fail；
X6/X7 只有局部 seed/loss pass，LineC / tail 稳定性不足。
按计划 Case B/C，尝试更集中地使用 delayed consolidation、readout-basis decoupling、reservoir proxy 与 split/merge。
```

执行：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_x5x6x7_casebc && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_x5x6x7_casebc --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K9-RAT-CoverSplitMerge-lite,K9-RAT-CoverSplitMerge-noMerge,K10-RAT-ParamSNRThenCover-3phase,K10-RAT-ParamSNRThenCover-slowConsolidate,K11-RAT-ReadoutBasisDecoupledSNR,K11-RAT-ReadoutFirstBasisConsolidate,K12-RAT-VarianceReservoirProxyCoverGrowth,K12-RAT-LowVarianceSignalCoverGrowth --train-steps 300 --batch-size 32 --log-interval 75 --mlp-hidden 160 --loss-interface CE --no-download
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
compute_budgeted_run = 1
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7194734811782837
snr_transfer_median_cos_group_vs_param = 0.6489424705505371
cover_purity_median = 0.03219109773635864
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

focused repair 逐行 pass：

```text
pass_s3 rows = 5
X5 pass_s3 rows = 0
X6 pass_s3 rows = 1
X7 pass_s3 rows = 4
```

解释：

```text
X5 仍被 NLL/ECE tail harm 拒绝；
X6 仍只有单 row pass；
X7 增加 source 但 LineC / family coverage 仍不足；
focused repair 不能写成 official success。
```

## 8. 结束前检查

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v137_boundary_conditioned_poprisk_training.py experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
```

结果：

```text
py_compile pass
```

最终可复现入口：

```text
Plan:
docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md

Runner:
experiments/run_v139_signal_to_cover_functional_substrate_architecture.py

Official artifacts:
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/

Non-RAT repair artifacts:
results/v13_09_signal_to_cover_functional_substrate_architecture/nonrat_exact_v139/

Focused repair artifacts:
results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_x5x6x7_casebc/
```

## 9. 用户再次追问后的 K13 cover-debt guard 修复

用户再次要求未达成则继续。本次先复核计划第 14-16 节：

```text
1. 当前 official route = R2-SignalRetentionPassCoverFormationFail。
2. 计划 Case A 的 K8/K9/K10/K11 已在 official 覆盖。
3. Case B/C 允许：延后 cover consolidation、readout follows AdamW/SNR、basis follows cover-SNR、减少 low-purity cover update、加强 Phase 3 consolidation。
4. 计划第 16 节写明：如果 v13.9 仍不能让 Rational 在 synthetic >=5/7 上达到 S3，需要更强 substrate/base architecture reset，而不是继续 functional update 搜索。
```

但复核实现时发现：

```text
底层已有 cover_policy() / would_increase_cover_debt() train-batch cover-debt guard；
K8-K12 alias 没有显式接上 CoverPhaseSchedule。
```

因此做一个额外 focused repair，不改变 official gate，不把 repair row 写成 promotion。

代码修改：

```text
experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
  新增 K13 alias：
  K13-RAT-CoverDebtGuardedSplitMerge-lite
  K13-RAT-ParamSNRSlowCoverPhaseGuard
  K13-RAT-LowVariancePhase3CoverGuard
  K13-RAT-ReadoutFirstCoverGuard
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
```

结果：

```text
py_compile pass
```

执行：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_k13_coverguard_x5x6x7 && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_k13_coverguard_x5x6x7 --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K13-RAT-CoverDebtGuardedSplitMerge-lite,K13-RAT-ParamSNRSlowCoverPhaseGuard,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard --train-steps 300 --batch-size 32 --log-interval 75 --mlp-hidden 160 --loss-interface CE --no-download
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
compute_budgeted_run = 1
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7194734811782837
snr_transfer_median_cos_group_vs_param = 0.6489424705505371
cover_purity_median = 0.03219109773635864
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

逐行 summary：

```text
v139_rat_signal_to_cover_summary rows = 90
pass_s3 rows = 3
pass_s4 rows = 0
X5 pass_s3 rows = 1
X6 pass_s3 rows = 0
X7 pass_s3 rows = 2
```

最接近 rows：

| task | seed | loss | method | source_vs_adamw | AUC_time_ratio | NLL delta | ECE delta | LineC majority | cover purity | pass_s3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| X7 | 2 | Brier | K13-LowVariancePhase3CoverGuard | 0.283445 | 0.732842 | -3.577312 | -0.296999 | 0 | 0.084654 | 0 |
| X7 | 1 | Brier | K13-LowVariancePhase3CoverGuard | 0.101607 | 0.899954 | -3.032255 | -0.249470 | 1 | 0.016306 | 1 |
| X5 | 0 | Brier | K13-ParamSNRSlowCoverPhaseGuard | 0.023444 | 0.969900 | -0.139494 | -0.022553 | 1 | 0.066040 | 1 |
| X6 | 2 | Brier | K13-ParamSNRSlowCoverPhaseGuard | 0.083134 | 0.841373 | -0.518049 | -0.048963 | 0 | 0.015432 | 0 |

判断：

```text
K13 cover-debt guard 没有打开 X5/X6/X7 family gate；
S3/S4 仍为 0/3；
不能作为 official success 或 promotion。
```

## 10. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.09 是否达成目标，若未达成则继续。本次复核计划 stop rule、official route 与 K13 repair route。

执行：

```bash
rg -n "8\\.4|14\\.|16\\.|stop|R2|R3|R4|substrate|architecture|Case A|Case B|Case C|Case E|不允许|禁止|recommend|推荐" docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
sed -n '930,1085p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md
python - <<'PY'
import json
from pathlib import Path
for p in [
    Path('results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_route_decision.json'),
    Path('results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_k13_coverguard_x5x6x7/v139_route_decision.json'),
]:
    print('---', p)
    d = json.loads(p.read_text())
    for k in [
        'route', 'minimum_success', 'official_success_reached', 'promotion_allowed',
        'final_stop_allowed', 'required_artifact_missing_count', 'snr_transfer_gate_pass',
        'cover_formation_gate_pass', 'kan_s3_task_pass_count', 'kan_s4_task_pass_count',
        'nonrat_substrate_health_pass_count', 'compute_budgeted_run',
    ]:
        if k in d:
            print(k, d[k])
PY
sed -n '240,520p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_实验结果复盘.md
sed -n '240,560p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_执行日志.md
```

结果：

```text
official_v139:
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 4
kan_s4_task_pass_count = 0
nonrat_substrate_health_pass_count = 0

repair_v139_k13_coverguard_x5x6x7:
route = R2-SignalRetentionPassCoverFormationFail
official_success_reached = 0
promotion_allowed = 0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
compute_budgeted_run = 1
```

计划 stop rule 对照：

```text
1. 第 14 节 Case A/B/C/E 已覆盖：
   K8-K12 official、Case B/C focused repair、K13 cover-debt guard focused repair、
   Non-RAT exact substrate repair。
2. 第 16 节写明：如果 v13.9 仍不能让 Rational 在 synthetic >=5/7 上达到 S3，
   需要 stronger substrate/base architecture reset，而不是继续 functional update 搜索。
3. 当前 official 只有 S3 family 4/7，K13 focused repair 仍为 0/3，Non-RAT substrate-health = 0。
```

判断：

```text
v13.09 未达成目标；
合法 route 仍为 R2-SignalRetentionPassCoverFormationFail；
promotion_allowed = 0；
real_short_run 不允许打开；
当前 v13.09 内继续排列 K-token / functional update 小修已越过计划边界。
```

## 11. 用户再次追问后的 Line S cross-plan substrate/base scout

用户再次要求未达成则继续。本次重新搜索仓库后发现 v13.6 曾经实现过 `Line S：Substrate / base architecture reset` 的 audited scout mapping：

```text
RAT-SNR1/2/3
CHE-SNR1/2/3
FOU-SNR1/2/3
RBF-SNR1/2/3
WAV-SNR1/2/3
```

这些 Line S candidate 是设计级 substrate scout，并映射到已有 v12.35/v1235 audited primitive。为继续尝试 v13.09 第 16 节推荐的 substrate/base architecture reset，本次在 v13.09 result 目录下补跑 v13.6 Line S scout。该结果只能作为 substrate/base diagnostic，不能写成 v13.09 functional promotion。

读取相关实现：

```bash
rg -n "Line S|substrate/base architecture scout|RAT-SNR|CHE-SNR|FOU-SNR|RBF-SNR|WAV-SNR|SNR1|SNR2|SNR3" docs/DG-KAN_v13.06_PopRiskSNR_BasisCoverBoundary_完整计划.md docs/DG-KAN_v13.06_PopRiskSNR_BasisCoverBoundary_执行日志.md docs/DG-KAN_v13.06_PopRiskSNR_BasisCoverBoundary_实验结果复盘.md experiments/run_v136_poprisk_snr_basis_cover_boundary.py experiments/run_v137_boundary_conditioned_poprisk_training.py experiments/run_v138_generic_poprisk_and_kan_snr_to_basis_lift.py experiments/run_v139_signal_to_cover_functional_substrate_architecture.py -S
sed -n '1680,1785p' experiments/run_v136_poprisk_snr_basis_cover_boundary.py
sed -n '1500,1625p' experiments/run_v136_poprisk_snr_basis_cover_boundary.py
sed -n '100,205p' experiments/run_v136_poprisk_snr_basis_cover_boundary.py
sed -n '1230,1325p' experiments/run_v136_poprisk_snr_basis_cover_boundary.py
```

执行：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/line_s_crossplan_substrate_scout_v139 && conda run -n kan python experiments/run_v136_poprisk_snr_basis_cover_boundary.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/line_s_crossplan_substrate_scout_v139 --synthetic-tasks X1 --synthetic-seeds 0 --synthetic-train-size 64 --synthetic-val-size 32 --checkpoint-steps 4 --future-steps 4 --operator-batch-size 8 --methods KAN-BasisChannelSNR --snr-variants Hard --loss-interfaces CE --controls Functional,SameActiveFractionRandomMask
```

结果：

```text
route = R4-PopRiskSNRNoGo
required_artifact_missing_count = 0
synthetic_rows = 1
synthetic_task_success_count = 0
promotion_allowed = 0
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
full_parameter_update_rows = 1
control_rows = 2
```

解析结果时先误用了系统 python + pandas：

```bash
python - <<'PY'
import json
import pandas as pd
...
PY
```

结果：

```text
ModuleNotFoundError: No module named 'pandas'
```

改用标准库 csv 解析：

```bash
python - <<'PY'
import csv, json
from pathlib import Path
base=Path('results/v13_09_signal_to_cover_functional_substrate_architecture/line_s_crossplan_substrate_scout_v139')
route=json.loads((base/'v136_route_decision.json').read_text())
print('route', route['route'])
for k in ['required_artifact_missing_count','synthetic_rows','synthetic_task_success_count','promotion_allowed','substrate_snr_gate_pass_count','nonrat_substrate_snr_gate_pass_count','full_parameter_update_rows','control_rows']:
    print(k, route.get(k))
rows=list(csv.DictReader((base/'v136_substrate_snr_gate.csv').open()))
print('rows', len(rows), 'line_s', sum(int(float(r.get('line_s_candidate') or 0)) for r in rows), 'pass', sum(int(float(r.get('substrate_snr_gate_pass') or 0)) for r in rows))
by={}
for r in rows:
    fam=r.get('family','')
    by.setdefault(fam,[0,0])
    by[fam][0]+=1
    by[fam][1]+=int(float(r.get('substrate_snr_gate_pass') or 0))
print('family_summary')
for fam,(cnt,ps) in sorted(by.items()): print(fam,cnt,ps)
PY
```

Family summary：

| family | rows | substrate-SNR pass |
|---|---:|---:|
| D-RAT | 4 | 4 |
| D-CHE | 4 | 0 |
| D-FOU | 4 | 0 |
| D-RBF | 4 | 0 |
| D-WAV | 3 | 0 |

通过的 rows 全部来自 Rational：

| candidate | mapped_candidate_id | incremental ratio | step ratio | SNR entropy | pass |
|---|---|---:|---:|---:|---:|
| D-RAT26-TangentTrustRegionNoCE | D-RAT26-TangentTrustRegionNoCE | 1.603821 | 1.291973 | 0.893051 | 1 |
| RAT-SNR1-denSlopeTelemetrySubstrate | D-RAT27-DenSlopeGuardNoCE | 1.593854 | 1.232687 | 0.893685 | 1 |
| RAT-SNR2-groupDiversityFloorSubstrate | D-RAT28-GroupDiversityPreservingRational | 1.603821 | 1.372908 | 0.893051 | 1 |
| RAT-SNR3-readoutRationalDecoupledSubstrate | D-RAT35-ReadoutRationalDecoupleNoCE | 1.349668 | 1.218606 | 0.893752 | 1 |

Non-RAT 结果：

```text
CHE/FOU/RBF/WAV Line S rows channel_snr_entropy 非零；
但 workspace_pass = 0，status = WorkspaceOnly_NotFunctionalSubstrate；
nonrat_substrate_snr_gate_pass_count = 0。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v136_poprisk_snr_basis_cover_boundary.py experiments/run_v137_boundary_conditioned_poprisk_training.py experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
```

结果：

```text
py_compile pass
```

判断：

```text
Line S cross-plan scout 没有改变 v13.09 结论；
它只证明 Rational substrate-SNR scout 仍可见，Non-RAT 仍不能进入 functional proof；
synthetic_task_success_count = 0，不能写成 v13.09 S3/S4/S5。
```

## 12. 用户再次追问后的 Rational Line S substrate-candidate focused repair

用户再次要求未达成则继续。本次进一步检查 v13.09 runner 后发现它支持：

```text
--rational-candidate
```

official_v139 默认只使用 `choose_primary_rational()`。既然上一节 Line S cross-plan scout 显示 RAT-SNR1/2/3 mapped substrates 通过 substrate-SNR gate，本次尝试把这些 Rational substrate/base reset candidates 直接接入 v13.09 signal-to-cover training。该实验仍是 focused repair / substrate-candidate scout，不替代 official。

候选：

```text
D-RAT27-DenSlopeGuardNoCE
D-RAT28-GroupDiversityPreservingRational
D-RAT35-ReadoutRationalDecoupleNoCE
```

先尝试较大的 D-RAT35 focused matrix：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat35_x5x6x7 && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat35_x5x6x7 --rational-candidate D-RAT35-ReadoutRationalDecoupleNoCE --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 128 --real-val-size 64 --real-test-size 64 --real-epochs 1 --synthetic-tasks X5,X6,X7 --synthetic-seeds 0,1,2 --synthetic-train-size 96 --synthetic-val-size 48 --k-losses CE,Brier --k-methods RAT-AdamW,K9-RAT-CoverSplitMerge-noMerge,K10-RAT-ParamSNRThenCover-slowConsolidate,K13-RAT-ParamSNRSlowCoverPhaseGuard,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard --train-steps 200 --batch-size 32 --log-interval 50 --mlp-hidden 160 --loss-interface CE --no-download
```

该 run 运行约 11 分钟仍未结束。为避免整轮卡在一个过大的 focused run 上，本次终止该进程：

```bash
ps -eo pid,ppid,etime,cmd | rg "run_v139_signal_to_cover_functional_substrate_architecture.py|repair_v139_lines_rational_substrate_drat35"
kill 3430814 3430794
```

说明：

```text
该中止 run 未产生可用 route artifact，不写入实验成功或失败指标；
只记录为 oversized focused run terminated。
```

改为 micro focused probes：

```bash
rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat35_micro && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat35_micro --rational-candidate D-RAT35-ReadoutRationalDecoupleNoCE --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 64 --real-val-size 32 --real-test-size 32 --real-epochs 1 --synthetic-tasks X5,X6,X7 --synthetic-seeds 0,1 --synthetic-train-size 64 --synthetic-val-size 32 --k-losses CE,Brier --k-methods RAT-AdamW,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard --train-steps 80 --batch-size 32 --log-interval 40 --mlp-hidden 120 --loss-interface CE --no-download

rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat27_micro && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat27_micro --rational-candidate D-RAT27-DenSlopeGuardNoCE --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 64 --real-val-size 32 --real-test-size 32 --real-epochs 1 --synthetic-tasks X5,X6,X7 --synthetic-seeds 0,1 --synthetic-train-size 64 --synthetic-val-size 32 --k-losses CE,Brier --k-methods RAT-AdamW,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard --train-steps 80 --batch-size 32 --log-interval 40 --mlp-hidden 120 --loss-interface CE --no-download

rm -rf results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat28_micro && conda run -n kan python experiments/run_v139_signal_to_cover_functional_substrate_architecture.py --out-dir results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat28_micro --rational-candidate D-RAT28-GroupDiversityPreservingRational --datasets MNIST --mlp-seeds 0 --mlp-seed-threshold 1 --mlp-methods MLP-AdamW --real-train-size 64 --real-val-size 32 --real-test-size 32 --real-epochs 1 --synthetic-tasks X5,X6,X7 --synthetic-seeds 0,1 --synthetic-train-size 64 --synthetic-val-size 32 --k-losses CE,Brier --k-methods RAT-AdamW,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard --train-steps 80 --batch-size 32 --log-interval 40 --mlp-hidden 120 --loss-interface CE --no-download
```

结果汇总：

| rational candidate | route | transfer gate | retention | cosine | cover purity | S3 task pass | S3 rows | S4 rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT27-DenSlopeGuardNoCE | R1-GenericSNROptimizerNoGo | 0 | 0.592759 | 0.532397 | 0.031367 | 0 | 3 | 0 |
| D-RAT28-GroupDiversityPreservingRational | R1-GenericSNROptimizerNoGo | 0 | 0.592886 | 0.532733 | 0.031374 | 0 | 2 | 0 |
| D-RAT35-ReadoutRationalDecoupleNoCE | R1-GenericSNROptimizerNoGo | 0 | 0.592714 | 0.532039 | 0.031379 | 0 | 3 | 0 |

每个 candidate 的 task-level S3 row count：

```text
D-RAT27: X5=1, X6=1, X7=1
D-RAT28: X5=0, X6=1, X7=1
D-RAT35: X5=1, X6=1, X7=1
```

最接近 rows 的共同模式：

```text
K13-LowVariancePhase3CoverGuard 有较强 source_vs_adamw；
但 X7 CE row noise delta > 0，因此不 pass；
cover_purity 仍显著低于 0.20；
micro run 的 K0 transfer retention/cos 也低于 official gate。
```

判断：

```text
把 Line S Rational mapped candidates 作为 v13.09 substrate/base focused repair 没有打开 gate；
逐行 source 局部 positive 不能写成 family success；
仍不能 promotion。
```

## 13. 用户再次追问后的最终复核

用户再次要求未达成则继续。本次只做 stop-contract 与进程复核，不再新增训练实验。原因是上一节已经把 v13.09 runner 内可安全指定的 Rational Line S substrate candidates 跑完；继续需要新的 substrate/base architecture reset runner，而不是继续在当前 v13.09 functional runner 内扩 probe。

执行：

```bash
python - <<'PY'
import json
from pathlib import Path
paths=[
('official_v139', Path('results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_route_decision.json')),
('k13_coverguard', Path('results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_k13_coverguard_x5x6x7/v139_route_decision.json')),
('line_s_crossplan', Path('results/v13_09_signal_to_cover_functional_substrate_architecture/line_s_crossplan_substrate_scout_v139/v136_route_decision.json')),
('drat27_micro', Path('results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat27_micro/v139_route_decision.json')),
('drat28_micro', Path('results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat28_micro/v139_route_decision.json')),
('drat35_micro', Path('results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_lines_rational_substrate_drat35_micro/v139_route_decision.json')),
]
for name,p in paths:
    d=json.loads(p.read_text())
    print('---', name)
    for k in ['route','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','required_artifact_missing_count','snr_transfer_gate_pass','cover_formation_gate_pass','kan_s3_task_pass_count','kan_s4_task_pass_count','synthetic_task_success_count','nonrat_substrate_health_pass_count','nonrat_substrate_snr_gate_pass_count','compute_budgeted_run']:
        if k in d:
            print(k, d[k])
PY

sed -n '1045,1070p' docs/DG-KAN_v13.09_SignalToCoverFunctional_SubstrateArchitecture_完整计划.md

ps -eo pid,ppid,etime,cmd | rg "run_v139_signal_to_cover_functional_substrate_architecture.py|run_v136_poprisk_snr_basis_cover_boundary.py" || true
```

结果：

```text
official_v139 route = R2-SignalRetentionPassCoverFormationFail
official_v139 kan_s3_task_pass_count = 4
official_v139 kan_s4_task_pass_count = 0
official_v139 promotion_allowed = 0

k13_coverguard route = R2-SignalRetentionPassCoverFormationFail
k13_coverguard kan_s3_task_pass_count = 0
k13_coverguard promotion_allowed = 0

line_s_crossplan route = R4-PopRiskSNRNoGo
line_s_crossplan synthetic_task_success_count = 0
line_s_crossplan nonrat_substrate_snr_gate_pass_count = 0
line_s_crossplan promotion_allowed = 0

drat27_micro route = R1-GenericSNROptimizerNoGo
drat28_micro route = R1-GenericSNROptimizerNoGo
drat35_micro route = R1-GenericSNROptimizerNoGo
drat27/28/35 promotion_allowed = 0

no v13.09/v13.6 experiment process running
```

计划第 16 节对照：

```text
如果 v13.9 仍不能让 Rational 在 synthetic >=5/7 上达到 S3，
需要更强的 substrate/base architecture reset，而不是继续 functional update 搜索。
```

最终判断：

```text
v13.09 没有达成目标；
不允许 promotion；
不允许 real short-run；
当前我已经不确定如何在当前 v13.09 runner 内继续安全推进，
而不把 compute-budgeted probe / substrate scout / mapped primitive diagnostic 编造成 official success。
```
