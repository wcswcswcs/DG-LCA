# DG-KAN v14.10 NonRAT FMS Transfer + FMSDefinitionReset 执行日志

生成时间：2026-05-30（Asia/Singapore）

本日志只记录本轮实际执行的命令、文件、artifact 与 blocker 修复动作；不把 smoke、repair、局部 positive row 写成 promotion。

## 1. 环境与计划读取

工作目录：

```text
/home/chengshun.wang/DG-LCA
```

计划文件：

```text
docs/DG-KAN_v14.10_NonRAT_FMS_Transfer_FMSDefinitionReset_完整计划.md
```

读取命令：

```bash
sed -n '1,240p' docs/DG-KAN_v14.10_NonRAT_FMS_Transfer_FMSDefinitionReset_完整计划.md
sed -n '241,520p' docs/DG-KAN_v14.10_NonRAT_FMS_Transfer_FMSDefinitionReset_完整计划.md
sed -n '520,920p' docs/DG-KAN_v14.10_NonRAT_FMS_Transfer_FMSDefinitionReset_完整计划.md
```

环境检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
PY
```

默认 `python` 报错：

```text
ModuleNotFoundError: No module named 'torch'
```

因此切换到已有 conda 环境：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
```

结果：

```text
torch 2.11.0+cu128
cuda_available True
device_count 4
NVIDIA RTX PRO 6000 Blackwell Server Edition
```

## 2. 代码修改

新增 runner：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

主要实现：

```text
1. v14.10 artifact surface：
   v1410_route_decision.json
   v1410_progress_table.csv
   v1410_forbidden_information_audit.csv
   v1410_no_action_search_audit.csv
   v1410_dche_fms_synthetic_results.csv
   v1410_dche_fms_synthetic_summary.csv
   v1410_dche_fms_real_results.csv
   v1410_dche_fms_real_summary.csv
   v1410_dche_fms_controls.csv
   v1410_dche_degree_telemetry.csv
   v1410_dche_projection_retention.csv
   v1410_mlp_generic_controls.csv
   v1410_all_basis_substrate_status.csv
   v1410_fourier_substrate_hardening.csv
   v1410_rbf_substrate_hardening.csv
   v1410_wavelet_substrate_hardening.csv
   v1410_rational_monitor.csv
   v1410_linec_tail_audit.csv
   v1410_failure_taxonomy.csv
   v1410_required_artifact_manifest.csv
   v1410_code_review_packet.zip
   v1410_no_go_boundary.md
   v1410_next_hypothesis_queue.md
```

```text
2. D-CHE official synthetic proof method set：
   C0-D-CHE-AdamW
   C1-D-CHE-AdamW-NoOpMatchedOverhead
   C2-D-CHE-AdamW-RandomMatchedNorm
   C3-D-CHE-AdamW-AdamWParallelDirectionControl
   C4-D-CHE-AdamW-GenericOptimizerStateControl
   F-CHE1-GenericParameterFMS-NoBasisProjection
   F-CHE2-DegreeWiseFMS
   F-CHE3-DegreeEnergyTrustRegionFMS
   F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS
   F-CHE5-ReadoutDegreeDecoupledFMS
   F-CHE6-PhaseScheduleDegreeFMS
   F-CHE7-ValuePreservingDegreeProjectionFMS
```

```text
3. Case A fallback method set：
   F-CHE-FB1-ValuePathOnly
   F-CHE-FB2-DegreeConstraintOnly
   F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
```

```text
4. MLP / generic controls：
   M0-MLP-AdamW
   M1-MLP-GenericFMS
   M2-MLP-DegreeAnalogFMS
   M3-MLP-RandomMatchedNorm
   M4-MLP-GenericOptimizerStateControl
```

合法性说明：

```text
1. FMS direction 只使用当前 train batch supervised loss / per-example gradient / train-stream FMS state。
2. LineC / CEp99 / NLL / ECE / AUCtime / Brier 只写入 audit/gate，不作为 direction。
3. 未新增 action token。
4. 未启动 controller。
5. 未执行 reset route。
6. 未按 dataset/task/seed branch 选择方法或 scale。
7. promotion_allowed 只在 S5 且非 compute_budgeted 且 artifact 完整时才可能为 1。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

结果：

```text
pass
```

## 3. Smoke

命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/smoke_v1410 \
  --device cuda:0 \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --loss-interfaces CE \
  --methods C0-D-CHE-AdamW,C2-D-CHE-AdamW-RandomMatchedNorm,F-CHE2-DegreeWiseFMS \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS \
  --train-steps 4 \
  --batch-size 8 \
  --synthetic-train-size 32 \
  --synthetic-val-size 16 \
  --trace-interval 2 \
  --fms-update-interval 2 \
  --skip-real 1 \
  --compute-budgeted-run 1
```

结果：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
synthetic_task_family_pass_count = 0
dche_substrate_official_fms_eligibility = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

说明：

```text
smoke 只证明 runner / artifact surface 可执行，不作为科学结论。
```

## 4. Official compute-budgeted synthetic

命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/official_v1410 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1,2 \
  --loss-interfaces CE,Brier \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-steps 200 \
  --batch-size 32 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --fms-update-interval 80 \
  --trace-interval 100 \
  --compute-budgeted-run 1
```

结果：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
synthetic_task_family_pass_count = 0
synthetic_task_pass = {}
dche_substrate_official_fms_eligibility = 1
dche_best_source_vs_adamw = 0.1652148962020874
mlp_best_source_vs_adamw = 0.2887003421783447
generic_fms_confound = 1
real_dataset_seed_pass_count = 0
official_s5_reached = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

因为 S3 synthetic 未过，real 3x3 没有打开。

产物目录：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/official_v1410/
```

## 5. Case A blocker 修复：amortized refresh repair

official failure decomposition 显示：

```text
source fail rows = 377
AUCtime fail rows = 368
step fail rows = 420
CEp99 fail rows = 175
NLL fail rows = 103
ECE fail rows = 92
LineC fail rows = 0
```

按计划 Case A，不新增 F-CHE8、不用 audit metric 做方向，只用预注册方法/FB 方法做修复。针对 step-time blocker 做全局 amortized refresh repair：

```text
fms_update_interval: 80 -> 200
methods: C0..C4 + F-CHE3 + F-CHE7
fallback: FB2 + FB3
real: skip，因为 synthetic 仍需先过 S3
```

命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_amortized_interval200_v1410 \
  --device cuda:0 \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1,2 \
  --loss-interfaces CE,Brier \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE3-DegreeEnergyTrustRegionFMS,F-CHE7-ValuePreservingDegreeProjectionFMS \
  --fallback-methods F-CHE-FB2-DegreeConstraintOnly,F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --train-steps 200 \
  --batch-size 32 \
  --synthetic-train-size 96 \
  --synthetic-val-size 64 \
  --fms-update-interval 200 \
  --trace-interval 100 \
  --skip-real 1 \
  --compute-budgeted-run 1
```

结果：

```text
route = R1-DCHESyntheticFMSFail
minimum_success = S0-ExecutionCompleteNoPromotion
synthetic_task_family_pass_count = 1
synthetic_task_pass = {X1: 1, X2: 0, X3: 0, X4: 0, X5: 0}
dche_substrate_official_fms_eligibility = 1
dche_best_source_vs_adamw = 0.13746285438537598
mlp_best_source_vs_adamw = 0.2882530689239502
generic_fms_confound = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

repair failure decomposition：

```text
source fail rows = 142
AUCtime fail rows = 141
CEp99 fail rows = 72
NLL fail rows = 35
ECE fail rows = 31
step fail rows = 0
LineC fail rows = 0
```

判断：

```text
amortized refresh 修掉了 step blocker，并把 synthetic pass 从 0/7 提到 1/7；
但 source / AUC / tail 仍不足，距离 S3 的 >=5/7 很远。
```

产物目录：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_amortized_interval200_v1410/
```

## 6. Required manifest

official：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/official_v1410/v1410_required_artifact_manifest.csv
missing_required_rows = 0
lines including header = 36
```

repair：

```text
results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_amortized_interval200_v1410/v1410_required_artifact_manifest.csv
missing_required_rows = 0
lines including header = 36
```

## 7. 最终执行状态

本轮执行到计划允许的停止点：

```text
1. D-CHE substrate eligibility confirmed from v14.9 artifact: 9/9, official_fms_eligibility = 1。
2. v14.10 D-CHE synthetic official proof executed。
3. S3 synthetic failed，route = R1-DCHESyntheticFMSFail。
4. Case A allowed fallbacks FB1/FB2/FB3 已执行。
5. 额外全局 amortized refresh repair 修复 step-time blocker，但仍只有 1/7。
6. real 3x3 未打开。
7. S5 未达成。
8. promotion_allowed = 0。
```

## 8. 继续推进：candidate / memory / S3 / real transfer

### 8.1 后续 repair 命令摘要

以下 run 均使用同一个 runner：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

strength / interval sensitivity：

```text
repair_interval200_strength002_v1410:
  same as repair_amortized_interval200_v1410, but --fms-strength 0.02。
repair_interval200_strength010_v1410:
  same as repair_amortized_interval200_v1410, but --fms-strength 0.10。
repair_interval120_v1410:
  same as repair_amortized_interval200_v1410, but --fms-update-interval 120。
```

结果：

```text
strength 0.02 / 0.10: route = R1，synthetic_task_family_pass_count = 1，pass X1 only。
interval120: route = R1，synthetic_task_family_pass_count = 0。
```

### 8.2 D-CHE17 candidate alignment

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  DEFAULT_D_CHE_CANDIDATE = D-CHE17-HighDegreeLateEnableSubstrate
  新增 --dche-candidate
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

结果：

```text
py_compile pass
```

D-CHE17 seed0 batch32 command：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed0_v1410 \
  --device cuda:0 --dche-candidate D-CHE17-HighDegreeLateEnableSubstrate \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0 --loss-interfaces CE,Brier \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE3-DegreeEnergyTrustRegionFMS,F-CHE7-ValuePreservingDegreeProjectionFMS \
  --fallback-methods F-CHE-FB2-DegreeConstraintOnly,F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --train-steps 200 --batch-size 32 --synthetic-train-size 96 --synthetic-val-size 64 \
  --fms-update-interval 200 --trace-interval 100 --skip-real 1 --compute-budgeted-run 1
```

结果：

```text
route = R1
synthetic_task_family_pass_count = 0
dche_best_source_vs_adamw = 0.3820805549621582
mlp_best_source_vs_adamw = 0.27991271018981934
generic_fms_confound = 0
blocker = memory ratio about 1.773 > 1.25
```

### 8.3 batch sensitivity

执行目录：

```text
repair_dche17_seed0_b8_v1410: same as D-CHE17 seed0, but --batch-size 8。
repair_dche17_seed012_b8_v1410: seed0,1,2 and --batch-size 8。
repair_dche17_seed012_b8_interval80_v1410: seed0,1,2, --batch-size 8, --fms-update-interval 80。
repair_dche17_seed0_b16_v1410: seed0, --batch-size 16。
```

结果：

```text
batch8 seed0: memory fixed but source weaker, synthetic pass 0/7。
batch8 seed012: synthetic_task_family_pass_count = 2，pass X5/X6。
batch8 interval80: synthetic_task_family_pass_count = 0。
batch16 seed0: source stronger but memory ratio about 1.393 > 1.25，synthetic pass 0/7。
```

### 8.4 streaming per-example gradient memory repair

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 collect_streaming_per_example_summary()
  新增 --streaming-per-example-gradients
  refresh 时流式累计 per-example gradient mean / variance / utilities，
  不再 materialize batch x params gradient matrix。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

结果：

```text
py_compile pass
```

关键 streaming runs：

```text
repair_dche17_seed0_streaming_b32_v1410:
  D-CHE17, seed0, batch32, selected methods, --streaming-per-example-gradients 1。
  route = R1, synthetic pass 0/7, median memory ratio = 1.0127660562170693。

repair_dche17_seed012_streaming_b32_v1410:
  D-CHE17, seed0/1/2, batch32, selected methods, --streaming-per-example-gradients 1。
  route = R1, synthetic_task_family_pass_count = 3，pass X1/X3/X7。

repair_dche17_seed012_streaming_allfche_v1410:
  D-CHE17, seed0/1/2, batch32, all F-CHE1..7 + FB1..3, --streaming-per-example-gradients 1。
  route = R1, synthetic_task_family_pass_count = 4，pass X1/X3/X5/X7。
```

### 8.5 300-step synthetic S3 repair

执行命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410 \
  --device cuda:0 --dche-candidate D-CHE17-HighDegreeLateEnableSubstrate \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 --synthetic-seeds 0,1,2 --loss-interfaces CE,Brier \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE1-GenericParameterFMS-NoBasisProjection,F-CHE2-DegreeWiseFMS,F-CHE3-DegreeEnergyTrustRegionFMS,F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS,F-CHE5-ReadoutDegreeDecoupledFMS,F-CHE6-PhaseScheduleDegreeFMS,F-CHE7-ValuePreservingDegreeProjectionFMS \
  --fallback-methods F-CHE-FB1-ValuePathOnly,F-CHE-FB2-DegreeConstraintOnly,F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M2-MLP-DegreeAnalogFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --train-steps 300 --batch-size 32 --synthetic-train-size 96 --synthetic-val-size 64 \
  --fms-update-interval 200 --streaming-per-example-gradients 1 --trace-interval 100 --skip-real 1 --compute-budgeted-run 1
```

结果：

```text
route = S3-DCHESyntheticFMSPass
minimum_success = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5
synthetic_task_pass = X1,X3,X4,X5,X7
dche_best_source_vs_adamw = 0.4909071922302246
mlp_best_source_vs_adamw = 0.3354611396789551
generic_fms_confound = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

## 9. Real 3x3 execution after S3

### 9.1 synthetic-source real-open support

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 --synthetic-source-dir。
  real run 可以读取已通过的 S3 synthetic artifact，而不是重复训练 synthetic。
  synthetic rows 写入 synthetic_reused_from_s3_artifact 与 synthetic_source_artifact_dir。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

结果：

```text
py_compile pass
```

### 9.2 real 200-step interval80

执行命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_v1410 \
  --device cuda:0 --dche-candidate D-CHE17-HighDegreeLateEnableSubstrate \
  --synthetic-source-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE1-GenericParameterFMS-NoBasisProjection,F-CHE2-DegreeWiseFMS,F-CHE3-DegreeEnergyTrustRegionFMS,F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS,F-CHE5-ReadoutDegreeDecoupledFMS,F-CHE6-PhaseScheduleDegreeFMS,F-CHE7-ValuePreservingDegreeProjectionFMS,F-CHE-FB1-ValuePathOnly,F-CHE-FB2-DegreeConstraintOnly,F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M2-MLP-DegreeAnalogFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --run-fallbacks 0 --train-steps 200 --batch-size 32 --train-size 1024 --val-size 512 --test-size 512 \
  --fms-update-interval 80 --streaming-per-example-gradients 1 --trace-interval 100 --compute-budgeted-run 1
```

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 0
```

### 9.3 real 200-step interval200 protocol alignment

执行命令同 9.2，只改变：

```bash
--out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410
--fms-update-interval 200
```

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 2
strict unique dataset-seeds = KMNIST seed0, KMNIST seed2
step_fail = 0
memory_fail = 0
promotion_allowed = 0
```

### 9.4 real 400-step confirmation

执行命令同 9.3，只改变：

```bash
--out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps400_interval200_v1410
--train-steps 400
```

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 1
promotion_allowed = 0
```

## 10. Case B RT fallback

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 REAL_TRANSFER_METHODS:
    F-CHE-RT1-TrainSplitAgreement
    F-CHE-RT2-ValueRetentionTrust
    F-CHE-RT3-DegreeEnergySafetyProjection
  RT1: 同一 train batch 切成两半，用 train-loss gradient split agreement 缩放 FMS scale。
  RT2: 用 train gradient generic/projected cos 与 retention 做 value-retention blend。
  RT3: 对 basis / degree tensor 做更窄的 train-stream degree-energy safety projection。
  no-action audit 将 RT1/RT2/RT3 记录为 pre-registered method set。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt123_steps200_interval200_v1410 \
  --device cuda:0 --dche-candidate D-CHE17-HighDegreeLateEnableSubstrate \
  --synthetic-source-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE-RT1-TrainSplitAgreement,F-CHE-RT2-ValueRetentionTrust,F-CHE-RT3-DegreeEnergySafetyProjection \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M2-MLP-DegreeAnalogFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --run-fallbacks 0 --train-steps 200 --batch-size 32 --train-size 1024 --val-size 512 --test-size 512 \
  --fms-update-interval 200 --streaming-per-example-gradients 1 --trace-interval 100 --compute-budgeted-run 1
```

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

## 12. 用户再次追问后的状态复核与停止边界

本次没有启动新的训练；只复核最新 v14.10 artifacts，确认前一轮结论是否仍成立。

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
paths = [
Path('results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410/v1410_route_decision.json'),
Path('results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410/v1410_route_decision.json'),
Path('results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt123_steps200_interval200_v1410/v1410_route_decision.json'),
Path('results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_composite_steps200_interval200_v1410/v1410_route_decision.json'),
Path('results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_strength010_steps200_interval200_v1410/v1410_route_decision.json'),
]
keys = ['route','minimum_success','synthetic_task_family_pass_count','real_dataset_seed_pass_count','official_s5_reached','promotion_allowed','required_artifact_missing_count','forbidden_information_violation_count','generic_fms_confound']
for p in paths:
    print('---', p)
    d = json.loads(p.read_text())
    for k in keys:
        if k in d:
            print(f'{k}={d[k]}')
PY
```

复核结果：

```text
best synthetic:
  route = S3-DCHESyntheticFMSPass
  synthetic_task_family_pass_count = 5
  official_s5_reached = 0
  promotion_allowed = 0

best real:
  route = R2-DCHESyntheticDoesNotTransfer
  minimum_success = S3-DCHESyntheticFMSPass
  real_dataset_seed_pass_count = 2
  official_s5_reached = 0
  promotion_allowed = 0
  required_artifact_missing_count = 0
  forbidden_information_violation_count = 0
  generic_fms_confound = 0

RT1/RT2/RT3:
  real_dataset_seed_pass_count = 0

RT4 strength 0.05:
  real_dataset_seed_pass_count = 0

RT4 strength 0.10:
  real_dataset_seed_pass_count = 0
```

停止边界：

```text
本次没有新增训练指令。
原因：计划内 Case B RT1/RT2/RT3、post-plan RT4、RT4 strength sensitivity 均未打开 real transfer。
继续推进很可能需要按 real dataset/seed fail pattern 调方向、
或用 CEp99/ECE/LineC/AUCtime audit metric 设计方向、
或新增 action/controller/reset route。
这些都违反 v14.10 当前计划边界。
```

## 11. Post-plan composite transfer trust diagnostic

说明：

```text
计划内 Case B 的 RT1/RT2/RT3 已执行且仍 <6/9。
用户要求继续思考修复，因此新增 post-plan diagnostic RT4。
RT4 不作为 promotion 依据；仍保持 all gates。
```

代码修改：

```text
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  新增 F-CHE-RT4-CompositeTransferTrust。
  RT4 组合 RT1 train-batch split-gradient agreement、
      RT2 value-retention trust、
      RT3 degree-energy safety projection。
  direction source 仍为 current train-stream supervised loss gradient。
```

语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

结果：

```text
py_compile pass
```

### 11.1 RT4 strength 0.05

执行命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_composite_steps200_interval200_v1410 \
  --device cuda:0 --dche-candidate D-CHE17-HighDegreeLateEnableSubstrate \
  --synthetic-source-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE-RT4-CompositeTransferTrust \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M2-MLP-DegreeAnalogFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --run-fallbacks 0 --train-steps 200 --batch-size 32 --train-size 1024 --val-size 512 --test-size 512 \
  --fms-update-interval 200 --streaming-per-example-gradients 1 --trace-interval 100 --compute-budgeted-run 1
```

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

### 11.2 RT4 strength 0.10

执行命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py \
  --out-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_rt4_strength010_steps200_interval200_v1410 \
  --device cuda:0 --dche-candidate D-CHE17-HighDegreeLateEnableSubstrate \
  --synthetic-source-dir results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --methods C0-D-CHE-AdamW,C1-D-CHE-AdamW-NoOpMatchedOverhead,C2-D-CHE-AdamW-RandomMatchedNorm,C3-D-CHE-AdamW-AdamWParallelDirectionControl,C4-D-CHE-AdamW-GenericOptimizerStateControl,F-CHE-RT4-CompositeTransferTrust \
  --mlp-methods M0-MLP-AdamW,M1-MLP-GenericFMS,M2-MLP-DegreeAnalogFMS,M3-MLP-RandomMatchedNorm,M4-MLP-GenericOptimizerStateControl \
  --run-fallbacks 0 --train-steps 200 --batch-size 32 --train-size 1024 --val-size 512 --test-size 512 \
  --fms-update-interval 200 --fms-strength 0.10 --streaming-per-example-gradients 1 --trace-interval 100 --compute-budgeted-run 1
```

结果：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```
