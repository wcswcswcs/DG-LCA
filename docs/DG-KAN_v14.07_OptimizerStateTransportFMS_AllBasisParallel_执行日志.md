# DG-KAN v14.7 OptimizerStateTransportFMS AllBasisParallel 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志记录实际执行过的命令、代码文件和 artifact 位置，便于复现；不补填未执行命令。

## 1. 计划与上下文读取

计划文件：

```bash
sed -n '980,1060p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
sed -n '500,545p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
sed -n '660,690p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
sed -n '900,940p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
```

关键前置输入：

```text
results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/
diagnostic_v1461_after_p1_transport/
```

## 2. 代码修改

修改文件：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py
```

主要修改：

```text
1. 新增 v14.7 runner：
   experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py

2. 在 v144 runner 增加 optimizer-state transport scope：
   affected
   all_fms_roles
   random_matched
   random_affected_fraction
   full_adamw

3. 增加 v14.7 strict gate / endpoint-vs-AdamW gate / specificity gate / efficiency gate。

4. 增加 route 复算能力：
   --reuse-existing-results 1

5. 修复 v14.7 发现的实现 blocker：
   optimizer-state transport 从 every-step reset 改为只在 FMS refresh/event step 执行。
```

## 3. 环境 blocker

首个默认 Python smoke：

```bash
python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147 \
  --datasets MNIST --seeds 0 \
  --methods K0-RAT-AdamW,K2-RAT-FMS-ZeroMomentReset-AffectedOnly,KCTRL-RandomMomentResetMatchedFraction \
  --mlp-methods M0-MLP-AdamW,M2-MLP-FMS-ZeroMomentResetAffected \
  --train-steps 4 --batch-size 8 --linec-mode none \
  --compute-budgeted-run 1 --no-download
```

结果：

```text
ModuleNotFoundError: No module named 'torch'
```

修复：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py
```

后续均使用：

```text
conda run -n kan python
```

## 4. Smoke blockers 与修复

### 4.1 load_real_split 调用签名

命令：

```bash
conda run -n kan python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147 \
  --datasets MNIST --seeds 0 \
  --methods K0-RAT-AdamW,K2-RAT-FMS-ZeroMomentReset-AffectedOnly,KCTRL-RandomMomentResetMatchedFraction \
  --mlp-methods M0-MLP-AdamW,M2-MLP-FMS-ZeroMomentResetAffected \
  --train-steps 4 --batch-size 8 --linec-mode none \
  --compute-budgeted-run 1 --no-download
```

blocker：

```text
TypeError: load_real_split() takes 4 positional arguments but 8 were given
```

修复：

```text
改为 v144.load_real_split(args, dataset, seed, device)，并把 input_dim/output_dim tensor 转成 int。
```

### 4.2 write_svg 调用签名

blocker：

```text
TypeError: write_svg() missing 1 required positional argument: 'lines'
```

修复：

```text
改为 write_svg(path, title, lines)。
```

### 4.3 smoke 通过

命令同 4.1，修复后结果：

```text
out_dir = results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147
route = R1-ZeroMomentResetNotReplicated
expected_dataset_seed_count = 1
k2_dataset_seed_pass_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 5. Official v14.7 首跑

命令：

```bash
conda run -n kan python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/official_v147 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K1-RAT-FMS-NoStateTransport,K2-RAT-FMS-ZeroMomentReset-AffectedOnly,K3-RAT-FMS-ZeroMomentReset-AllFMSRoles,K4-RAT-FMS-PartialMomentInterpolation,K5-RAT-FMS-RMSRecomputeMicrobatch,K6-RAT-FMS-MomentTransportProjectedGrad,KCTRL-RandomMomentResetMatchedFraction,KCTRL-ZeroMomentResetRandomCoords,KCTRL-FullAdamWStateReset,KCTRL-NoOpMatchedOverhead \
  --mlp-methods M0-MLP-AdamW,M1-MLP-FMS-NoStateTransport,M2-MLP-FMS-ZeroMomentResetAffected,M3-MLP-FMS-ZeroMomentResetRandomCoords,M4-MLP-FMS-FullAdamWStateReset \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --no-download
```

首跑发现：

```text
K2 endpoint-vs-AdamW = 9/9，但 strict = 0/9；
random/full reset controls endpoint-vs-AdamW 也达到 9/9；
实现层 transport 是 every-step reset，不符合“FMS event 后 transport”的计划语义。
```

route 复算命令：

```bash
conda run -n kan python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/official_v147 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --compute-budgeted-run 1 \
  --reuse-existing-results 1
```

复算后 route：

```text
route = R2-ResetWorksButNotFMSSpecific
k2_dataset_seed_pass_count = 0 / 9
k2_endpoint_vs_adamw_pass_count = 9 / 9
k2_specificity_pass_count = 0 / 9
k2_efficiency_pass_count = 1 / 9
control_endpoint_full_success_count = 3
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 6. Repair：event-only optimizer-state transport

触发原因：

```text
计划语义是 FMS event 后 transport；
首跑实现为 every-step reset，会把 K2 退化成 generic reset。
```

代码修复：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  optimizer-state transport 只在 refresh/FMS event step 执行。
```

重新执行：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py

conda run -n kan python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_transport \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K1-RAT-FMS-NoStateTransport,K2-RAT-FMS-ZeroMomentReset-AffectedOnly,K3-RAT-FMS-ZeroMomentReset-AllFMSRoles,K4-RAT-FMS-PartialMomentInterpolation,K5-RAT-FMS-RMSRecomputeMicrobatch,K6-RAT-FMS-MomentTransportProjectedGrad,KCTRL-RandomMomentResetMatchedFraction,KCTRL-ZeroMomentResetRandomCoords,KCTRL-FullAdamWStateReset,KCTRL-NoOpMatchedOverhead \
  --mlp-methods M0-MLP-AdamW,M1-MLP-FMS-NoStateTransport,M2-MLP-FMS-ZeroMomentResetAffected,M3-MLP-FMS-ZeroMomentResetRandomCoords,M4-MLP-FMS-FullAdamWStateReset \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --no-download
```

结果：

```text
route = R1-ZeroMomentResetNotReplicated
k2_dataset_seed_pass_count = 0 / 9
k2_endpoint_vs_adamw_pass_count = 4 / 9
k2_specificity_pass_count = 0 / 9
k2_efficiency_pass_count = 2 / 9
control_endpoint_full_success_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 7. 修复后 smoke

命令：

```bash
conda run -n kan python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147_event_only \
  --datasets MNIST --seeds 0 \
  --methods K0-RAT-AdamW,K2-RAT-FMS-ZeroMomentReset-AffectedOnly,KCTRL-RandomMomentResetMatchedFraction \
  --mlp-methods M0-MLP-AdamW,M2-MLP-FMS-ZeroMomentResetAffected \
  --train-steps 4 --batch-size 8 --linec-mode none \
  --compute-budgeted-run 1 --no-download
```

结果：

```text
route = R1-ZeroMomentResetNotReplicated
k2_dataset_seed_pass_count = 0 / 1
required_artifact_missing_count = 0
```

## 8. 输出目录

```text
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/official_v147/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_transport/
results/v14_7_optimizer_state_transport_fms_all_basis_parallel/smoke_v147_event_only/
```

## 9. 用户再次要求继续后的 affected-coordinate sparse mask 修复

触发原因：

```text
repair_v147_event_only_transport 显示 K2 affected-only reset fraction median 约 0.995，
说明当前 affected-coordinate identification 过宽，几乎等价 all-FMS-role reset。
计划 6.4 要求 affected coordinate 来自 train-stream FMS active mask / projected direction，
不能来自 audit metric / dataset / seed。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 optimizer-state transport scope:
    delta_top25
    random_delta_top25

experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py
  新增 --affected-coordinate-policy:
    delta_nonzero
    delta_top25
  当 policy = delta_top25:
    K2/M2 使用 projected-gradient delta top-25% affected mask；
    random controls 使用 random_delta_top25 matched fraction。
```

合法性：

```text
1. 不新增 action token。
2. 不使用 validation/test/future/query。
3. 不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier 生成 coordinate。
4. 不做 dataset/seed branch。
5. 不调 strength/lambda/lr/refresh 网格。
```

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py

conda run -n kan python experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py \
  --out-dir results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_delta_top25 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K1-RAT-FMS-NoStateTransport,K2-RAT-FMS-ZeroMomentReset-AffectedOnly,K3-RAT-FMS-ZeroMomentReset-AllFMSRoles,K4-RAT-FMS-PartialMomentInterpolation,K5-RAT-FMS-RMSRecomputeMicrobatch,K6-RAT-FMS-MomentTransportProjectedGrad,KCTRL-RandomMomentResetMatchedFraction,KCTRL-ZeroMomentResetRandomCoords,KCTRL-FullAdamWStateReset,KCTRL-NoOpMatchedOverhead \
  --mlp-methods M0-MLP-AdamW,M1-MLP-FMS-NoStateTransport,M2-MLP-FMS-ZeroMomentResetAffected,M3-MLP-FMS-ZeroMomentResetRandomCoords,M4-MLP-FMS-FullAdamWStateReset \
  --affected-coordinate-policy delta_top25 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --no-download
```

结果：

```text
out_dir = results/v14_7_optimizer_state_transport_fms_all_basis_parallel/repair_v147_event_only_delta_top25
route = R1-ZeroMomentResetNotReplicated
k2_dataset_seed_pass_count = 0 / 9
k2_endpoint_vs_adamw_pass_count = 2 / 9
k2_specificity_pass_count = 0 / 9
k2_efficiency_pass_count = 1 / 9
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

mask audit：

```text
K2 transport_reset_param_fraction:
  min = 0.18247486650943756
  median = 0.2487437129020691
  max = 0.25
```

判断：

```text
delta_top25 确实把 reset fraction 降到约 25%，
但 K2 endpoint-vs-AdamW 从 event-only 的 4/9 退到 2/9；
因此稀疏 affected mask 没有打开 v14.7 S5。
```

## 10. 最终核对

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v147_optimizer_state_transport_fms_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

route JSON 核对：

```text
official_v147:
  route = R2-ResetWorksButNotFMSSpecific
  k2_dataset_seed_pass_count = 0 / 9
  k2_endpoint_vs_adamw_pass_count = 9 / 9
  k2_specificity_pass_count = 0 / 9
  k2_efficiency_pass_count = 1 / 9
  control_endpoint_full_success_count = 3
  official_s5_reached = 0
  promotion_allowed = 0

repair_v147_event_only_transport:
  route = R1-ZeroMomentResetNotReplicated
  k2_dataset_seed_pass_count = 0 / 9
  k2_endpoint_vs_adamw_pass_count = 4 / 9
  k2_specificity_pass_count = 0 / 9
  k2_efficiency_pass_count = 2 / 9
  control_endpoint_full_success_count = 0
  official_s5_reached = 0
  promotion_allowed = 0

repair_v147_event_only_delta_top25:
  route = R1-ZeroMomentResetNotReplicated
  k2_dataset_seed_pass_count = 0 / 9
  k2_endpoint_vs_adamw_pass_count = 2 / 9
  k2_specificity_pass_count = 0 / 9
  k2_efficiency_pass_count = 1 / 9
  control_endpoint_full_success_count = 0
  official_s5_reached = 0
  promotion_allowed = 0
```

停止判断：

```text
按 v14.7 stop-go：
K2 < 9/9 时停止 zero_moment_reset route，不扩 action token。
当前 K2 strict 始终为 0/9，event-only 与 sparse repair 均未打开 S5。
继续调 strength/lambda/lr/refresh 会变成计划禁止的 grid search；
继续新增 action/controller 或用 audit metric 反推 direction 也被计划禁止。
因此本轮停止在 no-go，不写 promotion。
```

## 11. 用户再次追问后的 stop-go 复核

复核命令：

```bash
rg -n "stop|go|Stop|Gate|K2|K3|K5|K6|action|grid|controller|No-go|no-go|如果|若|failure|Failure|route|promotion" \
  docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
```

关键计划条款：

```text
If K2 9/9 and controls fail:
  S5 candidate, promotion allowed only after audit/code review.

If K2 <9/9:
  stop zero_moment_reset route; do not add action tokens.

If K3/K5/K6 beat K2:
  update mechanism interpretation; still no action search.

If all state transport fail:
  return to substrate or FMS design, not action search.
```

对照当前结果：

```text
official_v147:
  K2 strict = 0/9
  controls endpoint full success = 3

repair_v147_event_only_transport:
  K2 strict = 0/9
  K3/K5/K6 未打开 S5

repair_v147_event_only_delta_top25:
  K2 strict = 0/9
  K3/K5/K6 未打开 S5
```

执行判断：

```text
本次不新增训练。
原因是完整计划已给出 stop-go：
K2 < 9/9 时停止 zero_moment_reset route，不新增 action token。
继续尝试 strength/lambda/lr/refresh、controller、action token、
或 audit-metric-driven coordinate 都违反当前计划约束。
```

## 12. 用户再次追问后的计划原文复核

复核命令：

```bash
sed -n '470,740p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
sed -n '860,960p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
sed -n '960,1015p' docs/DG-KAN_v14.7_OptimizerStateTransportFMS_AllBasisParallel_完整计划.md
```

关键原文复核：

```text
5.4 Pass / fail:
  R0 triggers include action token, strength/lambda/lr/refresh grid,
  controller, audit metric direction, dataset/seed branch,
  diagnostic success as promotion.

6.7 failure route:
  R1-ZeroMomentResetNotReplicated: K2 < 9/9.
  R2-ResetWorksButNotFMSSpecific: K2 passes but random/full reset also passes.
  R3/R4/R5 only apply if K3/K5/K6 beat K2.

10.2 Stop-go:
  If K2 <9/9:
    stop zero_moment_reset route; do not add action tokens.

11 Case B:
  v14.6.1 S4c diagnostic did not reproduce in official runner.
  Do not continue with state reset variants unless new mechanism evidence appears.
```

执行判断：

```text
本次仍不新增训练。
当前结果落入：
  official_v147 = R2 generic reset control issue
  event-only / delta_top25 = R1 K2 < 9/9

没有 K3/K5/K6 beat K2 的证据；
继续 state reset variants 没有新的机制证据，违反 Case B 边界。
```
