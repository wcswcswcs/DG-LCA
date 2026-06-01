# DG-KAN v15.01 ControlResidualFunctionUpdate AllBasisAcceleration 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 CR-FU real-lite、substrate replay 或 MLP/generic control 写成 promotion。

## 1. 计划理解

v15.01/v15.1 的目标是检验：在 AdamW/Cautious/MGUP/second-moment/decay 等 controls 已经解释的方向被投掉后，FU 是否仍有独立 residual causal value。

## 2. 本轮代码修改

新增：

```text
experiments/run_v151_control_residual_function_update_allbasis_acceleration.py
```

修改：

```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.1 预注册 substrate-only candidates:
  D-FOU42..46, D-RBF40..44, D-WAV37..40。
```

过程修正：

```text
首次 official finalizer 完成实际训练矩阵后，route precedence 按中间诊断写成 R3-GenericOptimizerExplainsCRFU。
复核计划第 16 节 final decision rule 后，确认 F/K/D 同时失败且 generic controls explain 时，最终 decisive route 应为 R7-CurrentFUFamilyNoGo。
已修正 build_route precedence，并用 reuse-if-present=1 重写 route / manifest / 两份日志；该修正不改变任何训练指标。
用户再次追问后复核 Line O，发现 random_matched_decay_explains_fraction 初版固定为 0。
已改为使用当前 train-stream decay active mask 的 matched-random decay direction 做 metric projection readback；该 readback 不进入更新方向、不使用 audit metric、不新增 token。
再次复核发现 F6/M3 FunctionSpaceProximalResidual 初版使用一阶闭式近似 alpha，没有实际执行计划 8.2 的 train split B1 proximal objective。
已改为固定 alpha candidates = 0,0.05,0.10,0.25,0.50，在当前 train batch B1 上评估 CE + proximal penalty 选 alpha；不使用 validation/test/future/audit metric，不构成 action search。
用户再次追问后复核 control-residual 语义，发现 full residualizer 初版只投掉 Adam/Cautious/MGUP/decay，没有投掉计划中的 matched-random/same-active/same-projection/same-value-retention controls。
已将这些固定 train-stream matched control directions 纳入 F6/F7/M3/M4 full residualizer；不新增 FU token，不使用 audit metric，不构成 action search。
再次复核 Line O readback，发现 decoupled_decay_explains_fraction 初版是 decay_norm/raw_FU_norm，而不是 decay 对 raw FU 的 metric projection fraction。
已改为 decoupled decay direction 对 raw FU 的 metric projection readback；该指标只用于诊断，不进入方向。
```

CR-FU runner 实现：Line R/O/C0/F/M/K/D/C/Z；direction 只使用当前 train stream、当前 optimizer state 和 train-stream basis telemetry；LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate。

## 3. Line O / C0 结果

```text
direction_decomposition_rows = 540
mean_control_projection_norm_fraction = 0.3768205677114437
mean_control_residual_norm_fraction = 0.6231794322885563
mean_anti_alignment_fraction = 0.23809461378388935
mean_decoupled_decay_explains_fraction = 2.261822460070231e-05
mean_random_matched_decay_explains_fraction = 1.6123468965881815e-06
proximal_solver_used_train_split_rows = 54
full_residualizer_control_count_max = 8
```

## 4. Line F D-CHE CR-FU 结果

```text
line_f_candidate_count = 8
real_lite_pass_count = 1 / 9
source_vs_best_control_mean = -0.21726235416200426
control_equivalent_fraction = 0.9333333333333333
bad_event_fraction = 0.9777777777777777
line_f_exploration_gate_pass = 0
line_f_meaningful_gate_pass = 0
line_f_s4_gate_pass = 0
best_crfu_method = F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl
best_crfu_mean_source_vs_best_control = -0.002271784676445855
```

Method summary：

| method | rows | strict pass | dataset-seed pass | mean source vs best control | mean residual fraction |
|---|---:|---:|---:|---:|---:|
| F0-D-CHE-AdamW | 9 | 0 | 0 | -0.24988547298643324 | 0.5070169877218071 |
| F1-D-CHE-CautiousAdamW | 9 | 0 | 0 | -0.3026495708359612 | 1.0 |
| F2-D-CHE-MGUPControl | 9 | 0 | 0 | 0.0 | 1.0 |
| F3-D-CHE-RawFUResidualizedAgainstAdamW | 9 | 0 | 0 | -0.25830570194456315 | 0.5068081501711165 |
| F4-D-CHE-RawFUResidualizedAgainstCautious | 9 | 0 | 0 | -0.30511630905999076 | 0.4964823063336168 |
| F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl | 9 | 1 | 1 | -0.002271784676445855 | 0.48847922691768303 |
| F6-D-CHE-FunctionSpaceProximalResidual | 9 | 0 | 0 | -0.25899749994277954 | 0.49340971572912756 |
| F7-D-CHE-CRFU-AbstentionEnabled | 9 | 0 | 0 | -0.261620475186242 | 0.4932390714350997 |

## 5. Line M / K 结果

```text
mlp_generic_rows = 72
kan_specific_pass_count = 0
delta_kan_specific_mean = -0.23622695604960123
generic_optimizer_explains_crfu = 1
```

判断：MLP/generic controls 只作为 confound audit；不写成 KAN-specific promotion。

## 6. Line D all-basis 结果

```text
line_d_source = v151_actual_v149_substrate_acceleration
line_d_rows = 126
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 0 / 9
line_d_official_fu_eligible_family_count = 0
line_d_route = R6-AllBasisSubstrateBlocked
```

Line D family summary：

| family | rows | pass | max mean delta vs MLP | best LineC pass rate | official eligibility |
|---|---:|---:|---:|---:|---:|
| D-FOU | 45 | 0/9 | -0.109375 | 1.0 | 0 |
| D-RBF | 45 | 0/9 | -0.328125 | 1.0 | 0 |
| D-WAV | 36 | 0/9 | -0.0234375 | 1.0 | 0 |

## 7. 最终 route

```text
route = R7-CurrentFUFamilyNoGo
minimum_success = S1-CRFUSolverImplemented
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 8. 科学结论

```text
1. v15.01 已执行 Line R/O/C0/F/M/K/D/C/Z。
2. CR-FU 不允许写成 promotion，除非 S5 official gate 达成。
3. 本轮 route = R7-CurrentFUFamilyNoGo，promotion_allowed = 0。
4. 若 Line F / K / D 均未打开 gate，则当前 FU family 应关闭，不继续 FU9/FU10 或 action/controller/reset。
```

## 9. v15.01/v15.02 自查后的 B1 proximal 实现复核

本次自查发现一个实现语义 bug：

```text
发现问题：
  F6/M3 FunctionSpaceProximalResidual 已经改成固定 alpha candidates，
  但 crfu_update 内部实际用整批 current train batch 计算 proximal objective，
  而不是计划 8.2 写明的 train split B1。

修复内容：
  在 crfu_update 中将当前 train batch 固定切分为 B1 = batch[:batch_size/2]，
  F6/M3 的 CE + proximal penalty alpha selection 只在 B1 上评估。
  同时新增 resolve_cuda_device 防护：--device 非 cuda 或 CUDA 不可用时直接报错，
  不再静默 fallback 到 CPU。

合法性：
  只使用当前 train-stream batch；
  不使用 validation/test/future/query；
  不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier 生成方向；
  不新增 FU/F-CHE token、action bank、controller 或 reset route。
```

GPU 复跑后关键结果：

```text
device = cuda:0
direction_decomposition_rows = 540
F6-D-CHE-FunctionSpaceProximalResidual proximal rows = 27
M3-MLP-FunctionSpaceProximalResidual proximal rows = 27
F6/M3 proximal_solver_used_train_split rows = 54
F6/M3 alpha_set = 0.5
full_residualizer_control_count = 8

F6 mean source_vs_best_control = -0.25899749994277954
F6 strict_gate_pass = 0 / 9
F5 best_crfu_mean_source_vs_best_control = -0.002271784676445855
real_lite_pass_count = 1 / 9
control_equivalent_fraction = 0.9333333333333333
generic_optimizer_explains_crfu = 1
route = R7-CurrentFUFamilyNoGo
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

结论：

```text
该 bug 修复后，v15.01 的实验语义更符合计划；
但没有产生新的 positive conclusion。
当前 CR-FU family 仍没有达到 S2/S3/S4/S5。
```

## 10. 用户再次追问后的 GPU guard 入口修复

本次没有新增训练，也没有改变 v15.01 指标；只修复执行合同层缺口：

```text
发现问题：
  训练路径已要求 --device cuda，但 reuse-if-present/finalizer 路径可能先读取已有 artifact，
  从而绕过 GPU 校验。

修复：
  run(args) 入口新增 resolve_cuda_device(args.device)。
  --device 非 cuda 或 CUDA 不可用时直接报错。
```

验证：

```text
py_compile = pass
cuda_available = True
v15.01 route 结论不变：
  route = R7-CurrentFUFamilyNoGo
  promotion_allowed = 0
```
