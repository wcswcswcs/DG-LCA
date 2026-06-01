# DG-KAN v15.01 ControlResidualFunctionUpdate AllBasisAcceleration 执行日志

生成时间：2026-05-31（Asia/Singapore）

## 1. 计划文件

```text
/home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.01_ControlResidualFunctionUpdate_AllBasisAcceleration_完整计划.md
```

## 2. 代码修改

```text
experiments/run_v151_control_residual_function_update_allbasis_acceleration.py
experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. Line D substrate-only 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_1_control_residual_function_update_allbasis_acceleration/line_d_v151_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 12319500,12319501,12319502 --candidates D-FOU42-LowFreqResidualV5,D-FOU43-BandwiseSecondMomentWarmup,D-FOU44-PhaseStableLowBandOnly,D-FOU45-NoMaterializeLifetimeV4,D-FOU46-HighFreqQuarantineLateEnable,D-RBF40-CompactBumpIdentityResidualV2,D-RBF41-ActiveCenterOccupancySecondMoment,D-RBF42-WidthFloorTrustRegion,D-RBF43-GaussianLocalK4NoDenseV2,D-RBF44-CenterReadoutDecoupledWarmup,D-WAV37-TriangularSupportV5,D-WAV38-ScaleOccupancySecondMoment,D-WAV39-LocalSupportOverlapTrust,D-WAV40-FineScaleLateEnable
```

## 4. v15.01 official full-matrix 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v151_control_residual_function_update_allbasis_acceleration.py --out-dir results/v15_1_control_residual_function_update_allbasis_acceleration/official_v151 --line-d-out results/v15_1_control_residual_function_update_allbasis_acceleration/line_d_v151_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --trace-interval 60 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

## 5. route precedence 修正后重写指令

说明：该命令只复用已完成训练 artifact，重写 route / manifest / docs；不新增训练数据。

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v151_control_residual_function_update_allbasis_acceleration.py --out-dir results/v15_1_control_residual_function_update_allbasis_acceleration/official_v151 --line-d-out results/v15_1_control_residual_function_update_allbasis_acceleration/line_d_v151_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --trace-interval 60 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 1
```

## 5.1 用户再次追问后的 Line O decay readback 修复

```text
修复内容：random_matched_decay_explains_fraction 不再固定为 0；改为当前 train-stream decay active mask 下的 matched-random decay metric projection readback。
合法性：该 readback 不进入更新方向，不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier，不新增 FU/F-CHE token，不启动 action/controller/reset。
复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。
```

## 5.2 用户再次追问后的 F6/M3 proximal solver 修复

```text
发现问题：F6/M3 初版用一阶闭式近似 alpha，未实际评估计划 8.2 的 train split B1 proximal objective。
修复内容：固定 alpha candidates = 0,0.05,0.10,0.25,0.50，在当前 train batch B1 上评估 CE + proximal penalty 选择 alpha。
合法性：只使用当前 train stream；不使用 validation/test/future/query/LineC/CEp99/NLL/ECE/AUCtime/Brier；不新增 FU token，不做 action bank/controller/reset。
复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。
```

## 5.3 用户再次追问后的 full residualizer matched-control 修复

```text
发现问题：F6/F7/M3/M4 的 full residualizer 初版只包含 Adam/Cautious/MGUP/decay，没有纳入计划 control-residual 语义中的 matched-random/same-active/same-projection/same-value-retention controls。
修复内容：将固定 train-stream matched controls 纳入 full residualizer，记录 optimizer/matched/full residualizer control count。
合法性：只使用当前 train stream；不使用 validation/test/future/query/LineC/CEp99/NLL/ECE/AUCtime/Brier；不新增 FU/F-CHE token，不做 action bank/controller/reset。
复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。
```

## 5.4 用户再次追问后的 decoupled decay projection readback 修复

```text
发现问题：decoupled_decay_explains_fraction 初版是 decay_norm/raw_FU_norm，不是 decay direction 对 raw FU 的 metric projection fraction。
修复内容：改为 decoupled decay direction 对 raw FU 的 metric projection readback，并记录 decoupled_decay_projection_norm。
合法性：该 readback 不进入更新方向，不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier，不新增 FU/F-CHE token，不启动 action/controller/reset。
复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。
```

## 6. 关键输出目录

```text
official_out = results/v15_1_control_residual_function_update_allbasis_acceleration/official_v151
line_d_out = results/v15_1_control_residual_function_update_allbasis_acceleration/line_d_v151_allbasis_substrate
```

## 7. 关键 artifact

```text
v151_route_decision.json
v151_dche_crfu_results.csv
v151_dche_crfu_controls.csv
v151_mlp_generic_controls.csv
v151_kan_specificity_summary.csv
v151_allbasis_substrate_results.csv
v151_required_artifact_manifest.csv
```

## 8. 最终核验

```text
route = R7-CurrentFUFamilyNoGo
minimum_success = S1-CRFUSolverImplemented
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 9. v15.01/v15.02 自查后的 B1 proximal bug 修复与 GPU 重跑

检查与编译：

```bash
python -m py_compile experiments/run_v151_control_residual_function_update_allbasis_acceleration.py experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
```

修复内容：

```text
experiments/run_v151_control_residual_function_update_allbasis_acceleration.py
  crfu_update:
    F6/M3 FunctionSpaceProximalResidual 的 alpha objective
    从整批 xb/yb 改为当前 train batch 的 B1 = xb[:batch_size/2]。
  resolve_cuda_device:
    --device 非 cuda 或 CUDA 不可用时直接 RuntimeError，
    不再静默 fallback 到 CPU。
```

GPU full rerun：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v151_control_residual_function_update_allbasis_acceleration.py --out-dir results/v15_1_control_residual_function_update_allbasis_acceleration/official_v151 --line-d-out results/v15_1_control_residual_function_update_allbasis_acceleration/line_d_v151_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --trace-interval 60 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

复核结果：

```text
direction_decomposition_rows = 540
F6 proximal rows = 27
M3 proximal rows = 27
F6/M3 alpha_set = 0.5
full_residualizer_control_count = 8
real_lite_pass_count = 1 / 9
F6 strict_gate_pass = 0 / 9
F6 mean source_vs_best_control = -0.25899749994277954
route = R7-CurrentFUFamilyNoGo
promotion_allowed = 0
```

## 10. GPU guard entry 修复
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v151_control_residual_function_update_allbasis_acceleration.py experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py
```

结果：
```text
run(args) entry now calls resolve_cuda_device(args.device)
reuse-if-present/finalizer path also requires --device cuda:*
metric artifacts unchanged
route = R7-CurrentFUFamilyNoGo
promotion_allowed = 0
```
