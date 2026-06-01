# DG-KAN v15.02.1 ExploreOpenExecutionContract FunctionMetric AllBasis 执行日志

生成时间：2026-05-31（Asia/Singapore）

## 1. 计划文件

```text
/home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.02.1_ExploreOpenExecutionContract_FunctionMetric_AllBasis_完整计划.md
```

## 2. 代码修改

```text
experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py
experiments/run_v149_line_d_all_basis_substrate_repair.py
```

## 3. Line D substrate-only 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 12319500,12319501,12319502 --candidates D-FOU47-LowFreqIdentityResidualV5,D-FOU48-BandwiseSNRWarmupV3,D-FOU49-PhaseStableBandMixV3,D-FOU50-NoMaterializeLifetimeV3,D-FOU51-HighFreqQuarantineV2,D-RBF45-ActiveCenterOccupancyV3,D-RBF46-WidthConditionGuardV3,D-RBF47-CompactBumpNoDenseV3,D-RBF48-IdentityResidualV3,D-RBF49-GaussianLocalK4TaskHealth,D-WAV41-TriangularSupportV5,D-WAV42-ScaleOccupancyV4,D-WAV43-SupportOverlapDampingV3,D-WAV44-LocalTailCoverageAudit
```

## 4. v15.02.1 official 执行指令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py --out-dir results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/official_v1521 --line-d-out results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --trace-interval 30 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

## 5. 关键输出目录

```text
official_out = results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/official_v1521
line_d_out = results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate
```

## 6. 用户再次追问后的 finalizer 覆盖修复

说明：该命令复用已有训练 artifact，仅重写 route / manifest / audit / docs；不新增训练数据。

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py --out-dir results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/official_v1521 --line-d-out results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --trace-interval 30 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 1
```

修复内容：

```text
1. Line C audit 补齐 source_vs_control / AUCtime_ratio / LineC_fail_reason。
2. forbidden/no-action audit 补齐计划 5.3 readback 字段。
3. method surface manifest 补齐 Line D executed_rows。
4. route 显式加入 Line M controls explain positives 条件。
5. 再次追问后修正 G/P gate summary：只用实际 effect rows 计算 route 数值，audit/decomposition rows 不再参与 gate。
6. 补齐 v1521_dche_no_regression_monitor.csv 与 v1521_rational_no_regression_monitor.csv。
7. 补齐 v1521_line_z_no_go_taxonomy.csv，区分 promotion no-go / exploration no-go / budget-deferred / implementation blocker / theoretical no-go。
8. 将上述 monitor/taxonomy 纳入 v1521_required_artifact_manifest.csv。
9. 补齐 v1521_line_m_positive_row_control_map.csv：266 个 positive-looking rows 均有 matched generic controls，且 266/266 被 generic/MLP control 解释。
10. 补齐 v1521_line_d_family_exhaustion_certificates.csv 的 family-specific fallback 字段：D-FOU=5、D-RBF=8、D-WAV=7 个 fallback/readback 标记。
11. 补齐 v1521_line_x_transfer_operator_audit.csv 的 per-row failure_class，10/10 rows = X-Fail-LocalPositiveNoTransfer。
```

## 7. 最终核验

```text
route = R15_2_1-CurrentFunctionalDefinitionNoGo
minimum_success = S1-ExploreOpenExecutionContractExecuted
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 8. 用户再次追问后的 method/control surface 反方复核

说明：该命令只读取 official artifacts，核对完整计划中的 main methods、fallback ladder、generic controls 与 Line D candidates 是否存在漏跑；不新增训练数据。

```text
G_missing_prefixes = []
P_missing_prefixes = []
M_missing = []
D_missing_prefixes = []
G fallback missing = []
P fallback missing = []
```

## 9. 用户再次追问后的独立 gate 重算复核

说明：该复核由 finalizer 写入 v1521_gate_recompute_audit.csv，只重算 gate/route 条件；不新增训练数据。

```text
S2-FunctionMetricExplorationPositive: recomputed_pass=0 route_consistent=1
S3-FunctionMetricMeaningful: recomputed_pass=0 route_consistent=1
S4-RealTransferExploration: recomputed_pass=0 route_consistent=1
S5-Official: recomputed_pass=0 route_consistent=1
R-G-FunctionMetricExhausted: recomputed_pass=1 route_consistent=1
R-P-ProximalMetricExhausted: recomputed_pass=1 route_consistent=1
R-X-LocalPositiveNoTransfer: recomputed_pass=1 route_consistent=1
R-D-AllBasisSubstrateExhausted: recomputed_pass=1 route_consistent=1
R-M-GenericOptimizerExplainsGain: recomputed_pass=1 route_consistent=1
R15_2_1-CurrentFunctionalDefinitionNoGo: recomputed_pass=1 route_consistent=1
gate_route_inconsistent_rows = 0
```

## 10. 用户再次追问后的执行合同闭合复核

说明：该复核由 finalizer 写入 v1521_execution_contract_coverage_audit.csv，只读取计划内 artifact 覆盖状态；不新增训练数据。

```text
Line R artifact/provenance/audit: status=1
Line G main plus fallback ladder: status=1
Line P main plus fallback ladder: status=1
Line X transfer/operator audit: status=1
Line D all-basis fresh hardening and exhaustion: status=1
Line M positive-looking matched controls: status=1
Line C/tail audit: status=1
Per-fail failure taxonomy: status=1
Line Z no-go taxonomy and next queue: status=1
Required figures: status=1
Independent gate recompute: status=1
No remaining v15.02.1 executable fallback: status=1
contract_unclosed_rows = 0
```

## 11. v15.01/v15.02 自查后的 P-FB4 proximal fallback 修复与 GPU 重跑

检查与编译：

```bash
python -m py_compile experiments/run_v151_control_residual_function_update_allbasis_acceleration.py experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
```

第一次修复与中间重跑：

```text
修复内容：
  P-FB4 selected alpha but no commit 在 alpha selection 阶段不再跳过 trial update。

中间重跑命令：
```

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py --out-dir results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/official_v1521 --line-d-out results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate --device cuda:1 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --trace-interval 30 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

第二次复核发现并修复：

```text
发现问题：
  P-FB4 仍未真正进入 proximal solver，因为 train_case 只判断 line == "P"。

修复内容：
  将 proximal 分支判断改为 line.startswith("P")，
  使 P-FB4 真正执行 proximal objective / commit-effect audit。
  新增 resolve_cuda_device 防护：
    --device 非 cuda 或 CUDA 不可用时直接 RuntimeError，
    不再静默 fallback 到 CPU。
```

最终 GPU full rerun：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py --out-dir results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/official_v1521 --line-d-out results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --trace-interval 30 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

finalizer / docs rewrite：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py --out-dir results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/official_v1521 --line-d-out results/v15_02_1_explore_open_execution_contract_function_metric_allbasis/line_d_v1521_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 60 --trace-interval 30 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 1
```

最终复核结果：

```text
route = R15_2_1-CurrentFunctionalDefinitionNoGo
minimum_success = S1-ExploreOpenExecutionContractExecuted
line_p_source_vs_best_control_mean = -0.03708608945210775
line_p_control_equivalent_fraction = 0.6666666666666666
line_p_real_lite_pass_count = 0 / 9
best_p_method = P4-D-CHE-RandomSketchFunctionProx
P-FB1 rows = 35100
P-FB4 rows = 72
selected-no-commit objective rows = 5400
selected-no-commit objective nonzero B1 rows = 5400
selected-no-commit objective B1_improved rows = 5380
positive_rows_checked_by_m = 266
positive_row_map_missing_control_rows = 0
positive_row_map_generic_explained_rows = 266
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
promotion_allowed = 0
```

## 12. GPU guard entry 修复
```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v151_control_residual_function_update_allbasis_acceleration.py experiments/run_v1521_explore_open_execution_contract_function_metric_allbasis.py experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py
```

结果：
```text
run(args) entry now calls resolve_cuda_device(args.device)
reuse-if-present/finalizer path also requires --device cuda:*
metric artifacts unchanged
route = R15_2_1-CurrentFunctionalDefinitionNoGo
promotion_allowed = 0
```
