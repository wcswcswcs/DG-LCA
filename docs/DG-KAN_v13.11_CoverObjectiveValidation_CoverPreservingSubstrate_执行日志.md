# DG-KAN v13.11 CoverObjectiveValidation CoverPreservingSubstrate 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志只记录实际执行过的命令、输出目录和复现要点。

## 1. 计划与代码读取

```bash
sed -n '1,260p' docs/DG-KAN_v13.11_CoverObjectiveValidation_CoverPreservingSubstrate_完整计划.md
rg --files | rg 'v13\.11|v1311|CoverObjective|CoverPreserving|cover_objective|cover_preserving'
rg -n "route|Route|R[0-9]|S3|S4|S5|K19|K20|K21|K22|K23|artifact|manifest|stop|Stop|停止|执行|runner|Non-RAT|promotion|gate|修复|fallback|official|smoke|Cover" docs/DG-KAN_v13.11_CoverObjectiveValidation_CoverPreservingSubstrate_完整计划.md
sed -n '260,560p' docs/DG-KAN_v13.11_CoverObjectiveValidation_CoverPreservingSubstrate_完整计划.md
sed -n '560,940p' docs/DG-KAN_v13.11_CoverObjectiveValidation_CoverPreservingSubstrate_完整计划.md
```

相关代码读取：

```bash
sed -n '1,240p' experiments/run_v1310_cover_forming_substrate_architecture_reset.py
rg -n "def run_|def build_|def write_|def decide|route|cover_substrate|main\\(" experiments/run_v1310_cover_forming_substrate_architecture_reset.py
sed -n '240,620p' experiments/run_v1310_cover_forming_substrate_architecture_reset.py
sed -n '620,1120p' experiments/run_v1310_cover_forming_substrate_architecture_reset.py
```

## 2. 代码修改与语法检查

新增 runner：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

结果：

```text
py_compile pass
```

## 3. smoke

```bash
conda run -n kan python experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py \
  --out-dir results/v13_11_cover_objective_validation_cover_preserving_substrate/smoke_v1311 \
  --cover-objective-methods V0-AdamWControl,V1-FutureGradientClusterOracle,V5-RandomCoverControl,V6-PermutedOracleControl \
  --synthetic-tasks X1 \
  --synthetic-seeds 0 \
  --k-losses CE \
  --train-steps 4 \
  --batch-size 16 \
  --log-interval 2 \
  --synthetic-train-size 32 \
  --synthetic-val-size 16 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 64 \
  --real-val-size 32 \
  --real-test-size 32 \
  --real-epochs 1
```

第一次 smoke 发现 manifest 自引用顺序问题：

```text
v1311_required_manifest.csv 被列入 required manifest；
第一次 manifest 检查发生在该 manifest 写入前，导致 required_artifact_missing_count=1。
```

修复：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  写出 v1311_required_manifest.csv 后重新检查 manifest；
  回填 route.required_artifact_missing_count 并重写 route。
```

重跑同一 smoke 后：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_rows = 4
oracle_best_task_family_pass_count = 0
oracle_best_source_vs_best_control = 0.360177747701133
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
nonrat_vertical_pass_count = 0
mlp_cover_dataset_pass_count = 0 / 1
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 4. official_v1311 Line V

```bash
conda run -n kan python experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py \
  --out-dir results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311 \
  --cover-objective-methods V0-AdamWControl,V1-FutureGradientClusterOracle,V2-LineCSignalReservoirOracle,V3-TaskFamilyOracle,V4-LabelClassCentroidOracle,V5-RandomCoverControl,V6-PermutedOracleControl \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --k-losses CE,Brier \
  --train-steps 60 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --mlp-seeds 0,1 \
  --mlp-seed-threshold 2 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

输出目录：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311
```

route 摘要：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_rows = 98
oracle_best_task_family_pass_count = 1
oracle_best_source_vs_best_control = 0.23338876249709983
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
nonrat_vertical_pass_count = 0
mlp_cover_dataset_pass_count = 0 / 3
required_artifact_missing_count = 0
promotion_allowed = 0
final_stop_allowed = 1
```

## 5. Line V seed1 repeat

触发原因：official 有一个 V3/X5 CE 局部 pass row。为避免单 seed 偶然性误判，额外执行 seed1 repeat。该 run 仍是 Line V diagnostic，不替代 official，不允许 promotion。

```bash
conda run -n kan python experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py \
  --out-dir results/v13_11_cover_objective_validation_cover_preserving_substrate/linev_repeat_seed1_v1311 \
  --cover-objective-methods V0-AdamWControl,V1-FutureGradientClusterOracle,V2-LineCSignalReservoirOracle,V3-TaskFamilyOracle,V4-LabelClassCentroidOracle,V5-RandomCoverControl,V6-PermutedOracleControl \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 1 \
  --k-losses CE,Brier \
  --train-steps 40 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_rows = 98
oracle_best_task_family_pass_count = 1
oracle_best_source_vs_best_control = 0.14723053126705488
random_permuted_control_task_pass_count = 0
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 6. 结果解析与进程复核

```bash
python - <<'PY'
import csv,json
for run in ['official_v1311','linev_repeat_seed1_v1311']:
    base=f'results/v13_11_cover_objective_validation_cover_preserving_substrate/{run}'
    r=json.load(open(f'{base}/v1311_route_decision.json'))
    print(run, r['route'], r['cover_objective_valid'], r['oracle_best_task_family_pass_count'], r['required_artifact_missing_count'], r['promotion_allowed'])
    for row in csv.DictReader(open(f'{base}/v1311_cover_objective_validity.csv')):
        print(row['cover_objective_method'], row['task_family_pass_count'], row['max_source_vs_best_control'], row['median_cover_purity'], row['median_AUC_time_ratio'])
PY

ps -ef | rg 'run_v1311|run_v1310|run_v139|run_v136' | rg -v 'rg|pgrep' || true
```

最终进程复核：

```text
`ps -ef | rg ... | rg -v ...` 无输出；
无 v13.11 / v13.10 / v13.9 / v13.6 实验进程残留。
```

## 7. 用户再次追问后的 objective-reset diagnostic

触发原因：用户再次要求未达成则继续。复核计划第 16.1 节后，确认当前 route = R2-CoverObjectiveInvalid 时禁止继续 A-CPF / Line K architecture search；安全继续方向只能是 objective reset 诊断，而不是把未验证 objective 当作 substrate 搜索目标。

新增可复现分析脚本：

```text
experiments/analyze_v1311_objective_reset_alignment.py
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/analyze_v1311_objective_reset_alignment.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/analyze_v1311_objective_reset_alignment.py
```

输入 artifact：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_oracle_cover_results.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/official_v1311/v1311_random_permuted_cover_controls.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/linev_repeat_seed1_v1311/v1311_oracle_cover_results.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/linev_repeat_seed1_v1311/v1311_random_permuted_cover_controls.csv
```

输出 artifact：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/objective_reset_diagnostic_v1311/v1311_objective_reset_route.json
results/v13_11_cover_objective_validation_cover_preserving_substrate/objective_reset_diagnostic_v1311/v1311_objective_reset_metric_alignment.csv
results/v13_11_cover_objective_validation_cover_preserving_substrate/objective_reset_diagnostic_v1311/v1311_objective_reset_method_summary.csv
```

结果：

```text
diagnostic_route = D1-ObjectiveResetNeeded_CurrentCoverMetricsWeakValueAlignment
official_route_unchanged = R2-CoverObjectiveInvalid
uses_existing_artifacts_only = 1
new_training_executed = 0
line_a_k_architecture_search_executed = 0
promotion_allowed = 0
cover_metric_alignment_pass = 0
combined_rows = 168
combined_oracle_rows = 112
combined_control_rows = 56
combined_task_family_pass_rows = 2
combined_value_positive_rows = 3
official_cover_purity_spearman_oracle = -0.16657552973342446
repeat_cover_purity_spearman_oracle = -0.08448393711551606
combined_cover_purity_spearman_oracle = -0.10525506543205658
combined_signal_to_cover_spearman_oracle = -0.21369631325383537
```

解释：

```text
1. 该诊断只读既有 Line V artifact，没有新增训练。
2. cover_purity 与 signal_to_cover_score 对 source_vs_best_control 的相关性在 official / repeat / combined 中均为负。
3. 这支持 objective-reset 判断，不支持继续 A-CPF/K architecture search。
4. v13.11 official route 不变，不允许 promotion。
```

## 8. 用户再次追问后的 V7 train-stream value-cover proxy diagnostic

触发原因：用户再次要求未达成则继续。根据第 16.1 节，不能继续 A-CPF / Line K architecture search；因此本次尝试把第 18 节 useful negative 中的 “redefine cover objective” 具体化为一个新的 train-stream-only objective-reset diagnostic。

代码修改：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  新增 OBJECTIVE_RESET_METHODS:
    V7-TrainStreamValueCoverProxy
  新增 train_stream_value_cover_mask()
  新增 objective_reset_candidate / uses_train_stream_only_value_proxy 审计字段
```

合法性说明：

```text
1. V7 只使用当前 train batch 的 per-example gradients / group slices / parameter telemetry。
2. V7 不使用 validation/test/future/query batch 生成方向。
3. V7 不使用 LineC / CEp99 / NLL / ECE 生成方向。
4. V7 不是 v13.11 official oracle validity gate；即使有局部 row，也不能 promotion。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py \
  --out-dir results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v7_v1311 \
  --cover-objective-methods V0-AdamWControl,V5-RandomCoverControl,V6-PermutedOracleControl,V7-TrainStreamValueCoverProxy \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0,1 \
  --k-losses CE,Brier \
  --train-steps 60 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
mlp_cover_dataset_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
compute_budgeted_run = 1
```

validity rows：

| method | rows | task pass | control pass | median source | max source | median cover | median AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 28 | 0 | 0 | 0.0 | 0.0 | 0.0 | 1.0 |
| V5-RandomCoverControl | 28 | 0 | 0 | -0.22668853402137756 | 0.0 | 0.880892813205719 | 1.231992483139038 |
| V6-PermutedOracleControl | 28 | 0 | 0 | -0.17707574367523193 | 0.0 | 0.9499045014381409 | 1.2111748456954956 |
| V7-TrainStreamValueCoverProxy | 28 | 0 | 0 | -0.31016838550567627 | 0.0736189368426392 | 0.9442763328552246 | 1.241366982460022 |

artifact:

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v7_v1311/
```

进程复核：

```bash
ps -ef | rg 'run_v1311|analyze_v1311|run_v1310|run_v139|run_v136' | rg -v 'rg|pgrep' || true
```

结果：

```text
无输出；没有实验进程残留。
```

## 10. 用户再次追问后的 V10 near-AdamW residual value-cover proxy diagnostic

触发原因：V8/V9 soft floor 仍为负 source。为排除“proxy 仍然太多削弱 AdamW 基线”这一解释，本次继续实现 near-AdamW residual proxy。

代码修改：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  新增：
    V10-TrainStreamNearAdamWValueCoverProxy
  语义：
    train_stream_value_cover_mask(..., floor=0.85)
```

合法性说明：

```text
1. V10 只使用当前 train batch 的 per-example gradients / group slices。
2. V10 保留 85% 全梯度，仅做小幅 group reweight。
3. 不使用 validation/test/future/query batch 生成方向。
4. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
5. 只作为 post-R2 objective-reset diagnostic；不允许 promotion。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py \
  --out-dir results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v10_v1311 \
  --cover-objective-methods V0-AdamWControl,V5-RandomCoverControl,V6-PermutedOracleControl,V10-TrainStreamNearAdamWValueCoverProxy \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --k-losses CE,Brier \
  --train-steps 40 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
compute_budgeted_run = 1
```

validity rows：

| method | rows | task pass | control pass | median source | max source | median cover | median AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 14 | 0 | 0 | -0.0017003349494189024 | 0.0 | 0.0 | 1.0021430253982544 |
| V5-RandomCoverControl | 14 | 0 | 0 | -0.17035727202892303 | 0.0 | 0.5056086778640747 | 1.2395497560501099 |
| V6-PermutedOracleControl | 14 | 0 | 0 | -0.14652642607688904 | 0.0 | 0.9624462127685547 | 1.1120707988739014 |
| V10-TrainStreamNearAdamWValueCoverProxy | 14 | 0 | 0 | -0.10909856855869293 | 0.14819857915185564 | 0.9789875745773315 | 1.1364250183105469 |

artifact：

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v10_v1311/
```

进程复核：

```bash
ps -ef | rg 'run_v1311|analyze_v1311|run_v1310|run_v139|run_v136' | rg -v 'rg|pgrep' || true
```

结果：

```text
无输出；没有实验进程残留。
```

## 9. 用户再次追问后的 V8/V9 soft value-cover proxy diagnostic

触发原因：V7 是硬 mask，source 伤害明显。为排除“硬 cover 截断过激”这个实现层解释，本次继续实现更保守的 train-stream-only objective-reset proxies。

代码修改：

```text
experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
  新增：
    V8-TrainStreamSoftValueCoverProxy
    V9-TrainStreamResidualSafeValueCoverProxy
  train_stream_value_cover_mask() 增加 floor 参数。
```

语义：

```text
V8: train-stream value-cover group score + soft floor = 0.25。
V9: train-stream value-cover group score + residual-safe floor = 0.60。
```

合法性说明：

```text
1. V8/V9 只使用当前 train batch 的 per-example gradients / group slices。
2. 不使用 validation/test/future/query batch 生成方向。
3. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
4. 只作为 post-R2 objective-reset diagnostic；不允许 promotion。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v1311_cover_objective_validation_cover_preserving_substrate.py \
  --out-dir results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v8v9_v1311 \
  --cover-objective-methods V0-AdamWControl,V5-RandomCoverControl,V6-PermutedOracleControl,V8-TrainStreamSoftValueCoverProxy,V9-TrainStreamResidualSafeValueCoverProxy \
  --synthetic-tasks X1,X2,X3,X4,X5,X6,X7 \
  --synthetic-seeds 0 \
  --k-losses CE,Brier \
  --train-steps 40 \
  --batch-size 32 \
  --log-interval 20 \
  --synthetic-train-size 96 \
  --synthetic-val-size 48 \
  --datasets MNIST \
  --mlp-seeds 0 \
  --mlp-seed-threshold 1 \
  --real-train-size 128 \
  --real-val-size 64 \
  --real-test-size 64 \
  --real-epochs 1
```

结果：

```text
route = R2-CoverObjectiveInvalid
cover_objective_valid = 0
oracle_valid_candidate_count = 0
random_permuted_control_task_pass_count = 0
cover_preserving_substrate_pass_count = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
compute_budgeted_run = 1
```

validity rows：

| method | rows | task pass | control pass | median source | max source | median cover | median AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0-AdamWControl | 14 | 0 | 0 | -0.002191375009715557 | 0.0 | 0.0 | 1.002761960029602 |
| V5-RandomCoverControl | 14 | 0 | 0 | -0.17019987106323242 | 0.0 | 0.5056086778640747 | 1.2395715713500977 |
| V6-PermutedOracleControl | 14 | 0 | 0 | -0.14654985070228577 | 0.0 | 0.9624462127685547 | 1.1125260591506958 |
| V8-TrainStreamSoftValueCoverProxy | 14 | 0 | 0 | -0.24296048283576965 | -0.02572410368519984 | 0.9082890152931213 | 1.2377440929412842 |
| V9-TrainStreamResidualSafeValueCoverProxy | 14 | 0 | 0 | -0.2151571810245514 | -0.019695951689679747 | 0.9513776302337646 | 1.1870806217193604 |

artifact:

```text
results/v13_11_cover_objective_validation_cover_preserving_substrate/value_proxy_v8v9_v1311/
```

进程复核：

```bash
ps -ef | rg 'run_v1311|analyze_v1311|run_v1310|run_v139|run_v136' | rg -v 'rg|pgrep' || true
```

结果：

```text
无输出；没有实验进程残留。
```
