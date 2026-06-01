# DG-KAN v14.4 RealTransferFMS AllBasisSubstrate 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志只记录实际执行过的命令、修改文件和 artifact 路径；不把 smoke / compute-budgeted repair / S4b 写成 S5 或 promotion。

## 1. 计划读取

阅读计划：

```bash
sed -n '1,260p' docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md
sed -n '260,620p' docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md
sed -n '620,980p' docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md
sed -n '980,1160p' docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md
```

复核 v14.3 runner：

```bash
sed -n '1,240p' experiments/run_v143_real_short_run_gate.py
sed -n '240,620p' experiments/run_v143_real_short_run_gate.py
sed -n '1,260p' experiments/run_v143_functional_value_constraint_all_basis_substrate.py
```

计划关键 gate：

```text
S5 requires real 3x3 = 9/9 dataset-seed pass。
6/9 <= pass < 9/9 只能记为 S4b-RealTransferExplorationPositive。
LineC / CEp99 / NLL / ECE / Brier 只能 audit/gate，不能做 direction。
```

## 2. 代码修改

新增：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

实现内容：

```text
1. v14.4 real-transfer runner 与 v144 artifact surface。
2. K-RT1-TrainSplitAgreement。
3. K-RT2-TrainStreamTailTrust。
4. K-RT3-ProjectionValueRetention。
5. K-RT4-SourceTailCoState。
6. K-RT5-DelayedBasisConstraint。
7. MLP generic FMS controls。
8. v14.3 Non-RAT/Wavelet substrate artifact consolidation。
9. v144 required manifest / forbidden audit / code review manifest / figures。
```

合法性字段：

```text
direction_uses_validation_test_future_query = 0
direction_uses_linec_cep99_nll_ece = 0
uses_dataset_name_branch = 0
promotion_allowed = 0 unless true S5 non-compute-budgeted
```

## 3. 语法检查

第一次：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

结果：

```text
py_compile pass
```

## 4. Smoke 与实现 blocker

第一次 smoke：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/smoke_v144 \
  --datasets MNIST \
  --seeds 0 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,KCTRL-RandomMatchedProjection \
  --mlp-methods MLP-AdamW,MLP-FMS-SplitAgreement \
  --train-size 128 --val-size 64 --test-size 64 \
  --train-steps 4 --batch-size 8 \
  --fms-update-interval 2 --trace-interval 2 \
  --linec-mode exact --linec-batch-size 8 --linec-sketch-dim 4 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

失败：

```text
NameError: name 'eval_metrics' is not defined
```

修复：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  from experiments.run_v133_task_family_robust_basis_natural import eval_metrics
```

修复后语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

结果：

```text
py_compile pass
```

重跑 smoke：

```bash
rm -rf results/v14_4_real_transfer_fms_all_basis_substrate/smoke_v144
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/smoke_v144 \
  --datasets MNIST \
  --seeds 0 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,KCTRL-RandomMatchedProjection \
  --mlp-methods MLP-AdamW,MLP-FMS-SplitAgreement \
  --train-size 128 --val-size 64 --test-size 64 \
  --train-steps 4 --batch-size 8 \
  --fms-update-interval 2 --trace-interval 2 \
  --linec-mode exact --linec-batch-size 8 --linec-sketch-dim 4 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
expected_dataset_seed_count = 1
real_dataset_seed_pass_count = 0
real_short_run_pass_rows = 0
mean_source_vs_best_control_noncontrol = -1.1983697414398193
required_artifact_missing_count = 0
promotion_allowed = 0
```

解释：

```text
smoke 只证明 runner / artifact / K-RT surface 可执行；
不能作为 v14.4 no-go 或 success。
```

## 5. Official compute-budgeted v14.4

执行：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/official_v144 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT4-SourceTailCoState,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --mlp-methods MLP-AdamW,MLP-FMS-Amortized,MLP-FMS-SplitAgreement,MLP-FMS-SourceTailCoState \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.10 --fms-update-interval 80 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
expected_dataset_seed_count = 9
real_dataset_seed_pass_count = 3
real_short_run_pass_rows = 7
mean_source_vs_best_control_noncontrol = -0.09544556670718723
required_artifact_missing_count = 0
promotion_allowed = 0
```

失败分解：

```text
LineC 不是主要 blocker；多数 K-RT rows LineC majority pass。
主要 blocker:
  AUCtime = 36
  source = 31
  CEp99_tail = 26
  NLL_tail = 21
  ECE_tail = 10
  LineC = 1
```

## 6. Repair 1：low plasticity / lower lambda

触发原因：

```text
official 仍低于 v14.3 best 4/9；
source / AUC / tail 同时不稳，先排除 FMS plasticity 过强。
```

执行：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
expected_dataset_seed_count = 9
real_dataset_seed_pass_count = 6
real_short_run_pass_rows = 9
mean_source_vs_best_control_noncontrol = -0.03511386281914181
required_artifact_missing_count = 0
promotion_allowed = 0
```

## 7. Repair 2：strong tail trust

触发原因：

```text
low-plasticity 达到 6/9 但未达 S5；
剩余 blocker 主要是 AUC / CEp99 tail。
按计划尝试 train-stream tail trust，不使用 CEp99/NLL/ECE audit 作为 direction。
```

执行：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_tailtrust_strong \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-RT4-SourceTailCoState,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.35 \
  --rt-state-beta 0.80 --rt-risk-scale 8.0 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 5
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = -0.05162593832722417
promotion_allowed = 0
```

判断：

```text
更强 tail trust / lower lambda 牺牲了 source，未打开 S5。
```

## 8. Repair 3：moderate split / delayed

执行：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_split_delayed_moderate \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 40 --rt-lambda-max 0.65 \
  --rt-agreement-a0 -0.20 --rt-agreement-a1 0.40 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 5
real_short_run_pass_rows = 8
mean_source_vs_best_control_noncontrol = -0.06597101467627066
promotion_allowed = 0
```

## 9. Repair 4：slow refresh

触发原因：

```text
排除 FMS refresh 太频繁导致 real-transfer 波动。
```

执行：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_slowrefresh160_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 160 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = S4b-RealTransferExplorationPositive
real_dataset_seed_pass_count = 6
real_short_run_pass_rows = 9
mean_source_vs_best_control_noncontrol = -0.03129612737231784
promotion_allowed = 0
```

## 10. 结果读取命令

汇总 route：

```bash
python - <<'PY'
import csv, json, pathlib
base=pathlib.Path('results/v14_4_real_transfer_fms_all_basis_substrate')
for run in ['smoke_v144','official_v144','repair_v144_lowplasticity_lambda05','repair_v144_tailtrust_strong','repair_v144_split_delayed_moderate','repair_v144_slowrefresh160_lambda05']:
    p=base/run/'v144_route_decision.json'
    if p.exists():
        r=json.loads(p.read_text())
        print(run, r['route'], r['minimum_success'], r['real_dataset_seed_pass_count'], r['real_short_run_pass_rows'], r['mean_source_vs_best_control_noncontrol'], r['required_artifact_missing_count'], r['promotion_allowed'])
PY
```

best repair summary：

```bash
python - <<'PY'
import csv, pathlib
out=pathlib.Path('results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05')
for r in csv.DictReader((out/'v144_real_transfer_fms_summary.csv').open()):
    print(r['method'], r['dataset_seed_pass_count'], r['pass_rows'], r['mean_source_vs_best_control'], r['median_auc_time'], r['linec_majority_pass_rows'])
PY
```

## 11. Artifact 位置

```text
results/v14_4_real_transfer_fms_all_basis_substrate/smoke_v144/
results/v14_4_real_transfer_fms_all_basis_substrate/official_v144/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_tailtrust_strong/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_split_delayed_moderate/
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_slowrefresh160_lambda05/
```

每个 run 均输出：

```text
v144_route_decision.json
v144_required_manifest.csv
v144_forbidden_information_audit.csv
v144_code_review_manifest.csv
v144_real_transfer_fms_results.csv
v144_real_transfer_fms_summary.csv
v144_train_stream_proxy.csv
v144_projection_value_retention.csv
v144_source_tail_costate.csv
v144_split_agreement.csv
v144_real_3x3_failure_table.csv
v144_linec_audit.csv
v144_tail_calibration_audit.csv
fig_*.svg
```

## 12. 最终执行边界

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

当前已按计划尝试：

```text
K-RT1 split agreement
K-RT2 train-stream tail trust
K-RT3 projection value retention
K-RT4 source-tail co-state
K-RT5 delayed basis constraint
low plasticity / lower lambda
stronger train-stream tail trust
moderate split/delayed
slow FMS refresh
```

停止原因：

```text
我现在不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不把 6/9 exploration positive 写成 S5 official success。
```

## 13. 结果复盘日志写入

写入文件：

```text
docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_实验结果复盘.md
```

记录内容：

```text
1. v14.4 计划理解与硬约束。
2. 本轮 runner / K-RT methods / artifact surface 修改。
3. smoke import blocker 与修复结果。
4. official compute-budgeted 3/9 结果与 failure decomposition。
5. low plasticity / lower lambda repair 达到 S4b 6/9 的 best 结果。
6. stronger tail trust、moderate split/delayed、slow refresh repair 结果。
7. artifact 列表与最终 no-go boundary。
```

最终核对指令：

```bash
rg -n "best route|real_dataset_seed_pass_count|official_s5_reached|promotion_allowed|S4b|S5" \
  docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_实验结果复盘.md \
  docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_执行日志.md

jq '{route, minimum_success, official_s5_reached, real_dataset_seed_pass_count, expected_dataset_seed_count, real_short_run_pass_rows, mean_source_vs_best_control_noncontrol, required_artifact_missing_count, promotion_allowed}' \
  results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lambda05/v144_route_decision.json \
  results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_slowrefresh160_lambda05/v144_route_decision.json
```

## 14. 用户再次追问后的继续修复

用户再次要求若未达成则继续。本次重新对照计划 failure-response rules：

```text
If source improves but tail fails:
  Try train-stream tail trust / risk state.
If tail improves but source fails:
  Try projection value retention / split agreement.
If LineC fails:
  Audit role/basis movement; try delayed basis constraint or weaker constraint phase.
Do not use CEp99 / LineC / validation as direction.
Do not use seed-specific scaling.
```

已知 best lowplasticity 剩余失败：

```text
Fashion-MNIST seed0:
  K-RT3 source/AUC/NLL/ECE/LineC 可过，但 CEp99_delta = +0.166046。
Fashion-MNIST seed2:
  K-RT2 source/tail/LineC 可过，但 AUCtime = 1.232769。
KMNIST seed2:
  K-RT1 source/NLL/ECE/LineC 可过，但 AUCtime = 1.110939 且 CEp99_delta = +2.585526。
```

### 14.1 lowplasticity + train-stream output geometry

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_outputgeom_entropy \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --output-geometry-repair train_entropy_t080_100_else050 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 0 / 9
real_short_run_pass_rows = 0
mean_source_vs_best_control_noncontrol = -0.0064664847320980495
required_artifact_missing_count = 0
promotion_allowed = 0
```

failure counts：

```text
LineC = 32
source = 27
AUCtime = 22
ECE_tail = 14
CEp99_tail = 9
NLL_tail = 4
```

判断：

```text
train-stream entropy output geometry 在 v14.4 lowplasticity 组合中严重退化到 0/9；
它没有解决 S5，反而引入大量 LineC / source failure。
```

### 14.2 ultralow plasticity

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_ultralowplasticity_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.025 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 5 / 9
real_short_run_pass_rows = 8
mean_source_vs_best_control_noncontrol = -0.03621196746826172
required_artifact_missing_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.09185847308900622 | 1.0701243092652721 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.010098596413930258 | 1.0682185684632735 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021993809276156955 | 1.0507222223219834 | 8 |
| K-RT5-DelayedBasisConstraint | 3 | 3 | -0.08508180247412787 | 1.0947689743379645 | 9 |

failure counts：

```text
AUCtime = 22
CEp99_tail = 20
source = 14
NLL_tail = 10
ECE_tail = 4
LineC = 1
```

判断：

```text
ultralow plasticity 从 best 6/9 退化到 5/9；
继续简单降低 FMS strength 会压低 source，不能打开 S5。
```

### 14.3 更新后的最终执行边界

```text
best route 仍为 S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count 仍为 6 / 9
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

### 14.4 K-RT6 ProjectionTailTrust 修复补录

用户再次要求未达成则继续。本次按计划 failure-response 继续组合两个仍合法的 train-stream-only 机制：

```text
K-RT6-ProjectionTailTrust =
  K-RT3 projection value retention
  +
  K-RT2 train-stream tail/risk lambda
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-RT6-ProjectionTailTrust。
  rt_method_base 将 K-RT6 映射到 K8 Rational base。
  adaptive_projection_scales 将 K-RT6 走 K-RT3 value-retention projection。
  lambda_from_method 将 K-RT6 走 K-RT2 train-stream risk lambda。
```

合法性：

```text
K-RT6 direction 只使用 train-stream loss_q95 / margin_p10 / logit_rms、
FMS value vector 与 projection retention。
不使用 validation/test/future/query。
不使用 CEp99 / NLL / ECE / LineC / AUCtime 生成方向。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

结果：

```text
py_compile pass
```

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_krt6_projection_tailtrust_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,K-RT6-ProjectionTailTrust,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

route：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 11
mean_source_vs_best_control_noncontrol = -0.04666680892308553
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |
| K-RT6-ProjectionTailTrust | 2 | 2 | -0.09287859333886041 | 1.083393981663408 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.1188468403286404 | 1.0240534785437074 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.04685921139187283 | 1.0 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 1 | K-RT6 | 0.223744 | 0.908376 | -2.440582 | -0.237007 | -0.025605 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| MNIST | 2 | K-RT6 | 0.058370 | 0.932749 | -0.313580 | -0.125301 | -0.024170 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining best row by dataset-seed：

| dataset | seed | best method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | K-RT3 | 0 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 |
| Fashion-MNIST | 2 | K-RT2 | 0 | 0.162050 | 1.232769 | -0.704851 | -0.162050 | -0.007385 | 1 |
| KMNIST | 2 | K-RT1 | 0 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 |

K-RT6 row-level blocker：

| dataset | seed | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MNIST | 0 | 0 | -0.123465 | 1.104864 | -2.497648 | 0.012005 | -0.002448 | 1 | source,AUCtime |
| MNIST | 1 | 1 | 0.223744 | 0.908376 | -2.440582 | -0.237007 | -0.025605 | 1 | - |
| MNIST | 2 | 1 | 0.058370 | 0.932749 | -0.313580 | -0.125301 | -0.024170 | 1 | - |
| Fashion-MNIST | 0 | 0 | -0.167308 | 1.071968 | -0.327671 | -0.121784 | -0.032406 | 1 | source,AUCtime |
| Fashion-MNIST | 1 | 0 | -0.194636 | 1.102639 | 3.312584 | 0.174764 | 0.014478 | 1 | source,AUCtime,CEp99,NLL |
| Fashion-MNIST | 2 | 0 | -0.203331 | 1.111799 | 0.893652 | 0.203331 | 0.017452 | 1 | source,AUCtime,CEp99,NLL |
| KMNIST | 0 | 0 | 0.055100 | 1.025365 | -3.250263 | -0.055100 | 0.018707 | 1 | AUCtime |
| KMNIST | 1 | 0 | -0.342099 | 1.083394 | 0.090408 | -0.149531 | -0.030250 | 1 | source,AUCtime,CEp99 |
| KMNIST | 2 | 0 | -0.142282 | 1.243287 | 2.074610 | 0.064909 | 0.022072 | 1 | source,AUCtime,CEp99,NLL,ECE |

failure counts：

```text
AUCtime = 29
CEp99_tail = 22
source = 20
NLL_tail = 12
ECE_tail = 5
LineC = 0
```

判断：

```text
1. K-RT6 使 pass rows 增至 11，但 unique dataset-seed 仍为 6/9。
2. K-RT6 新增的 pass 只在 MNIST seed1/seed2；没有补上缺失的 Fashion-MNIST seed0/2 或 KMNIST seed2。
3. LineC blocker 在本轮为 0，说明 exact LineC 不是当前主 blocker。
4. 剩余主 blocker 是 AUCtime、CEp99 tail 与 source instability 的共同失败。
5. 不能把 S4b 写成 S5，也不能 promotion。
```

本次后的最终边界：

```text
best route 仍为 S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count 仍为 6 / 9
latest pass rows = 11
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

额外覆盖的合法 repair：

```text
train-stream entropy output geometry
ultralow FMS plasticity
```

停止原因更新：

```text
计划内 K-RT1..K-RT5、tail trust、projection retention、split agreement、
delayed basis、low/ultralow plasticity、slow refresh、train-stream output geometry 均已尝试。
我现在仍不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不把 CEp99/NLL/ECE/LineC audit metric 用作 direction、
不做 dataset-name branch、
不使用 seed-specific scaling、
不把 6/9 exploration positive 写成 S5 official success。
```

## 15. 用户再次追问后的计划覆盖复核

本次未新增训练，只复核 v14.4 完整计划中是否仍有未覆盖的合法推荐线。

执行指令：

```bash
sed -n '405,660p' docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md
sed -n '660,985p' docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md

jq -r '[.route,.minimum_success,.official_s5_reached,.real_dataset_seed_pass_count,.expected_dataset_seed_count,.real_short_run_pass_rows,.promotion_allowed] | @tsv' \
  results/v14_4_real_transfer_fms_all_basis_substrate/*/v144_route_decision.json | sort
```

复核结论：

```text
Line K:
  K-RT1..K-RT5 已全部实现并运行。
  source/tail/LineC/overhead failure-response 已覆盖：
    train-stream tail trust / risk state
    projection value retention
    split agreement
    delayed basis / weaker constraint phase
    low / ultralow plasticity
    slow refresh
    train-stream output geometry

Line W / Line D:
  计划明确 Non-RAT remains fail 时不要阻塞 Rational S5；
  保持 bounded parallel。
  v14.3 已完成 Wavelet/RBF/CHE/FOU/WAV 多轮 substrate repair；
  v14.4 runner 也输出 v144_wavelet_substrate_hardening、
  v144_rbf_substrate_repair、v144_chebyshev_lifetime_repair、
  v144_fourier_lifetime_repair、v144_all_basis_substrate_status。

Line G:
  official_v144 已运行 MLP generic controls；
  后续 repair 为节省预算使用 --skip-mlp-control，但不把 generic MLP 写成 KAN success。

Line C/R/Z:
  LineC / tail / calibration / route / manifest / forbidden audit 均已输出。
```

最新 route 汇总：

```text
best route = S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count = 6 / 9
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

最终停止依据：

```text
当前 v14.4 完整计划中的推荐修复方向已经覆盖；
新增 output geometry 与 ultralow plasticity 均退化；
继续推进需要新的 real-transfer mechanism 计划。

我现在不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不违反：
  no validation/test/future/query direction
  no CEp99/NLL/ECE/LineC direction
  no dataset-name branch
  no seed-specific scaling
  no 6/9 -> S5 造假
```

## 16. 用户再次追问后的 low-lr repair

用户再次要求继续。本次选择一个仍合法、非 dataset-specific、非 audit-direction 的窄修复：

```text
在 best lowplasticity 配置上进一步降低 lr 到 0.003，
检查是否能减少 CEp99/AUCtime 过冲，同时保留 source。
```

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_lr0003_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.003 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = R2-RealTransferFail
minimum_success = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 5 / 9
real_short_run_pass_rows = 5
mean_source_vs_best_control_noncontrol = 0.04683350192175971
required_artifact_missing_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 1 | 1 | 0.02774900197982788 | 1.0533332299728817 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.07578038507037693 | 1.0459694909969366 | 8 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.06553273068534003 | 1.0473440391487672 | 9 |
| K-RT5-DelayedBasisConstraint | 2 | 2 | 0.018271889951494005 | 1.031561781810233 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-RT1 | 0.065979 | 0.968803 | -1.047808 | -0.077472 | -0.015550 | 1 |
| Fashion-MNIST | 0 | K-RT3 | 0.284883 | 0.927610 | -0.864432 | -0.284883 | -0.044932 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.202265 | 0.934653 | -1.107346 | -0.202265 | -0.013246 | 1 |
| KMNIST | 0 | K-RT2 | 0.137554 | 0.976507 | -2.002703 | -0.137554 | -0.016042 | 1 |
| KMNIST | 1 | K-RT5 | 0.026668 | 0.968709 | -0.422087 | -0.111281 | -0.018716 | 1 |

failure counts：

```text
AUCtime = 24
CEp99_tail = 22
source = 10
NLL_tail = 3
ECE_tail = 2
LineC = 1
```

判断：

```text
lr=0.003 让 mean source 转正，但 unique dataset-seed pass 从 best 6/9 退到 5/9。
主要 blocker 仍是 AUCtime 与 CEp99 tail。
单纯降低 lr 不是打开 S5 的充分机制。
```

更新后的最终边界：

```text
best route 仍为 S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count 仍为 6 / 9
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

## 17. 用户再次追问后的 single-refresh 修复探针

目的：

```text
best lowplasticity / slowrefresh160 仍卡在 6/9；
主要失败项包含 AUCtime 与 CEp99 tail。
本次测试极低刷新频率 fms_update_interval=200，
让 200-step window 内只发生 single refresh，
检查是否能减少 direction churn 与 AUCtime / tail 过冲。
```

执行边界：

```text
1. 使用全 3x3：MNIST / Fashion-MNIST / KMNIST x seeds 0,1,2。
2. 使用统一参数，不做 dataset-name branch。
3. direction 只使用 train-stream loss/FMS/proxy/basis-role state。
4. LineC / CEp99 / NLL / ECE 只作为 audit/gate。
5. 不使用 validation/test/future/query batch 生成方向。
```

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_single_refresh200_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 200 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 9
mean_source_vs_best_control_noncontrol = -0.03191156850920783
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.08869858582814534 | 1.0727013843627924 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02159310711754693 | 1.0695566280761937 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.02195682790544298 | 1.050712801731164 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08249762323167589 | 1.0954464050942987 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.108119 | 0.991314 | -1.533506 | -0.219537 | -0.010946 | 1 |
| MNIST | 1 | K-RT1 | 0.223926 | 0.883772 | -6.946172 | -0.237098 | -0.017712 | 1 |
| MNIST | 1 | K-RT3 | 0.226802 | 0.898331 | -4.681934 | -0.239974 | -0.019997 | 1 |
| MNIST | 1 | K-RT5 | 0.182495 | 0.976379 | -5.419912 | -0.195667 | -0.028670 | 1 |
| MNIST | 2 | K-RT1 | 0.230200 | 0.943766 | -0.952721 | -0.297060 | -0.040520 | 1 |
| MNIST | 2 | K-RT5 | 0.057710 | 0.918460 | -0.084510 | -0.124570 | -0.018149 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027604 | 0.908469 | -2.434698 | -0.047252 | 0.000261 | 1 |
| KMNIST | 0 | K-RT2 | 0.051920 | 0.989902 | -6.277149 | -0.051920 | -0.002072 | 1 |
| KMNIST | 1 | K-RT5 | 0.027316 | 0.965192 | -0.075369 | -0.518898 | -0.045067 | 1 |

failure counts：

```text
AUCtime = 22
CEp99_tail = 18
source = 14
NLL_tail = 9
ECE_tail = 4
LineC = 0
```

判断：

```text
1. single-refresh200 与 best lowplasticity / slowrefresh160 一样只到 6/9。
2. LineC blocker 在本 run 中为 0，但 AUCtime / CEp99 / source 仍共同阻断 S5。
3. fms_update_interval=200 没有打开新的 dataset-seed coverage。
4. 不能把 S4b exploration positive 写成 S5 official success。
```

本次后的最终边界：

```text
best route 仍为 S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count 仍为 6 / 9
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

## 18. 用户再次追问后的 lowplasticity + K8/KRT4 覆盖修复

触发原因：

```text
前一轮 single-refresh200 仍为 6/9；
计划中 K8 baseline 与 K-RT4 SourceTailCoState 已在 official_v144 中执行，
但尚未在 best lowplasticity / lambda05 配置下与 K-RT1/2/3/5 同 run 覆盖。
本次不拼接不同 run 的局部 pass，而是统一全 3x3 重跑。
```

执行边界：

```text
1. 全 3x3：MNIST / Fashion-MNIST / KMNIST x seeds 0,1,2。
2. 不做 dataset-name branch。
3. 不使用 seed-specific scaling。
4. direction 只使用 train-stream loss/FMS/proxy/basis-role state。
5. LineC / CEp99 / NLL / ECE / AUC 只作为 audit/gate。
```

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_lowplasticity_plus_k8_krt4_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT4-SourceTailCoState,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

结果：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 10
mean_source_vs_best_control_noncontrol = -0.06702906445220665
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint | 1 | 1 | -0.1262306637234158 | 0.9864509986186465 | 8 |
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT4-SourceTailCoState | 0 | 0 | -0.13548827171325684 | 1.0664973035841765 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K8 | 0.087914 | 0.975628 | -5.826309 | -0.101176 | 0.002058 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining best row by dataset-seed：

| dataset | seed | best method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | K-RT3 | 0 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 |
| Fashion-MNIST | 2 | K-RT2 | 0 | 0.162050 | 1.232769 | -0.704851 | -0.162050 | -0.007385 | 1 |
| KMNIST | 2 | K-RT1 | 0 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 |

failure counts：

```text
source = 27
AUCtime = 34
CEp99_tail = 26
NLL_tail = 16
ECE_tail = 7
LineC = 1
```

判断：

```text
1. 加回 K8 与 K-RT4 后 pass rows 从 9 增加到 10，
   但 unique dataset-seed pass 仍是 6/9。
2. K8 只增加 MNIST seed1 的额外 pass，没有补上缺失的 Fashion-MNIST seed0/2 或 KMNIST seed2。
3. K-RT4 在 lowplasticity 配置下仍没有 dataset-seed pass。
4. 剩余 blocker 仍是 AUCtime 与 CEp99 tail；LineC 已不是主要 blocker。
5. 不能把 S4b 写成 S5，也不能 promotion。
```

本次后的最终边界：

```text
best route 仍为 S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count 仍为 6 / 9
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```

## 19. 当前最终执行边界（K-RT6 后）

K-RT6 的完整运行指令与结果已补录在 section 14.4；当前最新 artifact 为：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_krt6_projection_tailtrust_lambda05/
```

最新 route：

```text
route = S4b-RealTransferExplorationPositive
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 11
official_s5_reached = 0
promotion_allowed = 0
```

最终停止依据：

```text
K-RT6 增加 pass rows 到 11，但仍未补上 Fashion-MNIST seed0/2 与 KMNIST seed2。
当前 blocker 是 AUCtime、CEp99 tail 与 source instability 的共同失败。
我现在仍不确定如何在当前 v14.4 runner 内继续安全推进到 9/9 S5，
而不使用 audit metric 作 direction、不做 dataset-name branch、
不使用 seed-specific scaling、不拼接不同 run 的局部 positive。
```

## 20. 用户再次追问后的 K-RT7 LateProjectionTailTrust 修复

用户再次要求未达成则继续。本次继续尝试一个不使用 audit metric 的低 AUC-risk 机制：

```text
K-RT7-LateProjectionTailTrust =
  late FMS plasticity ramp
  +
  K-RT6 projection value retention
  +
  train-stream tail/risk lambda
```

设计目的：

```text
1. 前半程尽量保留 AdamW 学习曲线，减少 AUC_NLL 早期损伤。
2. 后半程逐步打开 FMS plasticity，尝试保留 final source。
3. ramp 只使用 train step phase；risk lambda 只使用 train-stream loss_q95 / margin_p10 / logit_rms。
4. 不使用 CEp99 / NLL / ECE / LineC / AUCtime 作为方向。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-RT7-LateProjectionTailTrust。
  新增 train_step_phase 到 train-stream proxy。
  K-RT7 使用 K-RT3/K-RT6 的 projection value retention。
  K-RT7 的 lambda = late_phase_ramp * train_stream_risk_trust。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

结果：

```text
py_compile pass
```

执行指令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v144_krt7_late_projection_tailtrust_lambda05 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT5-DelayedBasisConstraint,K-RT7-LateProjectionTailTrust,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-size 1024 --val-size 512 --test-size 512 \
  --train-steps 200 --batch-size 32 \
  --lr 0.005 --fms-strength 0.05 --fms-update-interval 80 --rt-lambda-max 0.5 \
  --trace-interval 100 --linec-mode exact --linec-batch-size 24 --linec-sketch-dim 8 \
  --compute-budgeted-run 1 \
  --device cuda:0
```

route：

```text
route = S4b-RealTransferExplorationPositive
minimum_success = S4b-RealTransferExplorationPositive
official_s5_reached = 0
real_dataset_seed_pass_count = 6 / 9
real_short_run_pass_rows = 10
mean_source_vs_best_control_noncontrol = -0.043665132257673475
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

| method | dataset-seed pass | pass rows | mean source | median AUCtime | LineC pass rows |
|---|---:|---:|---:|---:|---:|
| K-RT1-TrainSplitAgreement | 3 | 3 | -0.10150694184833103 | 1.0673764713627447 | 9 |
| K-RT2-TrainStreamTailTrust | 1 | 1 | 0.02155327796936035 | 1.0695467503638163 | 9 |
| K-RT3-ProjectionValueRetention | 1 | 1 | 0.021980126698811848 | 1.050731870366617 | 9 |
| K-RT5-DelayedBasisConstraint | 4 | 4 | -0.08248191409640843 | 1.095324834920501 | 9 |
| K-RT7-LateProjectionTailTrust | 1 | 1 | -0.07787021001180013 | 1.0802075331995027 | 9 |
| K0-RAT-AdamW | 0 | 0 | -0.1188468403286404 | 1.0240534785437074 | 9 |
| KCTRL-RandomMatchedProjection | 0 | 0 | -0.04685921139187283 | 1.0 | 9 |

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-RT1 | 0.113161 | 0.989009 | -1.502634 | -0.224621 | -0.014783 | 1 |
| MNIST | 1 | K-RT1 | 0.225544 | 0.883245 | -6.995262 | -0.238807 | -0.021089 | 1 |
| MNIST | 1 | K-RT3 | 0.226790 | 0.898311 | -4.681700 | -0.240052 | -0.020003 | 1 |
| MNIST | 1 | K-RT5 | 0.182434 | 0.976370 | -5.420082 | -0.195696 | -0.028673 | 1 |
| MNIST | 1 | K-RT7 | 0.075135 | 0.992399 | -2.819923 | -0.088398 | -0.010688 | 1 |
| MNIST | 2 | K-RT1 | 0.231780 | 0.943324 | -1.094994 | -0.298710 | -0.042674 | 1 |
| MNIST | 2 | K-RT5 | 0.057659 | 0.918473 | -0.084893 | -0.124589 | -0.018150 | 1 |
| Fashion-MNIST | 1 | K-RT5 | 0.027344 | 0.908514 | -2.434628 | -0.047216 | 0.000262 | 1 |
| KMNIST | 0 | K-RT2 | 0.051924 | 0.989902 | -6.277489 | -0.051924 | -0.002071 | 1 |
| KMNIST | 1 | K-RT5 | 0.027283 | 0.965196 | -0.074760 | -0.518914 | -0.045063 | 1 |

remaining best row by dataset-seed：

| dataset | seed | best method | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | 0 | K-RT3 | 0 | 0.184902 | 0.873954 | 0.166046 | -0.473994 | -0.046205 | 1 |
| Fashion-MNIST | 2 | K-RT7 | 0 | 0.233003 | 1.228868 | -0.889408 | -0.233003 | -0.032549 | 1 |
| KMNIST | 2 | K-RT1 | 0 | 0.094257 | 1.110939 | 2.585526 | -0.171631 | -0.016695 | 1 |

K-RT7 row-level：

| dataset | seed | pass | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MNIST | 0 | 0 | 0.008138 | 1.080208 | -3.253248 | -0.119598 | 0.003864 | 1 | AUCtime |
| MNIST | 1 | 1 | 0.075135 | 0.992399 | -2.819923 | -0.088398 | -0.010688 | 1 | - |
| MNIST | 2 | 0 | 0.042569 | 0.939440 | 0.454446 | -0.109499 | -0.029132 | 1 | CEp99 |
| Fashion-MNIST | 0 | 0 | -0.016016 | 1.125437 | 0.857180 | -0.273077 | -0.048155 | 1 | source,AUCtime,CEp99 |
| Fashion-MNIST | 1 | 0 | -0.682969 | 1.141081 | 2.715660 | 0.663097 | 0.055254 | 1 | source,AUCtime,CEp99,NLL,ECE |
| Fashion-MNIST | 2 | 0 | 0.233003 | 1.228868 | -0.889408 | -0.233003 | -0.032549 | 1 | AUCtime |
| KMNIST | 0 | 0 | -0.325062 | 1.057435 | -3.872623 | 0.325062 | 0.030242 | 1 | source,AUCtime,NLL,ECE |
| KMNIST | 1 | 0 | 0.070650 | 1.043607 | -2.214352 | -0.562281 | -0.055730 | 1 | AUCtime |
| KMNIST | 2 | 0 | -0.106281 | 1.104521 | 0.881664 | 0.028907 | -0.003424 | 1 | source,AUCtime,CEp99,NLL |

failure counts：

```text
AUCtime = 29
CEp99_tail = 22
source = 18
NLL_tail = 12
ECE_tail = 6
LineC = 0
```

判断：

```text
1. K-RT7 没有打开 S5，unique dataset-seed 仍为 6/9。
2. K-RT7 只新增/保留 MNIST seed1 的自身 pass；整体 pass rows 低于 K-RT6。
3. Fashion-MNIST seed2 的 source/tail 在 K-RT7 下更强，但 AUCtime 仍为 1.228868，不能过 gate。
4. Fashion-MNIST seed0 仍卡在 CEp99 tail；KMNIST seed2 仍卡在 AUCtime + CEp99/source。
5. late ramp 没有解决 AUCtime blocker。
6. 不能写成 S5，也不能 promotion。
```

本次后的最终边界：

```text
best route 仍为 S4b-RealTransferExplorationPositive
best real_dataset_seed_pass_count 仍为 6 / 9
best pass rows 仍为 11（来自 K-RT6 run）
latest K-RT7 pass rows = 10
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
```
