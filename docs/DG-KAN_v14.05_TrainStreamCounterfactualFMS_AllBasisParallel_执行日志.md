# DG-KAN v14.5 TrainStreamCounterfactualFMS AllBasisParallel 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志只记录实际执行过的命令、修改过的文件和产物路径；不补写未执行训练。

## 1. 计划阅读

读取计划文件：

```bash
sed -n '1,220p' docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_完整计划.md
sed -n '220,520p' docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_完整计划.md
sed -n '520,920p' docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_完整计划.md
sed -n '920,1320p' docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_完整计划.md
```

关键执行顺序：

```text
1. 先执行 Line O action-bank oracle decomposition。
2. 如果现有合法 action bank 的 oracle upper bound < 9/9，
   route = R2-ActionBankUpperBoundInsufficient。
3. R2 时不能继续调 controller policy；
   必须记录缺失 state classes，并提出新 action family。
```

## 2. 代码修改

新增文件：

```text
experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

实现内容：

```text
1. 读取 v14.4 已执行的合法 real-transfer artifacts。
2. 构造 v14.5 core action bank：
   A0 NoOp/K0
   A1 K8 lowplasticity
   A2 K-RT1
   A3 K-RT2
   A4 K-RT3
   A5 K-RT4
   A6 K-RT5
   A7 K-RT6
   A8 K-RT7
   A9 single-refresh200
   A10 slowrefresh160
3. 额外构造 extended legal-v144 diagnostic bank：
   纳入所有已存在 v14.4 非 smoke、非 control real-transfer rows。
4. 输出 O1 run-level oracle artifact。
5. 若 oracle upper bound < 9/9，则不执行 controller，并输出空 event/controller artifact 表头。
6. 输出 required manifest、forbidden audit、all-basis/MLP copy artifacts、figures、no-go 和 next queue。
```

## 3. 语法检查

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

结果：

```text
pass
```

## 4. v14.5 Line O Oracle Run

命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145 \
  --mode oracle \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

stdout route 摘要：

```json
{
  "route": "R2-ActionBankUpperBoundInsufficient",
  "minimum_success": "S4b-RealTransferExplorationPositive",
  "official_s5_reached": 0,
  "promotion_allowed": 0,
  "core_oracle_dataset_seed_pass_count": 6,
  "extended_legal_v144_oracle_dataset_seed_pass_count": 7,
  "best_oracle_dataset_seed_pass_count": 7,
  "best_oracle_level": "O1-extended-legal-v144-bank",
  "controller_executed": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0
}
```

## 5. 产物

主目录：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/
```

关键文件：

```text
v145_route_decision.json
v145_required_manifest.csv
v145_forbidden_information_audit.csv
v145_code_review_manifest.csv
v145_action_bank_rows.csv
v145_action_bank_oracle.csv
v145_oracle_summary.csv
v145_counterfactual_event_log.csv
v145_controller_real_results.csv
v145_controller_summary.csv
v145_controller_failure_table.csv
v145_wavelet_substrate_hardening.csv
v145_all_basis_status.csv
v145_mlp_generic_control.csv
v145_linec_tail_auc_audit.csv
v145_missing_state_classes.csv
v145_no_go_boundary.md
v145_next_hypothesis_queue.md
figures/*.svg
v145_code_review_packet.zip
```

manifest 检查：

```bash
awk -F, 'NR>1 && $3!=0 {print}' \
  results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_required_manifest.csv
```

结果：

```text
no output; required_artifact_missing_count = 0
```

## 6. 结果读取

命令：

```bash
cat results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_oracle_summary.csv
cat results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145/v145_missing_state_classes.csv
```

读取结果：

```text
O1-core-action-bank:
  dataset_seed_pass_count = 6 / 9
  remaining failures = source 1, AUC 1, CEp99 1

O1-extended-legal-v144-bank:
  dataset_seed_pass_count = 7 / 9
  remaining failures = AUC 2
```

未覆盖 state：

```text
core bank:
  Fashion-MNIST seed0: CEp99-only fail
  Fashion-MNIST seed2: AUC-only fail
  KMNIST seed2: source fail

extended legal-v144 bank:
  Fashion-MNIST seed2: AUC-only fail
  KMNIST seed2: AUC-only fail
```

## 7. Controller 未执行说明

根据 v14.5 计划第 5.5 与第 13.1：

```text
O1 oracle dataset_seed_pass_count < 9 时，
不能继续调 train-stream controller policy；
必须先判定 action family 不足。
```

因此本次没有执行 K-CF controller 训练或 replay。

对应 artifact 仍已输出表头：

```text
v145_counterfactual_event_log.csv
v145_controller_real_results.csv
v145_controller_summary.csv
v145_controller_failure_table.csv
```

`v145_controller_summary.csv` 记录：

```text
controller_executed = 0
not_executed_reason = oracle_action_bank_upper_bound_lt_9
```

## 8. 用户再次追问后的新 action family 尝试

触发原因：

```text
v14.5 O1 oracle 显示当前 action bank upper bound < 9/9。
计划第 13.1 要求此时不能调 controller policy，
应识别 missing state classes 并提出新 action family。
用户要求继续，因此本次实现并运行一个新的 train-stream-only action family。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-AUC1-SourceTailAUCColocation
  新增 K-AUC2-UltraLateAUCGuard
  lambda_from_method 新增 train_loss_rel_progress_ema 状态
  train-stream proxy artifact 新增 train_loss_rel_progress_ema 字段

experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
  修正 diagnostic oracle = 9/9 时不能直接写成 S5 的 route 逻辑；
  oracle 只用于决定是否进入 controller，不允许 promotion。
```

合法性说明：

```text
1. K-AUC1 / K-AUC2 只使用 train-stream loss mean/q95/margin/logit RMS、
   FMS value state、risk state、train step phase。
2. 不使用 validation/test/future/query batch 生成 direction。
3. 不使用 CEp99 / NLL / ECE / LineC / AUCtime audit metric 生成 direction。
4. 不做 dataset-name branch。
5. 不做 seed-specific scaling。
6. promotion_allowed = 0。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py
conda run -n kan python -m py_compile experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

结果：

```text
pass
```

新 action family 运行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_auc_colocation_action_family \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-AUC1-SourceTailAUCColocation,K-AUC2-UltraLateAUCGuard,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-RealTransferFail",
  "minimum_success": "S4-RealShortRunOpened",
  "official_s5_reached": 0,
  "real_dataset_seed_pass_count": 1,
  "real_short_run_pass_rows": 2,
  "mean_source_vs_best_control_noncontrol": -0.0183285309208764,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

新 action 纳入 v14.5 extended oracle 的命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_auc_action \
  --mode oracle \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-ActionBankUpperBoundInsufficient",
  "minimum_success": "S4b-RealTransferExplorationPositive",
  "official_s5_reached": 0,
  "promotion_allowed": 0,
  "core_oracle_dataset_seed_pass_count": 6,
  "extended_legal_v144_oracle_dataset_seed_pass_count": 7,
  "best_oracle_dataset_seed_pass_count": 7,
  "controller_executed": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0
}
```

新增产物：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_auc_colocation_action_family/
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_auc_action/
```

## 9. 用户再次追问后的 final-pulse action family 尝试

触发原因：

```text
上一轮纳入 K-AUC1 / K-AUC2 后，extended oracle 仍为 7/9。
remaining extended missing states 都是 AUC-only fail：
  Fashion-MNIST seed2
  KMNIST seed2
因此继续尝试一种更窄的 late/final-pulse action family：
  尽量只在训练最后阶段注入 FMS，
  降低对 AUC_NLL trajectory 的影响。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-AUC3-FinalPulseTailTrust。
  新增 K-AUC4-FinalPulseIdentityProjection。
  K-AUC3 使用 projection value retention + final-phase train-stream tail/source gate。
  K-AUC4 使用 identity projection + final-phase train-stream source gate。
```

合法性说明：

```text
1. phase ramp 只使用 train step phase。
2. source / tail gate 只使用 train-stream loss_q95、margin_p10、logit_rms、relative loss progress。
3. 不使用 validation/test/future/query batch 生成 direction。
4. 不使用 AUCtime / CEp99 / NLL / ECE / LineC 生成 direction。
5. 全 3x3 统一运行，不做 dataset-name branch，不做 seed-specific scaling。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

结果：

```text
pass
```

运行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_finalpulse_auc_action_family \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-AUC3-FinalPulseTailTrust,K-AUC4-FinalPulseIdentityProjection,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-RealTransferFail",
  "minimum_success": "S4-RealShortRunOpened",
  "official_s5_reached": 0,
  "real_dataset_seed_pass_count": 3,
  "real_short_run_pass_rows": 5,
  "mean_source_vs_best_control_noncontrol": -0.028244892756144207,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

method summary：

```text
K-AUC3-FinalPulseTailTrust: dataset_seed_pass=2, pass_rows=2, mean_source=-0.05224004056718615, median_AUCtime=1.0259966582387325, LineC_pass_rows=9
K-AUC4-FinalPulseIdentityProjection: dataset_seed_pass=3, pass_rows=3, mean_source=-0.004249744945102268, median_AUCtime=1.052606328166947, LineC_pass_rows=9
K0-RAT-AdamW: dataset_seed_pass=0
KCTRL-RandomMatchedProjection: dataset_seed_pass=0
```

新 action 纳入 v14.5 extended oracle 的命令：

```bash
conda run -n kan python -m py_compile experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py && \
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_finalpulse_action \
  --mode oracle \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-ActionBankUpperBoundInsufficient",
  "minimum_success": "S4b-RealTransferExplorationPositive",
  "official_s5_reached": 0,
  "promotion_allowed": 0,
  "core_oracle_dataset_seed_pass_count": 6,
  "extended_legal_v144_oracle_dataset_seed_pass_count": 7,
  "best_oracle_dataset_seed_pass_count": 7,
  "controller_executed": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0
}
```

manifest 缺失检查：

```bash
awk -F, 'NR>1 && $3!=0 {print}' \
  results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_finalpulse_action/v145_required_manifest.csv
```

结果：

```text
no output
```

新增产物：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_finalpulse_auc_action_family/
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_finalpulse_action/
```

## 10. 用户再次追问后的 basis-free / curvature-safe action family 尝试

触发原因：

```text
v14.5 计划在 oracle bank < 9/9 时建议新 action family，
包括 low-cost curvature-safe FMS 与 basis-free FMS + delayed projection。
前一轮 final-pulse 后 extended oracle 仍为 7/9，
因此继续覆盖这两个计划示例方向。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-BF1-BasisFreeDelayedProjection。
  新增 K-CURV1-CurvatureSafeBasisFreeFMS。
  在 train-stream proxy 中新增 train_grad_norm。
```

实现 blocker：

```text
首跑 K-CURV1 时，grad_growth 极端值导致 math.exp overflow。
修复：对 grad_growth 与 sigmoid exponent 做有限 clamp。
```

首跑失败命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_basisfree_curvature_action_family \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-BF1-BasisFreeDelayedProjection,K-CURV1-CurvatureSafeBasisFreeFMS,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

失败摘要：

```text
OverflowError: math range error
位置：lambda_from_method / K-CURV1 curvature_gate
```

修复后语法检查与重跑：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py && \
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_basisfree_curvature_action_family \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-BF1-BasisFreeDelayedProjection,K-CURV1-CurvatureSafeBasisFreeFMS,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-RealTransferFail",
  "minimum_success": "S4-RealShortRunOpened",
  "official_s5_reached": 0,
  "real_dataset_seed_pass_count": 1,
  "real_short_run_pass_rows": 2,
  "mean_source_vs_best_control_noncontrol": -0.04684809843699137,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

method summary：

```text
K-BF1-BasisFreeDelayedProjection: dataset_seed_pass=1, pass_rows=1, mean_source=0.0031835767957899305, median_AUCtime=1.087945079486236, LineC_pass_rows=9
K-CURV1-CurvatureSafeBasisFreeFMS: dataset_seed_pass=1, pass_rows=1, mean_source=-0.09687977366977268, median_AUCtime=1.0322478422814108, LineC_pass_rows=9
```

纳入 extended oracle：

```bash
conda run -n kan python -m py_compile experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py && \
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_basisfree_curvature_action \
  --mode oracle \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-ActionBankUpperBoundInsufficient",
  "extended_legal_v144_oracle_dataset_seed_pass_count": 7,
  "best_oracle_dataset_seed_pass_count": 7,
  "controller_executed": 0,
  "promotion_allowed": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0
}
```

新增产物：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_basisfree_curvature_action_family/
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_basisfree_curvature_action/
```

## 11. 用户再次追问后的 two-phase source-tail action family 尝试

触发原因：

```text
v14.5 计划还建议 two-phase source-tail separation。
K-BF1/K-CURV1 后 oracle 仍为 7/9，
因此继续实现并运行一个统一的 two-phase action family。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 K-2P1-TwoPhaseSourceTailSeparation。
  新增 K-2P2-TwoPhaseDelayedSourceTail。
```

运行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v144_real_transfer_fms_all_basis_substrate.py && \
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_twophase_action_family \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-2P1-TwoPhaseSourceTailSeparation,K-2P2-TwoPhaseDelayedSourceTail,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-RealTransferFail",
  "minimum_success": "S4-RealShortRunOpened",
  "official_s5_reached": 0,
  "real_dataset_seed_pass_count": 2,
  "real_short_run_pass_rows": 3,
  "mean_source_vs_best_control_noncontrol": -0.1334330207771725,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

method summary：

```text
K-2P1-TwoPhaseSourceTailSeparation: dataset_seed_pass=2, pass_rows=2, mean_source=-0.021079083283742268, median_AUCtime=1.038790706640546, LineC_pass_rows=9
K-2P2-TwoPhaseDelayedSourceTail: dataset_seed_pass=1, pass_rows=1, mean_source=-0.24578695827060276, median_AUCtime=1.0743407704918515, LineC_pass_rows=8
```

纳入 extended oracle：

```bash
conda run -n kan python -m py_compile experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py && \
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_twophase_action \
  --mode oracle \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route 摘要：

```json
{
  "route": "R2-ActionBankUpperBoundInsufficient",
  "extended_legal_v144_oracle_dataset_seed_pass_count": 7,
  "best_oracle_dataset_seed_pass_count": 7,
  "controller_executed": 0,
  "promotion_allowed": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0
}
```

manifest 缺失检查：

```bash
awk -F, 'NR>1 && $3!=0 {print}' \
  results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_twophase_action/v145_required_manifest.csv
```

结果：

```text
no output
```

新增产物：

```text
results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_twophase_action_family/
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_twophase_action/
```

## 12. 用户再次要求继续后的 Line W Wavelet bounded hardening

触发原因：

```text
1. v14.5 Line O/K 当前 best oracle 仍为 7/9，计划规定 oracle bank < 9/9 时不能 tune controller。
2. v14.5 计划 Line W 明确要求继续 Wavelet substrate hardening，且 Line W 不阻塞 Line K。
3. 旧 v145 oracle artifact 只复用 v14.3/v14.4 all-basis status，new_training_executed = 0；
   因此本次补做 Line W compute-budgeted bounded diagnostic。
```

代码修改：

```text
1. 新增：
   experiments/run_v145_wavelet_linew_hardening.py

2. 该 runner 记录 v14.5 Line W must-record 字段：
   candidate_id / dataset / seed
   workspace_raw_ratio / workspace_incremental_ratio / step_ratio
   mean_delta_vs_mlp / worst_delta_vs_mlp / AUCtime_ratio
   CEp99_delta / NLL_delta / ECE_delta
   LineC_pass_rate / RealSignalReservoirRatio / NoiseSignalLeak
   scale_occupancy_entropy / support_overlap / local_tail_coverage

3. 该 runner 明确：
   official_fms_proof_executed = 0
   promotion_allowed = 0
   linec_tail_auc_used_for_direction = 0
   labels_used_for_direction = 0
   validation_test_future_query_used_for_direction = 0

4. 修改：
   experiments/run_v143_nonrat_compact_task_health_probe.py
   新增 readout gradient gate:
   linear_readout_grad010
   linear_readout_grad005

5. 在 v145 Wavelet runner 中新增 W7-reservoir-balance：
   support = quantile_scale050
   readout gradient gate = linear_readout_grad010
   output geometry = none
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v143_nonrat_compact_task_health_probe.py \
  experiments/run_v145_wavelet_linew_hardening.py
```

结果：

```text
pass
```

### 12.1 v145 Wavelet runner smoke

执行命令：

```bash
conda run -n kan python experiments/run_v145_wavelet_linew_hardening.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_smoke_MNIST_seed0_v145 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --datasets MNIST \
  --seeds 0 \
  --configs W123-train-entropy,W4-support-entropy-readout,W5-scale-occupancy,W6-local-tail-guard \
  --candidates D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 3 \
  --hidden-override 256 \
  --linec-seeds 12319500 \
  --device cuda:0
```

route 摘要：

```json
{
  "route": "W0-WaveletSubstrateFail",
  "wavelet_dataset_seed_pass_count": 0,
  "wavelet_gate_pass_rows": 0,
  "official_fms_proof_executed": 0,
  "open_wavelet_fms_synthetic_proof": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

smoke blocker：

```text
MNIST seed0 在 W123/W4/W5/W6 下均未过 gate；
主要 blocker = task + LineC reservoir；
NoiseSignalLeak / RealSignalReservoirRatio 存在 tradeoff。
```

### 12.2 Line W full 3x3 W123/W4/W5/W6

执行命令：

```bash
conda run -n kan python experiments/run_v145_wavelet_linew_hardening.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_full_v145 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --configs W123-train-entropy,W4-support-entropy-readout,W5-scale-occupancy,W6-local-tail-guard \
  --candidates D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 3 \
  --hidden-override 256 \
  --linec-seeds 12319500 \
  --device cuda:0
```

route 摘要：

```json
{
  "route": "W0-WaveletSubstrateFail",
  "wavelet_dataset_seed_pass_count": 1,
  "wavelet_gate_pass_rows": 6,
  "official_fms_proof_executed": 0,
  "open_wavelet_fms_synthetic_proof": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

summary：

```text
W123-train-entropy: rows=27, dataset_seed_pass=1, gate_pass_rows=3, best_AUCtime=1.0365907904977243, best_LineC=1.0
W4-support-entropy-readout: rows=27, dataset_seed_pass=1, gate_pass_rows=3, best_AUCtime=1.0356365995316072, best_LineC=1.0
W5-scale-occupancy: rows=27, dataset_seed_pass=0, gate_pass_rows=0, best_AUCtime=1.0350109281750535, best_LineC=1.0
W6-local-tail-guard: rows=27, dataset_seed_pass=0, gate_pass_rows=0, best_AUCtime=1.0350107493680194, best_LineC=1.0
```

通过 rows：

```text
KMNIST seed0:
  W123 D-WAV17/18/19 pass
  W4 D-WAV17/18/19 pass
```

failure counts：

```text
LineC = 96
RealSignalReservoirRatio = 66
NoiseSignalLeak = 60
task = 42
```

### 12.3 W7 reservoir-balance 修复

触发原因：

```text
W123/W4/W5/W6 的主 blocker 是 LineC reservoir；
按计划 13.5，若 LineC reservoir fail，应尝试 train-stream reservoir proxy，
不能进入 Wavelet-FMS official proof。
```

先用 v143 局部探针检查 freeze readout：

```bash
conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_w7_freeze_support075_MNIST_seed0_v145 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --candidates D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --dataset MNIST \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 3 \
  --hidden-override 256 \
  --linec-seeds 12319500 \
  --wavelet-support-repair quantile_scale075 \
  --wavelet-role-constraint freeze_linear_readout \
  --output-geometry-repair none \
  --device cuda:1
```

局部结论：

```text
freeze readout 可使 LineC pass_rate=1.0，
但 MNIST seed0 task collapse，mean_delta_vs_MLP = -0.5；
因此不能作为 substrate repair。
```

其他局部探针均使用同一 v143 probe runner、MNIST seed0、同一 workspace csv、
同一 candidates、train_size=128、val_size=64、batch_size=32、epochs=3、
hidden_override=256、linec_seed=12319500；仅列出变更参数和结论：

| out dir | support | readout | output geometry | 结论 |
|---|---|---|---|---|
| `wavelet_linew_w5_scale_occupancy_MNIST_seed0_v145` | quantile_scale050 | linear_readout_grad025 | train_entropy_t085_100_else050 | gate=0；LineC=0；NoiseSignalLeak≈0.205-0.208；RSR≈0.771-0.773 |
| `wavelet_linew_w6_local_tail_guard_MNIST_seed0_v145` | quantile_scale050 | linear_readout_grad050 | train_topprob_t020_050_else100 | gate=0；与 W5 近似 |
| `wavelet_linew_w4b_support075_entropy085_MNIST_seed0_v145` | quantile_scale075 | linear_readout_grad025 | train_entropy_t085_100_else050 | gate=0；NoiseSignalLeak≈0.205-0.207；RSR≈0.783-0.788 |
| `wavelet_linew_w4c_support075_fixed050_MNIST_seed0_v145` | quantile_scale075 | linear_readout_grad025 | train_topprob_t020_050_else100 | gate=0；与 W4b 近似 |
| `wavelet_linew_support075_no_output_MNIST_seed0_v145` | quantile_scale075 | linear_readout_grad050 | none | gate=0；NoiseSignalLeak≈0.236；RSR≈0.717-0.721 |
| `wavelet_linew_support050_no_output_MNIST_seed0_v145` | quantile_scale050 | linear_readout_grad050 | none | gate=0；NoiseSignalLeak≈0.229-0.233；RSR≈0.706 |
| `wavelet_linew_w7_freeze_support050_MNIST_seed0_v145` | quantile_scale050 | freeze_linear_readout | none | gate=0；LineC=1.0；task collapse, delta=-0.5 |
| `wavelet_linew_w8_grad005_support050_MNIST_seed0_v145` | quantile_scale050 | linear_readout_grad005 | none | gate=0；与 grad010 近似，仍 LineC fail |

W7 full 执行命令：

```bash
conda run -n kan python experiments/run_v145_wavelet_linew_hardening.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_w7_reservoir_balance_full_v145 \
  --workspace-csv results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_wav_lowraw_compact_h256_workspace_v143/v143_nonrat_manual_kernel_substrate_probe.csv \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --configs W7-reservoir-balance \
  --candidates D-WAV17-Raw005SupportHealthSubstrate,D-WAV18-Raw002SupportHealthSubstrate,D-WAV19-Raw001SupportHealthSubstrate \
  --train-size 128 \
  --val-size 64 \
  --batch-size 32 \
  --epochs 3 \
  --hidden-override 256 \
  --linec-seeds 12319500 \
  --device cuda:0
```

route 摘要：

```json
{
  "route": "W0-WaveletSubstrateFail",
  "wavelet_dataset_seed_pass_count": 1,
  "wavelet_gate_pass_rows": 3,
  "official_fms_proof_executed": 0,
  "open_wavelet_fms_synthetic_proof": 0,
  "required_artifact_missing_count": 0,
  "forbidden_information_violation_count": 0,
  "promotion_allowed": 0
}
```

W7 通过 rows：

```text
Fashion-MNIST seed1:
  W7 D-WAV17/18/19 pass
```

W7 failure counts：

```text
LineC = 21
RealSignalReservoirRatio = 18
task = 12
NoiseSignalLeak = 12
```

### 12.4 Line W artifacts

新增主要产物：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_smoke_MNIST_seed0_v145/
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_full_v145/
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/wavelet_linew_w7_reservoir_balance_full_v145/
```

每个 v145 Line W run 输出：

```text
v145_wavelet_substrate_hardening.csv
v145_wavelet_linec_audit.csv
v145_wavelet_substrate_summary.csv
v145_wavelet_failure_table.csv
v145_wavelet_forbidden_information_audit.csv
v145_wavelet_required_manifest.csv
v145_wavelet_route_decision.json
```

最终判断：

```text
Line W 未达到 full 3x3 substrate pass；
open_wavelet_fms_synthetic_proof = 0；
promotion_allowed = 0；
不能进入 Wavelet-FMS official proof。
```

## 13. 用户再次要求继续后的 Line D All-basis bounded substrate repair

触发原因：

```text
v14.5 主线仍为 R2-ActionBankUpperBoundInsufficient；
Line W 也没有打开 full 3x3 substrate pass。
继续按计划覆盖 Line D：
  RBF / FastKAN compact center / width / local support repair；
  Chebyshev degree-energy / recurrence lifetime repair；
  Fourier low-frequency identity residual / phase stable repair。

本节不运行 official FMS proof，不运行 controller，不允许 promotion。
LineC / NLL / task-health 只作为 gate / audit。
```

### 13.1 Line D compact workspace

执行命令：

```bash
CANDS="D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate,D-CHE16-DegreeEnergyDampingSubstrate,D-CHE17-HighDegreeLateEnableSubstrate,D-CHE18-RoleDegreeEnergyCapSubstrate,D-CHE19-ChebyTangentTrustSubstrate,D-CHE20-DegreeNormalizedReadoutHealthSubstrate,D-FOU16-FrequencyBandDampingSubstrate,D-FOU17-PhaseStabilityCorrectionSubstrate,D-FOU18-LowFreqSignalTransportSubstrate,D-FOU19-HighFreqNoiseLeakVetoSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate"
conda run -n kan python experiments/run_v143_nonrat_manual_kernel_substrate_probe.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_workspace_v145 \
  --candidates "$CANDS" \
  --synthetic-task X1 \
  --seed 0 \
  --train-size 128 \
  --val-size 64 \
  --synthetic-dim 16 \
  --synthetic-classes 3 \
  --batch-size 32 \
  --profile-steps 1 \
  --workspace-warmup-steps 1 \
  --mlp-hidden 160 \
  --hidden-override 256 \
  --lr 0.002 \
  --weight-decay 0.001 \
  --adamw-foreach false \
  --device cuda:0
```

route 摘要：

```json
{
  "candidate_rows": 32,
  "manual_workspace_gate_pass_rows": 22,
  "official_fms_proof_executed": 0,
  "promotion_allowed": 0
}
```

manual_no_materialize workspace 结果：

```text
RBF: 6 / 6 pass
Chebyshev: 5 / 5 pass
Fourier: 5 / 5 pass
```

### 13.2 Line D full all-basis task-health

执行命令：

```bash
CANDS="D-RBF12-CenterOccupancyRebalanceSubstrate,D-RBF13-WidthConditionGuardSubstrate,D-RBF14-OOGBoundaryRepairSubstrate,D-RBF15-LocalCurvatureSmoothSubstrate,D-RBF16-ActiveCenterDiversityTransportSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate,D-CHE16-DegreeEnergyDampingSubstrate,D-CHE17-HighDegreeLateEnableSubstrate,D-CHE18-RoleDegreeEnergyCapSubstrate,D-CHE19-ChebyTangentTrustSubstrate,D-CHE20-DegreeNormalizedReadoutHealthSubstrate,D-FOU16-FrequencyBandDampingSubstrate,D-FOU17-PhaseStabilityCorrectionSubstrate,D-FOU18-LowFreqSignalTransportSubstrate,D-FOU19-HighFreqNoiseLeakVetoSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate"
BASE="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_task_trainstream_repair_full_v145"
WS="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_workspace_v145/v143_nonrat_manual_kernel_substrate_probe.csv"
for DATASET in MNIST Fashion-MNIST KMNIST; do
  for SEED in 0 1 2; do
    OUT="$BASE/${DATASET}_seed${SEED}"
    conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
      --out-dir "$OUT" \
      --workspace-csv "$WS" \
      --candidates "$CANDS" \
      --device cuda:0 \
      --data-root data \
      --no-download \
      --dataset "$DATASET" \
      --seed "$SEED" \
      --train-size 128 \
      --val-size 64 \
      --batch-size 32 \
      --epochs 3 \
      --mlp-hidden 160 \
      --hidden-override 256 \
      --lr 0.002 \
      --weight-decay 0.001 \
      --adamw-foreach false \
      --linec-batch-size 24 \
      --linec-sketch-dim 8 \
      --linec-seeds 12319500 \
      --rbf-center-repair quantile_width075 \
      --output-geometry-repair train_entropy_t080_100_else050
  done
done
```

route 摘要：

```text
line_d_dataset_seed_pass_count = 1 / 9
line_d_gate_pass_rows = 5
line_d_total_rows = 144
workspace_manual_gate_pass_rows = 144
official_fms_proof_executed = 0
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

failure counts：

```text
LineC = 66
mean_task = 139
worst_task = 137
```

### 13.3 Fourier low-frequency lr005 epochs6 repair

触发原因：

```text
full Line D 中通过行全部来自 Fourier；
最接近失败行多数是 D-FOU16/17/19/20，
且主要 failure 从 workspace 转为 task / LineC。
因此尝试统一的 low-frequency Fourier longer/higher-lr probe。
```

执行命令：

```bash
CANDS="D-FOU16-FrequencyBandDampingSubstrate,D-FOU17-PhaseStabilityCorrectionSubstrate,D-FOU18-LowFreqSignalTransportSubstrate,D-FOU19-HighFreqNoiseLeakVetoSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate"
BASE="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_fourier_lowfreq_lr005_epochs6_v145"
WS="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_workspace_v145/v143_nonrat_manual_kernel_substrate_probe.csv"
for DATASET in MNIST Fashion-MNIST KMNIST; do
  for SEED in 0 1 2; do
    OUT="$BASE/${DATASET}_seed${SEED}"
    conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
      --out-dir "$OUT" \
      --workspace-csv "$WS" \
      --candidates "$CANDS" \
      --device cuda:0 \
      --data-root data \
      --no-download \
      --dataset "$DATASET" \
      --seed "$SEED" \
      --train-size 128 \
      --val-size 64 \
      --batch-size 32 \
      --epochs 6 \
      --mlp-hidden 160 \
      --hidden-override 256 \
      --lr 0.005 \
      --weight-decay 0.001 \
      --adamw-foreach false \
      --linec-batch-size 24 \
      --linec-sketch-dim 8 \
      --linec-seeds 12319500 \
      --output-geometry-repair train_entropy_t080_100_else050
  done
done
```

route 摘要：

```text
line_d_dataset_seed_pass_count = 3 / 9
line_d_gate_pass_rows = 6
line_d_total_rows = 45
workspace_manual_gate_pass_rows = 45
promotion_allowed = 0
```

通过 dataset-seed：

```text
MNIST seed2
Fashion-MNIST seed0
Fashion-MNIST seed1
```

failure counts：

```text
LineC = 33
mean_task = 7
worst_task = 6
```

### 13.4 Fourier output-geometry probes

fixed050 执行命令：

```bash
CANDS="D-FOU16-FrequencyBandDampingSubstrate,D-FOU17-PhaseStabilityCorrectionSubstrate,D-FOU18-LowFreqSignalTransportSubstrate,D-FOU19-HighFreqNoiseLeakVetoSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate"
BASE="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_fourier_lowfreq_lr005_epochs6_fixed050_v145"
WS="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_workspace_v145/v143_nonrat_manual_kernel_substrate_probe.csv"
for DATASET in MNIST Fashion-MNIST KMNIST; do
  for SEED in 0 1 2; do
    OUT="$BASE/${DATASET}_seed${SEED}"
    conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
      --out-dir "$OUT" \
      --workspace-csv "$WS" \
      --candidates "$CANDS" \
      --device cuda:0 \
      --data-root data \
      --no-download \
      --dataset "$DATASET" \
      --seed "$SEED" \
      --train-size 128 \
      --val-size 64 \
      --batch-size 32 \
      --epochs 6 \
      --mlp-hidden 160 \
      --hidden-override 256 \
      --lr 0.005 \
      --weight-decay 0.001 \
      --adamw-foreach false \
      --linec-batch-size 24 \
      --linec-sketch-dim 8 \
      --linec-seeds 12319500 \
      --output-geometry-repair fixed050
  done
done
```

fixed050 结果：

```text
line_d_dataset_seed_pass_count = 3 / 9
line_d_gate_pass_rows = 6
failure counts:
  LineC = 38
  mean_task = 7
  worst_task = 6
```

fixed025 执行命令：

```bash
CANDS="D-FOU16-FrequencyBandDampingSubstrate,D-FOU17-PhaseStabilityCorrectionSubstrate,D-FOU18-LowFreqSignalTransportSubstrate,D-FOU19-HighFreqNoiseLeakVetoSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate"
BASE="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_fourier_lowfreq_lr005_epochs6_fixed025_v145"
WS="results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_workspace_v145/v143_nonrat_manual_kernel_substrate_probe.csv"
for DATASET in MNIST Fashion-MNIST KMNIST; do
  for SEED in 0 1 2; do
    OUT="$BASE/${DATASET}_seed${SEED}"
    conda run -n kan python experiments/run_v143_nonrat_compact_task_health_probe.py \
      --out-dir "$OUT" \
      --workspace-csv "$WS" \
      --candidates "$CANDS" \
      --device cuda:0 \
      --data-root data \
      --no-download \
      --dataset "$DATASET" \
      --seed "$SEED" \
      --train-size 128 \
      --val-size 64 \
      --batch-size 32 \
      --epochs 6 \
      --mlp-hidden 160 \
      --hidden-override 256 \
      --lr 0.005 \
      --weight-decay 0.001 \
      --adamw-foreach false \
      --linec-batch-size 24 \
      --linec-sketch-dim 8 \
      --linec-seeds 12319500 \
      --output-geometry-repair fixed025
  done
done
```

fixed025 结果：

```text
line_d_dataset_seed_pass_count = 2 / 9
line_d_gate_pass_rows = 5
failure counts:
  LineC = 36
  NLL = 1
  mean_task = 7
  worst_task = 6
```

### 13.5 Line D 汇总

汇总产物：

```text
results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/lined_allbasis_summary_v145/
```

关键汇总：

```text
single best legal config = lined_fourier_lowfreq_lr005_epochs6_v145
single best dataset-seed pass = 3 / 9
single best gate pass rows = 6

best across configs audit-only:
  dataset-seed pass = 4 / 9
  gate pass rows = 22
  不能作为 promotion；不能拼接不同配置写成成功。
```

best row by dataset-seed：

| dataset | seed | pass | config | candidate | blocker |
|---|---:|---:|---|---|---|
| MNIST | 0 | 0 | lr005_epochs6 | D-FOU17 | mean_task,worst_task |
| MNIST | 1 | 1 | lr005_epochs6_fixed025 | D-FOU17 | - |
| MNIST | 2 | 1 | lr005_epochs6 | D-FOU19 | - |
| Fashion-MNIST | 0 | 1 | lr005_epochs6 | D-FOU16 | - |
| Fashion-MNIST | 1 | 1 | lr005_epochs6 | D-FOU16 | - |
| Fashion-MNIST | 2 | 0 | lr005_epochs6_fixed050 | D-FOU17 | LineC |
| KMNIST | 0 | 0 | lr005_epochs6 | D-FOU19 | LineC |
| KMNIST | 1 | 0 | lr005_epochs6 | D-FOU17 | LineC |
| KMNIST | 2 | 0 | lr005_epochs6 | D-FOU19 | LineC |

最终判断：

```text
Line D 未达到 full 3x3 substrate pass。
RBF / Chebyshev workspace 已过，但 task-health 仍崩。
Fourier low-frequency 修复从 1/9 推到 single-config 3/9，
但仍被 MNIST seed0 的 task failure 与 Fashion-MNIST seed2 / KMNIST 的 LineC failure 阻断。
fixed050 / fixed025 显示输出尺度只是在不同 dataset-seed 之间交换 coverage，
不是稳定修复。

不进入 Non-RAT official FMS proof；
不允许 promotion。
```

## 14. 用户再次要求继续后的 micro-pulse / curvature low-lambda action probe

### 14.1 触发原因

用户再次要求未达成则继续。本次先复核 v14.5 最新 oracle：

```text
latest route = R2-ActionBankUpperBoundInsufficient
extended legal v144 oracle = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

剩余 oracle failure：

```text
Fashion-MNIST seed2:
  best row = official_v144 / K-RT2
  source = 0.233825
  AUCtime = 1.209973
  CEp99 delta = -1.428411
  NLL delta = -0.233825
  ECE delta = -0.019295
  LineC = 1
  blocker = AUC

KMNIST seed2:
  best row = repair_v145_basisfree_curvature_action_family / K-BF1
  source = 0.232619
  AUCtime = 1.122462
  CEp99 delta = -0.564575
  NLL delta = -0.309992
  ECE delta = -0.020928
  LineC = 1
  blocker = AUC
```

判断：

```text
剩余 blocker 不再是 source / tail / LineC，而是 AUCtime。
因此继续强化 tail/source 不合适；本次只尝试更弱、更接近 AdamW 轨迹的
micro-pulse / curvature-safe action。
```

### 14.2 micro-pulse curvature action run

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_micro_pulse_curvature_action_family \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-AUC3-FinalPulseTailTrust,K-AUC4-FinalPulseIdentityProjection,K-CURV1-CurvatureSafeBasisFreeFMS,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.25 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = -0.048930605252583824
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

```text
K-AUC3-FinalPulseTailTrust:
  dataset_seed_pass = 2
  pass_rows = 2
  mean_source = -0.05108500851525201
  median_source = 0.03684687614440918
  median_AUCtime = 1.0258133077999416
  LineC_pass_rows = 9

K-AUC4-FinalPulseIdentityProjection:
  dataset_seed_pass = 3
  pass_rows = 3
  mean_source = -0.0044083330366346575
  median_source = 0.013168573379516602
  median_AUCtime = 1.0525445131074915
  LineC_pass_rows = 9

K-CURV1-CurvatureSafeBasisFreeFMS:
  dataset_seed_pass = 1
  pass_rows = 1
  mean_source = -0.0912984742058648
  median_source = -0.016945600509643555
  median_AUCtime = 1.031439153944101
  LineC_pass_rows = 9
```

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-AUC4 | 0.050941 | 0.973495 | -3.536077 | -0.162381 | -0.020297 | 1 |
| MNIST | 1 | K-AUC3 | 0.175496 | 0.906495 | -5.737762 | -0.188712 | -0.016337 | 1 |
| MNIST | 1 | K-AUC4 | 0.162172 | 0.855069 | -5.133204 | -0.175389 | -0.014154 | 1 |
| MNIST | 1 | K-CURV1 | 0.272944 | 0.902376 | -5.382935 | -0.286160 | -0.033510 | 1 |
| MNIST | 2 | K-AUC3 | 0.036847 | 0.929429 | -1.094476 | -0.103742 | 0.000018 | 1 |
| MNIST | 2 | K-AUC4 | 0.096762 | 0.997539 | -1.112623 | -0.163657 | -0.024153 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 18
source = 14
CEp99_tail = 11
NLL_tail = 6
ECE_tail = 3
LineC = 0
```

### 14.3 纳入 v14.5 oracle

执行命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_micro_pulse_curvature_action \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

判断：

```text
micro-pulse curvature action 没有补上 Fashion-MNIST seed2 或 KMNIST seed2。
extended oracle 仍停在 7/9。
```

### 14.4 low-lr micro-pulse curvature action run

触发原因：

```text
micro-pulse lr=0.005 未打开 AUCtime blocker。
此前 low-lr repair 能提升 mean source，但没有配合末端/曲率微脉冲。
本次用统一 lr=0.003 做全 3x3 检查；不做 dataset-name branch，
不做 seed-specific scaling。
```

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_micro_pulse_curvature_lowlr_action_family \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-AUC3-FinalPulseTailTrust,K-AUC4-FinalPulseIdentityProjection,K-CURV1-CurvatureSafeBasisFreeFMS,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.003 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.25 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 5
mean_source_vs_best_control_noncontrol = 0.03533118963241577
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

```text
K-AUC3-FinalPulseTailTrust:
  dataset_seed_pass = 1
  pass_rows = 1
  mean_source = 0.051514506340026855
  median_source = 0.001583695411682129
  median_AUCtime = 1.014606097067032
  LineC_pass_rows = 8

K-AUC4-FinalPulseIdentityProjection:
  dataset_seed_pass = 2
  pass_rows = 2
  mean_source = 0.01567942566341824
  median_source = 0.029783308506011963
  median_AUCtime = 1.0550515719589157
  LineC_pass_rows = 8

K-CURV1-CurvatureSafeBasisFreeFMS:
  dataset_seed_pass = 2
  pass_rows = 2
  mean_source = 0.03879963689380222
  median_source = 0.07540583610534668
  median_AUCtime = 1.0207067148983873
  LineC_pass_rows = 9
```

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-AUC3 | 0.098271 | 0.970760 | -1.928946 | -0.109751 | -0.024200 | 1 |
| MNIST | 1 | K-AUC4 | 0.053912 | 0.932704 | -0.168503 | -0.065392 | -0.004021 | 1 |
| MNIST | 1 | K-CURV1 | 0.091432 | 0.992846 | -0.837555 | -0.102912 | -0.012476 | 1 |
| Fashion-MNIST | 1 | K-AUC4 | 0.107411 | 0.840245 | -0.847453 | -0.107411 | 0.011227 | 1 |
| KMNIST | 1 | K-CURV1 | 0.111598 | 0.953133 | -0.115360 | -0.196210 | -0.015661 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 20
CEp99_tail = 13
source = 9
NLL_tail = 6
LineC = 2
ECE_tail = 1
```

### 14.5 纳入 v14.5 oracle

执行命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_micro_pulse_curvature_lowlr_action \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. low-lr micro-pulse curvature action 也没有打开 S5。
2. 它让 mean source 转正，但 dataset-seed pass 仍为 3/9。
3. 纳入 v14.5 extended oracle 后，上界仍为 7/9。
4. 因此 controller 仍不能合法启动。
5. 不允许 promotion。
```

## 15. 用户再次要求继续后的 gradient-aligned trust-region action

### 15.1 触发原因

用户再次要求未达成则继续。当前剩余缺口仍是 AUC-only：

```text
Fashion-MNIST seed2:
  best = official_v144 / K-RT2
  source = 0.233825
  AUCtime = 1.209973
  tail / LineC 均过

KMNIST seed2:
  best = repair_v145_basisfree_curvature_action_family / K-BF1
  source = 0.232619
  AUCtime = 1.122462
  tail / LineC 均过
```

本次新增机制：

```text
K-TR1-GradientAlignedTrustRegion
K-TR2-LateGradientAlignedTrustRegion
```

核心思想：

```text
1. 使用当前 train mini-batch 的普通 AdamW gradient 作为 trajectory anchor。
2. FMS value direction 只允许在 train-gradient 附近做小幅增量。
3. 若 FMS direction 与当前 train gradient 对齐差，或增量范数过大，则收缩。
4. K-TR2 进一步只在 late phase 打开。
```

合法性：

```text
1. direction 只使用 train-stream gradient / FMS state / train proxy / phase。
2. 不使用 validation / test / future / query batch。
3. 不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction。
4. 不做 dataset-name branch。
5. 不做 seed-specific scaling。
```

### 15.2 代码修改

修改文件：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

主要修改：

```text
1. rt_method_base 新增 K-TR1 / K-TR2。
2. adaptive_projection_scales 对 K-TR1 / K-TR2 使用 identity projection，
   避免额外 basis projection 改写 trajectory。
3. lambda_from_method 新增：
   K-TR1 = phase * source_gate * tail_trust * progress_guard。
   K-TR2 = late_ramp * source_gate * tail_trust。
4. 新增 gradient_aligned_value_flat：
   value = generic_gradient + lambda * align_gate * clipped_delta。
   align_gate 来自 train-gradient 与 FMS direction 的 cosine。
   clipped_delta 范数按当前 train-gradient norm 限制。
5. train_case 中 value_flat 改为调用 gradient_aligned_value_flat。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

### 15.3 gradient-aligned trust-region run

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_gradient_aligned_trust_region_action_family \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-TR1-GradientAlignedTrustRegion,K-TR2-LateGradientAlignedTrustRegion,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = 0.01863147152794732
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

```text
K-TR1-GradientAlignedTrustRegion:
  dataset_seed_pass = 2
  pass_rows = 2
  mean_source = -0.022089693281385634
  median_source = 0.05983090400695801
  median_AUCtime = 1.0582657017722215
  LineC_pass_rows = 9

K-TR2-LateGradientAlignedTrustRegion:
  dataset_seed_pass = 4
  pass_rows = 4
  mean_source = 0.05935263633728027
  median_source = 0.006296634674072266
  median_AUCtime = 0.9682254863500114
  LineC_pass_rows = 9
```

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-TR1 | 0.223864 | 0.913624 | -4.776531 | -0.237126 | -0.028701 | 1 |
| MNIST | 1 | K-TR2 | 0.252104 | 0.856932 | -5.594723 | -0.265367 | -0.030127 | 1 |
| MNIST | 2 | K-TR2 | 0.214375 | 0.960684 | -2.075740 | -0.281305 | -0.040065 | 1 |
| KMNIST | 0 | K-TR2 | 0.201613 | 0.968225 | -6.414253 | -0.201613 | 0.016415 | 1 |
| KMNIST | 1 | K-TR1 | 0.162151 | 0.972546 | -3.030199 | -0.653782 | -0.041453 | 1 |
| KMNIST | 1 | K-TR2 | 0.006297 | 0.938703 | -1.484419 | -0.497927 | -0.044757 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 8
source = 7
CEp99_tail = 7
ECE_tail = 4
NLL_tail = 2
LineC = 0
```

判断：

```text
K-TR2 对 AUCtime 有正向信号，median_AUCtime = 0.968225；
但 unique dataset-seed pass 只有 4/9，没有达到 S5。
```

### 15.4 纳入 v14.5 oracle

执行命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_gradient_aligned_trust_region_action \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. Gradient-aligned trust-region action 没有打开 S5。
2. K-TR2 降低了本 run 的 AUCtime failure 数量，
   但没有补上 Fashion-MNIST seed2 / KMNIST seed2。
3. 纳入 extended oracle 后仍为 7/9。
4. controller 仍不能启动。
5. 不允许 promotion。
```

## 16. 用户再次要求继续后的 train-batch descent filter action

### 16.1 触发原因

用户再次要求未达成则继续。本次把 v14.5 counterfactual 思想更直接落到当前 train batch：

```text
K-CF1-TrainBatchDescentFilter
K-CF2-LateTrainBatchDescentFilter
```

核心思想：

```text
1. 先生成 FMS candidate gradient。
2. 用当前 train mini-batch AdamW gradient 计算一阶下降一致性：
   descent_ratio = dot(generic_grad, candidate_grad) / ||generic_grad||^2。
3. 如果 candidate 对当前 train loss 的一阶下降不可靠，则混回 AdamW gradient。
4. 如果 candidate 范数过大，则按当前 train-gradient norm 截断。
```

合法性：

```text
1. 只使用当前 train mini-batch gradient / FMS state / train proxy。
2. 不使用 validation / test / future / query batch。
3. 不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction。
4. 不做 dataset-name branch。
5. 不做 seed-specific scaling。
```

### 16.2 代码修改

修改文件：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

主要修改：

```text
1. rt_method_base 新增 K-CF1 / K-CF2。
2. adaptive_projection_scales 新增 K-CF1 / K-CF2，
   使用 existing value-retention projection path。
3. lambda_from_method 新增 K-CF1 / K-CF2。
4. 新增 train_batch_descent_filtered_grad。
5. projected_flat 赋值前调用 train_batch_descent_filtered_grad。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

### 16.3 train-batch descent filter run

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_train_batch_descent_filter_action_family \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-CF1-TrainBatchDescentFilter,K-CF2-LateTrainBatchDescentFilter,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 2 / 9
real_short_run_pass_rows = 3
mean_source_vs_best_control_noncontrol = -0.0806241167916192
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

```text
K-CF1-TrainBatchDescentFilter:
  dataset_seed_pass = 2
  pass_rows = 2
  mean_source = -0.05504324701097277
  median_source = -0.03355884552001953
  median_AUCtime = 1.0552308952393519
  LineC_pass_rows = 8

K-CF2-LateTrainBatchDescentFilter:
  dataset_seed_pass = 1
  pass_rows = 1
  mean_source = -0.10620498657226562
  median_source = 0.013661623001098633
  median_AUCtime = 1.0125294065992347
  LineC_pass_rows = 9
```

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 0 | K-CF1 | 0.204176 | 0.943531 | -3.758885 | -0.315636 | -0.011722 | 1 |
| MNIST | 1 | K-CF1 | 0.226195 | 0.896773 | -4.564541 | -0.239457 | -0.034714 | 1 |
| MNIST | 1 | K-CF2 | 0.301522 | 0.925878 | -8.713699 | -0.314784 | -0.021352 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 12
CEp99_tail = 10
source = 10
NLL_tail = 8
ECE_tail = 3
LineC = 1
```

判断：

```text
Train-batch descent filter 没有打开 S5；
单 run 退到 2/9，低于 K-TR。
```

### 16.4 纳入 v14.5 oracle

执行命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_train_batch_descent_filter_action \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. K-CF1 / K-CF2 没有打开 S5。
2. 纳入 extended oracle 后仍为 7/9。
3. 该 action family 低于 K-TR，不应继续沿此过滤方式扫阈值。
4. controller 仍不能启动。
5. 不允许 promotion。
```

## 17. 用户再次要求继续后的 front-loaded trajectory assist action

### 17.1 触发原因

用户再次要求未达成则继续。本次不继续调 K-CF 阈值，而尝试另一类 AUC-only 修复：

```text
K-FL1-FrontLoadedTrajectoryAssist
K-FL2-FrontLoadedGradientAlignedAssist
```

核心思想：

```text
1. AUC-only failure 可能来自早期 / 中期 trajectory cost。
2. 与 final pulse 相反，本次只在训练前半程给 FMS trajectory assist。
3. 后半程回到 AdamW，以避免 late functional action 拖高 AUC。
4. K-FL2 额外使用 train-gradient alignment 半径限制。
```

合法性：

```text
1. direction 只使用 train step phase / train proxy / FMS state / train gradient。
2. 不使用 AUCtime / LineC / CEp99 / NLL / ECE 作为 direction。
3. 不使用 validation / test / future / query batch。
4. 不做 dataset-name branch。
5. 不做 seed-specific scaling。
```

### 17.2 代码修改

修改文件：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

主要修改：

```text
1. rt_method_base 新增 K-FL1 / K-FL2。
2. adaptive_projection_scales 新增 K-FL1 / K-FL2。
3. K-FL2 走 identity projection + gradient_aligned_value_flat。
4. lambda_from_method 新增 front-loaded warmup/cooldown schedule：
   K-FL1: phase warmup 到 0.08，约 0.62 前关闭。
   K-FL2: phase warmup 到 0.08，约 0.55 前关闭。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

### 17.3 front-loaded trajectory action run

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_4_real_transfer_fms_all_basis_substrate/repair_v145_frontloaded_trajectory_action_family \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-FL1-FrontLoadedTrajectoryAssist,K-FL2-FrontLoadedGradientAlignedAssist,KCTRL-RandomMatchedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 6
mean_source_vs_best_control_noncontrol = 0.00047895643446180556
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
promotion_allowed = 0
```

method summary：

```text
K-FL1-FrontLoadedTrajectoryAssist:
  dataset_seed_pass = 2
  pass_rows = 2
  mean_source = -0.02782492505179511
  median_source = -0.00879669189453125
  median_AUCtime = 1.0286044314504503
  LineC_pass_rows = 9

K-FL2-FrontLoadedGradientAlignedAssist:
  dataset_seed_pass = 4
  pass_rows = 4
  mean_source = 0.028782837920718722
  median_source = 0.07313704490661621
  median_AUCtime = 1.0387034960633352
  LineC_pass_rows = 9
```

pass rows：

| dataset | seed | method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1 | K-FL1 | 0.219128 | 0.916936 | -6.472919 | -0.232391 | -0.021093 | 1 |
| MNIST | 1 | K-FL2 | 0.259067 | 0.994922 | -6.384107 | -0.272330 | -0.024174 | 1 |
| MNIST | 2 | K-FL1 | 0.182549 | 0.894147 | -2.439426 | -0.249479 | -0.036731 | 1 |
| MNIST | 2 | K-FL2 | 0.059128 | 0.953285 | -0.997680 | -0.126058 | -0.022846 | 1 |
| Fashion-MNIST | 1 | K-FL2 | 0.140299 | 0.880113 | -0.982815 | -0.160171 | 0.006390 | 1 |
| KMNIST | 0 | K-FL2 | 0.225074 | 0.962043 | -0.032822 | -0.225074 | 0.014608 | 1 |

failure counts（non-control rows）：

```text
AUCtime = 12
CEp99_tail = 7
source = 7
NLL_tail = 3
ECE_tail = 1
LineC = 0
```

判断：

```text
Front-loaded trajectory assist 没有打开 S5；
K-FL2 单方法 4/9，但没有超过 K-TR，也没有补上剩余 oracle 缺口。
```

### 17.4 纳入 v14.5 oracle

执行命令：

```bash
conda run -n kan python experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py \
  --out-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action \
  --include-extended-v144-bank 1 \
  --compute-budgeted-run 1
```

route：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
controller_not_executed_reason = oracle_action_bank_upper_bound_lt_9
official_s5_reached = 0
promotion_allowed = 0
```

仍未通过的 extended oracle rows：

| dataset | seed | best method | source | AUCtime | CEp99 delta | NLL delta | ECE delta | LineC | blocker |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Fashion-MNIST | 2 | K-RT2 | 0.233825 | 1.209973 | -1.428411 | -0.233825 | -0.019295 | 1 | AUC |
| KMNIST | 2 | K-BF1 | 0.232619 | 1.122462 | -0.564575 | -0.309992 | -0.020928 | 1 | AUC |

最终判断：

```text
1. K-FL1 / K-FL2 没有打开 S5。
2. 纳入 extended oracle 后仍为 7/9。
3. front-loaded 机制没有补上 Fashion-MNIST seed2 / KMNIST seed2。
4. controller 仍不能启动。
5. 不允许 promotion。
```

### 17.5 最终一致性检查

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v145_train_stream_counterfactual_fms_all_basis_parallel.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
git diff --check -- \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_执行日志.md \
  docs/DG-KAN_v14.5_TrainStreamCounterfactualFMS_AllBasisParallel_实验结果复盘.md
```

结果：

```text
git diff --check pass
```

route 复核命令：

```bash
jq '{route, best_oracle_dataset_seed_pass_count, controller_executed, official_s5_reached, promotion_allowed, required_artifact_missing_count, forbidden_information_violation_count}' \
  results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action/v145_route_decision.json
```

结果：

```text
route = R2-ActionBankUpperBoundInsufficient
best_oracle_dataset_seed_pass_count = 7
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
```
