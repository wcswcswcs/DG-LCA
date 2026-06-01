# DG-KAN v14.6.1 TrajectoryMechanismSufficiency NoActionSearch 执行日志

生成时间：2026-05-29（Asia/Singapore）

本日志只记录实际执行过的命令与产物位置；不补填未执行实验。

## 1. 计划读取

计划文件：

```text
docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md
```

读取命令：

```bash
sed -n '1,260p' docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md
sed -n '260,620p' docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md
sed -n '620,980p' docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md
sed -n '980,1320p' docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md
sed -n '1320,1500p' docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_完整计划.md
```

理解到的硬边界：

```text
1. v14.6.1 不是 action search。
2. 不新增 K-RT8 / K-AUC5 / K-FL3 类 action token。
3. action bank 只作为 upper-bound diagnostic artifact。
4. controller 在 coverage < 9/9 时不能执行。
5. AUCtime / LineC / CEp99 / NLL / ECE / Brier 只能作为 audit / gate。
6. diagnostic oracle / mechanism probe 不能写成 promotion。
```

## 2. 上游 artifact 检查

检查最新 v14.5 oracle route：

```bash
jq . \
  results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action/v145_route_decision.json
```

结果摘要：

```text
route = R2-ActionBankUpperBoundInsufficient
core_oracle_dataset_seed_pass_count = 6 / 9
extended_legal_v144_oracle_dataset_seed_pass_count = 7 / 9
best_oracle_dataset_seed_pass_count = 7 / 9
controller_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

检查 v14.5 oracle / audit / missing-state 文件：

```bash
python - <<'PY'
import csv, json
from pathlib import Path
p=Path('results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action')
print((p/'v145_route_decision.json').read_text())
for fn in ['v145_action_bank_oracle.csv','v145_linec_tail_auc_audit.csv','v145_missing_state_classes.csv','v145_action_bank_rows.csv']:
    path=p/fn
    print('\n==', fn, path.exists())
    if path.exists():
        with path.open() as f:
            r=list(csv.DictReader(f))
        print('rows', len(r), 'fields', r[0].keys() if r else [])
        for row in r[:5]: print(row)
PY
```

关键事实：

```text
extended oracle remaining failures:
  Fashion-MNIST seed2: dominant_failure = auc
  KMNIST seed2: dominant_failure = auc
```

检查 v14.4 train-stream proxy 可用性：

```bash
python - <<'PY'
import csv
from pathlib import Path
paths=list(Path('results/v14_4_real_transfer_fms_all_basis_substrate').glob('*/v144_train_stream_proxy.csv'))
print('proxy files', len(paths))
for path in paths[:3]:
    with path.open() as f: rows=list(csv.DictReader(f))
    print(path, len(rows), rows[0].keys() if rows else [])
    if rows: print(rows[0])
PY
```

观察：

```text
v144_train_stream_proxy.csv 存在，但只记录 step 1 / 100 / 200 的 train-stream proxy。
AdamW moment / RMS state snapshots 不存在。
因此 P1 optimizer-state transport 不能从历史 artifact 中真实计算，
不能伪造 moment_staleness / rms_mismatch。
```

## 3. 代码修改

新增文件：

```text
experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

核心功能：

```text
1. 读取 v14.5 最新 oracle 与既有 v14.4/v14.5 action-bank rows。
2. 生成 v1461 action-search violation audit。
3. 生成 existing bank lock manifest，锁定旧 rows 只可诊断，不可搜索。
4. 生成 missing-state trajectory autopsy。
5. 生成 AUC debt decomposition。
6. 生成 P1/P2/P3/P4 mechanism probe artifacts。
7. 生成 mechanism coverage certificate。
8. 在 coverage_after < 9/9 时禁止 controller。
9. 输出 required manifest / route / no-go / next hypothesis / figures。
```

代码合法性说明：

```text
1. 没有新增 K-token action。
2. 没有运行 controller。
3. 没有用 audit metric 生成 direction。
4. P1 缺 telemetry 时写明 unavailable，不补假数据。
5. P2/P3 使用已有 train-stream / oracle artifact 做 diagnostic。
6. P4 只记录 retrospective delayed rows 不足，不把它写成预注册 k=1/2/4 probe 成功。
```

语法检查命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

结果：

```text
py_compile pass
```

diff 检查命令：

```bash
git diff --check -- experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

结果：

```text
git diff --check pass
```

## 7. 用户再次要求继续后的 P1 optimizer-state transport profile

触发原因：

```text
初版 diagnostic_v1461 的 route = R2-MechanismUpperBoundInsufficient，
P1 因历史 artifact 没有 AdamW moment / RMS snapshots 被标记为 unavailable。
计划文件的 next hypothesis 明确要求：
  instrument optimizer-state snapshots directly。
因此继续做固定 P1 mechanism profile，而不是新增 action token 或 action search。
```

代码修改：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  1. 新增 v144_optimizer_state_transport_probe.csv required artifact。
  2. 新增 optimizer_state_flat / set_optimizer_state_from_flat。
  3. 新增 optimizer_state_mismatch_metrics。
  4. 新增 apply_optimizer_state_transport。
  5. 新增 --optimizer-state-transport-probe。
  6. 新增 --optimizer-state-transport-mode：
     none
     zero_moment_reset
     partial_moment_interpolation
     rms_recompute_microbatch
     moment_transport_projected_grad
  7. result row 新增 median_moment_staleness_before/after、
     median_rms_mismatch_before/after 等 P1 telemetry。

experiments/run_v1461_trajectory_mechanism_sufficiency.py
  1. 新增 --p1-probe-root。
  2. 读取 p1_transport_*_v1461/v144_real_transfer_fms_results.csv。
  3. 按 P1 gate 计算 missing-state AUCDebt reduction、
     pass-state regression、source/tail/LineC non-harm。
  4. 更新 v1461_optimizer_state_transport_probe.csv、
     v1461_optimizer_state_mismatch.csv、
     v1461_mechanism_probe_manifest.csv、
     v1461_mechanism_coverage_certificate.csv。
```

合法性说明：

```text
1. 没有新增 K-token action。
2. P1 modes 是计划预注册的 optimizer-state handling，不改变 functional direction。
3. 不使用 validation/test/future/query 生成 direction。
4. 不使用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成 direction。
5. 不启动 controller。
6. 不把 diagnostic mechanism probe 写成 promotion。
```

语法与 diff 检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py

git diff --check -- \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

结果：

```text
py_compile pass
git diff --check pass
```

P1 surface smoke：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/smoke_p1_transport_surface_v1461 \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST \
  --seeds 0 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust \
  --skip-mlp-control \
  --train-steps 4 \
  --batch-size 8 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 2 \
  --rt-lambda-max 0.5 \
  --linec-mode none \
  --compute-budgeted-run 1 \
  --optimizer-state-transport-probe 1 \
  --optimizer-state-transport-mode zero_moment_reset
```

结果摘要：

```text
required_artifact_missing_count = 0
v144_optimizer_state_transport_probe.csv rows = 3
smoke 只验证 output surface，不作为 official success 证据。
```

## 8. P1 fixed-mode 3x3 runs

共同配置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
methods = K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection
skip_mlp_control = 1
train_steps = 200
batch_size = 32
lr = 0.005
fms_strength = 0.05
fms_update_interval = 80
rt_lambda_max = 0.5
linec_mode = exact
compute_budgeted_run = 1
optimizer_state_transport_probe = 1
```

### 8.1 baseline：none

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/p1_transport_none_v1461 \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --optimizer-state-transport-probe 1 \
  --optimizer-state-transport-mode none
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 3 / 9
real_short_run_pass_rows = 3
mean_source_vs_best_control_noncontrol = 0.13121526771121556
required_artifact_missing_count = 0
```

### 8.2 zero_moment_reset

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/p1_transport_zero_moment_reset_v1461 \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --optimizer-state-transport-probe 1 \
  --optimizer-state-transport-mode zero_moment_reset
```

结果：

```text
route = S5-OfficialFunctionalSuccess  # v144 runner local route only
real_dataset_seed_pass_count = 9 / 9
real_short_run_pass_rows = 18
mean_source_vs_best_control_noncontrol = 1.2612722516059875
required_artifact_missing_count = 0
promotion_allowed = 0
```

说明：

```text
本 run 是 v14.6.1 P1 mechanism probe；
不能直接写成 official promotion。
最终以 v1461 diagnostic route 为准。
```

### 8.3 partial_moment_interpolation

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/p1_transport_partial_moment_interpolation_v1461 \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --optimizer-state-transport-probe 1 \
  --optimizer-state-transport-mode partial_moment_interpolation
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 2 / 9
real_short_run_pass_rows = 2
mean_source_vs_best_control_noncontrol = -0.1511165632141961
required_artifact_missing_count = 0
```

### 8.4 rms_recompute_microbatch

首跑 blocker：

```text
OverflowError: math range error
位置：lambda_from_method 中 train-stream risk_state 进入 math.exp。
```

修复：

```text
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
  新增 inverse_logistic(arg)，将 exp 输入 clamp 到 [-60, 60]。
  将 lambda_from_method 中的 logistic gate 改为 inverse_logistic。
```

修复后检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py

git diff --check -- experiments/run_v144_real_transfer_fms_all_basis_substrate.py
```

结果：

```text
py_compile pass
git diff --check pass
```

重跑命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/p1_transport_rms_recompute_microbatch_v1461 \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --optimizer-state-transport-probe 1 \
  --optimizer-state-transport-mode rms_recompute_microbatch
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 0 / 9
real_short_run_pass_rows = 0
mean_source_vs_best_control_noncontrol = -11170648.803024624
required_artifact_missing_count = 0
```

### 8.5 moment_transport_projected_grad

执行命令：

```bash
conda run -n kan python experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/p1_transport_moment_transport_projected_grad_v1461 \
  --device cuda:0 \
  --data-root data \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --methods K0-RAT-AdamW,K-RT2-TrainStreamTailTrust,K-BF1-BasisFreeDelayedProjection \
  --skip-mlp-control \
  --train-steps 200 \
  --batch-size 32 \
  --lr 0.005 \
  --fms-strength 0.05 \
  --fms-update-interval 80 \
  --rt-lambda-max 0.5 \
  --linec-mode exact \
  --compute-budgeted-run 1 \
  --optimizer-state-transport-probe 1 \
  --optimizer-state-transport-mode moment_transport_projected_grad
```

结果：

```text
route = R2-RealTransferFail
real_dataset_seed_pass_count = 1 / 9
real_short_run_pass_rows = 1
mean_source_vs_best_control_noncontrol = -0.6987721655103896
required_artifact_missing_count = 0
```

## 9. v14.6.1 diagnostic rerun after P1

执行命令：

```bash
conda run -n kan python experiments/run_v1461_trajectory_mechanism_sufficiency.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport \
  --v145-oracle-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action \
  --p1-probe-root results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search \
  --compute-budgeted-run 1
```

stdout route：

```text
route = S4c-MechanismSufficientDiagnostic
coverage_before = 7 / 9
coverage_after = 9 / 9
best_p1_state_transport_mode = zero_moment_reset
mechanism_probe_gate_pass_count = 1
controller_executed = 0
controller_not_executed_reason = mechanism_sufficient_diagnostic_no_controller_in_v1461
action_search_violation_count = 0
forbidden_information_violation_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

关键抽查命令：

```bash
jq . \
  results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport/v1461_route_decision.json

python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport')
for fn in ['v1461_optimizer_state_transport_probe.csv','v1461_mechanism_probe_manifest.csv','v1461_mechanism_coverage_certificate.csv','v1461_optimizer_state_mismatch.csv','v1461_failure_table.csv']:
    rows=list(csv.DictReader((base/fn).open()))
    print(fn, len(rows))
    for row in rows[:12]:
        print(row)
PY
```

抽查结论：

```text
v1461_optimizer_state_transport_probe.csv rows = 12
v1461_mechanism_probe_manifest.csv rows = 4
v1461_mechanism_coverage_certificate.csv rows = 9
v1461_optimizer_state_mismatch.csv rows = 8
v1461_failure_table.csv rows = 2
```

## 10. after-P1 最终一致性检查

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
git diff --check -- \
  experiments/run_v144_real_transfer_fms_all_basis_substrate.py \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py \
  docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_执行日志.md \
  docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_实验结果复盘.md
```

结果：

```text
git diff --check pass
```

route 抽查命令：

```bash
jq '.route, .coverage_before, .coverage_after, .best_p1_state_transport_mode, .required_artifact_missing_count, .official_s5_reached, .promotion_allowed' \
  results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport/v1461_route_decision.json
```

结果：

```text
route = S4c-MechanismSufficientDiagnostic
coverage_before = 7
coverage_after = 9
best_p1_state_transport_mode = zero_moment_reset
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

manifest 抽查命令：

```bash
python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461_after_p1_transport')
for fn in ['v1461_required_manifest.csv','v1461_code_review_manifest.csv','v1461_mechanism_probe_manifest.csv','v1461_optimizer_state_transport_probe.csv']:
    rows=list(csv.DictReader((base/fn).open()))
    print(fn, len(rows))
    if fn=='v1461_required_manifest.csv':
        print('missing', sum(1 for r in rows if r.get('exists')!='1'))
    if fn=='v1461_code_review_manifest.csv':
        print([r['path'] for r in rows])
PY
```

结果：

```text
v1461_required_manifest.csv rows = 31, missing = 0
v1461_code_review_manifest.csv rows = 7
v1461_mechanism_probe_manifest.csv rows = 4
v1461_optimizer_state_transport_probe.csv rows = 12
```

## 4. v14.6.1 diagnostic run

执行命令：

```bash
conda run -n kan python experiments/run_v1461_trajectory_mechanism_sufficiency.py \
  --out-dir results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461 \
  --v145-oracle-dir results/v14_5_train_stream_counterfactual_fms_all_basis_parallel/oracle_o1_v145_after_frontloaded_trajectory_action \
  --compute-budgeted-run 1
```

stdout route：

```text
route = R2-MechanismUpperBoundInsufficient
coverage_before = 7 / 9
coverage_after = 7 / 9
missing_state_count = 2
missing_state_classes = C1-AUCOnlyTrajectoryCost
line_t_mechanism_precondition_pass = 1
mechanism_probe_gate_pass_count = 0
controller_executed = 0
action_search_violation_count = 0
forbidden_information_violation_count = 0
required_artifact_missing_count = 0
official_s5_reached = 0
promotion_allowed = 0
```

## 5. artifact 抽查

route 复核命令：

```bash
jq . \
  results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461/v1461_route_decision.json
```

关键 CSV 抽查命令：

```bash
python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461')
for fn in ['v1461_required_manifest.csv','v1461_missing_state_trajectory_autopsy.csv','v1461_mechanism_probe_manifest.csv','v1461_mechanism_coverage_certificate.csv','v1461_failure_table.csv']:
    p=base/fn
    print('\n==',fn,p.exists())
    with p.open() as f: rows=list(csv.DictReader(f))
    print('rows',len(rows),'fields',list(rows[0].keys()) if rows else [])
    for r in rows[:6]: print(r)
PY
```

产物列表命令：

```bash
find results/v14_6_1_trajectory_mechanism_sufficiency_no_action_search/diagnostic_v1461 \
  -maxdepth 1 -type f | sort
```

抽查结论：

```text
required artifact rows = 31
required_artifact_missing_count = 0
v1461_failure_table.csv rows = 2
v1461_mechanism_probe_manifest.csv rows = 4
v1461_mechanism_coverage_certificate.csv rows = 9
```

## 6. 最终一致性检查

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
git diff --check -- \
  experiments/run_v1461_trajectory_mechanism_sufficiency.py \
  docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_执行日志.md \
  docs/DG-KAN_v14.6.1_TrajectoryMechanismSufficiency_NoActionSearch_实验结果复盘.md
```

结果：

```text
git diff --check pass
```
