# DG-KAN v15.0 FunctionUpdate AllBasis 并行加速执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 代码与计划

```text
plan = /home/chengshun.wang/DG-LCA/docs/DG-KAN_v15.0_FunctionUpdate_AllBasis_并行加速完整计划.md
runner = experiments/run_v150_function_update_allbasis_parallel.py
line_d_runner = experiments/run_v149_line_d_all_basis_substrate_repair.py
official_out = results/v15_0_function_update_allbasis_parallel/official_v150
line_d_out = results/v15_0_function_update_allbasis_parallel/line_d_v150_allbasis_substrate
```

## 1.1 本轮覆盖性修正

```text
1. FU3-D-CHE-CautiousFunctionUpdate 按计划拆成 hard / soft alignment 两个实际训练分支；
   M2-MLP-CautiousAdamW 同步拆成 hard / soft generic control。
2. Line C tail audit 增补 *_delta 字段，以及 signal_channel_energy、reservoir_energy、noise_leakage_proxy。
3. 上述字段只由实际训练 artifact 与同 family/dataset/seed baseline 计算；
   不用 LineC/CEp99/NLL/ECE/AUCtime 反推训练方向。
```

## 2. 执行命令

Line D substrate-only reconfirmation：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v15_0_function_update_allbasis_parallel/line_d_v150_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 12319500,12319501,12319502 --candidates D-FOU37-LowFreqIdentityResidualV4,D-FOU38-BandwiseSecondMomentWarmup,D-FOU39-PhaseStableCautiousUpdate,D-FOU40-NoMaterializeLifetimeV4,D-FOU41-HighFrequencyQuarantineV2,D-RBF35-ActiveCenterSecondMoment,D-RBF36-WidthConditionDecoupledDecay,D-RBF37-CompactBumpIdentityResidualV2,D-RBF38-GaussianLocalK4TaskHealthV2,D-RBF39-NoDenseCenterUpdate,D-WAV33-TriangularSupportV4,D-WAV34-ScaleSecondMomentOccupancy,D-WAV35-SupportOverlapCautiousUpdate,D-WAV36-LocalTailCoverageAuditOnly
```

v15.0 official finalizer：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v150_function_update_allbasis_parallel.py --out-dir results/v15_0_function_update_allbasis_parallel/official_v150 --line-d-out results/v15_0_function_update_allbasis_parallel/line_d_v150_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --trace-interval 60 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

## 3. 关键参数

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 256
val_size = 128
test_size = 128
train_steps = 120
batch_size = 32
lr = 0.003
weight_decay = 0.0001
linec_seeds = 12319500,12319501,12319502
device = cuda:0
official_fms_proof_executed_for_non_dche = 0
promotion_allowed_unless_S5 = 0
```

## 4. 输出文件

```text
v150_route_decision.json
v150_required_manifest.csv
v150_optimizer_mechanism_manifest.csv
v150_dche_fu_results.csv
v150_dche_fu_controls.csv
v150_mlp_controls.csv
v150_allbasis_substrate_results.csv
v150_rational_no_regression_monitor.csv
v150_linec_tail_audit.csv
v150_alignment_trace.csv
v150_block_preconditioner_trace.csv
v150_schedulefree_control_trace.csv
v150_failure_taxonomy.csv
v150_code_review_packet.zip
```

## 5. 运行结果摘要

```text
route = R8-CurrentFUDefinitionNoGo
minimum_success = S1-OptimizerMechanismImplemented
FU real_lite_pass_count = 0/9
Line D best = D-FOU 0/9
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 6. 用户再次追问后的补跑与复核命令

代码语法检查：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v150_function_update_allbasis_parallel.py
```

补齐 Line K `cautious` / `random_matched` decoupled decay controls 后，重跑 official finalizer：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v150_function_update_allbasis_parallel.py --out-dir results/v15_0_function_update_allbasis_parallel/official_v150 --line-d-out results/v15_0_function_update_allbasis_parallel/line_d_v150_allbasis_substrate --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 120 --trace-interval 60 --linec-seeds 12319500,12319501,12319502 --real-linec 1 --reuse-if-present 0
```

artifact 复核：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
base = Path('results/v15_0_function_update_allbasis_parallel/official_v150')
controls = list(csv.DictReader(open(base / 'v150_dche_fu_controls.csv')))
decay = list(csv.DictReader(open(base / 'v150_decoupled_decay_deconfound.csv')))
route = json.load(open(base / 'v150_route_decision.json'))
print('controls_rows', len(controls))
for method in sorted(set(r['method'] for r in controls)):
    group = [r for r in controls if r['method'] == method]
    print(method, len(group), sorted(set(r.get('decay_control_variant', '') for r in group)))
print('decay_rows', len(decay))
for r in decay:
    print({
        k: r.get(k)
        for k in [
            'dataset',
            'seed',
            'FU7_gain_vs_AdamW',
            'same_decoupled_decay_no_fu_gain',
            'cautious_decoupled_decay_no_fu_gain',
            'random_matched_decoupled_decay_no_fu_gain',
            'random_matched_decay_explains_gain',
            'any_decay_control_explains_gain',
        ]
    })
print(route)
PY
```

## 7. 覆盖矩阵复核结果

```text
v150_dche_fu_results.csv rows = 90
FU3 hard/soft = 9/9 + 9/9

v150_dche_fu_controls.csv rows = 108
C0/C1/C2/C3/C4/C5/C7/C8/C9 = 9 rows each
C6 standard/cautious/random_matched = 9/9 + 9/9 + 9/9

v150_mlp_controls.csv rows = 81
M2 hard/soft = 9/9 + 9/9

v150_decoupled_decay_deconfound.csv rows = 9
standard decay explains rows = 0
cautious decay explains rows = 0
random_matched decay explains rows = 2
any decay control explains rows = 2
decoupled_decay_explains_fraction = 0.2222222222222222
decoupled_decay_route = K-DecoupledDecayNotSufficient

v150_second_moment_trace.csv rows = 1917
v150_alignment_trace.csv rows = 837
v150_curvature_diag_trace.csv rows = 837
v150_block_preconditioner_trace.csv rows = 837
v150_schedulefree_control_trace.csv rows = 837
v150_linec_tail_audit.csv rows = 675
v150_allbasis_substrate_results.csv rows = 126
v150_rational_no_regression_monitor.csv rows = 8

v150_required_manifest.csv rows = 39, missing_sum = 0
v150_forbidden_information_audit.csv rows = 42, violation_sum = 0
v150_no_action_search_audit.csv rows = 42, violation_sum = 0
```
