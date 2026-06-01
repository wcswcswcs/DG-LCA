# DG-KAN v9.7.2 Exact Transfer Materializer / Core Expansion / Direct Update 实验复盘

> 本复盘记录 `DG-KAN_v9.7.2_ExactTransferMaterializer_CoreExpansion_DirectUpdate_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 exact-transfer materializer、linear/apply audit、exact score diagnostic、core-expansion diagnostic、direct-update boundary、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R3-ExactTransferNotBetterThanProxy
base_candidate = LQ-t2-h256
success_v9720_strict_purekan_functional = False
success_v9720_full_functional = False
success_v9720_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z/
```

核心结论：

1. P0 复现 v9.7.1 boundary：source route = `R1-ExactTransferArtifactMissing`，v9.7.1 exact rows = `0`，generated route = `stopped_no_new_objective`，field green/yellow/red = `14 / 4 / 0`。
2. P0 exact preflight 过：1 action x 8 check samples 完整落盘，`per_example_gradient_available = 1`，preflight exact rows = `8`，q90 grad compute = `0.43701985850930214 ms`。
3. P1 exact materializer 过：AP0 actions = `2876`，samples/action = `64`，exact linear transfer rows = `184064 / 184064`，completion = `1.0`，missing gradient/payload = `0 / 0`，nan/inf/duplicate = `0 / 0 / 0`。
4. P1 分阶段均过：P1a = `256 / 256` rows，P1b = `8192 / 8192` rows，P1c = `184064 / 184064` rows；P1 strong/weak pass = `1 / 1`。
5. P2 exact apply audit 过：64 actions x 64 samples = `4096` exact apply rows；linear/apply Pearson = `0.9997818368443041`，Spearman = `0.9978941645960946`，sign match = `0.961181640625`，MAE = `6.0028745734372246e-05`。
6. P2 per-dataset correlation 也过：Fashion-MNIST = `0.9997405389081275`，KMNIST = `0.9998186619484812`，MNIST = `0.9999851067179271`。
7. P3 exact transfer score 未过：best = `T4-core-safe-transfer`，TopK87 precision = `0.20689655172413793`，V LCB = `-0.22226101681921628`，longrisk UCB = `0.0`，LDO drop = `0.08045977011494251`；quality/value 不足。
8. P4 Core77 + exact transfer expansion 未过：best = `T2-SNR-transfer`，accepted = `87`，GradeAB precision = `0.8850574712643678`，V LCB = `0.0933837988474282`，risk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.39080459770114945`。
9. P5 exact-vs-old/proxy 比较未过：old R8A/R5B TopK87 仍是 high-quality diagnostic，precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，但 LDO drop = `0.37931034482758624`。
10. P5 proxy TransferLCB 诊断为 low-LDO / low-quality：TopK87 precision = `0.09195402298850575`，V LCB = `-0.24289252733948727`，LDO drop = `0.04597701149425287`。
11. P5 exact best 比 proxy precision 略高，但仍不可用：best exact = `exact-T4-core-safe`，TopK87 precision = `0.20689655172413793`，V LCB = `-0.22226101681921628`，LDO drop = `0.08045977011494251`，exact useful/strong pass = `0 / 0`。
12. P6 dataset shift 未解：best score = `T4-core-safe-transfer`，PSI mean = `0.05010046451625646`，但 per-dataset precision min = `0.10344827586206896`，macro V LCB = `-0.32560930582486725`。
13. P7 direct transfer-solved update gate-blocked：reason = `P2_or_P3_weak_pass_failed`，generated actions = `0`，branch-horizon rows = `0`。
14. P8-P12 gate-blocked：existing-action controller、selected runtime、system controller、paired replay、short/full boundary 均未打开。
15. P9 generated route 继续停止：`generated_route_status = stopped_no_new_objective`，APGU/APGV/APGW allowed = `0`。
16. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`exact_transfer_not_better_than_proxy`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9720_exact_transfer_materializer_core_expansion_direct_update.py` | v9.7.2 runner；读取 v9.7.1/v9.7.0/v9.5.5-v9.5.8/v9.3.3 artifacts，执行 exact preflight、full exact linear transfer materializer、exact apply audit、exact score/core-expansion/direct-update boundaries 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9720_exact_transfer_materializer_core_expansion_direct_update.py
```

正式运行：

```bash
python experiments/run_v9720_exact_transfer_materializer_core_expansion_direct_update.py \
  --out-dir results/real_rerun_20260506/v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "best_TopK87_precision": 0.20689655172413793,
  "best_core_expansion_precision": 0.8850574712643678,
  "best_score_id": "T4-core-safe-transfer",
  "direct_update_generated_actions": 0,
  "exact_linear_transfer_row_count": 184064,
  "linear_apply_correlation": 0.9997818368443041,
  "out_dir": "results/real_rerun_20260506/v9720_exact_transfer_materializer_core_expansion_direct_update_first_20260516T010000Z",
  "primary_blocker": "exact_transfer_not_better_than_proxy",
  "route": "R3-ExactTransferNotBetterThanProxy",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 fake rows；v9.7.0 proxy transfer 只作为诊断基线，没有提升为 official exact transfer。

## 2. Route

`route_decision_v9720.json` 摘要：

```json
{
  "route": "R3-ExactTransferNotBetterThanProxy",
  "source_route_v9710": "R1-ExactTransferArtifactMissing",
  "p0_boundary_pass": 1,
  "exact_transfer_preflight_pass": 1,
  "P1_exact_materializer_strong_pass": 1,
  "P1_exact_materializer_weak_pass": 1,
  "exact_linear_transfer_row_count": 184064,
  "P2_apply_strong_pass": 1,
  "P2_apply_weak_pass": 1,
  "linear_apply_correlation": 0.9997818368443041,
  "linear_apply_sign_match_rate": 0.961181640625,
  "P3_score_strong_pass": 0,
  "P3_score_weak_pass": 0,
  "best_score_id": "T4-core-safe-transfer",
  "best_TopK87_precision": 0.20689655172413793,
  "best_TopK87_LDO_drop": 0.08045977011494251,
  "P4_core_expansion_pass": 0,
  "best_core_expansion_score_id": "T2-SNR-transfer",
  "best_core_expansion_precision": 0.8850574712643678,
  "best_core_expansion_LDO_drop": 0.39080459770114945,
  "P5_exact_better_than_proxy": 0,
  "exact_transfer_fail": 1,
  "dataset_shift_solved": 0,
  "direct_update_weak_pass": 0,
  "direct_update_strong_pass": 0,
  "generated_route_status": "stopped_no_new_objective",
  "APGU_APGV_APGW_allowed": 0,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "exact_transfer_not_better_than_proxy",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.2 解决了 v9.7.1 的 artifact missing blocker，并证明 exact linear response 对 exact apply 是高度可预测的；但 exact transfer score 作为 selector/ranker 的质量很低，core expansion 虽能补到 87 且保持 clean risk，却无法改善 LDO。因此 route 停在 `R3-ExactTransferNotBetterThanProxy`。

## 3. P0 boundary 与 exact preflight

Artifacts：

```text
p0_boundary_reproduction_v9720.csv
p0_field_legality_audit_v9720.csv
p0_exact_transfer_preflight_1action_v9720.csv
```

Boundary summary：

```text
source_route_v9710 = R1-ExactTransferArtifactMissing
system_legal_controller_pass_v9710 = 0
generated_route_status_v9710 = stopped_no_new_objective
exact_per_sample_gradient_available_v9710 = 0
exact_apply_checkpoint_available_v9710 = 0
exact_linear_transfer_row_count_v9710 = 0
field green/yellow/red = 14 / 4 / 0
boundary_reproduction_pass = 1
```

Field legality summary：

```text
field_legality_pass = 1
uses_dataset_name_for_selector/controller = 0 / 0
uses_outcome_derived_field = 0
uses_validation_or_test = 0
fake/proxy/cpu_offload = 0 / 0 / 0
```

Exact preflight summary：

```text
preflight_action_count = 1
preflight_sample_count = 8
exact_linear_transfer_row_count = 8
completion_rate = 1.0
per_example_gradient_available = 1
missing_gradient_count = 0
missing_payload_count = 0
nan/inf/duplicate = 0 / 0 / 0
gradient_compute_ms_q50/q90 = 0.35408465191721916 / 0.43701985850930214
response_linear_mean = -4.781055866664996e-05
memory_mb_peak = 76.7431640625
exact_transfer_preflight_pass = 1
```

判断：P0 pass。v9.7.2 没有跳过 v9.7.1 boundary；同时真正打开了 exact per-example gradient preflight。

## 4. P1 exact linear transfer materializer

Artifact：

```text
p1_exact_linear_transfer_materializer_v9720.csv
```

Summary：

```text
action_count_requested/materialized = 2876 / 2876
sample_count_per_action = 64
exact_linear_transfer_row_count = 184064
expected_row_count = 184064
completion_rate = 1.0
missing_gradient_count = 0
missing_payload_count = 0
nan_count/inf_count/duplicate_row_count = 0 / 0 / 0
compute_ms_q50/q90 = 0.3252611495554447 / 0.3429688513278961
memory_mb_peak = 2398.28466796875
wallclock_sec = 90.75305555388331
P1a/P1b/P1c rows = 256 / 8192 / 184064
P1a/P1b/P1c pass = 1 / 1 / 1
P1_exact_materializer_strong/weak = 1 / 1
```

Dataset row distribution：

```text
Fashion-MNIST = 78464
KMNIST = 52992
MNIST = 52608
unique_actions = 2876
unique_check_groups = 1412
response_linear min/max = -0.21950623393058777 / 0.24883460998535156
response_linear mean/std = 0.0006200237920500581 / 0.008732016768161524
```

判断：P1 是本轮关键推进。v9.7.1 的 exact artifact missing 不再成立；exact transfer rows 已经完整落盘。

## 5. P2 exact apply audit

Artifact：

```text
p2_exact_apply_audit_subset_v9720.csv
```

Summary：

```text
exact_apply_action_count = 64
exact_apply_sample_count = 4096
linear_apply_correlation = 0.9997818368443041
linear_apply_spearman = 0.9978941645960946
linear_apply_sign_match_rate = 0.961181640625
linear_apply_mae = 6.0028745734372246e-05
exact_apply_compute_ms_q90 = 0.006908252544235438
wallclock_sec = 12.409300355240703
P2_apply_strong/weak = 1 / 1
```

Per-dataset correlation：

```text
Fashion-MNIST = 0.9997405389081275
KMNIST = 0.9998186619484812
MNIST = 0.9999851067179271
```

判断：P2 pass。linear transfer approximation 本身不是本轮失败原因；它几乎完美预测 exact apply audit subset。

## 6. P3 exact transfer score definitions

Artifact：

```text
p3_exact_transfer_score_definitions_v9720.csv
```

Summary：

```text
score_count = 5
evaluation_row_count = 20
score_strong_pass_count = 0
score_weak_pass_count = 0
best_score_id = T4-core-safe-transfer
best_TopK87_precision = 0.20689655172413793
best_TopK87_V_LCB = -0.22226101681921628
best_TopK87_longrisk_UCB = 0.0
best_TopK87_LDO_drop = 0.08045977011494251
P3_score_strong/weak = 0 / 0
```

TopK87 representative rows：

| score | precision | V LCB | longrisk UCB | bad/null UCB | memory/offdiag UCB | LDO |
|---|---:|---:|---:|---:|---:|---:|
| `T0-mean-transfer` | `0.05747126436781609` | `-1.1048019435793504` | `0.7763316932778618` | `0.3183007985873107 / 0.0` | `0.9433993751492317 / 0.9433993751492317` | `0.022988505747126436` |
| `T1-LCB-transfer` | `0.04597701149425287` | `-1.219124487767872` | `0.8180568506350425` | `0.3824481159806525 / 0.0` | `0.9433993751492317 / 0.9433993751492317` | `0.022988505747126436` |
| `T2-SNR-transfer` | `0.05747126436781609` | `-1.2407543562624466` | `0.7763316932778618` | `0.3950403220387326 / 0.03389313880385762` | `0.9433993751492317 / 0.9433993751492317` | `0.022988505747126436` |
| `T3-sign-agreement-transfer` | `0.034482758620689655` | `-1.1427300888230005` | `0.8283004628835223` | `0.3052085391932784 / 0.03389313880385762` | `0.8782246978377869 / 0.897499817714006` | `0.011494252873563218` |
| `T4-core-safe-transfer` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.0 / 0.0` | `0.0 / 0.0` | `0.08045977011494251` |

判断：P3 未过。exact transfer score 能通过 `T4` 把 risk/bad/null/memory/offdiag 压到 0，但 precision 和 V LCB 不够；其它 transfer scores 质量更差且 high-risk/high-memory/offdiag。

## 7. P4 Core77 + exact transfer expansion

Artifact：

```text
p4_core77_exact_transfer_expansion_v9720.csv
```

Summary：

```text
candidate_count = 4
core_count = 77
needed_expansion_to_87 = 10
strong_pass_count = 0
weak_pass_count = 0
best_score_id = T2-SNR-transfer
best_accepted_count = 87
best_GradeAB_precision = 0.8850574712643678
best_V_integrated_LCB = 0.0933837988474282
best_longrisk_UCB = 0.0
best_LDO_drop = 0.39080459770114945
P4_core_expansion_pass = 0
```

Candidate rows：

| score | accepted | precision | V LCB | risk/bad/null/memory/offdiag UCB | LDO | expansion dataset distribution |
|---|---:|---:|---:|---:|---:|---|
| `T1-LCB-transfer` | `87` | `0.8850574712643678` | `0.07843570471174309` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `{"Fashion-MNIST": 2, "KMNIST": 1, "MNIST": 7}` |
| `T2-SNR-transfer` | `87` | `0.8850574712643678` | `0.0933837988474282` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `{"Fashion-MNIST": 1, "KMNIST": 2, "MNIST": 7}` |
| `T3-sign-agreement-transfer` | `87` | `0.8850574712643678` | `0.08168995019910179` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `{"KMNIST": 1, "MNIST": 9}` |
| `T4-core-safe-transfer` | `87` | `0.8850574712643678` | `0.07843570471174309` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `{"Fashion-MNIST": 2, "KMNIST": 1, "MNIST": 7}` |

判断：exact transfer 能补齐 Core77 到 87，并保持 risk/bad/null/memory/offdiag clean；但 LDO drop 没有改善，V LCB 也低于旧 high-quality rank diagnostic。因此不能打开 controller。

## 8. P5 exact transfer vs old/proxy

Artifact：

```text
p5_exact_transfer_vs_old_proxy_v9720.csv
```

Summary：

```text
method_count = 6
old_rank_precision = 0.8735632183908046
old_rank_LDO_drop = 0.37931034482758624
proxy_transfer_precision = 0.09195402298850575
proxy_transfer_LDO_drop = 0.04597701149425287
best_exact_method = exact-T4-core-safe
best_exact_precision = 0.20689655172413793
best_exact_V_LCB = -0.22226101681921628
best_exact_longrisk_UCB = 0.0
best_exact_LDO_drop = 0.08045977011494251
exact_transfer_strong_pass = 0
exact_transfer_useful_diagnostic = 0
exact_transfer_fail = 1
P5_exact_better_than_proxy = 0
```

Representative TopK87 rows：

| method | precision | V LCB | longrisk UCB | LDO |
|---|---:|---:|---:|---:|
| `old-R8A/R5B` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `proxy-TransferLCB` | `0.09195402298850575` | `-0.24289252733948727` | `0.12221253967001162` | `0.04597701149425287` |
| `exact-T1-LCB` | `0.04597701149425287` | `-1.219124487767872` | `0.8180568506350425` | `0.022988505747126436` |
| `exact-T2-SNR` | `0.05747126436781609` | `-1.2407543562624466` | `0.7763316932778618` | `0.022988505747126436` |
| `exact-T3-sign` | `0.034482758620689655` | `-1.1427300888230005` | `0.8283004628835223` | `0.011494252873563218` |
| `exact-T4-core-safe` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.08045977011494251` |

判断：P5 未过。exact transfer materializer 是真实的，linear/apply audit 也是强的，但 exact scores 仍没有形成比旧 rank 或 proxy 更有部署价值的 selector。

## 9. P6 exact transfer dataset shift audit

Artifact：

```text
p6_exact_transfer_dataset_shift_audit_v9720.csv
```

Summary：

```text
score_id = T4-core-safe-transfer
dataset_count = 3
PSI_mean = 0.05010046451625646
per_dataset_precision_min = 0.10344827586206896
per_dataset_V_LCB_min = -0.4860695981244332
macro_V_LCB = -0.32560930582486725
dataset_shift_solved = 0
dataset_shift_unresolved = 1
```

Per dataset：

| dataset | TopK count | precision | V LCB | longrisk UCB | PSI |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | `87` | `0.39080459770114945` | `-0.4860695981244332` | `0.33129982990159157` | `0.08935498198854534` |
| KMNIST | `87` | `0.13793103448275862` | `-0.22349464525569646` | `0.0` | `0.046189123413228866` |
| MNIST | `87` | `0.10344827586206896` | `-0.26726367409447216` | `0.0` | `0.014757288146995172` |

判断：P6 未过。Exact transfer `T4` 降低了 PSI，但 dataset-level quality/value 崩，不能说明 dataset shift solved。

## 10. P7-P12 boundary

Artifacts：

```text
p7_direct_transfer_solved_small_update_v9720.csv
p8_minimal_existing_action_controller_v9720.csv
p9_selected_runtime_preflight_v9720.csv
p10_system_boundary_v9720.csv
p11_paired_replay_boundary_v9720.csv
p12_short_full_boundary_v9720.csv
allowed_next_gates_v9720.json
stop_conditions_v9720.json
```

Status：

```text
P7 direct update = not_run
reason = P2_or_P3_weak_pass_failed
generated_action_count = 0
branch_horizon_rows = 0
direct_update_weak/strong = 0 / 0

P8 controller = not_run
reason = P4_or_P5_weak_pass_failed
controller_pass = 0

P9 selected runtime = not_run
reason = P8_controller_not_selected
selected_runtime_pass = 0

P10 system = not_run
reason = P8_or_P9_not_passed
system_legal_controller_pass = 0

P11 paired replay = not_run
reason = P10_system_not_official
paired_replay_pass = 0

P12 short/full = not_run
reason = P11_paired_replay_not_open
short/full open = 0 / 0
```

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "generated_route_allowed": 0,
  "APGU_APGV_APGW_allowed": 0
}
```

Stop conditions：

```json
{
  "fix_materializer": 0,
  "stop_transfer_route": 1,
  "stop_generated_blind_variants": 1,
  "enter_system": 0
}
```

判断：Direct update 没有打开，因为 P3 exact score weak pass 失败；generated route 也继续停止，不允许 APGU/APGV/APGW blind run。

## 11. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9720.csv
```

Summary：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
sentinel_complete = 1
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 12. Figures

本轮额外落盘的诊断图：

```text
fig_p0_response_histogram.svg
fig_p0_grad_compute_time.svg
fig_p1_response_linear_histogram.svg
fig_p1_compute_time_by_action_size.svg
fig_p2_linear_apply_corr_by_dataset.svg
fig_p3_score_precision_topk87.svg
fig_p4_core_expansion_precision.svg
fig_p5_method_precision_ldo.svg
fig_p6_per_dataset_precision.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 13. No-fake audit

```text
rows_checked = 188349
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
v9710_boundary_pass = 1
field_legality_pass = 1
exact_preflight_pass = 1
exact_materializer_pass = 1
exact_apply_pass = 1
core_expansion_pass = 0
direct_update_pass = 0
controller/runtime/system = 0/0/0
generated_route_stop = 1
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
proxy_transfer_promoted_to_official = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R3-ExactTransferNotBetterThanProxy
F0_exact_transfer_preflight_missing = 0
F1_exact_transfer_materializer_incomplete = 0
F2_linear_transfer_not_predictive = 0
F3_exact_transfer_not_better_than_proxy = 1
F4_exact_transfer_quality_low = 0
F5_core_expansion_solved = 0
F6_direct_update_promising = 0
F7_system_pass = 0
F8_system_not_official = 1
F9_base_acc_catastrophic = 0
primary_blocker = exact_transfer_not_better_than_proxy
secondary_blocker = generated_route_stopped_no_new_objective
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `ee78219cf96fd9c45c5aab0e2fd3177f1ef5c3c0b212ccac43322338d14ef91e` |
| runner | `1b08797eab9b61899e91fbc45131a7559145953b3d549303b719d74513760873` |
| run manifest | `e54a8449fd415d1d34899d8971bd9856e36b9696be865d4c003994bb741dd517` |
| route | `538d8066ec9dbd7190659457f0c8dc317f3cd62181bb14fd44fd3b46b326afdb` |
| P0 boundary | `d281501d9be7d5e67e893332397e422342e6966781af817fb06237979da3caea` |
| P0 field legality | `951a7d7ba0c90b1373b9f496228e753d0c97bdffc347cd568fcaedb498734126` |
| P0 preflight | `d6f1440eeb9f0b4d6c6d55ec5a2dc2a63d2f1aa213bcb9a0af6513fc174e4f2f` |
| P1 exact materializer | `ad87ca67929eb1ca675c7b0a403bce7132552f664b2bf0636a0ee4c5d48aa9c9` |
| P2 exact apply audit | `91e12782defbaf8969c1020cd44bd006844d89326cd9db507d3eaa2b294a5181` |
| P3 score definitions | `135c81c02bc292b3a8e31aab4ceca3526e6c1a6de388ba890f074a7bbeb66aa8` |
| P4 core expansion | `ee0c663de25ce41513d26b02a639749a3be9a0c1fdce9f53c8c5b1a0a07cb5c0` |
| P5 exact vs old/proxy | `80fd4515c9c81d3f58690f44db536a351a63bbcfa1821a73e59647317506fe73` |
| P6 dataset shift | `c7270027398192b527954e452e31254989a8409f7220ce502b839e9c5e4a96ef` |
| P7 direct update | `72e7461232b9bcc045b97e479f22d1dba9ba08b48d48c9ab7d26b5abf04db4f2` |
| P8 controller | `3d40a55b8f885d8485113a1ef2ecd11d1a696639257437facd0d8ff748160a9a` |
| P9 runtime | `2626ec89f379bcd465bcd32817b1a0933b70231b5599f75b206f4132bada5c1c` |
| P10 system | `71a91b56b40c210d03cf23a0a2121f53e9ae9fcec621ee05fe8c4fb19c534629` |
| P11 paired replay | `083d894bff378fc960e238509ee8ea15829950edf01c9269e670acdddff68e2b` |
| P12 short/full | `c0231fb1792773674394f8343badef32cb56d0685c8d801ab0c796c099c4af84` |
| Base-Acc Sentinel | `874ac7a9f47142657710e3621f7044c5ffe08839e75bf6c54df73766bdaa16bd` |
| no-fake audit | `eb3f20ef43c0db2996fb8ec31698e8f7a059a0758dd22849b5885f76ead4f4f9` |
| contract audit | `8ff0048b4b70eacd6c8c52baa41b42654f7a1c3d0bd16d4b779d604c53f9b240` |
| failure taxonomy | `c6246d7deba5260c9fff11c5012d2b757b00d5980e64678d625a9ba407574165` |
| allowed next gates | `9b7665e62a6eeb1cab276eea3b5b38dffa2d3b8c61405b3ea74ca0c8058ec3dc` |
| stop conditions | `9c6130fabc0b86714fc3d18997337d0d5bb0097710d379dfa5e9d9cf652c041f` |

## 15. 最终分析结论

v9.7.2 的真实推进是：

```text
v9.7.1:
  exact per-sample gradient / exact apply checkpoint 未落地；
  exact selector、core+exact expansion、direct solved update 全部 gate-blocked；
  generated route 继续 no-new-objective stop。

v9.7.2:
  成功实现 exact per-example gradient preflight；
  对 2876 个 AP0 actions x 64 samples 完整 materialize exact linear response；
  用 4096 行 exact apply audit 证明 linear response 和 exact apply 高度一致；
  但 exact transfer score 本身选择出来的是 low-quality / value-negative region；
  Core77 + exact expansion 能补到 87 且保持 clean risk，但 LDO 仍高；
  direct update 因 exact score/core-expansion weak pass 失败而不打开；
  controller/runtime/system/paired replay 均未打开。
```

机制判断：

1. H0 成立：v9.7.1 boundary 被复现，没有跳过 exact artifact missing / generated stop / no system controller。
2. H1 成立：exact transfer materializer 已落地，full rows = `184064`，P1 strong/weak pass = `1 / 1`。
3. H2 成立：linear transfer 是 exact apply 的强 proxy，Pearson = `0.9997818368443041`，sign match = `0.961181640625`。
4. H3 未成立：exact transfer score 没有形成 deployable selector；best TopK87 precision 只有 `0.2069`，且 V LCB 为负。
5. H4 未成立：Core + exact expansion 补到 87 后仍 LDO = `0.3908`，不能 controller。
6. H5 未成立：exact transfer 没有提供比 old/proxy 更好的可用路线；old rank 高质量但 LDO 高，exact low-LDO 但 quality/value 低。
7. H6 未成立：dataset shift 没有 solved；T4 PSI mean 较低但 per-dataset precision/V 崩。
8. H7 未打开：direct transfer-solved update not_run，generated actions = `0`。
9. H8-H10 未打开：没有 controller/runtime/system pass，就不能打开 paired replay 或 short/full training。
10. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.2 真实执行后停在 `R3-ExactTransferNotBetterThanProxy`：exact transfer materializer 和 linear/apply audit 首次完整闭合，说明 v9.7.1 的 artifact missing 已解决；但 exact transfer score 选出的区域仍 low-quality / value-negative，Core77 exact expansion 也未解决 LDO，因此 strict PureKAN functional 仍未成功。
