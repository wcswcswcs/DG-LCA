# DG-KAN v9.7.4 ExistingAction LDO / Transfer Mismatch / Horizon Transfer 实验复盘

> 本复盘记录 `DG-KAN_v9.7.4_ExistingActionLDO_TransferMismatch_HorizonTransfer_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 exact-transfer mismatch diagnostic、Core77/Expansion10 diagnostic、dataset-shift stabilization diagnostic、horizon-transfer diagnostic、controller/runtime boundary、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-ExistingActionLDOBlocked
base_candidate = LQ-t2-h256
success_v9740_strict_purekan_functional = False
success_v9740_full_functional = False
success_v9740_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z/
```

核心结论：

1. P0 复现 v9.7.3 boundary：source route = `R3-ExistingActionStillLDOBlocked`，OldRankTop87 precision = `0.8735632183908046`，ExactT4Top87 precision = `0.20689655172413793`，E1 precision = `0.8850574712643678`，E1 LDO drop = `0.39080459770114945`，generated route = `stopped_no_new_objective`。
2. P1 exact-transfer mismatch audit 过：OldOnly precision = `0.855072463768116`，ExactOnly precision = `0.014492753623188406`，Intersection precision = `0.9444444444444444`。
3. P1 failure mode 仍是 transfer objective mismatch：Exact transfer 低 LDO 不是稳定好动作，而是把大量 OldRank 高价值动作排掉。
4. P2 Core77 + Expansion10 未过：Core77 precision = `1.0`，V LCB = `0.16402807605399972`，但 Core77 自身 LDO drop = `0.4415584415584416`。
5. P2 best = `C0+E1-87`，accepted = `87`，precision = `0.8850574712643678`，V LCB = `0.14120469391749882`，risk/bad/null/memory/offdiag UCB 全为 `0`，但 LDO drop = `0.39080459770114945`。
6. P2 证明 Expansion10 不是主因：`E1-old-rank-top10` 单独 LDO drop = `0.0`，但 Core77 已 LDO fail，因此 route 不能靠换 expansion10 解决。
7. P3 dataset-shift non-tuning 未过：best 仍是 `S0-raw-old-rank`，precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，LDO drop = `0.37931034482758624`。
8. P3 低 LDO 方案会破坏质量：`S5-exact-core-safe` LDO drop = `0.08045977011494251`，但 precision = `0.20689655172413793`，V LCB = `-0.22226101681921628`。
9. P4 horizon transfer materializer 真实执行：subset actions = `198`，windows = `1,5,20,80`，check samples/action = `32`，windowed action rows = `198`，nan/inf = `0 / 0`，materializer pass = `1`。
10. P4 horizon score 未过：best = `WT80-LCB`，TopK87 precision = `0.47126436781609193`，V LCB = `-0.10386482729963209`，longrisk UCB = `0.0`，LDO drop = `0.13793103448275862`；weak/strong pass = `0 / 0`。
11. P5 old-rank mechanism ablation 未形成可生成机制：old rank precision = `0.8735632183908046`，OldOnly good count = `59`，但 `P5_old_rank_mechanism_explained = 0`，`P5_generative_mechanism_ready = 0`。
12. P6/P7/P9/P10 gate-blocked：existing-action controller、selected runtime、paired replay、short/full boundary 均 not_run。
13. P8 generated route 继续停止：`generated_route_status = stopped_no_new_objective`，APGU/APGV/APGW/APGX run = `0`，direct solved sandbox allowed = `0`。
14. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
15. 当前 primary blocker：`existing_action_ldo_blocked`；secondary blocker：`transfer_objective_mismatch`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer.py` | v9.7.4 runner；读取 v9.7.3/v9.7.2/v9.7.0/v9.6.8/v9.6.6 artifacts，执行 OldRank/ExactTransfer mismatch、Core77/Expansion10 LDO responsibility、dataset-shift non-tuning、horizon transfer materializer、old-rank mechanism ablation 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer.py
```

正式运行：

```bash
python experiments/run_v9740_existing_action_ldo_transfer_mismatch_horizon_transfer.py \
  --out-dir results/real_rerun_20260506/v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "C0E1_LDO_drop": 0.39080459770114945,
  "Core77_LDO_drop": 0.4415584415584416,
  "best_horizon_precision": 0.47126436781609193,
  "best_horizon_score": "WT80-LCB",
  "oldonly_precision": 0.855072463768116,
  "out_dir": "results/real_rerun_20260506/v9740_existing_action_ldo_transfer_mismatch_horizon_transfer_first_20260516T030000Z",
  "primary_blocker": "existing_action_ldo_blocked",
  "route": "R2-ExistingActionLDOBlocked",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 fake rows；v9.7.0 proxy transfer 仍只作为 diagnostic baseline，没有提升为 official exact/windowed transfer。

## 2. Route

`route_decision_v9740.json` 摘要：

```json
{
  "route": "R2-ExistingActionLDOBlocked",
  "source_route_v9730": "R3-ExistingActionStillLDOBlocked",
  "p0_pass": 1,
  "H1_transfer_mismatch_pass": 1,
  "oldonly_precision": 0.855072463768116,
  "exactonly_precision": 0.014492753623188406,
  "Core77_LDO_drop": 0.4415584415584416,
  "C0E1_LDO_drop": 0.39080459770114945,
  "P2_core_expansion_ldo_pass": 0,
  "P3_dataset_shift_non_tuning_pass": 0,
  "P4_horizon_transfer_weak_pass": 0,
  "P4_horizon_transfer_strong_pass": 0,
  "best_horizon_score_id": "WT80-LCB",
  "best_horizon_precision": 0.47126436781609193,
  "best_horizon_V_LCB": -0.10386482729963209,
  "best_horizon_longrisk_UCB": 0.0,
  "P5_old_rank_mechanism_explained": 0,
  "P5_generative_mechanism_ready": 0,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_status": "stopped_no_new_objective",
  "direct_solved_sandbox_allowed": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "existing_action_ldo_blocked",
  "secondary_blocker": "transfer_objective_mismatch"
}
```

判断：v9.7.4 的 route 不是 exact materializer regression，也不是 windowed materializer failure。它说明 existing-action side 的高质量 clean-risk region 仍被 LDO block，且 blocker 主要来自 Core77 本身；transfer/windowed low-LDO signal 仍以 precision/value 代价换来，不能 official。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9740.csv
p0_field_legality_audit_v9740.csv
```

Summary：

```text
source_route_v9730 = R3-ExistingActionStillLDOBlocked
OldRankTop87_precision = 0.8735632183908046
ExactT4Top87_precision = 0.20689655172413793
OldExactIntersection_count = 18
OldOnly_count = 69
ExactOnly_count = 69
Core77_count = 77
E1_accepted_count = 87
E1_precision = 0.8850574712643678
E1_V_LCB = 0.14120469391749882
E1_LDO_drop = 0.39080459770114945
WT20_precision = 0.08045977011494253
WT20_V_LCB = -1.0243587575297828
generated_route_status = stopped_no_new_objective
field green/yellow/red = 14 / 4 / 0
dataset_name_used_in_controller = 0
outcome_derived_field_used_in_controller = 0
p0_pass = 1
```

判断：P0 pass。v9.7.4 没有跳过 v9.7.3 的 existing-action LDO / windowed transfer / generated stop boundary，也没有把 forbidden fields 带入 official path。

## 4. P1 OldRank / ExactTransfer mismatch

Artifact：

```text
p1_old_exact_transfer_mismatch_v9740.csv
p1_old_exact_transfer_mismatch_actions_v9740.csv
```

Summary：

```text
intersection_count = 18
oldonly_count = 69
exactonly_count = 69
intersection_precision = 0.9444444444444444
oldonly_precision = 0.855072463768116
oldonly_V_LCB = 0.13450092269587752
oldonly_longrisk_UCB = 0.0
oldonly_dataset_count = 3
oldonly_template_count = 69
exactonly_precision = 0.014492753623188406
exactonly_V_LCB = -0.3037141101011471
exactonly_longrisk_UCB = 0.0
H1_old_rank_not_explained_by_one_step_transfer_pass = 1
transfer_objective_mismatch = 1
```

Representative sets：

| set | count | precision | V LCB | longrisk UCB | LDO drop |
|---|---:|---:|---:|---:|---:|
| `Intersection` | `18` | `0.9444444444444444` | `0.0866484084297124` | `0.0` | `0.0` |
| `OldOnly` | `69` | `0.855072463768116` | `0.13450092269587752` | `0.0` | `0.37681159420289856` |
| `ExactOnly` | `69` | `0.014492753623188406` | `-0.3037141101011471` | `0.0` | `0.08695652173913046` |
| `OldRankTop87` | `87` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `ExactT4Top87` | `87` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.08045977011494251` |

判断：P1 是 exact-transfer side 的硬结论。Exact transfer 确实降低 LDO，但 ExactOnly 基本不是好动作；OldOnly 反而保持高 precision/value。因此 exact transfer 不能作为 old rank 的辅助 gate 或 official selector。

## 5. P2 Core77 / Expansion10 LDO responsibility

Artifacts：

```text
p2_core_expansion_ldo_responsibility_v9740.csv
p2_expansion10_ldo_detail_v9740.csv
```

Summary：

```text
set_count = 6
core_count = 77
need_expansion = 10
Core77_precision = 1.0
Core77_V_LCB = 0.16402807605399972
Core77_LDO_drop = 0.4415584415584416
C0E1_precision = 0.8850574712643678
C0E1_V_LCB = 0.14120469391749882
C0E1_LDO_drop = 0.39080459770114945
core77_ldo_stable = 0
expansion10_primary_ldo_cause = 0
core77_itself_ldo_fail = 1
P2_core_expansion_ldo_pass = 0
```

Sets：

| set | accepted | precision | V LCB | risk/bad/null/memory/offdiag UCB | LDO drop | coverage pass |
|---|---:|---:|---:|---|---:|---:|
| `C0-Core77` | `77` | `1.0` | `0.16402807605399972` | `0 / 0 / 0 / 0 / 0` | `0.4415584415584416` | `0` |
| `E1-old-rank-top10` | `10` | `0.0` | `-0.017804270882284985` | `0 / 0 / 0 / 0 / 0` | `0.0` | `0` |
| `C0+E1-87` | `87` | `0.8850574712643678` | `0.14120469391749882` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `1` |
| `C0+ExactT4-top10` | `87` | `0.8850574712643678` | `0.07843570471174309` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `1` |
| `C0+Intersection-first10` | `78` | `0.9871794871794872` | `0.16141672983734479` | `0 / 0 / 0 / 0 / 0` | `0.4358974358974359` | `0` |
| `C0+random-safe-diagnostic10` | `87` | `0.8850574712643678` | `0.10039704781884091` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `1` |

判断：P2 是本轮 primary blocker。Core77 本身已经 LDO fail，Expansion10 只是补 coverage；换成 exact-assisted 或 random-safe diagnostic expansion 也没有解决 LDO。

## 6. P3 dataset-shift non-tuning

Artifact：

```text
p3_dataset_shift_non_tuning_v9740.csv
```

Summary：

```text
method_count = 6
pass_count = 0
best_method_id = S0-raw-old-rank
best_precision = 0.8735632183908046
best_V_LCB = 0.14111334880346277
best_LDO_drop = 0.37931034482758624
P3_dataset_shift_non_tuning_pass = 0
```

Representative methods：

| method | precision | V LCB | longrisk UCB | LDO drop | PSI mean | pass |
|---|---:|---:|---:|---:|---:|---:|
| `S0-raw-old-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.31313509863308747` | `0` |
| `S1-global-quantile-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.29633198091493074` | `0` |
| `S2-step-bucket-normalized-rank` | `0.4367816091954023` | `-0.0711492861192822` | `0.0` | `0.21839080459770116` | `0.17755938134102164` | `0` |
| `S3-core-anchor-rank` | `0.7701149425287356` | `0.07576755853602393` | `0.0` | `0.29885057471264365` | `0.3090428749753942` | `0` |
| `S5-exact-core-safe` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.08045977011494251` | `0.05010046451625646` | `0` |

判断：P3 未过。非调参、非 dataset dispatch 的稳定化策略没有打开新 tradeoff；低 LDO 仍对应 low-quality/value-negative region。

## 7. P4 horizon-coupled transfer

Artifacts：

```text
p4_horizon_transfer_cohort_summary_v9740.csv
p4_horizon_transfer_materializer_v9740.csv
p4_horizon_transfer_score_eval_v9740.csv
```

Materializer summary：

```text
subset_action_count_requested/materialized = 198 / 198
windows = 1,5,20,80
check_sample_count_per_action = 32
windowed_action_rows = 198
nan/inf = 0 / 0
compute_ms_q50/q90 = 34.50229810550809 / 39.56810291856527
memory_mb_peak = 2264.0283203125
wallclock_sec = 27.252162854652852
windowed_materializer_pass = 1
```

Cohort summary：

| cohort | actions | precision | V LCB | WT80 mean | LDO drop |
|---|---:|---:|---:|---:|---:|
| `Intersection` | `18` | `0.9444444444444444` | `0.0866484084297124` | `0.4535725401176785` | `0.38888888888888884` |
| `OldOnly` | `69` | `0.855072463768116` | `0.13450092269587752` | `0.010067179508415913` | `0.391304347826087` |
| `ExactOnly` | `69` | `0.014492753623188406` | `-0.3037141101011471` | `0.10113173453042902` | `0.0` |
| `Core77` | `77` | `1.0` | `0.16402807605399972` | `0.09892247041447147` | `0.4415584415584416` |
| `Expansion10` | `10` | `0.0` | `-0.017804270882284985` | `0.12481853988068489` | `0.0` |
| `RandomLegalSafeDiagnostic` | `42` | `0.0` | `-0.3749055343469755` | `0.0043356916247338185` | `0.0` |

Score summary：

```text
score_count = 4
best_score_id = WT80-LCB
best_precision = 0.47126436781609193
best_V_LCB = -0.10386482729963209
best_longrisk_UCB = 0.0
best_LDO_drop = 0.13793103448275862
weak_pass_count = 0
strong_pass_count = 0
P4_horizon_transfer_weak/strong = 0 / 0
```

Windowed TopK87：

| score | precision | V LCB | longrisk UCB | LDO drop | pass |
|---|---:|---:|---:|---:|---:|
| `WT1-LCB` | `0.25287356321839083` | `-0.19153059051332547` | `0.0` | `0.022988505747126464` | `0` |
| `WT5-LCB` | `0.3218390804597701` | `-0.15768636622807328` | `0.0` | `0.08045977011494251` | `0` |
| `WT20-LCB` | `0.28735632183908044` | `-0.17709475079801107` | `0.0` | `0.05747126436781608` | `0` |
| `WT80-LCB` | `0.47126436781609193` | `-0.10386482729963209` | `0.0` | `0.13793103448275862` | `0` |

判断：P4 materializer 过，但 selector 不过。WT80 比 v9.7.3 windowed score 更接近 gate，但 precision 仍低于 weak gate，且 V LCB 为负；不能打开 full/direct update 或 generated route。

## 8. P5 old-rank mechanism ablation

Artifact：

```text
p5_old_rank_mechanism_ablation_v9740.csv
```

Summary：

```text
component_count = 6
old_rank_precision = 0.8735632183908046
old_rank_V_LCB = 0.14111334880346277
old_rank_LDO_drop = 0.37931034482758624
oldonly_good_count = 59
oldonly_safe_clean_fraction = 1.0
oldonly_memory_offdiag_safe_fraction = 1.0
oldonly_exact_compatible_fraction = 1.0
oldonly_template_count = 59
oldonly_dataset_count = 3
P5_old_rank_mechanism_explained = 0
P5_generative_mechanism_ready = 0
P5_pass = 0
```

Components：

| component | precision | V LCB | longrisk UCB | LDO drop | field legality |
|---|---:|---:|---:|---:|---|
| `A0-old-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `green` |
| `A1-remove-risk-penalty` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `green` |
| `A2-remove-memory-penalty` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `green` |
| `A3-safe-only-old-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `green` |
| `A4-exact-compatible-old-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `yellow_diagnostic` |
| `A5-memory-offdiag-safe-only` | `0.10344827586206896` | `-0.30356449672984037` | `0.0` | `0.022988505747126436` | `green` |

判断：P5 没有把 old-rank 高质量现象转成可生成机制。安全约束本身不足以重构 old-rank；low LDO 的 `A5` 质量崩。

## 9. Controller / runtime / generated route / system boundary

P6：

```text
p6_existing_action_minimal_controller_v9740.csv = not_run
reason = P2_P3_P4_P5_no_controller_open_gate
controller_pass = 0
source_controller_pass = 0
```

P7：

```text
p7_selected_runtime_boundary_v9740.csv = not_run
reason = P6_controller_not_selected
selected_runtime_pass = 0
```

P8：

```text
p8_generated_route_stop_reopen_rule_v9740.csv
generated_route_status = stopped_no_new_objective
APGU_APGV_APGW_APGX_run = 0
direct_solved_sandbox_allowed = 0
generated_action_count = 0
generated_route_stop_triggered = 1
```

P9-P10：

| artifact | status / reason |
|---|---|
| `p9_paired_replay_boundary_v9740.csv` | `not_run`, `P6_or_P7_not_passed` |
| `p10_short_full_boundary_v9740.csv` | `not_run`, `P9_paired_replay_not_open` |

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "generated_route_allowed": 0,
  "APGU_APGV_APGW_APGX_run_allowed": 0,
  "direct_solved_sandbox_allowed": 0
}
```

Stop conditions：

```json
{
  "enter_system": 0,
  "stop_exact_transfer_as_ranker": 1,
  "stop_generated_blind_variants": 1,
  "stop_threshold_tuning": 1
}
```

判断：没有 P2/P3/P4/P5 weak/strong pass，因此 controller/runtime/system/paired replay/short-full 全部关闭。Generated route 继续 no-new-objective stop，不允许 blind APG* 或 direct solved sandbox。

## 10. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9740.csv
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

## 11. Figures

本轮额外落盘诊断图：

```text
fig_p1_cohort_precision.svg
fig_p2_set_ldo.svg
fig_p3_non_tuning_precision.svg
fig_p4_horizon_precision.svg
fig_p5_component_precision.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 12. No-fake audit

```text
rows_checked = 552
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
v9730_boundary_pass = 1
transfer_mismatch_audit_pass = 1
core_expansion_ldo_pass = 0
dataset_shift_non_tuning_pass = 0
horizon_transfer_pass = 0/0
old_rank_mechanism_explained = 0
controller/runtime/system = 0/0/0
generated_route_stop = 1
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
proxy_transfer_promoted_to_official = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R2-ExistingActionLDOBlocked
F0_boundary_fail = 0
F1_old_rank_exact_transfer_mismatch = 1
F2_core77_itself_ldo_fail = 1
F3_dataset_shift_non_tuning_fail = 1
F4_horizon_transfer_fail = 1
F5_controller_runtime_blocked = 1
F6_generated_route_stopped = 1
F7_system_not_official = 1
primary_blocker = existing_action_ldo_blocked
```

## 13. Hash

| artifact | SHA256 |
|---|---|
| plan | `a9f2bec085c09ba08fcbde79f64198104862b10af54da95cf523cbd3d6b4521a` |
| runner | `032083e48e53ee51327d9da933e829792469518baa9171bcf195ba303f043774` |
| run manifest | `bfb23f50678c2e68fdc46628153f9f9289264856d655e3101b30027112c132b7` |
| route | `9cf749a3caa28e99dff63710e3418fc45e93e3a59bc9e21bbb112914c454ad71` |
| P0 boundary | `b2365618924d207679fee75fcbc3969cb294613670808d3fbe533cd239ff8993` |
| P0 field legality | `68943e016e4b3e355e3580a04f79e38974a8858097232169e6208a504df1d66b` |
| P1 mismatch | `7310bebe3fea8c48f7fa9fa89a3b1c724ed0bfab0e553aafc17b0d3b13f5b0f8` |
| P1 action detail | `46d16e782820706e73d63664bd6d0efa452037f6bbc70d5c94d002efc76b6d4a` |
| P2 LDO responsibility | `f6a45f1561ccf4ce4e0d68fe411715eb054d7f468e7061f50dbaf1597634c857` |
| P2 expansion detail | `d8eff942c275f8c48cb058a159830f65943921ad36d2a17a47a02e5dba4b9a99` |
| P3 non-tuning | `b901bf30d8973a0d7b695abec46a4677d6b5e9499e9136448089a0fbca9c43dd` |
| P4 cohort summary | `1206b15f7b0b978f9f0e704eed2602d82504f69fd217d1d3c0ab28d6bcd445fb` |
| P4 materializer | `9030e42ba1439fa9aa11d79112a04c02896c3adc4ecf757fa3ec07061b7c2f02` |
| P4 score eval | `bb5701325794b5ab4704c227be4873562b4f03049729dd1896e968fa025b1364` |
| P5 ablation | `62afaf1d4782b09df8a975287e5c8d04bd28dc50e551fdd47552e811040bfdc1` |
| P6 controller | `73b40feebaf59ce25c20b3a929e1577d7ea47c73daab5ec0b9a2576632a0639b` |
| P7 runtime | `d28626103d191c6cbcd238e90809be2ca6c798b9db22d1ea5cd325a2476a796c` |
| P8 generated stop | `32515524eb3e48319adc2d271254edfa67cb3fef2c1434588c8a89921d293b0e` |
| P9 paired replay | `b74c8701fdbb15ca121ad64187845877be6aca483f15f7d2c236fc322f82b939` |
| P10 short/full | `1b76a0485e39deeaf002b4e3f60fbfc232b4a06d27b79a3968397cf81e2eab88` |
| Base-Acc Sentinel | `f6635041faa1c30e307ea3b46c6deb4e1b39cb563f33acfef12b1a0374fd0696` |
| no-fake audit | `27fb5f9678b4a11cb5589182e74f71f3fa3e3f8fcb235c033ce9e85f9395c2ce` |
| contract audit | `c928c70ac65e99de12fecefcfc5bb75c4dab1a870099e880af45150e64dc28e2` |
| failure taxonomy | `53c9ac7ae34b9989cb200678e5b5cf1339d22f5eb03c10c6243878583e0fbbcd` |
| allowed next gates | `f082f8dd02c7582daef83d9faaa7d26a31c7bc69c09ed5df61d871d91d706a41` |
| stop conditions | `dbd90d0f77f6bfd89de9a82ac3e5a809efd436f1c9549feca5d38de77828a91c` |

## 14. 最终分析结论

v9.7.4 的真实推进是：

```text
v9.7.3:
  exact transfer autopsy 证明 ExactOnly low-quality；
  Core77+10 仍 high precision / clean risk 但 LDO 高；
  windowed transfer 子集 materializer 过，WT scores 不过；
  generated route 继续 no-new-objective stop。

v9.7.4:
  进一步把 old-rank/exact-transfer mismatch 拆成 Intersection / OldOnly / ExactOnly；
  证明 OldOnly 是跨 dataset/template 的真实高质量动作，不是 exact transfer 能解释的低维交集；
  证明 Core77 本身已经 LDO fail，Expansion10 不是主因；
  non-tuning dataset-shift stabilization 仍无法同时保住 high precision/value 与 low LDO；
  horizon transfer materializer 真实闭合，但 WT80 仍未达到 weak gate，且 V LCB 为负；
  old-rank mechanism ablation 没有形成可生成机制；
  controller/runtime/generated route/paired replay 全部 gate-blocked。
```

机制判断：

1. H0 成立：v9.7.3 boundary 被复现，没有跳过 existing-action LDO / generated stop / no system controller。
2. H1 成立：exact transfer 不是可用 auxiliary selector；ExactOnly precision = `0.0145`，OldOnly precision = `0.8551`。
3. H2 未成立：Core77+Expansion10 不能解决 LDO；Core77 自身 LDO drop = `0.4416`。
4. H3 未成立：dataset-shift non-tuning 不能打开 stable controller；低 LDO 方法 precision/value 崩。
5. H4 未成立：horizon transfer WT80 是本轮最接近的 transfer score，但 precision = `0.4713` 且 V LCB = `-0.1039`，不能 weak pass。
6. H5 未成立：old-rank ablation 没有给出可生成机制或 controller-ready feature。
7. H6/H7 未打开：没有 weak controller candidate，因此 existing-action controller 与 selected runtime 均 not_run。
8. H8 成立：generated route 继续 stopped_no_new_objective；APGU/APGV/APGW/APGX allowed = `0`。
9. H9-H10 未打开：没有 system controller，就不能打开 paired replay 或 short/full training。
10. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.4 真实执行后停在 `R2-ExistingActionLDOBlocked`：exact transfer mismatch 已被复盘清楚，Core77 本身而不是 expansion10 是 LDO blocker；horizon transfer 虽真实 materialize 但仍未形成可用 selector，generated route 继续停止，因此 strict PureKAN functional 仍未成功。
