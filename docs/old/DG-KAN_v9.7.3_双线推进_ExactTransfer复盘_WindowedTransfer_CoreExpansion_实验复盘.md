# DG-KAN v9.7.3 双线推进 / Exact Transfer 复盘 / Windowed Transfer / Core Expansion 实验复盘

> 本复盘记录 `DG-KAN_v9.7.3_双线推进_ExactTransfer复盘_WindowedTransfer_CoreExpansion_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 exact-transfer autopsy diagnostic、Core77 expansion diagnostic、windowed transfer diagnostic、dataset-shift stabilization diagnostic、direct-update boundary、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R3-ExistingActionStillLDOBlocked
base_candidate = LQ-t2-h256
success_v9730_strict_purekan_functional = False
success_v9730_full_functional = False
success_v9730_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9730_dual_line_exact_transfer_windowed_transfer_core_expansion_first_20260516T020000Z/
```

核心结论：

1. P0 复现 v9.7.2 boundary：source route = `R3-ExactTransferNotBetterThanProxy`，exact rows = `184064`，exact apply rows = `4096`，linear/apply Pearson = `0.9997818368443041`，generated route = `stopped_no_new_objective`。
2. P1 exact transfer failure autopsy 说明 exact transfer 不是可用辅助 gate：OldRankTop87 precision = `0.8735632183908046`，ExactT4Top87 precision = `0.20689655172413793`。
3. P1 交集很小但质量高：OldExactIntersection count = `18`，precision = `0.9444444444444444`；但 ExactOnly count = `69`，precision = `0.014492753623188406`，V LCB = `-0.3037141101011471`。
4. P1 failure mode = `exact_transfer_kills_high_value_actions`：OldOnly count = `69`，precision = `0.855072463768116`，说明 exact T4 排序排掉了大量旧 rank 高价值动作。
5. P2 Core77 + expansion10 未过：best = `E1-old-rank-top10`，accepted = `87`，precision = `0.8850574712643678`，V LCB = `0.14120469391749882`，longrisk/bad/null/memory/offdiag UCB 全为 `0`，但 LDO drop = `0.39080459770114945`。
6. P2 所有 exact-assisted expansion 仍 LDO fail：`E2/E3/E4/E5` 的 LDO drop 均为 `0.39080459770114945`，P2 weak/strong pass = `0 / 0`。
7. P3 windowed transfer materializer 真实执行：subset actions = `384`，windows = `1,5,20`，check samples/action = `32`，windowed rows = `384`，nan/inf = `0 / 0`，materializer pass = `1`。
8. P3 windowed score 未过：best = `WT20-LCB`，TopK87 precision = `0.08045977011494253`，V LCB = `-1.0243587575297828`，longrisk UCB = `0.7868704900942779`，memory/offdiag UCB = `0.9069621813964702 / 0.9162956312609247`。
9. P3 因 subset score fail，full windowed materializer 按 gate `not_run`。
10. P4 dataset-shift stabilization 未过：best = `S0-old-rank`，precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，LDO drop = `0.37931034482758624`；低 LDO 策略 value/precision 崩。
11. P5/P6 gate-blocked：existing-action controller 与 direct windowed-transfer solved update 均 `not_run`。
12. P7 generated route 继续停止：`generated_route_status = stopped_no_new_objective`，APGU/APGV/APGW allowed = `0`，generated actions = `0`。
13. P8-P10 gate-blocked：selected runtime、paired replay、short/full boundary 均未打开。
14. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
15. 当前 primary blocker：`existing_action_ldo_blocked`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion.py` | v9.7.3 runner；读取 v9.7.2/v9.7.0/v9.6.8 与 canonical AP0 artifacts，执行 exact-transfer failure autopsy、Core77+expansion10、windowed transfer subset materializer、dataset-shift stabilization、direct-update/system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion.py
```

正式运行：

```bash
python experiments/run_v9730_dual_line_exact_transfer_autopsy_windowed_transfer_core_expansion.py \
  --out-dir results/real_rerun_20260506/v9730_dual_line_exact_transfer_windowed_transfer_core_expansion_first_20260516T020000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "best_expansion_precision": 0.8850574712643678,
  "best_expansion_strategy": "E1-old-rank-top10",
  "best_windowed_precision": 0.08045977011494253,
  "best_windowed_score": "WT20-LCB",
  "out_dir": "results/real_rerun_20260506/v9730_dual_line_exact_transfer_windowed_transfer_core_expansion_first_20260516T020000Z",
  "primary_blocker": "existing_action_ldo_blocked",
  "route": "R3-ExistingActionStillLDOBlocked",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 fake rows；v9.7.0 proxy transfer 只作为 diagnostic baseline，没有提升为 official exact/windowed transfer。

## 2. Route

`route_decision_v9730.json` 摘要：

```json
{
  "route": "R3-ExistingActionStillLDOBlocked",
  "source_route_v9720": "R3-ExactTransferNotBetterThanProxy",
  "p0_pass": 1,
  "exact_transfer_auxiliary_useful": 0,
  "failure_mode": "exact_transfer_kills_high_value_actions",
  "P2_core_expansion_weak_pass": 0,
  "best_expansion_strategy_id": "E1-old-rank-top10",
  "best_expansion_precision": 0.8850574712643678,
  "best_expansion_V_LCB": 0.14120469391749882,
  "best_expansion_LDO_drop": 0.39080459770114945,
  "P3_windowed_transfer_pass": 0,
  "best_windowed_score_id": "WT20-LCB",
  "best_windowed_precision": 0.08045977011494253,
  "best_windowed_V_LCB": -1.0243587575297828,
  "best_windowed_LDO_drop": 0.011494252873563218,
  "dataset_shift_stabilization_pass": 0,
  "direct_update_weak_pass": 0,
  "generated_route_status": "stopped_no_new_objective",
  "APGU_APGV_APGW_allowed": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "existing_action_ldo_blocked",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.3 不是 exact materializer regression。Core+10 可以得到高 precision / clean risk 的 accepted region，但 LDO 仍高；windowed transfer 没有打开可用 selector。因此 route 停在 `R3-ExistingActionStillLDOBlocked`。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9730.csv
p0_field_legality_audit_v9730.csv
```

Summary：

```text
source_route_v9720 = R3-ExactTransferNotBetterThanProxy
source_primary_blocker_v9720 = exact_transfer_not_better_than_proxy
exact_linear_transfer_row_count_v9720 = 184064
exact_apply_sample_count_v9720 = 4096
linear_apply_correlation_v9720 = 0.9997818368443041
linear_apply_sign_match_rate_v9720 = 0.961181640625
best_exact_score_v9720 = T4-core-safe-transfer
best_exact_precision_v9720 = 0.20689655172413793
best_core_expansion_precision_v9720 = 0.8850574712643678
generated_route_status_v9720 = stopped_no_new_objective
system_legal_controller_pass_v9720 = 0
field green/yellow/red = 14 / 4 / 0
p0_pass = 1
```

判断：P0 pass。v9.7.3 没有跳过 v9.7.2 的 exact transfer failure / generated stop / no system boundary。

## 4. P1 exact transfer failure autopsy

Artifact：

```text
p1_exact_transfer_failure_autopsy_v9730.csv
```

Summary：

```text
set_count = 8
core_count = 77
old_top87_precision = 0.8735632183908046
old_top87_V_LCB = 0.14111334880346277
old_top87_LDO_drop = 0.37931034482758624
exact_top87_precision = 0.20689655172413793
exact_top87_V_LCB = -0.22226101681921628
exact_top87_LDO_drop = 0.08045977011494251
old_exact_intersection_count = 18
old_exact_intersection_precision = 0.9444444444444444
old_only_count = 69
old_only_precision = 0.855072463768116
exact_only_count = 69
exact_only_precision = 0.014492753623188406
exact_only_V_LCB = -0.3037141101011471
exact_transfer_auxiliary_useful = 0
failure_mode = exact_transfer_kills_high_value_actions
```

Representative sets：

| set | count | precision | V LCB | longrisk UCB | LDO |
|---|---:|---:|---:|---:|---:|
| `Core77` | `77` | `1.0` | `0.16402807605399972` | `0.0` | `0.4025974025974026` |
| `OldRankTop87` | `87` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `ExactT4Top87` | `87` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.08045977011494251` |
| `ProxyTransferTop87Diagnostic` | `87` | `0.09195402298850575` | `-0.24289252733948727` | `0.12221253967001162` | `0.04597701149425287` |
| `OldExactIntersection` | `18` | `0.9444444444444444` | `0.0866484084297124` | `0.0` | `0.0` |
| `OldOnly` | `69` | `0.855072463768116` | `0.13450092269587752` | `0.0` | `0.37681159420289856` |
| `ExactOnly` | `69` | `0.014492753623188406` | `-0.3037141101011471` | `0.0` | `0.08695652173913046` |

判断：Exact transfer 的低 LDO 不是“稳定好动作”，而是倾向于选 low-value/safe-looking actions。它保留了 18 个高质量交集，但不足以做 controller，也不能作为 expansion 的充分辅助信号。

## 5. P2 Core77 + expansion10 minimal fix

Artifacts：

```text
p2_core77_expansion10_minimal_fix_v9730.csv
p2_core77_expansion10_action_detail_v9730.csv
```

Summary：

```text
strategy_count = 5
core_count = 77
needed_expansion_to_87 = 10
strong_pass_count = 0
weak_pass_count = 0
best_strategy_id = E1-old-rank-top10
best_accepted_count = 87
best_GradeAB_precision = 0.8850574712643678
best_V_integrated_LCB = 0.14120469391749882
best_h240_longrisk_UCB = 0.0
best_LDO_drop = 0.39080459770114945
P2_core_expansion_strong/weak = 0 / 0
```

Strategies：

| strategy | accepted | precision | V LCB | longrisk/bad/null/memory/offdiag UCB | LDO | pass |
|---|---:|---:|---:|---|---:|---:|
| `E1-old-rank-top10` | `87` | `0.8850574712643678` | `0.14120469391749882` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `E2-exact-LCB-risk-clean-top10` | `87` | `0.8850574712643678` | `0.07843570471174309` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `E3-old-high-exact-nonnegative-top10` | `87` | `0.8850574712643678` | `0.12445341670828562` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `E4-old-high-exact-nonnegative-memory-offdiag-clean-top10` | `87` | `0.8850574712643678` | `0.12445341670828562` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |
| `E5-exact-core-safe-score-top10` | `87` | `0.8850574712643678` | `0.07843570471174309` | `0 / 0 / 0 / 0 / 0` | `0.39080459770114945` | `0` |

判断：P2 是 existing-action side 的硬结论。Core77 + 10 可以补到 coverage 并保持 clean risk，但无论 old-rank 还是 exact-assisted expansion 都不能降低 LDO。

## 6. P3 windowed transfer

Artifacts：

```text
p3_windowed_transfer_subset_materializer_v9730.csv
p3_windowed_transfer_score_eval_v9730.csv
```

Materializer summary：

```text
subset_action_count_requested/materialized = 384 / 384
windows = 1,5,20
check_sample_count_per_action = 32
windowed_action_rows = 384
nan/inf = 0 / 0
compute_ms_q50/q90 = 9.241705760359764 / 10.091810952872038
memory_mb_peak = 2303.7646484375
wallclock_sec = 25.139165493194014
windowed_materializer_pass = 1
```

Score summary：

```text
score_count = 3
windowed_pass_count = 0
best_windowed_score_id = WT20-LCB
best_windowed_precision = 0.08045977011494253
best_windowed_V_LCB = -1.0243587575297828
best_windowed_longrisk_UCB = 0.7868704900942779
best_windowed_LDO_drop = 0.011494252873563218
full_windowed_materializer_status = not_run
P3_windowed_transfer_pass = 0
```

Windowed TopK87：

| score | precision | V LCB | longrisk UCB | bad/null UCB | memory/offdiag UCB | LDO |
|---|---:|---:|---:|---|---|---:|
| `WT1-LCB` | `0.06896551724137931` | `-1.007834802576804` | `0.7868704900942779` | `0.23813471172069484 / 0.0` | `0.897499817714006 / 0.9069621813964702` | `0.022988505747126436` |
| `WT5-LCB` | `0.06896551724137931` | `-1.0257690923716491` | `0.7763316932778618` | `0.3052085391932784 / 0.03389313880385762` | `0.9069621813964702 / 0.9254890130967` | `0.0` |
| `WT20-LCB` | `0.08045977011494253` | `-1.0243587575297828` | `0.7868704900942779` | `0.2920178012860627 / 0.0` | `0.9069621813964702 / 0.9162956312609247` | `0.011494252873563218` |

判断：windowed transfer materializer 本身闭合，但 WT1/5/20 LCB 选出的动作仍是 value-negative / high-longrisk / high-memory-offdiag。P3 不允许扩展到 full windowed，也不能打开 direct update。

## 7. P4 dataset-shift recap / stabilization

Artifacts：

```text
p4_dataset_shift_recap_v9730.csv
p4_dataset_shift_stabilization_v9730.csv
```

Dataset recap sample：

```text
Fashion-MNIST:
  action_count = 1226
  GradeAB_density = 0.02936378466557912
  core_count = 34
  memory_safe_rate = 0.06525285481239804
  offdiag_safe_rate = 0.05301794453507341
  longrisk_rate = 0.8164763458401305
  old_score_mean = -6.021749447018721
```

Stabilization summary：

```text
strategy_count = 5
stabilization_pass_count = 0
best_strategy_id = S0-old-rank
best_precision = 0.8735632183908046
best_V_LCB = 0.14111334880346277
best_LDO_drop = 0.37931034482758624
dataset_shift_stabilization_pass = 0
```

Representative strategies：

| strategy | precision | V LCB | longrisk UCB | LDO |
|---|---:|---:|---:|---:|
| `S0-old-rank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `S1-template-demeaned-rank` | `0.011494252873563218` | `-0.9493532118031311` | `0.8684269299978762` | `0.0` |
| `S2-old-rank-hard-clean` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` |
| `S3-exact-core-safe` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.08045977011494251` |
| `S4-windowed-best` | `0.06896551724137931` | `-1.007834802576804` | `0.7868704900942779` | `0.022988505747126436` |

判断：低 LDO 的 stabilization 会把 precision/value/risk 打坏；高质量策略仍 LDO 高。没有 dataset-name-free stabilization pass。

## 8. P5-P10 boundary

P5：

```text
p5_existing_action_minimal_controller_v9730.csv = not_run
reason = P2_P3_P4_no_weak_pass
controller_selected = 0
existing_action_controller_pass = 0
```

P6：

```text
p6_direct_windowed_transfer_solved_update_v9730.csv = not_run
reason = P3_windowed_or_P5_controller_weak_pass_failed
subspace_count = 6
generated_action_count = 0
branch_horizon_rows_actual = 0
direct_update_weak/strong = 0 / 0
```

P7：

```text
generated_route_status = stopped_no_new_objective
APGU_APGV_APGW_allowed = 0
generated_action_count = 0
branch_horizon_rows_actual = 0
generated_route_stop_triggered = 1
reason = no_windowed_or_direct_update_objective_evidence
```

P8-P10：

| artifact | status / reason |
|---|---|
| `p8_selected_runtime_preflight_v9730.csv` | `not_run`, `P5_controller_not_selected` |
| `p9_paired_replay_boundary_v9730.csv` | `not_run`, `P5_P8_system_not_official` |
| `p10_short_full_boundary_v9730.csv` | `not_run`, `P9_paired_replay_not_open` |

判断：没有 P2/P3/P4 weak pass，因此 controller/runtime/system/paired replay/short-full 全部关闭。Generated route 继续 no-new-objective stop。

## 9. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9730.csv
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

## 10. Figures

本轮额外落盘诊断图：

```text
fig_p1_set_precision.svg
fig_p2_expansion_precision.svg
fig_p3_windowed_precision.svg
fig_p4_stabilization_precision.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 11. No-fake audit

```text
rows_checked = 592
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
v9720_boundary_pass = 1
exact_failure_autopsy_done = 1
core_expansion_pass = 0
windowed_transfer_pass = 0
dataset_shift_stabilization_pass = 0
controller/runtime/system = 0/0/0
direct_update_pass = 0
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
route = R3-ExistingActionStillLDOBlocked
F0_boundary_fail = 0
F1_exact_transfer_auxiliary_fail = 1
F2_core_expansion_still_LDO_blocked = 1
F3_windowed_transfer_fail = 1
F4_controller_runtime_blocked = 1
F5_direct_update_not_open = 1
F6_generated_route_stopped = 1
F7_system_not_official = 1
primary_blocker = existing_action_ldo_blocked
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `8e2372e3d0e212689304a8fb76253429e6f8ed9c57ccf415b67ab0915ab083d1` |
| runner | `fd4c573feecaab680911e0b9286bd4efae55a1cde16d1b851b4caa13204c7638` |
| run manifest | `8d15ae8786cb1f80ae2eb0e0e6fee6da536328faa28b0874dda974636f742687` |
| route | `a959dddb993a474de407143016fd203739d1956d6108cd2d81d5725c658af614` |
| P0 boundary | `08aebd32bd577f1409440ed3f1ee6dd7188897da5f96e542fe3aa3a36bb882e9` |
| P0 field legality | `dab001084e66052293a0d8db2b318b273bbbb193a4cc711ff6ca0aa9439cc1de` |
| P1 autopsy | `4bc73da102b3f439f2c46e27962b57dcf4d6143e1c3ec9c9296d0312f026a84e` |
| P2 expansion | `91e85d37fbde190dcfb3e53610453e85f9801a7c6f3def17f7d4fcd59c601b67` |
| P2 detail | `25792ec9b0017a99dee8ed895adc8fcf17be2397e2d94a0169e6805b7cf4de6c` |
| P3 materializer | `4bcca64c535a2ab1f6d3ffb3cf819bc7426a3583bb398dd057e77db3905ebb88` |
| P3 score eval | `0b2f5db93bb0763950cafe6d627021b5f849f7ac15cc064480487c803978fede` |
| P4 recap | `adad941d7d9e4af49ef5059ced0c2239c546bfd64dde7eb9b6ff5c460fa33954` |
| P4 stabilization | `1adeaf43c6b63fdd36d26a3befa45da73b4ca403fde8e73a2f9c35d4fe429d95` |
| P5 controller | `bcff37a0e104fbc9e4bd82dd0f1f5ffa1b8d9a8296058e5727ae61adad825bde` |
| P6 direct update | `b35d1890bb12db0f247aa5ed95b21f85e8d7fa1412c061cf264ad178599948ce` |
| P7 generated stop | `27f4f686c9b58cbd12cfd6a0e7e9f9478553999e16db389dec99c87e02806227` |
| P8 runtime | `427f3261d85cadcbfad97b930c82695e837157a2e971db83924410a8534566fb` |
| P9 paired replay | `6136690db320158a017a5fc2b2d3965b485c3bff2731b9012f8e4ce17382b234` |
| P10 short/full | `e676581a93dacb5075b99836a152dc69261805e1480112032c94ebdf2ed8a64f` |
| Base-Acc Sentinel | `7f9492635e3bdd1e75f6c344c854590f803727e9f6477594998da26630006504` |
| no-fake audit | `008f512d0f1ccc05fca6e5822f99a59049d314979ab88b5e0b73333f04a283c1` |
| contract audit | `4b6200f34f20f125e5c49481a4be879efc2a9a0dd4bba75421a7592d7d1c3a2b` |
| failure taxonomy | `7b63051357890e808209e154d806f86c34b07da38fb6d6e41d10af8ccb646778` |
| allowed next gates | `f7bf1b3aca0b3d87e82d5f73b5926f56ea36200169fe576e33173fbccdd2108f` |
| stop conditions | `7c399fc859aa6c3de4d0128caa778e44782292e0e0613db401e886d5701f2b6e` |

## 13. 最终分析结论

v9.7.3 的真实推进是：

```text
v9.7.2:
  exact transfer materializer 和 linear/apply audit 已完整闭合；
  one-step exact transfer score 本身 low-quality / value-negative；
  Core77 + exact expansion 仍 LDO 高。

v9.7.3:
  将 exact transfer failure 拆成 Old/Exact/Intersection/Only sets；
  证明 exact transfer 不是简单“轻微变差”，而是会排掉大量旧 rank 高价值动作；
  Core77 + 10 个 expansion 仍能得到高质量 clean-risk accepted region；
  但所有 expansion 策略 LDO 仍约 0.39；
  windowed transfer 子集 materializer 真实闭合，但 WT1/WT5/WT20 选择出来的是 high-longrisk / high-memory-offdiag / value-negative 动作；
  dataset-name-free stabilization 没有形成新 tradeoff；
  direct update 与 generated route 继续 gate-blocked。
```

机制判断：

1. H0 成立：v9.7.2 boundary 被复现，没有跳过 exact transfer failure / generated stop / no system controller。
2. H1 未成立：one-step exact transfer 不能作为 auxiliary expansion gate；ExactOnly precision = `0.0145`，OldOnly precision = `0.8551`。
3. H2 未成立：Core77+expansion10 能保持 high precision 与 clean risk，但不能解决 LDO，best LDO = `0.3908`。
4. H3 未成立：windowed transfer materializer 过 implementation gate，但 WT scores 选出的 TopK87 quality/risk 远低于 gate。
5. H4 未成立：dataset-shift stabilization 没有把 high-quality old-rank region 转成 low-LDO region。
6. H5/H6 未打开：没有 weak controller candidate，因此 existing-action controller、direct update、runtime 都不打开。
7. H7 成立：generated route 继续 stopped_no_new_objective；APGU/APGV/APGW allowed = `0`。
8. H8-H10 未打开：没有 system controller，就不能打开 paired replay 或 short/full training。
9. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.3 真实执行后停在 `R3-ExistingActionStillLDOBlocked`：exact transfer autopsy 证明它会排掉大量旧 rank 高价值动作；Core77+10 仍是高质量 clean-risk diagnostic 但 LDO 仍高；windowed transfer 子集实测没有产生可用 selector，因此 strict PureKAN functional 仍未成功。
