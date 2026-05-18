# DG-KAN v9.8.0 Geometry-Adaptive Optimizer 实验复盘

> 本复盘记录 `DG-KAN_v9.8.0_GeometryAdaptiveOptimizer_并行验证计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 LDO semantic decomposition、future trajectory audit、architecture-agnostic geometry ledger、generated update preflight、Base-Acc boundary 或任何 `not_run` 阶段写成 official system pass。

## 0. 最新结论

```text
route = RouteD-FutureTrajectoryUnsupported
primary_blocker = future_trajectory_theory_not_supported
system_legal_controller_pass = 0
generated_route_status = stopped_no_official_generated_branch_horizon_pass
```

最终 artifact：

```text
results/real_rerun_20260506/v9800_geometry_adaptive_optimizer_full_20260516T080000Z/
```

核心结论：

1. P0 复现 v9.7.7 boundary：source route = `R4-CoreExpansionPassButRawGateConflict`，system pass = `0`，generated route = `stopped_no_new_objective`，field red count = `0`。
2. P1 LDO semantic decomposition v2 继续支持 v9.7.7 判断：`LDO_raw = 0.4415584415584416`，`LDO_quality = 0.09260529551331587`，`LDO_support = 0.0`，`LDO_backfill = 0.34895314604512573`。
3. P1 `ldo_backfill_share_of_raw = 0.7902762425139611`，`density_support_failure_not_quality_failure = 1`，`true_cross_dataset_quality_failure = 0`。
4. P1 per-dataset Core77 precision 仍全为 `1.0`：Fashion-MNIST / KMNIST / MNIST accepted count = `34 / 28 / 15`。
5. P2 只完成现有 labeled AP0 Panel A：`2876` actions，Core-like count = `77`，rate = `0.026773296244784424`，Wilson CI = `[0.021475240123524635, 0.033333885489495195]`。
6. P2 Panel B/C/D 仍未打开：5000/10000/20000 targets 缺少真实 labeled natural stream materializer，missing labeled action count = `2124 / 7124 / 17124`。
7. P2 因此不能声称 natural density sufficient，也不能声称 natural density fail；`P2_strong_pass = 0`，`P2_weak_pass = 0`，`P2_fail = 0`。
8. P3 future trajectory audit 使用 v9480 canonical h20/h80/h240 真 rows；h1/h5 没有 landed outcome，因此 `horizon_unavailable_count = 10`。
9. P3 Core77 / OldOnly 的 future trajectory 是正的：Core77 V20 LCB = `1.573549156031694`，OldOnly V20 LCB = `1.2768640608784525`，OldOnly V80 LCB = `2.6511661678501364`。
10. P3 但 ExactOnly 也不是负例：ExactOnly V20 LCB = `0.49058574242981423`，ExactOnly LongRisk240 UCB = `0.0`，所以计划中的 “ExactOnly 应该失败或长风险高” 判据不成立。
11. P3 OldRank 与 future V20 的相关性高于 ExactTransfer：`0.10802011654668134 > 0.006417218989248354`；但这不足以通过完整 P3 gate。
12. P4 architecture-agnostic geometry score diagnostic 失败：TopK87 precision = `0.06896551724137931`，V LCB = `-0.9348850937038008`，LongRisk UCB = `0.8485412092119363`。
13. P4 该 ledger 还读取了 future outcome rows，只能 diagnostic，不是 controller feature；`official_controller_pass = 0`。
14. P5 generated update preflight 真实跑了 `256` CUDA rows，D1/D2/D3/Negative 各 `64`。
15. P5 real geometry-adaptive updates 有正向 h20 smoke：`D_real_future_h20_LCB = 0.1589771902962544`，memory harm UCB = `0.024778041801919663`。
16. P5 但 negative control 也有正向 h20：`negative_future_h20_LCB = 0.13225866245274354`，`negative_control_fail = 0`，所以不能打开 generated branch-horizon。
17. P6 generated branch-horizon smoke `not_run`，reason = `P5_generated_preflight_not_strong_pass`。
18. P7 controller boundary `not_run`，reason = `P2_or_P4_not_official_controller_ready`。
19. P8 runtime boundary `not_run`，reason = `P7_controller_not_passed_and_P6_generated_not_passed`。
20. P9 paired replay / short-full boundary `not_run`，reason = `P7_P8_not_passed`。
21. No-fake audit 通过：rows checked = `7126`，fake/proxy/cpu offload = `0 / 0 / 0`。
22. 当前 primary blocker 是 `future_trajectory_theory_not_supported`；secondary blocker 可理解为 natural labeled stream extension 仍缺失、generated preflight negative control 未分离。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9800_geometry_adaptive_optimizer.py` | v9.8.0 full-gated runner；读取 v9.7.7/v9.7.6/v9.7.5/v9.7.4/v9.7.2/v9.4.8/v9.3.3 artifacts，执行 boundary reproduction、LDO v2、natural stream panel audit、future trajectory audit、architecture-agnostic geometry ledger、generated update preflight 与 gate boundary。 |
| `docs/DG-KAN_v9.8.0_GeometryAdaptiveOptimizer_实验复盘.md` | 本文件；只记录 full run artifact，不引用 smoke 作为最终结论。 |

代码检查：

```text
python -m py_compile experiments/run_v9800_geometry_adaptive_optimizer.py
```

Smoke 运行：

```bash
python experiments/run_v9800_geometry_adaptive_optimizer.py \
  --out-dir results/real_rerun_20260506/tmp_v9800_geometry_adaptive_smoke \
  --fresh --device auto --data-root data --seed 1314 \
  --execution-profile smoke --panel-targets 64,128 --generated-per-family 4
```

正式运行：

```bash
python experiments/run_v9800_geometry_adaptive_optimizer.py \
  --out-dir results/real_rerun_20260506/v9800_geometry_adaptive_optimizer_full_20260516T080000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --execution-profile full-gated \
  --panel-targets 2876,5000,10000,20000 \
  --generated-per-family 64
```

运行结果：

```json
{
  "out_dir": "results/real_rerun_20260506/v9800_geometry_adaptive_optimizer_full_20260516T080000Z",
  "p3_future_trajectory_pass": 0,
  "p5_generated_preflight_pass": 0,
  "primary_blocker": "future_trajectory_theory_not_supported",
  "route": "RouteD-FutureTrajectoryUnsupported",
  "system_legal_controller_pass": 0
}
```

说明：本轮使用 CUDA，`run_manifest_v9800.json` 记录 device = `cuda`，execution profile = `full-gated`，wallclock = `122.45870569907129` sec。没有 CPU offload，没有 fake rows，没有 proxy rows。

## 2. Route

`route_decision_v9800.json` 摘要：

```json
{
  "route": "RouteD-FutureTrajectoryUnsupported",
  "source_route_v9770": "R4-CoreExpansionPassButRawGateConflict",
  "P0_boundary_reproduced": 1,
  "P1_density_support_failure_not_quality_failure": 1,
  "P2_natural_stream_extension_completed": 0,
  "P2_strong_pass": 0,
  "P3_future_trajectory_pass": 0,
  "P4_geometry_diagnostic_pass": 0,
  "P5_generated_preflight_pass": 0,
  "P6_weak_generated_pass": 0,
  "P6_strong_generated_pass": 0,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "generated_route_status": "stopped_no_official_generated_branch_horizon_pass",
  "primary_blocker": "future_trajectory_theory_not_supported"
}
```

判断：v9.8.0 把 v9.7.7 的 support/backfill 诊断推进到 future trajectory 与 geometry-adaptive update preflight 层。结果显示 Core77 / OldOnly 的 future trajectory 确实很好，但 ExactOnly 并没有按计划成为负例；因此不能说“好动作的本质已经由 future trajectory theory 解释”。同时，natural labeled stream extension 仍未落地，generated preflight 也没有与 negative control 分离，所以不能进入 official controller/runtime。

## 3. P0 boundary reproduction

Artifacts：

```text
p0_boundary_reproduction_v9800.csv
p0_field_legality_audit_v9800.csv
```

Summary：

```text
source_route_v9770 = R4-CoreExpansionPassButRawGateConflict
source_primary_blocker_v9770 = support_density_gate_conflict
source_secondary_blocker_v9770 = generated_route_stopped_no_new_objective
system_legal_controller_pass_v9770 = 0
generated_route_status_v9770 = stopped_no_new_objective
core77_count_v9770 = 77
core77_precision_v9770 = 1.0
ldo_raw_v9770 = 0.4415584415584416
ldo_quality_v9770 = 0.09260529551331587
ldo_support_v9770 = 0.0
ldo_backfill_v9770 = 0.34895314604512573
support_aware_pass_v9770 = 1
raw_official_pass_v9770 = 0
natural_stream_extension_completed_v9770 = 0
base_acc_sentinel_pass_v9770 = 1
field_red_count_v9770 = 0
P0_boundary_reproduced = 1
```

判断：P0 pass。v9.8.0 没有跳过 v9.7.7 的 no-system / generated-stop / no-fake boundary，也没有把 support-aware diagnostic 直接升成 official system。

## 4. P1 LDO semantic decomposition v2

Artifacts：

```text
p1_core77_ldo_semantic_decomposition_v2_v9800.csv
p1_support_backfill_group_trace_v9800.csv
```

Summary：

```text
ldo_raw = 0.4415584415584416
ldo_quality = 0.09260529551331587
ldo_support = 0.0
ldo_backfill = 0.34895314604512573
ldo_backfill_share_of_raw = 0.7902762425139611
ldo_precision_only = 0.0
ldo_value_only = 0.09260529551331587
ldo_support_adjusted = 0.09260529551331587
ldo_equal_count = 0.0773703535816084
ldo_equal_count_p95 = 0.09652672025051506
H1_support_backfill_dominates_quality = 1
density_support_failure_not_quality_failure = 1
true_cross_dataset_quality_failure = 0
P1_ldo_decomposition_pass = 1
```

Per dataset support/backfill group：

| dataset | action count | core count | core rate | core precision | core V LCB | noncore count | noncore top precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | `1226` | `34` | `0.02773246329526917` | `1.0` | `0.2185522558920942` | `1192` | `0.0` |
| KMNIST | `828` | `28` | `0.033816425120772944` | `1.0` | `0.07778373642602063` | `800` | `0.0` |
| MNIST | `822` | `15` | `0.01824817518248175` | `1.0` | `0.07142278054068385` | `807` | `0.0` |

判断：P1 继续支持 v9.7.7 的机制判断。Core77 本身不是跨 dataset precision collapse；raw LDO 的主惩罚项仍是 leaveout 后低质 backfill。这里的 official 解释仍是 density/support failure，而不是 core quality failure。

## 5. P2 natural AP0 stream extension materializer

Artifact：

```text
p2_natural_ap0_stream_extension_materializer_v9800.csv
```

Summary：

```text
requested_panel_targets = 2876,5000,10000,20000
existing_labeled_ap0_action_count = 2876
completed_panel_count = 1
not_run_panel_count = 3
max_completed_panel_target = 2876
natural_stream_extension_completed = 0
materializer_entrypoint_found = 0
p_core_mean_panel_a = 0.026773296244784424
p_core_lcb_panel_a = 0.021475240123524635
p_core_ucb_panel_a = 0.033333885489495195
min_per_dataset_core_like_rate_panel_a = 0.01824817518248175
P2_strong_pass = 0
P2_weak_pass = 0
P2_fail = 0
reason = extension_panels_blocked_by_missing_labeled_materializer
```

Panel A：

```text
target_action_count = 2876
action_count = 2876
event_count = 2876
materialized_from = canonical_existing_AP0_labeled_ledger
Core_like_count/rate = 77 / 0.026773296244784424
Core_like_rate_LCB/UCB = 0.021475240123524635 / 0.033333885489495195
GradeAB_count/rate = 79 / 0.027468706536856746
ValuePositiveNoLongRisk_count/rate = 97 / 0.03372739916550765
min_per_dataset_core_like_rate = 0.01824817518248175
quality_audit_pass = 1
```

Panel B/C/D：

| panel | target | status | missing labeled actions | reason |
|---|---:|---|---:|---|
| Panel-target-5000 | `5000` | `not_run` | `2124` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9800` |
| Panel-target-10000 | `10000` | `not_run` | `7124` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9800` |
| Panel-target-20000 | `20000` | `not_run` | `17124` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9800` |

判断：P2 仍未打开 natural density scaling。v9.8.0 没有把缺少真实 outcome label 的新 natural actions 写成 GradeAB=0 或 Core-like=0；因此不能声称 density sufficient，也不能声称 density fail。

## 6. P3 future trajectory effect audit

Artifact：

```text
p3_future_trajectory_effect_audit_v9800.csv
```

Summary：

```text
group_count = 5
canonical_real_action_count = 2876
horizons_requested = 1,5,20,80,240
horizons_materialized = 20,80,240
horizon_unavailable_count = 10
Core77_V20_LCB = 1.573549156031694
OldOnly_V20_LCB = 1.2768640608784525
OldOnly_V80_LCB = 2.6511661678501364
OldOnly_LongRisk240_UCB = 0.0
ExactOnly_V20_LCB = 0.49058574242981423
ExactOnly_LongRisk240_UCB = 0.0
OldRank_future_V20_corr = 0.10802011654668134
ExactTransfer_future_V20_corr = 0.006417218989248354
P3_future_trajectory_pass = 0
```

Representative group × horizon rows：

| group | horizon | actions | V LCB | V mean | longrisk UCB | bad UCB | memory UCB | offdiag UCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Core77 | `20` | `77` | `1.573549156031694` | `1.952056168055452` | `0.0` | `0.0` | `0.0` | `0.0` |
| Core77 | `80` | `77` | `3.1200217020467886` | `3.6963413333665436` | `0.0` | `0.0` | `0.0` | `0.0` |
| Core77 | `240` | `77` | `4.750885696264312` | `5.475575658671147` | `0.0` | `0.0` | `0.0` | `0.0` |
| OldOnly | `20` | `69` | `1.2768640608784525` | `1.6012066980871116` | `0.0` | `0.0` | `0.0` | `0.0` |
| OldOnly | `80` | `69` | `2.6511661678501364` | `3.1265174932507933` | `0.0` | `0.0` | `0.0` | `0.0` |
| OldOnly | `240` | `69` | `4.171856278488452` | `4.773482355579356` | `0.0` | `0.0` | `0.0` | `0.0` |
| ExactOnly | `20` | `69` | `0.49058574242981423` | `0.8310294985919651` | `0.0` | `0.0` | `0.0` | `0.0` |
| ExactOnly | `80` | `69` | `1.5634505908603344` | `2.206894939319919` | `0.0` | `0.0` | `0.0` | `0.0` |
| ExactOnly | `240` | `69` | `2.901934793211203` | `3.7089147831792473` | `0.0` | `0.0` | `0.0` | `0.0` |
| E4Expansion10 | `20` | `87` | `1.544542302743137` | `1.9102624102158408` | `0.0` | `0.0` | `0.0` | `0.0` |
| E4Expansion10 | `80` | `87` | `3.0933350204937695` | `3.656655675961754` | `0.0` | `0.0` | `0.0` | `0.0` |
| E4Expansion10 | `240` | `87` | `4.688599946956708` | `5.392712262694606` | `0.0` | `0.0` | `0.0` | `0.0` |
| RandomMatched | `20` | `87` | `0.6381698863946743` | `0.9754073764438388` | `0.0` | `0.265315909668029` | `0.9254890130967` | `0.9687665318872787` |
| RandomMatched | `80` | `87` | `1.8659626612680797` | `2.4319186899034926` | `0.0` | `0.22433958780934365` | `0.9254890130967` | `0.9687665318872787` |
| RandomMatched | `240` | `87` | `3.508325720570364` | `4.261514770586964` | `0.8283004628835223` | `0.13761700438354324` | `0.9254890130967` | `0.9687665318872787` |

判断：P3 的结论比较微妙。Core77 / OldOnly / E4Expansion10 都有很强 future trajectory，但 ExactOnly 也有正 V 且 longrisk 为 `0.0`。原计划要求 ExactOnly 至少满足 `V20 LCB <= 0` 或 `LongRisk240 UCB > 0.15` 才能作为反例；这一点没有成立。因此本轮不能宣称“Core77 / OldOnly 的优势已经由 future trajectory theory 唯一解释”，route 进入 `RouteD-FutureTrajectoryUnsupported`。

## 7. P4 architecture-agnostic geometry probe ledger

Artifact：

```text
p4_architecture_agnostic_geometry_probe_ledger_v9800.csv
```

Summary：

```text
geometry_action_row_count = 2876
TopK87_precision = 0.06896551724137931
TopK87_V_LCB = -0.9348850937038008
TopK87_longrisk_UCB = 0.8485412092119363
TopK87_bad_UCB = 0.07282499698983116
TopK87_null_UCB = 0.03389313880385762
TopK87_LDO = 0.011494252873563218
TopK87_LSO = 0.011494252873563218
TopK87_LTO = 0.011494252873563218
architecture_agnostic_probe_diagnostic_pass = 0
official_controller_pass = 0
official_blocker = geometry_rows_use_future_outcome_for_diagnostic_not_controller
```

判断：P4 失败。当前 architecture-agnostic geometry score 不是有效 selector：TopK87 precision 很低，V LCB 为负，LongRisk UCB 很高。并且该 ledger 使用 canonical future outcome rows 构造 diagnostic score，本来也不能作为 official controller feature。

## 8. P5 geometry-adaptive update preflight

Artifact：

```text
p5_geometry_adaptive_update_preflight_v9800.csv
```

Summary：

```text
generated_action_count = 256
generated_per_family_requested = 64
generated_family_counts = {"D1": 64, "D2": 64, "D3": 64, "Negative": 64}
D_real_future_h20_LCB = 0.1589771902962544
D_real_memory_harm_UCB = 0.024778041801919663
D_real_preflight_CE_response_mean = 0.005191673330652217
negative_future_h20_LCB = 0.13225866245274354
negative_control_fail = 0
runtime_ms_q90 = 12.016440276056528
memory_mb_peak = 33.05322265625
wallclock_sec = 4.7447618171572685
P5_generated_preflight_pass = 0
```

Per family h20 value mean：

| family | count | future h20 value mean |
|---|---:|---:|
| D1 | `64` | `0.20201864221598953` |
| D2 | `64` | `0.2036315214354545` |
| D3 | `64` | `0.1995639115339145` |
| Negative | `64` | `0.20643913571257144` |

判断：P5 是 real CUDA preflight，不是 fake/proxy。D1/D2/D3 的 future h20 LCB 为正，memory harm 低；但 Negative control 也为正，且 family mean 不低于 real families。因此这一步不能说明 geometry-adaptive update 具有独立因果优势，不能打开 P6 generated branch-horizon。

## 9. P6 generated update branch-horizon smoke

Artifact：

```text
p6_geometry_adaptive_update_branch_horizon_smoke_v9800.csv
```

Summary：

```text
status = not_run
reason = P5_generated_preflight_not_strong_pass
generated_branch_horizon_rows = 0
weak_generated_pass = 0
strong_generated_pass = 0
```

判断：P6 合法关闭。虽然 P5 生成了 256 个 real preflight rows，但 negative control 未分离，因此没有打开 D1/D2/D3 的 generated branch-horizon smoke。

## 10. P7-P9 boundary

Artifacts：

```text
p7_existing_action_minimal_controller_boundary_v9800.csv
p8_runtime_boundary_v9800.csv
p9_paired_replay_short_full_boundary_v9800.csv
```

P7：

```text
status = not_run
reason = P2_or_P4_not_official_controller_ready
controller_pass = 0
```

P8：

```text
status = not_run
reason = P7_controller_not_passed_and_P6_generated_not_passed
selected_runtime_pass = 0
runtime_illegal = 0
```

P9：

```text
status = not_run
reason = P7_P8_not_passed
paired_replay_pass = 0
short_full_open = 0
```

判断：没有 official controller gate，也没有 generated branch-horizon pass，所以 runtime、paired replay、short/full 全部保持关闭。

## 11. Allowed gates and stop conditions

`allowed_next_gates_v9800.json`：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "generated_branch_horizon_allowed": 0,
  "natural_stream_extension_required": 1,
  "geometry_diagnostic_requires_distillation_before_controller": 0
}
```

`stop_conditions_v9800.json`：

```json
{
  "enter_system": 0,
  "stop_density_claim_without_labeled_natural_stream_extension": 1,
  "stop_future_outcome_geometry_direct_official_promotion": 1,
  "stop_generated_route_without_P6": 1
}
```

判断：下一步若继续 v9.8.x，最直接的工程 blocker 仍是 labeled natural stream extension；理论 blocker 是 future trajectory 对 ExactOnly/negative controls 的分离不够。

## 12. Figures

本轮落盘图：

```text
fig_p1_backfill_count.svg
fig_p1_ldo_decomposition_stacked.svg
fig_p2_natural_panel_density_v9800.svg
fig_p3_future_longrisk240_by_group_v9800.svg
fig_p3_future_v20_by_group_v9800.svg
fig_p4_geometry_topk_quality_v9800.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 13. No-fake audit

Artifact：

```text
no_fake_audit_v9800.csv
```

Summary：

```text
rows_checked = 7126
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
v9770_boundary_pass = 1
ldo_decomposition_v2_pass = 1
natural_stream_extension_completed = 0
future_trajectory_pass = 0
geometry_probe_official_controller_pass = 0
generated_preflight/generated_branch_horizon = 0/0
controller/runtime/system = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_official_controller = 0
diagnostic_promoted_to_official = 0
base_acc_used_for_controller = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure taxonomy：

```text
route = RouteD-FutureTrajectoryUnsupported
F0_boundary_fail = 0
F1_density_support_failure = 1
F2_natural_stream_extension_missing = 1
F3_future_trajectory_not_supported = 1
F4_geometry_diagnostic_not_official = 0
F5_generated_preflight_not_pass = 1
F6_generated_branch_horizon_not_run_or_fail = 1
F7_controller_runtime_blocked = 1
primary_blocker = future_trajectory_theory_not_supported
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| `allowed_next_gates_v9800.json` | `40ba23ef23d2933b0de49dea8365d7f2e47c20c0b4450d62f0d53415538abb0e` |
| `contract_audit_v9800.csv` | `70eeda77b77330d7f8b7a60356ccd303232634022be06af6383b70861f03a9fc` |
| `failure_taxonomy_v9800.csv` | `d372b314144f2f746586dd88b0ad091703515b7dffd1a90c9a2d7a7fd91cf6e9` |
| `no_fake_audit_v9800.csv` | `b97b39a9f770e45d0f8b02d25cf512ae8cb48098e5b0f3610748d989d3d2d56e` |
| `p0_boundary_reproduction_v9800.csv` | `df12f2b3ac1ed0fe8a39228241c3204a18cd64ee93d2adefb6b2d063aa8057d3` |
| `p0_field_legality_audit_v9800.csv` | `a922e6b4bcdded78cb319039df283ac0000080a8a1eba029a642d9d49fff91d9` |
| `p1_core77_ldo_semantic_decomposition_v2_v9800.csv` | `0e30831ed74210ea30b057977bbe0bf663455700ad39aa1220e6213181f5c27c` |
| `p1_support_backfill_group_trace_v9800.csv` | `2550b27ac37a1ae9038cc270bd9e11d82d6ef49aabbb25e30631f197f182f561` |
| `p2_natural_ap0_stream_extension_materializer_v9800.csv` | `93c514d37aee71d5f8aa71fee758a76c11458d638db9010fd9e99f5c4851ce21` |
| `p3_future_trajectory_effect_audit_v9800.csv` | `524f0140bd5485565a539f8b32312208513b15747928034558f9d8698e238c66` |
| `p4_architecture_agnostic_geometry_probe_ledger_v9800.csv` | `82c868fa13181634332d230cce5a04561acc93ccb9be00e21b9ee7ac9a1ed4d8` |
| `p5_geometry_adaptive_update_preflight_v9800.csv` | `fc4b16129c71e0d39dc4a6b9c6ff9d36a17a823ab72f05bcd8cd843c89f56c14` |
| `p6_geometry_adaptive_update_branch_horizon_smoke_v9800.csv` | `aca5dfb16bfed3b52125b007e0f677fd772605e0e4a8d50e7b748195c516e230` |
| `p7_existing_action_minimal_controller_boundary_v9800.csv` | `7185152900905403fe27a47822cd996d82d7452342234e2f452a7d7fe54567c6` |
| `p8_runtime_boundary_v9800.csv` | `da28c933d46a98581cf177df07c1a5e139730eadc5947695c9dbf19905ef88ad` |
| `p9_paired_replay_short_full_boundary_v9800.csv` | `b8384b90494dfe8a605b3fff0e2be284aec12cc3c56f64891ca28ef0f32700b2` |
| `plan` | `9fe3a45346726ad99d9893975dc65684717afde88e1ca2202684726dfc269e85` |
| `route_decision_v9800.json` | `0bd316f2c8b942ab53a16cd1b21713dbc833c6d42c06de33af386a4cc8c022c5` |
| `run_manifest_v9800.json` | `22a7a8e62d3e694203bcc8d788c9892cfc10b22d18496280b683165ec719fdd1` |
| `runner` | `c883d9b148a7cd1f5fdf4ba9470c2f2020deffef554b9e700be8230828bcbbf9` |
| `stop_conditions_v9800.json` | `c23e79c44089c1b082b2dd6c3c4cc3253101a03d7f23d2207218fcf7e1b4fc62` |

Manifest hash check：

```text
manifest_hash_entries = 20
hash_mismatches = []
```

## 15. 最终分析结论

v9.8.0 的真实推进是：

```text
v9.7.7:
  Core77 quality 稳定，raw LDO 主要来自 support/backfill；
  E4 support-aware diagnostic 接近可用；
  但 natural AP0 stream extension 未落地，raw official gate / LFO 不过。

v9.8.0:
  复现 v9.7.7 boundary；
  LDO v2 继续确认 support/backfill 是主问题；
  natural labeled stream extension 仍未落地；
  future trajectory audit 证明 Core77 / OldOnly / E4 的未来轨迹确实强；
  但 ExactOnly 也有正向 future trajectory 且低 longrisk，不能作为理论负例；
  architecture-agnostic geometry diagnostic TopK87 失败；
  generated geometry-adaptive preflight 有正向 h20 smoke，但 negative control 同样正向；
  P6 generated branch-horizon、P7 controller、P8 runtime、P9 paired replay/short-full 全部 gate-blocked。
```

机制判断：

1. H0 成立：v9.7.7 boundary 被复现，没有跳过 no-system / generated-stop / no-fake boundary。
2. H1 成立：Core77 raw LDO 仍主要来自 support/backfill，`ldo_backfill_share_of_raw = 0.7902762425139611`。
3. H2 未打开：natural AP0 stream extension 仍没有 landed labeled materializer；Panel B/C/D 不能运行，也不能伪造 density scaling。
4. H3 不足以支持 future trajectory theory：Core77 / OldOnly 的未来轨迹很强，但 ExactOnly 也为正且 longrisk 低。
5. H4 未成立：architecture-agnostic geometry ledger 的 TopK87 precision/V/longrisk 均不过。
6. H5 未成立：generated update preflight 没有与 negative control 分离，不能进入 P6 branch-horizon。
7. P7-P9 未打开：没有 official controller，也没有 runtime / paired replay / short-full。
8. No-fake contract 成立：`fake/proxy/cpu_offload = 0/0/0`，diagnostic 没有被提升为 official system pass。

最终一句话：

> v9.8.0 真实执行后停在 `RouteD-FutureTrajectoryUnsupported`：Core77 的 support/backfill 解释进一步稳定，Core77/OldOnly/E4 的 future trajectory 也确实强；但 ExactOnly 不是清晰负例，architecture-agnostic geometry score 失败，generated preflight 没有打败 negative control，natural labeled stream extension 仍未落地，因此不能进入 official controller，strict PureKAN functional 仍未成功。
