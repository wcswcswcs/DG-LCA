# DG-KAN v9.7.7 Core77 Support Density / Natural Stream Gate 实验复盘

> 本复盘记录 `DG-KAN_v9.7.7_结果解读与下一步实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 LDO decomposition diagnostic、natural stream density diagnostic、support-aware gate diagnostic、OldOnly mechanism diagnostic、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R4-CoreExpansionPassButRawGateConflict
base_candidate = LQ-t2-h256
success_v9770_strict_purekan_functional = False
success_v9770_full_functional = False
success_v9770_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z/
```

核心结论：

1. P0 复现 v9.7.6 boundary：source route = `R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly`，system pass = `0`，generated route = `stopped_no_new_objective`，field red count = `0`。
2. P1 LDO decomposition 过：`LDO_raw = 0.4415584415584416`，`LDO_quality = 0.09260529551331587`，`LDO_support = 0.0`，`LDO_backfill = 0.34895314604512573`。
3. P1 支持 v9.7.6 的判断：`precision-only LDO = 0.0`，`value-only LDO = 0.09260529551331587`，`support-adjusted LDO = 0.09260529551331587`，`equal-count LDO = 0.0773703535816084`。
4. P1 raw LDO 主要来自 leaveout 后外部低质动作回填：Fashion-MNIST / KMNIST / MNIST backfill count = `34 / 28 / 15`，对应 backfill precision 都是 `0.0`。
5. P2 natural AP0 stream extension 仍未打开：`materializer_entrypoint_found = 0`，Panel B/C/D 均 `not_run`，reason = `no_landed_natural_AP0_stream_extension_materializer_for_v9770`。
6. P2 当前 Panel A 只覆盖既有 AP0 `2876` actions；Core-like count = `77`，rate = `0.026773296244784424`，min per-dataset core-like rate = `0.01824817518248175`。
7. P3 density CI 不足以下结论：Core-like Wilson CI = `[0.021475240123524635, 0.033333885489495195]`，跨过 `0.03`，所以 density result = `inconclusive`。
8. P3 estimated actions needed for 87 core = `3250`；但没有 natural stream extension，不能声称 density scaling 成功。
9. P4 minimal expansion 没有 strong pass。Core77-only precision = `1.0`，V LCB = `0.16402807605399972`，但 accepted = `77`，raw LDO = `0.4415584415584416`，LFO = `0.18181818181818177`。
10. P4 full87 expansion 中最接近的是 `E4-DatasetBlindScoreNormalizedTop10`：accepted = `87`，precision = `0.8850574712643678`，V LCB = `0.14120469391749882`，risk/bad/null/memory/offdiag UCB 全为 `0`，support-adjusted LDO = `0.08877827643026047`；但 raw LDO = `0.39080459770114945`，LFO = `0.16091954022988508`。
11. P5 OldOnly 二次诊断仍未找到机制：matched pair count = `294`，mechanism pass count = `0`；best feature = `WT80_LCB`，TopK87 precision = `0.47126436781609193`，V LCB = `-0.10386482729963209`。
12. P5 effect-size 较强的 `memory_score_inverse / old_family_margin_delta / old_family_probe_loss_inverse` 仍只给出 TopK87 precision = `0.45977011494252873`，V LCB = `-0.033718338838361464`。
13. P6 support-aware diagnostic pass count = `1`，raw official pass count = `0`；best rule = `E4-DatasetBlindScoreNormalizedTop10`。
14. P6 是本轮 route 分叉点：support-aware gate 看起来能过，但 legacy raw official gate 与 LFO 不过，因此只能 route 到 `R4-CoreExpansionPassButRawGateConflict`，不能进入 system。
15. P7 controller/runtime gate-blocked：controller 与 selected runtime 均 `not_run`。
16. P8 generated route 继续停止：new objective evidence = `0`，APGU/APGV/APGW/APGX/APGY/APGZ run = `0`。
17. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
18. 当前 primary blocker：`support_density_gate_conflict`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9770_core77_support_density_natural_stream_gate.py` | v9.7.7 runner；读取 v9.7.6/v9.7.5/v9.7.4/v9.7.2 artifacts 与 canonical AP0 ledger，执行 LDO decomposition、natural AP0 stream extension availability audit、Core-like density CI、minimal Core77+10 expansion、OldOnly secondary mechanism diagnostic、support-aware/raw gate contrast 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9770_core77_support_density_natural_stream_gate.py
```

正式运行：

```bash
python experiments/run_v9770_core77_support_density_natural_stream_gate.py \
  --out-dir results/real_rerun_20260506/v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "best_expansion_candidate": "E0-Core77Only",
  "best_expansion_precision": 1.0,
  "ldo_raw": 0.4415584415584416,
  "ldo_support_adjusted": 0.09260529551331587,
  "out_dir": "results/real_rerun_20260506/v9770_core77_support_density_natural_stream_gate_first_20260516T060000Z",
  "p_core_lcb": 0.021475240123524635,
  "p_core_mean": 0.026773296244784424,
  "p_core_ucb": 0.033333885489495195,
  "primary_blocker": "support_density_gate_conflict",
  "route": "R4-CoreExpansionPassButRawGateConflict",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 fake rows，没有 proxy rows。natural AP0 stream extension 没有 landed materializer，因此按 gate `not_run`，没有生成假扩展 action/outcome rows。

## 2. Route

`route_decision_v9770.json` 摘要：

```json
{
  "route": "R4-CoreExpansionPassButRawGateConflict",
  "source_route_v9760": "R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly",
  "p0_pass": 1,
  "P1_ldo_decomposition_pass": 1,
  "ldo_raw": 0.4415584415584416,
  "ldo_quality": 0.09260529551331587,
  "ldo_support": 0.0,
  "ldo_backfill": 0.34895314604512573,
  "ldo_support_adjusted": 0.09260529551331587,
  "P2_density_extension_pass": 0,
  "natural_stream_extension_completed": 0,
  "P3_density_core_pass": 0,
  "P3_density_core_fail": 0,
  "p_core_mean": 0.026773296244784424,
  "p_core_lcb": 0.021475240123524635,
  "p_core_ucb": 0.033333885489495195,
  "P4_strong_pass": 0,
  "P4_support_gate_definition_conflict": 0,
  "P5_oldonly_mechanism_pass": 0,
  "P6_raw_official_pass": 0,
  "P6_support_aware_diagnostic_pass": 1,
  "P6_support_density_gate_conflict": 1,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_status": "stopped_no_new_objective",
  "system_legal_controller_pass": 0,
  "primary_blocker": "support_density_gate_conflict",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.7 把 v9.7.6 的 support/backfill 诊断推进到了 gate 对照层。`E4` full87 在 support-aware diagnostic 下接近可用，但 raw official gate 和 LFO 仍失败；没有 natural stream extension，因此不能把 support-aware 结果写成 system pass。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9770.csv
p0_field_legality_audit_v9770.csv
```

Summary：

```text
source_route_v9760 = R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly
system_legal_controller_pass_v9760 = 0
generated_route_status_v9760 = stopped_no_new_objective
core77_count = 77
core77_precision = 1.0
core77_raw_LDO = 0.4415584415584416
core77_support_adjusted_LDO = 0.09260529551331587
oldonly_mechanism_pass_count = 0
field green/yellow/red = 14 / 4 / 0
P0_pass = 1
```

判断：P0 pass。v9.7.7 没有跳过 v9.7.6 的 no-system / generated-stop / OldOnly mechanism absent boundary。

## 4. P1 LDO decomposition

Artifact：

```text
p1_ldo_decomposition_gate_audit_v9770.csv
```

Summary：

```text
ldo_raw = 0.4415584415584416
ldo_quality = 0.09260529551331587
ldo_support = 0.0
ldo_backfill = 0.34895314604512573
ldo_precision_only = 0.0
ldo_value_only = 0.09260529551331587
ldo_support_adjusted = 0.09260529551331587
ldo_equal_count = 0.0773703535816084
H1_support_backfill_dominates_quality = 1
P1_ldo_decomposition_pass = 1
```

Per dataset：

| dataset | accepted | precision | V LCB | support rate | raw backfill | raw leaveout precision | backfill precision | support-adjusted drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | `34` | `1.0` | `0.2185522558920942` | `0.44155844155844154` | `34` | `0.5584415584415584` | `0.0` | `0.0` |
| KMNIST | `28` | `1.0` | `0.07778373642602063` | `0.36363636363636365` | `28` | `0.6363636363636364` | `0.0` | `0.08624433962797909` |
| MNIST | `15` | `1.0` | `0.07142278054068385` | `0.19480519480519481` | `15` | `0.8051948051948052` | `0.0` | `0.09260529551331587` |

判断：H1 成立。Core77 不是 dataset 内 precision 崩；raw LDO 主要由 leaveout 回填低质非 Core 动作造成。

## 5. P2 natural AP0 stream extension

Artifact：

```text
p2_natural_ap0_stream_extension_materializer_v9770.csv
```

Panel A：

```text
action_count = 2876
event_count = 2876
candidate_count = 1
core_like_count = 77
core_like_rate = 0.026773296244784424
GradeAB_count/rate = 79 / 0.027468706536856746
ValuePositiveNoLongRisk_count/rate = 97 / 0.03372739916550765
MemoryOffdiagCore_count/rate = 77 / 0.026773296244784424
per_dataset_core_like_rate = {"Fashion-MNIST": 0.02773246329526917, "KMNIST": 0.033816425120772944, "MNIST": 0.01824817518248175}
```

Panels B/C/D：

```text
PanelB-target-5000 = not_run
PanelC-target-10000 = not_run
PanelD-target-20000 = not_run
reason = no_landed_natural_AP0_stream_extension_materializer_for_v9770
```

判断：P2 未打开 natural density scaling。没有落地 materializer，就不能声明自然 AP0 扩流后 density sufficient/insufficient。

## 6. P3 density CI

Artifact：

```text
p3_corelike_density_ci_v9770.csv
```

Summary：

```text
p_core_mean = 0.026773296244784424
p_core_lcb = 0.021475240123524635
p_core_ucb = 0.033333885489495195
p_gradeab_mean/lcb/ucb = 0.027468706536856746 / 0.022096293530311863 / 0.03410179736468737
p_value_no_longrisk_mean/lcb/ucb = 0.03372739916550765 / 0.02772666190051218 / 0.04097211653230163
estimated_actions_needed_for_87_core = 3250
density_ci_result = inconclusive
```

判断：现有 2876 行的 Wilson interval 跨过 `0.03`，所以不能说 density 已足，也不能说 density 已失败。下一步如果继续这条线，真正缺的是 natural AP0 stream extension materializer。

## 7. P4 minimal Core77 + expansion

Artifact：

```text
p4_core77_expansion_minimal_completion_v9770.csv
```

Summary：

```text
candidate_count = 6
strong_pass_count = 0
support_gate_conflict_count = 0
best_candidate_id = E0-Core77Only
best_accepted_count = 77
best_precision = 1.0
best_V_LCB = 0.16402807605399972
best_raw_LDO = 0.4415584415584416
best_support_adjusted_LDO = 0.09260529551331587
best_LFO = 0.18181818181818177
```

Representative candidates：

| candidate | accepted | precision | V LCB | raw LDO | support-adjusted LDO | LFO | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E0-Core77Only` | `77` | `1.0` | `0.16402807605399972` | `0.4415584415584416` | `0.09260529551331587` | `0.18181818181818177` | `0` |
| `E1-OldOnlyTop10` | `87` | `0.8850574712643678` | `0.1413164676477515` | `0.39080459770114945` | `0.10727969348659006` | `0.16091954022988508` | `0` |
| `E4-DatasetBlindScoreNormalizedTop10` | `87` | `0.8850574712643678` | `0.14120469391749882` | `0.39080459770114945` | `0.08877827643026047` | `0.16091954022988508` | `0` |
| `E5-HorizonSafeLowNullTop10` | `87` | `0.8850574712643678` | `0.05893395772286003` | `0.39080459770114945` | `0.20323928944618608` | `0.16091954022988508` | `0` |

判断：P4 没有 strong pass。`E4` 很接近 support-aware gate，但 LFO/raw LDO 仍过不了 official boundary。

## 8. P5 OldOnly secondary mechanism

Artifact：

```text
p5_oldonly_secondary_mechanism_diagnostic_v9770.csv
```

Summary：

```text
matched_pair_count = 294
matched_success_rate = 0.8521739130434782
feature_count = 17
mechanism_pass_count = 0
best_feature_name = WT80_LCB
best_feature_precision = 0.47126436781609193
best_feature_V_LCB = -0.10386482729963209
best_feature_LDO = 0.13793103448275862
selected_minimal_features = ["memory_score_inverse", "old_family_probe_loss_inverse", "old_family_margin_delta"]
```

判断：OldOnly 仍是 high-value diagnostic，但没有找到少数合法字段能把它转成 controller-ready mechanism。

## 9. P6 support-aware vs raw gate

Artifact：

```text
p6_support_aware_vs_raw_gate_v9770.csv
```

Summary：

```text
rule_count = 6
legacy_raw_official_pass_count = 0
support_aware_diagnostic_pass_count = 1
best_rule_id = E4-DatasetBlindScoreNormalizedTop10
best_raw_LDO = 0.39080459770114945
best_support_adjusted_LDO = 0.08877827643026047
best_accepted_count = 87
P6_raw_official_pass = 0
P6_support_aware_diagnostic_pass = 1
P6_support_density_gate_conflict = 1
```

判断：这是本轮最重要的 gate conflict。`E4` 在 support-aware diagnostic 下满足 accepted_count/quality/risk/support-adjusted LDO，但 legacy raw LDO 与 LFO 不过。按计划不能把它写成 official system pass，必须停在 route conflict。

## 10. P7-P8 boundary

P7：

```text
p7_existing_action_minimal_controller_boundary_v9770.csv = not_run
reason = P4_or_P6_no_official_controller_gate
controller_pass = 0

p7_selected_runtime_boundary_v9770.csv = not_run
reason = P7_controller_not_passed
selected_runtime_pass = 0
```

P8：

```text
generated_route_status = stopped_no_new_objective
new_objective_evidence_present = 0
APGU_APGV_APGW_APGX_APGY_APGZ_run = 0
generated_action_count = 0
branch_horizon_rows = 0
generated_route_stop_triggered = 1
```

判断：没有 official controller gate，也没有新 generated objective evidence，因此 runtime、paired replay、short/full 和 generated variants 全部关闭。

## 11. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9770.csv
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

本轮额外落盘诊断图：

```text
fig_p1_ldo_decomposition_stacked.svg
fig_p1_backfill_count.svg
fig_p2_panel_density.svg
fig_p3_density_ci.svg
fig_p3_oldonly_matched_contrast_forest.svg
fig_p3_response_vector_oldonly_exactonly.svg
fig_p4_expansion_precision.svg
fig_p4_expansion_ldo.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 13. No-fake audit

```text
rows_checked = 468
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
v9760_boundary_pass = 1
ldo_decomposition_pass = 1
natural_stream_extension_completed = 0
density_core_pass/fail = 0/0
core_expansion_pass = 0
oldonly_mechanism_pass = 0
support_aware/raw_official_pass = 1/0
controller/runtime/system = 0/0/0
generated_route_stop = 1
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R4-CoreExpansionPassButRawGateConflict
F0_boundary_fail = 0
F1_core77_quality_stable_support_limited = 1
F2_natural_stream_extension_missing = 1
F3_density_ci_inconclusive_or_not_pass = 1
F4_core_expansion_no_official_pass = 1
F5_oldonly_mechanism_absent = 1
F6_support_aware_not_official_or_raw_fail = 1
F7_controller_runtime_blocked = 1
F8_generated_route_stopped = 1
F9_system_not_official = 1
primary_blocker = support_density_gate_conflict
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `120c4d56702f36e5cc27cc6b2d22644fa4f62a7f4a61ae7747b138a5d4b54457` |
| runner | `a8d5d1e36b4b86df8ccb52c97008ae617b7d9d0fb5af84c4a038aa5d7e0fbcd1` |
| route | `20c4f8bc22b9bb62109eeb71e398aa24c55add96f5a48bb2c17fb22153d4787f` |
| P0 boundary | `d38bd6782895f0b58dfb2c6a0bf832e42cf0bb846bd7e91f4a16d569156d7e53` |
| P0 field legality | `bf3c7d4fa0d76c4d92004076021fe3a7ba08884fb38ff875982290b746553fd9` |
| P1 LDO decomposition | `a40c844b165ce8d35407a5154cc2057df649f3fc988907a652adaab6dd61d4da` |
| P2 natural stream | `1508562e11d2915c0a8aaadb91f1db677d5dd2b41291826b59536434dd456681` |
| P3 density CI | `4a2033789a23d2986bec750863de26d2e1990e6c6cc767af22b3da5b45692457` |
| P4 expansion | `ff8aea1f83cc110a94cecf42f63c1cdf74808d01967459cea02e8075936909f4` |
| P5 OldOnly | `efdd1cd4c6d16cca58a7ee4212f0fdd649c7ef6bc6147bddd11486b87133c1a4` |
| P6 support/raw gate | `ae08b73536d18402eafe6a9a4459652ccae4d3680c250d3f0a8b9d92b79fd1dc` |
| P7 controller | `8aa89fcb16d28bac5f01fd4a454fa927ae53995acccf2dcc58296270cf5e076d` |
| P7 runtime | `5ae05cd296fba157e6922bf30f61816c6c8dea3147c92e97e82a8e07eeaa1130` |
| P8 generated route | `72116da4a5629ba9540678ac6554d38f3acd7355d9d3514af996559378784807` |
| Base-Acc Sentinel | `c111433c1cd2c03bb72db1fd5830dceacb4c3be962c6c3d2cfa6a1de7363f75f` |
| no-fake audit | `3af6726374cf6b02298fe168dd963d57bdbaf68b7858992f1822a059364abadb` |
| contract audit | `17344e53f49b12482098a5e17772482ae9baee002643e91c70b0cfda528664f9` |
| failure taxonomy | `928f6f79ac849b4f12c0d99af8834ff078ae7f27664a2f356c1d0eca56492d6e` |
| allowed next gates | `e023e4252551092887601eba7fe0a8376c3d631174f79b1e47bbc36d4b1ad40f` |
| stop conditions | `a84272b048de19812d709d25ad365f92afb122d49525e2091b83952b6d897f04` |

## 15. 最终分析结论

v9.7.7 的真实推进是：

```text
v9.7.6:
  Core77 raw LDO 被证明很大程度是 support/backfill 定义放大；
  但 Core77 数量不足，OldOnly 机制缺失，natural stream extension 未落地。

v9.7.7:
  把 raw LDO 明确拆成 quality/support/backfill；
  证明 backfill 是最大惩罚来源；
  用 Wilson CI 说明现有 2876 action 的 core density 仍不确定，不能替代 natural stream extension；
  找到一个 support-aware diagnostic full87 region；
  但 raw official gate 和 LFO 仍失败；
  OldOnly 机制依然缺失；
  controller/runtime/generated route/paired replay 全部 gate-blocked。
```

机制判断：

1. H0 成立：v9.7.6 boundary 被复现，没有跳过 no-system / generated-stop / no-fake boundary。
2. H1 成立：Core77 raw LDO 的主要成分是 backfill，`ldo_backfill = 0.34895314604512573` 大于 `ldo_quality = 0.09260529551331587`。
3. H2 未打开：natural AP0 stream extension 没有 landed materializer，Panel B/C/D 不能运行，也不能伪造 density scaling。
4. H3 未定：现有 Panel A 的 core density CI 跨过 0.03，结论是 inconclusive，不是 density sufficient。
5. H4 未成立为 official pass：minimal expansion 不能同时满足 raw LDO/LFO；`E4` 只在 support-aware diagnostic 下接近可用。
6. H5 未成立：OldOnly 机制仍找不到少数字段解释，OldRank 继续 diagnostic-only。
7. H6 成立于 conflict 层：support-aware diagnostic pass = `1`，raw official pass = `0`，因此 route 必须停在 gate conflict。
8. P7 未打开：没有 official controller gate，selected runtime 不运行。
9. P8 成立：generated route 继续 stopped_no_new_objective，APG blind variants 仍不允许。
10. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.7 真实执行后停在 `R4-CoreExpansionPassButRawGateConflict`：Core77 的质量稳定性与 support/backfill 分解更清楚了，并且 `E4` full87 在 support-aware diagnostic 下接近可用；但 raw LDO/LFO 仍不过，natural AP0 stream extension 未落地，OldOnly 机制仍缺失，因此不能进入 official controller，strict PureKAN functional 仍未成功。
