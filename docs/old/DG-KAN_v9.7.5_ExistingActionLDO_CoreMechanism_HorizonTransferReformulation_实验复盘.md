# DG-KAN v9.7.5 ExistingAction LDO / Core Mechanism / Horizon Transfer Reformulation 实验复盘

> 本复盘记录 `DG-KAN_v9.7.5_ExistingActionLDO_CoreMechanism_HorizonTransferReformulation_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 Core77 autopsy diagnostic、OldOnly mechanism diagnostic、response-vector diagnostic、dataset-blind stability repair、expansion v2、OldRank ablation、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R2-ExistingActionHighQualityButLDOBlocked
base_candidate = LQ-t2-h256
success_v9750_strict_purekan_functional = False
success_v9750_full_functional = False
success_v9750_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z/
```

核心结论：

1. P0 复现 v9.7.4 boundary：source route = `R2-ExistingActionLDOBlocked`，system pass = `0`，generated route = `stopped_no_new_objective`，field red count = `0`。
2. P1 Core77 LDO autopsy 过：Core77 count = `77`，precision = `1.0`，V LCB = `0.16402807605399972`，longrisk UCB = `0.0`，LDO drop = `0.4415584415584416`。
3. P1 判断 Core77 的问题是 dataset/support shift + structural LDO：H1a/H1b/H1c = `1 / 0 / 1`；没有发现 family/template/step hidden concentration 作为主解释。
4. P1 per dataset 都 precision = `1.0`，但 support/V 明显不均：Fashion-MNIST / KMNIST / MNIST count = `34 / 28 / 15`，V LCB = `0.2185522558920942 / 0.07778373642602063 / 0.07142278054068385`。
5. P1 bootstrap 结构性检查：bootstrap LDO p50 = `0.44680851063829785`，p95 = `0.5102040816326531`，P(LDO > 0.20) = `1.0`。
6. P2 OldOnly anatomy 继续支持 OldOnly 作为研究对象：OldOnly count = `69`，precision = `0.855072463768116`，V LCB = `0.13450092269587752`，dataset/template/family count = `3 / 69 / 34`。
7. P2 ExactOnly 仍是低质量对照：ExactOnly precision = `0.014492753623188406`，V LCB = `-0.3037141101011471`；OldOnly single-group pocket = `0`。
8. P3 transfer mismatch deep autopsy 过：OldOnly mean actual V = `0.18785132119344358`，ExactOnly mean actual V = `-0.25884870676985144`；H3a/H3b/H3c = `1 / 1 / 1`。
9. P3 说明 one-step/windowed response 仍错配：OldOnly mean WT80 = `0.010067179508415913`，ExactOnly mean WT80 = `0.10113173453042902`，但 ExactOnly actual value 仍为负。
10. P4 dataset-blind stability repair 未过：best = `S7-Core77-plus-frozen-expansion`，precision = `0.8850574712643678`，V LCB = `0.14120469391749882`，longrisk/bad/null/memory/offdiag UCB = `0`，但 LDO drop = `0.39080459770114945`。
11. P4 低 LDO 规则继续损伤质量：`S8-exact-core-safe` LDO drop = `0.08045977011494251`，但 precision = `0.20689655172413793`，V LCB = `-0.22226101681921628`。
12. P5 Core77 expansion v2 未过：best diagnostic = `E0-Core77-only`，precision = `1.0`，V LCB = `0.16402807605399972`，但 accepted = `77` 且 LDO drop = `0.4415584415584416`；所有补 10 的 full87 LDO 都仍为 `0.39080459770114945`。
13. P6 OldRank mechanism ablation v2 未解释机制：best = `A0-OldRank`，precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，OldOnly overlap = `1.0`，但 LDO drop = `0.37931034482758624`，mechanism pass count = `0`。
14. P7/P8/P10/P11 gate-blocked：existing-action controller、selected runtime、paired replay、short/full boundary 均 not_run。
15. P9 generated route 继续停止：new objective evidence = `0`，APGU/APGV/APGW/APGX run = `0`，direct solved sandbox allowed = `0`。
16. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`existing_action_ldo_blocked`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation.py` | v9.7.5 runner；读取 v9.7.4/v9.7.2 artifacts 与 canonical AP0 ledger，执行 Core77 LDO autopsy、OldOnly anatomy、response-vector transfer mismatch autopsy、dataset-blind stability repair、Core77 expansion v2、OldRank mechanism ablation v2 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation.py
```

正式运行：

```bash
python experiments/run_v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation.py \
  --out-dir results/real_rerun_20260506/v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z \
  --fresh --device auto --data-root data --seed 1314
```

运行结果：

```json
{
  "Core77_LDO_drop": 0.4415584415584416,
  "OldOnly_precision": 0.855072463768116,
  "best_ablation_id": "A0-OldRank",
  "best_stability_LDO_drop": 0.39080459770114945,
  "best_stability_precision": 0.8850574712643678,
  "best_stability_rule": "S7-Core77-plus-frozen-expansion",
  "out_dir": "results/real_rerun_20260506/v9750_existing_action_ldo_core_mechanism_horizon_transfer_reformulation_first_20260516T040000Z",
  "primary_blocker": "existing_action_ldo_blocked",
  "route": "R2-ExistingActionHighQualityButLDOBlocked",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有 CPU offload，没有 fake rows；v9.7.4 horizon/windowed artifacts 和 v9.7.2 exact artifacts 只作为已落盘 diagnostic 输入，没有提升为 official controller。

## 2. Route

`route_decision_v9750.json` 摘要：

```json
{
  "route": "R2-ExistingActionHighQualityButLDOBlocked",
  "source_route_v9740": "R2-ExistingActionLDOBlocked",
  "p0_pass": 1,
  "P1_core77_autopsy_pass": 1,
  "Core77_LDO_drop": 0.4415584415584416,
  "H1a_dataset_shift_or_support_shift": 1,
  "H1b_hidden_group_concentration": 0,
  "H1c_structural_not_random": 1,
  "P2_oldonly_continue_research_pass": 1,
  "OldOnly_precision": 0.855072463768116,
  "OldOnly_V_LCB": 0.13450092269587752,
  "ExactOnly_precision": 0.014492753623188406,
  "P3_transfer_autopsy_pass": 1,
  "H3a_immediate_transfer_misses_delayed_value": 1,
  "H3b_exactonly_low_value_safe_region": 1,
  "P4_dataset_blind_stability_weak_pass": 0,
  "P4_dataset_blind_stability_strong_pass": 0,
  "best_stability_rule": "S7-Core77-plus-frozen-expansion",
  "best_stability_precision": 0.8850574712643678,
  "best_stability_V_LCB": 0.14120469391749882,
  "best_stability_LDO_drop": 0.39080459770114945,
  "P5_expansion_v2_pass": 0,
  "best_expansion_id": "E0-Core77-only",
  "best_expansion_LDO_drop": 0.4415584415584416,
  "P6_oldrank_mechanism_explained": 0,
  "best_ablation_id": "A0-OldRank",
  "best_ablation_precision": 0.8735632183908046,
  "best_ablation_LDO_drop": 0.37931034482758624,
  "controller_pass": 0,
  "selected_runtime_pass": 0,
  "generated_route_status": "stopped_no_new_objective",
  "direct_solved_sandbox_allowed": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "existing_action_ldo_blocked",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.5 证明 high-quality existing-action signal 仍存在，但 dataset-blind repair、expansion v2 与 OldRank ablation 都没有把它变成 official controller。Route 因此停在 `R2-ExistingActionHighQualityButLDOBlocked`。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9750.csv
p0_field_legality_audit_v9750.csv
```

Summary：

```text
source_route_v9740 = R2-ExistingActionLDOBlocked
system_legal_controller_pass_v9740 = 0
Core77_LDO_drop_v9740 = 0.4415584415584416
C0E1_precision_v9740 = 0.8850574712643678
C0E1_LDO_drop_v9740 = 0.39080459770114945
best_horizon_score_id_v9740 = WT80-LCB
best_horizon_precision_v9740 = 0.47126436781609193
best_horizon_V_LCB_v9740 = -0.10386482729963209
generated_route_status_v9740 = stopped_no_new_objective
field green/yellow/red = 14 / 4 / 0
dataset_name_used_in_controller = 0
outcome_derived_field_used_in_controller = 0
p0_pass = 1
```

判断：P0 pass。v9.7.5 没有跳过 v9.7.4 的 LDO block / transfer mismatch / generated stop boundary，也没有把 dataset_name 或 outcome-derived fields 带入 official path。

## 4. P1 Core77 LDO autopsy

Artifacts：

```text
p1_core77_ldo_autopsy_v9750.csv
p1_core77_group_autopsy_v9750.csv
```

Summary：

```text
core77_count = 77
core77_precision = 1.0
core77_V_LCB = 0.16402807605399972
core77_longrisk_UCB = 0.0
core77_LDO_drop = 0.4415584415584416
min_dataset_precision = 1.0
min_dataset_V_LCB = 0.07142278054068385
min/max dataset support = 15 / 34
max_family_share = 0.18181818181818182
max_template_share = 0.012987012987012988
max_step_bucket_share = 0.11688311688311688
bootstrap_reps = 128
bootstrap_ldo_p50/p95 = 0.44680851063829785 / 0.5102040816326531
bootstrap_prob_LDO_gt_020 = 1.0
H1a/H1b/H1c = 1 / 0 / 1
P1_core77_autopsy_pass = 1
```

Per dataset：

| dataset | count | precision | V LCB | longrisk | score mean/std | templates | families |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Fashion-MNIST` | `34` | `1.0` | `0.2185522558920942` | `0.0` | `0.21676876682534327 / 0.2264243492171494` | `34` | `13` |
| `KMNIST` | `28` | `1.0` | `0.07778373642602063` | `0.0` | `0.07083746428633513 / 0.15094728005069832` | `28` | `19` |
| `MNIST` | `15` | `1.0` | `0.07142278054068385` | `0.0` | `0.09866469341472103 / 0.17221848217271524` | `15` | `10` |

Representative group rows：

| axis | group | base share | Core77 share | precision | V LCB | LDO if removed |
|---|---|---:|---:|---:|---:|---:|
| `dataset_id` | `Fashion-MNIST` | `0.42628650904033377` | `0.44155844155844154` | `1.0` | `0.2185522558920942` | `0.627906976744186` |
| `dataset_id` | `KMNIST` | `0.28789986091794156` | `0.36363636363636365` | `1.0` | `0.07778373642602063` | `0.6938775510204082` |
| `dataset_id` | `MNIST` | `0.2858136300417246` | `0.19480519480519481` | `1.0` | `0.07142278054068385` | `0.5483870967741935` |
| `family_id` | `873234` | `0.18219749652294853` | `0.18181818181818182` | `1.0` | `0.16115546166474742` | `0.46031746031746035` |
| `memory_bucket` | `memory_fail_0` | `0.17767732962447844` | `1.0` | `1.0` | `0.16402807605399972` | `0.0` |
| `offdiag_bucket` | `offdiag_fail_0` | `0.13803894297635605` | `1.0` | `1.0` | `0.16402807605399972` | `0.0` |

判断：P1 把 Core77 blocker 拆清楚了。它不是单 family/template/step pocket；它是一个质量很强但 support/value 在 dataset 间不均、且 bootstrap 下稳定 LDO fail 的 accepted region。

## 5. P2 OldOnly mechanism anatomy

Artifacts：

```text
p2_oldonly_mechanism_anatomy_v9750.csv
set_quality_table_v9750.md
```

Summary：

```text
OldOnly_count = 69
OldOnly_precision = 0.855072463768116
OldOnly_V_LCB = 0.13450092269587752
OldOnly_longrisk_UCB = 0.0
OldOnly_dataset_count = 3
OldOnly_template_count = 69
OldOnly_mean_exact_transfer_lcb = -0.00023737220316525016
ExactOnly_precision = 0.014492753623188406
Intersection_precision = 0.9444444444444444
P2_oldonly_continue_research_pass = 1
oldonly_single_group_pocket = 0
```

Sets：

| set | count | precision | V LCB | LDO | mean exact T4 | mean WT80 |
|---|---:|---:|---:|---:|---:|---:|
| `Intersection` | `18` | `0.9444444444444444` | `0.0866484084297124` | `0.38888888888888884` | `0.0022394793669712008` | `0.4535725401176785` |
| `OldOnly` | `69` | `0.855072463768116` | `0.13450092269587752` | `0.391304347826087` | `-0.00023737220316525016` | `0.010067179508415913` |
| `ExactOnly` | `69` | `0.014492753623188406` | `-0.3037141101011471` | `0.0` | `0.0002671318147075799` | `0.10113173453042902` |
| `Neither` | `2720` | `0.0007352941176470588` | `-0.8053213858144195` | `0.0` | `-911397058.8235402` | `0.0043356916247338185` |

判断：OldOnly 仍是最值得解释的 high-value diagnostic：它跨 3 个 dataset、69 个 template、34 个 family，且不是 single-group pocket。但当前 exact/windowed transfer 不能解释它。

## 6. P3 transfer mismatch deep autopsy

Artifacts：

```text
p3_transfer_mismatch_deep_autopsy_v9750.csv
p3_response_vector_actions_v9750.csv
```

Summary：

```text
action_row_count = 412
OldOnly_mean_T80 = 0.010067179508415913
OldOnly_mean_exact_T4 = -0.00023737220316525016
OldOnly_mean_actual_V = 0.18785132119344358
ExactOnly_mean_T80 = 0.10113173453042902
ExactOnly_mean_exact_T4 = 0.0002671318147075799
ExactOnly_mean_actual_V = -0.25884870676985144
H3a/H3b/H3c = 1 / 1 / 1
P3_transfer_autopsy_pass = 1
new_generated_objective_evidence = 0
```

Set response summary：

| set | rows | T1 | T5 | T20 | T80 | exact T4 | actual V | precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Intersection` | `18` | `0.019771466485516143` | `0.1803475689825304` | `0.3017556179215014` | `0.4535725401176785` | `0.0022394793669712008` | `0.19201479022934415` | `0.9444444444444444` |
| `OldOnly` | `69` | `-0.02750750261767131` | `-0.07741205840525932` | `-0.052716391152437515` | `0.010067179508415913` | `-0.00023737220316525016` | `0.18785132119344358` | `0.855072463768116` |
| `ExactOnly` | `69` | `0.013445590569027732` | `0.026958710986323797` | `0.05978104289846945` | `0.10113173453042902` | `0.0002671318147075799` | `-0.25884870676985144` | `0.014492753623188406` |
| `NeitherSample` | `256` | `-0.0008535288570166913` | `0.0003895854999520485` | `0.0007321842123660405` | `0.0013400778928845758` | `-929687500.0000095` | `-0.8438881581695569` | `0.0` |

判断：P3 证明 transfer objective 仍错配。ExactOnly 的 transfer response 更高，但 actual value/precision 极差；OldOnly 的 transfer response 弱甚至短窗为负，但 actual value 强。因此 transfer 不能作为 selector，也没有给出可生成 objective。

## 7. P4 dataset-blind stability repair

Artifact：

```text
p4_dataset_blind_stability_repair_v9750.csv
```

Summary：

```text
rule_count = 8
weak_pass_count = 0
strong_pass_count = 0
best_rule_id = S7-Core77-plus-frozen-expansion
best_precision = 0.8850574712643678
best_V_LCB = 0.14120469391749882
best_LDO_drop = 0.39080459770114945
P4_dataset_blind_stability_weak/strong = 0 / 0
```

Representative rules：

| rule | precision | V LCB | longrisk | LDO | max family share | pass |
|---|---:|---:|---:|---:|---:|---:|
| `S1-raw-OldRank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.1724137931034483` | `0` |
| `S2-family-step-z-normalized` | `0.42528735632183906` | `-0.1253167350532614` | `0.054480608015849245` | `0.21839080459770113` | `0.4367816091954023` | `0` |
| `S4-oldrank-memory-offdiag-veto` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `0.1724137931034483` | `0` |
| `S5-oldrank-response-stability-veto` | `0.5057471264367817` | `-0.014785440663710028` | `0.0` | `0.21839080459770122` | `0.1839080459770115` | `0` |
| `S6-support-balanced-topK` | `0.8045977011494253` | `0.12779778491856425` | `0.0` | `0.367816091954023` | `0.09195402298850575` | `0` |
| `S7-Core77-plus-frozen-expansion` | `0.8850574712643678` | `0.14120469391749882` | `0.0` | `0.39080459770114945` | `0.1724137931034483` | `0` |
| `S8-exact-core-safe` | `0.20689655172413793` | `-0.22226101681921628` | `0.0` | `0.08045977011494251` | `0.11494252873563218` | `0` |

判断：P4 未过。dataset-blind normalization/balancing/veto 可以移动 tradeoff，但没有同时保住 quality/value 与 LDO。

## 8. P5 Core77 expansion responsibility v2

Artifact：

```text
p5_core77_expansion_responsibility_v2_v9750.csv
```

Summary：

```text
candidate_count = 8
expansion_pass_count = 0
best_expansion_id = E0-Core77-only
best_full87_precision = 1.0
best_full87_V_LCB = 0.16402807605399972
best_full87_LDO_drop = 0.4415584415584416
P5_expansion_v2_pass = 0
existing_action_theory_reset_required = 1
```

Representative expansions：

| expansion | expansion count | expansion precision | full accepted | full precision | full V LCB | full LDO | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E0-Core77-only` | `0` | `0` | `77` | `1.0` | `0.16402807605399972` | `0.4415584415584416` | `0` |
| `E1-old-rank-top10` | `10` | `0.0` | `87` | `0.8850574712643678` | `0.14120469391749882` | `0.39080459770114945` | `0` |
| `E2-OldOnly-value-risk-top10` | `10` | `0.0` | `87` | `0.8850574712643678` | `0.1413164676477515` | `0.39080459770114945` | `0` |
| `E3-response-vector-stable-top10` | `10` | `0.0` | `87` | `0.8850574712643678` | `0.05893395772286003` | `0.39080459770114945` | `0` |
| `E5-worst-split-balanced-top10` | `10` | `0.0` | `87` | `0.8850574712643678` | `0.14049607449453821` | `0.39080459770114945` | `0` |
| `E7-random-clean-risk-diagnostic-top10` | `10` | `0.0` | `87` | `0.8850574712643678` | `0.10434591706098278` | `0.39080459770114945` | `0` |

判断：P5 未过。v9.7.4 的结论被加强：换 expansion10 不解决问题；Core77 自身 LDO 已经足以阻断 controller。

## 9. P6 OldRank mechanism ablation v2

Artifact：

```text
p6_oldrank_mechanism_ablation_v2_v9750.csv
```

Summary：

```text
ablation_count = 9
mechanism_pass_count = 0
best_ablation_id = A0-OldRank
best_precision = 0.8735632183908046
best_V_LCB = 0.14111334880346277
best_LDO_drop = 0.37931034482758624
best_OldOnly_overlap = 1.0
P6_oldrank_mechanism_explained = 0
P6_generative_mechanism_ready = 0
```

Representative ablations：

| ablation | precision | V LCB | longrisk | LDO | OldOnly overlap | pass |
|---|---:|---:|---:|---:|---:|---:|
| `A0-OldRank` | `0.8735632183908046` | `0.14111334880346277` | `0.0` | `0.37931034482758624` | `1.0` | `0` |
| `A1-value-only` | `0.4942528735632184` | `0.28517470943892703` | `0.5053405200342863` | `0.06896551724137934` | `0.4927536231884058` | `0` |
| `A2-signal-risk` | `0.09195402298850575` | `-0.2353681860871466` | `0.0` | `0.0` | `0.10144927536231885` | `0` |
| `A4-support-balanced-oldrank` | `0.8045977011494253` | `0.12779778491856425` | `0.0` | `0.367816091954023` | `0.927536231884058` | `0` |
| `A7-response-stable-oldrank` | `0.5057471264367817` | `-0.014785440663710028` | `0.0` | `0.21839080459770122` | `0.5072463768115942` | `0` |
| `A8-memory-offdiag-only` | `0.10344827586206896` | `-0.30356449672984037` | `0.0` | `0.022988505747126436` | `0.18840579710144928` | `0` |

判断：P6 未解释 OldRank。原始 OldRank 保留质量但 LDO 高；支持均衡能保留部分 OldOnly overlap，但 LDO 仍高；低 LDO 单因子规则质量崩。OldRank 仍只能作为 diagnostic，不能转 controller 或 generated objective。

## 10. Controller / runtime / generated route / system boundary

P7：

```text
p7_existing_action_minimal_controller_boundary_v9750.csv = not_run
reason = P4_P5_P6_no_weak_pass
controller_pass = 0
source_controller_pass = 0
```

P8：

```text
p8_selected_runtime_boundary_v9750.csv = not_run
reason = P7_controller_not_passed
selected_runtime_pass = 0
```

P9：

```text
p9_generated_route_stop_reopen_decision_v9750.csv
source_generated_route_status_v9740 = stopped_no_new_objective
new_objective_evidence_present = 0
oldrank_generative_mechanism_ready = 0
controller_pass = 0
generated_route_status = stopped_no_new_objective
APGU_APGV_APGW_APGX_run = 0
direct_solved_sandbox_allowed = 0
generated_action_count = 0
branch_horizon_rows = 0
generated_route_stop_triggered = 1
```

P10/P11：

| artifact | status / reason |
|---|---|
| `p10_paired_replay_boundary_v9750.csv` | `not_run`, `P7_or_P8_not_passed` |
| `p11_short_full_boundary_v9750.csv` | `not_run`, `P10_paired_replay_not_open` |

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "generated_route_allowed": 0,
  "direct_solved_sandbox_allowed": 0,
  "APGU_APGV_APGW_APGX_run_allowed": 0
}
```

Stop conditions：

```json
{
  "enter_system": 0,
  "stop_core_expansion_threshold_patching": 1,
  "stop_oldrank_controller_promotion": 1,
  "stop_transfer_selector_route": 1,
  "stop_generated_blind_variants": 1
}
```

判断：没有 P4/P5/P6 weak pass，因此 controller/runtime/system/paired replay/short-full 全部关闭。没有新 generated objective，因此 APG* blind variants 继续停止。

## 11. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9750.csv
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
fig_p1_core77_dataset_precision.svg
fig_p1_core77_group_share.svg
fig_p2_set_quality.svg
fig_p3_response_vector_precision.svg
fig_p4_rule_precision.svg
fig_p4_rule_ldo.svg
fig_p5_expansion_ldo.svg
fig_p6_oldrank_ablation_precision.svg
```

判断：这些图只用于复核诊断，不构成 official pass。

## 13. No-fake audit

```text
rows_checked = 611
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
v9740_boundary_pass = 1
core77_autopsy_pass = 1
oldonly_anatomy_pass = 1
transfer_autopsy_pass = 1
dataset_blind_stability_pass = 0/0
expansion_v2_pass = 0
oldrank_mechanism_explained = 0
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
route = R2-ExistingActionHighQualityButLDOBlocked
F0_boundary_fail = 0
F1_core77_ldo_root_cause_present = 1
F2_oldonly_high_quality_unexplained = 1
F3_transfer_objective_mismatch = 1
F4_dataset_blind_stability_fail = 1
F5_expansion_v2_fail = 1
F6_controller_runtime_blocked = 1
F7_generated_route_stopped = 1
F8_system_not_official = 1
primary_blocker = existing_action_ldo_blocked
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `c59169ee1d09749610136362a88b264939ac6a0424c6c4829d63b9e91e5575b2` |
| runner | `0f73c81a8ddfe0d329da9f95c90a8c9a758e093f8c050e2647f19df21cc5b4aa` |
| run manifest | `83dff3468804a15ba9455bd565d5ffd9f71c1ddb71ad5eae4093e22f967dd1e7` |
| route | `004a82f025b3b2062624f23f28f6b0c90c25ddc76631619c18210a16756e8c54` |
| P0 boundary | `cadc2fc8203de7191ec8a517b921e5df925e1e4617acbf0878dedc658b9f3c85` |
| P0 field legality | `d8de87c4d58435247ef0c9b85643a6f565cf59c9e9e77f1dbdedda15960364a3` |
| P1 Core77 autopsy | `3846b959deaf199bf17b654e5970b672fe7306b052d7665ef9ceae61cb49bdeb` |
| P1 Core77 group autopsy | `5d483a4d1d715f68d44ed6eda3fb6e9b914528704af02c4bc71abf531b2140dd` |
| P2 OldOnly anatomy | `60f71c5df29d3ffb78bf0f30d19d31ffbee2ba3d714a740653bb5c3fa4efa292` |
| P3 transfer autopsy | `d4b8ec5d6c73c21030701ea631912b63297bea9e69ffa17305987c062926b061` |
| P3 response actions | `6c615d4dc53348eaf8cedbb22059bef6098021dcdf4a968b15399e9bb9efbdd1` |
| P4 stability repair | `726d861a2e79b0ad5a749484b8c98dbc257b1028a645f12d88241fd4f6a21269` |
| P5 expansion v2 | `cf6b0b31eecd84c9475052d0a8bed88fd48c4a0e250a54a5eec40e9e9c9dda09` |
| P6 OldRank ablation v2 | `9e6bb0bf0d8905a940962d8111ff6f00102c9c86fe481efc90cccc8f5862161d` |
| P7 controller | `cc3a1f0bd37dd52de473d0a02372391acc7fc26eb0f3390ca2064fe5da050e7c` |
| P8 runtime | `b1ec3d34e12a14c00b20905604f459ca8445295094beb3d77d0aed0aeeb89d33` |
| P9 generated route | `6cb52b0991c27fd9da9288b77df45305c592f8cd2f28a5935ad22cc17f724283` |
| P10 paired replay | `b8cfdfbcc962acfc6d3ec37ab9fa1c242b66b0ada1fee35081daadb05076ef8b` |
| P11 short/full | `671dd51c9e9515b63ee7b8bc0c6ab8ed77bad4f98f3bbf992ebebf3064442c04` |
| Base-Acc Sentinel | `d91682d092782cbcdb8869446e16ebacf399706e24f820ff59f673cfe191af3f` |
| no-fake audit | `aada8a801772ec2bbf512960141ba92bc35b4fc40e6954e65724f886913255ce` |
| contract audit | `b5aeb7cfaf1cfb2b9da1c2593a85cca0cf65c4a2629d30bfa1069e4997b50394` |
| failure taxonomy | `bd2f4d65e1ee6641b5dd9749ebd420ab9d882f818989f0cea9fd90c0ca283646` |
| allowed next gates | `f082f8dd02c7582daef83d9faaa7d26a31c7bc69c09ed5df61d871d91d706a41` |
| stop conditions | `46b2f95fc5aa50cb51ef807a25adb4964fda702c2a58aeba0d4b2da7e23a1385` |

## 15. 最终分析结论

v9.7.5 的真实推进是：

```text
v9.7.4:
  exact transfer mismatch 已被复盘清楚；
  Core77 本身而不是 expansion10 是 LDO blocker；
  horizon transfer materializer 真实闭合但 selector 不过；
  generated route 继续 stopped_no_new_objective。

v9.7.5:
  进一步证明 Core77 的 LDO fail 是 dataset/support shift + structural LDO，而不是 family/template/step 单 pocket；
  OldOnly 仍是高质量、跨 dataset/template 的真实 diagnostic 集合；
  response-vector autopsy 显示 ExactOnly/WT 更偏向 safe-looking but low-value 动作；
  dataset-blind stability repair 不能同时保住 precision/value 与 low LDO；
  expansion v2 继续证明换补 10 个动作不解决 Core77 自身 LDO；
  OldRank ablation 没有提炼出 green-field minimal mechanism；
  controller/runtime/generated route/paired replay 全部 gate-blocked。
```

机制判断：

1. H0 成立：v9.7.4 boundary 被复现，没有跳过 existing-action LDO / transfer mismatch / generated stop。
2. H1a/H1c 成立：Core77 的 LDO fail 是 dataset/support shift 且结构性稳定；H1b 未成立，没有 single hidden group concentration 主解释。
3. H2 成立于诊断层：OldOnly 高质量、跨 dataset/template，不是 exact transfer 能解释的低维交集。
4. H3 成立：one-step/windowed transfer objective 与真实好动作错配；ExactOnly response 更高但 actual V 为负。
5. H4 未成立：dataset-blind repair 没有打开 weak/strong pass；高质量规则 LDO 高，低 LDO 规则 quality/value 崩。
6. H5 未成立：所有 expansion v2 仍失败；Core77 自身 LDO fail，不能靠补 10 个动作修。
7. H6 未成立：OldRank 的最小 green-field mechanism 未解释，不能转 controller 或 generated objective。
8. P7/P8 未打开：没有 weak controller candidate，因此 existing-action controller 与 selected runtime 均 not_run。
9. P9 成立：generated route 继续 stopped_no_new_objective；direct solved sandbox 和 APG* blind variants 都不允许。
10. P10/P11 未打开：没有 system controller，就不能打开 paired replay 或 short/full training。
11. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.5 真实执行后停在 `R2-ExistingActionHighQualityButLDOBlocked`：高质量 existing actions 确实存在，OldOnly 也不是单 pocket，但 Core77 的 LDO fail 是结构性 dataset/support shift；transfer response 与真实 value 错配，dataset-blind repair、expansion v2 和 OldRank ablation 都没有形成 official controller，因此 strict PureKAN functional 仍未成功。
