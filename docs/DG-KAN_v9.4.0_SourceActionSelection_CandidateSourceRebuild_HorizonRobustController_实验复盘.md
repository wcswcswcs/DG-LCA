# DG-KAN v9.4.0 Source Action Selection / Candidate Source Rebuild / Horizon-Robust Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.4.0_SourceActionSelection_CandidateSourceRebuild_HorizonRobustController_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 oracle source selector、biased source-panel diagnosis、legal-feature diagnostic 或未 materialize 的 AP0b-AP0f source generator 写成 official system pass。

## 0. 最新结论

```text
route = R3-LegalSourceSelectorOpaque
base_candidate = LQ-t2-h256
success_v9400_strict_purekan_functional = False
success_v9400_full_functional = False
success_v9400_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9400_source_action_selection_candidate_source_rebuild_horizon_controller_first_20260514T070000Z/
```

核心结论：

1. P0 复现 v9.3.9 boundary：source route = `R1-SourceAP0PanelBad`，source panel action = `64`，full AP0 action universe = `2876`。
2. P0 join 过：source panel joined action = `64 / 64`，但 source panel 明显不代表 full universe。
3. full AP0 universe row weak CP rate = `0.1035002318034307`，source panel weak CP row rate = `0.0625`；route flip = `1`。
4. source panel bias 很强：selection_bias_PSI = `25.80667683866949`，selection_bias_KL = `2.434227117471533`，max family gap = `0.38030250347705147`，max step bucket gap = `0.5111700278164117`。
5. P1 provenance 明确：v9.3.8/v9.3.9 source panel 是 `v9380_first_N_payload_available_slice`，reason = `smoke_convenience_slice_first64_payload_available`；没有 outcome/dataset/future usage。
6. P2 oracle 证明 AP0 source frontier 不是 absent：best oracle = `OS2-weak-CP-oracle-source-selector` at K=64，weak CP precision all rows = `0.734375`，V_ctrl LCB = `0.11206787715966769`，long-risk = `0.109375`。
7. P2 strict/weak official oracle gates 仍未全过：oracle_source_pass = `0`，oracle_source_weak_pass = `0`，因为 K=64 long-risk 略高于 strict `0.10`，K=256 V_ctrl LCB 为负；但 `oracle_source_present = 1`，不能再写 `AP0CandidateSourceOracleAbsent`。
8. P3 legal source selector failed：best legal selector = `LSS1-single-feature-topK-LSB-PayloadLinf`，weak CP h20 = `0.140625`，weak CP all = `0.171875`，V_ctrl LCB = `-0.24349444404221637`，long-risk = `0.2760416666666667`。
9. 当前 legal feature best AUC only `0.5861789765860932`；legal_source_selector_pass = `0`，legal_source_selector_weak_pass = `0`。
10. P4 AP0b-AP0f real source generators 未 materialize：source_generator_materialized = `0`；没有伪造 linearized gradients、cert tensors 或 h20 outcome rows。
11. P5-P13 全部 gate-blocked：没有 source generator / selected controller / selected runtime，因此 h20 smoke、horizon extension、certificate redesign、controller、runtime、paired replay、short/full 全部 not_run。
12. 当前 primary blocker：`legal_source_selector_opaque`；下一步需要实现真实 AP0b-AP0f value-producing source generators with commit-time certificates。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9400_source_action_selection_candidate_source_rebuild.py` | v9.4.0 runner；读取 v9.3.5 full AP0 outcome universe、v9.3.9 source panel 和 v9.3.3 durable payload，执行 full-vs-panel、source provenance、oracle upper-bound、legal selector capacity，并按 gate 阻断未 materialize 的 source generator / controller / runtime |

代码检查：

```text
python -m py_compile experiments/run_v9400_source_action_selection_candidate_source_rebuild.py
```

正式运行：

```bash
python experiments/run_v9400_source_action_selection_candidate_source_rebuild.py \
  --out-dir results/real_rerun_20260506/v9400_source_action_selection_candidate_source_rebuild_horizon_controller_first_20260514T070000Z \
  --fresh --device auto --data-root data --seed 1314 --source-panel-actions 64
```

运行结果：

```json
{
  "legal_source_selector_pass": 0,
  "oracle_source_weak_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9400_source_action_selection_candidate_source_rebuild_horizon_controller_first_20260514T070000Z",
  "route": "R3-LegalSourceSelectorOpaque"
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R3-LegalSourceSelectorOpaque",
  "source_route_v9390": "R1-SourceAP0PanelBad",
  "full_action_count": 2876,
  "source_panel_action_count": 64,
  "source_panel_join_pass": 1,
  "full_weak_CP_row_rate": 0.1035002318034307,
  "source_panel_weak_CP_row_rate": 0.0625,
  "selection_bias_PSI": 25.80667683866949,
  "selection_bias_KL": 2.434227117471533,
  "route_flip_full_vs_source_panel": 1,
  "source_panel_bias_pass": 0,
  "selection_rule_identified": 1,
  "source_selection_rule": "v9380_first_N_payload_available_slice",
  "oracle_source_present": 1,
  "oracle_source_pass": 0,
  "oracle_source_weak_pass": 0,
  "best_oracle_selector_id": "OS2-weak-CP-oracle-source-selector",
  "best_oracle_K": 64,
  "best_oracle_weak_CP_precision_all_rows": 0.734375,
  "best_oracle_V_ctrl_lcb_all": 0.11206787715966769,
  "best_oracle_long_risk_rate": 0.109375,
  "legal_source_selector_pass": 0,
  "best_legal_selector_id": "LSS1-single-feature-topK-LSB-PayloadLinf",
  "best_legal_weak_CP_precision_h20": 0.140625,
  "best_legal_weak_CP_precision_all": 0.171875,
  "best_legal_V_ctrl_lcb_all": -0.24349444404221637,
  "best_legal_long_risk_rate": 0.2760416666666667,
  "source_generator_materialized": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legal_source_selector_opaque"
}
```

判断：v9.4.0 把 v9.3.9 的 `source_AP0_same_panel_bad` 进一步拆开了。AP0 full universe 不是完全没有好 source，source panel 也确实 biased；但当前 legal commit-time selector 仍不能稳定找到这些 oracle-good actions。

## 3. P0 full-vs-source-panel diagnosis

Artifacts：

```text
p0_full_vs_source_panel_diagnosis.csv
source_panel_join_trace_v9400.csv
source_panel_bias_trace_v9400.csv
```

Summary：

```text
full_action_count = 2876
source_panel_action_count = 64
source_panel_joined_action_count = 64
source_panel_join_pass = 1
source_panel_fraction = 0.022253129346314324

full_row_count = 8628
full_weak_CP_row_rate = 0.1035002318034307
full_strong_CP_row_rate = 0.059921186833565135
full_long_risk_rate = 0.30076495132127956
full_V_ctrl_lcb = -0.6649802417404238
full_horizon_robust_action_rate = 0.004520166898470098

source_panel_row_count = 192
source_panel_weak_CP_row_rate = 0.0625
source_panel_strong_CP_row_rate = 0.057291666666666664
source_panel_long_risk_rate = 0.3072916666666667
source_panel_V_ctrl_lcb = -1.203869591724189
source_panel_horizon_robust_action_rate = 0.0

selection_bias_PSI = 25.80667683866949
selection_bias_KL = 2.434227117471533
max_family_gap = 0.38030250347705147
max_step_bucket_gap = 0.5111700278164117
max_score_bucket_gap = 0.26881954102920724
route_flip_full_vs_source_panel = 1
source_panel_bias_pass = 0
```

判断：P0 pass for diagnosis，fail for representativeness。source panel 是 full universe 的 `2.23%`，family / step / payload norm 分布偏移明显；v9.3.9 的 bad panel 不能代表 full AP0 universe。

## 4. P1 source selection provenance

Artifacts：

```text
p1_source_selection_provenance_audit.csv
source_selection_trace_v9400.csv
source_selection_unknown_reason_table_v9400.csv
```

Summary：

```text
selection_rule_identified = 1
unknown_selection_reason_count = 0
uses_outcome_in_source_selection = 0
uses_dataset_name_in_source_selection = 0
uses_future_step_in_source_selection = 0
source_panel_family_concentration = 0.5625
source_panel_step_concentration = 0.046875
source_panel_selection_bias_pass = 0
selection_rule = v9380_first_N_payload_available_slice
reason = source_panel_is_first_N_payload_available_convenience_slice_not_value_selector
```

判断：P1 解释了为什么 v9.3.8/v9.3.9 source panel 会差：它不是 legal value selector，而是 first-N payload-available smoke/convenience slice。它没有违规使用 outcome 或 dataset_name，但也没有选择 value-positive source 的能力。

## 5. P2 oracle source selector upper-bound

Artifacts：

```text
p2_oracle_source_selector_upper_bound.csv
oracle_source_selector_trace_v9400.csv
random_panel_bootstrap_trace_v9400.csv
```

Best oracle：

```text
oracle_selector_id = OS2-weak-CP-oracle-source-selector
K = 64
weak_CP_precision_all_rows = 0.734375
strong_CP_precision_all_rows = 0.3854166666666667
V_ctrl_lcb_all = 0.11206787715966769
long_risk_rate = 0.109375
support_balance_pass = 1
accepted_family_count = 34
max_family_share = 0.09375
weak_CP_h20 / h80 / h240 = 0.703125 / 0.828125 / 0.671875
oracle_source_present = 1
oracle_source_pass = 0
oracle_source_weak_pass = 0
```

判断：P2 的关键不是 official pass，而是排除 `AP0CandidateSourceOracleAbsent`。K=64 oracle source panel value LCB 为正、weak CP 很高，只是 long-risk `0.109375` 略高于 strict `0.10`；因此 full AP0 universe 里有足够强的 upper-bound 信号，但它仍是 oracle diagnostic，不能 official。

## 6. P3 legal source selector capacity

Artifacts：

```text
p3_legal_source_selector_capacity.csv
legal_source_feature_trace_v9400.csv
source_selector_frontier_trace_v9400.csv
source_selector_leaveout_trace_v9400.csv
```

Best legal selector：

```text
selector_id = LSS1-single-feature-topK-LSB-PayloadLinf
feature_set = LSB-PayloadLinf
K = 64
feature_auc_weak_CP = 0.5861789765860932
weak_CP_precision_h20 = 0.140625
weak_CP_precision_all_rows = 0.171875
strong_CP_precision_all_rows = 0.0625
V_ctrl_lcb_all = -0.24349444404221637
long_risk_rate = 0.2760416666666667
support_balance_pass = 1
legal_source_selector_pass = 0
legal_source_selector_weak_pass = 0
```

Other diagnostic references：

```text
LSA-StateNLL AUC = 0.6151712728688206, K64 weak_CP_all = 0.08333333333333333
LSA-StateCEp99 AUC = 0.6067577348832358, K64 weak_CP_all = 0.140625
LSA-StateMarginP10 AUC = 0.6108385679430719, K64 weak_CP_all = 0.10416666666666667
LSB-PayloadNorm AUC = 0.5886251866777592, K64 weak_CP_all = 0.15104166666666666
```

判断：P3 是本轮 final blocker。oracle 能找到，但现有 legal commit-time features 只能做到 h20 weak CP `0.140625`、all-row weak CP `0.171875`，且 V_ctrl LCB 为负、long-risk 高；不能 official，也不能继续跑 selected controller。

## 7. P4-P13 gated boundary

P4：

```text
stage = P4_VALUE_PRODUCING_SOURCE_GENERATOR
status = not_run
reason = legal_source_selector_failed_and_AP0b_AP0f_real_generators_not_materialized
source_generator_materialized = 0
next_required_implementation = implement_real_AP0b_AP0f_source_generators_with_commit_time_certificates
```

P5-P13：

| artifact | reason |
|---|---|
| `p5_h20_immediate_direction_smoke.csv` | `P4_source_generator_not_materialized` |
| `p6_horizon_extension_longrisk_audit.csv` | `P5_immediate_direction_not_available` |
| `p7_source_to_generated_transform_damage.csv` | `P6_source_survivor_not_available` |
| `p8_effect_valid_certificate_redesign.csv` | `P6_source_survivor_not_available` |
| `p9_minimal_source_certificate_controller.csv` | `P8_certificate_or_source_survivor_not_available` |
| `p10_selected_source_controller_online_runtime.csv` | `P9_controller_not_selected` |
| `p11_system_integration_gate_v9400.csv` | `P9_controller_not_selected` |
| `p12_leaveout_and_paired_replay_boundary.csv` | `P11_system_controller_not_official` |
| `p13_short_full_sampleeff_continual_robustness.csv` | `P12_official_paired_replay_not_open` |

没有把 oracle selector、biased-panel diagnosis、legal-feature diagnostic 或 missing generator placeholder 写成 downstream success。

## 8. No-fake audit

```text
rows_checked = 24276
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
full_vs_panel_join_pass = 1
source_selection_provenance_pass = 1
oracle_source_pass = 0
oracle_source_weak_pass = 0
legal_source_selector_pass = 0
source_generator_materialized = 0
h20_immediate_direction_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
source_measured_gap_used/formula_proxy_used = 0/0
diagnostic_promoted_to_official = 0
```

Failure table：

```text
F1_source_panel_sampling_bias = 1
F2_oracle_source_absent = 0
F3_legal_source_selector_opaque = 1
F4_source_generator_missing = 1
F5_immediate_direction_not_run = 1
F11_system_not_official = 1
primary_blocker = legal_source_selector_opaque
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `e008d2b676c11a08ea724ee52afd74f74e47052864acf32ed8754ae1b0a29b4b` |
| runner | `24701a1abb5b1e840cd9090af88ee3dd396a4f5ac5fb704105a697ffd66cf2e3` |
| run manifest | `97f846fecc6fb766a531ba903695c013361420cf88e8c392b80ab956c63da8d3` |
| route | `84557905521b17429903dfb2b3d1859dae485128a41c2c489e8bc1cd2a6cde57` |
| P0 full vs panel | `da7279d84d19b79f665fc5a9d11a610641a13814a44641ea6456c7ad528652c9` |
| P1 provenance | `4b0093cf990ea1573cf2e4cb6c67a7e11dd1e9addb93ea41597e961ffa90bceb` |
| P2 oracle | `754dc2ae8432ff7bd292040af5903f0e61848ae1d99ab3e0d9fec44152e960cf` |
| P3 legal selector | `0129f4f9ad50849807e601aa657a74137733c3d662e5d76fea4722805acecfb1` |
| P4 source generator | `f1aa102b4b38c70dc613f592176066fc9daafdd39b45ba3692d83dc2439c7562` |
| P11 system | `d14ebce5f6316426b0a2d57ed8d73ede22571eb8826dc5724d1ad3079fb64428` |
| contract audit | `7db8626a73c968fb5c7f8fdbce9b30179e26310f4d4d3da1144e11568039baf9` |
| provenance audit | `42c6973b3a7fd14193a48bcd340c2d85b4a644ffda416666a23ad1e49508c631` |
| failure table | `b273f6b38888c639a79ca1a49a7322130a3d4ab768fcdbe990352f4a55255d41` |

## 10. 最终分析结论

v9.4.0 的真实推进是：

```text
v9.3.9:
  同源 AP0 source panel 本身 value-poor / long-risk high；
  AP5-AP8 low-distortion generation 也不能恢复 frontier。

v9.4.0:
  证明 v9.3.9 source panel 是 biased first-N payload convenience slice；
  证明 full AP0 universe 下 oracle source frontier 仍存在；
  但当前 legal commit-time source selector 无法找到这个 frontier；
  repo 当前也没有真实 AP0b-AP0f value-producing source generator materialization。
```

机制判断：

1. H0 成立但不到 official：AP0 full universe 有 oracle source frontier，best K64 weak CP precision = `0.734375`。
2. H1 成立：source panel representativeness fail，PSI = `25.8067`，route flip = `1`。
3. H2 未成立：legal source selector failed；best legal all-row weak CP = `0.171875`，V_ctrl LCB = `-0.24349`。
4. H3-H7 未打开：没有真实 AP0b-AP0f source generator，因此不能跑 h20 immediate smoke、horizon extension、transform damage、certificate controller 或 selected runtime。
5. 本轮不能写成 source/controller official：oracle selector 和 legal-feature capacity 都只是 diagnostic。

最终一句话：

> v9.4.0 真实执行后停在 `R3-LegalSourceSelectorOpaque`：full AP0 universe 中存在 oracle-good source actions，但 v9.3.9 的 first-64 source panel 是明显偏置的 convenience slice，当前 legal commit-time features 又找不到 oracle frontier；strict PureKAN functional 仍未成功，下一步必须实现真实 AP0b-AP0f value-producing source generators。
