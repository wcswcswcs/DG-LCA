# DG-KAN v10.6 Local Future Response Operator 实验复盘

> 本复盘记录 `DG-KAN_v10.6_LocalFutureResponseOperator_结果解读与完整实验计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/manifest、v10.5/v10.3/v10.1 真实 artifact、本轮真实 LFRO 小扰动 payload replay 与真实 branch-horizon rows；没有 fake data、proxy rows、占位数据或 CPU offload。controller/runtime/paired replay 仅在硬 gate 通过后打开。

## 0. 最新结论

```text
route = CaseD-NoRiskSafeLocalFutureResponse
primary_blocker = lfro_no_risk_safe_positive_local_response
secondary_blocker = representation_or_basis_insufficient
system_legal_controller_pass = 0
generated_route_status = stopped_no_risk_safe_lfro
```

最终 artifact：`results/real_rerun_20260506/v1060_local_future_response_operator_full_20260518T230000Z`

核心结论：

1. P0 v10.5 boundary lock pass = `1`；source route = `CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient`。
2. P1 Fast/Slow/Risky/SafeLow/Bad = `4` / `22` / `321` / `26` / `14707`；discovery/strong = `0` / `0`。
3. P2 states/actions/branch rows = `8` / `320` / `5760`；materializer/weak/strong = `1` / `0` / `0`；best V20 = `1.8190300448331982`。
4. P3 weak/strong = `0` / `0`；best model = `LFRO-L3-robust-huber-clipped`，validation R2 V20/risk = `-0.6366622673887112` / `-0.9202397748287701`。
5. P4 weak/strong = `0` / `0`；all8 FastSlow precision/V240/longrisk = `0.0` / `-0.8158580619097676` / `1.0`。
6. P5 Stage4/8/16/32/64 opened = `0` / `0` / `0` / `0` / `0`；discovery = `0`。
7. P6 dominant failure = `F1-path-label-too-sparse-even-after-P1`；P7/P8/P9 = `not_run` / `not_run` / `not_run`；No-fake rows checked = `22248`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮命令

```text
python -m py_compile experiments/run_v1060_local_future_response_operator.py
```

```bash
python experiments/run_v1060_local_future_response_operator.py --out-dir results/real_rerun_20260506/v1060_local_future_response_operator_full_20260518T230000Z --fresh --device auto --data-root data --seed 1818 --execution-profile full-gated
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V1060",
  "status": "summary",
  "route": "CaseD-NoRiskSafeLocalFutureResponse",
  "primary_blocker": "lfro_no_risk_safe_positive_local_response",
  "secondary_blocker": "representation_or_basis_insufficient",
  "route_explanation": "True local perturbations did not expose a risk-safe positive V20 response state.",
  "source_route_v1050": "CaseD-GoodConeAndUnrollBothFailRepresentationInsufficient",
  "P0_boundary_pass": 1,
  "P1_discovery_pass": 0,
  "P1_strong_pass": 0,
  "P2_materializer_pass": 1,
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P3_weak_pass": 0,
  "P3_strong_pass": 0,
  "P4_weak_pass": 0,
  "P4_strong_pass": 0,
  "P5_generated_discovery_pass": 0,
  "P5_stage32_weak_pass": 0,
  "P5_stage64_strong_pass": 0,
  "P6_pass": 1,
  "P7_controller_pass": 0,
  "P8_runtime_pass": 0,
  "P9_paired_replay_pass": 0,
  "generated_route_status": "stopped_no_risk_safe_lfro",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Path Type Audit

| metric | value |
| --- | ---: |
| known_action_rows | 15080 |
| FastGood_count | 4 |
| SlowBurnGood_count | 22 |
| RiskyHighAUV_count | 321 |
| SafeLowValue_count | 26 |
| BadPath_count | 14707 |
| FastSlow_count | 26 |
| relaxed_v1_added_count | 11 |
| P1_discovery_pass | 0 |
| P1_strong_pass | 0 |
| reason | official_fixed_path_type_fastslow_below_32; relaxed_V1_diagnostic_not_promoted_to_official |

## 4. P2 LFRO Materializer

| metric | value |
| --- | ---: |
| state_count | 8 |
| basis_count | 16 |
| planned_perturbation_count | 320 |
| materialized_perturbation_count | 320 |
| validation_perturbation_count | 64 |
| branch_horizon_expected_rows | 5760 |
| branch_horizon_actual_rows | 5760 |
| perturb_linf_nonzero_count | 320 |
| max_apply_linf | 0.0 |
| cost_per_perturb_ms_q50 | 25.439 |
| cost_per_perturb_ms_q90 | 26.215 |
| state_with_positive_V20_count | 8 |
| state_with_positive_V20_and_risk_safe_count | 0 |
| state_best_V20_max | 1.8190300448331982 |
| state_longrisk_UCB_min | 0.7577722646420829 |
| P2_materializer_pass | 1 |
| P2_weak_pass | 0 |
| P2_strong_pass | 0 |

Top P2 perturbations by V20:

| state | alpha | V1 | V5 | V20 | risk20 | perturb_linf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1defe432f627ea3779130b49 | val_rand_05 | -0.8251650272868574 | -0.15515302494168282 | 1.8190300448331982 | 0 | 7.533682492066873e-06 |
| a67a43ae048190018cb5065d | train_e10_p | -0.3196718469262123 | -0.515121285803616 | 1.418895476963371 | 0 | 2.0040819435962476e-05 |
| 1defe432f627ea3779130b49 | train_e05_m | -0.06861416157335043 | -0.4969695247709751 | 1.2045317988377064 | 0 | 7.533682492066873e-06 |
| 0ee78f0cf64de7abeae36d5a | train_e05_p | -0.12901421333663166 | -0.5097090345807374 | 0.9360138247720897 | 0 | 5.9000922192353755e-06 |
| deee57c4798cfed67207331b | train_e08_p | -0.07479804358445108 | -0.19309076177887619 | 0.906589028192684 | 0 | 8.10243818705203e-06 |
| 1defe432f627ea3779130b49 | train_e09_m | -0.30869252420961857 | -0.6988255197647959 | 0.8503209920600057 | 0 | 7.533682492066873e-06 |
| 0d5f4d323330bde350b13f35 | train_e14_p | -0.026006717002019286 | -0.5476758070290089 | 0.786313399206847 | 0 | 5.879941909370245e-06 |
| a67a43ae048190018cb5065d | train_e07_p | -1.0650252494961023 | -1.8368344376794994 | 0.7341634933836758 | 0 | 2.004082125495188e-05 |
| 0d5f4d323330bde350b13f35 | train_e04_p | -0.11902473703958094 | 0.6280974647961557 | 0.7134350496344268 | 0 | 5.879941909370245e-06 |
| deee57c4798cfed67207331b | train_e13_m | -0.23242969042621553 | -0.24682654486969113 | 0.7081142808310688 | 0 | 8.10243818705203e-06 |

## 5. P3 LFRO Fit

| metric | value |
| --- | ---: |
| model_count | 32 |
| state_count | 8 |
| best_model_id | LFRO-L3-robust-huber-clipped |
| best_state_id | 0794b147a94cab877fa8fa1a |
| best_validation_R2_V20 | -0.6366622673887112 |
| best_validation_R2_longrisk | -0.9202397748287701 |
| best_rank_of_response_operator | 16 |
| best_condition_number | 1.0 |
| best_actual_validation_V20 | -0.3381594007369131 |
| best_actual_validation_longrisk | 1.0 |
| P3_weak_pass | 0 |
| P3_strong_pass | 0 |

Top fitted models:

| model | state | R2 V20 | R2 risk | rank | actual V20 | actual risk | weak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LFRO-L3-robust-huber-clipped | 0794b147a94cab877fa8fa1a | -0.6366622673887112 | -0.9202397748287701 | 16 | -0.3381594007369131 | 1.0 | 0 |
| LFRO-L3-robust-huber-clipped | 95642ea1f06dc379d3628d6f | -0.692198989838746 | -0.007514179281276512 | 16 | -3.4201835095882416 | 1.0 | 0 |
| LFRO-L3-robust-huber-clipped | 8b0099f54d014a1b1591e142 | -0.695801713019353 | 0.03748862961312838 | 16 | -0.38783658761531115 | 1.0 | 0 |
| LFRO-L3-robust-huber-clipped | 0ee78f0cf64de7abeae36d5a | -0.8283450445148242 | -0.07001594727146077 | 16 | -0.5977159857284278 | 1.0 | 0 |
| LFRO-L3-robust-huber-clipped | deee57c4798cfed67207331b | -0.9040354169925913 | -0.5951997887841061 | 16 | -0.9028168031945825 | 1.0 | 0 |
| LFRO-L2-ridge-quadratic-diag | 95642ea1f06dc379d3628d6f | -0.9064651270686432 | -0.008501282827203527 | 32 | -3.4201835095882416 | 1.0 | 0 |
| LFRO-L4-value-risk-monotone | 95642ea1f06dc379d3628d6f | -0.9064651270686432 | -0.008501282827203527 | 16 | -3.4201835095882416 | 1.0 | 0 |
| LFRO-L1-linear-ridge | 95642ea1f06dc379d3628d6f | -0.9075856108037352 | -0.008785857035785494 | 16 | -3.4201835095882416 | 1.0 | 0 |
| LFRO-L3-robust-huber-clipped | 0d5f4d323330bde350b13f35 | -0.943098410336497 | 0.0 | 16 | -0.4685664437711239 | 1.0 | 0 |
| LFRO-L2-ridge-quadratic-diag | 8b0099f54d014a1b1591e142 | -1.0399559374042662 | 0.03676980446911671 | 32 | -0.3316534401383251 | 1.0 | 0 |

## 6. P4 Constrained Solve

| metric | value |
| --- | ---: |
| solved_update_count | 8 |
| first4_FastSlow_count | 0 |
| first4_longrisk_created_rate | 1.0 |
| first4_V240_LCB | -1.0267852326355977 |
| all8_FastGood_count | 0 |
| all8_SlowBurnGood_count | 0 |
| all8_RiskyHighAUV_count | 0 |
| all8_SafeLowValue_count | 0 |
| all8_BadPath_count | 8 |
| all8_FastSlow_precision | 0.0 |
| all8_V20_LCB | -1.148678286079418 |
| all8_V80_LCB | -1.950021292519371 |
| all8_V240_LCB | -0.8158580619097676 |
| all8_longrisk_UCB | 1.0 |
| best_solver_id | S2-quadratic-trust |
| best_path_type | BadPath |
| P4_weak_pass | 0 |
| P4_strong_pass | 0 |

P4 solved update rows:

| solver | model | path | V20 | V80 | V240 | RAUV | risk | apply |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S4-pareto-holdout | LFRO-L3-robust-huber-clipped | BadPath | -0.14277547039091587 | -0.2683685689698905 | -0.07618846697732806 | -0.059207596071064475 | 1 | 0.0 |
| S1-linear-constrained | LFRO-L4-value-risk-monotone | BadPath | -1.0017966378945857 | -0.6337007242254913 | -1.177468164358288 | -0.9330144222360104 | 1 | 0.0 |
| S3-random-search | LFRO-L3-robust-huber-clipped | BadPath | -1.0335737294517457 | 0.14850531332194805 | -0.3914666483178735 | -0.2949372727423906 | 1 | 0.0 |
| S1-linear-constrained | LFRO-L3-robust-huber-clipped | BadPath | -0.601638636784628 | -0.4203336660284549 | -0.6382111730054021 | -0.3574333034455776 | 1 | 0.0 |
| S2-quadratic-trust | LFRO-L3-robust-huber-clipped | BadPath | -1.5545672920998186 | 0.27101160818710923 | -0.06464888155460358 | -0.7841155118774623 | 1 | 0.0 |
| S6-state-specific | LFRO-L2-ridge-quadratic-diag | BadPath | -0.08507428271695971 | -3.8928948929533362 | -0.07342442823573947 | -1.2882456297520548 | 1 | 0.0 |
| S5-worst-state | LFRO-L3-robust-huber-clipped | BadPath | -1.1133672632277012 | -1.2526206511538476 | -0.4457096247933805 | -0.6798399762250483 | 1 | 0.0 |
| S2-quadratic-trust | LFRO-L1-linear-ridge | BadPath | -0.883086911868304 | -1.928864975227043 | -1.1369632855057716 | -0.9881807223428041 | 1 | 0.0 |

## 7. P5/P6/P7 Gate

| stage | status | key |
| --- | --- | --- |
| P5 | not_run | Stage8=0 Stage32=0 Stage64=0 |
| P6 | summary | F1-path-label-too-sparse-even-after-P1 |
| P7 | not_run | P5_stage64_or_stage32_plus_P3_P4_strong_gates_not_passed |
| P8 | not_run | P7_controller_not_passed |
| P9 | not_run | P7_or_P8_not_passed |

Failure attribution:

| failure_class | evidence | recommended_fix |
| --- | --- | --- |
| F1-path-label-too-sparse-even-after-P1 | P1 FastSlow supervision below discovery gate | do not tune label thresholds; use active LFRO perturbations or broaden real state source |
| F3-basis-has-value-but-risk-kills-it | P2 active perturbation did not find risk-safe positive V20 state | add function-space/value basis or redesign representation before scaling generated actions |
| F4-LFRO-cannot-predict-heldout-perturbations | P3 holdout R2/value gate failed | increase orthogonal samples only if P2 has value signal; otherwise change basis |
| F8-SlowBurn-exists-but-not-predictable-at-commit-time | SlowBurn rows exist but LFRO holdout did not predict usable alpha | use trajectory/function-space response rather than commit-time scalar score |

## 8. No-Fake / Contract Audit

```text
rows_checked = 22248
fake/proxy/cpu = 0 / 0 / 0
```

## 9. Artifact Inventory

| artifact group | files |
| --- | --- |
| route / manifest / audit | `route_decision_v1060.json`, `run_manifest_v1060.json`, `no_fake_audit_v1060.csv`, `contract_audit_v1060.csv` |
| P0-P7 summaries | `p0_v1050_boundary_lock_v1060.csv`, `p1_path_type_redefinition_audit_v1060.csv`, `p2_lfro_materializer_v1060.csv`, `p3_lfro_fit_controllability_v1060.csv`, `p4_lfro_constrained_update_solve_v1060.csv`, `p5_lfro_solved_update_discovery_staircase_v1060.csv`, `p6_failure_attribution_v1060.csv`, `p7_controller_boundary_v1060.csv` |
| P2 LFRO real replay | `p2_lfro_perturbation_actions_v1060.csv`, `p2_lfro_basis_v1060.csv`, `p2_lfro_action_apply_replay_v1060.csv`, `p2_lfro_branch_horizon_v1060.csv` |
| P4 solved update real replay | `p4_lfro_solved_actions_v1060.csv`, `p4_lfro_solved_action_apply_replay_v1060.csv`, `p4_lfro_solved_branch_horizon_v1060.csv`, `p4_lfro_solved_labels_v1060.csv` |
| figures | `fig_p1_path_type_counts_v1060.svg`, `fig_p2_lfro_response_v1060.svg`, `fig_p3_lfro_fit_v1060.svg`, `fig_p4_solver_v1060.svg`, `fig_p5_staircase_v1060.svg`, `fig_p6_failure_v1060.svg`, `fig_v1060_route_matrix.svg` |

## 10. SHA256

| artifact | SHA256 |
| --- | --- |
| `contract_audit_v1060.csv` | `7bd817f74efb1f1a4f45668b57aaa2911c388f2ec46519a8bb047ee399ebad0b` |
| `fig_p1_path_type_counts_v1060.svg` | `03ce9cdf4d5d8203796a3cb2037ae742051e3ad311c5cc402cc546083c8b68da` |
| `fig_p2_lfro_response_v1060.svg` | `9a0b54feedbd2b2b2bc20886d1cfb18012d55f5c7a5de0e7846462d80e37bd3a` |
| `fig_p3_lfro_fit_v1060.svg` | `5f6c71c9cacd3c4a106fe06232d97c8fa7eb23c58c13c192628bd830a4e67e16` |
| `fig_p4_solver_v1060.svg` | `c4bdad67dfeeb37c7769aebbbb1dc24c8c79fd417069b83a045f3bda3fd20302` |
| `fig_p5_staircase_v1060.svg` | `a5ed42fadd2e206618479b19f5abb8d97c7e86051a3e0d1f0c9242b6c4e4b914` |
| `fig_p6_failure_v1060.svg` | `d1f2751a01e391535f43faa31b033904977e276063ef2b1803684028e8f367a8` |
| `fig_v1060_route_matrix.svg` | `d4862399ea47b3590f38a2d4e55801ba1995bdd5b3973ddb74fea3b5bd6ae4a6` |
| `materializer` | `6c7508011ba2fed26f66f232ec4adf9257c23f6968ad72b0864d29dda5d174c2` |
| `no_fake_audit_v1060.csv` | `8fc2c839c3783a181a496f913249ad9a7f8047d076c9ad1fa9ee2fd221efddc6` |
| `p0_v1050_boundary_lock_v1060.csv` | `7ad35856faf190dd2e7fad02e4759208a77e6dbd866218464fd282b9717dd340` |
| `p1_path_type_redefinition_audit_v1060.csv` | `2b7e17c4d07b599f47a53133e5292fba43819e76dc71d31c5468c02a8c3e98ba` |
| `p2_lfro_action_apply_replay_v1060.csv` | `666a32fa25c43a8283ade0755d1acfb3ac0c84dbf7fe9e0e6c5a1acc885a8875` |
| `p2_lfro_basis_v1060.csv` | `bbdd139aa9c3e2a828d78f868ce7708ca9709b60a7c0a0ca0f3e471fdd49bacf` |
| `p2_lfro_branch_horizon_v1060.csv` | `9bc665b8a3674fa8e7babcffd32d873390063c27438451d07469d36affe67cd1` |
| `p2_lfro_materializer_v1060.csv` | `c24011fcd7c8245e348d2773de4576bb87500e9420d48c9e983dab7e8874cf73` |
| `p2_lfro_perturbation_actions_v1060.csv` | `41fbe7f717415e6b2d802752faa4aadc1d8cb9e2f149ddc6fc82a685b40d3d9a` |
| `p3_lfro_fit_controllability_v1060.csv` | `9fc224c3d8e04bc9e6f1062374900b0ab5aaa4b224e811c7eb5ef2997cd2e0a8` |
| `p4_lfro_constrained_update_solve_v1060.csv` | `50605d843a961ed640b681a008531d2d3e2451d29b5e78a4276b2c27fb141a36` |
| `p4_lfro_solved_action_apply_replay_v1060.csv` | `796a726a169ec34735c026cfbd9469e04f620552ec89b672d01ad94091dd82b1` |
| `p4_lfro_solved_actions_v1060.csv` | `09c9638ca37e7664e56a2370a55e80b8b1e6b9ed9b2dfec154ba6171c1ab38db` |
| `p4_lfro_solved_branch_horizon_v1060.csv` | `9496a9c115d0970929b36bf59106aaf7f692f640d0ad89121cb30094ba19ef40` |
| `p4_lfro_solved_labels_v1060.csv` | `b49ed36de8467853421c0e89a6d428e82ed24e6329f8cbbd1999cac9cf367ab0` |
| `p5_lfro_solved_update_discovery_staircase_v1060.csv` | `1e14b7e97f779d0a1607ab50305b3d931af328669d312071f3ab8c1224f9e475` |
| `p6_failure_attribution_v1060.csv` | `622394472744f89c10a688b564fa46b07356fa56e8f471085766687c7f1687c8` |
| `p7_controller_boundary_v1060.csv` | `5b6f8307fd16735425b2511c6b92e68ece2025d8f176970758dab3ce42130fe8` |
| `p8_runtime_boundary_v1060.csv` | `d230ef1e57ecb04d5879a59a964d355d2583b9095c67faac766ce859c61bb511` |
| `p9_paired_replay_boundary_v1060.csv` | `6cb7c8759c6d8efe26baa78219ed5f7d27ae864ce22562296c28da8ebc70e395` |
| `plan` | `68f3b30eb76d82a89e9f79e54bc3607a6c21925baede77f859854ac73d3443a5` |
| `route_decision_v1060.json` | `31e33e27eed5481273933415882fd7e332f2d6fa6ecc46032fae56d972693c93` |
| `run_manifest_v1060.json` | `fd81c2a8f99845407d2998e7ca2b81e8e90e5564e6eddeeaa46e25a49a6bd972` |
| `runner` | `2a7f17efa1edd1347e09545171b80fc990257940b14e747953695fd688fad7a2` |

## 11. 最终分析结论

```text
1. v10.6 按计划从被动 GoodCone/FPO 分类转为主动 LFRO 系统辨识；P2 真实生成并回放小扰动，不使用 fake/proxy/placeholder。
2. P1 仍显示真实 Fast/Slow 标签稀疏；该结果只作为 LFRO 监督目标和失败归因，不被写成 controller。
3. P2 的 materializer gate 独立记录 payload reload、apply、branch-horizon、NaN/Inf 和成本；P2 weak/strong 只看真实 V20 与风险。
4. P3/P4 把拟合与求解分开：即使 LFRO 模型有预测，也必须由 P4 solved update 的真实 h240 branch-horizon 验证。
5. P7/P8/P9 仍受硬 gate 控制；没有 Stage64 strong 或 Stage32+P3/P4 strong 时保持 not_run。
```

最终一句话：v10.6 真实执行后停在 `CaseD-NoRiskSafeLocalFutureResponse`：True local perturbations did not expose a risk-safe positive V20 response state.
