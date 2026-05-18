# DG-KAN v9.8.9 Natural Stream / FuturePathOperator / Four-Line 实验复盘

> 本复盘记录 `DG-KAN_v9.8.9_自然扩流_FuturePathOperator_四线并行_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.8 reference artifacts；P1a 找不到真实 natural AP0 extension generator 时，后续 science stages 均显式 `not_run`，没有 fake data、proxy rows 或 CPU offload。

## 0. 最新结论

```text
route = R-C1-NaturalGeneratorEntryMissing
primary_blocker = natural_extension_generator_missing
secondary_blocker = science_full_run_stopped_by_fail_fast
route_recommendation = engineering_only_materializer_task
system_legal_controller_pass = 0
generated_route_status = stopped_P1a_generator_missing
```

最终 artifact：`results/real_rerun_20260506/v9890_natural_stream_futurepathoperator_four_line_full_20260517T000000Z`

核心结论：

1. P0 复现 v9.8.8 boundary：source route = `R7-EngineeringBlocked`，P4 weak = `1`，P5 weak = `0`，system = `0`。
2. P1a natural extension generator entrypoint 仍未落地：entrypoint_count = `0`，found = `0`。
3. P1b/P1c/P1d single/16/256 preflight 全部 `not_run`，原因 = `P1a_natural_extension_generator_missing_fail_fast`。
4. P2 只写 existing reference：PanelA CoreLike LCB/UCB = `0.021475240123524635` / `0.033333885489495195`；5000/10000/20000 natural panels `not_run`。
5. P3 future path decomposition = `not_run`；P4 FuturePathOperator Sketch = `not_run`，均因 P1a fail-fast 停止。
6. P5 controller = `not_run`；P6 generated sandbox allowed = `0`。
7. Engineering materializer task rows = `7`。
8. No-fake audit：rows checked = `254`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9890_natural_stream_futurepathoperator_four_line.py` | v9.8.9 runner；复现 v9.8.8 boundary，执行 P1a generator audit；若缺 generator，按 fail-fast 写 engineering-only blocker artifacts。 |

```text
python -m py_compile experiments/run_v9890_natural_stream_futurepathoperator_four_line.py
```

```bash
python experiments/run_v9890_natural_stream_futurepathoperator_four_line.py --out-dir results/real_rerun_20260506/v9890_natural_stream_futurepathoperator_four_line_full_20260517T000000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9890",
  "status": "summary",
  "route": "R-C1-NaturalGeneratorEntryMissing",
  "primary_blocker": "natural_extension_generator_missing",
  "secondary_blocker": "science_full_run_stopped_by_fail_fast",
  "route_recommendation": "engineering_only_materializer_task",
  "source_route_v9880": "R7-EngineeringBlocked",
  "P0_boundary_pass": 1,
  "P1a_entrypoint_search_pass": 0,
  "P1_weak_pass": 0,
  "P1_strong_pass": 0,
  "P2_density_sufficient": 0,
  "P2_density_insufficient": 0,
  "P2_density_inconclusive": 1,
  "P3_future_path_weak_pass": 0,
  "P3_future_path_strong_pass": 0,
  "P4_future_operator_sketch_weak_pass": 0,
  "P4_future_operator_sketch_strong_pass": 0,
  "P5_controller_pass": 0,
  "P6_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_P1a_generator_missing",
  "P7_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Engineering Contract

| task | contract |
|---|---|
| `land_generator_module` | Create experiments/natural_ap0_extension_materializer.py with generate_natural_ap0_extension_actions(seed,data_root,target_action_count,cursor). |
| `bind_action_lifecycle` | Use the existing AP0 action schema and write action_id/candidate_id/payload_hash/provenance without reusing old action rows. |
| `bind_action_apply` | Expose action apply replay hook and verify action_apply_linf_max <= 1e-7 on a single new action. |
| `bind_branch_horizon` | Materialize RealFunctional/AdamWParallel/bestLR/NoOp/Random/Shuffled branch-horizon rows for h1,h5,h20,h80,h240. |
| `single_action_smoke` | Run target_action_count=1 and verify unique action_id, payload_hash, no-transform sanity, no fake/proxy/cpu. |
| `sixteen_action_smoke` | Run target_action_count=16 and verify completion=1.0, duplicate counts=0, label exclusivity violations=0. |
| `two_fifty_six_smoke` | Run target_action_count=256 and record rows/sec, wallclock, peak GPU memory, NaN/Inf, exception count. |

判断：v9.8.9 没有继续跑完整 science runner。P1a 缺 generator 后，下一步必须是工程化 natural AP0 extension materializer，而不是继续复用旧 AP0 replay 做 density/controller claim。

## 4. P2 Density Boundary

```text
existing PanelA CoreLike LCB/UCB = 0.021475240123524635 / 0.033333885489495195
diagnostic PathGood LCB/UCB = 0.015918067732609058 / 0.06044229950650894
PanelB/C/D natural extension = not_run
density sufficient/insufficient/inconclusive = 0/0/1
```

判断：这些是 reference rows，不是 v9.8.9 新自然扩流结果；不能声称 density sufficient 或 insufficient。

## 5. Boundary

```text
P3 future path type decomposition = not_run
P4 FuturePathOperator Sketch v3 = not_run
P5 controller = not_run
P6 generated sandbox = not_run
P7 runtime = not_run
P8 paired replay = not_run
```

## 6. No-Fake / Contract / Failure

```text
rows_checked = 254
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9890.csv` | `6fabffc8f29c0a8ccda3c786beacb7f433b16fde97979caab1e17f8fadea39b4` |
| `engineering_materializer_task_list_v9890.csv` | `1a4a06677355a9657716e6c7f130190f208a8a0c99b962a7da2b1bbf4018bcdb` |
| `failure_taxonomy_v9890.csv` | `9f12afaf8b6c66f7dc1437efe50b09f277c9477d7004aa2fffa640be027a5105` |
| `field_legality_ledger_v9890.csv` | `6f3b15d688e012d43ccc664aadec50932874c02305f26ea9b7de70bc03e46a92` |
| `fig_p1_branch_horizon_completion.svg` | `35444d23b0cd43fc5566d17f4e1cbde20093b794c12528e96e075bf81994aa8e` |
| `fig_p1_generator_lifecycle_flow.svg` | `529554ad0b30933db435e212ad32dc8811723e512551ec2e7bd88b0e6a1db923` |
| `fig_p2_corelike_density_curve.svg` | `99d3e2dadb20ca4aa549a3b84d4cd864abf2046a1f9b3be75d6a577b511c3236` |
| `fig_p2_dataset_family_density_heatmap.svg` | `e83fbb801e734ed813fbfd525233eb6bc8cc5f41c7add39d820ac726b8255e55` |
| `fig_p2_density_CI_vs_panel_size.svg` | `6a25a40cdae8ddfaca0ced117f718dec1ea8062547bf6a545781bf16f6e8910e` |
| `fig_p2_pathgood_density_curve.svg` | `8b55c473d898b572cda857ee5c758f2d01ae42898dc56f4e47af64a93d0e3542` |
| `fig_p3_AUV_vs_RiskAdjustedAUV.svg` | `b0ef93cac3b39d58a2f98eb09ce72033e2f5d9b4c1199564be3b8ee201a5d9b3` |
| `fig_p3_delayed_gain_vs_longrisk.svg` | `a18f7ddc1074bc8a7eb1659ac4ca5464ef189856a8862e31831120300aa23c2a` |
| `fig_p3_future_path_V_curves.svg` | `b7d1203a056ed478ea4cd51beae4359fe1c97ea5885ece10002242da5feceb6d` |
| `fig_p3_future_path_risk_curves.svg` | `aa0867c50859f859b82d1ea0c550cb8d56f9591747337d5fad63f58359fd0668` |
| `fig_p4_proxy_failure_cases.svg` | `3926764267ab99de54978d8dfe90abba893b7189854f74cc4a9a638f25dca192` |
| `fig_p4_proxy_precision_cost_pareto.svg` | `c1ef20bd0ed7b33aa654b9eeca415e8280ec0287e770832a030ab1ccdab40440` |
| `fig_p4_proxy_score_vs_RAUV.svg` | `22445f58e8bd4b610ab599af1b0f09ce2de383f1bd96100742121be79c8a4327` |
| `fig_p5_controller_accepted_region_pareto.svg` | `a9013f745e2cc1cbd76f87cf2039bbc0f29c1ee76c18c88af08204534ebcc5c4` |
| `fig_p6_generated_sandbox_gate.svg` | `26d1f666cda6fe15ca419596bb42c9d58313d8653a99163b6c0dbae12b9cf2cc` |
| `fig_p7_runtime_component_stack.svg` | `c0bd68dc6cec1a9b86c0dc14a505f6db0124536282444fffa4c5f5acf5c94fe9` |
| `fig_stop_pivot_matrix.svg` | `d62400fb83b33f390387b50bef2829cf47b640d6d98e00760aa929cffabbaab3` |
| `fig_v9890_four_line_gate_matrix.svg` | `4f1825bd26082a7b29c417a95a39f0576381aac1270b1c83dab0a473ff53fc3e` |
| `no_fake_audit_v9890.csv` | `65f8d65e8cd2c0febb6fcf8e4b6497e22a6bb846f91932f2c574e0d5f03e3b68` |
| `p0_boundary_reproduction_v9890.csv` | `d7e0120bdb63ce96cee45fcbef4eddfdb3658bde677ad16ca1743e2a11f37bcb` |
| `p1_16_action_preflight_v9890.csv` | `5e86f7440d41188a6116f6326c0b22d59d4de7bea2920775e250672918067397` |
| `p1_256_action_smoke_v9890.csv` | `f64904db69d8f464d29b638d9e52e5a4d6c64214490dfd66e1741e5d09404ac8` |
| `p1_natural_generator_entrypoint_audit_v9890.csv` | `a0258cc7b23aefd21a593fbc33f011ecaddba8a12364ea9ed5679a23d11ab808` |
| `p1_single_action_preflight_v9890.csv` | `37d4ee8c22bb5d6ed08f87d33cbc0d92fa8becc12daa08060b03b7343a04bc13` |
| `p2_density_by_dataset_family_template_v9890.csv` | `cdb1be33e97f9612396247f00c5ea06f3270b1ae0ccbc6672945a3e57bfdc9bd` |
| `p2_natural_density_panel_v9890.csv` | `73dd92f473077ad71153026abbf2798fdcbaba4b508d433289063576e25be250` |
| `p3_future_path_type_decomposition_v9890.csv` | `6d6204df6a1d54d32b1e055cf194f61e6006a836627c4cd3a260452bc696f1c1` |
| `p4_future_operator_sketch_v9890.csv` | `b49dd9011737f1f7c6ba826a8379eda5eef344e62e3e154e389aa73008508129` |
| `p5_controller_gate_v9890.csv` | `14832815b4a2e69d7bc55308e64a8cffa148ff52b2c28a1efcf00c745472132e` |
| `p6_generated_sandbox_gate_v9890.csv` | `445eb1e51a7821e4de3239e4a47e5ee8d6aee4f25b7dbb63cf44e19b8146fdbb` |
| `p7_runtime_boundary_v9890.csv` | `46c98ebbb860a75510470932f9483a97c0705e893b6db3a38140aa3a8d1910ae` |
| `p8_paired_replay_boundary_v9890.csv` | `72b8bdceedca0b10229283be140203374467ea9c204511631f32aeccbf543c8a` |
| `plan` | `0586b5f3c0c522fc78c34e33c20a23a13dda2de049340861d06181dc02b6f7b0` |
| `route_decision_v9890.json` | `c02c5efb47e8b9ef8494b8fb6b5831c8ec5080ab6a24da2ff5116bc0becb4b30` |
| `run_manifest_v9890.json` | `0d1afea8dbbe11a9523d888153ff2e6c351f75176004c80fd6b08ef2584bf269` |
| `runner` | `37f785692cafab166ce07a465bcc61a823e682ae61c747a5c9014dfcf692379b` |

## 8. 最终分析结论

```text
1. v9.8.9 按计划执行了 P0/P1a，并确认 v9.8.8 的工程 blocker 仍存在。
2. P1a 仍找不到真实 natural AP0 extension action generator，因此按 fail-fast 停止 science full run。
3. P1b/P1c/P1d/P2B-D/P3/P4/P5/P6/P7/P8 全部显式 not_run，没有 fake/proxy rows。
4. 本轮有效推进是把下一步收敛为 engineering-only materializer task list 和 single-action smoke spec。
5. strict PureKAN functional 仍未成功，existing-action route 不能继续声称 density 或 controller closure。
```

最终一句话：

> v9.8.9 真实执行后停在 `R-C1-NaturalGeneratorEntryMissing`：自然 AP0 extension generator 仍未落地，因此本轮按计划停止完整 science runner，只留下 materializer 工程合同和 smoke 测试规格；没有进入 controller、generated sandbox、runtime 或 paired replay。
