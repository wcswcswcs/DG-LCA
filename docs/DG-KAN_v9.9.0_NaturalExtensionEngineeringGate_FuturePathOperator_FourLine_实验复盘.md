# DG-KAN v9.9.0 Natural Stream / FuturePathOperator / Four-Line 实验复盘

> 本复盘记录 `DG-KAN_v9.9.0_自然扩流工程门_FuturePathOperator_四线并行完整实验计划 (1).md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest、v9.8.9 reference artifacts 与本轮真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。P2 5000/10000/20000 panel 未运行时显式 `not_run`，没有把 256-action smoke 写成 density closure。

## 0. 最新结论

```text
route = R-C1B-NaturalGeneratorLandedSmokePassDensityPending
primary_blocker = natural_density_full_panel_pending
secondary_blocker = future_operator_sketch_not_run_after_engineering_smoke
route_recommendation = run_full_5000_10000_20000_natural_panels_or_extend_B_sketch
system_legal_controller_pass = 0
generated_route_status = stopped_density_full_panel_pending
```

最终 artifact：`results/real_rerun_20260506/v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z`

核心结论：

1. P0 复现 v9.8.9 boundary：source route = `R-C1-NaturalGeneratorEntryMissing`，P1a entrypoint = `0`，system = `0`，generated = `stopped_P1a_generator_missing`。
2. P1a natural extension generator entrypoint：entrypoint_count = `1`，found = `1`，generator module = `experiments/natural_ap0_extension_materializer.py`。
3. P1b single action smoke pass = `1`：branch-horizon rows = `30` / `30`，action_apply_linf_max = `0.0`，cpu_offload = `0`。
4. P1c 16-action smoke pass = `1`：branch-horizon rows = `480` / `480`，rows/sec = `23.66738807915214`，peak GPU MB = `122.0419921875`。
5. P1d 256-action smoke pass = `1`：branch-horizon rows = `7680` / `7680`，wallclock = `339.08833478810266` sec，peak GPU MB = `1445.3623046875`。
6. 本轮新增 natural actions = `273`，新增 branch-horizon rows = `8190`；old action/payload collision = `0` / `0`。
7. P2 natural smoke diagnostic：action count = `273`，CoreLike count/LCB/UCB = `2` / `0.0020113177709178186` / `0.0263139050279679`，PathGood count/LCB/UCB = `3` / `0.003744092637382641` / `0.031805477947384975`。
8. P2 full 5000/10000/20000 density panels 仍未运行，density sufficient/insufficient/inconclusive = `0` / `0` / `1`。
9. P3/P4/P5/P6 仍 gate-blocked：P3 = `not_run`，P4 = `not_run`，controller = `not_run`，generated sandbox allowed = `0`。
10. No-fake audit：rows checked = `9264`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/natural_ap0_extension_materializer.py` | 本轮新增 natural AP0 extension materializer；生成新 action rows、写 payload shard、执行 action apply 校验、materialize branch-horizon rows。 |
| `experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py` | v9.9.0 runner；复现 v9.8.9 boundary，执行 P1a/P1b/P1c/P1d engineering gate，写 density diagnostic、route、audits、manifest。 |

```text
python -m py_compile experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py
```

```bash
python experiments/run_v9900_natural_extension_engineering_gate_futurepathoperator.py --out-dir results/real_rerun_20260506/v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9900",
  "status": "summary",
  "route": "R-C1B-NaturalGeneratorLandedSmokePassDensityPending",
  "primary_blocker": "natural_density_full_panel_pending",
  "secondary_blocker": "future_operator_sketch_not_run_after_engineering_smoke",
  "route_recommendation": "run_full_5000_10000_20000_natural_panels_or_extend_B_sketch",
  "source_route_v9890": "R-C1-NaturalGeneratorEntryMissing",
  "P0_boundary_pass": 1,
  "P1a_entrypoint_search_pass": 1,
  "P1b_single_action_preflight_pass": 1,
  "P1c_16_action_preflight_pass": 1,
  "P1d_256_action_smoke_pass": 1,
  "P1_weak_pass": 1,
  "P1_strong_pass": 1,
  "new_natural_action_count": 273,
  "new_branch_horizon_rows": 8190,
  "natural_smoke_CoreLike_count": 2,
  "natural_smoke_CoreLike_LCB": 0.0020113177709178186,
  "natural_smoke_PathGood_count": 3,
  "natural_smoke_PathGood_LCB": 0.003744092637382641,
  "P2_density_sufficient": 0,
  "P2_density_insufficient": 0,
  "P2_density_inconclusive": 1,
  "P3_future_path_weak_pass": 0,
  "P3_future_path_strong_pass": 0,
  "P4_future_operator_sketch_weak_pass": 0,
  "P4_future_operator_sketch_strong_pass": 0,
  "P5_controller_pass": 0,
  "P6_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_density_full_panel_pending",
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

判断：v9.9.0 已经不再停在 P1a generator missing。P1b/P1c/P1d 都是真实新 action + branch-horizon replay smoke，并且全部通过；但这仍只是 engineering gate，不等价于 5000/10000/20000 density closure 或 controller pass。

## 4. P1 Smoke Results

| stage | actions | branch rows | completion | apply linf | rows/sec | peak GPU MB | NaN/Inf | cpu offload |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `P1b-single` | `1` | `30/30` | `1` | `0.0` | `11.924188489577846` | `33.83837890625` | `0/0` | `0` |
| `P1c-16` | `16` | `480/480` | `1` | `0.0` | `23.66738807915214` | `122.0419921875` | `0/0` | `0` |
| `P1d-256` | `256` | `7680/7680` | `1` | `0.0` | `22.648965511595247` | `1445.3623046875` | `0/0` | `0` |

## 5. P2 Density Boundary

```text
existing PanelA CoreLike LCB/UCB = 0.021475240123524635 / 0.033333885489495195
diagnostic PathGood LCB/UCB = 0.015918067732609058 / 0.06044229950650894
natural smoke actions = 273
natural smoke CoreLike count/LCB/UCB = 2 / 0.0020113177709178186 / 0.0263139050279679
natural smoke PathGood count/LCB/UCB = 3 / 0.003744092637382641 / 0.031805477947384975
PanelB/C/D 5000/10000/20000 natural extension = not_run
density sufficient/insufficient/inconclusive = 0/0/1
```

判断：本轮已有真实 new-natural-action smoke panel，但规模只有 273 action，不能替代 5000/10000/20000 full panel；因此 density 仍保持 inconclusive。

## 6. Boundary

```text
P3 future path type decomposition = not_run, reason = P2_full_density_not_adjudicated_after_real_materializer_smoke
P4 FuturePathOperator Sketch v3 = not_run, reason = P2_full_density_not_adjudicated_after_real_materializer_smoke
P5 controller = not_run
P6 generated sandbox = not_run
P7 runtime = not_run
P8 paired replay = not_run
```

## 7. No-Fake / Contract / Failure

```text
rows_checked = 9264
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
contract P1a/P1b/P1c/P1d = 1/1/1/1
failure route = R-C1B-NaturalGeneratorLandedSmokePassDensityPending
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9900.csv` | `9fe351fd99848ab6d6c33c5b460bd85ea412d2a14437795b64509ab1fbf020d9` |
| `engineering_materializer_task_list_v9900.csv` | `8031843b37282d0a7f34ce8f0dce92440fa9f6b3f22161a416298bc90d79a9a5` |
| `failure_taxonomy_v9900.csv` | `7bf2f5b2f01b54de3619d3ee065646412d13bcf8b2f864902cca7ec6c9fa79dc` |
| `field_legality_ledger_v9900.csv` | `422fc6b1aa1325d8a11b2f6b779bdcb30c5bf617559383303ad831d6fc5ed1fe` |
| `fig_p1_exception_type_bar.svg` | `c646c0967ea893fc127bad8b070e4eaaddf8aac65bd9210c4e96ad6dfc841c94` |
| `fig_p1_materializer_completion_by_stage.svg` | `e6f2e032540f852c23f75f9bfeab26a8dca6945bb87d492bffaf497ae105a9e9` |
| `fig_p1_rows_per_sec_by_panel.svg` | `ecd13016d57f921c074370738f5ea88cdc597c983a029e7fdebdd561edf6d9cb` |
| `fig_p2_dataset_family_density_heatmap.svg` | `67dfcd62410ce8aed95071f574531b98a58891c778b73b32e89beac0c48e7814` |
| `fig_p2_density_curve_corelike.svg` | `99d3e2dadb20ca4aa549a3b84d4cd864abf2046a1f9b3be75d6a577b511c3236` |
| `fig_p2_density_curve_pathgood.svg` | `8b55c473d898b572cda857ee5c758f2d01ae42898dc56f4e47af64a93d0e3542` |
| `fig_p2_template_support_histogram.svg` | `20350ed96607896f0d5aa17bfc1221f59dc8e3555839f92552cfb5b585cfe329` |
| `fig_p2_throughput_peak_memory.svg` | `e6a41de8933f4bcdf4fa736f9188da0ae497266f655579a5ef0f912709229f92` |
| `fig_p2_wilson_ci_by_panel.svg` | `6a25a40cdae8ddfaca0ced117f718dec1ea8062547bf6a545781bf16f6e8910e` |
| `fig_p3_delayed_gain_vs_risk_integral.svg` | `a18f7ddc1074bc8a7eb1659ac4ca5464ef189856a8862e31831120300aa23c2a` |
| `fig_p3_future_value_curve_by_group.svg` | `b7d1203a056ed478ea4cd51beae4359fe1c97ea5885ece10002242da5feceb6d` |
| `fig_p3_memory_offdiag_curve_by_group.svg` | `3cffc9e3abe2eb1ffc019308eea16de70e8ec193ac2519aff77cc310402498b5` |
| `fig_p3_rauv_vs_longrisk.svg` | `b0ef93cac3b39d58a2f98eb09ce72033e2f5d9b4c1199564be3b8ee201a5d9b3` |
| `fig_p3_risk_curve_by_group.svg` | `aa0867c50859f859b82d1ea0c550cb8d56f9591747337d5fad63f58359fd0668` |
| `fig_p4_proxy_V_vs_longrisk.svg` | `51bc29761391a96ca19a29e2c3953386d5c1751d1fb4f7fd224ba9f5be1cf232` |
| `fig_p4_proxy_cost_breakdown.svg` | `e701747f52e2facbfaaaa8378e177e87f471bfd05892b619067100d8a7935edd` |
| `fig_p4_proxy_leaveout_drop.svg` | `65a692828abf2b2b2c8a7f0a423a379f6473d4c5d3e6fb0702836cb57dc096b8` |
| `fig_p4_proxy_precision_vs_cost.svg` | `c1ef20bd0ed7b33aa654b9eeca415e8280ec0287e770832a030ab1ccdab40440` |
| `fig_p4_proxy_score_vs_AUV_scatter.svg` | `22445f58e8bd4b610ab599af1b0f09ce2de383f1bd96100742121be79c8a4327` |
| `fig_p5_controller_boundary.svg` | `a9013f745e2cc1cbd76f87cf2039bbc0f29c1ee76c18c88af08204534ebcc5c4` |
| `fig_p6_generated_gate_decision.svg` | `26d1f666cda6fe15ca419596bb42c9d58313d8653a99163b6c0dbae12b9cf2cc` |
| `fig_p7_runtime_component_stack.svg` | `c0bd68dc6cec1a9b86c0dc14a505f6db0124536282444fffa4c5f5acf5c94fe9` |
| `fig_stop_pivot_matrix.svg` | `26b30bcb9345c293772c430243c7bff5d8085b9c9172a9687ae72cda0787687f` |
| `fig_v9900_four_line_gate_matrix.svg` | `424c6ff4cad0093553a64e7359050f8eb21b3afe5384844ab30b38874d7a72a9` |
| `no_fake_audit_v9900.csv` | `1b26de14060970f5cc2c48daf712d9a0a3d823299f9f9cc3e8b5f765e29e0f62` |
| `p0_boundary_reproduction_v9900.csv` | `1f42e65fe771421c2e94c7c86c7c17fd5a6022653a14efc393b8b286d147779d` |
| `p1_16_action_preflight_v9900.csv` | `5efc61f74793329cb3ad6756ae62c91c1c4cb3ac131b45a8c96bd9b1328435d8` |
| `p1_256_action_smoke_v9900.csv` | `0b10dfd4a185d59da1755ea6952a0d7f8a5c11236267b2540429702147e057a9` |
| `p1_natural_extension_action_rows_v9900.csv` | `efe58dd6be362bb70ef21cedc355e87e48e6e1cedadddc293eafa74b61f688df` |
| `p1_natural_extension_apply_replay_v9900.csv` | `d18094f0e4f6086f36e62676b774af91fcc1a7346708ec3f8953fdea959fbec9` |
| `p1_natural_extension_branch_horizon_v9900.csv` | `791d6df2fdb1118d13ca7d2629e0fa3d688dc842287abc0d9cbd48cb523f83c9` |
| `p1_natural_generator_entrypoint_audit_v9900.csv` | `8b2a9d5040284ebcb5f654a6606038fc4265a5aacd68e257d0a786b1e840ff4e` |
| `p1_single_action_preflight_v9900.csv` | `3babde626a3e5bea44c4a87896fc66a2ea0d8da6371293830b5636b8327ddd60` |
| `p2_density_by_dataset_family_template_v9900.csv` | `bac6a6be87edc8e440a3bf0a7540fa2378f271393f38b78eab14bd52b04f7612` |
| `p2_natural_density_panel_v9900.csv` | `9844a801d4812a8d28e74b8e935996442344ad6cbe6b4866fe142e4e25783a0b` |
| `p2_natural_extension_label_diagnostic_v9900.csv` | `684cb0c90a3d42f081ba9b2a1f348a805fdc6bcdc89f6cd2654c2d826ce9e8c0` |
| `p3_future_path_type_decomposition_v9900.csv` | `59288225c2a1c59fc56f1e67db445b11e4e5efda4c7776909f99891dfb49846c` |
| `p4_future_operator_sketch_v9900.csv` | `c27cde731a124a0763f70975057224f512191104837ed7b924f3aad4f3f68f14` |
| `p5_controller_gate_v9900.csv` | `7fd37cabcd5ef9876ee7feeb65469d970c22804bd3094000fad255783bd39a54` |
| `p6_generated_sandbox_gate_v9900.csv` | `7c549c318e509190c9e389211dd12ac49887ad42c1252fb10cb64044c6950fe1` |
| `p7_runtime_boundary_v9900.csv` | `cf8438f6e7766fa111f2a6937f67467ffe2f6871b68ed2e8905b37119d98a445` |
| `p8_paired_replay_boundary_v9900.csv` | `b8dfc055c510ccffb3dbac1b5dfeddf39ac1b21611c37f880d120518b606b7ee` |
| `plan` | `90e31a87d7b1fde633611a24cb060bb8af28178c3a25c67f2a3376da979b5e08` |
| `route_decision_v9900.json` | `a4f33a3c4f2e21748566729556408abc967d22d36016fbf51cd359dd33615da5` |
| `run_manifest_v9900.json` | `f210ba309644639108875faa4d1d6b1cd336667f48e491eea0230ed54f51d09d` |
| `runner` | `60294a0a03b31563bb43e62f0871307fb12402ecb426979bcd251fec59907b42` |

## 9. 最终分析结论

```text
1. v9.9.0 真实落地了 natural AP0 extension materializer，而不是继续停在 generator missing。
2. P1b/P1c/P1d 都完成：single/16/256 action smoke 的 branch-horizon rows 全部 materialized，且没有旧 action/payload collision。
3. 256-action smoke 只是工程门和密度初诊断，不是 full natural stream panel；5000/10000/20000 仍未裁决。
4. P2 density 继续 inconclusive，P3/P4/P5/P6/P7/P8 不进入 official controller/runtime/generated。
5. strict PureKAN functional 仍未成功，但 primary blocker 已从 generator missing 推进为 full density panel pending。
```

最终一句话：

> v9.9.0 真实执行后停在 `R-C1B-NaturalGeneratorLandedSmokePassDensityPending`：本轮已经落地并验证 natural AP0 extension materializer，P1d 256-action smoke 真实通过；但 full 5000/10000/20000 density panel 尚未运行，FuturePathOperator sketch/controller/generated route 仍 gate-blocked，所以不能写成 official system pass。
