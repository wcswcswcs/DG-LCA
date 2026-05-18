# DG-KAN v9.8.3 Four-Line Future Mechanism / Natural Stream / Geometry Adaptive 实验复盘

> 本复盘记录 `DG-KAN_v9.8.3_四线并行_未来轨迹机制_自然动作扩流_几何自适应更新_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 future-path diagnostic、illegal mechanism、natural stream missing、gate semantics diagnostic 或 not_run controller/runtime 写成 official system pass。

## 0. 最新结论

```text
route = RouteB-FuturePathExistsMechanismAbsent
primary_blocker = no_legal_training_time_mechanism
secondary_blocker = natural_stream_materializer_missing
route_secondary = RouteD-NaturalStreamMaterializerMissing
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_natural_density_evidence
```

最终 artifact：`results/real_rerun_20260506/v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z`

核心结论：

1. P0 复现 v9.8.2 boundary：source route = `R-P2-NoMechanismExplainsGoodActions`，P1 rows expected/actual = `7260` / `7260`，fake/proxy/cpu = `0` / `0` / `0`。
2. P1 future path v2 weak/strong = `1` / `1`；Core77 AUV LCB = `2.8296932871691163`，RandomMatched AUV LCB = `1.8284728866377598`。
3. P1 immediate-not-dominant evidence = `1`；OldOnly V1 LCB = `-0.174770564086397`，OldOnly AUV LCB = `1.0287178879851673`。
4. P2 legal mechanism weak/strong pass count = `0` / `0`；best = `M5_oldrank_signal_diagnostic`，legality = `red:old_table_diagnostic`。
5. P3 natural AP0 extension still blocked：preflight pass/not_run = `0` / `3`，completed panel/not_run panel = `1` / `3`。
6. P4 gate modification allowed = `0`；backfill share = `0.7902762425139611`，density LCB = `0.021475240123524635`。
7. P5 discriminant matrix weak/strong/legal-strong = `1` / `0` / `0`；best feature = `OldRank_score`。
8. No-fake audit：rows checked = `1495`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9830_four_line_future_mechanism_natural_geometry.py` | v9.8.3 runner；读取 v9.8.2 full future path rows，执行 P1 mechanism decomposition v2、P2 legal mechanism discovery v2、P3 natural stream materializer audit、P4 gate semantics、P5 discriminant matrix，并按 gate 写 P6-P10。 |

代码检查：

```text
python -m py_compile experiments/run_v9830_four_line_future_mechanism_natural_geometry.py
```

正式运行：

```bash
python experiments/run_v9830_four_line_future_mechanism_natural_geometry.py --out-dir results/real_rerun_20260506/v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

说明：本轮 full-gated 是基于 v9.8.2 已真实落盘的 7260 条 future-path rows 做机制/扩流/gate 分析；P3 没有 landed natural AP0 extension materializer，因此 preflight 和 5000/10000/20000 panel 明确 `not_run`，没有生成假扩展动作。

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9830",
  "status": "summary",
  "route": "RouteB-FuturePathExistsMechanismAbsent",
  "route_secondary": "RouteD-NaturalStreamMaterializerMissing",
  "route_family": "mechanism_and_materializer",
  "primary_blocker": "no_legal_training_time_mechanism",
  "secondary_blocker": "natural_stream_materializer_missing",
  "source_route_v9820": "R-P2-NoMechanismExplainsGoodActions",
  "P0_boundary_pass": 1,
  "P1_weak_pass": 1,
  "P1_strong_pass": 1,
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P3_weak_pass": 0,
  "P3_strong_pass": 0,
  "P3_materializer_missing": 1,
  "P4_gate_modification_allowed": 0,
  "P5_weak_pass": 1,
  "P5_strong_pass": 0,
  "P5_legal_strong_pass": 0,
  "P6_controller_pass": 0,
  "P7_generated_reopen_allowed": 0,
  "P7_generated_pass": 0,
  "P8_runtime_pass": 0,
  "P9_paired_replay_pass": 0,
  "P10_short_full_pass": 0,
  "generated_route_status": "stopped_no_legal_mechanism_or_natural_density_evidence",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Future Path Mechanism v2

| group | actions | AUV LCB | V1 LCB | V20 LCB | V80 LCB | V240 LCB | mismatch | longrisk UCB | memory UCB | offdiag UCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Core77` | `77` | `2.8296932871691163` | `0.08815910625793319` | `1.573549156031694` | `3.1200217020467886` | `4.750885696264312` | `0.4155844155844156` | `0.03827572901862327` | `0.0` | `0.0` |
| `ExactOnly` | `68` | `1.5250480927868013` | `0.13699866747303657` | `0.4931219640179562` | `1.582392387764584` | `2.922090918869295` | `0.35294117647058826` | `0.0` | `0.0` | `0.0` |
| `OldOnly` | `10` | `1.0287178879851673` | `-0.174770564086397` | `0.30014089601530236` | `1.1093912874075194` | `2.0678046281088522` | `0.5` | `0.0` | `0.0` | `0.0` |
| `RandomMatched` | `87` | `1.8284728866377598` | `0.09405836253137967` | `0.5411237445554424` | `1.8306221718364828` | `3.5922551893613908` | `0.367816091954023` | `0.8485412092119363` | `0.9433993751492317` | `0.9605466806820027` |

判断：P1 strong pass = `1`。Core77 的完整 future path 仍强且风险低；OldOnly 出现 `V1_LCB <= 0` 但 `AUV_LCB > 0` 的 immediate-not-dominant 形态，支持“不是一步 loss descent，而是 future trajectory steering”的解释。

## 4. P2 Legal Mechanism Discovery v2

| mechanism | legality | arch specific | precision | V LCB | AUV LCB | longrisk UCB | memory UCB | offdiag UCB | LDO | weak | strong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_future_adamw_path_diagnostic` | `red:future_outcome_diagnostic` | `0` | `0.39080459770114945` | `-0.486183631655888` | `3.642533969726318` | `0.4075601452666917` | `0.4447068304756729` | `0.4569575346890791` | `0.04597701149425287` | `0` | `0` |
| `M2_hard_tail_relief_diagnostic` | `red:future_outcome_diagnostic` | `0` | `0.45977011494252873` | `-0.3758039097392462` | `4.187244879395712` | `0.3570360950674304` | `0.3824481159806525` | `0.3950403220387326` | `0.09195402298850575` | `0` | `0` |
| `M3_old_knowledge_memory_safety` | `green` | `0` | `0.41379310344827586` | `-0.11427822648412846` | `2.716555586178205` | `0.0` | `0.0` | `0.0` | `0.09195402298850575` | `0` | `0` |
| `M4_local_function_stability_diagnostic` | `red:future_outcome_diagnostic;yellow:kan_basis_specific` | `1` | `0.21839080459770116` | `-0.2234966269284317` | `1.0224280574295677` | `0.2517897676033668` | `0.2787225697162283` | `0.2920178012860627` | `0.011494252873563232` | `0` | `0` |
| `M5_oldrank_signal_diagnostic` | `red:old_table_diagnostic` | `0` | `0.8735632183908046` | `0.14061570456586386` | `2.7227479106902694` | `0.0` | `0.0` | `0.0` | `0.37931034482758624` | `0` | `0` |
| `M6_payload_norm_cost` | `green` | `0` | `0.22988505747126436` | `-0.21981076008236478` | `1.0988475479070037` | `0.21039105407869235` | `0.23813471172069484` | `0.2517897676033668` | `0.011494252873563204` | `0` | `0` |
| `M7_exacttransfer_signal_diagnostic` | `red:outcome_derived_transfer_diagnostic` | `0` | `0.20689655172413793` | `-0.2221212871334743` | `2.00055313882049` | `0.0` | `0.0` | `0.0` | `0.0` | `0` | `0` |

判断：P2 没有 legal-at-commit 机制通过。诊断类信号仍能贴近好区域，但带 red legality；green 的 memory/cost 类仍无法同时给出 precision/value。

## 5. P3 Natural AP0 Stream Extension Materializer

| row | status | target | labeled | expected rows | actual rows | missing labels | reason |
|---|---|---:|---:|---:|---:|---:|---|
| `Panel-target-2876` | `panel_row` | `2876` | `2876` | `` | `` | `0` | `` |
| `Panel-target-5000` | `not_run` | `5000` | `2876` | `150000` | `0` | `2124` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9830` |
| `Panel-target-10000` | `not_run` | `10000` | `2876` | `300000` | `0` | `7124` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9830` |
| `Panel-target-20000` | `not_run` | `20000` | `2876` | `600000` | `0` | `17124` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9830` |

判断：P3 strong pass = `0`。自然扩流 materializer 仍未落地，因此不能用 2876 panel 继续推断 5000/10000/20000 density。

## 6. P4 Gate Semantics

```text
raw_LDO = 0.4415584415584416
quality_LDO = 0.09260529551331587
support_LDO = 0.0
backfill_LDO = 0.34895314604512573
backfill_share = 0.7902762425139611
official_gate_modification_allowed = 0
```

判断：backfill 仍能解释 raw LDO 的主要部分，但 P3 strong pass 缺失，所以 official raw gate 不允许修改。

## 7. P5 Geometry Discriminant Matrix

| feature | legality | Core vs Random | Old vs Exact | precision | V LCB | AUV LCB | longrisk UCB | weak | strong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `action_norm` | `green` | `-0.08580526798040675` | `0.3323432686084884` | `0.22988505747126436` | `-0.21981076008236478` | `1.0988475479070037` | `0.21039105407869235` | `0` | `0` |
| `payload_norm` | `green` | `-0.08580526798040675` | `0.3323432686084884` | `0.22988505747126436` | `-0.21981076008236478` | `1.0988475479070037` | `0.21039105407869235` | `0` | `0` |
| `memory_offdiag_safe` | `green` | `1.7843693438168478` | `0.0` | `0.41379310344827586` | `-0.11427822648412846` | `2.716555586178205` | `0.0` | `0` | `0` |
| `OldRank_score` | `red:old_table_diagnostic` | `1.6911883002965806` | `1.3383587977893694` | `0.8735632183908046` | `0.14061570456586386` | `2.7227479106902694` | `0.0` | `1` | `0` |
| `ExactTransfer_score` | `red:outcome_derived_transfer_diagnostic` | `1.795240640272017` | `-0.3623455800872008` | `0.20689655172413793` | `-0.2221212871334743` | `2.00055313882049` | `0.0` | `0` | `0` |
| `WT80_score` | `red:windowed_future_diagnostic` | `0.41736285152280284` | `-0.3137943207152283` | `0.5057471264367817` | `-0.09437277710397` | `2.833760558362675` | `0.10637805008576995` | `0` | `0` |
| `AUV_future_label` | `red:future_outcome` | `0.5191551103657229` | `-0.01161356152511517` | `0.4482758620689655` | `-0.4357864398037876` | `4.381331480649709` | `0.369780988566077` | `0` | `0` |
| `hard_tail_h80` | `red:future_outcome` | `0.6030041262107941` | `0.24458550673638754` | `0.45977011494252873` | `-0.3758039097392462` | `4.187244879395712` | `0.3570360950674304` | `0` | `0` |
| `local_geometry_h80` | `red:future_outcome;yellow:kan_basis_specific` | `-0.044342137524046935` | `-0.9788509056568591` | `0.3333333333333333` | `-0.5676380959572139` | `2.4280161790877273` | `0.43239147769431` | `0` | `0` |
| `function_displacement_norm` | `red:future_outcome` | `-0.2952645512575107` | `0.17289185836557483` | `0.1724137931034483` | `-0.42467601119801085` | `1.0627715434283616` | `0.369780988566077` | `0` | `0` |

判断：P5 有若干 diagnostic discriminator，但没有 legal strong feature。继续说明 memory/offdiag 是安全门，不是收益源。

## 8. P6-P10 Boundary

```text
P6 controller = not_run
P7 generated = stopped
P8 runtime = not_run
P9 paired replay = not_run
P10 short/full = not_run
```

判断：P2/P5 没有 legal strong mechanism，P3 materializer missing，因此不允许 controller、generated、runtime、paired replay 或 short/full。

## 9. No-Fake / Contract / Failure

No-fake audit：

```text
rows_checked = 1495
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

Failure taxonomy：

```text
route = RouteB-FuturePathExistsMechanismAbsent
F1_future_path_not_strong = 0
F2_no_legal_mechanism = 1
F3_natural_stream_materializer_missing = 1
F4_gate_not_modifiable = 1
F5_generated_route_stopped = 1
primary_blocker = no_legal_training_time_mechanism
secondary_blocker = natural_stream_materializer_missing
```

## 10. Figures

```text
fig_p1_v9830_group_auv_lcb.svg
fig_p1_v9830_group_mismatch.svg
fig_p2_v9830_precision_by_mechanism.svg
fig_p3_v9830_density_curve.svg
fig_p4_v9830_ldo_decomposition.svg
fig_p5_v9830_effect_core_random.svg
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9830.csv` | `0b88d138c28aef79679265ecd0ae24b92e717686e789915bf165868fc276c081` |
| `failure_taxonomy_v9830.csv` | `713ae6d1041dd45498664bb7216cd4847bb86258d5187806dff4e0504f348700` |
| `no_fake_audit_v9830.csv` | `2396455716f777239ff2fe361e84ce17a4fdb97763daa8d3734710ab5e1ec644` |
| `p0_boundary_reproduction_v9830.csv` | `7f6b3c8465d3139bb9f629b8e318b96897e3c2a092b3e508460a0ccee4d0c066` |
| `p10_short_full_training_boundary_v9830.csv` | `87112248ef3241c20b7a4b6da10b0c80c7ffd1baf35fec204e4aa8821a573f4d` |
| `p1_future_path_mechanism_decomposition_v2_v9830.csv` | `d5e70a0e076c2cabc45eba28154d6ca32ec22af601cef61e67dfc01a47b878f8` |
| `p2_legal_mechanism_discovery_v2_v9830.csv` | `38005adea8d35b5a90e03ce587dbcc100f92a32adade599e022e3e274c166d86` |
| `p3_natural_ap0_stream_extension_materializer_v2_v9830.csv` | `a9cc1d0bc8b0d2dbdc49936f2649ad4d866256124e0276b4d891afcda5468d05` |
| `p4_gate_semantics_official_scope_v9830.csv` | `82353cc9e368ef344f20dee9de8708cc795c09943df5e4c4ebd399d7fe3041a2` |
| `p5_geometry_discriminant_matrix_v9830.csv` | `66e08d7f88189c90ed3887c84f34a14862e24bd007ba6f2cd78457663c57d672` |
| `p6_existing_action_minimal_controller_boundary_v9830.csv` | `733801712e12d1a227350369a28a64dd4b62dc5e0f69829b362a378d5c3879c5` |
| `p7_generated_update_reopen_gate_v9830.csv` | `69da217f919c4f140538f3f23522d4ca0c6936bbfc562a6804cc70884ff967c9` |
| `p8_selected_runtime_boundary_v9830.csv` | `d42e4e41c50e1239e23c822d489d82b85eb1c5236f4593f49570ff533e10dafa` |
| `p9_official_paired_replay_boundary_v9830.csv` | `4f97e634abf3360536b6019b0443b41e2f01256c10c64214e99990c454ea1f7d` |
| `plan` | `fbf941660967d1c4d93a2979b7f3990da0bb297d67eb2b8b0ba7a93ac09be8f0` |
| `route_decision_v9830.json` | `06118fe9a3995d4ce056da6991c52bc458c2291c3c27fdbc8009904009232153` |
| `run_manifest_v9830.json` | `15eb66eb2d4cc43964ec404c108b43b3d13b59405fdc0690cb9b5045e27ea60b` |
| `runner` | `14fb28f37a394a75abe606f40d24686afd2a8de1527229c45d9d293775baa616` |

## 12. 最终分析结论

v9.8.3 的真实推进是：

```text
1. P1 不再只是确认 future path 存在，而是把 AUV、Stability、Mismatch 拆到 action-level；Core77 继续强，OldOnly 显示 immediate-not-dominant。
2. P2 仍没有找到训练当下合法机制；诊断信号有效但不合法，合法安全信号不够选出收益动作。
3. P3 的自然 AP0 扩流 materializer 仍缺失，preflight 和 5000/10000/20000 panel 都不能伪造。
4. P4 不允许修改 official raw gate。
5. P5 判别矩阵确认若干 red diagnostic discriminator，但没有 legal strong mechanism。
6. P6-P10 全部 gate-blocked。
```

最终一句话：

> v9.8.3 真实执行后停在 `RouteB-FuturePathExistsMechanismAbsent`，并同时记录 `RouteD-NaturalStreamMaterializerMissing`：future path 机制现象更清楚了，但训练当下合法机制仍未闭合，自然 AP0 扩流 materializer 也未落地，因此不能进入 official controller，strict PureKAN functional 仍未成功。
