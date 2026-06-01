# DG-KAN v9.8.2 Future Path / Geometry Mechanism / Natural Stream 实验复盘

> 本复盘记录 `DG-KAN_v9.8.2_结果解读_未来轨迹几何机制_自然动作扩流_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；P1 h1/h5/h20/h80/h240 是真实 replay materialized rows；P3 5000/10000/20000 natural stream extension 没有 landed labeled materializer，因此显式 `not_run`，没有 fake data、proxy rows 或 CPU offload。

## 0. 最新结论

```text
route = R-P2-NoMechanismExplainsGoodActions
primary_blocker = no_legal_mechanism_explains_good_actions
secondary_blocker = natural_labeled_stream_extension_missing
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_generated_branch_horizon
```

最终 artifact：`results/real_rerun_20260506/v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z`

核心结论：

1. P0 复现 v9.8.1 boundary：source route = `R4-ArchitectureAgnosticGeometryFail`，system pass = `0`，field red count = `0`。
2. P1 真实补齐 full future path：selected actions = `242`，row count expected/actual = `7260` / `7260`，horizons = `1,5,20,80,240`。
3. P1 weak pass = `1`，missing horizon = `0`，NaN/Inf = `0` / `0`，no-transform sanity = `1`。
4. P1 observed AUV LCB：Core77 = `2.80616189380043`，OldOnly = `0.9993033070897774`，ExactOnly = `1.5125686938611076`，RandomMatched = `1.7813876735867753`。
5. P1 mechanism endpoint check = `0`；OldOnly h240 longrisk UCB = `0.0`。
6. P2 mechanism pass count = `0`；best mechanism = `M_signal`，TopK87 precision = `0.8735632183908046`，AUV LCB = `2.7227479106902694`。
7. P3 natural stream extension 仍未落地：completed panel = `1`，not-run panel = `3`，Panel A rate/LCB/UCB = `0.026773296244784424` / `0.021475240123524635` / `0.033333885489495195`。
8. P4 gate semantics pass = `1`；raw/support/density gate = `0` / `0` / `0`；official density proposal allowed = `0`。
9. P5 geometry weak/strong pass = `0` / `0`；best family = `G_stability`。
10. P6/P7/P8/P9/P10 均 gate-blocked：controller `not_run`，generated `not_run`，runtime `not_run`，paired replay `not_run`，short/full `not_run`。
11. No-fake audit：rows checked = `7555`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py` | v9.8.2 runner；真实 materialize selected groups 的 h1/h5/h20/h80/h240 branch-horizon rows，执行 mechanism contrast、natural stream extension audit、gate semantics、geometry family decomposition，并按 gate 写 P6-P10。 |

代码检查：

```text
python -m py_compile experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py
```

正式 full-gated 运行：

```bash
python experiments/run_v9820_future_path_geometry_mechanism_natural_stream.py --out-dir results/real_rerun_20260506/v9820_future_path_geometry_mechanism_natural_stream_full_20260516T120000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000 --generated-per-family 64
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9820",
  "status": "summary",
  "route": "R-P2-NoMechanismExplainsGoodActions",
  "route_family": "mechanism",
  "primary_blocker": "no_legal_mechanism_explains_good_actions",
  "secondary_blocker": "natural_labeled_stream_extension_missing",
  "source_route_v9810": "R4-ArchitectureAgnosticGeometryFail",
  "P0_boundary_pass": 1,
  "P1_weak_pass": 1,
  "P1_mechanism_pass": 0,
  "P2_mechanism_pass": 0,
  "P3_density_strong_pass": 0,
  "P3_density_weak_pass": 0,
  "P4_gate_semantics_pass": 1,
  "P5_weak_pass": 0,
  "P5_strong_pass": 0,
  "P6_controller_strong_pass": 0,
  "P7_generated_strong_pass": 0,
  "P8_runtime_pass": 0,
  "P9_paired_replay_pass": 0,
  "P10_short_full_pass": 0,
  "system_legal_controller_pass": 0,
  "generated_route_status": "stopped_no_legal_mechanism_or_generated_branch_horizon",
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Full Future Path Materializer

Artifacts：

```text
p1_full_future_path_materializer_v9820.csv
p1_future_path_completion_v9820.csv
p1_group_trajectory_summary_v9820.csv
```

Group/horizon summary：

| group | h | actions | V LCB | Vctrl LCB | CE mean | CEp99 mean | longrisk UCB | memory UCB | offdiag UCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Core77` | `1` | `77` | `0.08815910625793319` | `-0.38150928836076836` | `-0.07299900663315088` | `-0.048553257600053565` | `0.0` | `0.0` | `0.0` |
| `Core77` | `5` | `77` | `0.47094429986279623` | `-0.6437259799593326` | `-0.14389221720978038` | `-0.24370950257236307` | `0.0` | `0.0` | `0.0` |
| `Core77` | `20` | `77` | `1.573549156031694` | `-0.17648778612802843` | `-0.23736115767584218` | `-0.8544383628608344` | `0.0` | `0.0` | `0.0` |
| `Core77` | `80` | `77` | `3.1200217020467886` | `0.23509118073727725` | `-0.3359051822116236` | `-1.5418951683617257` | `0.0` | `0.0` | `0.0` |
| `Core77` | `240` | `77` | `4.750885696264312` | `0.2248144679958842` | `-0.4018768939014234` | `-2.0578310709059626` | `0.03827572901862327` | `0.0` | `0.0` |
| `ExactOnly` | `1` | `68` | `0.13699866747303657` | `-0.18949671283947203` | `-0.07137617701664567` | `-0.1207313452135114` | `0.0` | `0.0` | `0.0` |
| `ExactOnly` | `5` | `68` | `0.31093277492802823` | `-0.38529765836595015` | `-0.12314605293795466` | `-0.1420334471280084` | `0.0` | `0.0` | `0.0` |
| `ExactOnly` | `20` | `68` | `0.4931219640179562` | `-0.7328938433162533` | `-0.16705619184957707` | `-0.2437713340691784` | `0.0` | `0.0` | `0.0` |
| `ExactOnly` | `80` | `68` | `1.582392387764584` | `-0.35382639660922144` | `-0.22929339256028042` | `-0.7556909595561379` | `0.0` | `0.0` | `0.0` |
| `ExactOnly` | `240` | `68` | `2.922090918869295` | `0.015327829109752174` | `-0.260047403153936` | `-1.098049703218481` | `0.0` | `0.0` | `0.0` |
| `OldOnly` | `1` | `10` | `-0.174770564086397` | `-0.22933458962759967` | `-0.004013798572123051` | `-0.037525751441717145` | `0.0` | `0.0` | `0.0` |
| `OldOnly` | `5` | `10` | `-0.2015216336463978` | `-0.3768025026558646` | `-0.015950204990804195` | `-0.1904592476785183` | `0.0` | `0.0` | `0.0` |
| `OldOnly` | `20` | `10` | `0.30014089601530236` | `-0.2586713639752118` | `-0.042443423345685` | `-0.42846951335668565` | `0.0` | `0.0` | `0.0` |
| `OldOnly` | `80` | `10` | `1.1093912874075194` | `-0.042976464433907444` | `-0.08213981427252293` | `-1.0148171294480561` | `0.0` | `0.0` | `0.0` |
| `OldOnly` | `240` | `10` | `2.0678046281088522` | `-0.054483491929981` | `-0.09961161217652262` | `-1.1754935877397656` | `0.0` | `0.0` | `0.0` |
| `RandomMatched` | `1` | `87` | `0.09405836253137967` | `-0.2714390213452495` | `-0.04921209364701277` | `-0.06413099563669884` | `0.0` | `0.9433993751492317` | `0.9605466806820027` |
| `RandomMatched` | `5` | `87` | `0.14540610961983674` | `-0.7251219300571328` | `-0.09325921854496687` | `-0.060172725254776835` | `0.0` | `0.9433993751492317` | `0.9605466806820027` |
| `RandomMatched` | `20` | `87` | `0.5411237445554424` | `-1.0394428465347012` | `-0.15233070829390793` | `-0.23505213745366568` | `0.0` | `0.9433993751492317` | `0.9605466806820027` |
| `RandomMatched` | `80` | `87` | `1.8306221718364828` | `-1.1912445944181171` | `-0.23179052657736787` | `-0.8116410610576471` | `0.0` | `0.9433993751492317` | `0.9605466806820027` |
| `RandomMatched` | `240` | `87` | `3.5922551893613908` | `-0.7799894887792915` | `-0.30614706513675294` | `-1.510284047852131` | `0.8485412092119363` | `0.9433993751492317` | `0.9605466806820027` |

判断：P1 这轮不是 unavailable，而是真实补齐了 h1/h5/h20/h80/h240 selected-group trajectory。它证明 endpoint/path rows 可以落地；但这只是路径 materializer pass，不等价于 controller pass。

## 4. P2 Mechanism Contrast

| mechanism | Core vs Random effect | OldOnly vs ExactOnly effect | TopK87 precision | V LCB | AUV LCB | longrisk UCB | legal | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M_value_future` | `0.5191551103657229` | `-0.01161356152511517` | `0.4482758620689655` | `-0.4357864398037876` | `4.381331480649709` | `0.369780988566077` | `0` | `0` |
| `M_memory` | `0.5191551103657229` | `-0.01161356152511517` | `0.41379310344827586` | `-0.11427822648412846` | `2.716555586178205` | `0.0` | `1` | `0` |
| `M_hardtail` | `0.5191551103657229` | `-0.01161356152511517` | `0.45977011494252873` | `-0.3758039097392462` | `4.187244879395712` | `0.3570360950674304` | `0` | `0` |
| `M_cover` | `0.5191551103657229` | `-0.01161356152511517` | `0.21839080459770116` | `-0.2234966269284317` | `1.0224280574295677` | `0.2517897676033668` | `0` | `0` |
| `M_curvature` | `0.5191551103657229` | `-0.01161356152511517` | `0.3333333333333333` | `-0.5676380959572139` | `2.4280161790877273` | `0.43239147769431` | `0` | `0` |
| `M_signal` | `0.5191551103657229` | `-0.01161356152511517` | `0.8735632183908046` | `0.14061570456586386` | `2.7227479106902694` | `0.0` | `0` | `0` |
| `M_adamw_path` | `0.5191551103657229` | `-0.01161356152511517` | `0.39080459770114945` | `-0.486183631655888` | `3.642533969726318` | `0.4075601452666917` | `0` | `0` |

判断：P2 没有找到 legal-at-commit 且 precision/value/risk 同时满足的机制。OldRank / future path / response 类信号仍然强，但 diagnostic-only；memory/cost 类更合法，却不能选出足够好动作。

## 5. P3 Natural AP0 Labeled Stream Extension

| panel | target/action | status | labeled | CoreLike rate | LCB | UCB | reason |
|---|---:|---|---:|---:|---:|---:|---|
| `Panel-target-2876` | `2876` | `panel_row` | `2876` | `0.026773296244784424` | `0.021475240123524635` | `0.033333885489495195` | `` |
| `Panel-target-5000` | `5000` | `not_run` | `2876` | `` | `` | `` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9820` |
| `Panel-target-10000` | `10000` | `not_run` | `2876` | `` | `` | `` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9820` |
| `Panel-target-20000` | `20000` | `not_run` | `2876` | `` | `` | `` | `no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9820` |

判断：P3 仍未完成自然动作扩流。现有 2876 panel 可复核，5000/10000/20000 没有真实 labeled branch-horizon rows，所以不允许 density closure。

## 6. P4 Gate Semantics

```text
LDO_raw/quality/support/backfill = 0.4415584415584416 / 0.09260529551331587 / 0.0 / 0.34895314604512573
G_raw/G_support/G_density = 0 / 0 / 0
official_density_gate_proposal_allowed = 0
reason = raw_fail_decomposed_but_density_strong_pass_absent_keep_raw_gate
```

判断：raw/support/density gate 语义被拆开，但 density strong pass 不存在，因此不能提出 official gate 修改。

## 7. P5 Geometry Family Rebuild

| family | precision | V LCB | AUV LCB | longrisk UCB | legal | weak pass |
|---|---:|---:|---:|---:|---:|---:|
| `G_func` | `0.45977011494252873` | `-0.4020549613960917` | `4.110214135323428` | `0.3950403220387326` | `0` | `0` |
| `G_path` | `0.4482758620689655` | `-0.4357864398037876` | `4.381331480649709` | `0.369780988566077` | `0` | `0` |
| `G_memory` | `0.41379310344827586` | `-0.11427822648412846` | `2.716555586178205` | `0.0` | `1` | `0` |
| `G_stability` | `0.45977011494252873` | `-0.3758039097392462` | `4.187244879395712` | `0.3570360950674304` | `0` | `0` |
| `G_cost` | `0.22988505747126436` | `-0.21981076008236478` | `1.0988475479070037` | `0.21039105407869235` | `1` | `0` |

判断：P5 不再做单个大 score，而是拆成 family。结果仍没有 legal family 通过 precision/value/risk gate。

## 8. P6-P10 Boundary

```text
P6 controller = not_run, reason = P1_or_P2_or_P3_or_P4_or_P5_not_controller_ready
P7 generated = not_run, reason = P1_P2_P5_mechanism_not_established_generated_route_stopped
P8 runtime = not_run, reason = P6_controller_and_P7_generated_not_passed
P9 paired replay = not_run, reason = P8_runtime_not_passed
P10 short/full = not_run, reason = P9_paired_replay_not_passed
```

判断：没有 legal mechanism / geometry family / density pass，所以 controller、generated route、runtime、paired replay、short/full 全部保持关闭。

## 9. No-Fake / Contract / Failure

No-fake audit：

```text
rows_checked = 7555
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake/no_proxy = 1 / 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
p1_full_path_materialized = 1
p2_mechanism_pass = 0
p3_density_strong_pass = 0
controller/generated/runtime/system = 0/0/0/0
fake/proxy/cpu_offload = 0/0/0
```

Failure taxonomy：

```text
route = R-P2-NoMechanismExplainsGoodActions
F0_boundary_fail = 0
F1_future_path_materializer_fail = 0
F2_no_legal_mechanism = 1
F3_natural_stream_extension_missing = 1
F4_density_gate_not_resolved = 1
F5_geometry_family_no_legal_signal = 1
F6_controller_generated_runtime_blocked = 1
primary_blocker = no_legal_mechanism_explains_good_actions
secondary_blocker = natural_labeled_stream_extension_missing
```

## 10. Figures

```text
fig_p1_future_path_V_by_group_v9820.svg
fig_p1_future_path_longrisk_by_group_v9820.svg
fig_p2_mechanism_effect_forest_v9820.svg
fig_p3_density_curve_corelike_v9820.svg
fig_p4_gate_decomposition_waterfall_v9820.svg
fig_p5_geometry_family_quality_scatter_v9820.svg
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| `contract_audit_v9820.csv` | `1f0afb05fc5267b37a18ed7447a37b6b315d6208608a2ad532c939f424cacf11` |
| `failure_taxonomy_v9820.csv` | `73a693e24e790a573f664285cb3ac4488d1bd815223b45e9b9f27a9dfed7b4f2` |
| `no_fake_audit_v9820.csv` | `7355499648f93014d7faf7efb6a2a3b1eadaf3f7695b17e2ecf662ae6d127d07` |
| `p0_boundary_reproduction_v9820.csv` | `ce80abf9b0b286d63d487a229574cc89513ddaaf6a7ba7b40989929ee145669b` |
| `p0_field_legality_audit_v9820.csv` | `6eb8ddd4b2736a1829b761bab57715365e9c608314e2d07c82d31952cfe0a362` |
| `p10_short_full_training_boundary_v9820.csv` | `334e78c8a67a6537efb6b5f99120935354191c7fe2e0c9e45e215a53db3e5da6` |
| `p1_full_future_path_materializer_v9820.csv` | `6382f44a82be9e328187f2873fdcb0a4d4cba0834f464c4cb2b5e4bef22225e0` |
| `p1_future_path_completion_v9820.csv` | `e2d8e2346646c2919301f1e43b8aaf9e2895f45561461ca2156689aac93a7a03` |
| `p1_group_trajectory_summary_v9820.csv` | `59a28bdaf8e8726e2cf1b07eb26a96041e7a7ef5e0eb5305f1736c81e1ba32b3` |
| `p2_future_path_mechanism_contrast_v9820.csv` | `c7e812d5170c8da89a90b3018b5aecacf7d77e7965b2fa282d73888347c250ca` |
| `p3_natural_ap0_labeled_stream_extension_v9820.csv` | `bffb5702f2a17e384f28916cf590ec508418febc5e30d0cf8c328b37f424a30f` |
| `p4_gate_semantics_v9820.csv` | `85d6f9e3b13b97b69f5e65876d956eec3696ff4208d1890de173a2610f646556` |
| `p5_architecture_agnostic_geometry_rebuild_v9820.csv` | `152ceb6a2012137dc9a0dec4334cc24ebfba626806ba7aae4f8e0c9999a2e868` |
| `p6_existing_action_minimal_controller_boundary_v9820.csv` | `bd70680c59e77085df890260aa3d4306db1cd4e74c6821fe68846aaf078f5d59` |
| `p7_geometry_adaptive_generated_update_v9820.csv` | `deb7eea74efd822b50855e08d9b6dcb44f6675896080fbc788930ed7ca699c0e` |
| `p8_selected_runtime_boundary_v9820.csv` | `14b3b9de897692c52226d703b7fbd8d5dad24eb2a90b9105c59dc3e7247cc92b` |
| `p9_official_paired_replay_boundary_v9820.csv` | `1f809b50e28fd0ec2fe31ab2e14139d5046172842e98717892e0ad8ef281d23f` |
| `plan` | `08e672b7a343461c1dcd2e2a195e116bb65e944a446d5f822bd6be8225c9d8fc` |
| `route_decision_v9820.json` | `6b981a40039a6bda310cdc198aeb6f53dcdd036ef95cd34b489b681d0911d32e` |
| `run_manifest_v9820.json` | `bb64412d4ba6323afe6f4a305a849494305d9de0ee13fd083a2aed14327aca34` |
| `runner` | `b1062d38bfd91e1a6a6779b7e27f9234ae0743a7f80df6dcb1a6b6d705966c12` |

## 12. 最终分析结论

v9.8.2 的真实推进是：

```text
1. h1/h5 不再只是 unavailable：P1 对 selected groups 真实 materialize 了 h1/h5/h20/h80/h240。
2. Core77 / OldOnly 的 future path 端点和 AUV 仍然强，但 P2 没有找到 legal-at-commit 的机制能解释并选择这些动作。
3. Natural AP0 stream extension 仍未落地，5000/10000/20000 panel 没有真实 labels。
4. Gate semantics 已拆开，但 density strong pass 不存在，不能修改 official raw gate。
5. Geometry family decomposition 没有 legal family 通过 weak gate。
6. Controller / generated / runtime / replay / short-full 全部继续 gate-blocked。
```

最终一句话：

> v9.8.2 真实执行后停在 `R-P2-NoMechanismExplainsGoodActions`：本轮真正补上了 selected-action 的 h1/h5 full future path，这是比 v9.8.1 更实的推进；但 legal mechanism、natural action density、geometry family 和 generated branch-horizon 都没有过 gate，因此不能进入 official controller，strict PureKAN functional 仍未成功。
