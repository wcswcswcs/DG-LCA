# DG-KAN v9.8.4 Future Operator / Natural Stream / Geometry Adaptive Optimizer 实验复盘

> 本复盘记录 `DG-KAN_v9.8.4_未来轨迹算子_自然动作扩流_几何自适应优化器_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest；新增 A-line 组使用真实 branch-horizon replay；自然 AP0 扩流 materializer 缺失时显式 `not_run`，没有 fake data 或 proxy rows。

## 0. 最新结论

```text
route = R1-FuturePathMechanismFoundLegalMechanismAbsent
primary_blocker = no_legal_training_time_mechanism
secondary_blocker = natural_stream_materializer_missing
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_density_evidence
```

最终 artifact：`results/real_rerun_20260506/v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z`

核心结论：

1. P0 复现 v9.8.3 boundary：source route = `RouteB-FuturePathExistsMechanismAbsent`，P1/P2/P3/P5 = `1` / `0` / `1` / `0`。
2. A-line 新增组真实 replay：extra selected actions = `97`，extra future-path rows = `2910`。
3. P1 weak/strong = `1` / `0`；Core77 AUV LCB = `2.8296932871691163`，OldOnly AUV LCB = `1.0287178879851673`。
4. 新增 CoreExpansion10 AUV LCB = `0.6225266023303726`；RiskCleanButLowValue AUV LCB = `2.16033116127777`。
5. P2 legal mechanism weak/strong = `0` / `0`；best = `B2_signal_consistency_diagnostic`，legality = `red:old_table_or_windowed_diagnostic`。
6. P3 natural preflight C1d pass = `0`，materializer entrypoint found = `0`。
7. P4 panel completed/not_run = `1` / `3`，PanelA LCB = `0.021475240123524635`。
8. Gate modification allowed = `0`；backfill share = `0.7902762425139611`。
9. Virtual path probe = `not_run`；generated reopen allowed = `0`。
10. No-fake audit：rows checked = `3287`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py` | v9.8.4 runner；复现 v9.8.3 boundary，真实 materialize 新增 A-line 组，执行 legal mechanism / natural stream / virtual probe / generated reopen / controller-runtime boundary。 |

```text
python -m py_compile experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py
```

```bash
python experiments/run_v9840_future_operator_natural_stream_geometry_optimizer.py --out-dir results/real_rerun_20260506/v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9840",
  "status": "summary",
  "route": "R1-FuturePathMechanismFoundLegalMechanismAbsent",
  "route_secondary": "Stop1-NaturalStreamMaterializerMissing",
  "primary_blocker": "no_legal_training_time_mechanism",
  "secondary_blocker": "natural_stream_materializer_missing",
  "source_route_v9830": "RouteB-FuturePathExistsMechanismAbsent",
  "P0_boundary_pass": 1,
  "P1_weak_pass": 1,
  "P1_strong_pass": 0,
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P3_preflight_pass": 0,
  "P4_density_weak_pass": 0,
  "P4_density_strong_pass": 0,
  "P5_virtual_probe_pass": 0,
  "P6_generated_reopen_allowed": 0,
  "P7_generated_smoke_pass": 0,
  "P8_controller_pass": 0,
  "P9_runtime_pass": 0,
  "P10_paired_replay_pass": 0,
  "P11_short_full_pass": 0,
  "official_gate_modification_allowed": 0,
  "extra_selected_actions": 97,
  "extra_future_path_rows": 2910,
  "generated_route_status": "stopped_no_legal_mechanism_or_density_evidence",
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. Part A Future Path Mechanism

| group | actions | AUV LCB | V1 LCB | V20 LCB | V80 LCB | V240 LCB | RiskPath UCB | IND rate | memory/offdiag UCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Core77` | `77` | `2.8296932871691163` | `0.08815910625793319` | `1.573549156031694` | `3.1200217020467886` | `4.750885696264312` | `0.03827572901862327` | `0.4155844155844156` | `0.0` |
| `CoreExpansion10` | `10` | `0.6225266023303726` | `-0.04039991800257209` | `-0.029607953167106515` | `0.5669308435951039` | `1.623630340725588` | `0.0` | `0.2` | `0.0` |
| `ExactOnly` | `68` | `1.5250480927868013` | `0.13699866747303657` | `0.4931219640179562` | `1.582392387764584` | `2.922090918869295` | `0.0` | `0.35294117647058826` | `0.0` |
| `OldOnly` | `10` | `1.0287178879851673` | `-0.174770564086397` | `0.30014089601530236` | `1.1093912874075194` | `2.0678046281088522` | `0.0` | `0.5` | `0.0` |
| `RandomMatched` | `87` | `1.8284728866377598` | `0.09405836253137967` | `0.5411237445554424` | `1.8306221718364828` | `3.5922551893613908` | `0.8485412092119363` | `0.11494252873563218` | `1.9039460558312342` |
| `RiskCleanButLowValue` | `87` | `2.16033116127777` | `0.11800662600687634` | `0.592558920388403` | `2.268809387190667` | `4.269593559964293` | `0.054480608015849245` | `0.367816091954023` | `0.0` |

判断：A weak pass = `1`，A strong pass = `0`。Strong 未过的原因是 `OldOnly_AUV_not_significantly_above_ExactOnly_or_RandomMatched`；这不是 materializer 失败，而是更严格的 OldOnly / Core77 对照差异不够。

## 4. Part B Legal Mechanism

| mechanism | legality | precision | AUV LCB | V20/V80/V240 LCB | longrisk UCB | memory/offdiag UCB | LDO/LSO/LTO | weak | strong |
|---|---|---:|---:|---|---:|---:|---|---:|---:|
| `B1_function_response_current_diagnostic` | `red:future_replay_diagnostic` | `0.08045977011494253` | `1.1886694485089664` | `0.11803410433542048`/`1.1850809857471023`/`2.6093712625368672` | `0.23813471172069484` | `0.5171056772713958` | `0.034482758620689655`/`0.011494252873563218`/`0.011494252873563218` | `0` | `0` |
| `B2_signal_consistency_diagnostic` | `red:old_table_or_windowed_diagnostic` | `0.8735632183908046` | `2.8038542184231736` | `1.5456944935705927`/`3.104754449680345`/`4.703181791801234` | `0.0` | `0.0` | `0.37931034482758624`/`0.02298850574712652`/`0.011494252873563315` | `0` | `0` |
| `B3_memory_old_knowledge_safety` | `green` | `0.2413793103448276` | `2.716555586178205` | `1.4867790222027628`/`3.001260759694673`/`4.5826527579053815` | `0.0` | `0.0` | `0.08045977011494254`/`0.011494252873563232`/`0.011494252873563232` | `0` | `0` |
| `B4_path_proxy_probe_not_landed` | `green_but_not_materialized` | `0.16091954022988506` | `1.1183329597630347` | `0.2318320539301606`/`1.0971005148380197`/`2.355876159736847` | `0.13761700438354324` | `0.3494085592421139` | `0.0`/`0.011494252873563204`/`0.011494252873563204` | `0` | `0` |
| `B5_payload_norm_cost` | `green` | `0.16091954022988506` | `1.1183329597630347` | `0.2318320539301606`/`1.0971005148380197`/`2.355876159736847` | `0.13761700438354324` | `0.3494085592421139` | `0.0`/`0.011494252873563204`/`0.011494252873563204` | `0` | `0` |

判断：B 线没有 legal weak/strong pass。绿色 memory/cost 信号仍然更像安全门，不是收益源；red diagnostic 不进入 official controller。

## 5. Part C Natural AP0 Stream

```text
C1d_pass = 0
materializer_entrypoint_found = 0
PanelA_CoreLike_rate = 0.026773296244784424
PanelA_CoreLike_LCB = 0.021475240123524635
PanelB/C/D = not_run
```

判断：自然 AP0 扩流仍是工程 blocker；没有 5000/10000/20000 真实 labels，因此不能作 density closure。

## 6. Part D/E Generated / Controller Boundary

```text
virtual_probe_status = not_run
generated_route_reopen_allowed = 0
generated_smoke = not_run
controller/runtime/replay/shortfull = not_run
```

判断：B 线没有 legal mechanism，C 线没有 density evidence，virtual probe 未落地，所以 D/E 不打开。

## 7. Base-Acc / Audit / Failure

```text
base_acc_sentinel_pass = 1
rows_checked = 3287
route = R1-FuturePathMechanismFoundLegalMechanismAbsent
primary_blocker = no_legal_training_time_mechanism
secondary_blocker = natural_stream_materializer_missing
```

## 8. Figures

```text
fig_A1_group_future_path_V_curve.svg
fig_A2_group_future_path_risk_curve.svg
fig_A3_oldonly_immediate_vs_future.svg
fig_A4_exactonly_failure_path.svg
fig_A5_core77_vs_random_memory_offdiag_path.svg
fig_A6_path_shape_slope_scatter.svg
fig_A7_AUV_vs_longrisk_pareto.svg
fig_B1_mechanism_topK_quality_bar.svg
fig_B2_mechanism_auv_vs_risk.svg
fig_B3_signal_consistency_vs_future_AUV.svg
fig_B4_virtual_probe_cost_quality_pareto.svg
fig_B5_memory_safety_as_veto.svg
fig_B6_legal_vs_red_feature_comparison.svg
fig_C1_density_curve_panel_size.svg
fig_C2_corelike_rate_ci.svg
fig_C3_per_dataset_density.svg
fig_C4_per_family_density_heatmap.svg
fig_C5_materializer_throughput_curve.svg
fig_C6_missing_label_dashboard.svg
fig_D1_generated_quality_pareto.svg
fig_D2_generated_future_path_curves.svg
fig_D3_generated_memory_offdiag_dashboard.svg
fig_D4_generated_vs_existing_density.svg
fig_D5_runtime_apply_cost.svg
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| `base_acc_sentinel_v9840.csv` | `2b0190bdc4c147651e69a964a2783e744cb0bc6e3d5a2fc1522f70bb99ee7ec3` |
| `contract_audit_v9840.csv` | `a9f418bdf748721dffab2bff314cf74416913ca12c736c92ab83cde044d07e79` |
| `failure_taxonomy_v9840.csv` | `d2c13b8f56c52fdd48c7182314533045d2f94aa4ad710b54a547ecd144ac6b08` |
| `field_legality_ledger_v9840.csv` | `c2663c6d63bb64cff71e580787f34f2678195ad0e2b05fd7979982cea141bb70` |
| `no_fake_audit_v9840.csv` | `d4fbc8fe3803cbafdab6824aacfa8b8dd8c31dbe016fb2054a5a804d236f0a90` |
| `p0_boundary_reproduction_v9840.csv` | `dfb7fb391100738ab6f0759ff5ae19998d629e2b6bb81205828f5cae08ccefb4` |
| `p10_paired_replay_boundary_v9840.csv` | `8794f4f3912c057aa9b4b43eb1b6e17e75cc2219ddc851bfbf5431e36947364e` |
| `p11_short_full_boundary_v9840.csv` | `b9ec2c2f475f41dadc330bc31eec7d63d5096ef83d40ad5b9d048d3e09c76d70` |
| `p1_extra_future_path_materializer_v9840.csv` | `9dbe44cc749512bd7e2a84f3778ea911fc00c9ee92aee2bcee4032760e28dbba` |
| `p1_future_path_mechanism_decomposition_v3.csv` | `ea657558c9ee830af17d8854994d676eb25946e446f36649f0b2b8bcd47a2735` |
| `p2_legal_training_time_mechanism_v3.csv` | `b4e931ce00f0a0e96862d993f694f0995194956ac1d265a04f379e28fa169aac` |
| `p3_natural_ap0_stream_preflight_v9840.csv` | `48a50c362a0b4aeb373d7e373bbcd2c8cf6020574c394b63467487bd2eaf7877` |
| `p4_gate_semantics_v9840.csv` | `379c4dbd7d9aa2f368c7a8263c8a0a5c308d7d884da9541a6774a1cbf8bfb06d` |
| `p4_natural_ap0_density_panel_v9840.csv` | `e79b78dbabe746097f1a1ef62d3e7f1d818bdc5857be1b0cc9832309a1608996` |
| `p5_virtual_path_probe_v9840.csv` | `4750abdcdc1b0fd527f76890794a677f5dbe0889302eb6527f918a1449ca6f48` |
| `p6_geometry_adaptive_generated_reopen_gate_v9840.csv` | `0d2a53e0be6463a7aa9c57903d82d9a1585f04eaf99f28bfcd8bd2c3b3e6e29b` |
| `p7_generated_update_smoke_v9840.csv` | `a919f26e8cb8a419c085e2e1b3e1d514e2342825ee437aa0507375c46b9da7c0` |
| `p8_minimal_controller_boundary_v9840.csv` | `74eb1b242a73ff380b0e5e738ec5385ecb042656595f0ac4370345121951f970` |
| `p9_selected_runtime_boundary_v9840.csv` | `fd3fd686301ab3d312e2d9f81845450fa7c65bbaf56eaf01560275cac024aac3` |
| `plan` | `b800f6edd5d86ddb47ae705ad6d1aaa43016cc6ac87d8eb5b95cfbdf062fa5dc` |
| `route_decision_v9840.json` | `7bac704eeed6fc259e34e6284c63306ecce738ae46543d738e283e47ffcbed93` |
| `run_manifest_v9840.json` | `a4ba81731c1e06cde605a1ca5e1d341b83b858ba0ff7987a685c18f75097a743` |
| `runner` | `70b159f6f39477f92f2102857b9ef1509bce889e1d63e6916da3dab32ca87524` |

## 10. 最终分析结论

```text
1. v9.8.4 真实补跑了新增 A-line existing-action 组，而不是只复用旧表。
2. Core77 future path 仍强，新增 CoreExpansion10/RiskCleanButLowValue 给出额外对照；但 A strong 在严格 OldOnly/Random/Exact 对照下未完全闭合。
3. B 线没有找到 legal training-time mechanism。
4. C 线 natural AP0 扩流 materializer 仍未落地。
5. D 线 generated route 不允许重开。
6. E 线 controller/runtime/replay/short-full 全部 gate-blocked。
```

最终一句话：

> v9.8.4 真实执行后停在 `R1-FuturePathMechanismFoundLegalMechanismAbsent`：本轮把新增 A-line 组真实 replay 了，但训练当下合法机制、自然动作扩流和虚拟路径 probe 仍未闭合，因此不能进入 official controller，strict PureKAN functional 仍未成功。
