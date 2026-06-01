# DG-KAN v9.8.6 Future Path Mechanism / Natural Stream / Geometry Adaptive Optimizer 实验复盘

> 本复盘记录 `DG-KAN_v9.8.6_结果解读_未来路径机制_自然扩流_几何自适应优化器_完整计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.4/v9.8.5 已真实 materialized 的 landed rows；没有 fake data、proxy rows，也没有把 future-path diagnostic、future-operator proxy diagnostic、natural stream missing 或 generated not_run 写成 official controller pass。

## 0. 最新结论

```text
route = R5-LegalProxyFailNaturalStreamMissing
primary_blocker = legal_future_operator_proxy_failed
secondary_blocker = natural_stream_materializer_missing
route_recommendation = prioritize_C_materializer_or_pivot
system_legal_controller_pass = 0
generated_route_status = stopped_no_B_or_C_mechanism_evidence
```

最终 artifact：`results/real_rerun_20260506/v9860_future_path_mechanism_natural_stream_geometry_optimizer_full_20260516T200000Z`

核心结论：

1. P0 复现 v9.8.5 boundary：source route = `Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped`，A/B/C/D = `1/0/0/0`。
2. 本轮引用 landed future rows = `1695`，actions = `339`，horizons = `1,5,20,80,240`。
3. P1 重审组合路径：best = `Core77+RiskCleanButLowValueTop10`，AUV LCB = `3.2562463420439274`，support-adjusted LDO = `0.20084099015084253`。
4. P1 combo strong pass count = `0`，density dispute count = `0`。
5. P2 FO1-FO8 future-operator proxy 全部失败：weak/strong = `0` / `0`，best proxy = `FO8_FO123HardVetoNoWeightedScore`。
6. P2 best precision/V/cost q90 = `0.3563218390804598` / `-0.27147985776773054` / `155.12761380523443`。
7. P3 natural stream materializer entrypoint found = `0`；PanelA CoreLike LCB = `0.021475240123524635`。
8. P4 generated sandbox allowed = `0`，status = `stopped_no_B_or_C_mechanism_evidence`。
9. P5 controller = `not_run`，reason = `P1_P2_P3_not_official_controller_ready`。
10. No-fake audit：rows checked = `1737`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py` | v9.8.6 runner；读取 v9.8.5/v9.8.4/v9.8.2 landed rows，执行 Core77+Expansion combo reassessment、FO1-FO8 future-operator proxy、natural AP0 materializer audit、generated sandbox decision 与 controller/runtime boundary。 |

```text
python -m py_compile experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py
```

```bash
python experiments/run_v9860_future_path_mechanism_natural_stream_geometry_optimizer.py --out-dir results/real_rerun_20260506/v9860_future_path_mechanism_natural_stream_geometry_optimizer_full_20260516T200000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

说明：P1/P2 使用已真实落盘的 branch-horizon/train-probe rows；P3 因没有 natural labeled stream extension materializer，C0/C1/C2 与 5000/10000/20000 panel 都显式 `not_run`。

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9860",
  "status": "summary",
  "route": "R5-LegalProxyFailNaturalStreamMissing",
  "primary_blocker": "legal_future_operator_proxy_failed",
  "secondary_blocker": "natural_stream_materializer_missing",
  "route_recommendation": "prioritize_C_materializer_or_pivot",
  "source_route_v9850": "Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped",
  "P0_boundary_pass": 1,
  "P1_A_combo_strong_pass": 0,
  "P1_density_gate_dispute": 0,
  "P2_weak_pass": 0,
  "P2_strong_pass": 0,
  "P3_materializer_entrypoint_found": 0,
  "P3_density_weak_pass": 0,
  "P3_density_strong_pass": 0,
  "P3_density_fail": 0,
  "P4_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_no_B_or_C_mechanism_evidence",
  "P5_controller_pass": 0,
  "P6_runtime_pass": 0,
  "P7_paired_replay_pass": 0,
  "P8_short_full_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Future Path Combo Reassessment

| candidate | actions | AUV LCB | V80 LCB | V240 LCB | longrisk UCB | mem/off UCB | GradeAB precision | raw/support LDO | LSO/LTO/LFO | strong | density dispute |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `Core77` | `77` | `2.8296932871691163` | `3.1200217020467886` | `4.750885696264312` | `0.03827572901862327` | `0.0` | `1.0` | `0.4415584415584416`/`0.09260529551331587` | `0.038961038961038974`/`0.012987012987012991`/`0.18181818181818177` | `0` | `0` |
| `CoreExpansion10` | `10` | `0.6225266023303726` | `0.5669308435951039` | `1.623630340725588` | `0.0` | `0.0` | `0.0` | `0.0`/`0.009431541733713869` | `0.0`/`0.0`/`0.0` | `0` | `0` |
| `Core77+CoreExpansion10` | `87` | `2.722153601222041` | `2.993949373729339` | `4.588476154696239` | `0.03389313880385762` | `0.0` | `0.8850574712643678` | `0.39080459770114945`/`0.13505747126436785` | `0.034482758620689724`/`0.011494252873563204`/`0.16091954022988508` | `0` | `0` |
| `OldOnly` | `10` | `1.0287178879851673` | `1.1093912874075194` | `2.0678046281088522` | `0.0` | `0.0` | `0.0` | `0.0`/`0.0019193408800554056` | `0.0`/`0.0`/`0.0` | `0` | `0` |
| `ExactOnly` | `68` | `1.5250480927868013` | `1.582392387764584` | `2.922090918869295` | `0.0` | `0.0` | `0.0` | `0.0`/`0.33690788577680353` | `0.0`/`0.0`/`0.0` | `0` | `0` |
| `RandomMatched` | `87` | `1.8284728866377598` | `1.8306221718364828` | `3.5922551893613908` | `0.8485412092119363` | `1.9039460558312342` | `0.0` | `0.0`/`0.39471037761585115` | `0.0`/`0.0`/`0.0` | `0` | `0` |
| `RiskCleanButLowValue` | `87` | `2.16033116127777` | `2.268809387190667` | `4.269593559964293` | `0.054480608015849245` | `0.0` | `0.0` | `0.0`/`0.19904006890565495` | `0.0`/`0.0`/`0.0` | `0` | `0` |
| `Core77+RiskCleanButLowValueTop10` | `87` | `3.2562463420439274` | `3.632953825093594` | `5.462997655266469` | `0.03389313880385762` | `0.0` | `0.8850574712643678` | `0.39080459770114945`/`0.20084099015084253` | `0.034482758620689724`/`0.011494252873563204`/`0.16091954022988508` | `0` | `0` |

判断：P1 确认 Core77+CoreExpansion10 等组合可以形成更完整的 future-path diagnostic 对照，但 controller-ready 仍需要合法机制或 density gate closure。

## 4. P2 Future-Operator Proxy

| proxy | precision | V LCB | AUV LCB | RiskAdjustedAUV LCB | longrisk UCB | mem/off UCB | cost q90 ms | weak | strong | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `FO1_AdamWAlignedFutureResidual` | `0.3218390804597701` | `-0.576276588394034` | `3.9182654069288114` | `3.332756090383112` | `0.3442102410706317` | `0.7140721901348608` | `154.03002640232444` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05;cost_q90_above_1.5ms` |
| `FO2_MemoryBufferResponseStability` | `0.1724137931034483` | `-0.2566922907549677` | `1.1067854951505591` | `1.0930437425216268` | `0.0` | `0.0` | `152.9591870494187` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;cost_q90_above_1.5ms` |
| `FO3_HardTailContraction` | `0.3218390804597701` | `-0.5292339423122111` | `3.452220749850755` | `2.855550860171201` | `0.33129982990159157` | `0.7268170836335074` | `155.18634486943483` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05;cost_q90_above_1.5ms` |
| `FO4_OldFamilyMarginPreservation` | `0.26436781609195403` | `-0.6116343262566473` | `3.539460545953591` | `2.976337277216841` | `0.3183007985873107` | `0.7140721901348608` | `153.79969729110599` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05;cost_q90_above_1.5ms` |
| `FO5_MultiBatchAgreement` | `0.22988505747126436` | `-0.4787812712036512` | `2.2189246230150537` | `1.7145575987645398` | `0.265315909668029` | `0.6235093377805891` | `153.212146833539` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05;cost_q90_above_1.5ms` |
| `FO6_LowCostTwoSampleVirtualPath` | `0.26436781609195403` | `-0.6180973334711484` | `3.9048372067398134` | `3.185731172341303` | `0.4075601452666917` | `0.86478295538862` | `153.212146833539` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05;cost_q90_above_1.5ms` |
| `FO7_RiskVetoOnly` | `0.27586206896551724` | `-0.311212043098752` | `2.5347177158052454` | `2.522097241836135` | `0.0` | `0.0` | `153.2128108665347` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;cost_q90_above_1.5ms` |
| `FO8_FO123HardVetoNoWeightedScore` | `0.3563218390804598` | `-0.27147985776773054` | `3.3271077422640407` | `3.3271077422640407` | `0.0` | `0.0` | `155.12761380523443` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;cost_q90_above_1.5ms` |

判断：P2 这次不是复用 B1-B5，而是 FO1-FO8；结果仍没有 legal low-cost proxy pass，主要受 precision/value/risk 或 cost gate 阻断。

## 5. P3 Natural AP0 Stream

| panel | status | target | labeled | expected rows | actual rows | CoreLike rate | LCB/UCB | reason |
|---|---|---:|---:|---:|---:|---:|---|---|
| `Panel-2876` | `panel_row` | `2876` | `2876` | `` | `` | `0.026773296244784424` | `0.021475240123524635`/`0.033333885489495195` | `` |
| `Panel-5000` | `not_run` | `5000` | `2876` | `150000` | `0` | `` | ``/`` | `P3_preflight_not_passed_materializer_missing` |
| `Panel-10000` | `not_run` | `10000` | `2876` | `300000` | `0` | `` | ``/`` | `P3_preflight_not_passed_materializer_missing` |
| `Panel-20000` | `not_run` | `20000` | `2876` | `600000` | `0` | `` | ``/`` | `P3_preflight_not_passed_materializer_missing` |

判断：C 线仍未落地 materializer。PanelA 是既有 2876 labeled AP0 density panel，不是 5000/10000/20000 扩流结果。

## 6. P4-P8 Boundary

```text
generated_sandbox_allowed = 0
generated_route_status = stopped_no_B_or_C_mechanism_evidence
controller/runtime/paired replay/short-full = not_run
```

判断：没有 B legal proxy pass，也没有 C density sufficient/insufficient 证据，因此 generated sandbox、controller、runtime、paired replay、short/full 全部关闭。

## 7. Figures

```text
fig_A1_path_combo_V_curve.svg
fig_A2_path_combo_risk_curve.svg
fig_A3_core77_plus_expansion_pareto.svg
fig_A4_raw_LDO_vs_support_adjusted_LDO.svg
fig_A5_per_dataset_path_support.svg
fig_B1_proxy_precision_vs_cost.svg
fig_B2_proxy_V_vs_longrisk.svg
fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg
fig_B4_proxy_score_vs_future_AUV_scatter.svg
fig_B5_proxy_score_distribution_by_group.svg
fig_C1_density_curve_panel_size.svg
fig_C2_corelike_rate_ci.svg
fig_D1_generated_sandbox_gate.svg
```

## 8. No-Fake / Contract / Failure

```text
rows_checked = 1737
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| `base_acc_sentinel_v9860.csv` | `be8f2cecc4cc5689bdfadb9baaead46a14f77de78922f31cb666a91a61994ecc` |
| `contract_audit_v9860.csv` | `2728a86d28ed15ed7391634281d29eb6b0f538276032fc622d635b5917fb6765` |
| `failure_taxonomy_v9860.csv` | `7462dd796d09d61be02342c0e04da26b5f7a836942032474000e95d17e822881` |
| `field_legality_ledger_v9860.csv` | `735f9a2427fd311f905d5098a70672b4dcc2d87651b6b1cd4f16b7d8021be156` |
| `no_fake_audit_v9860.csv` | `4cc6323413a3df313a7f0051446035c97586b2b28f9e1f53cd6431abb5ffb868` |
| `p0_boundary_reproduction_v9860.csv` | `10367ec4234f81333228a544c31bb181b3b3585489d346193c4f8306db90ebd6` |
| `p0_landed_future_row_reference_v9860.csv` | `0fbe2b92125998e9f1f143dabd596a7094782fe4d5ba7f8037de4929d0c6bb6c` |
| `p1_future_path_combo_reassessment_v9860.csv` | `94edfaf2ac5c71a8c8f31b49b097b4d0ae86b589320c4ae02ac6cfd45d5a2559` |
| `p2_future_operator_proxy_v9860.csv` | `6c5028f9c234055fac0c5d18c6ee94f1e18ceae59ab12bd0eba136f62f51d910` |
| `p3_natural_ap0_density_panel_v9860.csv` | `4e6610896bba0235bfc282bb7298639090c9400a92c542c56c211241cf7dd29c` |
| `p3_natural_ap0_stream_materializer_v9860.csv` | `0e503325d3ba38f3b3f0291b49746d9c8119fbfee0863c4de7cd076105029d2e` |
| `p4_generated_conditional_sandbox_v9860.csv` | `6ac1aa8ce320251386b726372c72a437c1a8eda80e8bc8acddacc3b88ec97f6d` |
| `p5_minimal_controller_boundary_v9860.csv` | `57989e76bb522dfbd003735d8cf440b4a72612b125b6665cc05441187c28ec60` |
| `p6_selected_runtime_boundary_v9860.csv` | `07ab630bb5b75b4b9c3e0ef7a6601967797823d9de70dcad9dac5cf41f40339c` |
| `p7_paired_replay_boundary_v9860.csv` | `4bbab21d62a7cf162210165d22f8f270167a0e59d4b73a3b30e239fdfbce4b58` |
| `p8_short_full_boundary_v9860.csv` | `8800aa04eae69a9f6e0657416e3f33725e413f257785a35fdcb6f2361eae47ee` |
| `plan` | `45c08d347865eec258019e98f58fc40aa3b296550c2daa5aefee1d5d8274b08c` |
| `route_decision_v9860.json` | `dfa93af55323b947c6bd080a4b1672a8f9e6b5387c8cf4e17d7bfb241e8ed16d` |
| `run_manifest_v9860.json` | `b9924647ed5653c7a57babb2eb8c4b5429170d87768f8785110c1369f8962f35` |
| `runner` | `9a8a839b5e88eb5854f32c5fcc3bdfe6bee4173d7040999d1d974bd9781f051c` |
| `v9860_dashboard.md` | `7f980f92ad9cf8d1e7a656139ee6d5b1ab07edb1d388508b0e9ca9a05f48a9ed` |

## 10. 最终分析结论

```text
1. P1 证明 Core77+CoreExpansion10 等组合值得作为 future-path diagnostic candidate，但仍不是合法 controller。
2. P2 FO1-FO8 没有找到低成本、训练当下合法、可同时满足 precision/value/risk 的 future-operator proxy。
3. P3 natural AP0 stream materializer 仍缺失，density 不能裁决。
4. P4 generated sandbox 没有打开条件。
5. P5-P8 全部 gate-blocked，strict PureKAN functional 仍未成功。
```

最终一句话：

> v9.8.6 真实执行后停在 `R5-LegalProxyFailNaturalStreamMissing`：future-path combo 的诊断信号更完整，但合法 future-operator proxy 仍失败，自然 AP0 扩流 materializer 仍未落地，因此不能进入 official controller，generated route 继续停止。
