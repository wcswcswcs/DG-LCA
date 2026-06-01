# DG-KAN v9.8.7 Four-Line Natural Stream / FutureOperator Mechanism 实验复盘

> 本复盘记录 `DG-KAN_v9.8.7_四线并行_自然扩流_FutureOperator机制_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest；P1 preflight 使用真实 branch-horizon replay rows；5000/10000/20000 natural AP0 extension 因缺少新自然动作生成入口显式 `not_run`，没有 fake data、proxy rows 或 CPU offload。

## 0. 最新结论

```text
route = CaseE-MaterializerExtensionGeneratorMissing
primary_blocker = natural_stream_extension_generator_missing
secondary_blocker = legal_low_cost_proxy_failed
route_recommendation = prioritize_C_materializer_engineering
system_legal_controller_pass = 0
generated_route_status = stopped_no_B_or_C_or_A_reopen_condition
```

最终 artifact：`results/real_rerun_20260506/v9870_four_line_natural_stream_futureoperator_mechanism_full_20260516T220000Z`

核心结论：

1. P0 复现 v9.8.6 boundary：source route = `R5-LegalProxyFailNaturalStreamMissing`，system pass = `0`。
2. P1 真实 preflight materialized：selected actions = `256`，row count expected/actual = `7680` / `7680`，P1a/P1b/P1c = `1` / `1` / `1`。
3. P1d/P1e 扩展 panel 未打开：natural extension action generator found = `0`；reason = `existing_AP0_preflight_materialized_but_new_natural_AP0_extension_action_generator_missing`。
4. P2 density 仍 inconclusive：PanelA CoreLike LCB/UCB = `0.021475240123524635` / `0.033333885489495195`；P2 sufficient/insufficient/inconclusive = `0` / `0` / `1`。
5. P3 future path weak/strong = `1` / `0`；Core77 AUV LCB = `2.8296932871691163`。
6. P4 low-cost proxy weak/strong = `0` / `0`；best = `LC1_CheapMemoryHardTailResponse`，precision/V/cost q90 = `0.16091954022988506` / `-0.289129304605925` / `0.0011222437024116516`。
7. P5 controller = `not_run`；P6 generated sandbox allowed = `0`。
8. No-fake audit：rows checked = `9929`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py` | v9.8.7 runner；复现 v9.8.6 boundary，真实运行 existing AP0 natural preflight branch-horizon materializer，写 5000/10000/20000 extension blocker，执行 density/A-line/P4 low-cost proxy/controller/generated boundary。 |

```text
python -m py_compile experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py
```

```bash
python experiments/run_v9870_four_line_natural_stream_futureoperator_mechanism.py --out-dir results/real_rerun_20260506/v9870_four_line_natural_stream_futureoperator_mechanism_full_20260516T220000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9870",
  "status": "summary",
  "route": "CaseE-MaterializerExtensionGeneratorMissing",
  "primary_blocker": "natural_stream_extension_generator_missing",
  "secondary_blocker": "legal_low_cost_proxy_failed",
  "route_recommendation": "prioritize_C_materializer_engineering",
  "source_route_v9860": "R5-LegalProxyFailNaturalStreamMissing",
  "P0_boundary_pass": 1,
  "P1a_single_preflight_pass": 1,
  "P1b_16_preflight_pass": 1,
  "P1c_256_smoke_pass": 1,
  "P1d_5000_panel_weak_pass": 0,
  "P2_density_sufficient": 0,
  "P2_density_insufficient": 0,
  "P2_density_inconclusive": 1,
  "P3_A_line_weak_pass": 1,
  "P3_A_line_strong_mechanism_pass": 0,
  "P4_weak_pass": 0,
  "P4_strong_pass": 0,
  "P5_controller_pass": 0,
  "P6_generated_sandbox_allowed": 0,
  "generated_route_status": "stopped_no_B_or_C_or_A_reopen_condition",
  "P7_runtime_pass": 0,
  "P8_paired_replay_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. P1 Natural Materializer Preflight

| stage | target | labeled | expected rows | actual rows | completion | quality pass |
|---|---:|---:|---:|---:|---:|---:|
| `P1a` | `1` | `1` | `30` | `30` | `1.0` | `1` |
| `P1b` | `16` | `16` | `480` | `480` | `1.0` | `1` |
| `P1c` | `256` | `256` | `7680` | `7680` | `1.0` | `1` |

判断：P1a/P1b/P1c 是真实 replay preflight；但它只覆盖 existing AP0 动作。5000/10000/20000 需要新增 natural AP0 action generator，当前没有落地，不能伪造扩流 panel。

## 4. P2 Density Curve

| panel | status | size | CoreLike rate | CoreLike LCB/UCB | PathGood rate | PathGood LCB/UCB | reason |
|---|---|---:|---:|---|---:|---|---|
| `Panel-2876-existing` | `panel_row` | `2876` | `0.026773296244784424` | `0.021475240123524635`/`0.033333885489495195` | `` | ``/`` | `` |
| `Preflight-256-existing-AP0` | `diagnostic_preflight_row` | `256` | `0.0` | `0.0`/`0.014784391721725852` | `0.03125` | `0.015918067732609058`/`0.06044229950650894` | `` |
| `Panel-5000` | `not_run` | `5000` | `` | ``/`` | `` | ``/`` | `natural_AP0_extension_action_generator_missing_after_existing_2876_AP0_actions` |
| `Panel-10000` | `not_run` | `10000` | `` | ``/`` | `` | ``/`` | `natural_AP0_extension_action_generator_missing_after_existing_2876_AP0_actions` |
| `Panel-20000` | `not_run` | `20000` | `` | ``/`` | `` | ``/`` | `natural_AP0_extension_action_generator_missing_after_existing_2876_AP0_actions` |

判断：density 仍不能裁决。Panel-2876 只支持旧 CoreLike CI；preflight PathGood 是 diagnostic，不是扩展 panel。

## 5. P3 Future Path Mechanism

| group | actions | AUV LCB | V1 LCB | V80 LCB | V240 LCB | longrisk UCB | weak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Core77` | `77` | `2.8296932871691163` | `0.08815910625793319` | `3.1200217020467886` | `4.750885696264312` | `0.03827572901862327` | `1` |
| `CoreExpansion10` | `10` | `0.6225266023303726` | `-0.04039991800257209` | `0.5669308435951039` | `1.623630340725588` | `0.0` | `1` |
| `Core77+CoreExpansion10` | `87` | `2.722153601222041` | `0.11702769816449304` | `2.993949373729339` | `4.588476154696239` | `0.03389313880385762` | `1` |
| `RiskCleanButLowValue` | `87` | `2.16033116127777` | `0.11800662600687634` | `2.268809387190667` | `4.269593559964293` | `0.054480608015849245` | `0` |
| `Core77+RiskCleanButLowValueTop10` | `87` | `3.2562463420439274` | `0.16186559086365848` | `3.632953825093594` | `5.462997655266469` | `0.03389313880385762` | `1` |
| `OldOnly` | `10` | `1.0287178879851673` | `-0.174770564086397` | `1.1093912874075194` | `2.0678046281088522` | `0.0` | `1` |
| `ExactOnly` | `68` | `1.5250480927868013` | `0.13699866747303657` | `1.582392387764584` | `2.922090918869295` | `0.0` | `1` |
| `RandomMatched` | `87` | `1.8284728866377598` | `0.09405836253137967` | `1.8306221718364828` | `3.5922551893613908` | `0.8485412092119363` | `0` |
| `PathGoodFromNaturalPreflight` | `8` | `1.05868367387932` | `-0.006187013904225852` | `1.0437038207618778` | `2.3061338763449823` | `0.0` | `1` |
| `SlowBurnGoodFromNaturalPreflight` | `2` | `0.8666729951996963` | `-0.12934401752928565` | `0.828836497374846` | `1.883964165362788` | `0.0` | `1` |

## 6. P4 Low-Cost Proxy

| proxy | legality | precision | V LCB | AUV LCB | longrisk UCB | cost q90 ms | weak | strong | failure |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `LC1_CheapMemoryHardTailResponse` | `green:train_time_memory_hardtail_response` | `0.16091954022988506` | `-0.289129304605925` | `1.244672398943674` | `0.0` | `0.0011222437024116516` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive` |
| `LC2_LowRankFutureOperatorSketch` | `yellow:low_rank_sketch_from_recent_train_response_no_future_labels` | `0.16091954022988506` | `-0.8981221643414681` | `4.757971877946646` | `0.5761577373425045` | `0.000941101461648941` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05` |
| `LC3_SignalChannelAgreement` | `green:multi_channel_train_signal_agreement` | `0.10344827586206896` | `-0.6034471686218974` | `2.0818354385245463` | `0.5053405200342863` | `0.01966021955013275` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive;longrisk_above_0.05;memory_or_offdiag_above_0.05` |
| `LC4_HardVetoSignalAgreement` | `green:signal_agreement_with_memory_offdiag_veto` | `0.14942528735632185` | `-0.2883587492454408` | `1.1206657488817826` | `0.0` | `0.01931888982653618` | `0` | `0` | `precision_below_0.70;V_LCB_nonpositive` |

判断：P4 不再继续 FO1-FO8，而是测低成本 train-time proxy；结果仍未通过 precision/value/risk/leaveout gate。

## 7. P5-P8 Boundary

```text
P5 controller = not_run, reason = C_density_not_sufficient_B_proxy_not_passed_A_strong_plus_5000_panel_absent
P6 generated = not_run, allowed = 0, status = stopped_no_B_or_C_or_A_reopen_condition
P7 runtime = not_run
P8 paired replay = not_run
```

## 8. No-Fake / Contract / Failure

```text
rows_checked = 9929
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 9. Figures

```text
fig_p1_materializer_completion_by_stage.svg
fig_p1_rows_per_sec_by_panel.svg
fig_p1_exception_type_bar.svg
fig_p2_density_curve_corelike.svg
fig_p2_density_curve_pathgood.svg
fig_p2_wilson_ci_by_panel.svg
fig_p3_future_value_curve_by_group.svg
fig_p3_risk_curve_by_group.svg
fig_p4_proxy_precision_vs_cost.svg
fig_p4_proxy_V_vs_longrisk.svg
fig_p4_proxy_cost_breakdown.svg
fig_p4_proxy_score_vs_AUV_scatter.svg
fig_p4_proxy_leaveout_drop.svg
fig_p6_generated_gate_decision.svg
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| `base_acc_sentinel_v9870.csv` | `736d3e8790a4b31bb78de731947c9e5cdad075ce60acc04904d2b85942240af8` |
| `contract_audit_v9870.csv` | `54c69f9829cf59cf42208dc44d6e126d8ae4764a1d42d1dbcba53f1b4a0d05f3` |
| `failure_taxonomy_v9870.csv` | `08c73c09a7f0ebe51c7d99bc43f6a32e706f86235dd464b2a4257827223959d9` |
| `field_legality_ledger_v9870.csv` | `d9dfe043c3d1f04f5dfa0d7d740a42f585b33f740745b82bbeb7662cc6233985` |
| `no_fake_audit_v9870.csv` | `524443dda7b33dc9713d082711f839e4f09a8ec318d317142fa2ecc52b4957cf` |
| `p0_boundary_reproduction_v9870.csv` | `d629a66c8ec406d1370f12a892d2922599dc06302ea468fd33ebf089a780f60c` |
| `p0_landed_future_row_reference_v9870.csv` | `1d57eb4c05aa9dd64fcefa54a192a516eedbd913e74f8dfeaecb4ba346aa6dc5` |
| `p1_natural_materializer_completion_v9870.csv` | `0b3965009274e5ae14e8ae9eb5b4d646763cf390e4f02dc964ef8cdc12d877c6` |
| `p1_natural_materializer_preflight_v9870.csv` | `ef17e7b86636a9ea4f54c4fc440a36cce1b52e453734e8b613c9db22e95e1205` |
| `p1_natural_stream_panel_10000_v9870.csv` | `7eadc4c2da4d25f6ea5ee3124a7a09c290302b4a2e243bfe79a8c833bb1b8c9a` |
| `p1_natural_stream_panel_20000_v9870.csv` | `838030299ebd88e8a10cf00e1c8121e5d96e5933b598970e313cbd0eb56fec1e` |
| `p1_natural_stream_panel_5000_v9870.csv` | `1643825e5b6b1f273f68246de8fe22cbc6430243d6918c65497fd12d0f3d8594` |
| `p2_density_curve_v9870.csv` | `74489e62bcc7bd9cd3ca8ada93246d3383a74790f7614f1f6730380ff3555ee1` |
| `p2_pathgood_label_audit_v9870.csv` | `bb60b08bd863964bc5b3273ddb6f9fa0e04f04596fffee91684db543f2e0d784` |
| `p3_future_path_mechanism_v9870.csv` | `b6da0e589c915a6ca10e3cd33e715ed0391215dfe74a5597a827ce73402ed63b` |
| `p4_low_cost_proxy_v9870.csv` | `f247b7d3f0e0fa4fec5ef0bdd2e4b93b22869625c153aea0fa66f283e81d2356` |
| `p5_controller_candidates_v9870.csv` | `704185bba9afde4d72b64dffabc3ba83cf6cb0bd1fa3ed52be583d673967e145` |
| `p6_generated_reopen_decision_v9870.csv` | `92df27c33d5f2bbdb68ffa5e0e884ae6f07cdab7cf171541f365b7782e6f1425` |
| `p7_runtime_boundary_v9870.csv` | `f20ed57c6a826125352853e39657e434b6084f8cc7898ea24d1cbd4a2a81441d` |
| `p8_paired_replay_boundary_v9870.csv` | `8c9d76fd2f40f8d030fb3745bf2d540b930c0eb4ee3b28d871fd8a1d770f9ffd` |
| `plan` | `84907041ab6b1b5c7f3faa3c6eb70461545bd8a305c7f45d6d6f4a6bf4459bfd` |
| `route_decision_v9870.json` | `a5b449fc8a4611661c4ea7f35977025f499614043e6a10797ddc867f7ecb204a` |
| `run_manifest_v9870.json` | `77677837101c31cdc772af3f95de2b7f7911f68e65f53c3e5667e466cdda5ef4` |
| `runner` | `1abc11ccb9dc717bc0ff7f3844e60bb42fe3031c3639e7e62caeaae7bd93438d` |

## 11. 最终分析结论

```text
1. P1 确认 existing AP0 preflight materializer 可以真实跑，但这不是自然扩流 action generator。
2. 5000/10000/20000 panel 因缺少新增自然动作生成入口继续 not_run，density 不能裁决。
3. A 线 future path 仍有 weak signal，但 strong mechanism 仍依赖合法低成本 proxy。
4. B/P4 低成本 train-time proxy 仍未找到 controller-ready region。
5. Controller/generated/runtime/replay 全部 gate-blocked，strict PureKAN functional 仍未成功。
```

最终一句话：

> v9.8.7 真实执行后停在 `CaseE-MaterializerExtensionGeneratorMissing`：本轮把自然流的 existing-AP0 preflight 真正跑了出来，但 5000/10000/20000 需要的新自然动作扩流入口仍未落地，低成本合法 proxy 也未过 gate，因此不能进入 official controller 或 generated route。
