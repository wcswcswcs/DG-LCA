# DG-KAN v9.8.5 Four-Line Future Operator / Natural Stream Route Decision 实验复盘

> 本复盘记录 `DG-KAN_v9.8.5_结果解读_四线进展_未来算子与自然扩流实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.4 已真实 materialized 的 branch-horizon rows；没有 fake data、proxy rows，也没有把 future-path diagnostic、legal-probe diagnostic、natural stream missing 或 generated not_run 写成 official controller pass。

## 0. 最新结论

```text
route = Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped
primary_blocker = legal_virtual_probe_failed
secondary_blocker = natural_stream_materializer_missing
route_recommendation = pivot_or_engineering_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_B_C_D_conditions_not_met
```

最终 artifact：`results/real_rerun_20260506/v9850_four_line_future_operator_natural_stream_decision_full_20260516T180000Z`

核心结论：

1. P0 复现 v9.8.4 boundary：source route = `R1-FuturePathMechanismFoundLegalMechanismAbsent`，system pass = `0`。
2. A 线读取真实 landed future rows：realfunctional rows = `1695`，actions = `339`，A mechanism/strong/controller = `1` / `0` / `0`。
3. A 线 best path type = `Core77`；strong fail reason = `no_path_type_meets_full_count_value_risk_LDO_LSO_LTO_gate`。
4. B 线 legal virtual probe 全部失败：weak/strong = `0` / `0`，best probe = `B4_hard_tail_memory_veto_probe`，precision = `0.42528735632183906`，V LCB = `-0.23620231861434918`。
5. B 线 StopB triggered = `1`；best longrisk UCB = `0.0`，memory+offdiag UCB = `0.0`。
6. C 线 materializer entrypoint found = `0`；C preflight pass = `0`。
7. C panel completed/not_run = `1` / `3`；PanelA CoreLike LCB = `0.021475240123524635`。
8. D generated reopen allowed = `0`；status = `stopped_B_C_D_conditions_not_met`。
9. No-fake audit：rows checked = `2068`，fake/proxy/cpu = `0` / `0` / `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9850_four_line_future_operator_natural_stream_decision.py` | v9.8.5 runner；读取 v9.8.4/v9.8.2 landed rows，执行 A 路径类型分解、B 合法 virtual probe 评估、C natural stream materializer audit、D generated reopen decision、controller/runtime boundary、dashboard 与 audits。 |

```text
python -m py_compile experiments/run_v9850_four_line_future_operator_natural_stream_decision.py
```

```bash
python experiments/run_v9850_four_line_future_operator_natural_stream_decision.py --out-dir results/real_rerun_20260506/v9850_four_line_future_operator_natural_stream_decision_full_20260516T180000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000
```

说明：v9.8.5 没有重新伪造 branch rows；A/B 使用 v9.8.4 已真实 CUDA replay 的 landed rows，C/D/E 因 gate 未满足显式 not_run。

## 2. Route

```json
{
  "stage": "ROUTE_DECISION_V9850",
  "status": "summary",
  "route": "Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped",
  "primary_blocker": "legal_virtual_probe_failed",
  "secondary_blocker": "natural_stream_materializer_missing",
  "route_recommendation": "pivot_or_engineering_blocked",
  "source_route_v9840": "R1-FuturePathMechanismFoundLegalMechanismAbsent",
  "P0_boundary_pass": 1,
  "A_line_mechanism_pass": 1,
  "A_line_strong_pass": 0,
  "B_weak_pass": 0,
  "B_strong_pass": 0,
  "StopB_triggered": 1,
  "C_preflight_pass": 0,
  "C_weak_pass": 0,
  "C_strong_pass": 0,
  "C_materializer_entrypoint_found": 0,
  "D_generated_reopen_allowed": 0,
  "generated_route_status": "stopped_B_C_D_conditions_not_met",
  "controller_pass": 0,
  "runtime_pass": 0,
  "paired_replay_pass": 0,
  "short_full_pass": 0,
  "system_legal_controller_pass": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

## 3. A 线 Future Path Type

| group | actions | AUV LCB | RiskAdjustedAUV LCB | DelayedGain LCB | V80 LCB | V240 LCB | RiskPath UCB | mem/off UCB | LDO/LSO/LTO | path strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `Core77` | `77` | `2.8296932871691163` | `2.8166481132099395` | `4.53388384224294` | `3.1200217020467886` | `4.750885696264312` | `0.03827572901862327` | `0.0` | `0.4415584415584416`/`0.038961038961038974`/`0.012987012987012991` | `0` |
| `CoreExpansion10` | `10` | `0.6225266023303726` | `0.6225266023303726` | `1.6428470667862045` | `0.5669308435951039` | `1.623630340725588` | `0.0` | `0.0` | `0.0`/`0.0`/`0.0` | `0` |
| `ExactOnly` | `68` | `1.5250480927868013` | `1.5250480927868013` | `2.7310661478788507` | `1.582392387764584` | `2.922090918869295` | `0.0` | `0.0` | `0.0`/`0.0`/`0.0` | `0` |
| `OldOnly` | `10` | `1.0287178879851673` | `1.0287178879851673` | `2.1502249933631594` | `1.1093912874075194` | `2.0678046281088522` | `0.0` | `0.0` | `0.0`/`0.0`/`0.0` | `0` |
| `RandomMatched` | `87` | `1.8284728866377598` | `0.17625930908187082` | `3.382085914484093` | `1.8306221718364828` | `3.5922551893613908` | `0.8485412092119363` | `1.9003191436991538` | `0.0`/`0.0`/`0.0` | `0` |
| `RiskCleanButLowValue` | `87` | `2.16033116127777` | `2.1342641045596613` | `4.0890894687738975` | `2.268809387190667` | `4.269593559964293` | `0.054480608015849245` | `0.0` | `0.0`/`0.0`/`0.0` | `0` |

判断：A 线继续支持 future-path 现象，但没有任何 path type 同时满足 count/value/risk/LDO/LSO/LTO controller gate。

## 4. B 线 Legal Virtual Proxy

| probe | precision | V LCB | AUV LCB | longrisk UCB | memory/offdiag UCB | cost q90 ms | weak | strong | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `B1_current_response` | `0.28735632183908044` | `-0.5579475923979604` | `3.8598471101509695` | `0.3183007985873107` | `0.6625996598031831` | `154.58190022036433` | `0` | `0` | `probe_quality_too_low_or_risk_too_high` |
| `B2_virtual_1_step_probe` | `0.2988505747126437` | `-0.5618363755627628` | `3.8630446716729203` | `0.3442102410706317` | `0.7140721901348608` | `153.79969729110599` | `0` | `0` | `probe_quality_too_low_or_risk_too_high` |
| `B3_virtual_5_step_probe` | `0.3103448275862069` | `-0.5832567647917535` | `4.297504685331786` | `0.3442102410706317` | `0.764896231961305` | `154.58190022036433` | `0` | `0` | `probe_quality_too_low_or_risk_too_high` |
| `B4_hard_tail_memory_veto_probe` | `0.42528735632183906` | `-0.23620231861434918` | `3.525332265373713` | `0.0` | `0.0` | `155.12761380523443` | `0` | `0` | `probe_quality_too_low_or_risk_too_high` |
| `B5_cost_amortized_virtual_probe` | `0.3103448275862069` | `-0.5832567647917535` | `4.297504685331786` | `0.3442102410706317` | `0.764896231961305` | `154.58190022036433` | `0` | `0` | `probe_quality_too_low_or_risk_too_high` |

判断：B 线这次不再是空缺，而是用 train-time landed replay metrics 做了 legal virtual probe 评估；结果触发 StopB：best probe 的 V LCB <= 0，且其他 probe 还同时出现 precision 低或 longrisk/memory/offdiag 偏高。

## 5. C 线 Natural AP0 Stream

| panel | status | target | labeled | expected rows | actual rows | missing labels | CoreLike rate | LCB/UCB | reason |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `Panel-2876` | `panel_row` | `2876` | `2876` | `` | `` | `` | `0.026773296244784424` | `0.021475240123524635/0.033333885489495195` | `` |
| `Panel-5000` | `not_run` | `5000` | `2876` | `150000` | `0` | `2124` | `` | `` | `P3_C1d_preflight_not_passed_materializer_missing` |
| `Panel-10000` | `not_run` | `10000` | `2876` | `300000` | `0` | `7124` | `` | `` | `P3_C1d_preflight_not_passed_materializer_missing` |
| `Panel-20000` | `not_run` | `20000` | `2876` | `600000` | `0` | `17124` | `` | `` | `P3_C1d_preflight_not_passed_materializer_missing` |

判断：C 线仍未完成 natural stream extension；没有 5000/10000/20000 真实 labels，所以不能判定 density sufficient/insufficient。

## 6. D/E Boundary

```text
generated_route_reopen_allowed = 0
generated_route_status = stopped_B_C_D_conditions_not_met
controller/runtime/paired replay/short-full = not_run
```

判断：B strong 不成立，C density insufficient 未被证明，A 也没有合法 virtual objective，因此 generated route 不允许重开。

## 7. Dashboard / Figures

Dashboard：`v9850_dashboard.md`

```text
fig_A1_group_future_path_V_curve.svg
fig_A2_group_future_path_risk_curve.svg
fig_A4_AUV_vs_longrisk_pareto.svg
fig_B1_proxy_vs_true_AUV_scatter.svg
fig_B4_probe_cost_vs_quality.svg
fig_C1_density_curve_panel_size.svg
fig_C2_corelike_rate_ci.svg
fig_D1_generated_quality_pareto.svg
fig_stop_pivot_matrix.svg
```

## 8. No-Fake / Contract / Failure

```text
rows_checked = 2068
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| `base_acc_sentinel_v9850.csv` | `7c2731f3999a73d22bd54c9a9c1ff7a5cccfb6a6ac6e3299b4dbd74c70c3096b` |
| `contract_audit_v9850.csv` | `03038901a51c9fae298916fdd80d42b6104128a13a08a6f6f37a6f6b55e82a29` |
| `failure_taxonomy_v9850.csv` | `4f79c764289e19e1a922959fcfa375157c7bff5107c0d0d26ebcd85a1b746ed9` |
| `field_legality_ledger_v9850.csv` | `dc4018466e170ca0f9ce5608317304a83847620e7e1119d66ec6ef0650e8c30a` |
| `no_fake_audit_v9850.csv` | `1e6be103ed961cdaeac3fae7b4daa99d03eb5bd773cb19f68f39ee3c071806f8` |
| `p0_boundary_reproduction_v9850.csv` | `e1c05b6a35f4b5e5eff6b56cfa702a00ff8ec88cd299a18f92639d5b34703845` |
| `p1_future_path_type_decomposition_v9850.csv` | `444ceda56d3f08240e45361b85062bbc36ba35159b73248571f10311916a997c` |
| `p2_legal_virtual_path_proxy_v9850.csv` | `c93ee42a934707e430fb854fdd217d1616bdc684ccfad6f8a07dff5a684d3009` |
| `p3_natural_ap0_stream_materializer_audit_v9850.csv` | `66b994cde67e7d0e04bd222873dac47a3f0c84b81fee8e7aabc90e9f7bc1473e` |
| `p4_natural_ap0_density_panel_v9850.csv` | `0c89780bb7e56a20156014a10ced5824d595beda79796c590f09563673049dc5` |
| `p5_generated_route_reopen_decision_v9850.csv` | `f0d3dc322637c69ef76f8470100e486d3c6e43a20842a5492236ea485c4cb4ab` |
| `p6_minimal_controller_boundary_v9850.csv` | `fdc6286b9327523d9c44f743d29743950b5a4f6838f164c27ac53dc10a160763` |
| `p7_runtime_boundary_v9850.csv` | `1c568e56cd625c724885e4d36dd1d8b07a733ecd105b79bcc49ec3815af82b3b` |
| `p8_paired_replay_boundary_v9850.csv` | `898d40d41643b2afdf11315b0ddf14dfaf492f40183b9e7780a268fe58e2daab` |
| `p9_short_full_boundary_v9850.csv` | `cd25cc91b62e03c2ad7e76d87540d18b486be9ad44339ff4afe00a6fb1917ea0` |
| `plan` | `8d939b384801ea519fbfb31bfecd9cff8f04efe451aa4b1d11e8c39d846eaeca` |
| `route_decision_v9850.json` | `41d11200f6553e6b562368a2314fd7e01b62452f646944cbacd128e83fc4a05f` |
| `run_manifest_v9850.json` | `3fb434a48c58e10d546ba578b90b40435fff666edc49054dbaccbbdd99c9c63a` |
| `runner` | `cef7ffe7178842a0d2be6141cc5d53dc019474a0fafc3a886f8d1c9e1d2acbdd` |
| `v9850_dashboard.md` | `a4c31a0d7c1c2d94db261840e3867aff2b5fcfe9026b038294ac2d27e1a234ac` |

## 10. 最终分析结论

```text
1. A 线确认 future-path 现象仍存在，但没有 path type 达到 full controller gate。
2. B 线这次真正评估了 legal virtual probe，结果失败并触发 StopB。
3. C 线 natural AP0 stream materializer 仍未落地，因此 density 不能裁决。
4. D 线没有满足重开条件，generated route 正确停止。
5. v9.8.5 的路线裁决偏向 Case4：B 失败、C 工程 blocker、D 停止；下一步要么先工程化 C materializer，要么转向 optimizer-level theory，而不能继续调 red diagnostic。
```

最终一句话：

> v9.8.5 真实执行后停在 `Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped`：future-path 好动作现象仍可信，但合法 train-time virtual probe 不能选出安全好动作，自然扩流 materializer 仍缺失，generated route 没有重开条件，因此不能进入 official controller，strict PureKAN functional 仍未成功。
