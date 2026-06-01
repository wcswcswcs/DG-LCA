# DG-KAN v9.2.58 Risk-Support Sufficient Statistics 与 Exact-Signal Accept-Region Geometry 实验复盘

> 本复盘记录 `DG-KAN_v9.2.58_RiskSupportSufficientStatistics_ExactSignalAcceptRegion_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R10-ExactReferenceStillInfeasible
base_candidate = LQ-t2-h256
success_v9258_strict_purekan_functional = False
success_v9258_full_functional = False
success_v9258_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9258_risk_support_sufficient_statistics_exact_signal_accept_region_first_20260512T113000Z/
```

核心结论：

1. P0 复现 v9.2.57 boundary：source route = `R11-ExactSignalControllerInfeasible`，true-delta legality / predictivity / agreement pass 均保留，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；正式运行中 GPU 采样有活动：`NVIDIA L4, 31 %, 324-326 MiB`。
3. P1 exact-reference failure autopsy pass = `1`：false positive `732`，false negative `905`，accepted bad-event `560`，三类 attribution fraction 均为 `1.0`。
4. P2 risk sufficient statistics 有预测性：best = `RISK5-RiskLCB`，bad-event AUC = `0.809103`，corr = `0.563818`，risk_stat_pass = `1`；但 risk gate 只能得到 tiny/empty clean slice，utility pass = `0`。
5. P3 support sufficient statistics 未达到 official gate：best = `SUP2-FamilyReliabilityV2`，legal-oracle Jaccard = `0.506509`，diagnostic pass = `1`；但 precision = `0.572193`、bad-event = `0.364973`，support_stat_pass = `0`。
6. P4 exact reference feasibility v2 未恢复：best = `RISK6-HybridRiskSufficientStatistic + SUP0-ReferenceSupport`，precision = `1.0`、bad-event = `0.0`，但 coverage = `0.000992 < 0.03`，accepted strata = `1`，family = `3`。
7. P5 true-delta compute v3 仍未过：best true route = `TBD0-V9256CBD0Reference`，AUC = `0.888401`，agreement = `1.0`，step ratio = `2.863280 > 1.50`。
8. P6 oracle support 仍通过：precision = `1.0`，coverage = `0.120040`，bad-event = `0.0`；但 support measurement 仍失败：signal strata = `3`，balanced diagnostic rows = `36`。
9. P7-P10 因 `P4_exact_reference_still_infeasible` / `P4_exact_reference_infeasible` gate-blocked，均以 `not_run` 落盘。
10. 当前 blocker：`exact_reference_still_infeasible`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9258_risk_support_sufficient_statistics_exact_signal_accept_region.py` | v9.2.58 runner；执行 v9.2.57 boundary 复现、exact-reference failure autopsy、risk/support sufficient statistics factory、exact reference feasibility v2、true-delta compute v3、support/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9258_risk_support_sufficient_statistics_exact_signal_accept_region.py
```

正式运行：

```bash
python experiments/run_v9258_risk_support_sufficient_statistics_exact_signal_accept_region.py \
  --out-dir results/real_rerun_20260506/v9258_risk_support_sufficient_statistics_exact_signal_accept_region_first_20260512T113000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 168
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T11:30:03Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R10-ExactReferenceStillInfeasible",
  "base_candidate": "LQ-t2-h256",
  "v9257_boundary_pass": 1,
  "reference_failure_autopsy_pass": 1,
  "fp_attribution_fraction": 1.0,
  "fn_attribution_fraction": 1.0,
  "bad_event_attribution_fraction": 1.0,
  "best_risk_stat_id": "RISK5-RiskLCB",
  "risk_stat_pass": 1,
  "risk_stat_auc_bad_event": 0.8091028819692894,
  "risk_gate_bad_event": 0.0,
  "best_support_stat_id": "SUP2-FamilyReliabilityV2",
  "support_stat_pass": 0,
  "support_diagnostic_pass": 1,
  "legal_oracle_jaccard": 0.022857142857142857,
  "accepted_family_count": 3,
  "accepted_signal_strata_count": 1,
  "max_family_share": 0.6666666666666666,
  "exact_reference_feasible": 0,
  "reference_controller_precision": 1.0,
  "reference_controller_coverage": 0.000992063492063492,
  "reference_controller_bad_event": 0.0,
  "best_true_delta_id": "TBD0-V9256CBD0Reference",
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8884008136827201,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "oracle_support_pass": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.12003968253968254,
  "oracle_bad_event": 0.0,
  "support_measurement_pass": 0,
  "primary_blocker": "exact_reference_still_infeasible",
  "next_required_implementation": "reset_risk_support_target_decomposition"
}
```

判断：P2/P3 证明 risk/support statistics 不是完全无信号，但 P4 仍只能筛出 coverage 极低的 clean slice，无法形成 deployable accept region。

## 3. P1 exact reference failure autopsy

Artifacts：

```text
p1_exact_reference_controller_failure_autopsy.csv
```

Summary：

```text
accepted_count = 1299
false_positive_count = 732
false_negative_count = 905
coverage_loss_count = 889
bad_event_accepted_count = 560
false_positive_primary_mode = FPA5-delta_gain_mismatch
false_negative_primary_mode = FNA4-family_scarcity
bad_event_primary_mode = FPA6-tail_instability
false_positive_attribution_fraction = 1.0
false_negative_attribution_fraction = 1.0
bad_event_attribution_fraction = 1.0
reference_failure_autopsy_pass = 1
```

判断：v9.2.57 的 exact-reference failure 可以归因，不是没有 trace。主要冲突是 delta-gain mismatch、tail instability 和 family scarcity 的组合。

## 4. P2 risk sufficient statistics factory

Artifacts：

```text
p2_risk_sufficient_statistics_factory.csv
risk_stat_trace_v9258.csv
```

| risk stat | AUC bad-event | corr | precision | coverage | bad-event | stat pass | utility |
|---|---:|---:|---:|---:|---:|---:|---:|
| RISK0-ReferenceCurrentRisk | `0.612479` | `0.198931` | `1.000000` | `0.000083` | `0.000000` | `0` | `0` |
| RISK1-TailInstabilityLCB | `0.577902` | `0.117029` | `0.000000` | `0.000000` | `0.000000` | `0` | `0` |
| RISK2-BranchDisagreementRisk | `0.629090` | `0.008760` | `0.000000` | `0.000000` | `0.000000` | `0` | `0` |
| RISK3-ControlDominanceRisk | `0.612337` | `0.198618` | `1.000000` | `0.000165` | `0.000000` | `0` | `0` |
| RISK4-DeltaGainMismatch | `0.489158` | `-0.016766` | `0.470862` | `0.035466` | `0.519814` | `0` | `0` |
| RISK5-RiskLCB | `0.809103` | `0.563818` | `0.000000` | `0.000000` | `0.000000` | `1` | `0` |
| RISK6-HybridRiskSufficientStatistic | `0.725817` | `0.388906` | `1.000000` | `0.000992` | `0.000000` | `1` | `0` |

判断：bad-event 有 legal risk signature，`RISK5/RISK6` 过 predictivity gate；但这些 gate 太保守，coverage 远低于 `0.03`，不能直接作为 controller success。

## 5. P3 support sufficient statistics factory

Artifacts：

```text
p3_support_sufficient_statistics_factory.csv
support_stat_trace_v9258.csv
```

| support stat | precision | coverage | bad-event | jaccard | families | strata | max share | pass | diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUP0-ReferenceSupport | `0.389886` | `0.101356` | `0.528548` | `0.375491` | `28` | `2` | `0.140294` | `0` | `0` |
| SUP1-KNNFeatureDensity | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` | `0` | `0.000000` | `0` | `0` |
| SUP2-FamilyReliabilityV2 | `0.572193` | `0.061839` | `0.364973` | `0.506509` | `9` | `2` | `0.229947` | `0` | `1` |
| SUP3-LeaveFamilyOutReliability | `0.579130` | `0.047536` | `0.356522` | `0.434159` | `7` | `2` | `0.299130` | `0` | `0` |
| SUP4-BalancedSupportGate | `0.398156` | `0.098628` | `0.520536` | `0.382140` | `15` | `2` | `0.144174` | `0` | `0` |
| SUP5-SupportRiskJointPocket | `1.000000` | `0.000083` | `0.000000` | `0.001905` | `1` | `1` | `1.000000` | `0` | `0` |

判断：`SUP2` 对 oracle/legal overlap 有诊断价值，但 bad-event 仍高；`SUP5` 能找到极干净 row，但 coverage/family/strata 都太小。

## 6. P4 exact reference feasibility v2

Artifacts：

```text
p4_exact_reference_feasibility_v2.csv
reference_feasibility_trace_v9258.csv
accepted_region_geometry_trace_v9258.csv
```

Best summary：

```text
best_reference_controller_id = C-RISK6-HybridRiskSufficientStatistic+SUP0-ReferenceSupport
best_risk_stat_id = RISK6-HybridRiskSufficientStatistic
best_support_stat_id = SUP0-ReferenceSupport
exact_reference_feasible = 0
precision = 1.0
coverage = 0.000992063492063492
bad_event = 0.0
accepted_signal_strata_count = 1
accepted_family_count = 3
max_family_share = 0.6666666666666666
legal_oracle_jaccard = 0.022857142857142857
```

判断：P4 是本轮 terminal blocker。新 risk/support stats 能把 bad-event 压到 `0`，但代价是 coverage 几乎清零；accepted region 不是可部署的 safe coverage-preserving region。

## 7. P5 true-delta compute v3 parallel lane

Artifacts：

```text
p5_true_delta_compute_v3_parallel_lane.csv
true_delta_compute_trace_v9258.csv
```

| custom delta | true | source gap | formula | AUC | agreement | step q90 | compute pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| TBD0-V9256CBD0Reference | `1` | `0` | `0` | `0.888401` | `1.000000` | `2.863280` | `0` |
| TBD1-LayoutRepairedFullLogitSmallC-event-major | `1` | `0` | `0` | `0.888401` | `1.000000` | `3.580767` | `0` |
| TBD2-SelectedLogitExactV2 | `1` | `0` | `0` | `0.589056` | `0.448661` | `1.480000` | `0` |
| TBD3-BranchDeltaLogitsPlusK7dFused | `1` | `0` | `0` | `0.888401` | `1.000000` | `3.264817` | `0` |
| TBD6-HybridBestLayoutTrueDelta | `1` | `0` | `0` | `0.888401` | `1.000000` | `3.054184` | `0` |
| PX1-FormulaProxyNegativeControl | `0` | `0` | `1` | `0.589056` | `0.448661` | `1.350000` | `0` |
| PX2-SourceMeasuredGapInput | `0` | `1` | `0` | `0.888401` | `1.000000` | `1.020000` | `0` |

判断：compute lane 没有恢复。便宜的 `PX2` 仍是 source-measured gap proxy，不能转正；true delta exact route 仍超 system envelope。

## 8. P6/P7 downstream boundary

P6 support：

```text
natural_real_event_count = 12096
balanced_diagnostic_real_event_count = 36
measured_signal_strata_count = 3
measured_family_count = 45
support_measurement_pass = 0
accepted_signal_strata_count = 3
accepted_family_count = 3
max_family_share = 0.727273
oracle_support_pass = 1
oracle_precision = 1.0
oracle_coverage = 0.12003968253968254
oracle_bad_event = 0.0
```

P7-P10 均落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P4_exact_reference_infeasible` |
| `p8_leave_dataset_and_stratum_out.csv` | `P4_exact_reference_still_infeasible` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |

没有把 risk/support diagnostic、clean-but-tiny slice、source-gap proxy 或 oracle support 写成 LDO/LSO、paired replay、short-run 或 functional success。

## 9. No-fake audit

```text
rows_checked = 26039
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `26d02d5144bbadb0d4b9742f3ee1d83bd54c868bdcce02b54abfc0a02a60f6d3` |
| runner | `281608a22a31d13f9189f306925e50377110d365d2e5c1f625ef6f5a3cf2b83e` |
| run manifest | `072690e9c00fded56c0703aa118de363732193db0f1c865bacb9e64ccf0bca59` |
| route | `f8259c814b07f9af0892f73d387059e391aafc53a46522339ec9047f8df41bd4` |
| P0 boundary | `f1a715b0c671536fe499a93d2bf1b0bb6ebe74128d0d31775e83042a4d42226e` |
| P1 autopsy | `923fd1c73f1d825a7f7d045485053d8ac52836dbba689354c2173bbea9ecd333` |
| P2 risk factory | `bbc67f25e8a2c8ea371f135b550d11aab4d409085c76895866b0849e66b00a57` |
| P3 support factory | `6c43d146352e5381256ddb1841f292f44ed145cd258ba275cb71d6e62c594f14` |
| P4 reference feasibility | `fe4015b9681bc2f504cc677b4d1a1ef37911a89d45e74d1a5d1b19fa7a89b3e1` |
| P5 compute lane | `bd44479e42ee079ab9edc5974b1912ff15e8f185b39a6a96f881c708313ae6f1` |
| P6 support expansion | `7529d46cfdd2eae5109ef0474e45dc9372474592082a98675f7695216dfd966f` |
| P7 system controller | `6a881b23d4fa01cf4e94c0d4ea4d2fa470ea73799257ca0b98a2e693351099b7` |
| P8 boundary | `cf8c9981722247d15c24a0e6a997b0c5a4dfa18b4296ec7bfe7818a9ccbec537` |
| P9 boundary | `878f6c65fce01d1967b8bfc98f2769a13217ceb3bde13a75ccd37cb45f65cf1e` |
| P10 boundary | `a32abb2e8f1d5b7294fa061e2b063c25c00f9ffeb2c3d6cf5896d0335fec4905` |
| failure table | `471542a8efacc35b4c95970a24299a35f2821a31b1c71c895c8c1b41f74f3d76` |
| provenance audit | `3c33c545df434131f484c82f7594bcb213342540756b26424443a03f9959f495` |

## 11. 最终分析结论

v9.2.58 的真实推进是：

```text
v9.2.57: exact signal strong, but exact reference controller infeasible.
v9.2.58: failure autopsy closed; risk/support stats show diagnostic signal;
          but exact reference feasibility still does not recover.
```

机制判断：

1. H1 部分成立：bad-event 确实有 legal risk signature，`RISK5` AUC 达到 `0.809103`。
2. H2 不足以闭合：risk gate 能压 bad-event，但会把 coverage 压到 `0` 或 `0.000992` 量级。
3. H3 部分成立：`SUP2` 的 legal-oracle Jaccard 达到 `0.506509`，但 precision 和 bad-event 不够。
4. H4 仍是风险：support measurement 还是只有 `3` 个 signal strata，balanced diagnostic rows 只有 `36`。
5. H5 未闭合：true-delta compute 仍未进入 system envelope。
6. 本轮最重要的结论是：exact branch-delta value signal 依然只是 diagnostic；现有 risk/support sufficient statistics 还不能定义可部署 accept region。

最终一句话：

> v9.2.58 真实执行后停在 `R10-ExactReferenceStillInfeasible`：risk/support statistics 有诊断信号，但只能得到 clean-too-tiny slice，exact reference accept-region 仍不可部署，strict PureKAN functional 仍未成功。
