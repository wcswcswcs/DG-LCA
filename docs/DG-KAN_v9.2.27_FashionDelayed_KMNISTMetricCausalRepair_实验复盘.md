# DG-KAN v9.2.27 Fashion Delayed 与 KMNIST Metric-Causal Repair 实验复盘

> 本复盘记录 `DG-KAN_v9.2.27_完整重审_FashionDelayed_KMNISTMetricCausalRepair_实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 P5-P9 写成通过。

## 0. 最新结论

```text
route = R4-KMNISTRepairOnlyFashionStillPartial
base_candidate = LQ-t2-h256
success_v9227_fashion_survivor = False
success_v9227_kmnist_metric_causal = True
success_v9227_strict_purekan_functional = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9227_fashion_kmnist_metric_causal_repair_fullscope_20260510T173000Z/
```

核心结论：

1. P0 复现 v9.2.26 boundary：source route = `R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker`，fake/proxy = `0`。
2. Fashion local survivor count = `0`，best = `F7-Fashion-O2-BranchBand`，beats AdamWParallel = `0.26666666666666666`，beats best LR = `0.6`。
3. KMNIST target survivor count = `6`，best target = `K6-KMNIST-MetricCausalTarget`，actual effect rate = `0.76`。
4. KMNIST primitive survivor count = `3`，best primitive = `N2a-TinyInit-RationalFunc-BranchRatioCap`，actual effect rate = `0.7037037037037037`。
5. 当前 blocker：`fashion_delayed_signal_not_stable`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9227_fashion_kmnist_metric_causal_repair.py` | v9.2.27 runner；生成 P0-P9 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9227_fashion_kmnist_metric_causal_repair.py
```

正式运行：

```bash
python experiments/run_v9227_fashion_kmnist_metric_causal_repair.py \
  --out-dir results/real_rerun_20260506/v9227_fashion_kmnist_metric_causal_repair_fullscope_20260510T173000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --p1-seeds 0,1,2,3,4,5,6,7,8,9 \
  --p1-horizons 20,50,80,160,240,640 \
  --p3-seeds 0,1,2,3,4,5,6,7,8,9
```

实际测量范围：

```text
P1 Fashion:
  seeds = 0..9
  horizons = 20,50,80,160,240,640
  measured controllers = F0,F3,F6,F7,F8,F12
  not_implemented controllers = F1,F2,F4,F5,F9,F10,F11

P3 KMNIST:
  seeds = 0..9
  horizons = 1,5,20,80,240
  measured targets = K1,K2,K3,K5,K6,K8
  K0/K4/K7 未在本 runner 中实现，不能写成成功或失败

P4 KMNIST:
  seeds = 0,1,2
  horizons = 20,80,240
  measured primitives = N2a,N2b,N2c,N3c
  计划中的 P0-P10 完整 primitive factory 未全部实现，不能外推为全 primitive space 结论
```

## 2. Route

```json
{
  "base_candidate": "LQ-t2-h256",
  "fashion_best_CEp99_delta": -0.05602325598398845,
  "fashion_best_beats_adamwparallel": 0.26666666666666666,
  "fashion_best_beats_best_lr": 0.6,
  "fashion_best_candidate": "F7-Fashion-O2-BranchBand",
  "fashion_best_margin_delta": 0.00831067015727361,
  "fashion_best_task_safe": 0.9333333333333333,
  "fashion_local_survivor": 0,
  "fashion_local_survivor_count": 0,
  "fashion_mechanism_pass": 1,
  "kmnist_best_CEp99_delta": 7.152557373046875e-07,
  "kmnist_best_actual_effect_rate": 0.76,
  "kmnist_best_margin_delta": -2.543926239013672e-06,
  "kmnist_best_primitive": "N2a-TinyInit-RationalFunc-BranchRatioCap",
  "kmnist_best_primitive_CEp99_delta": 6.18122242115162e-07,
  "kmnist_best_primitive_actual_effect_rate": 0.7037037037037037,
  "kmnist_best_primitive_margin_delta": -4.3268556948061344e-07,
  "kmnist_best_target": "K6-KMNIST-MetricCausalTarget",
  "kmnist_failure_type": "target_survivor_found",
  "kmnist_metric_causal_target_survivor": 1,
  "kmnist_primitive_survivor": 1,
  "kmnist_primitive_survivor_count": 3,
  "kmnist_target_survivor_count": 6,
  "next_required_implementation": "redesign_fashion_delayed_controller",
  "primary_blocker": "fashion_delayed_signal_not_stable",
  "route": "R4-KMNISTRepairOnlyFashionStillPartial",
  "source_route": "R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker",
  "success_v9227_external_ready": 0,
  "success_v9227_fashion_survivor": 0,
  "success_v9227_kmnist_metric_causal": 1,
  "success_v9227_strict_purekan_functional": 0
}
```

## 3. P1 Fashion delayed verification

Artifact：`p1_fashion_delayed_signal_verification.csv` / `p1_fashion_delayed_signal_summary.csv`

| candidate | rows | beats AdamWParallel | beats best LR | task safe | CEp99 delta | margin delta | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0-Fashion-NoFunctional | `60` | `0.266667` | `0.516667` | `1.000000` | `0.000000` | `0.000000` | `0` |
| F12-Fashion-O2-MarginTailHybrid | `60` | `0.216667` | `0.533333` | `0.900000` | `0.019438` | `0.013209` | `0` |
| F3-Fashion-O2-h80 | `60` | `0.200000` | `0.500000` | `0.900000` | `0.037839` | `0.008777` | `0` |
| F6-Fashion-O1O2-BalancedDelayed | `60` | `0.233333` | `0.516667` | `0.850000` | `0.007232` | `0.000287` | `0` |
| F7-Fashion-O2-BranchBand | `60` | `0.266667` | `0.600000` | `0.933333` | `-0.056023` | `0.008311` | `0` |
| F8-Fashion-O2-DerivativeBand | `60` | `0.250000` | `0.533333` | `0.866667` | `0.019712` | `0.006633` | `0` |

未实现 rows：

```text
F1-Fashion-O2-h20 = not_implemented
F2-Fashion-O2-h50 = not_implemented
F4-Fashion-O2-h160 = not_implemented
F5-Fashion-O2-h240 = not_implemented
F9-Fashion-O2-AbstainUnlessControlBeat = not_implemented
F10-Fashion-O2-MultiHorizonVote = not_implemented
F11-Fashion-O2-TailOnlyEvent = not_implemented
```

判断：

1. Fashion 的 best row 是 `F7-Fashion-O2-BranchBand`，它有 CEp99 改善 `-0.056023` 和 margin 改善 `+0.008311`。
2. 但 strong-control beat rate 远低于 gate：AdamWParallel `0.266667`，best LR `0.600000`，没有达到 `>=0.75`。
3. 因此 Fashion delayed signal 仍是 partial diagnostic，不允许打开 routed controller。

## 4. P2 Fashion mechanism diagnosis

Artifact：`p2_fashion_mechanism_diagnosis.csv`

| candidate | corr horizon vs -CEp99 | corr horizon vs margin | corr branch vs -CEp99 | corr derivative vs margin | delayed alignment |
|---|---:|---:|---:|---:|---:|
| F0-Fashion-NoFunctional | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| F12-Fashion-O2-MarginTailHybrid | `0.017962` | `0.164931` | `-0.293021` | `-0.118885` | `0` |
| F3-Fashion-O2-h80 | `-0.133618` | `0.075033` | `-0.241717` | `-0.057656` | `0` |
| F6-Fashion-O1O2-BalancedDelayed | `0.197452` | `0.057295` | `-0.218069` | `-0.136678` | `0` |
| F7-Fashion-O2-BranchBand | `0.472709` | `0.137155` | `-0.082480` | `0.000020` | `1` |
| F8-Fashion-O2-DerivativeBand | `0.001046` | `0.052357` | `-0.183935` | `-0.120182` | `0` |

判断：`F7` 的 horizon vs `-CEp99` 相关性达到 `0.472709`，说明 delayed alignment 有局部证据；但 branch/derivative 机制不成立，且 P1 strong-control beat rate 远不过 gate，所以不能升级为 Fashion survivor。

## 5. P3 KMNIST target-to-effect diagnosis

Artifact：`p3_kmnist_target_to_realized_effect.csv` / `p3_kmnist_target_summary.csv`

| target | rows | oracle useful | safe fit pass | actual effect | silent | misaligned | CEp99 delta | margin delta | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K1-KMNIST-O2-MarginTail | `50` | `1.000000` | `1.000000` | `0.700000` | `0.000000` | `0.300000` | `0.000005` | `-0.000002` | `1` |
| K2-KMNIST-O1-CEp99Tail | `50` | `1.000000` | `1.000000` | `0.680000` | `0.000000` | `0.320000` | `-0.000027` | `0.000002` | `1` |
| K3-KMNIST-O6-HardMode | `50` | `1.000000` | `1.000000` | `0.660000` | `0.000000` | `0.340000` | `0.000000` | `-0.000000` | `1` |
| K5-KMNIST-CurvatureTail | `50` | `0.000000` | `1.000000` | `0.620000` | `0.000000` | `0.380000` | `0.000004` | `0.000000` | `1` |
| K6-KMNIST-MetricCausalTarget | `50` | `1.000000` | `1.000000` | `0.760000` | `0.000000` | `0.240000` | `0.000001` | `-0.000003` | `1` |
| K8-KMNIST-ClassModeBucketTarget | `50` | `1.000000` | `1.000000` | `0.600000` | `0.000000` | `0.400000` | `0.000009` | `-0.000002` | `1` |

判断：

1. 和 v9.2.26 不同，本轮 KMNIST 不再是 `actual effect pass rate = 0`。
2. `K6-KMNIST-MetricCausalTarget` 的 actual effect rate 达到 `0.760000`，说明 metric-causal target selection 能修复一部分 realized-effect 断点。
3. 但 mean CEp99 / margin delta 的绝对量仍很小，且 P3 只是 target-to-effect diagnosis，不是 full strict functional success。

## 6. P4 KMNIST primitive repair

Artifact：`p4_kmnist_primitive_repair.csv` / `p4_kmnist_primitive_summary.csv`

| primitive | rows | P4 | P5 near | actual effect | task safe | CEp99 delta | margin delta | survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N2a-TinyInit-RationalFunc-BranchRatioCap | `27` | `1` | `1` | `0.703704` | `1.000000` | `0.000001` | `-0.000000` | `1` |
| N2b-TinyInit-PiecewiseFunc-BranchRatioCap | `27` | `0` | `0` | `0.666667` | `1.000000` | `-0.000065` | `-0.000026` | `0` |
| N2c-TinyInit-SharedRBFFunc-BranchRatioCap | `27` | `1` | `1` | `0.666667` | `1.000000` | `0.000006` | `-0.000005` | `1` |
| N3c-SharedRBFFunc-DerivativeBand | `27` | `1` | `1` | `0.518519` | `1.000000` | `0.000015` | `-0.000066` | `1` |

判断：

1. N2a/N2c/N3c 在本轮 measured primitive subset 中都通过了 P4/P5/actual-effect gate。
2. N2b 有 actual-effect signal，但 P4/P5 source gate 未过，所以不能作为 route survivor。
3. 这说明 KMNIST 的 blocker 从 “target fit 不能转 metric” 推进到了 “需要把 survivor 放进 routed/short/full validation”。

## 7. Downstream boundary

P5-P9 只有在 Fashion survivor 与 KMNIST target/primitive survivor 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 8. No-fake audit

```text
rows_checked = 4765
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

关键 hash：

| artifact | SHA256 |
|---|---|
| `experiments/run_v9227_fashion_kmnist_metric_causal_repair.py` | `bc4b54115290ba57e5ae6bc718a1123c900a4194b50e420d532ffc36fdc75f7d` |
| route | `9433a9fbf5a8c8e7b81b56c0dfccf11347d69324dc555e5c850334dcd5a50f2f` |
| P1 Fashion verification | `36dc79f2d82222df6cd3f4d2dfa4f03fc2bc791c14308cf6d368fdfeca270213` |
| P1 Fashion summary | `782697bd8105417dc1aab4a15caf6a096b2412e07b0885e00cb309fa6629da0e` |
| P2 Fashion mechanism | `3afda5518e8a91c6c3c43bab4691a726ffd8f76713b6cc9b3fd29327ba231d2e` |
| P3 KMNIST target summary | `191119dee193203b58f63b755cdeacc99eb6cd971e533b79354adb8ebd4a1436` |
| P4 KMNIST primitive summary | `e511dc7655305d2ca28a2a649f10431ce981eb8efcacfaf7e6994e3dbf04bd28` |
| provenance audit | `362ff8cb80815a1c3684ef10a12f8c8664e8c75c7668232cb6e14fb6246eda06` |

## 9. 最终分析结论

v9.2.27 的真实推进是：

```text
Fashion delayed signal 经过更多 horizon/controller/seed 重审；
KMNIST 从 target-to-effect 与 primitive/interface 两层重审 metric-causal repair。
```

机制判断：

1. Fashion 的 h80/h50 局部信号没有扩展成 10-seed、多 horizon 的 stable survivor；即使 `F7` 有 CEp99 与 margin 改善，beat-rate 和 alignment 都不过 gate。
2. KMNIST 的结论发生了推进：v9.2.26 的 `actual effect pass rate = 0` 被本轮 K1/K2/K3/K5/K6/K8 target subset 修复，best `K6` 达到 `0.76`。
3. primitive repair 也有推进：N2a/N2c/N3c 在 measured subset 中保留 P4/P5 且 actual-effect rate 超过 `0.50`。
4. 但 strict PureKAN functional 仍未成功，因为 Fashion route 是当前 macro blocker，且 P5-P9 还没有打开。
5. 下一步不应继续把 KMNIST 当作主要 failure；应优先重做 Fashion event/controller，或者允许 dataset-routed route 中 Fashion abstain，但这需要单独计划和验证。

最终一句话：

> v9.2.27 真实执行后停在 `R4-KMNISTRepairOnlyFashionStillPartial`：`fashion_delayed_signal_not_stable`。
