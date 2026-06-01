# DG-KAN v9.2.28 Fashion-First Routed Functional 与 KMNIST Survivor Integration 实验复盘

> 本复盘记录 `DG-KAN_v9.2.28_FashionFirst_RoutedFunctional_KMNISTIntegration_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 P5-P8 写成通过。

## 0. 最新结论

```text
route = R5-KMNISTSurvivorNotPreserved
base_candidate = LQ-t2-h256
success_v9228_strict_purekan_functional = False
success_v9228_full_functional = False
success_v9228_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9228_fashionfirst_routed_functional_kmnist_integration_first_20260510T190000Z/
```

核心结论：

1. P0 复现 v9.2.27 boundary：source route = `R4-KMNISTRepairOnlyFashionStillPartial`，fake/proxy = `0`。
2. Fashion best = `F14-Fashion-F7-ControlBeatAbstain`，failure mode = `FashionEffectControlDominated`，stable survivor = `0`，abstain = `0`。
3. KMNIST preserved = `0`，best = `K6-KMNIST-MetricCausalTarget` / `N2a-TinyInit-RationalFunc-BranchRatioCap`。
4. Routed pass = `0`，best routed = `U0-GlobalBestSingle`。
5. 当前 blocker：`kmnist_survivor_not_preserved_under_stronger_replay`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9228_fashionfirst_routed_functional_kmnist_integration.py` | v9.2.28 runner；生成 P0-P8 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9228_fashionfirst_routed_functional_kmnist_integration.py
```

正式运行：

```bash
python experiments/run_v9228_fashionfirst_routed_functional_kmnist_integration.py \
  --out-dir results/real_rerun_20260506/v9228_fashionfirst_routed_functional_kmnist_integration_first_20260510T190000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

实际测量范围：

```text
P1 Fashion:
  seeds = 0..9
  horizons = 20,50,80,160,240,640
  controllers = F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18
  本轮不再保留 v9.2.27 的 F1/F2/F4/F5/F9/F10/F11 not_implemented blocker

P3 KMNIST:
  seeds = 0..9
  horizons = 1,5,20,80,240,640
  targets = K6,K2,K1,K8
  primitives = N2a,N2c,N3c,N2a-OrthogonalTail,N2c-OrthogonalTail
  N2a+N2c = not_implemented，因为跨 primitive 参数空间 hybrid step 未实现

P4 routed:
  datasets = MNIST,Fashion-MNIST,KMNIST
  seeds = 0,1,2
  horizons = 20,80,240,640
  controllers = U0,U1,U2,U3,U4,U5,U6,U7
  includes DatasetRouteShuffled / EventRouteShuffled controls
```

## 2. Route

```json
{
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "best_kmnist_primitive": "N2a-TinyInit-RationalFunc-BranchRatioCap",
  "best_kmnist_target": "K6-KMNIST-MetricCausalTarget",
  "best_routed_beats_adamwparallel": 0.5,
  "best_routed_beats_best_lr": 0.5,
  "best_routed_controller": "U0-GlobalBestSingle",
  "external_ready": 0,
  "fashion_abstain_pass": 0,
  "fashion_best_beats_adamwparallel": 0.2833333333333333,
  "fashion_best_beats_best_lr": 0.5666666666666667,
  "fashion_best_candidate": "F14-Fashion-F7-ControlBeatAbstain",
  "fashion_best_effect_size": -0.016765101750691732,
  "fashion_failure_mode": "FashionEffectControlDominated",
  "fashion_mechanism_pass": 1,
  "fashion_signal_confirmed": 0,
  "full_run_pass": 0,
  "functional_control_pass": 0,
  "functional_kmnist_repair_pass": 0,
  "functional_system_pass": 0,
  "functional_task_safe": 0,
  "kmnist_best_actual_effect_rate": 0.7666666666666667,
  "kmnist_best_beats_adamwparallel": 0.26666666666666666,
  "kmnist_best_beats_best_lr": 0.48333333333333334,
  "kmnist_integration_survivor_count": 0,
  "kmnist_survivor_preserved": 0,
  "next_required_implementation": "return_to_kmnist_effect_magnitude_repair",
  "paired_replay_pass": 0,
  "primary_blocker": "kmnist_survivor_not_preserved_under_stronger_replay",
  "robustness_pass": 0,
  "route": "R5-KMNISTSurvivorNotPreserved",
  "routing_overfit_controls_pass": 0,
  "short_run_pass": 0,
  "strong_baseline_pass": 0,
  "success_v9228_external_ready": 0,
  "success_v9228_full_functional": 0,
  "success_v9228_strict_purekan_functional": 0,
  "unified_routed_controller_pass": 0,
  "v9227_boundary_pass": 1
}
```

## 3. P1 Fashion controller completion

| candidate | rows | beats AdamWParallel | beats best LR | task safe | CEp99 delta | margin delta | survivor | abstain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0-Fashion-NoFunctional | `60` | `0.266667` | `0.516667` | `1.000000` | `0.000000` | `0.000000` | `0` | `0` |
| F1-Fashion-O2-h20 | `60` | `0.216667` | `0.583333` | `0.983333` | `-0.027228` | `-0.000418` | `0` | `0` |
| F10-Fashion-O2-MultiHorizonVote | `60` | `0.216667` | `0.516667` | `0.933333` | `-0.012574` | `-0.001669` | `0` | `0` |
| F11-Fashion-O2-TailOnlyEvent | `60` | `0.233333` | `0.550000` | `0.933333` | `-0.019786` | `-0.004016` | `0` | `0` |
| F12-Fashion-O2-MarginTailHybrid | `60` | `0.216667` | `0.533333` | `0.900000` | `0.019438` | `0.013209` | `0` | `0` |
| F13-Fashion-F7-BranchBandPlusHorizonVote | `60` | `0.250000` | `0.633333` | `0.933333` | `-0.059092` | `0.004543` | `0` | `0` |
| F14-Fashion-F7-ControlBeatAbstain | `60` | `0.283333` | `0.566667` | `0.933333` | `-0.016765` | `-0.012591` | `0` | `0` |
| F15-Fashion-F7-CEAndMarginJoint | `60` | `0.216667` | `0.600000` | `0.966667` | `-0.029219` | `0.006376` | `0` | `0` |
| F16-Fashion-F7-TailOverlapGate | `60` | `0.250000` | `0.566667` | `0.966667` | `-0.053014` | `0.003542` | `0` | `0` |
| F17-Fashion-F7-CurvatureProtected | `60` | `0.250000` | `0.550000` | `0.900000` | `-0.009720` | `-0.008577` | `0` | `0` |
| F18-Fashion-F7-DelayedEnsemble | `60` | `0.183333` | `0.566667` | `0.950000` | `-0.039790` | `0.001028` | `0` | `0` |
| F2-Fashion-O2-h50 | `60` | `0.266667` | `0.600000` | `0.933333` | `-0.056023` | `0.008311` | `0` | `0` |
| F3-Fashion-O2-h80 | `60` | `0.200000` | `0.500000` | `0.900000` | `0.037839` | `0.008777` | `0` | `0` |
| F4-Fashion-O2-h160 | `60` | `0.150000` | `0.450000` | `0.866667` | `0.033003` | `0.001329` | `0` | `0` |
| F5-Fashion-O2-h240 | `60` | `0.250000` | `0.533333` | `0.883333` | `0.006742` | `-0.015194` | `0` | `0` |
| F6-Fashion-O1O2-BalancedDelayed | `60` | `0.233333` | `0.516667` | `0.850000` | `0.007232` | `0.000287` | `0` | `0` |
| F7-Fashion-O2-BranchBand | `60` | `0.266667` | `0.600000` | `0.933333` | `-0.056023` | `0.008311` | `0` | `0` |
| F8-Fashion-O2-DerivativeBand | `60` | `0.250000` | `0.533333` | `0.866667` | `0.019712` | `0.006633` | `0` | `0` |
| F9-Fashion-O2-AbstainUnlessControlBeat | `60` | `0.216667` | `0.533333` | `0.983333` | `-0.003991` | `-0.004612` | `0` | `0` |

## 4. P2 Fashion mechanism attribution

Artifact：`p2_fashion_mechanism_attribution.csv`

代表性 rows：

| candidate | corr horizon vs -CEp99 | corr horizon vs margin | delayed alignment | failure mode |
|---|---:|---:|---:|---|
| F13-Fashion-F7-BranchBandPlusHorizonVote | `0.342913` | `0.136097` | `1` | FashionEffectExistsButControlDominated |
| F16-Fashion-F7-TailOverlapGate | `0.406422` | `0.033867` | `1` | FashionEffectExistsButControlDominated |
| F18-Fashion-F7-DelayedEnsemble | `0.300174` | `0.002641` | `1` | FashionEffectExistsButControlDominated |
| F2-Fashion-O2-h50 | `0.472709` | `0.137155` | `1` | FashionEffectExistsButControlDominated |
| F7-Fashion-O2-BranchBand | `0.472709` | `0.137155` | `1` | FashionEffectExistsButControlDominated |

判断：

1. Fashion 的 delayed alignment 确实存在，不是完全无信号。
2. 但所有代表性 delayed rows 都是 `FashionEffectExistsButControlDominated`，说明控制组仍能解释主要收益。
3. 因此 Fashion failure 不再是 missing controller，而是 event ranking / control superiority 不成立。

## 5. P3 KMNIST survivor integration

| target | primitive | rows | actual effect | beats AdamWParallel | beats best LR | task safe | survivor |
|---|---|---:|---:|---:|---:|---:|---:|
| K1-KMNIST-O2-MarginTail | N2a-OrthogonalTail | `60` | `0.666667` | `0.233333` | `0.483333` | `1.000000` | `0` |
| K1-KMNIST-O2-MarginTail | N2a-TinyInit-RationalFunc-BranchRatioCap | `60` | `0.700000` | `0.216667` | `0.450000` | `1.000000` | `0` |
| K1-KMNIST-O2-MarginTail | N2c-OrthogonalTail | `60` | `0.600000` | `0.233333` | `0.350000` | `1.000000` | `0` |
| K1-KMNIST-O2-MarginTail | N2c-TinyInit-SharedRBFFunc-BranchRatioCap | `60` | `0.600000` | `0.233333` | `0.350000` | `1.000000` | `0` |
| K1-KMNIST-O2-MarginTail | N3c-SharedRBFFunc-DerivativeBand | `60` | `0.683333` | `0.200000` | `0.400000` | `1.000000` | `0` |
| K2-KMNIST-O1-CEp99Tail | N2a-OrthogonalTail | `60` | `0.700000` | `0.216667` | `0.466667` | `1.000000` | `0` |
| K2-KMNIST-O1-CEp99Tail | N2a-TinyInit-RationalFunc-BranchRatioCap | `60` | `0.666667` | `0.216667` | `0.466667` | `1.000000` | `0` |
| K2-KMNIST-O1-CEp99Tail | N2c-OrthogonalTail | `60` | `0.616667` | `0.233333` | `0.383333` | `1.000000` | `0` |
| K2-KMNIST-O1-CEp99Tail | N2c-TinyInit-SharedRBFFunc-BranchRatioCap | `60` | `0.616667` | `0.233333` | `0.383333` | `1.000000` | `0` |
| K2-KMNIST-O1-CEp99Tail | N3c-SharedRBFFunc-DerivativeBand | `60` | `0.633333` | `0.116667` | `0.233333` | `1.000000` | `0` |
| K6-KMNIST-MetricCausalTarget | N2a-OrthogonalTail | `60` | `0.600000` | `0.233333` | `0.483333` | `1.000000` | `0` |
| K6-KMNIST-MetricCausalTarget | N2a-TinyInit-RationalFunc-BranchRatioCap | `60` | `0.766667` | `0.266667` | `0.483333` | `1.000000` | `0` |
| K6-KMNIST-MetricCausalTarget | N2c-OrthogonalTail | `60` | `0.583333` | `0.166667` | `0.383333` | `1.000000` | `0` |
| K6-KMNIST-MetricCausalTarget | N2c-TinyInit-SharedRBFFunc-BranchRatioCap | `60` | `0.583333` | `0.166667` | `0.383333` | `1.000000` | `0` |
| K6-KMNIST-MetricCausalTarget | N3c-SharedRBFFunc-DerivativeBand | `60` | `0.633333` | `0.183333` | `0.350000` | `1.000000` | `0` |
| K8-KMNIST-ClassModeBucketTarget | N2a-OrthogonalTail | `60` | `0.683333` | `0.283333` | `0.516667` | `1.000000` | `0` |
| K8-KMNIST-ClassModeBucketTarget | N2a-TinyInit-RationalFunc-BranchRatioCap | `60` | `0.600000` | `0.216667` | `0.466667` | `1.000000` | `0` |
| K8-KMNIST-ClassModeBucketTarget | N2c-OrthogonalTail | `60` | `0.516667` | `0.183333` | `0.316667` | `1.000000` | `0` |
| K8-KMNIST-ClassModeBucketTarget | N2c-TinyInit-SharedRBFFunc-BranchRatioCap | `60` | `0.516667` | `0.183333` | `0.316667` | `1.000000` | `0` |
| K8-KMNIST-ClassModeBucketTarget | N3c-SharedRBFFunc-DerivativeBand | `60` | `0.566667` | `0.250000` | `0.400000` | `1.000000` | `0` |

判断：

1. KMNIST 的 actual effect pass rate 仍高，best `K6 + N2a` 达到 `0.766667`。
2. 但 stronger replay 中它没有保留 control superiority：beats AdamWParallel `0.266667`，beats best LR `0.483333`。
3. 这说明 v9.2.27 的 KMNIST diagnosis survivor 没能升级成 integration survivor。

## 6. P4 Routed controller construction

| controller | rows | macro beats AdamWParallel | macro beats best LR | task safe | positive datasets | abstain-safe datasets | shuffle pass | routed pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U0-GlobalBestSingle | `36` | `0.500000` | `0.500000` | `0.972222` | `1` | `0` | `0` | `0` |
| U1-DatasetRouted | `36` | `0.472222` | `0.472222` | `0.972222` | `1` | `1` | `0` | `0` |
| U2-MetricRouted | `36` | `0.472222` | `0.472222` | `0.972222` | `1` | `1` | `0` | `0` |
| U3-EventRouted | `36` | `0.500000` | `0.500000` | `0.972222` | `1` | `0` | `0` | `0` |
| U4-AbstainHeavyRouter | `36` | `0.388889` | `0.388889` | `1.000000` | `1` | `2` | `0` | `0` |
| U5-RoleSignalRouted | `36` | `0.444444` | `0.444444` | `0.972222` | `1` | `1` | `0` | `0` |
| U6-KMNISTOnlyRouter | `36` | `0.388889` | `0.388889` | `1.000000` | `1` | `0` | `0` | `0` |
| U7-FashionOnlyRouter | `36` | `0.472222` | `0.472222` | `0.972222` | `1` | `0` | `0` | `0` |

判断：

1. 没有 routed controller 达到 macro `>=0.60` gate；best `U0` 也只有 `0.5/0.5`。
2. route-shuffle controls 没有通过，这一点是好事，但不是 positive causality。
3. P4 因 control superiority 不足失败，因此 P5-P8 不能打开。

## 7. Downstream boundary

P5-P8 只有在 P4 routed paired replay survivor 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 8. No-fake audit

```text
rows_checked = 31034
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
| `experiments/run_v9228_fashionfirst_routed_functional_kmnist_integration.py` | `7aadcbb3cc6c8723cce7f75c93ee62eb236e0a27e50511030ecbdfdb618a8bf2` |
| route | `9f229f99143c887619456622a5efc0b395d743f2f8232f6ec3f78ae27a9cefa5` |
| P1 Fashion completion | `8f40fd7c7a3e7e488c28a96622ac54fa07543b4c294f249c33b382bafe78b938` |
| P1 Fashion summary | `93cd0739925468e9b0cd28dc8250150fae96bcecd83e7686e25719b30d4a6cd4` |
| P2 Fashion mechanism | `4129b5687731e3089060701daa44a4e90bad89870660387f9f946cf90fd7e045` |
| P3 KMNIST integration replay | `469bcc4e826c9b3ac0219afe280a7e305f5eab3e06274a67251bf535d3c33931` |
| P3 KMNIST summary | `cab018d76dbebed17fce93428d70f9a71d0cac6d0bef33ead8ea9f3c7325a404` |
| P4 routed construction | `35cc666fba7b80d98635901dfaebdc6fce0d2604bc031ca1fc42cd0cc4870f49` |
| P4 routed summary | `4ee6e92e66d9a406bf52bb5a0516ef30f57bee1a31d41fc9df511d0127577609` |
| provenance audit | `bdf8c632af52234b287b820b28b336fe858745455cd3365ea8a40aa545304c6a` |

## 9. 最终分析结论

机制判断：

1. Fashion missing controllers 已经进入 measured replay，不再是 v9.2.27 的 not_implemented blocker。
2. KMNIST survivor 是否保留由 P3 stronger replay 决定，不能再只用 diagnosis rows 声明成功。
3. Routed controller 必须击败 AdamWParallel / best LR 且 route-shuffle controls 不能通过；否则就是 routing/control-equivalent。
4. 本轮 terminal blocker 是 `kmnist_survivor_not_preserved_under_stronger_replay`。

最终一句话：

> v9.2.28 真实执行后停在 `R5-KMNISTSurvivorNotPreserved`：`kmnist_survivor_not_preserved_under_stronger_replay`。
