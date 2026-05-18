# DG-KAN v9.2.22 Base-Qualified Strict PureKAN Functional Interface 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.22_BaseQualified_StrictPureKAN_FunctionalInterface_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = R3-FunctionalActuatabilityRetained
base_candidate = LQ-t2-h256
success_v9222_strict_purekan_functional = false
success_v9222_full_functional = false
success_v9222_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9222_basequalified_strict_purekan_functional_interface_first_20260510T120000Z/
```

核心结论：

1. P0 复现 v9.2.21 terminal boundary：I1c P4 pass 但 P5 near-pass fail。
2. P1 对 I1c miss rows 做了真实 autopsy，failure mechanism = `B5-hard_tail_amplification`。
3. P2 base-neutral interface factory 已执行，best base-neutral candidate = `N2a-TinyInit-RationalFunc-BranchRatioCap`。
4. P2 gate：contract `1`，P4 `1`，P5 near-pass `1`。
5. P3 functional actuatability pass = `1`；P4 paired replay pass = `0`。
6. 当前 blocker：`paired_replay_did_not_beat_adamwparallel_or_best_lr`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9222_basequalified_strict_purekan_functional_interface.py` | v9.2.22 runner；生成 P0-P7 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9222_basequalified_strict_purekan_functional_interface.py
```

正式运行：

```bash
python experiments/run_v9222_basequalified_strict_purekan_functional_interface.py \
  --out-dir results/real_rerun_20260506/v9222_basequalified_strict_purekan_functional_interface_first_20260510T120000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R3-FunctionalActuatabilityRetained",
  "base_candidate": "LQ-t2-h256",
  "v9221_boundary_pass": 1,
  "i1c_failure_mechanism": "B5-hard_tail_amplification",
  "best_base_neutral_candidate": "N2a-TinyInit-RationalFunc-BranchRatioCap",
  "interface_contract_pass": 1,
  "interface_p4_pass": 1,
  "interface_p5_nearpass": 1,
  "functional_actuatability_pass": 1,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 1.0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_kmnist_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "p2_best_near_pass_count": 9,
  "p2_best_row_count": 9,
  "p2_best_macro_delta": -0.003166655699412028,
  "p3_max_r_perp": 139.16336059570312,
  "p3_bad_event_rate": 0.0,
  "p4_real_beats_adamwparallel_rate": 0.17592592592592593,
  "p4_real_beats_best_lr_rate": 0.17592592592592593,
  "primary_blocker": "paired_replay_did_not_beat_adamwparallel_or_best_lr",
  "next_required_implementation": "redesign_base_neutral_interface_or_functional_event",
  "success_v9222_strict_purekan_functional": 0,
  "success_v9222_full_functional": 0,
  "success_v9222_external_ready": 0
}
```

## 3. P1 I1c failure autopsy

Artifact：

```text
p1_i1c_p5_failure_autopsy.csv
```

Summary：

```text
p1_pass = 1
i1c_failure_mechanism = B5-hard_tail_amplification
p1_miss_count = 2
p1_row_count = 9
```

判断：P1 是重新训练 current I1c 并记录 CE tail、margin、branch ratio、effective derivative、functional usage、lift condition 与 role update norm；不是只复述 v9.2.21 表格。

P1 miss rows：

| dataset | seed | KAN acc | MLP-match acc | delta | CEp99 | margin p10 | branch ratio | derivative p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KMNIST | 0 | `0.820500` | `0.838000` | `-0.017500` | `8.109769` | `-1.976641` | `0.129982` | `0.012682` |
| KMNIST | 1 | `0.815500` | `0.831500` | `-0.016000` | `8.013364` | `-2.061926` | `0.129592` | `0.012725` |

判断：本轮 current-I1c rerun 的 miss rows 与 v9.2.21 source boundary 不完全相同；v9.2.21 source miss 是 MNIST seed2 / KMNIST seed0，本轮 autopsy miss 集中在 KMNIST seed0/1。共同点是 hard-tail 指标恶化，因此归因为 `B5-hard_tail_amplification`，不是把 source miss 原样复制成新测量。

## 4. P2 base-neutral interface factory

Artifacts：

```text
p2_base_neutral_interface_factory.csv
interface_failure_trace_v9222.csv
branch_derivative_trace_v9222.csv
functional_channel_usage_trace_v9222.csv
```

Best candidate summary：

```text
best_base_neutral_candidate = N2a-TinyInit-RationalFunc-BranchRatioCap
interface_family = tiny_init_rational_branch_ratio_cap
P5_nearpass = 1
near_pass_count = 9
row_count = 9
macro_delta = -0.003166655699412028
repair_MNIST_seed2 = 1
repair_KMNIST_seed0 = 1
```

判断：P2 成功只表示 base qualification 恢复；不能自动写成 functional success。

P2 candidate summary：

| candidate | P5 near-pass | near rows | macro delta | repaired MNIST seed2 | repaired KMNIST seed0 |
|---|---:|---:|---:|---:|---:|
| N1a-ZeroInit-RationalFunc-FrozenBase | 0 | `6/9` | `-0.006167` | 0 | 0 |
| N2a-TinyInit-RationalFunc-BranchRatioCap | 1 | `9/9` | `-0.003167` | 1 | 1 |
| N2c-TinyInit-SharedRBFFunc-BranchRatioCap | 1 | `8/9` | `-0.005278` | 1 | 1 |
| N3a-RationalFunc-DerivativeBand | 0 | `7/9` | `-0.005889` | 1 | 1 |
| N3c-SharedRBFFunc-DerivativeBand | 1 | `9/9` | `-0.004889` | 1 | 1 |

说明：其他 candidate 没进入 summary，原因是 contract/P4 gate 未过或未实现；详见 `p2_base_neutral_interface_factory.csv`，没有把未实现分支补成失败训练。

## 5. P3/P4 downstream

P3 summary：

```text
functional_actuatability_pass = 1
max_r_perp = 139.16336059570312
bad_event_rate = 0.0
max_target_fit_R2 = 1.0
```

P4 summary：

```text
paired_replay_pass = 0
real_beats_adamwparallel_rate = 0.17592592592592593
real_beats_best_lr_rate = 0.17592592592592593
task_safe_rate = 1.0
```

P4 branch aggregate：

| branch | rows | CEp99 delta | margin delta | acc delta | task-safe |
|---|---:|---:|---:|---:|---:|
| RealFunctional | 108 | `+0.000010` | `+0.000003` | `0.000000` | `1.000000` |
| AdamWParallelTrustRatio-0.03 | 108 | `-0.003665` | `+0.001175` | `0.000000` | `1.000000` |
| LRScale-1.03 | 108 | `-0.003665` | `+0.001175` | `0.000000` | `1.000000` |
| RandomMatchedNorm | 108 | `+0.001802` | `+0.002940` | `+0.000109` | `1.000000` |
| NoOpMatchedOverhead | 108 | `0.000000` | `0.000000` | `0.000000` | `1.000000` |

判断：RealFunctional 是 task-safe，但没有 control superiority；`real_beats_adamwparallel_rate = 0.175926`、`real_beats_best_lr_rate = 0.175926`，远低于 paired replay gate。因此 P5 short/full validation 不允许打开。

未打开阶段均以 `not_run` row 落盘，没有倒灌成功。

## 6. No-fake audit

```text
rows_checked = 1203
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| v9.2.22 plan | `f95625644ced7856a161ba2115c7400b5346418c9a98abf5aa82f59497a43abd` |
| `experiments/run_v9222_basequalified_strict_purekan_functional_interface.py` | `3a2232654555bbb5a4aefc5048a64ab0a7fe43d668b07b2a6a26092a5bba3fd0` |
| route | `9598dd83379467209eb1af115d946ed431d5e0c70d513ca0a9bb9c9fa5502686` |
| P1 autopsy | `6d0f1cfe77149ddb5f278110cb09422b1dbd4a385a3f2761d9d296a2178a0aef` |
| P2 factory | `4e126739346ab69898cb833876d025d2964dabb72451214665bedf45ca5cd402` |
| P3 actuatability | `5e93fa054f369036c9c553c9482246bc2df27c4cfd63021a0a12901041399d7d` |
| P4 paired replay | `58de0b13fbef053fbb93ac58f67cb8c1f4938ede88b9ab6f1469336abac75371` |
| provenance audit | `0a528158e8368c510898df3c0924447b40b4e502903ee0f8c840e2797777db34` |

## 8. 最终分析结论

v9.2.22 的真实推进是：

```text
v9.2.21: strict interface contract/P4 可做，但 I1c 破坏 P5 base。
v9.2.22: 先归因 I1c failure，再测试 base-neutral / role-decoupled / warmstart attach interface。
```

机制判断：

1. 这轮的正确优先级是 base qualification；不能用 functional update 去掩盖一个还没 P5 near-pass 的 strict interface。
2. 如果 P2 找到 P5-qualified interface，它只说明 base 不再被 functional channel 破坏；仍必须经过 P3 actuatability 和 P4 strong-control paired replay。
3. 如果 P3/P4 fail，则说明 base-neutral repair 可能把 functional channel 静音，或 RealFunctional 仍被 AdamWParallel / LR controls 解释。
4. 当前 route 停在 `R3-FunctionalActuatabilityRetained`，原因是 `paired_replay_did_not_beat_adamwparallel_or_best_lr`。

最终一句话：

> v9.2.22 真实执行后停在 `R3-FunctionalActuatabilityRetained`：`paired_replay_did_not_beat_adamwparallel_or_best_lr`。
