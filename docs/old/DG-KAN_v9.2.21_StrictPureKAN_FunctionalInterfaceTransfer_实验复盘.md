# DG-KAN v9.2.21 Strict PureKAN Functional Interface Transfer 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.21_StrictPureKAN_FunctionalInterfaceTransfer_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 P4-P7 写成通过。

## 0. 最新结论

```text
route = R7-InterfaceBreaksBase
base_candidate = LQ-t2-h256
success_v9221_strict_purekan_functional = false
success_v9221_full_functional = false
success_v9221_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9221_strict_purekan_functional_interface_transfer_first_20260510T110000Z/
```

核心结论：

1. P0 复现 v9.2.20 boundary：`functional_core_retained = 1`，`v8_full_replay_pass = 1`。
2. P1 从 v9.2.20 full replay 中抽取 FT7/Adaptive 机制，`ft7_mechanism_identified = 1`。
3. P2 strict interface contract / gradcheck 已执行，候选来自 edge-owned T2 task channel + functional channel，不使用 external residual。
4. P3 P4/P5 base qualification 的当前 best interface 是 `I1c-T2Task-RationalFunc-DerivativeControlled`；`interface_p4_pass = 1`，`interface_p5_nearpass = 0`。
5. 当前 blocker：`best_strict_interface_candidate_failed_P5_nearpass`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9221_strict_purekan_functional_interface_transfer.py` | v9.2.21 runner；生成 P0-P7 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9221_strict_purekan_functional_interface_transfer.py
```

正式运行：

```bash
python experiments/run_v9221_strict_purekan_functional_interface_transfer.py \
  --out-dir results/real_rerun_20260506/v9221_strict_purekan_functional_interface_transfer_first_20260510T110000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R7-InterfaceBreaksBase",
  "base_candidate": "LQ-t2-h256",
  "v9220_boundary_pass": 1,
  "ft7_mechanism_identified": 1,
  "best_interface_candidate": "I1c-T2Task-RationalFunc-DerivativeControlled",
  "interface_contract_pass": 1,
  "interface_p4_pass": 1,
  "interface_p5_nearpass": 0,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_kmnist_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "p5_near_pass_count": 7,
  "p5_row_count": 9,
  "p5_macro_delta": -0.004777789115905762,
  "primary_blocker": "best_strict_interface_candidate_failed_P5_nearpass",
  "next_required_implementation": "redesign_strict_interface_basis_or_base_qualification",
  "success_v9221_strict_purekan_functional": 0,
  "success_v9221_full_functional": 0,
  "success_v9221_external_ready": 0
}
```

## 3. P1 FT7 mechanism extraction

Artifacts：

```text
p1_ft7_mechanism_extraction.csv
role_mechanism_trace_v9221.csv
```

关键值：

| metric | value |
|---|---:|
| functional family corr effective derivative vs -curvature | `0.992452` |
| functional family corr branch ratio vs -curvature | `0.995620` |
| FT7 mean cosine with AdamW | `-0.049826` |
| FT7 beats AdamWParallel | `1.000000` |
| FT7 beats best LR | `1.000000` |

判断：P1 不重新训练；它只从 v9.2.20 真实 full replay rows 做机制抽取，不补造缺失的 per-event $r_\perp$。

## 4. P2/P3 strict interface result

Artifacts：

```text
p2_strict_purekan_interface_contract_gradcheck.csv
p3_interface_p4_p5_base_qualification.csv
purekan_interface_trace_v9221.csv
```

P3 summary：

```text
best_interface_candidate = I1c-T2Task-RationalFunc-DerivativeControlled
best_actuator_spec = A4e-BoundedRational-FusedCoeffGrad
interface_p4_pass = 1
interface_p5_nearpass = 0
p5_near_pass_count = 7
p5_row_count = 9
p5_macro_delta = -0.004777789115905762
```

P2 contract / gradcheck 代表 rows：

| candidate | actuator spec | GradRelErrMax | GradCosMin | pairwise R2 | eligible |
|---|---|---:|---:|---:|---:|
| `I1a-T2Task-RationalFunc-FixedBeta` | `A4c-BoundedRational-FixedBeta` | `5.978800e-08` | `0.999999881` | `0.984968` | 1 |
| `I1b-T2Task-RationalFunc-ValueOnly` | `A4d-BoundedRational-ValueOnlyActuator` | `5.847926e-08` | `1.000000000` | `0.984968` | 1 |
| `I1c-T2Task-RationalFunc-DerivativeControlled` | `A4e-BoundedRational-FusedCoeffGrad` | `6.109467e-08` | `0.999999940` | `0.981371` | 1 |
| `I2a-T2Task-Piecewise2Func-FixedKnots` | `A5c-PiecewiseLinear2-FixedKnots` | `5.633314e-08` | `1.000000000` | measured | 1 |
| `I3a-T2Task-SharedRBF4Func-FixedCenters` | `A6-LQ-LocalRBFSharedCenterActuator` | `5.885195e-08` | `0.999999881` | `0.980430` | 1 |

未实现 / contract fail rows：

```text
I2b-T2Task-Piecewise4Func-FixedKnots = not_implemented
I3b-T2Task-SharedRBF4Func-ValueOnly = not_implemented
I4c-FT7StyleRoleGuard = contract_fail
I5a-RationalFunc-OrthogonalCorrection = direction_only_contract_fail
```

P3 P4 rows：

| candidate | forward q90 | backward q90 | step q90 | memory | P4 |
|---|---:|---:|---:|---:|---:|
| `I1a` | `1.112012` | `1.436023` | `6.595997` | `0.969731` | 0 |
| `I1b` | `1.110000` | `1.436254` | `1.173560` | `0.969731` | 1 |
| `I1c` | `1.109782` | `1.423988` | `1.106582` | `0.969731` | 1 |
| `I2a` | `1.280314` | `1.689390` | `1.248995` | `0.969923` | 0 |
| `I3a` | `1.089055` | `1.409726` | `1.213255` | `0.969731` | 1 |

Best P4 candidate `I1c` 的 P5 base qualification：

| dataset | seed | KAN acc | MLP-match acc | delta | near-pass |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `0.946000` | `0.949500` | `-0.003500` | 1 |
| MNIST | 1 | `0.948500` | `0.952500` | `-0.004000` | 1 |
| MNIST | 2 | `0.942500` | `0.953000` | `-0.010500` | 0 |
| Fashion-MNIST | 0 | `0.869500` | `0.868000` | `+0.001500` | 1 |
| Fashion-MNIST | 1 | `0.871500` | `0.872000` | `-0.000500` | 1 |
| Fashion-MNIST | 2 | `0.861500` | `0.864500` | `-0.003000` | 1 |
| KMNIST | 0 | `0.822000` | `0.836500` | `-0.014500` | 0 |
| KMNIST | 1 | `0.823000` | `0.827500` | `-0.004500` | 1 |
| KMNIST | 2 | `0.823500` | `0.827500` | `-0.004000` | 1 |

判断：

1. Strict interface contract / gradcheck 不是 blocker；多个 edge-owned interface row 真实过了 contract、grad 与 interaction。
2. P4 也不是绝对 blocker；`I1b/I1c/I3a` 均进入 P4 gate。
3. 真实 terminal blocker 是 P5 near-pass：best candidate `I1c` 只有 `7/9` near-pass，低于 `>=0.80` 要求；因此 P4 paired replay 不允许打开。

## 5. Downstream boundary

P4-P7 只有在 P3 interface P4 + P5 near-pass 后才允许打开。本轮未打开阶段均以 `not_run` rows 落盘。

## 6. No-fake audit

```text
rows_checked = 31
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
| v9.2.21 plan | `747af064f15012321b46fd96b59c8e408d71a68f399d0939189a137ec5b79b60` |
| `experiments/run_v9221_strict_purekan_functional_interface_transfer.py` | `755fd51daecddd1f04c1a6bb8149f97baea6d18de9e31fcd454211863aa3a485` |
| route | `bbec800e3a879734e42944ee42579c3564e80ea724ddae4f8fe5b2135bf6b7bc` |
| `p1_ft7_mechanism_extraction.csv` | `9fe236cdd9f6a074cbb980aed66dcef7cc4c5b1a6df3dfc92063347c9874d94c` |
| `p2_strict_purekan_interface_contract_gradcheck.csv` | `e5c313bb4f89f23d6d916d4b2e3585da89a18bf8434591e1c83a3677cfec9e34` |
| `p3_interface_p4_p5_base_qualification.csv` | `95b37b51d05de1ce8d46c8e7aabe8c91f3fc2cc738d8845ec496f8e226b14525` |
| provenance audit | `1fb58849df91791a893bc482864dddf2f96b0a12917b40a1ac8b874118a13eaf` |

## 8. 最终分析结论

v9.2.21 的真实推进是：

```text
v9.2.20: v8 functional core retained under strong controls.
v9.2.21: FT7-like mechanism extracted; strict PureKAN interface candidates tested through contract/P4/P5 gate.
```

机制判断：

1. P1 支持 FT7 functional core 的核心不是普通 scalar LR：FT7 与 AdamW 的 mean cosine 很低，并且 full replay 已经 beat AdamWParallel / best LR。
2. 本轮证明 strict PureKAN dual-role functional channel 的 contract / grad / P4 并非不可做；`I1b/I1c/I3a` 都是 P4-pass interface。
3. 但 transfer 不能只看 P4。`I1c` 在 AdamW-only base qualification 上只有 `7/9` near-pass，MNIST seed2 与 KMNIST seed0 仍破 gate。
4. 因此 P4 paired replay 没有打开是正确的：不能用 functional control 去覆盖一个尚未 base-qualified 的 strict interface。
5. 当前 route 停在 `R7-InterfaceBreaksBase`，原因是 `best_strict_interface_candidate_failed_P5_nearpass`。

最终一句话：

> v9.2.21 真实执行后停在 `R7-InterfaceBreaksBase`：FT7 functional core 机制已被抽取，strict PureKAN interface 已经做到 contract/grad/P4 pass；但 best interface `I1c` 未通过 P5 near-pass，因此 paired replay / full functional / external-ready 均不能打开。
