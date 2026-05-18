# DG-KAN v9.2.14 P4-Qualified Functional Actuator Closure 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.14_P4Qualified_Functional_Actuator_Closure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P6/P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.14 执行到一个新的可审计 route：

```text
route = R4-ActuatorP4ClosedBaseQualified
base_candidate = LQ-t2-h256
best_actuator_candidate = A7c-BasisEntropy-ValueOnly
success_v9214_p4_qualified_actuator = true
success_v9214_absorbable_actuator = false
success_v9214_functional_advantage = false
success_v9214_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9214_p4qualified_functional_actuator_closure_first_20260510T010000Z/
```

核心结论：

1. v9.2.13 boundary 被复现：source route 为 `R8-NoPureKANActuatorFound`，post-actuator controllability pass count 仍为 `273`，source actuator P4 pass count 为 `0`。
2. P1 phase attribution 完成，A4/A5/A7 的 component timing unknown fraction 均小于 `0.10`；A5 显示 backward derivative/coeffgrad dominant，A4/A7 在本次 component repeat 中 backward 未显著慢于 MLP-match。
3. P2 找到多个 P4-qualified + controllability-retaining actuator：A4b/A4d/A4e/A7c 均通过 q90 P4 与 controllability gate。
4. 最佳 route candidate 是 `A7c-BasisEntropy-ValueOnly`：forward q90 `1.104638`、backward q90 `1.387102`、step q90 `1.213123`、memory `0.969731`、target fit R2 `1.0`、rz `1.106568`。
5. P4 AdamW-only base qualification 已真实打开：`A7c` 三任务三 seed 共 9 rows，near-pass `8/9`，macro delta `-0.004167`。
6. P3 absorbable actuator 有大量 row-level fit signal，但按 candidate 聚合 bad-event rate 均高于 `0.05`，所以 `absorbable_pass = 0`。
7. P5 paired replay 已真实打开，但 RealFunctional 未击败 controls：Real CEp99 mean delta `+0.001747`，best control CEp99 mean delta `-0.008564`；Real margin mean delta `-0.000325`，best control margin mean delta `+0.000943`。
8. 因 P5 paired replay causality 未过，P6 full functional re-entry 与 P7 robustness/external-ready 均明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9214_p4qualified_functional_actuator_closure.py` | v9.2.14 runner；生成 P0-P7 required artifacts、route、failure/no-fake audit |

调整：

| 文件 | 改动 |
|---|---|
| `dgkan/models/fc_purekan_actuator.py` | 新增 A4b/A4c/A4d/A4e/A5b/A5c/A7c actuator spec；value-only actuator 显式 stopgrad basis、只训练 output-edge coefficient |

代码检查：

```bash
python -m py_compile dgkan/models/fc_purekan_actuator.py \
  experiments/run_v9214_p4qualified_functional_actuator_closure.py
```

已通过。

正式运行：

```bash
python experiments/run_v9214_p4qualified_functional_actuator_closure.py \
  --out-dir results/real_rerun_20260506/v9214_p4qualified_functional_actuator_closure_first_20260510T010000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --lr 0.0005 \
  --batch-size 128 \
  --hidden-dim 256 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 9984 \
  --eval-size 512 \
  --audit-batch-size 128 \
  --p4-batch-size 128 \
  --p4-warmup 5 \
  --p4-reps 20 \
  --p4-repeat-measurements 3 \
  --p1-phase-reps 20 \
  --p4-base-datasets MNIST,Fashion-MNIST,KMNIST \
  --p4-base-seeds 0,1,2 \
  --p5-train-size 9984 \
  --p5-test-size 2000 \
  --p5-epochs 20 \
  --p5-lr 0.0005 \
  --p5-replay-datasets MNIST,Fashion-MNIST,KMNIST \
  --p5-replay-seeds 0,1,2 \
  --p5-horizons 1,5,20,80 \
  --p5-warmup-steps 36 \
  --p5-eval-size 512 \
  --p5-functional-step-fraction 0.10
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R4-ActuatorP4ClosedBaseQualified",
  "base_candidate": "LQ-t2-h256",
  "best_actuator_candidate": "A7c-BasisEntropy-ValueOnly",
  "best_actuator_family": "basis_entropy_value_only",
  "best_absorbable_candidate": "AB2-ActuatorSolveThenAbsorb-LiftPlusT2",
  "best_output_target": "O6-KMNISTHardModeOutputTarget",
  "p4_closure_pass": 1,
  "controllability_retained": 1,
  "absorbable_pass": 0,
  "actuator_base_near_pass": 1,
  "paired_replay_pass": 0,
  "forward_ratio_q90": 1.1046384911977811,
  "backward_ratio_q90": 1.387101514775696,
  "step_ratio_q90": 1.2131227301969405,
  "memory_ratio": 0.9697312055736007,
  "target_fit_R2": 1.0,
  "output_displacement_ratio_rz": 1.106567684033476,
  "primary_blocker": "P4_qualified_actuator_base_nearpass_but_paired_replay_causality_failed"
}
```

判断：

1. v9.2.14 达到 minimum success：找到了 P4-qualified controllable actuator，并且 base near-pass 未被破坏。
2. 但没有达到 functional causality success：paired replay 没有击败 controls。
3. 当前 blocker 已从 “没有 P4-qualified actuator” 推进到 “有 P4-qualified actuator，但 functional causality 仍不成立”。

## 3. P0 v9.2.13 boundary reproduction

Artifact：

```text
p0_v9213_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `R8-NoPureKANActuatorFound` |
| direct logit oracle pass | `1` |
| current LQ raw max R2 | `0.9999945759773254` |
| current LQ raw max rz | `49.254014712137426` |
| current LQ safe max R2 | `0.18964403867721558` |
| current LQ safe max rz | `0.040227413979989614` |
| source actuator P4 pass count | `0` |
| post-actuator controllability pass count | `273` |
| fake/proxy count | `0` |

判断：v9.2.13 的边界稳定复现；本轮是在真实 actuator-system blocker 上继续推进。

## 4. P1 P4 failure phase attribution

Artifact：

```text
p1_actuator_p4_failure_attribution.csv
```

| candidate | unknown fraction | backward total ms | MLP backward ms | backward dominant |
|---|---:|---:|---:|---:|
| A4-LQ-BoundedRationalActuator | `0.020312` | `0.191810` | `0.195389` | 0 |
| A5-LQ-PiecewiseLinear2Actuator | `0.024016` | `0.223056` | `0.195139` | 1 |
| A7-LQ-BasisEntropyActuator | `0.020937` | `0.191432` | `0.195539` | 0 |

判断：

1. P1 attribution pass 成立：三行 unknown fraction 均小于 `0.10`。
2. A5 的 backward excess 明确更像 derivative/coeffgrad blocker。
3. A4/A7 在本轮 component repeat 中未复现出明显 backward excess，说明 v9.2.13 的 near-threshold P4 fail 有 timing/protocol 敏感性；这支持使用 q90 repeat，而不是单次 timing。

## 5. P2 P4 closure candidate factory

Artifact：

```text
p2_p4_closure_candidate_factory.csv
```

关键 rows：

| candidate | P4 | controllability | forward q90 | backward q90 | step q90 | R2 | rz | GradRelErrMax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A4b-BoundedRational-BranchlessDerivative | 1 | 1 | `1.112932` | `1.430346` | `1.175085` | `0.9999998` | `0.345196` | `6.414e-08` |
| A4c-BoundedRational-FixedBeta | 0 | 0 | `1.250530` | `1.691315` | `1.119179` | `0.9999998` | `0.345196` | `5.979e-08` |
| A4d-BoundedRational-ValueOnlyActuator | 1 | 1 | `1.112949` | `1.413886` | `1.175444` | `0.9999998` | `0.345196` | `5.848e-08` |
| A4e-BoundedRational-FusedCoeffGrad | 1 | 1 | `1.102240` | `1.427959` | `1.108534` | `0.9999998` | `0.345196` | `6.109e-08` |
| A5b-PiecewiseLinear2-Branchless | 0 | 0 | `1.253297` | `1.655878` | `1.247410` | `1.0` | `1.164402` | `5.633e-08` |
| A5c-PiecewiseLinear2-FixedKnots | 0 | 0 | `1.255108` | `1.672331` | `1.240714` | `1.0` | `1.164402` | `5.633e-08` |
| A7b-BasisEntropy-LowRank | 0 | 0 | `not_implemented` | `not_implemented` | `not_implemented` |  |  |  |
| A7c-BasisEntropy-ValueOnly | 1 | 1 | `1.104638` | `1.387102` | `1.213123` | `1.0` | `1.106568` | `6.377e-08` |

判断：

1. A4 family 的 P4 closure 是真实的：A4b/A4d/A4e 都过 P4 q90，且保留 controllability。
2. A7c 是本轮最佳 candidate，因为同时有更强 rz 与更低 backward q90。
3. A5 family 仍未过 P4，主要卡在 forward/backward q90。
4. A7b lowrank output-edge 没有实现，按 `not_implemented` 记录，没有伪装成失败测量或成功。
5. Value-only actuator 是关键系统设计：basis value 仍来自 FC-PureKAN edge coordinate，但 actuator channel 不再把 basis-shape derivative 传回 lift；这显著降低 backward，同时保留 output controllability。

## 6. P3 absorbable actuator audit

Artifact：

```text
p3_absorbable_actuator_audit.csv
```

总体：

```text
rows = 1125
row-level absorbable pass = 845
aggregate absorbable_pass = 0
```

按 candidate 聚合：

| candidate | rows | bad-event rate | fit/rz pass rate | row pass | max R2 | max rz |
|---|---:|---:|---:|---:|---:|---:|
| AB0-T2 | 225 | `0.097778` | `0.866667` | 173 | `1.0` | `1.012388` |
| AB1-Lift | 225 | `0.133333` | `0.866667` | 165 | `1.0` | `1.012389` |
| AB2-LiftPlusT2 | 225 | `0.106667` | `0.866667` | 171 | `1.0` | `1.012409` |
| AB3-ShadowLowRank | 225 | `0.133333` | `0.866667` | 165 | `1.0` | `1.012389` |
| AB4-EventOnly | 225 | `0.106667` | `0.866667` | 171 | `1.0` | `1.012409` |

判断：

1. Absorbable actuator 的 fit signal 很强，不是没有可吸收方向。
2. 但所有 aggregate bad-event rate 都超过计划阈值 `<=0.05`。
3. 因此不能把 row-level pass 写成 absorbable route success；`absorbable_pass = 0` 是正确结论。

## 7. P4 actuator base qualification

Artifact：

```text
p4_p4_qualified_actuator_base_qualification.csv
```

Candidate：

```text
A7c-BasisEntropy-ValueOnly
```

Rows：

| dataset | seed | KAN acc | MLP-match acc | delta | near-pass |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | `0.949000` | `0.949500` | `-0.000500` | 1 |
| MNIST | 1 | `0.948500` | `0.952500` | `-0.004000` | 1 |
| MNIST | 2 | `0.945000` | `0.953000` | `-0.008000` | 1 |
| Fashion-MNIST | 0 | `0.867500` | `0.868000` | `-0.000500` | 1 |
| Fashion-MNIST | 1 | `0.869000` | `0.872000` | `-0.003000` | 1 |
| Fashion-MNIST | 2 | `0.863000` | `0.864500` | `-0.001500` | 1 |
| KMNIST | 0 | `0.821500` | `0.836500` | `-0.015000` | 0 |
| KMNIST | 1 | `0.824000` | `0.827500` | `-0.003500` | 1 |
| KMNIST | 2 | `0.826000` | `0.827500` | `-0.001500` | 1 |

Summary：

```text
near-pass = 8/9
macro delta = -0.0041666627
actuator_base_near_pass = 1
```

判断：

1. A7c 没有破坏 LQ near-pass base，P4-qualified actuator base 成立。
2. 但它仍不是 full pass，KMNIST seed0 仍明显低于 MLP-match。
3. 这把 v9.2.13 的 blocker 从 “NoPureKANActuatorFound” 推进到 paired replay causality。

## 8. P5 paired replay causality

Artifact：

```text
p5_paired_replay_after_actuator_closure.csv
functional_event_trace_v9214.csv
```

Protocol：

```text
candidate = A7c-BasisEntropy-ValueOnly
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
targets = O1,O2,O3,O4,O6
horizons = 1,5,20,80
branches = AdamWOnly, RealFunctional, NoOpMatchedOverhead, RandomMatchedNorm,
           ShuffledTarget, AdamWParallelDirection, ActuatorNoAbsorbControl,
           AbsorbRandomTarget
functional_step_fraction = 0.10
```

P5 aggregate by branch：

| branch | rows | acc delta | loss delta | CEp99 delta | margin delta | task-safe |
|---|---:|---:|---:|---:|---:|---:|
| RealFunctional | 180 | `-0.000087` | `+1.588e-06` | `+0.001747` | `-0.000325` | `0.994444` |
| ActuatorNoAbsorbControl | 180 | `-0.000087` | `+1.589e-06` | `+0.001747` | `-0.000325` | `0.994444` |
| AdamWParallelDirection | 180 | `-0.000217` | `-0.000193` | `-0.008564` | `+0.000943` | `0.972222` |
| RandomMatchedNorm | 180 | `0.000000` | `-5.552e-06` | `+0.000885` | `+0.000630` | `1.000000` |
| ShuffledTarget | 180 | `-0.000109` | `+9.368e-06` | `+0.002242` | `-0.000799` | `0.988889` |
| AbsorbRandomTarget | 180 | `-0.000163` | `-2.702e-05` | `+0.002314` | `-0.001964` | `0.972222` |
| NoOpMatchedOverhead | 180 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |
| AdamWOnly | 180 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |

RealFunctional by horizon：

| horizon | rows | CEp99 delta | margin delta | task-safe |
|---:|---:|---:|---:|---:|
| 1 | 45 | `-9.007e-07` | `-1.893e-06` | `1.000000` |
| 5 | 45 | `-1.579e-06` | `+1.346e-06` | `1.000000` |
| 20 | 45 | `-3.788e-06` | `-2.061e-05` | `1.000000` |
| 80 | 45 | `+0.006995` | `-0.001280` | `0.977778` |

Gate summary：

```text
p5_task_safe_rate = 0.994444
p5_real_mean_CEp99_delta = +0.001747
p5_best_control_mean_CEp99_delta = -0.008564
p5_real_mean_margin_delta = -0.000325
p5_best_control_mean_margin_delta = +0.000943
p5_mechanism_pass = 0
paired_replay_pass = 0
```

判断：

1. A7c RealFunctional 是 task-safe 的，但机制上没有击败 controls。
2. RealFunctional 与 ActuatorNoAbsorbControl 几乎等价，说明本轮的 functional branch 没形成独立可归因机制。
3. AdamWParallelDirection 明显更好地改善 CE tail 和 margin，这说明 paired replay 的收益仍可由 task-gradient-parallel control 解释。
4. 因 P5 未过，P6 full re-entry 和 P7 robustness/external-ready 不允许打开。

## 9. Downstream boundary

这些 artifact 已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p6_functional_reentry_after_actuator_closure.csv` | `P5_paired_replay_causality_failed` |
| `p7_noise_robustness_external_ready.csv` | `P5_paired_replay_causality_failed` |

判断：没有用 full training、noise robustness 或 external validation 越过 P5 causality gate。

## 10. No-fake audit

`v9214_provenance_audit.csv`：

```text
rows_checked = 5158
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0 是 v9.2.13 source artifact recap，不伪装成新 rerun。
2. P1/P2/P3/P4/P5 均来自本轮真实 runner。
3. A7b 明确 `not_implemented`。
4. P6/P7 是明确 gate-blocked `not_run`。

## 11. Hash

| artifact | SHA256 |
|---|---|
| v9.2.14 plan | `d42828f4db64f30c619eb433368dea524b7cb7db907abd5711097be23173e899` |
| `dgkan/models/fc_purekan_actuator.py` | `031934e49964d947d028df4792aa1e6afdd542596121920db6fb71e205d15c96` |
| `experiments/run_v9214_p4qualified_functional_actuator_closure.py` | `30ecd56cb6a56ceae02540f472097b9616010716ed169218760bb67bdff44fe5` |
| route | `3fb66a05274059acdb2ace51dd74c044b273ba73424f8d0dd9d5a09ddc9b395b` |
| P0 recap | `aac45f8790323df8ec1b1f33acbea19ae46c9c97fae4dc2ab5d2467791eb325c` |
| P1 attribution | `874fb1c1a0edab70fc83d95d81e41351f812344e1f7c48e9146f9dcae69308ca` |
| P2 closure factory | `14335dd7743666350c19063b0a5e85beb7d2c1ed15a2f5cd9807402767736e31` |
| P3 absorbable audit | `92ea892b43f6406641cd94cf93674d0a8070ab2fa04a99958b633bfa165bc50d` |
| P4 base qualification | `208649fac5149dc5f42030218576184c7326b6fc6681fbb37b6f9afa57e2b371` |
| P5 paired replay | `f0c83af064e7814d39ccc1f7f7e7d150fdd15385544347a5384e97d640cbd0a5` |
| provenance audit | `fa509d7f1cd56cf503ff1f986c7ef94c00e90a5357c52fc5030a2a0a16efd4d9` |

## 12. 最终分析结论

v9.2.14 的真实推进是：

```text
v9.2.13: actuator controllability exists, but no actuator P4-qualified.
v9.2.14: value-only actuator closes P4 and keeps base near-pass, but paired replay still control-equivalent.
```

机制判断：

1. v9.2.13 的 “NoPureKANActuatorFound” 已经被推进：A4 family 和 A7c 都能形成 P4-qualified controllable actuator。
2. A7c 的关键是 value-only edge-owned channel：它保留 basis value 的 output controllability，但不把 actuator basis derivative 常驻传回 lift，从而关闭 backward gate。
3. 这说明常驻 actuator 的 P4 blocker 不是不可破；只要把 basis-shape gradient 从每步训练里移除，P4 envelope 可以成立。
4. 但 functional causality 仍未成立：RealFunctional 没有击败 matched controls，尤其输给 AdamWParallelDirection。
5. Absorbable actuator 也不是当前成功 route：虽然 fit/rz 行级信号强，但 aggregate bad-event rate 超过 `0.05`。
6. 当前问题本质已经转移：不再是 “找不到 P4 actuator”，而是 “P4 actuator 的 functional update 是否有独立于 task-gradient control 的因果收益”。

最终一句话：

> v9.2.14 找到了第一个 P4-qualified 且 base-near-pass 的 strict FC-PureKAN actuator：`A7c-BasisEntropy-ValueOnly`。但 P5 paired replay 未证明 functional causality，RealFunctional 的 tail/margin 改善不如 controls，因此 P6/P7 不能打开。
