# DG-KAN v9.2.11 Functional Causality Controller 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.11_Functional_Causality_Controller_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P4-P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.11 执行到一个可审计 terminal route：

```text
route = R5-FunctionalControlEquivalent
base_candidate = LQ-t2-h256
success_v9211_event_causality = false
success_v9211_short_run = false
success_v9211_full_functional = false
success_v9211_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9211_functional_causality_controller_first_20260509T220000Z/
```

核心结论：

1. P0 对 v9.2.10 final artifact 做 source recap，确认上一轮关键边界仍是：D9 one-step safe，但 P4 short-run step ratio `~1.43-1.46x` 且机制收益弱。
2. P1 归因明确：v9.2.10 的 P4 failure 主要是 `F1-overhead_dominant` 与 `F2-mechanism_weak`，没有证据表明 D9 在 short-run 中有独特机制优势。
3. 本轮新增 P2 paired event replay 是真实测量：`22680` rows，覆盖 LQ0/LQ1、三任务三 seed、7 个 direction、3 类 event、AdamW/real/no-op/random/geometry branches、1/5/20 step horizons。
4. P2 没有任何 event/direction/gate group 通过 causality gate：`p2_event_causality_pass_count = 0`。
5. Real functional branches 多数 task-safe，但不能按预注册阈值击败 matched controls；尤其 D9 的 task-safe projection 后 functional norm 极小，呈现 “safe but mostly neutered / control-equivalent”。
6. E5 high-confidence-SNR event 在本设置下 coverage 为 `0`，E0/E2 coverage 为 `1.0`，说明当前 event controller 没有形成高价值稀疏事件。
7. 因 P2 paired replay 未过，P3 controller promotion、P4 short-run、P5 full re-entry、P6 robustness、P7 external-ready 全部明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9211_functional_causality_controller.py` | v9.2.11 runner；生成 P0-P7 required artifacts、paired replay、route、failure/no-fake audit |

调整：

| 文件 | 改动 |
|---|---|
| `dgkan/functional/lq_functional_predictor.py` | 新增 D10-D16 direction family 与 E0/E1/E2/E3/E4/E5 event accept helper |

代码检查：

```bash
python -m py_compile experiments/run_v9211_functional_causality_controller.py \
  dgkan/functional/lq_functional_predictor.py
```

已通过。

正式运行：

```bash
python experiments/run_v9211_functional_causality_controller.py \
  --out-dir results/real_rerun_20260506/v9211_functional_causality_controller_first_20260509T220000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --batch-size 128 \
  --lr 0.0005 \
  --candidates LQ0,LQ1 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 9984 \
  --snr-warmup-steps 36 \
  --p2-event-steps 4 \
  --p2-horizons 1,5,20 \
  --p2-directions D9-SignalChannelProjection,D10-OrthogonalSignalGeometry,D11-SignalSubspaceCurvature,D13-LiftConditionCorrection,D14-QuadraticBasisEntropyCorrection,D15-OutputScaleTailCorrection,D16-KMNISTHardModeGeometry \
  --p2-event-types E0-uniform-stride,E2-margin-tail,E5-high-confidence-SNR \
  --p2-eval-size 512 \
  --step-fraction 0.01
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R5-FunctionalControlEquivalent",
  "base_candidate": "LQ-t2-h256",
  "p0_reproduction_pass": 1,
  "p1_failure_attribution_pass": 1,
  "p2_event_causality_pass": 0,
  "p2_event_causality_pass_count": 0,
  "p4_short_run_pass": 0,
  "primary_blocker": "paired_replay_real_functional_did_not_beat_matched_controls",
  "success_v9211_event_causality": 0,
  "success_v9211_short_run": 0,
  "success_v9211_full_functional": 0,
  "success_v9211_external_ready": 0
}
```

判断：

1. v9.2.11 的最小成功要求是 P2 paired replay causality pass；本轮没有达到。
2. 因 P2 未过，不能把任何 functional controller 推入 P4/P5。
3. route 写为 `R5-FunctionalControlEquivalent`，因为 real functional 在 paired replay 中 task-safe 但没有击败 controls。

## 3. P0 v9.2.10 source recap

Artifact：

```text
p0_v9210_reproduction.csv
```

说明：本轮 P0 是对 v9.2.10 final artifact 的 source recap，不伪装成新训练复现。

Source：

```text
results/real_rerun_20260506/v9210_functional_predictor_repair_p3p4_20260509T210000Z/
```

关键值：

| metric | value |
|---|---:|
| source route | `R5-FunctionalUnsafe` |
| source best direction | `D9-SignalChannelProjection` |
| source best gate | `G1-RoleSNROnly` |
| source step fraction | `0.01` |
| D9 bad-step rate | `0.027778` |
| D9 non-harm | `0.972222` |
| D9 prediction corr | `0.649843` |
| D9 overhead | `0.126000` |
| P4 best step ratio mean | `1.438416` |
| P4 best step ratio q90 | `1.457969` |
| P0 pass | `1` |

判断：上一轮 boundary 被正确纳入本轮：D9 one-step safe，但 P4 short-run 未成功。

## 4. P1 P4 failure attribution

Artifact：

```text
p1_p4_failure_attribution.csv
```

P1 归因：

| horizon | best step mean | best step q90 | best CEp99 delta | best margin delta | best control CEp99 delta | best control margin delta | factors |
|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | `1.429962` | `1.447735` | `-0.005691` | `+0.005776` | `-0.007958` | `+0.003386` | `F1-overhead_dominant,F2-mechanism_weak` |
| 240 | `1.446869` | `1.462239` | `+0.013035` | `+0.009931` | `+0.029286` | `+0.014728` | `F1-overhead_dominant,F2-mechanism_weak` |

判断：

1. P4 failure 不是 task accuracy 立刻崩坏，而是 overhead 明显超 strong gate。
2. 机制收益很弱：50-step CEp99 改善低于 RandomMatchedControl；240-step CEp99 反而变差。
3. v9.2.10 P4 source 没有记录 cosine，所以 `cos_with_adamw_step = not_measured_in_v9210_p4_source`，没有补造该值。

## 5. P2 paired event replay

Artifacts：

```text
p2_paired_event_replay_causality.csv
paired_replay_branch_trace_v9211.csv
p3_event_controller_direction_selection.csv
```

P2 protocol：

```text
base candidates = LQ0,LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
event steps = 4
horizons = 1,5,20
directions = D9,D10,D11,D13,D14,D15,D16
event types = E0-uniform-stride,E2-margin-tail,E5-high-confidence-SNR
branches = AdamWOnly, RealFunctional, NoOpMatchedOverhead, RandomMatchedNorm, GeometryD1Control
```

P2 row counts：

```text
p2_paired_event_replay_causality rows = 22680
p3_event_controller_direction_selection rows = 63
event_causality_pass_count = 0
```

Branch-level aggregate：

| horizon | branch | rows | accepted | acc delta | loss delta | CEp99 delta | margin delta | curvature delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | RealFunctional | 1512 | 1008 | `+0.000005` | `-8.65e-08` | `-2.11e-06` | `-4.34e-07` | `-9.05e-09` |
| 1 | RandomMatchedNorm | 1512 | 1008 | `0.000000` | `-9.93e-09` | `-7.29e-08` | `+8.55e-09` | `-6.90e-11` |
| 1 | GeometryD1Control | 1512 | 1008 | `0.000000` | `+1.88e-08` | `-7.59e-07` | `+3.48e-07` | `-5.00e-08` |
| 5 | RealFunctional | 1512 | 1008 | `+0.000005` | `-6.78e-08` | `-1.22e-05` | `-2.15e-06` | `-8.90e-09` |
| 5 | RandomMatchedNorm | 1512 | 1008 | `0.000000` | `+2.73e-08` | `-5.98e-07` | `+9.71e-07` | `-2.41e-10` |
| 5 | GeometryD1Control | 1512 | 1008 | `0.000000` | `+3.39e-08` | `-3.00e-07` | `-2.76e-09` | `-5.01e-08` |
| 20 | RealFunctional | 1512 | 1008 | `+0.000005` | `+3.17e-07` | `+2.98e-06` | `-1.39e-05` | `-9.18e-09` |
| 20 | RandomMatchedNorm | 1512 | 1008 | `0.000000` | `+4.44e-07` | `+3.43e-05` | `+9.36e-05` | `+3.10e-10` |
| 20 | GeometryD1Control | 1512 | 1008 | `0.000000` | `-1.69e-07` | `-7.26e-06` | `+9.05e-06` | `-5.11e-08` |

判断：

1. RealFunctional 的 task safety 基本成立，但 effect size 极小。
2. RealFunctional 在 horizon 20 的 CEp99/margin 不如 GeometryD1Control，不能过 causality gate。
3. Random / geometry controls 足以解释观测到的微小变化，因此不能声明 functional causality。

## 6. Direction / event summary

P2 summary 中没有任何 pass group。代表性 rows：

| direction | event | horizon | coverage | task safe | causal rate | real CEp99 delta | best control CEp99 delta | real margin delta | best control margin delta | pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D9 | E0 | 5 | `1.000000` | `1.000000` | `0.000000` | `+1.947e-06` | `-8.974e-07` | `+1.580e-06` | `+1.457e-06` | 0 |
| D9 | E0 | 20 | `1.000000` | `1.000000` | `0.000000` | `-5.307e-05` | `-1.089e-05` | `-5.573e-05` | `+1.404e-04` | 0 |
| D13 | E0 | 20 | `1.000000` | `1.000000` | `0.000000` | `+7.746e-05` | `-1.089e-05` | `-4.360e-05` | `+1.404e-04` | 0 |
| D16 | E0 | 20 | `1.000000` | `1.000000` | `0.000000` | `+7.693e-05` | `-1.089e-05` | `-8.093e-06` | `+1.404e-04` | 0 |
| D15 | E0 | 20 | `1.000000` | `1.000000` | `0.000000` | `-1.168e-05` | `-1.089e-05` | `+3.110e-06` | `+1.404e-04` | 0 |

Event behavior：

1. `E0-uniform-stride` 与 `E2-margin-tail` 在 early replay 中 coverage 都是 `1.0`，没有形成稀疏高价值事件。
2. `E5-high-confidence-SNR` coverage 为 `0.0`，说明当前 SNR threshold / warmup 下没有触发高置信 functional event。
3. 这支持 H3 的负面结论：当前 event controller 还不是 “high-value sparse event controller”。

Direction behavior：

| direction | h20 accepted | mean functional norm | mean cos with task grad | h20 CEp99 delta | h20 margin delta | h20 curvature delta |
|---|---:|---:|---:|---:|---:|---:|
| D9 | 144 | `3.010e-07` | `-0.075056` | `-3.538e-05` | `-3.716e-05` | `-2.759e-10` |
| D10 | 144 | `2.176e-06` | `~0.000000` | `-6.393e-06` | `-2.293e-06` | `-5.108e-08` |
| D13 | 144 | `2.176e-06` | `-0.000681` | `+5.164e-05` | `-2.907e-05` | `+3.104e-10` |
| D15 | 144 | `2.176e-06` | `-0.001147` | `-7.788e-06` | `+2.073e-06` | `-3.574e-08` |
| D16 | 144 | `2.176e-06` | `-0.000549` | `+5.129e-05` | `-5.395e-06` | `-2.784e-08` |

判断：

1. D9 的 update norm 很小，且 task-safe projection 后效果接近被中和；这解释了为什么它 one-step safe 但 short-run weak。
2. D10/D13/D16 更 orthogonal，但没有击败 controls；它们更像微小正则扰动。
3. D13/D16 在 h20 的 CEp99 反而变差，不能作为 hard-mode repair。

## 7. Downstream boundary

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | status | reason |
|---|---|---|
| `p4_short_run_controller_validation.csv` | `not_run` | `P2_event_causality_failed` |
| `p5_full_functional_reentry_10seed.csv` | `not_run` | `P2_event_causality_failed` |
| `p6_robustness_noisy_signal_validation.csv` | `not_run` | `P2_event_causality_failed` |
| `p7_strong_baseline_external_ready.csv` | `not_run` | `P2_event_causality_failed` |
| `functional_event_trace_v9211.csv` | `not_run` | `P2_event_causality_failed` |

判断：没有把 P4-P7 未打开阶段写成通过。

## 8. No-fake audit

`v9211_provenance_audit.csv`：

```text
rows_checked = 30313
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1 是 v9.2.10 source artifact recap / attribution，不伪装成新训练。
2. P2 是本轮真实 paired replay 测量。
3. P4-P7 是明确 `not_run`，没有 fake/proxy。
4. 本轮仍保持 CE-only，无 teacher/distillation/loss modification/sampler/class weight/offload/loss.backward。

## 9. Hash

| artifact | SHA256 |
|---|---|
| v9.2.11 plan | `01d7fc2593f1a3bedc722f59814820bbd5a3baf5ee3de789b08cffd7e2f7d20c` |
| `experiments/run_v9211_functional_causality_controller.py` | `fd9f6a4818ed97fc58d73dec44376c41e1db2ea4784782022d5a43bf818a65f6` |
| `dgkan/functional/lq_functional_predictor.py` | `caf7b9d72f1969a26654605cce91786d549dce7affc1c4ae3b837312fc06a799` |
| `dgkan/functional/snr_gated_lq.py` | `cd04043742c9e82e5e6a0bbb9d71010133bcd4c7753707cbf28e202087295c5e` |
| `dgkan/models/fc_purekan_lq.py` | `feb2bb205b0ad11155bca1c74d9f938d60cd6821c38a8c99a30c8db067669f8c` |
| route | `8d235ada17be623e4f6b89ddff39aeed702864d7fd4547ef2b51efd67e7d7268` |
| `p1_p4_failure_attribution.csv` | `b60c7ac3efc5ce1b0580bf7ab901f8daca0da15258be3e1324af4fa6a933e6bc` |
| `p2_paired_event_replay_causality.csv` | `d3194b409a2f01cc878dd1dcdff5e9757df3f0447ec775e97029dd2cef3e65f2` |
| `p3_event_controller_direction_selection.csv` | `30ed2f5dd7473e3ce32f8478f7b0e87e56d23ed4df6ac3af9c72aafee0ec1de1` |
| failure table | `fcbca89d9e118a3c06517d95718215b420b6671488f5c53f8903d6bf0010c3e3` |
| provenance audit | `347b57abb800787281cf70c054710c7bc5c9796dd674fa0cada133522aca36db` |

## 10. 最终分析结论

v9.2.11 的真实推进是：

```text
v9.2.10: D9 one-step safe，但 short-run weak / overhead high。
v9.2.11: paired replay 证明 real functional 没有击败 controls，因此不能再把 D9 当主线 functional causality。
```

机制判断：

1. 当前 functional 的核心问题已经不是 “会不会伤害 task”。
2. 核心问题变成：functional event 没有独立因果收益；real functional 的变化量太小，且被 geometry/random/no-op controls 解释。
3. D9 的安全性来自 task-safe projection，但 projection 也几乎中和了它的功能性；这就是 “safe but weak”。
4. Orthogonal / lift-condition / hard-mode directions 也没有通过 paired replay；它们没有形成强于 controls 的 CE-tail、margin 或 curvature 改善。
5. 当前 event controller 不合格：E0/E2 太宽，E5 太严；没有产生高价值稀疏事件。
6. 下一步不应继续调 D9 step fraction 或 SNR 阈值；应重新定义 functional signal，例如从训练轨迹中的真实 hard-mode transitions 学事件，或把 functional update 改为低频结构化 reparameterization，而不是每步微小 delta。

最终一句话：

> v9.2.11 真实执行后停在 `R5-FunctionalControlEquivalent`：paired event replay 没有发现任何 real functional 组合能按预注册阈值击败 NoOp/Random/Geometry controls。因此 P4-P7 不能打开，当前 functional causality 主线需要重新定义方向和事件，而不是继续小修 D9。

