# DG-KAN v9.2.12 Functional Direction Reconstruction 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.12_Functional_Direction_Reconstruction_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P4-P8 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.12 执行到一个可审计 terminal route：

```text
route = R5-ControlEquivalentAgain
base_candidate = LQ-t2-h256
success_v9212_output_direction = false
success_v9212_event_causality = false
success_v9212_full_functional = false
success_v9212_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9212_functional_direction_reconstruction_first_20260509T230000Z/
```

核心结论：

1. P0 复现 v9.2.11 terminal boundary：source route 仍是 `R5-FunctionalControlEquivalent`，source P2 causality pass count 为 `0`。
2. P1 control-equivalence autopsy 成立：旧 functional 主要失败机制是 event degeneracy、control-equivalence、output-effect too small 与 projection neutralization。
3. P2 已真实执行 output-space direction factory：`116640` measured rows，覆盖 LQ0/LQ1、三任务三 seed、12 个 output-target/subspace/solver priority 组合、3 类 calibrated sparse event、6 个 branch、1/5/20 horizons。
4. P2 没有任何 output-space direction 通过 gate：`p2_output_direction_pass_count = 0`。
5. Sparse event coverage 已修复到计划范围：group coverage 为 `0.033333` 或 `0.100000`；本轮不再是 v9.2.11 的 E0/E2 全开或 E5 全关问题。
6. 但 output-space directions 仍弱：group mean output displacement ratio 最大只有 `0.036009 < 0.05`，output target fit R2 最大 `0.059320`，说明当前 LQ 参数子空间很难实现预设 output target。
7. 即使部分 rows 有 CEp99 或 margin 改善，bad-event rate 仍高，且 RealFunctional 没有形成足以打开 P4 的稳定 control advantage。
8. 因 P2 未过，P4 paired replay causal confirmation、P5 short-run、P6 full re-entry、P7 robustness、P8 external-ready 全部明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `dgkan/functional/lq_output_space_functional.py` | output-space target、LQ logits VJP、parameter subspace、task-safe solver |
| `experiments/run_v9212_functional_direction_reconstruction.py` | v9.2.12 runner；生成 P0-P8 required artifacts、P2 direction factory、route、failure/no-fake audit |

复用：

| 文件 | 作用 |
|---|---|
| `dgkan/functional/lq_functional_predictor.py` | v9.2.10/v9.2.11 方向、mechanism metrics、event helper |
| `dgkan/functional/snr_gated_lq.py` | role scalar SNR、step norm/dot/projection 工具 |
| `dgkan/models/fc_purekan_lq.py` | LQ primitive manual forward/backward/update |

代码检查：

```bash
python -m py_compile experiments/run_v9212_functional_direction_reconstruction.py \
  dgkan/functional/lq_output_space_functional.py \
  dgkan/functional/lq_functional_predictor.py
```

已通过。

正式运行：

```bash
python experiments/run_v9212_functional_direction_reconstruction.py \
  --out-dir results/real_rerun_20260506/v9212_functional_direction_reconstruction_first_20260509T230000Z \
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
  --p2-event-steps 10 \
  --p2-target-coverage 0.10 \
  --p2-horizons 1,5,20 \
  --p2-event-types E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E6-CompositeSparseEvent \
  --p2-eval-size 256
```

说明：本轮 P2 使用 runner default priority set，共 12 个 output-target / subspace / solver 组合；不是 5-combo 缩小版。

## 2. Route

`route_decision.json`：

```json
{
  "route": "R5-ControlEquivalentAgain",
  "base_candidate": "LQ-t2-h256",
  "p0_reproduction_pass": 1,
  "p1_control_equivalence_autopsy_pass": 1,
  "p2_output_direction_pass": 0,
  "p2_output_direction_pass_count": 0,
  "p4_paired_replay_pass": 0,
  "p5_short_run_pass": 0,
  "p6_full_reentry_pass": 0,
  "primary_blocker": "output_space_directions_failed_safety_effect_or_control_gate",
  "success_v9212_output_direction": 0,
  "success_v9212_event_causality": 0,
  "success_v9212_full_functional": 0,
  "success_v9212_external_ready": 0
}
```

判断：

1. v9.2.12 的 minimum success 要求 P2 output direction pass；本轮未达到。
2. 这不是 event coverage failure：coverage 已进入 sparse window。
3. 也不是旧 D9/SNR overhead 问题：本轮重建了 output-space direction，但方向仍然弱且 control-equivalent。

## 3. P0 v9.2.11 reproduction

Artifact：

```text
p0_v9211_reproduction.csv
```

Source：

```text
results/real_rerun_20260506/v9211_functional_causality_controller_first_20260509T220000Z/
```

关键值：

| metric | value |
|---|---:|
| source route | `R5-FunctionalControlEquivalent` |
| source P2 causality pass count | `0` |
| source D9 mean coverage | `0.666667` |
| source D9 mean functional norm | `3.010341e-07` |
| source blocker | `paired_replay_real_functional_did_not_beat_matched_controls` |
| P0 pass | `1` |

判断：v9.2.11 的 control-equivalent terminal boundary 被正确纳入本轮；本轮不是在不稳定 source 上继续。

## 4. P1 control-equivalence autopsy

Artifact：

```text
p1_control_equivalence_autopsy.csv
```

Autopsy rows：

```text
rows = 63
```

Failure mechanism counts：

| mechanism | count |
|---|---:|
| `M3-event_degenerate` | `63` |
| `M5-control_equivalent` | `63` |
| `M4-output_effect_too_small` | `57` |
| `M1-projection_neutralization` | `33` |

说明：

1. v9.2.11 的旧 events 要么过宽、要么全不触发，P1 全部 rows 都带 `M3-event_degenerate`。
2. P1 全部 rows 都带 `M5-control_equivalent`，说明旧 RealFunctional 没有击败 matched controls。
3. 多数 rows output effect 太小；D9 的安全性主要来自 projection 后被中和。
4. 旧 source 未记录 `cos_with_adamw` / output displacement spectrum，本轮 CSV 明确写 `not_measured_in_v9211_source`，没有补造数值。

## 5. P2 output-space direction factory

Artifacts：

```text
p2_output_space_direction_factory.csv
p3_event_controller_calibration.csv
output_target_trace_v9212.csv
paired_replay_branch_trace_v9212.csv
```

P2 scope：

```text
base candidates = LQ0,LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
event checkpoints = 10 per dataset/seed/candidate
event controllers = E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E6-CompositeSparseEvent
functional candidates = 12 default priority output-target/subspace/solver combinations
branches = AdamWOnly, RealFunctional, NoOpMatchedOverhead, RandomMatchedNorm, ShuffledTarget, AdamWParallelDirection
horizons = 1,5,20
```

P2 row counts：

```text
p2_output_space_direction_factory rows = 116640
p3_event_controller_calibration rows = 108
output_target_trace rows = 2160
paired_replay_branch_trace rows = 38880
p2_output_direction_pass_count = 0
```

P2 summary ranges：

| metric | min | mean | max |
|---|---:|---:|---:|
| coverage | `0.033333` | `0.083333` | `0.100000` |
| bad-event rate | `0.000000` | `0.390432` | `0.833333` |
| holdout non-harm | `0.166667` | `0.609568` | `1.000000` |
| output displacement ratio vs AdamW logits | `0.008149` | `0.023436` | `0.036009` |
| output target fit R2 | `-0.000359` | `0.015340` | `0.059320` |

判断：

1. H4 sparse event coverage 被修好：coverage 落在 `0.033333-0.100000`，没有退化成 0 或 1。
2. H1 没过：group-level output displacement ratio 最大 `0.036009`，低于计划的 `0.05`。
3. output target fit 很弱：R2 最大 `0.059320`，说明 `J^T target` + 当前 subspace / task-safe projection 不能有效实现 target logits displacement。
4. safety 不稳定：mean bad-event rate `0.390432`，远高于 `<=0.05`。
5. control gate 没有形成稳定 survivor，因此 P4 不打开。

## 6. Representative P2 rows

Top output displacement group：

| target | subspace | event | horizon | coverage | bad | non-harm | output ratio | CEp99 delta | best control CEp99 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| O4 CurvatureOutputFlattening | S6 OrthogonalToAdamW | E1 | 1 | `0.100000` | `0.277778` | `0.722222` | `0.036009` | `-7.649e-05` | `-2.226e-05` |
| O4 CurvatureOutputFlattening | S6 OrthogonalToAdamW | E1 | 20 | `0.100000` | `0.444444` | `0.555556` | `0.036009` | `-4.304e-04` | `-9.978e-05` |
| O4 CurvatureOutputFlattening | S6 OrthogonalToAdamW | E6 | 20 | `0.100000` | `0.611111` | `0.388889` | `0.035970` | `-3.762e-04` | `-1.173e-04` |
| O6 KMNISTHardModeOutputTarget | S6 OrthogonalToAdamW | E1 | 5 | `0.033333` | `0.000000` | `1.000000` | `0.033570` | `-1.907e-05` | `-1.041e-06` |

Best safety rows：

| target | subspace | event | horizon | coverage | bad | non-harm | output ratio | CEp99 delta | margin delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| O6 KMNISTHardModeOutputTarget | S6 OrthogonalToAdamW | E1 | 5 | `0.033333` | `0.000000` | `1.000000` | `0.033570` | `-1.907e-05` | `+5.285e-06` |
| O6 KMNISTHardModeOutputTarget | S5 RecentSignalSubspace | E2 | 1 | `0.033333` | `0.000000` | `1.000000` | `0.032230` | `-1.589e-05` | `+5.138e-05` |
| O6 KMNISTHardModeOutputTarget | S6 OrthogonalToAdamW | E6 | 5 | `0.033333` | `0.000000` | `1.000000` | `0.028743` | `-4.458e-05` | `+2.420e-05` |

判断：

1. 有一些 one-step 或 sparse KMNIST rows 是安全的，也有 CEp99 / margin 改善信号。
2. 但这些 rows 的 output ratio 仍低于 `0.05`，而且不稳定扩展到 horizon 20。
3. O4/S6 的 CEp99 改善最大，但 bad-event rate 过高，不能作为 survivor。
4. O6/KMNIST hard-mode 更安全，但效果太弱，仍不能打开 P4。

## 7. Branch aggregate

Branch aggregate across all P2 rows：

| horizon | branch | rows | accepted | acc delta | loss delta | CEp99 delta | margin delta | output ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | RealFunctional | `6480` | `540` | `0.000000` | `-5.19e-07` | `-6.46e-06` | `-1.06e-06` | `0.019281` |
| 1 | ShuffledTarget | `6480` | `540` | `0.000000` | `-2.69e-07` | `-2.95e-06` | `-9.30e-07` | `0.018771` |
| 1 | AdamWParallel | `6480` | `540` | `0.000000` | `-1.37e-06` | `-1.26e-05` | `-4.14e-06` | `0.000000` |
| 5 | RealFunctional | `6480` | `540` | `0.000000` | `-1.72e-07` | `-1.45e-06` | `+4.66e-06` | `0.019281` |
| 5 | RandomMatchedNorm | `6480` | `540` | `0.000000` | `-5.63e-07` | `-4.30e-06` | `+7.76e-06` | `0.000000` |
| 20 | RealFunctional | `6480` | `540` | `-6.03e-07` | `-1.09e-07` | `+5.39e-07` | `-1.77e-05` | `0.019281` |
| 20 | RandomMatchedNorm | `6480` | `540` | `0.000000` | `-1.43e-06` | `-7.35e-05` | `-6.66e-05` | `0.000000` |
| 20 | ShuffledTarget | `6480` | `540` | `+1.21e-06` | `-5.46e-07` | `-2.43e-05` | `-2.15e-05` | `0.018771` |

判断：

1. RealFunctional 的 aggregate effect 仍然很小；horizon 20 下 CEp99 甚至略微变差。
2. RandomMatchedNorm / ShuffledTarget 在部分 aggregate 上不弱于 RealFunctional，因此不能声明 output-space target 有独立因果收益。
3. 这支持 `R5-ControlEquivalentAgain`，不是 functional advantage。

## 8. Downstream boundary

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | status | reason |
|---|---|---|
| `p4_paired_replay_causal_confirmation.csv` | `not_run` | `P2_output_direction_factory_failed` |
| `p5_short_run_multistep_validation.csv` | `not_run` | `P2_output_direction_factory_failed` |
| `p6_full_functional_reentry_10seed.csv` | `not_run` | `P2_output_direction_factory_failed` |
| `p7_noise_robustness_signal_validation.csv` | `not_run` | `P2_output_direction_factory_failed` |
| `p8_strong_baseline_external_ready.csv` | `not_run` | `P2_output_direction_factory_failed` |
| `functional_event_trace_v9212.csv` | `not_run` | `P2_output_direction_factory_failed` |

判断：没有把 P4-P8 未打开阶段写成通过。

## 9. No-fake audit

`v9212_provenance_audit.csv`：

```text
rows_checked = 157860
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1 是 v9.2.11 source artifact recap / autopsy，没有伪装成新训练。
2. P2 是本轮真实 output-space direction factory + paired branch replay。
3. P4-P8 是明确 gate-blocked `not_run`，没有 fake/proxy。
4. 本轮仍保持 CE-only，无 teacher/distillation/loss modification/sampler/class weight/offload/loss.backward。

## 10. Hash

| artifact | SHA256 |
|---|---|
| v9.2.12 plan | `c74441c216fd03a5c03db38e9be61ea54b8ac08ba682cee1ed171f6699f87014` |
| `experiments/run_v9212_functional_direction_reconstruction.py` | `5a36c5ec0078f0e912dfe2cc69d1e159dd3dddcd1eea026045f669418bc92401` |
| `dgkan/functional/lq_output_space_functional.py` | `0c55e598e7465dda65e4233d955a3756bb0adbea71de3b9e6295f86b54cd7d08` |
| `dgkan/functional/lq_functional_predictor.py` | `caf7b9d72f1969a26654605cce91786d549dce7affc1c4ae3b837312fc06a799` |
| `dgkan/functional/snr_gated_lq.py` | `cd04043742c9e82e5e6a0bbb9d71010133bcd4c7753707cbf28e202087295c5e` |
| `dgkan/models/fc_purekan_lq.py` | `feb2bb205b0ad11155bca1c74d9f938d60cd6821c38a8c99a30c8db067669f8c` |
| route | `3e95514b895c2c2534ace6d415b936638d634c66b5f469342bd6e8d3972802db` |
| P1 autopsy | `e3766a04e7fb7cf012a986f861f0982d6d875d0ff187d98e5a359b5097379fff` |
| P2 direction factory | `02a693f3514e51718b96c55ade075f6f59953ab8ee3a4c831d8294d8448be1da` |
| P3 calibration summary | `5ca7015c97b1af7c3ce78cd0a5d03ef370edb080c74f04a6357cc4bed539cad6` |
| output target trace | `82e76e28332ec9eb24c0bab6b612252942e58c23f3068f8c14e70b4b854dfe7c` |
| paired replay trace | `ee67935a22e5311e92e6fc8fb830f97f6a8ce36353eabc6895b86ae3564ceaeb` |
| provenance audit | `b04aedfbd62e93857bfbeaef4416736462639eb187a7da9848ae571299809d9e` |

## 11. 最终分析结论

v9.2.12 的真实推进是：

```text
v9.2.11: old functional direction/event control-equivalent。
v9.2.12: calibrated sparse events 成立，但 output-space target -> LQ parameter update 的实现能力太弱，仍不能击败 controls。
```

机制判断：

1. 本轮解决了一个旧问题：event controller 不再全开或全关，coverage 已进入 `0.033333-0.100000`。
2. 但核心问题没有解决：当前 LQ-t2-h256 的被测参数子空间无法把 output-space target 有效变成 logits displacement。group mean output ratio 最大 `0.036009`，target fit R2 最大 `0.059320`。
3. 这说明 $J^T\Delta z_{\text{target}}$ 的简单 VJP / projected-gradient 方向不足；它不是一个可控的 output-space solver。
4. 即使 hard-tail / margin / curvature target 有局部改善，matched controls 仍能解释相当多变化，aggregate horizon 20 下 RealFunctional 也没有稳定优于 Random 或 ShuffledTarget。
5. 因此不应继续小修 SNR 阈值、step fraction 或旧 D9；下一步应转向更强的求解结构或回到 primitive design。例如：显式小规模 least-squares / conjugate-gradient Jacobian sketch、可控输出校正 head 的 PureKAN 等价形式、或重新设计 LQ primitive 让 output-space correction 有足够可达性。

最终一句话：

> v9.2.12 真实执行后停在 `R5-ControlEquivalentAgain`：sparse event 已修复，但 12 个 output-space direction priority 组合没有任何一个通过 P2。当前问题不是 gate，而是 output target 到 LQ 参数子空间的可达性和因果收益仍不足；P4-P8 不能打开。

