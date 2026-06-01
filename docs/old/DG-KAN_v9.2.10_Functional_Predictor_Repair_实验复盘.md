# DG-KAN v9.2.10 Functional Predictor Repair 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.10_Functional_Predictor_Repair_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P5-P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.10 执行到一个可审计 terminal route：

```text
route = R5-FunctionalUnsafe
base_candidate = LQ-t2-h256
best_functional_candidate = D9-SignalChannelProjection
best_gate = G1-RoleSNROnly
best_step_fraction = 0.01
success_v9210_functional_predictor = true
success_v9210_functional_advantage = false
success_v9210_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9210_functional_predictor_repair_p3p4_20260509T210000Z/
```

核心结论：

1. v9.2.9 的 P2 failure 被真实复现：`D1-QuadraticCoeffDamping + G1 + rho0.10` 仍为 bad-step rate `0.819444`、prediction corr `0.231090`。
2. P2 direction factory 找到真实 one-step-safe survivor：`D9-SignalChannelProjection + G1` 在 `rho=0.01/0.03/0.10` 都通过 P2 safety/system/mechanism gate。
3. 最佳 P2 candidate 是 `D9 + G1 + rho0.01`：bad-step rate `0.027778`、holdout non-harm `0.972222`、prediction corr `0.649843`、overhead `0.126000`。
4. P3 calibrated predictor 通过：`PRED0-AllAcceptP2Survivor` 和 `PRED3-MarginRiskBottomQuartile` 都满足 precision-first calibration gate。
5. P4 short-run 没有通过：BestFunctional 在 50/240 steps 下 task-safe，但 step time ratio 分别约 `1.429962` / `1.446869`，超过本轮 `<=1.20` short-run system gate；所有 BestFunctional P4 rows 均未 pass。
6. P4 中 BestFunctional 的机制收益很小，且 GeometryD1Control / RandomMatchedControl 也出现相近甚至更好的 CEp99/margin 变化，不能证明 functional causality。
7. 因 P4 未过，P5 full re-entry、P6 robustness、P7 strong baseline / external ready 均明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `dgkan/functional/lq_functional_predictor.py` | LQ functional direction factory 与机制指标；避免把数学实现继续塞进 runner |
| `experiments/run_v9210_functional_predictor_repair.py` | v9.2.10 runner；生成 P0-P7 required artifacts、route、failure table、no-fake audit |

复用：

| 文件 | 作用 |
|---|---|
| `dgkan/functional/snr_gated_lq.py` | v9.2.9 low-cost role-scalar SNR / task-safe projection 工具 |
| `dgkan/models/fc_purekan_lq.py` | LQ primitive manual forward/backward/update |

代码检查：

```bash
python -m py_compile experiments/run_v9210_functional_predictor_repair.py \
  dgkan/functional/lq_functional_predictor.py \
  dgkan/functional/snr_gated_lq.py
```

已通过。

正式运行：

```bash
python experiments/run_v9210_functional_predictor_repair.py \
  --out-dir results/real_rerun_20260506/v9210_functional_predictor_repair_p3p4_20260509T210000Z \
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
  --steps 8 \
  --snr-warmup-steps 36 \
  --microbatch-size 64 \
  --run-p3-p4 \
  --p4-candidates best \
  --p4-datasets MNIST,Fashion-MNIST,KMNIST \
  --p4-seeds 0,1,2 \
  --p4-steps-list 50,240 \
  --p4-test-size 1000
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R5-FunctionalUnsafe",
  "base_candidate": "LQ-t2-h256",
  "best_direction": "D9-SignalChannelProjection",
  "best_gate": "G1-RoleSNROnly",
  "best_step_fraction": 0.01,
  "best_predictor": "PRED0-AllAcceptP2Survivor",
  "p0_repeat_pass": 1,
  "p2_survivor_count": 3,
  "p2_non_holdout_survivor_count": 3,
  "p3_calibration_pass": 1,
  "p4_short_run_pass": 0,
  "primary_blocker": "P4_short_run_functional_safety_failed"
}
```

判断：

1. v9.2.10 确实修复了 v9.2.9 的 one-step predictor/direction blocker。
2. 但修复只在 one-step holdout audit 成立；进入 short-run 后，没有形成可接受的系统/控制优势。
3. 因此 route 不是 success，也不是 functional advantage，而是 `R5-FunctionalUnsafe`。

## 3. P0 v9.2.9 failure reproduction

Artifact：

```text
p0_v929_p2_reproduction.csv
```

复现对象：

```text
D1-QuadraticCoeffDamping + G1-RoleSNROnly + rho0.10
```

结果：

| metric | value |
|---|---:|
| rows | `144` |
| accepted rows | `144` |
| bad-step rate | `0.819444` |
| holdout non-harm | `0.180556` |
| prediction corr | `0.231090` |
| mean holdout delta | `+8.309467e-07` |
| amortized overhead | `0.126000` |
| P2 safety pass | `0` |

判断：v9.2.9 的 P2 failure 稳定复现，不是旧 artifact 的随机误差。

## 4. P2 direction factory

Artifact：

```text
p2_direction_factory_one_step.csv
p2_direction_factory_summary.csv
```

P2 测试范围：

```text
directions = D1-D9 + NoOp + RandomMatchedNorm
gates = G1, G2, G4
step fractions = 0.01, 0.03, 0.10
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
candidates = LQ0,LQ1
```

P2 survivor 只有 `D9-SignalChannelProjection + G1-RoleSNROnly`：

| direction | gate | rho | bad-step | non-harm | corr | mean holdout delta | CEp99 delta | margin p10 delta | overhead | survivor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D9 | G1 | `0.01` | `0.027778` | `0.972222` | `0.649843` | `-6.535815e-07` | `-5.644229e-06` | `+2.882236e-06` | `0.126000` | 1 |
| D9 | G1 | `0.03` | `0.020833` | `0.979167` | `0.651083` | `-2.019314e-06` | `-1.678119e-05` | `+9.215333e-06` | `0.126000` | 1 |
| D9 | G1 | `0.10` | `0.020833` | `0.979167` | `0.652009` | `-6.732013e-06` | `-5.611943e-05` | `+3.224073e-05` | `0.126000` | 1 |

Holdout precheck rows 也很安全，但系统开销失败：

| direction | gate | rho | bad-step | non-harm | corr | overhead | survivor |
|---|---|---:|---:|---:|---:|---:|---:|
| D9 | G4 | `0.01` | `0.000000` | `1.000000` | `0.753482` | `7.550690` | 0 |
| D9 | G4 | `0.03` | `0.000000` | `1.000000` | `0.754980` | `7.564754` | 0 |
| D9 | G4 | `0.10` | `0.000000` | `1.000000` | `0.762471` | `7.548265` | 0 |

判断：

1. D9 是本轮关键修复：它不是继续 damping quadratic coeff，而是把 functional direction 投影到当前 task-gradient signal channel 上。
2. 这说明 v9.2.9 的主要问题不是 SNR estimator，而是 direction 本身和 population-risk predictor 错位。
3. 但 D9 的机制也值得警惕：它非常接近“额外的一小步任务梯度方向”，因此必须经 P4 controls 验证是否有真正 functional causality。

## 5. P3 calibrated predictor

Artifact：

```text
p3_calibrated_functional_predictor.csv
```

P3 结果：

| predictor | coverage | bad-step | non-harm | mean holdout delta | AUC | pass |
|---|---:|---:|---:|---:|---:|---:|
| PRED0-AllAcceptP2Survivor | `1.000000` | `0.027778` | `0.972222` | `-6.535815e-07` | `0.500000` | 1 |
| PRED1-PredictedImprovementScore | `0.909722` | `0.030534` | `0.969466` | `-7.184407e-07` | `0.058929` | 1 |
| PRED2-HighSNRTopQuartile | `0.256944` | `0.054054` | `0.945946` | `-1.511058e-06` | `0.412500` | 0 |
| PRED3-MarginRiskBottomQuartile | `0.256944` | `0.000000` | `1.000000` | `-1.565830e-06` | `0.501786` | 1 |

判断：

1. P3 通过的主要原因不是一个强 ranking predictor，而是 D9 one-step 方向本身已经 high precision。
2. `PRED0` 的 AUC 为 `0.5`，所以不能说它学到了强排序；它只是说明 “全接受 D9 survivor” 在 P2 rows 上已经足够安全。
3. `PRED3` 表明 margin-risk abstention 有信号，但样本仍少，不能外推到 full training。

## 6. P4 short-run functional safety

Artifact：

```text
p4_short_run_functional_safety.csv
functional_event_trace_v9210.csv
```

P4 protocol：

```text
candidate = LQ0-LQ-t2-h256
horizons = 50,240 steps
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
modes = AdamWOnly, BestFunctional, GeometryD1Control, RandomMatchedControl
BestFunctional = D9 + G1 + rho0.01
```

Aggregate by horizon：

| horizon | mode | acc delta | loss delta | CEp99 delta | margin delta | curvature delta | step ratio | memory ratio | task-safe | mechanism | system | P4 pass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | BestFunctional | `+0.000111` | `-0.000262` | `-0.005691` | `+0.005776` | `+1.610e-07` | `1.429962` | `1.006463` | `9/9` | `9/9` | `0/9` | `0/9` |
| 50 | GeometryD1Control | `+0.000222` | `-0.000039` | `-0.007658` | `+0.003386` | `-6.136e-06` | `1.348180` | `1.006454` | `9/9` | `9/9` | `0/9` | `0/9` |
| 50 | RandomMatchedControl | `+0.000000` | `-0.000411` | `-0.007958` | `-0.000909` | `+4.433e-07` | `1.323544` | `1.006314` | `9/9` | `9/9` | `0/9` | `0/9` |
| 240 | BestFunctional | `-0.000333` | `-0.001022` | `+0.013035` | `+0.009931` | `+4.116e-06` | `1.446869` | `1.006463` | `9/9` | `7/9` | `0/9` | `0/9` |
| 240 | GeometryD1Control | `-0.000889` | `+0.001298` | `+0.054372` | `+0.002062` | `-2.071e-05` | `1.358484` | `1.006454` | `8/9` | `9/9` | `0/9` | `0/9` |
| 240 | RandomMatchedControl | `+0.000111` | `+0.001008` | `+0.029286` | `+0.014728` | `+5.686e-06` | `1.327466` | `1.006314` | `9/9` | `8/9` | `0/9` | `0/9` |

BestFunctional by dataset：

| dataset | horizon | acc delta | loss delta | CEp99 delta | margin delta | step ratio |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 50 | `+0.000000` | `-0.000069` | `+0.003870` | `+0.010407` | `1.442208` |
| MNIST | 240 | `+0.000333` | `-0.000084` | `+0.024239` | `+0.044245` | `1.453420` |
| Fashion-MNIST | 50 | `+0.000333` | `-0.000762` | `-0.019955` | `+0.001237` | `1.404678` |
| Fashion-MNIST | 240 | `-0.000667` | `-0.003318` | `-0.011152` | `+0.007492` | `1.432137` |
| KMNIST | 50 | `+0.000000` | `+0.000046` | `-0.000990` | `+0.005683` | `1.443001` |
| KMNIST | 240 | `-0.000667` | `+0.000336` | `+0.026019` | `-0.021943` | `1.455050` |

判断：

1. BestFunctional 没有明显伤害 accuracy，task-safe 是成立的。
2. 但 short-run system gate 不成立：step time ratio 稳定在 `1.40-1.46x`，明显超过本轮 `<=1.20` gate。
3. 机制收益不稳定：50 steps 有 CEp99/margin 改善；240 steps 下 CEp99 在 MNIST/KMNIST 变差，KMNIST margin 也变差。
4. 控制组没有被清楚击败：GeometryD1Control 和 RandomMatchedControl 在多个 aggregate 指标上给出相近变化，说明 D9 的 functional causality 不够干净。
5. 因此 P4 失败是合理 terminal boundary，不应继续打开 P5。

## 7. Downstream boundary

这些 artifact 均已落盘，但明确为 `not_run`：

| artifact | status | reason |
|---|---|---|
| `p5_full_functional_reentry_10seed.csv` | `not_run` | `P4_short_run_functional_safety_failed` |
| `p6_noise_robustness_diagnostic.csv` | `not_run` | `P4_short_run_functional_safety_failed` |
| `p7_strong_baseline_external_ready.csv` | `not_run` | `P4_short_run_functional_safety_failed` |

判断：没有用 full training 或 external validation 越过 P4 short-run safety gate。

## 8. No-fake audit

`v9210_provenance_audit.csv`：

```text
rows_checked = 12734
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1/P2/P3/P4 rows 均来自本轮真实 runner 或 v9.2.9 source artifact recap。
2. P5-P7 是明确 gate-blocked `not_run`，没有伪装成通过。
3. 本轮仍保持 CE-only，无 teacher/distillation/geometry loss/sampler/class weight/offload/loss.backward。

## 9. Hash

| artifact | SHA256 |
|---|---|
| v9.2.10 plan | `88b961bec70d72c8cc99bba21649341fddaf8230ec014b01abb943f6b965df3f` |
| `experiments/run_v9210_functional_predictor_repair.py` | `98d03d3e89e82bd9c935d289a5e4863491e6c2734e69fac78cc430cd01a3a2d4` |
| `dgkan/functional/lq_functional_predictor.py` | `55dafc1e8165c9cbc5156a292915c48b82dde1318f776bd5572a5a2bea90de83` |
| `dgkan/functional/snr_gated_lq.py` | `cd04043742c9e82e5e6a0bbb9d71010133bcd4c7753707cbf28e202087295c5e` |
| `dgkan/models/fc_purekan_lq.py` | `feb2bb205b0ad11155bca1c74d9f938d60cd6821c38a8c99a30c8db067669f8c` |
| route | `98f24ffc03034f971c8a8bad317835bee6fbb4041d0b3e70b51697ba29fd44b3` |
| `p2_direction_factory_summary.csv` | `807e0a98fc9074c3f987c4492fdaaf86a5c426b29af7098cd066cf3df3730a63` |
| `p3_calibrated_functional_predictor.csv` | `1ec6b2262fa4570379c2803d594767e9143cd69bae01f8d42977eae6f22083d9` |
| `p4_short_run_functional_safety.csv` | `bcb42437f4e7744fc0645f5fb766e0b9066f10ca35989cd6adbd5a7876038703` |
| failure table | `b423f90274efb126e347595a97152dbaa16b1439581a9e7db77290f484832219` |
| provenance audit | `1686a8df59e922b3f9dc4c9fe9bae613ca2ad11ba8a580749a480ef48cc91f26` |

## 10. 最终分析结论

v9.2.10 的真实推进是：

```text
v9.2.9: low-cost SNR gate 过 P1，但原始 quadratic damping direction 过不了 P2。
v9.2.10: D9 signal-channel projection 过了 P2/P3，但过不了 P4 short-run safety/system/control。
```

机制判断：

1. 本轮证实了问题本质：v9.2.9 失败不是因为 SNR gate 太贵或完全无信号，而是 functional direction 不对。
2. D9 把 direction 改成 task-signal-aligned 后，one-step holdout safety 大幅改善：bad-step 从 `0.819444` 降到 `0.027778`。
3. 但 D9 也暴露出另一个本质问题：它太像一小步额外 task-gradient projection，缺乏独立 functional geometry 含义；进入 short-run 后收益很小，controls 也能得到类似变化。
4. P4 的主要硬 blocker 是系统开销：BestFunctional step ratio `1.43-1.45x`，说明即使 role-scalar SNR 本身便宜，direction construction / projection / per-step functional branch 仍然不够轻。
5. P4 的次要 blocker 是 causality：RandomMatchedControl 和 GeometryD1Control 没有被明确击败，因此不能说 D9 functional update 真的带来可归因 advantage。
6. 当前道路的正确部分是 “先 one-step population-risk，再 short-run，再 full re-entry”；错误部分是把 functional direction 做成 task-gradient shadow。下一步应重新定义 functional update 的几何目标，或者把 functional update 改成低频、事件级、离线校准的结构化校正，而不是每步附加一个小 task-aligned delta。

最终一句话：

> v9.2.10 真实执行后停在 `R5-FunctionalUnsafe`：D9 修复了 P2 one-step safety，但 P4 short-run 没有通过，主要因为 step time `~1.43x` 且控制组未被明确击败。因此 functional advantage 和 external readiness 仍不能打开。

