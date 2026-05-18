# DG-KAN v9.2.15 Control-Resistant Functional Causality 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.15_ControlResistant_Functional_Causality_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P4-P7 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.15 执行到一个可审计 terminal route：

```text
route = R10-ReturnToTargetOrPrimitiveDesign
base_candidate = LQ-t2-h256
success_v9215_paired_replay_causality = false
success_v9215_short_run = false
success_v9215_full_functional = false
success_v9215_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9215_control_resistant_functional_causality_first_20260510T030000Z/
```

核心结论：

1. v9.2.14 boundary 被真实复现：A7c 仍是 P4-qualified actuator，base near-pass `8/9`，但 source paired replay 中 RealFunctional 明显弱于 `AdamWParallelDirection` control。
2. P2 全矩阵已真实执行：`4 actuators x 5 targets x 3 solvers x 4 events x 3 datasets x 3 seeds x 4 horizons x 6 branches = 51840 rows`。
3. 没有任何 actuator / target / solver / event 组合在聚合口径下击败 best matched control：`p2_survivor_count = 0`。
4. 只有 `45/8640` 个 RealFunctional row 达到 row-level beat，全部集中在 `A4e + KMNIST + horizon80`；聚合后不成立，因此不能写成 causality survivor。
5. P3 control-contrastive posthoc 聚合也没有通过：`p3_pass_count = 0/240`。
6. 因 P2/P3 均无 survivor，P4 short-run、P5 full re-entry、P6 robustness、P7 external-ready 全部明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9215_control_resistant_functional_causality.py` | v9.2.15 runner；生成 P0-P7 required artifacts、P2 full causality matrix、route、failure/no-fake audit |

代码检查：

```bash
python -m py_compile experiments/run_v9215_control_resistant_functional_causality.py
```

已通过。

正式运行：

```bash
python experiments/run_v9215_control_resistant_functional_causality.py \
  --out-dir results/real_rerun_20260506/v9215_control_resistant_functional_causality_first_20260510T030000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --lr 0.0005 \
  --batch-size 128 \
  --train-size 9984 \
  --audit-batch-size 128 \
  --p2-eval-size 512 \
  --p2-actuators A4b-BoundedRational-BranchlessDerivative,A4d-BoundedRational-ValueOnlyActuator,A4e-BoundedRational-FusedCoeffGrad,A7c-BasisEntropy-ValueOnly \
  --p2-targets O1-HardTailLogitCorrection,O2-MarginTailExpansion,O3-CalibrationTailCompression,O4-CurvatureOutputFlattening,O6-KMNISTHardModeOutputTarget \
  --p2-solvers SOL1-LeastSquaresSketch,SOL2-ConstrainedLeastSquares,SOL3-TrustRegionQP \
  --p2-events E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E6-CompositeSparseEvent,E7-KMNISTHardModeEvent \
  --p2-datasets MNIST,Fashion-MNIST,KMNIST \
  --p2-seeds 0,1,2 \
  --p2-horizons 1,5,20,80 \
  --p2-warmup-steps 36 \
  --p2-functional-step-fraction 0.10 \
  --p2-trust-fraction 0.03
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R10-ReturnToTargetOrPrimitiveDesign",
  "base_candidate": "LQ-t2-h256",
  "p2_paired_replay_pass": 0,
  "p2_survivor_count": 0,
  "p3_control_contrastive_pass": 0,
  "p3_pass_count": 0,
  "real_mean_CEp99_delta": 0.00012183332884753193,
  "best_control": "AdamWParallelDirection",
  "best_control_mean_CEp99_delta": -0.0011878841453128392,
  "real_mean_margin_delta": 0.00011592911582233177,
  "best_control_mean_margin_delta": 0.0004845593098757996,
  "primary_blocker": "all_P4_qualified_actuators_remain_control_equivalent"
}
```

判断：

1. 本轮不是 P4 system failure；P4-qualified actuator 已经来自 v9.2.14。
2. 本轮也不是 target/solver 没有被测；P2 覆盖了计划中的 A4b/A4d/A4e/A7c、O1/O2/O3/O4/O6、SOL1/SOL2/SOL3、E1/E2/E6/E7。
3. 失败点是 control-resistant causality：RealFunctional 平均 CE tail 和 margin 都没有超过 best control。

## 3. P0 v9.2.14 boundary reproduction

Artifact：

```text
p0_v9214_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `R4-ActuatorP4ClosedBaseQualified` |
| best actuator | `A7c-BasisEntropy-ValueOnly` |
| A7c forward q90 | `1.1046384912` |
| A7c backward q90 | `1.3871015148` |
| A7c step q90 | `1.2131227302` |
| A7c memory | `0.9697312056` |
| A7c target fit R2 | `1.0` |
| A7c rz | `1.1065676840` |
| A7c base near-pass | `8` |
| A7c macro delta | `-0.0041666627` |

v9.2.14 paired replay source：

| source metric | RealFunctional | best control |
|---|---:|---:|
| CEp99 mean delta | `+0.0017473075` | `-0.0085636576` |
| margin mean delta | `-0.0003253869` | `+0.0009430163` |

判断：v9.2.15 的出发点成立。A7c P4/base 是真的，但 source functional causality 没成立。

## 4. P1 paired replay failure autopsy

Artifact：

```text
p1_paired_replay_failure_autopsy.csv
```

P1 从 v9.2.14 source P5 artifact 做 autopsy，不补造 source 没有记录的字段：

```text
cos_delta_adamw = not_measured_in_v9214_source
delta_norm = not_measured_in_v9214_source
rz = not_measured_in_v9214_p5
curvature_delta = not_measured_in_v9214_source
```

失败归因：

```text
p1_failure_factor = M4-control_dominance,M5-scale_failure
best_control = AdamWParallelDirection
```

按 target 的 RealFunctional source rows：

| target | CEp99 delta | margin delta | bad event |
|---|---:|---:|---:|
| O1 | `+0.0004068812` | `+0.0010316544` | 1 |
| O2 | `+0.0032847987` | `-0.0009884441` | 0 |
| O3 | `+0.0044568910` | `-0.0018128972` | 0 |
| O4 | `+0.0004706780` | `+0.0001531864` | 0 |
| O6 | `+0.0001172887` | `-0.0000104341` | 0 |

判断：P1 指向 control dominance，而不是简单的 task collapse。

## 5. P2 P4-qualified actuator causality matrix

Artifacts：

```text
p2_p4qualified_actuator_causality_matrix.csv
paired_replay_branch_trace_v9215.csv
control_rank_trace_v9215.csv
functional_event_trace_v9215.csv
```

总量：

```text
P2 rows = 51840
RealFunctional rows = 8640
row-level real_beats_best_control = 45/8640
aggregate survivor groups = 0
```

Branch aggregate：

| branch | rows | CEp99 delta | margin delta | NLL delta | acc delta | task-safe |
|---|---:|---:|---:|---:|---:|---:|
| AdamWOnly | 8640 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |
| RealFunctional | 8640 | `+0.0001218333` | `+0.0001159291` | `+0.0000008718` | `+0.0000076859` | `1.000000` |
| AdamWParallelDirection | 8640 | `-0.0011878841` | `+0.0004845593` | `+0.0001436396` | `-0.0002441406` | `0.986111` |
| RandomMatchedNorm | 8640 | `-0.0000419408` | `+0.0000226421` | `+0.0000083425` | `+0.0000463415` | `1.000000` |
| ShuffledTarget | 8640 | `+0.0000911933` | `+0.0002301994` | `+0.0000033766` | `+0.0000171803` | `1.000000` |
| NoOpMatchedOverhead | 8640 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |

RealFunctional by actuator：

| actuator | rows | CEp99 delta | margin delta | NLL delta | acc delta | row beats |
|---|---:|---:|---:|---:|---:|---:|
| A4b | 2160 | `+0.0003152044` | `+0.0001623342` | `-0.0000066423` | `+0.0000488281` | 0 |
| A4d | 2160 | `-0.0000600731` | `+0.0000009501` | `+0.0000001438` | `0.0000000000` | 0 |
| A4e | 2160 | `+0.0002183186` | `+0.0002609671` | `+0.0000094120` | `-0.0000180845` | 45 |
| A7c | 2160 | `+0.0000138835` | `+0.0000394651` | `+0.0000005737` | `0.0000000000` | 0 |

Row-level beat 分布：

```text
45 rows all from:
  actuator = A4e-BoundedRational-FusedCoeffGrad
  dataset = KMNIST
  horizon = 80
```

对应 target / solver / event 分布：

| dimension | values |
|---|---|
| targets | O1 `12`, O2 `12`, O6 `12`, O3 `9` |
| solvers | SOL1 `15`, SOL2 `15`, SOL3 `15` |
| events | E2 `12`, E6 `12`, E7 `12`, E1 `9` |

RealFunctional by dataset：

| dataset | rows | CEp99 delta | margin delta | NLL delta | acc delta | row beats |
|---|---:|---:|---:|---:|---:|---:|
| MNIST | 2880 | `-0.0000681404` | `+0.0000085057` | `-0.0000001066` | `0.0000000000` | 0 |
| Fashion-MNIST | 2880 | `+0.0002844882` | `+0.0002796355` | `+0.0000003013` | `+0.0000230577` | 0 |
| KMNIST | 2880 | `+0.0001491522` | `+0.0000596462` | `+0.0000024207` | `0.0000000000` | 45 |

RealFunctional by horizon：

| horizon | rows | CEp99 delta | margin delta | NLL delta | acc delta | row beats |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2160 | `-0.0000001220` | `+0.0000004357` | `-0.0000000633` | `0.0000000000` | 0 |
| 5 | 2160 | `-0.0000008326` | `+0.0000003119` | `-0.0000000250` | `0.0000000000` | 0 |
| 20 | 2160 | `-0.0000059541` | `+0.0000008085` | `-0.0000001409` | `0.0000000000` | 0 |
| 80 | 2160 | `+0.0004942419` | `+0.0004621604` | `+0.0000037164` | `+0.0000307436` | 45 |

判断：

1. RealFunctional 是 task-safe，但机制收益不够强。
2. A4d/A7c 的 CE tail 接近中性；A4e 有少量 KMNIST horizon80 row-level signal，但不是跨组合稳定优势。
3. `AdamWParallelDirection` 仍是 strongest matched control，CEp99 平均改善 `-0.0011878841`，强于 RealFunctional。
4. 因 group-level survivor 为 0，P2 paired replay causality 不成立。

## 6. P3 control-contrastive target / solver

Artifact：

```text
p3_control_contrastive_target_solver.csv
```

结果：

```text
rows = 240
control_contrastive_pass = 0
```

Top posthoc score rows 仍未过：

| actuator | target | solver | event | score | bad event | coverage | CE vs control | margin vs control |
|---|---|---|---|---:|---:|---:|---:|---:|
| A4d | O1 | SOL1 | E7 | `-0.000065` | `0.000000` | `0.101563` | `+0.000013` | `-0.000518` |
| A4d | O1 | SOL2 | E7 | `-0.000065` | `0.000000` | `0.101563` | `+0.000013` | `-0.000518` |
| A4d | O6 | SOL1 | E6 | `-0.000065` | `0.000000` | `0.126736` | `+0.000013` | `-0.000518` |

判断：P3 没有发现 control-contrastive survivor。最好的 posthoc rows 也只是非常接近 0，而不是正向击败 controls。

## 7. P4-P7 downstream boundary

这些 artifact 已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p4_short_run_causal_validation.csv` | `all_P4_qualified_actuators_remain_control_equivalent` |
| `p5_full_functional_reentry_10seed.csv` | same |
| `p6_noise_robustness_signal_separation.csv` | same |
| `p7_strong_baseline_external_ready.csv` | same |

判断：没有用 short-run、full training、robustness 或 external validation 越过 P2/P3 causality gate。

## 8. No-fake audit

`v9215_provenance_audit.csv`：

```text
rows_checked = 207612
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1 是 v9.2.14 source artifact recap / autopsy，不伪装成新训练。
2. P2 是本轮真实 full matrix replay。
3. P3 是从 P2 measured rows 生成的 posthoc control-contrastive summary，没有把它伪装成 SOL4-SOL6 新 solver 成功。
4. P4-P7 是明确 gate-blocked `not_run`。

## 9. Hash

| artifact | SHA256 |
|---|---|
| v9.2.15 plan | `a94df4bf814540adf4002005e75992bee8c8f63cd886a2f12d79bb08b5325143` |
| `experiments/run_v9215_control_resistant_functional_causality.py` | `cedf587f94d734a2cf14324b722ecb2237494618cd38b525c27636d46b7a9127` |
| route | `d59acd6158433a56482eb506af0fce7bfb8c4f6810278a5cd9737637481b7e0c` |
| P0 recap | `8bc443fa6305822b76a8ae581008187f321ede73f826af97a934abf3d0181c81` |
| P1 autopsy | `2f80c40cf4aecd247107941c527ce6fcdd65b747a2cedefc4b08b09c2cd90cb8` |
| P2 matrix | `316eb02f55bb72ed06834075ebe6ee3f5953c8959fdeaef8e6c4b98e2d5661a0` |
| P3 control-contrastive | `938a31ccba17018726a82b748c278609bbec539010166824f6401d33fa34abfd` |
| failure table | `30a857106bc60dd58234adaf14360b8bc6a2bec58904c2d51ca502c201a09552` |
| provenance audit | `61fcd278a1749646abcc52788383dd557ccb573c216dcc1464ac23bc4464630d` |

## 10. 最终分析结论

v9.2.15 的真实推进是：

```text
v9.2.14: P4-qualified actuator exists, but A7c paired replay loses to controls.
v9.2.15: all P4-qualified actuators / targets / solvers / events were tested;
          no aggregate RealFunctional survivor beats matched controls.
```

机制判断：

1. 当前问题已经不是 actuator system 或 actuator controllability：A4/A7c family 都能进入 P2 full matrix。
2. 当前问题也不是单一 A7c 选择错误：A4b/A4d/A4e/A7c 全部没有形成 aggregate causality survivor。
3. P2 中 RealFunctional 的平均变化非常小，且 CE tail 仍弱于 `AdamWParallelDirection` control；这说明当前 functional update 的可见收益仍可被 task-gradient-parallel control 解释。
4. A4e 在 KMNIST horizon80 出现 `45` 个 row-level beat，是一个值得记录的局部信号，但它太窄，不能支持打开 short-run/full-run。
5. P3 posthoc control-contrastive score 也未转正，说明继续在当前 O1-O6 / SOL1-SOL3 / E1-E7 空间内小调不太可能解决本质问题。
6. 下一步应回到 target/primitive design：要么构造真正 control-resistant 的 output target / solver，要么重新设计 functional actuator 的机制，使其产生 task-gradient control 解释不了的结构性变化。

最终一句话：

> v9.2.15 真实执行后停在 `R10-ReturnToTargetOrPrimitiveDesign`：P4-qualified actuator 已经不是 blocker，但全矩阵 paired replay 没有找到任何聚合口径下击败 controls 的 RealFunctional survivor；functional causality 主线不能继续小修阈值，应返回 target 或 primitive 设计。
