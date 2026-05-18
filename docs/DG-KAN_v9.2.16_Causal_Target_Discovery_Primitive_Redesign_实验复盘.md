# DG-KAN v9.2.16 Causal Target Discovery 与 Primitive Redesign 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.16_Causal_Target_Discovery_Primitive_Redesign_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P5-P8 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.16 执行到一个可审计 terminal route：

```text
route = R8-NoExtractableFunctionalAdvantage
base_candidate = LQ-t2-h256
success_v9216_paired_replay_causality = false
success_v9216_short_run = false
success_v9216_full_functional = false
success_v9216_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9216_causal_target_discovery_first_20260510T040000Z/
```

核心结论：

1. v9.2.15 terminal boundary 被复现：source route `R10-ReturnToTargetOrPrimitiveDesign`，`p2_survivor_count = 0`，`p3_pass_count = 0`，source row-level beat 仍集中在 `A4e + KMNIST + horizon80`。
2. P1 control-dominance autopsy 表明 best control 主要是 `AdamWParallelDirection`：`18007/34560` metric groups 由它获胜；四个指标 CEp99/ECE/Margin/NLL 中它都排第一。
3. P2 对 `A4e + KMNIST + horizon20/80/160` 做了 dedicated replication：`8640` rows，RealFunctional row-level beat `0`，因此 v9.2.15 的 `45` 个 row-level beat 没有复现。
4. P3 control-derived target discovery 做了在线 prototype replay：`3024` rows，CD1-CD7 均无 survivor；这些 prototype 没有全局聚类向量，因此 artifact 明确记录 `not_global_clustered_online_rows`，没有伪装成 cluster-quality pass。
5. P4 control-contrastive solver validation 有 `123` 个 row-level pass-like rows，但整体 prediction correlation 为 `-0.080205`，所以 `control_contrastive_pass = 0`。
6. 因 P2/P3/P4 都没有 paired replay survivor，P5 short-run、P6 full re-entry、P7 robustness、P8 external-ready 全部明确 `not_run`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9216_causal_target_discovery.py` | v9.2.16 runner；生成 P0-P8 required artifacts、A4e delayed-signal replication、control-derived prototype replay、route、failure/no-fake audit |

代码检查：

```bash
python -m py_compile experiments/run_v9216_causal_target_discovery.py
```

已通过。

正式运行：

```bash
python experiments/run_v9216_causal_target_discovery.py \
  --out-dir results/real_rerun_20260506/v9216_causal_target_discovery_first_20260510T040000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314 \
  --lr 0.0005 \
  --batch-size 128 \
  --train-size 9984 \
  --audit-batch-size 128 \
  --eval-size 512 \
  --warmup-steps 36 \
  --functional-step-fraction 0.10 \
  --trust-fraction 0.03 \
  --scale-brackets 0.003,0.01,0.03,0.10 \
  --p2-seeds 0,1,2,3,4,5,6,7,8,9 \
  --p2-horizons 20,80,160 \
  --p3-datasets MNIST,Fashion-MNIST,KMNIST \
  --p3-seeds 0,1,2 \
  --p3-horizons 20,80
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R8-NoExtractableFunctionalAdvantage",
  "base_candidate": "LQ-t2-h256",
  "best_control": "AdamWParallelDirection",
  "control_dominance_type": "D3-AdamWParallelDominance,D6-A4eKMNISTDelayedSignal",
  "a4e_kmnist_h80_signal_pass": 0,
  "a4e_row_level_beat_count": 0,
  "a4e_beat_rate": 0.0,
  "control_derived_target_pass": 0,
  "prototype_row_level_beat_count": 0,
  "control_contrastive_pass": 0,
  "control_contrastive_corr": -0.08020502152020542,
  "primary_blocker": "control_derived_targets_remain_control_equivalent"
}
```

判断：

1. A4e delayed signal 没有复制出来。
2. Control-derived targets 没有形成 paired replay survivor。
3. Control-contrastive solver 的局部 row signal 不具备预测稳定性，因此不能打开 P5。

## 3. P0 v9.2.15 boundary reproduction

Artifact：

```text
p0_v9215_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `R10-ReturnToTargetOrPrimitiveDesign` |
| source P2 pass | `0` |
| source P2 survivor count | `0` |
| source P3 pass count | `0` |
| source row-level beat count | `45` |
| source row-level beat region | `actuator=A4e-BoundedRational-FusedCoeffGrad;dataset=KMNIST;horizon=80` |
| source best control | `AdamWParallelDirection` |
| fake proxy count | `0` |
| P0 pass | `1` |

判断：v9.2.16 是在稳定的 v9.2.15 terminal boundary 上继续，不是重复跑一个不稳定分支。

## 4. P1 control-dominance autopsy

Artifact：

```text
p1_control_dominance_autopsy.csv
```

总量：

```text
rows = 34560
control_dominance_type = D3-AdamWParallelDominance,D6-A4eKMNISTDelayedSignal
```

Best control 分布：

| control | wins |
|---|---:|
| AdamWParallelDirection | `18007` |
| NoOpMatchedOverhead | `8672` |
| RandomMatchedNorm | `4415` |
| ShuffledTarget | `3466` |

按 metric：

| metric | top controls |
|---|---|
| CEp99 | AdamWParallel `4382`, NoOp `2150`, Random `1063`, Shuffled `1045` |
| ECE | AdamWParallel `4013`, NoOp `2310`, Random `1328`, Shuffled `989` |
| MarginP10 | AdamWParallel `4426`, NoOp `2223`, Random `1081`, Shuffled `910` |
| NLL | AdamWParallel `5186`, NoOp `1989`, Random `943`, Shuffled `522` |

判断：

1. Control dominance 不是单一 metric 偶然；AdamWParallel 在四个 metric 都领先。
2. v9.2.15 的 A4e/KMNIST/h80 局部信号仍被记录为 D6 clue，但只是 clue，不是 survivor。

## 5. P2 A4e + KMNIST + horizon signal replication

Artifact：

```text
p2_a4e_kmnist_h80_signal_replication.csv
```

Protocol：

```text
actuator = A4e-BoundedRational-FusedCoeffGrad
dataset = KMNIST
seeds = 0..9
horizons = 20,80,160
targets = O1,O2,O6,CD5
solvers = SOL2,SOL3,SOL4,SOL5
events = E7,E8,E9
branches = AdamWOnly,RealFunctional,NoOpMatchedOverhead,RandomMatchedNorm,ShuffledTarget,AdamWParallelDirection
```

Branch aggregate：

| branch | rows | CEp99 delta | margin delta | NLL delta | acc delta | task-safe |
|---|---:|---:|---:|---:|---:|---:|
| AdamWOnly | 1440 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |
| RealFunctional | 1440 | `-0.0001379708` | `-0.0002426716` | `-0.0000033884` | `-0.0000366211` | `1.000000` |
| AdamWParallelDirection | 1440 | `+0.0014261405` | `+0.0013084928` | `-0.0000166953` | `-0.0000651042` | `1.000000` |
| RandomMatchedNorm | 1440 | `-0.0008478251` | `-0.0005027257` | `-0.0000047974` | `+0.0000027127` | `1.000000` |
| ShuffledTarget | 1440 | `-0.0002204339` | `+0.0001744673` | `+0.0000028859` | `+0.0000515408` | `1.000000` |
| NoOpMatchedOverhead | 1440 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |

RealFunctional by horizon：

| horizon | rows | CEp99 delta | margin delta | row beats |
|---:|---:|---:|---:|---:|
| 20 | 480 | `-0.0000029107` | `+0.0000013272` | 0 |
| 80 | 480 | `+0.0000866463` | `-0.0000398936` | 0 |
| 160 | 480 | `-0.0004976481` | `-0.0006894484` | 0 |

RealFunctional by target：

| target | rows | CEp99 delta | margin delta | row beats |
|---|---:|---:|---:|---:|
| CD5 | 360 | `-0.0001197775` | `-0.0002732688` | 0 |
| O1 | 360 | `-0.0000865830` | `-0.0002000418` | 0 |
| O2 | 360 | `-0.0001854963` | `-0.0002363950` | 0 |
| O6 | 360 | `-0.0001600266` | `-0.0002609809` | 0 |

RealFunctional by solver：

| solver | rows | CEp99 delta | margin delta | row beats |
|---|---:|---:|---:|---:|
| SOL2 | 360 | `-0.0001183629` | `-0.0002628015` | 0 |
| SOL3 | 360 | `-0.0001498964` | `-0.0002070722` | 0 |
| SOL4 | 360 | `-0.0001165933` | `-0.0002562480` | 0 |
| SOL5 | 360 | `-0.0001670308` | `-0.0002445648` | 0 |

判断：

1. v9.2.15 的 `45/8640` row-level beat 没有在 dedicated replication 里复现。
2. A4e/KMNIST/h80 不是可以直接推进的 delayed causal signal。
3. P2 虽有 CEp99 小幅改善，但 margin 变差，且没有击败 controls，因此不能进入 P5。

## 6. P3 control-derived target discovery

Artifacts：

```text
p3_control_derived_target_discovery.csv
prototype_target_trace_v9216.csv
control_displacement_trace_v9216.csv
```

Protocol：

```text
actuators = A4e,A7c
targets = CD1,CD2,CD3,CD4,CD5,CD6,CD7
solvers = SOL2,SOL5
events = E9
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 20,80
```

重要边界：

```text
intra_cluster_cos_mean = not_global_clustered_online_rows
prototype_norm = computed_per_batch_not_stored_as_global_vector
```

说明：本轮 P3 是在线 control-derived prototype replay，没有完成全局 prototype clustering，因此没有把 cluster quality 写成通过。

Branch aggregate：

| branch | rows | CEp99 delta | margin delta | NLL delta | acc delta | task-safe |
|---|---:|---:|---:|---:|---:|---:|
| RealFunctional | 504 | `-0.0004549854` | `+0.0002393404` | `-0.0000465251` | `-0.0000387525` | `1.000000` |
| RandomMatchedNorm | 504 | `-0.0016468778` | `-0.0006924770` | `-0.0000132421` | `-0.0000930060` | `0.982143` |
| AdamWParallelDirection | 504 | `+0.0014424854` | `-0.0011008004` | `-0.0000450234` | `-0.0008680556` | `0.944444` |
| ShuffledTarget | 504 | `-0.0002320343` | `-0.0001163805` | `-0.0000045962` | `+0.0000155010` | `1.000000` |
| NoOpMatchedOverhead | 504 | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `1.000000` |

Prototype summary：

| prototype | CEp99 delta | margin delta | avg R2 | avg rz | row beats |
|---|---:|---:|---:|---:|---:|
| CD1 RandomWinner | `-0.0000401835` | `+0.0001444189` | `0.012335` | `0.001487` | 0 |
| CD2 AdamWParallelWinner | `-0.0008469158` | `+0.0004411546` | `0.008958` | `0.001386` | 0 |
| CD3 NoOpNull | `0.0000000000` | `0.0000000000` | `1.000000` | `0.000000` | 0 |
| CD4 BestControlMixture | `-0.0008462568` | `+0.0004407616` | `0.008992` | `0.001391` | 0 |
| CD5 A4e-H80 | `-0.0008270542` | `+0.0004304489` | `0.007193` | `0.001287` | 0 |
| CD6 ControlContrastiveTail | `-0.0005446672` | `+0.0001868312` | `0.003413` | `0.001224` | 0 |
| CD7 Curvature | `-0.0000798206` | `+0.0000317674` | `0.012251` | `0.001461` | 0 |

By actuator：

| actuator | CEp99 delta | margin delta | row beats |
|---|---:|---:|---:|
| A4e | `-0.0008569842` | `+0.0005929205` | 0 |
| A7c | `-0.0000529866` | `-0.0001142398` | 0 |

判断：

1. CD2/CD4/CD5 的 CEp99 和 margin 看起来比 old targets 更像 “有方向”，但它们没有击败 matched controls。
2. Prototype implementability 很弱：除 CD3 no-op 外，avg R2 约 `0.003-0.012`，avg rz 约 `0.0012-0.0015`，远低于计划的 `R2>=0.20`、`rz>=0.05`。
3. 因此 P3 失败点不是 task safety，而是 prototype 不能形成可实现且 control-resistant 的 functional displacement。

## 7. P4 control-contrastive solver validation

Artifact：

```text
p4_control_contrastive_solver_validation.csv
```

结果：

```text
rows = 1932
row-level control_contrastive_pass = 123
control_contrastive_corr = -0.0802050215
control_contrastive_pass = 0
```

Top actual Scc rows 多集中在 Fashion-MNIST / horizon80，例如：

| solver | target | actuator | dataset | horizon | actual Scc | CE gap vs control | margin gap vs control |
|---|---|---|---|---:|---:|---:|---:|
| SOL5 | CD2 | A4e | Fashion-MNIST | 80 | `0.028346` | `-0.030590` | `-0.022438` |
| SOL5 | CD5 | A4e | Fashion-MNIST | 80 | `0.028346` | `-0.030590` | `-0.022438` |
| SOL4/SOL2 variants | CD4/CD6 | A4e | Fashion-MNIST | 80 | positive local rows | mixed | mixed |

判断：

1. P4 有 row-level positive Scc，但 prediction correlation 为负，说明当前 predicted score 不能可靠选择 survivor。
2. 这些 positive rows 没有满足整体 paired replay survivor 条件，也没有通过 P3 prototype gate。
3. 因此不能打开 P5 short-run。

## 8. P5-P8 downstream boundary

这些 artifact 已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p5_short_run_causal_validation.csv` | `control_derived_targets_remain_control_equivalent` |
| `p6_full_functional_reentry_10seed.csv` | same |
| `p7_noise_robustness_signal_separation.csv` | same |
| `p8_strong_baseline_external_ready.csv` | same |

判断：没有用 short-run、full training、robustness 或 external validation 越过 paired replay/target discovery gate。

## 9. No-fake audit

`v9216_provenance_audit.csv`：

```text
rows_checked = 62858
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1 是 v9.2.15 source artifact recap / autopsy，不伪装成新训练。
2. P2/P3/P4 是本轮真实 replay / prototype / solver validation rows。
3. P3 未实现全局 clustering 的部分明确写为 `not_global_clustered_online_rows`，没有编造成 cluster pass。
4. P5-P8 是明确 gate-blocked `not_run`。

## 10. Hash

| artifact | SHA256 |
|---|---|
| v9.2.16 plan | `a511311cf57bc5d8506082f6ff7cf669d70eac2d89a2f5381025e1655fecaf1e` |
| `experiments/run_v9216_causal_target_discovery.py` | `a513816d3dc4d660475e2e65f2982f3e4bb94e8edbd462176a4b28e521d4c1b6` |
| route | `baae246f1b24a22edc9a7cf2ab5abc4d09421fe83348d29b4276b4bb34ed92c2` |
| P0 recap | `480c6043b694803324f847049dec8f921c30fdf0d0d140c8581b3cbbf2478a72` |
| P1 autopsy | `38395c703f82976ba7ae1434ffb77ec7b4015919a1f8cb2fe15fc2e61634b9f1` |
| P2 A4e replication | `32b14b4ec8a946185815c4373956068434ecceccf2ba6e3468948568c218bd24` |
| P3 target discovery | `82310dfd8989f6495cc280467447197836ea08cecf97feb4fe94af9dfbf6a089` |
| P4 solver validation | `39fec62416948cd9f2cc6e173f0038a55b0843c9e916aa92edc571c1b4290419` |
| prototype trace | `1f15fa674a832a323256188bea7cae341182a2198057c3de5fd90b2bf134c501` |
| failure table | `e0451346e3d4c3d5f8ef89e7b9627e3489f368a0aaa847e02d0e327c874d4dda` |
| provenance audit | `1df8759572bccfd01cc061b6caaf3ceb59f79b71ec656820ed44db627d384eb3` |

## 11. 最终分析结论

v9.2.16 的真实推进是：

```text
v9.2.15: full matrix found no aggregate survivor, but A4e/KMNIST/h80 had 45 row-level beats.
v9.2.16: dedicated A4e/KMNIST replication found 0 row-level beats;
          online control-derived prototypes also found 0 survivor.
```

机制判断：

1. A4e/KMNIST/h80 不是稳定 delayed functional clue；复验中 `row_level_beat_count = 0`。
2. Control-derived targets 能产生一些 CEp99 / margin 小改善，但 avg R2 和 rz 极低，说明它们没有形成可实现的 functional displacement。
3. `AdamWParallelDirection` 的 dominance 仍然是主要事实；这意味着当前 observed gain 更像 task-gradient shadow，而不是 functional target 的独立因果机制。
4. P4 中局部 positive Scc rows 不足以构成 solver success，因为预测相关性为负，不能可靠选择事件或 target。
5. 当前结果支持 H5：在现有 LQ/A4/A7c actuator-target-solver family 下，没有可提取 functional advantage。
6. 下一步不应继续调 event、SNR、step fraction 或现有 output target；应回到 primitive/basis design，或者先做真正的全局 control-prototype clustering，再决定是否值得重开 functional target discovery。

最终一句话：

> v9.2.16 真实执行后停在 `R8-NoExtractableFunctionalAdvantage`：A4e/KMNIST/h80 的局部信号没有复现，control-derived prototype 也没有形成 paired replay survivor；当前 functional target discovery 主线应暂停小修，返回 primitive/basis 或更基础的 target discovery 设计。
