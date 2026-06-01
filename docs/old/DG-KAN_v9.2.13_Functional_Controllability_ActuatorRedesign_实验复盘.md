# DG-KAN v9.2.13 Functional Controllability 与 PureKAN Actuator Redesign 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.13_Functional_Controllability_ActuatorRedesign_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 P5-P8 未打开阶段写成通过。

## 0. 最新结论

截至本轮，v9.2.13 执行到一个可审计 terminal route：

```text
route = R8-NoPureKANActuatorFound
base_candidate = LQ-t2-h256
success_v9213_controllability_audit = true
success_v9213_actuator_base = false
success_v9213_actuator_controllability = false
success_v9213_functional_advantage = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9213_functional_controllability_actuator_redesign_first_20260510T000000Z/
```

核心结论：

1. v9.2.12 terminal boundary 被复现：source route `R5-ControlEquivalentAgain`，`p2_output_direction_pass_count=0`，event coverage 仍在 `0.033333-0.100000`。
2. Direct logit oracle 通过：O1/O2/O3/O4/O6 每个 target 的 `27/27` rows 均有 useful signal，说明当前主要问题不是 output target 完全无效。
3. Current LQ raw LS 可拟合 target：raw `max R2=0.999995`、raw `max rz=49.254015`；但 task-safe / constrained 后 usable movement 不足：safe `max R2=0.189644`、safe `max rz=0.040227`，仍未达到 `rz>=0.05`。
4. A0-A7 actuator contract / gradcheck 均真实测量；A4/A5/A6/A7 保留 pairwise interaction，但没有任何 actuator 同时通过 P4 system gate。
5. Post-actuator LS controllability 有信号：`273/360` rows 达到 actuator controllability pass；代表性 A4 rows 的 `R2≈1.0` 且 `rz` 可到 `1.155372`。
6. 但 P4 是硬 blocker：所有 actuator P4 rows 均 fail，因此 P5 AdamW base qualification、P6 paired replay、P7 full re-entry、P8 external-ready 均不允许打开。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `dgkan/models/fc_purekan_actuator.py` | strict FC-PureKAN edge-owned actuator basis、manual forward/backward、LS controllability 工具 |
| `experiments/run_v9213_functional_controllability_actuator_redesign.py` | v9.2.13 runner；生成 P0-P8 required artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9213_functional_controllability_actuator_redesign.py dgkan/models/fc_purekan_actuator.py
```

已通过。

正式运行：

```bash
python experiments/run_v9213_functional_controllability_actuator_redesign.py \
  --out-dir results/real_rerun_20260506/v9213_functional_controllability_actuator_redesign_first_20260510T000000Z \
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
  --p4-reps 20
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R8-NoPureKANActuatorFound",
  "base_candidate": "LQ-t2-h256",
  "direct_logit_oracle_pass": 1,
  "current_lq_controllability_pass": 0,
  "current_lq_best_target_fit_R2": 0.18964403867721558,
  "current_lq_best_output_displacement_ratio": 0.040227413979989614,
  "current_lq_best_raw_target_fit_R2": 0.9999945759773254,
  "current_lq_best_raw_output_displacement_ratio": 49.254014712137426,
  "actuator_contract_pass": 1,
  "actuator_p4_pass": 0,
  "actuator_p5_near_pass": 0,
  "post_actuator_controllability_pass": 0,
  "primary_blocker": "actuator_controllability_signal_exists_but_no_candidate_passed_P4_system_gate"
}
```

判断：本轮没有进入 P6-P8。不是因为 target 无效，而是 tested actuators 虽有 controllability signal，却没有 P4-qualified candidate。

## 3. P0 v9.2.12 recap

Artifact：

```text
p0_v9212_reproduction.csv
```

| metric | value |
|---|---:|
| source route | `R5-ControlEquivalentAgain` |
| source rows | `116640` |
| source pass count | `0` |
| max output displacement ratio | `0.0360087525` |
| max output target fit R2 | `0.0593199465` |
| event coverage min/max | `0.033333 / 0.100000` |
| P0 reproduction pass | `1` |

判断：v9.2.12 的 control-equivalent boundary 稳定纳入本轮。

## 4. P1 current LQ controllability

Artifact：

```text
p1_functional_controllability_audit.csv
```

关键区分：

| scope | rows | max R2 | max rz | promising rows |
|---|---:|---:|---:|---:|
| raw LS | `195` | `0.999995` | `49.254015` | `117` |
| task-safe / constrained | `390` | `0.189644` | `0.040227` | `0` |

代表性 safe best row：

| target | subspace | solver | R2 | rz | projection ratio | holdout delta |
|---|---|---|---:|---:|---:|---:|
| O4 | S6 | SOL2 | `0.189644` | `0.040227` | `0.001601` | `-0.000082` |

判断：

1. Raw LS 说明 target 在局部 batch 上不是不可拟合。
2. 但 task-safe projection 把 usable movement 基本中和，`rz` 没过 `0.05`。
3. 这比 v9.2.12 更精确：问题不是纯 expressivity，而是 “task-safe controllability” 不足。

## 5. P2 direct logit oracle

Artifact：

```text
p2_direct_logit_oracle_target_audit.csv
```

Target useful counts：

| target | useful rows |
|---|---:|
| O1 HardTailLogitCorrection | `27/27` |
| O2 MarginTailExpansion | `27/27` |
| O3 CalibrationTailCompression | `27/27` |
| O4 CurvatureOutputFlattening | `27/27` |
| O6 KMNISTHardModeOutputTarget | `27/27` |

代表性 best CEp99 改善：

| target | dataset | seed | eps | CEp99 delta | margin delta | acc delta |
|---|---|---:|---:|---:|---:|---:|
| O2 | MNIST | 2 | `0.05` | `-0.998019` | `+0.393449` | `0.0` |
| O6 | KMNIST | 2 | `0.05` | `-0.743345` | `+0.516156` | `0.0` |

判断：output target 本身有 diagnostic usefulness；继续说 “target 全错” 不成立。

## 6. P3 actuator contract / gradcheck

Artifact：

```text
p3_actuator_contract_gradcheck.csv
```

| candidate | GradRelErrMax | pairwise R2 | basis cond | interaction pass |
|---|---:|---:|---:|---:|
| A0 current | `3.662e-06` | `0.976813` | `3.257` | 1 |
| A1 duplicate T2 | `1.652e-07` | `0.771356` | `33763832.0` | 0 |
| A2 normalized T2 | `9.408e-07` | `-2375806.25` | `2117.694` | 0 |
| A3 centered T2 | `1.673e-07` | `-180721.875` | `4167.333` | 0 |
| A4 bounded rational | `1.227e-09` | `0.983633` | `1368.566` | 1 |
| A5 piecewise linear 2 | `1.877e-06` | `0.976545` | `381.232` | 1 |
| A6 local RBF | `2.788e-08` | `0.984878` | `243.746` | 1 |
| A7 basis entropy | `1.007e-06` | `0.984690` | `44.206` | 1 |

判断：A4-A7 是真实 strict edge-basis actuator survivors at contract/grad/interaction level；A1-A3 暴露出 conditioning / interaction 失败。

## 7. P4 actuator system gate

Artifact：

```text
p4_actuator_p4_p5_base_qualification.csv
```

| candidate | forward | backward | step | compact memory | P4 |
|---|---:|---:|---:|---:|---:|
| A0 current | `1.002502` | `0.943293` | `2.248581` | `0.969501` | 0 |
| A1 duplicate T2 | `1.125948` | `1.587883` | `1.222466` | `0.969731` | 0 |
| A2 normalized T2 | `1.327233` | `1.511866` | `1.254456` | `0.969731` | 0 |
| A3 centered T2 | `1.240095` | `1.536814` | `1.242644` | `0.969731` | 0 |
| A4 bounded rational | `1.089077` | `1.514382` | `1.214271` | `0.969731` | 0 |
| A5 piecewise linear 2 | `1.301135` | `1.515704` | `1.253538` | `0.969923` | 0 |
| A6 local RBF | `1.114660` | `1.648638` | `1.222728` | `0.969731` | 0 |
| A7 basis entropy | `1.165031` | `1.544527` | `1.220802` | `0.969731` | 0 |

判断：

1. Memory 不是 blocker，所有 compact memory 都在 gate 内。
2. 主要 blocker 是 backward 或 forward/step，其中 A4/A5/A7 都接近但未过 `backward<=1.50`。
3. 因无 P4 pass candidate，P5 AdamW base qualification 不打开；CSV 中对应 rows 明确 `not_run`，不是把未跑写成失败训练。

## 8. P5 post-actuator controllability

Artifact：

```text
p5_post_actuator_controllability_audit.csv
```

结果：

```text
rows = 360
actuator_controllability_pass = 273
pass candidates = A1,A2,A3,A4,A5,A6,A7
```

代表性 pass rows：

| candidate | target | dataset | seed | R2 | rz | gain vs LQ |
|---|---|---|---:|---:|---:|---:|
| A4 bounded rational | O6 | KMNIST | 0 | `0.9999996` | `1.155372` | `+0.810356` |
| A4 bounded rational | O2 | KMNIST | 0 | `0.9999996` | `0.770263` | `+0.810356` |
| A4 bounded rational | O1 | KMNIST | 1 | `0.9999997` | `0.353909` | `+0.810356` |

判断：actuator route 不是没有可控性信号。相反，多个 actuator-only LS subspaces 能强拟合 output targets；但它们没有通过 P4，所以不能进入 paired replay。

## 9. Downstream boundary

这些 artifact 已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p6_paired_replay_with_actuator.csv` | `P5_post_actuator_controllability_failed_or_base_not_qualified` |
| `p7_functional_reentry_with_actuator.csv` | same |
| `p8_robustness_external_ready_gate.csv` | same |
| `functional_event_trace_v9213.csv` | same |

判断：没有用 functional / robustness / external 越过 P4 base gate。

## 10. No-fake audit

`v9213_provenance_audit.csv`：

```text
rows_checked = 2236
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0 是 v9.2.12 source artifact recap，不伪装成新 rerun。
2. P1/P2/P3/P4/P5 均来自本轮真实 runner。
3. P6-P8 是明确 gate-blocked `not_run`。

## 11. Hash

| artifact | SHA256 |
|---|---|
| v9.2.13 plan | `4457bbf32fb4a1e86daf29f10a095e7f3ca469ed625b068e33a5caf011d29f7c` |
| `experiments/run_v9213_functional_controllability_actuator_redesign.py` | `ab0fa9c63f52c67dd905fb88c1b27d35c71a451a7911be9a4660683148f9efc3` |
| `dgkan/models/fc_purekan_actuator.py` | `bca5a60610a462920cdd53ff62eb7321cbaf506654c3735bd53e85728679c5d3` |
| route | `b26c4e50ec6d9a1636dd65107eb54bb7307be10001e877a3cfa7c054f5de452c` |
| P1 controllability | `9b2c44dfcef378018186cd319988138ea167c6897118c9e7b26d1871412774ce` |
| P2 oracle | `bd1258752b717fdb0ca532e088e16688d68599b6c70525cc3f6ac2ab4be058d4` |
| P3 actuator contract | `98a6ebd8aaba2be1f853d051fead3208b81bc9d842096301c5457e6c6c252154` |
| P4 base qualification | `f26a86eb0b1dac6871c8ae761af75b8e0e41e256d0fb6e482e7b2c07c3111d27` |
| P5 post-actuator controllability | `bd3d7d5ea0b5dfc573c0fff40f677e0fb6082fbc8e3036658fdd1307b191a378` |
| provenance audit | `273823eb72279658924f41658a15a8e8370f6310be3c458483fe45b2c447a013` |

## 12. 最终分析结论

v9.2.13 的真实推进是：

```text
direct logit target 有效；
current LQ raw LS 能拟合 target；
task-safe constrained current LQ 位移仍不足；
edge-owned actuator 能带来 controllability signal；
但 tested actuators 没有 P4-qualified candidate。
```

机制判断：

1. v9.2.12 的 “output effect too small” 不应解释成 target 全错；P2 证明 target 直接作用在 logits 上有明确 CE tail / margin 改善。
2. Current LQ 的 raw output coefficient LS 有很高 fit，但一旦加入 task-safe / constrained condition，有效位移被压到 `rz=0.040227`，未过 `0.05`；这说明 “安全可控性” 才是本质。
3. A4-A7 这类新 actuator 确实能形成可控 output subspace，尤其 A4 在 KMNIST hard target 上很强；但 P4 backward/forward envelope 没闭合。
4. 因此 v9.2.13 不是 functional target failure，也不是纯 event failure，而是 “strict PureKAN actuator 的 controllability-system joint design” 还没成立。
5. 下一步不该继续调 SNR / event threshold；应回到 actuator basis factory，优先做能保持 A4/A7 controllability 但满足 P4 的 fused/cheaper derivative actuator，或把 actuator 设计成低频离线 reparameterization，而不是 per-step extra basis backward。

最终一句话：

> v9.2.13 真实执行后停在 `R8-NoPureKANActuatorFound`：output target 本身有效，actuator-only subspace 也有强 controllability signal，但没有任何 tested strict FC-PureKAN actuator 通过 P4 system gate，因此 P5/P6/P7/P8 均不能打开。

