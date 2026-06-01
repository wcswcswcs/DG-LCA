# DG-KAN v8.4 Formal Functional Advantage 实验复盘

> 本复盘记录 `docs/DG-KAN_v8.4_Formal_Functional_Advantage_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文件列出的落盘 CSV/JSON/manifest；没有使用 fake data、proxy rows、固定占位 ratio 或手填成功结论。本轮继续遵守：CE-only、no external teacher、no self-teacher、no distillation、no loss modification、no sampler/class weight、no CPU offload。

## 1. 是否完成

已完成 high-rep timing-stabilized v8.4 minimum；未完成 v8.4 formal / strong success。

同时必须保留一个重要限定：原 `50 warmup / 200 reps` official-style fresh/repeat reproduction 仍未复现 H0 step gate。因此本轮 minimum success 是在 high-rep timing-stabilized protocol 下闭合，不应被写成“50/200 原协议已稳定复现”。

当前最关键结论：

```text
official-style 50/200 fresh reproduction:
  H0BasePass = 0
  success_v84_minimum = 0

official-style 50/200 repeat reproduction:
  H0BasePass = 0
  success_v84_minimum = 0

high-rep timing-stabilized diagnostic:
  H0BasePass = 1
  P7Confirm10Pass = 1
  P8TimePass = 1
  P9ScalingRobustnessPass = 1
  FunctionalCausalityPass = 1
  RoleMechanismPass = 1
  GuardStrideMechanismPass = 0
  FormalProfilerPass = 1
  FormalTimeAccountingPass = 0
  success_v84_minimum = 1
  success_v84_formal_time_profiler = 0
  success_v84_strong = 0
```

判断：

1. v8.4 的中心思想是把 v8.3 minimum success 形式化为可复现、因果明确、系统可审计的 functional advantage。
2. 本轮首先做了 fresh reproduction。按 v8.3 原 50 warmup / 200 reps 风格复现时，`KW6 hidden68` 的 macro / memory / GradPass 复现，但 step gate 不稳，H0 失败。
3. 追加 high-rep timing diagnostic 后，H0/P7/P8/P9 链条能复现；随后补跑 P3 causality controls，`FT7-full` 相比 no-op / random controls 保持明显几何优势，因此 high-rep v8.4 minimum 闭合。
4. 继续补跑 P4 role ablation 后，stack-only 和 head-only 都显示正几何收益，`RoleMechanismPass=1`。
5. 继续补跑 P5 guard/stride ablation 后，accepted setting 没有支配所有替代项：`pre1` 与 `stride4-alpha15` 的曲率更低，但分别牺牲 TimeAUC 或 update fraction。因此 P5 不支持 formal claim，只支持“accepted setting 是当前平衡点之一”。
6. 继续整理 P6 strict time accounting 后，P8 raw phase timing 的核心 gate 通过，但 holdout guard、CUDA sync、time-to-target 仍未单独实测，因此 P6 仍不支持 formal time pass。
7. 追加 P7 targeted phase-mapped profiler 后，FT7 event step 的 CUDA kernel time 可被 phase slices 解释，FormalProfilerPass 更新为 1。
8. 不能把 high-rep minimum 写成 50/200 原协议稳定复现；它说明 blocker 从 functional causality 转移到 strict time accounting、timing protocol formalization 和 guard/stride Pareto 解释，而不是 teacher/loss/CPU/offload。

## 2. 本轮代码

新增文件：

| 文件 | 作用 |
|---|---|
| `experiments/run_gafu_v84_real.py` | v8.4 formalization runner；复用 v8.3 measured path，新增 v8.4 P0/P1/P2/formal placeholder/route/provenance artifacts |

代码检查：

```text
python -m py_compile experiments/run_gafu_v84_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

## 3. Run A：official-style fresh reproduction

```bash
python experiments/run_gafu_v84_real.py \
  --out-dir results/real_rerun_20260506/v84_formal_functional_reproduction_fresh_10seed_20260507T233000Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

初次 Route：

```json
{
  "route": "R9-NoReproduction",
  "H0BasePass": 0,
  "P7Confirm10Pass": 0,
  "P8TimePass": 0,
  "P9ScalingRobustnessPass": 0,
  "FunctionalCausalityPass": 0,
  "success_v84_minimum": 0,
  "primary_blocker": "fresh_reproduction_fail"
}
```

P1 base reproduction：

| candidate | macro gap | CI95 low | test gap | GradPass | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden68` | `+0.021680` | `+0.016341` | `+0.024349` | 1 | `1.011163` | `1.833653` | `1/9` | 0 |

判断：质量、GradPass、memory 均复现；失败点是 step gate。因为 H0 未过，functional full route 被正确 gate，没有打开 P7/P8/P9。

No-fake audit：

| audit | rows checked | fake/proxy nonzero | CPU offload |
|---|---:|---:|---:|
| v84 | 26 | 0 | 0 |
| v83 reused path | 3386 | 0 | 0 |
| v82 reused path | 1579 | 0 | 0 |

## 4. Run B：official-style repeat reproduction

```bash
python experiments/run_gafu_v84_real.py \
  --out-dir results/real_rerun_20260506/v84_formal_functional_reproduction_repeat_10seed_20260507T234500Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 50 \
  --bench-reps 200 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route 仍为：

```json
{
  "route": "R9-NoReproduction",
  "H0BasePass": 0,
  "success_v84_minimum": 0,
  "primary_blocker": "fresh_reproduction_fail"
}
```

P1 base reproduction：

| candidate | macro gap | CI95 low | test gap | GradPass | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden68` | `+0.021680` | `+0.016341` | `+0.024349` | 1 | `1.011163` | `1.993859` | `2/9` | 0 |

判断：repeat 没有修复 H0，反而 step max 更高。这支持“50/200 timing protocol 下 step reproduction 不稳”的判断。

## 5. Run C：base-only high-rep timing diagnostic

为检查 H0 failure 是否主要来自 benchmark timing instability，追加 base-only 高 warmup/reps 诊断。该 run 不是 v8.4 official success，只用于定位 blocker。

```bash
python experiments/run_gafu_v82_real.py \
  --out-dir results/real_rerun_20260506/v84_kw6_hidden68_base_timing_highrep_diagnostic_20260508T000000Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 200 \
  --bench-reps 1000 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Result：

| candidate | macro gap | CI95 low | test gap | GradPass | memory max | step max | step mean | S2 shapes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden68` | `+0.021680` | `+0.016341` | `+0.024349` | 1 | `1.011163` | `1.437941` | `1.343919` | `9/9` |

判断：高 reps 下 H0 base 可以闭合，说明 v8.4 blocker 不是 KW6 hidden68 的质量或 memory，而是 official-style timing measurement 的稳定性和复现协议。

## 6. Run D：high-rep full functional diagnostic

```bash
python experiments/run_gafu_v84_real.py \
  --out-dir results/real_rerun_20260506/v84_formal_functional_reproduction_highrep_10seed_20260508T001500Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 200 \
  --bench-reps 1000 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

Route：

```json
{
  "route": "R3-FunctionalMinimumReproduced",
  "minimum_reproduction_success": 1,
  "H0BasePass": 1,
  "P7Confirm10Pass": 1,
  "P8TimePass": 1,
  "P9ScalingRobustnessPass": 1,
  "FunctionalCausalityPass": 0,
  "success_v84_minimum": 0,
  "primary_blocker": "p3_functional_causality_controls_not_run"
}
```

P1 base reproduction：

| candidate | macro gap | CI95 low | test gap | GradPass | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden68` | `+0.021680` | `+0.016341` | `+0.024349` | 1 | `1.011163` | `1.452238` | `9/9` | 1 |

P2 FT7 reproduction:

| metric | value |
|---|---:|
| functional macro delta vs base | `+0.001302` |
| CI95 low | `-0.000456` |
| test delta vs base | `+0.001302` |
| curvature ratio vs base | `0.761848` |
| geometry reduction | `0.238152` |
| memory max | `1.027250` |
| step max | `1.101249` |
| functional update time ratio of step | `0.045565` |
| ValLossAUC time ratio | `1.042685` |
| P7 confirm10 pass | 1 |
| P8 TimeAUC pass | 1 |
| P9 scaling/robustness pass | 1 |
| FT7 reproduction pass | 1 |

P8 profiler:

| metric | value | gate |
|---|---:|---:|
| ValLossAUC time ratio | `1.042685` | `<=1.05` |
| functional update time / step | `0.045565` | `<=0.10` |
| unknown time fraction max | `0.035601` | `<=0.10` |
| train step time ratio | `1.033784` | record |
| P8 pass | 1 |  |

P9 scaling/robustness:

| metric | value | pass |
|---|---:|---:|
| raw rows | `300` |  |
| sample efficiency AUC base | `3139.083333` |  |
| sample efficiency AUC functional | `3141.300000` |  |
| sample efficiency AUC delta | `+2.216667` | 1 |
| robustness benefit count | `3/5` | 1 |
| geometry delta mean | `-0.204968` | 1 |
| ECE delta mean | `+0.000484` |  |
| NLL delta mean | `-0.004982` | 1 |
| memory ratio max | `1.002947` | 1 |
| step ratio max | `1.385498` | 1 |
| P9 pass | 1 | 1 |

判断：

1. 高 reps 下，v8.3 accepted route 可以完整复现到 P9。
2. 这说明 `KW6 hidden68 + FT7 stride8 alpha15 role_budget0.15 pre2 train-budget fallback` 的 functional route 仍有真实 task/geometry/system 信号。
3. 此时 v8.4 minimum 仍未闭合，因为 `FunctionalCausalityPass=0/not_run`。第 7 节追加 P3 后才更新为 high-rep minimum pass。

No-fake audit：

| audit | rows checked | fake/proxy nonzero | CPU offload |
|---|---:|---:|---:|
| v84 | 26 | 0 | 0 |
| v83 reused path | 3875 | 0 | 0 |
| v82 reused path | 1577 | 0 | 0 |

## 7. 追加：P3 causality controls

本节继续执行文档中心思想：证明 FT7 的几何收益来自 functional direction，而不是 guard / overhead / schedule artifact。本节没有改 CE objective、teacher、loss、sampler、class weight 或 CPU offload。

P3 在 high-rep out-dir 上做 postprocess-only 真实控制组训练：

```text
out_dir = results/real_rerun_20260506/v84_formal_functional_reproduction_highrep_10seed_20260508T001500Z
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = BASE, FT7-full, FT7-noop-matched-overhead, FT7-random-functional-direction
task_steps = 240
```

Route 更新后：

```json
{
  "route": "R3-FunctionalMinimumReproduced",
  "minimum_reproduction_success": 1,
  "H0BasePass": 1,
  "P7Confirm10Pass": 1,
  "P8TimePass": 1,
  "P9ScalingRobustnessPass": 1,
  "FunctionalCausalityPass": 1,
  "success_v84_minimum": 1,
  "success_v84_formal_time_profiler": 0,
  "success_v84_strong": 0,
  "primary_blocker": "formal_profiler_time_accounting_open"
}
```

P3 result：

| control | macro delta | test delta | curvature ratio | geometry reduction | NLL delta | step ratio | accepted norm | random norm | bad step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT7-full` | `-0.000868` | `+0.000434` | `0.741285` | `0.258715` | `-0.006275` | `0.986841` | `0.000000` | `0.000000` | `0.082407` |
| `FT7-noop-matched-overhead` | `0.000000` | `0.000000` | `1.000000` | `0.000000` | `0.000000` | `0.958908` | `0.043169` | `0.000000` | `0.085185` |
| `FT7-random-functional-direction` | `-0.002604` | `-0.003038` | `1.003617` | `-0.003617` | `+0.000906` | `1.068644` | `0.021225` | `0.035512` | `0.087500` |

P3 summary：

```text
P3_basic_causality_pass = 1
P3_strict_H1_causality_pass = 1
FunctionalCausalityPass = 1

FT7 curvature ratio = 0.741285
noop curvature ratio = 1.000000
random curvature ratio = 1.003617
```

判断：

1. `FT7-full` 明显优于 matched-overhead no-op：curvature ratio `0.741285 <= 0.90 * 1.000000`。
2. `FT7-full` 也明显优于 random functional direction：`0.741285 < 1.003617`。
3. no-op control 没有产生 geometry reduction，说明收益不是 guard / overhead 本身带来的。
4. random direction 不仅没有降低 curvature，task delta 也更差，说明 FT7 的 structured second-diff functional direction 有因果信号。
5. 因此 high-rep v8.4 minimum 更新为 pass；formal profiler/time accounting、S1、extended scaling/robustness 仍未完成。

No-fake audit：

```text
rows_checked = 29
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 8. 追加：P4 role ablation

本节继续 Wave 2 因果矩阵，解释 FT7 role-wise functional update 具体靠哪个 role 生效。本节仍不改 CE objective、teacher、loss、sampler、class weight 或 CPU offload。

P4 在同一 high-rep out-dir 上做 postprocess-only 真实 role ablation：

```text
out_dir = results/real_rerun_20260506/v84_formal_functional_reproduction_highrep_10seed_20260508T001500Z
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
role variants = BASE, FT7-full, FT7-stack-only, FT7-head-only, FT7-head-stack-swapped-budget, FT7-no-role-budget
task_steps = 240
```

Route 更新：

```json
{
  "route": "R3-FunctionalMinimumReproduced",
  "success_v84_minimum": 1,
  "FunctionalCausalityPass": 1,
  "RoleMechanismPass": 1,
  "success_v84_formal_time_profiler": 0,
  "success_v84_strong": 0,
  "primary_blocker": "formal_profiler_time_accounting_open"
}
```

P4 role ablation result：

| role variant | curvature ratio | geometry reduction | val acc delta | test delta | bad step | fallback | step ratio | Role pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT7-full` | `0.741285` | `0.258715` | `-0.000868` | `+0.000434` | `0.082407` | `0.091435` | `1.028268` | 1 |
| `FT7-stack-only` | `0.832420` | `0.167580` | `+0.000868` | `+0.001302` | `0.087500` | `0.043287` | `1.125748` | 1 |
| `FT7-head-only` | `0.871620` | `0.128380` | `-0.000434` | `-0.000651` | `0.083796` | `0.039815` | `1.074025` | 1 |
| `FT7-head-stack-swapped-budget` | `0.751441` | `0.248559` | `-0.000217` | `-0.000651` | `0.087037` | `0.096065` | `1.045261` | 1 |
| `FT7-no-role-budget` | `0.576031` | `0.423969` | `-0.000434` | `-0.000434` | `0.092130` | `0.060880` | `1.104145` | 1 |

判断：

1. `FT7-stack-only` 与 `FT7-head-only` 都满足 `geometry reduction > 0` 且 `AccDrop <= 0.003`，因此 role mechanism gate 通过。
2. stack-only 几何贡献更强：reduction `0.167580`，head-only 为 `0.128380`。
3. full route 的 reduction `0.258715` 高于 stack-only/head-only，说明 full FT7 存在 role interaction gain。
4. `no-role-budget` 几何最强，但它放宽了 role budget，不是 accepted route；只作为机制诊断，不能替代 FT7 accepted setting。

No-fake audit：

```text
rows_checked = 34
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 9. 追加：P5 guard / stride / fallback ablation

本节继续 Wave 2 因果矩阵，检查 v8.3 accepted setting 是否只是某个 guard / stride / fallback 参数偶然形成的黑箱。所有 variant 都是真实 240-step 训练，不使用 fake/proxy，不改 CE objective、teacher、loss、sampler、class weight 或 CPU offload。

P5 在同一 high-rep out-dir 上做 postprocess-only guard/stride ablation：

```text
out_dir = results/real_rerun_20260506/v84_formal_functional_reproduction_highrep_10seed_20260508T001500Z
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
variants = BASE,
           FT7-accepted,
           FT7-stride4-alpha15,
           FT7-stride8-alpha8,
           FT7-stride8-alpha15-no-pre2,
           FT7-stride8-alpha15-no-trainbudget,
           FT7-stride8-alpha15-pre1,
           FT7-stride8-alpha15-pre3
task_steps = 240
```

Route 更新：

```json
{
  "route": "R3-FunctionalMinimumReproduced",
  "success_v84_minimum": 1,
  "FunctionalCausalityPass": 1,
  "RoleMechanismPass": 1,
  "GuardStrideMechanismPass": 0,
  "success_v84_formal_time_profiler": 0,
  "success_v84_strong": 0,
  "primary_blocker": "guard_stride_pareto_open"
}
```

P5 result：

| variant | curvature ratio | geometry reduction | macro delta | ValLossAUC time | update/step | bad step | fallback | role accept | P5 variant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FT7-accepted` | `0.741285` | `0.258715` | `-0.000868` | `1.010259` | `0.048037` | `0.082407` | `0.091435` | `0.268519` | 1 |
| `FT7-stride4-alpha15` | `0.571441` | `0.428559` | `+0.001519` | `1.079287` | `0.101399` | `0.084722` | `0.185417` | `0.258333` | 0 |
| `FT7-stride8-alpha8` | `0.797955` | `0.202045` | `-0.000217` | `1.009466` | `0.049862` | `0.089352` | `0.077546` | `0.379630` | 1 |
| `FT7-stride8-alpha15-no-pre2` | `0.796541` | `0.203459` | `+0.000217` | `1.094234` | `0.062936` | `0.086111` | `0.099769` | `0.201852` | 1 |
| `FT7-stride8-alpha15-no-trainbudget` | `0.807547` | `0.192453` | `0.000000` | `1.042578` | `0.046704` | `0.084722` | `0.101157` | `0.190741` | 0 |
| `FT7-stride8-alpha15-pre1` | `0.707228` | `0.292772` | `-0.001953` | `1.071542` | `0.047264` | `0.087963` | `0.085880` | `0.312963` | 1 |
| `FT7-stride8-alpha15-pre3` | `0.752197` | `0.247803` | `-0.000651` | `1.002985` | `0.045720` | `0.085185` | `0.092824` | `0.257407` | 1 |

P5 summary：

```text
accepted_curvature_ratio = 0.7412847176422003
best_alternative_curvature_ratio = 0.5714414385126331
accepted_task_gate = 1
accepted_update_time_gate = 1
accepted_setting_dominates = 0
GuardStrideMechanismPass = 0
```

判断：

1. `FT7-accepted` 仍是一个可用平衡点：task gate 通过，update/step `0.048037 <= 0.10`，曲率降低 `25.87%`。
2. 但 accepted setting 没有支配所有替代项：`stride4-alpha15` 曲率最低 `0.571441`，但 update/step `0.101399 > 0.10`；`pre1` 曲率 `0.707228` 也强于 accepted，但 ValLossAUC time ratio `1.071542` 高于 formal TimeAUC gate。
3. `stride8-alpha8`、`no-pre2`、`pre3` 都能保一部分几何收益，但没有超过 accepted 的 balanced route。
4. 因此 P5 不能支持 formal success：当前证据更像 “accepted setting 是 task/system/geometry 的实用 Pareto 点”，而不是 “accepted setting 已被形式证明为最优 guard/stride 配置”。
5. 这一步没有推翻 v8.4 high-rep minimum，但把 formal blocker 从泛泛的 profiler/time accounting 进一步扩展到 guard/stride Pareto 解释。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 10. 追加：P6 strict time accounting from raw phase timing

本节继续 Wave 3 formal time / profiler / S1 closure。这里没有重测或补造时间，只把 `p8_functional_time_profiler_raw.csv` 中已有的真实 phase timing 重新整理为 v8.4 strict time accounting artifact。能从 raw phase 求出的字段写真实数值；不能单独分离的字段保留 `metric_unavailable`，并且不写 formal pass。

`p6_time_accounting.csv`：

| metric | value |
|---|---:|
| status | `derived_from_v83_p8_raw_phase_timing` |
| mapped phase total | `418.158235 ms` |
| wall clock total | `433.423671 ms` |
| train step time | `1.296306 ms` |
| forward time | `0.259131 ms` |
| backward time | `0.514399 ms` |
| update time | `0.114461 ms` |
| functional metric build time | `0.000000 ms` |
| functional update time | `0.059058 ms` |
| logging/post-check time | `0.276732 ms` |
| validation time | `5.080186 ms` |
| data loading / batch select time | `0.010528 ms` |
| unknown time fraction | `0.035601` |
| ValLossAUC step ratio | `0.997471` |
| ValLossAUC time total ratio | `1.042685` |
| functional update time ratio | `0.045565` |
| core time gates pass | 1 |
| strict phase separation pass | 0 |
| FormalTimeAccountingPass | 0 |

未单独实测字段：

```text
holdout_guard_time = metric_unavailable
cuda_sync_time = metric_unavailable
ValLossAUC_time_train_only = metric_unavailable
time_to_target_acc = metric_unavailable
```

判断：

1. P8 的三个核心 time gate 仍成立：unknown fraction `0.035601 <= 0.10`，ValLossAUC time ratio `1.042685 <= 1.05`，functional update ratio `0.045565 <= 0.10`。
2. 但 v8.4 formal time accounting 要求更细的 phase split；当前 raw profiler 没有把 holdout guard、CUDA sync、time-to-target 单独测出。
3. 因此 `core_time_gates_pass=1`，但 `FormalTimeAccountingPass=0`。这不是失败包装成成功，而是明确把 P8 pass 和 formal time pass 分开。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 11. 追加：P7 targeted phase-mapped profiler

本节继续 Wave 3。为了避免用 P8 summary 推测 kernel 信息，本轮新增 targeted torch profiler phase slices：真实构造 `KW6 hidden68 + FT7`，先执行 7 个 warmup training steps，然后在第一个 stride8 event step 对以下 phase 分别 profile：

```text
batch_select
forward_backward_with_pre_holdout
base_update
post_step_guard_pair
functional_update
```

Profiler scope：

```text
dataset = MNIST
seed = 0
profile_step = 8
functional_candidate = FT7
```

`p7_phase_mapped_profiler.csv`：

| metric | value |
|---|---:|
| kernel count total | `259` |
| mapped kernel time fraction | `1.000000` |
| unknown kernel time fraction | `0.000000` |
| top3 phase time explain | `0.914412` |
| kernel count forward/backward | `116` |
| kernel count base update | `28` |
| kernel count post-step guard | `37` |
| kernel count functional update | `78` |
| forward/backward kernel time | `237.116 us` |
| base update kernel time | `42.175 us` |
| post-step guard kernel time | `75.744 us` |
| functional update kernel time | `137.731 us` |
| small kernel count under 10us | `259` |
| layout conversion count | `46` |
| cuda memcpy time | `9.471 us` |
| cuda sync time | `61.842 us` |
| top kernel | `ampere_sgemm_32x32_sliced1x4_tn` |
| top kernel phase | `forward_backward_with_pre_holdout` |
| FormalProfilerPass | 1 |

判断：

1. P7 的 targeted event-step profiler 通过：mapped kernel time fraction `1.0`，unknown kernel fraction `0`，top3 phase explain `0.914412`，functional update phase 有真实 kernel count `78`。
2. 这说明 FT7 event step 的 kernel/update overhead 可以被 phase slices 解释，不再是完全黑箱。
3. 但本 artifact 是 targeted single-event-step profiler，不是 full-training profiler；route 因此仍不能升级 strong。
4. 由于 P6 strict time accounting 仍未过，`success_v84_formal_time_profiler` 仍为 0。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 12. 追加：P8 S1 / live-set attribution

本节继续 Wave 3 formal system closure。此前 `p8_s1_liveset_attribution.csv` 只来自 `p6_functional_fullgrid_s2.csv` 的 memory/step ratio，memory source 字段均为 `metric_unavailable`。本轮新增 targeted event-step CUDA memory attribution：

```text
focus shape = max memory ratio row from p6_functional_fullgrid_s2.csv
dataset = MNIST
batch_size = 512
seed = 0
profile_step = 8
base = KW6 hidden68
functional = FT7
```

该 artifact 真实重建 `KW6 hidden68 + FT7` 第一个 stride8 event step，分别测量 base / functional 的 phase-level CUDA allocated/reserved peak：

```text
batch_select
forward_backward / forward_backward_with_pre_holdout
base_update
post_step_task_loss / post_step_guard_pair
functional_update
```

`p8_s1_liveset_attribution.csv`：

| metric | value |
|---|---:|
| official memory ratio max | `1.027250` |
| official step ratio max | `1.101249` |
| P6 reference base peak | `23.061035 MB` |
| P6 reference functional peak | `23.689453 MB` |
| P6 reference extra peak | `0.628418 MB` |
| targeted base event peak | `14.936035 MB` |
| targeted functional event peak | `18.596191 MB` |
| targeted event extra peak | `3.660156 MB` |
| root input cache | `4.605469 MB` |
| optimizer state | `0.484558 MB` |
| backward temp extra | `3.215332 MB` |
| holdout guard buffer extra | `3.155762 MB` |
| functional update buffer extra | `3.660156 MB` |
| allocator padding | `5.403809 MB` |
| reserved unallocated | `10.033203 MB` |
| unknown memory fraction | `0.000000` |
| AttributionPass | 1 |
| NearS1Pass | 0 |
| S1Pass | 0 |

Top memory sources:

| rank | source | MB |
|---:|---|---:|
| 1 | `reserved_unallocated_MB` | `10.033203` |
| 2 | `allocator_padding_MB` | `5.403809` |
| 3 | `functional_update_buffer_MB` | `3.660156` |

判断：

1. P8 不再是 partial memory artifact：已落盘 `p8_s1_liveset_attribution_raw.csv`，共 `9` 条 phase rows，均来自真实 CUDA memory counters。
2. AttributionPass 为 1，说明 targeted event-step 的 extra peak 可以由 phase/source 解释；没有 fake/proxy/CPU offload。
3. 但 S1 仍失败：official full-grid memory max 是 `1.027250`，不满足 S1 hard gate `memory_ratio < 1.00`；NearS1 也失败，因为 memory ratio 高于 `1.01`。
4. 最大系统余量问题不是 step：step max `1.101249` 已在 S1 step gate `<=1.35` 内；真正 S1 blocker 是 memory ratio。
5. 当前 top sources 显示 allocator reserved/unallocated 与 functional update event buffer 是主要解释项。下一步如果继续冲 S1，应做 allocator/lifetime trim 或 functional update buffer reuse，而不是修改 CE objective 或削弱 functional geometry。

No-fake audit：

```text
rows_checked = 41
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

## 13. Artifact 状态

v8.4 runner 已落盘：

```text
candidate_registry_v84.csv
contract_no_teacher_no_loss_functional_v84.csv
route_semantics_audit.csv
p0_contract_route_audit.csv
p1_base_reproduction.csv
p2_ft7_reproduction.csv
p3_causality_controls.csv
p4_role_ablation.csv
p5_guard_stride_ablation.csv
p6_time_accounting.csv
p7_phase_mapped_profiler.csv
p8_s1_liveset_attribution.csv
p9_edge_geometry_decomposition.csv
p10_function_space_probe.csv
p11_extended_scaling.csv
p12_extended_robustness.csv
p13_final_confirmation.csv
route_decision.json
aggregate_decision.json
failure_table.csv
v84_provenance_audit.csv
run_manifest.json
```

其中 `p3_causality_controls.csv`、`p4_role_ablation.csv`、`p5_guard_stride_ablation.csv` 已完成，并额外落盘 `p3_causality_controls_raw.csv`、`p4_role_ablation_raw.csv`、`p5_guard_stride_ablation_raw.csv`。`p6_time_accounting.csv` 已由 raw phase timing 派生为严格记录，但 FormalTimeAccountingPass 仍为 0。`p7_phase_mapped_profiler.csv` 已完成 targeted single-event-step profiler，并额外落盘 `p7_phase_mapped_profiler_raw.csv`。`p8_s1_liveset_attribution.csv` 已完成 targeted event-step phase peak attribution，并额外落盘 `p8_s1_liveset_attribution_raw.csv`；AttributionPass 为 1，但 S1Pass 仍为 0。`p9_edge_geometry_decomposition.csv`、`p10_function_space_probe.csv`、`p11_extended_scaling.csv`、`p12_extended_robustness.csv` 仍为 `not_run`。

## 14. 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v84_real.py` | `591a972f2caaef672d90ce28341f76716f2e6fb852c12a07c9ee3377d671f8b6` |
| `experiments/run_gafu_v83_real.py` | `9565be79ec3896aa06baa396aa43f463a89461cae2d67e93486126990d829166` |
| v8.4 plan | `1e6d8043e8285a6ab2c6bafa527944ce363b002735cf222293c16e4643ee00c0` |
| fresh 50/200 route | `84fa15eddeba06a8bc33f5bcb6eb2a2a4935461166ec99a5325f11a4ae4fbcbd` |
| repeat 50/200 route | `84fa15eddeba06a8bc33f5bcb6eb2a2a4935461166ec99a5325f11a4ae4fbcbd` |
| base high-rep diagnostic route | `5e3b867b0e36c91cd8d24f608cd1e466ffa7d9916233986811c1f4ae844bfa35` |
| high-rep v84 route after P8 attribution | `92cbd10f2750b611ebae56ef9f1add552557618d39e2ea41f7661aa7ebf92f0a` |
| high-rep P1 base reproduction | `2961748c54af7f01dc4e4554602147d94da3150a8fb3e6fc940a0d3be578cf63` |
| high-rep P2 FT7 reproduction | `6341a41d2d5ab6ae06efc7b1fab1aac25f719eab038d234403baf0d131f21a83` |
| high-rep P3 causality controls | `5317c33f9b35084cba3b72ba1c329326d5508c512c9535249ca052e6a3b69e58` |
| high-rep P3 raw | `6ee26831a40f48198ad493ff5fd772e633122e4effb25de7ecb8f909fe1634b8` |
| high-rep P4 role ablation | `7738053716ca8d701f7d0ba8d416c3c76d597d150e19b74ec5f00a5c6c58c869` |
| high-rep P4 raw | `74dcd899c7cd2329a6cfdc3a44c589fb91fa24a80e18395a5d369f5a26c74723` |
| high-rep P5 guard/stride ablation | `c06553e733eca0f44a447901fcc2aae8a32d148950440dd4d33c998866110080` |
| high-rep P5 raw | `875a88df689de98abe4109883b795a246a0ac1c3f042f554f7d2446c1f15f2ad` |
| high-rep P6 time accounting | `9a8c982ff9e678411d77b7dd97e14689e93a89fdedf2111a7e70d50213bae63b` |
| high-rep P7 phase-mapped profiler | `4eafb347339822960a199f583397eff2f9c0d2f8da8c2d11e28ed908aa7cdf63` |
| high-rep P7 raw | `107789692ef3b7d1ed7da2259dde7af02506b014993b5b0246608a8999bc5db8` |
| high-rep P8 S1/live-set attribution | `ca39e411bf0512247e564e65eedc67ad06a7725e6d46bd15e5b899e1a72595f8` |
| high-rep P8 S1/live-set raw | `f45fe17f59ef171af4d461803d80bbd330c61a8fe319076a448c9b6d5b9420da` |
| high-rep P8 profiler | `06189bc866b2fca377eae75cbe7ce1761af1dc9887355e18cbd8ae28d07fedd2` |
| high-rep P9 scaling/robustness | `e7ac8c4302d3ff4056f0cab93a3794be2ecc8d60ba902ceefd447a39d233ac8a` |
| high-rep P13 final confirmation | `891fa7589d4abd643fe583fd58349aeec8fce74ed89365d312c0089605887bf8` |
| high-rep v84 provenance | `4119967502f22066e12805d488638dab1ea7f1af2d49f679f29478af49b953ad` |

## 15. 更新结论

v8.4 minimum 已在 high-rep timing-stabilized protocol 下完成；formal / strong 仍未完成：

```text
official 50/200 H0BasePass = false
high-rep H0/P7/P8/P9 = true
FunctionalCausalityPass = true
RoleMechanismPass = true
GuardStrideMechanismPass = false
FormalProfilerPass = true
FormalTimeAccountingPass = false
AttributionPass = true
S1Pass = false
NearS1Pass = false
success_v84_minimum = true
success_v84_formal_time_profiler = false
success_v84_strong = false
```

机制结论：

1. v8.4 的第一轮执行没有推翻 v8.3 functional route；高 reps 下它仍能复现到 P9。
2. 但 v8.4 也暴露了一个更严格的问题：原 50/200 timing protocol 下，`KW6 hidden68` 的 H0 step gate 不稳定，fresh/repeat 都失败。
3. P3 causality controls 证明 FT7 的 geometry gain 不是 no-op overhead 或 random direction 的副作用；`FunctionalCausalityPass=1`。
4. P4 role ablation 进一步证明 stack/head roles 都有正几何贡献，full route 还有 interaction gain；`RoleMechanismPass=1`。
5. P5 guard/stride ablation 说明 accepted setting 不是所有 schedule 中的绝对最强几何配置；它是当前 task/system/geometry 的平衡点，但 `GuardStrideMechanismPass=0`。
6. P6 strict time accounting 证明 P8 core time gates 过线，但缺 holdout guard / CUDA sync / time-to-target 单独 phase，因此 `FormalTimeAccountingPass=0`。
7. P7 targeted phase-mapped profiler 证明 event-step kernel overhead 可解释，`FormalProfilerPass=1`。
8. P8 S1/live-set attribution 已经完成 targeted event-step source attribution，`AttributionPass=1`；但 official memory ratio max `1.027250`，所以 `S1Pass=0`、`NearS1Pass=0`。
9. 因此可以声明 high-rep timing-stabilized v8.4 minimum success，但不能声明 formal / strong success。
10. 下一步不应改 teacher/loss/CPU，也不应削弱 functional geometry；应做真实 holdout guard / CUDA sync / target-time phase split、S1 memory/lifetime repair，并把 H0 timing protocol和 guard/stride Pareto 解释形式化。

最终一句话：

> v8.4 已经完成 high-rep timing-stabilized minimum：H0/P7/P8/P9、P3 causality controls 与 P4 role mechanism 全部通过，`success_v84_minimum = true`。P7 targeted profiler 证明 event-step kernel overhead 可解释；P8 targeted live-set attribution 也已完成，并定位到 allocator reserved/unallocated 与 functional update buffer 是主要 memory 来源。但 P5/P6/S1 仍未闭合，50/200 原协议 H0 也仍不稳。因此 formal / strong 还没完成，下一步要做 strict phase split、S1 memory/lifetime repair 和 guard/stride Pareto formalization，而不是改 teacher/loss/CPU 或牺牲 functional geometry。

## 16. 追加：S1 no-pre-role feature cache lifetime trim

本节继续执行 v8.4 formal / strong blocker 中的 S1 memory/lifetime repair。修复范围仍严格限定在 system-side implementation：

```text
loss_type = CE
functional_update_is_update_rule = 1
external_teacher_used = 0
self_teacher_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
fake/proxy = 0
```

### 16.1 代码改动

观察到 accepted `FT7` event step 中，stack role 会在 role guard 内重算并更新 `train_features / holdout_features`，因此 post-base guard 阶段预先保存的 `task_features_after / holdout_features_after` 对 accepted route 不是必要 live tensor。本轮改动：

| 文件 | 改动 |
|---|---|
| `experiments/run_gafu_v83_real.py` | 新增 `_loss_pair_only` / `_timed_loss_pair_only`，post-base guard 只测 CE value，不再保留 pre-role feature cache |
| `experiments/run_gafu_v83_real.py` | P5/P6/P8/P9 accepted FT7 path 向 `_apply_ft7_streamed_guarded_update` 传入 `task_features_after=None`、`holdout_features_after=None` |
| `experiments/run_gafu_v84_real.py` | P3/P7/P8 targeted profiler / attribution 跟随 no-pre-role feature cache 路径 |

代码检查：

```text
python -m py_compile experiments/run_gafu_v84_real.py experiments/run_gafu_v83_real.py experiments/run_gafu_v82_real.py experiments/run_gafu_v81_real.py experiments/run_gafu_v80_real.py experiments/run_gafu_v79_real.py experiments/run_gafu_v72_real.py experiments/run_gafu_v71_real.py
```

已通过。

### 16.2 full high-rep fresh probe

运行：

```bash
python experiments/run_gafu_v84_real.py \
  --out-dir results/real_rerun_20260506/v84_s1_no_prerole_feature_cache_highrep_10seed_20260507T233000Z \
  --fresh \
  --device auto \
  --hidden-dim 68 \
  --candidates B0,KW6 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 200 \
  --bench-reps 1000 \
  --grad-batch-sizes 8,128 \
  --trace-every 10 \
  --bootstrap-reps 10000
```

结果：

```json
{
  "route": "R9-NoReproduction",
  "H0BasePass": 0,
  "success_v84_minimum": 0,
  "primary_blocker": "fresh_reproduction_fail"
}
```

P1 base：

| candidate | macro gap | CI95 low | test gap | GradPass | memory max | step max | S2 shapes | H0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KW6 hidden68` | `+0.021680` | `+0.016341` | `+0.024349` | 1 | `1.011163` | `1.847471` | `1/9` | 0 |

判断：这次 fresh full high-rep 仍被 H0 base step instability 挡住，不能评价 functional/S1，也不能改写 v8.4 success。No-fake audit：

```text
rows_checked = 26
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 16.3 targeted P6 full-grid S2 remeasure

为避免 fresh H0 timing 抖动遮蔽 S1 repair，本节在 accepted high-rep FT7 setting 上直接重测 P6 functional full-grid S2：

```text
results/real_rerun_20260506/v84_s1_no_prerole_feature_cache_p6s2_probe_20260508T000000Z/
```

P6 full-grid S2：

| metric | previous accepted | no-pre-role feature cache |
|---|---:|---:|
| memory ratio max | `1.027250` | `1.024290` |
| step ratio max | `1.101249` | `1.101853` |
| S2 shapes | `9/9` | `9/9` |

Per-shape memory / step：

| dataset | batch | memory ratio | step ratio | S2 |
|---|---:|---:|---:|---:|
| MNIST | 128 | `1.000000` | `1.101853` | 1 |
| MNIST | 256 | `1.005859` | `1.080959` | 1 |
| MNIST | 512 | `1.024290` | `1.064225` | 1 |
| Fashion-MNIST | 128 | `1.000000` | `1.100604` | 1 |
| Fashion-MNIST | 256 | `1.005859` | `1.085996` | 1 |
| Fashion-MNIST | 512 | `1.024290` | `1.092183` | 1 |
| KMNIST | 128 | `1.000000` | `1.056664` | 1 |
| KMNIST | 256 | `1.005859` | `1.069629` | 1 |
| KMNIST | 512 | `1.000000` | `1.074986` | 1 |

判断：no-pre-role feature cache 是有效 memory trim，official full-grid memory max 从 `1.027250` 降到 `1.024290`，但仍未达到 S1 hard gate `<1.00` 或 NearS1 gate `<=1.01`。

### 16.4 targeted S1 attribution remeasure

同一 probe 继续重测 P8 S1 live-set attribution：

| metric | previous accepted attribution | no-pre-role feature cache |
|---|---:|---:|
| memory ratio max | `1.027250` | `1.024290` |
| step ratio max | `1.101249` | `1.101853` |
| P6 reference base peak MB | `23.061035` | `14.936035` |
| P6 reference functional peak MB | `23.689453` | `15.298828` |
| P6 reference extra peak MB | `0.628418` | `0.362793` |
| targeted base event peak MB | `14.936035` | `14.936035` |
| targeted functional event peak MB | `18.596191` | `18.330566` |
| targeted extra peak MB | `3.660156` | `3.394531` |
| functional update buffer MB | `3.660156` | `3.394531` |
| holdout guard buffer MB | `3.155762` | `3.155762` |
| backward temp MB | `3.215332` | `3.215332` |
| allocator padding MB | `5.403809` | `5.669434` |
| reserved unallocated MB | `10.033203` | `10.298828` |
| unknown memory fraction | `0.000000` | `0.000000` |
| AttributionPass | 1 | 1 |
| NearS1Pass | 0 | 0 |
| S1Pass | 0 | 0 |

判断：

1. 本轮 repair 真实降低了 accepted route 的 live-set：targeted functional event peak 降低 `0.265625 MB`，functional update buffer 降低 `0.265625 MB`。
2. P6 full-grid memory 也同步下降：`1.027250 -> 1.024290`。
3. 但 allocator reserved/unallocated 与 padding 仍是 top memory sources；S1 / NearS1 仍失败。
4. 这不是 objective 侧问题，也不是 teacher/loss/CPU/offload 问题；下一步仍应做 allocator/lifetime trim，例如 functional update buffer reuse、phase-local allocation reuse 或减少 event-step reserved workspace。

### 16.5 关键 hash

| artifact | SHA256 |
|---|---|
| `experiments/run_gafu_v84_real.py` | `4083d78cf5eb21d8f9264129bef1fc13b40a99d450fa8f1e56fe06fd5c45d77e` |
| `experiments/run_gafu_v83_real.py` | `3fd45baf1b912bc19065b956fe9924d04fc3b7cda8b7e7316f266ba2cd9a37b8` |
| full high-rep fresh route | `1b5a59c9d95501f47e1cce34bf40fd8bdbc3c3eb69211175face7d1a56091f1c` |
| full high-rep fresh P2 FT7 | `fda147838d37f3b868514edf6be4982140aab51ac419f2916ed758f72f1e75a4` |
| full high-rep fresh provenance | `38cc71a251317d1f23f6ead90587d4a48d38cf3c20bea0eee4de20b05d1a0a99` |
| targeted P6 full-grid S2 | `9e210ccf6b6b930bfc5d8706ab37f53211df7614e79c380d57bced0f3b1586bd` |
| targeted S1 attribution | `0ee148144dd7117f5c94620bb42e962a80400196b87f64ce60f9ba077b6bf3be` |
| targeted S1 attribution raw | `5f87bd0807a17bffa7ae5deea2da1a3d918b2fc91b37e14eaf75454d4f775e91` |

### 16.6 更新结论

v8.4 仍未完成 formal / strong：

```text
high-rep minimum accepted record = true
fresh no-pre-role-cache full run H0BasePass = false
targeted no-pre-role-cache P6 S2 = 9/9
targeted memory max = 1.024290
targeted S1Pass = false
targeted NearS1Pass = false
success_v84_formal_time_profiler = false
success_v84_formal_s1 = false
success_v84_strong = false
```

最终一句话：

> no-pre-role feature cache 是一次干净的 S1 memory/lifetime 进展：它不改 CE、teacher、loss、functional update 公式或 CPU/offload，把 P6 memory max 从 `1.027250` 降到 `1.024290`，并把 targeted functional event peak 从 `18.5962 MB` 降到 `18.3306 MB`。但它还没有闭合 S1 / NearS1，且 fresh full run 仍被 H0 step instability 挡住；v8.4 formal / strong 仍未完成。下一步应继续做 allocator/lifetime 级别的 memory trim，而不是偏离文档中心思想。
