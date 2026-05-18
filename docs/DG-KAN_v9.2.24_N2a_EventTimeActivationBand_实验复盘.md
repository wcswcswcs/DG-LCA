# DG-KAN v9.2.24 N2a Event-Time Activation Band 实验复盘

> 本复盘记录本轮针对 v9.2.23 `R2-FunctionalEventSilent` 的修复实验。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = R3-ActivationRepairedButControlEquivalent
base_candidate = LQ-t2-h256
success_v9224_event_activation_repair = true
success_v9224_strict_purekan_functional = false
success_v9224_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9224_n2a_event_time_activation_repair_first_20260510T140000Z/
```

核心结论：

1. v9.2.23 的 event silent boundary 被复现，source mean `r_z=0.07757615960306591`。
2. P1 activation sweep 真实执行；non-applicable O6 rows 在 MNIST/Fashion 中明确记录为 `target_applicable=0`，不作为成功或失败 proxy。
3. Activation policy = `{'Fashion-MNIST': 0.5, 'KMNIST': 0.5, 'MNIST': 0.5}`，policy type = `global`。
4. P1 activation band pass = `1`。
5. P2 paired replay pass = `0`；Real beats AdamWParallel rate = `0.20238095238095238`，beats best LR rate = `0.3333333333333333`。
6. 当前 blocker：`event_scale_repairs_silence_but_realfunctional_still_loses_controls`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9224_n2a_event_time_activation_repair.py` | v9.2.24 runner；执行 N2a event-time activation band sweep 与 repaired paired replay |

代码检查：

```text
python -m py_compile experiments/run_v9224_n2a_event_time_activation_repair.py
```

正式运行：

```bash
python experiments/run_v9224_n2a_event_time_activation_repair.py \
  --out-dir results/real_rerun_20260506/v9224_n2a_event_time_activation_repair_first_20260510T140000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

```json
{
  "route": "R3-ActivationRepairedButControlEquivalent",
  "base_candidate": "LQ-t2-h256",
  "source_route": "R2-FunctionalEventSilent",
  "source_mean_actual_r_z": 0.07757615960306591,
  "source_mean_actual_r_z_tail": 0.0782860946600084,
  "activation_band_pass": 1,
  "activation_policy_type": "global",
  "activation_policy": {
    "Fashion-MNIST": 0.5,
    "KMNIST": 0.5,
    "MNIST": 0.5
  },
  "eligible_rows": 105,
  "paired_replay_pass": 0,
  "real_row_count": 84,
  "real_beats_adamwparallel_rate": 0.20238095238095238,
  "real_beats_best_lr_rate": 0.3333333333333333,
  "task_safe_rate": 1.0,
  "real_mean_CEp99_delta": -0.0025468837647210983,
  "real_mean_margin_delta": 0.0011335256553831555,
  "primary_blocker": "event_scale_repairs_silence_but_realfunctional_still_loses_controls",
  "next_required_implementation": "redesign_functional_direction_not_only_activation_scale",
  "success_v9224_event_activation_repair": 1,
  "success_v9224_strict_purekan_functional": 0,
  "success_v9224_external_ready": 0
}
```

## 3. P1 activation band sweep

Artifacts：

```text
p1_n2a_event_time_activation_sweep.csv
p1_activation_band_summary.csv
```

Summary：

```text
activation_band_pass = 1
activation_policy_type = global
activation_policy = {'Fashion-MNIST': 0.5, 'KMNIST': 0.5, 'MNIST': 0.5}
eligible_rows = 105
```

关键 sweep 结果：

| dataset | fraction | mean r_z | mean tail r_z | silent rows | task-safe rows | eligible rows |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | `0.10` | `0.035658` | `0.032521` | `6` | `6` | `6` |
| Fashion-MNIST | `0.30` | `0.106772` | `0.097431` | `4` | `6` | `6` |
| Fashion-MNIST | `0.50` | `0.177968` | `0.162381` | `0` | `6` | `6` |
| MNIST | `0.10` | `0.073398` | `0.073070` | `6` | `6` | `6` |
| MNIST | `0.30` | `0.174126` | `0.179243` | `0` | `6` | `6` |
| MNIST | `0.50` | `0.280875` | `0.291124` | `0` | `6` | `6` |
| KMNIST | `0.10` | `0.160024` | `0.164464` | `0` | `9` | `9` |
| KMNIST | `0.50` | `0.593786` | `0.627937` | `0` | `9` | `9` |

Global pass row：

```text
fraction = 0.50
eligible_rows = 21
silent_rate = 0.0
task_safe_rate = 1.0
holdout_nonharm_rate = 0.952381
mean_actual_r_z = 0.385578
mean_actual_r_z_tail = 0.398689
```

判断：本轮解决的是 “event-time movement scale 是否太弱”，不是重新训练 base。Base 仍是 v9.2.22 的 N2a 路线。

机制判断：MNIST/Fashion 的 silent 可以通过 event cap 放大修复；KMNIST 原本就不是 silent。非 KMNIST 的 O6 target 在 CSV 中以 `target_applicable=0` 记录，没有拿来充当失败或成功。分裂现象的直接原因是同一 `0.10` cap 在 MNIST/Fashion 上产生的 realized logit ratio 不足，而 KMNIST 的 actuator-target coupling 更强。

## 4. P2 repaired paired replay

Artifact：

```text
p2_repaired_activation_paired_replay.csv
```

Summary：

```text
paired_replay_pass = 0
real_row_count = 84
real_beats_adamwparallel_rate = 0.20238095238095238
real_beats_best_lr_rate = 0.3333333333333333
task_safe_rate = 1.0
real_mean_CEp99_delta = -0.0025468837647210983
real_mean_margin_delta = 0.0011335256553831555
```

By dataset：

| dataset | beats AdamWParallel | beats best LR | CEp99 delta | margin delta | task-safe |
|---|---:|---:|---:|---:|---:|
| MNIST | `0.166667` | `0.500000` | `-0.000002` | `+0.000004` | `1.000000` |
| Fashion-MNIST | `0.333333` | `0.333333` | `-0.008912` | `+0.003964` | `1.000000` |
| KMNIST | `0.138889` | `0.222222` | `+0.000000` | `-0.000000` | `1.000000` |

By horizon：

| horizon | beats AdamWParallel | beats best LR | CEp99 delta | margin delta |
|---:|---:|---:|---:|---:|
| 1 | `0.238095` | `0.428571` | `-0.000004` | `+0.000001` |
| 5 | `0.142857` | `0.238095` | `-0.000010` | `+0.000006` |
| 20 | `0.095238` | `0.238095` | `-0.000419` | `+0.001759` |
| 80 | `0.333333` | `0.428571` | `-0.009754` | `+0.002769` |

判断：如果 P1 通过但 P2 不通过，说明 silence 可以修，但 causal superiority 仍未成立。

更具体地说，放大到 `0.50` 后 RealFunctional 不再“完全没动”，CE tail 和 margin 已有正向平均变化；但这个变化仍不足以通过 strong-control gate。Fashion-MNIST 有较明显 CEp99 改善，MNIST 对 best LR 接近 `0.50`，KMNIST 反而没有从放大中获得 control superiority。这说明下一步不该继续单纯加大 fraction，而要重做 functional direction / event selection。

## 5. No-fake audit

```text
rows_checked = 1169
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

## 6. 最终分析结论

v9.2.24 的真实推进是：

```text
v9.2.23: N2a event silent，mean r_z < 0.10。
v9.2.24: 只调整 event-time activation band，检查能否把 silent 修成 realized movement，并继续过 strong controls。
```

最终一句话：

> v9.2.24 真实执行后停在 `R3-ActivationRepairedButControlEquivalent`：`event_scale_repairs_silence_but_realfunctional_still_loses_controls`。

## 7. 下一步判断

本轮已经回答了用户提出的关键问题：

```text
N2a event-time activation 太弱可以通过 event cap 修复；
MNIST/Fashion silent 是 scale 不足，不是 target 完全无效；
KMNIST non-silent 是因为同一 cap 下 actuator-target coupling 更强；
但 scale 修复后仍没有 functional causality success。
```

下一步不应继续把 `0.50` 往上扫。原因是 P2 已经显示 task-safe movement 够了，但 strong-control superiority 不够。更合理的路线是：

1. 固定“non-silent activation band”作为已解决工程边界。
2. 改 functional direction，使它不是 AdamWParallel/LR 可解释的弱 tail displacement。
3. 优先分析 Fashion-MNIST 的 `CEp99=-0.008912` positive signal 和 KMNIST 的 amplification failure，做 target/direction 分叉，而不是全任务统一 cap。
