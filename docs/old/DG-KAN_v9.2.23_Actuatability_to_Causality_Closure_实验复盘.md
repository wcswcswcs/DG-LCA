# DG-KAN v9.2.23 Actuatability-to-Causality Closure 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.23_Actuatability_to_Causality_Closure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = R2-FunctionalEventSilent
base_candidate = LQ-t2-h256
success_v9223_strict_purekan_functional = false
success_v9223_full_functional = false
success_v9223_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9223_actuatability_to_causality_closure_first_20260510T130000Z/
```

核心结论：

1. P0 复现 v9.2.22 boundary：`N2a` 仍是 base-qualified / actuatability pass 的 strict PureKAN interface source，paired replay source 仍未通过。
2. P1 重新测量 actuatability proxy 到 realized logit movement 的关系，proxy-realized correlation = `0.869758`。
3. P1 mean actual logit ratio `r_z = 0.077576`，tail ratio `r_z_tail = 0.078286`，failure mode = `functional_event_silent`。
4. P2/P3 状态：best calibrated candidate = ``，best real beats AdamWParallel rate = `0.000000`，best real beats best LR rate = `0.000000`。
5. 当前 blocker：`realized_logit_or_tail_logit_displacement_below_threshold`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9223_actuatability_to_causality_closure.py` | v9.2.23 runner；生成 P0-P8 artifacts、actuatability-to-realized-effect calibration、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9223_actuatability_to_causality_closure.py
```

正式运行：

```bash
python experiments/run_v9223_actuatability_to_causality_closure.py \
  --out-dir results/real_rerun_20260506/v9223_actuatability_to_causality_closure_first_20260510T130000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R2-FunctionalEventSilent",
  "base_candidate": "LQ-t2-h256",
  "v9222_boundary_pass": 1,
  "actuatability_proxy_calibration_pass": 1,
  "failure_mode": "functional_event_silent",
  "best_calibrated_candidate": "",
  "best_interface_family": "",
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 1.0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_kmnist_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "p1_row_count": 27,
  "p1_proxy_actual_corr": 0.8697582483319393,
  "p1_mean_actual_r_z": 0.07757615960306591,
  "p1_mean_actual_r_z_tail": 0.0782860946600084,
  "p1_mean_actual_r_z_perp": 0.07702245901304262,
  "p1_silent_rate": 0.6666666666666666,
  "p1_misaligned_rate": 0.037037037037037035,
  "p2_activation_pass": 0,
  "p2_survivor_count": 0,
  "p3_interface_survivor_count": 0,
  "p2_best_real_beats_adamwparallel_rate": 0,
  "p2_best_real_beats_best_lr_rate": 0,
  "p2_best_bad_event_rate": 0,
  "p2_best_holdout_nonharm": 0,
  "p2_best_CEp99_delta": 0,
  "p2_best_margin_delta": 0,
  "primary_blocker": "realized_logit_or_tail_logit_displacement_below_threshold",
  "next_required_implementation": "increase_event_time_activation_band_without_breaking_base",
  "success_v9223_strict_purekan_functional": 0,
  "success_v9223_full_functional": 0,
  "success_v9223_external_ready": 0
}
```

## 3. P0 v9.2.22 boundary reproduction

Artifact：

```text
p0_v9222_boundary_reproduction.csv
```

关键值：

| metric | value |
|---|---:|
| source route | `R3-FunctionalActuatabilityRetained` |
| best candidate | `N2a-TinyInit-RationalFunc-BranchRatioCap` |
| interface P4/P5/actuatability | `1/1/1` |
| source paired replay pass | `0` |
| fake proxy count | `0` |

判断：P0 是 source artifact recap，不伪装成新 full training。

## 4. P1 actuatability-to-realized-effect calibration

Artifact：

```text
p1_actuatability_realized_effect_calibration.csv
actual_logit_displacement_trace_v9223.csv
```

Summary：

```text
rows = 27
proxy_actual_corr = 0.8697582483319393
mean_actual_r_z = 0.07757615960306591
mean_actual_r_z_tail = 0.0782860946600084
mean_actual_r_z_perp = 0.07702245901304262
silent_rate = 0.6666666666666666
misaligned_rate = 0.037037037037037035
proxy_calibration_pass = 1
```

关键分布：

| scope | proxy r_perp mean | actual r_z mean | actual tail r_z mean | failure |
|---|---:|---:|---:|---|
| all 27 rows | `34.908054` | `0.077576` | `0.078286` | silent `18/27`, misaligned `1/27` |
| MNIST | - | `0.048932` | `0.048714` | silent `9/9` |
| Fashion-MNIST | - | `0.023772` | `0.021680` | silent `9/9` |
| KMNIST | - | `0.160024` | `0.164464` | non-silent `8/9`, misaligned `1/9` |

按 target：

| target | proxy r_perp mean | actual r_z mean | actual tail r_z mean |
|---|---:|---:|---:|
| O1 HardTailLogitCorrection | `21.347115` | `0.088980` | `0.093154` |
| O2 MarginTailExpansion | `42.107043` | `0.090133` | `0.087811` |
| O6 KMNISTHardModeOutputTarget | `41.270003` | `0.053615` | `0.053893` |

判断：P1 不把 P3 的 large `r_perp` 直接当成功，而是重新测量 event 后 logits / tail logits 相对 AdamWParallel 的实际移动。

更具体地说，P1 没有证明 “P3 proxy 没信息”：`corr=0.869758` 反而说明 proxy 排序和 actual non-AdamW movement 有强相关。真正的问题是 scale：proxy r_perp 可以很大，但 cap / task-safe / base-neutral event 后 realized logit ratio 平均只有 `0.077576`，低于计划阈值 `0.10`。这就是本轮 route 写成 `R2-FunctionalEventSilent` 而不是 `R1-ActuatabilityProxyMiscalibrated` 的原因。

## 5. P2/P3 calibrated activation and interface comparison

Artifacts：

```text
p2_ft7_mechanism_calibrated_activation.csv
p3_interface_family_comparison.csv
branch_derivative_calibration_trace_v9223.csv
```

Summary：

```text
p2_activation_pass = 0
p2_survivor_count = 0
best_calibrated_candidate = 
best_interface_family = 
best_real_beats_adamwparallel_rate = 
best_real_beats_best_lr_rate = 
```

P2/P3 未打开原因：

```text
reason = P1_actuatability_calibration_failed_or_event_silent
```

P4-P8 downstream boundary：

| artifact | status | reason |
|---|---|---|
| `p4_strict_paired_replay_control_gate.csv` | `not_run` | P1 event silent |
| `p5_short_run_functional_validation.csv` | `not_run` | P1 event silent |
| `p6_full_10seed_functional_validation.csv` | `not_run` | P1 event silent |
| `p7_adamw_only_fullpass_repair.csv` | `not_run` | P1 event silent |
| `p8_robustness_external_ready.csv` | `not_run` | P1 event silent |

判断：P2/P3 没有被拿来“强行扫参数”。按计划，P1 已经命中 silence stop condition，因此本轮不能打开 strict paired replay、short/full validation 或 external-ready。

## 6. No-fake audit

`v9223_provenance_audit.csv`：

```text
rows_checked = 37
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

## 7. Hash

见 `artifact_hashes.csv`。关键 artifacts 包含 plan、runner、route、P0-P8、trace 与 provenance audit。

## 8. 最终分析结论

v9.2.23 的真实推进是：

```text
v9.2.22: base qualification 和 actuatability 已闭合，但 paired replay 输给 AdamWParallel / LR。
v9.2.23: 校准 P3 actuatability proxy 到 actual logits / tail logits，再决定是否打开 calibrated activation。
```

机制判断：

1. 当前应该把 `r_perp` 理解成 actuatability proxy，而不是 causal gain；本轮还进一步说明 proxy 有排序信息，但 realized movement scale 不够。
2. `N2a` 的 base-neutral 修复大概率把 functional branch 压得太保守：base 不再破坏 P5，但 event-time logit movement 平均低于 AdamWParallel 的 `0.10` ratio 阈值。
3. MNIST/Fashion-MNIST 全部 silent，KMNIST 多数 non-silent；这提示下一步不该无差别放大 functional channel，而应做 dataset/event-conditioned activation band，优先研究 KMNIST 的非 silent rows 是否可安全迁移。
4. 本轮没有打开 short/full validation，因为 paired replay 前的 P1 已经失败；这是防止把一个太弱的 functional event 通过后续训练噪声写成成功。

最终一句话：

> v9.2.23 真实执行后停在 `R2-FunctionalEventSilent`：`realized_logit_or_tail_logit_displacement_below_threshold`。
