# DG-KAN v9.2.26 Fashion Delayed Controller 与 KMNIST Target/Primitive Diagnosis 实验复盘

> 本复盘记录本轮针对 v9.2.25 `DelayedFashionSignalOnly` 的后续实验。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 KMNIST 诊断 rows 写成 functional success。

## 0. 最新结论

```text
route = R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker
base_candidate = LQ-t2-h256
success_v9226_fashion_delayed_controller = False
success_v9226_strict_purekan_functional = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9226_fashion_delayed_controller_kmnist_diagnosis_first_20260510T160000Z/
```

核心结论：

1. v9.2.25 boundary 复现通过：source route = `R2-DelayedFashionSignalOnly`，source best = `F3-Fashion-O2-Cap0.5` / `h80_only`。
2. Fashion 50/240-step ablation pass = `0`，best = `F3-Fashion-O2-DelayedController` / `h50`。
3. Fashion best CEp99 delta = `-0.0002285639444986979`，margin delta = `0.0011636714140574138`，beats AdamWParallel = `0.6666666666666666`，beats best LR = `0.6666666666666666`。
4. KMNIST diagnosis blocker = `KMNIST_realized_effect_not_metric_causal`；target oracle useful rate = `1.0`，safe fit pass rate = `1.0`，actual effect pass rate = `0.0`。
5. 当前 blocker：`fashion_signal_partial_and_kmnist_requires_target_primitive_redesign`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py` | Fashion delayed controller 50/240-step ablation；KMNIST target/primitive diagnosis |

代码检查：

```text
python -m py_compile experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py
```

正式运行：

```bash
python experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py \
  --out-dir results/real_rerun_20260506/v9226_fashion_delayed_controller_kmnist_diagnosis_first_20260510T160000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

```json
{
  "base_candidate": "LQ-t2-h256",
  "fashion_ablation_pass": 0,
  "fashion_all_scope_pass_count": 0,
  "fashion_best_CEp99_delta": -0.0002285639444986979,
  "fashion_best_beats_adamwparallel": 0.6666666666666666,
  "fashion_best_beats_best_lr": 0.6666666666666666,
  "fashion_best_candidate": "F3-Fashion-O2-DelayedController",
  "fashion_best_margin_delta": 0.0011636714140574138,
  "fashion_best_scope": "h50",
  "fashion_best_task_safe": 1.0,
  "fashion_horizon_pass_count": 1,
  "kmnist_actual_effect_pass_rate": 0.0,
  "kmnist_diagnosis_blocker": "KMNIST_realized_effect_not_metric_causal",
  "kmnist_max_cap_rz": 0.640339195728302,
  "kmnist_max_raw_R2": 1.0,
  "kmnist_max_safe_R2": 1.0,
  "kmnist_mean_actual_CEp99_delta": 2.1192762586805554e-07,
  "kmnist_mean_actual_margin_delta": 1.0596381293402777e-07,
  "kmnist_raw_fit_pass_rate": 1.0,
  "kmnist_safe_fit_pass_rate": 1.0,
  "kmnist_target_oracle_useful_rate": 1.0,
  "next_required_implementation": "refine_fashion_delayed_event_timing_and_redesign_kmnist_target_primitive",
  "primary_blocker": "fashion_signal_partial_and_kmnist_requires_target_primitive_redesign",
  "route": "R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker",
  "source_route": "R2-DelayedFashionSignalOnly",
  "success_v9226_external_ready": 0,
  "success_v9226_fashion_delayed_controller": 0,
  "success_v9226_strict_purekan_functional": 0
}
```

## 3. P1 Fashion delayed controller ablation

Artifact：

```text
p1_fashion_delayed_controller_ablation.csv
p1_fashion_delayed_controller_summary.csv
```

| candidate | scope | CEp99 delta | margin delta | beats AdamWParallel | beats best LR | task safe | pass |
|---|---|---:|---:|---:|---:|---:|---:|
| F1-Fashion-O1O2-DelayedController | 50_240_all | `-0.054052` | `-0.017115` | `0.250000` | `0.333333` | `0.916667` | `0` |
| F1-Fashion-O1O2-DelayedController | h240 | `-0.109354` | `-0.040738` | `0.000000` | `0.166667` | `0.833333` | `0` |
| F1-Fashion-O1O2-DelayedController | h50 | `0.001250` | `0.006509` | `0.500000` | `0.500000` | `1.000000` | `0` |
| F2-Fashion-O1-DelayedController | 50_240_all | `-0.051687` | `-0.064145` | `0.166667` | `0.166667` | `0.833333` | `0` |
| F2-Fashion-O1-DelayedController | h240 | `-0.106102` | `-0.140143` | `0.000000` | `0.000000` | `0.666667` | `0` |
| F2-Fashion-O1-DelayedController | h50 | `0.002728` | `0.011853` | `0.333333` | `0.333333` | `1.000000` | `0` |
| F3-Fashion-O2-DelayedController | 50_240_all | `-0.056417` | `0.029915` | `0.333333` | `0.500000` | `1.000000` | `0` |
| F3-Fashion-O2-DelayedController | h240 | `-0.112605` | `0.058667` | `0.000000` | `0.333333` | `1.000000` | `0` |
| F3-Fashion-O2-DelayedController | h50 | `-0.000229` | `0.001164` | `0.666667` | `0.666667` | `1.000000` | `1` |

判断：Fashion delayed signal 是否能进入下一步，取决于 50/240 两个 horizon 是否同时 beat AdamWParallel 和 best LR，并保持 CE tail 与 margin 的同向改善。本轮不把 h80 local signal 直接写成 full controller success。

## 4. P2 KMNIST target/primitive diagnosis

Artifact：

```text
p2_kmnist_target_primitive_diagnosis.csv
```

| metric | value |
|---|---:|
| diagnosis blocker | `KMNIST_realized_effect_not_metric_causal` |
| target oracle useful rate | `1.0` |
| raw fit pass rate | `1.0` |
| safe fit pass rate | `1.0` |
| actual effect pass rate | `0.0` |
| max raw R2 | `1.0` |
| max safe R2 | `1.0` |
| max actual r_z | `0.640339195728302` |
| mean actual CEp99 delta | `2.1192762586805554e-07` |
| mean actual margin delta | `1.0596381293402777e-07` |

按 target：

| target | oracle useful | raw fit | safe fit | actual effect | CEp99 delta | margin delta | max r_z |
|---|---:|---:|---:|---:|---:|---:|---:|
| O1 HardTailLogitCorrection | `1.0` | `1.0` | `1.0` | `0.0` | `0.000000` | `+3.97e-07` | `0.640339` |
| O2 MarginTailExpansion | `1.0` | `1.0` | `1.0` | `0.0` | `+4.77e-07` | `+2.38e-07` | `0.633945` |
| O6 KMNISTHardModeOutputTarget | `1.0` | `1.0` | `1.0` | `0.0` | `+1.59e-07` | `-3.18e-07` | `0.628278` |

判断：KMNIST 本轮只做诊断，不参与 success route。若 oracle 有效但 safe/actual 不过，说明继续调 event threshold 或 cap 不是主要方向，应回 target solver 或 primitive/control surface。

## 5. Downstream boundary

这些 artifact 已落盘，但明确为 `not_run`：

| artifact | reason |
|---|---|
| `p3_fashion_short_run_validation.csv` | route gate 未达到 strict controller success |
| `p4_strict_purekan_functional_reentry.csv` | same |
| `p5_external_ready.csv` | same |

## 6. No-fake audit

```text
rows_checked = 330
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py` | `2b86c605411bca3a10d59faf0b6f113c8174d6b84fcc77c1e8638e67f53186b5` |
| route | `41fd5524421059f8c2fd7d6b25ea8fa4b34c978c6b12541e57c41befaa187ec4` |
| P0 recap | `377f7dc56ef00468ecdbfa24f8fc1e55b7b825977d51dfebe8f61ea2c489d4c9` |
| P1 Fashion ablation | `f6e3ed65cd41e308661f1b6966a5d41172315a89b1d6bb03d6f47ace94b134ea` |
| P1 Fashion summary | `13656face8155a1af911007dcf240f0c768d6fbdb456c6b200171dcf033e67ab` |
| P2 KMNIST diagnosis | `4252f6a9200d2c900c9a79de2f1c266416a8343e5d11d2e1c217de8949bc22b8` |
| provenance audit | `858abecee510ce11787fcc3de06631e2f3c469cdb7fe8a69fae14db533367fb6` |

## 8. 最终分析结论

v9.2.26 的真实推进是：

```text
Fashion: h80 local signal 被提升到 50/240-step ablation；
KMNIST: 不再继续放大，退回 target/primitive diagnosis。
```

机制判断：

1. Fashion 是否能继续，取决于 delayed controller 是否在 50/240 两个 horizon 都能击败 AdamWParallel / LR controls。
2. KMNIST 的问题应从 target oracle、raw controllability、task-safe projection、actual metric effect 四层定位，而不是继续把 scale 当主因。
3. 本轮没有打开 full functional 或 external-ready；所有 downstream 都保持 gate-blocked。

最终一句话：

> v9.2.26 真实执行后停在 `R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker`：Fashion delayed controller ablation 与 KMNIST target/primitive diagnosis 已完成，但 strict PureKAN functional 仍未成功。
