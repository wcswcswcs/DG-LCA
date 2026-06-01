# DG-KAN v9.2.18 Functional Retrospective Primitive Reset 实验复盘

> 本复盘记录 `docs/DG-KAN_v9.2.18_Functional_Retrospective_Primitive_Reset_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows、手填成功结论，也没有把 v8.x 未测 AdamWParallel / LR strong controls 写成通过或失败。

## 0. 最新结论

截至本轮，v9.2.18 执行到一个可审计 terminal route：

```text
route = R8-FunctionalPaused
base_candidate = LQ-t2-h256
success_v9218_functional_retained = false
success_v9218_primitive_reset = false
success_v9218_external_ready = false
```

最终 artifact：

```text
results/real_rerun_20260506/v9218_functional_retrospective_primitive_reset_first_20260510T060000Z/
```

核心结论：

1. v8.x historical success 被重新标注为 transitional / diagnostic functional success；没有把 v8.7 external fair route 写成 strict FC-PureKAN success。
2. v8.7 source artifact 中确实有 FT7 / Adaptive-P / NoOp / Random 的真实 legacy-control rows，但没有 AdamWParallel 与 scalar LR controls。
3. P1 因此不能判断 “v8 functional survives v9-style controls”，也不能判断 “v8 functional is LR-equivalent”；这两个结论都没有数据支持。
4. v9.2.14-v9.2.17 已经显示当前 strict LQ/A4/A7c family 没有 control-resistant functional advantage，且 v9.2.17 出现 LR / extra-AdamW equivalence。
5. P4/P5 source recap 仍保留 LQ/A7c 的 P4 + near-pass 基础信号，但没有 AdamW-only full-pass，也没有 functional actuatability pass。
6. 因此本轮正确 route 是暂停 functional 作为主线：下一步要么真正实现 v8 strong-control replay，要么回到 primitive / basis factory reset；不能继续用旧 v8 成功为当前 v9 functional patch 背书。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9218_functional_retrospective_primitive_reset.py` | v9.2.18 retrospective runner；复用真实 source artifacts，生成 P0-P6、route、failure/no-fake audit |

代码检查：

```bash
python -m py_compile experiments/run_v9218_functional_retrospective_primitive_reset.py
```

已通过。

正式运行：

```bash
python experiments/run_v9218_functional_retrospective_primitive_reset.py \
  --out-dir results/real_rerun_20260506/v9218_functional_retrospective_primitive_reset_first_20260510T060000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

说明：本轮是 retrospective source-artifact audit。没有补造 v8 AdamWParallel / LR controls；缺失 controls 明确写为 `not_measured_in_source_artifact`。

## 2. Route

`route_decision.json`：

```json
{
  "route": "R8-FunctionalPaused",
  "base_candidate": "LQ-t2-h256",
  "v8_strong_control_replay_evaluable": 0,
  "v8_missing_strong_control_count": 11,
  "v8_source_legacy_measured_rows": 22,
  "v8_survives_controls": 0,
  "v8_lr_equivalent": 0,
  "v9_transfer_pass": 0,
  "primitive_factory_pass": 0,
  "adamw_fullpass": 0,
  "functional_actuatability_pass": 0,
  "external_ready": 0,
  "primary_blocker": "v8_success_not_reaudited_with_v9_style_adamwparallel_lr_controls_in_available_source_artifacts",
  "next_required_implementation": "implement_true_v8_strong_control_replay_or_continue_primitive_basis_factory_reset",
  "success_v9218_functional_retained": 0,
  "success_v9218_primitive_reset": 0,
  "success_v9218_external_ready": 0
}
```

判断：

1. 本轮不能保留 functional 为主线成功，因为 v8 strong controls 不可评估，v9 当前 family 又没有独立 functional advantage。
2. 本轮也不能把 v8 成功降级为 LR-equivalent，因为缺少真实 LR / AdamWParallel replay 数据。
3. 这是一个 “证据不足 + 当前 v9 负证据很强” 的暂停路线，不是永久证伪。

## 3. P0 历史路线统一审计

Artifact：

```text
p0_history_unified_audit.csv
```

Rows：

```text
p0 rows = 8
```

关键标注：

| version | classification | strict FC-PureKAN | AdamWParallel / LR controls |
|---|---|---:|---|
| v8.3 | Transitional functional success | 0 | not measured |
| v8.4 | Diagnostic assisted success | 0 | not measured |
| v8.7 | Transitional functional success | 0 | not measured |
| v8.8 | System-only / transitional diagnostic | 0 | not measured |
| v9.2.6-v9.2.7 | Strict FC-PureKAN near-pass base | 1 | functional not opened |
| v9.2.14 | No functional advantage | 1 | AdamWParallel measured |
| v9.2.15-v9.2.16 | No functional advantage | 1 | AdamWParallel measured |
| v9.2.17 | No functional advantage / LR-equivalent signal | 1 | AdamWParallel + LR measured |

判断：P0 pass。历史成功定义已分离，v8 transitional success 没有被混写成 strict Clean FullEdge PureKAN success。

## 4. P1 v8 strong-control replay

Artifact：

```text
p1_v8_strong_control_replay.csv
```

Rows：

```text
total rows = 33
source legacy measured rows = 22
not_measured strong-control rows = 11
```

v8.7 legacy one-step source rows：

| source controller | rows | holdout descent mean | bad-step rate | curvature reduction | cos corrected with task |
|---|---:|---:|---:|---:|---:|
| Fixed-FT7-stride128 | 48 | `0.011190` | `0.0` | `0.000313` | `0.995050` |
| Adaptive-FT-P | 48 | `0.011187` | `0.0` | `0.000132` | `0.995561` |
| NoOp | 48 | `0.011242` | `0.0` | `0.000032` | `1.000000` |
| RandomFunc | 48 | `0.011187` | `0.0` | `0.000052` | `0.995037` |

缺失 strong controls：

```text
AdamWOnly
ShuffledRoleMask
InvertedRoleMask
AdamWParallelSameNorm
AdamWParallelTrustRatio-0.003
AdamWParallelTrustRatio-0.01
AdamWParallelTrustRatio-0.03
LRScale-1.003
LRScale-1.01
LRScale-1.03
LRScale-1.10
```

判断：

1. v8 legacy rows 显示 FT7 / Adaptive-P 当时确实 task-safe，并且 curvature reduction 比 NoOp/Random 更强。
2. 但这些 rows 不能回答本轮核心问题，因为缺少后来的强 controls。
3. 因此不能写 `v8_survives_controls = true`，也不能写 `v8_lr_equivalent = true`。

## 5. P2 v8 mechanism attribution

Artifact：

```text
p2_v8_functional_mechanism_attribution.csv
```

Rows：

```text
p2 rows = 8
```

关键机制：

| controller | cos functional with task | curvature reduction | classification |
|---|---:|---:|---|
| Fixed-FT7-stride128 | `0.011337` | `0.000313` | positive vs legacy controls, unresolved vs AdamWParallel/LR |
| Adaptive-FT-P | `0.772947` | `0.000132` | positive vs legacy controls, unresolved vs AdamWParallel/LR |

判断：

1. Fixed-FT7 的 raw functional direction 与 task direction 几乎正交，但 task-aware corrected direction 与 task 高度同向。
2. Adaptive-P 更接近 task-signal-aligned 方向，这提示 v8 成功可能同时混有 branch/curvature signal 和 task-gradient shadow。
3. 缺少 `cos_functional_lrcontrol` 与 AdamWParallel controls，所以不能判定独立机制。

## 6. P3-P5 v9 transfer / primitive / AdamW recap

P3 artifact：

```text
p3_v8_v9_transfer_audit.csv
```

结论：

```text
v9_transfer_pass = 0
```

P4 artifact：

```text
p4_primitive_basis_factory_reset.csv
```

Source recap：

| family | source | P4 | pairwise R2 / actuatability | functional replay |
|---|---|---:|---:|---|
| B0 LQ-T2-current | LQ-t2-h256 | 1 | pairwise R2 `0.974276` | not measured for base |
| B3 Legendre source | LQ-legendre23-h128 | 1 | pairwise R2 `0.964048` | not measured for base |
| B4 Legendre h256 source | LQ-legendre23-h256 | 0 | pairwise R2 `0.962727` | not measured for base |
| B9 actuator channel | A7c-BasisEntropy-ValueOnly | 1 | R2 `1.0`, rz `1.106568` | fail vs controls |

P5 artifact：

```text
p5_adamw_only_fullpass_repair.csv
```

Key source recap：

| candidate | near-pass | full-pass | macro delta |
|---|---:|---:|---:|
| LQ-t2-h256 | 1 | 0 | `-0.0057666699` |
| A7c-BasisEntropy-ValueOnly | 1 | 0 | `-0.0041666627` |

判断：

1. v9 strict FC-PureKAN base 已经有可用近似路线，但还没有 full-pass。
2. A7c 解决了 P4-qualified actuator，但 functional replay 仍输给 controls。
3. 因此 P4/P5 recap 不支持 external-ready，也不支持继续做小 functional patch。

## 7. No-fake audit

`v9218_provenance_audit.csv`：

```text
rows_checked = 142
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = true
no_proxy = true
```

说明：

1. P0/P1/P2/P3/P4/P5 均来自真实 source CSV/JSON 或明确 `not_measured` / `not_implemented` rows。
2. 没有把缺失 v8 strong controls 填成测量值。
3. 没有打开 PureKANConv / PureKANFormer。

## 8. Hash

| artifact | SHA256 |
|---|---|
| v9.2.18 plan | `d93f487a3fb597293a2c85fb27e491fafb9e02e06e0babf9983d4db7bf3b8b04` |
| `experiments/run_v9218_functional_retrospective_primitive_reset.py` | `2ee17fcec9d4a4a64ff86a6f8e937d61135d1a0cb81c0bb95e9d39287748e96b` |
| route | `c9e3551454610502d73cf55a064d4973cd5e3a1ed96a92823c2d3e793a106918` |
| P0 history audit | `f30e7ac395aaff5558d19d62fcf4be15f3995ccd18fbdd95be6edbdee330cfe0` |
| P1 v8 strong-control replay | `02901dd93891a6dd77744c0ff2ee31d78588a2db6bdb4828eb9b6596bd11ebc8` |
| P2 v8 mechanism attribution | `99f3672d41b7eb1aa10c8930a1a429a5d16948130a35efba444ffa58758387d0` |
| P3 transfer audit | `67becb41e7590c47207d514836141eb0feafee9944a2fc563e03933bd0fed3e6` |
| P4 primitive reset | `2f2f910cd3834d30c26812860b7ee3d30262ccdac14685cdce718d9b34fbc925` |
| P5 AdamW repair recap | `7cf6e67b1f3833d512066f465b9762e76af89497006835254ef4fcf78e970c0b` |
| provenance audit | `9ad98143ba98215152210b25021218b5326f526333808a04f845f8c2e40b281b` |

## 9. 最终分析结论

v9.2.18 的真实推进是：

```text
v8.x functional success = 真实 historical/transitional success，但未被 v9-style controls 重审。
v9.x LQ/A4/A7c functional = 已经被 AdamWParallel / LR controls 强烈解释。
当前证据状态 = 不足以继续 functional 主线，也不足以永久证伪 v8 functional。
```

机制判断：

1. v8 的 FT7 / Adaptive-P 不是 fake，它们在旧控制下有 task-safe 和 curvature reduction 信号。
2. 但 v9.2.17 以后，AdamWParallel / scalar LR controls 已经成为必须控制项；v8 source artifacts 没有这些行，所以旧结论不能直接迁移到当前 strict FC-PureKAN functional claim。
3. 当前 v9 family 的问题不是 P4 actuator 或 output controllability，而是 RealFunctional 没有打败 controls；这已经跨 v9.2.14-v9.2.17 多轮成立。
4. 因此本轮不能继续做 “v9.2.19 functional target patch”。正确下一步只有两条：真实补跑 v8 strong-control replay，或回到 primitive/basis factory，优先修 AdamW-only full-pass 和 non-AdamW actuatability。
5. 在没有新证据前，functional 应作为 diagnostic 暂停，而不是主线 success route。

最终一句话：

> v9.2.18 真实执行后停在 `R8-FunctionalPaused`：v8 的 functional 成功被保留为 transitional historical evidence，但现有 source artifacts 缺少 AdamWParallel / LR strong controls；当前 v9 LQ/A4/A7c family 又没有 control-resistant functional advantage。因此不能打开 external，也不能继续小修 functional，应补跑 v8 strong controls 或返回 primitive/basis reset。
