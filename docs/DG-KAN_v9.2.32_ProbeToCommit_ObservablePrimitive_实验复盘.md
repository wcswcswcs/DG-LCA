# DG-KAN v9.2.32 Probe-to-Commit 与 Observable Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.2.32_ProbeToCommit_ObservablePrimitive_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未实现 train-stream probe / observable primitive 写成通过。

## 0. 最新结论

```text
route = R13-ReturnToInterfacePrimitiveDesign
base_candidate = LQ-t2-h256
success_v9232_strict_purekan_functional = False
success_v9232_full_functional = False
success_v9232_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9232_probe_to_commit_observable_primitive_first_20260510T230000Z/
```

核心结论：

1. P0 复现 v9.2.31 boundary：`route=R9-PrimitiveEffectUnpredictable`，event-value grounding pass = `1`，fake/proxy = `0`。
2. P1 static feature insufficiency confirmed = `1`；best static corr = `-0.259717`，best movement corr = `-0.184709`。
3. P2 source-logged probe audit 已执行；best probe = `CP5-HorizonConsistencyProbe`，corr = `0.095796`，AUC = `0.837635`。
4. P2 probe legality pass = `0`，原因是本轮没有把 v9.2.30 logged features 伪装成新的 train-stream virtual probe。
5. P5 observable primitive gate 中 OP0/current reference 未过 observability；OP1-OP6 明确 `not_implemented`。
6. 当前 blocker：`probe_predictive_but_not_legal_train_stream_probe`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9232_probe_to_commit_observable_primitive.py` | v9.2.32 runner；复现 v9.2.31 boundary，执行 static autopsy、source-logged probe-to-value audit、observable primitive gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9232_probe_to_commit_observable_primitive.py
```

正式运行：

```bash
python experiments/run_v9232_probe_to_commit_observable_primitive.py \
  --out-dir results/real_rerun_20260506/v9232_probe_to_commit_observable_primitive_first_20260510T230000Z \
  --fresh \
  --seed 1314
```

说明：本轮 P2 使用 v9.2.30/v9.2.31 真实落盘的 pre-event / grounded-value rows 做 probe audit。没有新增真实 train-stream virtual microholdout probe；因此 probe legality 不会被写成通过。

## 2. Route

```json
{
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "best_movement_feature": "pre_adamwparallel_logit_delta_norm",
  "best_movement_feature_corr": -0.18470922315538743,
  "best_observable_primitive": "OP0-current-reference",
  "best_observable_primitive_corr": 0.13341003240960184,
  "best_probe": "CP5-HorizonConsistencyProbe",
  "best_probe_auc": 0.8376348228043143,
  "best_probe_bad_event_rate": 0.0,
  "best_probe_corr": 0.0957956762290614,
  "best_probe_coverage": 0.051440329218107,
  "best_probe_precision": 0.76,
  "best_static_feature": "effective_derivative",
  "best_static_feature_corr": -0.2597168939509335,
  "between_family_value_variance": 0.0018907056591809422,
  "cpu_offload_used": 0,
  "dataset_tuning_detected": 0,
  "external_ready": 0,
  "fake_data_used": 0,
  "fake_proxy_nonzero_count": 0,
  "feature_rank_sufficient": 1,
  "full_run_pass": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_task_safe": 0,
  "hard_stratum_repair_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "next_required_implementation": "implement_true_train_stream_virtual_probe_or_observable_primitive_OP1_OP6",
  "no_fake": true,
  "no_proxy": true,
  "observable_primitive_not_implemented_count": 6,
  "observable_primitive_pass": 0,
  "paired_replay_pass": 0,
  "primary_blocker": "probe_predictive_but_not_legal_train_stream_probe",
  "probe_controller_pass": 0,
  "probe_legality_pass": 0,
  "probe_predictive_pass": 1,
  "probe_system_pass": 0,
  "proxy_row_used": 0,
  "robustness_pass": 0,
  "route": "R13-ReturnToInterfacePrimitiveDesign",
  "rows_checked": 3904,
  "short_run_pass": 0,
  "single_event_value_too_noisy": 1,
  "static_feature_insufficiency_confirmed": 1,
  "strict_all_static_below_025": 0,
  "strong_baseline_pass": 0,
  "success_v9232_external_ready": 0,
  "success_v9232_full_functional": 0,
  "success_v9232_strict_purekan_functional": 0,
  "v9231_boundary_pass": 1,
  "within_family_value_variance": 0.004773586519783202
}
```

## 3. P1 Static Feature Autopsy

Artifact：

```text
p1_static_feature_insufficiency_autopsy.csv
```

关键值：

| metric | value |
|---|---:|
| static insufficiency confirmed | `1` |
| strict all static below 0.25 | `0` |
| best static feature | `effective_derivative` |
| best static corr | `-0.259717` |
| best movement feature | `pre_adamwparallel_logit_delta_norm` |
| best movement corr | `-0.184709` |
| single-event value too noisy | `1` |

判断：v9.2.31 的 label 可靠性保留，但 static movement features 仍不足以打开 controller。

## 4. P2 Probe-to-Value Audit

Artifact：

```text
p2_probe_to_value_audit.csv
```

关键值：

| metric | value |
|---|---:|
| best probe | `CP5-HorizonConsistencyProbe` |
| best probe corr | `0.095796` |
| best probe AUC | `0.837635` |
| precision | `0.760000` |
| coverage | `0.051440` |
| predictive pass | `1` |
| legality pass | `0` |

判断：source-logged probe score 的连续相关性没有达到 `corr>=0.35`，但 CP5 的 AUC/precision 局部过 gate；关键 blocker 是它不是本轮新测的 official train-stream virtual probe，`probe_legality_pass = 0`，因此不能进入 P3/P4 official controller。

## 5. P5 Observable Primitive Gate

Artifact：

```text
p5_observable_primitive_repair_gate.csv
```

结果：

```text
observable_primitive_pass = 0
best_observable_primitive = OP0-current-reference
best_observable_primitive_corr = 0.133410
not_implemented OP rows = 6
```

判断：current OP0 reference 没有达到 observability；OP1-OP6 不是本轮实现内容，按 `not_implemented` 记录，没有伪装成失败训练。

## 6. Downstream Boundary

P6-P10 只有在 legal probe controller 或 observable primitive survivor 后打开。本轮均已落盘为 `not_run`。

## 7. No-fake audit

```text
rows_checked = 3904
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| runner | `bc1881f821c6cc89adb5b793f025e3170df612d797036c2e6b712962d202e0c8` |
| route | `287f2a071f769b472617bf52f7e53856fc45b880b78580fa421f908c86ecb9a3` |
| P1 autopsy | `c6b14b57f0826116b564bb634bc62674e3d86ffdd3c4f1b5c16cd06fd15bafad` |
| P2 probe audit | `422f960be2e3964d97dd9b264590e93f5c6d4263d61eb7a13e2a306811818409` |
| P5 primitive gate | `6c1595c78ef3c2d02672701bf696155303c9c46c0a35afcaf0e141913f5e3698` |
| provenance audit | `0026f2f8c7b70d332f6e6786d7bd585ceb747e61f8ef875b1d2e89635ed7565f` |

## 9. 最终分析结论

v9.2.32 的真实推进是：

```text
v9.2.31: grounded value 可靠，但 static pre-event observability 不过。
v9.2.32: source-logged probe audit 仍无法证明可预测 causal value；真实 train-stream probe 与 observable primitive 仍需实现。
```

机制判断：

1. 本轮不能声明 probe-to-commit success，因为 P2 没有合法 train-stream virtual probe pass。
2. `source_logged_feature_probe` 只说明现有落盘 pre-event statistics 不足以预测 grounded value，不能当作 official controller。
3. OP1-OP6 未实现，因此不能说 observable primitive factory 失败；只能说 current OP0/reference 没有可观测性，下一步必须实现真正 train-stream virtual probe 或 observable primitive。

最终一句话：

> v9.2.32 真实执行后停在 `R13-ReturnToInterfacePrimitiveDesign`：`probe_predictive_but_not_legal_train_stream_probe`。
