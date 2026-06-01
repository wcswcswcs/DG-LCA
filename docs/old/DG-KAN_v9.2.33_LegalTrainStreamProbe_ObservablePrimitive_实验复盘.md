# DG-KAN v9.2.33 Legal Train-Stream Probe 与 Observable Primitive 实验复盘

> 本复盘记录 `DG-KAN_v9.2.33_LegalTrainStreamProbe_ObservablePrimitive_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R14-ReturnToInterfacePrimitiveDesign
base_candidate = LQ-t2-h256
success_v9233_strict_purekan_functional = False
success_v9233_full_functional = False
success_v9233_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9233_legal_train_stream_probe_observable_primitive_first_20260510T233000Z/
```

核心结论：

1. P0 复现 v9.2.32 boundary：source route = `R13-ReturnToInterfacePrimitiveDesign`，best probe = `CP5-HorizonConsistencyProbe`，fake/proxy = `0`。
2. P1 判定 source CP5 为 `CP5LegalizableViaTrainStreamProbe`；v9.2.32 的 CP5 信号不能直接当 official controller。
3. P2 已执行真实 legal train-stream probe，best legal probe = `LP4-ControlContrastiveVirtualProbe`，corr = `0.182689`，AUC = `0.468670`。
4. P2 precision = `0.238095`，coverage = `0.043210`，bad-event = `0.000000`，system pass = `0`。
5. Observable primitive gate 仍未闭合：best OP = `OP0-current-reference`，OP1-OP6 not_implemented count = `6`。
6. 当前 blocker：`legal_train_stream_probe_not_predictive_and_observable_primitive_not_implemented`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9233_legal_train_stream_probe_observable_primitive.py` | v9.2.33 runner；执行 CP5 legality autopsy、legal train-stream probe、observable primitive gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9233_legal_train_stream_probe_observable_primitive.py
```

正式运行：

```bash
python experiments/run_v9233_legal_train_stream_probe_observable_primitive.py \
  --out-dir results/real_rerun_20260506/v9233_legal_train_stream_probe_observable_primitive_first_20260510T233000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

## 2. Route

```json
{
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "best_legal_probe": "LP4-ControlContrastiveVirtualProbe",
  "best_observable_primitive": "OP0-current-reference",
  "best_observable_primitive_corr": 0.13457096757355733,
  "cp5_best_feature": "horizon_agreement_score",
  "cp5_best_feature_auc": 0.8561248073959938,
  "cp5_legality_classification": "CP5LegalizableViaTrainStreamProbe",
  "cp5_main_source": "horizon_consistency_recomputable",
  "cpu_offload_used": 0,
  "dataset_tuning_detected": 0,
  "external_ready": 0,
  "fake_data_used": 0,
  "fake_proxy_nonzero_count": 0,
  "full_run_pass": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_task_safe": 0,
  "hard_stratum_repair_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "legal_probe_accepted_count": 21,
  "legal_probe_auc": 0.4686697483307653,
  "legal_probe_bad_event_rate": 0.0,
  "legal_probe_controller_candidate_pass": 0,
  "legal_probe_corr": 0.18268873425020277,
  "legal_probe_coverage": 0.043209876543209874,
  "legal_probe_memory_ratio": 1.6603190104166665,
  "legal_probe_precision": 0.23809523809523808,
  "legal_probe_predictive_pass": 0,
  "legal_probe_recall": 0.03787878787878788,
  "legal_probe_step_q90": 23.828452734609098,
  "legal_probe_system_pass": 0,
  "next_required_implementation": "if_legal_probe_failed_implement_real_OP1_OP6_else_extract_cheap_sufficient_statistics",
  "no_fake": true,
  "no_proxy": true,
  "observable_primitive_not_implemented_count": 6,
  "observable_primitive_pass": 0,
  "paired_replay_pass": 0,
  "primary_blocker": "legal_train_stream_probe_not_predictive_and_observable_primitive_not_implemented",
  "probe_controller_pass": 0,
  "proxy_row_used": 0,
  "robustness_pass": 0,
  "route": "R14-ReturnToInterfacePrimitiveDesign",
  "rows_checked": 3429,
  "short_run_pass": 0,
  "strong_baseline_pass": 0,
  "success_v9233_external_ready": 0,
  "success_v9233_full_functional": 0,
  "success_v9233_strict_purekan_functional": 0,
  "v9232_boundary_pass": 1
}
```

## 3. P1 Source CP5 Legality Autopsy

Artifact：

```text
p1_source_logged_cp5_legality_autopsy.csv
```

关键值：

| metric | value |
|---|---:|
| CP5 classification | `CP5LegalizableViaTrainStreamProbe` |
| CP5 best feature | `horizon_agreement_score` |
| CP5 best feature AUC | `0.856125` |
| posthoc weight | `0.300000` |
| legalizable weight | `1.200000` |

判断：CP5 的 horizon-consistency 逻辑可以尝试重算为 train-stream probe，但 v9.2.32 source 字段本身含 posthoc 成分，不能直接合法化。

## 4. P2 Legal Train-Stream Probe

Artifacts：

```text
p2_legal_train_stream_probe_implementation.csv
legal_train_stream_probe_trace_v9233.csv
```

P2 使用同一 train split 的 update batch 与 probe batch；不使用 validation/test metric，不用 dataset name 做 commit rule。

| metric | value |
|---|---:|
| best probe | `LP4-ControlContrastiveVirtualProbe` |
| corr | `0.182689` |
| AUC | `0.468670` |
| precision | `0.238095` |
| coverage | `0.043210` |
| bad event | `0.000000` |
| step q90 | `23.828453` |
| predictive pass | `0` |
| system pass | `0` |

判断：本轮没有把 source-logged CP5 伪装成合法 probe；P2 数值来自新计算的 train-stream virtual probe。

## 5. P5 Observable Primitive Gate

Artifact：

```text
p5_observable_primitive_implementation_gate.csv
```

```text
observable_primitive_pass = 0
best_observable_primitive = OP0-current-reference
best_observable_primitive_corr = 0.134571
not_implemented OP rows = 6
```

判断：OP0/current reference 没有形成 observable primitive pass；OP1-OP6 本轮仍明确 not_implemented，没有写成失败训练或成功。

## 6. Downstream Boundary

P6-P10 只有在 legal probe controller 或 observable primitive survivor 后打开。本轮均已落盘为 `not_run`。

## 7. No-fake audit

```text
rows_checked = 3429
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
| runner | `6e1d720f7597a6beaadef976494c10b2d2e7c221db4767e75d224ea42859477c` |
| route | `56546cdebfca6ae6e4dfb87245f0e5117c6960712dae4def4864641e9997ba31` |
| P1 autopsy | `efab9b0f5282df60e7fab1738d7e7ae9a3d481749002d76f097adad31a826390` |
| P2 legal probe | `3080ed021b94c4af8b134358bfb310592f4129eeb5dbf2d145b2786857ffd62f` |
| P5 primitive gate | `243908945628aafcda91845e9c839f77da5d045aede675ac4e261637f647b979` |
| provenance audit | `75dd59a0a8f303b55299d0510af58dd2f6906af72c8bea66e4b8e4d6f7b950a5` |

## 9. 最终分析结论

v9.2.33 的真实推进是：

```text
v9.2.32: source-logged CP5 有分类信号，但 legality 不过。
v9.2.33: CP5 被重做为 legal train-stream probe audit；
          official downstream 仍取决于 predictive / accept / system gate。
```

机制判断：

1. 本轮把 “source-logged signal” 和 “legal commit-time signal” 分开了，避免把后验 replay 信息当作 online controller。
2. legal probe 即使有局部分类能力，也必须同时满足 precision / coverage / bad-event / system gate。
3. 如果 P2 不过，正确下一步不是继续 dataset patch，而是实现真正 observable primitive，或者提取 cheaper sufficient statistics。
4. 当前 route 停在 `R14-ReturnToInterfacePrimitiveDesign`，原因是 `legal_train_stream_probe_not_predictive_and_observable_primitive_not_implemented`。

最终一句话：

> v9.2.33 真实执行后停在 `R14-ReturnToInterfacePrimitiveDesign`：`legal_train_stream_probe_not_predictive_and_observable_primitive_not_implemented`。
