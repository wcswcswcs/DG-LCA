# DG-KAN v9.2.41 Control-Gap Value Alignment 与 RoleWise Late-Attach 实验复盘

> 本复盘记录 `DG-KAN_v9.2.41_ControlGap_ValueAlignment_RoleWiseLateAttach_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R3-ControlGapScorePass
base_candidate = LQ-t2-h256
success_v9241_strict_purekan_functional = False
success_v9241_full_functional = False
success_v9241_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9241_controlgap_value_alignment_rolewise_lateattach_first_20260511T133000Z
```

核心结论：

1. P0 复现 v9.2.40 boundary：source route = `R11-BasePreservedButValueUnobservable`，base / attach / carrier gate 均为 `1`，value gate 为 `0`。
2. P1 value failure autopsy pass = `1`，primary mode = `F5-score_sign_mismatch`，primary fraction = `0.95`。
3. P2 并行 score matrix 中，best legal score = `S7-HybridMonotoneLegal`，value observability pass = `1`。
4. S7 指标：AUC `0.736068`，corr `0.343287`，accepted precision `0.777778`，coverage `0.04`，bad-event `0.0`。
5. P3 role-wise late attach audit pass = `1`，A3 已在 v9.2.40 source artifacts 中具备 strict contract / inactive equivalence / no-event preservation。
6. P4 oracle upper bound pass = `1`，oracle precision `1.0`，coverage `0.12`；shuffle controls 均未通过。
7. P5 leave-dataset-out pass = `0`，leave-stratum-out 不可评估：v9.2.40 source rows 只有一个 signal stratum。
8. 当前 blocker：`leave_dataset_or_stratum_out_failed_after_value_score_pass`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9241_controlgap_value_alignment_rolewise_lateattach.py` | v9.2.41 runner；从 v9.2.40 真实 carrier/value rows 生成 P0-P9 artifacts、score matrix、oracle/shuffle、leave-out gate、route、no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9241_controlgap_value_alignment_rolewise_lateattach.py
```

正式运行：

```bash
python experiments/run_v9241_controlgap_value_alignment_rolewise_lateattach.py \
  --out-dir results/real_rerun_20260506/v9241_controlgap_value_alignment_rolewise_lateattach_first_20260511T133000Z \
  --fresh --device auto --data-root data --seed 1314
```

说明：本轮 P1-P5 是对 v9.2.40 source-measured P5/P6 carrier/value rows 的 value alignment audit。没有补造 h640 rows；未测 ECE/NLL/curvature 在 P5 写为 `not_measured_in_v9240_source`。

## 2. Route

```json
{
  "route": "R3-ControlGapScorePass",
  "base_candidate": "LQ-t2-h256",
  "v9240_boundary_pass": 1,
  "source_route": "R11-BasePreservedButValueUnobservable",
  "value_failure_mode": "F5-score_sign_mismatch",
  "value_failure_attribution_pass": 1,
  "score_sign_mismatch_pass": 1,
  "best_score_id": "S7-HybridMonotoneLegal",
  "best_attach_candidate": "A3-LateAttachRoleWiseFT7EdgeCarrier",
  "value_observability_pass": 1,
  "value_auc": 0.736068376068376,
  "value_corr": 0.3432872708365391,
  "accepted_precision": 0.7777777777777778,
  "accepted_coverage": 0.04,
  "accepted_bad_event_rate": 0.0,
  "oracle_upper_bound_pass": 1,
  "rolewise_attach_pass": 1,
  "shuffle_control_any_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "external_ready": 0,
  "primary_blocker": "leave_dataset_or_stratum_out_failed_after_value_score_pass",
  "success_v9241_strict_purekan_functional": 0,
  "success_v9241_full_functional": 0,
  "success_v9241_external_ready": 0
}
```

判断：本轮把 v9.2.40 的 value failure 推进到 score 层面，说明 legal S7 在 source matrix 上可以识别一小批 good events。但它不能通过 leave-out，所以不能打开 official paired replay。

## 3. P1 value failure autopsy

Artifact：

```text
p1_value_failure_autopsy.csv
value_failure_trace_v9241.csv
```

关键值：

| metric | value |
|---|---:|
| current corr | `-0.259211` |
| current AUC | `0.443590` |
| current precision | `0.090909` |
| sign-flip corr | `0.259211` |
| sign-flip AUC | `0.556410` |
| control-gap corr | `0.984739` |
| accepted failed event count | `20` |
| primary failure mode | `F5-score_sign_mismatch` |
| primary failure fraction | `0.95` |

判断：v9.2.40 的 value score 不是单纯弱信号，而是方向错配；accepted events 大多 task-safe，但不是 control-resistant good events。

## 4. P2 parallel score redesign matrix

Artifact：

```text
p2_parallel_score_redesign_matrix.csv
score_component_trace_v9241.csv
```

Score summary：

| score | official eligible | official pass | AUC | corr | best precision | best coverage | bad event | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S0-CurrentV9240Score | 1 | 0 | `0.443590` | `-0.259211` | `0.142857` | `0.062222` | `0.0` |  |
| S1-SignFlippedCurrentScore | 0 | 0 | `0.556410` | `0.259211` | `0.666667` | `0.040000` | `0.0` | diagnostic only |
| S2-DirectControlGapField | 1 | 0 | `0.443590` | `-0.259211` | `0.142857` | `0.062222` | `0.0` |  |
| S3-ConservativeLCB-UCB | 1 | 0 | `0.210085` | `-0.175184` | `0.0` | `0.151111` | `0.058824` |  |
| S4-TailCEMarginPosthoc | 0 | 0 | `1.000000` | `0.070306` | `1.000000` | `0.120000` | `0.0` | posthoc outcome delta |
| S5-RoleWiseFT7Score | 1 | 0 | `0.576923` | `-0.152690` | `0.285714` | `0.031111` | `0.0` |  |
| S6-FamilyValueReliability | 0 | 0 | `0.500000` | `0.0` | `0.571429` | `0.031111` | `0.0` | posthoc family calibration |
| S7-HybridMonotoneLegal | 1 | 1 | `0.736068` | `0.343287` | `0.777778` | `0.040000` | `0.0` |  |
| S8-OracleUpperBound | 0 | 0 | `1.000000` | `1.000000` | `1.000000` | `0.120000` | `0.0` | oracle/posthoc |

判断：

1. S7 是本轮唯一 legal value observability survivor。
2. S4/S8 虽然强，但使用 posthoc outcome/oracle，不能进入 official route。
3. S7 过 P2 不等于 functional success；它必须继续通过 P5 leave-out 和 P6 paired replay。

## 5. P3 role-wise late attach audit

Artifact：

```text
p3_rolewise_late_attach_implementation.csv
rolewise_attach_trace_v9241.csv
```

| attach candidate | implemented | contract | inactive equivalence | no-event preservation | pass |
|---|---:|---:|---:|---:|---:|
| A3-LateAttachRoleWiseFT7EdgeCarrier | `1` | `1` | `1` | `1` | `1` |
| A5-FamilyValueLateAttach | `1` | `1` | `0` | `0` | `0` |

说明：P3 复用 v9.2.40 的 source-measured attach/equivalence/no-event rows；没有把 A5 未过的 inactive/no-event 写成通过。

## 6. P4 oracle upper-bound 与 shuffle controls

Artifact：

```text
p4_oracle_upper_bound_and_shuffle_controls.csv
oracle_upper_bound_trace_v9241.csv
```

| control | AUC | precision | coverage | bad event | pass |
|---|---:|---:|---:|---:|---:|
| OracleUpperBound | `1.000000` | `1.000000` | `0.120000` | `0.0` | `1` |
| ValueScoreShuffled | `0.604103` | `0.222222` | `0.080000` | `0.0` | `0` |
| FunctionalChannelShuffled | `0.470085` | `0.285714` | `0.031111` | `0.142857` | `0` |
| TailMaskShuffled | `0.452308` | `0.222222` | `0.080000` | `0.055556` | `0` |
| RoleScoreShuffled | `0.510256` | `0.428571` | `0.031111` | `0.0` | `0` |

判断：good events 确实存在，且 shuffle controls 未过；当前问题不是 carrier 完全无 value，而是 legal score 的泛化和稳定性没有闭合。

## 7. P5 leave-out gate

Artifact：

```text
p5_leave_dataset_and_stratum_out_validation.csv
leave_dataset_out_trace_v9241.csv
```

Leave-dataset-out：

| heldout | precision | coverage | bad event | task safe | beats AdamWParallel | pass |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST | `0.400000` | `0.666667` | `0.100000` | `0.900000` | `0.400000` | `0` |
| KMNIST | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |
| MNIST | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0.000000` | `0` |

Summary：

```text
leave_dataset_out_pass = 0
leave_dataset_out_pass_count = 0/3
leave_stratum_out_evaluable = 0
leave_stratum_out_pass = 0
reason = only_one_signal_stratum_measured_in_v9240_source
```

判断：S7 在全量 source rows 上能过 value gate，但在 leave-dataset-out 下完全不稳定。P6 official paired replay 因此不能打开。

## 8. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p6_official_paired_replay.csv` | `P5_leave_dataset_or_stratum_out_failed` |
| `p7_short_run_functional_validation.csv` | same |
| `p8_full_10seed_functional_validation.csv` | same |
| `p9_robustness_external_ready.csv` | same |

## 9. No-fake audit

```text
rows_checked = 2278
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `14828ea68c28f3cd228b934286126f886cc1c51cffb311e2954d6fe287ad377c` |
| runner | `e361197485da95501e860fea8d7afa6414a75f41bc0755efdb9475761cae9896` |
| run manifest | `e765ead0b391500091dc9e424a1b5b824fad0073217ec940e141f82a0f45c6b8` |
| route | `843eaa1aeaf03ef91ba18f01e839a5d95433c3672bfea3b390ea3488b6a95846` |
| P0 boundary | `a90c9b39cf2799c66042f7bb6be534781189541ac5ed7c10b89e9534ea6f2b0b` |
| P1 autopsy | `cdec17baf44067f0f7b319d3fa60b114406faa6d9c2a6e84859c589a924f3cbb` |
| P2 score matrix | `307cd12b804cc250ebabe99da616226e30b18c883a664d34927912f79810658c` |
| P3 rolewise attach | `ef89e0b63967ad5aa8ba5a126cbc2d2f554bea0649bc49e71cc06d77b25d702b` |
| P4 oracle/shuffle | `db52ce8e6e77a4ee94bb70b316dab92ab289a900ecacf67d2d9eb7bfc08ca128` |
| P5 leave-out | `3c1ce4c00e5b53ce560737b089a8ab4829e0241c866a4f97d328a2e73752bc5f` |
| P6 not-run boundary | `5a65f4ff895c603a82ed1fbbdfb4a145a994ea180ae9918088024f96c56c7708` |
| provenance audit | `1095ad123da29494f52900ee6a7351c78bcc0aee3bfd54b3ea39c43a6d8c5088` |

## 11. 最终分析结论

v9.2.41 的真实推进是：

```text
v9.2.40: base / snapshot attach / carrier 均过，value observability fail。
v9.2.41: value failure 被归因为 score sign mismatch；
          S7 legal hybrid score 在 source matrix 上过 value gate；
          但 leave-dataset/stratum gate fail，official paired replay 不允许打开。
```

机制判断：

1. 本轮确认 v9.2.40 不是 carrier 完全无 value。Oracle upper bound 说明 good events 存在。
2. 当前 score 的方向错配是真实 blocker；S7 说明 sign-corrected hybrid features 可以局部恢复 value observability。
3. 但 S7 没有通过 leave-dataset-out，尤其 heldout MNIST/KMNIST 没有保留 accepted good events，Fashion 也只有 precision `0.4` 且 bad-event `0.1`。
4. 因此不能进入 official paired replay、short-run、full-run 或 external-ready。
5. 下一步应稳定 legal score 的 leave-out 泛化，补足多 signal stratum / h640 source-measured features，再谈 official paired replay。

最终一句话：

> v9.2.41 真实执行后停在 `R3-ControlGapScorePass`：`S7-HybridMonotoneLegal` 通过 source value gate，但 `leave_dataset_or_stratum_out_failed_after_value_score_pass`，strict PureKAN functional 仍未成功。
