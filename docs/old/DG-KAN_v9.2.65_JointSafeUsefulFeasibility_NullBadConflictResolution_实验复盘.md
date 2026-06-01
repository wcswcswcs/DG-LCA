# DG-KAN v9.2.65 Joint Safe-Useful Feasibility 与 Null-Bad Conflict Resolution 实验复盘

> 本复盘记录 `DG-KAN_v9.2.65_JointSafeUsefulFeasibility_NullBadConflictResolution_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R14-JointSafeUsefulStillTiny
base_candidate = LQ-t2-h256
success_v9265_strict_purekan_functional = False
success_v9265_full_functional = False
success_v9265_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution_first_20260512T203000Z/
```

核心结论：

1. P0 复现 v9.2.64 boundary：source route = `R13-SafeUsefulStillTiny`，support-family densification pass = `1`，exact reference deployable = `0`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动，随后进入 CPU-heavy frontier 汇总阶段。
3. P1 oracle-bounded safe-useful feasibility 通过：oracle precision = `1.0`，coverage = `0.067336`，bad-event = `0.0`，null-rate = `0.0`，support-balanced pass = `1`。
4. P2 null-bad conflict autopsy 通过且检测到冲突：`P_bad_given_low_null = 0.573923`，`P_null_given_low_bad = 0.748132`，major attribution fraction = `1.0`。
5. P3 four-class joint statistic 有预测性：best = `JC4-HorizonPersistentSafeUseful`，safe-useful AUC = `0.836857`，precision = `0.839623`，null-rate = `0.028302`；但 coverage = `0.011684 < 0.03`，bad-event = `0.066038 > 0.05`，bad-event UCB = `0.130077`，utility/confidence pass = `0`。
6. P4 family-stable joint support audit 通过：natural rows = `24192`，balanced diagnostic rows = `6000`，signal strata = `318`，family = `2730`，duplicate rows = `0`；accepted support 也通过。
7. P5 exact-reference deployable frontier 未过：best = `C4-HorizonPersistentSafeUsefulController`，precision = `0.839623`，coverage = `0.011684`，bad-event = `0.066038`，null-rate = `0.028302`，precision LCB = `0.758099`，bad-event UCB = `0.130077`。
8. P6 true-delta compute v10 仍未过：best = `TBD0-V9256CBD0Reference`，true-delta AUC = `0.879072`，agreement = `1.0`，step ratio q90 = `2.863280 > 1.50`。
9. P7-P10 因 `P5_reference_deployable_frontier_failed` / `P5_reference_or_P6_compute_failed` gate-blocked，均以 `not_run` 落盘。
10. 当前 blocker：`exact_reference_still_tiny_after_joint_model`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution.py` | v9.2.65 runner；执行 v9.2.64 boundary 复现、oracle-bounded safe-useful feasibility、null-bad conflict autopsy、four-class joint statistic factory、family-stable joint support audit、exact-reference frontier v9、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution.py
```

正式运行：

```bash
python experiments/run_v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution.py \
  --out-dir results/real_rerun_20260506/v9265_joint_safe_useful_feasibility_null_bad_conflict_resolution_first_20260512T203000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T20:05:55Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R14-JointSafeUsefulStillTiny",
  "base_candidate": "LQ-t2-h256",
  "v9264_boundary_pass": 1,
  "oracle_safe_useful_feasible": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.06733630952380952,
  "oracle_bad_event": 0.0,
  "oracle_null_rate": 0.0,
  "oracle_support_balanced_pass": 1,
  "null_bad_conflict_detected": 1,
  "null_bad_conflict_attributed": 1,
  "P_bad_given_low_null": 0.5739225979733854,
  "P_null_given_low_bad": 0.7481317141522653,
  "best_joint_stat_id": "JC4-HorizonPersistentSafeUseful",
  "joint_stat_pass": 1,
  "joint_utility_pass": 0,
  "joint_confidence_pass": 0,
  "joint_auc_su": 0.8368571546494724,
  "joint_precision": 0.839622641509434,
  "joint_coverage": 0.011684303350970017,
  "joint_bad_event": 0.0660377358490566,
  "joint_null_rate": 0.02830188679245283,
  "precision_lcb": 0.7580987112154637,
  "bad_event_ucb": 0.13007652787510787,
  "support_joint_pass": 1,
  "natural_real_event_count": 24192,
  "balanced_diagnostic_real_event_count": 6000,
  "measured_signal_strata_count": 318,
  "measured_family_count": 2730,
  "duplicate_row_count": 0,
  "exact_reference_deployable": 0,
  "best_reference_controller_id": "C4-HorizonPersistentSafeUsefulController",
  "reference_precision": 0.839622641509434,
  "reference_coverage": 0.011684303350970017,
  "reference_bad_event": 0.0660377358490566,
  "reference_null_rate": 0.02830188679245283,
  "reference_precision_lcb": 0.7580987112154637,
  "reference_bad_event_ucb": 0.13007652787510787,
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8790720756550714,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "primary_blocker": "exact_reference_still_tiny_after_joint_model",
  "next_required_implementation": "repair_joint_safe_useful_decision_geometry"
}
```

判断：v9.2.65 证明 oracle-bounded safe-useful frontier 在本轮真实 rows 上存在，但当前可提交的 joint exact-reference frontier 仍太小且 bad-event confidence 不过，因此不能打开 system-legal controller。

## 3. P1 oracle-bounded safe-useful feasibility

Artifacts：

```text
p1_oracle_bounded_safe_useful_feasibility.csv
oracle_feasibility_trace_v9265.csv
```

Summary：

```text
oracle_safe_useful_feasible = 1
oracle_precision = 1.0
oracle_coverage = 0.06733630952380952
oracle_bad_event = 0.0
oracle_null_rate = 0.0
oracle_support_balanced_pass = 1
accepted_signal_strata_count = 37
accepted_family_count = 213
max_family_share = 0.0847145488029466
max_stratum_share = 0.18170656844689995
```

判断：P1 是本轮关键 positive upper bound。oracle-bounded region 满足 coverage / bad-event / null / support balance，说明问题不是数据中完全没有可部署上界，而是当前非-oracle joint decision geometry 还没有恢复这个 region。

## 4. P2 null-bad conflict autopsy

Artifacts：

```text
p2_null_bad_conflict_autopsy.csv
null_bad_conflict_trace_v9265.csv
```

Summary：

```text
conflict_matrix_complete = 1
null_bad_conflict_detected = 1
null_bad_conflict_attributed = 1
major_failure_attribution_fraction = 1.0
global_bad_rate = 0.40219907407407407
global_null_rate = 0.486317791005291
P_bad_given_low_null = 0.5739225979733854
P_null_given_low_bad = 0.7481317141522653
P_bad_null_given_low_null = 0.20913197411793433
P_risky_useful_given_low_null = 0.3647906238554511
P_safe_useful_given_low_null_low_bad = 0.5539947322212467
```

判断：null 与 bad 不是简单可分离轴。低 null 区仍有高 bad，低 bad 区又积累 high-null rows；因此 sequential null/bad gate 容易把 frontier 推成 clean-too-tiny 或 high-null low-precision。

## 5. P3 four-class joint statistic factory

Artifacts：

```text
p3_four_class_joint_statistic_factory.csv
four_class_joint_trace_v9265.csv
```

| joint stat | AUC SU | precision | coverage | bad-event | null-rate | precision LCB | bad UCB | pass | utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JC1-FourClassLCB-UCB | `0.964426` | `0.347458` | `0.117063` | `0.026365` | `0.580979` | `0.319415` | `0.037842` | `1` | `0` |
| JC2-JointMarginScore | `0.948604` | `0.541085` | `0.071098` | `0.043411` | `0.347287` | `0.502498` | `0.062026` | `1` | `0` |
| JC3-RiskWeightedUsefulDensity | `0.960788` | `0.343128` | `0.116292` | `0.022749` | `0.614218` | `0.315095` | `0.033627` | `1` | `0` |
| JC4-HorizonPersistentSafeUseful | `0.836857` | `0.839623` | `0.011684` | `0.066038` | `0.028302` | `0.758099` | `0.130077` | `1` | `0` |
| JC5-HybridJointSafeUseful | `0.976531` | `0.385021` | `0.104497` | `0.029536` | `0.539030` | `0.354568` | `0.042357` | `1` | `0` |

Best summary：

```text
best_joint_stat_id = JC4-HorizonPersistentSafeUseful
joint_stat_pass = 1
joint_utility_pass = 0
joint_confidence_pass = 0
joint_auc_su = 0.8368571546494724
joint_precision = 0.839622641509434
joint_coverage = 0.011684303350970017
joint_bad_event = 0.0660377358490566
joint_null_rate = 0.02830188679245283
```

判断：joint model 确实改善了 precision/null-rate，但它以低 coverage 和 bad-event UCB 失败。更宽的 JC1/JC2/JC3/JC5 有 coverage，却被 null-rate 或 precision 拉垮。

## 6. P4 family-stable joint support audit

Artifacts：

```text
p4_family_stable_joint_support_audit.csv
joint_support_trace_v9265.csv
```

Summary：

```text
natural_real_event_count = 24192
balanced_diagnostic_real_event_count = 6000
measured_signal_strata_count = 318
measured_family_count = 2730
duplicate_row_count = 0
support_joint_pass = 1
accepted_support_pass = 1
accepted_signal_strata_count = 18
accepted_family_count = 50
max_family_share = 0.07395498392282958
max_stratum_share = 0.18971061093247588
```

判断：v9.2.63 的 support regression 没有复现；v9.2.65 的 support density / family / strata / balance gate 都过了。当前 terminal blocker 不在 support 生成器。

## 7. P5 exact reference deployable frontier v9

Artifacts：

```text
p5_exact_reference_deployable_frontier_v9.csv
reference_frontier_v9_trace.csv
```

| controller | joint stat | precision | coverage | bad-event | null-rate | precision LCB | bad UCB | deployable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0-V9264BestReference | NASU2-SupportBackedValue | `0.773292` | `0.035494` | `0.055901` | `0.133540` | `0.724493` | `0.086624` | `0` |
| C2-JointMarginController | JC2-JointMarginScore | `0.541085` | `0.071098` | `0.043411` | `0.347287` | `0.502498` | `0.062026` | `0` |
| C3-ConflictPenalizedController | JC5-HybridJointSafeUseful | `0.385021` | `0.104497` | `0.029536` | `0.539030` | `0.354568` | `0.042357` | `0` |
| C4-HorizonPersistentSafeUsefulController | JC4-HorizonPersistentSafeUseful | `0.839623` | `0.011684` | `0.066038` | `0.028302` | `0.758099` | `0.130077` | `0` |
| C5-ConstrainedJointOptimizer | JC4-HorizonPersistentSafeUseful | `0.839623` | `0.011684` | `0.066038` | `0.028302` | `0.758099` | `0.130077` | `0` |

Best summary：

```text
best_reference_controller_id = C4-HorizonPersistentSafeUsefulController
exact_reference_deployable = 0
precision = 0.839622641509434
coverage = 0.011684303350970017
bad_event = 0.0660377358490566
null_rate = 0.02830188679245283
precision_lcb = 0.7580987112154637
bad_event_ucb = 0.13007652787510787
accepted_signal_strata_count = 14
accepted_family_count = 37
max_family_share = 0.09433962264150944
max_stratum_share = 0.2169811320754717
legal_oracle_jaccard = 0.15344827586206897
```

判断：P5 是本轮 terminal blocker。best reference controller 的 precision / support balance / null-rate 已可看，但 coverage 低于 `0.03`，bad-event point estimate 与 UCB 均不过。不能把 joint statistic 的高 precision 写成 deployable frontier。

## 8. P6 true-delta compute v10

Artifacts：

```text
p6_true_delta_compute_v10_parallel_lane.csv
true_delta_compute_v10_trace.csv
true_delta_correctness_trace_v9265.csv
true_delta_residual_trace_v9265.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8790720756550714
true_delta_safe_useful_auc = 0.7984599264206053
true_delta_joint_auc = 0.7984599264206053
true_delta_conditional_bad_auc = 0.526522839906146
true_delta_conditional_null_auc = 0.5002098952086693
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
```

判断：true-delta exact reference signal 仍存在，但 system envelope 没有恢复；`PX1/PX2` 这类 formula/source-gap proxy 没有被写成 official path。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P5_reference_or_P6_compute_failed` |
| `p8_leave_dataset_and_stratum_out.csv` | `P5_reference_deployable_frontier_failed` |
| `p9_official_paired_replay.csv` | `P5_reference_deployable_frontier_failed` |
| `p10_short_run_functional_validation.csv` | `P5_reference_deployable_frontier_failed` |

没有把 oracle feasibility、joint statistic predictivity、support pass、exact-reference non-deployable row 或 true-delta reference signal 写成 system-legal controller / LDO / paired replay / short-run success。

## 10. No-fake audit

```text
rows_checked = 145237
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
oracle_bounded_feasibility = 1
null_bad_conflict_resolution = 1
four_class_joint_model = 1
family_stable_joint_support = 1
exact_reference_deployable_frontier_v9 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `241bfa446424e891df61f0bf90fce72a3b40dac9a72dec310e75cd1d7e9caa0b` |
| runner | `38565726c86a4fcf527dac924a9ec2c43e68547e2cc348ce55b3234c8bbef658` |
| run manifest | `84bc2d9fa85892d98fec7959c8c173bc4ee48450c70717838807da952117a37b` |
| route | `4c29dbc2f9dde9f6b756012f8697a109b949084d7671244f0737bdb653e37056` |
| P0 boundary | `4ed35acf731c5cf08d5140c453719b5aaa81502c9d9c33801e98ea3ed14e9838` |
| P1 oracle feasibility | `ee26fa230dca88331007fbdd7a1c844ff1594f0b73ea091f64cdfede4ec450f4` |
| P2 null-bad conflict | `55034c8755a18c4c68b0aeabd99849a1ccfddb4b00245616b60e64787ac4c0cc` |
| P3 joint statistic | `822097d02ce0d68394ec0982e95246e02b4ce0cc7162a2f0b9c63b38d375d9b4` |
| P4 joint support | `5e1aa831ef5edb63de8551a30c8af0612481f6b9cfe78993027d2e483a2f2c04` |
| P5 deployable frontier | `6b4d4a7480bebf03a4fb7933313a5446a02975111432c8f13a4da6a61f955608` |
| P6 compute lane | `56b3c2245e40a9f7714e50494ff139307f2d143e7354cd1b36bba8a4c9be0d1f` |
| P7 system controller | `b3fbd5812335166dd17e798012fd3636094823250212d1ba81d9f59482aaf457` |
| P8 boundary | `7355b8a5d018f3e72aa74bee422ef5ab0bebff3bedc74bb85d60fed775985806` |
| P9 boundary | `ff68aa604b9f21f3a0867d744a0d0f15ec619098c33b0a4b9d8c94fd46737682` |
| P10 boundary | `fa51df76d30203df3a88bcb76431ad22929ca9c726f804d1cd5de1ca077eeac8` |
| failure table | `a16f4fbbbfdac4890115f8262e939e2250b455be379e2215ed82a9e361e9b918` |
| provenance audit | `4a96e83d5a796768d1697028ee1b12ffb84a84f217930a950d02d954e7f0533a` |

## 12. 最终分析结论

v9.2.65 的真实推进是：

```text
v9.2.64: conditional null separation 与 support family densification 有推进，
          但 safe-useful frontier 仍 tiny。
v9.2.65: oracle-bounded safe-useful frontier 已证明可行；
          null-bad conflict 已归因；
          support family/strata gate 继续稳定；
          但 joint exact-reference frontier 仍未达到 deployable coverage/safety。
```

机制判断：

1. H1 成立：oracle-bounded safe-useful region 存在，coverage `0.067336`、bad-event `0.0`、null-rate `0.0`。
2. H2 成立：null-bad conflict 确实存在，low-null 区 bad-event 仍高，low-bad 区 null 仍高。
3. H3 只部分成立：four-class joint statistics 有强 signal，但最佳可用 region coverage 只有 `0.011684`，bad-event UCB `0.130077`。
4. H4 成立：support regression 已修复，family count `2730`、strata `318`、balanced rows `6000`，accepted support 也过。
5. H5 未闭合：exact-reference deployable frontier 仍 failed，不能打开 system-legal controller。
6. H6 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.65 真实执行后停在 `R14-JointSafeUsefulStillTiny`：oracle 上界显示 safe-useful frontier 存在，support 也稳定，但当前 joint decision geometry 只能找到 coverage 太小且 bad-event confidence 不过的 reference slice，strict PureKAN functional 仍未成功。
