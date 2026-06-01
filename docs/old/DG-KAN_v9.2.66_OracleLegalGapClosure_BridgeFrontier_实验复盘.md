# DG-KAN v9.2.66 Oracle-Legal Gap Closure 与 Bridge Frontier 实验复盘

> 本复盘记录 `DG-KAN_v9.2.66_OracleLegalGapClosure_BridgeFrontier_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R12-C0BadTrimFail
base_candidate = LQ-t2-h256
success_v9266_strict_purekan_functional = False
success_v9266_full_functional = False
success_v9266_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9266_oracle_legal_gap_closure_bridge_frontier_first_20260512T213000Z/
```

核心结论：

1. P0 复现 v9.2.65 boundary：source route = `R14-JointSafeUsefulStillTiny`，oracle feasible = `1`，support joint pass = `1`，exact reference deployable = `0`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`。
3. P1 oracle-legal gap localization pass = `1`：oracle count = `1629`，C0 count = `930`，C4 count = `311`；oracle missed by both = `895`，C0 false positive = `197`，C4 false positive = `64`。
4. P1 归因闭合：oracle miss attribution fraction = `0.960894`，legal false-positive attribution fraction = `1.0`；主要 repairable miss mode = `M1-horizon-persistence-overreject`。
5. P2 C0 surgical trim 未过：best = `T2-ConflictRiskTrim`，precision = `0.831224`，bad-event = `0.008439`，bad-event UCB = `0.030242`；但 coverage = `0.026124 < 0.03`，null-rate = `0.151899 > 0.15`，因此 `C0_trim_pass = 0`。
6. P3 C4 oracle-neighbor expansion 未过：best = `E2-LegalNeighborExpansion`，coverage = `0.097884`，Jaccard = `0.486680`；但 precision = `0.534910`，bad-event = `0.268018`，因此 `C4_expansion_pass = 0`。
7. P4 bridge controller 未过：best route-level row 仍是 `C0-v9265BroadReference`，precision = `0.773292`，coverage = `0.035494`，bad-event = `0.055901`，bad-event UCB = `0.086624`。
8. P5 exact-reference deployable frontier 未过：best = `C0-v9265BroadReference`，precision LCB = `0.724493 < 0.75`，bad-event UCB = `0.086624 > 0.05`。
9. P6 true-delta compute v11 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879072`，agreement = `1.0`，step ratio q90 = `2.863280 > 1.50`。
10. P7-P10 因 `P5_reference_or_P6_compute_failed` / `P2_C0_bad_trim_failed` gate-blocked，全部以 `not_run` 落盘。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9266_oracle_legal_gap_closure_bridge_frontier.py` | v9.2.66 runner；执行 v9.2.65 boundary 复现、oracle-legal gap localization、C0 surgical bad-event trim、C4 oracle-neighbor recovery、bridge controller、exact-reference frontier v10、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9266_oracle_legal_gap_closure_bridge_frontier.py
```

正式运行：

```bash
python experiments/run_v9266_oracle_legal_gap_closure_bridge_frontier.py \
  --out-dir results/real_rerun_20260506/v9266_oracle_legal_gap_closure_bridge_frontier_first_20260512T213000Z \
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
completed_at = 2026-05-12T20:34:05Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R12-C0BadTrimFail",
  "base_candidate": "LQ-t2-h256",
  "v9265_boundary_pass": 1,
  "oracle_safe_useful_feasible": 1,
  "oracle_precision": 1.0,
  "oracle_coverage": 0.06733630952380952,
  "oracle_bad_event": 0.0,
  "oracle_null_rate": 0.0,
  "oracle_legal_gap_localized": 1,
  "oracle_miss_attribution_fraction": 0.9608938547486033,
  "legal_false_positive_attribution_fraction": 1.0,
  "C0_trim_pass": 0,
  "C0_trim_precision": 0.8312236286919831,
  "C0_trim_coverage": 0.026124338624338623,
  "C0_trim_bad_event": 0.008438818565400843,
  "C0_trim_null_rate": 0.1518987341772152,
  "C0_trim_precision_lcb": 0.7783414656382615,
  "C0_trim_bad_event_ucb": 0.030241920509300354,
  "C4_expansion_pass": 0,
  "C4_expansion_precision": 0.5349099099099099,
  "C4_expansion_coverage": 0.09788359788359788,
  "C4_expansion_bad_event": 0.268018018018018,
  "C4_expansion_null_rate": 0.11936936936936937,
  "best_bridge_controller_id": "C0-v9265BroadReference",
  "bridge_controller_pass": 0,
  "bridge_precision": 0.7732919254658385,
  "bridge_coverage": 0.035493827160493825,
  "bridge_bad_event": 0.055900621118012424,
  "bridge_null_rate": 0.13354037267080746,
  "bridge_precision_lcb": 0.7244928463797143,
  "bridge_bad_event_ucb": 0.08662425903435575,
  "exact_reference_deployable": 0,
  "true_delta_compute_pass": 0,
  "true_delta_auc": 0.8790720756550714,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "primary_blocker": "C0_bad_trim_fail"
}
```

判断：oracle-bounded safe-useful region 仍存在，但 legal controller 还不能把 C0/C4 之间的安全与 coverage 同时接上。C0 trim 已把 bad-event 压住，但 coverage/null gate 没过；C4 expansion 能扩 coverage，但 precision/bad-event 失败。

## 3. P1 oracle-legal gap localization

Artifacts：

```text
p1_oracle_legal_gap_localization.csv
oracle_legal_gap_trace_v9266.csv
```

Summary：

```text
A_oracle_count = 1629
A_C0_count = 930
A_C4_count = 311
A_oracle_intersect_C0_count = 733
A_oracle_intersect_C4_count = 247
A_oracle_missed_by_both_count = 895
A_C0_false_positive_count = 197
A_C4_false_positive_count = 64
legal_oracle_jaccard_C0 = 0.4014238773274918
legal_oracle_jaccard_C4 = 0.14589486119314826
repairable_oracle_miss_mode = M1-horizon-persistence-overreject
repairable_oracle_miss_coverage = 0.01707175925925926
oracle_miss_attribution_fraction = 0.9608938547486033
legal_false_positive_attribution_fraction = 1.0
oracle_legal_gap_localized = 1
```

判断：P1 说明 v9.2.65 的失败不是没有 oracle-safe region，而是 C0 broad reference 和 C4 precise reference 分别落在不同错误：C0 有 false positives，C4 丢 coverage。

## 4. P2 C0 surgical bad-event trim

Artifacts：

```text
p2_C0_surgical_bad_event_trim.csv
C0_trim_trace_v9266.csv
```

| trim | precision | coverage | bad-event | null-rate | precision LCB | bad UCB | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0-C0Reference | `0.773292` | `0.035494` | `0.055901` | `0.133540` | `0.724493` | `0.086624` | `0` |
| T2-ConflictRiskTrim | `0.831224` | `0.026124` | `0.008439` | `0.151899` | `0.778341` | `0.030242` | `0` |

Best summary：

```text
best_trim_id = T2-ConflictRiskTrim
C0_trim_confidence_pass = 1
C0_trim_pass = 0
rows_removed = 177
accepted_signal_strata_count = 14
accepted_family_count = 56
max_family_share = 0.11392405063291139
max_stratum_share = 0.14767932489451477
```

判断：`T2` 是真实推进：bad-event 和 confidence 都过了。但它把 coverage 从 `0.035494` 压到 `0.026124`，且 null-rate 刚好超过 `0.15`，所以不能转正。

## 5. P3 C4 oracle-neighbor recovery

Artifacts：

```text
p3_C4_oracle_neighbor_coverage_recovery.csv
C4_expansion_trace_v9266.csv
```

| expansion | precision | coverage | bad-event | null-rate | Jaccard | pass |
|---|---:|---:|---:|---:|---:|---:|
| C4 reference | `0.839623` | `0.011684` | `0.066038` | `0.028302` | `0.153448` | `0` |
| E2-LegalNeighborExpansion | `0.534910` | `0.097884` | `0.268018` | `0.119369` | `0.486680` | `0` |

Best summary：

```text
best_expansion_id = E2-LegalNeighborExpansion
rows_added = 2388
C4_expansion_pass = 0
C4_expansion_overlap_diagnostic_pass = 1
C4_expansion_precision_lcb = 0.5020235325263702
C4_expansion_bad_event_ucb = 0.2981044054561512
```

判断：`E2` 证明确实能 recover oracle-neighbor coverage，但带入大量 bad-event，precision 也明显不足。它只能作为 diagnostic expansion，不能作为 deployable recovery。

## 6. P4/P5 bridge controller 与 exact-reference frontier

Artifacts：

```text
p4_bridge_controller.csv
bridge_frontier_trace_v9266.csv
p5_exact_reference_deployable_frontier_v10.csv
reference_frontier_v10_trace.csv
```

| controller | precision | coverage | bad-event | null-rate | precision LCB | bad UCB | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-v9265BroadReference | `0.773292` | `0.035494` | `0.055901` | `0.133540` | `0.724493` | `0.086624` | `0` |
| C2-C0-MinimalBadTrim | `0.831224` | `0.026124` | `0.008439` | `0.151899` | `0.778341` | `0.030242` | `0` |
| C3-C4-LegalNeighborExpansion | `0.534910` | `0.097884` | `0.268018` | `0.119369` | `0.502024` | `0.298104` | `0` |
| C5-BridgeFrontierController | `0.208633` | `0.291116` | `0.209012` | `0.491480` | `0.193565` | `0.224936` | `0` |

Best summary：

```text
best_bridge_controller_id = C0-v9265BroadReference
bridge_controller_pass = 0
exact_reference_deployable = 0
accepted_signal_strata_count = 22
accepted_family_count = 92
max_family_share = 0.08385093167701864
max_stratum_share = 0.14596273291925466
legal_oracle_jaccard = 0.3915094339622642
oracle_gap_remaining = 0.0265652557319224
```

判断：bridge frontier 没有找到同时满足 coverage、precision LCB、bad-event UCB、null-rate 的 row。`C0` coverage 够但 safety confidence 不够；`C2` safety 够但 coverage/null 不够；`C3` coverage 够但 safety 崩掉。

## 7. P6 true-delta compute v11

Artifacts：

```text
p6_true_delta_compute_v11_parallel_lane.csv
true_delta_compute_v11_trace.csv
true_delta_correctness_trace_v9266.csv
true_delta_residual_trace_v9266.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.8790720756550714
true_delta_safe_useful_auc = 0.7984599264206053
true_delta_joint_auc = 0.7984599264206053
true_delta_bridge_auc = 0.8179433345998781
true_delta_conditional_bad_auc = 0.526522839906146
true_delta_conditional_null_auc = 0.5002098952086693
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.8632798851361203
true_delta_memory_ratio = 0.9695007261731864
true_delta_compute_pass = 0
```

判断：true-delta exact signal 仍存在，但 system envelope 没有恢复。低成本 formula/source-gap proxy rows 没有被转正。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P5_reference_or_P6_compute_failed` |
| `p8_leave_dataset_and_stratum_out.csv` | `P2_C0_bad_trim_failed` |
| `p9_official_paired_replay.csv` | `P2_C0_bad_trim_failed` |
| `p10_short_run_functional_validation.csv` | `P2_C0_bad_trim_failed` |

没有把 oracle feasibility、gap localization、C0 trimmed row、C4 expansion diagnostic、bridge non-deployable row 或 true-delta reference signal 写成 system-legal controller / LDO / paired replay / short-run success。

## 9. No-fake audit

```text
rows_checked = 48841
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
oracle_legal_gap_localization = 1
C0_surgical_bad_event_trim = 1
C4_oracle_neighbor_recovery = 1
bridge_controller = 1
exact_reference_deployable_frontier_v10 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `72211e69126d7dc91bb801a8e24333c1f5c76039f08897465772fd3a36823bd2` |
| runner | `51607c329ad4703905a8cb192da64c4cb848d5133974fc52a9b0cebe1e6a2383` |
| run manifest | `4d2fcd822730f4c3690397439a2b96b8e676a294c466913b1f8bdc3c5e14857a` |
| route | `15ac31fcc8c6a4dae5223f8aeb2177e0587cefda164a5a1566ee78bafae29234` |
| P0 boundary | `af28876c9d51f03ac14be900d98459c4f94ebe771bc9799e12486dbe112c655a` |
| P1 gap localization | `a3e8446eb9b3c2c4ce445d8cd96afbf30c4e1cdffe7de3466c67d01ff2d51419` |
| P2 C0 trim | `797a0e6fae17013949f1db7f764cb0582da36de9c16c04aa8a756fc4d033e3a5` |
| P3 C4 recovery | `f8ebe5d00993fcfa92bd88c47c96440a0e754b454848e69c94f86694d8803f42` |
| P4 bridge | `2c74aa542955b3f36b74a16bea80b2fc3264e9f3f0c6d2facdfe1cbc81351b26` |
| P5 reference frontier | `7df419d569d9d321f78d55414af00a0764190bc796c1d03fa01a8ce4fc5a351a` |
| P6 compute lane | `ade3d88ed5f0eeeef60858d50febc3428538ba26f4b1f72f8ff5f8c224946a0c` |
| P7 system controller | `b3fbd5812335166dd17e798012fd3636094823250212d1ba81d9f59482aaf457` |
| P8 boundary | `d66180902d7200bffeeb2d6ede672aca7b4a55d19f4cdc041beae34266c1142d` |
| P9 boundary | `f3532c9f287d3b157c2f4c33644304cee11046de4c178996ee44a5f0626a84f3` |
| P10 boundary | `d03970917a24eee82f1fffb5f6c0c969dfc5ebce5000e3b4dcd79566937ef084` |
| failure table | `e30ba201e5fb28e09d09e26fe3460540f380a3597f84878cf823cfcc886d7623` |
| provenance audit | `a672bcf75bd4f12f7ed828b502c0ed2719f4a1afb018867e43165d4cc5a7b9df` |

## 11. 最终分析结论

v9.2.66 的真实推进是：

```text
v9.2.65: oracle-bounded safe-useful frontier 存在，
          但 joint exact-reference frontier coverage/safety 仍未闭合。
v9.2.66: oracle-legal gap 被定位；
          C0 surgical trim 能压 bad-event；
          C4 neighbor expansion 能恢复 coverage；
          但二者没有被 bridge 成 deployable controller。
```

机制判断：

1. H1 成立：oracle/legal gap localization 通过，C0/C4 与 oracle 的错位被具体归因。
2. H2 部分成立：C0 trim 能把 bad-event 降到 `0.008439`，bad-event UCB 降到 `0.030242`；但 coverage 低于 `0.03`，null-rate 略高于 `0.15`。
3. H3 部分成立：C4 expansion 能把 coverage 提到 `0.097884`，Jaccard 提到 `0.486680`；但 precision/bad-event 失败。
4. H4 未闭合：bridge controller 没有找到同时满足 deployable coverage、precision LCB、bad-event UCB、null-rate 的 row。
5. H5 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.66 真实执行后停在 `R12-C0BadTrimFail`：oracle/legal gap 已定位，C0 trim 与 C4 expansion 各自解决一半问题，但 bridge frontier 仍无法同时满足 coverage 与 safety confidence，strict PureKAN functional 仍未成功。
