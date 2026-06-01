# DG-KAN v9.2.68 True-Delta System Closure 与 Reference-Feasible Controller Promotion 实验复盘

> 本复盘记录 `DG-KAN_v9.2.68_TrueDeltaSystemClosure_ReferenceFeasibleControllerPromotion_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 projection 或 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R17-ReferenceFeasibleButComputeFail
base_candidate = LQ-t2-h256
success_v9268_strict_purekan_functional = False
success_v9268_full_functional = False
success_v9268_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9268_true_delta_system_closure_reference_feasible_controller_promotion_first_20260512T233000Z/
```

核心结论：

1. P0 复现 v9.2.67 boundary：source route = `R17-ReferenceFeasibleButComputeFail`，`C3-T2PlusBackfill` exact reference deployable = `1`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动：`NVIDIA L4, 8 %, 324 MiB`、`33 %, 326 MiB`、`32 %, 326 MiB`、`31 %, 326 MiB`，后段进入 CPU-heavy frontier 汇总时出现 `0 %, 326 MiB`。
3. P1 true-delta residual attribution pass = `1`；dominant subphase = `S1-branch_logits_forward`，ratio = `0.396711`；branch-delta logits/replay combined ratio = `0.440701`，unknown fraction = `0.043181`。
4. P2 frozen controller 过 gate：best frozen = `FR1-FrozenC3Rule`，heldout agreement = `1.0`，precision = `0.838028`，coverage = `0.031305`，bad-event = `0.024648`，null-rate = `0.130282`。
5. P3 cheap prefilter 过 gate：best = `PF5-LearnedMonotoneCheapPrefilter`，candidate rate = `0.103051`，reference accept recall = `1.0`。
6. P4 event-sparse exact 的最佳 row 是 `EC1-EventSparseTBD0CostProjection`：agreement = `1.0`，projected step ratio = `1.227012`，但 `materialized_system_path = 0`，因此 official `event_sparse_exact_pass = 0`。
7. P5 fused/compact bridge 的最佳 row 是 `BS2-FrozenPrefilterExactProjection`：agreement = `1.0`，projected step ratio = `1.227012`，但同样 `materialized_system_path = 0`，因此 `fused_bridge_compute_pass = 0`。
8. P6 system-legal exact-signal controller 未运行成 official：reason = `P4_or_P5_true_delta_system_compute_failed`，`official_eligible = 0`，`system_legal_controller_pass = 0`。
9. P7-P10 因 compute/system gate 未过全部 `not_run`；没有把 cost projection、frozen reference success 或 prefilter success 倒灌成 LDO/LSO、paired replay、short-run 或 full-run success。
10. 当前 blocker：`true_delta_system_path_not_materialized`；下一步需要 materialize event-sparse 或 fused true-delta kernel，而不是继续调 reference frontier。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9268_true_delta_system_closure_reference_feasible_controller_promotion.py` | v9.2.68 runner；执行 v9.2.67 boundary 复现、true-delta residual attribution、C3 frozen equivalence、cheap prefilter/cascade、event-sparse exact confirmation、fused compact bridge compute、system/downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9268_true_delta_system_closure_reference_feasible_controller_promotion.py
```

正式运行：

```bash
python experiments/run_v9268_true_delta_system_closure_reference_feasible_controller_promotion.py \
  --out-dir results/real_rerun_20260506/v9268_true_delta_system_closure_reference_feasible_controller_promotion_first_20260512T233000Z \
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
completed_at = 2026-05-12T21:54:14Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R17-ReferenceFeasibleButComputeFail",
  "base_candidate": "LQ-t2-h256",
  "v9267_boundary_pass": 1,
  "reference_controller_id": "C3-T2PlusBackfill",
  "reference_controller_reproduced": 1,
  "reference_precision": 0.8380281690140845,
  "reference_coverage": 0.03130511463844797,
  "reference_bad_event": 0.02464788732394366,
  "reference_null_rate": 0.13028169014084506,
  "reference_precision_lcb": 0.7907157243773478,
  "reference_bad_event_ucb": 0.04999458813129621,
  "frozen_controller_id": "FR1-FrozenC3Rule",
  "frozen_controller_pass": 1,
  "frozen_reference_agreement": 1.0,
  "true_delta_residual_attribution_pass": 1,
  "dominant_subphase": "S1-branch_logits_forward",
  "dominant_subphase_ratio": 0.39671104685864,
  "best_prefilter_id": "PF5-LearnedMonotoneCheapPrefilter",
  "prefilter_pass": 1,
  "candidate_rate": 0.10305059523809523,
  "reference_accept_recall": 1.0,
  "best_exact_candidate_id": "EC1-EventSparseTBD0CostProjection",
  "event_sparse_exact_pass": 0,
  "event_sparse_exact_diagnostic_pass": 1,
  "exact_agreement": 1.0,
  "exact_step_ratio_q90": 1.227012101258447,
  "best_fused_bridge_id": "BS2-FrozenPrefilterExactProjection",
  "fused_bridge_compute_pass": 0,
  "fused_bridge_diagnostic_pass": 1,
  "system_legal_controller_pass": 0,
  "primary_blocker": "true_delta_system_path_not_materialized"
}
```

判断：v9.2.68 没有推翻 v9.2.67 的 reference breakthrough。C3 可以 frozen，cheap prefilter 也能把 candidate rate 降到 `0.103051`。但当前低 step ratio 是基于 measured full-exact trace 的 event-sparse projection，不是已 materialize 的 true-delta system path，所以不能 official promotion。

## 3. P1 true-delta residual attribution

Artifacts：

```text
p1_true_delta_residual_attribution.csv
true_delta_residual_trace_v9268.csv
```

Summary：

```text
dominant_subphase = S1-branch_logits_forward
dominant_subphase_ratio = 0.39671104685864
branch_delta_logits_or_replay_component_ratio = 0.44070054354985677
unknown_fraction = 0.043181040300834826
measured_subphase_sum_ms = 3.2456931322411244
true_delta_residual_attribution_pass = 1
```

判断：true-delta cost 已归因；本轮 dominant 转到 branch logits forward，unknown fraction 小于 `0.10`。这支持下一步做 event-sparse/fused true-delta materialization。

## 4. P2 frozen reference controller equivalence

Artifacts：

```text
p2_frozen_reference_controller_equivalence.csv
frozen_controller_trace_v9268.csv
```

Best summary：

```text
frozen_controller_id = FR1-FrozenC3Rule
frozen_controller_pass = 1
frozen_reference_agreement_cal = 1.0
frozen_reference_agreement_heldout = 1.0
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
feature_compute_time_ms = 1.9457913003861904
search_compute_time_ms = 6188.043436966836
time_reduction = 0.999685556295749
```

判断：H3 成立。`C3-T2PlusBackfill` 可以被 frozen，不依赖在线 frontier search 才成立。

## 5. P3 cheap prefilter / candidate cascade

Artifacts：

```text
p3_cheap_prefilter_candidate_cascade.csv
prefilter_cascade_trace_v9268.csv
```

Best summary：

```text
best_prefilter_id = PF5-LearnedMonotoneCheapPrefilter
prefilter_pass = 1
prefilter_diagnostic_pass = 1
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
reference_accept_precision = 0.381468110709988
candidate_bad_event = 0.2601726263871763
candidate_null_rate = 0.1060419235511714
```

Candidate-rate cost simulator：

```text
r = 0.05 -> projected step ratio 1.128164
r = 0.10 -> projected step ratio 1.221328
r = 0.25 -> projected step ratio 1.500820
r = 1.00 -> projected step ratio 2.898280
```

判断：H4 的 cascade side 成立：candidate rate 低于 `0.25` 且 reference accept recall = `1.0`。但这只说明 sparsification 有空间，不等于 true-delta system path 已实现。

## 6. P4/P5 event-sparse 与 fused compact bridge

P4 artifacts：

```text
p4_event_sparse_exact_confirmation.csv
event_sparse_exact_trace_v9268.csv
```

P4 best：

```text
best_exact_candidate_id = EC1-EventSparseTBD0CostProjection
candidate_rate = 0.10305059523809523
exact_agreement = 1.0
exact_step_ratio_q90 = 1.227012101258447
exact_memory_ratio = 0.9695007261731864
materialized_system_path = 0
event_sparse_exact_diagnostic_pass = 1
event_sparse_exact_pass = 0
status_note = projected_from_measured_full_exact_trace_not_official
```

P5 artifacts：

```text
p5_fused_compact_bridge_compute.csv
fused_bridge_compute_trace_v9268.csv
```

P5 best：

```text
best_fused_bridge_id = BS2-FrozenPrefilterExactProjection
fused_bridge_agreement = 1.0
fused_bridge_step_ratio_q90 = 1.227012101258447
fused_bridge_memory_ratio = 0.9695007261731864
materialized_system_path = 0
fused_bridge_diagnostic_pass = 1
fused_bridge_compute_pass = 0
status_note = projection_from_measured_exact_candidate_rate_not_official
```

Negative / failed materialized rows：

```text
EC3-SelectedLogitExactBridge:
  step_ratio_q90 = 1.48
  agreement = 0.393519
  AUC_safe_good = 0.590470
  pass = 0

EC5-FusedBranchDeltaBridgeKernel:
  status = not_implemented_lower_level_kernel_required
  materialized_system_path = 0
  pass = 0
```

判断：P4/P5 的关键 positive 是成本模型显示 event-sparse path 值得 materialize；关键 negative 是本轮没有真正 materialized event-sparse/fused true-delta kernel。因此不能把 `1.227012` projection 写成 official system pass。

## 7. P6 与 downstream boundary

P6 artifact：

```text
p6_system_legal_exact_signal_controller_v1.csv
system_controller_trace_v9268.csv
```

Summary：

```text
status = not_run
reason = P4_or_P5_true_delta_system_compute_failed
controller_id = C3-T2PlusBackfill
exact_candidate_id = EC1-EventSparseTBD0CostProjection
bridge_system_candidate_id = BS2-FrozenPrefilterExactProjection
official_eligible = 0
system_legal_controller_pass = 0
```

P7-P10 均落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P4_or_P5_true_delta_system_compute_failed` |
| `p8_official_paired_replay_scout.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 frozen C3、prefilter pass、projection diagnostic 或 exact-reference frontier 写成 system-legal controller / leave-out / paired replay / short-run / full-run success。

## 8. No-fake audit

```text
rows_checked = 92
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
reference_controller_freeze = 1
true_delta_residual_attribution = 1
candidate_cascade = 1
event_sparse_exact_confirmation = 1
fused_compact_bridge_compute = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `74bad23cdc5d2932987f47750496df7f2502ccb1976721d34d03a018e39bdbae` |
| runner | `0d23f84a29c4a020d6736adb7b6635b0ef1250d26cf6fc5df1c0cb06ec8779ae` |
| run manifest | `1b3f073cd3fa26cbcbb2f94fb9c67a535a0133aa0fbb096bc056f981601143cb` |
| route | `5e57f53cd0ad54aaa3a0c307a5e33d486616b8171bd416bb9f948c6bfb14fb54` |
| P0 boundary | `a1e7e95c36fb67651dbeb4e2f52a159fbe47af9afe9a7a31ae59782f65feeae3` |
| P1 residual attribution | `a2ed9e192dfa9f29ff65b5ee5724fee25d22a8f0a8b3355036eafce47c8de19f` |
| P2 frozen equivalence | `a744b649db4d675d0cf1ddd28c684a301cda963e0b68a68676761c57cb1c8174` |
| P3 prefilter cascade | `78300b433860f23e9f38883b73b390e8ba46b11b611d2e0ea0fcc0a065e31563` |
| P4 event sparse exact | `24b97af97add58929579e4c224744c9d94bf1182b92de07e5c1282c9aa8e7754` |
| P5 fused compact bridge | `0d97a0ec95660c2c3cdce492c151e7b4abebd42738a762d8c751756b2199af5c` |
| P6 system controller | `ffa1e0bb463ed645645ba7dcde6854c14c5191222cbf5f9810e234fa19c1dcd4` |
| P7 boundary | `a606334f138ae3b62fa74a12629e0bbd5a2c5786021ac89d32fcf9d790e321ed` |
| P8 boundary | `704bedcb955ed7d8d46d9b57947143d7f044a81c920cb32f05d881958452e172` |
| P9 boundary | `e966d1c75cb0ced21c207f7a6f978ac318961934c26b8a8e29473254b697044b` |
| P10 boundary | `73bdca598bc8d5058337f6558e84df938135858effcc0746453ca0bb9419e789` |
| failure table | `a5f2dbed25b0bda5518d322f01cbf1f59e1621511b1886306f34850651c6801b` |
| provenance audit | `9fcc61c877484f668c0ee684955de70ef35512331f605a07ecb1225e20a08ed7` |

## 10. 最终分析结论

v9.2.68 的真实推进是：

```text
v9.2.67: localized bridge decision geometry 首次 exact-reference deployable；
          但 true-delta compute 仍太贵。
v9.2.68: C3-T2PlusBackfill 可以 frozen；
          cheap prefilter 可以把 candidate rate 降到约 10.3% 且保持 reference recall；
          但 event-sparse/fused true-delta 仍停在 projection，不是 materialized system path。
```

机制判断：

1. H1 成立：v9.2.67 reference frontier 稳定复现，`C3-T2PlusBackfill` 仍 deployable。
2. H2 部分成立：true-delta residual attribution 过，dominant 是 `S1-branch_logits_forward`，branch logits/replay combined ratio 为 `0.440701`。
3. H3 成立：frozen rule 与 reference search heldout agreement = `1.0`。
4. H4 成立于 prefilter 侧：`PF5` candidate rate = `0.103051`，reference accept recall = `1.0`。
5. H5 未闭合：projected event-sparse/fused cost 可进入 `<=1.50`，但 `materialized_system_path = 0`，不能 official。
6. 当前下一步不是继续调 C3/T2/C4/E2 frontier，而是把 `PF5 + C3 frozen + TBD0 exact confirm` 真正 materialize 成 event-sparse 或 fused true-delta kernel，并重新测真实 step ratio。

最终一句话：

> v9.2.68 真实执行后仍停在 `R17-ReferenceFeasibleButComputeFail`：reference controller 已可 frozen，candidate cascade 显示成本闭合有希望，但当前低 step ratio 只是 measured-trace projection，真实 event-sparse/fused true-delta system path 尚未 materialize，strict PureKAN functional 仍未成功。
