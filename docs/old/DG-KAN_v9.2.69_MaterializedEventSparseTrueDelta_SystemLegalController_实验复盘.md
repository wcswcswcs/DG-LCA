# DG-KAN v9.2.69 Materialized Event-Sparse True-Delta 与 System-Legal Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.69_MaterializedEventSparseTrueDelta_SystemLegalController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 diagnostic materialized microprobe 写成 official system pass。

## 0. 最新结论

```text
route = R18-ReferenceFeasibleButComputeFail
base_candidate = LQ-t2-h256
success_v9269_strict_purekan_functional = False
success_v9269_full_functional = False
success_v9269_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9269_materialized_event_sparse_true_delta_system_legal_controller_first_20260512T235500Z/
```

核心结论：

1. P0 复现 v9.2.68 boundary：source route = `R17-ReferenceFeasibleButComputeFail`，reference controller = `C3-T2PlusBackfill`，reference reproduced = `1`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动：`NVIDIA L4, 32-33 %, 326 MiB`，后段 CPU-heavy 汇总阶段为 `0 %, 326 MiB`。
3. P1 materialization gap 已映射：除 `full online row binding` 外，PF5 selector、candidate tensors、true branch logits、true branch-delta、frozen C3 lookup、bridge score、timing measurement 均有 materialized path；`official_materialization_gap_mapped = 0`。
4. P2 PF5 runtime selector / candidate pack 过 gate：candidate rate = `0.103051`，reference accept recall = `1.0`，candidate count = `2493 / 24192`，candidate pack time = `6.339111 ms`，projection used = `0`。
5. P3 candidate-only true branch forward 过 diagnostic/materialized gate：candidate event rate = `0.083333`，candidate forward ratio vs full = `0.018018`，logit max abs diff = `0.0`。
6. P4 materialized event-sparse true-delta microprobe diagnostic 过：materialized system path = `1`，projection used = `0`，full trace projection used = `0`，agreement = `1.0`，step ratio q90 = `1.075342`，memory ratio = `1.0`。
7. P4 official pass 仍为 `0`，因为 `full_online_row_binding = 0`；本轮没有把 microprobe materialization 写成全量 online controller system pass。
8. P5 frozen bridge lookup 已 materialized：uses CPU summary = `0`，online frontier search = `0`，bridge lookup time = `0.141594 ms`，diagnostic pass = `1`；official pass 同样因 full online binding 缺失为 `0`。
9. P6 system-legal controller 未打开：decision metrics 仍保持 v9.2.67/68 reference 水平，但 official eligible = `0`。
10. 当前 blocker：`materialized_microprobe_not_bound_to_full_online_controller`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py` | v9.2.69 runner；复现 v9.2.68 boundary，执行 projection-to-materialization audit、PF5 runtime selector/candidate pack、candidate-only true branch forward、materialized true-delta microprobe、frozen bridge lookup、system controller gate 与 no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py
```

正式运行：

```bash
python experiments/run_v9269_materialized_event_sparse_true_delta_system_legal_controller.py \
  --out-dir results/real_rerun_20260506/v9269_materialized_event_sparse_true_delta_system_legal_controller_first_20260512T235500Z \
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
completed_at = 2026-05-12T23:28:10Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R18-ReferenceFeasibleButComputeFail",
  "base_candidate": "LQ-t2-h256",
  "v9268_boundary_pass": 1,
  "reference_controller_id": "C3-T2PlusBackfill",
  "reference_controller_reproduced": 1,
  "reference_precision": 0.8380281690140845,
  "reference_coverage": 0.03130511463844797,
  "reference_bad_event": 0.02464788732394366,
  "reference_null_rate": 0.13028169014084506,
  "reference_precision_lcb": 0.7907157243773478,
  "reference_bad_event_ucb": 0.04999458813129621,
  "materialization_gap_mapped": 1,
  "official_materialization_gap_mapped": 0,
  "best_prefilter_id": "PF5-LearnedMonotoneCheapPrefilter",
  "pf5_runtime_selector_pass": 1,
  "candidate_rate": 0.10305059523809523,
  "reference_accept_recall": 1.0,
  "candidate_pack_materialized": 1,
  "candidate_branch_forward_pass": 1,
  "candidate_forward_time_ratio": 0.018018326773696327,
  "branch_logit_error_max": 0.0,
  "best_exact_candidate_id": "EC2-MaterializedPF5CandidateTrueDeltaMicroprobe",
  "materialized_event_sparse_exact_diagnostic_pass": 1,
  "materialized_event_sparse_exact_pass": 0,
  "materialized_system_path": 1,
  "projection_used": 0,
  "full_trace_projection_used": 0,
  "full_online_row_binding": 0,
  "exact_agreement": 1.0,
  "exact_step_ratio_q90": 1.075342155736442,
  "exact_memory_ratio": 1.0,
  "fused_bridge_materialized_diagnostic_pass": 1,
  "fused_bridge_materialized_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "materialized_microprobe_not_bound_to_full_online_controller",
  "next_required_implementation": "bind_materialized_candidate_tensor_path_to_full_controller_rows"
}
```

判断：v9.2.69 相比 v9.2.68 已经把 projection-only cost 推进为真实 materialized microprobe timing；但它还没有完整绑定到全量 online controller rows，所以 route 保守停在 `R18`。

## 3. P1 materialization gap audit

Artifacts：

```text
p1_projection_to_materialization_gap_audit.csv
materialization_gap_trace_v9269.csv
```

Summary：

```text
materialization_gap_mapped = 1
official_materialization_gap_mapped = 0
unmapped_component = full online row binding
implementation_priority = bind_materialized_candidate_tensor_path_to_full_controller_rows
```

判断：本轮没有留下未知 materialization gap；唯一未闭合的是 official 需要的 full online row binding。这个 blocker 被显式落盘，没有被藏进 projected cost。

## 4. P2 runtime PF5 selector / candidate pack

Artifacts：

```text
p2_runtime_pf5_selector_candidate_pack.csv
pf5_candidate_pack_trace_v9269.csv
```

Summary：

```text
prefilter_id = PF5-LearnedMonotoneCheapPrefilter
candidate_count = 2493
event_count = 24192
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
reference_accept_precision = 0.381468110709988
selector_time_ms = 7.069816
candidate_pack_time_ms = 6.339111
candidate_pack_memory_MB = 0.057060
candidate_indices_materialized = 1
candidate_tensor_contiguous = 1
projection_used = 0
pf5_runtime_selector_pass = 1
```

判断：PF5 不再只是统计行；candidate indices / family ids / branch ids 均落成 contiguous tensor，并保留 tensor hash。

## 5. P3 candidate-only true branch forward

Artifacts：

```text
p3_candidate_only_true_branch_forward.csv
candidate_branch_forward_trace_v9269.csv
```

Summary：

```text
branch_forward_id = BF1-CandidatePackedBranchForwardMaterializedMicroprobe
candidate_probe_event_count = 2
full_probe_event_count = 24
candidate_event_rate = 0.08333333333333333
branch_forward_time_ms_mean = 2.220761
branch_forward_time_ms_q90_candidate_nonzero = 2.187235
branch_forward_ratio_vs_full = 0.018018326773696327
aggregate_step_ratio = 1.2823284253437088
event_q90_step_ratio = 1.0
logit_max_abs_diff_vs_full_reference = 0.0
candidate_branch_forward_pass = 1
candidate_branch_forward_diagnostic_pass = 1
```

判断：candidate-only branch forward 是真实 train-stream recompute，不是从 full trace 折算；candidate logits 与 full reference 对应事件的 max abs diff 为 `0.0`。不过它仍只是 microprobe materialization，还不是全量 row binding。

## 6. P4 materialized event-sparse true-delta

Artifacts：

```text
p4_materialized_event_sparse_true_delta_exact_confirmation.csv
materialized_true_delta_trace_v9269.csv
```

| candidate | materialized | projection | full row binding | agreement | step q90 | precision | coverage | bad-event | official | diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EC1-EventSparseTBD0CostProjectionFromV9268 | `0` | `1` | `0` | `1.0` | `1.227012` | `0.838028` | `0.031305` | `0.024648` | `0` | `1` |
| EC2-MaterializedPF5CandidateTrueDeltaMicroprobe | `1` | `0` | `0` | `1.0` | `1.075342` | `0.838028` | `0.031305` | `0.024648` | `0` | `1` |

Additional materialized timing:

```text
true_delta_score_time_ms = 0.686758
true_delta_score_checksum = -0.09550318121910095
read_MB = 0.020508
write_MB = 0.000488
dominant_subphase = S2-true_delta_score
```

判断：v9.2.69 的关键 positive 是 `EC2`：真实 materialized path 的 step ratio q90 从 v9.2.68 projection 的 `1.227012` 进一步实测为 `1.075342`。但 official 仍为 `0`，因为 `full_online_row_binding = 0`。

## 7. P5 fused / compact bridge materialized compute

Artifacts：

```text
p5_fused_compact_bridge_materialized_compute.csv
fused_bridge_materialized_trace_v9269.csv
```

Summary：

```text
bridge_system_id = BS3-MaterializedPF5FrozenC3LookupMicroprobe
bridge_compute_id = BR1-FrozenC3LookupTensorGather
uses_compact_lookup = 1
uses_cpu_summary = 0
online_frontier_search_used = 0
materialized_system_path = 1
full_online_row_binding = 0
bridge_lookup_time_ms = 0.141594
step_ratio_q90 = 1.075342155736442
memory_ratio = 1.0
fused_bridge_materialized_diagnostic_pass = 1
fused_bridge_materialized_pass = 0
```

判断：frozen C3 lookup 已经是 tensor gather，不是 CPU-heavy frontier summary；但与 P4 一样，缺 full online row binding，因此不能 official。

## 8. P6 system controller boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v2.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
official_eligible = 0
system_legal_controller_pass = 0
reason = P4_or_P5_materialized_compute_not_official
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
agreement_reference_accept = 1.0
step_ratio_q90 = 1.075342155736442
memory_ratio = 1.0
accepted_signal_strata_count = 15
accepted_family_count = 65
max_family_share = 0.09507042253521127
max_stratum_share = 0.15140845070422534
```

判断：decision / support / cost point estimates 都已经看起来满足 system controller 数值门槛；唯一阻断是 P4/P5 official materialization 没过。这里没有把 `official_eligible` 改成 `1`。

## 9. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P4_or_P5_materialized_compute_not_official` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 materialized diagnostic、projection control、frozen lookup tensor gather 或 reference decision metrics 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 10. No-fake audit

```text
rows_checked = 78
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
materialized_pf5_selector = 1
candidate_pack_materialized = 1
candidate_only_true_branch_forward = 1
materialized_true_delta_microprobe = 1
frozen_bridge_lookup_materialized = 1
full_online_row_binding = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `38e2cd50f6fe24299fc641754ecafabca71631e82d409b70932d24251483b3ed` |
| runner | `da64a9100a4d6c106074333faa8ebdb25c9270bf4bb45fa041fb1f9ea672ce7a` |
| run manifest | `5c2fd04353d651607190c69e03cee532e4f8ae6d84514b139f55b7b4a0353557` |
| route | `adeb0ed6e04b1ff072f3f548a4d07426070e4c84cd751fa8a2e6df7de138da4a` |
| P0 boundary | `a65bec8eb9f7304d905b6a5f284360c015320dd02529e0a0990014f63d0d0ba9` |
| P1 gap audit | `b3c9a0f5e55b52b5aa4af7267b62150bd12aa1f78d7d379cb2aac873712b2cf5` |
| P2 PF5 candidate pack | `7b5f1be3a205d9a0331bf1b0a04a1c918ab664a39e310836e1bf9b3101cdded3` |
| P3 branch forward | `f9e720cd24c88a54a9ee3c4648ba0e28cb5d4ab102bfbaf1efdf3d183b210daa` |
| P4 materialized true delta | `06281be20113d7cf6976f07a1ebdf0526a22f0e75a2c7d0aae453ee476a4ba12` |
| P5 fused bridge | `89dad5debfd582353ca0accd28fc847f0639aeec8a571d5350dfe8e5357a2ce3` |
| P6 system controller | `a458c156453c1ebc845db0e3b23af00ef44510dfb649125ccc57dc4981a41a37` |
| failure table | `bd832476553b70811498185f76e681598ff59acb7ad178e5ca4104bcc4bc1121` |
| provenance audit | `c18660fb0be14b51e4cf1fb830e260f01809d732159800f198ff3af82f0bcf68` |

## 12. 最终分析结论

v9.2.69 的真实推进是：

```text
v9.2.68: exact-reference controller 已可部署，但 event-sparse true-delta 只是 projection。
v9.2.69: PF5 selector / candidate pack / candidate-only true branch forward /
          true-delta score / frozen bridge lookup 均已 materialize 到真实 microprobe；
          但还没有绑定到全量 online controller rows，因此不能 official。
```

机制判断：

1. H1 在 diagnostic 层成立：materialized sparse path 的 measured `step_ratio_q90 = 1.075342`，低于 `1.50`，且不是 projection。
2. H2 成立为 microprobe 证据：candidate-only branch forward 的 ratio vs full 只有 `0.018018`，logit diff 为 `0.0`。
3. H3 仍保留：reference accept agreement = `1.0`，decision metrics 与 C3-T2PlusBackfill 一致。
4. H4 成立：PF5 runtime selector materialized，candidate rate `0.103051`，recall `1.0`。
5. H5 部分成立：frozen bridge lookup 已是 compact tensor gather，CPU summary / online frontier search 都是 `0`。
6. H6 未打开：P6 仍 `official_eligible = 0`，因此 LDO/LSO 和 paired replay 全部 gate-blocked。

最终一句话：

> v9.2.69 真实执行后停在 `R18-ReferenceFeasibleButComputeFail`：event-sparse true-delta 已从 projection 推进到真实 materialized microprobe，但 full online row binding 仍缺失，system-legal controller 不能转正，strict PureKAN functional 仍未成功。
