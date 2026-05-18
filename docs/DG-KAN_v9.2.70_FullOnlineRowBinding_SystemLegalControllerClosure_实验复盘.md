# DG-KAN v9.2.70 Full-Online Row Binding 与 System-Legal Controller Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.70_FullOnlineRowBinding_SystemLegalControllerClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 diagnostic microprobe 或 bridge lookup 单独写成 official system pass。

## 0. 最新结论

```text
route = R14-FullOnlineBindingFail
base_candidate = LQ-t2-h256
success_v9270_strict_purekan_functional = False
success_v9270_full_functional = False
success_v9270_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9270_full_online_row_binding_system_legal_controller_closure_first_20260513T010000Z/
```

核心结论：

1. P0 复现 v9.2.69 boundary：source route = `R18-ReferenceFeasibleButComputeFail`，reference controller = `C3-T2PlusBackfill`，reference reproduced = `1`，PF5 selector pass = `1`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；实际参数保持 MNIST/Fashion-MNIST/KMNIST、seeds `0..7`、microprobe steps `336`。
3. P1 full-online event identity binding 已过：event count = `24192`，candidate count = `2493`，event/global/candidate duplicate = `0`，event identity binding pass = `1`。
4. P1 official full-online row binding 未过：candidate tensor payload missing = `2493`，candidate branch logits missing = `2493`，candidate true-delta logits missing = `2493`，functional update payload missing = `951`。
5. P2 PF5 full-online selector 过 gate：candidate rate = `0.10305059523809523`，reference accept recall = `1.0`，candidate metadata bound = `1`；但 candidate payload bound = `0`。
6. P3 只能保留 v9.2.69 diagnostic microprobe：candidate forward ratio = `0.018018326773696327`，logit max abs diff = `0.0`，diagnostic pass = `1`，full-online pass = `0`。
7. P4 decision/cost diagnostic 仍满足：agreement = `1.0`，step ratio q90 = `1.075342155736442`，memory ratio = `1.0`，reference precision/coverage/bad-event 仍过；但 true-delta logits 未绑定全量 online rows，因此 full-online true-delta pass = `0`。
8. P5 frozen C3 bridge lookup 已 full-online materialized：uses CPU summary = `0`，online frontier search = `0`，bridge lookup time = `0.152631 ms`，fused bridge full-online pass = `1`。
9. P6 system-legal controller 未打开：official eligible = `0`，reason = `P1_or_P4_full_online_binding_failed`；没有把 bridge lookup pass 或 diagnostic true-delta microprobe 倒灌成 official controller。
10. 当前 blocker：`candidate_tensor_payload_missing_for_full_online_rows`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9270_full_online_row_binding_system_legal_controller_closure.py` | v9.2.70 runner；复现 v9.2.69 boundary，构造 full-online event table，审计 PF5 selector/candidate pack、candidate-only branch forward、true-delta logits、frozen bridge lookup 与 system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9270_full_online_row_binding_system_legal_controller_closure.py
```

正式运行：

```bash
python experiments/run_v9270_full_online_row_binding_system_legal_controller_closure.py \
  --out-dir results/real_rerun_20260506/v9270_full_online_row_binding_system_legal_controller_closure_first_20260513T010000Z \
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
completed_at = 2026-05-13T05:27:32Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R14-FullOnlineBindingFail",
  "base_candidate": "LQ-t2-h256",
  "v9269_boundary_pass": 1,
  "reference_controller_id": "C3-T2PlusBackfill",
  "reference_controller_reproduced": 1,
  "reference_precision": 0.8380281690140845,
  "reference_coverage": 0.03130511463844797,
  "reference_bad_event": 0.02464788732394366,
  "reference_null_rate": 0.13028169014084506,
  "reference_precision_lcb": 0.7907157243773478,
  "reference_bad_event_ucb": 0.04999458813129621,
  "full_online_row_binding_contract_pass": 0,
  "event_identity_binding_pass": 1,
  "pf5_full_online_selector_pass": 1,
  "candidate_pack_binding_pass": 0,
  "candidate_tensor_payload_missing_count": 2493,
  "candidate_branch_logits_missing_count": 2493,
  "candidate_true_delta_logits_missing_count": 2493,
  "functional_update_payload_missing_count": 951,
  "fused_bridge_full_online_pass": 1,
  "system_legal_controller_pass": 0,
  "primary_blocker": "candidate_tensor_payload_missing_for_full_online_rows",
  "next_required_implementation": "persist_candidate_x_y_logits_and_update_payload_for_full_online_rows"
}
```

判断：v9.2.70 把 full-online event identity 与 PF5/frozen bridge lookup 绑定清楚了，但 official 所需的 raw candidate tensor payload、branch logits、true-delta logits 和 accepted update payload 没有全量绑定，因此 route 必须停在 `R14-FullOnlineBindingFail`。

## 3. P1 full-online row binding contract

Artifacts：

```text
p1_full_online_row_binding_contract_audit.csv
full_online_binding_trace_v9270.csv
full_online_event_table_v9270.csv
```

| component | materialized | full binding | missing | note |
|---|---:|---:|---:|---|
| online event table | `1` | `1` | `0` | global row/event/source hash |
| PF5 candidate selector | `1` | `1` | `0` | candidate flag/index/rank/score |
| candidate id tensors | `1` | `1` | `0` | event/family/bucket/branch ids |
| candidate tensor payload | `0` | `0` | `2493` | `candidate_x/y/logits_base` not stored for full rows |
| candidate branch logits | `0` | `0` | `2493` | branch logits not materialized for all PF5 rows |
| candidate true-delta logits | `0` | `0` | `2493` | true delta logits not bound beyond microprobe |
| frozen C3 bridge lookup | `1` | `1` | `0` | compact accept lookup |
| bridge score and accept decision | `1` | `1` | `0` | tensor gather over row ids |
| functional update payload | `0` | `0` | `951` | accepted-row update payload not materialized |

Summary：

```text
event_count = 24192
candidate_count = 2493
accepted_count = 951
event_identity_binding_pass = 1
full_online_row_binding_contract_pass = 0
full_online_row_binding = 0
```

判断：这就是本轮 terminal blocker。row identity 不是问题；缺的是全量 candidate payload 和 accepted functional update payload。

## 4. P2 PF5 selector / candidate pack binding

Artifacts：

```text
p2_full_online_event_table_pf5_selector_binding.csv
pf5_full_online_selector_trace_v9270.csv
candidate_pack_full_online_trace_v9270.csv
```

Summary：

```text
prefilter_id = PF5-LearnedMonotoneCheapPrefilter
event_count = 24192
candidate_count = 2493
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
reference_accept_precision = 0.381468110709988
candidate_pack_time_ms = 7.448278
candidate_pack_memory_MB = 0.057060
event_id_duplicate_count = 0
candidate_index_duplicate_count = 0
candidate_missing_id_count = 0
candidate_event_id_mismatch_count = 0
pf5_full_online_selector_pass = 1
candidate_pack_binding_pass = 0
```

判断：PF5 selector 和 candidate id/metadata 是 full-online bound；但 candidate payload 没有绑定，不能把 P2 写成完整 candidate pack success。

## 5. P3/P4 candidate branch forward 与 true-delta

P3 artifacts：

```text
p3_full_online_candidate_only_branch_forward.csv
candidate_branch_forward_full_online_trace_v9270.csv
```

P3 summary：

```text
best_branch_forward_id = BF0-V9269MaterializedMicroprobeDiagnostic
candidate_branch_forward_diagnostic_pass = 1
candidate_branch_forward_full_online_pass = 0
candidate_forward_time_ratio = 0.018018326773696327
branch_logit_error_max = 0.0
primary_binding_blocker = candidate_x_y_logits_payload_missing_for_full_online_candidate_rows
```

P4 artifacts：

```text
p4_full_online_materialized_true_delta_exact_confirmation.csv
full_online_true_delta_trace_v9270.csv
```

P4 summary：

```text
best_exact_candidate_id = TD1-FullOnlinePF5CandidateTrueDelta
diagnostic_source_id = EC2-MaterializedPF5CandidateTrueDeltaMicroprobe
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
decision_cost_diagnostic_pass = 1
agreement_reference_accept = 1.0
step_ratio_q90 = 1.075342155736442
memory_ratio = 1.0
precision = 0.8380281690140845
coverage = 0.03130511463844797
bad_event = 0.02464788732394366
null_rate = 0.13028169014084506
full_online_materialized_true_delta_pass = 0
materialization_blocker = candidate_true_delta_logits_not_bound_to_full_online_rows
```

判断：v9.2.69 的 microprobe 数值仍然强，但 v9.2.70 没有把它升级成 full-online true-delta，因为 `2493` 个 candidate 的 true-delta logits 没有全量绑定。

## 6. P5 frozen bridge lookup

Artifacts：

```text
p5_full_online_fused_compact_bridge_compute.csv
full_online_bridge_lookup_trace_v9270.csv
```

Summary：

```text
bridge_system_id = BR1-FullOnlineFrozenC3LookupTensorGather
uses_compact_lookup = 1
uses_cpu_summary = 0
online_frontier_search_used = 0
materialized_system_path = 1
full_online_row_binding = 1
candidate_count = 2493
bridge_lookup_time_ms = 0.152631
agreement_reference_accept = 1.0
precision = 0.8380281690140845
coverage = 0.03130511463844797
bad_event = 0.02464788732394366
null_rate = 0.13028169014084506
fused_bridge_full_online_pass = 1
```

判断：bridge lookup 本身已 materialized 且绑定 full online rows。但 bridge lookup 不能代替缺失的 full-online true-delta/candidate payload，所以 P6 仍不能 official。

## 7. P6 system controller boundary

Artifact：

```text
p6_system_legal_exact_signal_controller_v3.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
official_eligible = 0
system_legal_controller_pass = 0
reason = P1_or_P4_full_online_binding_failed
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
```

判断：decision metrics / bridge lookup / cost diagnostic 都看起来满足门槛；但 official eligibility 仍为 `0`，因为 P1/P4 binding contract 不完整。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P1_full_online_binding_contract_failed` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 PF5 selector pass、bridge lookup pass、decision/cost diagnostic 或 v9.2.69 microprobe 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 9. No-fake audit

```text
rows_checked = 24244
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
full_online_event_table = 1
materialized_pf5_selector = 1
candidate_pack_metadata_materialized = 1
candidate_tensor_payload_bound = 0
candidate_only_branch_forward_full_online = 0
materialized_true_delta_full_online = 0
frozen_bridge_lookup_materialized = 1
full_online_row_binding = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `200de32f9a95a5e34dc6c55cd7be8005af961b924b30fe06d77d0baee2bb4382` |
| runner | `f45a02140dc75cc37196ccce0cc976eee926bf328309e36acb9d2bbd86e0d60b` |
| route | `14c43f23f8acbdc238ed72bfe4f4d18daea7b64d5e1c1847b15d89a671971c16` |
| run manifest | `cc4ef9ec69ecb98dea1c77a219a422976c4730c8df860e04a1be9a7e5209406f` |
| P1 binding audit | `09a050fbad9b4760c5177c45f4266f387907e474d447a2d3dc20675ef9f69c87` |
| P2 selector binding | `b63ade48625d259004b6a1a9c0d8dd4390bfa7a4889e634fb384db8ec1906eb2` |
| P3 branch forward | `bfa7058af93f49a45265e3be94d7c139964353c19bdc20812fdb2ef17c4e52e9` |
| P4 true delta | `0f29f79f9b900a5b9bba46820cbd98e78bb68940110f93c7f49ed189ad285758` |
| P5 bridge lookup | `072e0d8a6fe4c90b264a347fa5fe7d1a4cbd0304b85a75e6d50b9e6ace9fde54` |
| P6 system controller | `c134efb6e6f84ab75970edd33832043ce4897a66dce0dec665361af7efc650cc` |
| failure table | `77ea3c5e51848c02fc64e6781b93e97feaa0bcab0f202780b578fe44d3048c2b` |
| provenance audit | `6165cc93b15d6fffd5eddbf1a8adf1a75c240b1b80dbc0383d270ce8cebacca2` |

## 11. 最终分析结论

v9.2.70 的真实推进是：

```text
v9.2.69: event-sparse true-delta 已是真实 materialized microprobe，
          但缺 full online row binding。
v9.2.70: full-online event table / PF5 selector / frozen bridge lookup 已绑定；
          但 full-online candidate payload 和 true-delta logits 仍未全量落盘。
```

机制判断：

1. H1 部分成立：full-online row identity、candidate ids、PF5 selector 和 bridge lookup 都能绑定，duplicate/mismatch 为 `0`。
2. H2 未闭合：candidate tensor payload、candidate branch logits、candidate true-delta logits 均缺 `2493` rows，accepted functional update payload 缺 `951` rows。
3. H3 不能 official：v9.2.69 microprobe 的 agreement/step/memory 仍好，但本轮明确不把 microprobe 写成 full-online system pass。
4. H4 成立：frozen C3 lookup 已经是 full-online tensor gather，不使用 CPU summary 或 online frontier search。
5. H5 未打开：P6 `official_eligible = 0`，因此 LDO/LSO、paired replay、short/full validation 全部 gate-blocked。

最终一句话：

> v9.2.70 真实执行后停在 `R14-FullOnlineBindingFail`：full-online event identity、PF5 selector 和 frozen bridge lookup 已绑定，但 candidate payload / branch logits / true-delta logits / update payload 未全量 materialize，system-legal controller 仍不能转正。
