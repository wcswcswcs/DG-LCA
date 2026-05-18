# DG-KAN v9.2.71 Full-Online Payload Binding 与 Functional-Update Payload Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.71_FullOnlinePayloadBinding_FunctionalUpdatePayloadClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 payload-bound 但 system-too-expensive 的 controller 写成 official pass。

## 0. 最新结论

```text
route = R17-PayloadBoundSystemStillExpensive
base_candidate = LQ-t2-h256
success_v9271_strict_purekan_functional = False
success_v9271_full_functional = False
success_v9271_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9271_full_online_payload_binding_functional_update_payload_closure_first_20260513T063000Z/
```

追加 candidate-forward 优化尝试 artifact：

```text
results/real_rerun_20260506/v9271_full_online_payload_binding_functional_update_payload_closure_optimized_fix_20260513T073000Z/
results/real_rerun_20260506/v9271_full_online_payload_binding_functional_update_payload_closure_hybrid_20260513T074500Z/
```

核心结论：

1. P0 复现 v9.2.70 boundary：source route = `R14-FullOnlineBindingFail`，reference controller = `C3-T2PlusBackfill`，PF5 selector / frozen bridge lookup 已过，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；实际参数保持 MNIST/Fashion-MNIST/KMNIST、seeds `0..7`、microprobe steps `336`。
3. P1 full-online payload binding contract 通过：candidate tensor payload、branch/control logits、true-delta/selected-delta logits、accepted functional update payload missing count 全部为 `0`。
4. P2 candidate tensor payload 已全量 materialize：candidate count = `2493 / 24192`，candidate rate = `0.10305059523809523`，reference accept recall = `1.0`，candidate pack time = `608.782669 ms`。
5. P3 candidate branch/control logits 已全量绑定：branch/control missing count = `0`，branch forward mean = `1.410317 ms`，q90 = `1.482881 ms`，logit max abs diff = `0.0`。
6. P4 true-delta logits 已全量绑定：true-delta/selected-delta missing count = `0`，uses true branch delta = `1`，source gap / formula proxy = `0`，agreement = `1.0`。
7. P5 accepted functional update payload 已全量绑定：accepted count = `951`，functional update payload count = `951`，missing = `0`，payload for rejected rows = `0`，manual update ready = `1`。
8. P6 decision metrics 仍满足 reference frontier：precision = `0.838028`，coverage = `0.031305`，bad-event = `0.024648`，null-rate = `0.130282`，precision LCB = `0.790716`，bad-event UCB = `0.049995`。
9. P6 system-legal controller 仍未打开：payload-bound step ratio q90 = `4.248161404287591 > 1.50`，official eligible = `0`，system legal controller pass = `0`。
10. 追加尝试 exact-W2 delta logit materialization：24 个 direct-forward identity check 的 max abs error = `1.9073486328125e-06`，但 corrected step ratio q90 = `4.558396271484858`，没有过 system envelope。
11. 追加尝试 hybrid path：单 candidate step 使用 direct forward，2/3 candidate step 使用 exact-W2 shared path；branch forward q90 降到 `1.395499 ms`，但 step ratio q90 仍为 `4.53819861019958 > 1.50`。
12. 当前 blocker 仍是：`payload_bound_true_delta_step_ratio_fail`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9271_full_online_payload_binding_functional_update_payload_closure.py` | v9.2.71 runner；复现 v9.2.70 boundary，真实 replay train-stream，materialize PF5 candidate payload、branch/control logits、true-delta logits、frozen bridge decision 与 accepted functional update payload |

代码检查：

```text
python -m py_compile experiments/run_v9271_full_online_payload_binding_functional_update_payload_closure.py
```

正式运行：

```bash
python experiments/run_v9271_full_online_payload_binding_functional_update_payload_closure.py \
  --out-dir results/real_rerun_20260506/v9271_full_online_payload_binding_functional_update_payload_closure_first_20260513T063000Z \
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
completed_at = 2026-05-13T06:01:05Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R17-PayloadBoundSystemStillExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9270_boundary_pass": 1,
  "reference_controller_id": "C3-T2PlusBackfill",
  "reference_controller_reproduced": 1,
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_payload_pass": 1,
  "candidate_branch_logits_missing_count": 0,
  "candidate_control_logits_missing_count": 0,
  "candidate_true_delta_logits_payload_pass": 1,
  "candidate_true_delta_logits_missing_count": 0,
  "selected_delta_logits_missing_count": 0,
  "functional_update_payload_pass": 1,
  "functional_update_payload_missing_count": 0,
  "payload_for_rejected_rows_count": 0,
  "controller_precision": 0.8380281690140845,
  "controller_coverage": 0.03130511463844797,
  "controller_bad_event": 0.02464788732394366,
  "controller_null_rate": 0.13028169014084506,
  "controller_precision_lcb": 0.7907157243773478,
  "controller_bad_event_ucb": 0.04999458813129621,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 4.248161404287591,
  "true_delta_memory_ratio": 1.0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "primary_blocker": "payload_bound_true_delta_step_ratio_fail",
  "next_required_implementation": "optimize_payload_bound_candidate_forward"
}
```

判断：v9.2.71 已经把 v9.2.70 的 payload missing blocker 推进掉；但真实 full-online payload-bound path 的 step ratio 远高于 `1.50`，因此不能打开 system-legal controller。

## 3. P1 full-online payload binding contract

Artifacts：

```text
p1_payload_binding_contract_audit.csv
full_online_event_table_v9271.csv
end_to_end_payload_binding_trace_v9271.csv
payload_lifecycle_trace_v9271.csv
```

Summary：

```text
payload_binding_contract_pass = 1
unknown_blocker_count = 0
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
```

判断：row identity、candidate payload、branch logits、true-delta logits、update payload 均已绑定。`full_online_event_table_v9271.csv` 中 candidate rows 带有 `candidate_x_hash`、`candidate_y_hash`、`candidate_base_logits_hash`、`branch_logits_hash`、`control_logits_hash`、`true_delta_logits_hash`、`selected_delta_logits_hash`。

## 4. P2 candidate tensor payload

Artifacts：

```text
p2_candidate_tensor_payload_materialization.csv
candidate_payload_table_v9271.csv
```

Summary：

```text
payload_candidate_id = PB1-CandidateXYBaseLogitsPayload
event_count = 24192
candidate_count = 2493
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
candidate_x_bound = 1
candidate_y_bound = 1
candidate_base_logits_bound = 1
candidate_tensor_payload_missing_count = 0
candidate_pack_time_ms = 608.7826690636575
candidate_pack_memory_MB = 484.4794921875
projection_used = 0
posthoc_used_at_commit = 0
dataset_name_used = 0
candidate_tensor_payload_pass = 1
```

判断：v9.2.70 缺的 `candidate_x/y/base_logits` 在本轮已按 PF5 candidate 全量 materialize，并记录 shape/device/dtype/hash。

## 5. P3 candidate branch/control logits

Artifacts：

```text
p3_candidate_branch_logits_payload_materialization.csv
candidate_branch_logits_table_v9271.csv
```

Summary：

```text
branch_forward_id = BF1-FullOnlineCandidatePackedForwardV2
candidate_count = 2493
candidate_rate = 0.10305059523809523
uses_candidate_only_forward = 1
uses_full_row_forward = 0
candidate_branch_logits_missing_count = 0
candidate_control_logits_missing_count = 0
branch_forward_time_ms_mean = 1.4103165509405622
branch_forward_time_ms_q90 = 1.4828811399638653
branch_forward_ratio_vs_full = 0.10305059523809523
logit_max_abs_diff_vs_full_reference = 0.0
state_reproduction_extra_non_candidate_forward_count = 902
candidate_branch_logits_payload_pass = 1
```

判断：candidate-only branch/control logits 已绑定；但 replay 为复现 source train-stream state 额外执行了 `902` 个非 PF5 candidate forward，这个计数已落盘，没有隐藏。

## 6. P4 candidate true-delta logits

Artifacts：

```text
p4_candidate_true_delta_logits_materialization.csv
candidate_true_delta_logits_table_v9271.csv
```

Summary：

```text
true_delta_candidate_id = TD1-FullOnlineCandidateOnlyTBD0V2
candidate_count = 2493
candidate_true_delta_logits_missing_count = 0
selected_delta_logits_missing_count = 0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
agreement_reference_accept = 1.0
AUC_safe_good = 0.8790720756550714
AUC_bridge_accept = 0.8113610999018152
precision = 0.8380281690140845
coverage = 0.03130511463844797
bad_event = 0.02464788732394366
null_rate = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
step_ratio_q90 = 4.248161404287591
memory_ratio = 1.0
candidate_true_delta_logits_payload_pass = 1
true_delta_system_pass = 0
```

判断：true-delta logits binding 成功，且没有 source-measured gap / formula proxy；失败点不是 signal/decision，而是 full payload-bound system cost。

## 7. P5 accepted functional update payload

Artifacts：

```text
p5_accepted_functional_update_payload_materialization.csv
functional_update_payload_table_v9271.csv
```

Summary：

```text
update_payload_candidate_id = UP1-DeltaThetaBlockRefPayload
accepted_count = 951
functional_update_payload_count = 951
functional_update_payload_missing_count = 0
payload_for_rejected_rows_count = 0
delta_theta_block_ref_present = 1
functional_update_scale = 1.0
manual_update_ready = 1
payload_hash_present = 1
payload_apply_time_ms = 0.7214542036654759
payload_memory_MB = 746.68359375
functional_update_payload_pass = 1
```

判断：v9.2.70 缺的 accepted-row update payload 已闭合；没有为 rejected rows 生成 update payload。

## 8. P6 payload-bound system controller

Artifact：

```text
p6_payload_bound_system_legal_controller.csv
system_controller_trace_v9271.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
status = not_run
official_eligible = 0
system_legal_controller_pass = 0
reason = payload_bound_system_still_expensive
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
agreement_reference_accept = 1.0
step_ratio_q90 = 4.248161404287591
memory_ratio = 1.0
accepted_strata_count = 15
accepted_family_count = 65
full_online_row_binding = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
projection_used = 0
full_trace_projection_used = 0
source_measured_gap_used = 0
formula_proxy_used = 0
cpu_offload_used = 0
```

判断：decision / support / payload binding 都满足，但 system envelope 不满足。这里没有把 payload-bound controller 写成 official pass。

## 9. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_payload_bound_system_too_expensive` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 payload binding pass、decision metrics、true-delta agreement 或 update payload closure 写成 LDO/LSO、paired replay、short/full validation success。

## 10. No-fake audit

```text
rows_checked = 48556
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
candidate_payload_materialized = 1
candidate_branch_logits_materialized = 1
candidate_true_delta_logits_materialized = 1
functional_update_payload_materialized = 1
full_online_payload_binding = 1
full_online_update_payload_binding = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `d0e3f6587bf9731ba03c9ddf1d86534cf9f8b221f4ebb1300e1371071f7866d0` |
| runner | `a6d35a0a984b8d2a004049494b7f720a8789b589c19dda9c3469a0a7b0d1ec65` |
| run manifest | `cc4ef9ec69ecb98dea1c77a219a422976c4730c8df860e04a1be9a7e5209406f` |
| route | `14c43f23f8acbdc238ed72bfe4f4d18daea7b64d5e1c1847b15d89a671971c16` |
| P0 boundary | `1fbd06efac216af83df89ec85d3dc7bf1ba33ba156736dca6ac5d7cbbce820c2` |
| P1 binding audit | `62ee869025d068ab25bc8f98a071d6b6682928d1b0bc69f0ba4b74b2fc050f6e` |
| P2 candidate payload | `2128c58d37e1f2575d8169c2390d26691618937b82e3b400940e5402edf74bc4` |
| P3 branch logits | `136d296869b7c2004660577ff0cf24a94f287bf2fc05f8a288711932ed895db9` |
| P4 true delta | `f76cf5e19e6057ada4a3014400faf13fe280bc94f96c0f863e5d0daac4c8a651` |
| P5 update payload | `38c645d3f66bd1371176e2790456ad21ee61cf8860e9ab7efb25b69124e947b9` |
| P6 system controller | `288d031cf77fb0cf2477c568e30dff05b474d596ef94a30c9c1a988cd115e75c` |
| event table | `0525cdbbbb3b564ecbffd1dd96c0508d797a900783a14e2b5fec6ac4ff38b774` |
| candidate payload table | `4003778a5b27debbf133df039c07b1c1bc2907403c7a7263ccfdf3ebe71e6d97` |
| branch logits table | `d85c71fe2760bfb7af79cd15c2cf6e0ed76aea116cbdf39d3c16c7a6f9bdd020` |
| true delta logits table | `c220c366070a45ff57b827e14ef76474638c69edf6891437fe254636fb7682e3` |
| update payload table | `716ebd288789efb23cdd9a04dcdd0ddd193c21fbe9b885986f11b1cdad3d03f2` |
| bridge decision table | `778ca41cde81f04f8c22d8bbb23b0c30f6fddca9b4af53a000e979ecc3e68f9b` |
| failure table | `19884b0aeeb76a3bfa402dce15f3a8a024523ea8d5b0608fa4b6f32f84ff5f70` |
| provenance audit | `6df72cc31bf66655653ad3153146f7b4557264f7b8506f1696839c80cc304fa6` |

## 12. 最终分析结论

v9.2.71 的真实推进是：

```text
v9.2.70: full-online event identity / PF5 selector / frozen bridge lookup 已绑定，
          但 candidate payload / logits / update payload 缺失。
v9.2.71: full-online payload binding 与 functional-update payload closure 已完成；
          但 payload-bound true-delta system path 的 step ratio 过高。
```

机制判断：

1. H1 成立：candidate x/y/base logits payload 已为 `2493` 个 PF5 candidate 全量落盘，missing = `0`。
2. H2 成立：candidate branch/control logits 与 true-delta/selected-delta logits 已全量 materialize，source gap / formula proxy 均为 `0`。
3. H3 成立：accepted functional update payload 已为 `951` 个 accepted row 全量绑定，没有 rejected-row update payload。
4. H4 成立：reference decision metrics 没有 drift，agreement = `1.0`，precision/coverage/bad-event/null/LCB/UCB 仍满足 reference frontier。
5. H5 未闭合：full payload-bound measured `step_ratio_q90 = 4.248161`，显著高于 `1.50`，因此 system-legal controller 仍不能 official。
6. 当前下一步不是继续补 payload，而是优化 payload-bound candidate forward / true-delta compute path，把 full-online materialized route 压回 system envelope。

最终一句话：

> v9.2.71 真实执行后停在 `R17-PayloadBoundSystemStillExpensive`：full-online payload 与 functional-update payload 已全部绑定，但真实 payload-bound true-delta 路径太慢，strict PureKAN functional 仍未成功。

## 13. 追加修复尝试：candidate forward 优化

目标：在不使用 fake/proxy/source-measured gap 的前提下，尝试解决 “全量 payload-bound candidate forward 太慢”。

实现改动：

```text
exact-W2 delta logit materialization:
  因 functional delta 只改 W2，不改 A/W0；
  candidate logits 可精确写成 task_logits_probe + vals_probe[1] @ delta_W2。

hybrid materialization:
  单 candidate step 仍用 direct candidate forward；
  2/3 candidate step 使用 exact-W2 shared update/probe basis path。
```

这不是公式 proxy，也不是 source-measured gap；它仍然 materialize candidate branch logits / true-delta logits，并保留 direct-forward identity check。

Artifacts：

```text
results/real_rerun_20260506/v9271_full_online_payload_binding_functional_update_payload_closure_optimized_fix_20260513T073000Z/
results/real_rerun_20260506/v9271_full_online_payload_binding_functional_update_payload_closure_hybrid_20260513T074500Z/
```

对比结果：

| run | route | payload pass | branch q90 ms | step q90 | identity max error | system pass |
|---|---|---:|---:|---:|---:|---:|
| first direct full-online | `R17-PayloadBoundSystemStillExpensive` | `1` | `1.482881` | `4.248161` | n/a | `0` |
| exact-W2 corrected | `R17-PayloadBoundSystemStillExpensive` | `1` | `1.469111` | `4.558396` | `1.907349e-06` | `0` |
| hybrid direct/exact-W2 | `R17-PayloadBoundSystemStillExpensive` | `1` | `1.395499` | `4.538199` | `1.907349e-06` | `0` |

Hybrid summary：

```text
candidate_count = 2493
exact_w2_candidate_count = 2047
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
branch_forward_time_ms_mean = 1.0771113315225822
branch_forward_time_ms_q90 = 1.3954988680779934
true_delta_step_ratio_q90 = 4.53819861019958
optimized_identity_check_count = 24
optimized_identity_error_max = 1.9073486328125e-06
system_legal_controller_pass = 0
```

判断：

1. exact-W2 algebra path 数值正确：direct-forward identity check max error 只有 `1.907349e-06`。
2. hybrid 确实降低了 branch forward 局部 q90：从 `1.482881 ms` 降到 `1.395499 ms`。
3. 但 system step ratio 没有降入 envelope，反而仍高于 first direct run：`4.538199 > 4.248161 > 1.50`。
4. 这说明当前 blocker 不是单个 candidate logits 公式能解决的局部 forward 成本；step-level common-prep、functional delta 构造、同步/CPU norm/hash 以及多-candidate step 分布仍把 full payload-bound route 推出 system envelope。
5. 因此没有打开 system-legal controller，也没有运行 LDO/LSO、paired replay、short/full validation。

追加尝试 no-fake audit：

```text
rows_checked = 48556
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Hybrid artifact hashes：

| artifact | SHA256 |
|---|---|
| runner | `cca7f7b22242782fd99db4d287e400858c56492e6392fc1b46e14fa19e27b440` |
| P3 branch logits | `5930911283c4d972e85ea9866828fa8d02dbbd8d9388c883a16e052c60138238` |
| P4 true delta | `790ea1f2152c5cb4f460b1ace8dbf89feadf40b406aab4f070eca154acce1ac3` |
| P6 system controller | `2ebd6c59c565af0bacd57574eee537b97df303cac528195500ee78a9563b9f07` |
| provenance audit | `6df72cc31bf66655653ad3153146f7b4557264f7b8506f1696839c80cc304fa6` |

追加结论：

> 本轮 candidate-forward 优化尝试没有解决 blocker：exact-W2/hybrid 路径保持真实 payload binding 与数值一致性，但 full-online payload-bound system cost 仍远超 `1.50`，下一步需要更底层 fused/async payload-bound candidate forward，而不是 Python-level algebra patch。
