# DG-KAN v9.2.72 Payload-Bound System Cost Closure 与 Fused Candidate-Forward / True-Delta Pipeline 实验复盘

> 本复盘记录 `DG-KAN_v9.2.72_PayloadBoundSystemCostClosure_FusedCandidateTrueDeltaPipeline_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 payload-bound 但 system-too-expensive 的 controller 写成 official pass。

## 0. 最新结论

```text
route = R18-PayloadBoundSystemStillExpensive
base_candidate = LQ-t2-h256
success_v9272_strict_purekan_functional = False
success_v9272_full_functional = False
success_v9272_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_first_20260513T083000Z/
```

追加优化 artifact：

```text
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_batched_cuda_20260513T091500Z/
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_cuda_ext_20260513T103000Z/
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z/
```

核心结论：

1. P0 复现 v9.2.71 boundary：source route = `R17-PayloadBoundSystemStillExpensive`，payload binding pass = `1`，all missing payload counts = `0`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；正式运行中 GPU 有活动，后段进入 CPU/hash/CSV 汇总阶段。
3. P1 step-cost attribution 已闭合：unknown fraction = `0.0`，dominant cost subphase = `fused_candidate_delta_logits_ms`，total = `1621.190351 ms`。
4. P2 candidate payload lifecycle 过 gate：candidate count = `2493 / 24192`，candidate rate = `0.10305059523809523`，reference accept recall = `1.0`，candidate pack q90 = `0.279152 ms`。
5. P3 state reproduction extra forward 已消除：extra non-candidate forward 从 v9.2.71 的 `902` 降为 `0`；但仍有 replay delta construct count = `902`。
6. P4 fused candidate true-delta pipeline 仍保持数值一致：identity check count = `24`，max abs error = `1.9073486328125e-06`，branch q90 = `1.450533 ms`。
7. P5 compressed update payload 过 gate：accepted count = `951`，functional update payload missing = `0`，compressed W2-only payload memory = `9.287109 MB`。
8. P6 controller decision metrics 仍满足 reference frontier：precision = `0.838028`，coverage = `0.031305`，bad-event = `0.024648`，null-rate = `0.130282`。
9. P6 system-legal controller 仍未打开：step ratio q90 = `4.457842809866833 > 1.50`，official eligible = `0`。
10. 当前 blocker：`payload_bound_true_delta_step_ratio_fail`。
11. 追加 BF3 batched CUDA carrier path 后，step ratio q90 降到 `4.043819522772543`，但仍未过 `1.50`。
12. 追加 BF4 few-kernel CUDA extension 后，CUDA-vs-torch 数值对照通过：24 个 check 中 logits max error = `2.670288e-05`，delta max error = `1.280569e-09`。
13. BF4 将 fused candidate delta/logits total 从 BF3 的 `1134.071934 ms` 降到 `111.270273 ms`，branch q90 从 `1.377242 ms` 降到 `0.866299 ms`。
14. BF4 system step ratio q90 仍为 `2.969948325415854 > 1.50`，dominant 转为 `common_basis_prep_ms`，因此 strict PureKAN functional 仍未成功。
15. 追加 BF5 async combined basis+delta path 后，step ratio q90 进一步降到 `2.713295831053225`，branch q90 降到 `0.766299 ms`；但仍未过 `1.50`。
16. BF5 数值对照仍正确：CUDA-vs-torch logits max error = `2.670288e-05`，delta max error = `8.149073e-10`，no fake/proxy/offload = `0`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py` | v9.2.72 runner；复现 v9.2.71 boundary，执行 payload-bound step cost attribution、candidate payload lifecycle 优化、state replay extra forward elimination、fused candidate true-delta bridge pipeline、compressed update payload 与 system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py
```

正式运行：

```bash
python experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py \
  --out-dir results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_first_20260513T083000Z \
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
exact_w2_min_candidates = 1
completed_at = 2026-05-13T07:03:13Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R18-PayloadBoundSystemStillExpensive",
  "base_candidate": "LQ-t2-h256",
  "v9271_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "step_cost_attribution_pass": 1,
  "dominant_cost_subphase": "fused_candidate_delta_logits_ms",
  "cost_unknown_fraction": 0.0,
  "candidate_payload_lifecycle_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_control_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "selected_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "state_reproduction_pass": 1,
  "state_reproduction_extra_non_candidate_forward_count": 0,
  "state_reproduction_delta_construct_count": 902,
  "controller_precision": 0.8380281690140845,
  "controller_coverage": 0.03130511463844797,
  "controller_bad_event": 0.02464788732394366,
  "controller_null_rate": 0.13028169014084506,
  "controller_precision_lcb": 0.7907157243773478,
  "controller_bad_event_ucb": 0.04999458813129621,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 4.457842809866833,
  "true_delta_memory_ratio": 1.0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "primary_blocker": "payload_bound_true_delta_step_ratio_fail",
  "next_required_implementation": "lower_level_fused_async_payload_bound_candidate_forward"
}
```

判断：v9.2.72 没有出现 payload regression；失败点是 full payload-bound candidate true-delta path 的系统成本仍太高。

## 3. P1 payload-bound step cost attribution

Artifacts：

```text
p1_payload_bound_step_cost_attribution.csv
payload_bound_step_cost_trace_v9272.csv
```

Summary：

```text
step_cost_attribution_pass = 1
step_ratio_q50 = 1.0
step_ratio_q90 = 4.457842809866833
unknown_fraction = 0.0
dominant_subphase = fused_candidate_delta_logits_ms
dominant_subphase_ms_total = 1621.1903505027294
common_basis_prep_ms_total = 998.0899034999311
true_delta_selected_feature_ms_total = 298.55674831196666
candidate_pack_time_ms_q90 = 0.2791518345475197
branch_forward_time_ms_q90 = 1.4505330473184586
true_delta_time_ms_q90 = 0.1314687542617321
payload_apply_time_ms_q90 = 0.7461677305400372
```

判断：P1 回答了 v9.2.71 的关键问题。系统成本不是未知 overhead，也不是 candidate pack 被误计入 step q90；dominant 明确是 fused candidate delta/logits path，其次是 common basis prep。

## 4. P2 candidate payload lifecycle optimization

Artifacts：

```text
p2_candidate_payload_lifecycle_optimization.csv
candidate_payload_optimization_trace_v9272.csv
candidate_payload_table_v9272.csv
```

Summary：

```text
payload_candidate_id = PB2-StableRefCandidatePayload
event_count = 24192
candidate_count = 2493
candidate_rate = 0.10305059523809523
reference_accept_recall = 1.0
candidate_tensor_payload_missing_count = 0
candidate_pack_time_ms = 582.4845680035651
candidate_pack_time_ms_q90 = 0.2791518345475197
candidate_pack_memory_MB = 484.4794921875
candidate_payload_lifecycle_pass = 1
```

判断：payload lifecycle 没有 regression，candidate payload 仍全量绑定。优化后 pack q90 已不是 terminal blocker。

## 5. P3 state reproduction extra forward elimination

Artifacts：

```text
p3_state_reproduction_extra_forward_elimination.csv
state_reproduction_trace_v9272.csv
```

Summary：

```text
state_reproduction_extra_non_candidate_forward_count_before_v9271 = 902
state_reproduction_extra_non_candidate_forward_count_after = 0
state_reproduction_delta_construct_count = 902
candidate_forward_required_for_state_replay = 0
accept_agreement_after_state_replay = 1.0
state_reproduction_pass = 1
```

判断：v9.2.71 中的 `902` 个 extra non-candidate forward 已被消掉，说明当前 step ratio 失败不能再归因于非 candidate forward replay。

## 6. P4 fused branch-forward / true-delta / bridge pipeline

Artifacts：

```text
p4_fused_branch_forward_true_delta_bridge_pipeline.csv
fused_delta_bridge_trace_v9272.csv
candidate_branch_logits_table_v9272.csv
candidate_true_delta_logits_table_v9272.csv
bridge_decision_table_v9272.csv
```

Summary：

```text
branch_forward_id = BF2-FusedCandidateTrueDeltaPipeline
true_delta_candidate_id = TD2-FusedCandidateTrueDeltaBridge
payload_candidate_id = PB2-StableRefCandidatePayload
candidate_count = 2493
candidate_branch_logits_missing_count = 0
candidate_control_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
selected_delta_logits_missing_count = 0
exact_w2_candidate_count = 2493
branch_forward_time_ms_mean = 1.050653932612382
branch_forward_time_ms_q90 = 1.4505330473184586
optimized_identity_check_count = 24
optimized_identity_error_max = 1.9073486328125e-06
agreement_reference_accept = 1.0
uses_true_branch_delta = 1
uses_source_measured_gap = 0
uses_formula_proxy = 0
step_ratio_q90 = 4.457842809866833
true_delta_system_pass = 0
```

判断：fused path 数值正确且 payload-bound，但仍没有压进 system envelope。局部 branch q90 与 v9.2.71 hybrid 接近，full step ratio 仍高。

## 7. P5 compressed functional update payload

Artifacts：

```text
p5_compressed_functional_update_payload_lazy_apply.csv
compressed_update_payload_trace_v9272.csv
functional_update_payload_table_v9272.csv
```

Summary：

```text
update_payload_candidate_id = UP2-CompressedW2DeltaLazyApplyPayload
accepted_count = 951
functional_update_payload_count = 951
functional_update_payload_missing_count = 0
payload_for_rejected_rows_count = 0
compressed_w2_only_payload = 1
lazy_apply = 1
manual_update_ready = 1
payload_apply_time_ms = 0.7230835133374176
payload_memory_MB = 9.287109375
functional_update_payload_pass = 1
```

判断：P5 是本轮明确推进。update payload 从 full block payload 压成 W2-only compressed/lazy payload，missing 仍为 `0`，但这不足以解决 P6 system cost。

## 8. P6 payload-bound system controller

Artifacts：

```text
p6_payload_bound_system_legal_controller_v4.csv
system_controller_trace_v9272.csv
```

Boundary：

```text
controller_id = C3-T2PlusBackfill
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
step_ratio_q90 = 4.457842809866833
memory_ratio = 1.0
accepted_strata_count = 15
accepted_family_count = 65
```

判断：decision / support / payload binding / update payload 都过，但 system cost 不过。因此 official controller 仍不能转正。

## 9. Downstream boundary

这些 artifact 均已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_leave_dataset_and_stratum_out.csv` | `P6_payload_bound_system_too_expensive` |
| `p8_official_paired_replay.csv` | same |
| `p9_short_run_functional_validation.csv` | same |
| `p10_full_run_robustness_strong_baseline.csv` | same |

没有把 payload lifecycle pass、state replay pass、compressed update payload pass 或 fused path diagnostic 写成 LDO/LSO、paired replay、short/full validation success。

## 10. No-fake audit

```text
rows_checked = 73120
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
candidate_payload_lifecycle_optimized = 1
state_reproduction_extra_forward_eliminated = 1
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
| plan | `45133dadb41c0b184377f799169f5b833640dc731edb3b2b16e76fbf5c7364dc` |
| runner | `28577a3f92fc24f438399d3cbffef1aa4890c15a904a489b77bfabcf0b49e09b` |
| run manifest | `9cb9f250f5828fb8b185c12f2a6725ee2dd46aab12f600a29e50f74f03c4479c` |
| route | `362f119922fa9befef080d7b968702545cbf0be72ca9cc129d52ce556eb6fe51` |
| P1 cost attribution | `78ed06455b5b267f9bb8d4d3273e2924426125d6056b80162c9e4baec137b1aa` |
| P2 payload lifecycle | `927b1463051cde6a0161d743e617c5969c78158170037ba7d02f687726333611` |
| P3 state reproduction | `91d9f0a15da1dbfcbe6cbc9e4426117299d0c38b7ff39c036847d7383166fce1` |
| P4 fused pipeline | `501e9552f9e43163fa7008ef933bb82a692bd11e4672fa5dd9c7eb43cfa7b6b4` |
| P5 compressed update | `0035ce68414cf8905e781cada835b2d04e30b66df363bba6f2fb1ac38affc233` |
| P6 system controller | `382cd6294528e1c0bec8d25672345e002161a0ef75ff26b270be873fe32882e6` |
| provenance audit | `11d829f65e8d407f41bae6619e1c006bc14e2affe5e769b5e8f737c557e62607` |

## 12. 最终分析结论

v9.2.72 的真实推进是：

```text
v9.2.71: full-online payload/update payload 已闭合，
          但 payload-bound true-delta step ratio = 4.248161。
v9.2.72: step-cost attribution 闭合；
          candidate payload lifecycle、state extra forward、compressed update payload 均有推进；
          但 fused candidate true-delta path 仍未进入 system envelope。
```

机制判断：

1. H1 成立：cost attribution 没有留下 unknown overhead，unknown fraction = `0.0`。
2. H2 成立：candidate pack 不是当前 terminal blocker，candidate pack q90 = `0.279152 ms`。
3. H3 成立：state reproduction extra non-candidate forward 已从 `902` 降到 `0`。
4. H4 部分成立：compressed W2-only update payload 把 payload memory 降到 `9.287109 MB`，但 system cost 仍不过。
5. H5 未闭合：fused candidate true-delta pipeline 数值正确，但 step ratio q90 = `4.457843`，反而高于 v9.2.71 first run 的 `4.248161`。
6. 当前下一步不是继续 Python-level grouping 或 hash/schema 改造，而是 lower-level fused/async payload-bound candidate forward：需要把 common basis prep 与 candidate delta/logits 变成真正的 CUDA/Triton fused path，并减少 per-step CPU norm / Python loop / synchronization。

最终一句话：

> v9.2.72 真实执行后停在 `R18-PayloadBoundSystemStillExpensive`：payload-bound cost attribution 已闭合，state extra forward 和 update payload 都有推进，但 dominant fused candidate delta/logits path 仍太慢，strict PureKAN functional 仍未成功。

## 13. 追加低层优化尝试：batched CUDA carrier delta/logits path

目标：继续尝试解决 “全量 payload-bound candidate forward 太慢”，不再做 schema/hash 或 Python algebra 小修，而是把同一步的 candidate carriers 合并成 batched CUDA tensor path。

实现改动：

```text
BF2 Python carrier loop:
  每个 carrier 单独调用 _functional_delta_exact_w2；
  timed path 内部有 per-carrier CPU norm / metadata .cpu() 同步。

BF3 batched async CUDA carrier path:
  task_norm 保留为 GPU tensor；
  同一步 1-3 个 candidate carriers 的 signal stack / W2 delta / probe logits 一次性计算；
  timed fused path 不做 CPU norm / metadata .cpu()；
  direct-forward identity check 仍保留在 timed path 外，用于数值确认。
```

这不是 source-measured gap，也不是 formula proxy；仍然 materialize candidate branch logits / true-delta logits / selected-delta logits，并保留 direct-forward identity check。

追加 artifact：

```text
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_batched_cuda_20260513T091500Z/
```

代码检查：

```text
python -m py_compile experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py
```

正式运行：

```bash
python experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py \
  --out-dir results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_batched_cuda_20260513T091500Z \
  --fresh --device auto --data-root data --seed 1314
```

追加 route：

```json
{
  "route": "R18-PayloadBoundSystemStillExpensive",
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "state_reproduction_extra_non_candidate_forward_count": 0,
  "controller_precision": 0.8380281690140845,
  "controller_coverage": 0.03130511463844797,
  "controller_bad_event": 0.02464788732394366,
  "controller_null_rate": 0.13028169014084506,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 4.043819522772543,
  "system_legal_controller_pass": 0,
  "primary_blocker": "payload_bound_true_delta_step_ratio_fail"
}
```

对比结果：

| run | branch id | dominant total ms | branch q90 ms | step q90 | identity max error | system pass |
|---|---|---:|---:|---:|---:|---:|
| first BF2 grouped path | `BF2-FusedCandidateTrueDeltaPipeline` | `1621.190351` | `1.450533` | `4.457843` | `1.907349e-06` | `0` |
| batched CUDA BF3 path | `BF3-BatchedAsyncCudaCandidateTrueDeltaPipeline` | `1134.071934` | `1.377242` | `4.043820` | `1.907349e-06` | `0` |

Batched CUDA summary：

```text
branch_forward_id = BF3-BatchedAsyncCudaCandidateTrueDeltaPipeline
true_delta_candidate_id = TD3-BatchedAsyncCudaTrueDeltaBridge
candidate_count = 2493
exact_w2_candidate_count = 2493
uses_batched_cuda_carrier_pipeline = 1
branch_forward_time_ms_mean = 0.8505323339168487
branch_forward_time_ms_q90 = 1.3772416859865189
dominant_subphase = fused_candidate_delta_logits_ms
dominant_subphase_ms_total = 1134.0719335712492
common_basis_prep_ms_total = 986.305174883455
true_delta_selected_feature_ms_total = 301.651940215379
candidate_pack_time_ms_q90 = 0.29387371614575386
true_delta_time_ms_q90 = 0.13522477820515633
payload_apply_time_ms_q90 = 0.7644058205187321
optimized_identity_check_count = 24
optimized_identity_error_max = 1.9073486328125e-06
step_ratio_q90 = 4.043819522772543
```

追加 no-fake audit：

```text
rows_checked = 73120
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

追加 hash：

| artifact | SHA256 |
|---|---|
| runner | `bf64be36af4dc6dffc56486371b5dbe5fcf13563ce01b8da336972645384b038` |
| route | `1d7198016ed9e73e09888173b33ff47c18aafdf4cfb70de010fe0f521249f889` |
| P1 cost attribution | `a6dd55ac13cac2e2c02a7a8b647aca86bcd54daed35750fb38797e875da35aca` |
| P4 fused pipeline | `be4a32b90818829ad0b9b2259770b8a2522bbd1ae41228b27a9d0733ff842ec0` |
| P6 system controller | `fec09e24db4a72ef727183558bfc4460046dd15b476f4bf7ad59f8e4ac5e9309` |
| provenance audit | `11d829f65e8d407f41bae6619e1c006bc14e2affe5e769b5e8f737c557e62607` |

追加判断：

1. batched CUDA carrier path 是真实推进：dominant fused candidate delta/logits total 从 `1621.190351 ms` 降到 `1134.071934 ms`。
2. 局部 branch q90 也下降：`1.450533 ms -> 1.377242 ms`。
3. 数值一致性保持：24 个 direct-forward identity check 的 max abs error 仍是 `1.907349e-06`。
4. 但 system step ratio 仍为 `4.043820 > 1.50`，因此 official controller 仍不能打开。
5. 当前剩余成本不再是 payload/schema/hash，也不是 state extra forward；主要仍在 common basis prep 与 candidate delta/logits 的真实 GPU compute / synchronization 组合里。

追加结论：

> batched CUDA path 对 “全量 payload-bound candidate forward 太慢” 有真实改善，但没有解决 blocker：step ratio 仍大幅超过 `1.50`。下一步需要进一步把 common basis prep、carrier signal construction、W2 delta 和 probe logits 合成真正的 single/few-kernel Triton/CUDA implementation，并减少 per-step cuBLAS launch / quantile / indexing overhead。

## 14. 追加低层优化尝试：few-kernel CUDA extension 与非 CUDA batched 数值对照

目标：按计划继续尝试 “single/few-kernel Triton/CUDA”，不再继续 Python 层 grouping；同时按要求和非 CUDA/torch batched 路径做数值对照，确认 CUDA extension 没有改变 true-delta 数值语义。

实现改动：

```text
BF4 few-kernel CUDA extension:
  raw_delta_norm_kernel: carrier signal + raw W2 delta + norm2 accumulation；
  scale_delta_kernel: trust cap scaling；
  logits_kernel: vals_probe @ delta_W2 + task_probe_logits。

数值对照：
  对同一批 carrier candidates 同时跑 torch batched reference；
  比较 CUDA extension 输出 logits 与 delta_W2；
  对照结果写入 route 与 P4 summary。
```

追加 artifact：

```text
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_cuda_ext_20260513T103000Z/
```

代码检查：

```text
python -m py_compile experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py
```

正式运行：

```bash
python experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py \
  --out-dir results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_cuda_ext_20260513T103000Z \
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
completed_at = 2026-05-13T09:30:13Z
```

追加 route：

```json
{
  "route": "R18-PayloadBoundSystemStillExpensive",
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "cuda_extension_few_kernel_used": 1,
  "cuda_ext_candidate_count": 2493,
  "cuda_ext_step_count": 1250,
  "cuda_vs_torch_check_count": 24,
  "cuda_vs_torch_logits_error_max": 2.6702880859375e-05,
  "cuda_vs_torch_delta_error_max": 1.280568540096283e-09,
  "controller_precision": 0.8380281690140845,
  "controller_coverage": 0.03130511463844797,
  "controller_bad_event": 0.02464788732394366,
  "controller_null_rate": 0.13028169014084506,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.969948325415854,
  "system_legal_controller_pass": 0,
  "primary_blocker": "payload_bound_true_delta_step_ratio_fail"
}
```

与非 CUDA/torch batched path 对比：

| run | branch id | fused delta/logits total ms | common prep total ms | branch q90 ms | step q90 | numeric check | system pass |
|---|---|---:|---:|---:|---:|---|---:|
| BF3 torch batched | `BF3-BatchedAsyncCudaCandidateTrueDeltaPipeline` | `1134.071934` | `986.305175` | `1.377242` | `4.043820` | identity max error `1.907349e-06` | `0` |
| BF4 few-kernel CUDA ext | `BF4-FewKernelCudaCandidateTrueDeltaPipeline` | `111.270273` | `1026.540387` | `0.866299` | `2.969948` | logits `2.670288e-05`, delta `1.280569e-09` vs torch | `0` |

BF4 summary：

```text
candidate_count = 2493
cuda_ext_candidate_count = 2493
cuda_ext_step_count = 1250
cuda_extension_error = ""
cuda_vs_torch_check_count = 24
cuda_vs_torch_logits_error_max = 2.6702880859375e-05
cuda_vs_torch_delta_error_max = 1.280568540096283e-09
branch_forward_time_ms_mean = 0.4564021899698493
branch_forward_time_ms_q90 = 0.8662990294396877
dominant_subphase = common_basis_prep_ms
common_basis_prep_ms_total = 1026.540386956185
fused_candidate_delta_logits_ms_total = 111.27027263864875
true_delta_selected_feature_ms_total = 302.51913238316774
true_delta_time_ms_q90 = 0.138178002089262
step_ratio_q90 = 2.969948325415854
```

判断：

1. BF4 few-kernel CUDA extension 数值上与非 CUDA/torch batched reference 对齐：delta error 约 `1.28e-09`，logits error 约 `2.67e-05`，未使用 source-measured gap / formula proxy。
2. BF4 真实降低了 candidate delta/logits 局部成本：`1134.071934 ms -> 111.270273 ms`。
3. BF4 也降低了局部 branch q90：`1.377242 ms -> 0.866299 ms`。
4. 但 full system step ratio 仍是 `2.969948 > 1.50`，不能打开 official controller。
5. dominant cost 已从 fused candidate delta/logits 转移到 `common_basis_prep_ms`，说明下一步需要继续把 basis prep / quantile / row feature construction 纳入 fused/async path，而不是只优化 delta/logits kernel。

追加 no-fake audit：

```text
rows_checked = 73120
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

追加 hash：

| artifact | SHA256 |
|---|---|
| plan | `45133dadb41c0b184377f799169f5b833640dc731edb3b2b16e76fbf5c7364dc` |
| runner | `3a0564fd9e17c247d4c639bda1b877381140aa8ecbaa0a3056a97d2d5d8208eb` |
| run manifest | `0cdf2235facebed9ca929244f7ef8285a3b5432b0e6ee65fbeea7fafe074b929` |
| route | `6a656a6912c0b2dc58e06e143c858309ba6baacd48c20db57b9992f36a3e3cd2` |
| P1 cost attribution | `e4a22bb653e17092368ec20137a2d3497209eb91d07f7e7d9959ddf91570b56e` |
| P4 fused pipeline | `5a6be7f282c8b4c0ca23e3ee6c5f989d546f1b7cd7d49c29a75a579032679d45` |
| P6 system controller | `bdc31e60f4acfb36314efcd8e9f7e88370922108d3e61356162e026d28c59296` |
| provenance audit | `11d829f65e8d407f41bae6619e1c006bc14e2affe5e769b5e8f737c557e62607` |

追加结论：

> few-kernel CUDA extension 解决了 candidate delta/logits 局部慢的问题，并且和非 CUDA/torch batched reference 数值一致；但 full payload-bound system cost 仍未进 `1.50` envelope，当前 blocker 转移到 common basis prep / feature construction / step-level synchronization，system-legal controller 仍不能转正。

## 15. 追加低层优化尝试：async combined basis prep + few-kernel CUDA delta/logits

目标：继续尝试把 `basis prep / feature construction / step-level sync` 纳入更底层 fused/async 路径，而不是只优化 candidate delta/logits kernel。

实现改动：

```text
BF4:
  update basis prep 和 probe basis prep 分开；
  common_basis_prep_ms 单独 sync 计时；
  few-kernel CUDA extension 只覆盖 carrier signal / W2 delta / logits。

BF5:
  将 xu/xp 合并为 x_all，一次性计算 h_all / basis_all / logits_all；
  split 得到 update/probe 的 vals/logits；
  common basis prep + CUDA extension 放进同一个 async timed block；
  只在 block 末尾同步一次；
  仍保留 CUDA-vs-torch batched reference 数值对照。
```

这不是 source-measured gap，也不是 formula proxy；candidate payload、branch logits、true-delta logits、selected-delta logits 和 update payload 仍全部 materialize。

追加 artifact：

```text
results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z/
```

代码检查：

```text
python -m py_compile experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py
```

正式运行：

```bash
python experiments/run_v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline.py \
  --out-dir results/real_rerun_20260506/v9272_payload_bound_system_cost_closure_fused_candidate_true_delta_pipeline_async_basis_cuda_ext_20260513T110000Z \
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
disable_fused_common_basis_delta_pipeline = false
completed_at = 2026-05-13T10:27:21Z
```

追加 route：

```json
{
  "route": "R18-PayloadBoundSystemStillExpensive",
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "fused_common_basis_delta_pipeline_used": 1,
  "fused_common_basis_delta_step_count": 1250,
  "fused_common_basis_delta_candidate_count": 2493,
  "cuda_extension_few_kernel_used": 1,
  "cuda_ext_candidate_count": 2493,
  "cuda_ext_step_count": 1250,
  "cuda_vs_torch_check_count": 24,
  "cuda_vs_torch_logits_error_max": 2.6702880859375e-05,
  "cuda_vs_torch_delta_error_max": 8.149072527885437e-10,
  "controller_precision": 0.8380281690140845,
  "controller_coverage": 0.03130511463844797,
  "controller_bad_event": 0.02464788732394366,
  "controller_null_rate": 0.13028169014084506,
  "true_delta_agreement": 1.0,
  "true_delta_step_ratio_q90": 2.713295831053225,
  "system_legal_controller_pass": 0,
  "primary_blocker": "payload_bound_true_delta_step_ratio_fail"
}
```

与 BF4 对比：

| run | branch id | dominant | fused/common total ms | branch q90 ms | step q90 | CUDA-vs-torch logits/delta error | system |
|---|---|---|---:|---:|---:|---|---:|
| BF4 few-kernel CUDA ext | `BF4-FewKernelCudaCandidateTrueDeltaPipeline` | `common_basis_prep_ms` | common `1026.540387` + fused `111.270273` | `0.866299` | `2.969948` | `2.670288e-05` / `1.280569e-09` | `0` |
| BF5 async combined basis+delta | `BF5-AsyncCombinedBasisFewKernelCudaPipeline` | `fused_common_basis_delta_pipeline_ms` | fused common+delta `1086.943672` | `0.766299` | `2.713296` | `2.670288e-05` / `8.149073e-10` | `0` |

BF5 summary：

```text
candidate_count = 2493
fused_common_basis_delta_pipeline_used = 1
fused_common_basis_delta_step_count = 1250
fused_common_basis_delta_candidate_count = 2493
cuda_ext_candidate_count = 2493
cuda_ext_step_count = 1250
branch_forward_time_ms_mean = 0.4359982637677309
branch_forward_time_ms_q90 = 0.7662991993129253
dominant_subphase = fused_common_basis_delta_pipeline_ms
fused_common_basis_delta_pipeline_ms_total = 1086.9436715729535
common_basis_prep_ms_total = 0.0
fused_candidate_delta_logits_ms_total = 0.0
true_delta_selected_feature_ms_total = 323.1187085621059
true_delta_time_ms_q90 = 0.16031181439757347
optimized_identity_check_count = 24
optimized_identity_error_max = 2.574920654296875e-05
cuda_vs_torch_check_count = 24
cuda_vs_torch_logits_error_max = 2.6702880859375e-05
cuda_vs_torch_delta_error_max = 8.149072527885437e-10
step_ratio_q90 = 2.713295831053225
```

判断：

1. BF5 保持数值正确：与 torch batched reference 对照的 logits max error 为 `2.670288e-05`，delta max error 为 `8.149073e-10`。
2. BF5 进一步降低局部 branch q90：`0.866299 ms -> 0.766299 ms`。
3. BF5 进一步降低 system step q90：`2.969948 -> 2.713296`。
4. 但 BF5 仍未进入 `step_ratio_q90 <= 1.50` envelope，不能写成 system-legal controller。
5. 当前 blocker 从 “单独 candidate delta/logits kernel 太慢” 转为 “combined basis/delta pipeline 仍太慢 + true_delta_selected_feature/hash/step-level sync 仍有成本”；需要继续做更深的 basis/logit feature kernel fusion 或减少 full-row payload hashing/selected-feature materialization cost。

追加 no-fake audit：

```text
rows_checked = 73120
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

追加 hash：

| artifact | SHA256 |
|---|---|
| runner | `5ed71d490d807caa8cd89e3f179781a0dc2293b156218459dc37bf62b0ac2ee9` |
| run manifest | `3e64e61fd1850ac3a80fced1ee5aa6185eac4833dc8b172b2d735de13921bb8f` |
| route | `a80cb8f9b50a4a21e434e87fbbe4ebd47caaef37a2bd91f7bbfe2e71c622ee40` |
| P1 cost attribution | `cdebde56cc41c434421903b7015c354d43358a0cd50108bc44924fcef56505f7` |
| P4 fused pipeline | `fdfdec507d9b417036a0b606c1f849a77e8d2f55e01c788f7a0611fd18a83584` |
| P6 system controller | `08a52bdf3d7e7358252acb1e9f4ca96fc6a360f2a73ab929f95979a41bb3c5f1` |
| provenance audit | `11d829f65e8d407f41bae6619e1c006bc14e2affe5e769b5e8f737c557e62607` |

追加结论：

> BF5 async combined basis+delta path 继续推进了 “basis prep / feature construction / step-level sync” 的低层化：step q90 从 BF4 的 `2.969948` 降到 `2.713296`，并保持 CUDA-vs-torch 数值一致；但仍未达到 `1.50` system envelope，因此 v9.2.72 仍停在 `R18-PayloadBoundSystemStillExpensive`，strict PureKAN functional 仍未成功。
