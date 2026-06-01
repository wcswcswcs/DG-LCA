# DG-KAN v9.2.77 Quantile-Tail BasisNorm 与 Static-Bucket Bridge Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.2.77_QuantileTailBasisNorm_StaticBucketBridgeClosure_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 quantile-tail 局部优化、persistent workspace materialization 或 gate-blocked downstream 写成 official system pass。

## 0. 最新结论

```text
route = R17-BasisNormRuntimeStillInsufficient
base_candidate = LQ-t2-h256
success_v9277_strict_purekan_functional = False
success_v9277_full_functional = False
success_v9277_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_first_20260513T163000Z/
```

说明：本轮没有保存单独 stdout `run.log`。可审计日志是上面 artifact 目录中的 `run_manifest.json`、`route_decision.json`、各阶段 `p*.csv`、`*_trace_v9277.csv`、`failure_table.csv`、`artifact_hashes.csv` 和 `v9277_provenance_audit.csv`。

核心结论：

1. P0 复现 v9.2.76 boundary：source route = `R15-BasisNormDominantUnfixed`，payload binding pass = `1`，candidate/update payload missing count 全部为 `0`，system legal controller pass = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；实际参数保持 MNIST/Fashion-MNIST/KMNIST、seeds `0..7`、attribution steps per dataset = `8`、quantile-tail warmup steps = `1`。
3. P1 quantile-tail internal attribution 过 gate：unknown fraction = `0.001664`，dominant subcomponent = `threshold_compute_time_ms`，均值 `0.220606 ms`，component ratio = `0.381040`。
4. P1 quantile-tail 总均值 `0.578957 ms`；其他主要 subcomponents 为 `tail_dispatch_overhead_time_ms = 0.110434`、`logsumexp_time_ms = 0.090767`、`tail_context_lookup_time_ms = 0.038211`。
5. P2 kthvalue exact-tail runtime 过 gate：`quantile_tail_time_before = 0.578957 ms`，`quantile_tail_time_after = 0.194465 ms`，reduction = `0.664111`，audit agreement = `1.0`。
6. P2 数值对照通过：CUDA/torch-derived path 的 logits error = `0.0`，delta error = `0.0`；uses source measured gap / formula proxy / projection 全部为 `0`。
7. P3 basis_norm runtime integration 未过：`basis_norm_time_before = 0.535139 ms`，`basis_norm_time_after = 0.348024 ms`，reduction = `0.349657 < 0.50`，且 `basis_norm_bucketed = 0`、`basis_norm_fused_kernel_used = 0`。
8. P4 static bucket / persistent workspace 仍未 integrated：runtime bucket 和 persistent workspace 已 materialized，allocation count 从 `2493` 到 `1`，但 `basis_norm_bucketed = 0`、`W2_delta_bucketed = 0`、`bridge_score_bucketed = 0`，kernel/sync reduction 均为 `0.0`。
9. P5 single-pass quantile-tail/basisnorm/delta/bridge 未闭合：`quantile_tail_inside_kernel = 0`，`bridge_score_inside_kernel = 0`，`basis_norm_delta_bridge_single_pass = 0`。
10. P6/P7 不能 official：`materialized_runtime_path = 0`，`diagnostic_derived_from_measured_components = 1`，system controller `official_eligible = 0`，step ratio q90 仍为 source boundary 的 `2.713296 > 1.50`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9277_quantile_tail_basis_norm_static_bucket_bridge_closure.py` | v9.2.77 runner；复现 v9.2.76 boundary，执行 quantile-tail 内部分解、kthvalue exact-tail runtime、basis_norm integration、static bucket/persistent workspace、single-pass bridge 与 system controller gate |

代码检查：

```text
python -m py_compile experiments/run_v9277_quantile_tail_basis_norm_static_bucket_bridge_closure.py
```

正式运行：

```bash
python experiments/run_v9277_quantile_tail_basis_norm_static_bucket_bridge_closure.py \
  --out-dir results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_first_20260513T163000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
attribution_steps_per_dataset = 8
quantile_tail_warmup_steps = 1
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-13T14:08:37Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R17-BasisNormRuntimeStillInsufficient",
  "base_candidate": "LQ-t2-h256",
  "v9276_boundary_pass": 1,
  "payload_binding_contract_pass": 1,
  "candidate_tensor_payload_missing_count": 0,
  "candidate_branch_logits_missing_count": 0,
  "candidate_true_delta_logits_missing_count": 0,
  "functional_update_payload_missing_count": 0,
  "quantile_tail_internal_attribution_pass": 1,
  "dominant_quantile_tail_subcomponent": "threshold_compute_time_ms",
  "quantile_tail_unknown_fraction": 0.001663726558793046,
  "quantile_tail_runtime_pass": 1,
  "quantile_tail_strategy": "kthvalue_exact_tail_threshold",
  "quantile_tail_time_before": 0.5789566591071585,
  "quantile_tail_time_after": 0.1944652758538723,
  "quantile_tail_time_reduction": 0.664110822813977,
  "basis_norm_runtime_pass": 0,
  "basis_norm_time_before": 0.5351388632940749,
  "basis_norm_time_after": 0.3480239538475871,
  "basis_norm_time_reduction": 0.3496567382430278,
  "static_bucket_workspace_pass": 0,
  "basis_norm_bucketed": 0,
  "W2_delta_bucketed": 0,
  "bridge_score_inside_kernel": 0,
  "integrated_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "official_eligible": 0,
  "controller_step_ratio_q90": 2.713295831053225,
  "primary_blocker": "basis_norm_runtime_not_integrated",
  "next_required_implementation": "integrate_quantile_tail_with_bucketed_basis_norm"
}
```

判断：v9.2.77 确实解决了 quantile-tail 局部 runtime，但没有把它接入 bucketed/fused basis_norm runtime，因此 route 从 v9.2.76 的 `R15` 推进到 `R17`，仍不能打开 system controller。

## 3. P1 quantile-tail internal attribution

Artifacts：

```text
p1_quantile_tail_internal_attribution.csv
quantile_tail_internal_trace_v9277.csv
```

Summary：

```text
quantile_tail_attribution_id = QTA1-QuantileTailSubphaseTimer
sample_step_count = 24
quantile_tail_internal_attribution_pass = 1
quantile_tail_total_time_ms = 0.5789566591071585
quantile_tail_unknown_fraction = 0.001663726558793046
dominant_quantile_tail_subcomponent = threshold_compute_time_ms
dominant_quantile_tail_component_ratio = 0.3810398194670038
```

Subphase：

| subcomponent | mean ms |
|---|---:|
| `threshold_compute_time_ms` | `0.220606` |
| `tail_dispatch_overhead_time_ms` | `0.110434` |
| `logsumexp_time_ms` | `0.090767` |
| `tail_context_lookup_time_ms` | `0.038211` |
| `tail_mask_time_ms` | `0.028124` |
| `topk_time_ms` | `0.026367` |
| `tail_writeback_time_ms` | `0.025812` |
| `tail_kernel_launch_time_ms` | `0.020200` |
| `tail_temp_allocation_time_ms` | `0.014501` |
| `tail_sync_time_ms` | `0.003936` |
| `sort_time_ms` | `0.000000` |

判断：H1 成立。quantile-tail 内部成本不再是黑箱；dominant 是 threshold/quantile threshold compute，而不是 topk、sync 或 writeback。

## 4. P2 quantile-tail exact-tail runtime

Artifacts：

```text
p2_quantile_tail_cache_exact_tail_fusion_runtime.csv
quantile_tail_runtime_trace_v9277.csv
```

Best summary：

```text
quantile_tail_runtime_id = QTR3-KthValueExactTailRuntime
quantile_tail_strategy = kthvalue_exact_tail_threshold
cache_used = 0
cache_hit_rate = 0.0
fused_topk_tail_kernel_used = 0
quantile_tail_time_before = 0.5789566591071585
quantile_tail_time_after = 0.1944652758538723
quantile_tail_time_reduction = 0.664110822813977
audit_agreement = 1.0
cuda_vs_torch_check_count = 24
cuda_vs_torch_logits_error_max = 0.0
cuda_vs_torch_delta_error_max = 0.0
quantile_tail_runtime_pass = 1
```

判断：H2 成立。kthvalue exact-tail path 是本轮真实推进：它没有用 source-measured gap / formula proxy / projection，并且把 quantile-tail 局部成本降低超过 `50%`。

## 5. P3 basis_norm runtime integration

Artifacts：

```text
p3_basis_norm_runtime_integration.csv
basis_norm_runtime_trace_v9277.csv
```

Summary：

```text
basis_norm_runtime_id = BNR1-KthTailIntegratedBasisNorm
basis_norm_strategy = kth_tail_integrated_no_bucket_no_fused_kernel
basis_norm_time_before = 0.5351388632940749
basis_norm_time_after = 0.3480239538475871
basis_norm_time_reduction = 0.3496567382430278
audit_agreement = 1.0
basis_norm_bucketed = 0
basis_norm_fused_kernel_used = 0
basis_norm_runtime_pass = 0
```

判断：H3 未闭合。quantile-tail 子项过了，但 basis_norm 整体只下降 `34.97%`，没有达到 `>=50%`，也没有 materialize bucketed/fused basis_norm kernel。因此不能把 P2 的局部 pass 扩大成 basis_norm runtime pass。

## 6. P4 static bucket / persistent workspace

Artifacts：

```text
p4_static_bucket_persistent_workspace_integrated_runtime.csv
static_bucket_workspace_runtime_trace_v9277.csv
```

Summary：

```text
bucket_workspace_id = SB5-PersistentWorkspaceIntegratedV2NotConnected
runtime_bucket_used = 1
persistent_workspace_used = 1
workspace_memory_MB = 0.2988319396972656
allocation_count_before = 2493
allocation_count_after = 1
allocation_count_reduction = 0.9995988768551946
kernel_count_before/after = 3750 / 3750
sync_count_before/after = 1250 / 1250
kernel_count_reduction = 0.0
sync_count_reduction = 0.0
avg_candidates_per_kernel_after = 0.6648
basis_norm_bucketed = 0
W2_delta_bucketed = 0
bridge_score_bucketed = 0
static_bucket_workspace_pass = 0
```

Bucket sizes：

```text
{"1": 7260, "2": 365, "4": 439}
```

判断：persistent workspace 仍是真实 CUDA tensor materialization，并且 allocation reduction 很高；但它没有接入 basis_norm / W2_delta / bridge score runtime，kernel/sync 没下降，avg candidates per kernel 仍只有 `0.6648`，不能写成 static bucket closure。

## 7. P5/P6/P7 runtime 与 system boundary

Artifacts：

```text
p5_single_pass_quantile_tail_basisnorm_delta_bridge_runtime.csv
p6_integrated_materialized_runtime_candidates.csv
p7_system_legal_exact_signal_controller_v9.csv
```

P5 summary：

```text
basis_delta_bridge_runtime_id = BDB1-KthTailBasisNormReferenceDeltaBridgeNotSinglePass
quantile_tail_inside_kernel = 0
basis_norm_delta_bridge_single_pass = 0
bridge_score_inside_kernel = 0
accept_bit_inside_kernel = 1
agreement_reference_accept = 1.0
cuda_vs_torch_logits_error_max = 0.0
cuda_vs_torch_delta_error_max = 0.0
basis_delta_system_candidate_pass = 0
```

P6/P7 boundary：

```text
system_candidate_id = SYS7-KthTailNoBucketNoBridgeRuntime
materialized_runtime_path = 0
diagnostic_derived_from_measured_components = 1
integrated_runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = P6_integrated_runtime_or_step_ratio_failed
step_ratio_q90 = 2.713295831053225
memory_ratio = 1.0
```

Decision metrics still held：

```text
precision_heldout = 0.8380281690140845
coverage_heldout = 0.03130511463844797
bad_event_heldout = 0.02464788732394366
null_rate_heldout = 0.13028169014084506
precision_lcb = 0.7907157243773478
bad_event_ucb = 0.04999458813129621
accepted_signal_strata_count = 15
accepted_family_count = 65
max_family_share = 0.09507042253521127
max_stratum_share = 0.15140845070422534
```

判断：decision/support/payload 都没有 regression，但 runtime path 仍不是 materialized integrated system path。P7 只能保持 `not_run`，不能因为 P2 局部 runtime pass 而打开 official controller。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p8_leave_dataset_and_stratum_out.csv` | `P3_basis_norm_runtime_failed` |
| `p9_official_paired_replay.csv` | same |
| `p10_short_run_functional_validation.csv` | same |
| `p11_full_run_robustness_strong_baseline.csv` | same |

没有把 quantile-tail runtime pass、basis_norm diagnostic、persistent workspace materialization 或 decision metrics 写成 LDO/LSO、paired replay、short-run 或 full-run success。

## 9. No-fake audit

```text
rows_checked = 167
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
payload_binding_contract_pass = 1
candidate_tensor_payload_missing_count = 0
candidate_branch_logits_missing_count = 0
candidate_true_delta_logits_missing_count = 0
functional_update_payload_missing_count = 0
quantile_tail_internal_attribution_pass = 1
quantile_tail_runtime_pass = 1
basis_norm_runtime_pass = 0
static_bucket_workspace_used = 1
basis_norm_bucketed = 0
W2_delta_bucketed = 0
bridge_score_inside_kernel = 0
basis_norm_delta_bridge_single_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
projection_used_for_official = 0
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `bf38ebdf5af55a2348a7ede303d5b6a9c03d4b8fdd460b91899bd8faeca726ee` |
| runner | `cb808945e6c660ced10efa08e074146d92ab0616ab47c8331fe1c85589b7e582` |
| run manifest | `6d9f6a371d084876702ec83ca829c8c9dc29cdaf60977b5b0181bc69c601795e` |
| route | `070b1181b38ca27f0cd6d6b43746359cb3344c7a5df5e401ea54318c17f82567` |
| P0 boundary | `35ac996a495bd3220a4733fcce200494a1e84596aedf280c4698784456543d1e` |
| P1 quantile tail | `1e53c243f8f14a0ec06ac53bbe0862c1b998189da685764f1013aede670e1088` |
| P2 quantile tail runtime | `0b4c0d99d15c2407ef4c762815d89bc8831d6946970e025bf2736cc365ca3257` |
| P3 basis norm runtime | `19d0d0c20cd9bb4dfdc02e62f175bcbc540731c779fabbfd9007856585c29044` |
| P4 static bucket | `eed5c56751f4958c150835c9315bf08ae6a8221996da861f9dec55e32ae45b95` |
| P5 single pass bridge | `b662162fae7e6bb0632f04438e9ebfb193cade5cc5b4a29e794b27e3e2d9a620` |
| P6 integrated runtime | `e1c7e04b62905118c2c8f924ec8585368628ab6603a9a947962669860891b201` |
| P7 system controller | `669257988fb29026f69167763f578dc4aeec5318867ea425e56c01d7fca50d66` |
| failure table | `9f7e26a66037fb71fd12a0cfc604c87658a96089509f41ac84e0bfcb78000063` |
| provenance audit | `00d61e60fca0194c4f7e915609cf90e0ba6bb0a9122aa4200fd53bfbb5c38e34` |

## 11. 最终分析结论

v9.2.77 的真实推进是：

```text
v9.2.76: basis_norm dominant 已定位到 quantile/tail compute，
          但 no-clone/logsumexp/top2 只降低 27.12%。
v9.2.77: quantile-tail 内部 attribution 闭合；
          kthvalue exact-tail runtime 过 gate，局部降低 66.41%；
          但 basis_norm 整体 integration、static bucket、single-pass bridge 仍未闭合。
```

机制判断：

1. H1 成立：quantile-tail 内部 attribution 闭合，dominant 是 `threshold_compute_time_ms`，unknown fraction 只有 `0.001664`。
2. H2 成立：kthvalue exact-tail runtime 数值正确且过 runtime reduction gate，`0.578957 -> 0.194465 ms`。
3. H3 未成立：basis_norm 整体只降到 `0.348024 ms`，reduction `34.97%`，低于 `50%`；也没有 bucketed/fused kernel。
4. H4 未成立：persistent workspace materialized，但没有接入 bucketed basis_norm / W2_delta / bridge score，kernel/sync reduction 为 `0.0`。
5. H5 未成立：single-pass quantile-tail/basisnorm/delta/bridge 没有 materialize，bridge score 仍不在 kernel 内。
6. H6 未打开：P7 `official_eligible = 0`，因此 LDO/LSO、paired replay、short/full validation 全部 gate-blocked。

最终一句话：

> v9.2.77 真实执行后停在 `R17-BasisNormRuntimeStillInsufficient`：quantile-tail 局部 runtime 已真实过 gate，但 basis_norm 整体和 static-bucket/single-pass bridge 仍未 materialize 到 official runtime path，system-legal controller 仍不能转正。

## 12. 追加尝试：compiled single-pass basisnorm/delta/bridge

针对“quantile-tail 局部已过，但没接进 bucketed/fused basis_norm + single-pass bridge runtime”的困难，本轮继续做了追加尝试。

遇到的具体困难：

1. `QTR3-KthValueExactTailRuntime` 只解决 quantile-tail threshold/tail 子项；basis_norm 还包含 scale/shift、row norm、task norm、W2 delta、probe logits、bridge score 等路径。
2. P4 的 static bucket / persistent workspace 在 v9.2.77 仍只是 materialized workspace；`basis_norm_bucketed = 0`，`W2_delta_bucketed = 0`，`bridge_score_bucketed = 0`。
3. P5 的 bridge score 没有进入 kernel；所以即使 P2 局部过 gate，P6 仍是 `materialized_runtime_path = 0`。
4. 真正要解决，需要把 kth-tail、basis_norm、W2 delta、probe logits、bridge score 放到同一个真实 runtime path 中，并且再接入 static bucket/workspace。

追加实现：

```text
SP2-TorchCompileKthTailBasisNormDeltaBridge:
  kth-tail threshold
  basis_norm context
  W2 delta stack
  probe logits
  bridge score
  accept bit
```

这条路径使用真实 CUDA tensor，保留 eager-vs-compiled 数值对照；没有使用 fake data、proxy row、source-measured gap、formula proxy 或 CPU offload。

追加 artifact：

```text
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_compiled_single_pass_20260513T171500Z/
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_compiled_single_pass_q90_20260513T173000Z/
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_compiled_single_pass_warmup3_20260513T174500Z/
```

最佳追加结果来自 warmup=3 run：

```text
route = R17-BasisNormRuntimeStillInsufficient
compiled_single_pass_used = 1
compiled_single_pass_diagnostic_pass = 1
eager_single_pass_time_ms_mean = 1.4305428873437147
eager_single_pass_time_ms_q90 = 1.4389343559741974
compiled_single_pass_time_ms_mean = 1.1533999737973015
compiled_single_pass_time_ms_q90 = 1.1647101491689682
compiled_single_pass_time_mean_reduction = 0.19373268428255405
compiled_single_pass_time_q90_reduction = 0.19057450791045433
cuda_vs_torch_logits_error_max = 4.76837158203125e-07
cuda_vs_torch_delta_error_max = 3.4924596548080444e-10
bridge_score_error_max = 2.9802322387695312e-08
accept_disagreement_count = 0
tail_disagreement_count = 0
```

同一追加 run 的主 gate 仍未打开：

```text
quantile_tail_runtime_pass = 1
basis_norm_runtime_pass = 0
static_bucket_workspace_pass = 0
integrated_runtime_pass = 0
system_legal_controller_pass = 0
controller_step_ratio_q90 = 2.713295831053225
```

追加 no-fake audit：

```text
rows_checked = 217
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
| runner | `85266a88d454dba1ad008686d6a4110989e8acba7566b5dd307a587b9cccfc20` |
| run manifest | `71c3636fb3d82ed2555adcaf99ab3079edae45093fed6fd901c548c693f25d55` |
| route | `7e4d0a015032423f98d9f07f191be5ce710724a4735139e0acecc20335a9a38d` |
| P2 quantile tail runtime | `b655cffd4479fbe7817e5fc631375bc74123a67cc392e16b26037e7d1d3681f1` |
| P3 basis norm runtime | `ffe9d03eeaba4c8182118c518ac9f0f7e5dc8ec563e91836746e061b4b0e00a5` |
| P4 static bucket | `0b0fcccffab430c95c4529ead9550ed079a6cb0156f36060f4fe78e0d6aab882` |
| P5 single pass bridge | `461dae740e8982216038cad6ada82b580850c86f10647f0a7db0080aced2d9ed` |
| P6 integrated runtime | `bac716c2e9718a3d9e19a9a007328b06dd531ae66d4bf8b940bb7cddf6832e67` |
| P12 compiled single pass | `a8093ec239607278af213b19a8a320f990c4f7779f3f7c15a7ebd96157bc9366` |
| provenance audit | `b37c63507fb5ab7db915065c264e3e443dfc6ec08c7ef957377a5a9d364b520c` |

中间失败记录：

1. `compiled_single_pass_20260513T171500Z` 数值正确，compiled q90 低于 eager q90，但有一次真实 first-use outlier，使 mean reduction 为负；没有写成 diagnostic pass。
2. `compiled_single_pass_q90_20260513T173000Z` 按 q90 评价 compiled path 时，P2 quantile-tail runtime 在同轮 timing 中回退，route 变成 `R16-QuantileTailDominantUnfixed`；没有拿它转正。
3. `compiled_single_pass_warmup3_20260513T174500Z` 消除了 first-use outlier，compiled single-pass diagnostic pass = `1`，但仍不是 static bucket/full integrated system path。

追加判断：

1. 能解决一部分：compiled single-pass path 证明 kth-tail + basisnorm + W2 delta + logits + bridge score 可以在局部 q90 上再降约 `19.06%`，且数值保持正确。
2. 还没解决 official blocker：它不是 static-bucket workspace integrated kernel，`basis_norm_bucketed/W2_delta_bucketed/bridge_score_bucketed` 仍为 `0`，P6 仍 `materialized_runtime_path = 0`。
3. 不能轻易转正：controller 决策指标仍好，但 step ratio q90 仍沿用未闭合边界的 `2.713296 > 1.50`，LDO/paired replay/short-run 必须继续 gate-blocked。
4. 下一步真正要做的是把 SP2 的 single-pass 图进一步降到 static bucket / persistent workspace runtime：固定 bucket size，复用 workspace，把 basis_norm、delta、bridge score 的 kernel/sync/allocation 计数真实降下来，而不是只靠 `torch.compile` 局部融合。

追加一句话：

> 这轮追加尝试没有放弃：`torch.compile` single-pass 证明方向有局部收益，但它还不是 official 所需的 bucketed/fused integrated runtime；v9.2.77 仍停在 `R17-BasisNormRuntimeStillInsufficient`。

## 13. 追加尝试：static-bucket + persistent workspace single-pass runtime

继续针对“必须把 SP2 single-pass 图降到 static bucket + persistent workspace runtime，而不是只靠 `torch.compile` 局部融合”的要求，本轮追加实现了 workspace-bound single-pass path。

追加实现：

```text
SP3-StaticBucketPersistentCompiledSinglePass:
  static bucket sizes = 1,2,4
  persistent workspace:
    logits: bucket x 3 x batch x class
    delta: bucket x 3 x hidden x class
    bridge: bucket x 3
    accept: bucket x 3
    tail: bucket x batch
  runtime:
    compiled kth-tail/basis_norm/W2-delta/probe-logits/bridge/accept
    then write logits/delta/bridge/accept/tail into persistent workspace slot
```

这条路径把 basis_norm/W2-delta/bridge 的输出实际绑定到 static bucket workspace；同时保留 eager-vs-workspace path 数值对照。没有使用 fake data、proxy row、source-measured gap、formula proxy 或 CPU offload。

追加 artifact：

```text
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_static_bucket_single_pass_20260513T180000Z/
```

追加 route 仍为：

```text
route = R17-BasisNormRuntimeStillInsufficient
system_legal_controller_pass = 0
official_eligible = 0
controller_step_ratio_q90 = 2.713295831053225
```

P13 summary：

```text
runtime_id = SP3-StaticBucketPersistentCompiledSinglePass
static_bucket_single_pass_attempted = 1
static_bucket_single_pass_diagnostic_pass = 1
compiled_single_pass_used = 1
bucket_sizes = {"1": 11, "2": 6, "4": 7}
runtime_bucket_used = 1
persistent_workspace_used = 1
workspace_memory_MB = 0.25687503814697266
workspace_allocation_time_ms = 0.09289989247918129
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_bucketed = 1
bridge_score_inside_workspace_path = 1
```

Runtime 对比：

```text
eager_single_pass_time_ms_mean = 1.6802030343872805
eager_single_pass_time_ms_q90 = 1.92564120516181
bucket_workspace_single_pass_time_ms_mean = 1.2712367461062968
bucket_workspace_single_pass_time_ms_q90 = 1.3291193172335625
bucket_workspace_time_mean_reduction = 0.24340289828730216
bucket_workspace_time_q90_reduction = 0.3097783150512311
workspace_write_time_ms_mean = 0.0
```

数值对照：

```text
cuda_vs_torch_check_count = 24
cuda_vs_torch_logits_error_max = 4.76837158203125e-07
cuda_vs_torch_delta_error_max = 3.4924596548080444e-10
bridge_score_error_max = 2.9802322387695312e-08
accept_disagreement_count = 0
tail_disagreement_count = 0
```

但 full system gate 仍未打开：

```text
kernel_count_before/after = 3750 / 3750
sync_count_before/after = 1250 / 1250
allocation_count_before/after = 2493 / 1
kernel_count_reduction = 0.0
sync_count_reduction = 0.0
allocation_count_reduction = 0.9995988768551946
avg_candidates_per_kernel_after = 0.6648
static_bucket_workspace_pass = 0
integrated_runtime_pass = 0
system_legal_controller_pass = 0
```

追加 no-fake audit：

```text
rows_checked = 219
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
| runner | `9898bec9206b98e927803ff641654634879ffda421952c5ef84f797a93c35374` |
| run manifest | `3171630b7d2dc2b79ca33f904cbda5cf2eefe6da71c679327ceee9e51de93e65` |
| route | `87b51b84adcbc8749eeb4e74e70e88f624f8b8064d20cf1b56d6389c26f6136b` |
| P2 quantile tail runtime | `1c7e91b13ee97f5730cc5d7b0bb608da99f6e2ca122b7f7052aeaf363a23ee45` |
| P3 basis norm runtime | `97067fef7619815b2a71be86fdc1a4498ebe1ac28877bb0b9b7a26aaaa2636c5` |
| P4 static bucket | `b86c8162432bcb32c1f24b8a9a24cbc3cbe103eb5725004acde971312360670f` |
| P5 single pass bridge | `c1e7a52714d7660edea8f691bc0f3e31de6494cd367dc62fbb89cf8cf993f4cb` |
| P6 integrated runtime | `2571375ff7fb9397c6b3c8adb9c074f5df74306ae74ea47819014b9886f6228f` |
| P13 static bucket single pass | `97d3421c2c737d39e4103c3065f3d9c71744a51d3c0e1ad5a580af9101d6bb96` |
| provenance audit | `cf55238a3bfb610614ce0c154829a71b326fc9c8bb60e5d1efa7161d8861042c` |

追加判断：

1. 这次确实比上一轮更进一步：SP3 不只是 `torch.compile` 局部融合，它把 single-pass 输出真实写进 static bucket persistent workspace，workspace memory 和 allocation time 均已落盘。
2. SP3 有真实局部收益：workspace-bound q90 从 eager 的 `1.925641 ms` 降到 `1.329119 ms`，q90 reduction = `30.98%`，数值一致。
3. 但 official blocker 没闭合：kernel count 和 sync count 完全没降，avg candidates per kernel 仍为 `0.6648`，没有达到计划的 static-bucket runtime gate。
4. `allocation_count_after = 1` 说明 persistent workspace 起效，但单靠 workspace allocation reduction 不足以转正；真正缺的是 bucket 内多 candidate/多 step 的 fused kernel 或 CUDA graph capture，让 kernel/sync 数量下降。
5. 因此 P13 只能记为 diagnostic pass，不能写成 `static_bucket_workspace_pass = 1` 或 `system_legal_controller_pass = 1`。

追加一句话：

> SP3 已把 SP2 single-pass 图接进 static bucket persistent workspace，并获得真实局部 q90 改善；但 kernel/sync 没有下降，system step ratio 仍 `2.713296 > 1.50`，v9.2.77 仍不能转正。

## 14. 追加尝试：CUDA graph static-bucket replay

目标：继续尝试把 static bucket + persistent workspace runtime 往更低层推进，减少 Python launch / sync；不是继续 schema/hash 或单纯 `torch.compile` 局部融合。

实现尝试：

```text
SP4-CudaGraphStaticBucketPersistentSinglePass:
  fixed-shape static input buffers
  CUDAGraph capture of kth-tail / basis_norm / W2-delta / probe logits / bridge / accept
  replay output written to persistent bucket workspace
```

追加 artifact：

```text
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_cuda_graph_static_bucket_relaxed_20260513T190000Z/
```

结果：

```text
cuda_graph_static_bucket_attempted = 1
cuda_graph_capture_pass = 0
cuda_graph_static_bucket_diagnostic_pass = 0
cuda_graph_error = CUDA error: operation failed due to a previous error during capture
graph_cpu_launch_count_reduction = 0.0
kernel_count_reduction = 0.0
sync_count_reduction = 0.0
system_legal_controller_pass = 0
```

同轮主 route 仍为：

```text
route = R17-BasisNormRuntimeStillInsufficient
controller_step_ratio_q90 = 2.713295831053225
official_eligible = 0
```

判断：完整 SP4 graph capture 没有成功，因此不能把它写成 static bucket runtime pass。失败后没有用 launch reduction 派生字段冒充收益；`graph_cpu_launch_count_reduction` 已修正为 `0.0`。

No-fake audit：

```text
rows_checked = 221
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Hash：

| artifact | SHA256 |
|---|---|
| runner | `27eb6249821a73ca0eda56c417c7fe4995a88515d7249f7aa2adb5d53c50eff9` |
| run manifest | `f6588b2a5cbe0a84ad695908becd686dc291f7f59a26365895f004aa07238617` |
| route | `ed0741530412fd16987554e58460b653ee84e258bc02fbf5990b04d8d1dead61` |
| P14 CUDA graph static bucket | `47d8bb462a79da577f9161221298ef08435694981d83759b40e67d618ef851ed` |
| provenance audit | `16b478d0422189f76940ce8d517625864248bcf52c50ec2f1ec5cbf980106337` |

## 15. 追加隔离：precomputed-tail delta/bridge CUDA graph

为定位 P14 的困难，本轮又做了隔离实验：先用真实路径计算 `ce/pred/margin/tail`，然后只 capture W2-delta / logits / bridge / accept。为了排除动态 boolean scatter 的影响，delta 构造改成等价 dense one-hot 形式。

追加实现：

```text
SP5-CudaGraphPrecomputedTailDeltaBridgeIsolation:
  precomputed real ce / pred / margin / tail
  dense one-hot delta construction
  CUDAGraph capture only for delta/logits/bridge/accept
```

追加 artifact：

```text
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_cuda_graph_delta_bridge_dense_20260513T200000Z/
```

结果：

```text
cuda_graph_delta_bridge_isolation_attempted = 1
precomputed_tail_used = 1
cuda_graph_capture_pass = 0
cuda_graph_delta_bridge_diagnostic_pass = 0
cuda_graph_error = CUDA error: operation failed due to a previous error during capture
eager_delta_bridge_time_ms_mean = 0.66017957093815
eager_delta_bridge_time_ms_q90 = 0.7662680000066757
system_legal_controller_pass = 0
```

同轮主 route 仍为：

```text
route = R17-BasisNormRuntimeStillInsufficient
controller_step_ratio_q90 = 2.713295831053225
official_eligible = 0
```

判断：

1. capture 失败不是只由 kthvalue / quantile-tail 引起；即使 tail 预先真实计算、delta 改成 dense one-hot，PyTorch CUDAGraph 仍失败。
2. 当前困难不是“还有一个小 Python grouping 能修掉”，而是 PyTorch op 组合本身无法稳定 capture 成计划需要的 static-bucket graph runtime。
3. 因此下一步需要 native CUDA/Triton fused kernel，把 basis_norm / dense delta / bridge score / accept bit 做成真正的 bucket kernel；继续在 PyTorch CUDAGraph 外壳上尝试已没有可靠推进依据。
4. 不能把 P15 写成 official，因为 `precomputed_tail_used = 1`，它本身就是隔离诊断，不是 full single-pass runtime。

No-fake audit：

```text
rows_checked = 223
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Hash：

| artifact | SHA256 |
|---|---|
| runner | `1c8dcaee4d23b9b741773d1fba105f34388616a9cbde11922e0c88b6ac3e9c43` |
| run manifest | `cd4078ecb458a28d8f9eab9103a320c0b7eb683ef1e18801f9f2d7dfe4232f9c` |
| route | `ab3e29ac8a14e096d9f4be7b37c05d0b88e1ad4f8c12db9b73109e45fb2f4e2e` |
| P15 CUDA graph delta/bridge isolation | `e9faa33a9b905c223ff20a648eda90eb5c09b67b770ea82491946ba0db153f8d` |
| provenance audit | `54a508bd8394a5444466876767d247aa19ad33b3f5edb334f300f89fe11691ec` |

追加结论：

> v9.2.77 继续尝试到 CUDA graph static-bucket replay 与 precomputed-tail delta/bridge isolation：两条 CUDAGraph 路线都真实失败，且没有 fake/proxy/offload。当前可确认的困难是 PyTorch graph/capture 层不能承载计划要求的 bucketed single-pass runtime；下一步若继续推进，需要写 native CUDA/Triton bucket kernel，而不是继续依赖 `torch.compile` 或 CUDAGraph 外壳。

## 16. 追加尝试：native CUDA bucket basis_norm / dense-delta / bridge kernel

针对“为什么不直接执行 native CUDA/Triton bucket kernel”的要求，本轮继续实现并运行了 native CUDA extension。它不再依赖 `torch.compile` 或 CUDAGraph，而是用 `torch.utils.cpp_extension.load_inline` 编译 CUDA kernels。

实现内容：

```text
BK1-NativeCudaBasisNormDenseDeltaBridgeBucketKernel:
  update logits kernel
  CE / margin / pred kernel
  exact kth-tail threshold kernel
  row_abs / task_norm reduction kernels
  dense W2-delta kernel
  delta scale kernel
  probe logits kernel
  bridge score / accept kernel
```

追加 artifact：

```text
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_native_cuda_bucket_kernel_20260513T203000Z/
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_native_cuda_bucket_kernel_v2_20260513T210000Z/
results/real_rerun_20260506/v9277_quantile_tail_basis_norm_static_bucket_bridge_closure_native_cuda_bucket_kernel_v3_20260513T213000Z/
```

第一轮 native CUDA 结果：

```text
route = R17-BasisNormRuntimeStillInsufficient
native_cuda_bucket_kernel_attempted = 1
native_cuda_bucket_kernel_used = 1
basis_norm_bucketed = 1
W2_delta_bucketed = 1
bridge_score_inside_kernel = 1
accept_bit_inside_kernel = 1
eager_single_pass_time_ms_q90 = 0.882553867995739
native_cuda_bucket_kernel_time_ms_q90 = 0.5997372791171074
native_cuda_bucket_kernel_time_q90_reduction = 0.32045249489518646
kernel_count_after = 216
sync_count_after = 24
kernel_count_reduction = 0.9424
sync_count_reduction = 0.9808
```

数值对照：

```text
cuda_vs_torch_logits_error_max = 3.337860107421875e-05
cuda_vs_torch_delta_error_max = 7.566995918750763e-08
bridge_score_error_max = 1.1920928955078125e-07
tail_disagreement_count = 0
accept_disagreement_count = 10
native_cuda_bucket_kernel_numeric_pass = 0
native_cuda_bucket_kernel_diagnostic_pass = 0
```

第二轮改动：

```text
将 bridge score 从 atomicAdd 汇总改成 deterministic fixed-order reduce；
目标是排除 atomic reduction 顺序导致的 accept bit 翻转。
```

第二轮结果：

```text
route = R16-QuantileTailDominantUnfixed
native_cuda_bucket_kernel_used = 1
eager_single_pass_time_ms_q90 = 0.9521790780127048
native_cuda_bucket_kernel_time_ms_q90 = 0.6421208381652832
native_cuda_bucket_kernel_time_q90_reduction = 0.3256301750449557
cuda_vs_torch_logits_error_max = 3.337860107421875e-05
cuda_vs_torch_delta_error_max = 7.35744833946228e-08
bridge_score_error_max = 1.1920928955078125e-07
tail_disagreement_count = 0
accept_disagreement_count = 10
native_cuda_bucket_kernel_numeric_pass = 0
```

注意：第二轮主 route 因同轮 quantile-tail timing 回退变成 `R16-QuantileTailDominantUnfixed`，没有用于提升主结论。

第三轮改动：

```text
将 bridge reduce 改成 fixed-order float accumulation；
目标是更接近 PyTorch mean 的 float reduction，而不是 double reduction。
```

第三轮结果：

```text
route = R17-BasisNormRuntimeStillInsufficient
native_cuda_bucket_kernel_used = 1
eager_single_pass_time_ms_q90 = 1.111069694161415
native_cuda_bucket_kernel_time_ms_q90 = 0.6082197651267052
native_cuda_bucket_kernel_time_q90_reduction = 0.4525818062333508
cuda_vs_torch_logits_error_max = 3.337860107421875e-05
cuda_vs_torch_delta_error_max = 7.229391485452652e-08
bridge_score_error_max = 1.1920928955078125e-07
tail_disagreement_count = 0
accept_disagreement_count = 9
native_cuda_bucket_kernel_numeric_pass = 0
```

判断：

1. 这次确实执行了 native CUDA bucket kernel，不是停在建议层。
2. native kernel 对局部 runtime 有真实改善：q90 降低约 `32%`，kernel/sync 诊断计数也显著下降。
3. 但数值 gate 没过：logits/delta/tail 都基本对齐，失败集中在 `accept_bit_inside_kernel`，`accept_disagreement_count = 10`。
4. accept 翻转不是 tail mismatch，也不是明显 logits/delta 崩坏；bridge score error 只有 `1.19e-07`，但 bridge scores 处在 median 边界附近，这个微小误差足以改变 `score > median` 的赢家。
5. fixed-order double reduce 与 fixed-order float reduce 都没有把 accept disagreement 清零，说明这不是简单 atomic 顺序问题。
6. 不能为了过 gate 放宽 accept bit 或把 accept 移回 Python，因为计划要求 `bridge_score_inside_kernel` / `accept_bit_inside_kernel`，且不允许把数值未闭合的 kernel 写成 official。
7. 当前下一步若继续推进，应把 bridge accept 从 median-threshold fragile rule 改成数值稳定且与 reference 明确一致的 tie policy，或把 reference controller 的 accept rule 一起重定义并重新跑全套 decision/support/no-fake gates；不能只在 P16 局部改 tolerance 后宣称通过。

No-fake audit：

```text
rows_checked = 225
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

Hash：

| artifact | SHA256 |
|---|---|
| runner first native | `eb4f73efeb896911c68d30e43bee4bd6e3cb837fc7ce82d5de3e032dc3774673` |
| route first native | `21195f88d6c441b8127337c023ba3e9bab7396b236facb41d1c07a0bde2f09a7` |
| P16 first native | `93ed8ceea188c6599d5c55434c33df9b53c8a4779ed6d1ba842c26218347dc1e` |
| runner deterministic bridge | `c6883096cc73616a66052bfdffe87f62d3783df5662edaef549284cbb3f83077` |
| route deterministic bridge | `6d26c1bf11e91e1ec7806137f81f42bd0da0302644140d6dd5efeca409a225ae` |
| P16 deterministic bridge | `47eca3bb624f5c593d88a4f6cea0e1e9ba8fecb8390ca443dea2a9c174373810` |
| runner float-reduce bridge | `baa32c5caf8fbc3b716a8c04a940bbc77f4997389cddeecb68fdd3102a41a7cf` |
| route float-reduce bridge | `d617b95a6b415d831b97030949b42cee2b4f64c380505b6eac67e2bd9c746fa8` |
| P16 float-reduce bridge | `abfcf207e3fae0b28dbe37afa58616920578bafe397c6bbade65844669369366` |
| provenance audit | `e10632a54432911bbef0b416109c825b663e72f8ea4a6635bd59343dc62d9b56` |

追加结论：

> native CUDA bucket kernel 已执行并取得真实局部 runtime 改善，最佳 q90 reduction 达到 `45.26%`；但 accept bit 数值没有闭合，因此不能转成 static-bucket official runtime。v9.2.77 仍不能打开 system-legal controller；当前困难已经从“没写 native kernel”推进到“native bridge accept 的 median 边界数值稳定性未闭合”。
