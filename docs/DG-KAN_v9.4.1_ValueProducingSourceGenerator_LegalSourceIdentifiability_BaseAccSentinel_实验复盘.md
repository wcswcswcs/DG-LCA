# DG-KAN v9.4.1 Value-Producing Source Generator / Legal Source Identifiability / Base-Acc Sentinel 实验复盘

> 本复盘记录 `DG-KAN_v9.4.1_ValueProducingSourceGenerator_LegalSourceIdentifiability_BaseAccSentinel_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 stratified panel、Base-Acc Sentinel、source payload generation 或未 materialize 的 source outcome 写成 official system pass。

## 0. 最新结论

```text
route = R4-GeneratedSourceImmediateDirectionFail
base_candidate = LQ-t2-h256
success_v9410_strict_purekan_functional = False
success_v9410_full_functional = False
success_v9410_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9410_value_producing_source_generator_legal_source_identifiability_base_acc_first_20260514T080000Z/
```

核心结论：

1. P0 复现 v9.4.0 boundary：source route = `R3-LegalSourceSelectorOpaque`，full action count = `2876`，v9.4.0 system pass = `0`。
2. P1 stratified source panel 修复了 v9.3.9 first-64 convenience slice 的代表性问题：official panel = `PANEL-S256`，PSI = `0.01609085697443582`，KL = `0.007892893707147842`，max family gap = `0.007448496175243395`。
3. P1 `PANEL-S64` 未过，`PANEL-S256` / `PANEL-S864` 过；本轮 official source panel 使用 `PANEL-S256`。
4. P2 Base-Acc Sentinel 已真实运行，datasets = `MNIST,Fashion-MNIST,KMNIST`，seeds = `0,1,2`，model count = `3`，sentinel row count = `27`。
5. Base-Acc Sentinel 只作为健康监控，没有用于 selector/controller：`base_acc_used_for_controller = 0`。
6. Sentinel mean test acc：LQ-t2-h256 = `0.6553819444444444`，MatchedMLP = `0.5559895833333334`，LQ minus MLP = `0.09939236111111105`，`LQ_catastrophic_fail = 0`。
7. P3 AP0b-AP0f value-producing source generator 首次真实 materialize：primitive count = `5`，source input actions = `52`，generated action count = `260`。
8. P3 durable payload / certificate tensor 均落盘：`payload_tensor_written = 1`，`certificate_tensor_written = 1`，payload/certificate hash missing = `0 / 0`。
9. P3 action apply replay 数值闭合：`action_apply_error_linf_max = 0.0`，relative max = `0.0`，cosine min = `0.9999999999999938`。
10. P4/P5 source outcome 没有 materialize 出有效 branch-horizon rows：expected = `7020`，actual = `0`，`source_outcome_materialized = 0`。
11. 因 P4 actual rows = `0`，不能声明 generated source actions value-negative，也不能声明 h20/horizon extension 成功；本轮只说明 h20 immediate gate 未打开。
12. P6 certificate sufficiency 未运行，reason = `source_outcomes_not_materialized`。
13. P7-P14 均 gate-blocked：没有 source controller、selected runtime、system controller、paired replay、short/full/continual success。
14. 当前 primary blocker：`generated_source_immediate_direction_fail`；更具体地说，本轮 blocker 是 generated source outcome materializer 没有产出可评估 h20 rows。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9410_value_producing_source_generator.py` | v9.4.1 runner；读取 v9.4.0/v9.3.9/v9.3.5/v9.3.3 artifacts，执行 stratified panel、Base-Acc Sentinel、AP0b-AP0f source payload/certificate generation、source outcome gate、system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9410_value_producing_source_generator.py
```

正式运行：

```bash
python experiments/run_v9410_value_producing_source_generator.py \
  --out-dir results/real_rerun_20260506/v9410_value_producing_source_generator_legal_source_identifiability_base_acc_first_20260514T080000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --source-generator-actions 52 --smoke-actions-per-primitive 52 --seeds 0
```

说明：source outcome smoke 使用 `--seeds 0`；Base-Acc Sentinel 仍按 manifest 使用 `--sentinel-seeds 0,1,2`。实际参数记录在 `run_manifest.json`。

运行结果：

```json
{
  "generated_action_count_total": 260,
  "h20_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9410_value_producing_source_generator_legal_source_identifiability_base_acc_first_20260514T080000Z",
  "route": "R4-GeneratedSourceImmediateDirectionFail"
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R4-GeneratedSourceImmediateDirectionFail",
  "source_route_v9400": "R3-LegalSourceSelectorOpaque",
  "stratified_panel_pass": 1,
  "official_panel_PSI_vs_full": 0.01609085697443582,
  "official_panel_KL_vs_full": 0.007892893707147842,
  "base_acc_sentinel_complete": 1,
  "base_acc_used_for_controller": 0,
  "mean_test_acc_LQ": 0.6553819444444444,
  "mean_test_acc_MLP": 0.5559895833333334,
  "LQ_catastrophic_fail": 0,
  "source_generator_materialized": 1,
  "primitive_materialized_count": 5,
  "generated_action_count_total": 260,
  "payload_tensor_written": 1,
  "certificate_tensor_written": 1,
  "action_apply_error_linf_max": 0.0,
  "p3_materialization_pass": 1,
  "h20_immediate_direction_pass": 0,
  "horizon_extension_pass": 0,
  "certificate_sufficiency_pass": 0,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "generated_source_immediate_direction_fail"
}
```

判断：v9.4.1 真实推进了 source panel 代表性、Base-Acc Sentinel 和 AP0b-AP0f source payload/certificate 生成；但 source outcome materializer 没有产出 P4/P5 rows，所以 controller/system 必须阻断。

## 3. P1 stratified source panel

Artifacts：

```text
p1_stratified_source_panel_builder.csv
stratified_source_panel_trace_v9410.csv
```

Summary：

```text
panel_s64_pass = 0
panel_s256_pass = 1
panel_s864_pass = 1
official_panel_id = PANEL-S256
official_panel_action_count = 256
official_panel_PSI_vs_full = 0.01609085697443582
official_panel_KL_vs_full = 0.007892893707147842
official_panel_max_family_gap = 0.007448496175243395
official_panel_max_step_bucket_gap = 0.004531032684283731
official_panel_max_score_bucket_gap = 0.002151425591098738
official_panel_max_payload_bucket_gap = 0.004617958970792757
source_panel_used_for_official = 1
stratified_panel_pass = 1
```

判断：P1 解决了 v9.3.9 / v9.4.0 指出的 first-N biased panel 问题。`PANEL-S256` 已足够接近 full AP0 universe，可作为本轮 source generator 输入；但 panel pass 不等于 controller pass。

## 4. P2 Base-Acc Sentinel

Artifacts：

```text
p2_base_acc_sentinel_lq_vs_mlp.csv
base_acc_training_trace_v9410.csv
```

Summary：

```text
sentinel_row_count = 27
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
model_count = 3
sentinel_complete = 1
mean_test_acc_LQ = 0.6553819444444444
mean_test_acc_MLP = 0.5559895833333334
LQ_minus_MLP_mean_test_acc = 0.09939236111111105
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
hyperparams_fixed_before_run = 1
dataset_specific_tuning = 0
same_seed_schedule = 1
same_budget = 1
```

判断：Base-Acc Sentinel 没有发现 LQ-t2-h256 catastrophic fail；但它只是 fixed-config sentinel，不是 official full functional result，也没有用于任何 selector/controller。

## 5. P3 real value-producing source generator

Artifacts：

```text
p3_real_value_producing_source_generator.csv
source_payload_trace_v9410.csv
source_certificate_trace_v9410.csv
action_apply_replay_trace_v9410.csv
source_action_payload_shards_v9410/
```

Summary：

```text
source_generator_materialized = 1
primitive_materialized_count = 5
source_input_action_count = 52
generated_action_count_total = 260
max_generated_action_count_per_primitive = 52
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_measured = 1
action_apply_error_linf_max = 0.0
action_apply_error_relative_max = 0.0
action_apply_cosine_min = 0.9999999999999938
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
certificate_schema_version = v9410-source-cert-v1
certificate_fields_complete = 1
certificate_pass_action_count = 81
p3_materialization_pass = 1
```

判断：v9.4.0 的 `source_generator_materialized = 0` blocker 被真实推进。AP0b-AP0f 都生成了 durable payload 和 commit-time certificate，action apply replay 也闭合。这个结论只覆盖 generation/apply，不覆盖 source action value。

## 6. P4/P5 source outcome boundary

Artifacts：

```text
p4_h20_immediate_direction_smoke.csv
p5_horizon_extension_longrisk_audit.csv
source_outcome_trace_v9410.csv
branch_horizon_completion_trace_v9410.csv
```

P4 summary：

```text
stage = P4_H20_IMMEDIATE_DIRECTION_SMOKE
best_primitive_id = AP0b-LastEdgeLinearizedDescentSource
branch_horizon_row_count_expected = 7020
branch_horizon_row_count_actual = 0
source_outcome_materialized = 0
quality_audit_pass = 0
h20_immediate_direction_pass = 0
```

P5 summary：

```text
stage = P5_HORIZON_EXTENSION_LONGRISK_AUDIT
best_primitive_id = AP0b-LastEdgeLinearizedDescentSource
branch_horizon_row_count_actual = 0
horizon_extension_pass = 0
quality_audit_pass = 0
```

判断：P4/P5 没有得到可评估 source outcome rows。这里不能把 `weak_CP_precision = 0.0` 解释为真实 value-negative；它是 outcome rows absent 下的 gate-blocked 结果。正确结论是：generated source payload/certificate 已落盘，但 h20/h80/h240 source outcome materializer 没有闭合。

## 7. P6-P14 gated boundary

P6：

```text
stage = P6_EFFECT_VALID_CERTIFICATE_SUFFICIENCY
status = not_run
reason = source_outcomes_not_materialized
certificate_sufficiency_pass = 0
```

后续 boundary：

| artifact | status / reason |
|---|---|
| `p7_minimal_source_certificate_controller.csv` | controller not selected |
| `p8_selected_source_online_runtime.csv` | selected controller not available |
| `p9_system_integration_gate_v9410.csv` | official eligible = 0 |
| `p10_leaveout_boundary_v9410.csv` | P9 system controller not official |
| `p11_diagnostic_paired_replay_scout_v9410.csv` | same |
| `p12_official_paired_replay_v9410.csv` | same |
| `p13_short_full_training_mlp_comparison_v9410.csv` | same |
| `p14_continual_antiforgetting_v9410.csv` | same |

判断：没有把 source generator pass、Base-Acc Sentinel 或 empty outcome diagnostics 写成 source controller/runtime/system/downstream success。

## 8. No-fake audit

```text
rows_checked = 2081
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
stratified_panel_pass = 1
base_acc_sentinel_complete = 1
base_acc_used_for_controller = 0
source_generator_materialized = 1
p3_materialization_pass = 1
h20_immediate_direction_pass = 0
horizon_extension_pass = 0
certificate_sufficiency_pass = 0
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
source_measured_gap/formula_proxy = 0/0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
F1_source_panel_sampling_bias = 0
F2_static_legal_selector_opaque = 1
F3_source_generator_missing = 0
F4_action_apply_error = 0
F5_h20_immediate_direction_fail = 1
F6_horizon_long_risk_fail = 0
F7_certificate_no_effect_lift = 0
F8_controller_support_collapse = 1
F9_runtime_payload_apply_too_slow = 1
F10_base_acc_sentinel_catastrophic = 0
F14_short_full_not_open = 1
primary_blocker = generated_source_immediate_direction_fail
```

## 9. Hash

| artifact | SHA256 |
|---|---|
| plan | `1795fefd35078086be6fab162e8609c25bc6b86dc6f24b2e25564f8da8307d17` |
| runner | `a4d6fcbd6acefd9646aacfe5a4e02af2ac3965c1483a21f916d9dec8475885df` |
| run manifest | `72f1621ae31fb2ca0fa62417a92c9576b83b4af38c29c1a6304869d28ab01caa` |
| route | `1f5cd7baaac3b0d4519c6718a3afe8c1761b91c0ff2bffac94a7773138239382` |
| P0 boundary | `be26134a0f81435513621ac13f1ee265b23cc8a2584c93ae0906f21fbc444e94` |
| P1 stratified panel | `bcff3618bf5beac67adec32fc161c7f1b75423f3588ca4735c1188d1e36f05e1` |
| P2 Base-Acc Sentinel | `1d0a5bfb93584c74f184122271eb665f3f83bcbc06aaa36560bbff61aa75afb5` |
| P3 source generator | `591c1db0bd03014055d73d88828a6d18c921a6c51f7456740cffde08966734e1` |
| P4 h20 smoke | `da2e7cd61caaebd73d8e0fbb05b40f289fdd8167f328dd59ccd1753f62871147` |
| P5 horizon extension | `372cee16e4a2cc4978ec8aa7f189939cbafe282d4a9cbd6b7e2719926a724f32` |
| P6 certificate | `2a39c0530dee5f99ee8fea3397666f1a0ed307ee003f333e1b911b326c4a359f` |
| P7 controller | `066dda27625648ab17269662d3635720df2fe8857bbb74bb78b98ced046b64f8` |
| P8 runtime | `4b6d46656329d87bd7f0c8fe5fd21d2c6c61902e07f1dbcd5a77e4e638b45469` |
| P9 system | `da94459e9a0068a5132cf5c5e498facdd09d31efb33ea1b56b418a2b43bdfd73` |
| contract audit | `e97adf9385d7f829db493f9649efa51569e15ead0fc18d98e7379ee40150d70f` |
| provenance audit | `8719c574789a23d93239756ec163f5fce951c1d490e31fc89df0704a97366d9a` |
| failure table | `aad48617f18e5b2959a4bb68ef2878836ff698aeac7ea14a1fb900b352702d9a` |

## 10. 最终分析结论

v9.4.1 的真实推进是：

```text
v9.4.0:
  full AP0 universe 里存在 oracle-good source；
  v9.3.9 first-64 source panel 是 biased convenience slice；
  但 legal source selector 看不见 oracle frontier，source generator 未 materialize。

v9.4.1:
  用 stratified source panel 替代 biased source slice；
  增加 fixed-config Base-Acc Sentinel；
  首次 materialize AP0b-AP0f value-producing source payload/certificate/action apply replay；
  但 generated source branch-horizon outcomes 没有落盘成可评估 rows。
```

机制判断：

1. H0 成立：`PANEL-S256` 的 PSI/KL/gap 都很低，source panel bias blocker 已被推进。
2. H1 仍成立：Base-Acc Sentinel 没有 catastrophic fail，但这不是 controller 证据，也没有用于调参。
3. H2 部分成立：AP0b-AP0f source generators 已真实 materialize，payload/certificate/action apply 全部闭合。
4. H3 未打开：h20 immediate smoke 没有 materialize outcome rows，不能判断 generated source actions 是否 value-producing。
5. H4-H7 全部 gate-blocked：没有 h20 pass，就不能做 horizon extension、certificate sufficiency、controller、selected runtime 或 system integration。
6. P13/P14 未打开：没有 official system controller，不能进入 short/full training、MLP comparison 或 continual validation。

最终一句话：

> v9.4.1 真实执行后停在 `R4-GeneratedSourceImmediateDirectionFail`：代表性 source panel、Base-Acc Sentinel、AP0b-AP0f source payload/certificate generation 都真实推进了；但 source outcome materializer 没有产出 h20/horizon rows，因此 generated source value、certificate controller 和 strict PureKAN functional 仍不能转正。
