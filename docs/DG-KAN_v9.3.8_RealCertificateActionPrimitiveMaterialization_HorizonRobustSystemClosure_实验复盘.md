# DG-KAN v9.3.8 Real Certificate-Producing Action Primitive Materialization / Horizon-Robust System Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.8_RealCertificateActionPrimitiveMaterialization_HorizonRobustSystemClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 AP0 旧 outcome、diagnostic certificate 或未选中 controller 的 runtime estimate 写成 official system pass。

## 0. 最新结论

```text
route = R5-APGeneratedActionsValueFail
base_candidate = LQ-t2-h256
success_v9380_strict_purekan_functional = False
success_v9380_full_functional = False
success_v9380_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9380_real_certificate_action_primitive_materialization_horizon_system_closure_first_20260514T050000Z/
```

核心结论：

1. P0 复现 v9.3.7 boundary：source route = `R6-ActionPrimitiveRedesignRequired`，v9.3.7 generated action = `0`，system pass = `0`。
2. P1 已按计划停止 AP0 threshold/probe patch：best AP0 AUC = `0.6151712728688206`，best top273 CP precision = `0.28205128205128205`，`ap0_stop_condition = 1`。
3. P1 首次真实生成 AP1-AP4 action payload：source AP0 action = `64`，primitive = `4`，generated AP action = `256`，每个 primitive = `64`。
4. P1 durable payload / certificate tensor 均已落盘：`durable_payload_written = 1`，`certificate_tensor_written = 1`，payload/certificate hash missing = `0 / 0`。
5. P1 action apply replay 数值闭合：`action_apply_error_measured = 1`，L∞ max = `0.0`，relative max = `0.0`。
6. P2 certificate legality/minimality contract 过：certificate rows = `256`，fields complete = `1`，payload-hash binding = `1`，legality violation = `0`，certificate pass count = `83`。
7. P3 新 AP payload smoke outcomes 已真实 materialize：branch-horizon expected/actual = `6912 / 6912`，branch/horizon completion = `1.0 / 1.0`，rows/sec = `24.6675916566484`。
8. P3 不是 AP0 旧 outcome：outcome source = `same_run_generated_AP_smoke`，branches 包含 `RealAP`、controls、shuffled AP payload 和 certificate-pass-no-payload。
9. 但 generated AP certificate-pass rows 未形成 value/horizon frontier：weak CP precision = `0.10040160642570281`，strong CP precision = `0.028112449799196786`，horizon-robust CP precision = `0.0`，long-risk rate = `0.3092369477911647`，V_ctrl LCB = `-1.409779974372909`。
10. 最佳 primitive 是 `AP3-HorizonRobustTailMemoryCertificate`，但仍不过：weak CP precision = `0.19298245614035087`，strong CP precision = `0.05263157894736842`，horizon-robust = `0.0`，V_ctrl LCB = `-1.5802910145029976`。
11. P5 certificate calibration failed，P6 controller 未选中，P7 selected runtime 正确 `not_run`，没有写 runtime estimate。
12. P8 system controller 未打开：`official_eligible = 0`，`system_legal_controller_pass = 0`，primary blocker = `generated_AP_certificate_pass_rows_not_value_positive_or_horizon_safe`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9380_real_certificate_action_primitive_materialization.py` | v9.3.8 runner；从 v9.3.3 durable AP0 payload 真实生成 AP1-AP4 payload/certificate，disk replay，跑 generated AP branch-horizon smoke outcome，并执行 controller/runtime/system gate |

代码检查：

```text
python -m py_compile experiments/run_v9380_real_certificate_action_primitive_materialization.py
```

正式运行：

```bash
python experiments/run_v9380_real_certificate_action_primitive_materialization.py \
  --out-dir results/real_rerun_20260506/v9380_real_certificate_action_primitive_materialization_horizon_system_closure_first_20260514T050000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --actions-per-primitive 64 --smoke-actions-per-primitive 64 --runtime-actions 64
```

运行结果：

```json
{
  "ap_smoke_rows": 6912,
  "generated_action_count_total": 256,
  "out_dir": "results/real_rerun_20260506/v9380_real_certificate_action_primitive_materialization_horizon_system_closure_first_20260514T050000Z",
  "route": "R5-APGeneratedActionsValueFail"
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R5-APGeneratedActionsValueFail",
  "source_route_v9370": "R6-ActionPrimitiveRedesignRequired",
  "generated_action_count_total": 256,
  "primitive_materialized_count": 4,
  "max_generated_action_count_per_primitive": 64,
  "primitive_generation_pass": 1,
  "durable_payload_written": 1,
  "certificate_tensor_written": 1,
  "action_apply_error_linf_max": 0.0,
  "certificate_schema_contract_pass": 1,
  "certificate_pass_count": 83,
  "ap_smoke_outcome_pass": 1,
  "branch_horizon_row_count_actual": 6912,
  "certificate_pass_outcome_count": 249,
  "weak_CP_precision_certificate_pass": 0.10040160642570281,
  "strong_CP_precision_certificate_pass": 0.028112449799196786,
  "horizon_robust_CP_precision_certificate_pass": 0.0,
  "long_risk_rate_certificate_pass": 0.3092369477911647,
  "V_ctrl_lcb_certificate_pass": -1.409779974372909,
  "certificate_calibration_pass": 0,
  "certificate_controller_pass": 0,
  "selected_payload_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "generated_AP_certificate_pass_rows_not_value_positive_or_horizon_safe"
}
```

判断：v9.3.8 已经把 v9.3.7 的 “AP1-AP4 未实现” blocker 推进为真实 AP payload/outcome 实测；新的 blocker 是这些 generated AP + certificate rows 本身没有产生足够 value-positive、horizon-safe frontier。

## 3. P1 real AP generator

Artifacts：

```text
p1_real_ap_generator_smoke.csv
ap_action_payload_trace_v9380.csv
action_apply_replay_trace_v9380.csv
ap_action_payload_shards_v9380/
```

Summary：

```text
source_ap0_action_count = 64
primitive_count = 4
generated_action_count_total = 256
max_generated_action_count_per_primitive = 64
primitive_materialized_count = 4
durable_payload_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_measured = 1
action_apply_error_linf_max = 0.0
action_apply_error_relative_max = 0.0
certificate_pass_count = 83
primitive_generation_pass = 1
```

判断：H1/H2 的 implementation blocker 被真实推进。AP1/AP2/AP3/AP4 都生成了可 replay payload，并在 outcome 前写出 certificate/hash；没有使用 outcome-at-commit。

## 4. P2 certificate legality/minimality

Artifacts：

```text
p2_certificate_legality_minimality_audit.csv
ap_certificate_trace_v9380.csv
```

Summary：

```text
certificate_schema_version = v9380-real-ap-cert-v1
generated_action_count_total = 256
certificate_rows = 256
certificate_fields_complete = 1
certificate_hash_bound_to_payload_hash = 1
legality_violation_count = 0
certificate_pass_count = 83
certificate_schema_contract_pass = 1
```

判断：certificate schema 不再只是 AP0 diagnostic；它绑定了真实 generated AP payload。P2 pass 只说明证书合法且可审计，不说明证书有 functional value。

## 5. P3 generated AP smoke outcomes

Artifacts：

```text
p3_ap_smoke_outcome_materialization.csv
ap_smoke_outcome_trace_v9380.csv
branch_horizon_completion_trace_v9380.csv
```

Summary：

```text
materializer_id = APSMOKE1-GeneratedAPPayloadBranchHorizonSmoke
generated_action_count_input = 256
smoke_action_count_expected = 256
branch_horizon_row_count_expected = 6912
branch_horizon_row_count_actual = 6912
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
new_ap_payload_outcomes_materialized = 1
ap_smoke_outcome_pass = 1
certificate_pass_outcome_count = 249
weak_CP_precision_certificate_pass = 0.10040160642570281
strong_CP_precision_certificate_pass = 0.028112449799196786
horizon_robust_CP_precision_certificate_pass = 0.0
long_risk_rate_certificate_pass = 0.3092369477911647
V_ctrl_lcb_certificate_pass = -1.409779974372909
rows_per_sec = 24.6675916566484
branch_runtime_ms_q90 = 117.4008771777153
```

判断：P3 materialization 过，但 H3 未成立。证书通过的 generated AP rows 在真实 branch/horizon outcome 下 value 明显不足，且 long-risk 过高。

## 6. P5 certificate calibration

Artifact：

```text
p5_certificate_calibration_sufficiency.csv
```

Per primitive：

| primitive | cert-pass outcomes | weak CP precision | strong CP precision | horizon robust precision | long-risk rate | V_ctrl LCB | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP1 | 84 | 0.059524 | 0.011905 | 0.0 | 0.333333 | -1.299056 | 0 |
| AP2 | 57 | 0.087719 | 0.035088 | 0.0 | 0.333333 | -1.762332 | 0 |
| AP3 | 57 | 0.192982 | 0.052632 | 0.0 | 0.228070 | -1.580291 | 0 |
| AP4 | 51 | 0.078431 | 0.019608 | 0.0 | 0.333333 | -1.744541 | 0 |

判断：AP3 是相对最好，但仍远低于 weak gate：weak CP precision 只有 `0.192982`，V_ctrl LCB 为负，horizon robust precision 为 `0`。

## 7. Runtime / System Boundary

P7：

```text
runtime_candidate_id = not_run_certificate_controller_blocked
runtime_mode = not_run
selected_primitive_id = AP3-HorizonRobustTailMemoryCertificate
selected_controller_used = 0
selected_payload_apply_used = 0
selected_payload_runtime_pass = 0
reason = certificate_controller_not_selected
```

P8：

```text
system_candidate_id = SYS-v9380-real-certificate-action-primitive
primitive_id = AP3-HorizonRobustTailMemoryCertificate
controller_id = not_selected_certificate_calibration_failed
runtime_candidate_id = not_run_certificate_controller_blocked
official_eligible = 0
system_legal_controller_pass = 0
reason = generated_AP_certificate_pass_rows_not_value_positive_or_horizon_safe
```

判断：本轮没有把 runtime estimate 写成 official。由于 certificate controller 未选中，selected runtime 必须 gate-block。

## 8. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p9_leave_dataset_stratum_out.csv` | `P8_system_controller_not_official` |
| `p10_diagnostic_paired_replay_scout.csv` | same |
| `p11_official_paired_replay.csv` | same |
| `p12_short_full_sampleeff_continual_robustness.csv` | same |

没有把 AP generation pass、certificate contract pass 或 AP smoke outcome materialization pass 写成 LDO/paired replay/short-full success。

## 9. No-fake audit

```text
rows_checked = 7681
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
ap0_stop_condition = 1
primitive_generation_pass = 1
certificate_schema_contract_pass = 1
ap_smoke_outcome_pass = 1
certificate_calibration_pass = 0
certificate_controller_pass = 0
selected_payload_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
uses_validation_or_test = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
source_measured_gap/formula_proxy = 0/0
diagnostic_promoted_to_official = 0
```

## 10. Hash

| artifact | SHA256 |
|---|---|
| plan | `8b9043e198cf103f0d7a70b0536d1860c40733f1371e21f6d8ffed9e9ea1d79a` |
| runner | `3f18cbd424f921f4f7cd44d467414bc64777083730b38d9d4547e1da79ab5e2a` |
| run manifest | `1e4e092c7261f1ab0adff8e4c5ca7d186e37521e39e1dcc0cb9c293a1f119b9c` |
| route | `7d843b2b62300e4b1531c62de10b2c5c579d02a746c024094046a2933ce6908f` |
| P1 generator | `07c3733beeb892d5bc6ce62d3413d641e3c92082288e36e01d4cf4c56d72a580` |
| AP payload trace | `987cdb88dad2d118dfcc0182417ec2bf273aa187f6bec0f6d3ce349f2fd47231` |
| P2 certificate | `8c585e7ebef3c326312f738505e734ebef987418004b58cf4da36c5bfa33c960` |
| AP certificate trace | `84f54898c4b4fdde48f44ecb38aa26789c097d10fbdfad6a7e58f2d8557e15b1` |
| P3 smoke outcome | `c4d8f3f639a5a7716f79b7f5c967e2e43971d815aa32ef4dabc20d4a77dcb49f` |
| AP smoke outcome trace | `e3e48d09da440696e4497e88877d2aece50b7d192ce428319db918cb6433b4e3` |
| P5 calibration | `4c8aa53c4bca9b5ba72bb14643619eafde4fd14db4e4e6b583c3ca21d05d265f` |
| P7 runtime | `8ecb386f5fddd75050615619116357fc814d42f0ca08ed0d187e51d051f5678a` |
| P8 system | `32eb19fcad13ec945bf94490cc617c376ccc5ea8098e76089f7d838318f9b83e` |
| contract audit | `d426668f26f8c00ab68a9e715be73320e0b4fb08ea6087ad16c5c803c39e4783` |
| provenance audit | `dea22665ab131434185f6a2f6b82e8a0e02f46f99c615ddb0db8d15ae51783b2` |
| failure table | `6dcccf3de8e1e15f15a607a984e16aa212ef6bbf092f825edb30e869a9250188` |

## 11. 最终分析结论

v9.3.8 的真实推进是：

```text
v9.3.7:
  AP0 patch route 已停止；
  certificate schema 只能诊断 AP0；
  AP1-AP4 generated action count = 0。

v9.3.8:
  AP1-AP4 首次生成真实 payload/certificate/action apply trace；
  新 AP payload smoke outcomes 完整 materialize；
  但 certificate-pass generated AP rows 的 value / strong CP / horizon robustness 明显不足。
```

机制判断：

1. H0 成立：AP0 patch route 继续停止，没有执行 AP0 threshold search。
2. H1/H2 成立：真实 AP1-AP4 payload generator 和 commit-time certificate tensor 已落盘，hash 与 action apply replay 闭合。
3. H3 未成立：generated AP certificate-pass rows weak CP precision 只有 `0.1004`，strong CP precision `0.0281`，horizon robust precision `0.0`，long-risk `0.3092`。
4. H4 部分成立：certificate 字段可审计，但当前 certificate 不是 sufficient statistic。
5. H5 未打开：controller 未选中，因此 selected runtime 正确 not-run。
6. P8 未打开：不能进入 official paired replay 或 short/full functional validation。

最终一句话：

> v9.3.8 真实执行后停在 `R5-APGeneratedActionsValueFail`：AP1-AP4 不再是“未实现”，真实 payload/certificate/outcome 链路已经打通；但这些 generated AP action 的 certificate-pass 子集没有形成 value-positive、horizon-safe frontier，strict PureKAN functional 仍未成功。
