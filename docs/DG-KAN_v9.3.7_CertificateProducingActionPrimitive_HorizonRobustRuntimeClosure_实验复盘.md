# DG-KAN v9.3.7 Certificate-Producing Action Primitive / Horizon-Robust Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.7_CertificateProducingActionPrimitive_HorizonRobustRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 AP0 诊断证书、AP0 旧 outcome 或未实现的 AP1-AP4 primitive 写成 official system pass。

## 0. 最新结论

```text
route = R6-ActionPrimitiveRedesignRequired
base_candidate = LQ-t2-h256
success_v9370_strict_purekan_functional = False
success_v9370_full_functional = False
success_v9370_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9370_certificate_producing_action_primitive_horizon_runtime_first_20260514T040000Z/
```

核心结论：

1. P0 复现 v9.3.6 boundary：source route = `R12-CertificatePrimitiveFail`，weak CP coverage = `0.0984347442680776`，horizon-robust CP coverage = `0.004298941798941799`，system pass = `0`。
2. P1 明确停止 AP0 threshold/probe patch：检查 feature = `24`，best AUC 仍为 `F4-StateNLL = 0.6151712728688206`，best top273 CP precision = `0.28205128205128205`，`ap0_stop_condition = 1`。
3. P2 certificate schema 可以在 AP0 diagnostic rows 上落盘：diagnostic certificate rows = `3456`，primitive count = `4`，fields complete = `1`，legality audit pass = `1`。
4. 但 P2 不是 AP1-AP4 official contract：`primitive_materialized_count = 0`，certificate pass count = `0`，`certificate_schema_contract_pass = 0`。
5. P3 AP1-AP4 真实 action primitive generation 未 materialize：`generated_action_count_total = 0`，`primitive_generation_pass = 0`。
6. AP0 diagnostic certificate pass ignoring payload 共 `51` 行：AP1 = `4`，AP2 = `11`，AP3 = `36`，AP4 = `0`；这些不能写成新 AP payload。
7. P4 没有新 AP payload outcome：branch/horizon completion = `0.0 / 0.0`，`ap_smoke_outcome_pass = 0`。
8. P4 的 primitive outcome summary 只来自 AP0 existing outcomes diagnostic：AP1 weak CP precision = `0.25`，AP2 = `0.2727272727272727`，AP3 = `0.25`，AP4 = `0.0`。
9. P5-P8 因 AP primitive 未生成而 gate-blocked：没有 certificate calibration、full AP frontier、certificate controller 或 selected primitive runtime。
10. P9 system controller 未打开：`official_eligible = 0`，`system_legal_controller_pass = 0`，reason = `certificate_producing_action_primitive_not_materialized`。
11. 当前 blocker：`certificate_producing_action_primitive_not_materialized`；下一步必须实现真实 AP1-AP4 payload generators 与 commit-time certificate tensors，而不是继续在 AP0 上做事后诊断。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9370_certificate_producing_action_primitive_horizon_runtime.py` | v9.3.7 runner；读取 v9.3.6/v9.3.5 artifacts，执行 AP0 stop audit、certificate schema audit、AP1-AP4 primitive generation boundary、AP smoke outcome boundary、system gate |

代码检查：

```text
python -m py_compile experiments/run_v9370_certificate_producing_action_primitive_horizon_runtime.py
```

正式运行：

```bash
python experiments/run_v9370_certificate_producing_action_primitive_horizon_runtime.py \
  --out-dir results/real_rerun_20260506/v9370_certificate_producing_action_primitive_horizon_runtime_first_20260514T040000Z \
  --fresh --device auto --data-root data --seed 1314 --diagnostic-actions 864
```

运行结果：

```json
{
  "diagnostic_certificate_rows": 3456,
  "generated_action_count_total": 0,
  "out_dir": "results/real_rerun_20260506/v9370_certificate_producing_action_primitive_horizon_runtime_first_20260514T040000Z",
  "route": "R6-ActionPrimitiveRedesignRequired"
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R6-ActionPrimitiveRedesignRequired",
  "source_route_v9360": "R12-CertificatePrimitiveFail",
  "p0_boundary_reanalysis_pass": 1,
  "ap0_stop_condition": 1,
  "best_ap0_feature_auc_CP": 0.6151712728688206,
  "best_ap0_top273_CP_precision": 0.28205128205128205,
  "certificate_schema_rows_complete": 1,
  "certificate_schema_contract_pass": 0,
  "diagnostic_certificate_rows": 3456,
  "diagnostic_certificate_pass_ignoring_payload_count": 51,
  "primitive_generation_pass": 0,
  "generated_action_count_total": 0,
  "ap_smoke_outcome_pass": 0,
  "certificate_calibration_pass": 0,
  "full_ap_frontier_pass": 0,
  "certificate_controller_pass": 0,
  "selected_payload_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "certificate_producing_action_primitive_not_materialized",
  "next_required_implementation": "implement_real_AP1_AP4_payload_generators_with_commit_time_certificate_tensors",
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0
}
```

判断：v9.3.7 没有回退 v9.3.6 的结论。AP0 是 oracle-good but legally non-identifiable；本轮进一步确认 repo 当前没有真实 AP1-AP4 certificate-producing payload generator，不能继续 official promotion。

## 3. P0 v9.3.6 boundary reanalysis

Artifact：

```text
p0_v9360_boundary_reanalysis.csv
```

Summary：

```text
source_route = R12-CertificatePrimitiveFail
candidate/action/event = 2876 / 2876 / 24192
weak_CP_row_count = 893
weak_CP_coverage = 0.0984347442680776
weak_CP_coverage_lcb = 0.09247342078415305
weak_CP_V_ctrl_lcb = 0.2833107634852314
strong_CP_row_count = 353
strong_CP_coverage = 0.03891093474426808
horizon_robust_CP_action_count = 13
horizon_robust_CP_coverage = 0.004298941798941799
short_only_CP_action_count = 201
long_risk_action_count = 294
best_legal_capacity_feature = F4-StateNLL
best_legal_capacity_auc_CP = 0.6151712728688206
best_aef_feature = AEF6-ProbeReliability
best_aef_auc_CP = 0.5030687962043473
payload_apply_time_ms_q90_microbench = 0.0245068222284317
step_ratio_q90_microbench = 1.0305048474583738
p0_pass = 1
```

判断：P0 pass。上轮 blocker 不是 frontier absent，而是 AP0 legal identifiability / certificate primitive 缺失。

## 4. P1 AP0 threshold search ban

Artifacts：

```text
p1_ap0_threshold_search_ban_audit.csv
ap0_feature_capacity_trace_v9370.csv
```

Summary：

```text
feature_count_checked = 24
best_feature_by_auc = F4-StateNLL
best_auc_CP = 0.6151712728688206
best_feature_by_top273 = AEF6-ProbeReliability
best_top273_CP_precision = 0.28205128205128205
ap0_continuation_allowed = 0
ap0_stop_condition = 1
reason = no_AP0_feature_satisfies_continuation_gate
```

判断：P1 pass。计划要求停止 AP0 threshold/probe patch；本轮没有继续把 AP0 diagnostic feature 做成 controller。

## 5. P2 certificate schema / legality contract

Artifacts：

```text
p2_certificate_schema_legality_contract.csv
certificate_schema_trace_v9370.csv
```

Summary：

```text
certificate_schema_version = v9370-cert-schema-v1
diagnostic_certificate_rows = 3456
primitive_count = 4
certificate_fields_complete = 1
payload_hash_missing_count = 0
primitive_materialized_count = 0
diagnostic_certificate_pass_ignoring_payload_count = 51
certificate_pass_count = 0
legality_audit_pass = 1
certificate_schema_contract_pass = 0
reason = certificate_schema_rows_complete_but_AP1_AP4_generated_payloads_not_materialized
```

判断：schema 字段本身可以落盘，但它只是在 AP0 measured rows 上生成 diagnostic certificate。由于 AP1/AP2/AP3/AP4 的真实 payload 与 commit-time certificate tensors 没有 materialize，P2 不能 official。

## 6. P3 AP primitive smoke generation

Artifacts：

```text
p3_ap_primitive_smoke_generation.csv
certificate_action_trace_v9370.csv
```

Summary：

```text
primitive_count = 4
generated_action_count_total = 0
diagnostic_certificate_row_count = 3456
certificate_pass_count_total = 0
primitive_generation_pass = 0
reason = AP1_AP4_certificate_producing_action_generators_missing
```

Per primitive：

| primitive | generated action | diagnostic rows | diagnostic pass ignoring payload | true certificate pass |
|---|---:|---:|---:|---:|
| `AP1-LastEdgeLinearizedTailSafeCertificate` | `0` | `864` | `4` | `0` |
| `AP2-AdamWResidualOrthogonalBenefitCertificate` | `0` | `864` | `11` | `0` |
| `AP3-HorizonRobustTailMemoryCertificate` | `0` | `864` | `36` | `0` |
| `AP4-LowRankEdgeCertificate` | `0` | `864` | `0` | `0` |

判断：这是本轮 terminal blocker。没有真实 certificate-producing payload generator，不能生成新 action，也不能跑新 AP payload outcome。

## 7. P4 AP smoke outcome materialization

Artifacts：

```text
p4_ap_smoke_outcome_materialization.csv
primitive_outcome_diagnostic_trace_v9370.csv
```

Summary：

```text
branch_completion_rate = 0.0
horizon_completion_rate = 0.0
quality_audit_pass = 0
ap_smoke_outcome_pass = 0
reason = new_AP_payload_outcomes_not_materialized
```

Diagnostic-only AP0 existing outcomes：

| primitive | diagnostic accepted | weak CP precision | strong CP precision | horizon robust CP precision | long-risk rate |
|---|---:|---:|---:|---:|---:|
| `AP1-LastEdgeLinearizedTailSafeCertificate` | `4` | `0.25` | `0.0` | `0.0` | `0.25` |
| `AP2-AdamWResidualOrthogonalBenefitCertificate` | `11` | `0.2727272727272727` | `0.09090909090909091` | `0.09090909090909091` | `0.09090909090909091` |
| `AP3-HorizonRobustTailMemoryCertificate` | `36` | `0.25` | `0.05555555555555555` | `0.0` | `0.027777777777777776` |
| `AP4-LowRankEdgeCertificate` | `0` | `0.0` | `0.0` | `0.0` | `0.0` |

判断：这些 diagnostic rows 使用 AP0 既有 outcomes，不是 AP1-AP4 新 payload outcomes。它们只能说明“当前 certificate 条件套在 AP0 上也很弱”，不能作为 primitive pass。

## 8. P5-P8 gated boundary

这些 artifact 均按 blocker 落盘：

| artifact | status / reason |
|---|---|
| `p5_certificate_calibration_sufficient_statistic_audit.csv` | `not_run_certificate_primitive_missing` |
| `p6_full_ap_frontier_completion.csv` | `not_run_certificate_primitive_missing` |
| `p7_certificate_controller.csv` | `not_selected_certificate_primitive_missing` |
| `p8_selected_primitive_online_runtime.csv` | `not_run_certificate_primitive_missing` |

判断：没有 certificate primitive，就不能做 calibration、full AP frontier、controller 或 selected runtime。没有把 AP0 diagnostic certificate rows 传递成下游成功。

## 9. P9 system boundary

Artifact：

```text
p9_system_integration_gate.csv
```

Boundary：

```text
system_candidate_id = SYS-v9370-certificate-primitive-boundary
primitive_id = not_selected
controller_id = not_selected_certificate_primitive_missing
runtime_candidate_id = not_selected_certificate_primitive_missing
certificate_schema_version = v9370-cert-schema-v1
candidate_count = 2876
action_count = 0
accepted_count = 0
official_eligible = 0
system_legal_controller_pass = 0
certificate_legality_pass = 1
payload_binding_pass = 0
materialized_system_path = 0
reason = certificate_producing_action_primitive_not_materialized
```

判断：P9 未打开。schema legality pass 不等于 primitive pass；由于 action count = `0`，payload binding 和 materialized system path 均不能 official。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p10_leave_dataset_stratum_out.csv` | `P9_system_controller_not_official` |
| `p11_diagnostic_paired_replay_scout.csv` | same |
| `p12_official_paired_replay.csv` | same |
| `p13_short_full_sampleeff_continual_robustness.csv` | same |

没有把 AP0 diagnostic certificate、certificate schema 或 AP0 old outcomes 写成 LDO / paired replay / short-full functional success。

## 11. No-fake audit

```text
rows_checked = 3566
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
certificate_schema_rows_complete = 1
certificate_schema_contract_pass = 0
primitive_generation_pass = 0
ap_smoke_outcome_pass = 0
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
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
F0_AP0_stop_condition = 1
F1_certificate_schema_not_official_without_payload = 1
F2_AP1_AP4_payload_generators_missing = 1
F3_AP_smoke_outcome_missing = 1
F4_certificate_controller_blocked = 1
F5_selected_runtime_blocked = 1
primary_blocker = certificate_producing_action_primitive_not_materialized
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `0895532665ab20cc23d9a426fbc86a950e93b6dd59b361c2b3c79f3a88a24ac5` |
| runner | `5bc0aeca6b05d6e66c0a895aa8353379cd0b5505ddce88d8a80a0b0b183d8b2b` |
| run manifest | `b2867d481d730b139fd2e418f4621e9f8aef4d3678ceb2ac502b54722db64d40` |
| route | `931056c8a2629d06e8d58932a865d384c8a413722d5c262a0230e0728be4ad0f` |
| aggregate decision | `931056c8a2629d06e8d58932a865d384c8a413722d5c262a0230e0728be4ad0f` |
| contract audit | `89dc22f492f05467133fd4a624d9531c8cbe50c24885caddaf5345636deb8548` |
| provenance audit | `ab8db1a2bdf8c4645615a0d6380b9bb6439cc6b55db1af635cff52ed3c03ef4f` |
| P0 boundary | `3066381418ab2d0b61b94f289384d50c25e6821111418397180cf5350f8d827a` |
| P1 AP0 ban | `b77dd7bd31c423e7e1774ec7a8815e6f410f00f4b62403d6a9eaf75cb1d2f3ce` |
| AP0 feature trace | `469b3714a3172e2888798c22cd6c50ff224995ced25fb1f2c10297ad9d3f14b4` |
| P2 certificate schema | `98e89f5281c174e0ce52d679e32c59f413bd04567d33de859a9a8fb9501f9848` |
| certificate schema trace | `414b980f2b6cb88b2c33eb22389040365f3243e7f9fd7a1a29b683c4bedb1cfd` |
| P3 primitive generation | `bb174447c5b6afe2b8420497d0e97a9445e92b1861874a79f744277f3f804af3` |
| certificate action trace | `a666a4d45c2d8e079bb203fc6bd8f187ed799742feaa98afd6b127fdc474f08b` |
| P4 AP smoke outcome | `55f5466e18848b7b00c1f06931d3c347f1047313758c16219ac080fd82b43fbe` |
| primitive outcome diagnostic | `40278282423b566bfce5b5c4779e6a6e03e86ec7cd03b0775162a69fc6f23fff` |
| P5 certificate calibration | `a427aab5defeebc6561d00e01e8bb4b246ff1389bc6c9af1dda102ad1371179f` |
| P6 full AP frontier | `661cd5a222e40dc7159b0654fe3391f1c7ca8d971dfb98cbe837b9857cea1da6` |
| P7 certificate controller | `b60cc2345b7b497dd65f31b7984c11cefe27bef69587f9dd5c57e979dec9612c` |
| P8 selected runtime | `9616337587782e2ffaec3e115da5caf89377d981e76c13575581bde762271c56` |
| P9 system | `89efb114c5e7cf826d66175ef805534ec10b2eb71a890adfbad06467d21780df` |
| failure table | `a2621d6341b82edd3a6d0cf92d48d1f77c2f980e50c455c1eb6c1a8af27535e1` |

## 13. 最终分析结论

v9.3.7 的真实推进是：

```text
v9.3.6:
  AP0 full oracle frontier 仍存在；
  但 AP0 legal observability / AEF / microprobe 都无法识别 CP frontier；
  因此计划要求转向 certificate-producing action primitive。

v9.3.7:
  按计划停止 AP0 threshold/probe patch；
  建立 certificate schema 与 legality audit；
  但 repo 当前没有 AP1/AP2/AP3/AP4 的真实 payload generator 和 commit-time certificate tensors；
  因而无法 materialize new AP payload outcomes、certificate controller 或 selected runtime。
```

机制判断：

1. H0 成立：AP0 patch route 应停止，best legal AUC 仍只有 `0.6152`，best top273 CP precision 只有 `0.2821`。
2. H1 未成立：certificate schema rows complete，但没有 AP1-AP4 generated payloads，因此 schema 不能 official。
3. H2 未成立：AP1/AP2/AP3/AP4 generated action count 全为 `0`。
4. H3 未打开：新 AP payload outcome 没有 materialize；P4 只有 AP0 existing outcome diagnostic。
5. H4-H7 全部 gate-blocked：没有 full AP frontier、controller、selected primitive runtime 或 system integration。
6. 当前下一步必须实现真实 certificate-producing primitive：生成 payload、payload hash、certificate tensors、可 replay action apply，再跑 matched-control horizons。

最终一句话：

> v9.3.7 真实执行后停在 `R6-ActionPrimitiveRedesignRequired`：AP0 threshold/probe 路线已按计划停止，certificate schema 只能作为诊断落盘；AP1-AP4 真实 certificate-producing action primitive 尚未 materialize，因此 strict PureKAN functional 仍不能转正。
