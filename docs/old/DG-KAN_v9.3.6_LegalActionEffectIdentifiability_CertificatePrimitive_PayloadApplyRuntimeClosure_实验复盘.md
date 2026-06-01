# DG-KAN v9.3.6 Legal Action-Effect Identifiability / Certificate Primitive / PayloadApply Runtime Closure 实验复盘

> 本复盘记录 `DG-KAN_v9.3.6_LegalActionEffectIdentifiability_CertificatePrimitive_PayloadApplyRuntimeClosure_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 full oracle label、AEF/probe diagnostic 或 payload-apply microbench 写成 official system pass。

## 0. 最新结论

```text
route = R12-CertificatePrimitiveFail
base_candidate = LQ-t2-h256
success_v9360_strict_purekan_functional = False
success_v9360_full_functional = False
success_v9360_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9360_legal_action_effect_identifiability_certificate_runtime_first_20260514T030000Z/
```

核心结论：

1. P0 复现 v9.3.5 boundary：source route = `R8-StaticObservabilityFail`，full control outcome ready = `1`，control-positive oracle pass = `1`，system pass = `0`。
2. P1 CP label decomposition 过：weak CP row = `893`，coverage = `0.0984347442680776`，coverage LCB = `0.09247342078415305`，V_ctrl LCB = `0.2833107634852314`。
3. P1 同时显示 AP0 value 不 horizon-robust：strong CP row = `353`，strong coverage = `0.03891093474426808`；horizon-robust action = `13`，horizon-robust coverage = `0.004298941798941799`。
4. P1 分解出 short-only action = `201`，long-risk action = `294`，说明 weak CP frontier 不能直接写成 robust functional advantage。
5. P2 legal observability capacity 未过：best legal capacity feature = `F4-StateNLL`，AUC_CP = `0.6151712728688206`，best top273 CP precision = `0.12454212454212454`。
6. P3 action-effect fingerprint 未过：measured action = `864`，best AEF = `AEF6-ProbeReliability`，AUC_CP = `0.5030687962043473`，top273 CP precision = `0.28205128205128205`，cost q90 = `2.4969042278826237 ms`。
7. P4 cheap/legal microprobe 未过：best probe 仍是 `MP0-v9350-PR5-reference`，AUC_CP = `0.5030687962043473`，PR lift = `1.068825910931174`，cost q90 = `2.4969042278826237 ms`。
8. P5 controller 没有运行 threshold search：P2/P3/P4 legal signal 均失败，不能继续做 controller patch。
9. P6 判定 AP0 = oracle-good but non-identifiable：`ap0_oracle_good_but_nonidentifiable = 1`，dominant failure mode = `OF2-action_effect_feature_weak`。
10. P6 certificate primitive 未 materialize：`certificate_primitive_smoke_pass = 0`，下一步需要实现 certificate-producing action primitive，而不是事后猜 AP0 opaque action。
11. P7 payload apply microbench 有好信号但仍是 diagnostic：best runtime candidate = `RT3-LastLayerOnlyDiagnostic`，payload apply q90 = `0.0245068222284317 ms`，step ratio q90 = `1.0305048474583738`，error L∞ = `0.0`，cosine min = `0.9999998807907104`。
12. P7 没有 official：没有 selected legal controller，runtime path 是 microbench，不是 system selected-controller runtime；`payload_apply_runtime_pass = 0`。
13. P8 system controller 未打开：`official_eligible = 0`，`system_legal_controller_pass = 0`，reason = `ap0_legal_identifiability_failed_certificate_primitive_not_materialized`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9360_legal_action_effect_identifiability_certificate_runtime.py` | v9.3.6 runner；读取 v9.3.5 full control universe，执行 CP label decomposition、legal observability capacity、AEF/probe audit、observability failure autopsy、payload apply microbench 与 P8 gate |

代码检查：

```text
python -m py_compile experiments/run_v9360_legal_action_effect_identifiability_certificate_runtime.py
```

正式运行：

```bash
python experiments/run_v9360_legal_action_effect_identifiability_certificate_runtime.py \
  --out-dir results/real_rerun_20260506/v9360_legal_action_effect_identifiability_certificate_runtime_first_20260514T030000Z \
  --fresh --device auto --data-root data --seed 1314 --payload-runtime-actions 256
```

运行结果：

```json
{
  "best_aef_auc": 0.5030687962043473,
  "out_dir": "results/real_rerun_20260506/v9360_legal_action_effect_identifiability_certificate_runtime_first_20260514T030000Z",
  "route": "R12-CertificatePrimitiveFail",
  "weak_CP_coverage": 0.0984347442680776
}
```

## 2. Route

`route_decision.json` 摘要：

```json
{
  "route": "R12-CertificatePrimitiveFail",
  "base_candidate": "LQ-t2-h256",
  "source_route_v9350": "R8-StaticObservabilityFail",
  "full_control_outcome_ready": 1,
  "control_positive_oracle_pass_v9350": 1,
  "p1_cp_decomposition_pass": 1,
  "weak_CP_row_count": 893,
  "weak_CP_coverage": 0.0984347442680776,
  "weak_CP_coverage_lcb": 0.09247342078415305,
  "weak_CP_V_ctrl_lcb": 0.2833107634852314,
  "strong_CP_row_count": 353,
  "strong_CP_coverage": 0.03891093474426808,
  "horizon_robust_CP_action_count": 13,
  "horizon_robust_CP_coverage": 0.004298941798941799,
  "short_only_CP_action_count": 201,
  "long_risk_action_count": 294,
  "legal_observability_capacity_pass": 0,
  "best_legal_capacity_feature": "F4-StateNLL",
  "best_legal_capacity_auc_CP": 0.6151712728688206,
  "aef_feature_pass": 0,
  "best_aef_feature": "AEF6-ProbeReliability",
  "best_aef_auc_CP": 0.5030687962043473,
  "microprobe_pass": 0,
  "best_microprobe_id": "MP0-v9350-PR5-reference",
  "best_microprobe_auc_CP": 0.5030687962043473,
  "best_microprobe_cost_q90": 2.4969042278826237,
  "control_positive_controller_pass": 0,
  "ap0_oracle_good_but_nonidentifiable": 1,
  "certificate_primitive_smoke_pass": 0,
  "best_runtime_candidate_id": "RT3-LastLayerOnlyDiagnostic",
  "payload_apply_time_ms_q90": 0.0245068222284317,
  "step_ratio_q90": 1.0305048474583738,
  "payload_apply_runtime_pass": 0,
  "official_eligible": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "legal_action_effect_identifiability_failed",
  "next_required_implementation": "implement_certificate_producing_action_primitive_and_selected_controller_runtime"
}
```

判断：v9.3.6 没有回退 v9.3.5 的 oracle frontier；它把 blocker 推到 AP0 legal identifiability failure。当前不是缺 good action，而是 good action 在 AP0 中没有可部署 certificate / legal action-effect signal。

## 3. P1 CP label decomposition

Artifacts：

```text
p1_cp_label_decomposition_v9360.csv
cp_horizon_structure_trace_v9360.csv
```

Summary：

```text
action_count = 2876
weak_CP_action_count = 734
weak_CP_row_count = 893
weak_CP_coverage = 0.0984347442680776
weak_CP_coverage_lcb = 0.09247342078415305
weak_CP_V_ctrl_lcb = 0.2833107634852314
strong_CP_action_count = 34
strong_CP_row_count = 353
strong_CP_coverage = 0.03891093474426808
horizon_robust_CP_action_count = 13
horizon_robust_CP_coverage = 0.004298941798941799
short_only_CP_action_count = 201
long_risk_action_count = 294
horizon_missing_action_count = 0
weak_CP_label_quality_pass = 1
p1_cp_decomposition_pass = 1
```

判断：H1 成立但更精确了。AP0 weak CP density 足够，但 horizon-robust density 极低；controller 不能只追 weak CP，否则可能选到 short-only / long-risk action。

## 4. P2 legal observability capacity

Artifacts：

```text
p2_legal_observability_capacity_audit.csv
legal_feature_capacity_trace_v9360.csv
```

Summary：

```text
feature_group_count = 5
feature_count = 13
best_legal_capacity_feature = F4-StateNLL
best_legal_capacity_group = C_state_tail
best_legal_capacity_auc_CP = 0.6151712728688206
best_legal_capacity_pr_lift = 1.2117767597947864
best_top273_feature = G0-StateMarginP10
best_top273_CP_precision = 0.12454212454212454
legal_observability_capacity_weak_pass = 0
legal_observability_capacity_pass = 0
```

判断：P2 未过。state-tail/static/payload/support 这些 legal marginal features 仍远低于 `AUC >= 0.68` weak gate，更不到 strong gate；不能继续把这一路做 threshold patch。

## 5. P3 action-effect fingerprint

Artifacts：

```text
p3_action_effect_fingerprint_factory.csv
aef_feature_trace_v9360.csv
```

Summary：

```text
aef_feature_count = 6
measured_action_count = 864
full_action_count = 2876
missing_fingerprint_action_count = 2012
best_aef_feature = AEF6-ProbeReliability
best_aef_group = G6_minimal_certificate
best_aef_auc_CP = 0.5030687962043473
best_aef_pr_lift = 1.068825910931174
best_aef_top273_CP_precision = 0.28205128205128205
best_aef_cost_q90 = 2.4969042278826237
aef_feature_pass = 0
```

判断：本轮没有找到有效 action-effect fingerprint。AEF 只覆盖 v9.3.5 已测 probe 子集 `864` actions，而且最好的 AUC 接近随机；不能据此选择 official controller。

## 6. P4 cheap legal microprobe

Artifacts：

```text
p4_cheap_legal_microprobe.csv
microprobe_trace_v9360.csv
```

Summary：

```text
microprobe_measured = 1
probe_action_count = 864
best_microprobe_id = MP0-v9350-PR5-reference
best_microprobe_auc_CP = 0.5030687962043473
best_microprobe_pr_lift = 1.068825910931174
best_microprobe_top273_CP_precision = 0.28205128205128205
best_microprobe_cost_q90 = 2.4969042278826237
microprobe_pass = 0
```

判断：P4 未过。重用 v9.3.5 measured probe 的各个 low-level response 视角后，信号仍接近随机，而且成本仍远高于 `0.20 ms` budget。

## 7. P5 / P6 controller 与 certificate primitive boundary

P5：

```text
controller_measured = 0
controller_id = not_selected_observability_blocked
control_positive_controller_pass = 0
reason = P2_P3_P4_legal_observability_failed
```

P6：

```text
ap0_oracle_good_but_nonidentifiable = 1
dominant_failure_mode = OF2-action_effect_feature_weak
certificate_primitive_smoke_pass = 0
next_required_implementation = implement_certificate_producing_action_primitive
```

判断：P5 正确不运行 controller threshold search。P6 给出本轮主结论：AP0 是 oracle-good but legally opaque；下一步必须实现 AP1/AP2/AP3/AP4 这类 certificate-producing primitive，而不是继续事后猜 AP0 action。

## 8. P7 payload apply runtime

Artifacts：

```text
p7_payload_apply_runtime_closure.csv
payload_runtime_component_trace_v9360.csv
```

Measured reference：

```text
source_v9350_payload_apply_q90 = 0.7623820565640926
source_v9350_step_ratio_q90 = 2.068736718507292
```

Best measured microbench：

```text
best_runtime_candidate_id = RT3-LastLayerOnlyDiagnostic
payload_runtime_action_count = 256
best_payload_apply_time_ms_q90 = 0.0245068222284317
best_step_ratio_q90 = 1.0305048474583738
best_payload_apply_error_linf_max = 0.0
best_payload_apply_cosine_min = 0.9999998807907104
payload_apply_runtime_pass = 0
reason = runtime_microbench_not_official_without_legal_controller
```

判断：P7 有重要工程信号：payload apply microbench 可以很快。但它不是 official selected-controller online runtime；因此没有把它写成 `runtime pass`。

## 9. P8 system controller boundary

Artifact：

```text
p8_system_legal_controller_v9360.csv
system_controller_trace_v9360.csv
```

Boundary：

```text
system_candidate_id = SYS-v9360-legal-action-effect-identifiability
controller_id = not_selected_observability_blocked
runtime_candidate_id = RT3-LastLayerOnlyDiagnostic
primitive_id = AP0-current-action-reference
official_eligible = 0
control_positive_controller_pass = 0
payload_apply_runtime_pass = 0
system_legal_controller_pass = 0
reason = ap0_legal_identifiability_failed_certificate_primitive_not_materialized
```

判断：P8 未打开。没有 dataset_name，没有 future outcome feature，没有 diagnostic promoted to official；system failure 来自 legal identifiability，而不是 oracle density absent。

## 10. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p9_diagnostic_causal_scout.csv` | `P8_system_controller_not_official` |
| `p10_leave_dataset_stratum_out.csv` | same |
| `p11_official_paired_replay.csv` | same |
| `p12_short_full_sampleeff_continual_robustness.csv` | same |

没有把 CP oracle、AEF/probe diagnostic、payload apply microbench 或 certificate primitive placeholder 写成 LDO/paired replay/short/full success。

## 11. No-fake audit

```text
rows_checked = 3799
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
full_control_outcome_ready = 1
cp_decomposition_pass = 1
legal_observability_capacity_pass = 0
aef_feature_pass = 0
microprobe_pass = 0
control_positive_controller_pass = 0
certificate_primitive_smoke_pass = 0
payload_apply_runtime_pass = 0
system_legal_controller_pass = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
uses_test_or_validation = 0
uses_future_outcome_for_features = 0
source_measured_gap/formula_proxy = 0/0
diagnostic_promoted_to_official = 0
```

Failure table：

```text
F1_AP0_legal_identifiability_fail = 1
F2_static_capacity_fail = 1
F3_aef_feature_fail = 1
F4_microprobe_fail = 1
F5_certificate_primitive_missing = 1
F6_payload_runtime_not_official = 1
primary_blocker = legal_action_effect_identifiability_failed
```

## 12. Hash

| artifact | SHA256 |
|---|---|
| plan | `6b1921ce898e7afb70c392c9953f8fd812a67af47765c9ce7b6f41d1a22a716d` |
| runner | `0d5cda4f9824f28d1496e703626e4dab8d4aa25443604aefe01bd41bb3cbf445` |
| run manifest | `86c84bcd5627ddaf6f1cb08e86e2b520353f240f4a6fd653dba3e62b55792307` |
| route | `962868bc80f1fa206ee8f2f35e7dffef9d491cfffae989de8884ce9ff448abb0` |
| P0 reanalysis | `aed6dbb408414bf7db6b2914c0836ca2438b5a032cfa9bd0ba490ef8922eb693` |
| P1 CP decomposition | `8930ea6c452ad6d99bc89eb27f902452579102feb1de991ee7d2e7b4d32f00c0` |
| P2 observability | `63110215ee88d1919334b8e17cca71e5074ac7d2b217479752c148e43ad80024` |
| P3 AEF | `b22589f17301823f76cdafaa2d57058f94add360255f639ee86557f1117bf862` |
| P4 microprobe | `4c91d8639fe02ac80da69e0abfeb8daa996ae409d25c9d1bc461a5e33ba69164` |
| P6 autopsy | `1bf364e1d952fbbe49d424b8b8f72b77b27bac601e8f5d05b10004d6d99656df` |
| P7 runtime | `9b73f4ce88c4826c6c5286410673a4a3691d958a3efa95eabbae992be49c2cee` |
| P8 system | `8b2550adb089ca67bd4a479ef6b4e0ec9db523bf75ab0b4cec6723c3effdbb54` |
| contract audit | `3081ca542bd4c8993975c7a067f06314aa3dd441615b7645ed947ad714534748` |
| provenance audit | `9aff7809041a2c97005f5017b902119e9ed1003dd6c628f3c2fd8e759884361f` |
| failure table | `db8ba0b4d41ae0dc3f0492e6ee1da2f51bad10dd37342da983532d6fd51d8a87` |

## 13. 最终分析结论

v9.3.6 的真实推进是：

```text
v9.3.5:
  full control-positive frontier 已证明存在；
  但 static/probe observability 与 payload runtime 未闭合。

v9.3.6:
  将 CP label 拆成 weak / strong / horizon-robust / short-only / long-risk；
  确认 AP0 weak CP density 足够，但 horizon robustness 很弱；
  legal marginal features、action-effect fingerprints、cheap microprobe 均未识别 CP frontier；
  payload apply microbench 显示实现层面有优化空间，但没有 official controller runtime。
```

机制判断：

1. H1 成立但有边界：weak CP coverage = `0.0984`，但 horizon-robust coverage 只有 `0.0043`。
2. H2 未成立：当前 legal feature sigma-algebra 没有足够 CP information，best AUC 只有 `0.6152`。
3. H3 未成立：已测 AEF/probe response 接近随机，best AEF AUC = `0.5031`。
4. H4 未成立：cheap microprobe 没有改善，且成本 q90 仍 `2.4969 ms`。
5. H5 只在 diagnostic microbench 上有希望：payload apply q90 可到 `0.0245 ms`，但不是 selected-controller official path。
6. H6 触发：AP0 是 oracle-good but legally non-identifiable；下一步必须实现 certificate-producing action primitive。
7. P8 未打开：不能进入 official paired replay 或 short/full functional validation。

最终一句话：

> v9.3.6 真实执行后停在 `R12-CertificatePrimitiveFail`：AP0 的 full oracle frontier 仍存在，但当前 legal action-effect observable、AEF 和 microprobe 都看不见它；payload apply 有诊断级 runtime 改善，但没有 official controller，strict PureKAN functional 仍未成功。
