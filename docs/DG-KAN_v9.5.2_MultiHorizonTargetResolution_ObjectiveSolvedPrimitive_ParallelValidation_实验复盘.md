# DG-KAN v9.5.2 Multi-Horizon Target Resolution / Objective-Solved Primitive / Parallel Validation 实验复盘

> 本复盘记录 `DG-KAN_v9.5.2_MultiHorizonTargetResolution_ObjectiveSolvedPrimitive_ParallelValidation_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 target lattice diagnostic、legal upper-bound probe、mechanism anatomy、APY preflight、APY smoke、certificate diagnostic、runtime boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-CanonicalActionDensityInsufficientForMultiHorizonTarget
base_candidate = LQ-t2-h256
success_v9520_strict_purekan_functional = False
success_v9520_full_functional = False
success_v9520_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9520_multihorizon_target_resolution_objective_solved_primitive_first_20260515T020000Z/
```

核心结论：

1. P0 复现 v9.5.1 boundary：source route = `R1-TargetConflictUnresolved`，canonical truth base ready = `1`，v9.5.1 system pass = `0`。
2. P1 target lattice 真实扫描 `8000` 个候选 target，但 official / weak pass candidate 均为 `0`。
3. P1 选出的 best diagnostic target = `TE-a-0.05-b-0.05-g-0.05-e0.00-r0.00-w0.300.400.30`，action count = `27`，coverage = `0.009388038942976356`。
4. P1 best target 的 value/risk 有信号：V_integrated LCB = `0.24423130340766275`，h240 long-risk = `0.0`；但 coverage 低于 `0.03`，bad-event = `0.1111111111111111`，null-event = `0.25925925925925924`。
5. P1 terminal blocker = `no_target_candidate_satisfies_weak_pass`；route 因此停在 `R1-CanonicalActionDensityInsufficientForMultiHorizonTarget`。
6. P2 leaveout sanity 未过：target_leaveout_sanity_pass = `0`，main_dataset_weak_pass_count = `0`，value LCB drop max = `0.26589384528870547`。
7. P3 legal upper-bound probe v3 未过：best = `UB7-small-mlp-diagnostic-legal-v3`，AUC_target = `0.5228085228085229`，TopK64 target precision = `0.0`，TopK64 long-risk = `0.609375`。
8. P4 mechanism anatomy 未过：best cluster purity = `0.0625`，best AUC target = `0.5223080223080223`，dominant miss reason = `M8-outcome-only-pattern`。
9. P5 APX under selected target 重新打分未过：best APX = `APX7-EnsembleIntersectionPrimitive`，target precision = `0.015625`，V_integrated LCB = `-1.3027380854267743`，h240 long-risk = `0.84375`。
10. P6 APY1-APY8 objective-solved primitive spec/preflight 过：generated actions = `512`，payload/certificate hash missing = `0 / 0`，action apply L∞ max = `0.0`，no-transform equivalence pass = `1`。
11. P7 APY branch-horizon smoke 完整落盘：expected/actual rows = `9216 / 9216`，unresolved exception = `0`。
12. P7 APY smoke 未过：best = `APY3-LongRiskBarrierPrimitive`，target precision = `0.046875`，V_integrated LCB = `-0.9770587887784598`，h240 long-risk = `0.71875`。
13. P8 source-to-generated preservation/improvement 未过：best APY3 new-positive rate = `0.046875`，long-risk created rate = `0.421875`，Damage integrated LCB = `-0.6279387897953833`。
14. P9 vector certificate v5 未过：best = `CERT30-BasisEdgeLocalityCertificate`，AUC_target = `0.691913214990138`，TopK64 target precision = `0.015625`，TopK64 long-risk = `0.671875`。
15. P10-P16 gate-blocked：没有 source controller、selected runtime、official paired replay、short/full training 或 continual validation。
16. P15 Base-Acc Sentinel 复用前序真实 sentinel：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
17. 当前 primary blocker：`no_lattice_target_satisfies_weak_pass`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9520_multihorizon_target_resolution_objective_solved_primitive.py` | v9.5.2 runner；读取 v9.5.1/v9.5.0/v9.4.9/v9.4.8 canonical artifacts，执行 target lattice resolution、leaveout sanity、legal upper-bound probe v3、mechanism anatomy v3、APX rescore、APY1-APY8 objective-solved primitive、APY branch-horizon smoke、vector certificate v5、controller/runtime/system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9520_multihorizon_target_resolution_objective_solved_primitive.py
```

正式运行：

```bash
python experiments/run_v9520_multihorizon_target_resolution_objective_solved_primitive.py \
  --out-dir results/real_rerun_20260506/v9520_multihorizon_target_resolution_objective_solved_primitive_first_20260515T020000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --probe-action-limit 2876 --apy-actions-per-primitive 64
```

运行结果：

```json
{
  "best_apy_primitive": "APY3-LongRiskBarrierPrimitive",
  "best_apy_target_precision": 0.046875,
  "certificate_effect_valid_pass": 0,
  "out_dir": "results/real_rerun_20260506/v9520_multihorizon_target_resolution_objective_solved_primitive_first_20260515T020000Z",
  "route": "R1-CanonicalActionDensityInsufficientForMultiHorizonTarget",
  "selected_target_count": 27,
  "system_legal_controller_pass": 0,
  "weak_target_candidate_count": 0
}
```

## 2. Route

`route_decision_v9520.json` 摘要：

```json
{
  "route": "R1-CanonicalActionDensityInsufficientForMultiHorizonTarget",
  "source_route_v9510": "R1-TargetConflictUnresolved",
  "canonical_full_control_outcome_ready": 1,
  "target_lattice_candidate_count": 8000,
  "official_target_candidate_count": 0,
  "weak_target_candidate_count": 0,
  "selected_target_id": "TE-a-0.05-b-0.05-g-0.05-e0.00-r0.00-w0.300.400.30",
  "selected_target_action_count": 27,
  "selected_target_coverage": 0.009388038942976356,
  "selected_target_V_integrated_lcb": 0.24423130340766275,
  "selected_target_h240_longrisk": 0.0,
  "official_target_pass": 0,
  "weak_target_pass": 0,
  "target_leaveout_sanity_pass": 0,
  "legal_upper_bound_probe_pass": 0,
  "best_probe_id": "UB7-small-mlp-diagnostic-legal-v3",
  "best_probe_AUC_target": 0.5228085228085229,
  "mechanism_pass": 0,
  "apx_rescore_pass": 0,
  "apy_implementation_pass": 1,
  "apy_weak_pass": 0,
  "best_apy_primitive_id": "APY3-LongRiskBarrierPrimitive",
  "best_apy_target_precision": 0.046875,
  "best_apy_V_integrated_lcb": -0.9770587887784598,
  "best_apy_h240_longrisk": 0.71875,
  "certificate_effect_valid_pass": 0,
  "best_certificate_id": "CERT30-BasisEdgeLocalityCertificate",
  "best_certificate_AUC_target": 0.691913214990138,
  "source_controller_pass": 0,
  "selected_runtime_pass": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "no_lattice_target_satisfies_weak_pass"
}
```

判断：v9.5.2 的关键不是 APY implementation failure，而是 P1 没有找到满足 density / bad-null / long-risk / value / support 全部约束的 official 或 weak target。后续 P3-P9 是必要的并行诊断，但不能越过 P1 gate。

## 3. P0 v9.5.1 boundary

Artifact：

```text
p0_v9510_boundary_reproduction.csv
```

Summary：

```text
route_v9510 = R1-TargetConflictUnresolved
source_route_v9500 = R9-PrimitiveFamilyResetRequired
canonical_full_control_outcome_ready = 1
multi_horizon_objective_pass = 0
T_A/T_B/T_C/T_D counts = 120 / 25 / 16 / 64
best_raw_probe_id = UB4-gradient-action-bilinear-probe
best_raw_AUC_TB = 0.44422307962118557
best_raw_TopK64_precision_TB = 0.0
best_cluster_purity_TB = 0.0625
apx_primitive_family_spec_pass = 1
apx_preflight_pass = 1
apx_branch_horizon_rows_expected/actual = 12288 / 12288
best_apx_primitive = APX7-EnsembleIntersectionPrimitive
best_apx_h20_weak_CP = 0.125
best_apx_h20_V_ctrl_lcb = -1.6380800012017955
best_apx_h240_longrisk = 0.84375
best_certificate = CERT24-AdamWConflictReliefBound
best_certificate_AUC_TB = 0.6041257367387033
best_certificate_TopK64_TB_precision = 0.0
system_legal_controller_pass = 0
p0_pass = 1
```

判断：P0 pass。v9.5.2 没有跳过 v9.5.1 的 target conflict boundary。

## 4. P1 target lattice resolution

Artifacts：

```text
p1_target_lattice_resolution.csv
p1_target_lattice_trace_v9520.csv
```

Summary：

```text
target_lattice_candidate_count = 8000
official_target_candidate_count = 0
weak_target_candidate_count = 0
selected_target_id = TE-a-0.05-b-0.05-g-0.05-e0.00-r0.00-w0.300.400.30
selected_target_action_count = 27
selected_target_coverage = 0.009388038942976356
coverage_lcb = 0.006460034563184731
h20/h80/h240 weak rate = 0.8148148148148148 / 0.7777777777777778 / 0.8148148148148148
h20/h80/h240 V_lcb = 0.20138636510067395 / 0.2892031636318345 / 0.10183648386076839
V_integrated_mean = 0.35254755330151294
V_integrated_lcb = 0.24423130340766275
h240_longrisk_rate = 0.0
bad_event_rate = 0.1111111111111111
null_event_rate = 0.25925925925925924
accepted_family_count = 17
accepted_stratum_count = 26
accepted_dataset_count = 3
support_balance_pass = 1
jaccard_TA/TB/TC/TD = 0.176 / 0.6774193548387096 / 0.5925925925925926 / 0.3
reason_if_failed = coverage_lt_0.03;bad_gt_0.05;null_gt_0.15
```

判断：P1 是本轮 terminal blocker。Best target 在 integrated value 与 h240 long-risk 上比 v9.5.1 的 T_A/T_B/T_C/T_D 更平衡，但 action density 只有 `27 / 2876`，coverage 远低于 `0.03`；bad/null 也超过 gate。

## 5. P2 target leaveout sanity

Artifact：

```text
p2_target_leaveout_sanity.csv
```

Summary：

```text
target_leaveout_sanity_pass = 0
leaveout_row_count = 22
dataset_with_zero_coverage_count = 0
main_dataset_weak_pass_count = 0
longrisk_heldout_max = 0.0
value_lcb_drop_max = 0.26589384528870547
```

Dataset leaveout：

| heldout dataset | accepted count | coverage | V_integrated LCB | long-risk | bad | null | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Fashion-MNIST` | `8` | `0.0065252854812398045` | `0.4778527243863628` | `0.0` | `0.25` | `0.125` | `0` |
| `KMNIST` | `10` | `0.012077294685990338` | `0.15197571817525968` | `0.0` | `0.0` | `0.1` | `0` |
| `MNIST` | `9` | `0.010948905109489052` | `0.06824570529138442` | `0.0` | `0.1111111111111111` | `0.5555555555555556` | `0` |

判断：P2 未过。Best target 的 leaveout 没有出现 long-risk 爆炸，但 coverage 太低且 dataset leaveout weak pass 为 `0`。

## 6. P3 legal upper-bound probe v3

Artifact：

```text
p3_legal_upper_bound_probe_v3.csv
```

Summary：

```text
probe_count = 9
best_probe_id = UB7-small-mlp-diagnostic-legal-v3
best_AUC_target = 0.5228085228085229
best_AUC_longrisk = 0.5000412542514798
best_TopK64_target_precision = 0.0
best_TopK64_longrisk = 0.609375
legal_upper_bound_probe_pass = 0
legal_upper_bound_probe_weak_pass = 0
```

Probe rows：

| probe | AUC target | TopK64 target | TopK64 long-risk | pass |
|---|---:|---:|---:|---:|
| `UB0-raw-canonical-scalar-v3` | `0.40535340535340536` | `0.0` | `0.796875` | `0` |
| `UB1-payload-shape-v3` | `0.3973713973713974` | `0.0` | `0.6875` | `0` |
| `UB2-hard-tail-v3` | `0.3494923494923495` | `0.0` | `0.71875` | `0` |
| `UB3-raw-logit-tensor-v3` | `0.4805844805844806` | `0.0` | `0.671875` | `0` |
| `UB4-gradient-action-bilinear-v3` | `0.4520884520884521` | `0.0` | `0.765625` | `0` |
| `UB5-support-memory-v3` | `0.3389103389103389` | `0.0` | `0.59375` | `0` |
| `UB6-basis-edge-v3` | `0.3657033657033657` | `0.0` | `0.71875` | `0` |
| `UB7-small-mlp-diagnostic-legal-v3` | `0.5228085228085229` | `0.0` | `0.609375` | `0` |
| `UB8-full-legal-tensor-v3` | `0.32545532545532546` | `0.0` | `0.78125` | `0` |

判断：P3 未过。即使使用 small MLP diagnostic legal probe，TopK64 target precision 仍为 `0.0`。

## 7. P4 mechanism anatomy v3

Artifact：

```text
p4_mechanism_anatomy_v3.csv
```

Summary：

```text
positive_target_count = 27
near_miss_negative_count = 2849
cluster_count = 9
best_cluster_id = M5-low-support
best_cluster_purity = 0.0625
best_cluster_longrisk_rate = 0.6875
best_AUC_target = 0.5223080223080223
dominant_miss_reason = M8-outcome-only-pattern
mechanism_pass = 0
```

判断：P4 未过。Target positives 太稀疏，机制 cluster 的 target purity 仍只有 `0.0625`，不能蒸馏为 official mechanism。

## 8. P5 APX rescore under selected target

Artifact：

```text
p5_apx_rescore_under_target.csv
```

Summary：

```text
target_id = TE-a-0.05-b-0.05-g-0.05-e0.00-r0.00-w0.300.400.30
primitive_count = 8
best_primitive_id = APX7-EnsembleIntersectionPrimitive
best_target_precision = 0.015625
best_V_integrated_lcb = -1.3027380854267743
best_h240_longrisk = 0.84375
apx_rescore_pass = 0
```

判断：P5 未过。即便换成 P1 selected target，v9.5.1 APX family 的 value/risk 仍明显不达标。

## 9. P6 APY primitive spec / deterministic preflight

Artifact：

```text
p6_apy_primitive_spec_preflight.csv
```

Summary：

```text
primitive_count = 8
generated_action_count_expected/actual = 512 / 512
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_linf_max = 0.0
preflight_single_action_pass = 1
preflight_three_action_pass = 1
preflight_sixteen_action_pass = 1
no_transform_equivalence_pass = 1
negative_control_generated = 1
negative_control_divergence_present = 1
apy_implementation_pass = 1
```

判断：P6 pass。APY1-APY8 的 payload/certificate/action apply/preflight 链路不是本轮 blocker。

## 10. P7 APY branch-horizon smoke

Artifacts：

```text
p7_apy_branch_horizon_smoke_outcome.csv
apy_branch_horizon_outcome_trace_v9520.csv
```

Summary：

```text
generated_action_count_total = 512
branch_horizon_rows_expected/actual = 9216 / 9216
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1
unresolved_exception_count = 0
wallclock_sec = 441.3254356570542
rows_per_sec = 20.88254896230184
best_primitive_id = APY3-LongRiskBarrierPrimitive
best_target_precision = 0.046875
best_V_integrated_lcb = -0.9770587887784598
best_h240_longrisk = 0.71875
best_h20/h80/h240 weak = 0.140625 / 0.1875 / 0.1875
APY8_negative_control_weak_pass = 0
apy_weak_pass = 0
apy_strong_pass = 0
```

Per primitive：

| primitive | h20 weak | h80 weak | h240 weak | V_integrated LCB | h240 long-risk | target precision |
|---|---:|---:|---:|---:|---:|---:|
| `APY1-ConstrainedMultiHorizonQP` | `0.203125` | `0.171875` | `0.1875` | `-1.021495051023128` | `0.71875` | `0.015625` |
| `APY2-H80MiddleHorizonRepair` | `0.140625` | `0.09375` | `0.15625` | `-1.176527270605692` | `0.703125` | `0.0` |
| `APY3-LongRiskBarrierPrimitive` | `0.140625` | `0.1875` | `0.1875` | `-0.9770587887784598` | `0.71875` | `0.046875` |
| `APY4-AdamWCompatibleResidualQP` | `0.125` | `0.125` | `0.109375` | `-1.03322921690135` | `0.84375` | `0.0` |
| `APY5-SupportMemoryPrototypePrimitive` | `0.15625` | `0.15625` | `0.171875` | `-1.109022831509893` | `0.796875` | `0.0` |
| `APY6-BasisEdgeLocalityPrimitive` | `0.1875` | `0.140625` | `0.1875` | `-0.9843642732956008` | `0.703125` | `0.0` |
| `APY7-PortfolioMicroActionPrimitive` | `0.125` | `0.171875` | `0.140625` | `-1.2264285441317744` | `0.78125` | `0.015625` |
| `APY8-NegativeControlRandomOrthogonal` | `0.078125` | `0.171875` | `0.109375` | `-1.1542084898069274` | `0.859375` | `0.0` |

判断：P7 未过。APY materializer 完整闭合，但没有任何 primitive 形成 value-positive / horizon-safe / target-positive frontier。

## 11. P8 APY damage audit

Artifact：

```text
p8_apy_damage_audit.csv
```

Summary：

```text
best_primitive_id = APY3-LongRiskBarrierPrimitive
best_new_positive_created_rate = 0.046875
best_longrisk_created_rate = 0.421875
best_Damage_integrated_lcb = -0.6279387897953833
preservation_pass = 0
improvement_pass = 0
source_to_generated_preservation_or_improvement_pass = 0
```

判断：P8 未过。APY3 只产生少量 new positive，同时制造较高 long-risk，Damage integrated LCB 为负。

## 12. P9 vector certificate v5

Artifact：

```text
p9_vector_certificate_v5.csv
```

Summary：

```text
certificate_count = 8
best_certificate_id = CERT30-BasisEdgeLocalityCertificate
best_AUC_target = 0.691913214990138
best_AUC_longrisk = 0.5577380952380953
best_TopK64_target_precision = 0.015625
best_TopK64_longrisk = 0.671875
best_ECE_target = 0.6411295996902977
certificate_effect_valid_pass = 0
```

Certificate rows：

| certificate | AUC target | TopK64 target | TopK64 long-risk |
|---|---:|---:|---:|
| `CERT25-TargetLatticeScoreCertificate` | `0.6915187376725839` | `0.015625` | `0.671875` |
| `CERT26-MultiHorizonVectorMarginCertificate` | `0.632741617357002` | `0.015625` | `0.640625` |
| `CERT27-LongRiskBarrierCertificate` | `0.5893491124260355` | `0.015625` | `0.625` |
| `CERT28-LeaveoutStabilityCertificate` | `0.31676528599605525` | `0.0` | `0.65625` |
| `CERT29-APYPayloadNormCertificate` | `0.5518737672583827` | `0.0` | `0.6875` |
| `CERT30-BasisEdgeLocalityCertificate` | `0.691913214990138` | `0.015625` | `0.671875` |
| `CERT31-PortfolioConsensusCertificate` | `0.6658777120315582` | `0.015625` | `0.65625` |
| `CERT32-NegativeControlGuardCertificate` | `0.6272189349112426` | `0.015625` | `0.625` |

判断：P9 未过。Best AUC 看起来较高，但 TopK64 target precision 只有 `0.015625` 且 long-risk 高，不能作为 controller gate。

## 13. P10-P16 boundary

P10：

```text
p10_minimal_source_certificate_controller.csv = not_run
reason = P1_or_P7_or_P9_gate_failed
source_controller_pass = 0
```

P11：

```text
p11_selected_controller_runtime.csv = not_run
reason = P10_controller_not_selected
selected_runtime_pass = 0
```

P12：

```text
target_lattice_pass = 0
apy_primitive_generation_pass = 1
apy_branch_horizon_outcome_pass = 1
certificate_effect_valid_pass = 0
controller_pass = 0
runtime_pass = 0
official_eligible = 0
system_legal_controller_pass = 0
reason = target_absent_or_upstream_gate_failed
```

P13-P16：

| artifact | status / reason |
|---|---|
| `p13_leaveout_boundary.csv` | `not_run`, `P12_system_controller_not_official` |
| `p14_official_paired_replay_boundary.csv` | `not_run`, `P12_system_controller_not_official` |
| `p15_short_full_base_acc_boundary.csv` | Base-Acc Sentinel complete, functional short run `not_run` |
| `p16_continual_boundary.csv` | `not_run`, `P15_short_full_not_open` |

判断：没有把 APY preflight、完整 APY branch-horizon rows、certificate diagnostic 或 Base-Acc Sentinel 写成 official controller/runtime/system/downstream pass。

## 14. Base-Acc Sentinel

Artifact：

```text
p15_short_full_base_acc_boundary.csv
```

Summary：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
model_count = 4
sentinel_complete = 1
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
base_acc_reused_from_v9470/v9480/v9490/v9500/v9510 = 1/1/1/1/1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 15. No-fake audit

```text
rows_checked = 18204
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

Contract audit：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
v9510_boundary_pass = 1
target_lattice_resolution_pass = 0
target_leaveout_sanity_pass = 0
legal_upper_bound_probe_pass = 0
mechanism_pass = 0
apx_rescore_pass = 0
apy_implementation_pass = 1
apy_smoke_weak_pass = 0
source_to_generated_damage_pass = 0
certificate_effect_valid_pass = 0
source_controller/selected_runtime/system = 0/0/0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_old_table_for_official = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R1-CanonicalActionDensityInsufficientForMultiHorizonTarget
F0_reproduction_fail = 0
F1_target_absent = 1
F2_target_exists_legal_invisible = 0
F3_apx_target_mismatch_only = 0
F4_apx_universal_fail = 1
F5_apy_implementation_fail = 0
F6_apy_value_fail = 0
F7_certificate_fail = 0
F8_controller_runtime_blocked = 1
F9_base_acc_catastrophic = 0
primary_blocker = no_lattice_target_satisfies_weak_pass
```

## 16. Hash

| artifact | SHA256 |
|---|---|
| plan | `d40f17362fd85c28b85f8f3f7a3472cb075a9c62304ebe1d80c6325927076c61` |
| runner | `78153afc53878f5ec08430775578e6532a76d68cd9aab850fe94b4f5bd3fc53d` |
| run manifest | `bc14cd34189309f5fbdfe57c58d66c5ee7cf6d215554d8449c9ba28c442c5d1e` |
| route | `7e6c7e99240c96155f4dd85293c6ae8d3a8739d691ae4e8a934bef2f4cdffda2` |
| P0 boundary | `b3ca427a78f84894168bafedc7bf58b4036c7cc08eb9f11c751fd67ba61f1c93` |
| P1 target resolution | `b65edd64cbc14bbbd8fd043312bda7094a29f52f722e87f6d0129043e92626ef` |
| P1 target trace | `6dce7e5bab7b9e8fb558ecba13325f38cb2fdfeb6b808a8f6bfcdbfc59a5562e` |
| P2 leaveout | `2ce8d1c2f62f388ab7ce7e18f5e96826b12c4c2b8fdb06a9b05e2bd7f9d9b27d` |
| P3 legal probe | `8cdb0ed060e2c1ce1a0c7f145faf628cb2811bea77ba035504944d07b0a13705` |
| P4 mechanism | `e9420a5757f250c52e32953fd278ae026455a09157daf745ac7b5aaf07641027` |
| P5 APX rescore | `445ae6faa0a096a1ea933d47b2a5d21414bbcb26de9600a7bfa754c009b39d30` |
| P6 APY spec/preflight | `7f8e9209240e1268a077ae4caf18e5887765a796f1e266c3fafaf35a8f2d9a07` |
| P7 APY smoke | `55faf0af63f4399636b64aa5e4a4bc7b5cad41133ca1f84dcda51ea690409e3a` |
| P8 APY damage | `91afe661346d43f1f548858673a1a6afc3baf529349d596e044c0e82422d3467` |
| P9 certificate | `aaf18302e3f5745209584ac638787a6b2706f16c70eb4af919cb2a2a8e73cbf1` |
| P10 controller | `939a81c257ed5aa5cb5a9b80f1359a17a5be12405d77a1ebe7f432e3dd9c6c84` |
| P11 runtime | `62385b031bf500ecce70eb6e3aaf61575aa3d66c288b28dea963300930bdc9a9` |
| P12 system | `e4e93b9dac8b2a14ff9ba398590cec61a796cfe006c2752e288c797ef9a131ba` |
| P15 Base-Acc Sentinel | `22b7f2c7384b94210c88176170f99c08e8786445c805f58ca535f42be644d602` |
| no-fake audit | `ddaa6ea83ce34e46544d914c11c6ceae7b27c208f39f3191bbf39aacc6073334` |
| contract audit | `067df42b9a134edc7cca9ce39006978c3e9673c70e312cad9b0349d9981c2516` |
| failure table | `294b08fb992ad05e411a66d3c05f36288309b7ada69def6dba46b199e8a3fccf` |

## 17. 最终分析结论

v9.5.2 的真实推进是：

```text
v9.5.1:
  APX1-APX8 工程链路闭合；
  但 multi-horizon target 出现严格性与支持度冲突；
  legal probe、mechanism、APX generator、certificate 均未打开 controller。

v9.5.2:
  将 target 从四个手工候选扩展为 8000 点 lattice；
  找到一个 value/risk 更平衡的 diagnostic target；
  但没有任何 target 满足 weak pass；
  APY1-APY8 工程链路闭合并完整落盘；
  但 APY branch-horizon、damage audit、certificate 都没有形成可 official 的 selector/controller。
```

机制判断：

1. H0 成立：v9.5.1 boundary 被复现，canonical truth base 仍可用。
2. H1 未成立：target lattice 没有 official/weak pass candidate；best target 的 coverage 只有 `0.009388`。
3. H2 未成立：target leaveout sanity 未过，dataset leaveout weak pass count = `0`。
4. H3 未成立：legal upper-bound probe v3 对 selected target 不具备可用 observability，best TopK64 target precision = `0.0`。
5. H4 未成立：mechanism anatomy 没有找到 legal-stable cluster，best purity 只有 `0.0625`。
6. H5 未成立：APX under selected target 重新打分仍失败。
7. H6/H7 成立于 implementation 层：APY1-APY8 spec、payload/certificate/action apply、preflight 与 branch-horizon materializer 都闭合。
8. H8 未成立：APY smoke 没有产生 value-positive / horizon-safe / target-positive primitive；best APY3 的 V_integrated LCB 为负且 h240 long-risk 高。
9. H9 未成立：vector certificate v5 没有 effect-valid separation；best TopK64 target precision 只有 `0.015625`。
10. H10-H12 未打开：没有 source controller，就不能打开 selected runtime、official paired replay、short/full 或 continual validation。
11. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.5.2 真实执行后停在 `R1-CanonicalActionDensityInsufficientForMultiHorizonTarget`：target lattice 找到了更平衡但过稀疏的 diagnostic target，APY1-APY8 工程链路完整闭合；但没有任何 multi-horizon target 满足 weak pass，legal probe、APY generator 和 certificate 也未转成 official controller，strict PureKAN functional 仍未成功。
