# DG-KAN v9.2.30 Pre-Event Signal-Value Functional Controller 实验复盘

> 本复盘记录 `DG-KAN_v9.2.30_PreEvent_SignalValue_FunctionalController_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R7-PrimitiveEffectUnpredictable
base_candidate = LQ-t2-h256
success_v9230_strict_purekan_functional = False
success_v9230_full_functional = False
success_v9230_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/
```

核心结论：

1. P0 复现 v9.2.29 boundary：source route = `R1-SignalStrataExplainFailures`，event-value corr = `0.197001`，fake/proxy = `0`。
2. P1 新增真实 pre-event movement replay rows = `486`；pre-event feature pass = `0`，best movement abs corr = `0.299708`。
3. P2 failure reclassification pass = `0`，attributed fraction = `0.000000`，abstain precision = `0.000000`。
4. P3 best predictor = `None`，corr = `0.000000`，precision = `0.000000`，coverage = `0.000000`。
5. 当前 blocker：`pre_event_features_not_predictive`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9230_preevent_signal_value_functional_controller.py` | v9.2.30 runner；复现 v9.2.29 boundary，实测 pre-event movement features，执行 predictor / LDO gate、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9230_preevent_signal_value_functional_controller.py
```

正式运行：

```bash
python experiments/run_v9230_preevent_signal_value_functional_controller.py \
  --out-dir results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z \
  --fresh \
  --device auto \
  --data-root data \
  --seed 1314
```

实际测量范围：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 20,80,240
primitives = P0-N2a-Rational,P1-N2c-SharedRBF,P2-N3c-SharedRBFDerivativeBand
events = E1-CEHardTail,E2-MarginTail,E3-HardMode,E4-CurvatureTail,E5-CEPlusMargin,E6-OrthogonalCE
```

说明：本轮是 first-wave pre-event replay。没有把 `seed=3,4` 或 `horizon=640` 写成已测，也没有用它们推导 route。

## 2. Route

```json
{
  "adamw_fullpass": 0,
  "base_candidate": "LQ-t2-h256",
  "best_movement_feature_abs_corr": 0.2997081595094821,
  "dataset_tuning_detected": 0,
  "event_value_predictor_pass": 0,
  "external_ready": 0,
  "failure_mode_reclassification_pass": 0,
  "full_run_pass": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "functional_task_safe": 0,
  "hard_stratum_repair_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "movement_features_all_below_020": 0,
  "next_required_implementation": "return_to_interface_or_primitive_design",
  "p1_row_count": 486,
  "p2_attributed_fraction": 0.0,
  "paired_replay_pass": 0,
  "pre_event_feature_pass": 0,
  "pre_event_relevant_feature_count": 9,
  "primary_blocker": "pre_event_features_not_predictive",
  "robustness_pass": 0,
  "route": "R7-PrimitiveEffectUnpredictable",
  "short_run_pass": 0,
  "strong_baseline_pass": 0,
  "success_v9230_external_ready": 0,
  "success_v9230_full_functional": 0,
  "success_v9230_strict_purekan_functional": 0,
  "v9229_boundary_pass": 1
}
```

## 3. P1 pre-event movement feature extraction

P1 是本轮新增实测 replay，不是从 v9.2.29 缺失字段补值。官方 controller feature 不使用 dataset name / seed id / class name。

| feature | corr with actual control gap | abs corr | movement |
|---|---:|---:|---:|
| effective_derivative | `-0.339810` | `0.339810` | `0` |
| pre_adamwparallel_logit_delta_norm | `-0.299708` | `0.299708` | `1` |
| pre_bestlr_logit_delta_norm | `-0.299708` | `0.299708` | `1` |
| pre_real_nonadamw_delta_norm | `-0.249151` | `0.249151` | `1` |
| pre_real_logit_delta_norm | `-0.240042` | `0.240042` | `1` |
| pre_real_tail_logit_delta_norm | `-0.218750` | `0.218750` | `1` |
| projected_tail_component_ratio | `-0.205747` | `0.205747` | `1` |
| orthogonal_component_ratio | `0.205747` | `0.205747` | `1` |
| dominant_basis_fraction | `-0.205299` | `0.205299` | `0` |
| pre_real_vs_adamw_delta_ratio | `0.194462` | `0.194462` | `1` |

P1 gate：

```text
relevant_feature_count = 9
best_movement_feature_abs_corr = 0.299708
pre_event_feature_pass = 0
```

## 4. P2 failure-mode reclassification

```text
attributed_fraction = 0.000000
abstain_precision = 0.000000
unattributed_count = None
dataset_name_used = None
```

判断：P2 只把 rows 分到 silent / low magnitude / control dominated / misaligned / high uncertainty / positive candidate 等机制，不把数据集名作为 failure route。

## 5. P3 event-value predictor redesign

| predictor | corr | precision | coverage | bad event | pass |
|---|---:|---:|---:|---:|---:|

P3 判断：

```text
best_predictor = None
event_value_predictor_pass = 0
accepted_precision = 0.000000
accepted_coverage = 0.000000
accepted_bad_event_rate = 0.000000
```

## 6. P4 / downstream boundary

P4 leave-dataset-out pass = `0`，leave-stratum-out pass = `0`。

P5-P9 只有在 P3/P4 gate 通过后才允许打开。本轮未打开阶段均以 `not_run` row 落盘，没有把 short/full/external 写成通过。

## 7. No-fake audit

```text
rows_checked = 1491
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 8. Hash

| artifact | SHA256 |
|---|---|
| `experiments/run_v9230_preevent_signal_value_functional_controller.py` | `f759ed4987d99c1aee9ebc8e5d4df5e41c465cbe0956968afe5d47c6af83080a` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/contract_audit_v9230.csv` | `050f2f42c5ba33ae565bf244eb4737cd5606148180ae4585b7b47763e5c7bad5` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/event_value_prediction_trace_v9230.csv` | `0beec754e8d693a3089a125ee1d2c16092edab04460edda62166e4a321f25fbd` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/failure_table.csv` | `ec8da0eafe2599692ead7f33e625e2c77452b1005439115249926aff261e12a1` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/leave_dataset_out_trace_v9230.csv` | `b55498af14fdc02a4541ae11cbfd73c88fbb8a68fe1bc8e35208aa73330246d4` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p0_v9229_boundary_reproduction.csv` | `7c753a87c982084dd8c08a66c95a9f3c790a4e3b74b108ec116bc136f37771b7` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p1_pre_event_feature_correlation_summary.csv` | `d57b3fc4f27db6abad4225e53ff93266453bc4667f33635579c64589ccbf5aad` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p1_pre_event_movement_features.csv` | `0beec754e8d693a3089a125ee1d2c16092edab04460edda62166e4a321f25fbd` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p2_failure_mode_reclassification.csv` | `bdfdfa69cbf0332d03c8796e89a4419ce5ba085873736474b3948d44c39e0ad5` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p3_event_value_predictor_redesign.csv` | `b737c4fe4064911134a0382306474f7dd55dcf0105c8d9697039e5cde1659ba7` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p4_leave_dataset_and_stratum_out_validation.csv` | `b55498af14fdc02a4541ae11cbfd73c88fbb8a68fe1bc8e35208aa73330246d4` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p5_official_signal_routed_paired_replay.csv` | `8a321190262f20933cc2ab908b5fbb0765243a093c08ce6bd35cd8ac7d69e3ef` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p6_short_run_functional_validation.csv` | `9c03576951033f1e0c1e29295d93b44fecfe3c47a677761ba7df818a7aedded9` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p7_full_10seed_functional_validation.csv` | `d4e25369028b09ca42d5f5f9f627d61ad60dd76ca4e336f1e7b0feeb24c10098` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p8_adamw_only_fullpass_repair.csv` | `433eb8ce9a0eece64366b11f912b06c8aadaf372c28f9adb3f6114acbf1dbeda` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/p9_robustness_external_ready.csv` | `dd71f97b7734511ea3f3feb4d46dd45e68db632e43640a8815b1fe913805896d` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/paired_replay_branch_trace_v9230.csv` | `73e313624500978c534ddf2c943f27f5b96ac485d358595184dba7f03ca15a38` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/pre_event_feature_trace_v9230.csv` | `0beec754e8d693a3089a125ee1d2c16092edab04460edda62166e4a321f25fbd` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/route_decision.json` | `bf59c665e73433caafdcfab135679b7ffd4db6856ee576ac202678663b691dcc` |
| `results/real_rerun_20260506/v9230_preevent_signal_value_functional_controller_first_20260510T210000Z/v9230_provenance_audit.csv` | `4fa97f8142a434062bf140e5351e3576564f161bfc06068f82b2bb550d2053ea` |

## 9. 最终分析结论

v9.2.30 的真实推进是：

```text
v9.2.29: signal strata 能解释 failure，但 event-value corr 只有 0.197。
v9.2.30: 新增真实 pre-event movement replay，检验这些 feature 是否足以形成 dataset-agnostic controller。
```

机制判断：

1. 本轮不再回到 Fashion/KMNIST dataset-specific patch；dataset 只用于诊断和 LDO 评估。
2. P1 的 movement feature 是真实 one-step/replay 前测量，不是把 v9.2.29 的 `not_measured` 字段补成数值。
3. 当前 route 停在 `R7-PrimitiveEffectUnpredictable`，primary blocker = `pre_event_features_not_predictive`。
4. 因 downstream gate 未打开，strict PureKAN functional / full functional / external-ready 均不能声明成功。

最终一句话：

> v9.2.30 真实执行后停在 `R7-PrimitiveEffectUnpredictable`：`pre_event_features_not_predictive`。
