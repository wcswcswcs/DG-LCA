# DG-KAN v9.2.37 Training-Path Equivalent Functional Attach 实验复盘

> 本复盘记录 `DG-KAN_v9.2.37_TrainingPathEquivalent_FunctionalAttach_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R11-TPEAAllFailPrimitiveReset
base_candidate = LQ-t2-h256
success_v9237_strict_purekan_functional = False
success_v9237_full_functional = False
success_v9237_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z
```

核心结论：

1. P0 复现 v9.2.36 boundary：source route = `R11-BPFSAllFailPrimitiveReset`，source blocker = `no_bpfs_passed_P4_P5_base_qualification`。
2. P1 BPFS failure mode = `candidate_mismatch_or_LQ_base_gap`，failure attribution fraction = `1.0`。
3. P2 TPEA implemented count = `6`；P3 training-path equivalence / contract / grad = `1/1/1`。
4. P4 best TPEA = `TPEA1-RegisteredZeroExcluded`，P4 pass = `1`，P5 near-pass = `0`，base preservation = `0`。
5. P5 carrier pass = `0`，max r_z_tail = `0.000000`，max r_perp_tail = `0.000000`。
6. 当前 blocker：`all_tpea_failed_P4_P5_base_preservation`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `dgkan/models/fc_purekan_actuator.py` | 新增 TPEA1-TPEA6 zero-init functional attach specs |
| `experiments/run_v9237_training_path_equivalent_functional_attach.py` | v9.2.37 runner；生成 P0-P12 artifacts、training-path trace、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9237_training_path_equivalent_functional_attach.py
```

正式运行：

```bash
python experiments/run_v9237_training_path_equivalent_functional_attach.py \
  --out-dir results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R11-TPEAAllFailPrimitiveReset",
  "base_candidate": "LQ-t2-h256",
  "v9236_boundary_pass": 1,
  "source_route": "R11-BPFSAllFailPrimitiveReset",
  "source_primary_blocker": "no_bpfs_passed_P4_P5_base_qualification",
  "dataset_tuning_detected": 0,
  "bpfs_failure_mode": "candidate_mismatch_or_LQ_base_gap",
  "bpfs_failure_primary_fraction": 1.0,
  "tpea_implemented_count": 6,
  "best_tpea_candidate": "TPEA1-RegisteredZeroExcluded",
  "training_path_equivalence_pass": 1,
  "task_init_hash_match": 1,
  "optimizer_state_match": 1,
  "lambda_inactive_leak_detected": 0,
  "tpea_contract_pass": 1,
  "tpea_grad_pass": 1,
  "tpea_p4_pass": 1,
  "tpea_p5_nearpass": 0,
  "base_preservation_pass": 0,
  "lq_reference_p5_nearpass": 0,
  "lq_reference_macro_delta": -0.004499991734822591,
  "best_base_macro_delta": 0.0,
  "best_base_step_q90": 0.0,
  "functional_carrier_pass": 0,
  "max_r_z_tail": 0.0,
  "max_r_perp_tail": 0.0,
  "carrier_bad_event_rate": 0.0,
  "value_observability_pass": 0,
  "best_controller": "",
  "best_value_corr": 0.0,
  "best_value_auc": 0.5,
  "accepted_precision": 0.0,
  "accepted_coverage": 0.0,
  "accepted_bad_event_rate": 0.0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 1,
  "hard_stratum_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "primary_blocker": "all_tpea_failed_P4_P5_base_preservation",
  "next_required_implementation": "if_base_preserved_but_silent_redesign_carrier_else_if_value_unobservable_redesign_control_gap_statistic_else_extend_P7_P8",
  "success_v9237_strict_purekan_functional": 0,
  "success_v9237_full_functional": 0,
  "success_v9237_external_ready": 0,
  "rows_checked": 975,
  "fake_proxy_nonzero_count": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true,
  "completed_at": "2026-05-11T09:29:11Z"
}
```

## 3. P1 BPFS training-path failure autopsy

| failure mode | rows |
|---|---:|
| `candidate_mismatch_or_LQ_base_gap` | `54` |

## 4. P4 TPEA base preservation

| candidate | attach | P4 | P5 near | near rows | macro delta | step q90 | memory |
|---|---|---:|---:|---:|---:|---:|---:|
| TPEA0-LQ-reference | `lq_reference` | `1` | `0` | `6/9` | `-0.004500` | `1.082714` | `0.969501` |
| TPEA2-RegisteredZeroNoOpGroup | `registered_noop_optimizer_group` | `1` | `0` | `6/9` | `-0.004500` | `1.216044` | `0.969731` |
| TPEA3-LateAttachOrthogonalTail | `late_attach_after_base` | `1` | `0` | `6/9` | `-0.004500` | `1.082714` | `0.969501` |
| TPEA4-ShadowSpecUntilEvent | `shadow_spec_until_event` | `1` | `0` | `6/9` | `-0.004500` | `1.082714` | `0.969501` |
| TPEA5-RoleWiseFT7LateAttach | `rolewise_ft7_late_attach` | `1` | `0` | `6/9` | `-0.004500` | `1.082714` | `0.969501` |
| TPEA6-ControlGapLateAttach | `control_gap_late_attach` | `1` | `0` | `6/9` | `-0.004500` | `1.082714` | `0.969501` |

## 5. P5 functional carrier

rows = `0`

判断：P5 只在 P4/P5 survivor 上测试 event-time functional carrier 是否 non-silent，不把 movement 写成 paired replay success。

## 6. Downstream boundary

P7-P12 只有在 P6/P7/P8 gate 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 7. No-fake audit

```text
rows_checked = 975
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
| runner | `5fd9ddf71ade1cbfd3d810d57562d706cc7b7d7728d209ca899889634c7f13b6` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/run_manifest.json` | `983fa3ff85ac10be86d03f8e7c46010c921d67d5890502308c79f3345dcb3ea3` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/contract_audit_v9237.csv` | `c8bf78b780b17b514c00f4d37e6cc010c66b4f5a8c07c0bf77242888d578a557` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p0_v9236_boundary_reproduction.csv` | `36402a2cf15f36d086c4579f3eb6ede8de3fbaff765a278d7184dc083c986b42` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p1_bpfs_training_path_failure_autopsy.csv` | `3c8f9fcdfc5e88050c95c5880ddd789b3d84d5e1dc6fe8e307a07ceb721ddee2` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p2_tpea_implementation.csv` | `987b6d491bfca1c524dd6e03a6d58e6e935548533ed4e9e482d1d31efc85c997` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p3_contract_grad_training_path_equivalence.csv` | `ed7becc5da6d6d2868df78ee6c3e3a3acc6a1232e35462ede4398ddde5b1800a` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p4_tpea_p4_p5_base_preservation.csv` | `ddb33f72e2bd27e28f99f549d6ad34f38b215acf12a03cda959731d30cb94ca7` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p5_functional_carrier_actuatability.csv` | `4fe87ff555834513e147ebe289f70b489e93bbd4d9310b54b324582caa1ce394` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p6_value_observability_audit.csv` | `a0af26af334a0917a0c5a070ea90941e3d72adb2c0d698d49cce3015fa507ec2` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p7_leave_dataset_and_stratum_out_validation.csv` | `7469199ce9bfc8052a443037b98bf1f9d9677fa714b9324f320e4374ff9e4a51` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p8_official_training_path_equivalent_paired_replay.csv` | `527e41e74a6bc98b2feacbbb1519116912a81bfa6d0cc768353f715b8d0414e5` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p9_short_run_functional_validation.csv` | `7cee3e0aaf070f020a504527aa68c72a0d2120670fff537711b12141f3d4beab` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p10_full_10seed_functional_validation.csv` | `6c2ee4ac6fbd4a3ff82496804854757b17ac04b13038c76086bab56437add16c` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p11_adamw_only_fullpass_repair.csv` | `cc99e7b71000434d6aadf1e997437a8ca30739af05b3f865fa3bafd7ae36ef17` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/p12_robustness_external_ready.csv` | `9daf7bac7673115feb44fbc7ffb409dbf1730e953dee20d0b825e660980a519c` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/training_path_trace_v9237.csv` | `283ce1293b6354413a9279b3248084946c2900137fb7213896ca226302315182` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/optimizer_state_trace_v9237.csv` | `26fc1e32fa2feac407a07051165fef8084d052c502228e600242921bd7a97461` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/attach_mode_trace_v9237.csv` | `987b6d491bfca1c524dd6e03a6d58e6e935548533ed4e9e482d1d31efc85c997` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/functional_carrier_trace_v9237.csv` | `4fe87ff555834513e147ebe289f70b489e93bbd4d9310b54b324582caa1ce394` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/value_score_trace_v9237.csv` | `a0af26af334a0917a0c5a070ea90941e3d72adb2c0d698d49cce3015fa507ec2` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/leave_dataset_out_trace_v9237.csv` | `2cfd099b1fcbac7246f5c5a112366c6b8e64953f5b74076e47eed00d142ddaf6` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/paired_replay_branch_trace_v9237.csv` | `eb4fd64f74bbd43c846091aa5931a2bd2d45275a3dbd24551fdc76d0624cb443` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/route_decision.json` | `36ebd9f5cd1bd1a47ea50267d598a8b9047272c0678fc0e322709ccd90407c6a` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/aggregate_decision.json` | `36ebd9f5cd1bd1a47ea50267d598a8b9047272c0678fc0e322709ccd90407c6a` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/failure_table.csv` | `2bbac576a33fbd490574fbcf1746b6076a92bff6863f9a1322c6849b90a288ce` |
| `results/real_rerun_20260506/v9237_training_path_equivalent_functional_attach_first_20260511T100000Z/v9237_provenance_audit.csv` | `fa6dc8d313d308069ab879c76eea44deba069e1674484c7771b42a3a3c4444c0` |

## 9. 最终分析结论

v9.2.37 的真实推进是：

```text
v9.2.36: BPFS initial equivalence pass, but route-level P5 near-pass fail.
v9.2.37: training-path equivalence / attach mode / base qualification are separated and audited.
```

机制判断：

1. 不能再把 initial logit smoke 当成 base preservation；必须看 task param / grad / AdamW state trajectory。
2. 如果 TPEA 过 training-path equivalence 但 P5 仍失败，问题不是 functional carrier，而是 base qualification 或 matching protocol 仍未闭合。
3. 如果 TPEA 过 P4/P5 但 carrier 静音，下一步修 actuatability；如果 carrier 能动但 value 不过，再回 value statistic。

最终一句话：

> v9.2.37 真实执行后停在 `R11-TPEAAllFailPrimitiveReset`：`all_tpea_failed_P4_P5_base_preservation`。
