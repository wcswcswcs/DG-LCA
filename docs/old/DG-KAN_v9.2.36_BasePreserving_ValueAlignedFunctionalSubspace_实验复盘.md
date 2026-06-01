# DG-KAN v9.2.36 Base-Preserving Value-Aligned Functional Subspace 实验复盘

> 本复盘记录 `DG-KAN_v9.2.36_BasePreserving_ValueAlignedFunctionalSubspace_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R11-BPFSAllFailPrimitiveReset
base_candidate = LQ-t2-h256
success_v9236_strict_purekan_functional = False
success_v9236_full_functional = False
success_v9236_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z
```

核心结论：

1. P0 复现 v9.2.35 boundary：source route = `R9-VAOPAllFailPrimitiveReset`，primary blocker = `all_vaop_failed_P5_nearpass`。
2. P1 VAOP failure mode = `base_contamination`，base contamination rate = `1.0`。
3. P2 BPFS implemented count = `6`；P3 base equivalence / contract / grad = `1/1/1`。
4. P4 route-level BPFS base qualification = `0`；P4 pass = `1`，P5 near-pass = `0`，best BPFS = ``。
5. P5 carrier pass = `0`，max r_z_tail = `0.000000`，max r_perp_tail = `0.000000`。
6. 当前 blocker：`no_bpfs_passed_P4_P5_base_qualification`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `dgkan/models/fc_purekan_actuator.py` | 新增 BPFS1-BPFS6 zero-init base-preserving specs |
| `experiments/run_v9236_base_preserving_value_aligned_functional_subspace.py` | v9.2.36 runner；生成 P0-P12 artifacts、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile dgkan/models/fc_purekan_actuator.py experiments/run_v9236_base_preserving_value_aligned_functional_subspace.py
```

正式运行：

```bash
python experiments/run_v9236_base_preserving_value_aligned_functional_subspace.py \
  --out-dir results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R11-BPFSAllFailPrimitiveReset",
  "base_candidate": "LQ-t2-h256",
  "v9235_boundary_pass": 1,
  "source_route": "R9-VAOPAllFailPrimitiveReset",
  "source_primary_blocker": "all_vaop_failed_P5_nearpass",
  "dataset_tuning_detected": 0,
  "vaop_failure_mode": "base_contamination",
  "base_contamination_rate": 1.0,
  "bpfs_implemented_count": 6,
  "best_bpfs_candidate": "",
  "base_equivalence_pass": 1,
  "bpfs_contract_pass": 1,
  "bpfs_grad_pass": 1,
  "bpfs_p4_pass": 1,
  "bpfs_p5_nearpass": 0,
  "base_preservation_pass": 0,
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
  "primary_blocker": "no_bpfs_passed_P4_P5_base_qualification",
  "next_required_implementation": "if_carrier_silent_redesign_functional_carrier_else_if_value_unobservable_redesign_control_gap_statistic",
  "success_v9236_strict_purekan_functional": 0,
  "success_v9236_full_functional": 0,
  "success_v9236_external_ready": 0,
  "rows_checked": 163,
  "fake_proxy_nonzero_count": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true,
  "completed_at": "2026-05-11T08:10:11Z"
}
```

## 3. P4 BPFS base preservation

| candidate | P4 | P5 near | inactive/base-equivalence | near rows | macro delta | step q90 | memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| BPFS2-FrozenTaskFunctionalAttach | `1` | `0` | `1` | `6/9` | `-0.005167` | `1.171089` | `0.969731` |
| BPFS3-TaskOrthogonalFunctionalSubspace | `1` | `0` | `1` | `6/9` | `-0.005167` | `1.178622` | `0.969731` |
| BPFS4-ControlGapFunctionalChannel | `1` | `0` | `1` | `6/9` | `-0.005167` | `1.115252` | `0.969731` |
| BPFS5-FamilyValueFunctionalChannel | `1` | `0` | `1` | `6/9` | `-0.005167` | `1.117405` | `0.969731` |

## 4. P5 functional carrier

rows = `0`

判断：P5 只测试 event-time functional carrier 是否 non-silent，不把它写成 paired replay success。

## 5. Downstream boundary

P7-P12 只有在 P6/P7/P8 gate 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 6. No-fake audit

```text
rows_checked = 163
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 7. Hash

| artifact | SHA256 |
|---|---|
| runner | `043a62a386464b42541a3e7181f771cf9b62860f35cda404a8b87e5a304375d3` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/run_manifest.json` | `a134918102fff1208b5e160e7000f2cbb25223760e244550c118e1e268f81229` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/contract_audit_v9236.csv` | `c8bf78b780b17b514c00f4d37e6cc010c66b4f5a8c07c0bf77242888d578a557` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p0_v9235_boundary_reproduction.csv` | `08dafec5ee11381999ddec3c8b791708a41872462dec8bd1d5dd4897b5cf0427` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p1_vaop_failure_autopsy.csv` | `f5a096f513ce0534b4cafdb274192339c0fb746c37b9129e9852834abcfadccc` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p2_bpfs_implementation.csv` | `58d5fe53367d2fad7320e1fedb10a79d29ea7c0073d611e9db76bbaf20b10a75` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p3_contract_grad_base_equivalence.csv` | `a0c4d50c79d5e827a7a78178ecf779ef0bd2582ff21f00e3033a78eb0ce45da3` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p4_bpfs_p4_p5_base_preservation.csv` | `25fc6d55add7fc3e08ab38bf4f8ee82457ef4b76b5869d3e9e012f73a5fac098` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p5_functional_carrier_actuatability.csv` | `4cf378cb0f6e710d0e860baebfbf0fb3697738b49054ebde161cbc296a2023b2` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p6_value_observability_audit.csv` | `a0af26af334a0917a0c5a070ea90941e3d72adb2c0d698d49cce3015fa507ec2` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p7_leave_dataset_and_stratum_out_validation.csv` | `7469199ce9bfc8052a443037b98bf1f9d9677fa714b9324f320e4374ff9e4a51` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p8_official_base_preserving_paired_replay.csv` | `d84aea19d93ed2ea670c3c30b5021e562d34e5fd4c79bea99df2fae060f87513` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p9_short_run_functional_validation.csv` | `7cee3e0aaf070f020a504527aa68c72a0d2120670fff537711b12141f3d4beab` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p10_full_10seed_functional_validation.csv` | `6c2ee4ac6fbd4a3ff82496804854757b17ac04b13038c76086bab56437add16c` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p11_adamw_only_fullpass_repair.csv` | `cc99e7b71000434d6aadf1e997437a8ca30739af05b3f865fa3bafd7ae36ef17` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/p12_robustness_external_ready.csv` | `9daf7bac7673115feb44fbc7ffb409dbf1730e953dee20d0b825e660980a519c` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/bpfs_primitive_trace_v9236.csv` | `58d5fe53367d2fad7320e1fedb10a79d29ea7c0073d611e9db76bbaf20b10a75` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/base_preservation_trace_v9236.csv` | `25fc6d55add7fc3e08ab38bf4f8ee82457ef4b76b5869d3e9e012f73a5fac098` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/functional_carrier_trace_v9236.csv` | `4cf378cb0f6e710d0e860baebfbf0fb3697738b49054ebde161cbc296a2023b2` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/value_score_trace_v9236.csv` | `a0af26af334a0917a0c5a070ea90941e3d72adb2c0d698d49cce3015fa507ec2` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/leave_dataset_out_trace_v9236.csv` | `900c55f491ce111995d14b025e6226836d3f121fa9bd4c85b0e3b125a55576cf` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/paired_replay_branch_trace_v9236.csv` | `77205e74f9aa308fa8589ffb70facad48aa476cba8c7b070ef468cd91011ff86` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/route_decision.json` | `f42d6ae58b3365fe09c549f706df0e03480b014b0e5de51b812552b44a9b950a` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/aggregate_decision.json` | `f42d6ae58b3365fe09c549f706df0e03480b014b0e5de51b812552b44a9b950a` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/failure_table.csv` | `c9d8e3309b0d14fa357ef442814db5b7a662363ee3ade5c5d8a5369f4c5faf17` |
| `results/real_rerun_20260506/v9236_base_preserving_value_aligned_functional_subspace_first_20260511T090000Z/v9236_provenance_audit.csv` | `b8ad99792a3de0a76a7bc32cf4cc44b9dc055556879ef427e4f04ee09b968bbd` |

## 8. 最终分析结论

v9.2.36 的真实推进是：

```text
v9.2.35: VAOP 直接并入 base 后 P5 near-pass 全灭。
v9.2.36: BPFS inactive functional channel 按 zero-init/frozen update 重新审计 base preservation 与 carrier actuatability。
```

机制判断：

1. Base preservation 与 functional carrier 被分开 gate；functional update 不允许救 base。
2. 本轮行级 inactive/base-equivalence 没有失败；route-level 失败点是没有候选同时满足 P4 与 P5 near-pass。
3. 如果 BPFS 过 P4/P5 但 P5 carrier 静音，结论是 carrier 仍需重做，不是 dataset patch。
4. 如果 carrier 有 movement 但 value observability 失败，后续应回到 control-gap/value statistic，而不是把 movement 当成功。

最终一句话：

> v9.2.36 真实执行后停在 `R11-BPFSAllFailPrimitiveReset`：`no_bpfs_passed_P4_P5_base_qualification`。
