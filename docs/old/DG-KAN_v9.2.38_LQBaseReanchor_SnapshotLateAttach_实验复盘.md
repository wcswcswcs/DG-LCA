# DG-KAN v9.2.38 LQ Base Re-Anchor 与 Snapshot Late-Attach Functional Carrier 实验复盘

> 本复盘记录 `DG-KAN_v9.2.38_LQBaseReanchor_SnapshotLateAttach_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R1-LQBaseAnchorBroken
base_candidate = LQ-t2-h256
success_v9238_strict_purekan_functional = False
success_v9238_full_functional = False
success_v9238_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z
```

核心结论：

1. P0 复现 v9.2.37 boundary：source route = `R11-TPEAAllFailPrimitiveReset`，source blocker = `all_tpea_failed_P4_P5_base_preservation`。
2. P1 Current LQ re-anchor：near-pass `6/9`，macro delta `-0.004500`，anchor pass = `0`。
3. Historical reference：near-pass `8/9`，macro delta `-0.004000`；historical-current macro delta `-0.000500`。
4. TPEA-off P5 equivalence pass = `1`；protocol mismatch detected = `1`，mode = `historical_current_nearpass_drift`。
5. 因 LQ anchor 未过，本轮 snapshot late-attach / carrier / value / paired replay 全部未打开。
6. 当前 blocker：`current_lq_base_nearpass_rate_below_gate`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9238_lq_base_reanchor_snapshot_late_attach.py` | v9.2.38 runner；生成 P0-P12 artifacts、LQ anchor/protocol audit、route、failure/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9238_lq_base_reanchor_snapshot_late_attach.py
```

正式运行：

```bash
python experiments/run_v9238_lq_base_reanchor_snapshot_late_attach.py \
  --out-dir results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z \
  --fresh --device auto --data-root data --seed 1314
```

## 2. Route

```json
{
  "route": "R1-LQBaseAnchorBroken",
  "base_candidate": "LQ-t2-h256",
  "v9237_boundary_pass": 1,
  "source_route": "R11-TPEAAllFailPrimitiveReset",
  "source_primary_blocker": "all_tpea_failed_P4_P5_base_preservation",
  "dataset_tuning_detected": 0,
  "lq_base_anchor_pass": 0,
  "historical_current_lq_delta": -0.0004999770058525931,
  "protocol_mismatch_detected": 1,
  "protocol_mismatch_mode": "historical_current_nearpass_drift",
  "current_lq_nearpass": 0,
  "current_lq_near_count": 6,
  "current_lq_row_count": 9,
  "current_lq_near_rate": 0.6666666666666666,
  "current_lq_macro_delta": -0.004499991734822591,
  "historical_lq_near_count": 8,
  "historical_lq_row_count": 9,
  "historical_lq_macro_delta": -0.004000014728969998,
  "tpea_off_p5_equivalence_pass": 1,
  "snapshot_attach_implemented_count": 0,
  "best_late_attach_candidate": "",
  "checkpoint_inactive_equivalence_pass": 0,
  "no_event_replay_preservation_pass": 0,
  "functional_carrier_pass": 0,
  "value_observability_pass": 0,
  "leave_dataset_out_pass": 0,
  "leave_stratum_out_pass": 0,
  "paired_replay_pass": 0,
  "short_run_pass": 0,
  "full_run_pass": 0,
  "functional_task_safe": 0,
  "functional_control_pass": 0,
  "functional_system_pass": 0,
  "hard_stratum_repair_pass": 0,
  "adamw_fullpass": 0,
  "strong_baseline_pass": 0,
  "robustness_pass": 0,
  "external_ready": 0,
  "primary_blocker": "current_lq_base_nearpass_rate_below_gate",
  "next_required_implementation": "repair_current_lq_base_anchor_or_protocol_before_snapshot_late_attach",
  "success_v9238_strict_purekan_functional": 0,
  "success_v9238_full_functional": 0,
  "success_v9238_external_ready": 0,
  "rows_checked": 134,
  "fake_proxy_nonzero_count": 0,
  "fake_data_used": 0,
  "proxy_row_used": 0,
  "cpu_offload_used": 0,
  "no_fake": true,
  "no_proxy": true,
  "completed_at": "2026-05-11T10:25:10Z"
}
```

## 3. P1 LQ base anchor / protocol mismatch audit

| candidate | rows | near rows | macro delta | max acc diff vs Current LQ |
|---|---:|---:|---:|---:|
| CurrentLQReproduction | `9` | `6/9` | `-0.004500` | `0.000000` |
| TPEA1-off | `9` | `6/9` | `-0.004500` | `0.000000` |
| TPEA3-late-attach-off | `9` | `6/9` | `-0.004500` | `0.000000` |

P1 summary：

```text
lq_base_anchor_pass = 0
current_lq_near_rate = 0.666667
historical_current_lq_delta = -0.000500
tpea_off_p5_equivalence_pass = 1
protocol_mismatch_detected = 1
protocol_mismatch_mode = historical_current_nearpass_drift
```

判断：Current LQ 在本轮真实 runner 下没有达到 `near_pass_rate >= 0.80`，因此 functional route 按计划停止；没有把 snapshot late-attach 或 functional carrier 在未锚定 base 上继续推进。

## 4. Downstream boundary

P2-P12 只有在 LQ base anchor pass 后打开。本轮均以 `not_run` row 落盘，reason = `LQ_base_anchor_failed`。

## 5. No-fake audit

```text
rows_checked = 134
fake_proxy_nonzero_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
no_fake = True
no_proxy = True
```

## 6. Hash

| artifact | SHA256 |
|---|---|
| runner | `55901b68112cb06d55b9301350f0e07aa05c8912fc148bfcf0a2af473a8cb47c` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/run_manifest.json` | `28039f2eac4f8c7d4e16e624c5ba9272531d32881bbc132368ed83e42d7373d1` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/contract_audit_v9238.csv` | `c8bf78b780b17b514c00f4d37e6cc010c66b4f5a8c07c0bf77242888d578a557` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p0_v9237_boundary_reproduction.csv` | `cd6248726d715d843bfc1115a1e527143e98e6cd60ffffdd6a4c612a577071dc` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p1_lq_base_anchor_protocol_audit.csv` | `f315b515c1b9ca86d02937e9bee38e108b8a07e1230c7a5a66939ef1eab51a0a` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p2_snapshot_late_attach_implementation.csv` | `168229bc24baba4f4c69da40ebfae44f212a1241971994d20abae79bef0dca5b` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p3_checkpoint_inactive_equivalence.csv` | `c0f1ba29444ef31da48b86921559ce7e04d1f62cbad03eb211ab0329cc39a470` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p4_no_event_replay_base_preservation.csv` | `04b26ea4a88613c61bf1f737a534f94d760ad81456459b50ecc4317958745df5` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p5_functional_carrier_actuatability.csv` | `3061b18b3c3f39885667d65ebbf0265240b62d543a65fda834ef5e1148fe43be` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p6_value_observability_audit.csv` | `9ba9537b2f1dd97d346995b88ca82856156271848a7ba2d5427186912395ada9` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p7_leave_dataset_and_stratum_out_validation.csv` | `dd48f78f9cdeb87d81131d6add4db70f971456b89cd15682f0a885cf691b65b0` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p8_official_snapshot_late_attach_paired_replay.csv` | `30c2958df71cd7d2c0f81bbecfd0b6133cfca0e97c13ede713f9ef8b23656665` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p9_short_run_functional_validation.csv` | `345b6064f6f515108f4920186fce3302f6392e3cc4d8253556439be50115f672` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p10_full_10seed_functional_validation.csv` | `5d17a3cdcd4bacf07a37d55ccc3fd18c35571834f78bab42323a135251e99132` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p11_adamw_only_lq_fullpass_repair.csv` | `c68dbb0bebdc6b235c74c596544990b200d6966bf28eb27f321aaef5fdb54a75` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/p12_robustness_external_ready.csv` | `5dccc235068435ad877164e9d830d8b773bb1c4856ce47d9a17491e233517908` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/lq_anchor_trace_v9238.csv` | `f315b515c1b9ca86d02937e9bee38e108b8a07e1230c7a5a66939ef1eab51a0a` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/protocol_hash_diff_v9238.csv` | `9acf5b856e57cffcc5f9a80695fd84e7b02e457161f3c6a326fa07692d646c25` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/snapshot_attach_trace_v9238.csv` | `88fda24236452da980f58b75fa3ce7a2685aefd57d497cc2de3e5fa7fcb87c35` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/checkpoint_equivalence_trace_v9238.csv` | `988c1dead69f8c116127ddbb7479da7d4fe16126246b5eadaa7826795df1b077` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/functional_carrier_trace_v9238.csv` | `fba56647d02817d223693a4bdf6b11d81ac77a693483c1c84f6d65e63cf01034` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/value_score_trace_v9238.csv` | `01a2772b4df5d2e3a681c326dd9f02e254497a8b45634ae78af448b86b4c420b` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/leave_dataset_out_trace_v9238.csv` | `ae4981798d86a6d31a61d1e163b5b851ef97bd8649cfb6bcf0f50a9b99d832af` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/paired_replay_branch_trace_v9238.csv` | `2633103083fbf5370c795859448a79babd3d2531eb594d4b53c8f757d9b2c284` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/route_decision.json` | `03237e1bf88c1fd48b060efd00517c1f67a54cf4920e14b2066d863e7408fc45` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/aggregate_decision.json` | `03237e1bf88c1fd48b060efd00517c1f67a54cf4920e14b2066d863e7408fc45` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/failure_table.csv` | `942f342f0fe21d93f60b5288a02cbee2e6c89c0732b7e2e40e4847e7e775bf7e` |
| `results/real_rerun_20260506/v9238_lq_base_reanchor_snapshot_late_attach_first_20260511T110000Z/v9238_provenance_audit.csv` | `7eaa13aff893fb51a98d5d9edee1d9791759cd694402dd63e958b588edfb568b` |

## 7. 最终分析结论

v9.2.38 的真实推进是：

```text
v9.2.37: TPEA training-path equivalence pass, but P5 near-pass fail.
v9.2.38: LQ base re-anchor / protocol mismatch audit is performed before snapshot late attach.
```

机制判断：

1. 本轮最重要的结论不是 functional carrier failure，而是 current LQ base anchor 仍未闭合。
2. Current LQ macro delta 与 historical reference 接近，但 near-pass row count 从 historical `8/9` 变为 current `6/9`，没有达到计划的 anchor gate。
3. TPEA-off 与 current LQ 的 P5 等价性可以审计，但在 LQ anchor 未过前不能打开 snapshot late-attach。
4. 下一步应先修 LQ base/protocol re-anchor，而不是继续设计 value score 或 dataset-specific functional route。

最终一句话：

> v9.2.38 真实执行后停在 `R1-LQBaseAnchorBroken`：`current_lq_base_nearpass_rate_below_gate`。
