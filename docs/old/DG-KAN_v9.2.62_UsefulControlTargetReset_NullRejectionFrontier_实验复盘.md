# DG-KAN v9.2.62 Useful-Control Target Decomposition 与 Coverage-Constrained Null-Rejection Frontier 实验复盘

> 本复盘记录 `DG-KAN_v9.2.62_UsefulControlTargetReset_NullRejectionFrontier_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。

## 0. 最新结论

```text
route = R12-UsefulTargetStillTiny
base_candidate = LQ-t2-h256
success_v9262_strict_purekan_functional = False
success_v9262_full_functional = False
success_v9262_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9262_useful_control_target_reset_null_rejection_frontier_first_20260512T170000Z/
```

核心结论：

1. P0 复现 v9.2.61 boundary：source route = `R13-ValueRiskOrthogonalizationFail`，exact reference deployable = `0`，true-delta compute pass = `0`，fake/proxy/offload = `0`。
2. 本轮 manifest 记录 `device = cuda`，`triton_available = true`；运行中 GPU 采样有活动：`NVIDIA L4, 31-32 %, 326 MiB`。
3. P1 useful-control target autopsy pass = `1`：accepted count = `5997`，useful rejected = `230`，harmless-null accepted = `684`，bad-event accepted = `2846`，三类 attribution fraction 均为 `1.0`。
4. P2 null submode 有预测性：predictive null submode count = `4`；best null rejector = `N2-LowGainNullRejector`，null AUC = `0.720924`，null rate after gate = `0.139792`，diagnostic pass = `1`，official pass = `0`。
5. P3 useful/control statistic 有信号但不安全：best = `U1-RealGainLCBv2`，useful AUC = `0.703150`，coverage = `0.062707`，但 precision = `0.446935`，bad-event = `0.466051`，因此 utility pass = `0`。
6. P4 support stability 过关：natural rows = `24192`，balanced diagnostic rows = `6000`，signal strata = `27`，families = `741`，duplicate rows = `0`；accepted support pass = `1`。
7. P5 exact reference deployable frontier 仍未过：best = `C-U6-UsefulMixtureScore+N2-LowGainNullRejector+R2-UsefulConditionedBadRisk+S4-CoverageBalancedSupportCap`，precision = `0.763819`，coverage = `0.016452`，bad-event = `0.236181`。
8. P5 虽然 null rate 降到 `0.020101`，但 precision LCB = `0.719692 < 0.75`，bad-event UCB = `0.280308 > 0.05`，coverage 也低于 `0.03`。
9. P6 true-delta compute v7 仍未过：best = `TBD0-V9256CBD0Reference`，AUC = `0.879072`，agreement = `1.0`，step ratio q90 = `2.863280 > 1.50`。
10. P7-P10 因 P5/P3 gate-blocked，全部以 `not_run` 落盘；当前 blocker = `useful_stat_precision_fail`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9262_useful_control_target_reset_null_rejection_frontier.py` | v9.2.62 runner；执行 v9.2.61 boundary 复现、useful-control target autopsy、null submode/rejector factory、useful statistic reset、support stability audit、exact-reference frontier v6、true-delta compute lane、downstream/no-fake audit |

代码检查：

```text
python -m py_compile experiments/run_v9262_useful_control_target_reset_null_rejection_frontier.py
```

正式运行：

```bash
python experiments/run_v9262_useful_control_target_reset_null_rejection_frontier.py \
  --out-dir results/real_rerun_20260506/v9262_useful_control_target_reset_null_rejection_frontier_first_20260512T170000Z \
  --fresh --device auto --data-root data --seed 1314
```

实际参数来自 `run_manifest.json`：

```text
device = cuda
triton_available = true
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4,5,6,7
microprobe_steps = 336
interface_events = 24
interface_steps = 2
train_size = 2048
batch_size = 64
hidden_dim = 256
completed_at = 2026-05-12T16:30:00Z
```

## 2. Route

`route_decision.json`：

```json
{
  "route": "R12-UsefulTargetStillTiny",
  "base_candidate": "LQ-t2-h256",
  "v9261_boundary_pass": 1,
  "useful_target_autopsy_pass": 1,
  "best_null_rejector_id": "N2-LowGainNullRejector",
  "null_rejector_pass": 0,
  "null_rejector_diagnostic_pass": 1,
  "best_useful_stat_id": "U1-RealGainLCBv2",
  "useful_stat_pass": 1,
  "useful_utility_pass": 0,
  "useful_auc": 0.7031502441307241,
  "useful_precision_after_gate": 0.44693473961766644,
  "useful_coverage_after_gate": 0.0627066798941799,
  "useful_bad_event_after_gate": 0.46605141727092947,
  "support_stability_pass": 1,
  "accepted_support_pass": 1,
  "exact_reference_deployable": 0,
  "reference_precision": 0.7638190954773869,
  "reference_coverage": 0.01645171957671958,
  "reference_bad_event": 0.23618090452261306,
  "true_delta_compute_pass": 0,
  "true_delta_step_ratio_q90": 2.8632798851361203,
  "primary_blocker": "useful_stat_precision_fail",
  "next_required_implementation": "reset_useful_control_target_decomposition"
}
```

判断：support 生成器和 support stability 已经不再是本轮主 blocker；主失败点变成 useful/control target 的 precision 与 bad-event 无法同时达标。

## 3. P1 useful-control target autopsy

Artifacts：

```text
p1_useful_control_target_autopsy.csv
useful_target_trace_v9262.csv
```

Summary：

```text
accepted_count = 5997
useful_rejected_count = 230
harmless_null_accepted_count = 684
bad_event_accepted_count = 2846
positive_useful_submode_count = 4
real_gain_density = 0.540881
control_resistant_density = 0.545800
persistent_density = 0.591807
efficient_density = 0.596189
useful_target_autopsy_pass = 1
```

判断：四个 useful submode 都有足够密度，autopsy 能闭合；但当前 accepted slice 中 bad-event 数量很高，说明 useful target reset 还没形成安全 accept region。

## 4. P2 null submode / rejector factory

Artifacts：

```text
p2_null_submode_decomposition_and_rejector_factory.csv
null_submode_trace_v9262.csv
```

| null rejector | null AUC | corr | precision | coverage | bad-event | null rate | diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|
| N2-LowGainNullRejector | `0.720924` | `0.136753` | `0.550173` | `0.059730` | `0.406228` | `0.139792` | `1` |
| N1-ControlEquivalentNullRejector | `0.836970` | `0.326876` | `0.412773` | `0.079613` | `0.478193` | `0.181205` | `1` |
| N3-NonPersistentNullRejector | `0.859151` | `0.388734` | `0.412773` | `0.079613` | `0.478193` | `0.181205` | `1` |
| N4-DeltaSilentNullRejector | `0.985204` | `0.645778` | `0.343592` | `0.081928` | `0.486377` | `0.112008` | `1` |

Summary：

```text
predictive_null_submode_count = 4
null_rejector_pass = 0
null_rejector_diagnostic_pass = 1
```

判断：null submodes 确实可测，但 null rejector 不能单独解决 bad-event/precision；不能把 diagnostic null separation 写成 official success。

## 5. P3 useful/control statistic reset

Artifacts：

```text
p3_useful_control_statistic_reset.csv
useful_control_stat_trace_v9262.csv
```

| useful stat | reference only | useful AUC | precision | coverage | bad-event | null rate | pass | utility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U1-RealGainLCBv2 | `0` | `0.703150` | `0.446935` | `0.062707` | `0.466051` | `0.143705` | `1` | `0` |
| U2-ControlGapLCBv2 | `0` | `0.828059` | `0.550173` | `0.059730` | `0.406228` | `0.139792` | `1` | `0` |
| U3-PersistentControlGap | `0` | `0.829156` | `0.550173` | `0.059730` | `0.406228` | `0.139792` | `1` | `0` |
| U6-UsefulMixtureScore | `0` | `0.857489` | `0.550173` | `0.059730` | `0.406228` | `0.139792` | `1` | `0` |

判断：useful/control statistic 能找到 broader slice，coverage 过了 `0.03`，但 precision 和 bad-event 远离 official safety gate。这是本轮 route 的 primary blocker。

## 6. P4 support stability after generator pass

Artifacts：

```text
p4_support_stability_audit_after_generator_pass.csv
support_stability_trace_v9262.csv
```

Summary：

```text
natural_real_event_count = 24192
balanced_diagnostic_real_event_count = 6000
measured_signal_strata_count = 27
measured_family_count = 741
duplicate_row_count = 0
accepted_signal_strata_count = 11
accepted_family_count = 71
max_family_share = 0.094924
max_stratum_share = 0.560316
support_stability_pass = 1
accepted_support_pass = 1
```

判断：v9.2.62 保持并扩展了 v9.2.61 的 support generator 成果；support measurement / support stability 不再是阻断 downstream 的第一失败点。

## 7. P5 exact reference deployable frontier v6

Artifacts：

```text
p5_exact_reference_deployable_frontier_v6.csv
reference_frontier_v6_trace.csv
```

Best summary：

```text
best_reference_controller_id = C-U6-UsefulMixtureScore+N2-LowGainNullRejector+R2-UsefulConditionedBadRisk+S4-CoverageBalancedSupportCap
exact_reference_deployable = 0
precision = 0.763819
coverage = 0.016452
bad_event = 0.236181
null_rate = 0.020101
precision_lcb = 0.719692
bad_event_ucb = 0.280308
accepted_signal_strata_count = 5
accepted_family_count = 34
max_family_share = 0.133166
max_stratum_share = 0.600503
```

判断：null rate 明显被压低，但 bad-event 仍过高，coverage 也不够。这个 frontier 不能写成 deployable exact-reference controller。

## 8. P6 true-delta compute v7

Artifacts：

```text
p6_true_delta_compute_v7_parallel_lane.csv
true_delta_compute_v7_trace.csv
```

Summary：

```text
best_true_delta_id = TBD0-V9256CBD0Reference
true_delta_auc = 0.879072
true_delta_agreement = 1.0
true_delta_step_ratio_q90 = 2.863280
true_delta_memory_ratio = 0.969501
true_delta_compute_pass = 0
true_delta_compute_diagnostic_pass = 0
```

判断：exact/true delta signal 仍存在，但 compute path 继续超过 system envelope；即使 P3/P5 后续闭合，P6 仍需要单独解决。

## 9. Downstream boundary

这些 artifact 已落盘为 `not_run`：

| artifact | reason |
|---|---|
| `p7_system_legal_exact_signal_controller.csv` | `P5_reference_deployable_frontier_failed` |
| `p8_leave_dataset_and_stratum_out.csv` | `P3_useful_stat_failed` |
| `p9_official_paired_replay.csv` | `P3_useful_stat_failed` |
| `p10_short_run_functional_validation.csv` | `P3_useful_stat_failed` |

没有把 support stability pass、null diagnostic、useful statistic pass、exact-reference borderline row、oracle support 或 true-delta reference signal 写成 system-legal controller / LDO / paired replay / short-run success。

## 10. No-fake audit

```text
rows_checked = 108891
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
useful_control_target_reset = 1
null_submode_decomposition = 1
support_stability_after_generator_pass = 1
exact_reference_deployable_frontier_v6 = 1
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_controller = 0
```

## 11. Hash

| artifact | SHA256 |
|---|---|
| plan | `5886e89926873f95073954390129fbad56a2601dd4866d6dc6e97afc152a7383` |
| runner | `2e55238476930194db83f6127746ebdba4f1b6c4be4c04b21bef1d2c5e54490a` |
| run manifest | `245a7853f54aa9260be3b0b5e88b02a1869f6db680e0dbe0558bde5260b89670` |
| route | `4bd11b0292da121d016c7d6637315bddf948902e58b4d9233a91c7a8225e7450` |
| P0 boundary | `c102decaf5995031cd853443cfe1c061bbf9c6a0655a44058ced4619adbd885d` |
| P1 useful target | `4ff8bff4ac55ece25cbf6d1953a6207c158339a33de9cda2840815a58fa2f026` |
| P2 null rejector | `8fb4e77e315a60233a83a9771ebe6b38a5b02146a8a3d9707e467ff4036cd1d6` |
| P3 useful statistic | `753b9393487ad05b90ccb2a029b42cc1b342aba9f76a52d7534c10cb8eeabd71` |
| P4 support stability | `4ac392609bae891cbb5049c0ab311f705125311d9603b1262783de45b16cbf3a` |
| P5 deployable frontier | `dcb0a0cd57da616429f797858441b971a9b92e404e19d4865eeda09d17e2b652` |
| P6 compute lane | `ec532b410ff1c0427bf64ab5f76fc5217ccb7b17925b5301a6481fa453828f6c` |
| P7 system controller | `b02e624c8b00d27ac4816397f267440b9a714156325cc52718e34d8bd7b49d33` |
| P8 boundary | `b0e6775ecee5d71a90b7438954b8c99fc653d167d251372d0c2d243f005ab90e` |
| P9 boundary | `b5cf14c81b82a168dc9007bc712a2dd355a7fbfa3bfaa6f54f0ef4fb5bf0e226` |
| P10 boundary | `0ee0c6788f2ffc9d81f7e1acaae4cda4c74af1d4e56d140bcbf209b34a2ccd0d` |
| failure table | `ec8e04ae8e38755553a3363faad231d5520b090dd8660996a435cc9f4eb3f2a6` |
| provenance audit | `19db93920124c435491e6e26ac434ec94baf3ae5a120a10cfeed501806b8224c` |

## 12. 最终分析结论

v9.2.62 的真实推进是：

```text
v9.2.61: support-stratum generator reset 成功，但 value/risk/null/frontier 没闭合。
v9.2.62: support stability 继续过关，null submode 和 useful/control target 均有诊断信号；
          但 useful gate 的 precision/bad-event 失败，exact-reference frontier 仍不可部署。
```

机制判断：

1. H1 成立：useful-control target autopsy 闭合，四个 useful submode 都有足够密度。
2. H2 部分成立：null submode 可预测，`N2` 能把 null rate 压到 `0.139792`，但 bad-event 仍 `0.406228`。
3. H3 未成立：useful/control statistic 能扩大 coverage 到 `0.062707`，但 precision 只有 `0.446935`，bad-event 达到 `0.466051`。
4. H4 成立：support stability after generator pass 过关，rows/strata/family/balance/duplicate gate 都满足。
5. H5 未闭合：exact-reference deployable frontier 的 null rate 很低，但 bad-event/UCB/coverage gate 全部失败。
6. H6 继续未闭合：true-delta compute 仍 `step_ratio_q90 = 2.863280`，超过 `1.50`。

最终一句话：

> v9.2.62 真实执行后停在 `R12-UsefulTargetStillTiny`：support stability 已过关，null/useful 分解有诊断信号，但 useful-control target 仍把大量 bad-event 带入 accepted region，strict PureKAN functional 仍未成功。
