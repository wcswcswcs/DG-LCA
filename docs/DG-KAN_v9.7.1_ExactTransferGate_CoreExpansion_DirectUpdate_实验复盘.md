# DG-KAN v9.7.1 Exact Transfer Gate / Core Expansion / Direct Update 实验复盘

> 本复盘记录 `DG-KAN_v9.7.1_ExactTransferGate_CoreExpansion_DirectUpdate_完整实验计划.md` 的真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 v9.7.0 proxy transfer、exact-transfer missing diagnostic、direct-update boundary、generated-route boundary 或 Base-Acc Sentinel 写成 official system pass。

## 0. 最新结论

```text
route = R1-ExactTransferArtifactMissing
base_candidate = LQ-t2-h256
success_v9710_strict_purekan_functional = False
success_v9710_full_functional = False
success_v9710_external_ready = False
```

最终 artifact：

```text
results/real_rerun_20260506/v9710_exact_transfer_gate_core_expansion_direct_update_first_20260516T000000Z/
```

核心结论：

1. P0 复现 v9.7.0 boundary：source route = `R4-TransferPrincipleFail`，system pass = `0`，generated stop = `1`，field green/yellow/red = `14 / 4 / 0`。
2. P1 exact transfer artifact 没有落地：AP0 action count = `2876`，exact linear transfer rows = `0`，exact apply audit subset = `0`。
3. P1 明确 `exact_per_sample_gradient_available = 0`，`exact_apply_checkpoint_available = 0`，matched exact artifact count = `0`。
4. v9.7.0 proxy transfer ledger 可作为 diagnostic：proxy transfer rows = `2876`，但本轮 `proxy_promoted_to_official = 0`。
5. P2 exact-transfer existing-action selector 按 gate `not_run`；旧 R8A 与 proxy transfer 只作为诊断基线记录。
6. P2 诊断基线：old R8A precision = `0.8735632183908046`，V LCB = `0.14111334880346277`，LDO drop = `0.37931034482758624`。
7. P2 proxy transfer 诊断基线：precision = `0.17525773195876287`，V LCB = `-0.21558700787971152`，LDO drop = `0.07216494845360824`。
8. P3 Core + exact-transfer expansion 按 gate `not_run`；v9.7.0 core diagnostic 仍是 core count = `77`，needed expansion to 87 = `10`。
9. P4 exact no-tuning dataset transport 按 gate 未打开；只记录 v9.7.0 raw score diagnostic：PSI mean = `0.31313509863308747`。
10. P5 direct transfer-solved update 按 gate `not_run`：subspace count = `6`，generated actions = `0`，branch-horizon rows = `0`。
11. P6 proxy-vs-exact taxonomy 没有打开：proxy-exact pair count = `0`，dominant failure class = `F0-exact-transfer-artifact-missing`。
12. P7/P8/P10/P11 gate-blocked：existing-action controller、selected runtime、system controller、paired replay 和 short/full boundary 均未打开。
13. P9 generated route 继续停止：status = `stopped_no_new_objective`，APGV/APGW allowed = `0`，generated actions = `0`。
14. Base-Acc Sentinel 继续健康：rows = `120`，LQ mean test acc = `0.6537760416666667`，AdamWStrongLRGridMLP = `0.628515625`；没有用于 controller。
15. 当前 primary blocker：`exact_transfer_artifact_missing`；secondary blocker：`generated_route_stopped_no_new_objective`。

## 1. 本轮代码与命令

| 文件 | 作用 |
|---|---|
| `experiments/run_v9710_exact_transfer_gate_core_expansion_direct_update.py` | v9.7.1 runner；读取 v9.7.0 artifacts，执行 exact-transfer artifact gate、exact selector/core-expansion/direct-update boundaries、generated route decision 与 system boundary |

代码检查：

```text
python -m py_compile experiments/run_v9710_exact_transfer_gate_core_expansion_direct_update.py
```

正式运行：

```bash
python experiments/run_v9710_exact_transfer_gate_core_expansion_direct_update.py \
  --out-dir results/real_rerun_20260506/v9710_exact_transfer_gate_core_expansion_direct_update_first_20260516T000000Z \
  --fresh --device auto --data-root data --seed 1314 \
  --direct-actions-per-subspace 64
```

运行结果：

```json
{
  "direct_solved_generated_actions": 0,
  "exact_linear_transfer_row_count": 0,
  "exact_per_sample_gradient_available": 0,
  "generated_route_status": "stopped_no_new_objective",
  "out_dir": "results/real_rerun_20260506/v9710_exact_transfer_gate_core_expansion_direct_update_first_20260516T000000Z",
  "primary_blocker": "exact_transfer_artifact_missing",
  "route": "R1-ExactTransferArtifactMissing",
  "secondary_blocker": "generated_route_stopped_no_new_objective",
  "system_legal_controller_pass": 0
}
```

说明：本轮没有生成 fake exact rows，没有把 v9.7.0 proxy transfer 写成 exact transfer，也没有在 generated route stop 的状态下恢复 APGV/APGW blind run。

## 2. Route

`route_decision_v9710.json` 摘要：

```json
{
  "route": "R1-ExactTransferArtifactMissing",
  "source_route_v9700": "R4-TransferPrincipleFail",
  "p0_pass": 1,
  "P1_exact_transfer_weak_pass": 0,
  "P1_exact_transfer_strong_pass": 0,
  "exact_per_sample_gradient_available": 0,
  "exact_apply_checkpoint_available": 0,
  "exact_linear_transfer_row_count": 0,
  "exact_apply_audit_subset_count": 0,
  "p2_exact_selector_pass": 0,
  "p3_core_exact_expansion_pass": 0,
  "dataset_shift_score_scale_solved": 0,
  "direct_solved_weak_pass": 0,
  "direct_solved_strong_pass": 0,
  "new_objective_evidence_present": 0,
  "generated_route_status": "stopped_no_new_objective",
  "APGV_APGW_allowed": 0,
  "system_legal_controller_pass": 0,
  "primary_blocker": "exact_transfer_artifact_missing",
  "secondary_blocker": "generated_route_stopped_no_new_objective"
}
```

判断：v9.7.1 没有证明 exact transfer principle 失败；它证明的是 exact transfer 所需 landed artifact 仍缺失。由于 P1 没有 weak pass，P2/P3/P5 不能打开，route 必须停在 `R1-ExactTransferArtifactMissing`。

## 3. P0 boundary

Artifacts：

```text
p0_boundary_reproduction_v9710.csv
p0_field_legality_audit_v9710.csv
```

Summary：

```text
source_route_v9700 = R4-TransferPrincipleFail
system_legal_controller_pass_v9700 = 0
generated_stop_v9700 = 1
source_primary_blocker_v9700 = cross_sample_transfer_not_better_than_old_rank
source_secondary_blocker_v9700 = generated_route_stopped_no_new_objective
field_green/yellow/red = 14 / 4 / 0
uses_dataset_name_for_controller = 0
uses_outcome_derived_field = 0
uses_validation_or_test = 0
fake_row_count = 0
proxy_official_count = 0
cpu_offload_used = 0
manual_forward/manual_backward/manual_update = 1 / 1 / 1
p0_pass = 1
```

判断：P0 pass。v9.7.1 没有跳过 v9.7.0 的 transfer failure / generated stop boundary，也没有使用 dataset name、outcome-derived field 或 validation/test 信息进入 official path。

## 4. P1 exact transfer artifact

Artifact：

```text
p1_exact_transfer_artifact_v9710.csv
```

Summary：

```text
canonical_ap0_action_count = 2876
proxy_transfer_row_count_v9700 = 2876
exact_linear_transfer_row_count = 0
exact_apply_audit_subset_count = 0
missing_exact_linear_row_count = 2876
missing_required_field_count = 2876
nan_inf_count = 0
exact_per_sample_gradient_available = 0
exact_apply_checkpoint_available = 0
exact_artifact_candidate_count = 0
P1_exact_transfer_weak_pass = 0
P1_exact_transfer_strong_pass = 0
reason = exact_per_sample_gradient_and_apply_checkpoint_not_landed
proxy_transfer_available_for_diagnostic = 1
proxy_promoted_to_official = 0
```

Scan row：

```text
scan_scope = results/real_rerun_20260506/v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z
matched_exact_artifact_count = 0
v9700_exact_microprobe_status = not_run
v9700_exact_microprobe_reason = exact_per_sample_apply_checkpoint_not_landed
```

判断：P1 是本轮 terminal blocker。v9.7.0 的 2876 行 proxy transfer ledger 存在，但 exact per-sample gradient / exact apply checkpoint 没有落盘，因此不能构造 exact transfer selector、exact core expansion 或 direct solved update。

## 5. P2 exact-transfer selector

Artifact：

```text
p2_exact_transfer_selector_existing_action_v9710.csv
```

Status：

```text
status = not_run
reason = P1_exact_transfer_artifact_not_available
rule_count = 0
evaluation_row_count = 0
exact_selector_strong_pass = 0
exact_selector_weak_pass = 0
```

Diagnostic baselines from v9.7.0：

```text
old_R8A_precision = 0.8735632183908046
old_R8A_V_LCB = 0.14111334880346277
old_R8A_LDO_drop = 0.37931034482758624
proxy_transfer_precision = 0.17525773195876287
proxy_transfer_V_LCB = -0.21558700787971152
proxy_transfer_LDO_drop = 0.07216494845360824
```

判断：P2 没有执行 exact selector。旧 rank 与 proxy transfer 只是复现 v9.7.0 的诊断边界，不能被写成 exact transfer result。

## 6. P3 Core + exact-transfer expansion

Artifact：

```text
p3_core_expansion_exact_transfer_v9710.csv
```

Status：

```text
status = not_run
reason = P1_exact_transfer_artifact_not_available
candidate_count = 0
candidate_pass_count = 0
core_count_diagnostic_from_v9700 = 77
needed_expansion_to_87 = 10
p3_core_exact_expansion_pass = 0
proxy_core_expansion_best_candidate_diagnostic = B0-T3.1-CoreOnly
proxy_core_expansion_best_precision_diagnostic = 1.0
proxy_core_expansion_best_LDO_drop_diagnostic = 0.39080459770114945
```

判断：Core 77 的 diagnostic 仍然成立，但 exact expansion 没有 artifact 支撑，所以不能补 10 rows，也不能打开 controller。

## 7. P4 dataset-shift exact no-tuning

Artifact：

```text
p4_dataset_shift_exact_no_tuning_v9710.csv
```

Summary：

```text
exact_transport_available = 0
score_count = 0
raw_score_psi_mean_diagnostic_v9700 = 0.31313509863308747
dataset_shift_score_scale_solved = 0
dataset_specific_selector_produced = 0
reason = P1_exact_transfer_artifact_not_available
```

Raw score diagnostic from v9.7.0：

```text
TopK87_precision = 0.8735632183908046
TopK87_V_LCB = 0.14111334880346277
LDO_drop = 0.37931034482758624
per_dataset_TopK_count = {"Fashion-MNIST": 34, "KMNIST": 36, "MNIST": 17}
per_dataset_TopK_precision = {"Fashion-MNIST": 1.0, "KMNIST": 0.7777777777777778, "MNIST": 0.8235294117647058}
```

判断：没有 exact transport，因此 dataset-shift score-scale 不能被本轮 solved。dataset 仍只用于 diagnostic / leaveout，没有进入 selector。

## 8. P5 direct transfer-solved update

Artifact：

```text
p5_direct_transfer_solved_update_v9710.csv
```

Summary：

```text
subspace_count = 6
generated_action_count = 0
branch_horizon_rows_expected = 0
branch_horizon_rows_actual = 0
negative_control_generated = 0
direct_solved_weak_pass = 0
direct_solved_strong_pass = 0
new_objective_evidence_present = 0
reason = P1_exact_transfer_weak_pass_failed
```

Subspaces：

```text
D1-last-edge-coefficients-only = not_run
D2-final-KAN-basis-block-only = not_run
D3-low-rank-edge-residual-direction = not_run
D4-AdamW-orthogonal-residual-direction = not_run
D5-memory-gradient-orthogonal-residual-direction = not_run
D6-low-cost-transfer-positive-intersection = not_run
```

判断：P5 是诚实 gate-block。没有 exact transfer weak pass，就不能直接求 small transfer-optimal update，也不能生成 branch-horizon rows。

## 9. P6 proxy vs exact taxonomy

Artifact：

```text
p6_proxy_exact_failure_taxonomy_v9710.csv
```

Summary：

```text
taxonomy_opened = 0
proxy_exact_pair_count = 0
dominant_failure_class = F0-exact-transfer-artifact-missing
reason = P1_exact_transfer_artifact_not_available
```

判断：没有 proxy-exact matched pairs，因此不能判断 proxy sign/scale 是否错，也不能判断 exact transfer 本身是否 weak。当前只能归因到 exact artifact missing。

## 10. Controller / runtime / generated route / system

P7：

```text
p7_existing_action_minimal_controller_v9710.csv = not_run
reason = P2_P3_no_exact_transfer_selector_or_expansion_pass
controller_selected = 0
controller_pass = 0
source_controller_pass = 0
```

P8：

```text
p8_selected_runtime_v9710.csv = not_run
reason = P7_controller_not_selected
selected_runtime_pass = 0
```

P9：

```text
p9_generated_route_decision_v9710.csv
generated_route_status = stopped_no_new_objective
reason = P1_exact_transfer_fail_and_P5_not_open
new_objective_evidence_present = 0
APGV_APGW_allowed = 0
generated_action_count = 0
branch_horizon_rows_actual = 0
generated_route_stop_triggered = 1
```

P10/P11：

```text
p10_system_boundary_v9710.csv = not_run
reason = P7_or_P8_not_passed
system_legal_controller_pass = 0

p11_paired_replay_short_full_boundary_v9710.csv = not_run
reason = P10_system_not_official
paired_replay_pass = 0
short_run_boundary_open = 0
full_run_boundary_open = 0
```

Allowed next gates：

```json
{
  "selected_runtime_allowed": 0,
  "paired_replay_allowed": 0,
  "short_full_allowed": 0,
  "APGV_APGW_allowed": 0,
  "exact_transfer_artifact_required": 1,
  "direct_update_allowed": 0
}
```

判断：没有 exact-transfer weak pass，也没有 controller/runtime pass；generated route 继续停止，不允许 APGV/APGW blind run。

## 11. Base-Acc Sentinel

Artifact：

```text
base_acc_sentinel_v9710.csv
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
base_acc_sentinel_pass = 1
```

判断：Base-Acc Sentinel 继续健康，但仍是 isolated diagnostic，没有用于 selector、generator、certificate 或 controller。

## 12. Figures

本轮额外落盘的诊断图：

```text
fig_p1_linear_transfer_vs_exact_apply_scatter.svg
fig_p1_per_dataset_response_scale_hist.svg
fig_p1_response_distribution_violin.svg
fig_p1_response_mean_vs_std.svg
fig_p2_exact_transfer_precision_vs_ldo.svg
fig_p2_exact_vs_old_rank_frontier.svg
fig_p3_core_exact_expansion_sankey.svg
fig_p3_exact_expansion_quality_table.svg
fig_p4_exact_transport_psi.svg
fig_p5_direct_negative_control.svg
fig_p5_direct_solved_frontier.svg
fig_p6_proxy_exact_mismatch.svg
```

判断：这些图只用于记录 gate-block / missing-artifact 状态，不构成 official pass。

## 13. No-fake audit

```text
rows_checked = 142
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
v9700_boundary_pass = 1
field_legality_pass = 1
exact_transfer_artifact_pass = 0
exact_transfer_weak_pass = 0
exact_selector_pass = 0
core_exact_expansion_pass = 0
dataset_shift_score_scale_solved = 0
direct_solved_pass = 0
generated_route_stop = 1
controller/runtime/system = 0/0/0
base_acc_sentinel_pass = 1
base_acc_used_for_controller = 0
uses_loss_backward/teacher/loss_modification = 0/0/0
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test_for_controller = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
uses_old_table_for_official = 0
proxy_transfer_promoted_to_official = 0
diagnostic_promoted_to_official = 0
fake/proxy/cpu_offload = 0/0/0
```

Failure table：

```text
route = R1-ExactTransferArtifactMissing
F0_boundary_or_legality_fail = 0
F1_exact_transfer_artifact_missing = 1
F2_exact_transfer_selector_fail = 0
F3_core_expansion_exact_fail = 0
F4_direct_solved_update_not_open = 1
F5_generated_route_stopped_no_new_objective = 1
F6_controller_runtime_blocked = 1
F7_system_not_official = 1
F8_base_acc_catastrophic = 0
primary_blocker = exact_transfer_artifact_missing
secondary_blocker = generated_route_stopped_no_new_objective
```

## 14. Hash

| artifact | SHA256 |
|---|---|
| plan | `124cfc9018eecd7bd92ab846835430fbe45af0ae1102bfef538d86348e0bfc29` |
| runner | `1641dad9e54e0e1d404e86f5523dc008fe07007fe211f43b954f23c9359c23e1` |
| run manifest | `852fe63cacb9b9a1d3741ddf430979c8c9d18fad9f58686c2f9a619e9e845fed` |
| route | `e9a55ec2b80c1c9ee6388785fba5de9fdcca9ff8327a12a254b33029f7ff46ff` |
| P0 boundary | `10a872766dc13b72e08ef56c057a2cfd71ab4d55247bced68eecff978e18cdd0` |
| P0 field legality | `3e0b68abff2aa094c4585056d5af91703b174e4fb0b01b5e01953a0a9709642c` |
| P1 exact transfer | `c4e5407d94b6522c91c79cae93ded8532a5a2107019930b8fc475a787f02be59` |
| P2 exact selector | `5c21c6e477f976f089dec2b870ced3f45612d6d3d7906835d49f92d264a09cbe` |
| P3 core expansion | `f2062406caac5276241b60f4e85f7aff0a86516be114bf11ab3cc60857bb4e49` |
| P4 dataset shift | `b34fc2c395db409acf128d960e1d1d607e67b01995676000b3e6f4ff9107f80d` |
| P5 direct update | `33f91422e772ff49c857a2ec1f69cb5262a43293ef75419db2978f8750311ef8` |
| P6 taxonomy | `4cd48a7317de03254f004346702e46420f44abdd9c73ef8b64c04e8e0d69f712` |
| P7 controller | `c52a7f0db559f821ff017ac57777d8b85ff71c9ae0ef644d0b48b3d3a8245934` |
| P8 runtime | `675b476c02637d709d4b95cc1974ab2551d3b4b1cc1d33f33cd436efbb54a33f` |
| P9 generated route | `e091aafdaaa07eee6782d34e7cc82bbb5a3625fb794fa0a7a7c532455bdfcbb1` |
| P10 system | `c0a616b5798c96d18271796acba6ad0732f8fac06ea1173bb79b59244432d647` |
| P11 paired/short-full | `80122bdb8e5065b01cbdcca209e158652ddc8332daae82d2f2314b64b5f9f9a1` |
| Base-Acc Sentinel | `65a2fcb3707fcc1db56fad022f97d48f4f7e2bd556ef5882abfab3e360fa09f4` |
| no-fake audit | `712f3e5fd8f3e46c907c4688d530657d24bca2653d13424ea5225b7d8ad71d0d` |
| contract audit | `5d9a85570145093dffe8520da5d88a89a8fd8c15295a51041bc7ddfd0169113d` |
| failure taxonomy | `3a91f0b1077d3b6ee05a5d4500ef0a46d6d1d15e7a7a321c80f8667e287903b2` |
| allowed next gates | `408f53fc76b827fa381f4df5bc178bea1d3db86d2808c12950b5026d6b61f44e` |
| stop conditions | `acb6392b6c08d88c02110b008fd7ffb4824c9146351275fede6867a30df3f93f` |

## 15. 最终分析结论

v9.7.1 的真实推进是：

```text
v9.7.0:
  landed proxy transfer ledger 完整；
  old R8A rank 质量高但 LDO 不稳；
  proxy TransferLCB 降低 LDO 但 precision/value 崩；
  exact microprobe 与 direct solved update 未打开。

v9.7.1:
  按计划把 exact transfer gate 放在最前；
  复现 v9.7.0 boundary 与 field legality；
  扫描 v9.7.0 source artifact，确认 exact per-sample gradient / exact apply checkpoint 仍未落地；
  因 P1 exact artifact 缺失，P2 exact selector、P3 core+exact expansion、P5 direct solved update 全部 gate-blocked；
  generated route 继续保持 stopped_no_new_objective；
  controller/runtime/system/paired replay 均未打开。
```

机制判断：

1. H0 成立：v9.7.0 boundary 被复现，没有跳过 transfer failure / generated stop / no system controller。
2. H1 未成立：exact transfer artifact 没有落地，exact linear rows = `0`，exact apply subset = `0`。
3. H2 未打开：没有 exact transfer weak pass，就不能验证 exact selector 是否优于 old R8A 或 proxy TransferLCB。
4. H3 未打开：Core 77 + exact expansion 不能运行，因此无法判断 exact transfer 是否能补到 87 且降低 LDO。
5. H4 未打开：dataset-shift exact transport 不可运行；dataset 仍只作为 diagnostic / leaveout。
6. H5 未打开：direct small update 没有 exact objective evidence，generated actions = `0`。
7. H6 成立于边界层：proxy transfer 没有被提升为 official exact transfer，diagnostic_promoted_to_official = `0`。
8. H7 成立：generated route 继续停止，APGV/APGW allowed = `0`。
9. H8-H10 未打开：没有 controller/runtime/system pass，就不能打开 paired replay 或 short/full training。
10. Base-Acc Sentinel 继续健康，但没有用于 controller，也不是 functional success。

最终一句话：

> v9.7.1 真实执行后停在 `R1-ExactTransferArtifactMissing`：v9.7.0 的 proxy transfer ledger 仍只能作为 diagnostic，exact per-sample gradient 与 exact apply checkpoint 没有落地，因此 exact selector、core+exact expansion 和 direct solved update 全部不能打开；generated route 继续处于 no-new-objective stop，strict PureKAN functional 仍未成功。
