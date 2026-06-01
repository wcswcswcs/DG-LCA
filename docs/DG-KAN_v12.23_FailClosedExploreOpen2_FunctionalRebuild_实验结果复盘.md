# DG-KAN v12.23 FailClosedExploreOpen2 FunctionalRebuild 实验结果复盘

生成时间：2026-05-25（Asia/Singapore）

本复盘基于实际生成 artifacts，不编造未执行数据。最终结果不是 promotion success，而是硬预算完成后的 no-go 路由。

## 1. 最终路由

最终 route：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `minimum_success=Minimum Success F`
- `fail_reason=all mandatory exploration levels exhausted without promotion`
- `official_success_reached=0`
- `final_stop_allowed=1`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `required_artifact_missing_count=0`

解释：v12.23 的 final stop 不是因为成功，而是因为强制探索层级均已执行，required artifacts 不缺失，Line D 也不再被跳过。

## 2. Line A：Label-free signal frame

执行规模：

- Scout：`linea_scout_rows=126`
- Hardening：`linea_hardening_rows=63`
- 总 Line A 输入：`linea_rows=189`

Scout 全局 legal top-2：

1. `A45-StagedUnlabeledAdapt-warm1-r005`
2. `A41-UnlabeledFrameAdaptSchedule-labelFree`

Hardening 汇总：

| candidate | rows | mean_delta_vs_A0 | worst_delta_vs_A0 | max_AUC_time_ratio_vs_mlp | LineC pass rate | promotion |
|---|---:|---:|---:|---:|---:|---:|
| A1-noYForStats | 9 | -0.004557291666666667 | -0.021484375 | 1.1525883088110533 | 0.0 | 0 |
| A36-ResidualA1LowRankFrame-labelFree | 9 | -0.004774305555555556 | -0.021484375 | 1.1260724072238704 | 0.0 | 0 |
| A41-UnlabeledFrameAdaptSchedule-labelFree | 9 | -0.010199652777777778 | -0.03515625 | 1.0270575170134097 | 0.0 | 0 |
| A45-StagedUnlabeledAdapt-warm1-r005 | 9 | -0.00390625 | -0.021484375 | 1.0403984745484147 | 0.0 | 0 |

结论：

- A45 是本轮 hardening 最强候选，但仍未通过 official gate：worst delta 不达标，LineC pass rate 为 0。
- A45 也未通过 exploration gate：虽然 mean/worst/AUC_time 接近 exploration 阈值，但 LineC pass rate 需要至少 5/9，实际为 0/9。
- 当前 Line A 仍是“任务接近但 Line C 非撕裂条件不成立”的形态，不能升级为 label-free recovered。

## 3. Line T/C：Loss-agnostic visibility

关键数据：

- `T1A_auc_joint=0.3434466019417476`
- `T1A_v1223_official_gate_pass=0`
- `T1B_auc_joint=0.5045509708737864`
- `T1B_native_logged_rows=135`
- `T1B_v1223_exploration_gate_pass=0`
- `visibility_component_ablation_rows=6`
- `t1b_calibration_rows=2`

结论：

- T1A 远低于 official AUC 门槛 0.70。
- T1B 有 native optimizer update rows，但 AUC 只有约 0.505，低于 exploration 门槛 0.60。
- 本轮没有得到可用于 opening official P4 的 loss-agnostic value source。

## 4. Line I：Role-aware actuator

执行规模：

- `actuator_rows=1512`
- `actuator_role_safe_movement_rows=387`
- `actuator_role_safe_dataset_seed_count=9`
- `actuator_release_audit_rows=112`
- `actuator_release_dataset_seed_count=5`
- `actuator_exploratory_release_dataset_seed_count=0`
- `actuator_official_gate_rows=0`

结论：

- role-aware gate 修正后，movement 不再被 projector-angle 单一指标误杀：387 行通过 role-safe movement。
- 112 行产生 release audit 通过，但没有行同时满足 role-safe movement、release 和 matched-control gap，因此 exploratory release dataset/seed count 为 0。
- Line I 当前 blocker 是 control-resistant release，不是没有 actuator movement。

## 5. Line B：Functional P3/P4

关键数据：

- `functional_p3_rows=6`
- `functional_p3_pass_count=0`
- `p4_open=0`
- `p4_pass=0`

结论：

- 没有 actuator 行通过 cloned P3 gate，因此 P4 未打开。
- `v1223_shadow_to_cloned_p3.csv` 已写入 shadow-to-cloned 诊断建议，但 promotion 仍关闭。

## 6. Line D：Classic family smoke

v12.23 已执行每个 active family 的 smoke，不再是 deferred。

| family | status | candidate | val_acc | LineC CouplingR2 | NoiseSignalLeak | RealSignalReservoirRatio |
|---|---|---|---:|---:|---:|---:|
| Rational | FamilyNearPass | B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3 | 0.640625 | 0.1688038217034734 | 0.08171307295560837 | 0.8682652711868286 |
| Chebyshev | FamilyNearPass | B3e-ChebyKAN-K3-tritonL3-matmulTile | 0.125 | 0.03536240317259809 | 0.042757995426654816 | 0.6884616017341614 |
| Wavelet | FamilyNearPass | B5h-HatWaveletKAN-local-K4 | 0.125 | 0.01006836578887671 | 0.46543726325035095 | 0.6175047755241394 |
| RBF/FastKAN | FamilyNearPass | B2s-GaussianRBF-stream-K4-recompute | 0.125 | 0.012388832848461817 | 0.611052930355072 | 0.8501070141792297 |
| Fourier | FamilyNearPass | B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile | 0.296875 | 0.09257388126333477 | 0.230330228805542 | 0.7598191499710083 |

修复记录：

- 最初 Line D 复用 B320 `train_one`，五个 family 均被 `direct_readout` 属性错误挡住。
- 已修复为独立 autograd smoke，最终 `line_d_kernel_blocked_count=0`，`line_d_executed_family_count=5`。

结论：

- Rational 在极小 smoke 下 task 表现最好，但 RealSignalReservoirRatio 很高，不能视为机制 pass。
- Chebyshev/Wavelet/RBF/Fourier 仅达到 smoke 执行和指标记录，不满足 functional promotion。
- Line D 本轮完成“执行义务”，但没有 FamilyPass。

## 7. Line R：代码审计与 artifact 完整性

关键数据：

- `code_semantics_review_pass=1`
- `dirty_tree_rows=926`
- `dirty_tree_unexplained_change_count=0`
- `diff_isolation_rows=874`
- `feature_provenance_unknown_count=0`
- `actuator_role_gate_unknown_count=0`
- `required_artifact_missing_count=0`

注意：工作树存在大量历史 dirty/deleted docs 和既有代码修改。本轮没有回滚这些变化，而是在 `v1223_dirty_tree_audit.csv` 中逐项分类。v12.23 本轮依赖路径已写入 `v1223_core_symbol_map.json` 和 `v1223_diff_isolation_manifest.csv`。

代码审计包：

- 路径：`results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip`
- sha256：`a4ef59864511fab816a48a746aa517ce85f93f576e7c4de7535fc672b788e31a`
- entries：43
- 已包含新 runner、A43-A50 支撑代码、primitive 修改和 v12.23 计划文档。

## 8. 本轮代码修改是否合理

本轮修改均围绕 v12.23 计划要求：

- A43-A50 是计划指定候选，不是临时编造候选。
- `reslowrankpNNN` 和 `adaptframeschedp005` 是让计划中的 conservative strength 真正生效。
- v12.23 runner 产出计划要求 artifacts，并显式阻止重复读取 combined Line A CSV。
- Line D 独立 smoke 修复是为了避免 B320 专属训练路径污染经典 family 审计。

没有修改已有实验结果数据；所有汇总指标来自本轮实际 CSV/JSON。

## 9. 下一步建议

当前最明确 blocker：

1. Line A：A45 接近 task gate，但 LineC pass rate 为 0。下一轮应优先做 LineC-aware repair，而不是只提高 task accuracy。
2. Line I：movement 和 release 都存在，但 matched-control resistant gate 为 0。下一轮应针对 control gap 设计 role-matched controls 和 bisection。
3. Line T/C：T1B native logging 已有 135 行，但 AUC 仍接近随机。需要重构目标可见性特征，不应把 T1B 推进 official。
4. Line D：Rational 有 smoke 亮点，但 reservoir 过高；如果继续 Line D，应优先 Rational 的 reservoir 降噪/denominator-safe 方向。

## 10. 追加只读复核与完成判定（2026-05-25）

用户再次追问 v12.23 是否完成后，我执行了只读复核。第一次复核把 artifact manifest 文件误判为 JSON，实际文件是 `v1223_required_artifact_manifest.csv`，因此第一次命令失败并暴露了路径错误。该错误不是实验 blocker，也没有修改任何实验结果。

修正为读取 CSV manifest 后，复核结果如下：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `minimum_success=Minimum Success F`
- `final_stop_allowed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `required_artifact_missing_count=0`
- `manifest_rows=46`
- `manifest_missing_count=0`
- `linea_rows=189`
- `line_d_executed_family_count=5`
- `line_d_kernel_blocked_count=0`
- `actuator_rows=1512`
- `zip_entries=43`
- `has_runner=True`
- `has_ablation=True`
- `has_primitives=True`
- `zip_sha256=a4ef59864511fab816a48a746aa517ce85f93f576e7c4de7535fc672b788e31a`
- `exec_log_exists=True`
- `review_log_exists=True`

完成判定：v12.23 已完成。完成含义是计划内 mandatory exploration、fallback、Line D smoke、artifact manifest、代码审计 zip 与最终停止审计均已闭环；不是 promotion success。由于 `final_stop_allowed=1` 且 `required_artifact_missing_count=0`、`manifest_missing_count=0`，继续在 v12.23 上追加运行不符合本轮 fail-closed 结束语义。下一步应进入新版本计划，而不是在 v12.23 内无限延长。

## 11. 纠正后的继续执行结果（2026-05-25）

用户指出计划要求继续后，重新核对 v12.23 原计划，确认上一节的“完成判定”过早：我把 runner 的 `mandatory_exploration_levels_executed=1` 当作充分证据，但计划 fallback 表还要求 hardening fail 后继续 failure-specific repair，以及 T1A/T1B fail 后执行 T2 upper-bound diagnostic。随后已继续执行并重新生成最终 route/zip。

### 11.1 本次新增代码修改

修改内容：

- 在 `experiments/run_v1218_b320_label_free_ablation.py` 新增 A51-A54：
  - `A51-StagedUnlabeledAdapt-warm1-r002`
  - `A52-StagedUnlabeledAdapt-warm1-r001`
  - `A53-ResidualA1LowRankFrame-r002-fixedP`
  - `A54-A1FixedPBranchLowQuad020`
- 在 `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py` 注册 A51-A54，并新增 `v1223_t2_clone_probe_visibility_diagnostic.csv`。
- T2 artifact 明确是 diagnostic-only：`promotion_allowed=0`、`clone_probe_only=1`、`precommit_available=0`。它使用已执行 clone/LineC response observables，不是 online direction source。
- 重新打包后的 code review zip 已包含这些修改。

这些修改的合理性：

- A51/A52 对应计划建议的 lower residual energy + staged warmup。
- A53 对应 lower residual + freeze main projector。
- A54 对应 branch/role-energy 方向的 low-quad/fixedP 修复。
- T2 diagnostic 对应计划 fallback 表中 “T1B all fail -> T2 upper-bound diagnostic + no-go proof”。

### 11.2 Line A Level 3 repair 结果

本次新增 Level 3 repair hardening：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val/test = 512
epochs = 8
LineC batch = 64
sketch_dim = 24
```

新增 raw rows：81。最终纳入 official runner 后：

- `linea_rows=277`
- `linea_scout_rows=133`
- `linea_hardening_rows=144`

Level 3 aggregate：

| candidate | mean_delta_avg | worst_min | auc_max | LineC_avg | 结论 |
|---|---:|---:|---:|---:|---|
| A1-noYForStats | -0.0013020833333333337 | -0.017578125 | 1.1653115504975817 | 0.0 | mean 接近/过 official，但 worst、AUC、LineC fail |
| A45-StagedUnlabeledAdapt-warm1-r005 | -0.003038194444444444 | -0.029296875 | 1.1065331974809605 | 0.0 | 仍 fail，接近但未过 worst/LineC |
| A51-StagedUnlabeledAdapt-warm1-r002 | -0.0023871527777777775 | -0.0234375 | 1.1064237918237636 | 0.0 | 降低 strength 改善 mean，但 worst/LineC 仍 fail |
| A52-StagedUnlabeledAdapt-warm1-r001 | -0.002821180555555556 | -0.02734375 | 1.1063864086995177 | 0.0 | 同上，未打开 exploration |
| A53-ResidualA1LowRankFrame-r002-fixedP | -0.09483506944444446 | -0.166015625 | 2.0610389504270414 | 0.0 | fixedP lowrank repair 明显伤 task/AUC |
| A54-A1FixedPBranchLowQuad020 | -0.09657118055555554 | -0.13671875 | 2.2795644556702754 | 0.0 | branch-lowquad/fixedP 方向明显失败 |

关键解读：

- A51/A52 说明 residual strength 过高不是唯一 blocker；把 strength 从 0.005 降到 0.002/0.001 后，mean delta 可以接近 official gate，但 LineC 仍为 0，worst 仍不过。
- A53/A54 说明“冻结主 P + branch/lowquad”不是有效修复，反而显著破坏 task/AUC。
- 本轮新增修复后，`label_free_exploration_pass_count=0`，仍不能 promotion。

### 11.3 T2 clone-probe diagnostic 结果

新增 artifact：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_t2_clone_probe_visibility_diagnostic.csv
```

关键数据：

- `T2_clone_probe_support_count=277`
- `T2_clone_probe_positive_count=34`
- `T2_clone_probe_auc_joint=0.7762043088840475`
- `T2_clone_probe_precision_at_k=0.3235294117647059`
- `T2_clone_probe_recall_at_k=0.3235294117647059`
- `T2_clone_probe_promotion_allowed=0`

解读：

- T2 response 上界可以较好地区分部分 LineC non-tearing rows，说明“事后 response/LineC 可见性”并非完全不存在。
- 但 T2 使用 clone/response observables，`precommit_available=0`，不能作为 official functional direction source。
- 这强化了一个结论：当前真正 blocker 不是完全没有可见信号，而是缺少 precommit-safe、label-free、loss-agnostic 的可部署 visibility source。

### 11.4 重新生成的最终 route

重新运行 `official_explore_open2_continued` 后：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `minimum_success=Minimum Success F`
- `final_stop_allowed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `required_artifact_missing_count=0`
- `required_artifact_rows=47`
- `zip_entries=44`
- `zip_sha256=b585c681b5af29a3cda13b05352786c49b21e6502ffb145e42269c83b469dff8`

补充后的完成判定：

v12.23 现在才可以判定完成。完成含义仍不是 success，而是：

1. Line A 已从 scout/hardening 继续推进到 failure-specific repair；
2. T1A/T1B fail 后补了 T2 upper-bound diagnostic；
3. Line I/B、Line D、Line R 仍保持已执行状态；
4. 最终 required artifacts 无缺失；
5. code review zip 已更新并包含新增代码和 T2 artifact。

最终科学结论：

- label-free base 仍未恢复：A51/A52 接近 task mean gate，但 worst/AUC/LineC 不过。
- deployable value source 仍不可用：T1A/T1B 仍 fail，T2 只是 diagnostic upper bound。
- actuator 仍未打开：role-safe movement 和 release 各自存在，但没有 control-resistant exploratory release。
- functional P3/P4 仍关闭。
- Line D 已完成真实 smoke，但没有 FamilyPass。

因此 v12.23 的最终结论是更强的 no-go，而不是 promotion。

## 12. Line I controls-win fallback 继续执行结果（2026-05-25）

再次按计划第 13.7 与第 19 节审计后，发现 Line I 仍触发一个未充分闭合的 fallback：

```text
Actuator release but controls win -> AdamW-orthogonal residual + matched role controls
If logit drift exceeds gate -> norm bisection
```

旧结果中存在 safe+release rows，但没有任何行同时满足 `control_gap >= 0.002`。因此继续修改并执行。

### 12.1 新增修改

在 `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py` 新增：

- `I18-AdamWOrthogonalResidual-diagnostic`
- I18 从 residual 中减去 AdamW-parallel component。
- I18 标记为 diagnostic-only：
  - `diagnostic_only=1`
  - `uses_label=1`
  - `uses_ce_vector=1`
- I18 不允许打开 promotion/exploration gate。
- 新增 I18 低预算 bisection：`1e-05,2.5e-05,5e-05,0.0001,0.0002`
- 同预算补跑 matched controls，避免 control_gap 缺失。

修改合理性：

- 这正对应计划中 “subtract AdamW-parallel component; add matched role-energy controls”。
- 因 I18 使用 CE-derived AdamW update 作为正交化参考，不能作为 official loss-agnostic direction，只能是 diagnostic fallback。
- 因首轮 I18 drift 超过 gate，继续做 norm bisection，符合计划。

### 12.2 I18 结果

最终 run：

```text
run_id=official_explore_open2_continued_i18_bisect_controls
```

关键数据：

- `actuator_rows=2340`
- `actuator_release_audit_rows=150`
- `actuator_exploratory_release_dataset_seed_count=0`
- `I18 all rows=198`
- `I18 safe=0`
- `I18 release=38`
- `I18 exploratory=0`
- `I18 gap max=0.00018125027418136597`
- `I18 drift min=0.24471133947372437`
- `I18 drift <= 0.05 rows=0`
- `I18 bisection rows=90`
- `I18 bisection safe=0`
- `I18 bisection release=26`
- `I18 bisection exploratory=0`
- `I18 bisection gap max=0.00018125027418136597`
- `I18 bisection drift min=0.24620485305786133`
- low-budget matched control rows：630

解读：

- I18 的确能产生更多 release audit rows，但全部不满足 role-safe movement，因为 logit drift 远高于 0.05。
- 低预算 bisection 没能把 drift 降进 gate；最小 drift 仍为 `0.24471133947372437`。
- 即使补齐 matched controls，最大 control gap 也只有 `0.00018125027418136597`，低于 exploration gate `0.002`。
- 因 I18 使用 CE-derived AdamW update 作为正交化参考，即使出现好结果也不能直接 promotion；本轮实际也没有出现 gate pass。

### 12.3 最终 route 更新

最终 route：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `minimum_success=Minimum Success F`
- `final_stop_allowed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `mandatory_exploration_levels_executed=1`
- `required_artifact_missing_count=0`
- `zip_sha256=9ca9c4ac8a282bda300649e0686a3cf8816989d450967a34d1c12eb9e94ff420`

最终 Line I 结论：

- role-aware gate、controls-win fallback、AdamW-orthogonal diagnostic、norm bisection、matched controls 都已执行。
- release exists，但安全 drift 和 control gap 不能同时成立。
- 当前 Line I blocker 收敛为：`release/control/safety triangle conflict`，不是单纯 projector-angle gate 错判。

补充后的完成判定：

v12.23 现在完成条件比上一节更充分：Line A Level 3 repair、T2 diagnostic、Line I controls-win fallback、I18 norm bisection、Line B shadow/cloned P3 audit、Line D smoke、Line R audit 均已实际执行并纳入最终 route/zip。最终仍是更强 no-go，不是 promotion。

## 13. 非 I18 release/no-safe drift 补齐后的最终复盘

### 13.1 为什么还要继续

上一节的完成判定仍不够严格：I18 已执行 norm bisection，但 Line I 中还有非 I18 actuator 出现 `release_audit_pass=1` 且 `role_safe_movement_pass=0`。按计划的 fail-closed 规则，这类 blocker 必须继续尝试 “role-specific movement gate + norm bisection”，不能只用 I18 diagnostic 代表全部 actuator。

因此本轮补齐：

- `I11-BranchGainRedistributionRoleSafe`
- `I13-QuadResidualLowDriftRotation`
- `I15-NoiseReleaseReservoirReleaseQP`
- `I17-ShadowPositiveReplayMatchedControls`

同时保留 I18 的 diagnostic bisection，且补齐同预算 matched controls。

### 13.2 本轮修改是否合理

修改文件：

- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`

修改内容：

- 新增 fallback manifest：
  - `non_i18_release_no_safe_due_drift -> I11_I13_I15_I17_norm_bisection_with_matched_controls`
- 将 drift bisection actuator 从仅 I18 扩展到 I11/I13/I15/I17/I18。
- 对低预算 `1e-05,2.5e-05,5e-05,0.0001,0.0002` 同步补 matched controls。

合理性判断：

- 这是计划中 “Actuator release but no safety” blocker 的直接执行。
- 低预算 actuator 如果没有同预算 controls，`control_gap` 会不完整；因此 matched controls 补齐是必要审计修复。
- I18 仍标记为 diagnostic-only，不允许 promotion，避免把 CE-derived AdamW reference 伪装成 loss-agnostic official result。

### 13.3 最终运行与总览

最终 run：

```text
run_id=official_explore_open2_continued_all_drift_bisect
```

最终 route：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `final_stop_allowed=1`
- `mandatory_exploration_levels_executed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `actuator_rows=2700`
- `actuator_release_audit_rows=256`
- `actuator_exploratory_release_dataset_seed_count=0`
- `line_d_executed_family_count=5`
- `required_artifact_missing_count=0`
- `code_review_packet_sha256=e8b158c3bc8395e03c6f183aabb17cccd870d4abb3ec211c89a1680248b90ecc`

审计包：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip
```

审计包内容核对：

- zip entries：44
- 包含 v12.23 runner：`experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
- 包含 v12.18 Line A runner：`experiments/run_v1218_b320_label_free_ablation.py`
- 包含核心 primitive：`dgkan/models/fc_purekan_primitives.py`
- 包含 fused kernel：`dgkan/kernels/fused_hinge_quadratic.py`
- 包含 v12.23 计划文档：`docs/DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md`

### 13.4 Fallback 与 artifact 审计

`v1223_fallback_execution_manifest.csv`：

- fallback rows：10
- all executed：true

已执行 triggers：

- `scout_fail_or_incomplete`
- `hardening_task_fail`
- `T1B_auc_low_precision`
- `T1A_T1B_exploration_fail`
- `actuator_release_safety_conflict`
- `actuator_release_movement_but_controls_win`
- `I18_logit_drift_exceeds_gate`
- `non_i18_release_no_safe_due_drift`
- `P3_gate_fail`
- `classic_not_executed`

`v1223_required_artifact_manifest.csv`：

- required rows：47
- missing count：0

### 13.5 单 actuator 结论

`I11-BranchGainRedistributionRoleSafe`：

- 低范数 bisection rows：90
- bisection safe：90
- bisection release：28
- bisection exploratory：0
- bisection drift min：`3.8623809814453125e-05`
- bisection gap max：`0.0001363307237625122`
- 全预算 gap max：`0.0016209781169891357`

解读：

- I11 能做到 safe+release，但 control gap 仍低于 exploratory gate `0.002`。
- 低范数修复没有打开 exploration；最接近的全预算 gap 也只有 `0.0016209781169891357`。

`I13-QuadResidualLowDriftRotation`：

- 低范数 bisection rows：90
- bisection safe：0
- bisection release：26
- bisection exploratory：0
- bisection drift min：`0.24181252717971802`
- 全预算 gap max：`0.04676394909620285`
- drift <= 0.05 rows：0

解读：

- I13 的 release/large-gap 主要伴随过大 drift。
- 虽然全预算 gap max 很高，但完全不满足 role-safe movement，不能打开 promotion 或 exploration。

`I15-NoiseReleaseReservoirReleaseQP`：

- 低范数 bisection rows：90
- bisection safe：0
- bisection release：26
- bisection exploratory：0
- bisection drift min：`0.24253308773040771`
- 全预算 gap max：`0.0004449784755706787`
- drift <= 0.05 rows：0

解读：

- I15 既不能进入 drift gate，control gap 也低于 exploration gate。
- 该方向不是可修复的低范数安全释放候选。

`I17-ShadowPositiveReplayMatchedControls`：

- 低范数 bisection rows：90
- bisection safe：0
- bisection release：26
- bisection exploratory：0
- bisection drift min：`0.2449427843093872`
- 全预算 gap max：`0.019115328788757324`
- drift <= 0.05 rows：0

解读：

- I17 能制造较大 control gap，但 drift 完全越界。
- 这说明 shadow-positive replay 方向仍是 unsafe release，不是可部署 actuator。

`I18-AdamWOrthogonalResidual-diagnostic`：

- 低范数 bisection rows：90
- bisection safe：0
- bisection release：26
- bisection exploratory：0
- bisection drift min：`0.24620485305786133`
- 全预算 gap max：`0.00018125027418136597`
- drift <= 0.05 rows：0

解读：

- I18 的 AdamW-orthogonal diagnostic 没有改善 drift。
- 因其使用 CE-derived AdamW reference，本来就不能 official promotion；本轮数据也没有产生 exploration gate。

### 13.6 最终结论

v12.23 现在完成。

完成不是因为出现了 promotion，而是因为计划要求的 fail-closed fallback 已经全部实际执行，且新增审计未发现未执行的触发项：

- Line A：scout、top-2 hardening、Level 3 repair/no-go 已执行。
- Line T：T1A/T1B、component/sign/support calibration、T2 upper-bound diagnostic 已执行。
- Line I：role-aware gate、matched controls、AdamW-orthogonal diagnostic、I18 bisection、非 I18 bisection 全部执行。
- Line B：shadow P3 与 cloned P3 audit 已执行，P4 未打开。
- Line D：active families smoke 已执行。
- Line R：dirty tree/diff isolation/artifact/zip 审计已执行。

最终 no-go 边界更清楚：

- loss-agnostic / label-free Line A 没有找到可 promotion 候选。
- target visibility 的 upper-bound diagnostic 有信号，但 `promotion_allowed=0`。
- actuator 有 release，但无法同时满足 safety drift、matched-control gap、loss-agnostic provenance 三个条件。
- P3/P4 没有打开。

因此最终状态是：

```text
R4-FunctionalMechanismNoGoAfterAllFallbacks
```

这是一个完整执行后的 no-go，不是未完成停止。

## 14. 全线 continuation 复盘：每条线均追加推进

### 14.1 本轮为什么继续

用户指出“每个线都要推进”。因此本轮不只补 Line I，而是对 Line A/T/C/I/B/D/R 全部追加动作，并重新生成最终 route、fallback manifest、required artifact manifest 和 code review zip。

### 14.2 代码修复与新增动作

本轮修改：

- Line A：
  - 修复 `A51-A54` 被误标为 diagnostic-only 的问题。
  - 新增 `A55-ResidualA1LowRankFrame-r001-fixedP`。
  - 新增 `A56-A1FixedPBranchLowQuad010`。
- Line T/C：
  - 新增 `v1223_t3_visibility_support_stress.csv`，对 T2 upper-bound signal 做 dataset/seed support stress。
- Line I：
  - 新增 `BranchOnlyRandomControl`、`DirectOnlyRandomControl`、`GainOnlyRandomControl`。
- Line B：
  - 新增 `v1223_functional_p3_control_audit.csv`。
  - P3 gate 增加 source role-safe/release/exploratory/provenance 显式约束。
- Line D：
  - 从单 dataset/seed smoke 扩展为遍历 `--line-d-datasets` 与 `--line-d-seeds`。
- Line R：
  - required artifacts、fallback manifest、zip 都纳入新增 T3/P3 audit 产物。

合理性判断：

- `A51-A54` 是 label-free repair，不使用 label/CE vector，把它们列为 diagnostic 是审计错误；修正后重新跑 route 是必要的。
- A55/A56 是沿着上一轮 A51/A54 的失败模式继续降低 residual/quad branch 强度，没有降低 promotion gate。
- T3 使用完成后的 response/LineC observable，因此只能 audit-only，不能 promotion。
- Branch/direct/gain controls 是为了回答 Line I 的 role-control ambiguity，不能把 actuator 和 matched role controls 混为一谈。
- P3 source 必须来自安全、可释放、非 diagnostic、非 label/CE 的 actuator，否则不能转换为 cloned P3。

### 14.3 最终全线 run

最终 run：

```text
run_id=official_explore_open2_all_lines_continuation
```

最终 route：

- `route=R4-FunctionalMechanismNoGoAfterAllFallbacks`
- `final_stop_allowed=1`
- `mandatory_exploration_levels_executed=1`
- `official_success_reached=0`
- `hard_budget_exhausted=1`
- `required_artifact_missing_count=0`
- `code_review_packet_sha256=c8d45a8dbf9ace12e32418d7905f9db75680deba1561dacab23156c61b5bd1c7`

Fallback/artifact 审计：

- fallback rows：13
- fallback all executed：true
- required rows：49
- missing artifacts：`[]`
- zip entries：46
- zip 包含新增 T3 与 P3 control audit artifact。

### 14.4 Line A 结果

新增 hardening 数据：

- 三数据集：MNIST, Fashion-MNIST, KMNIST
- seeds：0,1,2
- train size：1024
- val/test size：512
- epochs：8
- LineC measured：1

汇总后：

- `linea_rows=331`
- `linea_hardening_rows=198`
- `label_free_candidate_rows=18`
- `label_free_official_pass_count=0`

关键候选：

- `A1-noYForStats`
  - mean delta vs A0：`-0.0029658564814814816`
  - worst delta vs A0：`-0.029296875`
  - LineC pass rate：`0.0`
  - diagnostic：`0`
- `A51-StagedUnlabeledAdapt-warm1-r002`
  - mean delta vs A0：`-0.002387152777777778`
  - worst delta vs A0：`-0.0234375`
  - LineC pass rate：`0.0`
  - diagnostic：`0`
- `A55-ResidualA1LowRankFrame-r001-fixedP`
  - mean delta vs A0：`-0.09288194444444445`
  - worst delta vs A0：`-0.15625`
  - LineC pass rate：`0.0`
  - diagnostic：`0`
- `A56-A1FixedPBranchLowQuad010`
  - mean delta vs A0：`-0.10221354166666667`
  - worst delta vs A0：`-0.150390625`
  - LineC pass rate：`0.0`
  - diagnostic：`0`

Line A 结论：

- 修复 diagnostic 标记后，A51 确认为合法 label-free repair candidate，但仍未过 official/exploration gate。
- A55/A56 的更低 residual/quad branch 没有修复 task/LineC，反而明显恶化。
- 当前最接近的仍是 A51/A1；失败边界更明确：继续降低 residual 或 quad branch 不是有效方向。

### 14.5 Line T/C 结果

新增 T3 support stress：

- rows：6
- stable rows：6
- AUC min：`0.7436998854524628`
- AUC max：`0.8489583333333334`

逐组结果：

- Fashion-MNIST：AUC `0.8104619565217391`, precision `0.375`, recall `0.375`
- KMNIST：AUC `0.8489583333333334`, precision `0.5`, recall `0.5`
- MNIST：AUC `0.843`, precision `0.5333333333333333`, recall `0.5333333333333333`
- seed 0：AUC `0.7436998854524628`, precision `0.3888888888888889`, recall `0.3888888888888889`
- seed 1：AUC `0.8348178137651822`, precision `0.38461538461538464`, recall `0.38461538461538464`
- seed 2：AUC `0.8003472222222222`, precision `0.3333333333333333`, recall `0.3333333333333333`

Line T/C 结论：

- T2/T3 的 response-level diagnostic 信号并不弱，且 dataset/seed support stress 全部 stable。
- 但 T2/T3 使用 completed response/LineC observables，不是 precommit-safe direction source，因此 `promotion_allowed=0`。
- 这把 blocker 从“没有可见信号”推进为“有 upper-bound 可见信号，但尚未找到 loss-agnostic/precommit-safe 生成方式”。

### 14.6 Line I 结果

新增 matched controls：

- `BranchOnlyRandomControl` rows：198
- `DirectOnlyRandomControl` rows：198
- `GainOnlyRandomControl` rows：198

总 actuator：

- `actuator_rows=3294`
- non-control exploratory actuator rows：0
- `actuator_exploratory_release_dataset_seed_count=0`

Line I 结论：

- 增加 role-specific controls 后，仍没有任何 non-control actuator 进入 exploratory gate。
- 最接近的安全释放仍是 I11，但 control gap 不够。
- unsafe high-gap 方向仍无法越过 drift gate。

### 14.7 Line B 结果

新增 P3 control audit：

- `functional_p3_rows=6`
- `functional_p3_control_audit_rows=30`
- `functional_p3_pass_count=0`
- `p4_open=0`

P3 audit top row：

- actuator：`I11-BranchGainRedistributionRoleSafe`
- dataset：MNIST
- seed：1
- budget：`0.005`
- role safe：1
- release：1
- exploratory：0
- control gap：`0.0016209781169891357`
- drift：`0.019960403442382812`
- NoiseSignalLeak delta：`-0.26113710552453995`
- Reservoir delta：`-0.11230021715164185`

Line B 结论：

- I11 top row 是真实 safe+release，但 control gap 低于 exploration gate `0.002`，更低于 P3 gate `0.005`。
- 因 source actuator 没有 exploratory gate，P3 不能打开，P4 也不能打开。
- 当前 Line B blocker 是 source actuator control-gap insufficiency，不是 P3 文件缺失。

### 14.8 Line D 结果

本轮 Line D 从 5 行扩展到 10 行：

- datasets：MNIST, Fashion-MNIST
- seed：0
- families：Rational, Chebyshev, Wavelet, RBF/FastKAN, Fourier
- kernel blocked：0

关键结果：

- Rational MNIST：val_acc `0.640625`, LineC CouplingR2 `0.1688038217034734`
- Rational Fashion-MNIST：val_acc `0.5625`, LineC CouplingR2 `0.20384892975350133`
- Fourier MNIST：val_acc `0.296875`, LineC CouplingR2 `0.09257388126333477`
- Fourier Fashion-MNIST：val_acc `0.40625`, LineC CouplingR2 `0.088131683128016`
- Chebyshev/Wavelet/RBF 在两个数据集上 task 较弱，但没有 kernel blocker。

Line D 结论：

- Classic no-BSpline portfolio 已真实执行，不再是 deferred。
- Rational 在 smoke 里明显强于其他 family，但这只是 smoke，不是 official promotion。
- 下一步如果继续 Line D，应围绕 Rational/Fourier 做 task+LineC hardening，而不是修 kernel。

### 14.9 最终判定

v12.23 在“每条线都推进”的口径下，现在完成。

完成原因不是出现成功，而是：

- Line A 有新增候选与修复后重算；
- Line T/C 有 T3 support stress；
- Line I 有 role-specific matched controls；
- Line B 有 P3 source control audit；
- Line D 有多数据集 family smoke；
- Line R 有 required artifact、fallback manifest、zip 全部更新；
- fallback manifest 13 行全部 executed；
- required artifact 49 行缺失 0。

最终仍是：

```text
R4-FunctionalMechanismNoGoAfterAllFallbacks
```

这是更强的 no-go：T2/T3 证明存在 response-level upper-bound signal，但 Line A 无法构造 label-free promotable base，Line I/B 无法把 safe release 转成足够 control gap 的 functional actuator，Line D 只打开了 Rational/Fourier 的后续 hardening 方向，没有本轮 promotion。

## 15. 2026-05-25 继续推进后的最终复盘：未达标，但完成 v12.23 预注册探索队列

### 15.1 本轮继续推进回答的问题

用户追问“达成目标了吗，没有请继续”。因此本轮没有把 section 14 的 no-go 当作终点，而是继续执行了三类工作：

1. 继续 Line A/C：从 A57-A59 到 A60-A65，围绕 A51 的 task 已恢复但 LineC non-tearing 为 0 的 blocker 做机制修复。
2. 继续 Line I/B：审计 actuator matched-control gap，发现并修复 actuator LineC delta 的 sketch seed 口径问题。
3. 重新跑官方 v12.23 runner，生成最终 route、required artifact manifest、fallback manifest、P3/P4、Line D、zip。

最终仍未达成 S1/S2/S3/S4/S5 任一路径：

```text
run_id = official_explore_open2_rmsq_seedfix_final
route = R4-FunctionalMechanismNoGoAfterAllFallbacks
official_success_reached = 0
fail_reason = all mandatory exploration levels exhausted without promotion
```

这不是成功结果，不能写成成功；它是一个更干净的机制 no-go。

### 15.2 关键代码修复：actuator LineC delta 的 sketch seed 污染

本轮最重要的代码审计发现不是新候选，而是 Line I 的测量口径：

```text
旧逻辑：
  base_metrics 用 seed + 7300
  trial metrics 用 seed + 8400 + budget

后果：
  即使 NoOpMatchedOverhead 没有移动模型，也会因为 LineC sketch seed 不同而产生非零 Noise/Reservoir delta。
```

这会污染：

- `release_audit_pass`
- `matched_control_gap`
- P3 source actuator selection

修复：

```text
每个 dataset/seed/budget 用同一个 metric_seed 计算 base 和 trial metrics。
base metrics 按 metric_seed 缓存。
```

修复后审计：

```text
NoOp max abs delta = 0.0
actuator_release_audit_rows: 309 -> 48
actuator_exploratory_release_dataset_seed_count = 0
```

解释：

- 修复前 I11 看起来“safe+release 且 control gap 只差一点”，但这一判断受 sketch-seed 噪声污染。
- 修复后 I11/I12/I14 等 role-safe actuator 不再通过 release；高 release/high gap 的 I13/I15/I17/I18 仍然 drift 太大或 role-safe 为 0。
- 因此 Line I 的真实 blocker 变为：安全 actuator release 不足，高 release actuator 不安全。

### 15.3 Line A/C：A57-A65 的真实结果

#### A57-A59

目标：

- A57：A51 + `boundq`
- A58：A51 + lower `signalBlock010`
- A59：lower `signalBroad025 + signalBlock010 + boundq`

三数据集 e12 结果：

| dataset | best candidate | best mean delta vs A0 |
|---|---|---:|
| MNIST | A51 | 0.006184895833333333 |
| Fashion-MNIST | A59 | 0.010416666666666666 |
| KMNIST | A51 | 0.0048828125 |

官方汇总后：

```text
label_free_best_hardening_candidate = A51-StagedUnlabeledAdapt-warm1-r002
label_free_best_hardening_mean_delta_vs_A0 = 0.003146701388888889
label_free_best_linec_pass_rate = 0.0
```

解释：

- A59 在 Fashion-MNIST 子任务上有 task delta 改善，但跨三数据集汇总不是 best。
- A57/A59 的 boundq/signal broad-block 调整没有打开 LineC non-tearing。
- A58 保留 A51 task 表现，但 LineC 仍 0。

#### A60-A62

目标：

- A60：A51 + `directreadinit125`
- A61：A51 + `identityamp150`
- A62：A58 + `directreadinit125`

结果：

| dataset | best candidate | best mean delta vs A0 |
|---|---|---:|
| MNIST | A51 | 0.006184895833333333 |
| Fashion-MNIST | A51 | -0.0016276041666666667 |
| KMNIST | A51 | 0.0048828125 |

关键结论：

```text
A60/A61/A62 全部 LineC pass rate = 0.0
```

解释：

- 单纯增强 direct/readout 并没有改变 LineC 的 reservoir tearing。
- A60 能维持 A51 的 task/AUC，但不能转化为合法 exploration path。

#### A63-A65

目标：

- A63：A51 + `rmsq`
- A64：A51 + `rmsq + boundq`
- A65：A58 + `rmsq`

结果：

| dataset | best candidate | best mean delta vs A0 |
|---|---|---:|
| MNIST | A51 | 0.006184895833333333 |
| Fashion-MNIST | A63 | 0.0022786458333333335 |
| KMNIST | A51 | 0.004557291666666667 |

A63/A65 的具体表现：

- Fashion-MNIST 上 A63/A65 将 worst delta 改到 `-0.0029296875`，AUC-time 约 `0.84`，这是 task/AUC 上的真实改善。
- 但 A63/A65 的 LineC pass rate 仍是 `0.0`。
- A64 明显伤 task，不是有效方向。

最终 Line A/C 结论：

```text
best hardening candidate = A51
best mean delta vs A0 = 0.003146701388888889
best LineC pass rate = 0.0
label_free official pass count = 0
label_free exploration pass count = 0
```

这说明当前 blocker 已从“label-free task 完全恢复不了”变为：

```text
label-free task 可以接近/超过 A0，但 LineC non-tearing 几何无法同时满足。
```

### 15.4 Line T/C：visibility 仍是 diagnostic upper bound，不是可部署 source

最终 official runner：

```text
T1A_auc_joint = 0.41019417475728154
T1B_auc_joint = 0.5032062807143713
T2_clone_probe_auc_joint = 0.7383255633255633
T2 precision/recall = 0.3181818181818182 / 0.3181818181818182
T3_visibility_support_auc_min = 0.6660482374768089
T3_visibility_support_stable_rows = 6
```

解释：

- T1A/T1B 仍不过 official/exploration。
- T2/T3 继续证明 completed-response 层存在可见信号，但它们 `promotion_allowed=0`，不能作为 direction source。
- 本轮没有把 diagnostic upper bound 冒充为 functional success。

### 15.5 Line I/B：seed-fix 后真实 blocker 更严格

最终 actuator 审计：

| actuator | safe rows | release rows | safe+release rows | exploratory rows | best gap | best drift |
|---|---:|---:|---:|---:|---:|---:|
| I11 | 204 | 0 | 0 | 0 | -0.016013488173484802 | 0.031005144119262695 |
| I12 | 126 | 0 | 0 | 0 | -0.018504545092582703 | 0.004106640815734863 |
| I13 | 0 | 13 | 0 | 0 | 0.010798055678606033 | 1.3926801681518555 |
| I14 | 126 | 0 | 0 | 0 | -0.0175180584192276 | 0.017914772033691406 |
| I15 | 0 | 10 | 0 | 0 | 0.033934324979782104 | 0.3914039134979248 |
| I17 | 0 | 13 | 0 | 0 | 0.051491498947143555 | 0.7304795980453491 |

结论：

- role-safe actuator 没有真实 release。
- 有 release/high gap 的 actuator drift 远超 `0.05`，role-safe 为 0。
- `actuator_exploratory_release_dataset_seed_count=0`
- `functional_p3_pass_count=0`
- `p4_open=0`

P3 source 选择了 high gap 的 I17：

```text
source_actuator = I17-ShadowPositiveReplayMatchedControls
source_role_safe_movement_pass = 0
source_release_audit_pass = 1
source_exploratory_release_gate = 0
control_gap = 0.051491498947143555
```

P3 正确 fail-closed，没有把 unsafe source 放行。

### 15.6 Line D：已推进，但不是本轮成功路径

最终 Line D：

```text
line_d_family_rows = 10
line_d_kernel_blocked_count = 0
datasets = MNIST, Fashion-MNIST
families = Rational, Chebyshev, Wavelet, RBF/FastKAN, Fourier
```

关键 smoke：

- Rational MNIST：val_acc `0.6875`, LineC CouplingR2 `0.1440191673907426`
- Rational Fashion-MNIST：val_acc `0.625`, LineC CouplingR2 `0.17414573268188527`
- Fourier MNIST：val_acc `0.4296875`, LineC CouplingR2 `0.09100964757886698`
- Fourier Fashion-MNIST：val_acc `0.5078125`, LineC CouplingR2 `0.13100685968572967`

结论：

- Line D 不再是未执行。
- Rational/Fourier 有继续 hardening 价值。
- 但 Line D 目前只是 smoke，不是 S1-S5 任一路径的 promotion。

### 15.7 最终 artifact 与 zip

最终 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip
```

审计：

```text
exists = True
size = 1224305
sha256 = 18fa7c952f3e4143b7d35bd9ef439039e38ba317db11693bbea97bfcb4410ff4
entries = 46
```

包含关键代码：

- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
- `experiments/run_v1218_b320_label_free_ablation.py`
- `dgkan/models/fc_purekan_primitives.py`
- `dgkan/kernels/fused_hinge_quadratic.py`

包含关键结果：

- `v1223_route_decision.json`
- `v1223_label_free_hardening.csv`
- `v1223_label_free_linec.csv`
- `v1223_linec_deployable_targets.csv`
- `v1223_actuator_safety_roleaware.csv`
- `v1223_functional_p3_control_audit.csv`
- `v1223_classic_family_status.csv`

### 15.8 最终结论与下一步 blocker

本轮 v12.23 计划执行到当前我能明确推进的边界。没有成功路径，不能写成成功。

最终判定：

```text
R4-FunctionalMechanismNoGoAfterAllFallbacks
```

但这个 no-go 比之前更可信，因为：

1. Line A 从 A51 扩到 A65，确认 task 可恢复但 LineC non-tearing 始终 0。
2. Line I 修复了 seed 口径污染后，确认 safe actuator 没有真实 release。
3. Line B/P3 没有拿 unsafe I17 作为合法 source。
4. Line D 已真实执行 10 行 family smoke。
5. required artifacts 缺失为 0，zip 已包含新代码和新结果。

下一步不是继续微调 A51 的 readout 或 RMSQ，因为这两类已经验证无效。更合理的下一轮应重新定义一个 precommit-safe、label-free 的 signal/reservoir 几何 source，或者把 Line D 的 Rational/Fourier smoke 升级为正式 hardening 线；如果继续 Line I，需要设计 drift-safe release actuator，而不是继续扩大 I17/I15 这类高 drift 方向。

## 16. 用户要求继续后的补充复盘：Line I dense scan + Line D hardening

### 16.1 是否达成 v12.23 目标

没有达成 official functional success，也没有打开 S1-S5 任一路线。

最终 official rerun：

```text
run_id = official_explore_open2_lined_hardening_final
route = R4-FunctionalMechanismNoGoAfterAllFallbacks
fail_reason = all mandatory exploration levels exhausted without promotion
official_success_reached = 0
final_stop_allowed = 1
hard_budget_exhausted = 1
mandatory_exploration_levels_executed = 1
required_artifact_missing_count = 0
```

这次继续推进后的关键区别是：前一版“Line D 只是 smoke”的不足被补了一层 Rational/Fourier hardening；Line I 也补了 dense drift-threshold scan。结果仍然 no-go，但不是因为没有继续执行，而是因为各 gate 的真实证据仍不闭合。

### 16.2 本节做了哪些代码修改

为了让后续复现和审计容易，本节新增了两个脚本，并把它们加入 v12.23 official zip：

- `experiments/run_v1223_line_d_hardening.py`
  - 用于按 family/dataset/seed/GPU 独立执行 Line D targeted hardening。
  - 复用官方 `classic_family_smoke_one()`，不重新发明训练/LineC 逻辑。
  - 输出顶层 `v1223_line_d_hardening_*` artifacts，确保官方 zip 可收集。
- `experiments/summarize_v1223_line_d_hardening.py`
  - 用于聚合四个 hardening shard。
  - 产出 `v1223_line_d_hardening_aggregate.csv`、summary CSV 和 summary JSON。
- `experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py`
  - 只修改 `package_zip()` 的固定代码清单，把上述两个脚本纳入 code-review packet。

语法检查全部通过：

```text
py_compile pass
```

审计风险：这次没有改 gate 阈值，没有降低 promotion 条件；只增加可复现 driver、汇总脚本和 zip 纳入规则。

### 16.3 Line I dense scan 结论

dense scan 使用 budgets：

```text
0.0003,0.0005,0.0008,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.007,0.008,0.009,0.01
```

最终 official actuator 汇总：

```text
actuator_rows = 6534
actuator_role_safe_movement_rows = 1228
actuator_release_audit_rows = 120
actuator_exploratory_release_dataset_seed_count = 0
actuator_official_gate_rows = 0
functional_p3_pass_count = 0
p4_open = 0
```

逐 actuator 审计：

| actuator | safe_rows | release_rows | safe+release | exploratory | best_control_gap | min_drift |
|---|---:|---:|---:|---:|---:|---:|
| I11 | 366 | 0 | 0 | 0 | -0.016013488173484802 | 0.000038623809814453125 |
| I12 | 288 | 0 | 0 | 0 | -0.018504545092582703 | 0.00008755922317504883 |
| I13 | 0 | 32 | 0 | 0 | 0.010798055678606033 | 0.23476892709732056 |
| I14 | 288 | 0 | 0 | 0 | -0.0175180584192276 | 0.0006197690963745117 |
| I15 | 0 | 28 | 0 | 0 | 0.033934324979782104 | 0.22774165868759155 |
| I17 | 0 | 30 | 0 | 0 | 0.051491498947143555 | 0.2368667721748352 |
| I18 diagnostic | 0 | 30 | 0 | 0 | 0.014657139778137207 | 0.24330437183380127 |

解释：

- I11/I12/I14 是 drift-safe / role-safe，但 release rows 为 0，control gap 也为负。
- I13/I15/I17/I18 有 release 或正 gap，但 min drift 已远超 `0.05`，所以 role-safe 为 0。
- dense threshold scan 没有发现 “drift-safe release” 的交叉区域。
- NoOp seed-fix 审计：`noop_rows=378`，`max_abs_metric_delta_sum=0.0`。这说明 seed 口径污染修复后，NoOp 不再产生假 delta。

因此 Line I 的 blocker 不是预算格点太粗，而是当前 actuator family 的安全方向和 release 方向仍然分离。

### 16.4 Line D hardening 结论

本节补了文档要求的 Rational/Fourier hardening，而不是只停在 smoke。

执行矩阵：

```text
Rational x MNIST x seeds 0,1,2 on cuda:0
Rational x Fashion-MNIST x seeds 0,1,2 on cuda:1
Fourier x MNIST x seeds 0,1,2 on cuda:2
Fourier x Fashion-MNIST x seeds 0,1,2 on cuda:3
train/val = 1024/512
epochs = 8
LineC batch/sketch = 64/24
```

聚合结果：

| family | dataset | rows | mean_val_acc | min_val_acc | max_val_acc | mean_NLL | mean_linec_CouplingR2 | mean_linec_NoiseSignalLeak | mean_linec_RealSignalReservoirRatio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fourier | Fashion-MNIST | 3 | 0.5846354166666666 | 0.5546875 | 0.609375 | 1.3872931798299153 | 0.12844373058968392 | 0.0521984708805879 | 0.6052201688289642 |
| Fourier | MNIST | 3 | 0.6647135416666666 | 0.650390625 | 0.68359375 | 1.6000018914540608 | 0.12411807126846226 | 0.06800038740038872 | 0.6920690933863322 |
| Rational | Fashion-MNIST | 3 | 0.6588541666666666 | 0.630859375 | 0.67578125 | 1.3111967245737712 | 0.10925706826505344 | 0.05980806487301985 | 0.4520403246084849 |
| Rational | MNIST | 3 | 0.6744791666666666 | 0.6484375 | 0.701171875 | 1.7119425932566326 | 0.13741879786845176 | 0.07816234479347865 | 0.5534500678380331 |

最佳单行：

- best val_acc：Rational / MNIST / seed 0，`val_acc=0.701171875`，`linec_CouplingR2=0.13238496612899686`
- best LineC CouplingR2：Fourier / Fashion-MNIST / seed 2，`linec_CouplingR2=0.17223205548419696`，`val_acc=0.609375`

解释：

- Line D 不再是预算饥饿或未执行；本节 12/12 rows 都是 `FamilyNearPass`，kernel_blocked_rows 为 0。
- Rational 在 task 上更强，尤其 MNIST/Fashion-MNIST 的 mean val_acc 比 Fourier 更好。
- Fourier 的 best single-row LineC CouplingR2 更高，但 task 不占优。
- v12.23 的 S1-S5 promotion 仍不允许由 Line D hardening 直接打开；这些结果只能说明 Rational/Fourier 值得作为下一轮正式 classic-family hardening 线，而不是本轮 functional success。

### 16.5 S1-S5 gate 状态

| route | 状态 | 证据 |
|---|---|---|
| S1 LabelFreeNearRecovered | 未打开 | `label_free_official_pass_count=0`，`label_free_exploration_pass_count=0`，best LineC pass rate `0.0` |
| S2 ValueSourceExploratoryVisible | 未打开 | T1A `0.41019417475728154`，T1B `0.5032062807143713`；T2/T3 仍是 diagnostic/support evidence，promotion_allowed `0` |
| S3 RoleSafeActuatorSurvivor | 未打开 | `actuator_exploratory_release_dataset_seed_count=0`；safe 与 release 没有交集 |
| S4 FunctionalP3Opened | 未打开 | `functional_p3_pass_count=0` |
| S5 OfficialFunctionalSuccess | 未打开 | `p4_open=0`，`official_success_reached=0` |

最终判断：

```text
R4-FunctionalMechanismNoGoAfterAllFallbacks
```

### 16.6 最终 zip 与审计包

最终 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_code_review_packet.zip
```

审计结果：

```text
exists = true
size = 1857914
entries = 63
sha256 = 6a65395e8cb2cd851de80821aaac395db48dc80cf82e574d5805ea236f7cd424
```

确认新代码和 hardening aggregate 已入包：

- `experiments/run_v1223_line_d_hardening.py`
- `experiments/summarize_v1223_line_d_hardening.py`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate_summary.csv`
- `results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2/v1223_line_d_hardening_aggregate_summary.json`

### 16.7 本轮补充推进后的结论

这次继续推进没有把 no-go 改成 success，但它补掉了两个之前还可能被质疑的空档：

1. Line I 不是因为预算格点没扫到才失败；dense scan 后仍没有 drift-safe release。
2. Line D 不是因为没执行才不能判断；Rational/Fourier hardening 已完成 12 行有效结果，显示有继续研究价值，但没有打开 v12.23 functional promotion path。

下一轮如果继续 v12.23 同一范式，优先方向不应再是简单扩大 I15/I17 的高 drift release，也不应继续只微调 A51/A63 的 readout/RMSQ。更合理的下一步是两条之一：

- 将 Rational/Fourier 升为正式 classic-family hardening protocol，明确 task、LineC、efficiency 三重 gate；
- 重新设计 drift-safe release actuator，使 movement/release/control-gap 三者在同一预算区间有交集。


## 17. 继续推进复盘：从 R4 推进到 S4，但 P4 短训未通过

本节复盘对应执行日志第 13 节。所有数字均来自实际产物，不手填虚构数据。

### 17.1 本轮继续推进前的状态

上一轮 official full run 的最终状态：

```text
route = R4-FunctionalMechanismNoGoAfterAllFallbacks
official_success_reached = 0
required_artifact_missing_count = 0
```

主要 blocker 是 Line I 的 safe movement 与 release/control-gap 仍不闭合；Line D hardening 已完成，但不打开 v12.23 functional promotion path。

### 17.2 I19-I25 的实验判断

| line | 结果 | 判断 |
|---|---|---|
| I19-I21 | drift-safe，但 release/control-gap 不足 | 不能 survivor |
| I22/I23 | release 较强，但 mean compensation 后 drift 仍 unsafe | 不能 survivor |
| I24/I25 | direct_readout logit compensation 能把 drift 压入安全区，并出现 survivor | 值得跨 seed/dataset 验证 |

I24/I25 的方向记录为：

```text
uses_label = 0
uses_ce_vector = 0
direct_logit_compensation_applied = 1
```

审计风险：它使用 unlabeled query features / base logits 做 direct_readout compensation，因此只能说 label-free / loss-agnostic，不能说完全 precommit-independent。

### 17.3 control scope 修复的重要性

分片 KMNIST seed0/1 曾打开 S4，但合并 targeted official 初始只到 S3。审计发现原因是继承自 v12.21 的 matched_control_gap 在全 out-dir 比较 control。

修复后 scope 为：

```text
dataset / seed / budget / signed_direction
```

修复后合并 route：

```text
route = S4-FunctionalP3Opened
minimum_success = Minimum Success D
functional_p3_pass_count = 6
p4_open = 1
actuator_exploratory_release_dataset_seed_count = 6
actuator_official_gate_rows = 13
actuator_official_gate_pass = 1
required_artifact_missing_count = 0
```

该修复没有降低 gate；它修正的是 matched control 的比较范围。

### 17.4 P3 source 的真实含义

最终 P3 source：

```text
I24-DirectLogitCompensatedQuadRelease / KMNIST / seed 0 / budget 0.009 / sign +1
```

关键指标：

```text
CouplingR2_delta = 0.05668966566648037
NoiseSignalLeak_delta = -0.02430429309606552
Reservoir_delta = -0.06281167268753052
control_gap = 0.04396222531795502
CEp99_delta = -0.07897424697875977
ECE_delta = -0.031210273504257202
role_safe_movement_pass = 1
release_audit_pass = 1
exploratory_release_gate = 1
```

这说明 v12.23 不再是完全找不到 role-safe actuator survivor。现在至少存在一个可审计的 cloned P3 source。

### 17.5 P4 short-run 结果

按文档要求，P3 打开后继续执行 short-run P4。结果没有通过。

1-epoch：

```text
source_final_acc = 0.609375
noop_final_acc = 0.5859375
control_final_acc = 0.625
source_vs_noop_acc_delta = +0.0234375
source_vs_control_acc_delta = -0.015625
p4_pass = 0
```

3-epoch：

```text
source_final_acc = 0.703125
noop_final_acc = 0.7421875
control_final_acc = 0.703125
source_vs_noop_acc_delta = -0.0390625
source_vs_control_acc_delta = 0.0
p4_pass = 0
```

解释：P3 source 有 functional audit 信号，但短训后不具备 control-resistant task advantage。1 epoch 时 source 比 NoOp 好但输 control；3 epoch 时 source 不输 control，但明显输 NoOp。因此不能写成 P4 success，更不能写成 S5 official success。

### 17.6 当前最终状态

最终 targeted out-dir：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted
```

最终 route：

```text
route = S4-FunctionalP3Opened
minimum_success = Minimum Success D
fail_reason = functional P3 gate opened
official_success_reached = 0
final_stop_allowed = 0
functional_p3_pass_count = 6
p4_open = 1
p4_executed = 1
p4_pass = 0
```

最终 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
sha256 = f9f0478bd2392f0f1bdc778820e1d1f7813c7757735d7a16eea09d42963ef3b0
entries = 58
```

### 17.7 结论

本轮不是 official success，但确实推进了 v12.23：

```text
R4 -> S4
```

成立的结论：

```text
1. Direct-logit compensated I24/I25 可以制造 drift-safe release survivor。
2. local matched-control scope 修复后，合并 9 dataset-seed targeted run 打开 Functional P3。
3. P4 short-run 已执行，且明确未通过。
```

不能成立的结论：

```text
1. 不能说 S5 official functional success。
2. 不能说 P4 pass。
3. 不能说 I24/I25 已经具备 short-run task control advantage。
```

下一步如果继续 v12.23 同线，重点不应再是单纯增加 I24 budget，而应解决 P4 中 source 输给 NoOp/control 的问题：要么把 P3 source 从 query-logit compensation 改成 precommit-safe 的 low-drift compensation，要么设计 P4 source/control 共用同一 compensation substrate，避免 source 的 advantage 被 compensation 本身的 matched control 吃掉。


---

# 18. 继续推进复盘：P3 与 P4 解耦的直接证据

## 18.1 本轮追加是否达成 v12.23 目标？

没有达成 S5 official functional success。

最终两个关键 out-dir 都停在：

```text
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_pass = 0
required_artifact_missing_count = 0
```

但是，这次追加不是原地重复。它把失败边界从“P3 打开但 P4 没过”推进到了更具体的机制结论：

```text
P3 几何 gate 与 P4 short-run task advantage 在当前 I24/I25 direct-logit compensation family 中发生解耦。
```

## 18.2 原 P3 source 的 P4 失败不是 compensation substrate 单点问题

原 official targeted P3 source：

```text
KMNIST seed0 / I24-DirectLogitCompensatedQuadRelease / norm0.009 / sign+1
```

原 P4 short-run 已经失败：

```text
1 epoch: source=0.609375, noop=0.5859375, control=0.625, p4_pass=0
3 epoch: source=0.703125, noop=0.7421875, control=0.703125, p4_pass=0
```

本轮进一步测试 5 种 compensation/reference 模式：

```text
per_variant_query
shared_source_query
per_variant_train
shared_source_train
none
```

3 epoch 全部失败，且 source 在所有模式下都输给 noop 或 control：

```text
per_variant_query source-control = -0.0390625
shared_source_query source-control = -0.046875
per_variant_train source-control = -0.0546875
shared_source_train source-control = -0.0859375
none source-control = -0.0390625
```

这说明不能把 P4 fail 简单归因于 query-batch compensation、shared substrate 缺失、或 train/query reference 不一致。

## 18.3 official row scan 找到 P4 pass，但它不是 official success

对原 targeted out-dir 的 13 个 official release survivors 做 3 epoch scan，找到唯一 P4 pass：

```text
KMNIST seed1 / I24 / norm0.03 / sign-1
source_final_acc = 0.6796875
noop_final_acc = 0.640625
control_final_acc = 0.6328125
source_vs_noop = +0.0390625
source_vs_control = +0.046875
p4_pass = 1
```

但这行不能升 S5，因为它不满足 P3 hard gate：

```text
CouplingR2_delta = -0.022635042354497537
```

v12.23 文档明确要求：P4 official short-run 只能在 P3 通过后作为 official gate；shadow P4 可以运行，但 promotion_allowed 必须为 0。

因此该结果的正确解读是：

```text
有 task advantage 的 actuator state 存在，但它没有当前 Functional P3 几何资格。
```

不能解读为：

```text
S5 已达成
P4 official success
可以放宽 P3 gate
```

## 18.4 train-batch compensation actuator 修复失败

新增 I26/I27 是为了排除 query-batch compensation 引入的 reference mismatch：

```text
I26-TrainDirectLogitCompensatedQuadRelease
I27-TrainDirectLogitCompensatedShadowRelease
TrainDirectLogitCompensatedRandomControl
```

它们的实际表现：

| split | release rows | role_safe_movement_rows | outcome |
|---|---:|---:|---|
| MNIST s0-s2 | 6 | 0 | no P3 |
| Fashion s0-s2 | 37 | 0 | no P3 |
| KMNIST s0-s1 | 16 | 0 | no P3 |
| KMNIST s2 | 13 | 0 | no P3 |

结论：把 direct compensation reference 从 query batch 改到 train batch，不能保留 role-safe movement。它产生 release，但 release 与 movement gate 冲突更严重。

## 18.5 shadow-P4-to-P3 repair 的成功与失败

基于 18.3 的 shadow P4 pass，追加 targeted repair：只扫 KMNIST seed1 的 I24/I25 邻域，budget 从 0.004 到 0.04。

修复成功打开新的 P3 source：

```text
KMNIST seed1 / I24 / norm0.027 / sign+1
CouplingR2_delta = 0.0782001797938564
NoiseSignalLeak_delta = -0.05513792112469673
Reservoir_delta = -0.05831503868103027
control_gap = 0.04508481174707413
CEp99_delta = -0.762779951095581
ECE_delta = 0.0001386404037475586
```

这是一个比原 P3 source 更强的几何/audit P3 候选。

但是 P4 仍失败：

```text
1 epoch: source=0.4921875, noop=0.5, control=0.5234375, p4_pass=0
3 epoch: source=0.625, noop=0.6875, control=0.671875, p4_pass=0
```

5 种 compensation mode 也全部失败：

```text
per_variant_query source-noop = -0.0234375, source-control = -0.0234375
shared_source_query source-noop = -0.0234375, source-control = -0.0078125
per_variant_train source-noop = -0.015625, source-control = -0.0078125
shared_source_train source-noop = -0.03125, source-control = -0.0078125
none source-noop = -0.0234375, source-control = -0.015625
```

这给出一个很强的负结论：

```text
在 KMNIST seed1 / I24 邻域，P3 pass 点和 P4 pass 点不是同一个点。
```

## 18.6 同一邻域的 P3/P4 tradeoff

repair out-dir official row scan 只有两条 official survivor：

| source | P3 gate | P4 outcome |
|---|---|---|
| KMNIST seed1 / I24 / norm0.027 / sign+1 | pass | P4 fail |
| KMNIST seed1 / I24 / norm0.03 / sign-1 | fail, CouplingR2 negative | P4 pass |

这说明当前 family 的问题不是“没有任何动作有用”，而是：

```text
有用的 task update 与当前 loss-agnostic functional geometry gate 不重合。
```

这比单纯 P4 fail 更有诊断价值，因为它给出了下一轮要解决的精确矛盾：

```text
要么调整 functional geometry target，使 task-useful direction 也能保留正向 CouplingR2；
要么保留 P3 geometry，但改变 online update，使 P3 direction 不再在短训中输给 noop/control。
```

## 18.7 修复与审计记录

本轮代码修改：

```text
1. 新增 experiments/run_v1223_p4_compensation_modes.py。
2. 新增 experiments/run_v1223_p4_official_row_scan.py。
3. 在 experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py 中加入 I26/I27 train-batch compensation actuator。
4. 在 package_zip members 中加入两个新 P4 audit 脚本，修复 zip 缺脚本问题。
```

审计说明：

```text
apply_patch 两次尝试均被 sandbox helper 的 bwrap RTM_NEWADDR 错误阻断。
package list 修复改用 conda run -n kan python 做确定性文本替换。
该替换只增加两个 package member，不改变 gate、指标或实验结果。
```

复现包最终状态：

| zip | sha256 | contains new P4 scripts |
|---|---|---|
| results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip | 2ad392c3b87825ca557625d0f066905c1ca3e7f0600414d1830a1b5095c5ff35 | yes |
| results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip | 3e58b41d6e7b9f82acf0d7377880afaa90940854a100675a851e04756aeea5b2 | yes |

## 18.8 最终判断

本轮追加后，v12.23 的目标仍未完全达成：

```text
S5 official functional success = no
P4 official pass = no
```

但 v12.23 当前已经达成并巩固：

```text
S4 FunctionalP3Opened = yes
P4 diagnostic executed = yes
Line D executed = yes
required artifacts missing = 0
code review packet refreshed = yes
```

本轮最重要的科学结论：

```text
当前 I24/I25 direct-logit compensation family 中，P3 几何成功并不推出 P4 task success；反过来，P4 task success 的 shadow row 又破坏 P3 CouplingR2 gate。
```

下一步不应继续盲目扩大同一 I24/I25 budget grid。更合理的下一步是围绕 P3/P4 解耦做机制级修复：

```text
1. 为 P4-pass shadow row 设计 CouplingR2-preserving projection，而不是只调 norm。
2. 为 P3-pass row 设计 post-P3 online update，使它不再输给 noop/control。
3. 将 P3 gate 增加 task-useful shadow contrast 的诊断列，但 promotion 仍保持 fail-closed。
```


---

# 19. Blend 诊断复盘：简单 P3/P4 方向混合仍无法闭合

## 19.1 为什么继续做 blend？

18 节已经看到同一 I24 / KMNIST seed1 邻域的矛盾：

```text
norm0.027 / sign+：P3 pass，但 P4 fail
norm0.03 / sign-：P4 pass，但 P3 fail
```

因此一个自然修复方向是：用 sign+ 方向保留 CouplingR2 与 release geometry，再少量加入 sign- 的 task-useful shadow 方向，看看是否能得到 both P3/P4。

## 19.2 结果

blend 网格：

```text
p3_budget in 0.018,0.021,0.024,0.027,0.03
shadow_budget in 0.006,0.012,0.018,0.024,0.03
组合数 = 25
```

总结果：

```text
any_p3_pass = 1
any_p4_pass = 0
any_both_p3_p4_pass = 0
```

最佳 P3 行：

```text
p3_budget = 0.018
shadow_budget = 0.006
CouplingR2_delta = 0.02181613985160702
NoiseSignalLeak_delta = -0.02925848215818405
Reservoir_delta = -0.11880582571029663
control_gap = 0.02241254597902298
source_acc = 0.6640625
noop_acc = 0.6875
control_acc = 0.6171875
```

解读：它通过了 P3，并且 source 明显优于 random control；但 source 仍输 NoOp 0.0234375，所以 P4 不成立。

## 19.3 结论边界

这次 blend 诊断关闭了一个低成本假设：

```text
简单二段式 I24 sign+ / sign- blend 不能把 P3 pass 与 P4 pass 合并到同一个 candidate。
```

注意，这不是证明所有 coupling-preserving projection 都不可能。它只证明当前实现：

```text
I24 sign+ p3_budget -> I24 sign- shadow_budget -> direct_readout compensation
```

在 25 个小网格组合里没有找到 both pass。

## 19.4 最终状态更新

最终状态仍是：

```text
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_pass = 0
```

最终复现包：

```text
main zip = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
main sha256 = 336b4a3d4c702214b9482f1937fb2b700adfbe2ddf32089fdb8b649f4067d15c
repair zip = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
repair sha256 = 7cf9f90a86366b9490e0996a3c068498d01e45c7110d7ea8bfbf6614ce7897de
```

两个 zip 均确认包含：

```text
experiments/run_v1223_p4_compensation_modes.py
experiments/run_v1223_p4_official_row_scan.py
experiments/run_v1223_shadowp4_coupling_preserving_blend.py
```

## 19.5 下一步判断

本轮已经从三个角度尝试修复 P4：

```text
1. compensation/reference mode 改变：失败。
2. shadow-P4-to-P3 targeted repair：P3 成功，P4 失败。
3. coupling-preserving sign blend：P3 可恢复，P4 仍失败。
```

因此当前 v12.23 内继续扩同一 I24/I25 机制的边际价值很低。下一轮若继续，应转向结构性修复：不要只混合参数方向，而要显式优化 “P3 gate + NoOp-resistant task gain” 的联合目标，并保持 promotion fail-closed。


# 20. P4 深挖复盘：quad_only 打开 audit-only task gain，但官方 S5 仍未成立

## 20.1 本轮为什么继续？

19 节之后仍是：

```text
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_pass = 0
```

用户要求未达成目标不得停止，因此继续沿“post-P3 online update”方向推进，而不是再扩同一 I24/I25 norm grid。新的问题从：

```text
P3 source 为什么短训输给 NoOp？
```

变成：

```text
是否存在一种不改 loss、不用 teacher、不用 distillation、source/noop/control 公平共享的训练 role policy，使 P3 source 在任务收益、NLL/CEp99、LineC 和时间上同时不输 NoOp？
```

## 20.2 optimizer schedule 不是关键修复

e3/e5 的 optimizer schedule scan 对主 out-dir 和 repair out-dir 都失败：

```text
main: any_p4_pass = 0 / 20
repair: any_p4_pass = 0 / 20
```

这说明早期 P4 fail 不是简单 lr/weight decay/epochs=3/5 的问题。尤其 repair P3 source 在 schedule grid 中仍然低于 NoOp/control，说明“强 P3 几何”不会自动转化成短训优势。

## 20.3 e5 的小样本 pass 不可靠

repair e3/e5 trainable-role scan 找到一条 freeze_quad/e5 audit pass：

```text
source=0.3359375, noop=0.3125, control=0.3046875
```

但 512/256 confirmation 失败：

```text
any_p4_pass = 0
```

其中 direct_only/direct_gain/direct_branch_gain 有 accuracy advantage，但 NLL/CEp99 比 NoOp 差。同 temperature calibration 也没有修复：同一个温度作用于 source/noop/control 后，best direct_branch_gain 仍然是 source_NLL 和 source_CEp99 均劣于 NoOp。

结论：e5 小样本 pass 不能推广，不能作为 S5。

## 20.4 e8/e12 quad_only 是本轮真正新信号

延长到 e8/e12 后，两个 out-dir 都出现了 quad_only audit pass：

| out-dir | source | role | epochs | lr | source acc | noop acc | control acc | status |
|---|---|---|---:|---:|---:|---:|---:|---|
| main | KMNIST s0 / I24 / norm0.009 / sign+ | quad_only | 8 | 0.002 | 0.75 | 0.73828125 | 0.7265625 | audit-only pass |
| repair | KMNIST s1 / I24 / norm0.027 / sign+ | quad_only | 8 | 0.002 | 0.80859375 | 0.78515625 | 0.7890625 | audit-only pass |
| repair | KMNIST s1 / I24 / norm0.027 / sign+ | quad_only | 12 | 0.002 | 0.80859375 | 0.7734375 | 0.78125 | audit-only pass |

这是重要推进：

```text
P3 source 并非只能赢 P3 audit；在 quad_only post-P3 online update 下，它可以赢 NoOp 和 matched random control。
```

但这些结果最初仍是 `promotion_allowed=0`，所以不能直接写 S5。

## 20.5 strict trajectory verifier 的审计结果

为了避免把 audit pass 误写成 official success，本轮新增 trajectory verifier。它额外检查：

```text
1. per-epoch trajectory 与 AUC-error-time；
2. source/noop step-time q90 ratio；
3. final Acc/NLL/ECE/CEp99；
4. final LineC: CouplingR2, NoiseSignalLeak, RealSignalReservoirRatio；
5. matched control accuracy；
6. exact seed replay 与 seed-shift replay。
```

seed-shift replay 结果先失败：

```text
main seed-shift: source=0.7265625 < noop=0.734375/control=0.73828125
repair seed-shift: source acc still higher, but CEp99, step_time_ratio, NoiseSignalLeak/Reservoir fail
```

这说明 pass 对随机训练顺序有敏感性，不能写成 robust official success。

exact seed replay 后，主 targeted source 通过 strict trajectory gate：

```text
source=0.75, noop=0.73828125, control=0.7265625
source_NLL=1.2607176303863525 <= noop_NLL=1.2737209796905518
source_CEp99=10.830883026123047 <= noop_CEp99=14.015250205993652
source_ECE=0.25108960270881653 <= noop_ECE+0.02
step_time_ratio=0.9758790537510409
auc_error_time_ratio=0.9876783565783177
CouplingR2 source=0.3259254463806682 > noop=0.29090090994226603
NoiseSignalLeak source=0.13140901923179626 < noop=0.23948727548122406
RealSignalReservoirRatio source=0.8273053169250488 < noop=0.8910343050956726
strict_trajectory_gate_pass=1
promotion_allowed=0
```

repair exact replay 仍失败，尽管任务指标更强：

```text
source=0.80859375, noop=0.7734375, control=0.78125
source_NLL=0.8303794264793396 <= noop_NLL=0.9941673874855042
source_CEp99=9.003666877746582 <= noop_CEp99=10.317561149597168
source_ECE=0.19563405215740204 <= noop_ECE=0.20045700669288635
step_time_ratio=0.9808645769696466
auc_error_time_ratio=0.8903285087745244
CouplingR2 source=0.265527624440711 > noop=0.25369382153472153
NoiseSignalLeak source=0.08920775353908539 < noop=0.11743558943271637
RealSignalReservoirRatio source=0.913444995880127 > noop=0.8790651559829712
strict_trajectory_gate_pass=0
```

repair blocker 很清楚：RealSignalReservoirRatio 比 NoOp 更差。

## 20.6 为什么仍不能写 S5？

本轮不能把主 targeted strict pass 写成 official S5，原因不是数据不够好看，而是 provenance 与 promotion 口径仍未闭合：

```text
1. 相关脚本明确写入 promotion_allowed=0，route 未改 S5。
2. I24 DirectLogitCompensatedQuadRelease 使用 query-batch direct-logit compensation reference；它是 audit/diagnostic path，不是已经完成的 official precommit-only path。
3. 之前尝试的 train-batch compensation I26/I27 没有打开 P3：release rows 有，但 role_safe_movement_rows=0。
4. seed-shift replay 不稳定，说明当前 pass 还不能称为 robust official functional success。
```

因此最终状态应写为：

```text
S5 official functional success = no
S4 FunctionalP3Opened = yes
P4 trainable-role audit pass = yes
strict trajectory gate audit pass on main exact replay = yes
promotion_allowed = 0
```

这不是退步；它把之前“P3/P4 完全解耦”的结论推进为更精确的边界：

```text
P3 source 在 quad_only online training 下可以产生 NoOp-resistant task gain；
当前阻塞点转移到 official provenance、seed robustness、以及 train/precommit-only compensation reference。
```

## 20.7 本轮代码修复/新增的审计说明

本轮新增脚本均为诊断/复核脚本，不修改官方 gate 阈值，不改 loss，不引入 teacher/distillation，也不把 audit result 写成 route success。

```text
experiments/run_v1223_p4_optimizer_schedule_scan.py
  用于证明 e3/e5 lr/wd schedule 不是充分修复。

experiments/run_v1223_p4_trainable_role_scan.py
  用于公平比较 source/noop/control 在相同 role-policy 下的 post-P3 online update。

experiments/run_v1223_p4_temperature_calibration_scan.py
  用于验证 e5 acc-advantage 是否只是同温度校准问题。

experiments/run_v1223_p4_trajectory_gate_verifier.py
  用于 exact/seed-shift replay、trajectory/time/LineC strict gate 审计。

experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
  仅更新 package_zip members，确保复现包包含新增脚本。
```

`apply_patch` 在本环境多次被 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` 阻断；本轮文件写入继续使用已批准的 `conda run -n kan python` 做确定性文本写入/替换。该方式没有改 gate 阈值或实验数据。

## 20.8 最终产物与复现包

最终主 zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
sha256 = 6476520cee8efcbba167a316937538ca108b74a1ccfb8f03ae57b62d50c49a5b
entries = 85
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_trainable_role_scan_any_pass = 1
p4_trajectory_gate_last_strict_pass = 1
p4_trajectory_gate_promotion_allowed = 0
required_artifact_missing_count = 0
```

最终 repair zip：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
sha256 = 5d33fadd3e55d4395d056cfcdbc2685ffb8d492ec56fcb6c504c57409e50423f
entries = 81
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_trainable_role_scan_any_pass = 1
p4_trajectory_gate_last_strict_pass = 0
p4_trajectory_gate_promotion_allowed = 0
required_artifact_missing_count = 0
```

两个 zip 均已确认包含本轮新增脚本。

## 20.9 下一步结论

v12.23 现在可以更精确地收口，不应再说“P4 没有任何信号”。正确结论是：

```text
1. P4 task gain exists under quad_only post-P3 update。
2. Main exact replay 甚至通过了 strict trajectory gate。
3. 但 official S5 仍被 promotion provenance 和 robustness 阻断。
4. 之前的 train-batch official-like compensation I26/I27 没有打开 P3，所以下一步不是简单把 query reference 换成 train reference。
```

下一轮应优先做：

```text
A. 设计 precommit/train-stream-only direct compensation reference，替代 I24 的 query-batch reference，同时保留 I24 的 P3/P4 task gain。
B. 对 main strict pass 做至少 3 个 train-shuffle seed replay，确认稳定性；若不稳，route 仍保持 audit-only。
C. 对 repair source 设计 reservoir-vetoed quad_only update，使 RealSignalReservoirRatio 不再劣于 NoOp。
D. 只有当 precommit provenance、strict gate、seed robustness 同时通过，才能把 route 从 S4 升到 S5。
```


# 21. 继续推进复盘：audit pass 被收紧为不稳健信号，S5 仍不成立

## 21.1 本轮问题定义

上一节最强结果是 main exact replay 出现了 strict trajectory gate audit pass，但仍有两个硬阻塞：

```text
1. I24 使用 query-batch direct-logit compensation，provenance 不能 official promotion。
2. seed-shift replay 已显示不稳定。
```

因此本轮不是重复证明 “有一个好看的行”，而是继续推进三个实际修复方向：

```text
A. 训练顺序鲁棒性：main strict pass 是否能跨 train-shuffle seed 复现？
B. provenance 修复：I26 train-stream-only compensation 是否能替代 I24 query reference？
C. repair 线 reservoir blocker：是否能通过 checkpoint/veto 找到同时任务收益和 reservoir 不坏化的点？
```

## 21.2 main strict pass 不具备 train-shuffle robustness

主线固定：

```text
KMNIST seed0 / I24 / norm0.009 / sign+1
quad_only, epochs=8, lr=0.002
linec_seed_base=12239500
```

3 个 train-shuffle seed 结果：

| train_seed_base | source acc | noop acc | control acc | pass | blocker |
|---:|---:|---:|---:|---:|---|
| 12240400 | 0.75 | 0.734375 | 0.75 | 0 | source 未超过 control，且 NLL/CEp99 劣于 NoOp |
| 12241400 | 0.72265625 | 0.7421875 | 0.734375 | 0 | task accuracy 失败 |
| 12242400 | 0.72265625 | 0.70703125 | 0.7421875 | 0 | control 胜 source |

结论：

```text
main exact strict pass = real but not robust
train-shuffle pass_count = 0/3
不能 official promotion
```

这把上一轮的 main strict pass 从 “可能 S5 候选” 下修为：

```text
audit-only seed-sensitive signal。
```

## 21.3 train-stream-only compensation 修复失败

为解决 I24 query reference provenance，本轮把 source/control 切到：

```text
I26-TrainDirectLogitCompensatedQuadRelease
TrainDirectLogitCompensatedRandomControl
compensation_reference = train
compensation_batch = 32, 128, 256
```

结果：

| batch | source acc | noop acc | control acc | strict pass |
|---:|---:|---:|---:|---:|
| 32 | 0.734375 | 0.73828125 | 0.75 | 0 |
| 128 | 0.73046875 | 0.73828125 | 0.75 | 0 |
| 256 | 0.73046875 | 0.73828125 | 0.75 | 0 |

结论：

```text
I26 train-stream-only compensation 没有保留 I24 query-reference 的 task gain。
```

这说明当前 query compensation 可能在利用 validation/query geometry 的偶然对齐；简单改成 train-stream reference 会让 source 输给 NoOp/control。官方 provenance 修复仍未完成。

## 21.4 repair reservoir-veto checkpoint 找到 audit pass，但不稳健

repair source：

```text
KMNIST seed1 / I24 / norm0.027 / sign+1
quad_only, epochs=12, lr=0.002
```

per-epoch checkpoint scan 找到 epoch 11：

```text
source_acc = 0.80859375
noop_acc = 0.77734375
control_acc = 0.78515625
source_NLL = 0.8233494758605957 <= noop_NLL = 0.9744518399238586
source_CEp99 = 8.874238967895508 <= noop_CEp99 = 10.221264839172363
source_ECE = 0.1958829164505005 <= noop_ECE = 0.1987992823123932
source_CouplingR2 = 0.27166296495386577 > noop_CouplingR2 = 0.256553534179773
source_NoiseSignalLeak = 0.16414161026477814 < noop_NoiseSignalLeak = 0.2372053563594818
source_RealSignalReservoirRatio = 0.793380856513977 < noop_RealSignalReservoirRatio = 0.8506437540054321
```

这是一个真实的 audit pass，但不能 promotion，因为：

```text
1. 它来自 checkpoint scan，等于用 epoch/audit gate 做选择。
2. 它仍使用 query-batch I24 compensation。
3. LineC sketch seed 和 train-shuffle seed 复核不稳定。
```

## 21.5 repair e11 的 LineC 与 train-shuffle 稳定性

LineC sketch seed 复核：

```text
pass_count = 3/5
fail seeds: default 12239500, 12242600
```

这说明 reservoir/Noise 判定仍受 LineC sketch seed 影响。默认 seed 下 e11 失败点是：

```text
source_RealSignalReservoirRatio = 0.9068674445152283
noop_RealSignalReservoirRatio = 0.8772277235984802
```

固定一个通过的 LineC seed `12241600` 后，train-shuffle seed 复核：

```text
pass_count = 1/3
```

失败原因分别包括：

```text
12240400: NoiseSignalLeak source > NoOp
12241400: CouplingR2 source < NoOp and CEp99 source > NoOp + 0.05
```

结论：repair e11 的 checkpoint pass 是有价值的机制信号，但仍不是稳定 functional success。

## 21.6 lr=0.0015 稳定性修复失败

为了降低 quad_only update 波动，本轮追加 lr=0.0015：

```text
exact/default LineC seed: fail
exact/linec12241600: fail
trainseed12242400/linec12241600: fail
pass_count = 0/3
```

其中 lr=0.0015 虽然保持部分 task gain，但经常在 CouplingR2、Reservoir 或 control comparison 上失败。降低 lr 不是足够修复。

## 21.7 本轮代码修复/新增审计说明

本轮新增/修改没有改变 official gate 阈值，也没有把 audit result 写成 route success：

```text
experiments/run_v1223_p4_trajectory_gate_verifier.py
  新增 source/control actuator override 与 train/query compensation reference 参数。
  用于复核 I26 train-stream-only compensation，不改变原 I24 结果。

experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py
  新增 per-epoch checkpoint audit；promotion_allowed 固定为 0。
  它的结果用于定位机制，不用于 official promotion。

experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py
  package_zip 增加 reservoir-veto 脚本，便于复现。
```

执行环境仍为：

```text
conda env = kan
GPU = cuda:0/1/2/3 并行
no_fake = 1
promotion_allowed = 0
```

## 21.8 最终状态

最终状态仍是：

```text
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_pass = 0
required_artifact_missing_count = 0
```

新增汇总 artifact：

```text
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_continuation_p4_robustness_and_repair_summary.json
results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_continuation_p4_robustness_and_repair_summary.json
```

最终 zip：

```text
main zip = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip
main sha256 = 6ba42be4a7bc68a816b2e89e613a503bc867a2d51031fef908642e1016528206
entries = 99

repair zip = results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip
repair sha256 = e303966aeb723de1df7973760aed5a5da1e69c36083b67a4ca33e3e081fea4b5
entries = 107
```

## 21.9 下一步判断

本轮推进后，v12.23 的边界进一步变清楚：

```text
1. P4 task gain 不是幻觉，但高度依赖 query-reference、train-shuffle seed、LineC sketch seed。
2. train-stream-only compensation 简单替代失败，说明 provenance 修复不是换 batch 就够。
3. reservoir-veto checkpoint 能找到机制上很接近的 audit pass，但不具备 robustness。
4. lr 降低不能稳定修复。
```

下一步如果继续，应转向更结构化的修复，而不是继续同一 verifier 暴力扫：

```text
A. 构造 train-stream ensemble compensation reference：多个 train micro-batch 的 ridge/median update，而不是单一 batch。
B. 对 LineC metric 做 multi-sketch aggregation gate，减少单 sketch seed 偶然性；promotion 前必须通过 aggregation，不可 cherry-pick。
C. 对 quad_only update 加 reservoir regularizer 或 reservoir-vetoed checkpoint policy 的 precommit proxy；该 proxy 不能使用 audit labels/LineC target。
```


# 22. Train-stream ensemble compensation 复盘：provenance 修复仍未打开

## 22.1 为什么做 ensemble？

17 节中，I26 train-stream-only compensation 的 batch 32/128/256 都失败。但单一 train reference 可能过于局部，因此继续测试多个 train micro-batch 的 ridge compensation delta ensemble：

```text
source = I26-TrainDirectLogitCompensatedQuadRelease
control = TrainDirectLogitCompensatedRandomControl
microbatch = 32
ensemble_count = 2,4,8
quad_only, e8, lr=0.002
```

## 22.2 结果

```text
any_pass = 0
```

| ensemble_count | source acc | noop acc | control acc | outcome |
|---:|---:|---:|---:|---|
| 2 | 0.73046875 | 0.73828125 | 0.75 | fail |
| 4 | 0.734375 | 0.73828125 | 0.74609375 | fail |
| 8 | 0.734375 | 0.73828125 | 0.74609375 | fail |

虽然 NLL/CEp99 比 NoOp 好，但 source 没有赢 NoOp/control，因此 P4 strict gate 不成立。

## 22.3 结论

这进一步关闭了一个清晰假设：

```text
不是因为 train reference 单 batch 太窄导致 I26 fail；即使用 2/4/8 个 train micro-batch 做 ensemble，task gain 仍不能保留。
```

当前最可信的边界是：

```text
I24 query-reference compensation 可以制造 audit-only task gain；
train-stream-only reference，包括 single batch 和 ensemble batch，目前都不能复制该 gain。
```

因此 official S5 仍不能成立：

```text
route = S4-FunctionalP3Opened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
```

最终复现包：

```text
main sha256 = 6ba42be4a7bc68a816b2e89e613a503bc867a2d51031fef908642e1016528206
repair sha256 = e303966aeb723de1df7973760aed5a5da1e69c36083b67a4ca33e3e081fea4b5
```


# 23. Multi-sketch 与 precommit proxy 复盘：P4 仍未达成 official S5

## 23.1 本轮为什么继续

前一轮已经看到 I24 query-reference 在单一 LineC seed 下可出现 strict pass，但 train-shuffle、LineC seed 和 provenance 都不稳定。按计划不能在这种状态下停止或 promotion。因此本轮继续推进三个问题：

- 单 sketch seed 是否导致误判。
- 事后 reservoir-veto checkpoint 能否变成不看 label、CE、LineC 的 precommit proxy。
- train-stream compensation 是否能复制 query-reference 的 task gain。

## 23.2 代码修改审计

新增脚本 experiments/run_v1223_p4_multisketch_aggregate_gate.py。

用途：同一 source、noop、control 训练后，对多个 LineC sketch seed 做聚合审计。输出 strict_all_sketch_pass、strict_majority_sketch_pass、linec_seed_pass_count、task_gate_pass。promotion_allowed 固定为 0。

新增脚本 experiments/run_v1223_p4_precommit_proxy_checkpoint.py。

用途：用不含 label、CE、loss、LineC、audit target 的 unlabeled train-probe logit proxy 选择 checkpoint。选完后才做 task 和 LineC 审计。脚本显式记录 proxy_uses_labels=0、proxy_uses_ce_or_loss=0、proxy_uses_linec_or_audit_target=0、promotion_allowed=0。

后续修复：初版 best_score 选择早期 epoch，出现早停偏差。因此加入 latest_eligible、best_after_warmup、min_selection_epoch，作为 planned repair direction 的真实尝试。

打包修复：experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py 将两个新脚本加入 package_zip，最终 zip 已确认包含新代码。

## 23.3 multi-sketch aggregation 结果

batch32 sketch_dim8：

- main epoch 8：task gate 1，LineC pass 4/5，strict all 0，strict majority 1。关键数值：source 0.75，noop 0.73828125，control 0.7265625。
- repair epoch 11：task gate 1，LineC pass 3/5，strict all 0，strict majority 1。关键数值：source 0.80859375，noop 0.77734375，control 0.78515625。

解读：query-reference I24 确实有 task gain，并且不是单一 seed 全部偶然。但它不能通过 all-sketch gate。不能 promotion 的原因是 strict_all_sketch_pass=0、source 仍是 query-reference compensation，而且 train-shuffle robustness 之前已经失败。

## 23.4 precommit proxy 结果

best_score：

- main selected_epoch 2：task gate 0，LineC pass 2/5。失败模式：source 输 NoOp。
- repair selected_epoch 4：task gate 1，LineC pass 0/5。失败模式：task 过但 LineC 全 fail。

latest_eligible 加 min epoch 6：

- main selected_epoch 12：task gate 0，LineC pass 3/5。失败模式：source 输 control。
- repair selected_epoch 12：task gate 1，LineC pass 2/5。失败模式：task 过但 LineC 不稳。

best_after_warmup 加 min epoch 8：

- main selected_epoch 8：task gate 1，LineC pass 4/5。失败模式：majority pass，但 all fail。
- repair selected_epoch 8：task gate 1，LineC pass 0/5。失败模式：task 过但 LineC 全 fail。

结论：当前 unlabeled logit proxy 不能替代 reservoir-veto checkpoint。它可以通过 warmup 把 main 拉回 epoch 8 的 majority-positive 状态，但不能给出 all-sketch functional geometry。repair 线也没有稳定。

## 23.5 train-stream compensation 结果

train-stream I26 precommit proxy：

- selected_epoch 8。
- source_actuator I26-TrainDirectLogitCompensatedQuadRelease。
- control_actuator TrainDirectLogitCompensatedRandomControl。
- source acc 0.734375。
- noop acc 0.73828125。
- control acc 0.75。
- task_gate_pass 0。
- LineC pass 0/5。
- strict_all_sketch_pass 0。
- strict_majority_sketch_pass 0。

结论：query-reference I24 的 task gain 仍不能被 train-stream compensation 复制。provenance blocker 未解除。

## 23.6 larger LineC batch/sketch 结果

batch64 sketch_dim24：

- main epoch 8：task gate 0，LineC pass 0/5。source task 不再满足 gate，LineC 全 fail。
- repair epoch 11：task gate 1，LineC pass 2/5。task 仍强，但 LineC 仍不稳。

结论：失败不能简单归因于 small sketch_dim=8 的随机噪声。更大 batch/sketch 下没有打开 official gate。

## 23.7 当前达成情况

本轮继续后，v12.23 仍未达成目标：

- route = S4-FunctionalP3Opened。
- official_success_reached = 0。
- p4_pass = 0。
- promotion_allowed = 0。

新增边界：

- I24 query-reference P4 有真实 task gain，但只达到 majority-sketch，不满足 all-sketch。
- precommit-safe proxy 没能稳定选择 all-sketch checkpoint。
- train-stream provenance 仍不能复制 task gain。
- larger LineC batch/sketch 不会自动修复。

因此不能写 S5，也不能把 majority-sketch 结果包装成 official success。

## 23.8 最终 zip

- main zip：results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted/v1223_code_review_packet.zip。
- main sha256：f14d33cd3cf9fab2823e7d744c56a83cde3ef35036fdecfaf9b3f11f13889c39。
- main entries：116。
- repair zip：results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25/v1223_code_review_packet.zip。
- repair sha256：b403579e714f318b36d7f7ce44fde2e1f98ae75f44078dc3bcb8d5a5f12ea4c4。
- repair entries：120。

审计包新增确认包含：

- experiments/run_v1223_p4_multisketch_aggregate_gate.py。
- experiments/run_v1223_p4_precommit_proxy_checkpoint.py。
- experiments/run_v1223_p4_train_ensemble_compensation_scan.py。

## 23.9 下一步不应继续的无效方向

不建议继续：

- 单纯增加 I24 query-reference epoch 或 seed 来找单 seed pass。这会扩大 audit-only cherry-pick 风险。
- 继续调当前 unlabeled logit proxy 的阈值。只要没有新的 precommit observable，它已经显示不能稳定对应 LineC all-pass。
- 继续 train-stream compensation batch count 的简单扩展。single batch、ensemble、precommit checkpoint 都失败。

更合理的后续方向：

- 重新定义 precommit-safe signal/reservoir observable，而不是只看 logit entropy、margin、usage。
- 如果继续 P4，应先解决 provenance。source 必须来自 train-stream 或真正 precommit observable，不能依赖 query-reference compensation。
- Line D Rational/Fourier 已有 hardening 价值，下一轮可作为独立 classic-family functional 线推进，而不是继续拿 I24 query compensation 做 official。
