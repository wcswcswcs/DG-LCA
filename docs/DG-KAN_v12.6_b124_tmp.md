# DG-KAN v12.6 Lower-Level Fused Hinge/Quadratic + Functional Geometry 结果复盘

> 本复盘记录 `DG-KAN_v12.6_LowerLevelFusedHingeQuadratic_FunctionalGeometry_完整计划.md` 的真实执行。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位成功或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R2-AUCStepAndTaskFail
base_qualified = False
functional_open = False
artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b124_h160_absdiag050_fixedbranch_fixedgain_identitytailquad020_hingeamp025_f3_workspace_final075_20260520T213000Z
next_recommended_action = return to Line A timing if AUC-step passed; otherwise return to Line P/C primitive trajectory
```

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B51a-SimpleFastTaskGeometry-h128-fixedP-temp075`，并让 `SimpleFastTaskGeometryKAN` 支持 `fixedp` projection buffer | 对应 v12.6 A3 “先固定 P 降低 backward 成本”；不降低 gate，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 `B54a` / `B55a` learnable-P task-geometry repair candidates | 在 lower-level F2 打开后测试 IdentityTail 与 quad050；仍不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | fixed-P 时 manual backward 不写 `quad_proj.grad`，functional direction 与 trainable params 对齐 | 避免 fixed-P 变体在 optimizer/functional diagnostic 中错配参数；这是审计性修复，不改变 B48a。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | 新增 FHQ Triton kernel module | 将 FHQ forward、delta、direct/quad/proj grad 从 runner 下沉到 `dgkan`；runner 不再承载关键 kernel 实现。 |
| `dgkan/kernels/fused_hinge_quadratic.py` | F2 projection-gradient tile 调整为 `64x64`，并加入 two-hinge / sqdiag direct forward/grad 支持 | 继续 v12.6 lower-level kernel repair；不改变 loss/gate，只扩展可测 primitive。 |
| `experiments/run_v126_lowerlevel_fhq_functional_geometry.py` | 新增 v12.6 runner，写 `v126_*` artifacts | 独立于 v12.5.2 runner；只负责 protocol/gate/artifact/report，调用 `dgkan.kernels.fused_hinge_quadratic`。 |
| `experiments/run_v126_lowerlevel_fhq_functional_geometry.py` | 新增 v12.6 AUC-step / AUC-time attribution | 若 task 被合法打开，将区分 P4-A timing blocker 与 P4-B trajectory blocker。 |
| `experiments/run_v126_lowerlevel_fhq_functional_geometry.py` | full-step memory 改为 warmup reset 后 incremental peak accounting | 去除常驻数据/模型分配对 step memory gate 的污染；仍使用 `memory_ratio <= 0.80`，不降低 gate。 |

## 2. Fused Forward

| candidate_id | implementation_id | forward_ratio_vs_mlp_q90 | logits_max_abs_err_vs_reference | logits_relerr_vs_reference | official_forward_pass |
|---|---|---|---|---|---|
| `MLP-same-param-AdamW` | `MLP-reference` | `1.0` | `0.0` | `0.0` | `1` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `torch-reference-simple-fast` | `8.021457680685316` | `0.0` | `0.0` | `0` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `FHQ-triton-two-kernel-forward` | `2.0137900544102347` | `3.5762786865234375e-07` | `3.3183044934048667e-07` | `0` |


## 3. Backward / Full-Step

| candidate_id | implementation_id | step_ratio_q90 | memory_ratio_q90 | gradient_correctness_pass | official_efficiency_pass | failure |
|---|---|---|---|---|---|---|
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `2.3941780918285374` | `2.7195238095238095` | `1` | `0` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F2-triton-forward-delta-readout-proj-grad-learnableP` | `1.055328062956001` | `0.5957142857142858` | `1` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `1.4433949663893555` | `0.5957142857142858` | `1` | `0` | `` |
| `B124a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F1-triton-forward-delta-readout-grad-fixedP` | `0.9569840822556168` | `0.12904761904761905` | `1` | `1` | `` |


Gradient rows 摘要：

| candidate_id | implementation_id | grad_role | grad_relerr | grad_cos | official_backward_pass | status |
|---|---|---|---|---|---|---|
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `direct_readout` | `9.116264010344821e-08` | `1.0` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `quad_proj` | `9.19438392088523e-08` | `0.9999998807907104` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `quad_readout` | `5.5491724992862146e-08` | `1.0` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `bias` | `1.0580949805216733e-07` | `1.0` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `direct_readout` | `0.000799425586592406` | `0.9999998807907104` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `quad_proj` | `0.0016229540342465043` | `0.9999998807907104` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `quad_readout` | `0.0009105305070988834` | `1.0` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `bias` | `5.6931888536837505e-08` | `1.0` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F3-triton-workspace-learnableP-proj-grad` | `direct_readout` | `0.0008221303578466177` | `0.9999998807907104` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F3-triton-workspace-learnableP-proj-grad` | `quad_proj` | `0.0015982706099748611` | `0.9999999403953552` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F3-triton-workspace-learnableP-proj-grad` | `quad_readout` | `0.0008792526787146926` | `1.0` | `1` | `` |
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `F3-triton-workspace-learnableP-proj-grad` | `bias` | `1.0677234030254112e-07` | `1.0` | `1` | `` |


## 4. A4 Expression

| candidate_id | A4_pass | A4_expression_pass | status | reason |
|---|---|---|---|---|
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `1` | `1` | `` | `` |
| `B124a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `1` | `1` | `` | `` |


## 5. A5 Task / AUC Attribution

| candidate_id | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok | A5_task_pass | status | reason |
|---|---|---|---|---|---|---|---|---|---|
| `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `0.014756944444444444` | `-0.00390625` | `1.0` | `1` | `0` | `0` | `0` | `` | `` |
| `B124a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `-0.12456597222222222` | `-0.173828125` | `0.0` | `0` | `0` | `0` | `0` | `` | `` |


## 6. Line C / Functional

Line C:

| candidate_id | CouplingR2 | CouplingCorr | KernelDrift | official_gate_open |
|---|---|---|---|---|
| `MLP-same-param-AdamW` | `0.21133869467893396` | `0.45971596240997314` | `6.950242042541504` | `0` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.27571050261201546` | `0.5250893831253052` | `1.2660470008850098` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.27571062973572524` | `0.5250895619392395` | `1.2660470008850098` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.21044565166676954` | `0.45874372124671936` | `4.435661792755127` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0.21624587671220385` | `0.4650229513645172` | `1.6140456199645996` | `0` |


Signal / reservoir:

| candidate_id | RealSignalReservoirRatio | NoiseSignalLeak | signal_effective_rank | official_gate_open |
|---|---|---|---|---|
| `MLP-same-param-AdamW` | `0.08000694215297699` | `0.4323427677154541` | `3.4235739707946777` | `0` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.8621050715446472` | `0.13670609891414642` | `1.8789453506469727` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.8621050119400024` | `0.13670629262924194` | `1.8789451122283936` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.7805043458938599` | `0.10373576730489731` | `1.0603744983673096` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0.8700215816497803` | `0.001413449877873063` | `1.2137125730514526` | `0` |


Functional control diagnostic:

| candidate_id | best_functional_score | best_control_score | control_gap_vs_best_control | beats_controls | official_gate_open |
|---|---|---|---|---|---|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.40474163833291255` | `0.5930645875632763` | `-0.9978062258961888` | `0` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.40474167894897184` | `0.593064546585083` | `-0.9978062255340548` | `0` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-0.7320637674881989` | `0.2663341388106346` | `-0.9983979062988335` | `0` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `-0.7034559227676102` | `0.2943388670682907` | `-0.9977947898359009` | `0` | `0` |


解释：Functional 在 `base_qualified = false` 时继续保持 diagnostic；即使 control gap 为正，也不能写成 official functional success。

## 7. Failure Table

| failure_code | candidate_id | reason | action_recommended |
|---|---|---|---|
| `A5_task_gate_fail` | `B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `` | `if AUC-step fails return to Line P/C; if AUC-step passes and AUC-time fails return to Line A` |
| `A5_task_gate_fail` | `B124a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075` | `` | `if AUC-step fails return to Line P/C; if AUC-step passes and AUC-time fails return to Line A` |


## 8. No-Fake / Hash

Provenance first row:

```text
artifact = v126_auc_attribution.csv
rows_checked = 18
no_fake_pass = 1
no_proxy_pass = 1
no_cpu_offload_pass = 1
```

Selected hashes:

| artifact | sha256_prefix |
|---|---|
| `v126_auc_attribution.csv` | `f357dba5321e` |
| `v126_candidate_manifest.csv` | `c6f4aa5c18c5` |
| `v126_expression_audit.csv` | `35342ef45c20` |
| `v126_expression_summary.csv` | `e24dccf66391` |
| `v126_failure_table.csv` | `0319aab17430` |
| `v126_frozen_readout.csv` | `ef25a27e5729` |
| `v126_fullstep_profile.csv` | `21e2415f67d9` |
| `v126_functional_control_matrix.csv` | `83c55637cae6` |
| `v126_functional_five_step.csv` | `8b17899e90de` |
| `v126_functional_one_step.csv` | `28c3b1a22108` |
| `v126_fused_backward_correctness.csv` | `a5f18a473169` |
| `v126_fused_forward_profile.csv` | `ab9ca1fba742` |


## 9. 分析结论

1. 本轮已执行 v12.6 的第一层升级：新增真实 Triton FHQ0 forward smoke，并尝试 fixed-P F1 backward/full-step profile。
2. A4/A5 只有在 v12.6 lower-level full-step official pass 后才会打开；如果未打开，属于合法关闭，不是 task success 或 failure 伪造。
3. `B51a fixedP` 是按计划的 backward cost repair，不是降低 gate；如果它不能同时通过 correctness、step 与 memory gate，下一步应继续 lower-level backward/update fusion。
4. Functional 仍然只作为 cloned diagnostic；base 未合格时不允许 official route。
