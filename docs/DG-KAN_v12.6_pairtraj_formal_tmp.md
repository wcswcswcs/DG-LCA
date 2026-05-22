# DG-KAN v12.6 Lower-Level Fused Hinge/Quadratic + Functional Geometry 结果复盘

> 本复盘记录 `DG-KAN_v12.6_LowerLevelFusedHingeQuadratic_FunctionalGeometry_完整计划.md` 的真实执行。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位成功或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R4-LowCostOrFixedPEfficiencyPassA4Fail
base_qualified = False
functional_open = False
artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b131_h160_pairtraj_absdiag050_f1_f4_final075_20260520T231000Z
next_recommended_action = restore structural interaction without hidden-size brute force; do not enter task
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
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `torch-reference-simple-fast` | `5.422877532765527` | `0.0` | `0.0` | `0` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `FHQ-triton-two-kernel-forward` | `3.626103527398592` | `1.430511474609375e-06` | `1.7703416688163998e-07` | `0` |


## 3. Backward / Full-Step

| candidate_id | implementation_id | step_ratio_q90 | memory_ratio_q90 | gradient_correctness_pass | official_efficiency_pass | failure |
|---|---|---|---|---|---|---|
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `1.2170065893125566` | `2.7185714285714284` | `1` | `0` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-forward-delta-readout-proj-grad-learnableP` | `1.0218999172944936` | `0.5966666666666667` | `1` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `0.974534355477866` | `0.5966666666666667` | `1` | `1` | `` |
| `B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F1-triton-forward-delta-readout-grad-fixedP` | `0.9621187912614572` | `0.13` | `1` | `1` | `` |
| `B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F4-triton-workspace-forward-delta-readout-grad-fixedP` | `0.685358064760707` | `0.13` | `1` | `1` | `` |


Gradient rows 摘要：

| candidate_id | implementation_id | grad_role | grad_relerr | grad_cos | official_backward_pass | status |
|---|---|---|---|---|---|---|
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `direct_readout` | `8.91909905931243e-08` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `quad_proj` | `9.743497741965257e-08` | `1.0000001192092896` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `quad_readout` | `5.545191328337751e-08` | `0.9999999403953552` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `branch_scale` | `3.722325914168323e-08` | `0.9999998807907104` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `logit_gain` | `4.731066738372647e-08` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `bias` | `2.9695980785504617e-08` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `direct_readout` | `0.0007938210037536919` | `0.9999998807907104` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `quad_proj` | `0.0014420590596273541` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `quad_readout` | `0.0007872309652157128` | `0.9999998807907104` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `branch_scale` | `2.9720604288741015e-07` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `logit_gain` | `2.563679117884021e-07` | `0.9999998211860657` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `bias` | `8.960396513657543e-08` | `0.9999999403953552` | `1` | `` |


## 4. A4 Expression

| candidate_id | A4_pass | A4_expression_pass | status | reason |
|---|---|---|---|---|
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `0` | `0` | `` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `0` | `0` | `` | `` |
| `B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `0` | `0` | `` | `` |
| `B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `0` | `0` | `` | `` |


## 5. A5 Task / AUC Attribution

| candidate_id | mean_delta | worst_delta | near_pass_rate | ece_ok | auc_step_ok | auc_time_ok | A5_task_pass | status | reason |
|---|---|---|---|---|---|---|---|---|---|
| `` | `` | `` | `` | `` | `` | `` | `` | `not_run` | `A4 expression gate not opened; A5 task legally closed` |


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
| `A5_task_legally_closed` | `` | `A4 expression gate not opened; A5 task legally closed` | `` |
| `A4_not_open_or_failed` | `` | `` | `only run A4 after official lower-level full-step; if cheap primitive fails A4 restore structural interaction without increasing hidden brute-force` |


## 8. No-Fake / Hash

Provenance first row:

```text
artifact = v126_auc_attribution.csv
rows_checked = 1
no_fake_pass = 1
no_proxy_pass = 1
no_cpu_offload_pass = 1
```

Selected hashes:

| artifact | sha256_prefix |
|---|---|
| `v126_auc_attribution.csv` | `36aa5f9912a2` |
| `v126_candidate_manifest.csv` | `9f86a5e56dd3` |
| `v126_expression_audit.csv` | `737017a00d94` |
| `v126_expression_summary.csv` | `ba5cf2f6f020` |
| `v126_failure_table.csv` | `aa2e1950b10b` |
| `v126_frozen_readout.csv` | `362bba9456d0` |
| `v126_fullstep_profile.csv` | `c1610d7b9768` |
| `v126_functional_control_matrix.csv` | `83c55637cae6` |
| `v126_functional_five_step.csv` | `8b17899e90de` |
| `v126_functional_one_step.csv` | `28c3b1a22108` |
| `v126_fused_backward_correctness.csv` | `3981ab244150` |
| `v126_fused_forward_profile.csv` | `aefae73f1d5d` |


## 9. 分析结论

1. 本轮已执行 v12.6 的第一层升级：新增真实 Triton FHQ0 forward smoke，并尝试 fixed-P F1 backward/full-step profile。
2. A4/A5 只有在 v12.6 lower-level full-step official pass 后才会打开；如果未打开，属于合法关闭，不是 task success 或 failure 伪造。
3. `B51a fixedP` 是按计划的 backward cost repair，不是降低 gate；如果它不能同时通过 correctness、step 与 memory gate，下一步应继续 lower-level backward/update fusion。
4. Functional 仍然只作为 cloned diagnostic；base 未合格时不允许 official route。
