# DG-KAN v12.6 Lower-Level Fused Hinge/Quadratic + Functional Geometry 结果复盘

> 本复盘记录 `DG-KAN_v12.6_LowerLevelFusedHingeQuadratic_FunctionalGeometry_完整计划.md` 的真实执行。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位成功或 CPU offload。Base 未合格时 Functional 只允许 diagnostic，不允许写成 official success。

## 0. 最新结论

```text
route = R4-LowCostOrFixedPEfficiencyPassA4Fail
base_qualified = False
functional_open = False
artifact = results/v12_6_lowerlevel_fhq_functional_geometry/v126_b131_pairtraj_smoke_20260520T230000Z
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
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `torch-reference-simple-fast` | `4.5006760898451175` | `0.0` | `0.0` | `0` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `FHQ-triton-two-kernel-forward` | `1.7404817091343057` | `1.430511474609375e-06` | `2.919722135175107e-07` | `0` |


## 3. Backward / Full-Step

| candidate_id | implementation_id | step_ratio_q90 | memory_ratio_q90 | gradient_correctness_pass | official_efficiency_pass | failure |
|---|---|---|---|---|---|---|
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `1.0777089472864` | `1.0609523809523809` | `1` | `0` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-forward-delta-readout-proj-grad-learnableP` | `0.6121829586259829` | `0.5966666666666667` | `1` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F3-triton-workspace-forward-delta-readout-proj-grad-learnableP` | `2.1853255154418165` | `0.5966666666666667` | `1` | `0` | `` |
| `B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F1-triton-forward-delta-readout-grad-fixedP` | `0.5462580164376075` | `0.13` | `1` | `1` | `` |
| `B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F4-triton-workspace-forward-delta-readout-grad-fixedP` | `0.6370805189316201` | `0.13` | `1` | `1` | `` |


Gradient rows 摘要：

| candidate_id | implementation_id | grad_role | grad_relerr | grad_cos | official_backward_pass | status |
|---|---|---|---|---|---|---|
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `direct_readout` | `9.835806480396059e-08` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `quad_proj` | `9.186771166014296e-08` | `1.0000001192092896` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `quad_readout` | `6.99365187983858e-08` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `branch_scale` | `4.491343830181904e-08` | `0.9999998807907104` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `logit_gain` | `5.1822375723986625e-08` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `B48a-existing-manual-ce-torch-reduction` | `bias` | `8.72209966473747e-08` | `1.0000001192092896` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `direct_readout` | `0.0010081890504807234` | `0.9999998807907104` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `quad_proj` | `0.0013686207821592689` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `quad_readout` | `0.0006932762917131186` | `0.9999999403953552` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `branch_scale` | `2.0919704013522278e-07` | `1.0` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `logit_gain` | `2.0684358048583817e-07` | `1.0000001192092896` | `1` | `` |
| `B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075` | `F2-triton-learnableP-proj-grad` | `bias` | `8.996936884386741e-08` | `1.0` | `1` | `` |


## 4. A4 Expression

| candidate_id | A4_pass | A4_expression_pass | status | reason |
|---|---|---|---|---|
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
| `MLP-same-param-AdamW` | `0.6962752903168432` | `0.8344374895095825` | `8.486576080322266` | `0` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.8001197592205573` | `0.8945083618164062` | `1.9034978151321411` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.8001196873578679` | `0.8945083022117615` | `1.9034979343414307` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.6665533716200089` | `0.816428005695343` | `4.355406284332275` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0.6581272675518215` | `0.811265230178833` | `2.717763900756836` | `0` |


Signal / reservoir:

| candidate_id | RealSignalReservoirRatio | NoiseSignalLeak | signal_effective_rank | official_gate_open |
|---|---|---|---|---|
| `MLP-same-param-AdamW` | `0.561637282371521` | `0.21929393708705902` | `1.244736671447754` | `0` |
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `0.6940956115722656` | `0.0645122155547142` | `1.1507974863052368` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `0.6940955519676208` | `0.06451229751110077` | `1.1507974863052368` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `0.9940088987350464` | `3.4297154005713537e-09` | `1.1465721130371094` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `0.8500789999961853` | `0.09205583482980728` | `1.3830969333648682` | `0` |


Functional control diagnostic:

| candidate_id | best_functional_score | best_control_score | control_gap_vs_best_control | beats_controls | official_gate_open |
|---|---|---|---|---|---|
| `B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075` | `-0.34202604065301956` | `0.6557829827070236` | `-0.9978090233600432` | `0` | `0` |
| `B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050` | `-0.3420262032495144` | `0.655783161520958` | `-0.9978093647704723` | `0` | `0` |
| `B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050` | `-1.202250203687902` | `-0.20445966720581055` | `-0.9977905364820914` | `0` | `0` |
| `B48a-SimpleFastTaskGeometry-h128-temp075` | `-0.7116374053029679` | `0.28529350459575653` | `-0.9969309098987245` | `0` | `0` |


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
| `v126_candidate_manifest.csv` | `1697c4063388` |
| `v126_expression_audit.csv` | `c34c8cf7d48a` |
| `v126_expression_summary.csv` | `2570cd7b0844` |
| `v126_failure_table.csv` | `aa2e1950b10b` |
| `v126_frozen_readout.csv` | `34a90466afe8` |
| `v126_fullstep_profile.csv` | `b20ca76fee7e` |
| `v126_functional_control_matrix.csv` | `088f55006c7e` |
| `v126_functional_five_step.csv` | `8b17899e90de` |
| `v126_functional_one_step.csv` | `b98366d000d2` |
| `v126_fused_backward_correctness.csv` | `d4ea1a49adcf` |
| `v126_fused_forward_profile.csv` | `2e33bc8c7f7f` |


## 9. 分析结论

1. 本轮已执行 v12.6 的第一层升级：新增真实 Triton FHQ0 forward smoke，并尝试 fixed-P F1 backward/full-step profile。
2. A4/A5 只有在 v12.6 lower-level full-step official pass 后才会打开；如果未打开，属于合法关闭，不是 task success 或 failure 伪造。
3. `B51a fixedP` 是按计划的 backward cost repair，不是降低 gate；如果它不能同时通过 correctness、step 与 memory gate，下一步应继续 lower-level backward/update fusion。
4. Functional 仍然只作为 cloned diagnostic；base 未合格时不允许 official route。
