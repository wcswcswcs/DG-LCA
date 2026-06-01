# DG-KAN v12.18：B320 代码审查后重锚 + Loss-Agnostic Functional Update 可见性重建完整计划

> 版本：v12.18 execution plan  
> 基于：v12.17.2 实验复盘、`v1217_code_review_packet.zip` 本地解压代码审查、此前 B320/B314/B109/FHQ 主线结果。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 总原则：B320 base 当前不再小修；functional update 必须 loss-agnostic；CE / ECE / CEp99 / Brier / label-defined hard release 只能作为审计指标，不能作为 functional direction 的设计目标；不得按 MNIST / Fashion-MNIST / KMNIST 做数据集专用调参。

---

# 0. 一句话结论

v12.17.2 不是没有进展。它把 functional update 的问题进一步缩小到：

$$
\boxed{\text{B320 base 已经不是主 blocker；当前 blocker 是 loss-agnostic value source 不可见。}}
$$

但本地解压代码后，我认为还必须加一个更严肃的结论：

$$
\boxed{\text{当前 v12.17.2 的“loss-agnostic visibility”实现语义还需要重审。}}
$$

原因不是 Codex 输出的 CSV 不完整，而是核心实现路径里存在三个必须人工确认的点：

```text
1. B320 base 使用了 label-informed trainprobe 初始化；这会影响 base 贡献归因。
2. 当前 Line C 的部分 sketch / reservoir / noise 指标由 CE gradient 和真实/置乱 label CE 构造；它们可以作为审计，但不能作为 deployable loss-agnostic feature。
3. v12.17.2 runner 本身是 offline diagnostic/postprocess，没有重新执行 B320、fused kernel、P3 actuator 或 P4 functional short-run。
```

因此，v12.18 的核心不是继续调 lambda、window、rank 或 branch damping，而是：

$$
\boxed{\text{先把代码语义审清楚，再重建真正可部署的 loss-agnostic observables。}}
$$

---

# 1. 这次实验结果怎么理解

## 1.1 明确进展

v12.17.2 的进展是实验纪律和问题定位，而不是 functional success。

已确认：

```text
P0 B320 anchor pass = 1
P1 calibration pass = 1，仅 repair_sketchdim24_rank5_b64_w3_5_10 source 通过
P2 visibility pass = 0
P3 response pass = 0
P4 open = 0
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
promotion_allowed = 0
```

重要数据：

```text
B320 step_ratio_q90_max = 0.39931987348441034
B320 memory_ratio_q90_max = 0.1295238095238095
B320 mean_delta_vs_mlp_min = 0.027669270833333332
B320 worst_delta_vs_mlp_min = -0.001953125
```

P2 不是完全无信号：

```text
best repair protocol = repair_nonrole_geometry/mean_noise_reservoir_rank
best AUC_noise_min = 0.6720854377104377
best AUC_reservoir_min = 0.7308912627551021
best AUC_joint_min = 0.6732622663551402
precision_joint_min = 0.0
recall_joint_min = 0.0
```

这说明当前 features 对 hard release 有弱排序信号，但不能稳定定位 top-k 可执行 release row。

## 1.2 当前失败不是 B320 base 失败

当前 B320 指标说明：

```text
效率足够；
任务轨迹足够；
Line C nontearing 已经站住；
继续小修 B320 architecture 不能解决 functional blocker。
```

所以 v12.18 不再把“找 base”作为主目标，也不允许通过修改 B320 架构制造新进展。B320 只做 anchor monitor 与 label-init ablation。

## 1.3 当前失败也不是 actuator 完全动不了

v12.15/v12.16 continuation 已经显示 fused primitive actuator 可以产生 sketch / projector movement；v12.17.2 又纳入 v12.15 continuation rows 做 audit-only expansion。但结果仍然没有 exploratory/hard joint release row，也没有 P3 survivor。

所以 blocker 更准确地写成：

$$
\boxed{\text{executor 可以产生 movement，但 current target / selector 看不到 release basis。}}
$$

## 1.4 为什么你仍然会觉得进展慢

因为现在已经不是“训练一个模型看 accuracy”的阶段。项目的目标是证明：

$$
\text{B320 + functional update} > \text{B320 + ordinary AdamW / controls}
$$

并且 functional update 必须：

```text
loss-agnostic；
不读取 CE vector；
不读取 label；
不读取 validation/test/future outcome；
不按 dataset name 分支；
能降低 NoiseSignalLeak；
能降低 RealSignalReservoirRatio；
能击败 AdamWParallel / RandomMatchedNorm / SNR-only；
不伤 AUC / ECE / CEp99 / overhead。
```

这个难度比普通 optimizer 调参高很多。慢是正常的，但不能用慢来合理化小修；v12.18 必须换成更严格的代码语义审计和 observable 设计。

---

# 2. 本地代码审查后的关键发现

我已在本地解压 `v1217_code_review_packet.zip`。包内包含：

```text
code/dgkan/models/fc_purekan_primitives.py
code/dgkan/kernels/fused_hinge_quadratic.py
code/dgkan/optim/manual_adamw.py
code/dgkan/profiling/timing.py
code/experiments/run_v1215_*.py
code/experiments/run_v1216_*.py
code/experiments/run_v1217_*.py
code/experiments/run_v1252_*.py
code/experiments/run_v1283_*.py
review_artifacts/v1217_*.csv/json/md
```

## 2.1 发现 A：v12.17.2 runner 是 offline diagnostic，不是新的 functional 实验

`run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py` 明确写着：

```text
This runner is intentionally an offline diagnostic over real v12.16 artifacts.
```

它的 `run_main` 主要读取 v12.16 CSV：

```text
v1216_anchor_monitor.csv
v1216_linec_v2_sketch_targets.csv
v1216_actuator_response_matrix.csv
v1216_explicit_target_calibration.csv
v1216_loss_agnostic_audit.csv
v1216_classic_family_status.csv
```

然后执行：

```text
P0 anchor monitor
P1 Line C calibration
P2 target visibility atlas
P2 visibility repair
P3 actuator response postprocess
P4 gated-not-run artifacts
```

它没有新训练 B320，没有重新生成 P3 actuator，也没有运行 P4 short-run。这个实现是合理的，但结论必须被限定为：

```text
v12.17.2 是 visibility / gate / code-review diagnostic；
不是 functional update candidate generation；
不是 online training success。
```

## 2.2 发现 B：代码包不完整，B320 构造链还不能端到端审完

`run_v1283_b109_classic_family_functional_geometry.py::_make_model` 调用：

```text
v124._make_model(..., y_stats)
```

但本次 code packet 没有包含：

```text
experiments/run_v124_multibasis_functional_dual.py
```

因此我无法仅凭这次 zip 完整确认：

```text
B320 在 actual training runner 中如何传入 y_stats；
PrimitiveSpec 如何被 v124._make_model 转成 SimpleFastTaskGeometryKAN；
strict PureKAN / nonKAN count / parameter coverage 是否在实际构造处被完整执行。
```

v12.18 必须要求 Codex 下次提交 **transitive code bundle**，而不是只提交自己认为关键的几个文件。

## 2.3 发现 C：B320 base 存在 label-informed trainprobe 初始化

在 `SimpleFastTaskGeometryKAN.__init__` 中，若 `"trainprobe" in variant_lower and y_for_stats is not None`，代码会计算每个 class 的 train-stream 均值方向：

```text
mask = y_stats == class_idx
vec = z_stats[mask].mean(dim=0) - global_center
```

并将这些方向写入：

```text
direct_readout
quad_proj
```

相关逻辑包括：

```text
trainprobe_signal_init_uses_labels buffer
trainprobe_signal_init_applied buffer
trainprobedirect
trainprobeP
signalBroad
signalBlock
```

这不是 functional update 的 label leakage，但它是 **base initialization 的 label dependence**。这件事必须被公开、量化和消融。否则 B320 的 base advantage 会被质疑为：

```text
label-informed initialization advantage
```

而不是：

```text
KAN primitive / architecture advantage
```

v12.18 不要求立刻抛弃 B320，但要求：

```text
B320 当前 anchor 继续用于 functional diagnostic；
同时必须跑 B320-label-init ablation；
如果 label-free B320 明显掉 gate，则 external / next-gen MLP claim 必须降级。
```

## 2.4 发现 D：当前 Line C 的部分“loss-agnostic features”实际上有 label / CE provenance

`run_v1252_efficiency_functional_manifold.py::_sample_grad_sketch` 使用：

```text
loss = F.cross_entropy(model(x[i:i+1]), y[i:i+1])
grads = torch.autograd.grad(loss, params, ...)
```

`_signal_reservoir_metrics` 又使用：

```text
ce_real = F.cross_entropy(logits, y, reduction="none")
y_noise = y[torch.randperm(...)]
ce_noise = F.cross_entropy(logits, y_noise, reduction="none")
r_real = ce_real - mean
r_noise = ce_noise - mean
```

这些是很有用的 **审计指标**，但它们不是 loss-agnostic deployable observables。当前 v12.17.2 的 P2 feature table 如果复用了这些字段，就不能被称为严格 label-free feature family。

正确分层应该是：

```text
Audit metrics:
  可以使用 label / CE / permuted-label CE。
  用来判断 update 是否真的降低 NoiseSignalLeak / RealSignalReservoirRatio。

Deployable observables:
  不许使用 label / CE / permuted-label CE。
  只能使用 logits、random cotangent logit-Jacobian、augmentation drift、parameter role response、unlabeled batch statistics。

Functional direction:
  只能使用 deployable observables。
```

v12.18 必须把 Line C 拆成这三层，否则后续会继续在“loss-agnostic”语义上混乱。

## 2.5 发现 E：当前 P2 features 包含 actual response 与 method identity，不能直接作为 deployable pre-commit selector

`build_visibility_features` 里包含：

```text
actual_CouplingR2_delta
method_is_control
method_is_random
method_is_cotangent_vjp
method_is_role_actuator
method_is_projector
cotangent_family_hash
sketch_family_hash
```

这类字段用于 offline atlas 可以接受，但不能直接成为 online selector 的依据，除非在线系统真的能在 commit 前用 cloned probe 得到相同 response，并且成本可接受。

尤其 `cotangent_family_hash` / `sketch_family_hash` 是字符串 hash 后的连续数值，这种编码不具备稳定几何意义。因为 P2 最终失败，它没有造成 false success；但下一版如果 P2 pass，必须进一步要求：

```text
leave-method-out；
leave-source-run-out；
禁止 hash-as-continuous feature 作为 promotion score；
只允许 predeclared physical/geometric features。
```

## 2.6 发现 F：P3 actuator response 没有 matched actuator controls

`run_p3_actuator_response` 写明：

```text
control_gap_checked = 0
control_gap_not_checked_reason = v12.16 actuator matrix has no matched NoOp/Random actuator-control rows; P2 failed so no P3 promotion attempted
```

这很诚实，但也意味着：即使未来 P3 出现 safe movement，也不能只用当前 matrix 宣称 control-resistant actuator success。v12.18 若重开 P3，必须同步生成 matched actuator controls。

---

# 3. v12.18 的整体目标

v12.18 的总目标不是再找 base，也不是继续小修 functional。总目标是：

$$
\boxed{\text{在 B320 locked base 上，建立严格可审计的 loss-agnostic functional value pipeline。}}
$$

这个 pipeline 必须回答四个问题：

```text
Q1: B320 的优势是否依赖 label-informed initialization？
Q2: 不使用 label / CE 的 observables 是否能预测 hard release？
Q3: fused primitive actuator 是否能在 matched controls 下执行这些 observables 指向的 update？
Q4: 如果 P2/P3 成立，P4 short-run 是否真的优于 B320+AdamW / controls？
```

v12.18 的输出不是“functional success”或“failure”二选一，而是以下 route 之一：

```text
R0-CodeAuditIncomplete
R1-B320AnchorRegression
R2-B320LabelInitDependenceDetected
R3-StrictLossAgnosticObservablesNotVisible
R4-ActuatorExecutorNotControlResistant
R5-P3SurvivorFoundP4Opened
R6-P4FunctionalShortRunSuccess
R7-DowngradeToLabelFreeGeometryMaintenance
```

---

# 4. 实验结构

v12.18 分成六条线，可以并行执行，但 promotion 顺序必须严格。

```text
Line A：B320 Anchor + Label-Init Audit
Line R：Critical Code Review / Transitive Dependency Closure
Line C：Audit-only Line C Calibration
Line T：Strict Loss-Agnostic Target Visibility v2
Line I：Actuator Response Dictionary v2 with Controls
Line B：Functional Candidate Construction + P4 Short-run
Line D：Classic No-BSpline Portfolio Monitor
```

Line D 继续保留，但不抢主线资源；B-spline 继续 frozen。

---

# 5. Line R：核心代码审查与依赖闭包

## 5.1 目标

防止再次出现：

```text
runner artifact 完整，但核心 model / metric / actuator / optimizer 代码没有被真正审查。
```

## 5.2 Codex 必须提交的代码包

下次 code review packet 必须包含所有 transitive imports：

```text
experiments/run_v1218_*.py
experiments/run_v1217_*.py
experiments/run_v1216_*.py
experiments/run_v1215_*.py
experiments/run_v1283_*.py
experiments/run_v1252_*.py
experiments/run_v124_multibasis_functional_dual.py
experiments/run_v120_good_geometry_battery.py
experiments/data loading utilities actually used

dgkan/models/fc_purekan_primitives.py
dgkan/kernels/fused_hinge_quadratic.py
dgkan/optim/manual_adamw.py
dgkan/profiling/timing.py
all fused kernels imported by active candidates
```

如果任何实际 import 没有进入 zip，route 必须是：

```text
R0-CodeAuditIncomplete
```

## 5.3 我需要重点审查的核心代码面

Codex 必须指出实际 file / symbol / line range；不能只写 summary。

```text
CR0 runner / route decision / artifact writing
CR1 B320 candidate registry and v124._make_model actual construction
CR2 SimpleFastTaskGeometryKAN label-informed init and buffers
CR3 FHQ fused forward/backward/update kernels
CR4 manual AdamW and optimizer-state semantics
CR5 Line C audit metric: CE/label provenance
CR6 deployable loss-agnostic observable construction
CR7 hard release labels: audit-only isolation
CR8 P2 visibility scorer and split protocol
CR9 P2 feature provenance checker
CR10 P3 actuator generation and response matrix
CR11 P3 matched controls
CR12 P4 short-run event integration and overhead accounting
CR13 timing/memory profiler and hook separation
CR14 provenance/hash/no fake/no proxy/no CPU offload audit
CR15 dataset-agnostic discipline / no dataset-name branch
```

## 5.4 必须落盘 artifact

```text
v1218_transitive_dependency_manifest.csv
v1218_core_code_review_manifest.csv
v1218_symbol_line_map.json
v1218_feature_provenance_audit.csv
v1218_b320_label_init_audit.csv
v1218_manual_review_packet.md
v1218_review_blocker_table.csv
```

## 5.5 通过标准

```text
all_required_files_present = 1
CR0-CR15 unknown_or_not_inspected = 0
core_code_review_pass = 1
manual_review_pending_count = 0 before promotion
```

如果 Codex 找不到某段代码，必须写：

```text
unknown_or_not_inspected = 1
```

不能猜。

---

# 6. Line A：B320 Anchor + Label-Init Audit

## 6.1 假设

H-A1：B320 当前 anchor 仍然稳定。  
H-A2：B320 的 base success 不完全依赖 label-informed trainprobe initialization。  
H-A3：如果 label init 是必要条件，则当前 B320 仍可作为 functional diagnostic anchor，但不能作为 label-free next-gen MLP external claim。

## 6.2 实验设计

固定 B320 当前实现，新增 ablation：

```text
A0 B320-current-labelInit
A1 B320-current-noYForStats
A2 B320-current-randomP-labelFree
A3 B320-current-PCA-P-labelFree
A4 B320-current-lowfreqP-labelFree
A5 MLP same-param AdamW
A6 MLP same-step/FLOP AdamW
```

所有 candidate 使用同一数据、同一 seed、同一训练预算。不得按 dataset 调参。

## 6.3 必须记录字段

```text
candidate_id
uses_y_for_stats
trainprobe_signal_init_uses_labels
trainprobe_signal_init_applied
trainprobeP_enabled
trainprobeDirect_enabled
signalBroad
signalBlock
step_ratio_q90
memory_ratio_q90
mean_delta_vs_mlp
worst_delta_vs_mlp
near_pass_rate
AUC_step_ratio
AUC_time_ratio
ECE_delta
CEp99_delta
LineC_nontearing_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
```

## 6.4 通过标准

B320 current anchor no-regression：

$$
step\_ratio_{q90}\le 0.50
$$

$$
memory\_ratio_{q90}\le 0.20
$$

$$
mean\_delta\ge 0
$$

$$
worst\_delta\ge -0.003
$$

$$
AUC\_step\le 1.00,\quad AUC\_time\le 1.00
$$

Label-free ablation strong pass：

$$
mean\_delta_{labelFree}\ge mean\_delta_{current}-0.005
$$

$$
AUC\_time_{labelFree}\le 1.00
$$

$$
LineC\_nontearing_{labelFree}=1
$$

如果 label-free ablation 失败，但 current B320 仍过，则 route 不阻断 functional diagnostic，但必须标记：

```text
B320_claim_scope = label-informed supervised initialization anchor
external_ready_base_claim = 0
```

## 6.5 失败后 Codex 先尝试

```text
如果 current B320 fail：
  检查 artifact/hash/protocol 错配；
  复现 v12.14/v12.17 anchor；
  不修改 architecture 继续。

如果 only label-free ablation fail：
  尝试 PCA/lowfreq/orthogonal label-free init；
  不允许再加入 label-derived class mean trick；
  不改变 functional line 的 loss-agnostic 约束。
```

## 6.6 可视化

```text
fig_A_b320_label_init_scorecard.svg
fig_A_b320_label_vs_labelFree_auc.svg
fig_A_b320_label_init_linec.svg
fig_A_base_claim_scope_table.svg
```

---

# 7. Line C：Audit-only Line C Calibration

## 7.1 目标

把审计指标和可部署 observables 分开。

Audit-only 指标允许使用：

```text
label
CE vector
permuted-label CE
ECE / CEp99 / Brier
```

但它们只能用于：

```text
hard release label
post-hoc audit
badification constraint
```

不能进入 feature construction 或 functional direction。

## 7.2 记录 artifact

```text
v1218_linec_audit_metrics.csv
v1218_linec_null_distribution.csv
v1218_linec_threshold_sensitivity.csv
v1218_linec_hard_support.csv
v1218_audit_metric_provenance.csv
```

## 7.3 指标

```text
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
Brier_delta
margin_p10_delta
logit_max_abs_drift
hard_noise_release
hard_reservoir_release
hard_joint_release
control_false_positive_rate
bootstrap_ci_low/high
```

## 7.4 通过标准

```text
NoOp hard joint release rows = 0
RandomMatchedNorm hard joint release rows = 0
control false positive rate <= 0.01
Noise hard support >= 5% or mark sparse-support mode
Reservoir hard support >= 5% or mark sparse-support mode
```

如果 official source 出现 false positive，可以使用 repair source，但必须记录：

```text
source_calibration_pass
source_config
why source excluded
```

---

# 8. Line T：Strict Loss-Agnostic Target Visibility v2

## 8.1 目标

判断真正 label-free / CE-free 的 observables 是否能预测 audit-only hard release。

这一步的关键是：feature 可以被 label-free 计算；target 可以用 label/CE 审计。

## 8.2 禁止进入 feature 的字段

```text
label
y
CE
NLL
Brier
permuted-label CE
SNR computed from CE gradient
cross_entropy gradient sketch
hard release label
actual NoiseSignalLeak / ReservoirRatio labels
validation/test outcome
future outcome
dataset name branch
numeric hash of method/family as continuous feature
```

## 8.3 允许的 deployable observables

```text
logit spectrum: logits covariance, entropy, margin-free dispersion
random-cotangent logit-Jacobian sketch: grad of random logit projection, not CE
augmentation consistency: unlabeled x -> aug(x) logit drift
train-probe logit displacement coupling from cloned label-free perturbation
parameter-role response norm / projector angle
output-subspace drift
basis occupancy / role norm / projection condition
kernel drift from logits / features, not CE gradient
```

Random-cotangent sketch 示例：

$$
v\sim\mathcal{N}(0,I_C),\quad s(x)=v^Tf_\theta(x)
$$

$$
g_i=\nabla_\theta s(x_i)
$$

这里不使用 label，也不使用 CE。

## 8.4 实验设计

构造三类 feature table：

```text
T1 strict_precommit_unlabeled_features
T2 cloned_probe_unlabeled_response_features
T3 offline_audit_only_label_features, forbidden for direction, only for comparison
```

必须分别输出可见性结果，不能混表。

Split 必须包括：

```text
leave-dataset-out
leave-seed-out
leave-window-out
leave-method-out
leave-source-run-out
leave-actuator-family-out
```

## 8.5 记录字段

```text
row_id
feature_set
feature_family
feature_name
feature_value
feature_provenance
uses_label_for_feature
uses_ce_for_feature
uses_permuted_label_for_feature
uses_dataset_name_for_feature
precommit_available
clone_probe_required
clone_probe_cost_ms
hard_noise_release
hard_reservoir_release
hard_joint_release
split_protocol
heldout_value
AUC_noise
AUC_reservoir
AUC_joint
precision_at_k_joint
recall_at_k_joint
support_concentration_seed
support_concentration_window
support_concentration_method
```

## 8.6 通过标准

Exploratory：

$$
AUC_{joint,min}\ge 0.65
$$

Hard：

$$
precision@k_{joint,min}\ge 0.25
$$

$$
recall@k_{joint,min}\ge 0.20
$$

Robustness：

```text
support not concentrated in a single seed/window/method/source;
leave-method-out pass;
leave-source-run-out pass;
feature_provenance all strict loss-agnostic.
```

如果只出现：

```text
AUC_joint ~ 0.65-0.70
precision/recall = 0
support concentrated in one seed/window
```

则 route 保持：

```text
R3-StrictLossAgnosticObservablesNotVisible
```

## 8.7 失败后 Codex 先尝试

```text
如果 label-free features AUC 低：
  新增随机 cotangent logit-Jacobian ensemble；
  新增 augmentation consistency；
  新增 output-subspace drift；
  不允许引入 CE gradient sketch。

如果 AUC 高但 precision/recall 低：
  检查 support concentration；
  增加 windows / seeds / actuator dictionary；
  不降低 release threshold。

如果 feature provenance audit 发现 CE/label：
  该 feature family 标记 forbidden_for_direction；
  只能保留为 audit-only。
```

## 8.8 可视化

```text
fig_T_feature_provenance_heatmap.svg
fig_T_auc_by_feature_family.svg
fig_T_precision_recall_by_split.svg
fig_T_support_concentration.svg
fig_T_labelFree_vs_auditOnly_comparison.svg
fig_T_leave_method_source_out.svg
```

---

# 9. Line I：Actuator Response Dictionary v2 with Controls

## 9.1 目标

在 P2 visibility 有可靠 label-free observable 后，验证 fused primitive actuator 是否能执行这些方向，并击败 matched controls。

## 9.2 必须生成的 actuator families

```text
I1 direct-readout role actuator
I2 quad-readout role actuator
I3 quad-proj low-rank actuator
I4 branch/gain actuator
I5 mixed role actuator
I6 random matched-norm actuator
I7 AdamW-parallel matched actuator
I8 SNR-only matched actuator
```

## 9.3 记录字段

```text
actuator_id
actuator_family
dataset
seed
window
sketch_delta_fro
projector_angle_deg
logit_max_abs_drift
predicted_label_free_score_delta
audit_NoiseSignalLeak_delta
audit_RealSignalReservoirRatio_delta
audit_CouplingR2_delta
control_gap_vs_NoOp
control_gap_vs_RandomMatchedNorm
control_gap_vs_AdamWParallel
control_gap_vs_SNR
safe_movement_gate
exploratory_release_gate
hard_release_gate
```

## 9.4 通过标准

Safe movement：

$$
sketch\_delta\_fro\ge 0.01
$$

$$
projector\_angle\ge 1^\circ
$$

$$
logit\_max\_abs\_drift\le 0.05
$$

Release audit：

$$
\Delta NoiseSignalLeak\le -0.01
$$

$$
\Delta RealSignalReservoirRatio\le -0.01
$$

Control resistance：

$$
control\_gap\ge 0.005
$$

Group gate：

```text
safe movement pass in >=6/9 dataset-seed groups;
release pass in >=3/9 groups for exploratory;
no pass concentrated in one dataset or seed;
matched controls present for every row.
```

## 9.5 失败后 Codex 先尝试

```text
如果 actuator movement 小：
  扩展 role basis，但保持 logit drift cap；
  不放大 lambda 到破坏 safety。

如果 movement 大但 release 没有：
  回到 Line T target design，不继续调 actuator。

如果 beat controls 失败：
  记录 control解释；
  不允许进入 P4。
```

---

# 10. Line B：Functional Candidate Construction + P4 Short-run

## 10.1 开启条件

只有当：

```text
Line R pass
Line A current B320 pass
Line C calibration pass
Line T hard visibility pass
Line I response pass
```

才允许 P4。

## 10.2 Functional candidate 原则

Functional candidate 只能由 label-free observables 和 actuator response dictionary 构造：

$$
\Delta\theta_{func}=\sum_j a_j A_j
$$

其中 $A_j$ 是 actuator basis，$a_j$ 只能由 label-free objective 求得。

禁止：

```text
CE vector
label
permuted-label CE
validation/test/future outcome
dataset-name branch
single dataset special threshold
```

## 10.3 P4 short-run 设计

```text
base = B320-current
functional_event_interval = 24, 48
seeds = 0,1,2
datasets = MNIST,Fashion-MNIST,KMNIST
controls = NoOpMatchedOverhead, RandomMatchedNorm, AdamWParallel, SNR-only
```

## 10.4 必须记录字段

```text
dataset
seed
step
event_id
event_accepted
candidate_id
functional_norm_ratio
amortized_overhead
train_loss
val_loss
AUC_step_ratio
AUC_time_ratio
val_acc_delta_vs_B320
ECE_delta_vs_B320
CEp99_delta_vs_B320
Brier_delta_vs_B320
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
control_gap
accepted_event_count
rejected_event_count
no_op_equivalent_event_count
fail_reason
```

## 10.5 通过标准

Task / cost non-harm：

$$
AUCtime_{func}\le AUCtime_{B320}
$$

$$
AUCstep_{func}\le AUCstep_{B320}
$$

$$
Acc_{func}\ge Acc_{B320}-0.003
$$

$$
ECE_{func}\le ECE_{B320}+0.02
$$

$$
CEp99_{func}\le CEp99_{B320}+0.05
$$

$$
T_{amortized,func}/T_{B320}\le 1.05
$$

Geometry audit improvement：

$$
\Delta CouplingR^2\ge 0.02
$$

$$
\Delta NoiseSignalLeak\le -0.01
$$

$$
\Delta RealSignalReservoirRatio\le -0.01
$$

Control resistance：

$$
control\_gap\ge 0.005
$$

No-op exclusion：

```text
accepted_event_count > 0
no_op_equivalent_event_count / accepted_event_count <= 0.20
```

## 10.6 失败后 Codex 先尝试

```text
如果 P4 task / AUC fail：
  区分 AUC-step 和 AUC-time；
  AUC-step fail = mechanism fail；
  AUC-time only fail = overhead/kernel issue；
  不做 dataset-specific repair。

如果 P4 geometry fail：
  回到 Line T / I；
  不继续 lambda 网格。

如果 controls 解释：
  记录被哪个 control 解释；
  不允许 official success。
```

## 10.7 可视化

```text
fig_B_p4_val_loss_vs_step.svg
fig_B_p4_val_loss_vs_time.svg
fig_B_p4_event_timeline.svg
fig_B_p4_linec_trajectory.svg
fig_B_p4_control_gap.svg
fig_B_p4_overhead_breakdown.svg
```

---

# 11. Line D：Classic No-BSpline Portfolio Monitor

## 11.1 目标

经典基函数不作为本轮主 blocker，但要保留 no-regression monitor。

```text
Active families: Rational, Chebyshev, Wavelet, RBF/FastKAN, Fourier
Frozen: B-spline
```

## 11.2 本轮规则

只允许执行有新 hypothesis 的 family repair；不重复旧 focused candidates。

必须记录：

```text
family
status
new_hypothesis_implemented
best_candidate
L3 efficiency
A4 expression
A5 task
LineC geometry
blocker
```

如果没有新 hypothesis：

```text
new_hypothesis_implemented = 0
status unchanged
```

---

# 12. 总 route 决策

v12.18 最终 route：

```text
R0-CodeAuditIncomplete:
  transitive code bundle or CR0-CR15 incomplete.

R1-B320AnchorRegression:
  current B320 no-regression failed.

R2-B320LabelInitDependenceDetected:
  current B320 passes, but label-free ablation fails strongly.
  Functional diagnostic may continue, external-ready base claim closed.

R3-StrictLossAgnosticObservablesNotVisible:
  Line T strict label-free features fail heldout visibility.

R4-ActuatorExecutorNotControlResistant:
  Line T passes, but Line I actuator cannot safely execute or beat controls.

R5-P3SurvivorFoundP4Opened:
  Line T/I pass and P4 started; no success claim yet.

R6-P4FunctionalShortRunSuccess:
  P4 passes task, geometry, cost, controls.

R7-DowngradeToLabelFreeGeometryMaintenance:
  hard release is not observable under strict loss-agnostic features, but label-free geometry maintenance may still be useful as a different, weaker claim.
```

---

# 13. v12.18 的最重要执行顺序

```text
Step 1:
  Codex 生成完整 transitive code review packet。
  如果 run_v124 或任何 actual import 缺失，停止。

Step 2:
  B320 label-init audit。
  确认当前 B320 是否依赖 y_for_stats。

Step 3:
  重新定义 Line C：audit-only vs deployable observable 分层。

Step 4:
  严格 loss-agnostic feature table。
  禁止 CE-gradient sketch 和 label-derived fields 进入 features。

Step 5:
  Target visibility v2。
  必须 leave-dataset / seed / window / method / source out。

Step 6:
  只有 P2 pass，才生成 matched-control actuator dictionary。

Step 7:
  只有 P3 pass，才运行 P4 short-run。
```

---

# 14. 本计划的 justification

这份计划不是把路线改乱，而是把 v12.17.2 的真实问题向下挖了一层。

v12.17.2 的报告结论是：

```text
loss-agnostic observable 看不到 hard release。
```

本地代码审查后，我认为更准确的是：

```text
当前“loss-agnostic observable”这个概念在实现上还混合了三类对象：
  1. label/CE-derived audit metric；
  2. cloned probe response；
  3. 真正可部署的 label-free observable。
```

如果不先拆开这三类对象，继续实验会很容易出现两种错误：

```text
错误 A：把 CE/label-derived audit score 包装成 loss-agnostic direction。
错误 B：把 offline response atlas 包装成 online precommit selector。
```

v12.18 的关键是阻止这两种错误，同时保留正确方向：

```text
B320 作为当前 strongest PureKAN base anchor；
functional update 继续作为最终差异化目标；
Line C 继续作为 signal/reservoir/noise audit；
但方向生成必须严格 label-free / CE-free。
```

如果 v12.18 证明 strict label-free observables 仍然看不到 hard release，那并不等于项目失败，而是说明：

$$
\boxed{\text{当前 hard-release functional update 目标在 strict loss-agnostic 条件下不可观测。}}
$$

那时 functional update 应降级为：

```text
label-free geometry maintenance / stability regularized update
```

而不是继续追 noise/reservoir hard release。

